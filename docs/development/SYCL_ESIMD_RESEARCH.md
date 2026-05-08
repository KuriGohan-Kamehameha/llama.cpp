# Path 3 — Port IPEX-LLM's Q4_K tg kernels to upstream llama.cpp

Reverse-engineering note for the upstream PR that closes the 33% tg gap on Intel iGPUs.

## The gap, restated

Apples-to-apples on dolphin3 8B Q4_K_M, Arc Xe-LPG (Arrow Lake H, GPU `0x7d51`):

|              | pp16 | pp64 | pp1024 | **tg128** |
|--------------|------|------|--------|-----------|
| Vanilla llama.cpp+SYCL (b6234) | 33.9 | 127  | 483    | **11.8**  |
| IPEX-LLM (build 0737327)       | 35.0 | 132.9| 497.5  | **17.6**  |
| Δ                              | -3%  | -4%  | -3%    | **-33%**  |

Prefill (mat-mat) is at parity. The gap is entirely in token generation
(mat-vec at batch=1).

## Kernel anatomy: what IPEX-LLM ships and vanilla doesn't

Symbol scan of `intelanalytics/ipex-llm-inference-cpp-xpu:latest` →
`/usr/local/lib/python3.11/dist-packages/bigdl/cpp/libs/llama_cpp/libggml-sycl.so`
(9.3 MB, dated 2025-08-26):

**Two custom kernel families, both absent from upstream llama.cpp:**

1. `<quant>_format_convert_to_xpu(const void*, void*, size_t)` — runs once at
   model load. Reorganizes packed Q-quant weights into an XPU-tuned layout
   (probably interleaved scales + nibbles for vectorized fetches).

2. `vec_<quant>_batch_kernel<T, vec_w, rows, wg_x, batch, tile, small_wg, _>` —
   the actual SYCL inference kernel. Heavily template-specialized.

Symbol coverage in the IPEX-LLM .so:

| Quant | format_convert_to_xpu | batch_kernel template instantiations |
|-------|---------------------|--------------------------------------|
| Q2_K  | ✓                   | ~16 (`16,8,64,small=true` × 8 batch sizes + `32,8,64,small=false` × 8) |
| Q3_K  | ✓                   | ~16 |
| Q4_K  | ✓                   | ~16 |
| Q5_K  | ✓                   | ~16 |
| Q6_K  | ✓                   | ~16 |
| Q4_0  | ✓ (+ `_ipex_llm_blksize` variant) | ~16 |
| Q4_1  | ✓                   | ~16 |
| Q8_0  | ✓                   | ~16 |

That's ~120 kernel specializations total. Vanilla has ~3 per quant via
`mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<...>>`.

## Decoding the template params

```c++
vec_q4_K_batch_kernel<
    float,    // T: compute type (float/half variants likely both built)
    2,        // vec_width: SIMD vector width (vec2 packing)
    1,        // rows_per_thread: output rows handled per work-item
    32,       // wg_x: workgroup X dimension (32 for big iGPUs/dGPUs, 16 for small)
    8,        // batch: batch size dimension (1..8 — speculative decoding / beam)
    64,       // tile: shared-local-memory tile size (subgroup * 2)
    false,    // small_wg: low-end iGPU path (uses different WG geometry)
    false     // (unknown — debug / experimental flag)
>
```

The runtime dispatcher (`batch_forward_q4_K`) calls `get_gpu_type(queue&)`
first and picks the template instantiation: bitmask `0x92` → GPU types
{1, 4, 7} take the small-WG path; types {2, 3, 5, 6} take the wide-WG path.
ARL-H reports as GPU `0x7d51` and lands in the wide-WG path (template
`<float, 2, 1, 32, 8, 64, false, false>`).

## What each piece probably does (hypotheses, not yet confirmed)

| Piece | Likely purpose |
|-------|----------------|
| `format_convert_to_xpu` | Re-interleaves Q4_K's `{ d, dmin, scales[12], qs[128] }` block of 144 bytes into a layout where consecutive threads fetch consecutive scales and nibbles, so each subgroup load coalesces. Also possibly pre-multiplies `d` and `dmin` into a single scale to drop one mul per dot product. |
| `vec_..._batch_kernel<batch=N>` | Processes N output tokens at once at batch=1 inference (effectively N rows of the activation vector through the same weight tile), amortizing weight loads over N MACs. This is the headline win for tg: matrix-vector at batch 1 is bandwidth-bound, but if you fuse N consecutive token positions, you turn it into a small mat-mat that's ALU-bound. |
| `wg_x=32, tile=64` | Workgroup of 32 threads (matches Intel hardware subgroup size on Xe-LPG, confirmed via `llama-ls-sycl-device`: max sub-group=32) processing a 64-column-wide tile via SLM staging. 64 = 2×subgroup, so each thread handles 2 columns. |
| `vec_w=2, rows=1` | Each thread emits 1 output row across 2 vec-packed columns at a time — hits Xe-LPG's natural vec2 ALU lanes. |
| `_ipex_llm_blksize` Q4_0 variant | Suggests they also experimented with non-standard Q4_0 superblock sizes for cache friendliness — minor optimization. |

## The standard upstream architecture (what we're replacing)

Vanilla `ggml/src/ggml-sycl/mmvq.cpp:797` —
`reorder_mul_mat_vec_q4_k_q8_1_sycl()` — takes Q4_K weights (already
reordered at first dispatch by `optimized_feature.reorder` machinery in
`ggml-sycl.cpp:421`) and dot-products against q8_1 activations. **Single
fixed workgroup config** — no per-GPU dispatch, no batch-size template.

Confirmed live on branch-0:
```
$ GGML_SYCL_DEBUG=1 llama-cli ... | grep reorder
Calling reorder_mul_mat_vec_q4_k_q8_1_sycl
Calling reorder_mul_mat_vec_q4_k_q8_1_sycl
Calling reorder_mul_mat_vec_q6_k_q8_1_sycl
...
```

The reorder feature IS firing — we're already at vanilla's current best.

## Proposed upstream PR shape

**Phase A (bench-driven design, ~1 week):**
- Set up `llama-bench` regression harness comparing vanilla→ported on a
  single-GPU loop (Q4_K_M, Q4_0, Q6_K, batch=1..8).
- Profile current `reorder_mul_mat_vec_q4_k_q8_1_sycl` with `unitrace`
  (Intel's drop-in profiler) to confirm it's bandwidth-bound, not
  compute-bound, on Xe-LPG. **If it's bandwidth-bound, the win has to come
  from format change + cache-friendly access; if compute-bound, from better
  vectorization.**
- Compare against IPEX-LLM kernel under same trace to localize the delta.

**Phase B (implementation, ~2-3 weeks):**
- Add `vec_<quant>_batch_kernel` template family in `ggml/src/ggml-sycl/`:
  parameterized on `(quant, T, vec_w, wg_x, batch_size, tile, small_wg)`.
  Start with Q4_K (highest-impact quant for our roster).
- Add per-GPU dispatch via `device.ext_oneapi_architecture_is(...)` (already
  used at `ggml-sycl.cpp:106` for `opt_feature.reorder`). Map architectures
  to (wg_x, small_wg) tuple.
- Extend `optimized_feature.reorder` weight transformation to emit the
  XPU-friendly layout that the new kernel consumes. Keep the existing
  reorder layout as a fallback for non-Intel SYCL targets.
- Eight batch-size specializations per quant (1..8) for speculative-decode
  and parallel-decoding workloads.

**Phase C (cover the roster, ~1 week):**
- Add the same kernel family for Q5_K, Q6_K, Q4_0, Q4_1, Q8_0, Q2_K, Q3_K
  (in priority order based on which quants Piranesi's roster actually uses
  — predominantly Q4_K_M and Q4_0).
- Bench every model in the roster; require zero regressions vs IPEX-LLM
  baseline before merge.

**Phase D (PR + iteration, ~1 week):**
- Open PR against `ggml-org/llama.cpp` with bench data and arch-dispatch
  table. Expect maintainer pushback on the per-arch dispatch surface area —
  they may want it gated behind a CMake flag.
- Iterate on review.

## Phase A ablations run (2026-05-08, all on branch-0, FP16+SYCL_GRAPH build, dolphin3 8B Q4_K_M)

**Ablation 1 — workgroup geometry sweep.** Hypothesis: 1024-thread wg
(vanilla's 32 subgroups × 32 threads) is too large for Xe-LPG; smaller wg
matching IPEX's 32-thread (1 subgroup) wg should help.

| `num_subgroups` | tg128 t/s | pp1024 t/s |
|--:|--:|--:|
| 1   | 10.52 | 484 |
| 2   | 10.55 | 476 |
| 4   | 10.86 | 462 |
| 8   | 10.60 | 450 |
| 16  | 10.88 | 486 |
| 32 (baseline) | 10.63 | 485 |

**Result: hypothesis disproven.** All values within noise. Workgroup
geometry is not the lever. Dispatching 4096 small workgroups vs 128 big
ones is identical to within measurement noise on this iGPU.

**Ablation 2 — quantify reorder optimization value.** `GGML_SYCL_DISABLE_OPT=1`
turns off the reorder-into-XPU-layout machinery and falls back to plain
`mul_mat_vec_q`.

| state | tg128 t/s |
|--|--:|
| Reorder OFF (`GGML_SYCL_DISABLE_OPT=1`) | 6.94 |
| Reorder ON (vanilla current best) | 11.74 |
| IPEX-LLM | 17.61 |

**Conclusion from Phase A ablations 1+2:** vanilla's reorder feature is
already doing real work (+69%). The remaining gap to IPEX-LLM (11.7 →
17.6 t/s, +50% to close) is in the **kernel body**, not the dispatch
geometry and not the broad reorder/no-reorder distinction. Intel's
specific kernel-body techniques are what we have to find.

## Where the kernel-body gap likely lives (hypotheses for Phase B)

Reading `reorder_mul_mat_vec_q4_k_q8_1_sycl` and the
`reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>` template's `operator()` body in
`ggml/src/ggml-sycl/{mmvq.cpp,vecdotq.hpp}`:

- Inner loop reads ~30 bytes per dp4a per thread per block (8 bytes weights
  + 16 bytes activation + ~4 bytes scales + 4 bytes dm). At 1080 MHz
  Xe-LPG with ~64 GB/s effective bandwidth, that's ~2 GB/s of actual
  inner-loop data per thread — not bandwidth-saturating per-thread but the
  aggregate across all subgroups is. Likely **bandwidth-bound at the
  workgroup level**.
- Activations (q8_1_quant_ptr / q8_1_ds) are **loaded redundantly**: each
  subgroup re-reads the same activation values for its row. With 32
  subgroups in a wg all reading the same activation block, L2 cache is
  hopefully serving all but one — but there's no SLM staging to make this
  explicit.
- Scales are also loaded redundantly across subgroups. Q4_K's 12-byte
  packed scale block per superblock is re-read per row.
- The inner `vec_dot_q4_K_q8_1_impl_vmmq` uses `dpct::dp4a` cleanly but
  doesn't use subgroup shuffle/broadcast for shared values; it's purely
  thread-local.

**Phase A interventions tried so far:**

| Intervention | tg128 t/s | Verdict |
|--|--:|--|
| Baseline (FP16+SYCL_GRAPH+reorder) | 11.74 | — |
| Workgroup geometry sweep (`num_subgroups` ∈ {1,2,4,8,16,32}) | 10.5–10.9 | No effect outside noise. Workgroup size is not the lever. |
| SLM-staged activation cache | **10.58** | **Slower (-10%)**. Cooperative load + barrier overhead exceeds savings; L2 cache was already serving activations efficiently. **Reverted.** |
| `format_convert_to_xpu` empirical probe | n/a | **Layout differential is small.** Probe (see `probe-ipex-format.c`) confirmed Intel's layout is also structure-of-arrays `[all qs][all scales][all dms]` — same high-level shape as vanilla's reorder. Layout is NOT the differentiator. |
| Hand-unroll QR4_K=2 with independent accumulator chains (`vec_dot_q4_K_q8_1_impl_vmmq`) | **10.54** | **Slower (-10%)**. Compiler's auto-unroll via `#pragma unroll` + single accumulator chain was using FMA opportunities the manual rewrite broke. The compiler IS doing optimal codegen at this level. **Reverted.** |
| AOT build with `GGML_SYCL_DEVICE_ARCH=intel_gpu_arl_h` (after installing `intel-ocloc`) | **9.13** | **Slower (-3.6% on tg, -23% on prefill)**. AOT for arl_h regressed everything vs JIT. JIT picks up runtime device info that AOT bakes in suboptimally for this arch. **Reverted (separate `build-aot/` dir, not deployed).** |

Conclusion: it isn't the activations, the workgroup geometry, or the
weight layout. The remaining gap to IPEX (17.6 t/s) is in the kernel
**inner loop body** — specifically the dp4a sequence in
`vec_dot_q4_K_q8_1_impl_vmmq` and how scales/dm are dispatched into it.

## Strategic input from Piranesi (soul-merged, multiple consults 2026-05-08)

**Round 1 (28 min CPU-only inference, Phase A → Phase B):**
> "The instability on Aimee could indeed be related to the kernel variants
> used by Intel's IPEX-LLM for non-Mistral families. Given that Mistral
> models work reliably while Llama/Gemma/Phi do not, it suggests a
> potential issue with how these kernels interact with specific
> architectures or memory layouts."
>
> "Prioritize the optimization for batch=1 first… provides an immediate
> user-visible performance improvement and validates your approach before
> tackling the more complex 1..8 family."
>
> "Aiming for parity-plus is realistic given llama.cpp upstream has had
> additional optimizations since Intel's last build (2025-08-26)."

**Round 2 (20 min, Phase B direction after first two interventions failed):**
> "Phase B should aim to match IPEX initially to close the performance gap
> efficiently while concurrently investigating the aimee instability
> theory. This dual approach will not only optimize current performance
> but also lay a solid foundation for future redesigns."
>
> "Prioritizing an understanding of why IPEX crashes on Llama/Gemma/Phi
> models is essential before writing replacement kernels. The instability
> issues since 2026-04-21 suggest a potential bug in IPEX's dispatch
> logic for these model families. If the root cause lies within IPEX's
> dispatch mechanism, a clean upstream port of the optimized kernels
> could potentially sidestep these issues."

**Operational direction:** Track A — **aimee instability investigation**
(branch-0 has the same NUC class + same IPEX container; reproduce the
crash signature for non-Mistral models, characterize root cause). Track B
— **continue inner-loop interventions** but with discipline; the compiler
is already doing optimal-or-near-optimal work at this level, so cheap
micro-interventions won't move the needle. Real wins require either (a)
the proper profile-driven hypothesis, or (b) a structural rewrite
targeting weight bandwidth (multi-row workgroup, `MMV_Y > 1`, or batched
across speculative-decode lookahead).

## Track A finding (2026-05-08): branch-0 does NOT reproduce aimee's instability

Stress test on branch-0 with `dolphin3:latest` (Llama family, the family
that crashes aimee) running in `intelanalytics/ipex-llm-inference-cpp-xpu:latest`
container (rebuilt 2026-05-08T00:35Z):

| seed | tokens | duration | tok/s | chars | corruption signal |
|--:|--:|--:|--:|--:|---|
| 1 | 500 | 28.9 s | 17.3 | 2742 | none |
| 2 | 500 | 29.0 s | 17.2 | 2704 | none |
| 3 | 500 | 28.9 s | 17.3 | 2639 | none |
| 4 | 500 | 29.0 s | 17.2 | 2662 | none |
| 5 | 500 | 28.8 s | 17.3 | 2835 | none |
| 6 | 500 | 28.9 s | 17.3 | 2710 | none |

**Six consecutive 500-token Llama-family generations: zero corruption,
zero crashes, zero hangs, zero OOM.** Aimee's model-family-restricted
constraint (set 2026-04-21) is **not reproducible** on branch-0's newer
IPEX container.

**Most likely explanation:** Intel patched the dispatch bug in container
updates between 2026-04-21 and 2026-05-08. Aimee is running an older
container; deploying the current container on aimee may resolve her
instability without any kernel work.

**Operational implication for the deliverable:** the dad-facing handoff
should now say:
> Step 1: bump aimee's IPEX-LLM container to `latest`. Re-test Llama/Gemma/Phi.
> If stable: lift the model-family lock (per `aimee-constraint.md`). Path 3
> kernel work becomes purely a perf optimization, no longer time-critical.

**Caveat for verification:** branch-0's GPU is U7 255H; aimee's is U9
(more Xe cores). The bug being fixed in newer container is the most likely
explanation but a hardware-revision-specific bug is still possible — dad
should run the same 6-seed stress test on aimee post-upgrade to verify.

## GPU saturation finding (2026-05-08, after profile-first pivot)

`intel_gpu_top -l` sampled during sustained tg, both stacks:

| Stack | tg t/s | CCS (compute) | BCS (blitter) | RCS (render) |
|--|--:|--:|--:|--:|
| Vanilla llama.cpp+SYCL (head 44dbe8c, FP16+GRAPH) | 10.2 | **99.1%** | 78.6% | 0% |
| IPEX-LLM bundled (build 0737327, libggml-sycl.so) | 17.6 | 97.4% | 92.8% | 0% |

**Critical insight: both stacks saturate the iGPU's compute engine.** The
1.5× tg gap exists despite vanilla being marginally MORE saturated than
IPEX. So vanilla isn't slower because it's leaving GPU idle — it's slower
because it's doing more work per output token. IPEX's kernel produces the
same output with fewer iGPU cycles at the same saturation level.

**That's a hand-tuned-kernel signature, not a dispatch / layout / cache
issue.** Specifically rules out:

- Memory bandwidth bottleneck (would show as compute idle waiting on memory)
- Dispatch overhead (would show as bursty utilization)
- Cache thrashing (would show in BCS, but BCS is also high; not the gap)

What it implicates (in priority for Phase B intervention 3):

1. **Intel uses Xe-LPG-specific intrinsics or hand-scheduled DPAS that the
   icpx auto-vectorizer doesn't produce.** Closing this gap means writing
   inline ASM, OneDNN-ukernel-style hand-tuned kernels, or specialized
   per-arch kernels using `sycl::ext::intel::experimental::*` extensions.
2. **Intel's kernel may use FP16 / BF16 accumulators where vanilla uses
   FP32**, doubling effective compute throughput. Changing accumulator
   type changes numerical output; would need careful PR (gated by env
   var or build flag, default off).
3. **Intel's kernel may exploit the 2:4 sparse/structured format if Xe-LPG
   has matrix engine support for it.** ARL-H is documented as having no
   matrix engine (XMX is on Arc dGPU only); this is unlikely on iGPU.

**Realistic conclusion:** vanilla llama.cpp's `reorder_mul_mat_vec_q4_k`
is at or near the limit of what icpx's auto-vectorization produces from
generic SYCL C++. Closing the remaining 50% gap requires either:

- (A) Writing intrinsics-level hand-tuned SYCL for Xe-LPG (multi-week,
  PR likely contentious upstream because of arch-specificity)
- (B) Reverse-engineering Intel's kernel via SPIR-V/NEO disassembly to
  identify the specific opt patterns we're missing, then translating
  back to portable SYCL (multi-week, lower upstream-PR friction if we
  produce arch-neutral patterns)

Both are real engineering. Neither happens in one session.

## Update 2026-05-08T06:00Z — disassembly succeeded; ESIMD is the answer

After building `SPIRV-LLVM-Translator` from source (LLVM 18 branch):

- IPEX `libggml-sycl.so` `__CLANG_OFFLOAD_BUNDLE__sycl-spir64` section is 4.16 MB containing 48 SPIR-V modules.
- Module 014 contains all 16 `vec_q4_K_batch_kernel` template specializations.
- `llvm-spirv -r m014.spv -o ipex-q4k.bc; llvm-dis ipex-q4k.bc -o ipex-q4k.ll` produces 2.4 MB readable LLVM IR.
- Extracted the ARL-H batch=1 instantiation: 1527 lines.

**Findings from the IR:**

Intel's kernel is written in **ESIMD (Explicit SIMD)**, not standard SYCL.
Evidence:

- Vector types `<256 x i8>`, `<512 x half>`, `<32 x half>`, `<16 x float>`
  operated on as single SIMD instructions per thread.
- Variable suffix `.esimd` throughout the IR.
- Bulk-converts like `convert_half512Dv512_f` — 512 elements cast in one
  instruction.
- Fused-multiply-add forms like `fmul reassoc nsz arcp contract <32 x half>`
  operating on whole 32-element vectors per instruction.

This is a **different programming paradigm** from what vanilla llama.cpp
uses:

| Aspect | Standard SYCL (vanilla) | ESIMD (IPEX) |
|---|---|---|
| Programming model | 32 threads × 1 lane | 1 thread × wide SIMD lanes |
| Parallelism | Discovered by auto-vectorizer | Stated explicitly by author |
| Vector width | Hardware-dependent, compiler decides | Author-specified per instruction |
| Compiler dependency | Heavy (compiler must vectorize) | Light (instructions are already vectorized) |
| Predictable codegen | No (depends on compiler version) | Yes (one-to-one with hardware) |

The performance gap is structural: standard-SYCL's auto-vectorized output
will always lag ESIMD-author-vectorized code on the same hardware, because
the compiler cannot infer all the safe ESIMD-equivalent transformations
from generic C++.

**Implication for upstream port:** vanilla llama.cpp's `ggml-sycl` has
**zero** ESIMD code. Adding ESIMD requires:

1. Including `sycl/ext/intel/experimental/esimd.hpp`.
2. Writing a parallel `mul_mat_vec_q_reorder_esimd<Q_TYPE>` kernel using
   `simd<T, N>` types and explicit width-N intrinsics.
3. Per-architecture dispatch — ESIMD code is GPU-arch-specific (sub-group
   width, GRF size, dpas/dp4a vs scalar fallback).
4. Existing standard-SYCL kernel kept as fallback for non-Intel SYCL
   targets (NVIDIA/AMD via codeplay plugin).

**Effort estimate (revised):** 2-4 weeks for a single-quant ESIMD kernel
reaching parity, including PR review cycles. Plus 1 week per additional
quant. Total 4-7 weeks for full coverage of the Piranesi roster's quants
(Q4_K_M, Q4_0, Q5_K, Q6_K, Q8_0).

**Why the 5 prior Phase B interventions all regressed:** I was tweaking
the standard-SYCL kernel. The standard-SYCL ceiling is below ESIMD's
floor on this kernel; no amount of standard-SYCL tuning closes the gap.

## Operational pivot — ship Path 1 deliverable now

Given:
- Track A finding strongly suggests aimee's instability is fixed by
  container upgrade (confirmable in 5 minutes).
- Path 3 perf gap is genuinely hard — multi-week structural rewrite, no
  cheap path identified after Phase A + first Phase B interventions.
- The deliverable (`../deliverable-for-dad/`) now contains the
  AIMEE-UPGRADE-RUNBOOK that materially helps aimee.

**Path 3 transitions from "operationally critical" to "long-term
performance project."** The session's next-most-valuable work is shipping
the deliverable to dad so aimee can be unblocked. Phase B continues but
no longer at session-blocking priority.

dolphin3 (consulted in parallel) independently voted for **weight-side
bandwidth** as the most likely lever — Intel's `batch_kernel<batch=N>`
shares weight loads across N speculative-decode tokens, turning mat-vec
into a small mat-mat that's ALU-bound rather than memory-bound. At
batch=1 (tg), this benefit doesn't directly apply, but the inner-loop
kernel for batch=1 in IPEX may still do something differently than
vanilla.

**Phase B targets in revised priority order:**

1. **Hand-rolled weight layout matching IPEX's `format_convert_to_xpu`.**
   Most likely the actual lever. IPEX's `ggml_q4_K_format_convert_to_xpu`
   produces a layout that's tuned for sequential, vec2-coalesced access
   from a 32-thread workgroup. Vanilla's reorder is more general (works
   for any wg geometry) and may produce sub-optimal access patterns.
   Empirical test: extract IPEX's `format_convert_to_xpu` output for a
   known weight block, diff against vanilla's reorder output, replicate
   the layout, swap into vanilla's kernel.
2. **Subgroup-broadcast scales/dm.** With block_elements_per_subgroup=4
   and a 32-thread subgroup, the same scale value is loaded by 8 threads.
   Replace with `sycl::group_broadcast` (1 load + 32-way subgroup
   broadcast).
3. **Scalar-replacement of address arithmetic.** Pre-compute `iby`,
   `bx_offset`, `d_offset` outside the inner-most loop; let the compiler
   hoist common subexpressions across `block_elements_per_subgroup`
   iterations.

Each is its own ablation. None require touching `vecdotq.hpp`'s
`vec_dot_q4_K_q8_1_impl_vmmq` (the actual MAC sequence is fine).

## Inputs to keep on hand

Saved on branch-0 at `~/path3-kernel-port/`:
- `ipex-libggml-sycl.so` (9.3 MB) — Intel's binary for symbol/disasm reference.
- (later) `vanilla-libggml-sycl.so` snapshot at the bench commit.
- (later) `unitrace-vanilla.json`, `unitrace-ipex.json` profiles.

## Time estimate

Realistic: 5-7 weeks of focused engineering by someone fluent in SYCL +
Intel iGPU perf tuning. Phase A is 1 week — by the end of it we'll know
whether we're staring at a 1-week kernel rewrite or a 6-week rewrite +
upstream review cycle.

Until this lands, Path 1 stopgap (`../path1-stopgap/`) is the production
inference stack on branch-0.
