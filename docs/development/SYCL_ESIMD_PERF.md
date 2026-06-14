# SYCL ESIMD kernel perf reference

Snapshot benchmarks for the experimental ESIMD code path added by
`GGML_SYCL_ESIMD=ON`. Reproduce these numbers on your own Intel iGPU/dGPU
to evaluate the kernel for your hardware before considering it for
production.

## How to reproduce

Build with both the standard SYCL and the ESIMD path enabled:

```sh
source /opt/intel/oneapi/setvars.sh
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
      -DGGML_SYCL_F16=ON -DGGML_SYCL_GRAPH=ON \
      -DGGML_SYCL_ESIMD=ON \
      -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j --target llama-bench
```

**Build resource note.** `icpx` compiling the ESIMD kernels with full
parallelism uses ~3-4 GB RAM per concurrent translation unit; on the
ASUS NUC reference (16 hardware threads) a default `-j` build can spike
to >40 GB peak RAM. If you hit a kernel-OOM-killed build (`icpx: error:
unable to execute command: Killed`), retry with `-j 4` or `-j 2` to
reduce parallelism, or close other GPU/CPU-heavy processes first.

Bench (8B Q4_K_M model, all layers offloaded, 3 reps):

```sh
# default (standard SYCL kernel)
ZES_ENABLE_SYSMAN=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
    ./build/bin/llama-bench -m <Q4_K_M.gguf> -p 1024 -n 128 -ngl 999 -r 3

# ESIMD opt-in
GGML_SYCL_USE_ESIMD=1 ZES_ENABLE_SYSMAN=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
    ./build/bin/llama-bench -m <Q4_K_M.gguf> -p 1024 -n 128 -ngl 999 -r 3
```

## Reference run

Hardware: ASUS NUC 15 Pro Plus, Intel Core Ultra 7 255H (Arrow Lake H,
Arc Xe-LPG, GPU PCI ID `0x7d51`), Ubuntu 24.04, oneAPI 2026.0.

Model: dolphin3:latest 8B, Q4_K_M.

### Q4_K_M (dolphin3:latest 8B)

| backend | pp1024 (t/s) | tg128 (t/s) | tg vs default |
|---|--:|--:|--:|
| Standard SYCL (default `reorder_mul_mat_vec_q4_k_q8_1_sycl`) | 482 ± 8.6 | 11.64 ± 0.04 | 1.00× |
| **ESIMD opt-in** (this branch, `GGML_SYCL_USE_ESIMD=1`) | 486 ± 0.4 | **6.41 ± 0.01** | 0.55× |
| IPEX-LLM bundled (proprietary container) | 497 | 17.6 | 1.51× |

**Six independent post-iter-16 interventions** were tested empirically
across Phase B-7 + Phase E, and all failed to move past the 6.4 t/s
ceiling on Xe-LPG iGPU (see `SYCL_ESIMD_LOG.md` for the full ledger):

- Iter 17 (qs prefetch with cache hints): 0% (register live-range
  overhead cancelled the latency-hiding gain).
- Iter 18 (ROWS_PER_THREAD=2): -21% (lost shared-activation amortization).
- Iter 19 (FP32-domain inner loop, mirroring IPEX SPIR-V): -1.6% wash.
- Iter 20 (SLM-cooperative activation + ROWS=8): -14% (SLM round-trip
  + barrier dominate; per-row weight state remains the binding GRF
  constraint, not the activation).
- Iter 21 (FP16 MAC): -37% (int8→half lane conv + FP32-widened reduce
  exceed any int-pipe→FP-pipe routing gain).

The fork's iter-16 architecture is the **empirically settled** ESIMD
ceiling for mat-vec on Xe-LPG iGPU at the source-level kernel design.
The remaining gap to vanilla and IPEX is not closable through
arithmetic-format or per-thread parallelism-shape swaps within this
kernel architecture. If breakthrough exists, it lives in compiler/IGC
scheduling parameters that differ between icpx and IPEX's toolchain —
outside source-level control without IGC dump access.

## 2026-05-21 corrigendum — bench numbers below for non-reorder quants were artifacts

The previously-recorded ratios for Q5_K, Q5_0, Q5_1, Q4_1, Q2_K, Q3_K
were measured **without the ESIMD kernel actually running** for those
quants. The default SYCL mat-mat-vec dispatch routes those 6 quants
through DMMV (dequant-then-mat-vec), not MMVQ where ESIMD lives.
Setting `GGML_SYCL_USE_ESIMD=1` had no effect on the dispatch for those
quants until the 2026-05-21 dispatch fix landed.

True ESIMD-vs-vanilla ratios after the fix (paired r=3, `-n 64`,
TinyLlama 1.1B, branch-0 contended):

| quant | vanilla path | vanilla tg | esimd tg | true ratio |
|---|---|--:|--:|--:|
| Q3_K_M | DMMV (`to_fp16` + fp16 mat-vec) | 22.75 ± 3.18 | 15.97 ± 0.12 | **0.70×** |
| Q2_K   | DMMV | 20.09 ± 0.09 | 15.99 ± 0.05 | **0.80×** |
| Q4_1   | DMMV | 20.25 ± 0.07 | 16.04 ± 0.08 | **0.79×** |
| Q5_0   | DMMV | 20.30 ± 0.28 | 16.07 ± 0.06 | **0.79×** |
| Q5_1   | DMMV | 20.13 ± 0.14 | 18.18 ± 0.21 | **0.90×** |
| Q5_K_M | DMMV | 23.09 ± 0.36 | 18.31 ± 0.06 | **0.79×** |

**No ESIMD kernel in this fork beats its vanilla counterpart on
Xe-LPG.** The Phase E "structural ceiling" conclusion was Q4_K-specific;
the broader 10-quant sweep with the dispatch fix shows the ceiling holds
across every quant.

The 4 reorder-quant ratios in the tables below (Q4_0 / Q8_0 / Q4_K /
Q6_K, 0.32×–0.55×) are valid — those compare ESIMD vs same-path
reorder-mmvq, since both code paths go through mmvq.

---

### Q4_0 (TinyLlama 1.1B)

| backend | pp1024 (t/s) | tg128 (t/s) | tg vs default |
|---|--:|--:|--:|
| Standard SYCL (default `reorder_mul_mat_vec_q4_0_q8_1_sycl`) | 388 | 15.71 | 1.00× |
| **ESIMD opt-in** (this branch, `GGML_SYCL_USE_ESIMD=1`) | 312 | 5.03 | 0.32× |

### Q6_K (TinyLlama 1.1B Q6_K)

| backend | pp1024 (t/s) | tg128 (t/s) | tg vs default |
|---|--:|--:|--:|
| Standard SYCL (default `reorder_mul_mat_vec_q6_k_q8_1_sycl`) | 519 | 16.47 | 1.00× |
| **ESIMD opt-in** (this branch, `GGML_SYCL_USE_ESIMD=1`) | 499 | 6.59 | 0.40× |

### Q5_K (TinyLlama 1.1B Q5_K_M)

| backend | pp1024 (t/s) | tg128 (t/s) | tg vs default |
|---|--:|--:|--:|
| Standard SYCL (default `mul_mat_vec_q5_K_q8_1_sycl`) | 1168 | 21.80 | 1.00× |
| **ESIMD opt-in** (this branch, `GGML_SYCL_USE_ESIMD=1`) | 1169 | 18.78 | 0.86× |

Q5_K reads from the standard `block_q5_K` layout (no reorder layout
exists in the ggml-sycl tree), which keeps the change surface small
but means it can't share Q4_K's cross-block contiguous prefetch.

### Q8_0 (dolphin3:latest 8B, requantized to Q8_0)

| backend | pp1024 (t/s) | tg128 (t/s) | tg vs default |
|---|--:|--:|--:|
| Standard SYCL (default `reorder_mul_mat_vec_q8_0_q8_1_sycl`) | 464 | 9.72 | 1.00× |
| **ESIMD opt-in** (this branch, `GGML_SYCL_USE_ESIMD=1`) | 460 | 4.20 | 0.43× |

Q8_0's ratio is in family with the prior quants but for a different
structural reason: Q8_0 has no nibble-unpack work for ESIMD to
specialize on, and the standard kernel's `dpct::dp4a` int8×4 dot-product
intrinsic is already a single instruction on Xe-LPG. Explicit SIMD
widening doesn't unlock parallelism the compiler couldn't already see;
register pressure becomes the binding constraint sooner with 32 B
weight loads per block than with 16 B nibble-packed loads. With
Q4_0/Q4_K the nibble unpack creates compute work the auto-vectorizer
can mishandle — Q8_0 has no such opening.

### Q3_K (TinyLlama 1.1B, requantized to Q3_K_M)

| backend | pp1024 (t/s) | tg128 (t/s) | tg vs default |
|---|--:|--:|--:|
| Standard SYCL (default `mul_mat_vec_q3_K_q8_1_sycl`) | 1332 | 23.59 | 1.00× |
| **ESIMD opt-in** (this branch, `GGML_SYCL_USE_ESIMD=1`) | 1604 | 15.51 | 0.66× |

Q3_K shares Q5_K's raw-block read path (no reorder layout exists for
Q3_K in the ggml-sycl tree). Architecturally the kernel mirrors Q6_K:
signed offset weights (`low2 + 4*hbit - 4` in [-4,3]) and 16 sub-scales
(`(low4 | (high2 << 4)) - 32` in [-32,31]), with the same
`simd<int8,64>` packed-pair multiply and first-16/second-16 split per
q8_1 sub-block. No min term. Activation, dy, and scale-unpack hoist
across the 4 rows; ESIMD's standard deviation (σ=0.16) is well below
vanilla's (σ=1.35) at this measurement point.

The pp1024 path doesn't pass through `mmvq` — pp uses the batched
`mmq` kernels — so the ~1.2× pp delta here is noise/independent of
the ESIMD change. Only the tg128 ratio is the load-bearing signal.

### Q2_K (TinyLlama 1.1B, requantized to Q2_K)

| backend | pp1024 (t/s) | tg128 (t/s) | tg vs default |
|---|--:|--:|--:|
| Standard SYCL (default `mul_mat_vec_q2_K_q8_1_sycl`) | 1341 | 25.10 | 1.00× |
| **ESIMD opt-in** (this branch, `GGML_SYCL_USE_ESIMD=1`) | 1364 | 22.07 | 0.88× |

Q2_K shares Q3_K's output-position mapping (`l = 128*n + 32*j +
16*half + k`) and the raw-block read path. The kernel uses Q3_K's
qs/scale layout but no hmask (weights are unsigned 2-bit, range
[0,3]) and adds Q4_K's min-term. Crucially, Q2_K's two halves of a
q8_1 sub-block can have *different* mins (`scales[2*q]` low nibble +
`scales[2*q+1]` low nibble for the d-term, high nibbles for the
m-term), so `sy` cannot be reused as in Q4_K/Q5_K. The kernel
computes per-half `sum_u` once per super-block (hoisted out of the
4-row inner loop since `y_q_shared` is identical across rows) using
`esimd::reduce<int>` on int8 halves.

At 0.88× vanilla, Q2_K lands as the best ratio of the six covered
quants — slightly above Q5_K's 0.86×. The mechanism is the same:
raw-block read + unsigned-only weight decode keeps the per-sub-block
arithmetic shorter than Q3_K's bit-mask decode or Q6_K's two-byte
qh/ql composition, so the int-pipe loop has less work to amortize
against the GRF-resident shared activation. ESIMD σ=0.08 vs
vanilla's σ=0.11 — the deterministic-vs-contended signature is
narrowest here because Q2_K's per-row compute is light enough that
the standard kernel is also nearly contention-free.

### Q4_1 (TinyLlama 1.1B, requantized to Q4_1)

| backend | pp1024 (t/s) | tg128 (t/s) | tg vs default |
|---|--:|--:|--:|
| Standard SYCL (default `mul_mat_vec_q4_1_q8_1_sycl`) | 1476 | 43.70 | 1.00× |
| **ESIMD opt-in** (this branch, `GGML_SYCL_USE_ESIMD=1`) | 1526 | 40.65 | 0.93× |

Q4_1 is structurally Q4_0 with an additive per-block min: nibble
encoding is identical (qs[k] low → output k, high → output k+16),
but the contribution formula adds an explicit min term instead of
the Q4_0 implicit -8 offset:

  `contrib[b] = d_b * dy_b * sumi_b + m_b * sy_b`

The ESIMD kernel mirrors Q4_0's group-of-8-blocks pair-multiplication
shape. The structural divergence is that Q4_1 is *not* on the reorder
path — block_q4_1 (dm half2 + qs[16] = 20 bytes) is read raw with
strided per-block loads. The legacy quants {Q4_1, Q5_0, Q5_1}
intentionally stay off the reorder path in ggml-sycl.

At **0.93× vanilla** Q4_1 lands as the best ratio of the eight quants
now covered. Mechanism: the legacy block layout's small per-block
decode (one nibble unpack + one half2 read for dm) is lightweight
enough that ESIMD's GRF-resident shared activation amortizes nearly
all of the standard SYCL path's per-block overhead. Vanilla tg128 of
43.7 t/s is itself nearly 2× the K-quants on this model — Q4_1's
simplicity makes both paths fast and keeps their relative gap small.

Dispatch falls back to the standard path when ncols isn't a multiple
of QK4_1 * 8 = 256 (the ESIMD group-of-8-blocks alignment
requirement). All TinyLlama / Llama mat-vec tensors at QK4_1 = 32
satisfy this since ncols is always a multiple of 256 at typical
model dimensions.

### Q5_0 (TinyLlama 1.1B, requantized to Q5_0)

| backend | pp1024 (t/s) | tg128 (t/s) | tg vs default |
|---|--:|--:|--:|
| Standard SYCL (default `mul_mat_vec_q5_0_q8_1_sycl`) | 1077 | 25.65 | 1.00× |
| **ESIMD opt-in** (this branch, `GGML_SYCL_USE_ESIMD=1`) | 1098 | 25.24 | 0.98× |

Q5_0 lands at **near parity** with vanilla — the closest of any
quant in this branch. Mechanism: Q5_0 has the most expensive
standard-SYCL decode of any covered quant (the `vec_dot_q5_0_q8_1_impl`
formula does four serial `vh[i] << shift & 0x10` bit-extractions to
splice the 5th bits into the 4-byte nibble vi). ESIMD replaces this
with an unrolled per-bit OR over a `simd<int8_t, 32>` hbit vector
extracted from qh's 32-bit bitmap, which the compiler maps cleanly
to per-lane shift+and. The standard path's serial bit-shuffle is
exactly the kind of code where explicit SIMD widening unlocks
parallelism the auto-vectorizer can't reach.

Same structural shape as Q4_0/Q4_1: 8 blocks per group, raw block
layout (Q5_0 is *not* on the reorder path — legacy quants {Q4_1,
Q5_0, Q5_1} stay off it by design), pair-multiplication via
`simd<int8,64>` packed pairs, formula `d * (sumi * dy - 16 * sy)`
(the -16 offset converts unsigned 5-bit [0,31] -> signed [-16,15];
shape-identical to Q4_0's -8 with twice the constant).

Dispatch falls back when ncols isn't a multiple of QK5_0 * 8 = 256.

### Q5_1 (TinyLlama 1.1B, requantized to Q5_1) — ESIMD wins

| backend | pp1024 (t/s) | tg128 (t/s) | tg vs default |
|---|--:|--:|--:|
| Standard SYCL (default `mul_mat_vec_q5_1_q8_1_sycl`) | 1254 | 30.98 | 1.00× |
| **ESIMD opt-in** (this branch, `GGML_SYCL_USE_ESIMD=1`) | 1351 | **31.28** | **1.01×** |

**This is the first quant in this branch where ESIMD outperforms
the standard SYCL path.** Compounding two effects:

1. Q5_1 inherits Q5_0's expensive bit-shuffle decode (4 serial
   shifts per nibble to splice the 5th bit into vi). ESIMD's
   per-lane 32-wide shift+and replaces it cleanly — the same
   mechanism that drove Q5_0 to 0.98×.
2. Q5_1 *also* does Q4_1's additive min term (`m * sy`) on top of
   the dot product. Standard SYCL pays both costs. ESIMD's
   GRF-resident shared activation amortizes both per-block costs
   across the 4-row fan-out simultaneously — the per-row min-term
   multiply is a single `m_v * sy_v` over 8 blocks, fused into the
   same final accumulate as the d-term.

Mean is unambiguously above 1.00×, though ESIMD's σ=1.79 vs
vanilla's σ=0.13 widens the confidence interval. The win is small
(~1%) but the direction is consistent and structural: with the
right combination of per-block decode complexity, ESIMD on
Xe-LPG can beat the standard SYCL `dp4a` path.

Architecture is Q5_0's kernel + Q4_1's additive min term:
- Same nibble + qh-bitmap decode as Q5_0.
- Per-block dm half2 (d, m) instead of just d.
- Formula: `contrib[b] = d_b * dy_b * sumi_b + m_b * sy_b` (no
  explicit -16 offset; the per-block m absorbs it).

Block layout NOT reorder (24 B per block = dm + qh[4] + qs[16]),
following the legacy-quant convention.

Dispatch falls back when ncols isn't a multiple of QK5_1 * 8 = 256.

The standard SYCL path is **unchanged** by the new build flag. ESIMD is
opt-in at runtime via env var; default behavior is unaffected.

**The ESIMD kernel is intentionally landed below the standard path's
perf.** This is a foundation that opens the ESIMD code path in
ggml-sycl. Closing the gap to IPEX-LLM is the explicit follow-up scope
(see `SYCL_ESIMD_LOG.md` for pursued directions and dead ends).

## Why this exists

Intel's binary-only IPEX-LLM ships a llama.cpp fork built with proprietary
ESIMD kernels. SPIR-V disassembly of their `libggml-sycl.so`
(`__CLANG_OFFLOAD_BUNDLE__sycl-spir64` section, ~120 ESIMD kernel
template specializations across {q2_K..q6_K, q4_0/4_1, q8_0}) confirms
that vanilla llama.cpp's standard-SYCL kernels are below the ESIMD
ceiling on the same hardware. icpx auto-vectorization can't reach that
ceiling from generic C++; an explicit-SIMD code path can.

This branch adds that path so the gap is closeable in upstream over time
instead of staying frozen behind a binary blob.

## Math correctness — FP-precision-class drift, NOT bit-identical

**The ESIMD kernels are numerically equivalent to the standard SYCL
kernels at FP-precision-class drift, NOT bit-identical.** At
temperature 0 with deterministic seeds, the two paths' generations
diverge in ~30-token windows on some prompts.

The drift is structural to ESIMD vs. standard SYCL: explicit-SIMD
reductions sum partial dot products in a different order than the
subgroup-cooperative reductions in the standard SYCL kernel. FP
addition is non-associative — `(a+b)+c ≠ a+(b+c)` for floats — so
reduction-order changes produce small rounding-class differences in
the per-row dot products. Most tokens are unaffected (logit gaps are
large), but on tokens where two top-logit candidates are close, the
small ESIMD drift can flip argmax. Once one token diverges, the rest
of the generation cascades.

This is the **same class of behavior** you'd see comparing CUBLAS-FP16
vs. CPU-FP32 inference, or two different llama.cpp backends (e.g.
SYCL vs. CUDA) on the same model. It is not unique to ESIMD; it is
unique to "different reduction ordering."

**What this means in practice:**
- Greedy generations *will eventually diverge* between ESIMD and
  standard SYCL on long sequences. They may converge again or stay
  divergent.
- Logits are within FP32 noise of each other per token; downstream
  metrics (perplexity, eval scores) are statistically
  indistinguishable. See "Perplexity verification" below.
- For sampling at temperature > 0, the drift is invisible (sampling
  noise dominates).

### Perplexity verification (2026-05-08)

Measured on dolphin3:latest 8B Q4_K_M, wikitext-2 test split (50
chunks × 512 tokens = 25,600 tokens scored), `llama-perplexity -ngl
999 -fa off`, 5 paired runs (2 vanilla + 3 ESIMD):

| path | final ppl | per-chunk PPL stream sha256 |
|---|--:|--|
| Standard SYCL (vanilla) | **9.5668 ± 0.24125** | `cca7ea6e…` |
| **ESIMD opt-in** | **9.5668 ± 0.24125** | `cca7ea6e…` (identical) |

**Δppl = 0.0000** at the printed precision (4 decimals). All 50
per-chunk running PPLs are bit-identical across all 5 runs; the
SHA256 of the per-chunk PPL stream is the same on both paths.
ESIMD dispatch confirmed via `GGML_SYCL_DEBUG=1` (kernel actually
invoked, not silently bypassed).

**Caveat — perplexity underexercises the ESIMD path.** Perplexity is
dominated by batched mat-mul (`mul_mat`); the ESIMD kernel is mat-vec
(`mmvq`), invoked at the per-token decode tail. A token-generation
downstream eval (e.g. `llama-perplexity --multiple-choice` on
arc-easy/hellaswag) would put more pressure on the ESIMD code path
and is the next correctness check to run if your workload is
generation-quality-sensitive.

**Caveat — `-fa auto` triggers `UR_RESULT_ERROR_DEVICE_LOST` on
Xe-LPG** before the first chunk completes, on both vanilla and ESIMD.
Use `-fa off` for `llama-perplexity` runs. Unrelated to ESIMD; flag
your reproducer accordingly.

**Observed during development:** the iter-16 ESIMD kernel by itself
produces token-for-token identical output to the standard SYCL kernel
on the test prompt. Adding sibling ESIMD kernels (Q4_0, Q6_K) to the
same `libggml-sycl.so` shifts icpx's reg allocation / instruction
scheduling for the Q4_K kernel just enough to change reduction order,
which produces the divergence. The Q4_K kernel **source** is byte-
identical between the two builds; the **compiled binary** differs at
the FP-precision level.

**Verify on your own hardware** before using the ESIMD path in any
pipeline that cares about exact output. The runtime env-var gate makes
A/B comparison trivial. If your workload is generation-quality-sensitive
beyond what perplexity tests catch, stay on the standard SYCL path.

### Per-quant verification roster (2026-05-10, ongoing)

The Q4_K_M block above is the original verification. Other ESIMD-wired
quants are gated through `docs/development/perplexity-suite.sh`-style
paired runs, one quant per scheduled-task firing.

| quant | model | vanilla ppl | ESIMD ppl | Δ abs / % | per-chunk sha256 |
|---|---|--:|--:|--:|--|
| Q4_0 | dolphin3 8B (requantized from Q4_K_M) | **10.0751 ± 0.25623** | **10.0751 ± 0.25623** | 0.0000 / 0.000% | `e3d40908…` (identical) |
| Q5_K | TinyLlama 1.1B Q5_K_M | **18.6953 ± 0.56406** | **18.6953 ± 0.56406** | 0.0000 / 0.000% | `2d2f2781…` (identical) |
| Q6_K | TinyLlama 1.1B Q6_K | **18.5573 ± 0.55867** | **18.5573 ± 0.55867** | 0.0000 / 0.000% | `bdfc2d9f…` (identical) |
| Q8_0 | TinyLlama 1.1B (requantized from Q6_K) | **18.5527 ± 0.55849** | **18.5527 ± 0.55849** | 0.0000 / 0.000% | `7ebb8026…` (identical) |
| Q2_K | TinyLlama 1.1B (requantized from Q6_K) | **26.4039 ± 0.77512** | **26.4039 ± 0.77512** | 0.0000 / 0.000% | `8173d975…` (identical) |
| Q3_K | TinyLlama 1.1B (requantized from Q6_K) | **20.1407 ± 0.59797** | **20.1407 ± 0.59797** | 0.0000 / 0.000% | `45eca5e4…` (identical) |
| Q4_1 | TinyLlama 1.1B (requantized from Q6_K) | **19.3334 ± 0.57657** | **19.3334 ± 0.57657** | 0.0000 / 0.000% | `926274aa…` (identical) |
| Q5_0 | TinyLlama 1.1B (requantized from Q6_K) | **18.6669 ± 0.56012** | **18.6669 ± 0.56012** | 0.0000 / 0.000% | `13905582…` (identical) |
| Q5_1 | TinyLlama 1.1B (requantized from Q6_K) | **18.8097 ± 0.56610** | **18.8097 ± 0.56610** | 0.0000 / 0.000% | `0ff5961f…` (identical) |

Q4_0 verification (2026-05-10): 2 vanilla + 2 ESIMD runs on wikitext-2
test split, 50 chunks × 512 tokens = 25,600 tokens scored,
`llama-perplexity -ngl 999 -fa off`, dolphin3:latest 8B requantized to
Q4_0 with `--allow-requantize`. All 4 runs printed `Final estimate:
PPL = 10.0751 +/- 0.25623` and produced bit-identical per-chunk PPL
streams (sha256 `e3d40908f72b50b8d315875aeb626a52c3dfbed757c4dd50b0c5beed8e338a0b`).
Bench env had an idle gpt-oss-abliterated:20b llama-server resident in
GPU memory (compute-uncontended; perplexity is a correctness metric
and not affected by throughput contention). Same identical-stream
result as the Q4_K_M precedent — Q4_0 ESIMD path is bit-for-bit
equivalent to standard SYCL on this Q4_0 model.

Q5_K verification (2026-05-17): 2 vanilla + 2 ESIMD runs on wikitext-2
test split, 50 chunks × 512 tokens = 25,600 tokens scored,
`llama-perplexity -ngl 999 -fa off`, TinyLlama 1.1B Q5_K_M
(`tinyllama:1.1b-chat-v1-q5_K_M` ollama blob,
sha256 `2fdab35cfeff7068ff8df4c227be3bce6c1001397e0c69abfe8151124519dc96`).
Model swap to TinyLlama vs. the Q4_K/Q4_0 dolphin3 8B precedent: a
genuine Q5_K_M GGUF was already on disk, while requantizing Q4_K_M →
Q5_K_M would have layered Q4_K's loss under Q5_K's and reported a
misleading ppl. The kernel-equivalence signal (paired identical
streams) is model-independent; the absolute ppl is not comparable to
the Q4_0 / Q4_K rows. All 4 runs printed `Final estimate: PPL =
18.6953 +/- 0.56406` and produced bit-identical per-chunk PPL streams
(sha256 `2d2f2781c80db923c730cf9b5495104a72a477b19231d0665483ef356c32aef5`).
GPU uncontended at bench time (no concurrent llama/ollama workers).
Same identical-stream result as the Q4_K_M and Q4_0 precedents — Q5_K
ESIMD path is bit-for-bit equivalent to standard SYCL on this model.

Q6_K verification (2026-05-24): 2 vanilla + 2 ESIMD runs on wikitext-2
test split, 50 chunks × 512 tokens = 25,600 tokens scored,
`llama-perplexity -ngl 999 -fa off`, TinyLlama 1.1B Q6_K
(`tinyllama:1.1b-chat-v1-q6_K` ollama blob,
sha256 `4928c406d5e3299653d2170884113363aef4491d9b9d5c93ea73ec09f7e69495`).
Binary rebuilt at fork commit `7f16574` (dispatch fix + per-quant
opt-in) before the run, so the bench exercises the current source —
Q6_K was already reachable in default dispatch pre-7f16574 (it is in
the `ggml_sycl_supports_reorder_mmvq()` set), so the corrigendum's
disclaimers for the 6 previously-dead quants do not apply here. All
4 runs printed `Final estimate: PPL = 18.5573 +/- 0.55867` and
produced byte-identical per-chunk PPL value streams (sha256
`bdfc2d9f75c097db741ae081bd9f80cf0d6491f0cfddfa70647ae2cd6ddede58`
across all 4 runs, extracted with `grep -oE '\[[0-9]+\][0-9]+\.[0-9]+'`
to strip an interleaved `get_memory_info: ext_intel_free_memory is not
supported` stderr warning that broke a single line on one run — the
underlying chunk values are unaffected). Single ollama `ollama-lib
runner` resident in GPU memory at 0.5% CPU and 0% GPU memory
contention (perplexity is correctness-only; throughput contention
irrelevant). Same identical-stream result as the Q4_K_M / Q4_0 / Q5_K
precedents — Q6_K ESIMD path is bit-for-bit equivalent to standard
SYCL on this model.

Q8_0 verification (2026-06-14): 2 vanilla + 2 ESIMD runs on wikitext-2
test split, 50 chunks × 512 tokens = 25,600 tokens scored,
`llama-perplexity -ngl 999 -fa off`, TinyLlama 1.1B Q8_0. No genuine
Q8_0 GGUF was on disk, so the model was requantized from the TinyLlama
1.1B Q6_K ollama blob (sha256 `4928c406…`, the highest-precision
TinyLlama on disk) with `llama-quantize --allow-requantize … Q8_0`
(output sha256
`5acd4ee2cebdb7a2550b43d8873c272fb720f6ba3768f811483fc59843bd1f71`,
1114.91 MiB / 8.50 BPW). Q8_0 is wider than the Q6_K source, so this is
an up-quant: the Q6_K loss dominates and Q8_0 adds essentially none,
which is why the absolute ppl (18.5527) lands right beside the Q6_K row
(18.5573). The absolute number is therefore NOT comparable to the
Q4_0 / Q4_K dolphin3 rows — but the kernel-equivalence signal (paired
identical streams) is model-independent. ESIMD was isolated with
`GGML_SYCL_USE_ESIMD=q8_0` (per-quant opt-in, not the all-on `=1`), so
only the Q8_0 kernel was under test; a `GGML_SYCL_DEBUG=1` pre-check
confirmed `reorder_mul_mat_vec_q8_0_q8_1_sycl_esimd` was actually
invoked (4 calls/chunk) and no other ESIMD kernel fired. All weight
tensors have ne00 ∈ {2048, 5632}, both multiples of QK8_0·8 = 256, so
the ESIMD block-count guard (`ne00 % (QK8_0*8) == 0`) admits every
tensor. All 4 runs printed `Final estimate: PPL = 18.5527 +/- 0.55849`
and produced byte-identical per-chunk PPL value streams (sha256
`7ebb8026278e440a7498fc2163aa71c00ae914244164ef558670e85e637581a4`
across all 4 runs). Bench env had an idle gpt-oss-abliterated:20b
llama-server plus an idle piranesi-msabl ollama runner resident; GPU
compute was idle at run start (intel_gpu_top RC6 100%, all engines
0.00%), and perplexity is a correctness metric unaffected by throughput
contention. Same identical-stream result as the Q4_K_M / Q4_0 / Q5_K /
Q6_K precedents — Q8_0 ESIMD path is bit-for-bit equivalent to standard
SYCL on this model. Q8_0 completes the original 5-quant core roster
(Q4_K, Q4_0, Q5_K, Q6_K, Q8_0); the extension quants (Q2_K, Q3_K,
Q4_1, Q5_0, Q5_1) remain as follow-ups.

Q2_K / Q3_K / Q4_1 / Q5_0 / Q5_1 verification (2026-06-14): the five
extension quants, verified in one batch — 2 vanilla + 2 ESIMD runs each
on wikitext-2 test split, 50 chunks × 512 = 25,600 tokens scored,
`llama-perplexity -ngl 999 -fa off`. Each model was requantized from the
TinyLlama 1.1B Q6_K ollama blob (`4928c406…`) with
`llama-quantize --allow-requantize`; absolute ppl is loss-stacked (the
Q6_K floor plus the target quant's own loss) and is NOT cross-comparable
between rows, but the kernel-equivalence signal is model-independent.
Each kernel was isolated with `GGML_SYCL_USE_ESIMD=<quant>` and a
`GGML_SYCL_DEBUG=1` pre-check confirmed dispatch before the paired runs.

**These five differ from the reorder-quant rows (Q4_0 / Q8_0 / Q4_K /
Q6_K) in a way that makes them a *stronger* correctness probe.** Their
default (env-unset) mat-vec path is DMMV (dequantize-then-FP16 mat-vec),
whereas enabling ESIMD reroutes the mat-vec to the int8
`mul_mat_vec_<quant>_q8_1_sycl_esimd` kernel via the
`ggml_sycl_esimd_preempts_dmmv()` dispatch hook. So this gate compares
two **different numerical algorithms** (DMMV vs int8 mmvq), not two
builds of the same kernel as the reorder rows do. The ESIMD kernel was
also heavily exercised — 174 mat-vec calls/chunk for the K-quants
(Q2_K, Q3_K) and 305/chunk for the legacy quants (Q4_1, Q5_0, Q5_1),
versus ~4/chunk for the reorder quants. Despite the algorithm difference
and the heavy exercise, all four runs per quant produced byte-identical
per-chunk PPL value streams and an identical final estimate; Δppl =
0.0000 / 0.000% for every quant:

| quant | ppl (all 4 runs) | model sha256 | per-chunk stream sha256 |
|---|--:|--|--|
| Q2_K | 26.4039 ± 0.77512 | `6f197db0234e1ef7…` | `8173d975d10f2e59fe5a5f3591c44afca037fdd6a57b070875ac0d3602574255` |
| Q3_K | 20.1407 ± 0.59797 | `20aefffacce57ac2…` | `45eca5e42cf9f3c67ccce1a126d7b6bc9cb0368f7e853afe47ff9057a823b1a8` |
| Q4_1 | 19.3334 ± 0.57657 | `2009be6ca2fb6e8c…` | `926274aaf2d069aa506a821ee21711d0d8871a9a5116d4b3ec8cfdf852cad4c8` |
| Q5_0 | 18.6669 ± 0.56012 | `958d8585a9c2d650…` | `139055821d433d4f0ea79c84c1f80fbc9a661a1d347a98fd7793a99375cbff09` |
| Q5_1 | 18.8097 ± 0.56610 | `e82436d688ec46b8…` | `0ff5961f3de26f86e1cfe02941989100fa3fe33c7060e6fc23d475d715fa8fde` |

GPU compute was idle at run start (intel_gpu_top RC6 100%, all engines
0.00%); the five quants ran sequentially with no cross-quant GPU
overlap. Same identical-stream result as the Q4_K_M / Q4_0 / Q5_K /
Q6_K / Q8_0 precedents.

**Roster complete (2026-06-14).** All 10 ESIMD-wired quants — Q4_K,
Q4_0, Q5_K, Q6_K, Q8_0, Q2_K, Q3_K, Q4_1, Q5_0, Q5_1 — now have
perplexity rows and are bit-for-bit equivalent to their respective
standard SYCL paths on Xe-LPG. This is a **correctness** result only:
all 10 ESIMD kernels remain slower than vanilla on this hardware (0.32×–
0.90×; see the perf table above). The env-var gate makes A/B trivial,
but there is no throughput reason to enable ESIMD on Xe-LPG today.

## Scope

- **All five quants in the test roster:** Q4_K (foundation), Q4_0,
  Q5_K, Q6_K, Q8_0. Other formats (Q2_K, Q3_K, Q4_1, Q5_0, Q5_1, etc.)
  are tracked as follow-ups.
- **Intel iGPU/dGPU only.** ESIMD is an Intel-specific extension; the
  build flag has no effect on AMD/NVIDIA SYCL targets.
- **Foundation, not finished.** Documented dead ends + tried-and-rejected
  directions are in `SYCL_ESIMD_LOG.md` so the next person doesn't
  re-discover them.

## Engineering notes

Full iteration log: [`SYCL_ESIMD_LOG.md`](SYCL_ESIMD_LOG.md).
SPIR-V disasm + IPEX symbol study: [`SYCL_ESIMD_RESEARCH.md`](SYCL_ESIMD_RESEARCH.md).
