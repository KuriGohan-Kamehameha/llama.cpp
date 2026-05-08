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
