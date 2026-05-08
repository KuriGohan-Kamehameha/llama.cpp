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

| backend | pp1024 (t/s) | tg128 (t/s) | tg vs default |
|---|--:|--:|--:|
| Standard SYCL (default `reorder_mul_mat_vec_q4_k_q8_1_sycl`) | 477 | 11.14 | 1.00× |
| **ESIMD opt-in** (this branch, `GGML_SYCL_USE_ESIMD=1`) | 477 | **6.41** | 0.58× |
| IPEX-LLM bundled (`intelanalytics/ipex-llm-inference-cpp-xpu` container, proprietary) | 497 | 17.6 | 1.58× |

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

## Math correctness

The ESIMD kernel produces token-for-token identical output to the
standard SYCL kernel on 30+ token completions across multiple seeds and
prompts, at temperature 0. Verified by isolated `llama-cli` diff with
the IPEX-LLM container paused (so the standard kernel has the iGPU
exclusively).

Don't trust this — verify on your own hardware before using ESIMD path
in any pipeline that cares about output. The runtime env-var gate makes
A/B comparison trivial.

## Scope

- **Q4_K only** at present. Q4_0/Q5_K/Q6_K/Q8_0 are tracked as follow-ups.
- **Intel iGPU/dGPU only.** ESIMD is an Intel-specific extension; the
  build flag has no effect on AMD/NVIDIA SYCL targets.
- **Foundation, not finished.** Documented dead ends + tried-and-rejected
  directions are in `SYCL_ESIMD_LOG.md` so the next person doesn't
  re-discover them.

## Engineering notes

Full iteration log: [`SYCL_ESIMD_LOG.md`](SYCL_ESIMD_LOG.md).
SPIR-V disasm + IPEX symbol study: [`SYCL_ESIMD_RESEARCH.md`](SYCL_ESIMD_RESEARCH.md).
