// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Sat Bajaj. Released under the MIT license; see LICENSE.
//
// ESIMD (Explicit SIMD) Q4_K mat-vec kernel for the SYCL backend.
//
// This file provides an alternative to `reorder_mul_mat_vec_q4_k_q8_1_sycl`
// implemented using Intel ESIMD instead of standard SYCL. ESIMD lets the
// kernel state SIMD width explicitly per instruction (each thread is a SIMD
// unit), bypassing icpx's auto-vectorizer. On Intel iGPUs this opens a code
// path that the standard SYCL backend cannot reach by definition.
//
// Status: foundation; opt-in via `GGML_SYCL_ESIMD=ON` build flag and
// `GGML_SYCL_USE_ESIMD=1` runtime env var. Default OFF; standard-SYCL path
// is unaffected when this kernel is not compiled or not selected at runtime.
//
// Performance reference (Arc Xe-LPG, Arrow Lake H, GPU 0x7d51, llama-bench
// dolphin3 8B Q4_K_M, ngl=999):
//   - Vanilla `reorder_mul_mat_vec_q4_k_q8_1_sycl`: 11.85 tg128 t/s
//   - This kernel: 6.35 tg128 t/s
//   - IPEX-LLM bundled (proprietary): 17.6 tg128 t/s
//
// Math correctness: bit-identical to the standard-SYCL kernel on
// 30+ token completions across multiple seeds and prompts (verified against
// the same upstream `reorder_mul_mat_vec_q4_k_q8_1_sycl` running on the
// same hardware).
//
// This is intentionally landed below vanilla perf — the goal is to add the
// ESIMD code path to ggml-sycl. Subsequent work (per-arch DPAS, prefetch,
// other quants) is tracked as follow-ups. See the PR description for a
// breakdown of which architectural choices were tried and which paid off.

#ifdef GGML_SYCL_ESIMD

#include "mmvq.hpp"
#include "ggml.h"
#include "common.hpp"
#include "quants.hpp"

#include <sycl/ext/intel/esimd.hpp>

namespace esimd = sycl::ext::intel::esimd;

// Number of output rows handled per thread. Activation loads are shared
// across the rows owned by the same thread; this is the largest single
// perf lever in this kernel.
//
// Sweep on Arc Xe-LPG ARL-H showed ROWS=4 is the local maximum for tg128
// (ROWS=1: 4.14, ROWS=2: 5.18, ROWS=4: 5.55, ROWS=8: 4.30 due to register
// pressure). Other Intel iGPU/dGPU classes may have different optima;
// this is a tuning knob.
constexpr int ROWS_PER_THREAD = 4;

template <int VEC_W>
static void esimd_q4_k_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float * __restrict__ dst,
        const int ncols, const int nrows,
        const sycl::nd_item<1> & nd_item) SYCL_ESIMD_KERNEL {

    const int thread_id = nd_item.get_global_id(0);
    const int row_base  = thread_id * ROWS_PER_THREAD;
    if (row_base >= nrows) return;

    const int n_sblocks = ncols / QK_K;

    // Region pointers per the existing reorder layout
    // (ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_K>):
    //   [all blocks' qs (128 bytes each)]
    //   [all scales (12 bytes each)]
    //   [all dm (4 bytes each)]
    const int total_blocks = nrows * n_sblocks;
    const auto * qs_base     = static_cast<const uint8_t *>(vx);
    const auto * scales_base = qs_base + total_blocks * (QK_K / 2);
    const auto * dm_base     = scales_base + total_blocks * K_SCALE_SIZE;

    // q8_1 activation y: [ncols int8 quants] [(ncols/QK8_1) sycl::half2 ds]
    const auto * y_qs = static_cast<const int8_t *>(vy);
    const auto * y_ds = reinterpret_cast<const sycl::half2 *>(
        static_cast<const uint8_t *>(vy) + ncols);

    float acc[ROWS_PER_THREAD] = { 0.0f };

    for (int sb = 0; sb < n_sblocks; ++sb) {
        // Activation loads — shared across all rows in this thread.
        esimd::simd<int8_t, 256> y_q_shared;
        y_q_shared.copy_from(y_qs + sb * QK_K);

        constexpr int N_Y_DS = QK_K / QK8_1;
        const uint16_t * y_ds_u16 =
            reinterpret_cast<const uint16_t *>(y_ds + sb * N_Y_DS);

        // Per-sub-block dy and sy = (d, s) of q8_1 ds half2 (sy holds
        // d * sum(y_quants), saving the per-sub-block sum-of-y reduce).
        esimd::simd<float, 8> dy_v, sy_v;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            dy_v[i] = static_cast<float>(sycl::bit_cast<sycl::half>(y_ds_u16[i * 2]));
            sy_v[i] = static_cast<float>(sycl::bit_cast<sycl::half>(y_ds_u16[i * 2 + 1]));
        }

        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int row = row_base + r;
            if (row >= nrows) break;
            const int block_idx = row * n_sblocks + sb;

            // Per-row weight loads.
            esimd::simd<uint8_t, 128> qs;
            qs.copy_from(qs_base + block_idx * (QK_K / 2));

            esimd::simd<uint8_t, 12> scales;
            scales.copy_from(scales_base + block_idx * K_SCALE_SIZE);

            // dm via raw uint16 + bit_cast (ESIMD doesn't allow taking the
            // address of a simd_view, and sycl::vec<half,2>::operator[] is
            // not supported in ESIMD context).
            const uint16_t * dm_u16 = reinterpret_cast<const uint16_t *>(
                dm_base + block_idx * sizeof(sycl::half2));
            const float d_f    = static_cast<float>(sycl::bit_cast<sycl::half>(dm_u16[0]));
            const float dmin_f = static_cast<float>(sycl::bit_cast<sycl::half>(dm_u16[1]));

            // Q4_K packed scales[12]: 8 sub-block (6-bit scale, 6-bit min) pairs
            // packed across 12 bytes per the standard layout. Sub-blocks 0..3 are
            // direct-encoded (6 bits in scales[i]); sub-blocks 4..7 take their low
            // 4 bits from scales[i+8] and their high 2 bits from scales[i+0]&0xC0.
            uint8_t s[12];
            #pragma unroll
            for (int k = 0; k < 12; ++k) s[k] = scales[k];

            esimd::simd<float, 8> sub_scales_v;
            esimd::simd<float, 8> sub_mins_v;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sub_scales_v[i]     = static_cast<float>(s[i] & 0x3F);
                sub_mins_v  [i]     = static_cast<float>(s[i + 4] & 0x3F);
                sub_scales_v[i + 4] = static_cast<float>((s[i + 8] & 0x0F) | ((s[i + 0] & 0xC0) >> 2));
                sub_mins_v  [i + 4] = static_cast<float>((s[i + 8] >> 4)   | ((s[i + 4] & 0xC0) >> 2));
            }

            esimd::simd<float, 8> dot_v;

            // Pack low+high nibbles of each pair into simd<int8,64> for one
            // wide multiply per pair, and matching activation slice. 4 iters
            // per super-block (one per pair of sub-blocks).
            #pragma unroll
            for (int sub = 0; sub < 8; sub += 2) {
                const int qs_offset = (sub / 2) * 32;
                esimd::simd<uint8_t, 32> qs_pair = qs.template select<32, 1>(qs_offset);
                esimd::simd<int8_t, 32> w_lo = qs_pair & uint8_t(0x0F);
                esimd::simd<int8_t, 32> w_hi = (qs_pair >> 4) & uint8_t(0x0F);

                esimd::simd<int8_t, 64> w_pair;
                w_pair.template select<32, 1>(0)  = w_lo;
                w_pair.template select<32, 1>(32) = w_hi;

                esimd::simd<int8_t, 64> y_pair;
                y_pair.template select<32, 1>(0)  = y_q_shared.template select<32, 1>(sub * 32);
                y_pair.template select<32, 1>(32) = y_q_shared.template select<32, 1>((sub + 1) * 32);

                esimd::simd<int16_t, 64> prod_pair = w_pair * y_pair;

                // Materialize the two halves as simd<> (esimd::reduce does
                // not accept simd_view).
                esimd::simd<int16_t, 32> prod_lo_s = prod_pair.template select<32, 1>(0);
                esimd::simd<int16_t, 32> prod_hi_s = prod_pair.template select<32, 1>(32);

                dot_v[sub]     = static_cast<float>(esimd::reduce<int>(prod_lo_s, std::plus<>{}));
                dot_v[sub + 1] = static_cast<float>(esimd::reduce<int>(prod_hi_s, std::plus<>{}));
            }

            // sumf_d term: dot * sub_scale * dy
            // sumf_m term: sub_min * sy   (q8_1's sy already absorbs dy)
            const esimd::simd<float, 8> contrib_d = dot_v * sub_scales_v * dy_v;
            const esimd::simd<float, 8> contrib_m = sy_v  * sub_mins_v;
            acc[r] += d_f    * esimd::reduce<float>(contrib_d, std::plus<>{})
                    - dmin_f * esimd::reduce<float>(contrib_m, std::plus<>{});
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = row_base + r;
        if (row < nrows) {
            dst[row] = acc[r];
        }
    }
}

void reorder_mul_mat_vec_q4_k_q8_1_sycl_esimd(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr int WG_SIZE = 32;
    const int n_threads = (nrows + ROWS_PER_THREAD - 1) / ROWS_PER_THREAD;
    const sycl::range<1> global_size((n_threads + WG_SIZE - 1) / WG_SIZE * WG_SIZE);
    const sycl::range<1> workgroup_size(WG_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(global_size, workgroup_size),
            [=](sycl::nd_item<1> nd_item) SYCL_ESIMD_KERNEL {
                esimd_q4_k_kernel<32>(vx, vy, dst, ncols, nrows, nd_item);
            });
    });
}

#endif  // GGML_SYCL_ESIMD
