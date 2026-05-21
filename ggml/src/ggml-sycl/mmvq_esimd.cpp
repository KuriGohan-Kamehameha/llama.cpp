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
//
// IPEX-LLM Q4_K kernel SPIR-V comparison (2026-05-08) shows the perf
// gap is NOT a missing non-public intrinsic. Both kernels use only
// `rdregion`/`wrregion` genx ops (no DPAS, no LSC hints, no SLM, no
// subgroup ops, sub_group_size=1 same as ours). IPEX's win is
// algorithmic: FP-domain inner loop (convert weights to FP32, fmul,
// FP16 accumulate) instead of our INT8 mul -> INT16 -> reduce -> FP
// scale. See SYCL_ESIMD_LOG.md "iter 19" for the FP-domain rewrite plan.
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

// Phase C addition — Q4_0 ESIMD kernel sibling to the Q4_K iter-16 kernel.
//
// Q4_0 block layout (per ggml-common.h):
//   typedef struct { ggml_half d; uint8_t qs[QK4_0/2]; } block_q4_0;
//
// Reorder layout (per ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_0>):
//   [all qs (16 bytes/block, ncols/QK4_0 blocks/row)]
//   [all d  (sizeof(ggml_half) per block)]
//
// Per-block math (matching reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>):
//   sumi = dot(int4 low-nibbles, q8_1[0..15]) + dot(int4 high-nibbles, q8_1[16..31])
//   block_contrib = d * (sumi * ds_d - 8 * ds_s)
// where (ds_d, ds_s) are the half2 (d, d*sum_y) of the matching q8_1 sub-block.
//
// We process 8 Q4_0 blocks at a time per super-iteration so we get a 256-byte
// shared activation load (same trick as Q4_K's QK_K=256 super-block iter).

template <int VEC_W>
static void esimd_q4_0_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float * __restrict__ dst,
        const int ncols, const int nrows,
        const sycl::nd_item<1> & nd_item) SYCL_ESIMD_KERNEL {

    const int thread_id = nd_item.get_global_id(0);
    const int row_base  = thread_id * ROWS_PER_THREAD;
    if (row_base >= nrows) return;

    constexpr int Q40_BLOCKS_PER_GROUP = 8;     // 8 blocks * 32 weights = 256
    constexpr int GROUP_QS_BYTES       = Q40_BLOCKS_PER_GROUP * (QK4_0 / 2);  // 128
    constexpr int GROUP_Y_QS_BYTES     = Q40_BLOCKS_PER_GROUP * QK4_0;        // 256

    const int n_blocks_per_row = ncols / QK4_0;
    const int n_groups         = n_blocks_per_row / Q40_BLOCKS_PER_GROUP;
    const int total_blocks     = nrows * n_blocks_per_row;

    const auto * qs_base = static_cast<const uint8_t *>(vx);
    const auto * d_base  = qs_base + total_blocks * (QK4_0 / 2);

    // q8_1 activation y: [ncols int8 quants] [(ncols/QK8_1) sycl::half2 ds]
    const auto * y_qs = static_cast<const int8_t *>(vy);
    const auto * y_ds_u8 = static_cast<const uint8_t *>(vy) + ncols;

    float acc[ROWS_PER_THREAD] = { 0.0f };

    for (int g = 0; g < n_groups; ++g) {
        // Activation load — shared across all rows in this thread.
        esimd::simd<int8_t, 256> y_q_shared;
        y_q_shared.copy_from(y_qs + g * GROUP_Y_QS_BYTES);

        // q8_1 (d, s) per sub-block, 8 sub-blocks per group.
        const uint16_t * y_ds_u16 = reinterpret_cast<const uint16_t *>(
            y_ds_u8 + g * Q40_BLOCKS_PER_GROUP * sizeof(sycl::half2));
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
            const int block0 = row * n_blocks_per_row + g * Q40_BLOCKS_PER_GROUP;

            // 8 contiguous Q4_0 blocks of qs = 128 bytes.
            esimd::simd<uint8_t, 128> qs;
            qs.copy_from(qs_base + block0 * (QK4_0 / 2));

            // Per-block d (half) — 8 of them.
            const uint16_t * d_u16 = reinterpret_cast<const uint16_t *>(
                d_base + block0 * sizeof(ggml_half));
            esimd::simd<float, 8> d_v;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                d_v[i] = static_cast<float>(sycl::bit_cast<sycl::half>(d_u16[i]));
            }

            // sumi per block: int4 nibble dot int8 activations.
            // Pack two adjacent blocks at once into simd<int8,64> (32 weights
            // each) — 4 iterations cover all 8 blocks in this group.
            esimd::simd<float, 8> sumi_v;
            #pragma unroll
            for (int sub = 0; sub < 8; sub += 2) {
                // 32 bytes of qs cover 2 blocks (16 bytes/block).
                esimd::simd<uint8_t, 32> qs_pair = qs.template select<32, 1>(sub * 16);

                // For block in {sub, sub+1}: 16 low-nibble weights + 16 high-nibble
                // weights, i.e. byte k of that block's qs gives weight[k] (low) and
                // weight[k+16] (high). The natural q8_1 layout: y_q_shared bytes
                // [sub*32 .. (sub+1)*32) for block `sub`.
                esimd::simd<int8_t, 16> w0_lo = qs_pair.template select<16, 1>(0)  & uint8_t(0x0F);
                esimd::simd<int8_t, 16> w0_hi = (qs_pair.template select<16, 1>(0)  >> 4) & uint8_t(0x0F);
                esimd::simd<int8_t, 16> w1_lo = qs_pair.template select<16, 1>(16) & uint8_t(0x0F);
                esimd::simd<int8_t, 16> w1_hi = (qs_pair.template select<16, 1>(16) >> 4) & uint8_t(0x0F);

                esimd::simd<int8_t, 64> w_pair;
                w_pair.template select<16, 1>(0)  = w0_lo;
                w_pair.template select<16, 1>(16) = w0_hi;
                w_pair.template select<16, 1>(32) = w1_lo;
                w_pair.template select<16, 1>(48) = w1_hi;

                esimd::simd<int8_t, 64> y_pair;
                y_pair.template select<32, 1>(0)  = y_q_shared.template select<32, 1>(sub       * 32);
                y_pair.template select<32, 1>(32) = y_q_shared.template select<32, 1>((sub + 1) * 32);

                esimd::simd<int16_t, 64> prod_pair = w_pair * y_pair;

                // Reduce two halves separately: each half = 32 int16 products
                // covering one whole Q4_0 block.
                esimd::simd<int16_t, 32> prod_b0 = prod_pair.template select<32, 1>(0);
                esimd::simd<int16_t, 32> prod_b1 = prod_pair.template select<32, 1>(32);

                sumi_v[sub]     = static_cast<float>(esimd::reduce<int>(prod_b0, std::plus<>{}));
                sumi_v[sub + 1] = static_cast<float>(esimd::reduce<int>(prod_b1, std::plus<>{}));
            }

            // Per-block contrib: d * (sumi * dy - 8 * sy)
            const esimd::simd<float, 8> contrib = d_v * (sumi_v * dy_v - 8.0f * sy_v);
            acc[r] += esimd::reduce<float>(contrib, std::plus<>{});
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

void reorder_mul_mat_vec_q4_0_q8_1_sycl_esimd(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % (QK4_0 * 8) == 0);  // need multiple of 8 blocks per row

    constexpr int WG_SIZE = 32;
    const int n_threads = (nrows + ROWS_PER_THREAD - 1) / ROWS_PER_THREAD;
    const sycl::range<1> global_size((n_threads + WG_SIZE - 1) / WG_SIZE * WG_SIZE);
    const sycl::range<1> workgroup_size(WG_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(global_size, workgroup_size),
            [=](sycl::nd_item<1> nd_item) SYCL_ESIMD_KERNEL {
                esimd_q4_0_kernel<32>(vx, vy, dst, ncols, nrows, nd_item);
            });
    });
}

// Phase C addition — Q6_K ESIMD kernel sibling to the Q4_K iter-16 kernel.
//
// Q6_K block layout (per ggml-common.h):
//   typedef struct {
//       uint8_t ql[QK_K/2];      // 128 bytes — quants, lower 4 bits per weight
//       uint8_t qh[QK_K/4];      // 64  bytes — quants, upper 2 bits per weight
//       int8_t  scales[QK_K/16]; // 16 bytes — sub-block scales (signed int8)
//       ggml_half d;             //  2 bytes — global delta
//   } block_q6_K;
//
// Reorder layout (per ggml_sycl_reordered::block_q_t<GGML_TYPE_Q6_K>):
//   [all blocks' ql       (128 bytes/block)]
//   [all blocks' qh       ( 64 bytes/block)]
//   [all blocks' scales   ( 16 bytes/block)]
//   [all blocks' d        (sizeof(ggml_half) per block)]
//
// Math (matches dequantize_row_q6_K + reorder_vec_dot_q_sycl<Q6_K>):
//   each super-block has 16 sub-blocks of 16 weights each (signed scales[16]);
//   per-weight unsigned 6-bit = (ql_nibble | (qh_2bits << 4)); signed weight
//   = unsigned_6bit - 32. q8_1 covers 32 weights per sub-block (8 q8_1 subs
//   per super-block). Q6_K has no min term — symmetric quant.
//
//   per super-block per row:
//     for each q8_1 sub-block s in [0..8):
//       sumi_first16  = sum_{k=0..15}  (w_k - 32) * y_qs[s*32 + k]
//       sumi_second16 = sum_{k=16..31} (w_k - 32) * y_qs[s*32 + k]
//       contrib[s]    = (scales[2*s] * sumi_first16 + scales[2*s+1] * sumi_second16) * d8[s]
//     dst[row] += d * sum_{s=0..8} contrib[s]
//
// Within each "half" of 128 weights (super-block weights 0..127 or 128..255):
//   half consumes ql bytes [base+0 .. base+63] and qh bytes [base+0 .. base+31].
//   For l = 0..31:
//     w[l + 0]  = (ql[l + 0]  & 0x0F) | (((qh[l] >> 0) & 0x03) << 4)
//     w[l + 32] = (ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4)
//     w[l + 64] = (ql[l + 0]  >> 4)   | (((qh[l] >> 4) & 0x03) << 4)
//     w[l + 96] = (ql[l + 32] >> 4)   | (((qh[l] >> 6) & 0x03) << 4)
//   These 4 chunks of 32 weights map 1:1 to 4 consecutive q8_1 sub-blocks.

template <int VEC_W>
static void esimd_q6_k_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float * __restrict__ dst,
        const int ncols, const int nrows,
        const sycl::nd_item<1> & nd_item) SYCL_ESIMD_KERNEL {

    const int thread_id = nd_item.get_global_id(0);
    const int row_base  = thread_id * ROWS_PER_THREAD;
    if (row_base >= nrows) return;

    const int n_sblocks    = ncols / QK_K;
    const int total_blocks = nrows * n_sblocks;

    // Reorder region pointers: ql | qh | scales | d
    const auto * ql_base     = static_cast<const uint8_t *>(vx);
    const auto * qh_base     = ql_base     + total_blocks * (QK_K / 2);
    const auto * scales_base = qh_base     + total_blocks * (QK_K / 4);
    const auto * d_base      = scales_base + total_blocks * (QK_K / 16);

    // q8_1 activation y: [ncols int8 quants] [(ncols/QK8_1) sycl::half2 ds]
    const auto * y_qs = static_cast<const int8_t *>(vy);
    const auto * y_ds = reinterpret_cast<const sycl::half2 *>(
        static_cast<const uint8_t *>(vy) + ncols);

    float acc[ROWS_PER_THREAD] = { 0.0f };

    for (int sb = 0; sb < n_sblocks; ++sb) {
        // Activation load — shared across all rows in this thread.
        esimd::simd<int8_t, 256> y_q_shared;
        y_q_shared.copy_from(y_qs + sb * QK_K);

        constexpr int N_Y_DS = QK_K / QK8_1;  // 8
        const uint16_t * y_ds_u16 =
            reinterpret_cast<const uint16_t *>(y_ds + sb * N_Y_DS);

        // Per-q8_1-sub-block dy. Q6_K doesn't use sy (no min term).
        esimd::simd<float, 8> dy_v;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            dy_v[i] = static_cast<float>(sycl::bit_cast<sycl::half>(y_ds_u16[i * 2]));
        }

        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int row = row_base + r;
            if (row >= nrows) break;
            const int block_idx = row * n_sblocks + sb;

            // Per-row weight loads.
            esimd::simd<uint8_t, 128> ql_vec;
            ql_vec.copy_from(ql_base + block_idx * (QK_K / 2));

            esimd::simd<uint8_t, 64> qh_vec;
            qh_vec.copy_from(qh_base + block_idx * (QK_K / 4));

            // 16 signed int8 sub-scales.
            esimd::simd<int8_t, 16> sc_vec;
            sc_vec.copy_from(reinterpret_cast<const int8_t *>(
                scales_base + block_idx * (QK_K / 16)));

            // Convert to float once for the final fmadd.
            esimd::simd<float, 16> sc_f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                sc_f[i] = static_cast<float>(sc_vec[i]);  // signed int8 -> float
            }

            // d (super-block delta) via raw uint16 + bit_cast.
            const uint16_t * d_u16 = reinterpret_cast<const uint16_t *>(
                d_base + block_idx * sizeof(ggml_half));
            const float d_f = static_cast<float>(sycl::bit_cast<sycl::half>(*d_u16));

            // Per q8_1 sub-block sumi values (sumi_first16, sumi_second16).
            esimd::simd<float, 8> sumi_first_v;
            esimd::simd<float, 8> sumi_second_v;

            // Process each of 2 halves (128 weights each).
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                // ql slice: bytes [h*64 .. h*64+63] (64 bytes).
                // qh slice: bytes [h*32 .. h*32+31] (32 bytes).
                esimd::simd<uint8_t, 32> ql0 = ql_vec.template select<32, 1>(h * 64 + 0);
                esimd::simd<uint8_t, 32> ql1 = ql_vec.template select<32, 1>(h * 64 + 32);
                esimd::simd<uint8_t, 32> qh0 = qh_vec.template select<32, 1>(h * 32);

                // Build the four 32-weight chunks; subtract 32 to get signed.
                // Chunk A: w[l +  0] = (ql0 & 0x0F) | ((qh0 >> 0) & 0x03) << 4
                // Chunk B: w[l + 32] = (ql1 & 0x0F) | ((qh0 >> 2) & 0x03) << 4
                // Chunk C: w[l + 64] = (ql0 >> 4)   | ((qh0 >> 4) & 0x03) << 4
                // Chunk D: w[l + 96] = (ql1 >> 4)   | ((qh0 >> 6) & 0x03) << 4
                esimd::simd<int8_t, 32> wA = (ql0 & uint8_t(0x0F)) | ((qh0 << 4) & uint8_t(0x30));
                esimd::simd<int8_t, 32> wB = (ql1 & uint8_t(0x0F)) | ((qh0 << 2) & uint8_t(0x30));
                esimd::simd<int8_t, 32> wC = (ql0 >> 4)            | ((qh0     ) & uint8_t(0x30));
                esimd::simd<int8_t, 32> wD = (ql1 >> 4)            | ((qh0 >> 2) & uint8_t(0x30));

                wA = wA - int8_t(32);
                wB = wB - int8_t(32);
                wC = wC - int8_t(32);
                wD = wD - int8_t(32);

                // Each chunk maps to a q8_1 sub-block with 32 activation int8s.
                // half h spans q8_1 sub-blocks [h*4 .. h*4 + 4).
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int q8_1_idx = h * 4 + q;
                    esimd::simd<int8_t, 32> w_chunk;
                    if      (q == 0) w_chunk = wA;
                    else if (q == 1) w_chunk = wB;
                    else if (q == 2) w_chunk = wC;
                    else             w_chunk = wD;

                    esimd::simd<int8_t, 32> y_chunk =
                        y_q_shared.template select<32, 1>(q8_1_idx * 32);

                    esimd::simd<int16_t, 32> prod = w_chunk * y_chunk;

                    // Split into first-16 / second-16 for the two sub-block scales.
                    esimd::simd<int16_t, 16> prod_a = prod.template select<16, 1>(0);
                    esimd::simd<int16_t, 16> prod_b = prod.template select<16, 1>(16);

                    sumi_first_v [q8_1_idx] =
                        static_cast<float>(esimd::reduce<int>(prod_a, std::plus<>{}));
                    sumi_second_v[q8_1_idx] =
                        static_cast<float>(esimd::reduce<int>(prod_b, std::plus<>{}));
                }
            }

            // Pack scales: scales[2*s] for first-16, scales[2*s+1] for second-16.
            esimd::simd<float, 8> sc_first_v;
            esimd::simd<float, 8> sc_second_v;
            #pragma unroll
            for (int s = 0; s < 8; ++s) {
                sc_first_v [s] = sc_f[2 * s];
                sc_second_v[s] = sc_f[2 * s + 1];
            }

            // contrib[s] = (sc_first[s]*sumi_first[s] + sc_second[s]*sumi_second[s]) * dy[s]
            const esimd::simd<float, 8> contrib =
                (sc_first_v * sumi_first_v + sc_second_v * sumi_second_v) * dy_v;
            acc[r] += d_f * esimd::reduce<float>(contrib, std::plus<>{});
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

void reorder_mul_mat_vec_q6_k_q8_1_sycl_esimd(
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
                esimd_q6_k_kernel<32>(vx, vy, dst, ncols, nrows, nd_item);
            });
    });
}

// Phase C addition - Q5_K ESIMD kernel sibling to the Q4_K iter-16 kernel.
//
// Q5_K block layout (per ggml-common.h):
//   typedef struct {
//       ggml_half2 dm;                 // 4  bytes - super-block (d, dmin)
//       uint8_t   scales[K_SCALE_SIZE];// 12 bytes - 8 packed (sub_scale, sub_min) pairs (same as Q4_K)
//       uint8_t   qh[QK_K/8];          // 32 bytes - high bit per weight
//       uint8_t   qs[QK_K/2];          // 128 bytes - low 4 bits per weight
//   } block_q5_K;                       // 176 bytes total
//
// Q5_K has no reorder layout in the ggml-sycl tree (unlike Q4_K/Q6_K/Q4_0/Q8_0),
// so this kernel reads directly from the standard struct-of-arrays layout: each
// row is `n_sblocks` contiguous block_q5_K structs at vx + row*n_sblocks*176.
//
// Math (matches dequantize_row_q5_K + vec_dot_q5_K_q8_1_impl_vmmq):
//   per super-block, 8 sub-blocks of 32 weights each. sub-scales/sub-mins are
//   the same packed[12] encoding as Q4_K. Each weight w_l in sub `s` is
//   unsigned 5-bit:
//     low 4 bits  = qs[(s/2)*32 + l] >> ((s & 1) * 4)) & 0x0F
//     high  1 bit = (qh[l] >> s) & 1, value = 16 if set
//   per super-block per row:
//     sumf_d = sum_{s=0..8} sub_scale[s] * dy[s] * dot(w_s, y_qs_s)
//     sumf_m = sum_{s=0..8} sub_min[s]   * sy[s]   (q8_1 sy already absorbs dy)
//     dst[row] += d * sumf_d - dmin * sumf_m
//
// Kernel architecture mirrors Q4_K iter 16:
//   ROWS_PER_THREAD=4, WG=32, single-super-block per outer iter, shared
//   activation load across rows, and packing low+high nibbles into
//   simd<int8,64> for one wide multiply per sub-block pair.

template <int VEC_W>
static void esimd_q5_k_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float * __restrict__ dst,
        const int ncols, const int nrows,
        const sycl::nd_item<1> & nd_item) SYCL_ESIMD_KERNEL {

    const int thread_id = nd_item.get_global_id(0);
    const int row_base  = thread_id * ROWS_PER_THREAD;
    if (row_base >= nrows) return;

    const int n_sblocks = ncols / QK_K;

    // block_q5_K is 176 bytes: dm(4) + scales(12) + qh(32) + qs(128).
    constexpr int BLOCK_BYTES   = 4 + K_SCALE_SIZE + (QK_K / 8) + (QK_K / 2);
    constexpr int OFF_SCALES    = 4;
    constexpr int OFF_QH        = OFF_SCALES + K_SCALE_SIZE;   // 16
    constexpr int OFF_QS        = OFF_QH + (QK_K / 8);          // 48
    static_assert(BLOCK_BYTES == 176, "block_q5_K size assumption");

    const auto * x_base = static_cast<const uint8_t *>(vx);

    // q8_1 activation y: [ncols int8 quants] [(ncols/QK8_1) sycl::half2 ds]
    const auto * y_qs = static_cast<const int8_t *>(vy);
    const auto * y_ds = reinterpret_cast<const sycl::half2 *>(
        static_cast<const uint8_t *>(vy) + ncols);

    float acc[ROWS_PER_THREAD] = { 0.0f };

    for (int sb = 0; sb < n_sblocks; ++sb) {
        // Activation loads - shared across all rows in this thread.
        esimd::simd<int8_t, 256> y_q_shared;
        y_q_shared.copy_from(y_qs + sb * QK_K);

        constexpr int N_Y_DS = QK_K / QK8_1;  // 8
        const uint16_t * y_ds_u16 =
            reinterpret_cast<const uint16_t *>(y_ds + sb * N_Y_DS);

        // Per-sub-block dy and sy = (d, s) of q8_1 ds half2.
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
            const auto * blk = x_base + block_idx * BLOCK_BYTES;

            // dm via raw uint16 + bit_cast.
            const uint16_t * dm_u16 = reinterpret_cast<const uint16_t *>(blk);
            const float d_f    = static_cast<float>(sycl::bit_cast<sycl::half>(dm_u16[0]));
            const float dmin_f = static_cast<float>(sycl::bit_cast<sycl::half>(dm_u16[1]));

            // Packed scales[12] - same encoding as Q4_K.
            esimd::simd<uint8_t, 12> scales_v;
            scales_v.copy_from(blk + OFF_SCALES);
            uint8_t s[12];
            #pragma unroll
            for (int k = 0; k < 12; ++k) s[k] = scales_v[k];

            esimd::simd<float, 8> sub_scales_v;
            esimd::simd<float, 8> sub_mins_v;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sub_scales_v[i]     = static_cast<float>(s[i] & 0x3F);
                sub_mins_v  [i]     = static_cast<float>(s[i + 4] & 0x3F);
                sub_scales_v[i + 4] = static_cast<float>((s[i + 8] & 0x0F) | ((s[i + 0] & 0xC0) >> 2));
                sub_mins_v  [i + 4] = static_cast<float>((s[i + 8] >> 4)   | ((s[i + 4] & 0xC0) >> 2));
            }

            // qh: 32 bytes. Bit `s` of qh[l] is the high bit for weight l in sub-block s.
            esimd::simd<uint8_t, 32> qh_v;
            qh_v.copy_from(blk + OFF_QH);

            // qs: 128 bytes. qs[pair*32 + l] -> low4 = sub(2*pair) weight l,
            //                                   high4 = sub(2*pair+1) weight l.
            esimd::simd<uint8_t, 128> qs_v;
            qs_v.copy_from(blk + OFF_QS);

            esimd::simd<float, 8> dot_v;

            // 4 pair iterations - each handles two sub-blocks (sub, sub+1).
            #pragma unroll
            for (int sub = 0; sub < 8; sub += 2) {
                const int qs_offset = (sub / 2) * 32;
                esimd::simd<uint8_t, 32> qs_pair = qs_v.template select<32, 1>(qs_offset);

                // Low 4-bit nibble for sub, high 4-bit nibble for sub+1.
                esimd::simd<int8_t, 32> w_lo = qs_pair & uint8_t(0x0F);
                esimd::simd<int8_t, 32> w_hi = (qs_pair >> 4) & uint8_t(0x0F);

                // High bit: bit `sub` of qh -> w_lo, bit `sub+1` -> w_hi.
                // Add 16 where the qh bit is set.
                esimd::simd<int8_t, 32> hi_lo = ((qh_v >> sub)       & uint8_t(0x01)) << 4;
                esimd::simd<int8_t, 32> hi_hi = ((qh_v >> (sub + 1)) & uint8_t(0x01)) << 4;
                w_lo = w_lo + hi_lo;  // unsigned 5-bit, range 0..31
                w_hi = w_hi + hi_hi;

                // Pack into simd<int8,64>.
                esimd::simd<int8_t, 64> w_pair;
                w_pair.template select<32, 1>(0)  = w_lo;
                w_pair.template select<32, 1>(32) = w_hi;

                esimd::simd<int8_t, 64> y_pair;
                y_pair.template select<32, 1>(0)  = y_q_shared.template select<32, 1>(sub * 32);
                y_pair.template select<32, 1>(32) = y_q_shared.template select<32, 1>((sub + 1) * 32);

                esimd::simd<int16_t, 64> prod_pair = w_pair * y_pair;

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

void mul_mat_vec_q5_k_q8_1_sycl_esimd(
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
                esimd_q5_k_kernel<32>(vx, vy, dst, ncols, nrows, nd_item);
            });
    });
}

// Phase C addition - Q8_0 ESIMD kernel sibling to the Q4_K iter-16 kernel.
//
// Q8_0 block layout (per ggml-common.h):
//   typedef struct {
//       ggml_half d;          //  2 bytes - delta
//       int8_t    qs[QK8_0];  // 32 bytes - 32 signed int8 weights
//   } block_q8_0;             // 34 bytes total
//
// Reorder layout (per ggml_sycl_reordered::block_q_t<GGML_TYPE_Q8_0> /
// reorder_qw_q8_0 in ggml-sycl.cpp):
//   [all blocks' qs (32 bytes/block, ncols/QK8_0 blocks/row)]
//   [all blocks' d  (sizeof(ggml_half) per block)]
//
// Math (matches reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>):
//   per block: sumi = dot(int8 weights[0..32), q8_1.qs[0..32))
//              block_contrib = d * sumi * ds_d   (no min/bias term)
//   row sum: dst[row] = sum over all blocks of block_contrib
//
// Q8_0 is the simplest of the five formats: weights are already signed int8
// (no nibble unpack), no qh high-bit, no packed sub-scales, no min term,
// no global super-block scale. We process 8 contiguous Q8_0 blocks per
// super-iteration so the activation load is the same 256-byte simd<int8,256>
// shared across all rows in the thread (matches Q4_K / Q4_0 / Q5_K / Q6_K).
//
// QK8_0 = 32, QK8_1 = 32 -- so 8 Q8_0 blocks aligns with 8 q8_1 sub-blocks.

template <int VEC_W>
static void esimd_q8_0_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float * __restrict__ dst,
        const int ncols, const int nrows,
        const sycl::nd_item<1> & nd_item) SYCL_ESIMD_KERNEL {

    const int thread_id = nd_item.get_global_id(0);
    const int row_base  = thread_id * ROWS_PER_THREAD;
    if (row_base >= nrows) return;

    constexpr int Q80_BLOCKS_PER_GROUP = 8;     // 8 blocks * 32 weights = 256
    constexpr int GROUP_QS_BYTES       = Q80_BLOCKS_PER_GROUP * QK8_0;        // 256
    constexpr int GROUP_Y_QS_BYTES     = Q80_BLOCKS_PER_GROUP * QK8_1;        // 256

    const int n_blocks_per_row = ncols / QK8_0;
    const int n_groups         = n_blocks_per_row / Q80_BLOCKS_PER_GROUP;
    const int total_blocks     = nrows * n_blocks_per_row;

    // Reorder region pointers: qs | d
    const auto * qs_base = static_cast<const int8_t *>(vx);
    const auto * d_base  = reinterpret_cast<const uint8_t *>(qs_base) + total_blocks * QK8_0;

    // q8_1 activation y: [ncols int8 quants] [(ncols/QK8_1) sycl::half2 ds]
    const auto * y_qs    = static_cast<const int8_t *>(vy);
    const auto * y_ds_u8 = static_cast<const uint8_t *>(vy) + ncols;

    float acc[ROWS_PER_THREAD] = { 0.0f };

    for (int g = 0; g < n_groups; ++g) {
        // Activation load - shared across all rows in this thread.
        esimd::simd<int8_t, 256> y_q_shared;
        y_q_shared.copy_from(y_qs + g * GROUP_Y_QS_BYTES);

        // q8_1 (d, s) per sub-block, 8 sub-blocks per group. Q8_0 has no min
        // term so we only need d (sy is unused).
        const uint16_t * y_ds_u16 = reinterpret_cast<const uint16_t *>(
            y_ds_u8 + g * Q80_BLOCKS_PER_GROUP * sizeof(sycl::half2));
        esimd::simd<float, 8> dy_v;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            dy_v[i] = static_cast<float>(sycl::bit_cast<sycl::half>(y_ds_u16[i * 2]));
        }

        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int row = row_base + r;
            if (row >= nrows) break;
            const int block0 = row * n_blocks_per_row + g * Q80_BLOCKS_PER_GROUP;

            // 8 contiguous Q8_0 blocks of qs = 256 bytes of int8 weights.
            esimd::simd<int8_t, 256> w;
            w.copy_from(qs_base + block0 * QK8_0);

            // Per-block d (half) - 8 of them.
            const uint16_t * d_u16 = reinterpret_cast<const uint16_t *>(
                d_base + block0 * sizeof(ggml_half));
            esimd::simd<float, 8> d_v;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                d_v[i] = static_cast<float>(sycl::bit_cast<sycl::half>(d_u16[i]));
            }

            // sumi per block: int8 weight dot int8 activation (32 each).
            // Process 2 blocks per pair (64 weights) to bound GRF pressure -
            // a single 256-wide w*y multiply needs simd<int16,256> = 512 B
            // of intermediate per row, which spills under 4-row fan-out.
            esimd::simd<float, 8> sumi_v;
            #pragma unroll
            for (int sub = 0; sub < 8; sub += 2) {
                esimd::simd<int8_t, 64> w_pair;
                w_pair.template select<32, 1>(0)  = w.template select<32, 1>(sub       * 32);
                w_pair.template select<32, 1>(32) = w.template select<32, 1>((sub + 1) * 32);

                esimd::simd<int8_t, 64> y_pair;
                y_pair.template select<32, 1>(0)  = y_q_shared.template select<32, 1>(sub       * 32);
                y_pair.template select<32, 1>(32) = y_q_shared.template select<32, 1>((sub + 1) * 32);

                esimd::simd<int16_t, 64> prod_pair = w_pair * y_pair;

                esimd::simd<int16_t, 32> prod_b0 = prod_pair.template select<32, 1>(0);
                esimd::simd<int16_t, 32> prod_b1 = prod_pair.template select<32, 1>(32);

                sumi_v[sub]     = static_cast<float>(esimd::reduce<int>(prod_b0, std::plus<>{}));
                sumi_v[sub + 1] = static_cast<float>(esimd::reduce<int>(prod_b1, std::plus<>{}));
            }

            // Per-block contrib: d * sumi * dy   (no min/offset term)
            const esimd::simd<float, 8> contrib = d_v * sumi_v * dy_v;
            acc[r] += esimd::reduce<float>(contrib, std::plus<>{});
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

void reorder_mul_mat_vec_q8_0_q8_1_sycl_esimd(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % (QK8_0 * 8) == 0);  // need multiple of 8 blocks per row

    constexpr int WG_SIZE = 32;
    const int n_threads = (nrows + ROWS_PER_THREAD - 1) / ROWS_PER_THREAD;
    const sycl::range<1> global_size((n_threads + WG_SIZE - 1) / WG_SIZE * WG_SIZE);
    const sycl::range<1> workgroup_size(WG_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(global_size, workgroup_size),
            [=](sycl::nd_item<1> nd_item) SYCL_ESIMD_KERNEL {
                esimd_q8_0_kernel<32>(vx, vy, dst, ncols, nrows, nd_item);
            });
    });
}



// =========================================================================
// Q3_K — 3-bit K-quant (raw block layout, no reorder).
//
// block_q3_K = 110 bytes:
//   hmask[QK_K/8] = 32 B  (high bit per weight, bit-packed)
//   qs   [QK_K/4] = 64 B  (low 2 bits per weight, bit-packed)
//   scales[12]              (16 sub-scales, 6-bit packed: low4 in 0..7, high2 in 8..11)
//   d    (ggml_half)        (super-block delta)
//
// Layout (from ggml-quants.c dequantize_row_q3_K):
//   For output position l = 128*n + 32*j + 16*half + k  (n in {0,1}, j in 0..3,
//   half in {0,1}, k in 0..15):
//     low2(l)  = (qs[32*n + 16*half + k] >> (2*j)) & 0x3
//     hbit(l)  = (hmask[16*half + k] >> (4*n + j)) & 0x1
//     signed   = low2(l) + 4*hbit(l) - 4    in [-4, 3]
//
// Per q8_1 sub-block q (q in 0..7, covering outputs q*32 .. q*32+31), we have
// n = q/4, j = q%4, and the 32 weights split into low-16 (half=0) and high-16
// (half=1). 32 weights map to qs bytes [q_qs_off .. q_qs_off+31] with bit shift
// q_shift = 2*(q%4), and hmask byte index = same 0..31 range, bit position q.
//
// 16 signed sub-scales (range -32..31): scale[2*q] applies to low-16,
// scale[2*q+1] to high-16. No min term — Q3_K only uses dy from q8_1's ds.
//
// Pattern follows Q6_K (signed offset, no min) and Q5_K (raw-block load,
// bit-extract from a mask byte).
// =========================================================================

template <int VEC_W>
static void esimd_q3_k_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float * __restrict__ dst,
        const int ncols, const int nrows,
        const sycl::nd_item<1> & nd_item) SYCL_ESIMD_KERNEL {

    const int thread_id = nd_item.get_global_id(0);
    const int row_base  = thread_id * ROWS_PER_THREAD;
    if (row_base >= nrows) return;

    const int n_sblocks = ncols / QK_K;

    // block_q3_K is 110 bytes: hmask(32) | qs(64) | scales(12) | d(2)
    constexpr int BLOCK_BYTES  = (QK_K / 8) + (QK_K / 4) + 12 + sizeof(ggml_half);
    constexpr int OFF_HMASK    = 0;
    constexpr int OFF_QS       = QK_K / 8;               // 32
    constexpr int OFF_SCALES   = OFF_QS + (QK_K / 4);    // 96
    constexpr int OFF_D        = OFF_SCALES + 12;        // 108
    static_assert(BLOCK_BYTES == 110, "block_q3_K size assumption");

    const auto * x_base = static_cast<const uint8_t *>(vx);

    // q8_1 activation y: [ncols int8 quants] [(ncols/QK8_1) sycl::half2 ds]
    const auto * y_qs = static_cast<const int8_t *>(vy);
    const auto * y_ds = reinterpret_cast<const sycl::half2 *>(
        static_cast<const uint8_t *>(vy) + ncols);

    float acc[ROWS_PER_THREAD] = { 0.0f };

    for (int sb = 0; sb < n_sblocks; ++sb) {
        // Activation load — shared across all rows in this thread.
        esimd::simd<int8_t, 256> y_q_shared;
        y_q_shared.copy_from(y_qs + sb * QK_K);

        constexpr int N_Y_DS = QK_K / QK8_1;  // 8
        const uint16_t * y_ds_u16 =
            reinterpret_cast<const uint16_t *>(y_ds + sb * N_Y_DS);

        // Per-q8_1-sub-block dy. Q3_K has no min term (no sy use).
        esimd::simd<float, 8> dy_v;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            dy_v[i] = static_cast<float>(sycl::bit_cast<sycl::half>(y_ds_u16[i * 2]));
        }

        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int row = row_base + r;
            if (row >= nrows) break;
            const int block_idx = row * n_sblocks + sb;
            const auto * blk = x_base + block_idx * BLOCK_BYTES;

            // Per-row weight loads.
            esimd::simd<uint8_t, 32> hmask_v;
            hmask_v.copy_from(blk + OFF_HMASK);

            esimd::simd<uint8_t, 64> qs_v;
            qs_v.copy_from(blk + OFF_QS);

            esimd::simd<uint8_t, 12> scales_v;
            scales_v.copy_from(blk + OFF_SCALES);

            // Super-block delta d.
            const uint16_t * d_u16 = reinterpret_cast<const uint16_t *>(blk + OFF_D);
            const float d_f = static_cast<float>(sycl::bit_cast<sycl::half>(*d_u16));

            // Unpack 16 signed sub-scales: 6-bit unsigned scale[i] then -32 offset.
            //   low 4 bits : scales[i & 7] >> ((i >> 3) * 4)
            //   high 2 bits: scales[8 + (i & 3)] >> ((i >> 2) * 2)
            //   signed     : (low | (high << 4)) - 32
            uint8_t s[12];
            #pragma unroll
            for (int k = 0; k < 12; ++k) s[k] = scales_v[k];

            esimd::simd<float, 16> sc_f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const uint8_t lo = (s[i & 7]        >> ((i >> 3) * 4)) & uint8_t(0x0F);
                const uint8_t hi = (s[8 + (i & 3)]  >> ((i >> 2) * 2)) & uint8_t(0x03);
                const int8_t sc = static_cast<int8_t>(static_cast<int>(lo | (hi << 4)) - 32);
                sc_f[i] = static_cast<float>(sc);
            }

            // Per q8_1 sub-block sumi values (first-16 / second-16 of 32).
            esimd::simd<float, 8> sumi_first_v;
            esimd::simd<float, 8> sumi_second_v;

            // 8 q8_1 sub-blocks, 32 weights each. For sub-block q:
            //   n = q / 4, j = q % 4
            //   qs slice base = 32 * n  (32 bytes, low-16 + high-16 stacked)
            //   qs bit shift  = 2 * j
            //   hmask bit     = q   (covers both halves through hmask[0..15] + hmask[16..31])
            #pragma unroll
            for (int q = 0; q < 8; ++q) {
                const int q_qs_offset = (q < 4) ? 0 : 32;
                const int q_shift     = (q & 3) * 2;

                // 32 packed bytes of qs (low-16 weights from bytes 0..15, high-16 from 16..31).
                esimd::simd<uint8_t, 32> qs_slice = qs_v.template select<32, 1>(q_qs_offset);

                // Low 2 bits per byte at this q's shift.
                esimd::simd<int8_t, 32> w_low =
                    (qs_slice >> uint8_t(q_shift)) & uint8_t(0x03);

                // High bit per weight from hmask: bit q of each of 32 hmask bytes.
                esimd::simd<int8_t, 32> hbit =
                    (hmask_v >> uint8_t(q)) & uint8_t(0x01);

                // signed = low + 4*hbit - 4    (hbit=0 → low-4; hbit=1 → low)
                esimd::simd<int8_t, 32> w_chunk = w_low + (hbit << 2) - int8_t(4);

                // Activation chunk: y_q_shared[q*32 .. q*32+31] is laid out output-major,
                // matching weight output positions.
                esimd::simd<int8_t, 32> y_chunk =
                    y_q_shared.template select<32, 1>(q * 32);

                esimd::simd<int16_t, 32> prod = w_chunk * y_chunk;

                // Split first-16 / second-16 for the 2 sub-scales (2*q, 2*q+1).
                esimd::simd<int16_t, 16> prod_a = prod.template select<16, 1>(0);
                esimd::simd<int16_t, 16> prod_b = prod.template select<16, 1>(16);

                sumi_first_v [q] =
                    static_cast<float>(esimd::reduce<int>(prod_a, std::plus<>{}));
                sumi_second_v[q] =
                    static_cast<float>(esimd::reduce<int>(prod_b, std::plus<>{}));
            }

            // Pack scales: sc_f[2*q] for first-16, sc_f[2*q+1] for second-16.
            esimd::simd<float, 8> sc_first_v;
            esimd::simd<float, 8> sc_second_v;
            #pragma unroll
            for (int q = 0; q < 8; ++q) {
                sc_first_v [q] = sc_f[2 * q];
                sc_second_v[q] = sc_f[2 * q + 1];
            }

            // contrib[q] = (sc_first[q]*sumi_first[q] + sc_second[q]*sumi_second[q]) * dy[q]
            const esimd::simd<float, 8> contrib =
                (sc_first_v * sumi_first_v + sc_second_v * sumi_second_v) * dy_v;
            acc[r] += d_f * esimd::reduce<float>(contrib, std::plus<>{});
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

void mul_mat_vec_q3_k_q8_1_sycl_esimd(
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
                esimd_q3_k_kernel<32>(vx, vy, dst, ncols, nrows, nd_item);
            });
    });
}



// =========================================================================
// Q2_K — 2-bit K-quant (raw block layout, no reorder).
//
// block_q2_K = 84 bytes:
//   scales[QK_K/16] = 16 B  (4-bit sub-scale low nibble, 4-bit sub-min high nibble)
//   qs    [QK_K/4]  = 64 B  (2-bit unsigned weights, 4 per byte)
//   dm = (d, dmin) ggml_half2
//
// Layout (from ggml-quants.c dequantize_row_q2_K):
//   For output position l = 128*n + 32*j + 16*half + k   (n in {0,1},
//   j in 0..3, half in {0,1}, k in 0..15):
//     low2(l)     = (qs[32*n + 16*half + k] >> (2*j)) & 0x3   unsigned [0,3]
//     scale-byte  = scales[is] where is = 8*n + 2*j + half    (i.e. 0..15)
//     sub-scale   = scale-byte & 0x0F
//     sub-min     = scale-byte >> 4
//     weight_l    = d*sub-scale*low2(l) - dmin*sub-min       (NO offset; unsigned)
//
// Per q8_1 sub-block q (q in 0..7, covering outputs q*32 .. q*32+31):
//   n = q/4, j = q%4
//   qs_off = 32*n (32 bytes covering 16+16 weight half-blocks at this shift)
//   shift = 2*j
//   Low-16 (k=0..15) uses scales[2*q].
//   High-16 (k=16..31) uses scales[2*q+1].
//
// Combined contribution (mirrors impl_mmvq's two terms):
//   sum_d  = sum_{q,h} dy_q * sub_scale_{q,h} * sumi_{q,h}
//   sum_m  = sum_{q,h} dy_q * sub_min_{q,h}   * sum_u_{q,h}
//   result = d * sum_d - dmin * sum_m
// where sumi_{q,h} = dp(weights, u) on half h of q8_1 sub-block q,
// and sum_u_{q,h} = plain sum of u quants in that half.
//
// Q2_K differs from Q4_K's existing min-term path: Q4_K uses sy (the
// precomputed sum element of q8_1 ds half2) because both halves of a
// Q4_K q8_1 sub-block share one min. Q2_K's two halves can have
// different mins (`scales[2*q]` vs `scales[2*q+1]`), so we must compute
// sum_u for each half separately. These half-sums depend only on
// y_q_shared which is identical across all 4 rows in this thread, so
// we hoist them out of the row loop.
//
// Pattern: Q3_K layout (raw block, no reorder, 16 sub-scales, first-16/
//   second-16 split per q8_1 sub-block) + Q4_K-style min term.
// =========================================================================

template <int VEC_W>
static void esimd_q2_k_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float * __restrict__ dst,
        const int ncols, const int nrows,
        const sycl::nd_item<1> & nd_item) SYCL_ESIMD_KERNEL {

    const int thread_id = nd_item.get_global_id(0);
    const int row_base  = thread_id * ROWS_PER_THREAD;
    if (row_base >= nrows) return;

    const int n_sblocks = ncols / QK_K;

    // block_q2_K is 84 bytes: scales(16) | qs(64) | dm(4)
    constexpr int BLOCK_BYTES  = (QK_K / 16) + (QK_K / 4) + 2 * sizeof(ggml_half);
    constexpr int OFF_SCALES   = 0;
    constexpr int OFF_QS       = QK_K / 16;                  // 16
    constexpr int OFF_DM       = OFF_QS + (QK_K / 4);        // 80
    static_assert(BLOCK_BYTES == 84, "block_q2_K size assumption");

    const auto * x_base = static_cast<const uint8_t *>(vx);

    // q8_1 activation y: [ncols int8 quants] [(ncols/QK8_1) sycl::half2 ds]
    const auto * y_qs = static_cast<const int8_t *>(vy);
    const auto * y_ds = reinterpret_cast<const sycl::half2 *>(
        static_cast<const uint8_t *>(vy) + ncols);

    float acc[ROWS_PER_THREAD] = { 0.0f };

    for (int sb = 0; sb < n_sblocks; ++sb) {
        // Activation load — shared across all rows in this thread.
        esimd::simd<int8_t, 256> y_q_shared;
        y_q_shared.copy_from(y_qs + sb * QK_K);

        constexpr int N_Y_DS = QK_K / QK8_1;  // 8
        const uint16_t * y_ds_u16 =
            reinterpret_cast<const uint16_t *>(y_ds + sb * N_Y_DS);

        // Per-q8_1-sub-block dy. (sy unused — Q2_K has per-half mins,
        // we need split sums of u, not the full sum sy encodes.)
        esimd::simd<float, 8> dy_v;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            dy_v[i] = static_cast<float>(sycl::bit_cast<sycl::half>(y_ds_u16[i * 2]));
        }

        // Per-half sums of u for the min term: hoisted (y_q_shared is identical
        // across rows). Computed once per super-block, used by all 4 rows.
        esimd::simd<float, 8> y_sum_first_v;
        esimd::simd<float, 8> y_sum_second_v;
        #pragma unroll
        for (int q = 0; q < 8; ++q) {
            esimd::simd<int8_t, 32> y_chunk =
                y_q_shared.template select<32, 1>(q * 32);
            esimd::simd<int16_t, 16> y_lo16 = y_chunk.template select<16, 1>(0);
            esimd::simd<int16_t, 16> y_hi16 = y_chunk.template select<16, 1>(16);
            y_sum_first_v [q] =
                static_cast<float>(esimd::reduce<int>(y_lo16, std::plus<>{}));
            y_sum_second_v[q] =
                static_cast<float>(esimd::reduce<int>(y_hi16, std::plus<>{}));
        }

        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int row = row_base + r;
            if (row >= nrows) break;
            const int block_idx = row * n_sblocks + sb;
            const auto * blk = x_base + block_idx * BLOCK_BYTES;

            // Per-row weight loads.
            esimd::simd<uint8_t, 16> scales_v;
            scales_v.copy_from(blk + OFF_SCALES);

            esimd::simd<uint8_t, 64> qs_v;
            qs_v.copy_from(blk + OFF_QS);

            // dm = (d, dmin) via raw uint16 + bit_cast.
            const uint16_t * dm_u16 = reinterpret_cast<const uint16_t *>(blk + OFF_DM);
            const float d_f    = static_cast<float>(sycl::bit_cast<sycl::half>(dm_u16[0]));
            const float dmin_f = static_cast<float>(sycl::bit_cast<sycl::half>(dm_u16[1]));

            // Unpack 16 (sub-scale, sub-min) pairs.
            esimd::simd<float, 16> sub_scale_v;
            esimd::simd<float, 16> sub_min_v;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const uint8_t sc_byte = scales_v[i];
                sub_scale_v[i] = static_cast<float>(sc_byte & uint8_t(0x0F));
                sub_min_v  [i] = static_cast<float>(sc_byte >> 4);
            }

            // Per q8_1 sub-block sumi values (first-16 / second-16 of 32).
            esimd::simd<float, 8> sumi_first_v;
            esimd::simd<float, 8> sumi_second_v;

            // 8 q8_1 sub-blocks. For sub-block q:
            //   n = q/4, j = q%4
            //   qs slice base = 32*n     (32 bytes, low-16 + high-16 stacked)
            //   qs bit shift  = 2*j
            //   No hmask — Q2_K weights are unsigned 2-bit [0,3].
            #pragma unroll
            for (int q = 0; q < 8; ++q) {
                const int q_qs_offset = (q < 4) ? 0 : 32;
                const int q_shift     = (q & 3) * 2;

                esimd::simd<uint8_t, 32> qs_slice = qs_v.template select<32, 1>(q_qs_offset);
                esimd::simd<int8_t, 32> w_chunk =
                    (qs_slice >> uint8_t(q_shift)) & uint8_t(0x03);  // unsigned [0,3]

                esimd::simd<int8_t, 32> y_chunk =
                    y_q_shared.template select<32, 1>(q * 32);

                esimd::simd<int16_t, 32> prod = w_chunk * y_chunk;

                esimd::simd<int16_t, 16> prod_a = prod.template select<16, 1>(0);
                esimd::simd<int16_t, 16> prod_b = prod.template select<16, 1>(16);

                sumi_first_v [q] =
                    static_cast<float>(esimd::reduce<int>(prod_a, std::plus<>{}));
                sumi_second_v[q] =
                    static_cast<float>(esimd::reduce<int>(prod_b, std::plus<>{}));
            }

            // Pack per-half scale/min (sc[2*q] for first-16, sc[2*q+1] for second-16).
            esimd::simd<float, 8> sc_first_v,  sc_second_v;
            esimd::simd<float, 8> min_first_v, min_second_v;
            #pragma unroll
            for (int q = 0; q < 8; ++q) {
                sc_first_v  [q] = sub_scale_v[2 * q];
                sc_second_v [q] = sub_scale_v[2 * q + 1];
                min_first_v [q] = sub_min_v  [2 * q];
                min_second_v[q] = sub_min_v  [2 * q + 1];
            }

            // d-term : dy_q * (sc_first * sumi_first + sc_second * sumi_second)
            // m-term : dy_q * (min_first * y_sum_first + min_second * y_sum_second)
            const esimd::simd<float, 8> contrib_d =
                dy_v * (sc_first_v  * sumi_first_v  + sc_second_v  * sumi_second_v);
            const esimd::simd<float, 8> contrib_m =
                dy_v * (min_first_v * y_sum_first_v + min_second_v * y_sum_second_v);

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

void mul_mat_vec_q2_k_q8_1_sycl_esimd(
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
                esimd_q2_k_kernel<32>(vx, vy, dst, ncols, nrows, nd_item);
            });
    });
}


#endif  // GGML_SYCL_ESIMD
