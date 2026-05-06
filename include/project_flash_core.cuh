#pragma once

#include "flash_attn.cuh"

#include <mma.h>

namespace project_flash {

namespace wmma = nvcuda::wmma;

// Shared-memory padding (in half elements) for the tensor-core paths.
// The padded leading dimensions must remain multiples of 8 half values so that
// WMMA load_matrix_sync stays legal, while also shifting rows away from the
// worst shared-memory bank-alignment pattern.
constexpr int SMEM_PAD = 8;

// ---- Legacy helpers used by ablation kernels (unpadded strides) --------

__device__ inline void load_kv_block(
    project_in_t* s_kt,
    project_in_t* s_v,
    const project_in_t* k_base,
    const project_in_t* v_base,
    int kv_start,
    int N,
    int d
) {
    const project_in_t zero = __float2half(0.0f);
    for (int idx = threadIdx.x; idx < PROJECT_BLOCK_N * d; idx += blockDim.x) {
        int row = idx / d;
        int col = idx % d;
        int global_row = kv_start + row;

        project_in_t k_val = zero;
        project_in_t v_val = zero;
        if (global_row < N) {
            k_val = k_base[global_row * d + col];
            if (v_base != nullptr) {
                v_val = v_base[global_row * d + col];
            }
        }

        s_kt[col + row * d] = k_val;
        if (s_v != nullptr) {
            s_v[row * d + col] = v_val;
        }
    }
}

__device__ inline void compute_score_tile_tensor_core(
    float* s_scores,
    const project_in_t* s_q,
    const project_in_t* s_kt,
    int d,
    int score_stride,
    float scale
) {
    wmma::fragment<wmma::accumulator, PROJECT_TILE, PROJECT_TILE, PROJECT_TILE, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < d; k0 += PROJECT_TILE) {
        wmma::fragment<wmma::matrix_a, PROJECT_TILE, PROJECT_TILE, PROJECT_TILE, project_in_t, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, PROJECT_TILE, PROJECT_TILE, PROJECT_TILE, project_in_t, wmma::col_major> b_frag;

        wmma::load_matrix_sync(a_frag, s_q + k0, d);
        wmma::load_matrix_sync(b_frag, s_kt + k0, d);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag.x[i] *= scale;
    }

    wmma::store_matrix_sync(s_scores, c_frag, score_stride, wmma::mem_row_major);
}

__device__ inline void compute_score_block_tensor_core(
    float* s_scores,
    const project_in_t* s_q,
    const project_in_t* s_kt,
    int d,
    float scale
) {
    #pragma unroll
    for (int tile_idx = 0; tile_idx < PROJECT_K_TILES_PER_BLOCK; tile_idx++) {
        compute_score_tile_tensor_core(
            s_scores + tile_idx * PROJECT_TILE,
            s_q,
            s_kt + tile_idx * PROJECT_TILE * d,
            d,
            PROJECT_BLOCK_N,
            scale
        );
    }
}

// ========================================================================
// Core FA1 kernel — register-resident O / scores with native-like memory use
//
// Key optimizations in the main path:
//   1. Compile-time head-dim dispatch, mirroring native FA1's specialized
//      kernels for common head dimensions.
//   2. 16-byte vectorized Q/K/V tile loads (the official code uses uint4
//      global-memory transactions in its tile loaders).
//   3. Shared K/V SRAM backed by V fragments kept in registers, following the
//      native FA1 pattern more closely without leaving the WMMA programming
//      model.
//   4. Loop ordering that reuses loaded Q fragments across K tiles and reuses
//      loaded P fragments across output-dimension tiles.
//   5. exp2-based online softmax scaling, matching the official softmax path.
// ========================================================================

template<int ROWS, int COLS, int COLS_PADDED, bool UseVecLoads>
__device__ inline void load_padded_rowmajor_tile(
    project_in_t* dst,
    const project_in_t* src,
    int global_row_start,
    int N
) {
    const project_in_t zero = __float2half(0.0f);
    if constexpr (UseVecLoads) {
        static_assert((COLS % 8) == 0);
        static_assert((COLS_PADDED % 8) == 0);
        constexpr int kVecElems = 8;
        constexpr int kVecsPerRow = COLS / kVecElems;

        for (int idx = threadIdx.x; idx < ROWS * kVecsPerRow; idx += blockDim.x) {
            int row = idx / kVecsPerRow;
            int vec = idx % kVecsPerRow;
            int gr = global_row_start + row;
            uint4 data = make_uint4(0u, 0u, 0u, 0u);
            if (gr < N) {
                const uint4* src_vec =
                    reinterpret_cast<const uint4*>(src + gr * COLS);
                data = src_vec[vec];
            }
            uint4* dst_vec = reinterpret_cast<uint4*>(dst + row * COLS_PADDED);
            dst_vec[vec] = data;
        }
    } else {
        for (int idx = threadIdx.x; idx < ROWS * COLS; idx += blockDim.x) {
            int row = idx / COLS;
            int col = idx % COLS;
            int gr = global_row_start + row;
            dst[row * COLS_PADDED + col] = (gr < N)
                ? src[gr * COLS + col]
                : zero;
        }
    }

    for (int idx = threadIdx.x; idx < ROWS * (COLS_PADDED - COLS);
         idx += blockDim.x) {
        int row = idx / (COLS_PADDED - COLS);
        int pad = idx % (COLS_PADDED - COLS);
        dst[row * COLS_PADDED + COLS + pad] = zero;
    }
}

template<int HEAD_DIM, bool UseVecLoads>
static __global__ void flash_attention_core_kernel(
    const project_in_t* __restrict__ Q,
    const project_in_t* __restrict__ K,
    const project_in_t* __restrict__ V,
    project_out_t* __restrict__ O,
    int N,
    float scale_l2,
    bool causal
) {
    constexpr int d = HEAD_DIM;
    constexpr int num_o_frags = HEAD_DIM / PROJECT_TILE;
    constexpr int d_padded = HEAD_DIM + SMEM_PAD;
    constexpr int bn_padded = PROJECT_BLOCK_N + SMEM_PAD;

    const int batch_head = blockIdx.z;
    const int warp_id = threadIdx.x / PROJECT_WARP_SIZE;
    const int lane = threadIdx.x % PROJECT_WARP_SIZE;
    const int q_block_start = blockIdx.x * PROJECT_BLOCK_M;
    const int q_start = q_block_start + warp_id * PROJECT_TILE;

    const int fg = lane / 4;
    const int fp = lane % 4;
    const int frow0 = fg;
    const int frow1 = fg + 8;
    const int global_row0 = q_start + frow0;
    const int global_row1 = q_start + frow1;

    const project_in_t* q_base = Q + batch_head * N * d;
    const project_in_t* k_base = K + batch_head * N * d;
    const project_in_t* v_base = V + batch_head * N * d;
    project_out_t* o_base = O + batch_head * N * d;

    extern __shared__ __align__(32) unsigned char smem_raw[];
    project_in_t* s_q  = reinterpret_cast<project_in_t*>(smem_raw);
    project_in_t* s_kv = s_q + PROJECT_BLOCK_M * d_padded;
    project_in_t* s_p  = s_kv;

    project_in_t* s_q_warp = s_q + warp_id * PROJECT_TILE * d_padded;
    project_in_t* s_p_warp = s_p + warp_id * PROJECT_TILE * bn_padded;

    load_padded_rowmajor_tile<PROJECT_BLOCK_M, d, d_padded, UseVecLoads>(
        s_q, q_base, q_block_start, N
    );

    wmma::fragment<wmma::accumulator, PROJECT_TILE, PROJECT_TILE,
                   PROJECT_TILE, float> o_frag[num_o_frags];
    #pragma unroll
    for (int f = 0; f < num_o_frags; f++) {
        wmma::fill_fragment(o_frag[f], 0.0f);
    }

    float m0 = -FLT_MAX, m1 = -FLT_MAX;
    float l0 = 0.0f, l1 = 0.0f;

    __syncthreads();

    wmma::fragment<wmma::matrix_a, PROJECT_TILE, PROJECT_TILE,
                   PROJECT_TILE, project_in_t,
                   wmma::row_major> q_frag[d / PROJECT_TILE];
    #pragma unroll
    for (int k0 = 0; k0 < d; k0 += PROJECT_TILE) {
        wmma::load_matrix_sync(q_frag[k0 / PROJECT_TILE], s_q_warp + k0, d_padded);
    }

    int num_kv_tiles = cdiv(N, PROJECT_BLOCK_N);
    if (causal) {
        int q_end = q_block_start + PROJECT_BLOCK_M - 1;
        if (q_end >= N) q_end = N - 1;
        int causal_limit = cdiv(q_end + 1, PROJECT_BLOCK_N);
        if (causal_limit < num_kv_tiles) num_kv_tiles = causal_limit;
    }

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        const int kv_start = kv_tile * PROJECT_BLOCK_N;

        load_padded_rowmajor_tile<PROJECT_BLOCK_N, d, d_padded, UseVecLoads>(
            s_kv, v_base, kv_start, N
        );
        __syncthreads();

        wmma::fragment<wmma::matrix_b, PROJECT_TILE, PROJECT_TILE,
                       PROJECT_TILE, project_in_t,
                       wmma::row_major> v_frag[PROJECT_K_TILES_PER_BLOCK][num_o_frags];
        #pragma unroll
        for (int kk = 0; kk < PROJECT_BLOCK_N; kk += PROJECT_TILE) {
            #pragma unroll
            for (int dd = 0; dd < d; dd += PROJECT_TILE) {
                wmma::load_matrix_sync(
                    v_frag[kk / PROJECT_TILE][dd / PROJECT_TILE],
                    s_kv + kk * d_padded + dd,
                    d_padded
                );
            }
        }

        __syncthreads();
        load_padded_rowmajor_tile<PROJECT_BLOCK_N, d, d_padded, UseVecLoads>(
            s_kv, k_base, kv_start, N
        );
        __syncthreads();

        wmma::fragment<wmma::accumulator, PROJECT_TILE, PROJECT_TILE,
                       PROJECT_TILE, float> sf[PROJECT_K_TILES_PER_BLOCK];
        #pragma unroll
        for (int tn = 0; tn < PROJECT_K_TILES_PER_BLOCK; tn++) {
            wmma::fill_fragment(sf[tn], 0.0f);
        }

        #pragma unroll
        for (int k0 = 0; k0 < d; k0 += PROJECT_TILE) {
            #pragma unroll
            for (int tn = 0; tn < PROJECT_K_TILES_PER_BLOCK; tn++) {
                wmma::fragment<wmma::matrix_b, PROJECT_TILE, PROJECT_TILE,
                               PROJECT_TILE, project_in_t,
                               wmma::col_major> b;
                wmma::load_matrix_sync(
                    b, s_kv + tn * PROJECT_TILE * d_padded + k0, d_padded
                );
                wmma::mma_sync(sf[tn], q_frag[k0 / PROJECT_TILE], b, sf[tn]);
            }
        }

        #pragma unroll
        for (int tn = 0; tn < PROJECT_K_TILES_PER_BLOCK; tn++) {
            #pragma unroll
            for (int i = 0; i < sf[tn].num_elements; i++) {
                sf[tn].x[i] *= scale_l2;
            }
        }

        __syncwarp();

        const bool r0v = (global_row0 < N);
        const bool r1v = (global_row1 < N);

        float mx0 = -FLT_MAX, mx1 = -FLT_MAX;
        #pragma unroll
        for (int tn = 0; tn < PROJECT_K_TILES_PER_BLOCK; tn++) {
            int bc = kv_start + tn * PROJECT_TILE;
            int c0 = bc + fp * 2;
            int c1 = c0 + 1;
            int c4 = bc + fp * 2 + 8;
            int c5 = c4 + 1;
            if (r0v) {
                if (c0 < N && (!causal || c0 <= global_row0)) mx0 = fmaxf(mx0, sf[tn].x[0]);
                if (c1 < N && (!causal || c1 <= global_row0)) mx0 = fmaxf(mx0, sf[tn].x[1]);
                if (c4 < N && (!causal || c4 <= global_row0)) mx0 = fmaxf(mx0, sf[tn].x[4]);
                if (c5 < N && (!causal || c5 <= global_row0)) mx0 = fmaxf(mx0, sf[tn].x[5]);
            }
            if (r1v) {
                if (c0 < N && (!causal || c0 <= global_row1)) mx1 = fmaxf(mx1, sf[tn].x[2]);
                if (c1 < N && (!causal || c1 <= global_row1)) mx1 = fmaxf(mx1, sf[tn].x[3]);
                if (c4 < N && (!causal || c4 <= global_row1)) mx1 = fmaxf(mx1, sf[tn].x[6]);
                if (c5 < N && (!causal || c5 <= global_row1)) mx1 = fmaxf(mx1, sf[tn].x[7]);
            }
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFF, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFF, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFF, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFF, mx1, 2));

        const float nm0 = fmaxf(m0, mx0);
        const float nm1 = fmaxf(m1, mx1);
        const float a0 = exp2f(m0 - nm0);
        const float a1 = exp2f(m1 - nm1);

        #pragma unroll
        for (int f = 0; f < num_o_frags; f++) {
            o_frag[f].x[0] *= a0; o_frag[f].x[1] *= a0;
            o_frag[f].x[2] *= a1; o_frag[f].x[3] *= a1;
            o_frag[f].x[4] *= a0; o_frag[f].x[5] *= a0;
            o_frag[f].x[6] *= a1; o_frag[f].x[7] *= a1;
        }

        float sum0 = 0.0f, sum1 = 0.0f;
        #pragma unroll
        for (int tn = 0; tn < PROJECT_K_TILES_PER_BLOCK; tn++) {
            int bc = kv_start + tn * PROJECT_TILE;
            int c0 = bc + fp * 2;
            int c1 = c0 + 1;
            int c4 = bc + fp * 2 + 8;
            int c5 = c4 + 1;

            float p0 = 0.0f, p1 = 0.0f, p4 = 0.0f, p5 = 0.0f;
            float p2 = 0.0f, p3 = 0.0f, p6 = 0.0f, p7 = 0.0f;
            if (r0v) {
                if (c0 < N && (!causal || c0 <= global_row0)) p0 = exp2f(sf[tn].x[0] - nm0);
                if (c1 < N && (!causal || c1 <= global_row0)) p1 = exp2f(sf[tn].x[1] - nm0);
                if (c4 < N && (!causal || c4 <= global_row0)) p4 = exp2f(sf[tn].x[4] - nm0);
                if (c5 < N && (!causal || c5 <= global_row0)) p5 = exp2f(sf[tn].x[5] - nm0);
            }
            if (r1v) {
                if (c0 < N && (!causal || c0 <= global_row1)) p2 = exp2f(sf[tn].x[2] - nm1);
                if (c1 < N && (!causal || c1 <= global_row1)) p3 = exp2f(sf[tn].x[3] - nm1);
                if (c4 < N && (!causal || c4 <= global_row1)) p6 = exp2f(sf[tn].x[6] - nm1);
                if (c5 < N && (!causal || c5 <= global_row1)) p7 = exp2f(sf[tn].x[7] - nm1);
            }

            sum0 += p0 + p1 + p4 + p5;
            sum1 += p2 + p3 + p6 + p7;

            int pc = tn * PROJECT_TILE;
            s_p_warp[frow0 * bn_padded + pc + fp * 2] = __float2half(p0);
            s_p_warp[frow0 * bn_padded + pc + fp * 2 + 1] = __float2half(p1);
            s_p_warp[frow0 * bn_padded + pc + fp * 2 + 8] = __float2half(p4);
            s_p_warp[frow0 * bn_padded + pc + fp * 2 + 9] = __float2half(p5);
            s_p_warp[frow1 * bn_padded + pc + fp * 2] = __float2half(p2);
            s_p_warp[frow1 * bn_padded + pc + fp * 2 + 1] = __float2half(p3);
            s_p_warp[frow1 * bn_padded + pc + fp * 2 + 8] = __float2half(p6);
            s_p_warp[frow1 * bn_padded + pc + fp * 2 + 9] = __float2half(p7);
        }

        sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1);
        sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
        sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 1);
        sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 2);

        l0 = l0 * a0 + sum0;
        l1 = l1 * a1 + sum1;
        m0 = nm0;
        m1 = nm1;

        __syncwarp();

        #pragma unroll
        for (int kk = 0; kk < PROJECT_BLOCK_N; kk += PROJECT_TILE) {
            wmma::fragment<wmma::matrix_a, PROJECT_TILE, PROJECT_TILE,
                           PROJECT_TILE, project_in_t,
                           wmma::row_major> pf;
            wmma::load_matrix_sync(pf, s_p_warp + kk, bn_padded);

            #pragma unroll
            for (int dd = 0; dd < d; dd += PROJECT_TILE) {
                wmma::mma_sync(o_frag[dd / PROJECT_TILE], pf,
                               v_frag[kk / PROJECT_TILE][dd / PROJECT_TILE],
                               o_frag[dd / PROJECT_TILE]);
            }
        }

        __syncthreads();
    }

    const float inv_l0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
    const float inv_l1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;

    if (global_row0 < N) {
        #pragma unroll
        for (int f = 0; f < num_o_frags; f++) {
            int bc = f * PROJECT_TILE;
            o_base[global_row0 * d + bc + fp * 2] = o_frag[f].x[0] * inv_l0;
            o_base[global_row0 * d + bc + fp * 2 + 1] = o_frag[f].x[1] * inv_l0;
            o_base[global_row0 * d + bc + fp * 2 + 8] = o_frag[f].x[4] * inv_l0;
            o_base[global_row0 * d + bc + fp * 2 + 9] = o_frag[f].x[5] * inv_l0;
        }
    }
    if (global_row1 < N) {
        #pragma unroll
        for (int f = 0; f < num_o_frags; f++) {
            int bc = f * PROJECT_TILE;
            o_base[global_row1 * d + bc + fp * 2] = o_frag[f].x[2] * inv_l1;
            o_base[global_row1 * d + bc + fp * 2 + 1] = o_frag[f].x[3] * inv_l1;
            o_base[global_row1 * d + bc + fp * 2 + 8] = o_frag[f].x[6] * inv_l1;
            o_base[global_row1 * d + bc + fp * 2 + 9] = o_frag[f].x[7] * inv_l1;
        }
    }
}

template<int HEAD_DIM, bool UseVecLoads>
inline void launch_flash_attention_core_hdim(
    const project_in_t* d_Q,
    const project_in_t* d_K,
    const project_in_t* d_V,
    project_out_t* d_O,
    int B,
    int H,
    int N,
    float scale_l2,
    bool causal
) {
    constexpr int d_padded = HEAD_DIM + SMEM_PAD;
    constexpr int bn_padded = PROJECT_BLOCK_N + SMEM_PAD;
    constexpr int kv_buf_stride = (d_padded > bn_padded) ? d_padded : bn_padded;
    constexpr int kv_buf_elems = PROJECT_BLOCK_N * kv_buf_stride;

    const int BH = B * H;
    const int num_q_tiles = cdiv(N, PROJECT_BLOCK_M);

    size_t smem_bytes = 0;
    smem_bytes += PROJECT_BLOCK_M * d_padded * sizeof(project_in_t);
    smem_bytes += kv_buf_elems * sizeof(project_in_t);

    dim3 block(PROJECT_THREADS);
    dim3 grid(num_q_tiles, 1, BH);

    if (smem_bytes >= 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            flash_attention_core_kernel<HEAD_DIM, UseVecLoads>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes
        ));
    }

    flash_attention_core_kernel<HEAD_DIM, UseVecLoads><<<grid, block, smem_bytes>>>(
        d_Q, d_K, d_V, d_O, N, scale_l2, causal
    );
    CUDA_CHECK(cudaGetLastError());
}

template<bool UseVecLoads>
inline void launch_flash_attention_core_impl(
    const project_in_t* d_Q,
    const project_in_t* d_K,
    const project_in_t* d_V,
    project_out_t* d_O,
    int B,
    int H,
    int N,
    int d,
    float scale,
    bool causal
) {
    const float scale_l2 = scale * PROJECT_LOG2E;
    switch (d) {
        case 32:
            launch_flash_attention_core_hdim<32, UseVecLoads>(
                d_Q, d_K, d_V, d_O, B, H, N, scale_l2, causal
            );
            break;
        case 64:
            launch_flash_attention_core_hdim<64, UseVecLoads>(
                d_Q, d_K, d_V, d_O, B, H, N, scale_l2, causal
            );
            break;
        case 128:
            launch_flash_attention_core_hdim<128, UseVecLoads>(
                d_Q, d_K, d_V, d_O, B, H, N, scale_l2, causal
            );
            break;
        default:
            fprintf(stderr, "Unsupported head dimension d=%d in FA1 launcher.\n", d);
            exit(EXIT_FAILURE);
    }
}

inline void launch_flash_attention_core(
    const project_in_t* d_Q,
    const project_in_t* d_K,
    const project_in_t* d_V,
    project_out_t* d_O,
    int B,
    int H,
    int N,
    int d,
    float scale,
    bool causal
) {
    check_supported_head_dim(d);
    launch_flash_attention_core_impl<true>(
        d_Q, d_K, d_V, d_O, B, H, N, d, scale, causal
    );
}

inline void launch_flash_attention_core_no_vectorized_loads(
    const project_in_t* d_Q,
    const project_in_t* d_K,
    const project_in_t* d_V,
    project_out_t* d_O,
    int B,
    int H,
    int N,
    int d,
    float scale,
    bool causal
) {
    check_supported_head_dim(d);
    launch_flash_attention_core_impl<false>(
        d_Q, d_K, d_V, d_O, B, H, N, d, scale, causal
    );
}

}  // namespace project_flash
