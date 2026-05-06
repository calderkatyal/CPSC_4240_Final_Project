#pragma once

#include "flash_attn.cuh"

#include <cstdint>
#include <mma.h>

namespace project_flash {

namespace wmma = nvcuda::wmma;

// Shared-memory padding (in half elements) for the tensor-core paths.
// The padded leading dimensions must remain multiples of 8 half values so that
// WMMA load_matrix_sync stays legal, while also shifting rows away from the
// worst shared-memory bank-alignment pattern.
constexpr int SMEM_PAD = 8;
constexpr float LOG2E_F = 1.4426950408889634f;

__device__ inline float fast_exp(float x) {
    return exp2f(x * LOG2E_F);
}

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
// Core FA1 kernel — register-resident O and softmax
//
// Key optimisations vs. the previous version:
//   1. O accumulator lives in WMMA register fragments, not shared memory.
//      Eliminates 16 KB smem and all per-KV-block O load/store traffic.
//   2. Scores stay in register fragments; softmax runs on registers using
//      the documented m16n16k16 accumulator layout (sm_70+).
//      Eliminates 16 KB smem for s_scores.
//   3. Shared-memory strides are padded by SMEM_PAD to avoid bank conflicts
//      (original stride d=64 ≡ 0 mod 32 → 16-way conflicts).
//   4. Total smem drops from ~56 KB to ~25 KB, raising occupancy from
//      ~2 blocks/SM to ~5 blocks/SM.
// ========================================================================

template<int HEAD_DIM, bool IS_CAUSAL>
static __global__ void flash_attention_core_kernel(
    const project_in_t* __restrict__ Q,
    const project_in_t* __restrict__ K,
    const project_in_t* __restrict__ V,
    project_out_t* __restrict__ O,
    int N,
    float scale
) {
    const int batch_head = blockIdx.z;
    const int warp_id    = threadIdx.x / PROJECT_WARP_SIZE;
    const int lane       = threadIdx.x % PROJECT_WARP_SIZE;
    const int q_block_start = blockIdx.x * PROJECT_BLOCK_M;
    const int q_start       = q_block_start + warp_id * PROJECT_TILE;

    // ---- WMMA m16n16k16 float-accumulator element-to-matrix mapping ----
    // group g = lane/4  (0-7),  thread-in-group p = lane%4  (0-3)
    //   elts 0,1 → row g,   cols p*2,   p*2+1
    //   elts 2,3 → row g+8, cols p*2,   p*2+1
    //   elts 4,5 → row g,   cols p*2+8, p*2+9
    //   elts 6,7 → row g+8, cols p*2+8, p*2+9
    const int fg = lane / 4;
    const int fp = lane % 4;
    const int frow0 = fg;            // first row this thread owns
    const int frow1 = fg + 8;        // second row this thread owns
    const int global_row0 = q_start + frow0;
    const int global_row1 = q_start + frow1;

    const project_in_t* q_base = Q + batch_head * N * HEAD_DIM;
    const project_in_t* k_base = K + batch_head * N * HEAD_DIM;
    const project_in_t* v_base = V + batch_head * N * HEAD_DIM;
    project_out_t*      o_base = O + batch_head * N * HEAD_DIM;

    // Padded strides
    constexpr int d_padded = HEAD_DIM + SMEM_PAD;
    constexpr int half_dim_pairs = HEAD_DIM / 2;
    const int bn_padded = PROJECT_BLOCK_N + SMEM_PAD;

    // Buffer for s_kt / s_p must fit both layouts
    constexpr int kv_elems = PROJECT_BLOCK_N * d_padded;
    constexpr int p_elems = PROJECT_BLOCK_M * (PROJECT_BLOCK_N + SMEM_PAD);
    int kv_buf_elems = kv_elems;
    if (p_elems > kv_buf_elems)
        kv_buf_elems = p_elems;

    // ---- Shared memory layout ----
    extern __shared__ __align__(32) unsigned char smem_raw[];
    project_in_t* s_q  = reinterpret_cast<project_in_t*>(smem_raw);
    project_in_t* s_kt = s_q + PROJECT_BLOCK_M * d_padded;
    project_in_t* s_v  = s_kt + kv_buf_elems;
    project_in_t* s_p  = s_kt;   // alias — reused after scores are consumed

    project_in_t* s_q_warp = s_q + warp_id * PROJECT_TILE * d_padded;
    project_in_t* s_p_warp = s_p + warp_id * PROJECT_TILE * bn_padded;

    const project_in_t zero = __float2half(0.0f);
    const uint32_t zero2 = 0;

    // ---- Load Q to shared memory (vectorized, with pad columns zeroed) ----
    for (int idx = threadIdx.x; idx < PROJECT_BLOCK_M * half_dim_pairs;
         idx += blockDim.x) {
        int row = idx / half_dim_pairs;
        int pair = idx % half_dim_pairs;
        int gr  = q_block_start + row;
        uint32_t val = zero2;
        if (gr < N) {
            val = reinterpret_cast<const uint32_t*>(
                q_base + gr * HEAD_DIM)[pair];
        }
        reinterpret_cast<uint32_t*>(s_q + row * d_padded)[pair] = val;
    }
    for (int idx = threadIdx.x; idx < PROJECT_BLOCK_M * SMEM_PAD;
         idx += blockDim.x) {
        int row = idx / SMEM_PAD;
        int col = HEAD_DIM + (idx % SMEM_PAD);
        s_q[row * d_padded + col] = zero;
    }

    // ---- Register-resident O accumulators ----
    constexpr int num_o_frags = HEAD_DIM / PROJECT_TILE;
    wmma::fragment<wmma::accumulator, PROJECT_TILE, PROJECT_TILE,
                   PROJECT_TILE, float> o_frag[num_o_frags];
    for (int f = 0; f < num_o_frags; f++)
        wmma::fill_fragment(o_frag[f], 0.0f);

    // Per-thread running max / sum for the two rows this thread tracks
    float m0 = -FLT_MAX, m1 = -FLT_MAX;
    float l0 = 0.0f,     l1 = 0.0f;

    __syncthreads();

    // ---- KV iteration bounds (with causal early termination) ----
    int num_kv_tiles = cdiv(N, PROJECT_BLOCK_N);
    if (IS_CAUSAL) {
        int q_end = q_block_start + PROJECT_BLOCK_M - 1;
        if (q_end >= N) q_end = N - 1;
        int causal_limit = cdiv(q_end + 1, PROJECT_BLOCK_N);
        if (causal_limit < num_kv_tiles) num_kv_tiles = causal_limit;
    }

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        const int kv_start = kv_tile * PROJECT_BLOCK_N;

        // ============================================================
        // 1.  Load K and V to shared memory
        // ============================================================
        for (int idx = threadIdx.x; idx < PROJECT_BLOCK_N * half_dim_pairs;
             idx += blockDim.x) {
            int row = idx / half_dim_pairs;
            int pair = idx % half_dim_pairs;
            int gr  = kv_start + row;
            uint32_t kv = zero2, vv = zero2;
            if (gr < N) {
                kv = reinterpret_cast<const uint32_t*>(
                    k_base + gr * HEAD_DIM)[pair];
                vv = reinterpret_cast<const uint32_t*>(
                    v_base + gr * HEAD_DIM)[pair];
            }
            reinterpret_cast<uint32_t*>(s_kt + row * d_padded)[pair] = kv;
            reinterpret_cast<uint32_t*>(s_v + row * d_padded)[pair] = vv;
        }
        for (int idx = threadIdx.x; idx < PROJECT_BLOCK_N * SMEM_PAD;
             idx += blockDim.x) {
            int row = idx / SMEM_PAD;
            int col = HEAD_DIM + (idx % SMEM_PAD);
            s_kt[row * d_padded + col] = zero;
            s_v[row * d_padded + col] = zero;
        }
        __syncthreads();

        // ============================================================
        // 2.  S = Q @ K^T   →  register score fragments
        // ============================================================
        wmma::fragment<wmma::accumulator, PROJECT_TILE, PROJECT_TILE,
                       PROJECT_TILE, float> sf[PROJECT_K_TILES_PER_BLOCK];

        #pragma unroll
        for (int tn = 0; tn < PROJECT_K_TILES_PER_BLOCK; tn++) {
            wmma::fill_fragment(sf[tn], 0.0f);
            #pragma unroll
            for (int k0 = 0; k0 < HEAD_DIM; k0 += PROJECT_TILE) {
                wmma::fragment<wmma::matrix_a, PROJECT_TILE, PROJECT_TILE,
                               PROJECT_TILE, project_in_t,
                               wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, PROJECT_TILE, PROJECT_TILE,
                               PROJECT_TILE, project_in_t,
                               wmma::col_major> b;
                wmma::load_matrix_sync(a, s_q_warp + k0, d_padded);
                wmma::load_matrix_sync(
                    b, s_kt + tn * PROJECT_TILE * d_padded + k0, d_padded);
                wmma::mma_sync(sf[tn], a, b, sf[tn]);
            }
            #pragma unroll
            for (int i = 0; i < sf[tn].num_elements; i++)
                sf[tn].x[i] *= scale;
        }

        // All warps must finish reading s_kt before P overwrites it
        __syncthreads();

        // ============================================================
        // 3.  Online softmax  (entirely in registers + warp shuffles)
        // ============================================================
        const bool r0v = (global_row0 < N);
        const bool r1v = (global_row1 < N);

        // --- row max ---
        float mx0 = -FLT_MAX, mx1 = -FLT_MAX;
        const bool kv_tile_no_tail = (kv_start + PROJECT_BLOCK_N <= N);
        const bool full_valid0 = r0v && kv_tile_no_tail
            && (!IS_CAUSAL || kv_start + PROJECT_BLOCK_N - 1 <= global_row0);
        const bool full_valid1 = r1v && kv_tile_no_tail
            && (!IS_CAUSAL || kv_start + PROJECT_BLOCK_N - 1 <= global_row1);
        #pragma unroll
        for (int tn = 0; tn < PROJECT_K_TILES_PER_BLOCK; tn++) {
            int bc = kv_start + tn * PROJECT_TILE;
            int c0 = bc + fp * 2, c1 = c0 + 1;
            int c4 = bc + fp * 2 + 8, c5 = c4 + 1;
            if (full_valid0) {
                mx0 = fmaxf(mx0, sf[tn].x[0]);
                mx0 = fmaxf(mx0, sf[tn].x[1]);
                mx0 = fmaxf(mx0, sf[tn].x[4]);
                mx0 = fmaxf(mx0, sf[tn].x[5]);
            } else if (r0v) {
                if (c0 < N && (!IS_CAUSAL || c0 <= global_row0)) mx0 = fmaxf(mx0, sf[tn].x[0]);
                if (c1 < N && (!IS_CAUSAL || c1 <= global_row0)) mx0 = fmaxf(mx0, sf[tn].x[1]);
                if (c4 < N && (!IS_CAUSAL || c4 <= global_row0)) mx0 = fmaxf(mx0, sf[tn].x[4]);
                if (c5 < N && (!IS_CAUSAL || c5 <= global_row0)) mx0 = fmaxf(mx0, sf[tn].x[5]);
            }
            if (full_valid1) {
                mx1 = fmaxf(mx1, sf[tn].x[2]);
                mx1 = fmaxf(mx1, sf[tn].x[3]);
                mx1 = fmaxf(mx1, sf[tn].x[6]);
                mx1 = fmaxf(mx1, sf[tn].x[7]);
            } else if (r1v) {
                if (c0 < N && (!IS_CAUSAL || c0 <= global_row1)) mx1 = fmaxf(mx1, sf[tn].x[2]);
                if (c1 < N && (!IS_CAUSAL || c1 <= global_row1)) mx1 = fmaxf(mx1, sf[tn].x[3]);
                if (c4 < N && (!IS_CAUSAL || c4 <= global_row1)) mx1 = fmaxf(mx1, sf[tn].x[6]);
                if (c5 < N && (!IS_CAUSAL || c5 <= global_row1)) mx1 = fmaxf(mx1, sf[tn].x[7]);
            }
        }
        // Reduce across p = 0..3 (lanes within same group share a row)
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFF, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFF, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFF, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFF, mx1, 2));

        const float nm0 = fmaxf(m0, mx0);
        const float nm1 = fmaxf(m1, mx1);
        const float a0  = fast_exp(m0 - nm0);
        const float a1  = fast_exp(m1 - nm1);

        // --- rescale O accumulators ---
        #pragma unroll
        for (int f = 0; f < num_o_frags; f++) {
            o_frag[f].x[0] *= a0; o_frag[f].x[1] *= a0;
            o_frag[f].x[2] *= a1; o_frag[f].x[3] *= a1;
            o_frag[f].x[4] *= a0; o_frag[f].x[5] *= a0;
            o_frag[f].x[6] *= a1; o_frag[f].x[7] *= a1;
        }

        // --- exp + write P to shared memory ---
        float sum0 = 0.0f, sum1 = 0.0f;
        #pragma unroll
        for (int tn = 0; tn < PROJECT_K_TILES_PER_BLOCK; tn++) {
            int bc = kv_start + tn * PROJECT_TILE;
            int c0 = bc + fp * 2, c1 = c0 + 1;
            int c4 = bc + fp * 2 + 8, c5 = c4 + 1;

            float p0 = 0, p1 = 0, p4 = 0, p5 = 0;
            float p2 = 0, p3 = 0, p6 = 0, p7 = 0;
            if (full_valid0) {
                p0 = fast_exp(sf[tn].x[0] - nm0);
                p1 = fast_exp(sf[tn].x[1] - nm0);
                p4 = fast_exp(sf[tn].x[4] - nm0);
                p5 = fast_exp(sf[tn].x[5] - nm0);
            } else if (r0v) {
                if (c0 < N && (!IS_CAUSAL || c0 <= global_row0)) p0 = fast_exp(sf[tn].x[0] - nm0);
                if (c1 < N && (!IS_CAUSAL || c1 <= global_row0)) p1 = fast_exp(sf[tn].x[1] - nm0);
                if (c4 < N && (!IS_CAUSAL || c4 <= global_row0)) p4 = fast_exp(sf[tn].x[4] - nm0);
                if (c5 < N && (!IS_CAUSAL || c5 <= global_row0)) p5 = fast_exp(sf[tn].x[5] - nm0);
            }
            if (full_valid1) {
                p2 = fast_exp(sf[tn].x[2] - nm1);
                p3 = fast_exp(sf[tn].x[3] - nm1);
                p6 = fast_exp(sf[tn].x[6] - nm1);
                p7 = fast_exp(sf[tn].x[7] - nm1);
            } else if (r1v) {
                if (c0 < N && (!IS_CAUSAL || c0 <= global_row1)) p2 = fast_exp(sf[tn].x[2] - nm1);
                if (c1 < N && (!IS_CAUSAL || c1 <= global_row1)) p3 = fast_exp(sf[tn].x[3] - nm1);
                if (c4 < N && (!IS_CAUSAL || c4 <= global_row1)) p6 = fast_exp(sf[tn].x[6] - nm1);
                if (c5 < N && (!IS_CAUSAL || c5 <= global_row1)) p7 = fast_exp(sf[tn].x[7] - nm1);
            }

            sum0 += p0 + p1 + p4 + p5;
            sum1 += p2 + p3 + p6 + p7;

            int pc = tn * PROJECT_TILE;
            s_p_warp[frow0 * bn_padded + pc + fp * 2]     = __float2half(p0);
            s_p_warp[frow0 * bn_padded + pc + fp * 2 + 1] = __float2half(p1);
            s_p_warp[frow0 * bn_padded + pc + fp * 2 + 8] = __float2half(p4);
            s_p_warp[frow0 * bn_padded + pc + fp * 2 + 9] = __float2half(p5);
            s_p_warp[frow1 * bn_padded + pc + fp * 2]     = __float2half(p2);
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

        // ============================================================
        // 4.  O += P @ V   (WMMA, O stays in register fragments)
        // ============================================================
        #pragma unroll
        for (int dd = 0; dd < HEAD_DIM; dd += PROJECT_TILE) {
            #pragma unroll
            for (int kk = 0; kk < PROJECT_BLOCK_N; kk += PROJECT_TILE) {
                wmma::fragment<wmma::matrix_a, PROJECT_TILE, PROJECT_TILE,
                               PROJECT_TILE, project_in_t,
                               wmma::row_major> pf;
                wmma::fragment<wmma::matrix_b, PROJECT_TILE, PROJECT_TILE,
                               PROJECT_TILE, project_in_t,
                               wmma::row_major> vf;
                wmma::load_matrix_sync(pf, s_p_warp + kk, bn_padded);
                wmma::load_matrix_sync(
                    vf, s_v + kk * d_padded + dd, d_padded);
                wmma::mma_sync(o_frag[dd / PROJECT_TILE], pf, vf,
                               o_frag[dd / PROJECT_TILE]);
            }
        }

        __syncthreads();
    }

    // ================================================================
    // 5.  Write O / l to global memory
    // ================================================================
    const float inv_l0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
    const float inv_l1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;

    if (global_row0 < N) {
        for (int f = 0; f < num_o_frags; f++) {
            int bc = f * PROJECT_TILE;
            o_base[global_row0 * HEAD_DIM + bc + fp * 2]     = o_frag[f].x[0] * inv_l0;
            o_base[global_row0 * HEAD_DIM + bc + fp * 2 + 1] = o_frag[f].x[1] * inv_l0;
            o_base[global_row0 * HEAD_DIM + bc + fp * 2 + 8] = o_frag[f].x[4] * inv_l0;
            o_base[global_row0 * HEAD_DIM + bc + fp * 2 + 9] = o_frag[f].x[5] * inv_l0;
        }
    }
    if (global_row1 < N) {
        for (int f = 0; f < num_o_frags; f++) {
            int bc = f * PROJECT_TILE;
            o_base[global_row1 * HEAD_DIM + bc + fp * 2]     = o_frag[f].x[2] * inv_l1;
            o_base[global_row1 * HEAD_DIM + bc + fp * 2 + 1] = o_frag[f].x[3] * inv_l1;
            o_base[global_row1 * HEAD_DIM + bc + fp * 2 + 8] = o_frag[f].x[6] * inv_l1;
            o_base[global_row1 * HEAD_DIM + bc + fp * 2 + 9] = o_frag[f].x[7] * inv_l1;
        }
    }
}

template<int HEAD_DIM, bool IS_CAUSAL>
inline void launch_flash_attention_core_typed(
    const project_in_t* d_Q,
    const project_in_t* d_K,
    const project_in_t* d_V,
    project_out_t* d_O,
    int B,
    int H,
    int N,
    float scale
) {
    const int BH = B * H;
    const int num_q_tiles = cdiv(N, PROJECT_BLOCK_M);
    constexpr int d_padded = HEAD_DIM + SMEM_PAD;
    const int bn_padded = PROJECT_BLOCK_N + SMEM_PAD;

    int kv_buf_elems = PROJECT_BLOCK_N * d_padded;
    if (PROJECT_BLOCK_M * bn_padded > kv_buf_elems)
        kv_buf_elems = PROJECT_BLOCK_M * bn_padded;

    size_t smem_bytes = 0;
    smem_bytes += PROJECT_BLOCK_M * d_padded  * sizeof(project_in_t); // s_q
    smem_bytes += kv_buf_elems                * sizeof(project_in_t); // s_kt / s_p
    smem_bytes += PROJECT_BLOCK_N * d_padded  * sizeof(project_in_t); // s_v

    dim3 block(PROJECT_THREADS);
    dim3 grid(num_q_tiles, 1, BH);

    if (smem_bytes >= 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            flash_attention_core_kernel<HEAD_DIM, IS_CAUSAL>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes
        ));
    }

    flash_attention_core_kernel<HEAD_DIM, IS_CAUSAL><<<grid, block, smem_bytes>>>(
        d_Q, d_K, d_V, d_O, N, scale
    );

    CUDA_CHECK(cudaGetLastError());
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

    switch (d) {
        case 16:
            if (causal) launch_flash_attention_core_typed<16, true>(d_Q, d_K, d_V, d_O, B, H, N, scale);
            else launch_flash_attention_core_typed<16, false>(d_Q, d_K, d_V, d_O, B, H, N, scale);
            break;
        case 32:
            if (causal) launch_flash_attention_core_typed<32, true>(d_Q, d_K, d_V, d_O, B, H, N, scale);
            else launch_flash_attention_core_typed<32, false>(d_Q, d_K, d_V, d_O, B, H, N, scale);
            break;
        case 48:
            if (causal) launch_flash_attention_core_typed<48, true>(d_Q, d_K, d_V, d_O, B, H, N, scale);
            else launch_flash_attention_core_typed<48, false>(d_Q, d_K, d_V, d_O, B, H, N, scale);
            break;
        case 64:
            if (causal) launch_flash_attention_core_typed<64, true>(d_Q, d_K, d_V, d_O, B, H, N, scale);
            else launch_flash_attention_core_typed<64, false>(d_Q, d_K, d_V, d_O, B, H, N, scale);
            break;
        case 80:
            if (causal) launch_flash_attention_core_typed<80, true>(d_Q, d_K, d_V, d_O, B, H, N, scale);
            else launch_flash_attention_core_typed<80, false>(d_Q, d_K, d_V, d_O, B, H, N, scale);
            break;
        case 96:
            if (causal) launch_flash_attention_core_typed<96, true>(d_Q, d_K, d_V, d_O, B, H, N, scale);
            else launch_flash_attention_core_typed<96, false>(d_Q, d_K, d_V, d_O, B, H, N, scale);
            break;
        case 112:
            if (causal) launch_flash_attention_core_typed<112, true>(d_Q, d_K, d_V, d_O, B, H, N, scale);
            else launch_flash_attention_core_typed<112, false>(d_Q, d_K, d_V, d_O, B, H, N, scale);
            break;
        case 128:
            if (causal) launch_flash_attention_core_typed<128, true>(d_Q, d_K, d_V, d_O, B, H, N, scale);
            else launch_flash_attention_core_typed<128, false>(d_Q, d_K, d_V, d_O, B, H, N, scale);
            break;
    }
}

}  // namespace project_flash
