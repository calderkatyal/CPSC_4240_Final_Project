#pragma once

#include "flash_attn.cuh"

#include <mma.h>

namespace project_flash {

namespace wmma = nvcuda::wmma;

__device__ inline void load_q_tile(
    project_in_t* s_q,
    const project_in_t* q_base,
    int q_start,
    int N,
    int d
) {
    const project_in_t zero = __float2half(0.0f);
    for (int idx = threadIdx.x; idx < PROJECT_TILE * d; idx += blockDim.x) {
        int row = idx / d;
        int col = idx % d;
        int global_row = q_start + row;
        s_q[idx] = (global_row < N) ? q_base[global_row * d + col] : zero;
    }
}

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

static __global__ void flash_attention_core_kernel(
    const project_in_t* __restrict__ Q,
    const project_in_t* __restrict__ K,
    const project_in_t* __restrict__ V,
    project_out_t* __restrict__ O,
    int N,
    int d,
    float scale,
    bool causal
) {
    int batch_head = blockIdx.z;
    int warp_id = threadIdx.x / PROJECT_WARP_SIZE;
    int lane = threadIdx.x % PROJECT_WARP_SIZE;
    int q_block_start = blockIdx.x * PROJECT_BLOCK_M;
    int q_start = q_block_start + warp_id * PROJECT_TILE;

    const project_in_t* q_base = Q + batch_head * N * d;
    const project_in_t* k_base = K + batch_head * N * d;
    const project_in_t* v_base = V + batch_head * N * d;
    project_out_t* o_base = O + batch_head * N * d;

    extern __shared__ unsigned char smem_raw[];
    project_in_t* s_q = reinterpret_cast<project_in_t*>(smem_raw);
    project_in_t* s_kt0 = s_q + PROJECT_BLOCK_M * d;
    project_in_t* s_v0 = s_kt0 + PROJECT_BLOCK_N * d;
    float* s_scores = reinterpret_cast<float*>(s_v0 + PROJECT_BLOCK_N * d);

    const project_in_t zero = __float2half(0.0f);
    for (int idx = threadIdx.x; idx < PROJECT_BLOCK_M * d; idx += blockDim.x) {
        int row = idx / d;
        int col = idx % d;
        int global_row = q_block_start + row;
        s_q[idx] = (global_row < N) ? q_base[global_row * d + col] : zero;
    }
    __syncthreads();

    project_in_t* s_q_warp = s_q + warp_id * PROJECT_TILE * d;
    float* s_scores_warp = s_scores + warp_id * PROJECT_TILE * PROJECT_BLOCK_N;

    float row_m = -FLT_MAX;
    float row_l = 0.0f;
    float row_o[PROJECT_MAX_D];
    #pragma unroll
    for (int dd = 0; dd < PROJECT_MAX_D; dd++) {
        row_o[dd] = 0.0f;
    }

    int num_kv_tiles = cdiv(N, PROJECT_BLOCK_N);
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * PROJECT_BLOCK_N;
        load_kv_block(s_kt0, s_v0, k_base, v_base, kv_start, N, d);

        __syncthreads();
        compute_score_block_tensor_core(s_scores_warp, s_q_warp, s_kt0, d, scale);
        __syncwarp();

        if (lane < PROJECT_TILE) {
            int row_in_tile = lane;
            int row = q_start + row_in_tile;
            if (row < N) {
                float row_max = -FLT_MAX;
                for (int j = 0; j < PROJECT_BLOCK_N; j++) {
                    int kv_idx = kv_start + j;
                    float score = -FLT_MAX;
                    if (kv_idx < N && (!causal || kv_idx <= row)) {
                        score = s_scores_warp[row_in_tile * PROJECT_BLOCK_N + j];
                    }
                    row_max = fmaxf(row_max, score);
                }

                float m_new = fmaxf(row_m, row_max);
                float alpha = expf(row_m - m_new);

                for (int dd = 0; dd < d; dd++) {
                    row_o[dd] *= alpha;
                }

                float l_new = row_l * alpha;
                for (int j = 0; j < PROJECT_BLOCK_N; j++) {
                    int kv_idx = kv_start + j;
                    if (kv_idx >= N || (causal && kv_idx > row)) {
                        continue;
                    }
                    float p = expf(s_scores_warp[row_in_tile * PROJECT_BLOCK_N + j] - m_new);
                    l_new += p;
                    for (int dd = 0; dd < d; dd++) {
                        row_o[dd] += p * __half2float(s_v0[j * d + dd]);
                    }
                }

                row_m = m_new;
                row_l = l_new;
            }
        }

        __syncthreads();
    }

    if (lane < PROJECT_TILE) {
        int row_in_tile = lane;
        int row = q_start + row_in_tile;
        if (row < N) {
            float inv_l = 1.0f / row_l;
            for (int dd = 0; dd < d; dd++) {
                o_base[row * d + dd] = row_o[dd] * inv_l;
            }
        }
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

    int BH = B * H;
    int num_q_tiles = cdiv(N, PROJECT_BLOCK_M);
    size_t smem_bytes = (PROJECT_BLOCK_M * d) * sizeof(project_in_t);
    smem_bytes += (PROJECT_BLOCK_N * d) * sizeof(project_in_t);
    smem_bytes += (PROJECT_BLOCK_N * d) * sizeof(project_in_t);
    smem_bytes += (PROJECT_Q_WARPS * PROJECT_TILE * PROJECT_BLOCK_N) * sizeof(float);

    dim3 block(PROJECT_THREADS);
    dim3 grid(num_q_tiles, 1, BH);

    if (smem_bytes >= 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            flash_attention_core_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes
        ));
    }

    flash_attention_core_kernel<<<grid, block, smem_bytes>>>(
        d_Q, d_K, d_V, d_O, N, d, scale, causal
    );

    CUDA_CHECK(cudaGetLastError());
}

}  // namespace project_flash
