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

    int kv_p_elems = PROJECT_BLOCK_N * d;
    if (PROJECT_BLOCK_M * PROJECT_BLOCK_N > kv_p_elems)
        kv_p_elems = PROJECT_BLOCK_M * PROJECT_BLOCK_N;

    project_in_t* s_q = reinterpret_cast<project_in_t*>(smem_raw);
    project_in_t* s_kt = s_q + PROJECT_BLOCK_M * d;
    project_in_t* s_v = s_kt + kv_p_elems;
    float* s_scores = reinterpret_cast<float*>(s_v + PROJECT_BLOCK_N * d);
    float* s_o = s_scores + PROJECT_BLOCK_M * PROJECT_BLOCK_N;
    float* s_m = s_o + PROJECT_BLOCK_M * d;
    float* s_l = s_m + PROJECT_BLOCK_M;
    project_in_t* s_p = s_kt;

    project_in_t* s_q_warp = s_q + warp_id * PROJECT_TILE * d;
    float* s_scores_warp = s_scores + warp_id * PROJECT_TILE * PROJECT_BLOCK_N;
    float* s_o_warp = s_o + warp_id * PROJECT_TILE * d;
    project_in_t* s_p_warp = s_p + warp_id * PROJECT_TILE * PROJECT_BLOCK_N;

    const project_in_t zero = __float2half(0.0f);
    for (int idx = threadIdx.x; idx < PROJECT_BLOCK_M * d; idx += blockDim.x) {
        int row = idx / d;
        int col = idx % d;
        int global_row = q_block_start + row;
        s_q[idx] = (global_row < N) ? q_base[global_row * d + col] : zero;
    }
    for (int idx = threadIdx.x; idx < PROJECT_BLOCK_M * d; idx += blockDim.x) {
        s_o[idx] = 0.0f;
    }
    for (int idx = threadIdx.x; idx < PROJECT_BLOCK_M; idx += blockDim.x) {
        s_m[idx] = -FLT_MAX;
        s_l[idx] = 0.0f;
    }
    __syncthreads();

    int row_in_tile = lane % PROJECT_TILE;
    int col_half = lane / PROJECT_TILE;
    int half_cols = PROJECT_BLOCK_N / 2;
    int col_start = col_half * half_cols;
    int half_d = d / 2;
    int d_start = col_half * half_d;

    int num_kv_tiles = cdiv(N, PROJECT_BLOCK_N);
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * PROJECT_BLOCK_N;

        load_kv_block(s_kt, s_v, k_base, v_base, kv_start, N, d);
        __syncthreads();

        compute_score_block_tensor_core(s_scores_warp, s_q_warp, s_kt, d, scale);
        __syncthreads();

        {
            int global_row = q_start + row_in_tile;
            bool row_valid = (global_row < N);
            int m_idx = warp_id * PROJECT_TILE + row_in_tile;

            float local_max = -FLT_MAX;
            if (row_valid) {
                for (int j = col_start; j < col_start + half_cols; j++) {
                    int kv_idx = kv_start + j;
                    if (kv_idx < N && (!causal || kv_idx <= global_row)) {
                        local_max = fmaxf(local_max,
                            s_scores_warp[row_in_tile * PROJECT_BLOCK_N + j]);
                    }
                }
            }
            float partner = __shfl_xor_sync(0xFFFFFFFF, local_max, PROJECT_TILE);
            float row_max = fmaxf(local_max, partner);

            float old_m = s_m[m_idx];
            float new_m = fmaxf(old_m, row_max);
            float alpha = expf(old_m - new_m);

            if (row_valid) {
                for (int dd = d_start; dd < d_start + half_d; dd++) {
                    s_o_warp[row_in_tile * d + dd] *= alpha;
                }
            }

            float local_sum = 0.0f;
            if (row_valid) {
                for (int j = col_start; j < col_start + half_cols; j++) {
                    int kv_idx = kv_start + j;
                    float p_val;
                    if (kv_idx >= N || (causal && kv_idx > global_row)) {
                        p_val = 0.0f;
                    } else {
                        p_val = expf(
                            s_scores_warp[row_in_tile * PROJECT_BLOCK_N + j] - new_m);
                    }
                    local_sum += p_val;
                    s_p_warp[row_in_tile * PROJECT_BLOCK_N + j] = __float2half(p_val);
                }
            } else {
                for (int j = col_start; j < col_start + half_cols; j++) {
                    s_p_warp[row_in_tile * PROJECT_BLOCK_N + j] = zero;
                }
            }

            partner = __shfl_xor_sync(0xFFFFFFFF, local_sum, PROJECT_TILE);
            float row_sum = local_sum + partner;

            if (col_half == 0) {
                s_l[m_idx] = s_l[m_idx] * alpha + row_sum;
                s_m[m_idx] = new_m;
            }
        }
        __syncwarp();

        #pragma unroll
        for (int dd = 0; dd < d; dd += PROJECT_TILE) {
            wmma::fragment<wmma::accumulator, PROJECT_TILE, PROJECT_TILE,
                           PROJECT_TILE, float> c_frag;
            wmma::load_matrix_sync(c_frag, s_o_warp + dd, d,
                                   wmma::mem_row_major);

            #pragma unroll
            for (int kk = 0; kk < PROJECT_BLOCK_N; kk += PROJECT_TILE) {
                wmma::fragment<wmma::matrix_a, PROJECT_TILE, PROJECT_TILE,
                               PROJECT_TILE, project_in_t,
                               wmma::row_major> p_frag;
                wmma::fragment<wmma::matrix_b, PROJECT_TILE, PROJECT_TILE,
                               PROJECT_TILE, project_in_t,
                               wmma::row_major> v_frag;

                wmma::load_matrix_sync(p_frag, s_p_warp + kk,
                                       PROJECT_BLOCK_N);
                wmma::load_matrix_sync(v_frag, s_v + kk * d + dd, d);

                wmma::mma_sync(c_frag, p_frag, v_frag, c_frag);
            }

            wmma::store_matrix_sync(s_o_warp + dd, c_frag, d,
                                    wmma::mem_row_major);
        }

        __syncthreads();
    }

    for (int idx = threadIdx.x; idx < PROJECT_BLOCK_M * d; idx += blockDim.x) {
        int row = idx / d;
        int col = idx % d;
        int global_row = q_block_start + row;
        if (global_row < N) {
            float l = s_l[row];
            float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
            o_base[global_row * d + col] = s_o[row * d + col] * inv_l;
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
    int kv_p_elems = PROJECT_BLOCK_N * d;
    if (PROJECT_BLOCK_M * PROJECT_BLOCK_N > kv_p_elems)
        kv_p_elems = PROJECT_BLOCK_M * PROJECT_BLOCK_N;
    size_t smem_bytes = (PROJECT_BLOCK_M * d) * sizeof(project_in_t);
    smem_bytes += kv_p_elems * sizeof(project_in_t);
    smem_bytes += (PROJECT_BLOCK_N * d) * sizeof(project_in_t);
    smem_bytes += (PROJECT_BLOCK_M * PROJECT_BLOCK_N) * sizeof(float);
    smem_bytes += (PROJECT_BLOCK_M * d) * sizeof(float);
    smem_bytes += PROJECT_BLOCK_M * sizeof(float);
    smem_bytes += PROJECT_BLOCK_M * sizeof(float);

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
