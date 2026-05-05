#pragma once

#include "project_flash_core.cuh"

namespace project_flash {

static __global__ void flash_attention_splitkv_partial_kernel(
    const project_in_t* __restrict__ Q,
    const project_in_t* __restrict__ K,
    const project_in_t* __restrict__ V,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_o,
    int num_splits,
    int N,
    int d,
    float scale,
    bool causal
) {
    int batch_head = blockIdx.z;
    int split_idx = blockIdx.y;
    int warp_id = threadIdx.x / PROJECT_WARP_SIZE;
    int lane = threadIdx.x % PROJECT_WARP_SIZE;
    int q_block_start = blockIdx.x * PROJECT_BLOCK_M;
    int q_start = q_block_start + warp_id * PROJECT_TILE;
    int q_end = q_block_start + PROJECT_BLOCK_M - 1;
    if (q_end >= N) { q_end = N - 1; }

    const project_in_t* q_base = Q + batch_head * N * d;
    const project_in_t* k_base = K + batch_head * N * d;
    const project_in_t* v_base = V + batch_head * N * d;

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
    int tiles_per_split = cdiv(num_kv_tiles, num_splits);
    int kv_tile_begin = split_idx * tiles_per_split;
    int kv_tile_end = (split_idx + 1) * tiles_per_split;
    if (kv_tile_end > num_kv_tiles) {
        kv_tile_end = num_kv_tiles;
    }
    if (causal) {
        int causal_limit = cdiv(q_end + 1, PROJECT_BLOCK_N);
        if (kv_tile_end > causal_limit) {
            kv_tile_end = causal_limit;
        }
    }

    for (int kv_tile = kv_tile_begin; kv_tile < kv_tile_end; kv_tile++) {
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

                if (row_max > -FLT_MAX) {
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
        }

        __syncthreads();
    }

    if (lane < PROJECT_TILE) {
        int row = q_start + lane;
        if (row < N) {
            int split_row_idx = ((split_idx * gridDim.z + batch_head) * N) + row;
            partial_m[split_row_idx] = row_m;
            partial_l[split_row_idx] = row_l;
            int o_offset = split_row_idx * d;
            for (int dd = 0; dd < d; dd++) {
                partial_o[o_offset + dd] = row_o[dd];
            }
        }
    }
}

static __global__ void flash_attention_splitkv_combine_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_o,
    project_out_t* __restrict__ O,
    int num_splits,
    int N,
    int d
) {
    int batch_head = blockIdx.z;
    int row = blockIdx.x * PROJECT_BLOCK_M + threadIdx.x;
    if (row >= N) {
        return;
    }

    float global_m = -FLT_MAX;
    for (int split_idx = 0; split_idx < num_splits; split_idx++) {
        int split_row_idx = ((split_idx * gridDim.z + batch_head) * N) + row;
        global_m = fmaxf(global_m, partial_m[split_row_idx]);
    }

    float global_l = 0.0f;
    for (int split_idx = 0; split_idx < num_splits; split_idx++) {
        int split_row_idx = ((split_idx * gridDim.z + batch_head) * N) + row;
        float l_local = partial_l[split_row_idx];
        if (l_local > 0.0f) {
            global_l += l_local * expf(partial_m[split_row_idx] - global_m);
        }
    }

    project_out_t* o_base = O + batch_head * N * d;
    if (global_l == 0.0f) {
        for (int dd = 0; dd < d; dd++) {
            o_base[row * d + dd] = 0.0f;
        }
        return;
    }

    for (int dd = 0; dd < d; dd++) {
        float accum = 0.0f;
        for (int split_idx = 0; split_idx < num_splits; split_idx++) {
            int split_row_idx = ((split_idx * gridDim.z + batch_head) * N) + row;
            float l_local = partial_l[split_row_idx];
            if (l_local > 0.0f) {
                accum += partial_o[split_row_idx * d + dd]
                    * expf(partial_m[split_row_idx] - global_m);
            }
        }
        o_base[row * d + dd] = accum / global_l;
    }
}

inline int choose_splitkv_splits(int B, int H, int N) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int num_q_tiles = cdiv(N, PROJECT_BLOCK_M);
    int num_kv_tiles = cdiv(N, PROJECT_BLOCK_N);
    int effective_sms = prop.multiProcessorCount * 2;
    return project_num_splits_heuristic(B * H * num_q_tiles, effective_sms, num_kv_tiles, 8);
}

inline size_t splitkv_workspace_bytes(int B, int H, int N, int d, int num_splits) {
    return static_cast<size_t>(num_splits) * B * H * N * (2 * sizeof(float) + d * sizeof(float));
}

inline void launch_flash_attention_splitkv(
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
    int num_splits = choose_splitkv_splits(B, H, N);
    if (num_splits <= 1) {
        launch_flash_attention_core(
            d_Q, d_K, d_V, d_O, B, H, N, d, scale, causal
        );
        return;
    }

    float* partial_m = nullptr;
    float* partial_l = nullptr;
    float* partial_o = nullptr;
    CUDA_CHECK(tracked_cuda_malloc(reinterpret_cast<void**>(&partial_m), (size_t)num_splits * BH * N * sizeof(float)));
    CUDA_CHECK(tracked_cuda_malloc(reinterpret_cast<void**>(&partial_l), (size_t)num_splits * BH * N * sizeof(float)));
    CUDA_CHECK(tracked_cuda_malloc(reinterpret_cast<void**>(&partial_o), (size_t)num_splits * BH * N * d * sizeof(float)));

    size_t partial_smem = (PROJECT_BLOCK_M * d) * sizeof(project_in_t);
    partial_smem += (PROJECT_BLOCK_N * d) * sizeof(project_in_t);
    partial_smem += (PROJECT_BLOCK_N * d) * sizeof(project_in_t);
    partial_smem += (PROJECT_Q_WARPS * PROJECT_TILE * PROJECT_BLOCK_N) * sizeof(float);

    dim3 block(PROJECT_THREADS);
    dim3 grid_partial(num_q_tiles, num_splits, BH);
    if (partial_smem >= 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            flash_attention_splitkv_partial_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            partial_smem
        ));
    }
    flash_attention_splitkv_partial_kernel<<<grid_partial, block, partial_smem>>>(
        d_Q, d_K, d_V, partial_m, partial_l, partial_o, num_splits, N, d, scale, causal
    );
    CUDA_CHECK(cudaGetLastError());

    dim3 grid_combine(num_q_tiles, 1, BH);
    dim3 combine_block(PROJECT_BLOCK_M);
    flash_attention_splitkv_combine_kernel<<<grid_combine, combine_block>>>(
        partial_m, partial_l, partial_o, d_O, num_splits, N, d
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(tracked_cuda_free(partial_m));
    CUDA_CHECK(tracked_cuda_free(partial_l));
    CUDA_CHECK(tracked_cuda_free(partial_o));
}

}  // namespace project_flash
