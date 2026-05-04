#pragma once

#include "project_flash_core.cuh"

namespace project_flash {

template <bool DoubleBuffer>
__global__ void flash_attention_splitkv_partial_kernel(
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
    int q_tile = blockIdx.x;
    int lane = threadIdx.x;
    int q_start = q_tile * PROJECT_TILE;
    int q_end = q_start + PROJECT_TILE - 1;
    if (q_end >= N) {
        q_end = N - 1;
    }

    const project_in_t* q_base = Q + batch_head * N * d;
    const project_in_t* k_base = K + batch_head * N * d;
    const project_in_t* v_base = V + batch_head * N * d;

    extern __shared__ unsigned char smem_raw[];
    project_in_t* s_q = reinterpret_cast<project_in_t*>(smem_raw);
    project_in_t* s_kt0 = s_q + PROJECT_TILE * d;
    project_in_t* s_v0 = s_kt0 + PROJECT_TILE * d;
    project_in_t* s_kt1 = DoubleBuffer ? (s_v0 + PROJECT_TILE * d) : nullptr;
    project_in_t* s_v1 = DoubleBuffer ? (s_kt1 + PROJECT_TILE * d) : nullptr;
    float* s_scores = reinterpret_cast<float*>(
        DoubleBuffer ? (s_v1 + PROJECT_TILE * d) : (s_v0 + PROJECT_TILE * d)
    );
    float* s_o = s_scores + PROJECT_TILE * PROJECT_TILE;
    float* s_m = s_o + PROJECT_TILE * d;
    float* s_l = s_m + PROJECT_TILE;

    load_q_tile(s_q, q_base, q_start, N, d);
    for (int idx = lane; idx < PROJECT_TILE * d; idx += blockDim.x) {
        s_o[idx] = 0.0f;
    }
    if (lane < PROJECT_TILE) {
        s_m[lane] = -FLT_MAX;
        s_l[lane] = 0.0f;
    }
    __syncthreads();

    int num_kv_tiles = cdiv(N, PROJECT_TILE);
    int tiles_per_split = cdiv(num_kv_tiles, num_splits);
    int kv_tile_begin = split_idx * tiles_per_split;
    int kv_tile_end = (split_idx + 1) * tiles_per_split;
    if (kv_tile_end > num_kv_tiles) {
        kv_tile_end = num_kv_tiles;
    }
    if (causal) {
        int causal_limit = cdiv(q_end + 1, PROJECT_TILE);
        if (kv_tile_end > causal_limit) {
            kv_tile_end = causal_limit;
        }
    }

    if constexpr (DoubleBuffer) {
        if (kv_tile_begin < kv_tile_end) {
            load_kv_tile(
                s_kt0, s_v0, k_base, v_base, kv_tile_begin * PROJECT_TILE, N, d
            );
        }
        __syncthreads();
    }

    for (int kv_tile = kv_tile_begin; kv_tile < kv_tile_end; kv_tile++) {
        int kv_start = kv_tile * PROJECT_TILE;
        project_in_t* s_kt_cur = s_kt0;
        project_in_t* s_v_cur = s_v0;

        if constexpr (DoubleBuffer) {
            int local_idx = kv_tile - kv_tile_begin;
            int cur_buf = local_idx & 1;
            int next_buf = 1 - cur_buf;
            s_kt_cur = (cur_buf == 0) ? s_kt0 : s_kt1;
            s_v_cur = (cur_buf == 0) ? s_v0 : s_v1;

            if (kv_tile + 1 < kv_tile_end) {
                int next_start = (kv_tile + 1) * PROJECT_TILE;
                project_in_t* s_kt_next = (next_buf == 0) ? s_kt0 : s_kt1;
                project_in_t* s_v_next = (next_buf == 0) ? s_v0 : s_v1;
                load_kv_tile(s_kt_next, s_v_next, k_base, v_base, next_start, N, d);
            }
        } else {
            load_kv_tile(s_kt0, s_v0, k_base, v_base, kv_start, N, d);
        }

        __syncthreads();
        compute_score_tile_tensor_core(s_scores, s_q, s_kt_cur, d, scale);
        __syncthreads();

        if (lane < PROJECT_TILE) {
            int row_in_tile = lane;
            int row = q_start + row_in_tile;
            if (row < N) {
                float row_max = -FLT_MAX;
                for (int j = 0; j < PROJECT_TILE; j++) {
                    int kv_idx = kv_start + j;
                    float score = -FLT_MAX;
                    if (kv_idx < N && (!causal || kv_idx <= row)) {
                        score = s_scores[row_in_tile * PROJECT_TILE + j];
                    }
                    row_max = fmaxf(row_max, score);
                }

                if (row_max > -FLT_MAX) {
                    float m_old = s_m[row_in_tile];
                    float m_new = fmaxf(m_old, row_max);
                    float alpha = expf(m_old - m_new);

                    for (int dd = 0; dd < d; dd++) {
                        s_o[row_in_tile * d + dd] *= alpha;
                    }

                    float l_new = s_l[row_in_tile] * alpha;
                    for (int j = 0; j < PROJECT_TILE; j++) {
                        int kv_idx = kv_start + j;
                        if (kv_idx >= N || (causal && kv_idx > row)) {
                            continue;
                        }
                        float p = expf(s_scores[row_in_tile * PROJECT_TILE + j] - m_new);
                        l_new += p;
                        for (int dd = 0; dd < d; dd++) {
                            s_o[row_in_tile * d + dd] +=
                                p * __half2float(s_v_cur[j * d + dd]);
                        }
                    }

                    s_m[row_in_tile] = m_new;
                    s_l[row_in_tile] = l_new;
                }
            }
        }

        __syncthreads();
    }

    if (lane < PROJECT_TILE) {
        int row = q_start + lane;
        if (row < N) {
            int split_row_idx = ((split_idx * gridDim.z + batch_head) * N) + row;
            partial_m[split_row_idx] = s_m[lane];
            partial_l[split_row_idx] = s_l[lane];
            int o_offset = split_row_idx * d;
            for (int dd = 0; dd < d; dd++) {
                partial_o[o_offset + dd] = s_o[lane * d + dd];
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
    int q_tile = blockIdx.x;
    int lane = threadIdx.x;
    int q_start = q_tile * PROJECT_TILE;
    int row = q_start + lane;
    if (lane >= PROJECT_TILE || row >= N) {
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

template <bool DoubleBuffer>
inline int choose_splitkv_splits(int B, int H, int N) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int num_q_tiles = cdiv(N, PROJECT_TILE);
    int num_kv_tiles = cdiv(N, PROJECT_TILE);
    int effective_sms = prop.multiProcessorCount * 2;
    return project_num_splits_heuristic(B * H * num_q_tiles, effective_sms, num_kv_tiles, 8);
}

template <bool DoubleBuffer>
inline size_t splitkv_workspace_bytes(int B, int H, int N, int d, int num_splits) {
    return static_cast<size_t>(num_splits) * B * H * N * (2 * sizeof(float) + d * sizeof(float));
}

template <bool DoubleBuffer>
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
    int num_q_tiles = cdiv(N, PROJECT_TILE);
    int num_splits = choose_splitkv_splits<DoubleBuffer>(B, H, N);
    if (num_splits <= 1) {
        launch_flash_attention_core<false, DoubleBuffer>(
            d_Q, d_K, d_V, d_O, B, H, N, d, scale, causal
        );
        return;
    }

    float* partial_m = nullptr;
    float* partial_l = nullptr;
    float* partial_o = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&partial_m), (size_t)num_splits * BH * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&partial_l), (size_t)num_splits * BH * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&partial_o), (size_t)num_splits * BH * N * d * sizeof(float)));

    size_t partial_smem = (PROJECT_TILE * d) * sizeof(project_in_t);
    partial_smem += (DoubleBuffer ? 2 : 1) * (PROJECT_TILE * d) * sizeof(project_in_t);
    partial_smem += (DoubleBuffer ? 2 : 1) * (PROJECT_TILE * d) * sizeof(project_in_t);
    partial_smem += (PROJECT_TILE * PROJECT_TILE) * sizeof(float);
    partial_smem += (PROJECT_TILE * d) * sizeof(float);
    partial_smem += 2 * PROJECT_TILE * sizeof(float);

    dim3 block(32);
    dim3 grid_partial(num_q_tiles, num_splits, BH);
    flash_attention_splitkv_partial_kernel<DoubleBuffer><<<grid_partial, block, partial_smem>>>(
        d_Q, d_K, d_V, partial_m, partial_l, partial_o, num_splits, N, d, scale, causal
    );
    CUDA_CHECK(cudaGetLastError());

    dim3 grid_combine(num_q_tiles, 1, BH);
    flash_attention_splitkv_combine_kernel<<<grid_combine, block>>>(
        partial_m, partial_l, partial_o, d_O, num_splits, N, d
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(partial_m));
    CUDA_CHECK(cudaFree(partial_l));
    CUDA_CHECK(cudaFree(partial_o));
}

}  // namespace project_flash
