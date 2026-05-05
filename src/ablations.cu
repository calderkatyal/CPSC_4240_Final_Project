// Ablation variants for the simplified FlashAttention project.
//
// These isolate key ideas from the main FA1-style kernel:
//   1. No online softmax: keep tiling and tensor-core score tiles, but use a
//      two-pass softmax instead of the online recurrence.
//   2. No SRAM tiling: keep exact attention and online softmax, but load every
//      key/value row directly from global memory instead of staging tiles.

#include "project_flash_core.cuh"

using namespace project_flash;

namespace {

__global__ void two_pass_find_max_kernel(
    const project_in_t* __restrict__ Q,
    const project_in_t* __restrict__ K,
    float* __restrict__ row_max,
    int N,
    int d,
    float scale,
    bool causal
) {
    int batch_head = blockIdx.z;
    int q_tile = blockIdx.x;
    int lane = threadIdx.x;
    int q_start = q_tile * PROJECT_TILE;

    const project_in_t* q_base = Q + batch_head * N * d;
    const project_in_t* k_base = K + batch_head * N * d;

    extern __shared__ unsigned char smem_raw[];
    project_in_t* s_q = reinterpret_cast<project_in_t*>(smem_raw);
    project_in_t* s_kt = s_q + PROJECT_TILE * d;
    float* s_scores = reinterpret_cast<float*>(s_kt + PROJECT_TILE * d);

    load_q_tile(s_q, q_base, q_start, N, d);
    __syncthreads();

    float local_max = -FLT_MAX;
    if (lane < PROJECT_TILE) {
        int row = q_start + lane;
        if (row < N) {
            local_max = -FLT_MAX;
        }
    }

    int num_kv_tiles = cdiv(N, PROJECT_TILE);
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * PROJECT_TILE;
        load_kv_tile(s_kt, nullptr, k_base, nullptr, kv_start, N, d);
        __syncthreads();
        compute_score_tile_tensor_core(s_scores, s_q, s_kt, d, scale);
        __syncthreads();

        if (lane < PROJECT_TILE) {
            int row = q_start + lane;
            if (row < N) {
                for (int j = 0; j < PROJECT_TILE; j++) {
                    int kv_idx = kv_start + j;
                    if (kv_idx >= N || (causal && kv_idx > row)) {
                        continue;
                    }
                    local_max = fmaxf(local_max, s_scores[lane * PROJECT_TILE + j]);
                }
            }
        }

        __syncthreads();
    }

    if (lane < PROJECT_TILE) {
        int row = q_start + lane;
        if (row < N) {
            row_max[batch_head * N + row] = local_max;
        }
    }
}

__global__ void two_pass_attn_kernel(
    const project_in_t* __restrict__ Q,
    const project_in_t* __restrict__ K,
    const project_in_t* __restrict__ V,
    const float* __restrict__ row_max,
    project_out_t* __restrict__ O,
    int N,
    int d,
    float scale,
    bool causal
) {
    int batch_head = blockIdx.z;
    int q_tile = blockIdx.x;
    int lane = threadIdx.x;
    int q_start = q_tile * PROJECT_TILE;

    const project_in_t* q_base = Q + batch_head * N * d;
    const project_in_t* k_base = K + batch_head * N * d;
    const project_in_t* v_base = V + batch_head * N * d;
    project_out_t* o_base = O + batch_head * N * d;

    extern __shared__ unsigned char smem_raw[];
    project_in_t* s_q = reinterpret_cast<project_in_t*>(smem_raw);
    project_in_t* s_kt = s_q + PROJECT_TILE * d;
    project_in_t* s_v = s_kt + PROJECT_TILE * d;
    float* s_scores = reinterpret_cast<float*>(s_v + PROJECT_TILE * d);
    float* s_o = s_scores + PROJECT_TILE * PROJECT_TILE;
    float* s_l = s_o + PROJECT_TILE * d;

    load_q_tile(s_q, q_base, q_start, N, d);
    for (int idx = lane; idx < PROJECT_TILE * d; idx += blockDim.x) {
        s_o[idx] = 0.0f;
    }
    if (lane < PROJECT_TILE) {
        s_l[lane] = 0.0f;
    }
    __syncthreads();

    int num_kv_tiles = cdiv(N, PROJECT_TILE);
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * PROJECT_TILE;
        load_kv_tile(s_kt, s_v, k_base, v_base, kv_start, N, d);
        __syncthreads();
        compute_score_tile_tensor_core(s_scores, s_q, s_kt, d, scale);
        __syncthreads();

        if (lane < PROJECT_TILE) {
            int row = q_start + lane;
            if (row < N) {
                float m = row_max[batch_head * N + row];
                for (int j = 0; j < PROJECT_TILE; j++) {
                    int kv_idx = kv_start + j;
                    if (kv_idx >= N || (causal && kv_idx > row)) {
                        continue;
                    }

                    float p = expf(s_scores[lane * PROJECT_TILE + j] - m);
                    s_l[lane] += p;
                    for (int dd = 0; dd < d; dd++) {
                        s_o[lane * d + dd] += p * __half2float(s_v[j * d + dd]);
                    }
                }
            }
        }

        __syncthreads();
    }

    if (lane < PROJECT_TILE) {
        int row = q_start + lane;
        if (row < N) {
            float inv_l = 1.0f / s_l[lane];
            for (int dd = 0; dd < d; dd++) {
                o_base[row * d + dd] = s_o[lane * d + dd] * inv_l;
            }
        }
    }
}

__global__ void flash_attn_no_tiling_kernel(
    const project_in_t* __restrict__ Q,
    const project_in_t* __restrict__ K,
    const project_in_t* __restrict__ V,
    project_out_t* __restrict__ O,
    int N,
    int d,
    float scale,
    bool causal
) {
    int batch_head = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) {
        return;
    }

    const project_in_t* q_base = Q + batch_head * N * d;
    const project_in_t* k_base = K + batch_head * N * d;
    const project_in_t* v_base = V + batch_head * N * d;
    project_out_t* o_base = O + batch_head * N * d;

    float m = -FLT_MAX;
    float l = 0.0f;
    float o_acc[PROJECT_MAX_D];
    for (int dd = 0; dd < d; dd++) {
        o_acc[dd] = 0.0f;
    }

    int max_j = causal ? (row + 1) : N;
    for (int j = 0; j < max_j; j++) {
        float dot = 0.0f;
        for (int dd = 0; dd < d; dd++) {
            dot += __half2float(q_base[row * d + dd]) * __half2float(k_base[j * d + dd]);
        }

        float score = dot * scale;
        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);

        for (int dd = 0; dd < d; dd++) {
            o_acc[dd] *= alpha;
        }

        float p = expf(score - m_new);
        l = l * alpha + p;
        for (int dd = 0; dd < d; dd++) {
            o_acc[dd] += p * __half2float(v_base[j * d + dd]);
        }
        m = m_new;
    }

    float inv_l = 1.0f / l;
    for (int dd = 0; dd < d; dd++) {
        o_base[row * d + dd] = o_acc[dd] * inv_l;
    }
}

}  // namespace

void flash_attention_v1_no_online_softmax(
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
    float* d_row_max = nullptr;
    CUDA_CHECK(tracked_cuda_malloc(reinterpret_cast<void**>(&d_row_max), (size_t)BH * N * sizeof(float)));

    size_t pass1_smem = 2 * PROJECT_TILE * d * sizeof(project_in_t);
    pass1_smem += PROJECT_TILE * PROJECT_TILE * sizeof(float);
    size_t pass2_smem = 3 * PROJECT_TILE * d * sizeof(project_in_t);
    pass2_smem += PROJECT_TILE * PROJECT_TILE * sizeof(float);
    pass2_smem += PROJECT_TILE * d * sizeof(float);
    pass2_smem += PROJECT_TILE * sizeof(float);

    dim3 block(32);
    dim3 grid(num_q_tiles, 1, BH);

    two_pass_find_max_kernel<<<grid, block, pass1_smem>>>(
        d_Q, d_K, d_row_max, N, d, scale, causal
    );
    CUDA_CHECK(cudaGetLastError());

    two_pass_attn_kernel<<<grid, block, pass2_smem>>>(
        d_Q, d_K, d_V, d_row_max, d_O, N, d, scale, causal
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(tracked_cuda_free(d_row_max));
}

void flash_attention_v1_no_tiling(
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
    dim3 block(256);
    dim3 grid(cdiv(N, 256), BH);

    flash_attn_no_tiling_kernel<<<grid, block>>>(
        d_Q, d_K, d_V, d_O, N, d, scale, causal
    );

    CUDA_CHECK(cudaGetLastError());
}
