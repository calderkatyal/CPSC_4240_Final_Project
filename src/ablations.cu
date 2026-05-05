// Ablation variants for the simplified FlashAttention project.
//
// These isolate key ideas from the main FA1-style kernel:
//   1. No online softmax: keep tiling and tensor-core score tiles, but use a
//      two-pass softmax instead of the online recurrence.
//   2. No SRAM tiling: keep the same 64-row / 4-warp outer structure and the
//      same online-softmax recurrence, but fetch every key/value row directly
//      from global memory instead of staging K/V tiles in shared memory. In
//      this simplified codebase, removing the staged K/V tile also means the
//      score path falls back to scalar dot products instead of the WMMA helper.

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
    int warp_id = threadIdx.x / PROJECT_WARP_SIZE;
    int lane = threadIdx.x % PROJECT_WARP_SIZE;
    int q_block_start = blockIdx.x * PROJECT_BLOCK_M;
    int q_start = q_block_start + warp_id * PROJECT_TILE;

    const project_in_t* q_base = Q + batch_head * N * d;
    const project_in_t* k_base = K + batch_head * N * d;

    extern __shared__ unsigned char smem_raw[];
    project_in_t* s_q = reinterpret_cast<project_in_t*>(smem_raw);
    project_in_t* s_kt = s_q + PROJECT_BLOCK_M * d;
    float* s_scores = reinterpret_cast<float*>(s_kt + PROJECT_TILE * d);

    const project_in_t zero = __float2half(0.0f);
    for (int idx = threadIdx.x; idx < PROJECT_BLOCK_M * d; idx += blockDim.x) {
        int row = idx / d;
        int col = idx % d;
        int global_row = q_block_start + row;
        s_q[idx] = (global_row < N) ? q_base[global_row * d + col] : zero;
    }
    __syncthreads();

    project_in_t* s_q_warp = s_q + warp_id * PROJECT_TILE * d;
    float* s_scores_warp = s_scores + warp_id * PROJECT_TILE * PROJECT_TILE;

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
        compute_score_tile_tensor_core(s_scores_warp, s_q_warp, s_kt, d, scale);
        __syncwarp();

        if (lane < PROJECT_TILE) {
            int row = q_start + lane;
            if (row < N) {
                for (int j = 0; j < PROJECT_TILE; j++) {
                    int kv_idx = kv_start + j;
                    if (kv_idx >= N || (causal && kv_idx > row)) {
                        continue;
                    }
                    local_max = fmaxf(local_max, s_scores_warp[lane * PROJECT_TILE + j]);
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
    project_in_t* s_kt = s_q + PROJECT_BLOCK_M * d;
    project_in_t* s_v = s_kt + PROJECT_TILE * d;
    float* s_scores = reinterpret_cast<float*>(s_v + PROJECT_TILE * d);
    float* s_o = s_scores + PROJECT_Q_WARPS * PROJECT_TILE * PROJECT_TILE;
    float* s_l = s_o + PROJECT_BLOCK_M * d;

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
    if (threadIdx.x < PROJECT_BLOCK_M) {
        s_l[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    project_in_t* s_q_warp = s_q + warp_id * PROJECT_TILE * d;
    float* s_scores_warp = s_scores + warp_id * PROJECT_TILE * PROJECT_TILE;
    float* s_o_warp = s_o + warp_id * PROJECT_TILE * d;
    float* s_l_warp = s_l + warp_id * PROJECT_TILE;

    int num_kv_tiles = cdiv(N, PROJECT_TILE);
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * PROJECT_TILE;
        load_kv_tile(s_kt, s_v, k_base, v_base, kv_start, N, d);
        __syncthreads();
        compute_score_tile_tensor_core(s_scores_warp, s_q_warp, s_kt, d, scale);
        __syncwarp();

        if (lane < PROJECT_TILE) {
            int row = q_start + lane;
            if (row < N) {
                float m = row_max[batch_head * N + row];
                for (int j = 0; j < PROJECT_TILE; j++) {
                    int kv_idx = kv_start + j;
                    if (kv_idx >= N || (causal && kv_idx > row)) {
                        continue;
                    }

                    float p = expf(s_scores_warp[lane * PROJECT_TILE + j] - m);
                    s_l_warp[lane] += p;
                    for (int dd = 0; dd < d; dd++) {
                        s_o_warp[lane * d + dd] += p * __half2float(s_v[j * d + dd]);
                    }
                }
            }
        }

        __syncthreads();
    }

    if (lane < PROJECT_TILE) {
        int row = q_start + lane;
        if (row < N) {
            float inv_l = 1.0f / s_l_warp[lane];
            for (int dd = 0; dd < d; dd++) {
                o_base[row * d + dd] = s_o_warp[lane * d + dd] * inv_l;
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
    float* s_o = reinterpret_cast<float*>(s_q + PROJECT_BLOCK_M * d);
    float* s_m = s_o + PROJECT_BLOCK_M * d;
    float* s_l = s_m + PROJECT_BLOCK_M;

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
    if (threadIdx.x < PROJECT_BLOCK_M) {
        s_m[threadIdx.x] = -FLT_MAX;
        s_l[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    project_in_t* s_q_warp = s_q + warp_id * PROJECT_TILE * d;
    float* s_o_warp = s_o + warp_id * PROJECT_TILE * d;
    float* s_m_warp = s_m + warp_id * PROJECT_TILE;
    float* s_l_warp = s_l + warp_id * PROJECT_TILE;

    if (lane < PROJECT_TILE) {
        int row_in_tile = lane;
        int row = q_start + row_in_tile;
        if (row < N) {
            int max_j = causal ? (row + 1) : N;
            int num_kv_tiles = cdiv(max_j, PROJECT_TILE);

            for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
                int kv_start = kv_tile * PROJECT_TILE;
                float row_max = -FLT_MAX;

                for (int j = 0; j < PROJECT_TILE; j++) {
                    int kv_idx = kv_start + j;
                    if (kv_idx >= max_j) {
                        continue;
                    }

                    float dot = 0.0f;
                    for (int dd = 0; dd < d; dd++) {
                        dot += __half2float(s_q_warp[row_in_tile * d + dd])
                            * __half2float(k_base[kv_idx * d + dd]);
                    }
                    row_max = fmaxf(row_max, dot * scale);
                }

                float m_old = s_m_warp[row_in_tile];
                float m_new = fmaxf(m_old, row_max);
                float alpha = expf(m_old - m_new);

                for (int dd = 0; dd < d; dd++) {
                    s_o_warp[row_in_tile * d + dd] *= alpha;
                }

                float l_new = s_l_warp[row_in_tile] * alpha;
                for (int j = 0; j < PROJECT_TILE; j++) {
                    int kv_idx = kv_start + j;
                    if (kv_idx >= max_j) {
                        continue;
                    }

                    float dot = 0.0f;
                    for (int dd = 0; dd < d; dd++) {
                        dot += __half2float(s_q_warp[row_in_tile * d + dd])
                            * __half2float(k_base[kv_idx * d + dd]);
                    }

                    float p = expf(dot * scale - m_new);
                    l_new += p;
                    for (int dd = 0; dd < d; dd++) {
                        s_o_warp[row_in_tile * d + dd] +=
                            p * __half2float(v_base[kv_idx * d + dd]);
                    }
                }

                s_m_warp[row_in_tile] = m_new;
                s_l_warp[row_in_tile] = l_new;
            }

            float inv_l = 1.0f / s_l_warp[row_in_tile];
            for (int dd = 0; dd < d; dd++) {
                o_base[row * d + dd] = s_o_warp[row_in_tile * d + dd] * inv_l;
            }
        }
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
    int num_q_tiles = cdiv(N, PROJECT_BLOCK_M);
    float* d_row_max = nullptr;
    CUDA_CHECK(tracked_cuda_malloc(reinterpret_cast<void**>(&d_row_max), (size_t)BH * N * sizeof(float)));

    size_t pass1_smem = (PROJECT_BLOCK_M * d) * sizeof(project_in_t);
    pass1_smem += (PROJECT_TILE * d) * sizeof(project_in_t);
    pass1_smem += (PROJECT_Q_WARPS * PROJECT_TILE * PROJECT_TILE) * sizeof(float);
    size_t pass2_smem = (PROJECT_BLOCK_M * d) * sizeof(project_in_t);
    pass2_smem += (PROJECT_TILE * d) * sizeof(project_in_t);
    pass2_smem += (PROJECT_TILE * d) * sizeof(project_in_t);
    pass2_smem += (PROJECT_Q_WARPS * PROJECT_TILE * PROJECT_TILE) * sizeof(float);
    pass2_smem += (PROJECT_BLOCK_M * d) * sizeof(float);
    pass2_smem += PROJECT_BLOCK_M * sizeof(float);

    dim3 block(PROJECT_THREADS);
    dim3 grid(num_q_tiles, 1, BH);

    if (pass1_smem >= 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            two_pass_find_max_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            pass1_smem
        ));
    }

    two_pass_find_max_kernel<<<grid, block, pass1_smem>>>(
        d_Q, d_K, d_row_max, N, d, scale, causal
    );
    CUDA_CHECK(cudaGetLastError());

    if (pass2_smem >= 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            two_pass_attn_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            pass2_smem
        ));
    }

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
    size_t smem_bytes = (PROJECT_BLOCK_M * d) * sizeof(project_in_t);
    smem_bytes += (PROJECT_BLOCK_M * d) * sizeof(float);
    smem_bytes += 2 * PROJECT_BLOCK_M * sizeof(float);
    dim3 block(PROJECT_THREADS);
    dim3 grid(cdiv(N, PROJECT_BLOCK_M), 1, BH);

    if (smem_bytes >= 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            flash_attn_no_tiling_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes
        ));
    }

    flash_attn_no_tiling_kernel<<<grid, block, smem_bytes>>>(
        d_Q, d_K, d_V, d_O, N, d, scale, causal
    );

    CUDA_CHECK(cudaGetLastError());
}
