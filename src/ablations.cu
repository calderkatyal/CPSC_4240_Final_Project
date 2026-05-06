// Ablation variants for the simplified FlashAttention project.
//
// These isolate key ideas from the main FA1-style kernel:
//   1. No online softmax: keep tiling and tensor-core score tiles, but use a
//      two-pass softmax instead of the online recurrence.
//   2. No tensor cores: keep the same tiled, fused, online-softmax structure,
//      but replace the WMMA score/output updates with scalar per-thread math.
//   3. No vectorized loads: keep the same FA1 core kernel, but disable the
//      16-byte Q/K/V tile loads used by the main path.
//   4. No SRAM tiling: keep the same 64-row / 4-warp outer structure and the
//      same online-softmax recurrence, but fetch every key/value row directly
//      from global memory instead of staging K/V blocks in shared memory.

#include "project_flash_core.cuh"

using namespace project_flash;

namespace {

__global__ void two_pass_find_max_kernel(
    const project_in_t* __restrict__ Q,
    const project_in_t* __restrict__ K,
    float* __restrict__ row_max,
    int N,
    int d,
    float scale_l2,
    bool causal
) {
    int batch_head = blockIdx.z;
    int warp_id = threadIdx.x / PROJECT_WARP_SIZE;
    int lane = threadIdx.x % PROJECT_WARP_SIZE;
    int q_block_start = blockIdx.x * PROJECT_BLOCK_M;
    int q_start = q_block_start + warp_id * PROJECT_TILE;

    const project_in_t* q_base = Q + batch_head * N * d;
    const project_in_t* k_base = K + batch_head * N * d;

    extern __shared__ __align__(32) unsigned char smem_raw[];
    project_in_t* s_q = reinterpret_cast<project_in_t*>(smem_raw);
    project_in_t* s_kt = s_q + PROJECT_BLOCK_M * d;
    float* s_scores = reinterpret_cast<float*>(s_kt + PROJECT_BLOCK_N * d);

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

    float local_max = -FLT_MAX;
    if (lane < PROJECT_TILE) {
        int row = q_start + lane;
        if (row < N) {
            local_max = -FLT_MAX;
        }
    }

    int num_kv_tiles = cdiv(N, PROJECT_BLOCK_N);
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * PROJECT_BLOCK_N;
        load_kv_block(s_kt, nullptr, k_base, nullptr, kv_start, N, d);
        __syncthreads();
        compute_score_block_tensor_core(s_scores_warp, s_q_warp, s_kt, d, scale_l2);
        __syncwarp();

        if (lane < PROJECT_TILE) {
            int row = q_start + lane;
            if (row < N) {
                for (int j = 0; j < PROJECT_BLOCK_N; j++) {
                    int kv_idx = kv_start + j;
                    if (kv_idx >= N || (causal && kv_idx > row)) {
                        continue;
                    }
                    local_max = fmaxf(local_max, s_scores_warp[lane * PROJECT_BLOCK_N + j]);
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
    float scale_l2,
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

    extern __shared__ __align__(32) unsigned char smem_raw[];
    project_in_t* s_q = reinterpret_cast<project_in_t*>(smem_raw);
    project_in_t* s_kt = s_q + PROJECT_BLOCK_M * d;
    project_in_t* s_v = s_kt + PROJECT_BLOCK_N * d;
    float* s_scores = reinterpret_cast<float*>(s_v + PROJECT_BLOCK_N * d);

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

    float row_l = 0.0f;
    float row_o[PROJECT_MAX_D];
    #pragma unroll
    for (int dd = 0; dd < PROJECT_MAX_D; dd++) {
        row_o[dd] = 0.0f;
    }

    int num_kv_tiles = cdiv(N, PROJECT_BLOCK_N);
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * PROJECT_BLOCK_N;
        load_kv_block(s_kt, s_v, k_base, v_base, kv_start, N, d);
        __syncthreads();
        compute_score_block_tensor_core(s_scores_warp, s_q_warp, s_kt, d, scale_l2);
        __syncwarp();

        if (lane < PROJECT_TILE) {
            int row = q_start + lane;
            if (row < N) {
                float m = row_max[batch_head * N + row];
                for (int j = 0; j < PROJECT_BLOCK_N; j++) {
                    int kv_idx = kv_start + j;
                    if (kv_idx >= N || (causal && kv_idx > row)) {
                        continue;
                    }

                    float p = exp2f(s_scores_warp[lane * PROJECT_BLOCK_N + j] - m);
                    row_l += p;
                    for (int dd = 0; dd < d; dd++) {
                        row_o[dd] += p * __half2float(s_v[j * d + dd]);
                    }
                }
            }
        }

        __syncthreads();
    }

    if (lane < PROJECT_TILE) {
        int row = q_start + lane;
        if (row < N) {
            float inv_l = 1.0f / row_l;
            for (int dd = 0; dd < d; dd++) {
                o_base[row * d + dd] = row_o[dd] * inv_l;
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
    float scale_l2,
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

    extern __shared__ __align__(32) unsigned char smem_raw[];
    project_in_t* s_q = reinterpret_cast<project_in_t*>(smem_raw);

    const project_in_t zero = __float2half(0.0f);
    for (int idx = threadIdx.x; idx < PROJECT_BLOCK_M * d; idx += blockDim.x) {
        int row = idx / d;
        int col = idx % d;
        int global_row = q_block_start + row;
        s_q[idx] = (global_row < N) ? q_base[global_row * d + col] : zero;
    }
    __syncthreads();

    project_in_t* s_q_warp = s_q + warp_id * PROJECT_TILE * d;
    float row_m = -FLT_MAX;
    float row_l = 0.0f;
    float row_o[PROJECT_MAX_D];
    #pragma unroll
    for (int dd = 0; dd < PROJECT_MAX_D; dd++) {
        row_o[dd] = 0.0f;
    }

    if (lane < PROJECT_TILE) {
        int row_in_tile = lane;
        int row = q_start + row_in_tile;
        if (row < N) {
            int max_j = causal ? (row + 1) : N;
            int num_kv_tiles = cdiv(max_j, PROJECT_BLOCK_N);

            for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
                int kv_start = kv_tile * PROJECT_BLOCK_N;
                float row_max = -FLT_MAX;

                for (int j = 0; j < PROJECT_BLOCK_N; j++) {
                    int kv_idx = kv_start + j;
                    if (kv_idx >= max_j) {
                        continue;
                    }

                    float dot = 0.0f;
                    for (int dd = 0; dd < d; dd++) {
                        dot += __half2float(s_q_warp[row_in_tile * d + dd])
                            * __half2float(k_base[kv_idx * d + dd]);
                    }
                    row_max = fmaxf(row_max, dot * scale_l2);
                }

                float m_new = fmaxf(row_m, row_max);
                float alpha = exp2f(row_m - m_new);

                for (int dd = 0; dd < d; dd++) {
                    row_o[dd] *= alpha;
                }

                float l_new = row_l * alpha;
                for (int j = 0; j < PROJECT_BLOCK_N; j++) {
                    int kv_idx = kv_start + j;
                    if (kv_idx >= max_j) {
                        continue;
                    }

                    float dot = 0.0f;
                    for (int dd = 0; dd < d; dd++) {
                        dot += __half2float(s_q_warp[row_in_tile * d + dd])
                            * __half2float(k_base[kv_idx * d + dd]);
                    }

                    float p = exp2f(dot * scale_l2 - m_new);
                    l_new += p;
                    for (int dd = 0; dd < d; dd++) {
                        row_o[dd] += p * __half2float(v_base[kv_idx * d + dd]);
                    }
                }

                row_m = m_new;
                row_l = l_new;
            }

            float inv_l = 1.0f / row_l;
            for (int dd = 0; dd < d; dd++) {
                o_base[row * d + dd] = row_o[dd] * inv_l;
            }
        }
    }
}

template<int HEAD_DIM>
__global__ void flash_attn_no_tensor_cores_kernel(
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
    constexpr int kv_buf_stride = (d_padded > bn_padded) ? d_padded : bn_padded;
    constexpr int kv_buf_elems = PROJECT_BLOCK_N * kv_buf_stride;

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
    project_in_t* s_kt = s_q + PROJECT_BLOCK_M * d_padded;
    project_in_t* s_v  = s_kt + kv_buf_elems;
    project_in_t* s_p  = s_kt;

    project_in_t* s_q_warp = s_q + warp_id * PROJECT_TILE * d_padded;
    project_in_t* s_p_warp = s_p + warp_id * PROJECT_TILE * bn_padded;

    load_padded_rowmajor_tile<PROJECT_BLOCK_M, d, d_padded, true>(
        s_q, q_base, q_block_start, N
    );

    float o_accum[num_o_frags][8];
    #pragma unroll
    for (int f = 0; f < num_o_frags; f++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            o_accum[f][i] = 0.0f;
        }
    }

    float m0 = -FLT_MAX;
    float m1 = -FLT_MAX;
    float l0 = 0.0f;
    float l1 = 0.0f;

    __syncthreads();

    int num_kv_tiles = cdiv(N, PROJECT_BLOCK_N);
    if (causal) {
        int q_end = q_block_start + PROJECT_BLOCK_M - 1;
        if (q_end >= N) q_end = N - 1;
        int causal_limit = cdiv(q_end + 1, PROJECT_BLOCK_N);
        num_kv_tiles = (causal_limit < num_kv_tiles) ? causal_limit : num_kv_tiles;
    }

    #pragma unroll
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        const int kv_start = kv_tile * PROJECT_BLOCK_N;

        load_padded_rowmajor_tile<PROJECT_BLOCK_N, d, d_padded, true>(
            s_kt, k_base, kv_start, N
        );
        load_padded_rowmajor_tile<PROJECT_BLOCK_N, d, d_padded, true>(
            s_v, v_base, kv_start, N
        );
        __syncthreads();

        float sf[PROJECT_K_TILES_PER_BLOCK][8];
        #pragma unroll
        for (int tn = 0; tn < PROJECT_K_TILES_PER_BLOCK; tn++) {
            float s00 = 0.0f, s01 = 0.0f, s04 = 0.0f, s05 = 0.0f;
            float s10 = 0.0f, s11 = 0.0f, s14 = 0.0f, s15 = 0.0f;
            const int col0 = tn * PROJECT_TILE + fp * 2;
            const int col1 = col0 + 1;
            const int col4 = tn * PROJECT_TILE + fp * 2 + 8;
            const int col5 = col4 + 1;
            #pragma unroll
            for (int dd = 0; dd < d; dd++) {
                const float q0 = __half2float(s_q_warp[frow0 * d_padded + dd]);
                const float q1 = __half2float(s_q_warp[frow1 * d_padded + dd]);
                const float k0 = __half2float(s_kt[col0 * d_padded + dd]);
                const float k1 = __half2float(s_kt[col1 * d_padded + dd]);
                const float k4 = __half2float(s_kt[col4 * d_padded + dd]);
                const float k5 = __half2float(s_kt[col5 * d_padded + dd]);
                s00 += q0 * k0;
                s01 += q0 * k1;
                s04 += q0 * k4;
                s05 += q0 * k5;
                s10 += q1 * k0;
                s11 += q1 * k1;
                s14 += q1 * k4;
                s15 += q1 * k5;
            }
            sf[tn][0] = s00 * scale_l2;
            sf[tn][1] = s01 * scale_l2;
            sf[tn][2] = s10 * scale_l2;
            sf[tn][3] = s11 * scale_l2;
            sf[tn][4] = s04 * scale_l2;
            sf[tn][5] = s05 * scale_l2;
            sf[tn][6] = s14 * scale_l2;
            sf[tn][7] = s15 * scale_l2;
        }

        const bool r0v = (global_row0 < N);
        const bool r1v = (global_row1 < N);

        float mx0 = -FLT_MAX;
        float mx1 = -FLT_MAX;
        #pragma unroll
        for (int tn = 0; tn < PROJECT_K_TILES_PER_BLOCK; tn++) {
            const int bc = kv_start + tn * PROJECT_TILE;
            const int c0 = bc + fp * 2;
            const int c1 = c0 + 1;
            const int c4 = bc + fp * 2 + 8;
            const int c5 = c4 + 1;
            if (r0v) {
                if (c0 < N && (!causal || c0 <= global_row0)) mx0 = fmaxf(mx0, sf[tn][0]);
                if (c1 < N && (!causal || c1 <= global_row0)) mx0 = fmaxf(mx0, sf[tn][1]);
                if (c4 < N && (!causal || c4 <= global_row0)) mx0 = fmaxf(mx0, sf[tn][4]);
                if (c5 < N && (!causal || c5 <= global_row0)) mx0 = fmaxf(mx0, sf[tn][5]);
            }
            if (r1v) {
                if (c0 < N && (!causal || c0 <= global_row1)) mx1 = fmaxf(mx1, sf[tn][2]);
                if (c1 < N && (!causal || c1 <= global_row1)) mx1 = fmaxf(mx1, sf[tn][3]);
                if (c4 < N && (!causal || c4 <= global_row1)) mx1 = fmaxf(mx1, sf[tn][6]);
                if (c5 < N && (!causal || c5 <= global_row1)) mx1 = fmaxf(mx1, sf[tn][7]);
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
            o_accum[f][0] *= a0; o_accum[f][1] *= a0;
            o_accum[f][4] *= a0; o_accum[f][5] *= a0;
            o_accum[f][2] *= a1; o_accum[f][3] *= a1;
            o_accum[f][6] *= a1; o_accum[f][7] *= a1;
        }

        float sum0 = 0.0f;
        float sum1 = 0.0f;
        #pragma unroll
        for (int tn = 0; tn < PROJECT_K_TILES_PER_BLOCK; tn++) {
            const int bc = kv_start + tn * PROJECT_TILE;
            const int c0 = bc + fp * 2;
            const int c1 = c0 + 1;
            const int c4 = bc + fp * 2 + 8;
            const int c5 = c4 + 1;

            float p0 = 0.0f, p1 = 0.0f, p4 = 0.0f, p5 = 0.0f;
            float p2 = 0.0f, p3 = 0.0f, p6 = 0.0f, p7 = 0.0f;
            if (r0v) {
                if (c0 < N && (!causal || c0 <= global_row0)) p0 = exp2f(sf[tn][0] - nm0);
                if (c1 < N && (!causal || c1 <= global_row0)) p1 = exp2f(sf[tn][1] - nm0);
                if (c4 < N && (!causal || c4 <= global_row0)) p4 = exp2f(sf[tn][4] - nm0);
                if (c5 < N && (!causal || c5 <= global_row0)) p5 = exp2f(sf[tn][5] - nm0);
            }
            if (r1v) {
                if (c0 < N && (!causal || c0 <= global_row1)) p2 = exp2f(sf[tn][2] - nm1);
                if (c1 < N && (!causal || c1 <= global_row1)) p3 = exp2f(sf[tn][3] - nm1);
                if (c4 < N && (!causal || c4 <= global_row1)) p6 = exp2f(sf[tn][6] - nm1);
                if (c5 < N && (!causal || c5 <= global_row1)) p7 = exp2f(sf[tn][7] - nm1);
            }

            sum0 += p0 + p1 + p4 + p5;
            sum1 += p2 + p3 + p6 + p7;

            const int pc = tn * PROJECT_TILE;
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
        for (int kk = 0; kk < PROJECT_BLOCK_N; kk++) {
            const float p_row0 = __half2float(s_p_warp[frow0 * bn_padded + kk]);
            const float p_row1 = __half2float(s_p_warp[frow1 * bn_padded + kk]);

            #pragma unroll
            for (int f = 0; f < num_o_frags; f++) {
                const int bc = f * PROJECT_TILE;
                const float v0 = __half2float(s_v[kk * d_padded + bc + fp * 2]);
                const float v1 = __half2float(s_v[kk * d_padded + bc + fp * 2 + 1]);
                const float v4 = __half2float(s_v[kk * d_padded + bc + fp * 2 + 8]);
                const float v5 = __half2float(s_v[kk * d_padded + bc + fp * 2 + 9]);
                o_accum[f][0] += p_row0 * v0;
                o_accum[f][1] += p_row0 * v1;
                o_accum[f][4] += p_row0 * v4;
                o_accum[f][5] += p_row0 * v5;
                o_accum[f][2] += p_row1 * v0;
                o_accum[f][3] += p_row1 * v1;
                o_accum[f][6] += p_row1 * v4;
                o_accum[f][7] += p_row1 * v5;
            }
        }

        __syncthreads();
    }

    const float inv_l0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
    const float inv_l1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
    if (global_row0 < N) {
        #pragma unroll
        for (int f = 0; f < num_o_frags; f++) {
            const int bc = f * PROJECT_TILE;
            o_base[global_row0 * d + bc + fp * 2] = o_accum[f][0] * inv_l0;
            o_base[global_row0 * d + bc + fp * 2 + 1] = o_accum[f][1] * inv_l0;
            o_base[global_row0 * d + bc + fp * 2 + 8] = o_accum[f][4] * inv_l0;
            o_base[global_row0 * d + bc + fp * 2 + 9] = o_accum[f][5] * inv_l0;
        }
    }
    if (global_row1 < N) {
        #pragma unroll
        for (int f = 0; f < num_o_frags; f++) {
            const int bc = f * PROJECT_TILE;
            o_base[global_row1 * d + bc + fp * 2] = o_accum[f][2] * inv_l1;
            o_base[global_row1 * d + bc + fp * 2 + 1] = o_accum[f][3] * inv_l1;
            o_base[global_row1 * d + bc + fp * 2 + 8] = o_accum[f][6] * inv_l1;
            o_base[global_row1 * d + bc + fp * 2 + 9] = o_accum[f][7] * inv_l1;
        }
    }
}

template<int HEAD_DIM>
void launch_no_tensor_cores_hdim(
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
    smem_bytes += PROJECT_BLOCK_N * d_padded * sizeof(project_in_t);

    dim3 block(PROJECT_THREADS);
    dim3 grid(num_q_tiles, 1, BH);
    if (smem_bytes >= 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            flash_attn_no_tensor_cores_kernel<HEAD_DIM>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes
        ));
    }

    flash_attn_no_tensor_cores_kernel<HEAD_DIM><<<grid, block, smem_bytes>>>(
        d_Q, d_K, d_V, d_O, N, scale_l2, causal
    );
    CUDA_CHECK(cudaGetLastError());
}

void launch_no_tensor_cores(
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
            launch_no_tensor_cores_hdim<32>(d_Q, d_K, d_V, d_O, B, H, N, scale_l2, causal);
            break;
        case 64:
            launch_no_tensor_cores_hdim<64>(d_Q, d_K, d_V, d_O, B, H, N, scale_l2, causal);
            break;
        case 128:
            launch_no_tensor_cores_hdim<128>(d_Q, d_K, d_V, d_O, B, H, N, scale_l2, causal);
            break;
        default:
            fprintf(stderr, "Unsupported head dimension d=%d in no-tensor-core ablation.\n", d);
            exit(EXIT_FAILURE);
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
    const float scale_l2 = scale * PROJECT_LOG2E;

    int BH = B * H;
    int num_q_tiles = cdiv(N, PROJECT_BLOCK_M);
    float* d_row_max = nullptr;
    CUDA_CHECK(tracked_cuda_malloc(reinterpret_cast<void**>(&d_row_max), (size_t)BH * N * sizeof(float)));

    size_t pass1_smem = (PROJECT_BLOCK_M * d) * sizeof(project_in_t);
    pass1_smem += (PROJECT_BLOCK_N * d) * sizeof(project_in_t);
    pass1_smem += (PROJECT_Q_WARPS * PROJECT_TILE * PROJECT_BLOCK_N) * sizeof(float);
    size_t pass2_smem = (PROJECT_BLOCK_M * d) * sizeof(project_in_t);
    pass2_smem += (PROJECT_BLOCK_N * d) * sizeof(project_in_t);
    pass2_smem += (PROJECT_BLOCK_N * d) * sizeof(project_in_t);
    pass2_smem += (PROJECT_Q_WARPS * PROJECT_TILE * PROJECT_BLOCK_N) * sizeof(float);

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
        d_Q, d_K, d_row_max, N, d, scale_l2, causal
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
        d_Q, d_K, d_V, d_row_max, d_O, N, d, scale_l2, causal
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
    const float scale_l2 = scale * PROJECT_LOG2E;

    int BH = B * H;
    size_t smem_bytes = (PROJECT_BLOCK_M * d) * sizeof(project_in_t);
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
        d_Q, d_K, d_V, d_O, N, d, scale_l2, causal
    );

    CUDA_CHECK(cudaGetLastError());
}

void flash_attention_v1_no_tensor_cores(
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
    launch_no_tensor_cores(d_Q, d_K, d_V, d_O, B, H, N, d, scale, causal);
}

void flash_attention_v1_no_vectorized_loads(
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
    launch_flash_attention_core_no_vectorized_loads(
        d_Q, d_K, d_V, d_O, B, H, N, d, scale, causal
    );
}
