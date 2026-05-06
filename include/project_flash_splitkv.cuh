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
    float scale_l2,
    bool causal
) {
    const int batch_head = blockIdx.z;
    const int split_idx  = blockIdx.y;
    const int warp_id    = threadIdx.x / PROJECT_WARP_SIZE;
    const int lane       = threadIdx.x % PROJECT_WARP_SIZE;
    const int q_block_start = blockIdx.x * PROJECT_BLOCK_M;
    const int q_start       = q_block_start + warp_id * PROJECT_TILE;
    int q_end = q_block_start + PROJECT_BLOCK_M - 1;
    if (q_end >= N) q_end = N - 1;

    const int fg = lane / 4;
    const int fp = lane % 4;
    const int frow0 = fg;
    const int frow1 = fg + 8;
    const int global_row0 = q_start + frow0;
    const int global_row1 = q_start + frow1;

    const project_in_t* q_base = Q + batch_head * N * d;
    const project_in_t* k_base = K + batch_head * N * d;
    const project_in_t* v_base = V + batch_head * N * d;

    const int d_padded  = d + SMEM_PAD;
    const int bn_padded = PROJECT_BLOCK_N + SMEM_PAD;

    int kv_buf_elems = PROJECT_BLOCK_N * d_padded;
    if (PROJECT_BLOCK_M * bn_padded > kv_buf_elems)
        kv_buf_elems = PROJECT_BLOCK_M * bn_padded;

    extern __shared__ __align__(32) unsigned char smem_raw[];
    project_in_t* s_q  = reinterpret_cast<project_in_t*>(smem_raw);
    project_in_t* s_kt = s_q + PROJECT_BLOCK_M * d_padded;
    project_in_t* s_v  = s_kt + kv_buf_elems;
    project_in_t* s_p  = s_kt;

    project_in_t* s_q_warp = s_q + warp_id * PROJECT_TILE * d_padded;
    project_in_t* s_p_warp = s_p + warp_id * PROJECT_TILE * bn_padded;

    const project_in_t zero = __float2half(0.0f);

    for (int idx = threadIdx.x; idx < PROJECT_BLOCK_M * d_padded;
         idx += blockDim.x) {
        int row = idx / d_padded;
        int col = idx % d_padded;
        int gr  = q_block_start + row;
        s_q[idx] = (col < d && gr < N) ? q_base[gr * d + col] : zero;
    }

    const int num_o_frags = d / PROJECT_TILE;
    wmma::fragment<wmma::accumulator, PROJECT_TILE, PROJECT_TILE,
                   PROJECT_TILE, float> o_frag[PROJECT_MAX_D / PROJECT_TILE];
    for (int f = 0; f < num_o_frags; f++)
        wmma::fill_fragment(o_frag[f], 0.0f);

    float m0 = -FLT_MAX, m1 = -FLT_MAX;
    float l0 = 0.0f,     l1 = 0.0f;

    __syncthreads();

    int num_kv_tiles = cdiv(N, PROJECT_BLOCK_N);
    int tiles_per_split = cdiv(num_kv_tiles, num_splits);
    int kv_tile_begin = split_idx * tiles_per_split;
    int kv_tile_end = (split_idx + 1) * tiles_per_split;
    if (kv_tile_end > num_kv_tiles) kv_tile_end = num_kv_tiles;
    if (causal) {
        int causal_limit = cdiv(q_end + 1, PROJECT_BLOCK_N);
        if (kv_tile_end > causal_limit) kv_tile_end = causal_limit;
    }

    for (int kv_tile = kv_tile_begin; kv_tile < kv_tile_end; kv_tile++) {
        const int kv_start = kv_tile * PROJECT_BLOCK_N;

        for (int idx = threadIdx.x; idx < PROJECT_BLOCK_N * d_padded;
             idx += blockDim.x) {
            int row = idx / d_padded;
            int col = idx % d_padded;
            int gr  = kv_start + row;
            project_in_t kv = zero, vv = zero;
            if (col < d && gr < N) {
                kv = k_base[gr * d + col];
                vv = v_base[gr * d + col];
            }
            s_kt[idx] = kv;
            s_v[idx]  = vv;
        }
        __syncthreads();

        wmma::fragment<wmma::accumulator, PROJECT_TILE, PROJECT_TILE,
                       PROJECT_TILE, float> sf[PROJECT_K_TILES_PER_BLOCK];

        #pragma unroll
        for (int tn = 0; tn < PROJECT_K_TILES_PER_BLOCK; tn++)
            wmma::fill_fragment(sf[tn], 0.0f);

        #pragma unroll
        for (int k0 = 0; k0 < d; k0 += PROJECT_TILE) {
            wmma::fragment<wmma::matrix_a, PROJECT_TILE, PROJECT_TILE,
                           PROJECT_TILE, project_in_t,
                           wmma::row_major> a;
            wmma::load_matrix_sync(a, s_q_warp + k0, d_padded);
            #pragma unroll
            for (int tn = 0; tn < PROJECT_K_TILES_PER_BLOCK; tn++) {
                wmma::fragment<wmma::matrix_b, PROJECT_TILE, PROJECT_TILE,
                               PROJECT_TILE, project_in_t,
                               wmma::col_major> b;
                wmma::load_matrix_sync(
                    b, s_kt + tn * PROJECT_TILE * d_padded + k0, d_padded);
                wmma::mma_sync(sf[tn], a, b, sf[tn]);
            }
        }
        #pragma unroll
        for (int tn = 0; tn < PROJECT_K_TILES_PER_BLOCK; tn++) {
            #pragma unroll
            for (int i = 0; i < sf[tn].num_elements; i++)
                sf[tn].x[i] *= scale_l2;
        }

        __syncthreads();

        const bool r0v = (global_row0 < N);
        const bool r1v = (global_row1 < N);

        float mx0 = -FLT_MAX, mx1 = -FLT_MAX;
        #pragma unroll
        for (int tn = 0; tn < PROJECT_K_TILES_PER_BLOCK; tn++) {
            int bc = kv_start + tn * PROJECT_TILE;
            int c0 = bc + fp * 2, c1 = c0 + 1;
            int c4 = bc + fp * 2 + 8, c5 = c4 + 1;
            if (r0v) {
                if (c0 < N && (!causal || c0 <= global_row0))
                    mx0 = fmaxf(mx0, sf[tn].x[0]);
                if (c1 < N && (!causal || c1 <= global_row0))
                    mx0 = fmaxf(mx0, sf[tn].x[1]);
                if (c4 < N && (!causal || c4 <= global_row0))
                    mx0 = fmaxf(mx0, sf[tn].x[4]);
                if (c5 < N && (!causal || c5 <= global_row0))
                    mx0 = fmaxf(mx0, sf[tn].x[5]);
            }
            if (r1v) {
                if (c0 < N && (!causal || c0 <= global_row1))
                    mx1 = fmaxf(mx1, sf[tn].x[2]);
                if (c1 < N && (!causal || c1 <= global_row1))
                    mx1 = fmaxf(mx1, sf[tn].x[3]);
                if (c4 < N && (!causal || c4 <= global_row1))
                    mx1 = fmaxf(mx1, sf[tn].x[6]);
                if (c5 < N && (!causal || c5 <= global_row1))
                    mx1 = fmaxf(mx1, sf[tn].x[7]);
            }
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFF, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFF, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFF, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFF, mx1, 2));

        const float nm0 = fmaxf(m0, mx0);
        const float nm1 = fmaxf(m1, mx1);
        const float a0  = exp2f(m0 - nm0);
        const float a1  = exp2f(m1 - nm1);

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
            int c0 = bc + fp * 2, c1 = c0 + 1;
            int c4 = bc + fp * 2 + 8, c5 = c4 + 1;

            float p0 = 0, p1 = 0, p4 = 0, p5 = 0;
            float p2 = 0, p3 = 0, p6 = 0, p7 = 0;
            if (r0v) {
                if (c0 < N && (!causal || c0 <= global_row0))
                    p0 = exp2f(sf[tn].x[0] - nm0);
                if (c1 < N && (!causal || c1 <= global_row0))
                    p1 = exp2f(sf[tn].x[1] - nm0);
                if (c4 < N && (!causal || c4 <= global_row0))
                    p4 = exp2f(sf[tn].x[4] - nm0);
                if (c5 < N && (!causal || c5 <= global_row0))
                    p5 = exp2f(sf[tn].x[5] - nm0);
            }
            if (r1v) {
                if (c0 < N && (!causal || c0 <= global_row1))
                    p2 = exp2f(sf[tn].x[2] - nm1);
                if (c1 < N && (!causal || c1 <= global_row1))
                    p3 = exp2f(sf[tn].x[3] - nm1);
                if (c4 < N && (!causal || c4 <= global_row1))
                    p6 = exp2f(sf[tn].x[6] - nm1);
                if (c5 < N && (!causal || c5 <= global_row1))
                    p7 = exp2f(sf[tn].x[7] - nm1);
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

        #pragma unroll
        for (int kk = 0; kk < PROJECT_BLOCK_N; kk += PROJECT_TILE) {
            wmma::fragment<wmma::matrix_a, PROJECT_TILE, PROJECT_TILE,
                           PROJECT_TILE, project_in_t,
                           wmma::row_major> pf;
            wmma::load_matrix_sync(pf, s_p_warp + kk, bn_padded);
            #pragma unroll
            for (int dd = 0; dd < d; dd += PROJECT_TILE) {
                wmma::fragment<wmma::matrix_b, PROJECT_TILE, PROJECT_TILE,
                               PROJECT_TILE, project_in_t,
                               wmma::row_major> vf;
                wmma::load_matrix_sync(
                    vf, s_v + kk * d_padded + dd, d_padded);
                wmma::mma_sync(o_frag[dd / PROJECT_TILE], pf, vf,
                               o_frag[dd / PROJECT_TILE]);
            }
        }

        __syncthreads();
    }

    // Write partial O, m, l — each thread writes its 2 rows
    if (fp == 0) {
        if (global_row0 < N) {
            int sri = ((split_idx * gridDim.z + batch_head) * N) + global_row0;
            partial_m[sri] = m0;
            partial_l[sri] = l0;
        }
        if (global_row1 < N) {
            int sri = ((split_idx * gridDim.z + batch_head) * N) + global_row1;
            partial_m[sri] = m1;
            partial_l[sri] = l1;
        }
    }
    if (global_row0 < N) {
        int sri = ((split_idx * gridDim.z + batch_head) * N) + global_row0;
        int o_off = sri * d;
        for (int f = 0; f < num_o_frags; f++) {
            int bc = f * PROJECT_TILE;
            partial_o[o_off + bc + fp * 2]     = o_frag[f].x[0];
            partial_o[o_off + bc + fp * 2 + 1] = o_frag[f].x[1];
            partial_o[o_off + bc + fp * 2 + 8] = o_frag[f].x[4];
            partial_o[o_off + bc + fp * 2 + 9] = o_frag[f].x[5];
        }
    }
    if (global_row1 < N) {
        int sri = ((split_idx * gridDim.z + batch_head) * N) + global_row1;
        int o_off = sri * d;
        for (int f = 0; f < num_o_frags; f++) {
            int bc = f * PROJECT_TILE;
            partial_o[o_off + bc + fp * 2]     = o_frag[f].x[2];
            partial_o[o_off + bc + fp * 2 + 1] = o_frag[f].x[3];
            partial_o[o_off + bc + fp * 2 + 8] = o_frag[f].x[6];
            partial_o[o_off + bc + fp * 2 + 9] = o_frag[f].x[7];
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
    if (row >= N) return;

    float global_m = -FLT_MAX;
    for (int split_idx = 0; split_idx < num_splits; split_idx++) {
        int sri = ((split_idx * gridDim.z + batch_head) * N) + row;
        global_m = fmaxf(global_m, partial_m[sri]);
    }

    float global_l = 0.0f;
    for (int split_idx = 0; split_idx < num_splits; split_idx++) {
        int sri = ((split_idx * gridDim.z + batch_head) * N) + row;
        float l_local = partial_l[sri];
        if (l_local > 0.0f)
            global_l += l_local * exp2f(partial_m[sri] - global_m);
    }

    project_out_t* o_base = O + batch_head * N * d;
    if (global_l == 0.0f) {
        for (int dd = 0; dd < d; dd++)
            o_base[row * d + dd] = 0.0f;
        return;
    }

    for (int dd = 0; dd < d; dd++) {
        float accum = 0.0f;
        for (int split_idx = 0; split_idx < num_splits; split_idx++) {
            int sri = ((split_idx * gridDim.z + batch_head) * N) + row;
            float l_local = partial_l[sri];
            if (l_local > 0.0f)
                accum += partial_o[sri * d + dd]
                    * exp2f(partial_m[sri] - global_m);
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

    float* partial_m_ptr = nullptr;
    float* partial_l_ptr = nullptr;
    float* partial_o_ptr = nullptr;
    CUDA_CHECK(tracked_cuda_malloc(reinterpret_cast<void**>(&partial_m_ptr), (size_t)num_splits * BH * N * sizeof(float)));
    CUDA_CHECK(tracked_cuda_malloc(reinterpret_cast<void**>(&partial_l_ptr), (size_t)num_splits * BH * N * sizeof(float)));
    CUDA_CHECK(tracked_cuda_malloc(reinterpret_cast<void**>(&partial_o_ptr), (size_t)num_splits * BH * N * d * sizeof(float)));

    const int d_padded  = d + SMEM_PAD;
    const int bn_padded = PROJECT_BLOCK_N + SMEM_PAD;
    int kv_buf_elems = PROJECT_BLOCK_N * d_padded;
    if (PROJECT_BLOCK_M * bn_padded > kv_buf_elems)
        kv_buf_elems = PROJECT_BLOCK_M * bn_padded;

    size_t partial_smem = 0;
    partial_smem += PROJECT_BLOCK_M * d_padded * sizeof(project_in_t);
    partial_smem += kv_buf_elems * sizeof(project_in_t);
    partial_smem += PROJECT_BLOCK_N * d_padded * sizeof(project_in_t);

    dim3 block(PROJECT_THREADS);
    dim3 grid_partial(num_q_tiles, num_splits, BH);
    if (partial_smem >= 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            flash_attention_splitkv_partial_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            partial_smem
        ));
    }
    const float scale_l2 = scale * 1.4426950408889634f;
    flash_attention_splitkv_partial_kernel<<<grid_partial, block, partial_smem>>>(
        d_Q, d_K, d_V, partial_m_ptr, partial_l_ptr, partial_o_ptr,
        num_splits, N, d, scale_l2, causal
    );
    CUDA_CHECK(cudaGetLastError());

    dim3 grid_combine(num_q_tiles, 1, BH);
    dim3 combine_block(PROJECT_BLOCK_M);
    flash_attention_splitkv_combine_kernel<<<grid_combine, combine_block>>>(
        partial_m_ptr, partial_l_ptr, partial_o_ptr, d_O, num_splits, N, d
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(tracked_cuda_free(partial_m_ptr));
    CUDA_CHECK(tracked_cuda_free(partial_l_ptr));
    CUDA_CHECK(tracked_cuda_free(partial_o_ptr));
}

}  // namespace project_flash
