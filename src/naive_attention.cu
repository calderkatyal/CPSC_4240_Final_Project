// Naive mixed-precision attention baseline.
//
// This baseline keeps the exact attention math but materializes the full
// N x N score matrix in global memory and uses three separate kernels:
//   1. QK^T score matrix
//   2. row-wise softmax
//   3. P @ V
//
// Inputs use FP16 to match the project kernels, while the score matrix,
// softmax, and output accumulation stay in FP32.

#include "flash_attn.cuh"

__global__ void naive_qk_kernel(
    const project_in_t* __restrict__ Q,
    const project_in_t* __restrict__ K,
    float* __restrict__ S,
    int N,
    int d,
    float scale,
    bool causal
) {
    int batch_head = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) {
        return;
    }

    const project_in_t* q = Q + batch_head * N * d;
    const project_in_t* k = K + batch_head * N * d;
    float* s = S + batch_head * N * N;

    if (causal && col > row) {
        s[row * N + col] = -FLT_MAX;
        return;
    }

    float dot = 0.0f;
    for (int i = 0; i < d; i++) {
        dot += __half2float(q[row * d + i]) * __half2float(k[col * d + i]);
    }
    s[row * N + col] = dot * scale;
}

__global__ void naive_softmax_kernel(float* __restrict__ S, int N) {
    int batch_head = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N) {
        return;
    }

    float* s = S + batch_head * N * N + row * N;
    float max_val = -FLT_MAX;
    for (int j = 0; j < N; j++) {
        max_val = fmaxf(max_val, s[j]);
    }

    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        s[j] = expf(s[j] - max_val);
        sum += s[j];
    }

    float inv_sum = 1.0f / sum;
    for (int j = 0; j < N; j++) {
        s[j] *= inv_sum;
    }
}

__global__ void naive_pv_kernel(
    const float* __restrict__ P,
    const project_in_t* __restrict__ V,
    project_out_t* __restrict__ O,
    int N,
    int d
) {
    int batch_head = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= d) {
        return;
    }

    const float* p = P + batch_head * N * N;
    const project_in_t* v = V + batch_head * N * d;
    project_out_t* o = O + batch_head * N * d;

    float val = 0.0f;
    for (int j = 0; j < N; j++) {
        val += p[row * N + j] * __half2float(v[j * d + col]);
    }
    o[row * d + col] = val;
}

void naive_attention(
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
    float* d_S = nullptr;
    CUDA_CHECK(tracked_cuda_malloc(reinterpret_cast<void**>(&d_S), (size_t)BH * N * N * sizeof(float)));

    {
        dim3 block(16, 16);
        dim3 grid(cdiv(N, 16), cdiv(N, 16), BH);
        naive_qk_kernel<<<grid, block>>>(d_Q, d_K, d_S, N, d, scale, causal);
    }

    {
        dim3 block(256);
        dim3 grid(cdiv(N, 256), BH);
        naive_softmax_kernel<<<grid, block>>>(d_S, N);
    }

    {
        dim3 block(16, 16);
        dim3 grid(cdiv(d, 16), cdiv(N, 16), BH);
        naive_pv_kernel<<<grid, block>>>(d_S, d_V, d_O, N, d);
    }

    CUDA_CHECK(tracked_cuda_free(d_S));
    CUDA_CHECK(cudaGetLastError());
}
