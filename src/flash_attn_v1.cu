// FA1-style forward kernel.
//
// Core ideas represented here:
//   1. Shared-memory tiling for Q/K/V state
//   2. Online softmax across KV tiles
//   3. Fused exact attention without materializing the full N x N matrix
//   4. Mixed precision with FP16 inputs and FP32 accumulation
//   5. Tensor-core-assisted QK score tiles via WMMA

#include "project_flash_core.cuh"

void flash_attention_v1(
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
    project_flash::launch_flash_attention_core(
        d_Q, d_K, d_V, d_O, B, H, N, d, scale, causal
    );
}
