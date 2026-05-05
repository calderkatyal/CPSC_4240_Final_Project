#pragma once

#include "utils.cuh"

// This project implements a simplified forward-only family of FlashAttention-
// inspired kernels in CUDA C++. The scope is intentionally narrower than the
// official FlashAttention codebase:
//   - dense self-attention only
//   - FP16 Q/K/V inputs with FP32 softmax and output accumulation
//   - tensor-core-assisted score tiles for d that is a multiple of 16
//   - no backward pass / dropout / varlen / MQA-GQA / KV cache
//   - no Hopper-specific TMA / warp-specialized / FP8 kernels

// ============================================================================
// Simplified FA1-style kernel:
//   - shared-memory tiling for K/V (and Q staging)
//   - online softmax
//   - fused forward pass
//   - no explicit N×N score matrix materialization
// ============================================================================
void flash_attention_v1(
    const project_in_t* d_Q, const project_in_t* d_K, const project_in_t* d_V,
    project_out_t* d_O,
    int B, int H, int N, int d, float scale, bool causal
);

// ============================================================================
// FA2-inspired extension:
//   - retains the simplified forward-only scope above
//   - keeps the dense exact attention computation
//   - adds sequence-parallel split-KV execution with a partial-statistics combine
//
// Note: this is not a full implementation of official FlashAttention-2.
// ============================================================================
void flash_attention_v2(
    const project_in_t* d_Q, const project_in_t* d_K, const project_in_t* d_V,
    project_out_t* d_O,
    int B, int H, int N, int d, float scale, bool causal
);

// ============================================================================
// Ablation variants
// ============================================================================

// V1 without online softmax (two-pass: compute max first, then softmax)
void flash_attention_v1_no_online_softmax(
    const project_in_t* d_Q, const project_in_t* d_K, const project_in_t* d_V,
    project_out_t* d_O,
    int B, int H, int N, int d, float scale, bool causal
);

// V1 with no shared-memory K/V tiling
void flash_attention_v1_no_tiling(
    const project_in_t* d_Q, const project_in_t* d_K, const project_in_t* d_V,
    project_out_t* d_O,
    int B, int H, int N, int d, float scale, bool causal
);
