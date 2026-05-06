// Later-version split-KV forward extension.
//
// The official FlashAttention-2 contribution is a more ambitious redesign of
// work partitioning and non-matmul overhead. The narrow idea we model here is:
//   - sequence-parallel split-KV execution, where multiple thread blocks
//     process disjoint KV partitions for the same query tile and then combine
//     the partial log-sum-exp/output statistics
//
// This keeps one later-version-style split/combine mechanism without claiming
// to reproduce the full native FA2 kernel family.

#include "project_flash_splitkv.cuh"

void flash_attention_v2(
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
    project_flash::launch_flash_attention_splitkv(
        d_Q, d_K, d_V, d_O, B, H, N, d, scale, causal
    );
}
