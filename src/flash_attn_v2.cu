// FA2-inspired forward extension.
//
// In this project, the later-version-inspired change is wider query ownership
// per CTA. That keeps the compact WMMA kernel structure, but repartitions work
// so each block amortizes K/V tile loads across more query rows, echoing
// FlashAttention-2's broader emphasis on better work partitioning.

#include "project_flash_core.cuh"

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
    project_flash::launch_flash_attention_fa2_extension(
        d_Q, d_K, d_V, d_O, B, H, N, d, scale, causal
    );
}
