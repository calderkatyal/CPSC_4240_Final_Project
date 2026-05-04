// FA3-inspired forward extension.
//
// Official FlashAttention-3 is Hopper-specific and relies on mechanisms such
// as TMA, warp specialization, and warpgroup MMA. We do not reproduce those
// machine-specific paths here. Instead, we model the high-level staging idea:
//   - software-pipelined double buffering of K/V tiles while keeping the
//     FA2-style split-KV sequence-parallel structure
//
// This is a portable approximation of the overlap direction in FA3, not a
// reproduction of the Hopper-specific implementation.

#include "project_flash_splitkv.cuh"

void flash_attention_v3(
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
    project_flash::launch_flash_attention_splitkv<true>(
        d_Q, d_K, d_V, d_O, B, H, N, d, scale, causal
    );
}
