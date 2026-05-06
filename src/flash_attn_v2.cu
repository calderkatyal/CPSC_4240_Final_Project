// FA2-inspired forward extension.
//
// Official FlashAttention-2's forward path improves parallelism by splitting
// work along the sequence / K-V dimension when a single attention head does not
// expose enough CTAs to fill the GPU. The simplified project analogue here is a
// split-KV path with a small combine kernel. In this project we keep that
// sequence-parallel idea explicit by always routing the FA2-inspired entrypoint
// through the split-KV path.

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
