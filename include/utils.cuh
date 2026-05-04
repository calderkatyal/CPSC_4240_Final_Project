#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

inline int cdiv(int a, int b) { return (a + b - 1) / b; }

using project_in_t = half;
using project_out_t = float;

constexpr int PROJECT_TILE = 16;
constexpr int PROJECT_MAX_D = 128;

inline void check_supported_head_dim(int d) {
    if (d <= 0 || d > PROJECT_MAX_D || (d % PROJECT_TILE) != 0) {
        fprintf(stderr,
                "Unsupported head dimension d=%d. This simplified project "
                "supports tensor-core-friendly head dimensions that are "
                "multiples of %d with d <= %d.\n",
                d, PROJECT_TILE, PROJECT_MAX_D);
        exit(EXIT_FAILURE);
    }
}

inline void check_tensor_core_capability(const cudaDeviceProp& prop) {
    if (prop.major < 7) {
        fprintf(stderr,
                "This project's CUDA kernels require Volta-or-newer tensor "
                "cores (compute capability >= 7.0). Detected %d.%d.\n",
                prop.major, prop.minor);
        exit(EXIT_FAILURE);
    }
}

inline void convert_float_to_project_input(
    const float* src, project_in_t* dst, int n
) {
    for (int i = 0; i < n; i++) {
        dst[i] = __float2half(src[i]);
    }
}

inline void print_project_precision_summary(int d) {
    printf("Project kernel precision: FP16 Q/K/V inputs, FP32 softmax/output accumulation\n");
    printf("Tensor-core score path: enabled for d=%d (multiple of %d)\n", d, PROJECT_TILE);
}

inline int project_num_splits_heuristic(
    int batch_nheads_mblocks, int effective_sms, int num_n_blocks, int max_splits
) {
    if (batch_nheads_mblocks >= static_cast<int>(0.8f * effective_sms)) {
        return 1;
    }
    if (max_splits > effective_sms) {
        max_splits = effective_sms;
    }
    if (max_splits > num_n_blocks) {
        max_splits = num_n_blocks;
    }
    float max_efficiency = 0.0f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv_host = [](int a, int b) { return (a + b - 1) / b; };
    auto is_split_eligible = [&](int num_splits) {
        return num_splits == 1
            || ceildiv_host(num_n_blocks, num_splits)
                != ceildiv_host(num_n_blocks, num_splits - 1);
    };

    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.0f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / float(effective_sms);
            float eff = n_waves / ceilf(n_waves);
            if (eff > max_efficiency) {
                max_efficiency = eff;
            }
            efficiency.push_back(eff);
        }
    }

    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            continue;
        }
        if (efficiency[num_splits - 1] >= 0.85f * max_efficiency) {
            return num_splits;
        }
    }
    return 1;
}

struct AttentionParams {
    int batch_size;
    int num_heads;
    int seq_len;
    int head_dim;
    float scale;
    bool causal;
};

inline void fill_random(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
    }
}

inline void reference_attention_host(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int N, int d, float scale, bool causal
) {
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int offset = (b * H + h) * N * d;
            const float* q = Q + offset;
            const float* k = K + offset;
            const float* v = V + offset;
            float* o = O + offset;

            for (int i = 0; i < N; i++) {
                float row_max = -FLT_MAX;
                float* scores = (float*)malloc(N * sizeof(float));

                for (int j = 0; j < N; j++) {
                    if (causal && j > i) {
                        scores[j] = -FLT_MAX;
                        continue;
                    }
                    float dot = 0.0f;
                    for (int dd = 0; dd < d; dd++) {
                        dot += q[i * d + dd] * k[j * d + dd];
                    }
                    scores[j] = dot * scale;
                    if (scores[j] > row_max) row_max = scores[j];
                }

                float sum_exp = 0.0f;
                for (int j = 0; j < N; j++) {
                    scores[j] = expf(scores[j] - row_max);
                    sum_exp += scores[j];
                }
                for (int j = 0; j < N; j++) {
                    scores[j] /= sum_exp;
                }

                for (int dd = 0; dd < d; dd++) {
                    float val = 0.0f;
                    for (int j = 0; j < N; j++) {
                        val += scores[j] * v[j * d + dd];
                    }
                    o[i * d + dd] = val;
                }
                free(scores);
            }
        }
    }
}

inline float max_abs_diff(const float* a, const float* b, int n) {
    float maxd = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > maxd) maxd = d;
    }
    return maxd;
}
