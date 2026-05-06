#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cstddef>
#include <unordered_map>

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

__host__ __device__ inline int cdiv(int a, int b) { return (a + b - 1) / b; }

inline std::unordered_map<void*, size_t>& cuda_allocations() {
    static std::unordered_map<void*, size_t> allocations;
    return allocations;
}

inline size_t& cuda_current_bytes() {
    static size_t current = 0;
    return current;
}

inline size_t& cuda_peak_bytes() {
    static size_t peak = 0;
    return peak;
}

inline void cuda_memory_tracking_clear() {
    cuda_allocations().clear();
    cuda_current_bytes() = 0;
    cuda_peak_bytes() = 0;
}

inline void cuda_memory_tracking_reset_peak() {
    cuda_peak_bytes() = cuda_current_bytes();
}

inline size_t cuda_memory_tracking_peak_bytes() {
    return cuda_peak_bytes();
}

inline cudaError_t tracked_cuda_malloc(void** ptr, size_t size) {
    cudaError_t err = cudaMalloc(ptr, size);
    if (err == cudaSuccess && ptr != nullptr && *ptr != nullptr) {
        cuda_allocations()[*ptr] = size;
        cuda_current_bytes() += size;
        if (cuda_current_bytes() > cuda_peak_bytes()) {
            cuda_peak_bytes() = cuda_current_bytes();
        }
    }
    return err;
}

inline cudaError_t tracked_cuda_free(void* ptr) {
    auto& allocations = cuda_allocations();
    auto it = allocations.find(ptr);
    if (it != allocations.end()) {
        cuda_current_bytes() -= it->second;
        allocations.erase(it);
    }
    return cudaFree(ptr);
}

using project_in_t = half;
using project_out_t = half;

constexpr int PROJECT_TILE = 16;
constexpr int PROJECT_WARP_SIZE = 32;
constexpr int PROJECT_Q_WARPS = 4;
constexpr int PROJECT_BLOCK_M = PROJECT_TILE * PROJECT_Q_WARPS;
constexpr int PROJECT_BLOCK_N = 128;
constexpr int PROJECT_K_TILES_PER_BLOCK = PROJECT_BLOCK_N / PROJECT_TILE;
constexpr int PROJECT_THREADS = PROJECT_WARP_SIZE * PROJECT_Q_WARPS;
constexpr int PROJECT_MAX_D = 128;
constexpr float PROJECT_LOG2E = 1.4426950408889634f;

inline bool is_supported_head_dim(int d) {
    return d == 32 || d == 64 || d == 128;
}

inline void check_supported_head_dim(int d) {
    if (!is_supported_head_dim(d)) {
        fprintf(stderr,
                "Unsupported head dimension d=%d. Supported specializations "
                "are {32, 64, 128}.\n",
                d);
        exit(EXIT_FAILURE);
    }
}

inline void check_tensor_core_capability(const cudaDeviceProp& prop) {
    if (prop.major < 7) {
        fprintf(stderr,
                "These kernels require Volta-or-newer tensor cores "
                "(compute capability >= 7.0). Detected %d.%d.\n",
                prop.major, prop.minor);
        exit(EXIT_FAILURE);
    }
}

inline void convert_float_to_input(
    const float* src, project_in_t* dst, int n
) {
    for (int i = 0; i < n; i++) {
        dst[i] = __float2half(src[i]);
    }
}

inline void print_precision_summary(int d) {
    printf("Kernel precision: FP16 Q/K/V/O tensors, FP32 softmax/output accumulators\n");
    printf("Tensor-core score path: enabled for supported head dims {32, 64, 128}; current d=%d\n", d);
    printf("FA1 thread-block tiles: Q-block=%d rows, KV-block=%d rows\n",
           PROJECT_BLOCK_M, PROJECT_BLOCK_N);
}

inline void convert_output_to_float(
    const project_out_t* src, float* dst, int n
) {
    for (int i = 0; i < n; i++) {
        dst[i] = __half2float(src[i]);
    }
}

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
