// Benchmark suite
//
// This benchmark exercises:
//   - a naive mixed-precision baseline that materializes the full N x N score matrix
//   - a simplified FA1-style kernel with tensor-core score tiles
//   - FA2-style split-KV sequence parallelism
//   - FA1 ablations for online softmax and SRAM tiling
//
// Correctness is checked against a host-side float32 reference.

#include "flash_attn.cuh"
#include "project_flash_splitkv.cuh"

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

using ProjectKernelFn = void (*)(
    const project_in_t*, const project_in_t*, const project_in_t*, project_out_t*,
    int, int, int, int, float, bool
);

float benchmark_kernel(
    ProjectKernelFn fn,
    const project_in_t* d_Q,
    const project_in_t* d_K,
    const project_in_t* d_V,
    project_out_t* d_O,
    int B,
    int H,
    int N,
    int d,
    float scale,
    bool causal,
    int warmup_iters = 3,
    int bench_iters = 10
) {
    for (int i = 0; i < warmup_iters; i++) {
        fn(d_Q, d_K, d_V, d_O, B, H, N, d, scale, causal);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; i++) {
        fn(d_Q, d_K, d_V, d_O, B, H, N, d, scale, causal);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / bench_iters;
}

void run_correctness_check(
    const char* name,
    ProjectKernelFn fn,
    const float* h_ref,
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
    size_t total = (size_t)B * H * N * d;

    fn(d_Q, d_K, d_V, d_O, B, H, N, d, scale, causal);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_O(total);
    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, total * sizeof(float), cudaMemcpyDeviceToHost));

    float err = max_abs_diff(h_ref, h_O.data(), (int)total);
    printf("  %-35s max_abs_error = %.6e  %s\n", name, err, err < 3e-3 ? "PASS" : "FAIL");
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("=== Flash Attention Benchmark Suite ===\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    check_tensor_core_capability(prop);

    printf("Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Global memory: %.1f GB\n\n", prop.totalGlobalMem / 1e9);

    int B = 2;
    int H = 8;
    int d = 64;
    bool causal = true;
    check_supported_head_dim(d);
    print_project_precision_summary(d);
    printf("\n");

    std::vector<int> seq_lens = {128, 256, 512, 1024, 2048, 4096};

    std::filesystem::create_directories("results");
    FILE* csv = fopen("results/benchmark_results.csv", "w");
    if (csv == nullptr) {
        fprintf(stderr, "Failed to open results/benchmark_results.csv for writing.\n");
        return 1;
    }
    fprintf(csv, "method,seq_len,time_ms,memory_bytes,max_error,causal\n");

    struct MethodEntry {
        const char* name;
        ProjectKernelFn fn;
        size_t extra_mem;
    };

    MethodEntry methods[] = {
        {"Naive", naive_attention, 0},
        {"Simplified FA1", flash_attention_v1, 0},
        {"FA2-inspired extension", flash_attention_v2, 0},
        {"Ablation: no online softmax", flash_attention_v1_no_online_softmax, 0},
        {"Ablation: no SRAM tiling", flash_attention_v1_no_tiling, 0},
    };

    for (int N : seq_lens) {
        printf("--- N = %d (B=%d, H=%d, d=%d, causal=%d) ---\n", N, B, H, d, causal);

        size_t total = (size_t)B * H * N * d;
        size_t input_bytes = total * sizeof(project_in_t);
        size_t output_bytes = total * sizeof(project_out_t);

        std::vector<float> h_Qf(total), h_Kf(total), h_Vf(total), h_ref(total);
        std::vector<project_in_t> h_Q(total), h_K(total), h_V(total);

        srand(42);
        fill_random(h_Qf.data(), total);
        fill_random(h_Kf.data(), total);
        fill_random(h_Vf.data(), total);

        convert_float_to_project_input(h_Qf.data(), h_Q.data(), (int)total);
        convert_float_to_project_input(h_Kf.data(), h_K.data(), (int)total);
        convert_float_to_project_input(h_Vf.data(), h_V.data(), (int)total);

        float scale = 1.0f / sqrtf((float)d);
        reference_attention_host(
            h_Qf.data(), h_Kf.data(), h_Vf.data(), h_ref.data(),
            B, H, N, d, scale, causal
        );

        project_in_t* d_Q = nullptr;
        project_in_t* d_K = nullptr;
        project_in_t* d_V = nullptr;
        project_out_t* d_O = nullptr;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_Q), input_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_K), input_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_V), input_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_O), output_bytes));

        CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), input_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), input_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), input_bytes, cudaMemcpyHostToDevice));

        printf("Correctness checks:\n");
        for (const auto& method : methods) {
            run_correctness_check(
                method.name, method.fn, h_ref.data(),
                d_Q, d_K, d_V, d_O, B, H, N, d, scale, causal
            );
        }

        printf("Benchmarks:\n");
        for (const auto& method : methods) {
            float ms = benchmark_kernel(
                method.fn, d_Q, d_K, d_V, d_O, B, H, N, d, scale, causal
            );

            method.fn(d_Q, d_K, d_V, d_O, B, H, N, d, scale, causal);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> h_O(total);
            CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, output_bytes, cudaMemcpyDeviceToHost));
            float err = max_abs_diff(h_ref.data(), h_O.data(), (int)total);

            size_t mem = 3 * input_bytes + output_bytes + method.extra_mem;
            if (std::string(method.name) == "Naive") {
                mem += (size_t)B * H * N * N * sizeof(float);
            }
            if (std::string(method.name) == "FA2-inspired extension") {
                int nsplits = project_flash::choose_splitkv_splits<false>(B, H, N);
                mem += project_flash::splitkv_workspace_bytes<false>(B, H, N, d, nsplits);
            }
            if (std::string(method.name) == "Ablation: no online softmax") {
                mem += (size_t)B * H * N * sizeof(float);
            }

            printf("  %-35s %8.3f ms  mem=%zu bytes  err=%.2e\n",
                   method.name, ms, mem, err);
            fprintf(csv, "%s,%d,%.4f,%zu,%.6e,%d\n",
                    method.name, N, ms, mem, err, causal);
        }

        printf("\n");

        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V));
        CUDA_CHECK(cudaFree(d_O));
    }

    fclose(csv);
    printf("Project kernel results written to results/benchmark_results.csv\n");
    printf("Run python/run_benchmarks.py on a CUDA machine to add the optional\n");
    printf("official flash_attn comparison and regenerate plots/report inputs.\n");

    return 0;
}
