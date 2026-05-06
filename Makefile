NVCC = nvcc
NVCC_FLAGS = -O3 --use_fast_math -std=c++17 -I include

# Keep the default build fast for the target GPU used in this project.
# Override, e.g. `make benchmark CUDA_ARCHS="80 89"` if you want a wider fatbin.
CUDA_ARCHS ?= 89
ARCH_FLAGS = $(foreach arch,$(CUDA_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))

SOURCES = src/flash_attn_v1.cu src/flash_attn_v2.cu src/ablations.cu src/benchmark.cu

.PHONY: all clean run

all: benchmark

benchmark: $(SOURCES) include/flash_attn.cuh include/utils.cuh
	mkdir -p results
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) $(SOURCES) -o benchmark

run: benchmark
	./benchmark

clean:
	rm -f benchmark
	rm -f results/benchmark_results.csv \
	      results/gpu_comparison_results.csv \
	      results/table_rows.tex
	rm -f results/*.pdf
