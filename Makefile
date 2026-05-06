NVCC = nvcc
NVCC_FLAGS = -O3 --use_fast_math -std=c++17 -I include
ARCH_FLAGS = -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=sm_89 \
             -gencode arch=compute_90,code=sm_90

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
