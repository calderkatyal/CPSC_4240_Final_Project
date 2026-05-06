cmake_minimum_required(VERSION 3.18)
project(flash_attention CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Auto-detect GPU architecture, fall back to common ones
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

# Collect all CUDA source files
set(SOURCES
    src/flash_attn_v1.cu
    src/ablations.cu
    src/benchmark.cu
)

add_executable(benchmark ${SOURCES})
target_compile_options(benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
