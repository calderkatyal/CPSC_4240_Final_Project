# Simplified FlashAttention Project

This repository contains a CUDA C++ implementation of a simplified FlashAttention benchmark suite, together with plotting scripts and a NeurIPS-style report. The project compares:

- a naive exact attention baseline
- a simplified FlashAttention-1-style kernel
- an FA2-inspired split-KV extension
- ablations of key FlashAttention-1 ideas
- official FlashAttention-1 and FlashAttention-2 baselines

## Yale Cluster Setup

These instructions are written for a general user on the Yale cluster and assume access to an NVIDIA GPU node.

### 1. Clone the repository

```bash
git clone git@github.com:calderkatyal/CPSC_4240_Final_Project.git CPSC_4240_Final_Project
cd CPSC_4240_Final_Project
```

### 2. Load the required modules

Use the GCC 12 / CUDA 12.1 / Python 3.10 toolchain:

```bash
module purge
module load GCC/12.2.0
module load CUDA/12.1.1
module load Python/3.10.8-GCCcore-12.2.0
```

You can verify the compiler and Python versions with:

```bash
gcc --version
g++ --version
python --version
which gcc
which g++
which python
```

### 3. Create the virtual environment

Create the environment in the project root:

```bash
python -m venv venv
source venv/bin/activate
```

### 4. Install Python dependencies

Install PyTorch with CUDA 12.1 wheels, then the remaining packages:

```bash
pip install --upgrade pip
pip install setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.2
pip install -r requirements.txt
```

### 5. Clone the official FlashAttention source trees

The project benchmarks two official versions from the same upstream repository, checked out at different tags:

```bash
mkdir -p external
cd external

git clone --branch v1.0.9 --depth 1 --recurse-submodules https://github.com/Dao-AILab/flash-attention.git flash-attn-fa1
git clone --branch v2.8.3 --depth 1 --recurse-submodules https://github.com/Dao-AILab/flash-attention.git flash-attn-fa2

cd ..
```

### 6. Force CUDA builds to use the loaded GCC toolchain

This step is important because CUDA 12.1 does not support GCC versions newer than 12.

```bash
export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$(which g++)
```

### 7. Build the official FlashAttention baselines

Build both official versions in place:

```bash
cd external/flash-attn-fa1
python setup.py build_ext --inplace

cd ../flash-attn-fa2
MAX_JOBS=4 python setup.py build_ext --inplace

cd ../..
```

### 8. Benchmark official FlashAttention-1 and FlashAttention-2

Run the official baselines first:

```bash
python python/benchmark_official_flash_attn.py --version fa1 --source-dir external/flash-attn-fa1
python python/benchmark_official_flash_attn.py --version fa2 --source-dir external/flash-attn-fa2
```

This writes:

- `results/official_flash_attn_v1_results.csv`
- `results/official_flash_attn_v2_results.csv`

### 9. Build and run the project benchmark

```bash
make benchmark
python python/run_benchmarks.py
```

This produces:

- `results/benchmark_results.csv`
- `results/gpu_comparison_results.csv`

### 10. Generate plots

```bash
python python/plot_results.py
```

This writes the runtime, speedup, memory, and ablation figures into `results/`.

### 11. Rebuild the report

```bash
cd report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
cd ..
```

The rebuilt PDF will be available at:

- `report/report.pdf`