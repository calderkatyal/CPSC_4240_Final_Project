# Simplified FlashAttention Project

This repository contains a CUDA C++ implementation of a simplified FlashAttention benchmark suite, together with plotting scripts and a NeurIPS-style report. The project compares:

- a shared PyTorch baseline that matches the official FlashAttention benchmark style
- Our Flash Attention
- ablations of key FlashAttention-1 ideas
- an official FlashAttention-1 baseline

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

### 5. Clone the official FlashAttention-1 source tree

Clone the official FA1 source tree with submodules:

```bash
mkdir -p external
cd external

git clone --branch v1.0.9 --depth 1 --recurse-submodules https://github.com/Dao-AILab/flash-attention.git flash-attn-fa1

cd ..
```

### 6. Force CUDA builds to use the loaded GCC toolchain

This step is important because CUDA 12.1 does not support GCC versions newer than 12.

```bash
export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$(which g++)
```

### 7. Build the official FlashAttention-1 extension

Build the official FA1 extension in place:

```bash
cd external/flash-attn-fa1
python setup.py build_ext --inplace
cd ../..
```

### 8. Benchmark official FlashAttention-1

Run the official baseline first. FA1 is imported from its source tree:

```bash
python python/benchmark_official_flash_attn.py --source-dir external/flash-attn-fa1
```

This writes:

- `results/official_pytorch_baseline_results.csv`
- `results/official_flash_attn_v1_results.csv`

The shared PyTorch baseline is adapted directly from the official
`benchmarks/benchmark_flash_attention.py` comparison in the FlashAttention
repository. It materializes the full score matrix with PyTorch ops
(`baddbmm`, softmax, and value contraction), so the merged speedup tables and
figures use one common baseline for Our FA and the official
FA1 kernel.

### 9. Build and run the project benchmark

```bash
make clean benchmark
python python/run_benchmarks.py
```

This produces:

- `results/benchmark_results.csv`
- `results/gpu_comparison_results.csv`
- `results/table_rows.tex`

### 10. Generate plots

```bash
python python/plot_results.py
```

This writes the runtime, speedup, memory, and ablation figures into `results/`.

### 11. Rebuild the report

```bash
python python/build_report.py
```

The rebuilt PDF will be available at:

- `report/report.pdf`

This build expects the merged comparison outputs in `results/gpu_comparison_results.csv`
plus `results/table_rows.tex` and `results/gpu_comparison_*.pdf`; it does not
fall back to local-only report data. It reads those files but does not rewrite
anything inside `results/`.

## Resume a Later Cluster Session

If you already completed the one-time setup in an earlier session (packages
installed, virtual environment created, and official FA1 source cloned), you do
not need to repeat the install steps above.

From a fresh shell on the cluster, run:

```bash
cd /path/to/CPSC_4240_Final_Project

module purge
module load GCC/12.2.0
module load CUDA/12.1.1
module load Python/3.10.8-GCCcore-12.2.0
module load texlive/20220321-GCC-12.2.0

source venv/bin/activate

export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$(which g++)
```

Then rerun the full benchmark/report pipeline:

```bash
python python/benchmark_official_flash_attn.py --source-dir external/flash-attn-fa1
make clean benchmark
python python/run_benchmarks.py
python python/plot_results.py
python python/build_report.py
```

That sequence assumes all dependencies are already installed and produces:

- refreshed benchmark CSVs in `results/`
- refreshed plot PDFs in `results/`
- refreshed table macros in `results/table_rows.tex`
- refreshed report PDF at `report/report.pdf`
