"""
GPU-only benchmark runne

It performs three steps:
  1. Runs the local CUDA benchmark binary to produce project timings.
  2. Optionally benchmarks the official `flash_attn` Python package.
  3. Merges both CSVs into a single comparison CSV for plotting/reporting.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from flash_attn import flash_attn_func as official_flash_attn_func
except Exception:
    official_flash_attn_func = None


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CSV_FIELDS = ["method", "seq_len", "time_ms", "memory_bytes", "max_error", "causal"]


def write_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def ensure_cuda_available() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run this script on a CUDA-enabled machine.")


def run_project_benchmark_binary() -> None:
    benchmark_bin = ROOT_DIR / "benchmark"
    if not benchmark_bin.exists():
        raise FileNotFoundError(
            "Missing ./benchmark. Build it first with `make benchmark` or `make run`."
        )

    print("Running local CUDA benchmark binary...")
    subprocess.run([str(benchmark_bin)], cwd=ROOT_DIR, check=True)


def torch_reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if causal:
        nq = q.shape[-2]
        nk = k.shape[-2]
        mask = torch.triu(torch.ones((nq, nk), device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float())


def benchmark_torch_callable(fn, warmup: int = 10, bench_iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(bench_iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / bench_iters


def run_official_flash_attn_benchmarks() -> list[dict]:
    if official_flash_attn_func is None:
        print("Skipping official flash_attn benchmarks: package import failed.")
        return []

    print("Running official flash_attn package benchmarks...")
    device = torch.device("cuda")
    dtype = torch.float16
    B, H, d = 2, 8, 64
    seq_lens = [128, 256, 512, 1024, 2048, 4096]
    causal = True

    rows = []
    for N in seq_lens:
        torch.manual_seed(42)
        q = torch.randn((B, N, H, d), device=device, dtype=dtype)
        k = torch.randn((B, N, H, d), device=device, dtype=dtype)
        v = torch.randn((B, N, H, d), device=device, dtype=dtype)

        with torch.no_grad():
            out = official_flash_attn_func(q, k, v, dropout_p=0.0, causal=causal)
            ref = torch_reference_attention(
                q.permute(0, 2, 1, 3),
                k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
                causal=causal,
            ).permute(0, 2, 1, 3)
            err = (out.float() - ref).abs().max().item()

            t_ms = benchmark_torch_callable(
                lambda: official_flash_attn_func(q, k, v, dropout_p=0.0, causal=causal)
            )

        io_bytes = 4 * B * N * H * d * torch.finfo(dtype).bits // 8
        rows.append(
            {
                "method": "Official flash_attn (fp16)",
                "seq_len": N,
                "time_ms": f"{t_ms:.4f}",
                "memory_bytes": io_bytes,
                "max_error": f"{err:.6e}",
                "causal": 1,
            }
        )
        print(f"  N={N}: {t_ms:.4f} ms, max error={err:.3e}")

    out_path = RESULTS_DIR / "official_flash_attn_results.csv"
    write_rows(out_path, rows)
    print(f"Official flash_attn results written to {out_path}")
    return rows


def merge_results() -> list[dict]:
    project_path = RESULTS_DIR / "benchmark_results.csv"
    official_path = RESULTS_DIR / "official_flash_attn_results.csv"

    if not project_path.exists():
        raise FileNotFoundError("Expected results/benchmark_results.csv after running ./benchmark.")

    merged = load_rows(project_path)
    if official_path.exists():
        merged.extend(load_rows(official_path))

    out_path = RESULTS_DIR / "gpu_comparison_results.csv"
    write_rows(out_path, merged)
    print(f"Merged GPU comparison results written to {out_path}")
    return merged


def main() -> None:
    ensure_cuda_available()
    run_project_benchmark_binary()
    run_official_flash_attn_benchmarks()
    merge_results()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
