"""
GPU benchmark runner for the simplified FlashAttention project.

This script runs the local CUDA benchmark binary and then merges any separately
generated official FlashAttention CSVs into one comparison file.

Official FlashAttention-1 and FlashAttention-2 should be benchmarked from
separate invocations of `python/benchmark_official_flash_attn.py`. In this
project setup, FA1 is imported from a source tree via `--source-dir`, while
FA2 is imported from the installed `flash_attn` wheel.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


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


def run_project_benchmark_binary() -> None:
    benchmark_bin = ROOT_DIR / "benchmark"
    if not benchmark_bin.exists():
        raise FileNotFoundError(
            "Missing ./benchmark. Build it first with `make benchmark` or `make run`."
        )

    print("Running local CUDA benchmark binary...")
    subprocess.run([str(benchmark_bin)], cwd=ROOT_DIR, check=True)


def merge_results() -> list[dict]:
    project_path = RESULTS_DIR / "benchmark_results.csv"
    if not project_path.exists():
        raise FileNotFoundError("Expected results/benchmark_results.csv after running ./benchmark.")

    merged = load_rows(project_path)
    optional_paths = [
        RESULTS_DIR / "official_flash_attn_v1_results.csv",
        RESULTS_DIR / "official_flash_attn_v2_results.csv",
    ]
    for path in optional_paths:
        if path.exists():
            merged.extend(load_rows(path))

    out_path = RESULTS_DIR / "gpu_comparison_results.csv"
    write_rows(out_path, merged)
    print(f"Merged GPU comparison results written to {out_path}")
    return merged


def main() -> None:
    run_project_benchmark_binary()
    merge_results()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
