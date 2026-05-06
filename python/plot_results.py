"""
Generate GPU benchmark plots for the FA1 kernel suite.

The script plots:
  - Our FA results from `results/benchmark_results.csv`
  - merged comparison data from `results/gpu_comparison_results.csv` when available
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SPEEDUP_BASELINE = "PyTorch attention baseline (fp16)"

COLORS = {
    SPEEDUP_BASELINE: "#ff7f0e",
    "Our FA": "#1f77b4",
    "Official FlashAttention-1 (fp16)": "#9467bd",
    "Ablation: no tensor cores": "#d62728",
    "Ablation: no vectorized loads": "#bcbd22",
    "Ablation: no online softmax": "#7f7f7f",
    "Ablation: no SRAM tiling": "#17becf",
}

MARKERS = {
    SPEEDUP_BASELINE: "D",
    "Our FA": "o",
    "Official FlashAttention-1 (fp16)": "P",
    "Ablation: no tensor cores": "s",
    "Ablation: no vectorized loads": "v",
    "Ablation: no online softmax": "x",
    "Ablation: no SRAM tiling": "h",
}

ALLOWED_METHODS = set(COLORS)


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def group_by_method(rows: list[dict]) -> dict[str, dict[str, list[float]]]:
    groups = defaultdict(lambda: {"seq_lens": [], "times": [], "memory": [], "errors": []})
    for row in rows:
        method = row["method"]
        if method not in ALLOWED_METHODS:
            continue
        groups[method]["seq_lens"].append(int(row["seq_len"]))
        groups[method]["times"].append(float(row["time_ms"]))
        groups[method]["memory"].append(int(row["memory_bytes"]))
        groups[method]["errors"].append(float(row["max_error"]))
    return groups

def plot_runtime_scaling(groups, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, data in sorted(groups.items()):
        ax.plot(
            data["seq_lens"],
            data["times"],
            marker=MARKERS.get(method, "o"),
            color=COLORS.get(method, "#333333"),
            label=method,
            linewidth=1.6,
            markersize=6,
        )
    ax.set_xlabel("Sequence Length (N)")
    ax.set_ylabel("Time (ms)")
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_memory_scaling(groups, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    priority = [
        SPEEDUP_BASELINE,
        "Our FA",
        "Official FlashAttention-1 (fp16)",
    ]
    for method in priority:
        if method not in groups:
            continue
        data = groups[method]
        mem_mb = [m / 1e6 for m in data["memory"]]
        ax.plot(
            data["seq_lens"],
            mem_mb,
            marker=MARKERS.get(method, "o"),
            color=COLORS.get(method, "#333333"),
            label=method,
            linewidth=1.6,
            markersize=6,
        )
    ax.set_xlabel("Sequence Length (N)")
    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_speedup_vs_baseline(groups, title: str, output_path: Path) -> None:
    if SPEEDUP_BASELINE not in groups:
        return

    baseline_map = dict(zip(groups[SPEEDUP_BASELINE]["seq_lens"], groups[SPEEDUP_BASELINE]["times"]))
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, data in sorted(groups.items()):
        if method == SPEEDUP_BASELINE:
            continue
        seq_lens = []
        speedups = []
        for seq_len, time_ms in zip(data["seq_lens"], data["times"]):
            if seq_len in baseline_map and time_ms > 0:
                seq_lens.append(seq_len)
                speedups.append(baseline_map[seq_len] / time_ms)
        if seq_lens:
            ax.plot(
                seq_lens,
                speedups,
                marker=MARKERS.get(method, "o"),
                color=COLORS.get(method, "#333333"),
                label=method,
                linewidth=1.6,
                markersize=6,
            )

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6, label="PyTorch baseline")
    ax.set_xlabel("Sequence Length (N)")
    ax.set_ylabel("Speedup vs PyTorch baseline")
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_ablation_bars(groups, title: str, output_path: Path) -> None:
    candidates = [
        "Our FA",
        "Ablation: no tensor cores",
        "Ablation: no vectorized loads",
        "Ablation: no online softmax",
        "Ablation: no SRAM tiling",
    ]
    available = [method for method in candidates if method in groups]
    if len(available) < 2:
        return

    target_n = max(groups[available[0]]["seq_lens"])
    labels = []
    times = []
    colors = []
    for method in available:
        data = groups[method]
        if target_n not in data["seq_lens"]:
            continue
        idx = data["seq_lens"].index(target_n)
        labels.append(method.replace("Ablation: ", "- "))
        times.append(data["times"][idx])
        colors.append(COLORS.get(method, "#333333"))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(labels)), times, color=colors, height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"{title} (N={target_n})")
    ax.grid(True, axis="x", alpha=0.3)
    offset = max(times) * 0.01 if times else 0.0
    for bar, time_ms in zip(bars, times):
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{time_ms:.2f}",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def generate_from_csv(path: Path, label: str, prefix: str) -> None:
    rows = load_csv(path)
    groups = group_by_method(rows)
    plot_runtime_scaling(groups, f"Attention Runtime Scaling ({label})", RESULTS_DIR / f"{prefix}_runtime.pdf")
    plot_memory_scaling(groups, f"Peak GPU Memory ({label})", RESULTS_DIR / f"{prefix}_memory.pdf")
    plot_speedup_vs_baseline(groups, f"Speedup vs PyTorch baseline ({label})", RESULTS_DIR / f"{prefix}_speedup.pdf")
    plot_ablation_bars(groups, f"Ablation Comparison ({label})", RESULTS_DIR / f"{prefix}_ablation.pdf")


def main() -> None:
    generated_any = False
    specs = [
        ("benchmark_results.csv", "Our FA kernels", "project_gpu"),
        ("gpu_comparison_results.csv", "GPU comparison", "gpu_comparison"),
    ]
    for csv_name, label, prefix in specs:
        path = RESULTS_DIR / csv_name
        if path.exists():
            print(f"Using {csv_name}:")
            generate_from_csv(path, label, prefix)
            print()
            generated_any = True

    if not generated_any:
        print("No GPU benchmark CSVs found. Run `python python/run_benchmarks.py` on a CUDA machine first.")


if __name__ == "__main__":
    main()
