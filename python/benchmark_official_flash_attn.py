"""
Benchmark one installed official FlashAttention version.

Examples:
  python python/benchmark_official_flash_attn.py --version fa1
  python python/benchmark_official_flash_attn.py --version fa2
  python python/benchmark_official_flash_attn.py --version fa1 --source-dir external/flash-attn-fa1
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CSV_FIELDS = ["method", "seq_len", "time_ms", "memory_bytes", "max_error", "causal"]
PYTORCH_BASELINE_LABEL = "PyTorch attention baseline (fp16)"


def maybe_prepend_source_dir(source_dir: str | None) -> None:
    if source_dir is None:
        return
    source_path = Path(source_dir).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_path}")
    sys.path.insert(0, str(source_path))


def write_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


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


def attention_pytorch_qkvpacked(qkv: torch.Tensor, causal: bool) -> torch.Tensor:
    """PyTorch eager attention baseline adapted from the official FA benchmark script."""
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = q.permute(0, 2, 1, 3).reshape(batch_size * nheads, seqlen, d)
    k = k.permute(0, 2, 3, 1).reshape(batch_size * nheads, d, seqlen)
    softmax_scale = 1.0 / math.sqrt(d)
    scores = torch.empty(
        batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device
    )
    scores = torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale).reshape(
        batch_size, nheads, seqlen, seqlen
    )
    if causal:
        causal_mask = torch.triu(
            torch.full((seqlen, seqlen), -10000.0, device=scores.device),
            diagonal=1,
        )
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    output = torch.einsum("bhts,bshd->bthd", attention, v)
    return output.to(dtype=qkv.dtype)


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


def measure_peak_memory_bytes(fn, warmup: int = 10) -> int:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def get_fa1_callable():
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func

    def call(qkv_unpadded: torch.Tensor, cu_seqlens: torch.Tensor, seqlen: int, causal: bool) -> torch.Tensor:
        out = flash_attn_unpadded_qkvpacked_func(
            qkv_unpadded, cu_seqlens, seqlen, 0.0, softmax_scale=None, causal=causal
        )
        bsz = cu_seqlens.numel() - 1
        _, _, nheads, d = qkv_unpadded.shape
        return out.reshape(bsz, seqlen, nheads, d)

    return call


def get_fa2_callable():
    from flash_attn import flash_attn_qkvpacked_func

    def call(qkv: torch.Tensor, causal: bool) -> torch.Tensor:
        return flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=causal)

    return call


def run_benchmarks(version: str, source_dir: str | None) -> list[dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run this on a CUDA-enabled machine.")

    maybe_prepend_source_dir(source_dir)

    device = torch.device("cuda")
    dtype = torch.float16
    B, H, d = 2, 8, 64
    seq_lens = [128, 256, 512, 1024, 2048, 4096]
    causal = True

    if version == "fa1":
        label = "Official FlashAttention-1 (fp16)"
        flash_fn = get_fa1_callable()
        out_path = RESULTS_DIR / "official_flash_attn_v1_results.csv"
    elif version == "fa2":
        label = "Official FlashAttention-2 (fp16)"
        flash_fn = get_fa2_callable()
        out_path = RESULTS_DIR / "official_flash_attn_v2_results.csv"
    else:
        raise ValueError(f"Unsupported version: {version}")

    rows = []
    baseline_rows = []
    print(f"Running {label} benchmarks...")
    for N in seq_lens:
        torch.manual_seed(42)
        q = torch.randn((B, N, H, d), device=device, dtype=dtype)
        k = torch.randn((B, N, H, d), device=device, dtype=dtype)
        v = torch.randn((B, N, H, d), device=device, dtype=dtype)
        qkv = torch.stack([q, k, v], dim=2)
        qkv_unpadded = qkv.reshape(B * N, 3, H, d)
        cu_seqlens = torch.arange(0, (B + 1) * N, step=N, dtype=torch.int32, device=device)

        with torch.no_grad():
            if version == "fa1":
                out = flash_fn(qkv_unpadded, cu_seqlens, N, causal=causal)
                timed_flash = lambda: flash_fn(qkv_unpadded, cu_seqlens, N, causal=causal)
            else:
                out = flash_fn(qkv, causal=causal)
                timed_flash = lambda: flash_fn(qkv, causal=causal)
            ref = torch_reference_attention(
                q.permute(0, 2, 1, 3),
                k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
                causal=causal,
            ).permute(0, 2, 1, 3)
            err = (out.float() - ref).abs().max().item()
            t_ms = benchmark_torch_callable(timed_flash)
            mem_bytes = measure_peak_memory_bytes(timed_flash)
            baseline_out = attention_pytorch_qkvpacked(qkv, causal=causal)
            baseline_err = (baseline_out.float() - ref).abs().max().item()
            baseline_t_ms = benchmark_torch_callable(lambda: attention_pytorch_qkvpacked(qkv, causal=causal))
            baseline_mem_bytes = measure_peak_memory_bytes(lambda: attention_pytorch_qkvpacked(qkv, causal=causal))

        rows.append(
            {
                "method": label,
                "seq_len": N,
                "time_ms": f"{t_ms:.4f}",
                "memory_bytes": mem_bytes,
                "max_error": f"{err:.6e}",
                "causal": 1,
            }
        )
        baseline_rows.append(
            {
                "method": PYTORCH_BASELINE_LABEL,
                "seq_len": N,
                "time_ms": f"{baseline_t_ms:.4f}",
                "memory_bytes": baseline_mem_bytes,
                "max_error": f"{baseline_err:.6e}",
                "causal": 1,
            }
        )
        print(f"  N={N}: {t_ms:.4f} ms, peak memory={mem_bytes} bytes, max error={err:.3e}")

    write_rows(out_path, rows)
    write_rows(RESULTS_DIR / "official_pytorch_baseline_results.csv", baseline_rows)
    print(f"Wrote {out_path}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["fa1", "fa2"], required=True)
    parser.add_argument(
        "--source-dir",
        help="Optional built official flash-attn source tree to import from, used for FA1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmarks(args.version, args.source_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
