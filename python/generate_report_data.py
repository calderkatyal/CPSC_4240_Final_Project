"""Generate LaTeX table rows from benchmark CSV results."""

from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

RUNTIME_METHODS = [
    ("Naive", "Naive"),
    ("Simplified FA1", "Simplified FA1"),
    ("FA2-inspired extension", "FA2-inspired extension"),
    ("Official FlashAttention-1 (fp16)", "Official FlashAttention-1"),
    ("Official FlashAttention-2 (fp16)", "Official FlashAttention-2"),
]

SPEEDUP_METHODS = [
    ("Simplified FA1", "Simplified FA1"),
    ("FA2-inspired extension", "FA2-inspired extension"),
    ("Official FlashAttention-1 (fp16)", "Official FlashAttention-1"),
    ("Official FlashAttention-2 (fp16)", "Official FlashAttention-2"),
]

ABLATION_METHODS = [
    ("Simplified FA1", "Simplified FA1"),
    ("Ablation: no online softmax", "Ablation: no online softmax"),
    ("Ablation: no SRAM tiling", "Ablation: no SRAM tiling"),
]

TABLE_SEQS = [512, 1024, 2048, 4096]
ABLATION_SEQ = 2048


def load_rows() -> list[dict[str, str]]:
    preferred = RESULTS_DIR / "gpu_comparison_results.csv"
    source = preferred
    if not source.exists():
        raise FileNotFoundError(
            "Could not find merged comparison CSV. Expected "
            f"{preferred}."
        )

    with source.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {source}.")
    return rows


def build_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    return {(row["method"], int(row["seq_len"])): row for row in rows}


def tex_escape(text: str) -> str:
    return text.replace("&", r"\&")


def format_ms(value: str | None) -> str:
    if value is None:
        return r"\blankcell"
    return f"{float(value):.4f}"


def format_speedup(value: float | None) -> str:
    if value is None:
        return r"\blankcell"
    return f"{value:.2f}$\\times$"


def format_error(value: str | None) -> str:
    if value is None:
        return r"\blankcell"
    return f"{float(value):.2e}"


def row_value(
    lookup: dict[tuple[str, int], dict[str, str]], method: str, seq: int, field: str
) -> str | None:
    row = lookup.get((method, seq))
    if row is None:
        return None
    return row[field]


def make_runtime_rows(lookup: dict[tuple[str, int], dict[str, str]]) -> str:
    lines = []
    for method_key, label in RUNTIME_METHODS:
        values = [
            format_ms(row_value(lookup, method_key, seq, "time_ms")) for seq in TABLE_SEQS
        ]
        lines.append(f"{tex_escape(label)} & " + " & ".join(values) + r" \\%")
    return "\n".join(lines) + "\n"


def make_speedup_rows(lookup: dict[tuple[str, int], dict[str, str]]) -> str:
    lines = []
    for method_key, label in SPEEDUP_METHODS:
        values = []
        for seq in TABLE_SEQS:
            naive = row_value(lookup, "Naive", seq, "time_ms")
            method = row_value(lookup, method_key, seq, "time_ms")
            if naive is None or method is None:
                values.append(r"\blankcell")
            else:
                values.append(format_speedup(float(naive) / float(method)))
        lines.append(f"{tex_escape(label)} & " + " & ".join(values) + r" \\%")
    return "\n".join(lines) + "\n"


def make_ablation_rows(lookup: dict[tuple[str, int], dict[str, str]]) -> str:
    naive = row_value(lookup, "Naive", ABLATION_SEQ, "time_ms")
    lines = []
    for method_key, label in ABLATION_METHODS:
        runtime = row_value(lookup, method_key, ABLATION_SEQ, "time_ms")
        error = row_value(lookup, method_key, ABLATION_SEQ, "max_error")
        if naive is None or runtime is None:
            speedup = r"\blankcell"
        else:
            speedup = format_speedup(float(naive) / float(runtime))
        lines.append(
            f"{tex_escape(label)} & {format_ms(runtime)} & {speedup} & {format_error(error)} \\\\%"
        )
    return "\n".join(lines) + "\n"


def make_table_macro_file(lookup: dict[tuple[str, int], dict[str, str]]) -> str:
    return "\n".join(
        [
            r"\newcommand{\runtimeTableRows}{%",
            make_runtime_rows(lookup).rstrip(),
            "}",
            r"\newcommand{\speedupTableRows}{%",
            make_speedup_rows(lookup).rstrip(),
            "}",
            r"\newcommand{\ablationTableRows}{%",
            make_ablation_rows(lookup).rstrip(),
            "}",
            "",
        ]
    )


def write_results_file(name: str, contents: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / name).write_text(contents)


def main() -> None:
    rows = load_rows()
    lookup = build_lookup(rows)
    write_results_file("table_rows.tex", make_table_macro_file(lookup))
    print(f"Wrote report table data to {RESULTS_DIR / 'table_rows.tex'}")


if __name__ == "__main__":
    main()
