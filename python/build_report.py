"""Build the PDF report from existing results artifacts."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = REPO_ROOT / "python"
REPORT_DIR = REPO_ROOT / "report"
RESULTS_DIR = REPO_ROOT / "results"

REQUIRED_RESULT_FILES = [
    RESULTS_DIR / "gpu_comparison_results.csv",
    RESULTS_DIR / "gpu_comparison_runtime.pdf",
    RESULTS_DIR / "gpu_comparison_speedup.pdf",
    RESULTS_DIR / "gpu_comparison_memory.pdf",
    RESULTS_DIR / "gpu_comparison_ablation.pdf",
]


def run_step(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd or REPO_ROOT, check=True)


def main() -> None:
    if shutil.which("pdflatex") is None:
        raise SystemExit("pdflatex is not available on PATH.")

    missing = [path for path in REQUIRED_RESULT_FILES if not path.exists()]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "Missing required results artifacts. Generate benchmarks/plots first:\n"
            f"{missing_list}"
        )

    python = sys.executable
    run_step([python, str(PYTHON_DIR / "generate_report_data.py")])
    run_step(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
        cwd=REPORT_DIR,
    )
    run_step(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
        cwd=REPORT_DIR,
    )
    print(f"Built report PDF at {REPORT_DIR / 'report.pdf'}")


if __name__ == "__main__":
    main()
