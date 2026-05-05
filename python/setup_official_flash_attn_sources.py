"""
Download and extract official FlashAttention source distributions side by side.

This script keeps FA1 and FA2 in separate folders so they can be built and
benchmarked from a single Python environment by swapping `--source-dir`.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DEST = ROOT_DIR / "external"
SPECS = [
    ("1.0.9", "flash-attn-fa1"),
    ("2.8.3", "flash-attn-fa2"),
]


def extract_tarball(tar_path: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        top_level = members[0].name.split("/")[0]
        for member in members:
            relative = Path(member.name).relative_to(top_level)
            if not str(relative):
                continue
            member.name = str(relative)
            tar.extract(member, target_dir)


def main() -> None:
    dest = DEFAULT_DEST
    dest.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for version, folder_name in SPECS:
            print(f"Downloading flash-attn=={version} source distribution...")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "download",
                    "--no-binary",
                    ":all:",
                    "--no-deps",
                    f"flash-attn=={version}",
                    "-d",
                    str(tmp_path),
                ],
                check=True,
            )
            tarballs = sorted(tmp_path.glob(f"flash_attn-{version}*.tar.gz"))
            if not tarballs:
                raise FileNotFoundError(f"Could not find downloaded tarball for flash-attn=={version}")
            extract_tarball(tarballs[-1], dest / folder_name)
            print(f"  Extracted to {dest / folder_name}")

    print("\nNext steps:")
    print(f"  cd {dest / 'flash-attn-fa1'} && python setup.py build_ext --inplace")
    print(f"  cd {dest / 'flash-attn-fa2'} && MAX_JOBS=4 python setup.py build_ext --inplace")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
