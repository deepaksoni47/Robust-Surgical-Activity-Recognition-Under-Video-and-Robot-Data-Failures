"""Helper utility to assemble the minimal SurgMotion GPU transfer package.

Copies only the necessary pre-tested code, scripts, checkpoint, pilot video,
and metadata into a standalone directory ('surgmotion_gpu_package/') ready to be
copied to a USB drive or transferred to the mentor's computer.

Usage:
    python scripts/package_surgmotion_transfer.py
    python scripts/package_surgmotion_transfer.py --zip
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_transfer_package(output_dir: Path, make_zip: bool = False) -> None:
    print("=" * 60)
    print("Assembling SurgMotion Minimum GPU Transfer Package")
    print("=" * 60)

    # Clean previous package directory if present
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Manifest of files to copy: (source_rel_path, target_rel_path)
    file_manifest = [
        # Checkpoint and config
        ("models/SurgMotion-L/SurgMotion-vitl.pt", "models/SurgMotion-L/SurgMotion-vitl.pt"),
        ("models/SurgMotion-L/config.json", "models/SurgMotion-L/config.json"),
        # Pilot video
        ("dataset/Suturing/video/Suturing_B001_capture1.avi", "dataset/Suturing/video/Suturing_B001_capture1.avi"),
        # Metadata manifest
        ("artifacts/manifest.csv", "artifacts/manifest.csv"),
        # Dependencies
        ("requirements-surgmotion.txt", "requirements-surgmotion.txt"),
        # Extraction & validation scripts
        ("scripts/extract_surgmotion_features.py", "scripts/extract_surgmotion_features.py"),
        ("scripts/validate_surgmotion_cache.py", "scripts/validate_surgmotion_cache.py"),
        # Protocol docs
        ("docs/surgmotion_gpu_handoff.md", "README_GPU.md"),
    ]

    total_bytes = 0

    for src_rel, dst_rel in file_manifest:
        src = PROJECT_ROOT / src_rel
        dst = output_dir / dst_rel
        if not src.is_file():
            raise FileNotFoundError(f"Required transfer file missing: {src.resolve()}")

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        size = dst.stat().st_size
        total_bytes += size
        print(f"  [+] {dst_rel} ({size / (1024*1024):.2f} MB)" if size > 1024*1024 else f"  [+] {dst_rel} ({size:,} B)")

    # 2. Copy necessary source modules (src/ without __pycache__)
    src_dir = PROJECT_ROOT / "src"
    dst_src_dir = output_dir / "src"
    for py_file in src_dir.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        rel_py = py_file.relative_to(src_dir)
        target_py = dst_src_dir / rel_py
        target_py.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(py_file, target_py)
        size = target_py.stat().st_size
        total_bytes += size

    print(f"  [+] src/ (all Python modules copied without cache)")

    print("-" * 60)
    print(f"Transfer Package assembled at: {output_dir.resolve()}")
    print(f"Total payload size: {total_bytes / (1024 * 1024):.2f} MB ({total_bytes / (1024**3):.2f} GB)")

    if make_zip:
        zip_path = PROJECT_ROOT / "artifacts" / "surgmotion_gpu_package"
        print(f"Creating zip archive at {zip_path}.zip ...")
        shutil.make_archive(str(zip_path), "zip", output_dir)
        zip_file = Path(str(zip_path) + ".zip")
        print(f"Archive ready: {zip_file.name} ({zip_file.stat().st_size / (1024*1024):.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble minimal SurgMotion GPU transfer package.")
    parser.add_argument("--output-dir", type=str, default="surgmotion_gpu_package",
                        help="Target directory for assembled package")
    parser.add_argument("--zip", action="store_true",
                        help="Also produce a compressed .zip archive in artifacts/")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    create_transfer_package(out_dir, make_zip=args.zip)


if __name__ == "__main__":
    main()
