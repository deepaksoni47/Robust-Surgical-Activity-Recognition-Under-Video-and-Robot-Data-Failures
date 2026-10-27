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


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hex digest of a file in chunks."""
    import hashlib
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_transfer_package(
    output_dir: Path,
    all_videos: bool = False,
    include_capture2: bool = False,
    with_pilot: bool = True,
    make_zip: bool = False,
    zip_name: str = "surgmotion_gpu_package",
) -> None:
    print("=" * 65)
    print("Assembling SurgMotion GPU Transfer Package (Laptop -> Mentor Machine)")
    print("=" * 65)

    # Clean previous package directory if present
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Base manifest of essential files
    file_manifest = [
        # Checkpoint and config
        ("models/SurgMotion-L/SurgMotion-vitl.pt", "models/SurgMotion-L/SurgMotion-vitl.pt"),
        ("models/SurgMotion-L/config.json", "models/SurgMotion-L/config.json"),
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

    # Pilot video (always included if not all_videos)
    if not all_videos:
        file_manifest.append(
            ("dataset/Suturing/video/Suturing_B001_capture1.avi", "dataset/Suturing/video/Suturing_B001_capture1.avi")
        )

    # Pilot outputs for reference and safe resumption
    if with_pilot:
        pilot_npz = PROJECT_ROOT / "data/cache/features_surgmotion/Suturing_B001_capture1.npz"
        pilot_json = PROJECT_ROOT / "data/cache/features_surgmotion/extraction_summary_Suturing.json"
        if pilot_npz.is_file():
            file_manifest.append(("data/cache/features_surgmotion/Suturing_B001_capture1.npz", "data/cache/features_surgmotion/Suturing_B001_capture1.npz"))
        if pilot_json.is_file():
            file_manifest.append(("data/cache/features_surgmotion/extraction_summary_Suturing.json", "data/cache/features_surgmotion/extraction_summary_Suturing.json"))

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

    # 2. Add complete video inputs from manifest if requested
    if all_videos:
        print("\nAdding all JIGSAWS video trials from manifest.csv ...")
        manifest_csv = PROJECT_ROOT / "artifacts/manifest.csv"
        df = pd.read_csv(manifest_csv)
        video_count = 0
        video_bytes = 0

        for _, row in df.iterrows():
            # Capture 1 (canonical benchmark camera)
            p1_rel = row["video_capture1_path"]
            src_p1 = PROJECT_ROOT / p1_rel
            if src_p1.is_file():
                dst_p1 = output_dir / p1_rel
                dst_p1.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_p1, dst_p1)
                sz = src_p1.stat().st_size
                video_count += 1
                video_bytes += sz
                total_bytes += sz

            # Capture 2 (optional stereo stream)
            if include_capture2:
                p2_rel = row["video_capture2_path"]
                src_p2 = PROJECT_ROOT / p2_rel
                if src_p2.is_file():
                    dst_p2 = output_dir / p2_rel
                    dst_p2.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_p2, dst_p2)
                    sz = src_p2.stat().st_size
                    video_count += 1
                    video_bytes += sz
                    total_bytes += sz

        print(f"  [+] Added {video_count} video files ({video_bytes / (1024*1024):.2f} MB)")

    # 3. Copy necessary source modules (src/ without __pycache__)
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

    # 4. Generate SHA256 integrity manifest
    print("\nComputing SHA-256 integrity manifest for all package files ...")
    checksum_lines = []
    for fpath in sorted(output_dir.rglob("*")):
        if fpath.is_file():
            rel = fpath.relative_to(output_dir).as_posix()
            digest = compute_sha256(fpath)
            checksum_lines.append(f"{digest}  {rel}")

    checksum_file = output_dir / "file_integrity_manifest.sha256"
    checksum_file.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(f"  [+] file_integrity_manifest.sha256 ({len(checksum_lines)} files indexed)")

    # 5. Generate Overnight Execution Guide
    readme_content = f"""# SurgMotion Full JIGSAWS Overnight Extraction Guide

**Environment:** Mentor's Workstation (Quadro RTX 5000 / Xeon, Windows 11)  
**Package Origin:** Laptop primary development environment (`D:\\miniproject-5thsem`)  
**Target:** 103 JIGSAWS trials (284,644 video frames -> 17,433 clips)  
**Estimated Time:** ~1.75 hours (106 minutes) at batch size 8 on CUDA FP16  

---

## Phase 1 — Verify Integrity on Mentor's Computer
Open PowerShell in the unpacked package directory:
```powershell
python -c "
with open('file_integrity_manifest.sha256') as f:
    for line in f:
        sha, rel = line.strip().split('  ')
        import hashlib, pathlib
        p = pathlib.Path(rel)
        if not p.is_file():
            print(f'MISSING: {{rel}}')
            continue
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != sha:
            print(f'CORRUPT: {{rel}}')
print('Integrity check complete.')
"
```

## Phase 2 — Quick Pre-Flight Check (Smoke Test + Single Trial)
1. Single-clip GPU diagnostic:
```powershell
python scripts/extract_surgmotion_features.py --smoke-test --device cuda --fp16
```
2. Verify with a second trial (e.g. Suturing_B002):
```powershell
python scripts/extract_surgmotion_features.py --task Suturing --trial-id Suturing_B002 --device cuda --fp16 --batch-size 8 --stride 16
python scripts/validate_surgmotion_cache.py data/cache/features_surgmotion/Suturing_B002_capture1.npz
```

## Phase 3 — Full Overnight Extraction Commands
Execute task by task for clean progress tracking and separate summaries:

```powershell
# 1. Suturing (39 trials, ~8,097 clips, ~49 mins)
# Note: Suturing_B001 is automatically skipped as it is already present in cache!
python scripts/extract_surgmotion_features.py --task Suturing --device cuda --fp16 --batch-size 8 --stride 16 --output-dir data/cache/features_surgmotion

# 2. Knot Tying (36 trials, ~3,738 clips, ~23 mins)
python scripts/extract_surgmotion_features.py --task Knot_Tying --device cuda --fp16 --batch-size 8 --stride 16 --output-dir data/cache/features_surgmotion

# 3. Needle Passing (28 trials, ~5,598 clips, ~34 mins)
python scripts/extract_surgmotion_features.py --task Needle_Passing --device cuda --fp16 --batch-size 8 --stride 16 --output-dir data/cache/features_surgmotion
```

## Phase 4 — Validate Entire Output Directory
```powershell
python scripts/validate_surgmotion_cache.py --dir data/cache/features_surgmotion
```
*Expected:* 103 valid `.npz` files, 0 NaNs, 0 Infs, total size ~65 MB.

## Phase 5 — Return to Laptop
Copy only the `data/cache/features_surgmotion/` folder back to the laptop:
`D:\\miniproject-5thsem\\data\\cache\\features_surgmotion\\`
"""
    (output_dir / "README_OVERNIGHT.md").write_text(readme_content, encoding="utf-8")
    print(f"  [+] README_OVERNIGHT.md generated")

    print("-" * 65)
    print(f"Transfer Package assembled at: {output_dir.resolve()}")
    print(f"Total payload size: {total_bytes / (1024 * 1024):.2f} MB ({total_bytes / (1024**3):.2f} GB)")

    if make_zip:
        zip_path = PROJECT_ROOT / "artifacts" / zip_name
        print(f"\nCreating zip archive at {zip_path}.zip ...")
        shutil.make_archive(str(zip_path), "zip", output_dir)
        zip_file = Path(str(zip_path) + ".zip")
        print(f"Archive ready: {zip_file.name} ({zip_file.stat().st_size / (1024*1024):.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble minimal SurgMotion GPU transfer package.")
    parser.add_argument("--output-dir", type=str, default="surgmotion_gpu_package",
                        help="Target directory for assembled package")
    parser.add_argument("--all-videos", action="store_true",
                        help="Include all 103 JIGSAWS video trials from manifest.csv")
    parser.add_argument("--include-capture2", action="store_true",
                        help="Also include capture2 stereo videos")
    parser.add_argument("--no-pilot", action="store_true",
                        help="Do not include pilot cache and summary")
    parser.add_argument("--zip", action="store_true",
                        help="Also produce a compressed .zip archive in artifacts/")
    parser.add_argument("--zip-name", type=str, default="surgmotion_gpu_package",
                        help="Filename for the zip archive in artifacts/")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    create_transfer_package(
        output_dir=out_dir,
        all_videos=args.all_videos,
        include_capture2=args.include_capture2,
        with_pilot=not args.no_pilot,
        make_zip=args.zip,
        zip_name=args.zip_name,
    )


if __name__ == "__main__":
    import pandas as pd
    main()

