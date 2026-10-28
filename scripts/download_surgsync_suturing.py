"""
Safe, Resumable Downloader for SurgSync Suturing Subset (ITER-34).

Downloads exclusively the 'single_interrupted_stitch' recordings (92 episodes)
and dataset metadata from Hugging Face (jackzhy/surgsync), avoiding the 340+ GB
of unrelated tasks (dissection, peg transfer, tissue manipulation).

Includes:
- Disk space preflight check with conservative safety margin.
- Automatic resume capability on interrupted connections.
- Optional '--skip-video' mode to download only parquets & metadata (~80 MB).
- Post-download integrity verification against official meta/manifest.json.
- Dynamic destination directory (does not hardcode drive letters).

Usage:
    # 1. Preflight check (check free disk space and list required files without downloading):
    python scripts/download_surgsync_suturing.py --dest-dir E:/surgsync_v10 --preflight-only

    # 2. Download parquets and metadata only (~80 MB, suitable for development laptop):
    python scripts/download_surgsync_suturing.py --dest-dir data/surgsync_v10 --skip-video

    # 3. Full suturing download on mentor machine (~372 GB raw FFV1 video + parquets):
    python scripts/download_surgsync_suturing.py --dest-dir E:/surgsync_v10 --min-free-gb 450.0

    # 4. Verify integrity of an already-downloaded folder:
    python scripts/download_surgsync_suturing.py --dest-dir E:/surgsync_v10 --verify-only
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    from huggingface_hub import snapshot_download, HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


REPO_ID = "jackzhy/surgsync"
REPO_TYPE = "dataset"

# Expected size constants based on official meta/manifest.json audit
EXPECTED_SUTURING_TOTAL_BYTES = 372_412_000_000  # ~372.4 GB with video
EXPECTED_SUTURING_PARQUET_BYTES = 85_000_000     # ~85 MB without video
DEFAULT_SAFETY_BUFFER_GB = 75.0                  # Buffer beyond expected download size


class PreflightError(Exception):
    """Raised when preflight assertions (disk space, connectivity) fail."""
    pass


def get_disk_free_gb(path: Path) -> float:
    """Return available free disk space in gigabytes for the path's drive."""
    target = path.resolve()
    while not target.exists() and target.parent != target:
        target = target.parent
    usage = shutil.disk_usage(target)
    return usage.free / (1024 ** 3)


def check_preflight_space(dest_dir: Path, skip_video: bool, min_free_gb: Optional[float] = None) -> Tuple[bool, float, float]:
    """
    Check if destination volume has sufficient free space.
    
    Returns:
        (is_sufficient, free_gb, required_gb)
    """
    free_gb = get_disk_free_gb(dest_dir)
    if skip_video:
        required_gb = 1.0  # ~85 MB needed, 1 GB requirement
    else:
        base_gb = EXPECTED_SUTURING_TOTAL_BYTES / (1024 ** 3)
        required_gb = min_free_gb if min_free_gb is not None else (base_gb + DEFAULT_SAFETY_BUFFER_GB)
        
    return (free_gb >= required_gb), free_gb, required_gb


def verify_download_integrity(
    dest_dir: Path,
    manifest_path: Optional[Path] = None,
    verify_hashes: bool = False
) -> Dict[str, Any]:
    """
    Verify downloaded files against official manifest.json.
    
    Args:
        dest_dir: Directory containing downloaded dataset.
        manifest_path: Path to manifest.json (defaults to dest_dir/meta/manifest.json).
        verify_hashes: If True, computes full SHA-256 of all files.
        
    Returns:
        Dictionary with verification statistics and missing/mismatched files.
    """
    if manifest_path is None:
        manifest_path = dest_dir / "meta" / "manifest.json"

    if not manifest_path.is_file():
        return {
            "status": "FAILED",
            "error": f"Manifest file not found at {manifest_path}",
            "verified_count": 0,
            "missing_files": [],
        }

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    all_files: Dict[str, Dict[str, Any]] = manifest_data.get("files", {})
    # Filter for files that belong to single_interrupted_stitch or meta/
    relevant_files = {
        p: info for p, info in all_files.items()
        if "single_interrupted_stitch" in p or p.startswith("meta/")
    }

    missing = []
    size_mismatches = []
    hash_mismatches = []
    verified_count = 0

    print(f"\nVerifying {len(relevant_files)} files against manifest...")
    for rel_path, info in relevant_files.items():
        local_file = dest_dir / rel_path
        expected_size = info.get("size_bytes")
        expected_sha = info.get("sha256")

        if not local_file.is_file():
            missing.append(rel_path)
            continue

        actual_size = local_file.stat().st_size
        if expected_size is not None and actual_size != expected_size:
            size_mismatches.append({
                "file": rel_path,
                "expected": expected_size,
                "actual": actual_size
            })
            continue

        if verify_hashes and expected_sha:
            hasher = hashlib.sha256()
            with open(local_file, "rb") as bf:
                while chunk := bf.read(1024 * 1024):
                    hasher.update(chunk)
            actual_sha = hasher.hexdigest()
            if actual_sha != expected_sha:
                hash_mismatches.append({
                    "file": rel_path,
                    "expected": expected_sha,
                    "actual": actual_sha
                })
                continue

        verified_count += 1

    status = "PASSED" if (len(missing) == 0 and len(size_mismatches) == 0 and len(hash_mismatches) == 0) else "INCOMPLETE"
    return {
        "status": status,
        "total_relevant_files": len(relevant_files),
        "verified_count": verified_count,
        "missing_count": len(missing),
        "size_mismatch_count": len(size_mismatches),
        "hash_mismatch_count": len(hash_mismatches),
        "missing_files": missing[:20],
        "size_mismatches": size_mismatches[:10],
    }


def download_surgsync_suturing(
    dest_dir: Path,
    skip_video: bool = False,
    min_free_gb: Optional[float] = None,
    preflight_only: bool = False,
    verify_only: bool = False,
    verify_hashes: bool = False,
    max_workers: int = 4,
) -> int:
    """
    Main download execution routine.
    """
    dest_dir = dest_dir.resolve()
    print("=" * 70)
    print("  SurgSync Suturing Subset Acquisition Engine (ITER-34)")
    print("=" * 70)
    print(f"Target Repository : {REPO_ID} ({REPO_TYPE})")
    print(f"Destination Path  : {dest_dir}")
    print(f"Mode              : {'Parquets & Meta ONLY (~85 MB)' if skip_video else 'Full Suturing Subset (~372 GB with raw FFV1)'}")

    # 1. Preflight Disk Space Check
    is_sufficient, free_gb, required_gb = check_preflight_space(dest_dir, skip_video, min_free_gb)
    print(f"\n[Preflight] Free Disk Space    : {free_gb:.2f} GB")
    print(f"[Preflight] Required Threshold  : {required_gb:.2f} GB")

    if not is_sufficient:
        err_msg = (
            f"ERROR: Insufficient disk space on {dest_dir.anchor}.\n"
            f"Available: {free_gb:.2f} GB, Required: {required_gb:.2f} GB.\n"
            "Aborting acquisition to prevent disk overflow."
        )
        print(f"\n[FAIL] {err_msg}", file=sys.stderr)
        if not preflight_only:
            raise PreflightError(err_msg)
        return 1

    print("[Preflight] Disk Space Check    : PASSED")

    if preflight_only:
        print("\nPreflight-only requested. All storage checks passed. Exiting without downloading.")
        return 0

    if verify_only:
        print("\nVerify-only requested. Scanning destination directory...")
        res = verify_download_integrity(dest_dir, verify_hashes=verify_hashes)
        print(f"Integrity Status: {res['status']}")
        print(f"Verified Files  : {res['verified_count']} / {res['total_relevant_files']}")
        if res['missing_count'] > 0:
            print(f"Missing Files   : {res['missing_count']}")
        return 0 if res['status'] == "PASSED" else 1

    if not HF_AVAILABLE:
        print("\n[FAIL] 'huggingface_hub' package is required. Install via: pip install huggingface_hub", file=sys.stderr)
        return 1

    # 2. Configure Snapshot Download Patterns
    allow_patterns = [
        "meta/*",
        "online_data/episodes/single_interrupted_stitch/*",
        "offline_data/episodes/single_interrupted_stitch/*",
    ]
    ignore_patterns = ["*.mkv"] if skip_video else []

    dest_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"\nInitiating resumable download with {max_workers} worker threads...")
    print("Files will be checked and resumed if partial.")

    try:
        downloaded_path = snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            local_dir=str(dest_dir),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_workers=max_workers,
            resume_download=True,
        )
        elapsed = time.time() - t0
        print(f"\n[SUCCESS] Download completed in {elapsed:.1f} seconds.")
        print(f"Local Directory: {downloaded_path}")

        # Post-download verification
        print("\nRunning post-download manifest verification...")
        verif = verify_download_integrity(dest_dir, verify_hashes=False)
        print(f"Manifest Verification: {verif['status']} ({verif['verified_count']}/{verif['total_relevant_files']} files verified)")
        return 0 if verif['status'] == "PASSED" else 1

    except Exception as e:
        print(f"\n[ERROR] Download encountered an exception: {e}", file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(description="Download SurgSync Suturing subset from Hugging Face.")
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path("data/surgsync_v10"),
        help="Destination directory for downloaded dataset."
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Skip downloading large .mkv video files (~372 GB). Downloads only parquets and metadata (~85 MB)."
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        help="Minimum free disk space threshold in GB (defaults to 450 GB for full, 1 GB for skip-video)."
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check disk space and requirements without downloading."
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify integrity of an existing download against meta/manifest.json."
    )
    parser.add_argument(
        "--verify-hashes",
        action="store_true",
        help="When verifying, compute full SHA-256 hashes instead of size-only checks."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of concurrent download worker threads."
    )

    args = parser.parse_args()
    ret = download_surgsync_suturing(
        dest_dir=args.dest_dir,
        skip_video=args.skip_video,
        min_free_gb=args.min_free_gb,
        preflight_only=args.preflight_only,
        verify_only=args.verify_only,
        verify_hashes=args.verify_hashes,
        max_workers=args.max_workers,
    )
    sys.exit(ret)


if __name__ == "__main__":
    main()
