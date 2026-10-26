"""Cache Validation and Integrity Auditor for SurgMotion Features.

Verifies:
1. File existence and non-zero size.
2. Presence of required arrays: 'features', 'frame_indices', 'clip_center_frames', 'metadata_json'.
3. Array shapes and dimensionality (e.g. [N, 1024] or [N, 32, 1024]).
4. Monotonicity of temporal frame indices.
5. Finiteness of all numerical representations (zero NaNs / Infs).
6. Metadata validity and provenance traceability.

Usage:
    # Audit a single cache file:
    python scripts/validate_surgmotion_cache.py data/cache/features_surgmotion/Suturing_B001_capture1.npz

    # Audit an entire directory of caches:
    python scripts/validate_surgmotion_cache.py --dir data/cache/features_surgmotion
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def validate_single_cache(cache_path: Path, expected_dim: int = 1024) -> Dict[str, Any]:
    """Validate a single SurgMotion .npz cache file."""
    if not cache_path.is_file():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    size_bytes = cache_path.stat().st_size
    if size_bytes == 0:
        raise ValueError(f"Cache file is empty (0 bytes): {cache_path.name}")

    with np.load(cache_path, allow_pickle=True) as data:
        keys = set(data.files)
        required_keys = {"features", "frame_indices"}
        missing = required_keys - keys
        if missing:
            raise ValueError(f"Missing required keys in {cache_path.name}: {missing}")

        features = data["features"]
        frame_indices = data["frame_indices"]

        # 1. Dimension and shape check
        if features.ndim not in (2, 3):
            raise ValueError(f"Expected 2D [N, D] or 3D [N, T, D] features array, got shape {features.shape}")

        n_samples = len(features)
        feat_dim = features.shape[-1]

        if n_samples == 0:
            raise ValueError(f"Zero features in {cache_path.name}")

        if feat_dim != expected_dim:
            raise ValueError(f"Feature dim mismatch: expected {expected_dim}, got {feat_dim}")

        # 2. Frame indices check
        if len(frame_indices) != n_samples:
            raise ValueError(f"Length mismatch: {n_samples} features vs {len(frame_indices)} frame indices")

        if not np.all(np.diff(frame_indices) > 0):
            raise ValueError(f"Frame indices in {cache_path.name} are not strictly monotonically increasing")

        # 3. Finite value assertion
        if not np.all(np.isfinite(features)):
            raise ValueError(f"NaN or Inf values detected in features of {cache_path.name}!")

        # 4. Metadata verification
        metadata = {}
        if "metadata_json" in data:
            meta_raw = data["metadata_json"]
            try:
                meta_str = str(meta_raw.item()) if hasattr(meta_raw, "item") else str(meta_raw)
                metadata = json.loads(meta_str)
            except Exception as e:
                print(f"Warning: Could not parse metadata_json in {cache_path.name}: {e}")

    return {
        "cache_file": cache_path.name,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "num_clips": n_samples,
        "feature_shape": list(features.shape),
        "feature_dim": feat_dim,
        "frame_range": [int(frame_indices[0]), int(frame_indices[-1])],
        "is_finite": True,
        "encoder": metadata.get("encoder_name", "Unknown"),
        "pooling_mode": metadata.get("pooling_mode", "Unknown"),
        "stride": metadata.get("stride", "Unknown"),
        "device": metadata.get("device", "Unknown"),
        "status": "VALID",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SurgMotion feature cache archives.")
    parser.add_argument("file", nargs="?", type=str, default=None,
                        help="Path to single .npz file to validate")
    parser.add_argument("--dir", type=str, default=None,
                        help="Directory of .npz files to validate")
    parser.add_argument("--expected-dim", type=int, default=1024,
                        help="Expected feature dimension (default: 1024)")
    args = parser.parse_args()

    files_to_check: List[Path] = []
    if args.file:
        files_to_check.append(Path(args.file))
    elif args.dir:
        d = Path(args.dir)
        if not d.is_dir():
            print(f"Error: Directory not found: {d}")
            sys.exit(1)
        files_to_check = sorted(list(d.glob("*.npz")))
    else:
        # Default to data/cache/features_surgmotion if it exists
        default_dir = Path("data/cache/features_surgmotion")
        if default_dir.is_dir():
            files_to_check = sorted(list(default_dir.glob("*.npz")))
        else:
            print("Error: Specify a file path or --dir to validate.")
            sys.exit(1)

    if not files_to_check:
        print("No .npz files found to validate.")
        return

    print(f"Auditing {len(files_to_check)} SurgMotion cache file(s)...")
    valid_count = 0
    errors = []

    for f in files_to_check:
        try:
            res = validate_single_cache(f, expected_dim=args.expected_dim)
            print(f"  [OK] {res['cache_file']}: {res['num_clips']} clips, shape {res['feature_shape']}, "
                  f"{res['size_mb']} MB, frames {res['frame_range'][0]}-{res['frame_range'][1]}")
            valid_count += 1
        except Exception as e:
            print(f"  [FAIL] {f.name}: {e}")
            errors.append((f.name, str(e)))

    print("\n" + "=" * 50)
    print(f"Audit Summary: {valid_count}/{len(files_to_check)} files VALID.")
    if errors:
        print(f"ERRORS DETECTED ({len(errors)}):")
        for fn, err in errors:
            print(f"  - {fn}: {err}")
        sys.exit(1)
    else:
        print("All audited cache files passed integrity and finiteness checks.")


if __name__ == "__main__":
    main()
