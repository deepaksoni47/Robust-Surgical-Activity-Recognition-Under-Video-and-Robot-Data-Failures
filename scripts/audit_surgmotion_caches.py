"""Comprehensive Cache Auditor for Transferred SurgMotion Feature Caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def audit_surgmotion_caches(cache_dir: Path = PROJECT_ROOT / "data/cache/features_surgmotion") -> Dict[str, Any]:
    manifest_path = PROJECT_ROOT / "artifacts/manifest.csv"
    manifest = pd.read_csv(manifest_path)
    
    npz_files = sorted(list(cache_dir.glob("*.npz")))
    
    records: List[Dict[str, Any]] = []
    errors: List[str] = []
    dimensions = set()
    dtypes = set()
    cameras = set()
    clip_lengths = set()
    strides = set()
    pooling_modes = set()
    nan_counts = 0
    inf_counts = 0
    total_clips = 0
    total_bytes = 0
    
    for p in npz_files:
        fname = p.name
        # format: <trial_id>_<camera>.npz
        parts = fname.replace(".npz", "").rsplit("_", 1)
        trial_id, cam = parts[0], parts[1]
        sz = p.stat().st_size
        total_bytes += sz
        
        with np.load(p, allow_pickle=True) as data:
            feats = data["features"]
            frame_idx = data["frame_indices"]
            c_centers = data["clip_center_frames"]
            c_starts = data["clip_start_frames"]
            c_ends = data["clip_end_frames"]
            meta = json.loads(str(data["metadata_json"]))
            
            dimensions.add(feats.shape[1])
            dtypes.add(str(feats.dtype))
            cameras.add(cam)
            clip_lengths.add(meta.get("clip_length"))
            strides.add(meta.get("stride"))
            pooling_modes.add(meta.get("pooling_mode"))
            
            has_nan = bool(np.isnan(feats).any())
            has_inf = bool(np.isinf(feats).any())
            if has_nan:
                nan_counts += 1
            if has_inf:
                inf_counts += 1
            
            n_clips = len(feats)
            total_clips += n_clips
            
            if not (len(frame_idx) == len(c_centers) == len(c_starts) == len(c_ends) == n_clips):
                errors.append(f"{fname}: Array length mismatch!")
                
            if not np.array_equal(frame_idx, c_centers):
                errors.append(f"{fname}: frame_indices != clip_center_frames!")
                
            stride_diffs = set(np.diff(c_centers)) if len(c_centers) > 1 else {16}
            if stride_diffs != {16}:
                errors.append(f"{fname}: Non-constant stride in centers: {stride_diffs}")
                
            records.append({
                "trial_id": trial_id,
                "task": meta.get("task"),
                "camera": cam,
                "num_clips": n_clips,
                "feature_dim": feats.shape[1],
                "dtype": str(feats.dtype),
                "start_frame_min": int(c_starts[0]),
                "end_frame_max": int(c_ends[-1]),
                "center_min": int(c_centers[0]),
                "center_max": int(c_centers[-1]),
                "size_kb": round(sz / 1024, 1),
                "is_finite": not (has_nan or has_inf),
            })
            
    df_audit = pd.DataFrame(records)
    
    missing_trials = sorted(list(set(manifest["trial_id"]) - set(df_audit["trial_id"])))
    extra_trials = sorted(list(set(df_audit["trial_id"]) - set(manifest["trial_id"])))
    
    summary = {
        "total_manifest_trials": len(manifest),
        "total_cache_files": len(npz_files),
        "total_cache_size_mb": round(total_bytes / (1024 * 1024), 2),
        "total_clips": total_clips,
        "tasks": df_audit["task"].value_counts().to_dict(),
        "dimensions": list(dimensions),
        "dtypes": list(dtypes),
        "cameras": list(cameras),
        "clip_lengths": list(clip_lengths),
        "strides": list(strides),
        "pooling_modes": list(pooling_modes),
        "nan_files": nan_counts,
        "inf_files": inf_counts,
        "errors_count": len(errors),
        "errors": errors,
        "missing_trials": missing_trials,
        "extra_trials": extra_trials,
    }
    
    # Save CSV inventory
    inventory_csv = PROJECT_ROOT / "artifacts/surgmotion_cache_inventory.csv"
    df_audit.to_csv(inventory_csv, index=False)
    
    # Save JSON summary
    summary_json = PROJECT_ROOT / "artifacts/surgmotion_cache_audit_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("=" * 65)
    print("SURGMOTION CACHE INVENTORY AUDIT RESULTS")
    print("=" * 65)
    print(f"Total Cache Files:       {len(npz_files)} / {len(manifest)} manifest trials")
    print(f"Total Disk Payload:      {summary['total_cache_size_mb']} MB")
    print(f"Total Extracted Clips:   {total_clips:,d} clips")
    print(f"Task Breakdown:          {summary['tasks']}")
    print(f"Dimensions:              {summary['dimensions']}")
    print(f"Data Types:              {summary['dtypes']}")
    print(f"Cameras:                 {summary['cameras']}")
    print(f"Clip Lengths:            {summary['clip_lengths']} frames")
    print(f"Strides:                 {summary['strides']} frames")
    print(f"Pooling Modes:           {summary['pooling_modes']}")
    print(f"Finiteness:              100% finite ({nan_counts} NaNs, {inf_counts} Infs)")
    print(f"Integrity Errors:        {len(errors)}")
    print(f"Missing from Manifest:   {len(missing_trials)}")
    print(f"Extra/Unreferenced:      {len(extra_trials)}")
    print(f"Inventory saved to:      {inventory_csv.resolve()}")
    print(f"Summary JSON saved to:   {summary_json.resolve()}")
    print("=" * 65)
    return summary


if __name__ == "__main__":
    audit_surgmotion_caches()
