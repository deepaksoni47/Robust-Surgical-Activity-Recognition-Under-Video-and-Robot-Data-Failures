"""Script to extract and cache frozen visual features for JIGSAWS trials.

Reads trial metadata from the manifest, streams video frames in mini-batches,
extracts representations using the frozen visual encoder (ResNet-18 or SurgMotion),
and atomically saves compressed .npz archives with comprehensive provenance metadata.

Usage:
    # Extract features for 1 Suturing trial (smoke test)
    python scripts/extract_visual_features.py --task Suturing --limit-trials 1

    # Extract all Suturing trials
    python scripts/extract_visual_features.py --task Suturing

    # Audit and verify existing caches
    python scripts/extract_visual_features.py --verify-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from src.data.feature_cache import (
    extract_and_cache_trial,
    get_cache_path,
    validate_feature_cache,
)
from src.data.manifest import build_manifest
from src.models.visual_encoder import build_visual_encoder
from src.utils.reproducibility import load_config, save_json, set_seed


def run_feature_extraction(
    config_path: str = "configs/encoder.yaml",
    task: str = "Suturing",
    trial_id: Optional[str] = None,
    limit_trials: Optional[int] = None,
    limit_frames: Optional[int] = None,
    batch_size: Optional[int] = None,
    camera: str = "capture1",
    device: Optional[str] = None,
    force: bool = False,
    verify_only: bool = False,
    output_audit: str = "artifacts/feature_cache_audit.json",
) -> Dict[str, Any]:
    """Execute feature extraction across selected trials and write an audit report."""
    set_seed(42)
    cfg = load_config(config_path)

    cache_dir = Path(cfg.get("extraction", {}).get("cache_dir", "data/cache/features"))
    bs = batch_size or cfg.get("extraction", {}).get("batch_size", 32)
    target_device = device or cfg.get("extraction", {}).get("device", "cpu")

    # 1. Build encoder
    print(f"Loading visual encoder from config: {config_path} ...")
    encoder = build_visual_encoder(cfg, device=target_device)
    print(f"Visual encoder active: {encoder.encoder_name} (Output Dim: {encoder.output_dim}, Device: {encoder.device})")
    assert encoder.is_frozen(), "Encoder parameters must be frozen!"

    # 2. Load manifest
    manifest_csv = Path("artifacts/manifest.csv")
    if manifest_csv.is_file():
        manifest_df = pd.read_csv(manifest_csv)
    else:
        manifest_df = build_manifest()

    # Filter trials
    if task != "all":
        manifest_df = manifest_df[manifest_df["task"] == task].reset_index(drop=True)

    if trial_id:
        manifest_df = manifest_df[manifest_df["trial_id"] == trial_id].reset_index(drop=True)
        if len(manifest_df) == 0:
            raise ValueError(f"Trial '{trial_id}' not found in manifest.")

    if limit_trials is not None and limit_trials > 0:
        manifest_df = manifest_df.head(limit_trials).reset_index(drop=True)

    print(f"\nProcessing {len(manifest_df)} trials for task '{task}' (Camera: {camera}, Batch Size: {bs}) ...")

    # 3. Execution / Verification Loop
    results: List[Dict[str, Any]] = []
    total_frames_extracted = 0
    t0_all = time.perf_counter()

    for idx, row in manifest_df.iterrows():
        t_id = row["trial_id"]
        target_path = get_cache_path(cache_dir, t_id, encoder.encoder_name, camera=camera)

        if verify_only:
            try:
                val = validate_feature_cache(
                    target_path, expected_dim=encoder.output_dim, expected_trial_id=t_id
                )
                print(f"[{idx+1}/{len(manifest_df)}] VALID: {t_id} ({val['num_frames']} frames, {val['file_size_bytes']/1024/1024:.2f} MB)")
                results.append({"trial_id": t_id, "status": "VALID", **val})
            except Exception as e:
                print(f"[{idx+1}/{len(manifest_df)}] MISSING/INVALID: {t_id} ({e})")
                results.append({"trial_id": t_id, "status": "INVALID", "error": str(e)})
            continue

        print(f"\n[{idx+1}/{len(manifest_df)}] Processing {t_id} ...")
        t_start = time.perf_counter()
        res = extract_and_cache_trial(
            trial_row=row,
            encoder=encoder,
            cache_dir=cache_dir,
            camera=camera,
            batch_size=bs,
            max_frames=limit_frames,
            force=force,
        )
        elapsed = time.perf_counter() - t_start

        if res["status"] == "EXTRACTED":
            fps = res["num_frames"] / elapsed if elapsed > 0 else 0.0
            size_mb = res["file_size_bytes"] / (1024 * 1024)
            print(f"  -> Extracted {res['num_frames']} frames in {elapsed:.1f}s ({fps:.1f} FPS, {size_mb:.2f} MB) [Saved: {res['cache_path']}]")
            total_frames_extracted += res["num_frames"]
        else:
            print(f"  -> Skipped: {res.get('reason')} ({res['num_frames']} frames present)")

        results.append(res)

    total_time = time.perf_counter() - t0_all

    audit_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task": task,
        "encoder_name": encoder.encoder_name,
        "output_dim": encoder.output_dim,
        "trials_processed": len(results),
        "extracted_count": sum(1 for r in results if r.get("status") == "EXTRACTED"),
        "skipped_count": sum(1 for r in results if r.get("status") == "SKIPPED"),
        "valid_count": sum(1 for r in results if r.get("status") in ("VALID", "EXTRACTED", "SKIPPED")),
        "total_frames_extracted": total_frames_extracted,
        "total_elapsed_sec": round(total_time, 2),
        "results": results,
    }

    # Save audit report
    audit_file = Path(output_audit)
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    save_json(audit_summary, audit_file)
    print(f"\nAudit summary saved to: {audit_file}")

    return audit_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and cache visual features for JIGSAWS trials.")
    parser.add_argument("--config", type=str, default="configs/encoder.yaml", help="Path to encoder.yaml")
    parser.add_argument("--task", type=str, default="Suturing", help="Task name ('Suturing', 'Knot_Tying', 'Needle_Passing', 'all')")
    parser.add_argument("--trial", type=str, default=None, help="Specific trial_id (e.g., Suturing_B001)")
    parser.add_argument("--limit-trials", type=int, default=None, help="Limit number of trials to extract")
    parser.add_argument("--limit-frames", type=int, default=None, help="Limit frames per trial")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for frame extraction")
    parser.add_argument("--camera", type=str, default="capture1", help="Camera name (capture1 or capture2)")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    parser.add_argument("--force", action="store_true", help="Force re-extraction of existing caches")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing caches without extracting")
    parser.add_argument("--output-audit", type=str, default="artifacts/feature_cache_audit.json", help="Path to audit JSON")
    args = parser.parse_args()

    summary = run_feature_extraction(
        config_path=args.config,
        task=args.task,
        trial_id=args.trial,
        limit_trials=args.limit_trials,
        limit_frames=args.limit_frames,
        batch_size=args.batch_size,
        camera=args.camera,
        device=args.device,
        force=args.force,
        verify_only=args.verify_only,
        output_audit=args.output_audit,
    )
    print(f"\nExtraction complete: {summary['extracted_count']} extracted, {summary['skipped_count']} skipped out of {summary['trials_processed']} trials.")


if __name__ == "__main__":
    main()
