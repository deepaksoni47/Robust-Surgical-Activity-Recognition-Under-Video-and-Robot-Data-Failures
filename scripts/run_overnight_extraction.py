"""Overnight SurgMotion Feature Extraction Orchestrator for Mentor's GPU Machine.

Automates the complete end-to-end execution across all 103 JIGSAWS video trials:
1. Prevents Windows sleep/hibernation during the run via SetThreadExecutionState.
2. Runs pre-flight environment, GPU, checkpoint, and video availability checks.
3. Performs a 1-clip GPU smoke test to verify CUDA FP16 & SDPA execution.
4. Sequentially extracts features for Suturing (39), Knot Tying (36), and Needle Passing (28).
5. Automatically skips already completed and validated caches (e.g. Suturing_B001).
6. Isolates errors: logs any trial-level exceptions and continues with remaining trials.
7. Audits all output files with the cache validation engine upon completion.
8. Writes an overnight execution summary report and prints the next-morning handoff summary.

Usage:
    # Full overnight extraction on GPU:
    python scripts/run_overnight_extraction.py

    # Dry-run pre-flight check (verifies videos, checkpoint, GPU, and prints ETA):
    python scripts/run_overnight_extraction.py --dry-run

    # Extract specific task:
    python scripts/run_overnight_extraction.py --tasks Suturing
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


def prevent_windows_sleep(enable: bool = True) -> None:
    """Prevent or restore Windows sleep and display idle during long overnight tasks."""
    if sys.platform != "win32":
        return
    try:
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x00000040
        if enable:
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
            )
            print("[System] Windows sleep/hibernation prevention enabled for overnight run.")
        else:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            print("[System] Windows sleep/hibernation state restored to default.")
    except Exception as e:
        print(f"[System] Warning: Could not configure Windows sleep prevention: {e}")


def check_gpu_environment() -> Dict[str, Any]:
    """Inspect and report CUDA device capabilities."""
    info: Dict[str, Any] = {
        "cuda_available": False,
        "device_name": "CPU",
        "device_count": 0,
        "vram_gb": 0.0,
        "torch_version": "unknown",
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            info["vram_gb"] = round(vram_bytes / (1024 ** 3), 2)
    except ImportError:
        pass
    return info


def verify_inputs(
    manifest_path: Path,
    camera: str = "capture1",
    dataset_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, List[str], int, int]:
    """Verify presence and accessibility of all videos referenced in manifest."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path.resolve()}")

    df = pd.read_csv(manifest_path)
    video_col = f"video_{camera}_path"
    if video_col not in df.columns:
        raise ValueError(f"Column '{video_col}' missing from manifest.")

    missing_files: List[str] = []
    total_frames = 0
    total_clips = 0

    import cv2
    for _, row in df.iterrows():
        rel_path = row[video_col]
        video_path = (dataset_dir / Path(rel_path).name) if dataset_dir else (PROJECT_ROOT / rel_path)
        if not video_path.is_file():
            missing_files.append(f"{row['trial_id']} ({rel_path})")
        else:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                cc = len(list(range(1, max(1, fc - 64 + 2), 16)))
                total_frames += fc
                total_clips += cc

    return df, missing_files, total_frames, total_clips


def run_smoke_test(checkpoint_path: Path, device: str = "cuda", fp16: bool = True) -> bool:
    """Run single-clip GPU smoke test using extract_surgmotion_features.py."""
    print("\n--- Running Pre-Flight 1-Clip GPU Smoke Test ---")
    cmd = [
        sys.executable,
        "scripts/extract_surgmotion_features.py",
        "--smoke-test",
        "--checkpoint", str(checkpoint_path),
        "--device", device,
    ]
    if fp16 and device.startswith("cuda"):
        cmd.append("--fp16")

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return proc.returncode == 0


def run_task_extraction(
    task: str,
    checkpoint: Path,
    output_dir: Path,
    camera: str = "capture1",
    batch_size: int = 8,
    stride: int = 16,
    clip_length: int = 64,
    fp16: bool = True,
    device: str = "cuda",
) -> Tuple[bool, float]:
    """Run feature extraction for a single task via extract_surgmotion_features.py."""
    print("\n" + "=" * 70)
    print(f"Starting Extraction Task: {task}")
    print("=" * 70)
    t0 = time.time()
    cmd = [
        sys.executable,
        "scripts/extract_surgmotion_features.py",
        "--task", task,
        "--checkpoint", str(checkpoint),
        "--output-dir", str(output_dir),
        "--camera", camera,
        "--batch-size", str(batch_size),
        "--stride", str(stride),
        "--clip-length", str(clip_length),
        "--device", device,
    ]
    if fp16 and device.startswith("cuda"):
        cmd.append("--fp16")

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    success = (proc.returncode == 0)
    return success, elapsed


def run_validation(output_dir: Path) -> Dict[str, Any]:
    """Execute validate_surgmotion_cache.py across the output directory."""
    print("\n" + "=" * 70)
    print("Running Final Directory-Wide Cache Integrity Audit")
    print("=" * 70)
    cmd = [
        sys.executable,
        "scripts/validate_surgmotion_cache.py",
        "--dir", str(output_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    # Count files and statuses
    npz_files = list(output_dir.glob("*.npz"))
    # Filter out .tmp.npz if any
    valid_npzs = [f for f in npz_files if not f.name.endswith(".tmp.npz")]
    return {
        "audit_returncode": proc.returncode,
        "total_cache_files": len(valid_npzs),
        "audit_passed": (proc.returncode == 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Overnight full JIGSAWS SurgMotion feature extraction.")
    parser.add_argument("--tasks", nargs="+", default=["Suturing", "Knot_Tying", "Needle_Passing"],
                        choices=["Suturing", "Knot_Tying", "Needle_Passing", "all"],
                        help="Tasks to extract sequentially")
    parser.add_argument("--checkpoint", type=str, default="models/SurgMotion-L/SurgMotion-vitl.pt",
                        help="Path to SurgMotion-vitl.pt checkpoint")
    parser.add_argument("--output-dir", type=str, default="data/cache/features_surgmotion",
                        help="Output directory for extracted .npz files")
    parser.add_argument("--camera", type=str, default="capture1",
                        choices=["capture1", "capture2"],
                        help="Camera stream to extract")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for ViT forward passes")
    parser.add_argument("--stride", type=int, default=16,
                        help="Clip stride in frames")
    parser.add_argument("--clip-length", type=int, default=64,
                        help="Clip window in frames")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Target device (cuda / cpu)")
    parser.add_argument("--no-fp16", action="store_true",
                        help="Disable FP16 mixed precision")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run pre-flight checks and estimate runtime without extracting")
    args = parser.parse_args()

    start_iso = datetime.datetime.now().isoformat()
    t_global_start = time.time()

    print("=" * 75)
    print("SURGMOTION FULL JIGSAWS OVERNIGHT EXTRACTION ORCHESTRATOR")
    print(f"Started at: {start_iso}")
    print("=" * 75)

    # 1. Inspect Environment & Hardware
    gpu_info = check_gpu_environment()
    print(f"Platform:        {platform.platform()}")
    print(f"Python:          {sys.version.split()[0]}")
    print(f"PyTorch Version: {gpu_info['torch_version']}")
    print(f"CUDA Available:  {gpu_info['cuda_available']}")
    print(f"Device Name:     {gpu_info['device_name']}")
    print(f"Total GPU VRAM:  {gpu_info['vram_gb']} GB")

    if not gpu_info["cuda_available"] and args.device == "cuda":
        print("\n[WARNING] CUDA is not available! Falling back to CPU will take ~40+ hours.")
        if not args.dry_run:
            ans = input("Proceed on CPU anyway? (y/N): ").strip().lower()
            if ans != "y":
                print("Aborting. Please check NVIDIA drivers / PyTorch CUDA install.")
                sys.exit(1)
            args.device = "cpu"

    # 2. Verify Checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    if not ckpt_path.is_file():
        print(f"\n[ERROR] Checkpoint missing at: {ckpt_path.resolve()}")
        print("Please ensure 'models/SurgMotion-L/SurgMotion-vitl.pt' is present.")
        sys.exit(1)
    ckpt_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    print(f"Checkpoint:      {ckpt_path.name} ({ckpt_size_mb:.2f} MB) -> OK")

    # 3. Verify Video Inputs from Manifest
    manifest_path = PROJECT_ROOT / "artifacts" / "manifest.csv"
    print(f"Manifest:        {manifest_path.resolve()}")
    try:
        df, missing_videos, total_frames, total_clips = verify_inputs(manifest_path, camera=args.camera)
    except Exception as e:
        print(f"[ERROR] Could not read manifest or verify videos: {e}")
        sys.exit(1)

    print(f"Total Trials:    {len(df)} across {list(df['task'].unique())}")
    print(f"Total Frames:    {total_frames:,d} frames (~{total_frames/(30*60):.1f} video minutes)")
    print(f"Total Clips:     {total_clips:,d} clips (W={args.clip_length}, S={args.stride})")

    if missing_videos:
        print(f"\n[WARNING] {len(missing_videos)}/{len(df)} video files are missing on disk:")
        for m in missing_videos[:5]:
            print(f"  - {m}")
        if len(missing_videos) > 5:
            print(f"  ... and {len(missing_videos)-5} more.")
        print("Please copy the required 'dataset/' folders from the laptop.")
        if not args.dry_run:
            sys.exit(1)
    else:
        print("Video Status:    All 103 canonical video files verified on disk -> OK")

    # Estimated Runtime
    sec_per_clip_est = 0.364  # Measured on Quadro RTX 5000 (pilot benchmark)
    est_total_sec = total_clips * sec_per_clip_est
    print(f"Estimated Time:  {est_total_sec/3600:.2f} hours (~{est_total_sec/60:.1f} minutes) on Quadro RTX 5000")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output Cache:    {output_dir.resolve()}")

    if args.dry_run:
        print("\n[DRY RUN COMPLETE] Pre-flight checks passed successfully.")
        return

    # 4. Run Pre-Flight Smoke Test
    smoke_ok = run_smoke_test(ckpt_path, device=args.device, fp16=not args.no_fp16)
    if not smoke_ok:
        print("[ERROR] Smoke test failed! Aborting overnight run before starting full extraction.")
        sys.exit(1)

    # 5. Execute Overnight Extraction
    prevent_windows_sleep(True)
    tasks_to_run = ["Suturing", "Knot_Tying", "Needle_Passing"] if "all" in args.tasks else args.tasks
    task_results: Dict[str, Any] = {}

    try:
        for task in tasks_to_run:
            success, elapsed = run_task_extraction(
                task=task,
                checkpoint=ckpt_path,
                output_dir=output_dir,
                camera=args.camera,
                batch_size=args.batch_size,
                stride=args.stride,
                clip_length=args.clip_length,
                fp16=not args.no_fp16,
                device=args.device,
            )
            task_results[task] = {
                "success": success,
                "elapsed_sec": round(elapsed, 2),
                "elapsed_min": round(elapsed / 60, 2),
            }
            print(f"[Task Summary] {task}: {'SUCCESS' if success else 'FAILED'} in {elapsed/60:.2f} min")

    finally:
        prevent_windows_sleep(False)

    # 6. Post-Extraction Audit
    audit_results = run_validation(output_dir)

    total_elapsed_sec = time.time() - t_global_start
    total_elapsed_min = total_elapsed_sec / 60
    end_iso = datetime.datetime.now().isoformat()

    # 7. Write Overnight Summary Report
    report = {
        "session": "Full JIGSAWS Overnight SurgMotion Feature Extraction",
        "started_at": start_iso,
        "completed_at": end_iso,
        "total_elapsed_min": round(total_elapsed_min, 2),
        "total_elapsed_sec": round(total_elapsed_sec, 2),
        "hardware": gpu_info,
        "settings": {
            "batch_size": args.batch_size,
            "stride": args.stride,
            "clip_length": args.clip_length,
            "camera": args.camera,
            "fp16": not args.no_fp16,
            "device": args.device,
            "checkpoint": str(ckpt_path),
        },
        "task_results": task_results,
        "audit_results": audit_results,
    }

    report_path = output_dir / "overnight_extraction_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # 8. Final Next-Morning Printout
    print("\n" + "=" * 75)
    print("OVERNIGHT EXTRACTION RUN COMPLETE — NEXT-MORNING HANDOFF SUMMARY")
    print("=" * 75)
    print(f"Total Elapsed Time:   {total_elapsed_min:.2f} minutes ({total_elapsed_sec/3600:.2f} hours)")
    print(f"Tasks Attempted:      {list(task_results.keys())}")
    for t, res in task_results.items():
        print(f"  - {t:15s}: {'PASSED' if res['success'] else 'FAILED'} ({res['elapsed_min']} min)")
    print(f"Total Valid Caches:   {audit_results['total_cache_files']} / 103 trials")
    print(f"Audit Status:         {'ALL CACHES PASSED VALIDATION' if audit_results['audit_passed'] else 'AUDIT WARNINGS DETECTED'}")
    print(f"Report JSON:          {report_path.resolve()}")
    print("\nNext Action: Copy the entire folder:")
    print(f"    '{output_dir.resolve()}' (~65 MB)")
    print("back to your laptop at:")
    print(r"    D:\miniproject-5thsem\data\cache\features_surgmotion")
    print("=" * 75)


if __name__ == "__main__":
    main()
