"""scripts/run_iter35_imbalance_experiment.py

Single-Command Orchestrator for ITER-35 Class-Imbalance Experiment on Mentor GPU.

Combines:
1. Windows sleep/hibernation prevention during training.
2. CUDA & feature cache preflight verification.
3. Sequential execution of all 4 Phase 1 loss formulations on Suturing:
   - Run 1.1: Standard Cross-Entropy ('ce') — unweighted control
   - Run 1.2: Class-Weighted Cross-Entropy ('class_weighted') — inverse frequency
   - Run 1.3: Multi-Class Focal Loss ('focal', gamma=2.0) — primary candidate
   - Run 1.4: Class-Balanced Focal Loss ('cb_focal', gamma=2.0, beta=0.9999) — effective number
4. Cross-condition comparative summary & artifact generation.
5. Automated Git add, commit, and push to GitHub origin main.

Usage:
    # 1. Full Production GPU Run (All 4 conditions, 8 LOUO folds, ~25-30 min on RTX 5000):
    python scripts/run_iter35_imbalance_experiment.py

    # 2. Fast Smoke Test (1 fold, 1 epoch to verify end-to-end execution in ~1 min):
    python scripts/run_iter35_imbalance_experiment.py --smoke-test

    # 3. Dry-run without git push:
    python scripts/run_iter35_imbalance_experiment.py --no-push
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("imbalance_runner")


def prevent_windows_sleep(enable: bool = True) -> None:
    """Prevent or restore Windows sleep and display idle during GPU execution."""
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
            logger.info("[System] Windows sleep/hibernation prevention enabled.")
        else:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            logger.info("[System] Windows sleep/hibernation state restored to default.")
    except Exception as e:
        logger.warning(f"[System] Could not adjust Windows execution state: {e}")


def run_preflight_check(device_name: str, task: str = "Suturing") -> bool:
    """Verify hardware, CUDA availability, caches, and repository state."""
    logger.info("=" * 70)
    logger.info("RUNNING PREFLIGHT VERIFICATION FOR MENTOR GPU EXPERIMENT")
    logger.info("=" * 70)

    import torch
    cuda_avail = torch.cuda.is_available()
    logger.info(f"Python: {sys.version.split()[0]} | PyTorch: {torch.__version__}")
    logger.info(f"Target Device: {device_name} | CUDA Available: {cuda_avail}")

    if device_name == "cuda":
        if not cuda_avail:
            logger.error("CUDA requested but torch.cuda.is_available() is False. Aborting.")
            return False
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"VRAM: {vram_gb:.2f} GB")

    # Verify feature caches
    cache_dir = PROJECT_ROOT / "data/cache/features_surgmotion"
    if not cache_dir.is_dir():
        logger.error(f"SurgMotion cache directory missing at: {cache_dir}")
        return False

    cached_files = list(cache_dir.glob(f"{task}_*.npz"))
    logger.info(f"Found {len(cached_files)} cached {task} SurgMotion feature files.")
    if len(cached_files) < 30:
        logger.error(f"Insufficient cached feature files for {task} ({len(cached_files)} < 30). Aborting.")
        return False

    logger.info("[Preflight] All environment assertions passed.")
    return True


def execute_subcommand(cmd: List[str], desc: str) -> bool:
    """Execute shell command with streaming output and error logging."""
    logger.info("-" * 70)
    logger.info(f"STARTING: {desc}")
    logger.info(f"COMMAND: {' '.join(cmd)}")
    logger.info("-" * 70)

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in proc.stdout:
        print(line, end="", flush=True)

    proc.wait()
    elapsed = time.time() - t0

    if proc.returncode != 0:
        logger.error(f"Command failed with exit code {proc.returncode} after {elapsed:.1f}s: {desc}")
        return False

    logger.info(f"SUCCESS: {desc} (elapsed: {elapsed:.1f}s)")
    return True


def git_commit_and_push(commit_message: str) -> bool:
    """Stage generated artifacts, commit, and push to GitHub origin main."""
    logger.info("=" * 70)
    logger.info("AUTOMATIC GIT COMMIT & PUSH")
    logger.info("=" * 70)

    try:
        # Check current branch
        branch_proc = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        current_branch = branch_proc.stdout.strip()
        logger.info(f"Current Git Branch: {current_branch}")

        # Stage artifacts and checkpoints
        subprocess.run(
            ["git", "add", "artifacts/", "checkpoints/", "docs/", "memory.md"],
            cwd=str(PROJECT_ROOT),
            check=True,
        )

        # Check if there are changes to commit
        status_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )

        if not status_proc.stdout.strip():
            logger.info("No new changes detected to commit.")
            return True

        # Commit
        subprocess.run(
            ["git", "commit", "-m", commit_message],
            cwd=str(PROJECT_ROOT),
            check=True,
        )
        logger.info(f"Committed changes: '{commit_message}'")

        # Push to origin
        logger.info(f"Pushing to origin {current_branch}...")
        push_proc = subprocess.run(
            ["git", "push", "origin", current_branch],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"Push successful:\n{push_proc.stdout.strip()}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Git command failed: {e}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False


def build_comparative_summary(task: str = "Suturing") -> Dict[str, Any]:
    """Collate and compare results from all evaluated loss formulations."""
    loss_types = ["ce", "class_weighted", "focal", "cb_focal"]
    summary: Dict[str, Any] = {
        "task": task,
        "timestamp": datetime.datetime.now().isoformat(),
        "conditions": {},
    }

    for lt in loss_types:
        suffix = f"_{lt}" if lt != "class_weighted" else ""
        art_path = PROJECT_ROOT / f"artifacts/surgmotion_fusion_adaptive_{task}{suffix}_results.json"
        if art_path.is_file():
            with open(art_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                summary["conditions"][lt] = {
                    "accuracy_mean": data["metrics"]["accuracy"]["mean"],
                    "accuracy_std": data["metrics"]["accuracy"]["std"],
                    "balanced_accuracy_mean": data["metrics"]["balanced_accuracy"]["mean"],
                    "balanced_accuracy_std": data["metrics"]["balanced_accuracy"]["std"],
                    "macro_f1_mean": data["metrics"]["macro_f1"]["mean"],
                    "macro_f1_std": data["metrics"]["macro_f1"]["std"],
                    "edit_score_mean": data["metrics"]["edit_score"]["mean"],
                    "edit_score_std": data["metrics"]["edit_score"]["std"],
                }

    out_file = PROJECT_ROOT / f"artifacts/iter35_imbalance_comparison_{task}_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved comparative summary to: {out_file}")

    # Print summary comparison table
    print("\n" + "=" * 85)
    print(f"ITER-35 CLASS-IMBALANCE COMPARISON SUMMARY: {task} (8 LOUO FOLDS)")
    print("=" * 85)
    print(f"| Loss Formulation | Accuracy (%) | Balanced Acc (%) | Macro-F1 (%) | Edit Score (%) |")
    print(f"|:---|:---:|:---:|:---:|:---:|")
    for lt, m in summary["conditions"].items():
        print(
            f"| {lt:20} | {m['accuracy_mean']:5.2f} ± {m['accuracy_std']:4.2f} "
            f"| {m['balanced_accuracy_mean']:5.2f} ± {m['balanced_accuracy_std']:4.2f} "
            f"| {m['macro_f1_mean']:5.2f} ± {m['macro_f1_std']:4.2f} "
            f"| {m['edit_score_mean']:5.2f} ± {m['edit_score_std']:4.2f} |"
        )
    print("=" * 85 + "\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-Command All-in-One Runner for ITER-35 Class-Imbalance Experiment.")
    parser.add_argument("--task", type=str, default="Suturing", help="Target task (default: Suturing)")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs per fold (default: 25)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu'). Autodetects GPU if None.")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 fold and 1 epoch for rapid pipeline verification")
    parser.add_argument("--no-push", action="store_true", help="Skip automated Git push at the end")
    parser.add_argument("--preflight-only", action="store_true", help="Run environment preflight check only")
    args = parser.parse_args()

    import torch
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Preflight assertion
    if not run_preflight_check(device_name=device, task=args.task):
        sys.exit(1)

    if args.preflight_only:
        logger.info("Preflight check passed. Exiting (--preflight-only specified).")
        sys.exit(0)

    # 2. Prevent Windows sleep during execution
    prevent_windows_sleep(True)

    epochs = 1 if args.smoke_test else args.epochs
    t_start = time.time()

    # Define the 4 experimental runs
    runs = [
        {
            "name": "Run 1.1: Standard Cross-Entropy Baseline (ce)",
            "args": ["--loss-type", "ce"],
        },
        {
            "name": "Run 1.2: Class-Weighted Cross-Entropy (class_weighted)",
            "args": ["--loss-type", "class_weighted"],
        },
        {
            "name": "Run 1.3: Multi-Class Focal Loss (gamma=2.0) [Primary Candidate]",
            "args": ["--loss-type", "focal", "--gamma", "2.0"],
        },
        {
            "name": "Run 1.4: Class-Balanced Focal Loss (gamma=2.0, beta=0.9999)",
            "args": ["--loss-type", "cb_focal", "--gamma", "2.0", "--beta", "0.9999"],
        },
    ]

    failed_runs = []

    try:
        python_exe = sys.executable
        extra_args = ["--fold", "1", "--no-robustness", "--no-save-artifacts"] if args.smoke_test else []
        for r in runs:
            cmd = [
                python_exe,
                "scripts/train_surgmotion_multimodal_fusion.py",
                "--task", args.task,
                "--fusion-type", "adaptive",
                "--epochs", str(epochs),
                "--batch-size", str(args.batch_size),
                "--device", device,
            ] + r["args"] + extra_args

            success = execute_subcommand(cmd, r["name"])
            if not success:
                failed_runs.append(r["name"])
                logger.error(f"Aborting sequence due to failure in {r['name']}.")
                break

        # 3. Post-execution synthesis
        if not failed_runs and not args.smoke_test:
            build_comparative_summary(task=args.task)

        total_elapsed = time.time() - t_start
        logger.info(f"All scheduled runs completed in {total_elapsed/60:.1f} minutes.")

        # 4. Automated Git Commit & Push
        if not failed_runs and not args.no_push and not args.smoke_test:
            commit_msg = (
                f"feat(iter35): complete Phase 1 class-imbalance ablation on {args.task} "
                f"(ce, class_weighted, focal, cb_focal)"
            )
            git_commit_and_push(commit_msg)

    finally:
        prevent_windows_sleep(False)

    if failed_runs:
        logger.error(f"Experiment finished with failures in: {failed_runs}")
        sys.exit(1)
    else:
        logger.info("ITER-35 Phase 1 experiment workflow finished successfully.")


if __name__ == "__main__":
    main()
