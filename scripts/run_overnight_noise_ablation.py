"""scripts/run_overnight_noise_ablation.py

Single-Command Overnight Orchestrator for ITER-31 Noise Ablation on Mentor GPU.

Combines preflight check, primary continuous-noise ablation, control baseline,
trajectory plot generation, and automated Git commit/push into one seamless command.

Workflow:
1. Prevents Windows sleep / hibernation (SetThreadExecutionState).
2. Runs preflight assertion (GPU, CUDA, VRAM, dataset, SurgMotion caches).
3. Executes Primary Ablation: Continuous-Noise Trained Adaptive Fusion (Suturing, Knot, Needle).
4. Executes Control Baseline: Existing Discrete Binary Dropout (Suturing, Knot, Needle).
5. Generates publication-ready gate response and temporal dynamics figures.
6. Automatically commits and pushes all generated artifacts and figures to GitHub.
7. Restores Windows sleep state and prints next-morning verification handoff.

Usage:
    # 1. Single Command Overnight Run (GPU):
    python scripts/run_overnight_noise_ablation.py

    # 2. Fast 1-Fold Smoke Test (verifies whole orchestrator in ~1 min):
    python scripts/run_overnight_noise_ablation.py --smoke-test

    # 3. Preflight Check Only:
    python scripts/run_overnight_noise_ablation.py --preflight-only
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
logger = logging.getLogger("overnight_runner")


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
            logger.info("[System] Windows sleep/hibernation prevention enabled for overnight run.")
        else:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            logger.info("[System] Windows sleep/hibernation state restored to default.")
    except Exception as e:
        logger.warning(f"[System] Could not configure Windows sleep state: {e}")


def run_command(cmd: List[str], description: str) -> bool:
    """Execute shell command with real-time output streaming and error trapping."""
    logger.info("=" * 70)
    logger.info(f"STARTING: {description}")
    logger.info(f"COMMAND: {' '.join(cmd)}")
    logger.info("=" * 70)

    start_time = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        ret = proc.wait()
        elapsed = time.time() - start_time
        if ret == 0:
            logger.info(f"SUCCESS: {description} (Elapsed: {elapsed / 60:.2f} min)")
            return True
        else:
            logger.error(f"FAILED (Exit Code {ret}): {description} (Elapsed: {elapsed / 60:.2f} min)")
            return False
    except Exception as e:
        logger.error(f"EXCEPTION running {description}: {e}")
        return False


def run_git_sync(commit_message: str) -> bool:
    """Stage artifacts and push to origin main."""
    logger.info("=" * 70)
    logger.info("GIT SYNC: Staging artifacts and pushing to origin main...")
    logger.info("=" * 70)

    try:
        # 1. Add artifacts and figures
        subprocess.run(["git", "add", "artifacts/iter31_noise_ablation/"], cwd=str(PROJECT_ROOT), check=True)
        subprocess.run(["git", "add", "paper/figures/"], cwd=str(PROJECT_ROOT), check=True)

        # 2. Check if anything to commit
        status_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        if not status_proc.stdout.strip():
            logger.info("Git: No changes to commit.")
            return True

        # 3. Commit
        subprocess.run(["git", "commit", "-m", commit_message], cwd=str(PROJECT_ROOT), check=True)
        logger.info("Git: Commit successful.")

        # 4. Push
        push_proc = subprocess.run(["git", "push", "origin", "main"], cwd=str(PROJECT_ROOT), check=True)
        logger.info("Git: Successfully pushed to origin main!")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Git sync failed: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-Command Overnight Noise Ablation Runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-fold 1-epoch smoke test to verify pipeline")
    parser.add_argument("--preflight-only", action="store_true", help="Run preflight environment check and exit")
    parser.add_argument("--no-push", action="store_true", help="Skip automatic git commit and push")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device ('cuda' or 'cpu')")
    parser.add_argument("--epochs", type=int, default=25, help="Epochs per condition (default 25)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default 32)")
    args = parser.parse_args()

    python_bin = sys.executable
    ablation_script = str(PROJECT_ROOT / "scripts" / "train_surgmotion_noise_ablation.py")
    plot_script = str(PROJECT_ROOT / "scripts" / "plot_gate_dynamics.py")

    # 1. Preflight assertion
    logger.info("PHASE 1: Running Preflight Environment Assertion...")
    preflight_cmd = [python_bin, ablation_script, "--preflight-only"]
    if not run_command(preflight_cmd, "Preflight Environment Check"):
        logger.error("Preflight check FAILED. Aborting overnight run.")
        sys.exit(1)

    if args.preflight_only:
        logger.info("Preflight check successful. Exiting as requested.")
        sys.exit(0)

    # 2. Enable sleep prevention
    prevent_windows_sleep(True)
    overall_start = time.time()
    artifacts_dir = PROJECT_ROOT / "artifacts" / "iter31_noise_ablation"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 3. Execute Primary Ablation (continuous_noise)
        logger.info("\nPHASE 2: Executing Primary Ablation: Continuous-Noise Trained Adaptive Fusion...")
        cond1_cmd = [
            python_bin, ablation_script,
            "--task", "all",
            "--condition", "continuous_noise",
            "--noise-min", "0.0",
            "--noise-max", "1.5",
            "--noise-prob", "0.5",
            "--sequence-length", "8",
            "--epochs", str(1 if args.smoke_test else args.epochs),
            "--batch-size", str(8 if args.smoke_test else args.batch_size),
            "--device", "cpu" if args.smoke_test else args.device,
        ]
        if args.smoke_test:
            cond1_cmd.append("--smoke-test")

        success_cond1 = run_command(cond1_cmd, "Primary Ablation (continuous_noise)")
        if not success_cond1:
            logger.error("Primary ablation failed! Proceeding with control baseline anyway...")

        # 4. Execute Control Baseline (existing)
        logger.info("\nPHASE 3: Executing Control Baseline: Discrete Binary Dropout Only...")
        cond2_cmd = [
            python_bin, ablation_script,
            "--task", "all",
            "--condition", "existing",
            "--sequence-length", "8",
            "--epochs", str(1 if args.smoke_test else args.epochs),
            "--batch-size", str(8 if args.smoke_test else args.batch_size),
            "--device", "cpu" if args.smoke_test else args.device,
        ]
        if args.smoke_test:
            cond2_cmd.append("--smoke-test")

        success_cond2 = run_command(cond2_cmd, "Control Baseline (existing)")
        if not success_cond2:
            logger.error("Control baseline failed!")

        # 5. Generate Figures
        logger.info("\nPHASE 4: Generating Gate Response and Temporal Dynamics Figures...")
        traj_suturing = artifacts_dir / "gate_trajectories_continuous_noise_Suturing.json"
        plot_cmd = [python_bin, plot_script]
        if traj_suturing.is_file():
            plot_cmd.extend(["--trajectory-file", str(traj_suturing)])
        run_command(plot_cmd, "Publication Figures Generation")

        # 6. Git Commit & Push
        if not args.no_push and not args.smoke_test:
            logger.info("\nPHASE 5: Committing and Pushing Results to GitHub...")
            msg = "ITER-31: Completed mentor GPU noise ablation and gate trajectory run"
            run_git_sync(msg)
        elif args.smoke_test:
            logger.info("Smoke test completed. Skipping Git push.")

        total_elapsed = time.time() - overall_start
        logger.info("=" * 70)
        logger.info(f"ALL OVERNIGHT PHASES COMPLETE! Total Runtime: {total_elapsed / 3600:.2f} hours")
        logger.info(f"Artifacts saved to: {artifacts_dir.resolve()}")
        logger.info("=" * 70)

    finally:
        prevent_windows_sleep(False)


if __name__ == "__main__":
    main()
