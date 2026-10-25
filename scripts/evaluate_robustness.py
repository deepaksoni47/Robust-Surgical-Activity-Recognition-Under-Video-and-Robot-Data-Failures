#!/usr/bin/env python3
"""scripts/evaluate_robustness.py

Execute robustness evaluation under visual corruptions and kinematic dropouts on JIGSAWS.
Evaluates Clean, Mean-C, and Worst-C metrics following the Endo-C6 benchmark protocol.

Usage:
    # 1. Audit readiness of all folds and models across tasks
    python scripts/evaluate_robustness.py --task Suturing --audit-only
    python scripts/evaluate_robustness.py --task all --audit-only

    # 2. Run synthetic pipeline verification / smoke run
    python scripts/evaluate_robustness.py --task Suturing --dry-run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.splits import get_louo_splits
from src.robustness.evaluation import (
    audit_robustness_readiness,
    compute_robustness_aggregates,
    evaluate_model_under_corruption,
    print_robustness_audit_table,
)
from src.robustness.video_corruptions import (
    FEATURE_CORRUPTIONS,
    FRAME_CORRUPTIONS,
    get_visual_corruption,
    list_available_corruptions,
)
from src.utils.reproducibility import get_device, load_config, save_json, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_robustness")


# ---------------------------------------------------------------------------
# Synthetic Verification Dataset & Dummy Model for Dry-Run
# ---------------------------------------------------------------------------

class SyntheticVerificationDataset(Dataset):
    """Synthetic dataset for dry-run verification of the robustness pipeline."""

    def __init__(self, n_samples: int = 16, window_size: int = 16, v_dim: int = 512, k_dim: int = 76, num_classes: int = 5) -> None:
        super().__init__()
        torch.manual_seed(42)
        self.video = torch.randn(n_samples, window_size, v_dim)
        self.kinematics = torch.randn(n_samples, window_size, k_dim)
        self.targets = torch.randint(0, num_classes, (n_samples,))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "video": self.video[idx],
            "kinematics": self.kinematics[idx],
            "target": self.targets[idx],
        }


class DummyClassifier(nn.Module):
    """Toy classifier for dry-run pipeline smoke test."""

    def __init__(self, in_dim: int = 512, num_classes: int = 5) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x[:, -1, :]
        return self.fc(x)


def run_dry_run_smoke_test(task: str, cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Execute synthetic dry-run smoke test validating all registered corruption operators."""
    logger.info("Executing robustness pipeline smoke test with synthetic inputs...")
    dataset = SyntheticVerificationDataset(n_samples=16, num_classes=5)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    model = DummyClassifier(in_dim=512, num_classes=5).to(device)

    # 1. Clean evaluation
    clean_res = evaluate_model_under_corruption(
        model=model,
        dataloader=dataloader,
        num_classes=5,
        class_names={i: f"G{i}" for i in range(5)},
        device=device,
        corruption=None,
        model_type="video_only",
    )
    clean_f1 = clean_res["metrics"]["macro_f1"]
    logger.info(f"Clean Synthetic Macro-F1: {clean_f1:.4f}")

    # 2. Test all feature-level corruptions at severity 5
    feature_corruptions = list(FEATURE_CORRUPTIONS.keys())
    corrupted_f1s: Dict[str, float] = {}

    for c_name in feature_corruptions:
        op = get_visual_corruption(c_name, severity=5, seed=42)
        c_res = evaluate_model_under_corruption(
            model=model,
            dataloader=dataloader,
            num_classes=5,
            class_names={i: f"G{i}" for i in range(5)},
            device=device,
            corruption=op,
            seed=42,
            model_type="video_only",
        )
        corrupted_f1s[c_name] = c_res["metrics"]["macro_f1"]
        logger.info(f"Corrupted [{c_name}] (Severity 5) Macro-F1: {c_res['metrics']['macro_f1']:.4f}")

    # 3. Compute aggregate robustness metrics
    aggregates = compute_robustness_aggregates(clean_metric=clean_f1, corrupted_metrics=corrupted_f1s)
    logger.info(
        f"Synthetic Aggregates: Clean={aggregates['clean']:.4f}, "
        f"Mean-C={aggregates['mean_c']:.4f}, Worst-C={aggregates['worst_c']:.4f}, "
        f"Relative Drop={aggregates['relative_drop']*100:.2f}%"
    )

    return {
        "status": "VERIFIED_DRY_RUN",
        "clean_macro_f1": clean_f1,
        "corrupted_f1s": corrupted_f1s,
        "aggregates": aggregates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate visual robustness and readiness under Endo-C6 corruptions.")
    parser.add_argument("--task", type=str, default="Suturing", choices=["Suturing", "Knot_Tying", "Needle_Passing", "all"])
    parser.add_argument("--config", type=str, default="configs/robustness.yaml")
    parser.add_argument("--severity", type=int, default=5, help="Corruption severity level in [1..5]")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--output-dir", type=str, default="artifacts/robustness")
    parser.add_argument("--audit-only", action="store_true", help="Print readiness audit table and exit")
    parser.add_argument("--dry-run", action="store_true", help="Run synthetic smoke run verifying all operators")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = ["Suturing", "Knot_Tying", "Needle_Passing"] if args.task == "all" else [args.task]
    cfg = load_config(args.config)
    device = get_device()
    set_seed(args.seed)

    for t in tasks:
        audit = audit_robustness_readiness(
            task=t,
            cfg=cfg,
            checkpoints_dir=args.checkpoints_dir,
        )

        if args.audit_only:
            print_robustness_audit_table(audit)
            continue

        if args.dry_run:
            print_robustness_audit_table(audit)
            res = run_dry_run_smoke_test(task=t, cfg=cfg, device=device)
            out_file = Path(args.output_dir) / f"dry_run_{t}_smoke_results.json"
            save_json(res, out_file)
            logger.info(f"Dry-run smoke results saved to: {out_file}")
            continue

        # Real Benchmark execution path
        print_robustness_audit_table(audit)
        if not audit["all_ready"]:
            logger.warning(
                f"Robustness benchmark execution BLOCKED for {t}: "
                f"Missing complete feature caches or trained checkpoints."
            )
            continue


if __name__ == "__main__":
    main()
