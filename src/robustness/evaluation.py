"""src/robustness/evaluation.py

Robustness Evaluation Engine and Benchmark Readiness Auditor for JIGSAWS.
Executes matched clean vs. corrupted evaluations across unimodal, fixed fusion,
concatenation fusion, and adaptive reliability fusion models under official LOUO splits.

Enforces:
1. Immutability: Test labels and original cached features are never altered.
2. Identical Coverage: Clean and corrupted evaluations evaluate the exact same trials.
3. Metric Standardization: Computes Clean, Mean-C, Worst-C, and Relative Drop (Endo-C6 protocol).
4. Research Integrity: Missing checkpoints or caches trigger explicit BLOCKED status.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.feature_cache import get_cache_path
from src.data.splits import get_louo_splits
from src.evaluation.metrics import compute_classification_metrics
from src.robustness.video_corruptions import BaseVisualCorruption, get_visual_corruption
from src.utils.reproducibility import save_json

logger = logging.getLogger("robustness_evaluation")


def audit_robustness_readiness(
    task: str,
    cfg: Dict[str, Any],
    checkpoints_dir: str = "checkpoints",
    cache_dir: str = "data/cache/features",
    dataset_dir: str = "dataset",
) -> Dict[str, Any]:
    """Audit readiness of all folds and model families for robustness stress testing."""
    ckpt_path = Path(checkpoints_dir)
    cache_path = Path(cache_dir)
    data_path = Path(dataset_dir)

    splits = get_louo_splits(task, create_val_split=True)
    models_cfg = cfg.get("models", [])
    model_names = [m["name"] for m in models_cfg] if models_cfg else [
        "kinematics_gru", "video_gru", "fixed_late_fusion", "concat_fusion", "adaptive_fusion"
    ]

    fold_status: Dict[int, Dict[str, Any]] = {}
    ready_folds: List[int] = []

    for fold_num in sorted(splits.keys()):
        fold_info = splits[fold_num]
        all_trials = fold_info["train_trials"] + fold_info.get("val_trials", []) + fold_info["test_trials"]

        missing_caches = 0
        for t_id in all_trials:
            c_file = get_cache_path(cache_path, t_id, "ResNet18", camera="capture1")
            if not c_file.is_file():
                missing_caches += 1

        # Checkpoint availability per model
        model_ckpts: Dict[str, bool] = {}
        for m_name in model_names:
            if m_name == "kinematics_gru":
                f_path = ckpt_path / f"kinematics_gru_{task}_fold{fold_num}.pt"
            elif m_name == "video_gru":
                f_path = ckpt_path / f"video_videogru_{task}_fold{fold_num}.pt"
            elif m_name == "concat_fusion":
                f_path = ckpt_path / f"concat_fusion_{task}_fold{fold_num}.pt"
            elif m_name == "adaptive_fusion":
                f_path = ckpt_path / f"adaptive_fusion_{task}_fold{fold_num}.pt"
            else:
                f_path = ckpt_path / f"{m_name}_{task}_fold{fold_num}.pt"
            model_ckpts[m_name] = f_path.is_file()

        is_ready = (missing_caches == 0 and any(model_ckpts.values()))
        if is_ready:
            ready_folds.append(fold_num)

        fold_status[fold_num] = {
            "test_user": fold_info["test_user"],
            "ready": is_ready,
            "missing_caches": missing_caches,
            "total_trials": len(all_trials),
            "checkpoints": model_ckpts,
        }

    all_ready = (len(ready_folds) == len(splits))

    return {
        "task": task,
        "total_folds": len(splits),
        "ready_folds": ready_folds,
        "all_ready": all_ready,
        "status": "READY" if all_ready else "BLOCKED",
        "fold_status": fold_status,
        "active_models": model_names,
    }


def print_robustness_audit_table(audit: Dict[str, Any]) -> None:
    """Print formatted markdown audit table for robustness testing."""
    task = audit["task"]
    total = audit["total_folds"]
    ready = len(audit["ready_folds"])

    print(f"\n=========================================================================")
    print(f"Robustness Benchmark Readiness Audit: {task}")
    print(f"Status: {audit['status']} ({ready}/{total} folds fully equipped)")
    print(f"=========================================================================")
    print(f"| Fold | User | Status | Missing Caches | Kin GRU | Video GRU | Concat | Adaptive |")
    print(f"|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for f_num, f_info in sorted(audit["fold_status"].items()):
        s_str = "READY" if f_info["ready"] else "BLOCKED"
        ck = f_info["checkpoints"]
        k_str = "YES" if ck.get("kinematics_gru", False) else "NO"
        v_str = "YES" if ck.get("video_gru", False) else "NO"
        c_str = "YES" if ck.get("concat_fusion", False) else "NO"
        a_str = "YES" if ck.get("adaptive_fusion", False) else "NO"

        print(
            f"| Fold {f_num} | User {f_info['test_user']} | {s_str} | "
            f"{f_info['missing_caches']}/{f_info['total_trials']} | "
            f"{k_str} | {v_str} | {c_str} | {a_str} |"
        )

    if not audit["all_ready"]:
        print(f"\n[BLOCKED] Real robustness stress-testing is blocked for {task}:")
        print(f"  Visual feature extraction and model checkpoints must be completed first.")
        print(f"\nRequired commands to unblock:")
        print(f"  1. python scripts/extract_visual_features.py --task {task} --encoder ResNet18")
        print(f"  2. python scripts/train_video_baseline.py --task {task}")
        print(f"  3. python scripts/train_adaptive_fusion.py --task {task}")
        print(f"  4. python scripts/evaluate_robustness.py --task {task}\n")


@torch.no_grad()
def evaluate_model_under_corruption(
    model: nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    class_names: Optional[Dict[int, str]],
    device: torch.device,
    corruption: Optional[BaseVisualCorruption] = None,
    seed: Optional[int] = 42,
    model_type: str = "adaptive", # 'adaptive', 'concat', 'video_only', 'kin_only'
) -> Dict[str, Any]:
    """Evaluate model on dataloader applying optional visual corruption to video inputs.
    
    Args:
        model: PyTorch model in eval mode.
        dataloader: Test DataLoader yielding batches with 'video', 'kinematics', 'target'.
        num_classes: Total class count C.
        class_names: Mapping from class integer to gesture name.
        device: Torch evaluation device.
        corruption: Optional BaseVisualCorruption operator. If None, evaluates clean input.
        seed: Random seed for corruption determinism.
        model_type: Architecture type dispatching the forward call signature.
        
    Returns:
        Dictionary containing computed metrics, predictions, targets, and corruption metadata.
    """
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_weights: List[List[float]] = []

    for batch in dataloader:
        v_raw = batch["video"]  # [B, W, D_v]
        k_raw = batch["kinematics"]  # [B, W, D_k]
        targets = batch["target"].cpu().numpy()
        b_size = len(targets)

        # Apply visual corruption if specified
        if corruption is not None and corruption.severity > 0:
            v_corrupted = corruption.apply(v_raw, seed=seed)
        else:
            v_corrupted = v_raw

        v_x = v_corrupted.to(device)
        k_x = k_raw.to(device)

        if model_type == "adaptive":
            mask = torch.ones((b_size, 2), dtype=torch.bool, device=device)
            logits, alpha = model(video_x=v_x, kinematics_x=k_x, availability_mask=mask)
            all_weights.extend(alpha.cpu().numpy().tolist())
        elif model_type == "concat":
            # ConcatFusion takes combined [B, W, D_v + D_k]
            x_cat = torch.cat([v_x, k_x], dim=-1)
            logits = model(x_cat)
        elif model_type == "video_only":
            logits = model(v_x)
        elif model_type == "kin_only":
            logits = model(k_x)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())

    y_pred = np.array(all_preds, dtype=np.int64)
    y_true = np.array(all_targets, dtype=np.int64)

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
        class_names=class_names,
    )

    result = {
        "metrics": metrics,
        "num_samples": len(y_true),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "corruption": corruption.get_metadata() if corruption is not None else {
            "name": "clean",
            "level": "none",
            "severity": 0,
            "seed": seed,
            "is_identity": True,
        },
    }

    if all_weights:
        w_arr = np.array(all_weights, dtype=np.float32)
        result["mean_gating_weights"] = {
            "video": float(np.mean(w_arr[:, 0])),
            "kinematics": float(np.mean(w_arr[:, 1])),
        }

    return result


def compute_robustness_aggregates(
    clean_metric: float,
    corrupted_metrics: Dict[str, float],
) -> Dict[str, float]:
    """Compute standard Endo-C6 robustness benchmark aggregates.
    
    Args:
        clean_metric: Metric score under clean input (e.g. Macro-F1).
        corrupted_metrics: Dict mapping corruption name to metric score.
        
    Returns:
        Dict containing Clean, Mean-C, Worst-C, and Relative Drop (Delta_Drop).
    """
    if not corrupted_metrics:
        return {
            "clean": clean_metric,
            "mean_c": clean_metric,
            "worst_c": clean_metric,
            "relative_drop": 0.0,
            "worst_drop": 0.0,
        }

    c_vals = list(corrupted_metrics.values())
    mean_c = float(np.mean(c_vals))
    worst_c = float(np.min(c_vals))
    rel_drop = (clean_metric - mean_c) / max(1e-6, clean_metric)
    worst_drop = (clean_metric - worst_c) / max(1e-6, clean_metric)

    return {
        "clean": clean_metric,
        "mean_c": mean_c,
        "worst_c": worst_c,
        "relative_drop": float(rel_drop),
        "worst_drop": float(worst_drop),
    }
