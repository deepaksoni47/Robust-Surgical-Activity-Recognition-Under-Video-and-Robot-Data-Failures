#!/usr/bin/env python3
"""scripts/train_adaptive_fusion.py

Train and evaluate the Adaptive Reliability-Aware Fusion Architecture on JIGSAWS.
Integrates unimodal sequence representations with dynamic gating weights,
explicit modality availability masking, and single-modality fallback verification.

Usage:
    # Audit fold readiness
    python scripts/train_adaptive_fusion.py --task Suturing --audit-only

    # Dry-run / smoke run on available data
    python scripts/train_adaptive_fusion.py --task Suturing --dry-run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.feature_cache import get_cache_path
from src.data.multimodal_dataset import (
    get_multimodal_fold_dataloaders,
    get_multimodal_fold_datasets,
)
from src.data.splits import get_louo_splits
from src.data.transcriptions import get_gesture_vocabulary
from src.evaluation.benchmark_comparison import audit_task_multimodal_readiness
from src.evaluation.metrics import (
    collapse_to_segments,
    compute_classification_metrics,
    compute_edit_score,
)
from src.models.kinematics_gru import KinematicsGRU
from src.models.reliability_gate import (
    AdaptiveReliabilityFusion,
    BothModalitiesUnavailableError,
    ReliabilityGate,
)
from src.models.video_head import VideoGRU
from src.utils.reproducibility import get_device, load_config, save_json, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_adaptive_fusion")


def check_adaptive_fusion_readiness(
    task: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Audit readiness of all folds for adaptive multimodal training."""
    data_cfg = cfg.get("data", {})
    cache_dir = Path(data_cfg.get("cache_dir", "data/cache/features"))
    dataset_dir = Path(data_cfg.get("dataset_dir", "dataset"))
    encoder_name = data_cfg.get("encoder_name", "ResNet18")
    camera = data_cfg.get("camera", "capture1")

    splits = get_louo_splits(task, create_val_split=True)
    fold_status: Dict[int, Dict[str, Any]] = {}
    ready_folds: List[int] = []

    for fold_num in sorted(splits.keys()):
        fold_info = splits[fold_num]
        all_trials = fold_info["train_trials"] + fold_info.get("val_trials", []) + fold_info["test_trials"]

        missing_caches: List[str] = []
        missing_kin: List[str] = []

        for t_id in all_trials:
            c_path = get_cache_path(cache_dir, t_id, encoder_name, camera=camera)
            if not c_path.is_file():
                missing_caches.append(t_id)

            k_path = dataset_dir / task / "kinematics" / "AllGestures" / f"{t_id}.txt"
            if not k_path.is_file():
                missing_kin.append(t_id)

        is_ready = (len(missing_caches) == 0 and len(missing_kin) == 0)
        if is_ready:
            ready_folds.append(fold_num)

        k_ckpt_path = Path("checkpoints") / f"kinematics_gru_{task}_fold{fold_num}.pt"
        v_ckpt_path = Path("checkpoints") / f"video_videogru_{task}_fold{fold_num}.pt"

        fold_status[fold_num] = {
            "test_user": fold_info["test_user"],
            "ready": is_ready,
            "missing_cache_count": len(missing_caches),
            "missing_kinematics_count": len(missing_kin),
            "missing_caches": missing_caches,
            "has_kinematics_checkpoint": k_ckpt_path.is_file(),
            "has_video_checkpoint": v_ckpt_path.is_file(),
        }

    return {
        "task": task,
        "total_folds": len(splits),
        "ready_folds": ready_folds,
        "fold_status": fold_status,
        "all_ready": (len(ready_folds) == len(splits)),
    }


def print_readiness_audit(audit: Dict[str, Any]) -> None:
    """Print a clean Markdown-formatted readiness audit table."""
    task = audit["task"]
    total = audit["total_folds"]
    ready = len(audit["ready_folds"])
    status_overall = "READY" if audit["all_ready"] else "BLOCKED"

    print(f"\n=======================================================")
    print(f"LOUO Fold Readiness Audit for Task: {task}")
    print(f"Status: {status_overall} ({ready}/{total} folds ready)")
    print(f"=======================================================")
    print(f"| Fold | Test User | Status | Missing Caches | Missing Kinematics | Kin Checkpoint | Video Checkpoint |")
    print(f"|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for f_num, f_info in sorted(audit["fold_status"].items()):
        s_str = "READY" if f_info["ready"] else "BLOCKED"
        k_str = "YES" if f_info["has_kinematics_checkpoint"] else "NO"
        v_str = "YES" if f_info["has_video_checkpoint"] else "NO"
        print(
            f"| Fold {f_num} | User {f_info['test_user']} | {s_str} | "
            f"{f_info['missing_cache_count']} | {f_info['missing_kinematics_count']} | {k_str} | {v_str} |"
        )

    if not audit["all_ready"]:
        print(f"\n[BLOCKED] Real multimodal training/evaluation cannot proceed for {task}:")
        print(f"  Only {ready}/{total} folds have complete feature caches.")
        print(f"\nRequired commands to unblock:")
        print(f"  1. Extract visual features:")
        print(f"     python scripts/extract_visual_features.py --task {task} --encoder ResNet18")
        print(f"  2. Train video baseline (if submodel checkpoints required):")
        print(f"     python scripts/train_video_baseline.py --task {task}")
        print(f"  3. Train adaptive reliability fusion:")
        print(f"     python scripts/train_adaptive_fusion.py --task {task}\n")


def verify_checkpoint_reproducibility(
    checkpoint_path: Path,
    original_model: AdaptiveReliabilityFusion,
    dataloader: DataLoader,
    cfg: Dict[str, Any],
    num_classes: int,
    device: torch.device,
    tolerance: float = 1e-6,
) -> bool:
    """Verify that reloading checkpoint into a fresh model reproduces exact evaluation outputs within tolerance."""
    twin_model = build_adaptive_model(cfg=cfg, num_classes=num_classes, device=device)
    twin_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    twin_model.eval()
    original_model.eval()

    for batch in dataloader:
        v_x = batch["video"].to(device)
        k_x = batch["kinematics"].to(device)
        b_size = v_x.shape[0]
        mask = torch.ones((b_size, 2), dtype=torch.bool, device=device)

        with torch.no_grad():
            orig_logits, orig_alpha = original_model(video_x=v_x, kinematics_x=k_x, availability_mask=mask)
            twin_logits, twin_alpha = twin_model(video_x=v_x, kinematics_x=k_x, availability_mask=mask)

        diff_logits = (orig_logits - twin_logits).abs().max().item()
        diff_alpha = (orig_alpha - twin_alpha).abs().max().item()

        if diff_logits > tolerance or diff_alpha > tolerance:
            raise ValueError(
                f"Checkpoint reload failed reproducibility test: logits max diff={diff_logits:.2e}, "
                f"alpha max diff={diff_alpha:.2e} (tolerance={tolerance:.2e})"
            )
        break

    return True


def build_adaptive_model(
    cfg: Dict[str, Any],
    num_classes: int,
    device: torch.device,
) -> AdaptiveReliabilityFusion:
    """Build AdaptiveReliabilityFusion with modality-specific GRU heads and ReliabilityGate."""
    sub_cfg = cfg.get("submodels", {})
    v_cfg = sub_cfg.get("video", {})
    k_cfg = sub_cfg.get("kinematics", {})
    gate_cfg = cfg.get("model", {})

    v_model = VideoGRU(
        input_dim=v_cfg.get("input_dim", 512),
        proj_dim=v_cfg.get("proj_dim", 128),
        hidden_dim=v_cfg.get("hidden_dim", 128),
        num_classes=num_classes,
        num_layers=v_cfg.get("num_layers", 2),
        dropout=v_cfg.get("dropout", 0.2),
    )

    k_model = KinematicsGRU(
        input_dim=k_cfg.get("input_dim", 76),
        proj_dim=k_cfg.get("proj_dim", 128),
        hidden_dim=k_cfg.get("hidden_dim", 128),
        num_classes=num_classes,
        num_layers=k_cfg.get("num_layers", 2),
        dropout=k_cfg.get("dropout", 0.2),
    )

    gate = ReliabilityGate(
        video_dim=num_classes,
        kinematics_dim=num_classes,
        hidden_dim=gate_cfg.get("hidden_dim", 32),
        use_linear_direct=gate_cfg.get("use_linear_direct", True),
        dropout=gate_cfg.get("dropout", 0.1),
    )

    model = AdaptiveReliabilityFusion(
        num_classes=num_classes,
        video_model=v_model,
        kinematics_model=k_model,
        gate=gate,
    ).to(device)

    return model


def train_one_epoch(
    model: AdaptiveReliabilityFusion,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    missing_prob: float = 0.2,
) -> float:
    """Execute one training epoch with optional simulated missingness."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        v_x = batch["video"].to(device)
        k_x = batch["kinematics"].to(device)
        targets = batch["target"].to(device)
        b_size = v_x.shape[0]

        # Generate availability masks
        mask = torch.ones((b_size, 2), dtype=torch.bool, device=device)
        if missing_prob > 0.0:
            rand_vals = torch.rand(b_size, device=device)
            # Mask out video with probability missing_prob / 2
            v_drop = rand_vals < (missing_prob / 2.0)
            mask[v_drop, 0] = False
            # Mask out kinematics with probability missing_prob / 2
            k_drop = (rand_vals >= (missing_prob / 2.0)) & (rand_vals < missing_prob)
            mask[k_drop, 1] = False

        optimizer.zero_grad()
        logits, _ = model(video_x=v_x, kinematics_x=k_x, availability_mask=mask)
        loss = criterion(logits, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_adaptive_model(
    model: AdaptiveReliabilityFusion,
    dataloader: DataLoader,
    num_classes: int,
    class_names: Optional[Dict[int, str]],
    device: torch.device,
    fixed_mask: Optional[Tuple[bool, bool]] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate adaptive model under clean or forced fallback condition."""
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_weights: List[List[float]] = []

    for batch in dataloader:
        v_x = batch["video"].to(device) if (fixed_mask is None or fixed_mask[0]) else None
        k_x = batch["kinematics"].to(device) if (fixed_mask is None or fixed_mask[1]) else None
        targets = batch["target"].cpu().numpy()
        b_size = len(targets)

        if fixed_mask is not None:
            mask = torch.tensor([fixed_mask], dtype=torch.bool, device=device).expand(b_size, 2)
        else:
            mask = torch.ones((b_size, 2), dtype=torch.bool, device=device)

        logits, alpha = model(video_x=v_x, kinematics_x=k_x, availability_mask=mask)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())
        all_weights.extend(alpha.cpu().numpy().tolist())

    y_pred = np.array(all_preds, dtype=np.int64)
    y_true = np.array(all_targets, dtype=np.int64)
    weights_arr = np.array(all_weights, dtype=np.float32)

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
        class_names=class_names,
    )

    return metrics, y_true, y_pred, weights_arr


def train_and_eval_adaptive_fold(
    task: str,
    fold_num: int,
    cfg: Dict[str, Any],
    device: torch.device,
    save_checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train and evaluate adaptive model on a single LOUO fold."""
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    window_size = data_cfg.get("window_size", 16)
    train_stride = data_cfg.get("train_stride", 8)
    eval_stride = data_cfg.get("eval_stride", 8)
    cache_dir = data_cfg.get("cache_dir", "data/cache/features")
    encoder_name = data_cfg.get("encoder_name", "ResNet18")
    camera = data_cfg.get("camera", "capture1")
    normalize = data_cfg.get("normalize", True)

    epochs = train_cfg.get("epochs", 15)
    batch_size = train_cfg.get("batch_size", 32)
    lr = train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    patience = train_cfg.get("patience", 4)
    missing_prob = train_cfg.get("train_missing_prob", 0.2)

    splits = get_louo_splits(task, create_val_split=True)
    fold_info = splits[fold_num]
    vocab = get_gesture_vocabulary(task)
    num_classes = len(vocab)
    inv_vocab = {v: k for k, v in vocab.items()}

    loaders = get_multimodal_fold_dataloaders(
        task=task,
        fold=fold_num,
        batch_size=batch_size,
        window_size=window_size,
        train_stride=train_stride,
        eval_stride=eval_stride,
        cache_dir=cache_dir,
        encoder_name=encoder_name,
        camera=camera,
        normalize=normalize,
        use_class_weights=train_cfg.get("use_class_weights", True),
    )

    train_loader = loaders["dataloaders"]["train"]
    val_loader = loaders["dataloaders"]["val"]
    test_loader = loaders["dataloaders"]["test"]

    model = build_adaptive_model(cfg, num_classes=num_classes, device=device)

    if train_cfg.get("use_class_weights", True):
        criterion = nn.CrossEntropyLoss(weight=loaders["class_weights"].to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training with early stopping on validation Macro-F1
    best_val_score = -1.0
    best_state: Optional[Dict[str, Any]] = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device, missing_prob=missing_prob)

        if val_loader is not None and len(loaders["datasets"]["val"]) > 0:
            val_metrics, _, _, _ = evaluate_adaptive_model(model, val_loader, num_classes, inv_vocab, device)
            val_score = val_metrics["macro_f1"]

            if val_score > best_val_score:
                best_val_score = val_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break
        else:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_checkpoint_path is not None:
        save_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_checkpoint_path)
        logger.info(f"Saved best model checkpoint to {save_checkpoint_path}")

        # Verify checkpoint reload reproducibility
        verify_checkpoint_reproducibility(
            checkpoint_path=save_checkpoint_path,
            original_model=model,
            dataloader=test_loader,
            cfg=cfg,
            num_classes=num_classes,
            device=device,
        )
        logger.info("Checkpoint reload reproducibility verified within numerical tolerance.")

    # 1. Clean Test Evaluation
    clean_metrics, y_true, y_pred, weights = evaluate_adaptive_model(
        model, test_loader, num_classes, inv_vocab, device, fixed_mask=(True, True)
    )

    # 2. Video-Only Fallback Evaluation
    v_metrics, _, _, _ = evaluate_adaptive_model(
        model, test_loader, num_classes, inv_vocab, device, fixed_mask=(True, False)
    )

    # 3. Kinematics-Only Fallback Evaluation
    k_metrics, _, _, _ = evaluate_adaptive_model(
        model, test_loader, num_classes, inv_vocab, device, fixed_mask=(False, True)
    )

    clean_metrics["fold"] = fold_num
    clean_metrics["test_user"] = fold_info["test_user"]
    clean_metrics["video_only_fallback"] = {
        "accuracy": v_metrics["accuracy"],
        "macro_f1": v_metrics["macro_f1"],
    }
    clean_metrics["kinematics_only_fallback"] = {
        "accuracy": k_metrics["accuracy"],
        "macro_f1": k_metrics["macro_f1"],
    }
    clean_metrics["mean_gating_weights"] = {
        "video": float(np.mean(weights[:, 0])),
        "kinematics": float(np.mean(weights[:, 1])),
    }

    logger.info(
        f"Fold {fold_num} (User {fold_info['test_user']}): "
        f"Clean Acc={clean_metrics['accuracy']*100:.2f}%, F1={clean_metrics['macro_f1']:.4f} | "
        f"Video-Only Acc={v_metrics['accuracy']*100:.2f}% | "
        f"Kin-Only Acc={k_metrics['accuracy']*100:.2f}%"
    )

    return clean_metrics


def run_adaptive_fusion_benchmark(
    task: str = "Suturing",
    config_path: str = "configs/adaptive_fusion.yaml",
    output_dir: str = "artifacts",
    checkpoints_dir: str = "checkpoints",
    specific_fold: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute adaptive fusion benchmark or readiness audit."""
    cfg = load_config(config_path)
    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)
    device = get_device()

    logger.info("=" * 70)
    logger.info(f"ADAPTIVE RELIABILITY-AWARE FUSION BENCHMARK: {task}")
    logger.info(f"Device: {device} | Seed: {seed}")
    logger.info("=" * 70)

    # 1. Audit readiness
    audit = check_adaptive_fusion_readiness(task, cfg)
    splits = get_louo_splits(task, create_val_split=True)
    target_folds = [specific_fold] if specific_fold is not None else sorted(splits.keys())

    print_readiness_audit(audit)

    if not audit["all_ready"] and not dry_run:
        logger.warning(
            f"Benchmark execution BLOCKED for {task}: Only {len(audit['ready_folds'])}/{len(splits)} folds have complete feature caches."
        )
        return {
            "status": "BLOCKED",
            "task": task,
            "ready_folds": audit["ready_folds"],
            "total_folds": len(splits),
            "audit": audit,
        }

    # 2. Training loop across available folds
    fold_results: List[Dict[str, Any]] = []
    ckpt_dir = Path(checkpoints_dir)

    for f_num in target_folds:
        if not audit["fold_status"][f_num]["ready"] and not dry_run:
            continue

        ckpt_file = ckpt_dir / f"adaptive_fusion_{task}_fold{f_num}.pt"
        f_res = train_and_eval_adaptive_fold(
            task=task,
            fold_num=f_num,
            cfg=cfg,
            device=device,
            save_checkpoint_path=ckpt_file,
        )
        fold_results.append(f_res)

        if dry_run:
            logger.info("Dry-run requested: stopped after first fold.")
            break

    if not fold_results:
        return {"status": "BLOCKED", "task": task, "audit": audit}

    accs = [r["accuracy"] * 100 for r in fold_results]
    f1s = [r["macro_f1"] for r in fold_results]

    summary = {
        "model": "AdaptiveReliabilityFusion",
        "task": task,
        "n_folds": len(fold_results),
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_macro_f1": float(np.mean(f1s)),
        "std_macro_f1": float(np.std(f1s)),
        "folds": fold_results,
    }

    out_path = Path(output_dir) / f"adaptive_fusion_{task}_results.json"
    save_json(summary, out_path)
    logger.info(f"Results saved to: {out_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Adaptive Reliability-Aware Fusion on JIGSAWS.")
    parser.add_argument("--task", type=str, default="Suturing", choices=["Suturing", "Knot_Tying", "Needle_Passing", "all"])
    parser.add_argument("--config", type=str, default="configs/adaptive_fusion.yaml")
    parser.add_argument("--fold", type=int, default=None, help="Specific fold number to evaluate")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--audit-only", action="store_true", help="Print fold readiness audit and exit")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 epoch smoke test on first available fold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = ["Suturing", "Knot_Tying", "Needle_Passing"] if args.task == "all" else [args.task]

    for t in tasks:
        if args.audit_only:
            cfg = load_config(args.config)
            audit = check_adaptive_fusion_readiness(t, cfg)
            print_readiness_audit(audit)
        else:
            run_adaptive_fusion_benchmark(
                task=t,
                config_path=args.config,
                output_dir=args.output_dir,
                checkpoints_dir=args.checkpoints_dir,
                specific_fold=args.fold,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
