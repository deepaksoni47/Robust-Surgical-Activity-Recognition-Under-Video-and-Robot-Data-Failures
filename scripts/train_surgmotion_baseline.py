"""Train and evaluate downstream SurgMotion gesture classification models on JIGSAWS.

Supports:
1. VideoMLP: Clip-level sanity baseline (evaluates individual 64-frame tubelets).
2. VideoGRU: Temporal recurrent sequence model over sliding sequences of SurgMotion clips.

Strict Evaluation Protocol:
- Official Leave-One-User-Out (LOUO) cross-validation (8 folds for Suturing & Knot Tying;
  7 active folds for Needle Passing, excluding Fold 6 User G).
- Zero data leakage: normalization statistics are fitted strictly on training trials of each fold.
- Evaluation metrics: Accuracy, Balanced Accuracy, Macro-F1, and Levenshtein Edit Score.
- Saves model checkpoints, fold predictions, and machine-readable aggregate summaries.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import FOLD_USERS, get_louo_splits
from src.data.surgmotion_dataset import JIGSAWSSurgMotionDataset, get_surgmotion_dataloaders
from src.data.transcriptions import get_gesture_vocabulary
from src.evaluation.metrics import compute_classification_metrics, compute_edit_score
from src.models.video_head import VideoGRU, VideoMLP
from src.utils.reproducibility import get_device, save_json, set_seed


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch over mini-batches of SurgMotion sequences."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        feats = batch["features"].to(device)  # [B, S, 1024]
        targets = batch["target"].to(device)  # [B]

        optimizer.zero_grad()
        logits = model(feats)
        loss = criterion(logits, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    class_names: Optional[Dict[int, str]],
    device: torch.device,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate model on a DataLoader and compute classification metrics."""
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []

    for batch in dataloader:
        feats = batch["features"].to(device)
        targets = batch["target"].cpu().numpy()

        logits = model(feats)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())

    y_true = np.array(all_targets, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
        class_names=class_names,
    )
    return metrics, y_true, y_pred


def compute_surgmotion_edit_score(
    model: nn.Module,
    dataset: JIGSAWSSurgMotionDataset,
    device: torch.device,
) -> float:
    """Compute segmental edit score across test trials in a SurgMotion dataset."""
    model.eval()
    true_seqs: List[List[int]] = []
    pred_seqs: List[List[int]] = []

    with torch.no_grad():
        for rec in dataset.trial_records:
            # rec["features"] shape: (N_clips, 1024)
            feats_t = torch.from_numpy(rec["features"]).float().to(device)
            # Evaluate sequential predictions
            if hasattr(model, "forward_sequence"):
                logits_seq = model.forward_sequence(feats_t.unsqueeze(0)).squeeze(0)  # (N_clips, C)
                preds = torch.argmax(logits_seq, dim=-1).cpu().numpy().tolist()
            else:
                logits = model(feats_t)  # (N_clips, C)
                preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()

            true_labels = rec["labels"].tolist()
            true_seqs.append(true_labels)
            pred_seqs.append(preds)

    return compute_edit_score(true_seqs, pred_seqs)


def train_and_eval_surgmotion_fold(
    task: str,
    fold: int,
    model_name: str = "VideoGRU",
    sequence_length: int = 8,
    batch_size: int = 32,
    epochs: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device_name: Optional[str] = None,
    save_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train and evaluate downstream SurgMotion model on one official LOUO fold."""
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab = get_gesture_vocabulary(task=task, include_background=True)
    num_classes = len(vocab)
    class_names = {idx: g for g, idx in vocab.items()}

    # 1. Build DataLoaders with zero-leakage training-only normalizer
    train_loader, val_loader, test_loader, normalizer, fold_info = get_surgmotion_dataloaders(
        task=task,
        fold=fold,
        sequence_length=sequence_length,
        sequence_stride=1,
        batch_size=batch_size,
    )

    if len(fold_info["test_trials"]) == 0:
        return {
            "fold": fold,
            "task": task,
            "test_user": fold_info["test_user"],
            "is_active": False,
            "reason": "Official test set empty",
        }

    # 2. Build Model
    if model_name == "VideoGRU":
        model = VideoGRU(
            input_dim=1024,
            proj_dim=128,
            hidden_dim=128,
            num_classes=num_classes,
            num_layers=2,
            dropout=0.2,
        ).to(device)
    elif model_name == "VideoMLP":
        model = VideoMLP(
            input_dim=1024,
            hidden_dim=128,
            num_classes=num_classes,
            dropout=0.2,
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Compute class weights for imbalanced cross-entropy
    train_targets = []
    for b in train_loader:
        train_targets.extend(b["target"].numpy().tolist())
    counts = np.bincount(train_targets, minlength=num_classes)
    total_samples = len(train_targets)
    weights = np.where(counts > 0, total_samples / (num_classes * counts), 1.0)
    class_weights_t = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_macro_f1 = -1.0
    best_state: Optional[Dict[str, Any]] = None
    best_epoch = 0

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _ = evaluate_dataloader(model, val_loader, num_classes, class_names, device)
        val_f1 = val_metrics["macro_f1"]

        scheduler.step(val_f1)

        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    # 4. Test Evaluation using best validation checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, y_true, y_pred = evaluate_dataloader(model, test_loader, num_classes, class_names, device)
    edit_score = compute_surgmotion_edit_score(model, test_loader.dataset, device)
    test_metrics["edit_score"] = float(round(edit_score, 2))

    # 5. Save artifacts if save_dir provided
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f"surgmotion_{model_name}_{task}_fold{fold}.pt"
        torch.save({
            "model_state": best_state,
            "fold": fold,
            "task": task,
            "model_name": model_name,
            "sequence_length": sequence_length,
            "best_epoch": best_epoch,
            "test_metrics": test_metrics,
            "normalizer": normalizer.to_dict(),
        }, ckpt_path)

        pred_path = save_dir / f"preds_surgmotion_{model_name}_{task}_fold{fold}.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump({
                "trial_id": fold_info["test_trials"],
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist(),
                "metrics": test_metrics,
            }, f, indent=2)

    return {
        "fold": fold,
        "task": task,
        "test_user": fold_info["test_user"],
        "is_active": True,
        "best_epoch": best_epoch,
        "accuracy": round(float(test_metrics["accuracy"]) * 100.0, 2),
        "balanced_accuracy": round(float(test_metrics["balanced_accuracy"]) * 100.0, 2),
        "macro_f1": round(float(test_metrics["macro_f1"]) * 100.0, 2),
        "edit_score": test_metrics["edit_score"],
        "n_train_trials": len(fold_info["train_trials"]),
        "n_val_trials": len(fold_info["val_trials"]),
        "n_test_trials": len(fold_info["test_trials"]),
    }


def run_full_cross_validation(
    task: str = "Suturing",
    model_name: str = "VideoGRU",
    sequence_length: int = 8,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: Optional[str] = None,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """Execute complete official LOUO cross-validation for a task."""
    set_seed(42)
    t0_task = time.time()
    splits = get_louo_splits(task=task, create_val_split=True)
    save_dir = PROJECT_ROOT / f"checkpoints/surgmotion_{model_name}_{task}" if save_artifacts else None

    print("=" * 70)
    print(f"SURGMOTION LOUO BENCHMARK: Task={task} | Model={model_name} | W={sequence_length}")
    print("=" * 70)

    fold_results = []
    for fold_num in sorted(splits.keys()):
        fold_info = splits[fold_num]
        if len(fold_info["test_trials"]) == 0:
            print(f"Fold {fold_num} (User {fold_info['test_user']}): Omitted (empty test set in official split)")
            continue

        print(f"\n--- Training Fold {fold_num} / 8 (Test User: {fold_info['test_user']}, {len(fold_info['test_trials'])} trials) ---")
        res = train_and_eval_surgmotion_fold(
            task=task,
            fold=fold_num,
            model_name=model_name,
            sequence_length=sequence_length,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            device_name=device,
            save_dir=save_dir,
        )
        fold_results.append(res)
        print(f"Result Fold {fold_num}: Acc={res['accuracy']:.2f}%, BalAcc={res['balanced_accuracy']:.2f}%, F1={res['macro_f1']:.2f}%, Edit={res['edit_score']:.2f}%")

    # Aggregate across active folds
    accs = [r["accuracy"] for r in fold_results]
    bal_accs = [r["balanced_accuracy"] for r in fold_results]
    f1s = [r["macro_f1"] for r in fold_results]
    edits = [r["edit_score"] for r in fold_results]

    summary = {
        "task": task,
        "model": model_name,
        "encoder": "SurgMotion-ViTL (Frozen)",
        "feature_dim": 1024,
        "sequence_length": sequence_length,
        "active_folds": len(fold_results),
        "total_elapsed_sec": round(time.time() - t0_task, 2),
        "metrics": {
            "accuracy": {"mean": round(float(np.mean(accs)), 2), "min": round(float(np.min(accs)), 2), "max": round(float(np.max(accs)), 2)},
            "balanced_accuracy": {"mean": round(float(np.mean(bal_accs)), 2), "min": round(float(np.min(bal_accs)), 2), "max": round(float(np.max(bal_accs)), 2)},
            "macro_f1": {"mean": round(float(np.mean(f1s)), 2), "min": round(float(np.min(f1s)), 2), "max": round(float(np.max(f1s)), 2)},
            "edit_score": {"mean": round(float(np.mean(edits)), 2), "min": round(float(np.min(edits)), 2), "max": round(float(np.max(edits)), 2)},
        },
        "fold_results": fold_results,
    }

    # Save summary artifact
    if save_artifacts:
        summary_path = PROJECT_ROOT / f"artifacts/surgmotion_{model_name}_{task}_results.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary results to: {summary_path.resolve()}")

    print("\n" + "=" * 70)
    print(f"FINAL LOUO AGGREGATE SUMMARY: {task} ({model_name})")
    print("=" * 70)
    print(f"Accuracy:          {summary['metrics']['accuracy']['mean']:.2f}% (min: {summary['metrics']['accuracy']['min']:.2f}%, max: {summary['metrics']['accuracy']['max']:.2f}%)")
    print(f"Balanced Accuracy: {summary['metrics']['balanced_accuracy']['mean']:.2f}% (min: {summary['metrics']['balanced_accuracy']['min']:.2f}%, max: {summary['metrics']['balanced_accuracy']['max']:.2f}%)")
    print(f"Macro-F1:          {summary['metrics']['macro_f1']['mean']:.2f}% (min: {summary['metrics']['macro_f1']['min']:.2f}%, max: {summary['metrics']['macro_f1']['max']:.2f}%)")
    print(f"Edit Score:        {summary['metrics']['edit_score']['mean']:.2f}% (min: {summary['metrics']['edit_score']['min']:.2f}%, max: {summary['metrics']['edit_score']['max']:.2f}%)")
    print(f"Total Time:        {summary['total_elapsed_sec']:.1f} s")
    print("=" * 70)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate SurgMotion downstream classifiers on JIGSAWS.")
    parser.add_argument("--task", type=str, default="Suturing",
                        choices=["Suturing", "Knot_Tying", "Needle_Passing", "all"],
                        help="Target task")
    parser.add_argument("--model", type=str, default="VideoGRU",
                        choices=["VideoGRU", "VideoMLP"],
                        help="Downstream temporal architecture")
    parser.add_argument("--sequence-length", type=int, default=8,
                        help="Number of consecutive SurgMotion clips per sequence window")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default=None,
                        help="Execution device (cuda or cpu)")
    args = parser.parse_args()

    tasks = ["Suturing", "Knot_Tying", "Needle_Passing"] if args.task == "all" else [args.task]
    for t in tasks:
        seq_len = 1 if args.model == "VideoMLP" else args.sequence_length
        run_full_cross_validation(
            task=t,
            model_name=args.model,
            sequence_length=seq_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )


if __name__ == "__main__":
    main()
