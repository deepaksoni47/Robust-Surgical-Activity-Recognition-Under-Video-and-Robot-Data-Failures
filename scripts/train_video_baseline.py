from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.feature_cache import get_cache_path, validate_feature_cache
from src.data.splits import get_louo_splits
from src.data.video_dataset import get_video_fold_dataloaders
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_edit_score,
)
from src.models.video_head import VideoGRU, VideoMLP
from src.utils.reproducibility import get_device, load_config, save_json, set_seed


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch over batches of cached video feature sequences."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        video_feats = batch["video"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        logits = model(video_feats)
        loss = criterion(logits, targets)
        loss.backward()

        # Gradient clipping for recurrent and deep trunk stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    class_names: Optional[Dict[int, str]],
    device: torch.device,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate model on a DataLoader and compute full classification metrics."""
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []

    for batch in dataloader:
        video_feats = batch["video"].to(device)
        targets = batch["target"].cpu().numpy()

        logits = model(video_feats)
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

    return metrics, y_true, y_pred


def compute_trial_edit_scores(
    model: nn.Module,
    test_dataset: Any,
    device: torch.device,
) -> float:
    """Compute Segmental Edit Score (%) across trials in test dataset."""
    model.eval()
    true_seqs: List[List[int]] = []
    pred_seqs: List[List[int]] = []

    with torch.no_grad():
        for record in test_dataset.trial_records:
            # Video features shape: (T, D)
            feat_arr = torch.from_numpy(record["video"]).float().to(device)
            # Forward sequence through model: (1, T, D) -> (T, num_classes)
            logits_seq = model.forward_sequence(feat_arr.unsqueeze(0)).squeeze(0)
            preds = torch.argmax(logits_seq, dim=-1).cpu().numpy().tolist()
            true_labels = record["labels"].tolist()

            true_seqs.append(true_labels)
            pred_seqs.append(preds)

    return compute_edit_score(true_seqs, pred_seqs)


def check_fold_caches_available(
    task: str,
    fold_num: int,
    cache_dir: Union[str, Path] = "data/cache/features",
    encoder_name: str = "ResNet18",
    camera: str = "capture1",
    dataset_dir: Union[str, Path] = "dataset",
) -> Tuple[bool, List[str]]:
    """Check if all required trial feature caches exist for a specific fold."""
    splits = get_louo_splits(task, create_val_split=True)
    if fold_num not in splits:
        return False, []

    fold_info = splits[fold_num]
    required_trials = fold_info["train_trials"] + fold_info.get("val_trials", []) + fold_info["test_trials"]
    missing_trials: List[str] = []

    for t_id in required_trials:
        path = get_cache_path(cache_dir, t_id, encoder_name, camera=camera)
        if not path.is_file():
            missing_trials.append(t_id)

    is_complete = (len(missing_trials) == 0)
    return is_complete, missing_trials


def train_and_eval_fold(
    task: str,
    fold_num: int,
    cfg: Dict[str, Any],
    device: torch.device,
    save_checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train video classifier on one fold and evaluate test performance."""
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    model_name = model_cfg.get("name", "VideoGRU")
    window_size = data_cfg.get("window_size", 16)
    train_stride = data_cfg.get("train_stride", 8)
    eval_stride = data_cfg.get("eval_stride", 8)
    cache_dir = data_cfg.get("cache_dir", "data/cache/features")
    encoder_name = data_cfg.get("encoder_name", "ResNet18")
    camera = data_cfg.get("camera", "capture1")
    expected_dim = data_cfg.get("expected_dim", 512)
    normalize = data_cfg.get("normalize", True)

    batch_size = train_cfg.get("batch_size", 32)
    epochs = train_cfg.get("epochs", 15)
    lr = train_cfg.get("learning_rate", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    use_class_weights = train_cfg.get("use_class_weights", True)
    patience = train_cfg.get("patience", 4)

    # 1. Load DataLoaders
    loaders = get_video_fold_dataloaders(
        task=task,
        fold_num=fold_num,
        batch_size=batch_size,
        window_size=window_size,
        train_stride=train_stride,
        eval_stride=eval_stride,
        dataset_dir=data_cfg.get("dataset_dir", "dataset"),
        cache_dir=cache_dir,
        encoder_name=encoder_name,
        camera=camera,
        expected_dim=expected_dim,
        create_val_split=True,
        normalize=normalize,
    )

    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]
    vocab = loaders["vocab"]
    inv_vocab = {idx: g for g, idx in vocab.items()}
    num_classes = len(vocab)
    fold_info = loaders["fold_info"]

    if len(loaders["datasets"]["test"]) == 0:
        print(f"Skipping fold {fold_num}: 0 test trials.")
        return {}

    # 2. Build Model
    if model_name == "VideoMLP":
        model = VideoMLP(
            input_dim=model_cfg.get("input_dim", 512),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_classes=num_classes,
            dropout=model_cfg.get("dropout", 0.2),
        ).to(device)
    else:
        model = VideoGRU(
            input_dim=model_cfg.get("input_dim", 512),
            proj_dim=model_cfg.get("proj_dim", 128),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_classes=num_classes,
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.2),
            bidirectional=model_cfg.get("bidirectional", False),
        ).to(device)

    # 3. Loss & Optimizer
    if use_class_weights:
        weights = loaders["class_weights"].to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 4. Training loop with early stopping on validation Macro-F1
    best_val_score = -1.0
    best_state: Optional[Dict[str, Any]] = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        if val_loader is not None and len(loaders["datasets"]["val"]) > 0:
            val_metrics, _, _ = evaluate_model(model, val_loader, num_classes, inv_vocab, device)
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

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint if requested
    if save_checkpoint_path is not None:
        save_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_checkpoint_path)

    # 5. Final Test Evaluation
    test_metrics, _, _ = evaluate_model(model, test_loader, num_classes, inv_vocab, device)
    edit_score = compute_trial_edit_scores(model, loaders["datasets"]["test"], device)

    test_metrics["edit_score"] = edit_score
    test_metrics["fold"] = fold_num
    test_metrics["test_user"] = fold_info["test_user"]
    test_metrics["encoder"] = encoder_name
    test_metrics["feature_dim"] = model_cfg.get("input_dim", 512)

    print(
        f"Fold {fold_num} (User {fold_info['test_user']}): "
        f"Acc={test_metrics['accuracy']*100:.2f}%, "
        f"BalAcc={test_metrics['balanced_accuracy']*100:.2f}%, "
        f"Macro-F1={test_metrics['macro_f1']:.4f}, "
        f"Edit={edit_score:.2f}%"
    )

    return test_metrics


def run_video_baseline_benchmark(
    task: str = "Suturing",
    config_path: str = "configs/video_baseline.yaml",
    output_dir: str = "artifacts",
    checkpoints_dir: str = "checkpoints",
    specific_fold: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run LOUO benchmark for Video baseline classifier on a task."""
    cfg = load_config(config_path)
    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)
    device = get_device()

    model_name = cfg.get("model", {}).get("name", "VideoGRU")
    encoder_name = cfg.get("data", {}).get("encoder_name", "ResNet18")
    feat_dim = cfg.get("model", {}).get("input_dim", 512)
    cache_dir = cfg.get("data", {}).get("cache_dir", "data/cache/features")
    camera = cfg.get("data", {}).get("camera", "capture1")

    splits = get_louo_splits(task, create_val_split=True)
    target_folds = [specific_fold] if specific_fold is not None else sorted(splits.keys())

    print(f"\n==========================================")
    print(f"Benchmarking {model_name} (Encoder: {encoder_name}, Dim: {feat_dim}) on Task: {task}")
    print(f"Device: {device}")
    print(f"Folds to evaluate: {target_folds}")
    print(f"==========================================")

    # 1. Check cache availability across folds
    available_folds: List[int] = []
    missing_folds_info: Dict[int, List[str]] = {}

    for f_num in target_folds:
        is_complete, missing = check_fold_caches_available(
            task=task,
            fold_num=f_num,
            cache_dir=cache_dir,
            encoder_name=encoder_name,
            camera=camera,
        )
        if is_complete:
            available_folds.append(f_num)
        else:
            missing_folds_info[f_num] = missing

    if missing_folds_info:
        all_missing = sorted(list({t for miss in missing_folds_info.values() for t in miss}))
        print(f"\n[CACHE STATUS] {len(available_folds)}/{len(target_folds)} folds have complete feature caches.")
        print(f"Missing caches detected for {len(all_missing)} trials across folds: {list(missing_folds_info.keys())}")
        print(f"To extract cached features for all trials in task '{task}', execute:")
        print(f"    python scripts/extract_visual_features.py --task {task} --encoder {encoder_name}\n")

    if dry_run:
        return {
            "status": "DRY_RUN",
            "task": task,
            "encoder": encoder_name,
            "available_folds": available_folds,
            "missing_folds": missing_folds_info,
        }

    if not available_folds:
        raise FileNotFoundError(
            f"No folds have complete visual feature caches for task '{task}'. "
            f"Run 'python scripts/extract_visual_features.py --task {task} --encoder {encoder_name}' first."
        )

    # 2. Benchmark execution
    fold_results: List[Dict[str, Any]] = []

    for fold_num in available_folds:
        ckpt_path = (
            Path(checkpoints_dir) / f"video_{model_name.lower()}_{task}_fold{fold_num}.pt"
            if fold_num == 1
            else None
        )
        res = train_and_eval_fold(
            task=task,
            fold_num=fold_num,
            cfg=cfg,
            device=device,
            save_checkpoint_path=ckpt_path,
        )
        if res:
            fold_results.append(res)

    # 3. Aggregate metrics
    accs = [f["accuracy"] * 100 for f in fold_results]
    bal_accs = [f["balanced_accuracy"] * 100 for f in fold_results]
    macro_f1s = [f["macro_f1"] for f in fold_results]
    edits = [f["edit_score"] for f in fold_results]

    summary = {
        "model": model_name,
        "encoder": encoder_name,
        "feature_dim": feat_dim,
        "task": task,
        "seed": seed,
        "n_folds_evaluated": len(fold_results),
        "total_folds": len(target_folds),
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "min_accuracy": float(np.min(accs)),
        "max_accuracy": float(np.max(accs)),
        "mean_balanced_accuracy": float(np.mean(bal_accs)),
        "std_balanced_accuracy": float(np.std(bal_accs)),
        "min_balanced_accuracy": float(np.min(bal_accs)),
        "max_balanced_accuracy": float(np.max(bal_accs)),
        "mean_macro_f1": float(np.mean(macro_f1s)),
        "std_macro_f1": float(np.std(macro_f1s)),
        "min_macro_f1": float(np.min(macro_f1s)),
        "max_macro_f1": float(np.max(macro_f1s)),
        "mean_edit_score": float(np.mean(edits)),
        "std_edit_score": float(np.std(edits)),
        "folds": fold_results,
        "missing_folds_info": {k: len(v) for k, v in missing_folds_info.items()},
        "config": cfg,
    }

    print(f"\n---> {task} {model_name} ({encoder_name}) Summary ({len(fold_results)} folds):")
    print(f"Accuracy:          {summary['mean_accuracy']:.2f}% ± {summary['std_accuracy']:.2f}% (Min: {summary['min_accuracy']:.2f}%, Max: {summary['max_accuracy']:.2f}%)")
    print(f"Balanced Accuracy: {summary['mean_balanced_accuracy']:.2f}% ± {summary['std_balanced_accuracy']:.2f}% (Min: {summary['min_balanced_accuracy']:.2f}%, Max: {summary['max_balanced_accuracy']:.2f}%)")
    print(f"Macro-F1:          {summary['mean_macro_f1']:.4f} ± {summary['std_macro_f1']:.4f} (Min: {summary['min_macro_f1']:.4f}, Max: {summary['max_macro_f1']:.4f})")
    print(f"Edit Score:        {summary['mean_edit_score']:.2f}% ± {summary['std_edit_score']:.2f}%")

    # 4. Save results JSON
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"video_{model_name.lower()}_{task}_results.json"
    save_json(summary, out_file)
    print(f"Saved benchmark results to {out_file}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Video Baseline on cached features.")
    parser.add_argument("--task", type=str, default="Suturing", choices=["Suturing", "Knot_Tying", "Needle_Passing"])
    parser.add_argument("--config", type=str, default="configs/video_baseline.yaml")
    parser.add_argument("--model-type", type=str, default="VideoGRU", choices=["VideoGRU", "VideoMLP"])
    parser.add_argument("--fold", type=int, default=None, help="Evaluate a single fold (e.g. 1)")
    parser.add_argument("--dry-run", action="store_true", help="Inspect cache availability without training")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")

    args = parser.parse_args()

    # Load and override config if model-type specified
    cfg = load_config(args.config)
    if "model" not in cfg:
        cfg["model"] = {}
    cfg["model"]["name"] = args.model_type

    run_video_baseline_benchmark(
        task=args.task,
        config_path=args.config,
        output_dir=args.output_dir,
        checkpoints_dir=args.checkpoints_dir,
        specific_fold=args.fold,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
