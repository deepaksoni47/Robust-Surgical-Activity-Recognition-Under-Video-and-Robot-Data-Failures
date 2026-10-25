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

from src.data.feature_cache import get_cache_path
from src.data.multimodal_dataset import (
    get_multimodal_fold_dataloaders,
    get_multimodal_fold_datasets,
)
from src.data.splits import get_louo_splits
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_edit_score,
)
from src.models.concat_fusion import ConcatFusionGRU, ConcatFusionMLP
from src.utils.reproducibility import get_device, load_config, save_json, set_seed


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch over batches of synchronized multimodal feature windows."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        video_feats = batch["video"].to(device)
        kinematics = batch["kinematics"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        logits = model(video_feats, kinematics)
        loss = criterion(logits, targets)
        loss.backward()

        # Gradient clipping for recurrent and projection trunk stability
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
    """Evaluate multimodal model on DataLoader and compute full classification metrics."""
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []

    for batch in dataloader:
        video_feats = batch["video"].to(device)
        kinematics = batch["kinematics"].to(device)
        targets = batch["target"].cpu().numpy()

        logits = model(video_feats, kinematics)
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


def compute_multimodal_trial_edit_scores(
    model: nn.Module,
    test_dataset: Any,
    device: torch.device,
) -> float:
    """Compute Segmental Edit Score (%) across full test trial sequences."""
    model.eval()
    true_seqs: List[List[int]] = []
    pred_seqs: List[List[int]] = []

    with torch.no_grad():
        for record in test_dataset.trial_records:
            v_arr = torch.from_numpy(record["video"]).float().to(device).unsqueeze(0)  # (1, T, D_v)
            k_arr = torch.from_numpy(record["kinematics"]).float().to(device).unsqueeze(0)  # (1, T, 76)

            logits_seq = model.forward_sequence(v_arr, k_arr).squeeze(0)  # (T, num_classes)
            preds = torch.argmax(logits_seq, dim=-1).cpu().numpy().tolist()
            true_labels = record["labels"].tolist()

            true_seqs.append(true_labels)
            pred_seqs.append(preds)

    return compute_edit_score(true_seqs, pred_seqs)


def check_concat_fusion_readiness(
    task: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Audit readiness: check feature cache presence and kinematics file existence."""
    splits = get_louo_splits(task, create_val_split=True)
    cache_dir = Path(cfg.get("data", {}).get("cache_dir", "data/cache/features"))
    dataset_dir = Path(cfg.get("data", {}).get("dataset_dir", "dataset"))
    encoder_name = cfg.get("data", {}).get("encoder_name", "ResNet18")
    camera = cfg.get("data", {}).get("camera", "capture1")

    fold_status: Dict[int, Dict[str, Any]] = {}
    ready_folds: List[int] = []

    for fold_num in sorted(splits.keys()):
        fold_info = splits[fold_num]
        all_trials = fold_info["train_trials"] + fold_info.get("val_trials", []) + fold_info["test_trials"]

        missing_caches: List[str] = []
        missing_kinematics: List[str] = []

        for t_id in all_trials:
            c_path = get_cache_path(cache_dir, t_id, encoder_name, camera=camera)
            if not c_path.is_file():
                missing_caches.append(t_id)

            k_path = dataset_dir / task / "kinematics" / "AllGestures" / f"{t_id}.txt"
            if not k_path.is_file():
                missing_kinematics.append(t_id)

        is_ready = (len(missing_caches) == 0 and len(missing_kinematics) == 0)
        if is_ready:
            ready_folds.append(fold_num)

        fold_status[fold_num] = {
            "test_user": fold_info["test_user"],
            "ready": is_ready,
            "missing_cache_trials": missing_caches,
            "missing_cache_count": len(missing_caches),
            "missing_kinematics_count": len(missing_kinematics),
        }

    return {
        "task": task,
        "total_folds": len(splits),
        "ready_folds": ready_folds,
        "fold_status": fold_status,
        "all_ready": (len(ready_folds) == len(splits)),
    }


def train_and_eval_fold(
    task: str,
    fold_num: int,
    cfg: Dict[str, Any],
    device: torch.device,
    save_checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train ConcatFusionGRU on one fold and evaluate test performance."""
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    model_name = model_cfg.get("name", "ConcatFusionGRU")
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
    loaders = get_multimodal_fold_dataloaders(
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
    if model_name == "ConcatFusionMLP":
        model = ConcatFusionMLP(
            video_dim=model_cfg.get("video_dim", 512),
            kinematics_dim=model_cfg.get("kinematics_dim", 76),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_classes=num_classes,
            dropout=model_cfg.get("dropout", 0.2),
        ).to(device)
    else:
        model = ConcatFusionGRU(
            video_dim=model_cfg.get("video_dim", 512),
            kinematics_dim=model_cfg.get("kinematics_dim", 76),
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
    edit_score = compute_multimodal_trial_edit_scores(model, loaders["datasets"]["test"], device)

    test_metrics["edit_score"] = edit_score
    test_metrics["fold"] = fold_num
    test_metrics["test_user"] = fold_info["test_user"]
    test_metrics["encoder"] = encoder_name
    test_metrics["video_dim"] = model_cfg.get("video_dim", 512)
    test_metrics["kinematics_dim"] = model_cfg.get("kinematics_dim", 76)

    print(
        f"Fold {fold_num} (User {fold_info['test_user']}): "
        f"Acc={test_metrics['accuracy']*100:.2f}%, "
        f"BalAcc={test_metrics['balanced_accuracy']*100:.2f}%, "
        f"Macro-F1={test_metrics['macro_f1']:.4f}, "
        f"Edit={edit_score:.2f}%"
    )

    return test_metrics


def run_concat_fusion_benchmark(
    task: str = "Suturing",
    config_path: str = "configs/concat_fusion.yaml",
    output_dir: str = "artifacts",
    checkpoints_dir: str = "checkpoints",
    specific_fold: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run full LOUO benchmark for early/intermediate feature concatenation fusion."""
    cfg = load_config(config_path)
    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)
    device = get_device()

    model_name = cfg.get("model", {}).get("name", "ConcatFusionGRU")
    encoder_name = cfg.get("data", {}).get("encoder_name", "ResNet18")
    v_dim = cfg.get("model", {}).get("video_dim", 512)
    k_dim = cfg.get("model", {}).get("kinematics_dim", 76)

    print(f"\n==========================================")
    print(f"Benchmarking {model_name} [Video ({v_dim}-D) + Kin ({k_dim}-D)] on Task: {task}")
    print(f"Device: {device}")
    print(f"==========================================")

    # 1. Audit readiness
    audit = check_concat_fusion_readiness(task, cfg)
    splits = get_louo_splits(task, create_val_split=True)
    target_folds = [specific_fold] if specific_fold is not None else sorted(splits.keys())

    ready_target_folds = [f for f in target_folds if f in audit["ready_folds"]]
    missing_folds = [f for f in target_folds if f not in audit["ready_folds"]]

    print(f"Readiness: {len(ready_target_folds)}/{len(target_folds)} folds ready.")
    if missing_folds:
        print(f"Folds lacking prerequisites: {missing_folds}")
        for mf in missing_folds:
            st = audit["fold_status"][mf]
            print(
                f"  - Fold {mf} (User {st['test_user']}): "
                f"Missing Video Caches={st['missing_cache_count']} trials"
            )

    if dry_run or not ready_target_folds:
        print(f"\n[AUDIT STATUS] REAL CONCATENATION FUSION BENCHMARK STATUS:")
        if not ready_target_folds:
            print(
                "Real multimodal concatenation evaluation cannot proceed because prerequisite "
                "video feature caches are not yet extracted for all trials in this task.\n"
                "To extract all required features:\n"
                f"    python scripts/extract_visual_features.py --task {task} --encoder {encoder_name}\n"
                f"Then re-run:\n"
                f"    python scripts/train_concat_fusion.py --task {task}\n"
            )
        return {
            "status": "BLOCKED" if not ready_target_folds else "DRY_RUN",
            "task": task,
            "readiness": audit,
        }

    # 2. Run benchmark on available folds
    fold_results: List[Dict[str, Any]] = []

    for fold_num in ready_target_folds:
        ckpt_path = (
            Path(checkpoints_dir) / f"concat_fusion_{task}_fold{fold_num}.pt"
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
        "method": "FeatureConcatenation",
        "encoder": encoder_name,
        "video_dim": v_dim,
        "kinematics_dim": k_dim,
        "input_dim": v_dim + k_dim,
        "task": task,
        "n_folds_evaluated": len(fold_results),
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_balanced_accuracy": float(np.mean(bal_accs)),
        "std_balanced_accuracy": float(np.std(bal_accs)),
        "mean_macro_f1": float(np.mean(macro_f1s)),
        "std_macro_f1": float(np.std(macro_f1s)),
        "mean_edit_score": float(np.mean(edits)),
        "std_edit_score": float(np.std(edits)),
        "fold_results": fold_results,
        "config": cfg,
    }

    out_file = Path(output_dir) / f"concat_fusion_{task}_results.json"
    save_json(summary, out_file)
    print(f"\nSaved concatenation fusion results to {out_file}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate feature concatenation fusion on JIGSAWS.")
    parser.add_argument("--task", type=str, default="Suturing", choices=["Suturing", "Knot_Tying", "Needle_Passing"])
    parser.add_argument("--config", type=str, default="configs/concat_fusion.yaml")
    parser.add_argument("--model-type", type=str, default="ConcatFusionGRU", choices=["ConcatFusionGRU", "ConcatFusionMLP"])
    parser.add_argument("--fold", type=int, default=None, help="Evaluate a specific fold")
    parser.add_argument("--dry-run", action="store_true", help="Audit readiness without training")
    parser.add_argument("--check-readiness", action="store_true", help="Audit caches and files")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")

    args = parser.parse_args()

    cfg = load_config(args.config)
    if "model" not in cfg:
        cfg["model"] = {}
    cfg["model"]["name"] = args.model_type

    run_concat_fusion_benchmark(
        task=args.task,
        config_path=args.config,
        output_dir=args.output_dir,
        checkpoints_dir=args.checkpoints_dir,
        specific_fold=args.fold,
        dry_run=args.dry_run or args.check_readiness,
    )


if __name__ == "__main__":
    main()
