import argparse
import sys
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

from src.data.dataset import get_fold_dataloaders
from src.data.splits import get_louo_splits
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_edit_score,
)
from src.models.kinematics_gru import KinematicsGRU
from src.utils.reproducibility import get_device, load_config, save_json, set_seed


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch over batches."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        kinematics = batch["kinematics"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        logits = model(kinematics)
        loss = criterion(logits, targets)
        loss.backward()

        # Gradient clipping for recurrent stability
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
        kinematics = batch["kinematics"].to(device)
        targets = batch["target"].cpu().numpy()

        logits = model(kinematics)
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
            # Kinematics shape: (T, 76)
            kin_arr = torch.from_numpy(record["kinematics"]).float().to(device)
            # Forward sequence through GRU
            logits_seq = model.forward_sequence(kin_arr.unsqueeze(0)).squeeze(0)  # (T, num_classes)
            preds = torch.argmax(logits_seq, dim=-1).cpu().numpy().tolist()
            true_labels = record["labels"].tolist()

            true_seqs.append(true_labels)
            pred_seqs.append(preds)

    return compute_edit_score(true_seqs, pred_seqs)


def train_and_eval_fold(
    task: str,
    fold_num: int,
    cfg: Dict[str, Any],
    device: torch.device,
    save_checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train KinematicsGRU on one fold and evaluate test performance."""
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    window_size = data_cfg.get("window_size", 16)
    train_stride = data_cfg.get("train_stride", 8)
    eval_stride = data_cfg.get("eval_stride", 8)
    batch_size = train_cfg.get("batch_size", 32)
    epochs = train_cfg.get("epochs", 15)
    lr = train_cfg.get("learning_rate", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    use_class_weights = train_cfg.get("use_class_weights", True)
    patience = train_cfg.get("patience", 4)

    # 1. Load DataLoaders
    loaders = get_fold_dataloaders(
        task=task,
        fold_num=fold_num,
        batch_size=batch_size,
        window_size=window_size,
        train_stride=train_stride,
        eval_stride=eval_stride,
        dataset_dir=data_cfg.get("dataset_dir", "dataset"),
        create_val_split=True,
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
    model = KinematicsGRU(
        input_dim=model_cfg.get("input_dim", 76),
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

    print(
        f"Fold {fold_num} (User {fold_info['test_user']}): "
        f"Acc={test_metrics['accuracy']*100:.2f}%, "
        f"BalAcc={test_metrics['balanced_accuracy']*100:.2f}%, "
        f"Macro-F1={test_metrics['macro_f1']:.4f}, "
        f"Edit={edit_score:.2f}%"
    )

    return test_metrics


def run_kinematics_gru_benchmark(
    task: str = "Suturing",
    config_path: str = "configs/kinematics_gru.yaml",
    output_dir: str = "artifacts",
    checkpoints_dir: str = "checkpoints",
) -> Dict[str, Any]:
    """Run full 8-fold LOUO benchmark for KinematicsGRU on a task."""
    cfg = load_config(config_path)
    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)
    device = get_device()

    splits = get_louo_splits(task, create_val_split=True)
    fold_results: List[Dict[str, Any]] = []

    print(f"\n==========================================")
    print(f"Benchmarking KinematicsGRU on Task: {task}")
    print(f"Device: {device}")
    print(f"==========================================")

    for fold_num in sorted(splits.keys()):
        ckpt_path = Path(checkpoints_dir) / f"kinematics_gru_{task}_fold{fold_num}.pt" if fold_num == 1 else None
        res = train_and_eval_fold(
            task=task,
            fold_num=fold_num,
            cfg=cfg,
            device=device,
            save_checkpoint_path=ckpt_path,
        )
        if res:
            fold_results.append(res)

    # Aggregate metrics
    accs = [f["accuracy"] * 100 for f in fold_results]
    bal_accs = [f["balanced_accuracy"] * 100 for f in fold_results]
    macro_f1s = [f["macro_f1"] for f in fold_results]
    edits = [f["edit_score"] for f in fold_results]

    summary = {
        "model": "KinematicsGRU",
        "task": task,
        "n_folds": len(fold_results),
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
    }

    print(f"\n---> {task} KinematicsGRU Summary:")
    print(f"Accuracy:          {summary['mean_accuracy']:.2f}% ± {summary['std_accuracy']:.2f}% (Min: {summary['min_accuracy']:.2f}%, Max: {summary['max_accuracy']:.2f}%)")
    print(f"Balanced Accuracy: {summary['mean_balanced_accuracy']:.2f}% ± {summary['std_balanced_accuracy']:.2f}% (Min: {summary['min_balanced_accuracy']:.2f}%, Max: {summary['max_balanced_accuracy']:.2f}%)")
    print(f"Macro-F1:          {summary['mean_macro_f1']:.4f} ± {summary['std_macro_f1']:.4f} (Min: {summary['min_macro_f1']:.4f}, Max: {summary['max_macro_f1']:.4f})")
    print(f"Edit Score:        {summary['mean_edit_score']:.2f}% ± {summary['std_edit_score']:.2f}%")

    # Export
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, out_dir / f"kinematics_gru_{task}_results.json")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate KinematicsGRU on JIGSAWS")
    parser.add_argument("--task", type=str, default="Suturing", help="Task name ('Suturing', 'Knot_Tying', 'Needle_Passing', or 'all')")
    parser.add_argument("--config", type=str, default="configs/kinematics_gru.yaml", help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    tasks = ["Suturing", "Knot_Tying", "Needle_Passing"] if args.task == "all" else [args.task]
    all_summaries = {}

    for t in tasks:
        summary = run_kinematics_gru_benchmark(
            task=t,
            config_path=args.config,
            output_dir=args.output_dir,
            checkpoints_dir=args.checkpoints_dir,
        )
        all_summaries[t] = summary

    # Save overall summary markdown
    md_lines = [
        "# JIGSAWS Kinematics-Only GRU Baseline Benchmark",
        "",
        "Architecture: 2-Layer GRU [76 -> 128 -> GRU(128) -> C], LayerNorm, Dropout(0.2).",
        "Evaluation: Official 8-fold Leave-One-User-Out cross-validation.",
        "",
        "| Task | Accuracy (%) | Balanced Acc (%) | Macro-F1 | Edit Score (%) |",
        "|---|---:|---:|---:|---:|",
    ]

    for t, s in all_summaries.items():
        md_lines.append(
            f"| **{t}** | {s['mean_accuracy']:.2f} ± {s['std_accuracy']:.2f} | "
            f"{s['mean_balanced_accuracy']:.2f} ± {s['std_balanced_accuracy']:.2f} | "
            f"{s['mean_macro_f1']:.4f} ± {s['std_macro_f1']:.4f} | "
            f"{s['mean_edit_score']:.2f} ± {s['std_edit_score']:.2f} |"
        )

    md_path = Path(args.output_dir) / "kinematics_gru_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\nSaved overall summary to {md_path}")


if __name__ == "__main__":
    main()
