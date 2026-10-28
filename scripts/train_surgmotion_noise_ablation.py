"""scripts/train_surgmotion_noise_ablation.py

ITER-31 Workstream A: Continuous-Noise Reliability-Gate Ablation.

Tests whether training the multimodal adaptive reliability gate with continuous Gaussian
feature noise (sigma ~ U(0.0, 1.5)) on normalized visual features enables true continuous
corruption detection and improves robustness compared to discrete binary dropout.

Supported Ablation Conditions:
1. 'continuous_noise': Binary modality dropout + continuous Gaussian noise on video features.
2. 'existing': Original ITER-28 training (binary modality dropout only; clean inputs).
3. 'clean_only': Control condition with pristine inputs (no dropout, no noise).

Features:
- Official JIGSAWS Leave-One-User-Out (LOUO) splits.
- Multi-severity robustness stress testing (clean, sigma in {0.2, 0.5, 1.0, 1.5, 2.0}, kin noise, missing video, missing kin).
- Full time-resolved gate weight logging on held-out trials across all conditions.
- Strict CPU/GPU portability with deterministic seeding.
- Fast smoke-test mode for local development validation.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import get_louo_splits
from src.data.surgmotion_dataset import JIGSAWSSurgMotionDataset, get_surgmotion_dataloaders
from src.data.transcriptions import get_gesture_vocabulary
from src.evaluation.metrics import compute_classification_metrics
from src.models.surgmotion_fusion import (
    GateTrajectoryLogger,
    SurgMotionAdaptiveFusionModel,
    inject_feature_noise,
)
from src.utils.reproducibility import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("noise_ablation")


# ==============================================================================
# 1. Edit Score Metric for Sequential Inference
# ==============================================================================

def levenshtein_distance(seq1: List[int], seq2: List[int]) -> int:
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def compress_sequence(seq: List[int]) -> List[int]:
    if not seq:
        return []
    compressed = [seq[0]]
    for x in seq[1:]:
        if x != compressed[-1]:
            compressed.append(x)
    return compressed


def compute_edit_score(true_seqs: List[List[int]], pred_seqs: List[List[int]]) -> float:
    scores = []
    for y_t, y_p in zip(true_seqs, pred_seqs):
        c_t = compress_sequence(y_t)
        c_p = compress_sequence(y_p)
        denom = max(len(c_t), len(c_p))
        if denom == 0:
            scores.append(100.0)
        else:
            dist = levenshtein_distance(c_t, c_p)
            scores.append(max(0.0, (1.0 - dist / denom) * 100.0))
    return float(np.mean(scores)) if scores else 0.0


# ==============================================================================
# 2. Training Loop with Configurable Noise Injection
# ==============================================================================

def train_ablation_epoch(
    model: SurgMotionAdaptiveFusionModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    condition: str = "continuous_noise",
    noise_min: float = 0.0,
    noise_max: float = 1.5,
    noise_prob: float = 0.5,
) -> float:
    """Train single epoch under the designated ablation condition."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        v_in = batch["features"].to(device)
        k_in = batch["kinematics"].to(device)
        targets = batch["target"].to(device)
        b_size = v_in.shape[0]

        if condition == "clean_only":
            mask = torch.ones((b_size, 2), dtype=torch.bool, device=device)
            v_in_pass = v_in
            k_in_pass = k_in

        else:
            # Binary availability dropout: 10% video missing, 10% kin missing, 80% both
            r = np.random.rand()
            if r < 0.10:
                mask = torch.tensor([[0, 1]], dtype=torch.bool, device=device).expand(b_size, 2)
                v_in_pass = torch.zeros_like(v_in)
                k_in_pass = k_in
            elif r < 0.20:
                mask = torch.tensor([[1, 0]], dtype=torch.bool, device=device).expand(b_size, 2)
                v_in_pass = v_in
                k_in_pass = torch.zeros_like(k_in)
            else:
                mask = torch.ones((b_size, 2), dtype=torch.bool, device=device)
                if condition == "continuous_noise" and np.random.rand() < noise_prob:
                    # Sample noise std uniformly from [noise_min, noise_max]
                    sigma = float(np.random.uniform(noise_min, noise_max))
                    v_in_pass = inject_feature_noise(v_in, noise_std=sigma)
                else:
                    v_in_pass = v_in
                k_in_pass = k_in

        optimizer.zero_grad()
        logits, _ = model(v_in_pass, k_in_pass, availability_mask=mask)
        loss = criterion(logits, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


# ==============================================================================
# 3. Evaluation & Gate Trajectory Logging
# ==============================================================================

@torch.no_grad()
def eval_ablation_loader(
    model: SurgMotionAdaptiveFusionModel,
    loader: DataLoader,
    num_classes: int,
    class_names: Dict[int, str],
    device: torch.device,
    availability_mask: Optional[torch.Tensor] = None,
    corrupt_video_std: float = 0.0,
    corrupt_kin_std: float = 0.0,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, List[float], List[float]]:
    """Evaluate model on windowed batches and collect batch-level average gate weights."""
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    v_alphas: List[float] = []
    k_alphas: List[float] = []

    for batch in loader:
        v_in = batch["features"].to(device).clone()
        k_in = batch["kinematics"].to(device).clone()
        targets = batch["target"].cpu().numpy()
        b_size = v_in.shape[0]

        mask = availability_mask
        if mask is not None:
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(b_size, 2).to(device)
            else:
                mask = mask.to(device)

        if mask is not None:
            if not mask[0, 0]:
                v_in = torch.zeros_like(v_in)
            if not mask[0, 1]:
                k_in = torch.zeros_like(k_in)

        if corrupt_video_std > 0.0:
            v_in = inject_feature_noise(v_in, noise_std=corrupt_video_std)
        if corrupt_kin_std > 0.0:
            k_in = inject_feature_noise(k_in, noise_std=corrupt_kin_std)

        logits, alpha = model(v_in, k_in, availability_mask=mask)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())
        v_alphas.extend(alpha[:, 0].cpu().numpy().tolist())
        k_alphas.extend(alpha[:, 1].cpu().numpy().tolist())

    y_true = np.array(all_targets, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)

    metrics = compute_classification_metrics(
        y_true=y_true, y_pred=y_pred, num_classes=num_classes, class_names=class_names
    )
    return metrics, y_true, y_pred, v_alphas, k_alphas


def compute_ablation_trial_trajectories(
    model: SurgMotionAdaptiveFusionModel,
    dataset: JIGSAWSSurgMotionDataset,
    fold: int,
    task: str,
    device: torch.device,
    condition_name: str,
    noise_std: float,
    availability_mask: Optional[torch.Tensor] = None,
    corrupt_video_std: float = 0.0,
    corrupt_kin_std: float = 0.0,
    logger_obj: Optional[GateTrajectoryLogger] = None,
) -> float:
    """Evaluate sequential inference on full trials, compute Edit Score, and log gate trajectories."""
    model.eval()
    true_seqs: List[List[int]] = []
    pred_seqs: List[List[int]] = []

    with torch.no_grad():
        for rec in dataset.trial_records:
            v_t = torch.from_numpy(rec["features"]).float().to(device).unsqueeze(0)
            k_t = torch.from_numpy(rec["kinematics"]).float().to(device).unsqueeze(0)

            mask = availability_mask
            if mask is not None:
                mask = mask.unsqueeze(0).expand(1, 2).to(device)
                if not mask[0, 0]:
                    v_t = torch.zeros_like(v_t)
                if not mask[0, 1]:
                    k_t = torch.zeros_like(k_t)

            if corrupt_video_std > 0.0:
                v_t = inject_feature_noise(v_t, noise_std=corrupt_video_std)
            if corrupt_kin_std > 0.0:
                k_t = inject_feature_noise(k_t, noise_std=corrupt_kin_std)

            logits_seq, alpha_seq = model.forward_sequence(
                v_t, k_t, availability_mask=mask, return_alpha=True
            )
            preds = torch.argmax(logits_seq.squeeze(0), dim=-1).cpu().numpy()
            alphas = alpha_seq.squeeze(0).cpu().numpy()  # [T, 2]

            gt_labels = rec["labels"]
            true_seqs.append(gt_labels.tolist())
            pred_seqs.append(preds.tolist())

            if logger_obj is not None:
                logger_obj.log_trial_trajectory(
                    fold=fold,
                    task=task,
                    trial_id=rec["trial_id"],
                    condition=condition_name,
                    noise_std=noise_std,
                    alpha_weights=alphas,
                    center_frames=rec["center_frames"],
                    ground_truth=gt_labels,
                    predictions=preds,
                )

    return compute_edit_score(true_seqs, pred_seqs)


# ==============================================================================
# 4. Cross-Validation Runner for Ablation
# ==============================================================================

def run_task_ablation(
    task: str,
    condition: str = "continuous_noise",
    sequence_length: int = 8,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    noise_min: float = 0.0,
    noise_max: float = 1.5,
    noise_prob: float = 0.5,
    test_noise_levels: Tuple[float, ...] = (0.2, 0.5, 1.0, 1.5, 2.0),
    evaluate_robustness: bool = True,
    smoke_test: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run full Leave-One-User-Out cross-validation for the designated task and ablation condition."""
    logger.info("=" * 70)
    logger.info(f"STARTING ABLATION: Task={task} | Condition={condition} | Device={device}")
    logger.info(f"Hyperparameters: seq_len={sequence_length}, epochs={epochs}, batch={batch_size}, lr={lr}")
    logger.info(f"Noise params: [{noise_min}, {noise_max}], p={noise_prob} | Smoke test={smoke_test}")
    logger.info("=" * 70)

    vocab = get_gesture_vocabulary(task=task)
    num_classes = len(vocab)
    class_names = {idx: name for name, idx in vocab.items()}

    splits_dict = get_louo_splits(task=task, create_val_split=True)
    all_folds = sorted(list(splits_dict.keys()))
    if smoke_test:
        all_folds = [all_folds[0]]  # Run only fold 1 for smoke test
        epochs = min(epochs, 1)

    fold_results: List[Dict[str, Any]] = []
    robustness_results: Dict[str, List[Dict[str, Any]]] = {}
    trajectory_logger = GateTrajectoryLogger()

    # Pre-configure robustness test conditions
    eval_conditions: List[Tuple[str, float, Dict[str, Any]]] = [
        ("clean", 0.0, {}),
    ]
    if evaluate_robustness:
        for lvl in test_noise_levels:
            eval_conditions.append((f"video_noise_{lvl}", lvl, {"corrupt_video_std": lvl}))
        eval_conditions.append(("kinematics_corrupted", 1.0, {"corrupt_kin_std": 1.0}))
        eval_conditions.append(("missing_video", 0.0, {"availability_mask": torch.tensor([0, 1])}))
        eval_conditions.append(("missing_kinematics", 0.0, {"availability_mask": torch.tensor([1, 0])}))

    for cond_name, _, _ in eval_conditions:
        robustness_results[cond_name] = []

    for fold_num in all_folds:
        f_info = splits_dict[fold_num]
        if not f_info["test_trials"]:
            logger.info(f"Fold {fold_num} has 0 test trials (Subject {f_info['test_user']}). Skipping.")
            continue

        logger.info(f"\n--- Fold {fold_num}/{len(splits_dict)} (Held-out User: {f_info['test_user']}) ---")

        train_loader, val_loader, test_loader, _, fold_info = get_surgmotion_dataloaders(
            task=task,
            fold=fold_num,
            sequence_length=sequence_length,
            batch_size=batch_size,
        )

        # Compute balanced class weights
        train_targets = []
        for b in train_loader:
            train_targets.extend(b["target"].numpy().tolist())
        counts = np.bincount(train_targets, minlength=num_classes)
        weights = np.where(counts > 0, len(train_targets) / (num_classes * counts), 1.0)
        class_weights_t = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)

        model = SurgMotionAdaptiveFusionModel(
            num_classes=num_classes,
            video_dim=1024,
            kinematics_dim=76,
            hidden_dim=128,
            dropout=0.2,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )

        best_val_f1 = -1.0
        best_state = None
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            train_loss = train_ablation_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                condition=condition,
                noise_min=noise_min,
                noise_max=noise_max,
                noise_prob=noise_prob,
            )

            val_m, _, _, _, _ = eval_ablation_loader(
                model=model,
                loader=val_loader,
                num_classes=num_classes,
                class_names=class_names,
                device=device,
            )

            val_f1 = val_m["macro_f1"]
            scheduler.step(val_f1)

            if val_f1 > best_val_f1 or best_state is None:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch

        if best_state is not None:
            model.load_state_dict(best_state)

        # 1. Clean Test Evaluation
        test_m, y_true, y_pred, v_al, k_al = eval_ablation_loader(
            model=model,
            loader=test_loader,
            num_classes=num_classes,
            class_names=class_names,
            device=device,
        )

        clean_edit = compute_ablation_trial_trajectories(
            model=model,
            dataset=test_loader.dataset,
            fold=fold_num,
            task=task,
            device=device,
            condition_name="clean",
            noise_std=0.0,
            logger_obj=trajectory_logger,
        )

        fold_res = {
            "fold": fold_num,
            "task": task,
            "test_user": fold_info["test_user"],
            "best_epoch": best_epoch,
            "accuracy": round(float(test_m["accuracy"]) * 100.0, 2),
            "balanced_accuracy": round(float(test_m["balanced_accuracy"]) * 100.0, 2),
            "macro_f1": round(float(test_m["macro_f1"]) * 100.0, 2),
            "edit_score": round(float(clean_edit), 2),
            "mean_alpha_video": round(float(np.mean(v_al)), 4),
            "mean_alpha_kin": round(float(np.mean(k_al)), 4),
            "n_test_trials": len(fold_info["test_trials"]),
        }
        fold_results.append(fold_res)
        logger.info(
            f"Fold {fold_num} Clean: Acc={fold_res['accuracy']:.2f}%, "
            f"BalAcc={fold_res['balanced_accuracy']:.2f}%, F1={fold_res['macro_f1']:.2f}%, "
            f"Edit={fold_res['edit_score']:.2f}%, alpha_v={fold_res['mean_alpha_video']:.3f}"
        )

        # 2. Robustness Stress Testing across designated conditions
        for cond_name, noise_s, cond_kwargs in eval_conditions:
            cm, _, _, cv_al, ck_al = eval_ablation_loader(
                model=model,
                loader=test_loader,
                num_classes=num_classes,
                class_names=class_names,
                device=device,
                **cond_kwargs,
            )
            cedit = compute_ablation_trial_trajectories(
                model=model,
                dataset=test_loader.dataset,
                fold=fold_num,
                task=task,
                device=device,
                condition_name=cond_name,
                noise_std=noise_s,
                logger_obj=trajectory_logger if cond_name != "clean" else None,
                **cond_kwargs,
            )
            robustness_results[cond_name].append({
                "fold": fold_num,
                "accuracy": round(float(cm["accuracy"]) * 100.0, 2),
                "balanced_accuracy": round(float(cm["balanced_accuracy"]) * 100.0, 2),
                "macro_f1": round(float(cm["macro_f1"]) * 100.0, 2),
                "edit_score": round(float(cedit), 2),
                "mean_alpha_video": round(float(np.mean(cv_al)), 4),
                "mean_alpha_kin": round(float(np.mean(ck_al)), 4),
            })

    # Summary Aggregates
    accs = [r["accuracy"] for r in fold_results]
    bal_accs = [r["balanced_accuracy"] for r in fold_results]
    f1s = [r["macro_f1"] for r in fold_results]
    edits = [r["edit_score"] for r in fold_results]

    robustness_summary: Dict[str, Any] = {}
    for cond_name, records in robustness_results.items():
        c_accs = [r["accuracy"] for r in records]
        c_f1s = [r["macro_f1"] for r in records]
        c_edits = [r["edit_score"] for r in records]
        c_alphas = [r["mean_alpha_video"] for r in records]
        robustness_summary[cond_name] = {
            "accuracy_mean": round(float(np.mean(c_accs)), 2),
            "accuracy_std": round(float(np.std(c_accs)), 2),
            "macro_f1_mean": round(float(np.mean(c_f1s)), 2),
            "macro_f1_std": round(float(np.std(c_f1s)), 2),
            "edit_score_mean": round(float(np.mean(c_edits)), 2),
            "edit_score_std": round(float(np.std(c_edits)), 2),
            "alpha_video_mean": round(float(np.mean(c_alphas)), 4),
            "delta_acc_from_clean": round(float(np.mean(c_accs) - np.mean(accs)), 2),
            "delta_f1_from_clean": round(float(np.mean(c_f1s) - np.mean(f1s)), 2),
        }

    summary = {
        "task": task,
        "condition": condition,
        "n_active_folds": len(fold_results),
        "noise_range": [noise_min, noise_max],
        "noise_prob": noise_prob,
        "clean_summary": {
            "accuracy_mean": round(float(np.mean(accs)), 2),
            "accuracy_std": round(float(np.std(accs)), 2),
            "balanced_accuracy_mean": round(float(np.mean(bal_accs)), 2),
            "balanced_accuracy_std": round(float(np.std(bal_accs)), 2),
            "macro_f1_mean": round(float(np.mean(f1s)), 2),
            "macro_f1_std": round(float(np.std(f1s)), 2),
            "edit_score_mean": round(float(np.mean(edits)), 2),
            "edit_score_std": round(float(np.std(edits)), 2),
        },
        "robustness_summary": robustness_summary,
        "fold_results": fold_results,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"surgmotion_ablation_{condition}_{task}_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved results summary to {out_file}")

        # Save trajectory records
        traj_file = output_dir / f"gate_trajectories_{condition}_{task}.json"
        trajectory_logger.save_json(traj_file)
        logger.info(f"Saved {len(trajectory_logger.records)} gate trajectory records to {traj_file}")

    return summary


# ==============================================================================
# 5. Preflight Environment Check
# ==============================================================================

def run_preflight_check() -> bool:
    """Validate GPU availability, directory structure, caches, and dependencies."""
    logger.info("=" * 60)
    logger.info("PREFLIGHT CHECK — ITER-31 NOISE ABLATION RUNNER")
    logger.info("=" * 60)

    # 1. Python & Torch
    logger.info(f"Python version: {sys.version.split()[0]}")
    logger.info(f"PyTorch version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    logger.info(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        logger.warning("CUDA not detected. Running on CPU.")

    # 2. Check JIGSAWS directory
    dataset_dir = PROJECT_ROOT / "dataset"
    if not dataset_dir.is_dir():
        logger.error(f"Dataset directory missing: {dataset_dir}")
        return False
    logger.info(f"Dataset directory found: {dataset_dir}")

    # 3. Check SurgMotion feature cache directory
    cache_dir = PROJECT_ROOT / "data" / "cache" / "features_surgmotion"
    if not cache_dir.is_dir():
        logger.error(f"SurgMotion cache directory missing: {cache_dir}")
        return False

    npz_files = list(cache_dir.glob("*.npz"))
    logger.info(f"SurgMotion feature caches found: {len(npz_files)} .npz files")
    if len(npz_files) < 100:
        logger.warning(f"Expected 103 caches, found {len(npz_files)}. Some trials may be missing.")

    logger.info("Preflight check SUCCESSFUL. Ready for execution.\n")
    return True


# ==============================================================================
# 6. Main CLI Entry Point
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="ITER-31 Continuous-Noise Reliability-Gate Ablation")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "Suturing", "Knot_Tying", "Needle_Passing"],
        help="JIGSAWS task name or 'all'",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="continuous_noise",
        choices=["continuous_noise", "existing", "clean_only"],
        help="Training ablation condition",
    )
    parser.add_argument("--noise-min", type=float, default=0.0, help="Min noise std for continuous training")
    parser.add_argument("--noise-max", type=float, default=1.5, help="Max noise std for continuous training")
    parser.add_argument("--noise-prob", type=float, default=0.5, help="Probability of injecting noise into batch")
    parser.add_argument("--sequence-length", type=int, default=8, help="Window sequence length (default 8)")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs (default 25)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default 1e-3)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', or 'auto')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/iter31_noise_ablation",
        help="Directory to save result JSONs and trajectory logs",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 fold, 1 epoch for rapid validation")
    parser.add_argument("--preflight-only", action="store_true", help="Run preflight check and exit")

    args = parser.parse_args()

    if args.preflight_only:
        success = run_preflight_check()
        sys.exit(0 if success else 1)

    set_seed(args.seed)

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)

    tasks = ["Suturing", "Knot_Tying", "Needle_Passing"] if args.task == "all" else [args.task]
    out_dir = PROJECT_ROOT / args.output_dir

    for t in tasks:
        run_task_ablation(
            task=t,
            condition=args.condition,
            sequence_length=args.sequence_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=dev,
            noise_min=args.noise_min,
            noise_max=args.noise_max,
            noise_prob=args.noise_prob,
            evaluate_robustness=True,
            smoke_test=args.smoke_test,
            output_dir=out_dir,
        )

    logger.info("\nALL ABLATION RUNS COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
