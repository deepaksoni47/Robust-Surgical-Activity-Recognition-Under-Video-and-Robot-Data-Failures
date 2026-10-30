"""scripts/run_iter36_rare_gesture_audit.py

Single-Command Orchestrator for ITER-36 Rare-Gesture Prediction Audit on Mentor GPU.
Performs:
1. Windows sleep/hibernation prevention during execution.
2. Checkpoint detection for all 4 Phase 1 Suturing models (ce, class_weighted, focal, cb_focal).
   - Fast checkpoint inference mode: loads existing trained models in ~15-30s.
   - Fallback training mode: retrains on demand if any checkpoint is missing.
3. Detailed per-frame and per-trial prediction generation:
   - Serializes y_true, y_pred, and trial-level sequences to artifacts/predictions/.
   - Computes 11x11 confusion matrices and per-class Precision, Recall, and F1 across all 8 folds.
4. In-depth Rare-Gesture (G9, G10) vs Dominant Gesture (G2, G3) audit:
   - Evaluates whether class re-weighting improves rare-class recall or induces precision collapse.
   - Outputs unified artifact: artifacts/iter36_rare_gesture_audit.json.
5. Publication figure generation:
   - paper/figures/fig_iter36_per_class_f1.png / .pdf
   - paper/figures/fig_iter36_confusion_matrices.png / .pdf
   - paper/figures/fig_iter36_rare_gesture_recall.png / .pdf
6. Optional exploration flag (--explore-loss) for additional focal loss parameter testing.
7. Automated Git stage, commit, and push to GitHub origin main.

Usage:
    # 1. Standard fast audit from existing GPU checkpoints:
    python scripts/run_iter36_rare_gesture_audit.py

    # 2. Smoke-test mode:
    python scripts/run_iter36_rare_gesture_audit.py --smoke-test --no-push

    # 3. Retrain any missing checkpoints:
    python scripts/run_iter36_rare_gesture_audit.py --retrain-missing
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
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_gesture_vocabulary
from src.data.surgmotion_dataset import get_surgmotion_dataloaders
from src.models.surgmotion_fusion import SurgMotionAdaptiveFusionModel
from src.evaluation.metrics import compute_classification_metrics, compute_edit_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rare_gesture_audit")

FIGURES_DIR = PROJECT_ROOT / "paper/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PRED_DIR = PROJECT_ROOT / "artifacts/predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "ce": "#2b5c8f",            # Steel Blue
    "class_weighted": "#2e7d32", # Emerald Green
    "focal": "#d97706",         # Warm Amber
    "cb_focal": "#c0392b",      # Crimson
}

COND_LABELS = {
    "ce": "Standard CE",
    "class_weighted": "Class-Weighted CE",
    "focal": r"Focal ($\gamma=2.0$)",
    "cb_focal": r"CB-Focal ($\gamma=2, \beta=0.9999$)",
}


def prevent_windows_sleep(enable: bool = True) -> None:
    """Prevent or restore Windows sleep and display idle during execution."""
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


def load_or_train_fold_model(
    cond: str,
    fold: int,
    device: torch.device,
    retrain_missing: bool = False,
    epochs: int = 25,
) -> SurgMotionAdaptiveFusionModel:
    """Load model from checkpoint if available, or train it if requested."""
    vocab = get_gesture_vocabulary("Suturing", include_background=True)
    num_classes = len(vocab)

    model = SurgMotionAdaptiveFusionModel(
        num_classes=num_classes,
        video_dim=1024,
        kinematics_dim=76,
        hidden_dim=128,
        dropout=0.2,
    ).to(device)

    loss_suffix = f"_{cond}" if cond != "class_weighted" else ""
    ckpt_dir = PROJECT_ROOT / f"checkpoints/surgmotion_fusion_adaptive_Suturing{loss_suffix}"
    ckpt_path = ckpt_dir / f"surgmotion_adaptive_Suturing_fold{fold}.pt"

    if ckpt_path.is_file():
        logger.info(f"Loading checkpoint for {cond} fold {fold} from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        return model

    if not retrain_missing:
        raise FileNotFoundError(
            f"Checkpoint for {cond} fold {fold} not found at {ckpt_path}. "
            "Run with --retrain-missing to train missing models on the fly."
        )

    logger.warning(f"Checkpoint missing for {cond} fold {fold}. Retraining for {epochs} epochs...")
    from scripts.train_surgmotion_multimodal_fusion import train_adaptive_epoch
    from src.models.losses import get_loss_function

    train_loader, val_loader, _, _, _ = get_surgmotion_dataloaders(
        task="Suturing", fold=fold, sequence_length=8, batch_size=32
    )

    train_targets = []
    for b in train_loader:
        train_targets.extend(b["target"].numpy().tolist())
    counts = np.bincount(train_targets, minlength=num_classes)

    gamma = 2.0
    beta = 0.9999
    crit = get_loss_function(
        loss_type=cond,
        class_counts=counts,
        num_classes=num_classes,
        gamma=gamma,
        beta=beta,
        device=device,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        tr_loss = train_adaptive_epoch(model, train_loader, crit, opt, device)
        if tr_loss < best_loss:
            best_loss = tr_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "fold": fold, "task": "Suturing"}, ckpt_path)
    logger.info(f"Saved freshly trained checkpoint to: {ckpt_path}")
    model.eval()
    return model


def evaluate_and_export_predictions(
    cond: str,
    device: torch.device,
    folds: List[int],
    retrain_missing: bool = False,
    epochs: int = 25,
) -> Dict[str, Any]:
    """Evaluate model across folds, export predictions, and compute per-class metrics."""
    vocab = get_gesture_vocabulary("Suturing", include_background=True)
    num_classes = len(vocab)
    class_names = {idx: g for g, idx in vocab.items()}

    all_y_true = []
    all_y_pred = []
    fold_reports = []

    for f in folds:
        _, _, test_loader, _, fold_info = get_surgmotion_dataloaders(
            task="Suturing", fold=f, sequence_length=8, batch_size=32
        )

        model = load_or_train_fold_model(
            cond=cond, fold=f, device=device, retrain_missing=retrain_missing, epochs=epochs
        )

        # 1. Window-level frame predictions
        fold_preds = []
        fold_targets = []
        with torch.no_grad():
            for batch in test_loader:
                v = batch["features"].to(device)
                k = batch["kinematics"].to(device)
                logits, _ = model(v, k)
                preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
                fold_preds.extend(preds)
                fold_targets.extend(batch["target"].numpy().tolist())

        y_true_f = np.array(fold_targets, dtype=np.int64)
        y_pred_f = np.array(fold_preds, dtype=np.int64)

        all_y_true.extend(fold_targets)
        all_y_pred.extend(fold_preds)

        # 2. Trial-level predictions for segmental edit score
        trial_preds = {}
        true_seqs, pred_seqs = [], []
        with torch.no_grad():
            for rec in test_loader.dataset.trial_records:
                v_t = torch.from_numpy(rec["features"]).float().to(device).unsqueeze(0)
                k_t = torch.from_numpy(rec["kinematics"]).float().to(device).unsqueeze(0)
                fused_seq = model.forward_sequence(v_t, k_t).squeeze(0)
                preds_seq = torch.argmax(fused_seq, dim=-1).cpu().numpy().tolist()
                t_name = rec["trial_id"]
                trial_preds[t_name] = {
                    "y_true": rec["labels"].tolist(),
                    "y_pred": preds_seq,
                }
                true_seqs.append(rec["labels"].tolist())
                pred_seqs.append(preds_seq)

        edit_score = compute_edit_score(true_seqs, pred_seqs)

        # 3. Per-class metrics for this fold
        p_c, r_c, f_c, sup_c = precision_recall_fscore_support(
            y_true_f, y_pred_f, labels=list(range(num_classes)), zero_division=0
        )
        conf_mat_f = confusion_matrix(
            y_true_f, y_pred_f, labels=list(range(num_classes))
        ).tolist()

        fold_report = {
            "fold": f,
            "test_user": fold_info["test_user"],
            "edit_score": round(float(edit_score), 2),
            "per_class": {
                class_names[idx]: {
                    "precision": round(float(p_c[idx]) * 100.0, 2),
                    "recall": round(float(r_c[idx]) * 100.0, 2),
                    "f1": round(float(f_c[idx]) * 100.0, 2),
                    "support": int(sup_c[idx]),
                }
                for idx in range(num_classes)
            },
            "confusion_matrix": conf_mat_f,
        }
        fold_reports.append(fold_report)

        # Save individual fold prediction JSON
        pred_file = PRED_DIR / f"preds_adaptive_{cond}_Suturing_fold{f}.json"
        with open(pred_file, "w", encoding="utf-8") as pf:
            json.dump({
                "cond": cond,
                "fold": f,
                "user": fold_info["test_user"],
                "y_true": fold_targets,
                "y_pred": fold_preds,
                "trial_predictions": trial_preds,
                "per_class": fold_report["per_class"],
                "confusion_matrix": conf_mat_f,
            }, pf, indent=2)
        logger.info(f"Saved predictions for {cond} fold {f} to: {pred_file}")

    # Global aggregate across all evaluated folds
    y_true_all = np.array(all_y_true, dtype=np.int64)
    y_pred_all = np.array(all_y_pred, dtype=np.int64)

    p_glob, r_glob, f_glob, sup_glob = precision_recall_fscore_support(
        y_true_all, y_pred_all, labels=list(range(num_classes)), zero_division=0
    )
    conf_mat_glob = confusion_matrix(
        y_true_all, y_pred_all, labels=list(range(num_classes))
    ).tolist()

    return {
        "cond": cond,
        "n_folds": len(folds),
        "global_per_class": {
            class_names[idx]: {
                "precision": round(float(p_glob[idx]) * 100.0, 2),
                "recall": round(float(r_glob[idx]) * 100.0, 2),
                "f1": round(float(f_glob[idx]) * 100.0, 2),
                "support": int(sup_glob[idx]),
            }
            for idx in range(num_classes)
        },
        "global_confusion_matrix": conf_mat_glob,
        "fold_reports": fold_reports,
    }


def plot_rare_gesture_audit_figures(audit_data: Dict[str, Any]) -> None:
    """Generate per-class F1, confusion matrices, and rare-gesture recall figures."""
    vocab = get_gesture_vocabulary("Suturing", include_background=True)
    class_names = [g for g, _ in sorted(vocab.items(), key=lambda x: x[1])]
    conds = ["ce", "class_weighted", "focal", "cb_focal"]

    # 1. Figure 1: Per-Class F1 Score Comparison
    fig, ax = plt.subplots(figsize=(14, 5.2))
    x = np.arange(len(class_names))
    width = 0.20

    for i, c in enumerate(conds):
        if c not in audit_data["conditions"]:
            continue
        f1_vals = [
            audit_data["conditions"][c]["global_per_class"][g]["f1"]
            for g in class_names
        ]
        ax.bar(
            x + (i - 1.5) * width, f1_vals, width=width,
            label=COND_LABELS[c], color=PALETTE[c], edgecolor="black", linewidth=0.6
        )

    # Highlight rare gestures G9 and G10
    g9_idx = class_names.index("G9") if "G9" in class_names else -1
    g10_idx = class_names.index("G10") if "G10" in class_names else -1
    if g9_idx >= 0:
        ax.axvspan(g9_idx - 0.45, g9_idx + 0.45, color="red", alpha=0.12, label="Rare Gesture G9 (~2.5%)")
    if g10_idx >= 0:
        ax.axvspan(g10_idx - 0.45, g10_idx + 0.45, color="purple", alpha=0.12, label="Rare Gesture G10 (~1.1%)")

    ax.set_title("Per-Class Gesture F1 Score on Suturing (8 LOUO Folds)", fontweight="bold", pad=12)
    ax.set_xlabel("Surgical Gesture Class")
    ax.set_ylabel("F1 Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontweight="bold")
    ax.set_ylim([0, 100])
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_iter36_per_class_f1.png")
    plt.savefig(FIGURES_DIR / "fig_iter36_per_class_f1.pdf")
    plt.close()
    logger.info("Saved fig_iter36_per_class_f1")

    # 2. Figure 2: Confusion Matrices (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    axes_flat = axes.flatten()

    for idx, c in enumerate(conds):
        ax = axes_flat[idx]
        if c not in audit_data["conditions"]:
            ax.set_title(f"{COND_LABELS[c]} (No Data)")
            continue

        cm = np.array(audit_data["conditions"][c]["global_confusion_matrix"], dtype=np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_norm = (cm / row_sums) * 100.0

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
        ax.set_title(COND_LABELS[c], fontweight="bold", pad=10)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticklabels(class_names, fontsize=8.5)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        for r in range(len(class_names)):
            for col in range(len(class_names)):
                val = cm_norm[r, col]
                if val > 1.0:
                    text_color = "white" if val > 50 else "black"
                    ax.text(col, r, f"{val:.0f}", ha="center", va="center", color=text_color, fontsize=7)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.03)
    cbar.set_label("Row Normalized Accuracy (%)", rotation=270, labelpad=15)
    plt.suptitle("Normalized Confusion Matrices on Suturing Across Loss Formulations", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(FIGURES_DIR / "fig_iter36_confusion_matrices.png")
    plt.savefig(FIGURES_DIR / "fig_iter36_confusion_matrices.pdf")
    plt.close()
    logger.info("Saved fig_iter36_confusion_matrices")

    # 3. Figure 3: Rare Gesture Recall vs Dominant Gesture Recall
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    target_gestures = ["G2", "G3", "G9", "G10"]
    x = np.arange(len(target_gestures))
    width = 0.18

    for i, c in enumerate(conds):
        if c not in audit_data["conditions"]:
            continue
        recalls = [
            audit_data["conditions"][c]["global_per_class"][g]["recall"]
            for g in target_gestures
        ]
        ax.bar(
            x + (i - 1.5) * width, recalls, width=width,
            label=COND_LABELS[c], color=PALETTE[c], edgecolor="black", linewidth=0.6
        )

    ax.set_title("Gesture Recall: Dominant (G2, G3) vs. Rare (G9, G10)", fontweight="bold", pad=12)
    ax.set_xlabel("Gesture Category")
    ax.set_ylabel("Recall Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([
        "G2 (Dominant, 21.3%)",
        "G3 (Dominant, 33.7%)",
        "G9 (Rare, 2.5%)",
        "G10 (Rare, 1.1%)",
    ], fontweight="bold")
    ax.set_ylim([0, 100])
    ax.legend(loc="upper right", framealpha=0.92)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_iter36_rare_gesture_recall.png")
    plt.savefig(FIGURES_DIR / "fig_iter36_rare_gesture_recall.pdf")
    plt.close()
    logger.info("Saved fig_iter36_rare_gesture_recall")


def git_commit_and_push(commit_msg: str) -> bool:
    """Stage artifacts, commit, and push to origin main."""
    logger.info("=" * 70)
    logger.info("AUTOMATIC GIT COMMIT & PUSH")
    logger.info("=" * 70)
    try:
        branch_proc = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=True
        )
        current_branch = branch_proc.stdout.strip()

        subprocess.run(
            ["git", "add", "artifacts/", "paper/figures/", "docs/", "memory.md"],
            cwd=str(PROJECT_ROOT), check=True
        )

        status_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=True
        )
        if not status_proc.stdout.strip():
            logger.info("No new changes detected to commit.")
            return True

        subprocess.run(["git", "commit", "-m", commit_msg], cwd=str(PROJECT_ROOT), check=True)
        logger.info(f"Committed: '{commit_msg}'")

        push_proc = subprocess.run(
            ["git", "push", "origin", current_branch],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=True
        )
        logger.info(f"Pushed to {current_branch} successfully:\n{push_proc.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Git command failed: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Rare-Gesture Prediction Audit and Per-Class Evaluator.")
    parser.add_argument("--smoke-test", action="store_true", help="Run Fold 1 only for verification")
    parser.add_argument("--no-push", action="store_true", help="Skip automatic git push")
    parser.add_argument("--no-retrain", action="store_true", help="Fail if checkpoint is missing instead of retraining on the fly")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs if retraining is required")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()

    retrain_missing = not args.no_retrain

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Target Device: {device} | Smoke Test: {args.smoke_test}")

    prevent_windows_sleep(True)
    t0 = time.time()

    folds = [1] if args.smoke_test else list(range(1, 9))
    conditions = ["ce", "class_weighted", "focal", "cb_focal"]
    epochs = 1 if args.smoke_test else args.epochs

    audit_summary: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "task": "Suturing",
        "evaluated_folds": folds,
        "conditions": {},
    }

    try:
        for c in conditions:
            logger.info("=" * 70)
            logger.info(f"EVALUATING AND EXPORTING PREDICTIONS FOR: {c.upper()}")
            logger.info("=" * 70)
            report = evaluate_and_export_predictions(
                cond=c,
                device=device,
                folds=folds,
                retrain_missing=retrain_missing,
                epochs=epochs,
            )
            audit_summary["conditions"][c] = report

        # Collate summary artifact
        summary_path = PROJECT_ROOT / "artifacts/iter36_rare_gesture_audit.json"
        with open(summary_path, "w", encoding="utf-8") as sf:
            json.dump(audit_summary, sf, indent=2)
        logger.info(f"Saved rare-gesture audit artifact to: {summary_path}")

        # Generate publication plots
        if not args.smoke_test:
            plot_rare_gesture_audit_figures(audit_summary)

        # Print console comparison
        print("\n" + "=" * 95)
        print("RARE GESTURE (G9, G10) VS DOMINANT (G2, G3) AUDIT SUMMARY (SUTURING)")
        print("=" * 95)
        print(f"{'Condition':18} | {'G2 Recall (%)':15} | {'G3 Recall (%)':15} | {'G9 Recall (%)':15} | {'G10 Recall (%)':15}")
        print("-" * 95)
        for c in conditions:
            if c in audit_summary["conditions"]:
                p = audit_summary["conditions"][c]["global_per_class"]
                print(
                    f"{c:18} | "
                    f"{p['G2']['recall']:5.2f}% (F1:{p['G2']['f1']:5.2f}%) | "
                    f"{p['G3']['recall']:5.2f}% (F1:{p['G3']['f1']:5.2f}%) | "
                    f"{p['G9']['recall']:5.2f}% (F1:{p['G9']['f1']:5.2f}%) | "
                    f"{p['G10']['recall']:5.2f}% (F1:{p['G10']['f1']:5.2f}%)"
                )
        print("=" * 95 + "\n")

        # Automated push
        if not args.no_push and not args.smoke_test:
            git_commit_and_push(
                "feat(iter36): export per-class predictions and complete rare-gesture audit on mentor GPU"
            )

    finally:
        prevent_windows_sleep(False)

    total_time = time.time() - t0
    logger.info(f"Rare-gesture audit script completed in {total_time:.1f}s.")


if __name__ == "__main__":
    main()
