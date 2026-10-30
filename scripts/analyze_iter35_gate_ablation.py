"""Audited Static versus Dynamic Gate Ablation across All Active LOUO Folds (ITER-35).

Scientific Motivation (Workstream A Audit):
The exploratory gate comparison in ITER-34 calculated the mean gating weight alpha
from the held-out test batch / test trajectories. This introduced test-time adaptation
(data leakage), conditioning test predictions on empirical test-set gating statistics.

This script implements the strictly valid, leak-free evaluation:
1. Mean gating weights (alpha_train_v, alpha_train_k) are determined EXCLUSIVELY on training data.
2. Weights are frozen BEFORE evaluating the held-out test subject.
3. Evaluates all 23 active LOUO folds across Suturing (8), Knot Tying (8), and Needle Passing (7).
4. Compares:
   - Dynamic Gate: alpha(t) dynamically predicted per timestep.
   - Training-Derived Static Gate (VALID): alpha_train frozen from training fold.
   - Test-Derived Static Gate (EXPLORATORY / HISTORICAL): alpha_test derived from test batch (audited for comparison).
   - Uniform Fixed Fusion (50/50): alpha = [0.5, 0.5].
5. Saves fold-level results to 'artifacts/iter35_static_gate_ablation.json' without modifying historical files.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.multimodal_dataset import get_multimodal_fold_dataloaders
from src.data.splits import get_louo_splits
from src.data.transcriptions import get_gesture_vocabulary
from src.evaluation.metrics import compute_classification_metrics, compute_edit_score
from src.models.kinematics_gru import KinematicsGRU
from src.models.reliability_gate import AdaptiveReliabilityFusion, ReliabilityGate
from src.models.video_head import VideoGRU
from src.utils.reproducibility import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("iter35_gate_ablation")


def load_adaptive_model(
    task: str,
    fold_num: int,
    num_classes: int,
    device: torch.device,
) -> AdaptiveReliabilityFusion:
    """Load pretrained AdaptiveReliabilityFusion checkpoint."""
    v_model = VideoGRU(
        input_dim=512,
        proj_dim=128,
        hidden_dim=128,
        num_classes=num_classes,
        num_layers=2,
        dropout=0.2,
    )
    k_model = KinematicsGRU(
        input_dim=76,
        proj_dim=128,
        hidden_dim=128,
        num_classes=num_classes,
        num_layers=2,
        dropout=0.2,
    )
    gate = ReliabilityGate(
        video_dim=num_classes,
        kinematics_dim=num_classes,
        hidden_dim=32,
        use_linear_direct=True,
        dropout=0.1,
    )
    model = AdaptiveReliabilityFusion(
        num_classes=num_classes,
        video_model=v_model,
        kinematics_model=k_model,
        gate=gate,
    )

    ckpt_path = PROJECT_ROOT / f"checkpoints/adaptive_fusion_{task}_fold{fold_num}.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def compute_training_mean_alpha(
    model: AdaptiveReliabilityFusion,
    train_loader: Any,
    device: torch.device,
) -> Tuple[float, float]:
    """Compute mean gating weight alpha strictly on training fold."""
    model.eval()
    alpha_list: List[np.ndarray] = []

    for batch in train_loader:
        vx = batch["video"].to(device)
        kx = batch["kinematics"].to(device)
        b_size = vx.shape[0]
        mask = torch.ones((b_size, 2), dtype=torch.bool, device=device)

        vl = model.video_model(vx)
        kl = model.kinematics_model(kx)
        a = model.gate(vl, kl, availability_mask=mask)
        alpha_list.append(a.cpu().numpy())

    all_alphas = np.vstack(alpha_list)
    mean_alpha = all_alphas.mean(axis=0)
    return float(mean_alpha[0]), float(mean_alpha[1])


@torch.no_grad()
def evaluate_gate_variants_on_test(
    model: AdaptiveReliabilityFusion,
    test_loader: Any,
    alpha_train_v: float,
    alpha_train_k: float,
    num_classes: int,
    inv_vocab: Dict[int, str],
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate test set under dynamic, training-derived static, test-derived static, and 50/50 fusion."""
    model.eval()

    all_vl, all_kl, all_dyn_a = [], [], []
    targets: List[int] = []

    for batch in test_loader:
        vx = batch["video"].to(device)
        kx = batch["kinematics"].to(device)
        b_size = vx.shape[0]
        mask = torch.ones((b_size, 2), dtype=torch.bool, device=device)

        vl = model.video_model(vx)
        kl = model.kinematics_model(kx)
        a = model.gate(vl, kl, availability_mask=mask)

        all_vl.append(vl.cpu().numpy())
        all_kl.append(kl.cpu().numpy())
        all_dyn_a.append(a.cpu().numpy())
        targets.extend(batch["target"].numpy().tolist())

    v_logits = np.vstack(all_vl)        # [N, C]
    k_logits = np.vstack(all_kl)        # [N, C]
    dyn_alphas = np.vstack(all_dyn_a)   # [N, 2]
    y_true = np.array(targets, dtype=np.int64)

    # 1. Dynamic Gate: alpha(t) dynamically predicted per timestep
    dyn_fused = dyn_alphas[:, 0:1] * v_logits + dyn_alphas[:, 1:2] * k_logits
    dyn_preds = np.argmax(dyn_fused, axis=-1)

    # 2. Training-Derived Static Gate (VALID, ZERO LEAKAGE):
    stat_train_fused = alpha_train_v * v_logits + alpha_train_k * k_logits
    stat_train_preds = np.argmax(stat_train_fused, axis=-1)

    # 3. Test-Derived Static Gate (EXPLORATORY / HISTORICAL LEAKAGE):
    alpha_test_v = float(np.mean(dyn_alphas[:, 0]))
    alpha_test_k = float(np.mean(dyn_alphas[:, 1]))
    stat_test_fused = alpha_test_v * v_logits + alpha_test_k * k_logits
    stat_test_preds = np.argmax(stat_test_fused, axis=-1)

    # 4. Uniform Fixed 50/50 Fusion:
    uniform_fused = 0.5 * v_logits + 0.5 * k_logits
    uniform_preds = np.argmax(uniform_fused, axis=-1)

    m_dyn = compute_classification_metrics(y_true, dyn_preds, num_classes=num_classes, class_names=inv_vocab)
    m_stat_train = compute_classification_metrics(y_true, stat_train_preds, num_classes=num_classes, class_names=inv_vocab)
    m_stat_test = compute_classification_metrics(y_true, stat_test_preds, num_classes=num_classes, class_names=inv_vocab)
    m_uniform = compute_classification_metrics(y_true, uniform_preds, num_classes=num_classes, class_names=inv_vocab)

    # Compute trial-level Edit Scores for each variant in a single unified pass
    edit_dyn, edit_stat_train, edit_stat_test, edit_uniform = compute_all_trial_edits(
        model=model,
        dataset=test_loader.dataset,
        alpha_train_v=alpha_train_v,
        alpha_train_k=alpha_train_k,
        alpha_test_v=alpha_test_v,
        alpha_test_k=alpha_test_k,
        device=device,
    )

    return {
        "alpha_train": {"video": round(alpha_train_v, 4), "kinematics": round(alpha_train_k, 4)},
        "alpha_test": {"video": round(alpha_test_v, 4), "kinematics": round(alpha_test_k, 4)},
        "dynamic": {
            "accuracy": round(float(m_dyn["accuracy"]) * 100.0, 2),
            "balanced_accuracy": round(float(m_dyn["balanced_accuracy"]) * 100.0, 2),
            "macro_f1": round(float(m_dyn["macro_f1"]) * 100.0, 2),
            "edit_score": round(float(edit_dyn), 2),
        },
        "static_training_derived": {
            "accuracy": round(float(m_stat_train["accuracy"]) * 100.0, 2),
            "balanced_accuracy": round(float(m_stat_train["balanced_accuracy"]) * 100.0, 2),
            "macro_f1": round(float(m_stat_train["macro_f1"]) * 100.0, 2),
            "edit_score": round(float(edit_stat_train), 2),
        },
        "static_test_derived_exploratory": {
            "accuracy": round(float(m_stat_test["accuracy"]) * 100.0, 2),
            "balanced_accuracy": round(float(m_stat_test["balanced_accuracy"]) * 100.0, 2),
            "macro_f1": round(float(m_stat_test["macro_f1"]) * 100.0, 2),
            "edit_score": round(float(edit_stat_test), 2),
        },
        "uniform_50_50": {
            "accuracy": round(float(m_uniform["accuracy"]) * 100.0, 2),
            "balanced_accuracy": round(float(m_uniform["balanced_accuracy"]) * 100.0, 2),
            "macro_f1": round(float(m_uniform["macro_f1"]) * 100.0, 2),
            "edit_score": round(float(edit_uniform), 2),
        },
    }


@torch.no_grad()
def compute_all_trial_edits(
    model: AdaptiveReliabilityFusion,
    dataset: Any,
    alpha_train_v: float,
    alpha_train_k: float,
    alpha_test_v: float,
    alpha_test_k: float,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float, float, float]:
    """Compute segmental edit score across full trials for all 4 gating variants in a single pass."""
    true_seqs: List[List[int]] = []
    dyn_preds: List[List[int]] = []
    stat_train_preds: List[List[int]] = []
    stat_test_preds: List[List[int]] = []
    uniform_preds: List[List[int]] = []

    for record in dataset.trial_records:
        v_seq = torch.from_numpy(record["video"]).float().unsqueeze(0).to(device)
        k_seq = torch.from_numpy(record["kinematics"]).float().unsqueeze(0).to(device)
        labels = record["labels"].tolist()

        # Modality logits through sequential forward passes (computed once per trial)
        vl_seq = model.video_model.forward_sequence(v_seq).squeeze(0)  # [T, C]
        kl_seq = model.kinematics_model.forward_sequence(k_seq).squeeze(0)  # [T, C]

        t_len = vl_seq.shape[0]
        mask = torch.ones((t_len, 2), dtype=torch.bool, device=device)
        alphas = model.gate(vl_seq, kl_seq, availability_mask=mask)  # [T, 2]

        dyn_fused = alphas[:, 0:1] * vl_seq + alphas[:, 1:2] * kl_seq
        stat_tr_fused = alpha_train_v * vl_seq + alpha_train_k * kl_seq
        stat_te_fused = alpha_test_v * vl_seq + alpha_test_k * kl_seq
        unif_fused = 0.5 * vl_seq + 0.5 * kl_seq

        true_seqs.append(labels)
        dyn_preds.append(torch.argmax(dyn_fused, dim=-1).cpu().numpy().tolist())
        stat_train_preds.append(torch.argmax(stat_tr_fused, dim=-1).cpu().numpy().tolist())
        stat_test_preds.append(torch.argmax(stat_te_fused, dim=-1).cpu().numpy().tolist())
        uniform_preds.append(torch.argmax(unif_fused, dim=-1).cpu().numpy().tolist())

    return (
        compute_edit_score(true_seqs, dyn_preds),
        compute_edit_score(true_seqs, stat_train_preds),
        compute_edit_score(true_seqs, stat_test_preds),
        compute_edit_score(true_seqs, uniform_preds),
    )



def run_gate_audit_for_task(task: str, device: torch.device) -> Dict[str, Any]:
    """Run gate audit across all official LOUO folds for a task."""
    vocab = get_gesture_vocabulary(task)
    num_classes = len(vocab)
    inv_vocab = {v: k for k, v in vocab.items()}
    splits = get_louo_splits(task, create_val_split=True)

    fold_results = []
    logger.info(f"\n=======================================================")
    logger.info(f"AUDITING GATE ABLATION: {task} ({len(splits)} LOUO Folds)")
    logger.info(f"=======================================================")

    for fold_num in sorted(splits.keys()):
        fold_info = splits[fold_num]
        if len(fold_info["test_trials"]) == 0:
            logger.info(f"Fold {fold_num} (User {fold_info['test_user']}): Omitted (empty test set in official split)")
            continue

        model = load_adaptive_model(task, fold_num, num_classes, device)
        loaders = get_multimodal_fold_dataloaders(
            task=task,
            fold_num=fold_num,
            batch_size=32,
            window_size=16,
            train_stride=8,
            eval_stride=8,
            cache_dir="data/cache/features",
            encoder_name="ResNet18",
            camera="capture1",
            normalize=True,
        )

        # 1. Compute training-derived alpha (ZERO LEAKAGE)
        av_train, ak_train = compute_training_mean_alpha(model, loaders["train_loader"], device)

        # 2. Evaluate all variants on held-out test fold
        res = evaluate_gate_variants_on_test(
            model=model,
            test_loader=loaders["test_loader"],
            alpha_train_v=av_train,
            alpha_train_k=ak_train,
            num_classes=num_classes,
            inv_vocab=inv_vocab,
            device=device,
        )
        res["fold"] = fold_num
        res["test_user"] = fold_info["test_user"]
        fold_results.append(res)

        logger.info(
            f"Fold {fold_num} (User {fold_info['test_user']}): "
            f"Train Alpha=[{av_train:.3f}, {ak_train:.3f}] | "
            f"Dyn: Acc={res['dynamic']['accuracy']}%, F1={res['dynamic']['macro_f1']}% | "
            f"Static(Train): Acc={res['static_training_derived']['accuracy']}%, F1={res['static_training_derived']['macro_f1']}% | "
            f"Static(Test/Legacy): Acc={res['static_test_derived_exploratory']['accuracy']}%, F1={res['static_test_derived_exploratory']['macro_f1']}%"
        )

    # Compute task summary aggregates
    def get_task_stats(variant_key: str, metric_key: str) -> Dict[str, float]:
        vals = [r[variant_key][metric_key] for r in fold_results]
        return {
            "mean": round(float(np.mean(vals)), 2),
            "std": round(float(np.std(vals)), 2),
            "min": round(float(np.min(vals)), 2),
            "max": round(float(np.max(vals)), 2),
        }

    # Statistical hypothesis test: Dynamic vs Static (Training-Derived)
    dyn_f1s = [r["dynamic"]["macro_f1"] for r in fold_results]
    stat_train_f1s = [r["static_training_derived"]["macro_f1"] for r in fold_results]
    dyn_accs = [r["dynamic"]["accuracy"] for r in fold_results]
    stat_train_accs = [r["static_training_derived"]["accuracy"] for r in fold_results]

    diff_f1 = np.array(dyn_f1s) - np.array(stat_train_f1s)
    diff_acc = np.array(dyn_accs) - np.array(stat_train_accs)

    # Paired t-test and Wilcoxon signed-rank test
    t_f1, p_t_f1 = stats.ttest_rel(dyn_f1s, stat_train_f1s)
    t_acc, p_t_acc = stats.ttest_rel(dyn_accs, stat_train_accs)
    try:
        w_f1, p_w_f1 = stats.wilcoxon(dyn_f1s, stat_train_f1s)
    except Exception:
        w_f1, p_w_f1 = 0.0, 1.0

    task_summary = {
        "task": task,
        "n_active_folds": len(fold_results),
        "mean_training_alpha": {
            "video": round(float(np.mean([r["alpha_train"]["video"] for r in fold_results])), 4),
            "kinematics": round(float(np.mean([r["alpha_train"]["kinematics"] for r in fold_results])), 4),
        },
        "mean_test_alpha_legacy": {
            "video": round(float(np.mean([r["alpha_test"]["video"] for r in fold_results])), 4),
            "kinematics": round(float(np.mean([r["alpha_test"]["kinematics"] for r in fold_results])), 4),
        },
        "variants": {
            "dynamic": {m: get_task_stats("dynamic", m) for m in ["accuracy", "balanced_accuracy", "macro_f1", "edit_score"]},
            "static_training_derived": {m: get_task_stats("static_training_derived", m) for m in ["accuracy", "balanced_accuracy", "macro_f1", "edit_score"]},
            "static_test_derived_exploratory": {m: get_task_stats("static_test_derived_exploratory", m) for m in ["accuracy", "balanced_accuracy", "macro_f1", "edit_score"]},
            "uniform_50_50": {m: get_task_stats("uniform_50_50", m) for m in ["accuracy", "balanced_accuracy", "macro_f1", "edit_score"]},
        },
        "statistical_tests": {
            "dynamic_minus_static_train_acc": {
                "mean_diff": round(float(np.mean(diff_acc)), 2),
                "t_stat": round(float(t_acc), 4),
                "p_val_ttest": round(float(p_t_acc), 4),
                "statistically_significant": bool(p_t_acc < 0.05),
            },
            "dynamic_minus_static_train_f1": {
                "mean_diff": round(float(np.mean(diff_f1)), 2),
                "t_stat": round(float(t_f1), 4),
                "p_val_ttest": round(float(p_t_f1), 4),
                "wilcoxon_stat": round(float(w_f1), 4),
                "p_val_wilcoxon": round(float(p_w_f1), 4),
                "statistically_significant": bool(p_t_f1 < 0.05),
            },
        },
        "fold_results": fold_results,
    }

    return task_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audited Static vs Dynamic Gate Ablation across All Active LOUO Folds.")
    parser.add_argument("--tasks", nargs="+", default=["Suturing", "Knot_Tying", "Needle_Passing"], help="Tasks to audit")
    parser.add_argument("--output", type=str, default="artifacts/iter35_static_gate_ablation.json", help="Output JSON path")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cpu")
    logger.info(f"Running leak-free gate ablation audit on device: {device}")

    all_tasks_summary = {}
    for task in args.tasks:
        t_summary = run_gate_audit_for_task(task, device)
        all_tasks_summary[task] = t_summary

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_tasks_summary, f, indent=2)
    logger.info(f"\nSaved audited gate ablation report to: {output_path.resolve()}")

    # Print summary table
    print("\n" + "=" * 90)
    print("AUDITED GATE ABLATION SUMMARY — CLEAN INPUT CONDITIONS (23 ACTIVE LOUO FOLDS)")
    print("=" * 90)
    print(f"| Task | Gating Variant | Accuracy (%) | Macro-F1 (%) | Edit Score (%) | Stat vs Dyn (p-val) |")
    print(f"|:---|:---|:---:|:---:|:---:|:---:|")

    for task, t_data in all_tasks_summary.items():
        v = t_data["variants"]
        p_val = t_data["statistical_tests"]["dynamic_minus_static_train_f1"]["p_val_ttest"]
        sig = " (Not Sig)" if p_val >= 0.05 else " (Significant)"
        print(f"| {task:14} | Dynamic Gate (Predicted)        | {v['dynamic']['accuracy']['mean']:5.2f} ± {v['dynamic']['accuracy']['std']:4.2f} | {v['dynamic']['macro_f1']['mean']:5.2f} ± {v['dynamic']['macro_f1']['std']:4.2f} | {v['dynamic']['edit_score']['mean']:5.2f} ± {v['dynamic']['edit_score']['std']:4.2f} | Reference |")
        print(f"|                | Static Gate (Training-Derived)  | {v['static_training_derived']['accuracy']['mean']:5.2f} ± {v['static_training_derived']['accuracy']['std']:4.2f} | {v['static_training_derived']['macro_f1']['mean']:5.2f} ± {v['static_training_derived']['macro_f1']['std']:4.2f} | {v['static_training_derived']['edit_score']['mean']:5.2f} ± {v['static_training_derived']['edit_score']['std']:4.2f} | p={p_val:.3f}{sig} |")
        print(f"|                | Static Gate (Test/Exploratory)  | {v['static_test_derived_exploratory']['accuracy']['mean']:5.2f} ± {v['static_test_derived_exploratory']['accuracy']['std']:4.2f} | {v['static_test_derived_exploratory']['macro_f1']['mean']:5.2f} ± {v['static_test_derived_exploratory']['macro_f1']['std']:4.2f} | {v['static_test_derived_exploratory']['edit_score']['mean']:5.2f} ± {v['static_test_derived_exploratory']['edit_score']['std']:4.2f} | Legacy Audit |")
        print(f"|                | Uniform Fixed (50/50)           | {v['uniform_50_50']['accuracy']['mean']:5.2f} ± {v['uniform_50_50']['accuracy']['std']:4.2f} | {v['uniform_50_50']['macro_f1']['mean']:5.2f} ± {v['uniform_50_50']['macro_f1']['std']:4.2f} | {v['uniform_50_50']['edit_score']['mean']:5.2f} ± {v['uniform_50_50']['edit_score']['std']:4.2f} | Baseline |")
        print(f"|---")
    print("=" * 90)


if __name__ == "__main__":
    main()
