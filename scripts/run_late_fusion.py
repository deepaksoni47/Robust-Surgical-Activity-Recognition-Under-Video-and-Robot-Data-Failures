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
from src.models.fusion import (
    FixedLateFusion,
    fuse_logits,
    search_optimal_alpha,
    validate_class_mappings,
)
from src.models.kinematics_gru import KinematicsGRU
from src.models.video_head import VideoGRU
from src.utils.reproducibility import get_device, load_config, save_json, set_seed


def check_fusion_readiness(
    task: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Audit availability of checkpoints, cached features, and alignment for late fusion."""
    splits = get_louo_splits(task, create_val_split=True)
    cache_dir = Path(cfg.get("data", {}).get("cache_dir", "data/cache/features"))
    encoder_name = cfg.get("data", {}).get("encoder_name", "ResNet18")
    camera = cfg.get("data", {}).get("camera", "capture1")

    video_pattern = cfg.get("models", {}).get("video", {}).get("checkpoint_pattern", "")
    kin_pattern = cfg.get("models", {}).get("kinematics", {}).get("checkpoint_pattern", "")

    fold_status: Dict[int, Dict[str, Any]] = {}
    ready_folds: List[int] = []

    for fold_num in sorted(splits.keys()):
        fold_info = splits[fold_num]
        all_trials = fold_info["train_trials"] + fold_info.get("val_trials", []) + fold_info["test_trials"]

        # 1. Check checkpoints
        v_ckpt = Path(video_pattern.format(task=task, fold=fold_num))
        k_ckpt = Path(kin_pattern.format(task=task, fold=fold_num))

        v_ckpt_exists = v_ckpt.is_file()
        k_ckpt_exists = k_ckpt.is_file()

        # 2. Check feature caches
        missing_caches: List[str] = []
        for t_id in all_trials:
            c_path = get_cache_path(cache_dir, t_id, encoder_name, camera=camera)
            if not c_path.is_file():
                missing_caches.append(t_id)

        is_fold_ready = (v_ckpt_exists and k_ckpt_exists and len(missing_caches) == 0)
        if is_fold_ready:
            ready_folds.append(fold_num)

        fold_status[fold_num] = {
            "test_user": fold_info["test_user"],
            "ready": is_fold_ready,
            "video_checkpoint_exists": v_ckpt_exists,
            "video_checkpoint_path": str(v_ckpt),
            "kinematics_checkpoint_exists": k_ckpt_exists,
            "kinematics_checkpoint_path": str(k_ckpt),
            "missing_cache_trials": missing_caches,
            "missing_cache_count": len(missing_caches),
        }

    return {
        "task": task,
        "total_folds": len(splits),
        "ready_folds": ready_folds,
        "fold_status": fold_status,
        "all_ready": (len(ready_folds) == len(splits)),
    }


@torch.no_grad()
def collect_modality_logits_and_targets(
    video_model: nn.Module,
    kinematics_model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pass dataloader batches through both models and collect paired logits and targets."""
    video_model.eval()
    kinematics_model.eval()

    all_v_logits: List[np.ndarray] = []
    all_k_logits: List[np.ndarray] = []
    all_targets: List[int] = []

    for batch in dataloader:
        v_in = batch["video"].to(device)
        k_in = batch["kinematics"].to(device)
        targets = batch["target"].cpu().numpy()

        v_out = video_model(v_in).cpu().numpy()
        k_out = kinematics_model(k_in).cpu().numpy()

        all_v_logits.append(v_out)
        all_k_logits.append(k_out)
        all_targets.extend(targets.tolist())

    v_logits = np.vstack(all_v_logits)
    k_logits = np.vstack(all_k_logits)
    targets_arr = np.array(all_targets, dtype=np.int64)

    return v_logits, k_logits, targets_arr


@torch.no_grad()
def compute_multimodal_trial_edit_scores(
    video_model: nn.Module,
    kinematics_model: nn.Module,
    test_dataset: Any,
    alpha: float,
    device: torch.device,
) -> float:
    """Compute Segmental Edit Score (%) across full test trial sequences with late fusion."""
    video_model.eval()
    kinematics_model.eval()
    true_seqs: List[List[int]] = []
    pred_seqs: List[List[int]] = []

    for record in test_dataset.trial_records:
        v_arr = torch.from_numpy(record["video"]).float().to(device).unsqueeze(0)  # (1, T, D_v)
        k_arr = torch.from_numpy(record["kinematics"]).float().to(device).unsqueeze(0)  # (1, T, 76)

        v_seq = video_model.forward_sequence(v_arr).squeeze(0)  # (T, C)
        k_seq = kinematics_model.forward_sequence(k_arr).squeeze(0)  # (T, C)

        fused_seq = fuse_logits(v_seq, k_seq, alpha=alpha)
        preds = torch.argmax(fused_seq, dim=-1).cpu().numpy().tolist()
        true_labels = record["labels"].tolist()

        true_seqs.append(true_labels)
        pred_seqs.append(preds)

    return compute_edit_score(true_seqs, pred_seqs)


def evaluate_late_fusion_fold(
    task: str,
    fold_num: int,
    cfg: Dict[str, Any],
    device: torch.device,
    fixed_alpha: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute fixed late fusion on one fold using validation-searched or pre-declared alpha."""
    fusion_cfg = cfg.get("fusion", {})
    models_cfg = cfg.get("models", {})
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("evaluation", {})

    window_size = data_cfg.get("window_size", 16)
    train_stride = data_cfg.get("train_stride", 8)
    eval_stride = data_cfg.get("eval_stride", 8)
    cache_dir = data_cfg.get("cache_dir", "data/cache/features")
    encoder_name = data_cfg.get("encoder_name", "ResNet18")
    camera = data_cfg.get("camera", "capture1")
    expected_dim = data_cfg.get("expected_dim", 512)
    normalize = data_cfg.get("normalize", True)
    batch_size = eval_cfg.get("batch_size", 32)

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

    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]
    vocab = loaders["vocab"]
    inv_vocab = {idx: g for g, idx in vocab.items()}
    num_classes = len(vocab)
    fold_info = loaders["fold_info"]

    # 2. Instantiate and load trained models
    v_cfg = models_cfg.get("video", {})
    k_cfg = models_cfg.get("kinematics", {})

    v_model = VideoGRU(
        input_dim=v_cfg.get("input_dim", 512),
        proj_dim=v_cfg.get("proj_dim", 128),
        hidden_dim=v_cfg.get("hidden_dim", 128),
        num_classes=num_classes,
        num_layers=v_cfg.get("num_layers", 2),
        dropout=v_cfg.get("dropout", 0.2),
    ).to(device)

    k_model = KinematicsGRU(
        input_dim=k_cfg.get("input_dim", 76),
        proj_dim=k_cfg.get("proj_dim", 128),
        hidden_dim=k_cfg.get("hidden_dim", 128),
        num_classes=num_classes,
        num_layers=k_cfg.get("num_layers", 2),
        dropout=k_cfg.get("dropout", 0.2),
    ).to(device)

    v_ckpt_path = Path(v_cfg.get("checkpoint_pattern", "").format(task=task, fold=fold_num))
    k_ckpt_path = Path(k_cfg.get("checkpoint_pattern", "").format(task=task, fold=fold_num))

    if not v_ckpt_path.is_file():
        raise FileNotFoundError(f"Video checkpoint missing: {v_ckpt_path}")
    if not k_ckpt_path.is_file():
        raise FileNotFoundError(f"Kinematics checkpoint missing: {k_ckpt_path}")

    v_model.load_state_dict(torch.load(v_ckpt_path, map_location=device))
    k_model.load_state_dict(torch.load(k_ckpt_path, map_location=device))

    # 3. Determine alpha: validation grid search or pre-declared
    chosen_alpha: float = 0.5
    search_info: Optional[Dict[str, Any]] = None

    if fixed_alpha is not None:
        chosen_alpha = float(fixed_alpha)
        print(f"Fold {fold_num}: Using pre-declared fixed alpha = {chosen_alpha:.2f}")
    elif fusion_cfg.get("alpha_search", True) and val_loader is not None and len(loaders["datasets"]["val"]) > 0:
        val_v_logits, val_k_logits, val_targets = collect_modality_logits_and_targets(
            v_model, k_model, val_loader, device
        )
        candidates = fusion_cfg.get("alpha_candidates", [round(a, 2) for a in np.linspace(0.0, 1.0, 11)])
        search_res = search_optimal_alpha(
            val_video_logits=val_v_logits,
            val_kinematics_logits=val_k_logits,
            val_targets=val_targets,
            metric=fusion_cfg.get("metric", "macro_f1"),
            alpha_candidates=candidates,
            num_classes=num_classes,
            class_names=inv_vocab,
        )
        chosen_alpha = search_res["optimal_alpha"]
        search_info = search_res
        print(
            f"Fold {fold_num}: Optimal alpha selected on validation set = {chosen_alpha:.2f} "
            f"(Val {search_res['metric_name']} = {search_res['best_val_score']:.4f})"
        )
    else:
        chosen_alpha = float(fusion_cfg.get("default_alpha", 0.5))
        print(f"Fold {fold_num}: Using default alpha = {chosen_alpha:.2f}")

    # 4. Evaluate Test Set across modalities for matched comparison
    test_v_logits, test_k_logits, test_targets = collect_modality_logits_and_targets(
        v_model, k_model, test_loader, device
    )

    # 4a. Fixed Late Fusion evaluation
    fused_test_logits = fuse_logits(test_v_logits, test_k_logits, alpha=chosen_alpha)
    fused_preds = np.argmax(fused_test_logits, axis=-1)
    fusion_metrics = compute_classification_metrics(
        y_true=test_targets,
        y_pred=fused_preds,
        num_classes=num_classes,
        class_names=inv_vocab,
    )
    fusion_edit = compute_multimodal_trial_edit_scores(
        v_model, k_model, loaders["datasets"]["test"], alpha=chosen_alpha, device=device
    )
    fusion_metrics["edit_score"] = fusion_edit
    fusion_metrics["alpha"] = chosen_alpha

    # 4b. Pure Kinematics comparison (alpha = 0.0)
    kin_preds = np.argmax(test_k_logits, axis=-1)
    kin_metrics = compute_classification_metrics(
        y_true=test_targets,
        y_pred=kin_preds,
        num_classes=num_classes,
        class_names=inv_vocab,
    )
    kin_edit = compute_multimodal_trial_edit_scores(
        v_model, k_model, loaders["datasets"]["test"], alpha=0.0, device=device
    )
    kin_metrics["edit_score"] = kin_edit

    # 4c. Pure Video comparison (alpha = 1.0)
    vid_preds = np.argmax(test_v_logits, axis=-1)
    vid_metrics = compute_classification_metrics(
        y_true=test_targets,
        y_pred=vid_preds,
        num_classes=num_classes,
        class_names=inv_vocab,
    )
    vid_edit = compute_multimodal_trial_edit_scores(
        v_model, k_model, loaders["datasets"]["test"], alpha=1.0, device=device
    )
    vid_metrics["edit_score"] = vid_edit

    print(
        f"Fold {fold_num} (User {fold_info['test_user']}) Results:\n"
        f"  - Kinematics-Only (alpha=0.0): Acc={kin_metrics['accuracy']*100:.2f}%, "
        f"Macro-F1={kin_metrics['macro_f1']:.4f}, Edit={kin_edit:.2f}%\n"
        f"  - Video-Only      (alpha=1.0): Acc={vid_metrics['accuracy']*100:.2f}%, "
        f"Macro-F1={vid_metrics['macro_f1']:.4f}, Edit={vid_edit:.2f}%\n"
        f"  - Late Fusion     (alpha={chosen_alpha:.2f}): Acc={fusion_metrics['accuracy']*100:.2f}%, "
        f"Macro-F1={fusion_metrics['macro_f1']:.4f}, Edit={fusion_edit:.2f}%"
    )

    return {
        "fold": fold_num,
        "test_user": fold_info["test_user"],
        "alpha": chosen_alpha,
        "search_info": search_info,
        "n_test_samples": len(test_targets),
        "n_test_trials": len(loaders["datasets"]["test"].trial_records),
        "test_trial_ids": [r["trial_id"] for r in loaders["datasets"]["test"].trial_records],
        "predictions": fused_preds.tolist(),
        "ground_truth": test_targets.tolist(),
        "fusion_metrics": fusion_metrics,
        "kinematics_metrics": kin_metrics,
        "video_metrics": vid_metrics,
    }


def run_late_fusion_benchmark(
    task: str = "Suturing",
    config_path: str = "configs/late_fusion.yaml",
    output_dir: str = "artifacts",
    specific_fold: Optional[int] = None,
    fixed_alpha: Optional[float] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute late fusion benchmark across official LOUO folds."""
    cfg = load_config(config_path)
    seed = cfg.get("evaluation", {}).get("seed", 42)
    set_seed(seed)
    device = get_device()

    print(f"\n==========================================")
    print(f"JIGSAWS Fixed Late-Logit Fusion Benchmark: {task}")
    print(f"Device: {device}")
    print(f"==========================================")

    # 1. Audit readiness
    audit = check_fusion_readiness(task, cfg)
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
                f"Video Ckpt={st['video_checkpoint_exists']}, "
                f"Kin Ckpt={st['kinematics_checkpoint_exists']}, "
                f"Missing Video Caches={st['missing_cache_count']} trials"
            )

    if dry_run or not ready_target_folds:
        print(f"\n[AUDIT STATUS] REAL FUSION BENCHMARK STATUS:")
        if not ready_target_folds:
            print(
                "Real multimodal benchmark evaluation cannot proceed because prerequisite unimodal "
                "checkpoints and/or video feature caches are unavailable.\n"
                "To prepare all prerequisites:\n"
                f"  1. Extract video features:  python scripts/extract_visual_features.py --task {task} --encoder ResNet18\n"
                f"  2. Train video baseline:    python scripts/train_video_baseline.py --task {task}\n"
                f"  3. Re-run late fusion:      python scripts/run_late_fusion.py --task {task}\n"
            )
        return {
            "status": "BLOCKED" if not ready_target_folds else "DRY_RUN",
            "task": task,
            "readiness": audit,
        }

    # 2. Run ready folds
    fold_results: List[Dict[str, Any]] = []
    pred_dir = Path(output_dir) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for f_num in ready_target_folds:
        res = evaluate_late_fusion_fold(
            task=task,
            fold_num=f_num,
            cfg=cfg,
            device=device,
            fixed_alpha=fixed_alpha,
        )
        if res:
            pred_file = pred_dir / f"late_fusion_{task}_fold{f_num}.json"
            save_json({
                "method": "FixedLateLogitFusion",
                "task": task,
                "fold": f_num,
                "test_user": res["test_user"],
                "alpha": res["alpha"],
                "n_test_trials": res["n_test_trials"],
                "n_test_samples": res["n_test_samples"],
                "test_trial_ids": res["test_trial_ids"],
                "accuracy": res["fusion_metrics"]["accuracy"],
                "balanced_accuracy": res["fusion_metrics"]["balanced_accuracy"],
                "macro_f1": res["fusion_metrics"]["macro_f1"],
                "edit_score": res["fusion_metrics"]["edit_score"],
                "predictions": res["predictions"],
                "ground_truth": res["ground_truth"],
            }, pred_file)
            fold_results.append(res)

    # 3. Aggregate metrics across folds
    f_accs = [r["fusion_metrics"]["accuracy"] * 100 for r in fold_results]
    f_bal_accs = [r["fusion_metrics"]["balanced_accuracy"] * 100 for r in fold_results]
    f_f1s = [r["fusion_metrics"]["macro_f1"] for r in fold_results]
    f_edits = [r["fusion_metrics"]["edit_score"] for r in fold_results]
    alphas = [r["alpha"] for r in fold_results]

    k_accs = [r["kinematics_metrics"]["accuracy"] * 100 for r in fold_results]
    k_f1s = [r["kinematics_metrics"]["macro_f1"] for r in fold_results]
    k_edits = [r["kinematics_metrics"]["edit_score"] for r in fold_results]

    v_accs = [r["video_metrics"]["accuracy"] * 100 for r in fold_results]
    v_f1s = [r["video_metrics"]["macro_f1"] for r in fold_results]
    v_edits = [r["video_metrics"]["edit_score"] for r in fold_results]

    summary = {
        "method": "FixedLateLogitFusion",
        "task": task,
        "n_folds_evaluated": len(fold_results),
        "mean_alpha": float(np.mean(alphas)),
        "fusion": {
            "mean_accuracy": float(np.mean(f_accs)),
            "std_accuracy": float(np.std(f_accs)),
            "mean_balanced_accuracy": float(np.mean(f_bal_accs)),
            "std_balanced_accuracy": float(np.std(f_bal_accs)),
            "mean_macro_f1": float(np.mean(f_f1s)),
            "std_macro_f1": float(np.std(f_f1s)),
            "mean_edit_score": float(np.mean(f_edits)),
            "std_edit_score": float(np.std(f_edits)),
        },
        "unimodal_kinematics": {
            "mean_accuracy": float(np.mean(k_accs)),
            "mean_macro_f1": float(np.mean(k_f1s)),
            "mean_edit_score": float(np.mean(k_edits)),
        },
        "unimodal_video": {
            "mean_accuracy": float(np.mean(v_accs)),
            "mean_macro_f1": float(np.mean(v_f1s)),
            "mean_edit_score": float(np.mean(v_edits)),
        },
        "fold_results": fold_results,
        "config": cfg,
    }

    out_file = Path(output_dir) / f"late_fusion_{task}_results.json"
    save_json(summary, out_file)
    print(f"\nSaved late fusion benchmark results to {out_file}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed late-logit fusion on JIGSAWS.")
    parser.add_argument("--task", type=str, default="Suturing", choices=["Suturing", "Knot_Tying", "Needle_Passing"])
    parser.add_argument("--config", type=str, default="configs/late_fusion.yaml")
    parser.add_argument("--fold", type=int, default=None, help="Evaluate a specific fold")
    parser.add_argument("--alpha", type=float, default=None, help="Pre-declared alpha override [0.0, 1.0]")
    parser.add_argument("--dry-run", action="store_true", help="Audit prerequisites without evaluating")
    parser.add_argument("--check-readiness", action="store_true", help="Audit checkpoints and caches")
    parser.add_argument("--output-dir", type=str, default="artifacts")

    args = parser.parse_args()

    run_late_fusion_benchmark(
        task=args.task,
        config_path=args.config,
        output_dir=args.output_dir,
        specific_fold=args.fold,
        fixed_alpha=args.alpha,
        dry_run=args.dry_run or args.check_readiness,
    )


if __name__ == "__main__":
    main()
