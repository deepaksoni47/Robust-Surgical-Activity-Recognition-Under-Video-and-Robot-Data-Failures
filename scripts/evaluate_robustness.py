#!/usr/bin/env python3
"""scripts/evaluate_robustness.py

Execute robustness evaluation under visual corruptions and kinematic dropouts on JIGSAWS.
Evaluates Clean, Mean-C, and Worst-C metrics following the Endo-C6 benchmark protocol.

Usage:
    # 1. Audit readiness of all folds and models across tasks
    python scripts/evaluate_robustness.py --task Suturing --audit-only
    python scripts/evaluate_robustness.py --task all --audit-only

    # 2. Run synthetic pipeline verification / smoke run
    python scripts/evaluate_robustness.py --task Suturing --dry-run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.multimodal_dataset import get_multimodal_fold_dataloaders
from src.data.splits import get_louo_splits
import cv2
from src.data.video_dataset import VideoFeatureNormalizer
from src.evaluation.metrics import compute_classification_metrics, compute_edit_score
from src.models.concat_fusion import ConcatFusionGRU
from src.models.kinematics_gru import KinematicsGRU
from src.models.reliability_gate import AdaptiveReliabilityFusion, ReliabilityGate
from src.models.video_head import VideoGRU
from src.models.visual_encoder import BaseVisualEncoder, ResNet18VisualEncoder, build_visual_encoder
from src.robustness.evaluation import (
    audit_robustness_readiness,
    compute_robustness_aggregates,
    evaluate_model_under_corruption,
    print_robustness_audit_table,
)
from src.robustness.video_corruptions import (
    FEATURE_CORRUPTIONS,
    FRAME_CORRUPTIONS,
    get_visual_corruption,
    list_available_corruptions,
)
from src.utils.reproducibility import get_device, load_config, save_json, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_robustness")


# ---------------------------------------------------------------------------
# Synthetic Verification Dataset & Dummy Model for Dry-Run
# ---------------------------------------------------------------------------

class SyntheticVerificationDataset(Dataset):
    """Synthetic dataset for dry-run verification of the robustness pipeline."""

    def __init__(self, n_samples: int = 16, window_size: int = 16, v_dim: int = 512, k_dim: int = 76, num_classes: int = 5) -> None:
        super().__init__()
        torch.manual_seed(42)
        self.video = torch.randn(n_samples, window_size, v_dim)
        self.kinematics = torch.randn(n_samples, window_size, k_dim)
        self.targets = torch.randint(0, num_classes, (n_samples,))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "video": self.video[idx],
            "kinematics": self.kinematics[idx],
            "target": self.targets[idx],
        }


class DummyClassifier(nn.Module):
    """Toy classifier for dry-run pipeline smoke test."""

    def __init__(self, in_dim: int = 512, num_classes: int = 5) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x[:, -1, :]
        return self.fc(x)


def run_dry_run_smoke_test(task: str, cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Execute synthetic dry-run smoke test validating all registered corruption operators."""
    logger.info("Executing robustness pipeline smoke test with synthetic inputs...")
    dataset = SyntheticVerificationDataset(n_samples=16, num_classes=5)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    model = DummyClassifier(in_dim=512, num_classes=5).to(device)

    # 1. Clean evaluation
    clean_res = evaluate_model_under_corruption(
        model=model,
        dataloader=dataloader,
        num_classes=5,
        class_names={i: f"G{i}" for i in range(5)},
        device=device,
        corruption=None,
        model_type="video_only",
    )
    clean_f1 = clean_res["metrics"]["macro_f1"]
    logger.info(f"Clean Synthetic Macro-F1: {clean_f1:.4f}")

    # 2. Test all feature-level corruptions at severity 5
    feature_corruptions = list(FEATURE_CORRUPTIONS.keys())
    corrupted_f1s: Dict[str, float] = {}

    for c_name in feature_corruptions:
        op = get_visual_corruption(c_name, severity=5, seed=42)
        c_res = evaluate_model_under_corruption(
            model=model,
            dataloader=dataloader,
            num_classes=5,
            class_names={i: f"G{i}" for i in range(5)},
            device=device,
            corruption=op,
            seed=42,
            model_type="video_only",
        )
        corrupted_f1s[c_name] = c_res["metrics"]["macro_f1"]
        logger.info(f"Corrupted [{c_name}] (Severity 5) Macro-F1: {c_res['metrics']['macro_f1']:.4f}")

    # 3. Compute aggregate robustness metrics
    aggregates = compute_robustness_aggregates(clean_metric=clean_f1, corrupted_metrics=corrupted_f1s)
    logger.info(
        f"Synthetic Aggregates: Clean={aggregates['clean']:.4f}, "
        f"Mean-C={aggregates['mean_c']:.4f}, Worst-C={aggregates['worst_c']:.4f}, "
        f"Relative Drop={aggregates['relative_drop']*100:.2f}%"
    )

    return {
        "status": "VERIFIED_DRY_RUN",
        "clean_macro_f1": clean_f1,
        "corrupted_f1s": corrupted_f1s,
        "aggregates": aggregates,
    }


def build_adaptive_model_for_robustness(
    num_classes: int,
    device: torch.device,
) -> AdaptiveReliabilityFusion:
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
    return AdaptiveReliabilityFusion(
        num_classes=num_classes,
        video_model=v_model,
        kinematics_model=k_model,
        gate=gate,
    ).to(device)


def run_real_latent_robustness_benchmark(
    task: str,
    cfg: Dict[str, Any],
    checkpoints_dir: str = "checkpoints",
    output_dir: str = "artifacts/robustness",
    severities: Optional[List[int]] = None,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Execute real-data robustness benchmark under latent-feature perturbations across available models."""
    if device is None:
        device = get_device()
    if severities is None:
        severities = cfg.get("protocol", {}).get("eval_severities", [1, 3, 5])

    ckpt_path = Path(checkpoints_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = get_louo_splits(task, create_val_split=True)
    feature_corruptions = list(FEATURE_CORRUPTIONS.keys())

    target_models = ["video_gru", "kinematics_gru", "concat_fusion", "adaptive_fusion"]
    benchmark_results: Dict[str, Any] = {
        "benchmark": "LatentFeaturePerturbations",
        "task": task,
        "severities": severities,
        "seed": seed,
        "models": {},
    }

    for model_name in target_models:
        logger.info(f"\n---> Evaluating Robustness for Model: {model_name} on Task: {task}")
        fold_summaries = []

        for fold_num in sorted(splits.keys()):
            fold_info = splits[fold_num]
            if len(fold_info.get("test_trials", [])) == 0:
                continue

            if model_name == "video_gru":
                ckpt_file = ckpt_path / f"video_videogru_{task}_fold{fold_num}.pt"
                m_type = "video_only"
            elif model_name == "kinematics_gru":
                ckpt_file = ckpt_path / f"kinematics_gru_{task}_fold{fold_num}.pt"
                m_type = "kin_only"
            elif model_name == "concat_fusion":
                ckpt_file = ckpt_path / f"concat_fusion_{task}_fold{fold_num}.pt"
                m_type = "concat"
            elif model_name == "adaptive_fusion":
                ckpt_file = ckpt_path / f"adaptive_fusion_{task}_fold{fold_num}.pt"
                m_type = "adaptive"
            else:
                continue

            if not ckpt_file.is_file():
                continue

            try:
                loaders = get_multimodal_fold_dataloaders(
                    task=task,
                    fold_num=fold_num,
                    batch_size=32,
                    window_size=16,
                    eval_stride=8,
                    dataset_dir=cfg.get("data", {}).get("dataset_dir", "dataset"),
                    cache_dir=cfg.get("data", {}).get("cache_dir", "data/cache/features"),
                    encoder_name="ResNet18",
                    camera="capture1",
                    expected_dim=512,
                    create_val_split=True,
                    normalize=True,
                )
            except Exception as e:
                logger.warning(f"Could not load multimodal dataloader for fold {fold_num}: {e}")
                continue

            test_loader = loaders["test_loader"]
            vocab = loaders["vocab"]
            inv_vocab = {idx: g for g, idx in vocab.items()}
            num_classes = len(vocab)

            if model_name == "video_gru":
                model = VideoGRU(
                    input_dim=512, proj_dim=128, hidden_dim=128, num_classes=num_classes, num_layers=2, dropout=0.2
                ).to(device)
            elif model_name == "kinematics_gru":
                model = KinematicsGRU(
                    input_dim=76, proj_dim=128, hidden_dim=128, num_classes=num_classes, num_layers=2, dropout=0.2
                ).to(device)
            elif model_name == "concat_fusion":
                model = ConcatFusionGRU(
                    video_dim=512, kinematics_dim=76, proj_dim=128, hidden_dim=128, num_classes=num_classes, num_layers=2, dropout=0.2
                ).to(device)
            elif model_name == "adaptive_fusion":
                model = build_adaptive_model_for_robustness(num_classes=num_classes, device=device)
            else:
                continue

            model.load_state_dict(torch.load(ckpt_file, map_location=device))
            model.eval()

            # Clean evaluation
            clean_eval = evaluate_model_under_corruption(
                model=model,
                dataloader=test_loader,
                num_classes=num_classes,
                class_names=inv_vocab,
                device=device,
                corruption=None,
                model_type=m_type,
            )
            clean_f1 = clean_eval["metrics"]["macro_f1"]

            # Corruptions across severities
            sev_results: Dict[int, Dict[str, float]] = {}
            for sev in severities:
                c_f1s: Dict[str, float] = {}
                for c_name in feature_corruptions:
                    op = get_visual_corruption(c_name, severity=sev, seed=seed)
                    cor_eval = evaluate_model_under_corruption(
                        model=model,
                        dataloader=test_loader,
                        num_classes=num_classes,
                        class_names=inv_vocab,
                        device=device,
                        corruption=op,
                        seed=seed,
                        model_type=m_type,
                    )
                    c_f1s[c_name] = cor_eval["metrics"]["macro_f1"]
                sev_results[sev] = c_f1s

            primary_c_f1s = sev_results.get(5, sev_results[max(severities)])
            aggregates = compute_robustness_aggregates(clean_metric=clean_f1, corrupted_metrics=primary_c_f1s)

            fold_summaries.append({
                "fold": fold_num,
                "test_user": fold_info["test_user"],
                "clean_f1": clean_f1,
                "aggregates": aggregates,
                "corruptions_by_severity": sev_results,
            })

        if fold_summaries:
            clean_vals = [f["clean_f1"] for f in fold_summaries]
            mean_c_vals = [f["aggregates"]["mean_c"] for f in fold_summaries]
            worst_c_vals = [f["aggregates"]["worst_c"] for f in fold_summaries]
            rel_drop_vals = [f["aggregates"]["relative_drop"] for f in fold_summaries]

            benchmark_results["models"][model_name] = {
                "n_folds": len(fold_summaries),
                "clean_macro_f1_mean": float(np.mean(clean_vals)),
                "clean_macro_f1_std": float(np.std(clean_vals)),
                "mean_c_mean": float(np.mean(mean_c_vals)),
                "mean_c_std": float(np.std(mean_c_vals)),
                "worst_c_mean": float(np.mean(worst_c_vals)),
                "worst_c_std": float(np.std(worst_c_vals)),
                "relative_drop_mean": float(np.mean(rel_drop_vals)),
                "relative_drop_std": float(np.std(rel_drop_vals)),
                "folds": fold_summaries,
            }
            logger.info(
                f"[{model_name}] Robustness (Sev 5): Clean={np.mean(clean_vals):.4f}, "
                f"Mean-C={np.mean(mean_c_vals):.4f}, Worst-C={np.mean(worst_c_vals):.4f}, "
                f"RelDrop={np.mean(rel_drop_vals)*100:.2f}%"
            )

    out_file = out_dir / f"latent_perturbations_{task}_results.json"
    save_json(benchmark_results, out_file)
    logger.info(f"Saved real latent robustness results to: {out_file}")

    return benchmark_results


def run_real_image_robustness_benchmark(
    task: str,
    cfg: Dict[str, Any],
    checkpoints_dir: str = "checkpoints",
    output_dir: str = "artifacts/robustness",
    severity: int = 5,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Execute raw image-level optical corruptions benchmark (corrupting raw frames before visual encoding)."""
    if device is None:
        device = get_device()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = ResNet18VisualEncoder(device=device)
    frame_corruptions = list(FRAME_CORRUPTIONS.keys())
    splits = get_louo_splits(task, create_val_split=True)

    fold_num = 1
    fold_info = splits[fold_num]
    test_trials = fold_info["test_trials"][:2]

    logger.info(f"\n---> Evaluating Pre-Encoder Optical Corruptions for Task: {task} (Trials: {test_trials})")
    optical_results: Dict[str, Any] = {
        "benchmark": "PreEncoderFrameLevelOpticalCorruptions",
        "task": task,
        "fold": fold_num,
        "test_trials": test_trials,
        "severity": severity,
        "seed": seed,
        "corruptions": {},
    }

    for c_name in frame_corruptions:
        op = get_visual_corruption(c_name, severity=severity, seed=seed)
        drift_scores = []
        for trial_id in test_trials:
            v_path = Path("dataset") / task / "video" / f"{trial_id}_capture1.avi"
            if not v_path.is_file():
                continue
            cap = cv2.VideoCapture(str(v_path))
            frames_list = []
            while len(frames_list) < 64:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (224, 224))
                frames_list.append(frame_resized)
            cap.release()
            if not frames_list:
                continue
            clean_tensors = torch.cat([BaseVisualEncoder.preprocess_image(f) for f in frames_list], dim=0)
            clean_feats = encoder.extract_features(clean_tensors).cpu().numpy()
            corrupted_frames = [op.apply(f, seed=seed + i) for i, f in enumerate(frames_list)]
            corrupted_tensors = torch.cat([BaseVisualEncoder.preprocess_image(f) for f in corrupted_frames], dim=0)
            corrupted_feats = encoder.extract_features(corrupted_tensors).cpu().numpy()

            cosine_sim = np.mean(
                np.sum(clean_feats * corrupted_feats, axis=-1) /
                (np.linalg.norm(clean_feats, axis=-1) * np.linalg.norm(corrupted_feats, axis=-1) + 1e-8)
            )
            drift_scores.append(float(cosine_sim))

        optical_results["corruptions"][c_name] = {
            "mean_feature_cosine_similarity": float(np.mean(drift_scores)) if drift_scores else 0.0,
            "feature_distortion": float(1.0 - np.mean(drift_scores)) if drift_scores else 0.0,
        }

    out_file = out_dir / f"image_corruptions_{task}_results.json"
    save_json(optical_results, out_file)
    logger.info(f"Saved real image optical corruption results to: {out_file}")

    return optical_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate visual robustness and readiness under Endo-C6 corruptions.")
    parser.add_argument("--task", type=str, default="Suturing", choices=["Suturing", "Knot_Tying", "Needle_Passing", "all"])
    parser.add_argument("--config", type=str, default="configs/robustness.yaml")
    parser.add_argument("--severity", type=int, default=5, help="Corruption severity level in [1..5]")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--output-dir", type=str, default="artifacts/robustness")
    parser.add_argument("--audit-only", action="store_true", help="Print readiness audit table and exit")
    parser.add_argument("--dry-run", action="store_true", help="Run synthetic smoke run verifying all operators")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = ["Suturing", "Knot_Tying", "Needle_Passing"] if args.task == "all" else [args.task]
    cfg = load_config(args.config)
    device = get_device()
    set_seed(args.seed)

    for t in tasks:
        audit = audit_robustness_readiness(
            task=t,
            cfg=cfg,
            checkpoints_dir=args.checkpoints_dir,
        )

        if args.audit_only:
            print_robustness_audit_table(audit)
            continue

        if args.dry_run:
            print_robustness_audit_table(audit)
            res = run_dry_run_smoke_test(task=t, cfg=cfg, device=device)
            out_file = Path(args.output_dir) / f"dry_run_{t}_smoke_results.json"
            save_json(res, out_file)
            logger.info(f"Dry-run smoke results saved to: {out_file}")
            continue

        # Real Benchmark execution path
        print_robustness_audit_table(audit)
        if not audit["all_ready"]:
            logger.warning(
                f"Robustness benchmark execution BLOCKED for {t}: "
                f"Missing complete feature caches or trained checkpoints."
            )
            continue

        run_real_latent_robustness_benchmark(
            task=t,
            cfg=cfg,
            checkpoints_dir=args.checkpoints_dir,
            output_dir=args.output_dir,
            severities=[1, 3, 5],
            seed=args.seed,
            device=device,
        )
        run_real_image_robustness_benchmark(
            task=t,
            cfg=cfg,
            checkpoints_dir=args.checkpoints_dir,
            output_dir=args.output_dir,
            severity=args.severity,
            seed=args.seed,
            device=device,
        )


if __name__ == "__main__":
    main()
