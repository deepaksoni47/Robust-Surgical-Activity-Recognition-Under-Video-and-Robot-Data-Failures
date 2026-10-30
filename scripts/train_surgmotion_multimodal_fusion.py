"""Train and evaluate SurgMotion + Kinematics Multimodal Fusion on JIGSAWS (ITER-28).

Supports three fusion architectures:
1. Fixed Late Fusion ('late'):
   Combines unimodal video and kinematics logits:
       L_fused = alpha * L_video + (1 - alpha) * L_kin
   Selects alpha* in [0.0, 1.0] strictly using validation Macro-F1 (zero test-set tuning).

2. Feature Concatenation Fusion ('concat'):
   Concatenates 1024D SurgMotion features + 76D robot kinematics (1100D input)
   and models joint temporal interactions via ConcatFusionGRU.

3. Adaptive Reliability-Aware Fusion ('adaptive'):
   Uses a gating network to dynamically weight video and kinematics based on
   observable representations and explicit availability masks:
       g_m = (w_raw_m * a_m) / sum_j (w_raw_j * a_j + eps)
   Guarantees exact single-modality fallback under optical or robotic telemetry drop.

Robustness Stress Testing:
- Clean: both modalities intact (a = [1, 1])
- Video Corrupted: visual feature Gaussian perturbation
- Kinematics Corrupted: kinematic sensor noise perturbation
- Video Missing: a = [0, 1], optical telemetry unavailable
- Kinematics Missing: a = [1, 0], robotic telemetry unavailable
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import KinematicsNormalizer
from src.data.splits import get_louo_splits
from src.data.surgmotion_dataset import JIGSAWSSurgMotionDataset, get_surgmotion_dataloaders
from src.data.transcriptions import get_gesture_vocabulary
from src.evaluation.metrics import compute_classification_metrics, compute_edit_score
from src.models.concat_fusion import ConcatFusionGRU
from src.models.fusion import fuse_logits
from src.models.kinematics_gru import KinematicsGRU
from src.models.reliability_gate import (
    AdaptiveReliabilityFusion,
    ReliabilityGate,
    validate_availability_mask,
)
from src.models.losses import get_loss_function
from src.models.video_head import VideoGRU
from src.utils.reproducibility import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("surgmotion_fusion")


# ==============================================================================
# 1. Feature Concatenation Fusion Training & Evaluation
# ==============================================================================

def train_concat_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        v_in = batch["features"].to(device)     # [B, S, 1024]
        k_in = batch["kinematics"].to(device)   # [B, S, 76]
        targets = batch["target"].to(device)    # [B]

        optimizer.zero_grad()
        logits = model(v_in, k_in)
        loss = criterion(logits, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def eval_concat_loader(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    class_names: Dict[int, str],
    device: torch.device,
    corrupt_video: bool = False,
    corrupt_kin: bool = False,
    missing_video: bool = False,
    missing_kin: bool = False,
    noise_std: float = 1.0,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []

    for batch in loader:
        v_in = batch["features"].to(device).clone()
        k_in = batch["kinematics"].to(device).clone()
        targets = batch["target"].cpu().numpy()

        if missing_video:
            v_in = torch.zeros_like(v_in)
        elif corrupt_video:
            v_in = v_in + torch.randn_like(v_in) * noise_std

        if missing_kin:
            k_in = torch.zeros_like(k_in)
        elif corrupt_kin:
            k_in = k_in + torch.randn_like(k_in) * noise_std

        logits = model(v_in, k_in)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())

    y_true = np.array(all_targets, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)

    metrics = compute_classification_metrics(
        y_true=y_true, y_pred=y_pred, num_classes=num_classes, class_names=class_names
    )
    return metrics, y_true, y_pred


def compute_concat_edit_score(
    model: nn.Module,
    dataset: JIGSAWSSurgMotionDataset,
    device: torch.device,
    corrupt_video: bool = False,
    corrupt_kin: bool = False,
    missing_video: bool = False,
    missing_kin: bool = False,
    noise_std: float = 1.0,
) -> float:
    model.eval()
    true_seqs: List[List[int]] = []
    pred_seqs: List[List[int]] = []

    with torch.no_grad():
        for rec in dataset.trial_records:
            v_t = torch.from_numpy(rec["features"]).float().to(device).unsqueeze(0)
            k_t = torch.from_numpy(rec["kinematics"]).float().to(device).unsqueeze(0)

            if missing_video:
                v_t = torch.zeros_like(v_t)
            elif corrupt_video:
                v_t = v_t + torch.randn_like(v_t) * noise_std

            if missing_kin:
                k_t = torch.zeros_like(k_t)
            elif corrupt_kin:
                k_t = k_t + torch.randn_like(k_t) * noise_std

            if hasattr(model, "forward_sequence"):
                logits_seq = model.forward_sequence(v_t, k_t).squeeze(0)
                preds = torch.argmax(logits_seq, dim=-1).cpu().numpy().tolist()
            else:
                logits = model(v_t, k_t).squeeze(0)
                preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()

            true_seqs.append(rec["labels"].tolist())
            pred_seqs.append(preds)

    return compute_edit_score(true_seqs, pred_seqs)


# ==============================================================================
# 2. Adaptive Reliability-Aware Gating Network Training & Evaluation
# ==============================================================================

class SurgMotionAdaptiveFusionModel(nn.Module):
    """End-to-end adaptive multimodal classifier combining SurgMotion VideoGRU and Kinematics GRU."""

    def __init__(
        self,
        num_classes: int,
        video_dim: int = 1024,
        kinematics_dim: int = 76,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Visual sequence encoder
        self.video_encoder = VideoGRU(
            input_dim=video_dim,
            proj_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=2,
            dropout=dropout,
        )

        # Kinematics sequence encoder
        self.kin_encoder = nn.Sequential(
            nn.Linear(kinematics_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.kin_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.kin_classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Dynamic Reliability Gate: inputs [v_repr (128), k_repr (128), a (2)] -> raw weights [2]
        self.gate = ReliabilityGate(
            video_dim=hidden_dim,
            kinematics_dim=hidden_dim,
            hidden_dim=64,
            dropout=dropout,
        )

    def forward(
        self,
        v_in: torch.Tensor,
        k_in: torch.Tensor,
        availability_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through adaptive fusion.
        
        Returns:
            fused_logits: [B, num_classes]
            weights: [B, 2] dynamic modality weights [g_video, g_kin]
        """
        # 1. Video representations and logits
        v_proj = self.video_encoder.input_proj(v_in)
        v_gru_out, _ = self.video_encoder.gru(v_proj)
        v_repr = v_gru_out[:, -1, :]  # [B, 128]
        v_logits = self.video_encoder.classifier(v_repr)

        # 2. Kinematics representations and logits
        k_proj = self.kin_encoder(k_in)
        k_gru_out, _ = self.kin_gru(k_proj)
        k_repr = k_gru_out[:, -1, :]  # [B, 128]
        k_logits = self.kin_classifier(k_repr)

        # 3. Dynamic Reliability Gate
        alpha = self.gate(
            video_repr=v_repr,
            kinematics_repr=k_repr,
            availability_mask=availability_mask,
        )  # [B, 2]

        alpha_v = alpha[:, 0].unsqueeze(-1)
        alpha_k = alpha[:, 1].unsqueeze(-1)

        fused_logits = alpha_v * v_logits + alpha_k * k_logits
        return fused_logits, alpha

    def forward_sequence(
        self,
        v_in: torch.Tensor,
        k_in: torch.Tensor,
        availability_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sequential inference for trial-level segmental edit score computation."""
        v_proj = self.video_encoder.input_proj(v_in)
        v_gru_out, _ = self.video_encoder.gru(v_proj)
        v_logits_seq = self.video_encoder.classifier(v_gru_out)  # [B, T, C]

        k_proj = self.kin_encoder(k_in)
        k_gru_out, _ = self.kin_gru(k_proj)
        k_logits_seq = self.kin_classifier(k_gru_out)  # [B, T, C]

        # Gate at each timestep
        b, t, d = v_gru_out.shape
        v_flat = v_gru_out.reshape(b * t, d)
        k_flat = k_gru_out.reshape(b * t, d)

        mask_flat = None
        if availability_mask is not None:
            if availability_mask.ndim == 2:
                mask_flat = availability_mask.unsqueeze(1).expand(b, t, 2).reshape(b * t, 2)
            else:
                mask_flat = availability_mask.reshape(b * t, 2)

        alpha_flat = self.gate(video_repr=v_flat, kinematics_repr=k_flat, availability_mask=mask_flat)
        alpha = alpha_flat.reshape(b, t, 2)

        alpha_v = alpha[:, :, 0].unsqueeze(-1)
        alpha_k = alpha[:, :, 1].unsqueeze(-1)

        fused_seq = alpha_v * v_logits_seq + alpha_k * k_logits_seq
        return fused_seq


def train_adaptive_epoch(
    model: SurgMotionAdaptiveFusionModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        v_in = batch["features"].to(device)
        k_in = batch["kinematics"].to(device)
        targets = batch["target"].to(device)

        # Standard training on clean data with random dropouts during training to regularize gate
        b_size = v_in.shape[0]
        # In 80% batches both available, 10% video drop, 10% kin drop
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


@torch.no_grad()
def eval_adaptive_loader(
    model: SurgMotionAdaptiveFusionModel,
    loader: DataLoader,
    num_classes: int,
    class_names: Dict[int, str],
    device: torch.device,
    availability_mask: Optional[torch.Tensor] = None,
    corrupt_video: bool = False,
    corrupt_kin: bool = False,
    noise_std: float = 1.0,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []

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

        if corrupt_video:
            v_in = v_in + torch.randn_like(v_in) * noise_std
        if corrupt_kin:
            k_in = k_in + torch.randn_like(k_in) * noise_std

        logits, _ = model(v_in, k_in, availability_mask=mask)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())

    y_true = np.array(all_targets, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)

    metrics = compute_classification_metrics(
        y_true=y_true, y_pred=y_pred, num_classes=num_classes, class_names=class_names
    )
    return metrics, y_true, y_pred


def compute_adaptive_edit_score(
    model: SurgMotionAdaptiveFusionModel,
    dataset: JIGSAWSSurgMotionDataset,
    device: torch.device,
    availability_mask: Optional[torch.Tensor] = None,
    corrupt_video: bool = False,
    corrupt_kin: bool = False,
    noise_std: float = 1.0,
) -> float:
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

            if corrupt_video:
                v_t = v_t + torch.randn_like(v_t) * noise_std
            if corrupt_kin:
                k_t = k_t + torch.randn_like(k_t) * noise_std

            fused_seq = model.forward_sequence(v_t, k_t, availability_mask=mask).squeeze(0)
            preds = torch.argmax(fused_seq, dim=-1).cpu().numpy().tolist()

            true_seqs.append(rec["labels"].tolist())
            pred_seqs.append(preds)

    return compute_edit_score(true_seqs, pred_seqs)


# ==============================================================================
# 3. Fixed Late-Logit Fusion Implementation
# ==============================================================================

def run_fixed_late_fusion_fold(
    task: str,
    fold: int,
    device: torch.device,
    sequence_length: int = 8,
    batch_size: int = 32,
    save_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute Fixed Late-Logit Fusion using pre-trained unimodal models with validation-searched alpha."""
    vocab = get_gesture_vocabulary(task=task, include_background=True)
    num_classes = len(vocab)
    class_names = {idx: g for g, idx in vocab.items()}

    train_loader, val_loader, test_loader, feat_norm, fold_info = get_surgmotion_dataloaders(
        task=task,
        fold=fold,
        sequence_length=sequence_length,
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

    # 1. Load Pretrained Unimodal Models
    v_ckpt_path = PROJECT_ROOT / f"checkpoints/surgmotion_VideoGRU_{task}/surgmotion_VideoGRU_{task}_fold{fold}.pt"
    k_ckpt_path = PROJECT_ROOT / f"checkpoints/kinematics_gru_{task}_fold{fold}.pt"

    # Instantiate video model
    video_model = VideoGRU(
        input_dim=1024,
        proj_dim=128,
        hidden_dim=128,
        num_classes=num_classes,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    if v_ckpt_path.is_file():
        v_ckpt = torch.load(v_ckpt_path, map_location=device)
        video_model.load_state_dict(v_ckpt["model_state"] if "model_state" in v_ckpt else v_ckpt)
    else:
        logger.warning(f"Video checkpoint not found at {v_ckpt_path}. Training a quick video model...")
        # Train quick video head if checkpoint missing
        opt = torch.optim.AdamW(video_model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        for _ in range(10):
            train_concat_epoch(
                lambda v, k: video_model(v), train_loader, crit, opt, device
            )

    # Instantiate kinematics model
    kin_model = KinematicsGRU(
        input_dim=76,
        hidden_dim=128,
        num_classes=num_classes,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    if k_ckpt_path.is_file():
        k_ckpt = torch.load(k_ckpt_path, map_location=device)
        kin_model.load_state_dict(k_ckpt["model_state"] if "model_state" in k_ckpt else k_ckpt)
    else:
        logger.warning(f"Kinematics checkpoint not found at {k_ckpt_path}. Using trained surrogate...")
        # Train quick kinematics head if checkpoint missing
        opt = torch.optim.AdamW(kin_model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        for _ in range(10):
            kin_model.train()
            for b in train_loader:
                opt.zero_grad()
                loss = crit(kin_model(b["kinematics"].to(device)), b["target"].to(device))
                loss.backward()
                opt.step()

    video_model.eval()
    kin_model.eval()

    # 2. Collect Validation Logits for Alpha Grid Search
    val_v_logits, val_k_logits, val_targets = [], [], []
    with torch.no_grad():
        for b in val_loader:
            v_logits = video_model(b["features"].to(device)).cpu().numpy()
            k_logits = kin_model(b["kinematics"].to(device)).cpu().numpy()
            val_v_logits.append(v_logits)
            val_k_logits.append(k_logits)
            val_targets.extend(b["target"].numpy().tolist())

    v_val = np.vstack(val_v_logits)
    k_val = np.vstack(val_k_logits)
    y_val = np.array(val_targets, dtype=np.int64)

    # Grid search alpha on validation only
    alphas = np.linspace(0.0, 1.0, 21)
    best_alpha = 0.5
    best_val_f1 = -1.0

    for a in alphas:
        fused_val = a * v_val + (1.0 - a) * k_val
        preds = np.argmax(fused_val, axis=-1)
        m = compute_classification_metrics(y_true=y_val, y_pred=preds, num_classes=num_classes)
        f1 = m["macro_f1"]
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_alpha = float(a)

    # 3. Evaluate on Held-out Test Set using best_alpha
    test_v_logits, test_k_logits, test_targets = [], [], []
    with torch.no_grad():
        for b in test_loader:
            v_logits = video_model(b["features"].to(device)).cpu().numpy()
            k_logits = kin_model(b["kinematics"].to(device)).cpu().numpy()
            test_v_logits.append(v_logits)
            test_k_logits.append(k_logits)
            test_targets.extend(b["target"].numpy().tolist())

    v_test = np.vstack(test_v_logits)
    k_test = np.vstack(test_k_logits)
    y_test = np.array(test_targets, dtype=np.int64)

    fused_test = best_alpha * v_test + (1.0 - best_alpha) * k_test
    test_preds = np.argmax(fused_test, axis=-1)

    test_metrics = compute_classification_metrics(
        y_true=y_test, y_pred=test_preds, num_classes=num_classes, class_names=class_names
    )

    # Compute Edit Score on test trials
    true_seqs, pred_seqs = [], []
    with torch.no_grad():
        for rec in test_loader.dataset.trial_records:
            v_t = torch.from_numpy(rec["features"]).float().to(device).unsqueeze(0)
            k_t = torch.from_numpy(rec["kinematics"]).float().to(device).unsqueeze(0)
            v_seq = video_model.forward_sequence(v_t).squeeze(0).cpu().numpy()
            k_seq = kin_model.forward_sequence(k_t).squeeze(0).cpu().numpy()
            f_seq = best_alpha * v_seq + (1.0 - best_alpha) * k_seq
            seq_preds = np.argmax(f_seq, axis=-1).tolist()
            true_seqs.append(rec["labels"].tolist())
            pred_seqs.append(seq_preds)

    edit_score = compute_edit_score(true_seqs, pred_seqs)

    # 4. Save artifacts
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        pred_path = save_dir / f"preds_late_fusion_{task}_fold{fold}.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump({
                "fold": fold,
                "task": task,
                "best_alpha": best_alpha,
                "y_true": y_test.tolist(),
                "y_pred": test_preds.tolist(),
            }, f, indent=2)

    return {
        "fold": fold,
        "task": task,
        "test_user": fold_info["test_user"],
        "is_active": True,
        "best_alpha": round(best_alpha, 3),
        "accuracy": round(float(test_metrics["accuracy"]) * 100.0, 2),
        "balanced_accuracy": round(float(test_metrics["balanced_accuracy"]) * 100.0, 2),
        "macro_f1": round(float(test_metrics["macro_f1"]) * 100.0, 2),
        "edit_score": round(float(edit_score), 2),
        "n_test_trials": len(fold_info["test_trials"]),
    }


# ==============================================================================
# 4. Multimodal Fusion Benchmark Orchestrator
# ==============================================================================

def run_multimodal_benchmark(
    task: str = "Suturing",
    fusion_type: str = "concat",
    sequence_length: int = 8,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    loss_type: str = "class_weighted",
    gamma: float = 2.0,
    beta: float = 0.9999,
    fold: Optional[int] = None,
    evaluate_robustness: bool = True,
    device_name: Optional[str] = None,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """Execute complete official LOUO multimodal cross-validation for a task."""
    set_seed(42)
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    splits = get_louo_splits(task=task, create_val_split=True)
    vocab = get_gesture_vocabulary(task=task, include_background=True)
    num_classes = len(vocab)
    class_names = {idx: g for g, idx in vocab.items()}

    loss_suffix = f"_{loss_type}" if loss_type != "class_weighted" else ""
    save_dir = PROJECT_ROOT / f"checkpoints/surgmotion_fusion_{fusion_type}_{task}{loss_suffix}" if save_artifacts else None
    t0_task = time.time()

    print("=" * 75)
    print(f"SURGMOTION MULTIMODAL FUSION: Task={task} | Type={fusion_type.upper()} | Loss={loss_type} (gamma={gamma}, beta={beta}) | Device={device}")
    print("=" * 75)

    fold_results = []
    robustness_results: Dict[str, List[Dict[str, float]]] = {
        "clean": [],
        "video_corrupted": [],
        "kinematics_corrupted": [],
        "video_missing": [],
        "kinematics_missing": [],
    }

    folds_to_run = [fold] if fold is not None else sorted(splits.keys())
    for fold_num in folds_to_run:
        fold_info = splits[fold_num]
        if len(fold_info["test_trials"]) == 0:
            print(f"Fold {fold_num} (User {fold_info['test_user']}): Omitted (empty test set in official split)")
            continue

        print(f"\n--- Training Fold {fold_num} / {len(splits)} (Test User: {fold_info['test_user']}, {len(fold_info['test_trials'])} trials) ---")


        if fusion_type == "late":
            res = run_fixed_late_fusion_fold(
                task=task,
                fold=fold_num,
                device=device,
                sequence_length=sequence_length,
                batch_size=batch_size,
                save_dir=save_dir,
            )
            fold_results.append(res)
            print(f"Result Fold {fold_num}: Acc={res['accuracy']:.2f}%, BalAcc={res['balanced_accuracy']:.2f}%, F1={res['macro_f1']:.2f}%, Edit={res['edit_score']:.2f}%, Alpha={res['best_alpha']:.2f}")

        elif fusion_type in ("concat", "adaptive"):
            train_loader, val_loader, test_loader, _, fold_info = get_surgmotion_dataloaders(
                task=task,
                fold=fold_num,
                sequence_length=sequence_length,
                batch_size=batch_size,
            )

            # Class frequencies computed strictly from training fold only (zero leakage)
            train_targets = []
            for b in train_loader:
                train_targets.extend(b["target"].numpy().tolist())
            counts = np.bincount(train_targets, minlength=num_classes)
            criterion = get_loss_function(
                loss_type=loss_type,
                class_counts=counts,
                num_classes=num_classes,
                gamma=gamma,
                beta=beta,
                device=device,
            )

            if fusion_type == "concat":
                model = ConcatFusionGRU(
                    video_dim=1024,
                    kinematics_dim=76,
                    proj_dim=128,
                    hidden_dim=128,
                    num_classes=num_classes,
                    num_layers=2,
                    dropout=0.2,
                ).to(device)
            else:
                model = SurgMotionAdaptiveFusionModel(
                    num_classes=num_classes,
                    video_dim=1024,
                    kinematics_dim=76,
                    hidden_dim=128,
                    dropout=0.2,
                ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

            best_val_f1 = -1.0
            best_state = None
            best_epoch = 0

            for epoch in range(1, epochs + 1):
                if fusion_type == "concat":
                    train_loss = train_concat_epoch(model, train_loader, criterion, optimizer, device)
                    val_metrics, _, _ = eval_concat_loader(model, val_loader, num_classes, class_names, device)
                else:
                    train_loss = train_adaptive_epoch(model, train_loader, criterion, optimizer, device)
                    val_metrics, _, _ = eval_adaptive_loader(model, val_loader, num_classes, class_names, device)

                val_f1 = val_metrics["macro_f1"]
                scheduler.step(val_f1)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch

            if best_state is not None:
                model.load_state_dict(best_state)
                if save_dir is not None:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_file = save_dir / f"model_fold{fold_num}.pt"
                    torch.save({
                        "model_state": best_state,
                        "fold": fold_num,
                        "task": task,
                        "fusion_type": fusion_type,
                        "loss_type": loss_type,
                        "gamma": gamma,
                        "beta": beta,
                    }, ckpt_file)


            # Test Evaluation
            if fusion_type == "concat":
                test_m, y_true, y_pred = eval_concat_loader(model, test_loader, num_classes, class_names, device)
                edit_score = compute_concat_edit_score(model, test_loader.dataset, device)
            else:
                test_m, y_true, y_pred = eval_adaptive_loader(model, test_loader, num_classes, class_names, device)
                edit_score = compute_adaptive_edit_score(model, test_loader.dataset, device)

            res = {
                "fold": fold_num,
                "task": task,
                "test_user": fold_info["test_user"],
                "is_active": True,
                "best_epoch": best_epoch,
                "accuracy": round(float(test_m["accuracy"]) * 100.0, 2),
                "balanced_accuracy": round(float(test_m["balanced_accuracy"]) * 100.0, 2),
                "macro_f1": round(float(test_m["macro_f1"]) * 100.0, 2),
                "edit_score": round(float(edit_score), 2),
                "n_test_trials": len(fold_info["test_trials"]),
            }
            fold_results.append(res)
            print(f"Result Fold {fold_num}: Acc={res['accuracy']:.2f}%, BalAcc={res['balanced_accuracy']:.2f}%, F1={res['macro_f1']:.2f}%, Edit={res['edit_score']:.2f}%")

            # Save Checkpoint
            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = save_dir / f"surgmotion_{fusion_type}_{task}_fold{fold_num}.pt"
                torch.save({"model_state": best_state, "fold": fold_num, "task": task, "metrics": res}, ckpt_path)

            # Robustness Stress Testing
            if evaluate_robustness:
                conditions = [
                    ("clean", {}),
                    ("video_corrupted", {"corrupt_video": True}),
                    ("kinematics_corrupted", {"corrupt_kin": True}),
                    ("video_missing", {"missing_video": True} if fusion_type == "concat" else {"availability_mask": torch.tensor([0, 1])}),
                    ("kinematics_missing", {"missing_kin": True} if fusion_type == "concat" else {"availability_mask": torch.tensor([1, 0])}),
                ]
                for cond_name, cond_kwargs in conditions:
                    if fusion_type == "concat":
                        cm, _, _ = eval_concat_loader(model, test_loader, num_classes, class_names, device, **cond_kwargs)
                        cedit = compute_concat_edit_score(model, test_loader.dataset, device, **cond_kwargs)
                    else:
                        cm, _, _ = eval_adaptive_loader(model, test_loader, num_classes, class_names, device, **cond_kwargs)
                        cedit = compute_adaptive_edit_score(model, test_loader.dataset, device, **cond_kwargs)

                    robustness_results[cond_name].append({
                        "accuracy": round(float(cm["accuracy"]) * 100.0, 2),
                        "macro_f1": round(float(cm["macro_f1"]) * 100.0, 2),
                        "edit_score": round(float(cedit), 2),
                    })

    # Summary Aggregates
    accs = [r["accuracy"] for r in fold_results]
    bal_accs = [r["balanced_accuracy"] for r in fold_results]
    f1s = [r["macro_f1"] for r in fold_results]
    edits = [r["edit_score"] for r in fold_results]

    summary = {
        "task": task,
        "fusion_type": fusion_type,
        "encoder": "SurgMotion-ViTL (Frozen 1024D) + Robot Kinematics (76D)",
        "active_folds": len(fold_results),
        "total_elapsed_sec": round(time.time() - t0_task, 2),
        "loss_config": {
            "loss_type": loss_type,
            "gamma": gamma,
            "beta": beta,
        },
        "metrics": {
            "accuracy": {"mean": round(float(np.mean(accs)), 2), "std": round(float(np.std(accs)), 2), "min": round(float(np.min(accs)), 2), "max": round(float(np.max(accs)), 2)},
            "balanced_accuracy": {"mean": round(float(np.mean(bal_accs)), 2), "std": round(float(np.std(bal_accs)), 2), "min": round(float(np.min(bal_accs)), 2), "max": round(float(np.max(bal_accs)), 2)},
            "macro_f1": {"mean": round(float(np.mean(f1s)), 2), "std": round(float(np.std(f1s)), 2), "min": round(float(np.min(f1s)), 2), "max": round(float(np.max(f1s)), 2)},
            "edit_score": {"mean": round(float(np.mean(edits)), 2), "std": round(float(np.std(edits)), 2), "min": round(float(np.min(edits)), 2), "max": round(float(np.max(edits)), 2)},
        },
        "fold_results": fold_results,
    }

    if evaluate_robustness and fusion_type in ("concat", "adaptive"):
        rob_summary: Dict[str, Dict[str, float]] = {}
        for cname, crecs in robustness_results.items():
            if crecs:
                rob_summary[cname] = {
                    "accuracy": round(float(np.mean([x["accuracy"] for x in crecs])), 2),
                    "macro_f1": round(float(np.mean([x["macro_f1"] for x in crecs])), 2),
                    "edit_score": round(float(np.mean([x["edit_score"] for x in crecs])), 2),
                }
        summary["robustness"] = rob_summary

    if save_artifacts:
        summary_path = PROJECT_ROOT / f"artifacts/surgmotion_fusion_{fusion_type}_{task}{loss_suffix}_results.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved multimodal fusion results to: {summary_path.resolve()}")

    print("\n" + "=" * 75)
    print(f"FINAL MULTIMODAL LOUO SUMMARY: {task} ({fusion_type.upper()}) | Loss: {loss_type}")
    print("=" * 75)
    print(f"Accuracy:          {summary['metrics']['accuracy']['mean']:.2f}% (std: {summary['metrics']['accuracy']['std']:.2f}%)")
    print(f"Balanced Accuracy: {summary['metrics']['balanced_accuracy']['mean']:.2f}% (std: {summary['metrics']['balanced_accuracy']['std']:.2f}%)")
    print(f"Macro-F1:          {summary['metrics']['macro_f1']['mean']:.2f}% (std: {summary['metrics']['macro_f1']['std']:.2f}%)")
    print(f"Edit Score:        {summary['metrics']['edit_score']['mean']:.2f}% (std: {summary['metrics']['edit_score']['std']:.2f}%)")
    print(f"Total Time:        {summary['total_elapsed_sec']:.1f} s")
    print("=" * 75)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="SurgMotion + Kinematics Multimodal Fusion Trainer.")
    parser.add_argument("--task", type=str, default="Suturing",
                        choices=["Suturing", "Knot_Tying", "Needle_Passing", "all"],
                        help="Target task")
    parser.add_argument("--fusion-type", type=str, default="concat",
                        choices=["late", "concat", "adaptive", "all"],
                        help="Multimodal fusion architecture")
    parser.add_argument("--sequence-length", type=int, default=8,
                        help="Sequence window length")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--loss-type", type=str, default="class_weighted",
                        choices=["ce", "class_weighted", "focal", "cb_focal"],
                        help="Loss function formulation for class imbalance mitigation")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Focusing parameter for focal loss formulations")
    parser.add_argument("--beta", type=float, default=0.9999,
                        help="Class-balanced effective number beta parameter")
    parser.add_argument("--fold", type=int, default=None,
                        help="Specific LOUO fold to run (default: all folds)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda or cpu)")
    parser.add_argument("--no-robustness", action="store_true",
                        help="Skip robustness evaluation")
    parser.add_argument("--no-save-artifacts", action="store_true",
                        help="Do not save model checkpoints and results JSONs")
    args = parser.parse_args()

    tasks = ["Suturing", "Knot_Tying", "Needle_Passing"] if args.task == "all" else [args.task]
    fusions = ["late", "concat", "adaptive"] if args.fusion_type == "all" else [args.fusion_type]

    for f_type in fusions:
        for t in tasks:
            run_multimodal_benchmark(
                task=t,
                fusion_type=f_type,
                sequence_length=args.sequence_length,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                loss_type=args.loss_type,
                gamma=args.gamma,
                beta=args.beta,
                fold=args.fold,
                evaluate_robustness=not args.no_robustness,
                device_name=args.device,
                save_artifacts=not args.no_save_artifacts,
            )


if __name__ == "__main__":
    main()

