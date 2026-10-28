"""src/models/surgmotion_fusion.py

SurgMotion Multimodal Adaptive Fusion Architecture with Gate Trajectory Logging
and Continuous-Noise Injection for Robustness Ablation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.models.reliability_gate import ReliabilityGate
from src.models.video_head import VideoGRU


def inject_feature_noise(
    features: torch.Tensor,
    noise_std: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Inject zero-mean additive Gaussian noise into normalized features.
    
    Formula:
        x_noisy = x + epsilon,  epsilon ~ N(0, noise_std^2 * I)
        
    Args:
        features: Tensor of normalized features (mean ~ 0, var ~ 1).
        noise_std: Standard deviation of Gaussian perturbation.
        generator: Optional torch.Generator for bitwise determinism.
        
    Returns:
        torch.Tensor: Perturbed feature tensor of identical shape and dtype.
    """
    if noise_std <= 0.0:
        return features
    noise = torch.randn(
        features.shape,
        dtype=features.dtype,
        device=features.device,
        generator=generator,
    )
    return features + noise * float(noise_std)


class SurgMotionAdaptiveFusionModel(nn.Module):
    """End-to-end adaptive multimodal classifier combining SurgMotion VideoGRU and Kinematics GRU.
    
    Processes 1024-D SurgMotion tubelets and 76-D robot kinematics through dedicated recurrent trunks,
    then combines prediction streams dynamically using the ReliabilityGate:
        y_fused = alpha_vis * y_vis + alpha_kin * y_kin
    where [alpha_vis, alpha_kin] = ReliabilityGate(h_vis, h_kin, a).
    """

    def __init__(
        self,
        num_classes: int,
        video_dim: int = 1024,
        kinematics_dim: int = 76,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        use_linear_direct_gate: bool = True,
    ) -> None:
        """Initialize SurgMotion adaptive multimodal fusion model.
        
        Args:
            num_classes: Total gesture classes C.
            video_dim: Dimension of frozen visual representation (1024 for SurgMotion-L).
            kinematics_dim: Dimension of robot kinematics (76 for JIGSAWS).
            hidden_dim: Recurrent and feature projection dimension (128).
            dropout: Dropout probability.
            use_linear_direct_gate: Whether ReliabilityGate uses direct linear mapping (default True).
        """
        super().__init__()
        self.num_classes = int(num_classes)
        self.video_dim = int(video_dim)
        self.kinematics_dim = int(kinematics_dim)
        self.hidden_dim = int(hidden_dim)

        # 1. Visual sequence encoder (VideoGRU trunk)
        self.video_encoder = VideoGRU(
            input_dim=video_dim,
            proj_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=2,
            dropout=dropout,
        )

        # 2. Kinematics sequence encoder
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

        # 3. Dynamic Reliability Gate: inputs [v_repr (128), k_repr (128), a (2)] -> weights [2]
        self.gate = ReliabilityGate(
            video_dim=hidden_dim,
            kinematics_dim=hidden_dim,
            hidden_dim=64,
            use_linear_direct=use_linear_direct_gate,
            dropout=dropout,
        )

    def forward(
        self,
        v_in: torch.Tensor,
        k_in: torch.Tensor,
        availability_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for batch sequence classification.
        
        Args:
            v_in: Visual features of shape [B, S, video_dim].
            k_in: Kinematics of shape [B, S, kinematics_dim].
            availability_mask: Optional availability mask of shape [B, 2] or [2].
            
        Returns:
            fused_logits: Fused classification logits of shape [B, num_classes].
            alpha: Dynamic modality weights of shape [B, 2] [alpha_vis, alpha_kin].
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
        return_alpha: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sequential inference for trial-level segmental edit score and trajectory logging.
        
        Args:
            v_in: Visual features of shape [B, T, video_dim].
            k_in: Kinematics of shape [B, T, kinematics_dim].
            availability_mask: Optional availability mask of shape [B, 2] or [2].
            return_alpha: If True, returns (fused_seq, alpha_seq).
            
        Returns:
            fused_seq: Sequential logits of shape [B, T, num_classes].
            alpha_seq (optional): Sequential gate weights of shape [B, T, 2].
        """
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
            elif availability_mask.ndim == 1:
                mask_flat = availability_mask.unsqueeze(0).expand(b * t, 2)
            else:
                mask_flat = availability_mask.reshape(b * t, 2)

        alpha_flat = self.gate(video_repr=v_flat, kinematics_repr=k_flat, availability_mask=mask_flat)
        alpha = alpha_flat.reshape(b, t, 2)

        alpha_v = alpha[:, :, 0].unsqueeze(-1)
        alpha_k = alpha[:, :, 1].unsqueeze(-1)

        fused_seq = alpha_v * v_logits_seq + alpha_k * k_logits_seq
        if return_alpha:
            return fused_seq, alpha
        return fused_seq


class GateTrajectoryLogger:
    """Structured recorder for held-out trial time-resolved gate weights across conditions."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def log_trial_trajectory(
        self,
        fold: int,
        task: str,
        trial_id: str,
        condition: str,
        noise_std: float,
        alpha_weights: np.ndarray,  # [T, 2]
        center_frames: np.ndarray,  # [T]
        ground_truth: np.ndarray,   # [T]
        predictions: np.ndarray,    # [T]
    ) -> None:
        """Record full time-resolved trajectory for a trial."""
        n_steps = len(alpha_weights)
        for t in range(n_steps):
            self.records.append({
                "fold": int(fold),
                "task": str(task),
                "trial_id": str(trial_id),
                "time_idx": int(t),
                "center_frame": int(center_frames[t]),
                "condition": str(condition),
                "noise_std": float(noise_std),
                "alpha_video": round(float(alpha_weights[t, 0]), 4),
                "alpha_kinematics": round(float(alpha_weights[t, 1]), 4),
                "ground_truth": int(ground_truth[t]),
                "prediction": int(predictions[t]),
            })

    def save_json(self, output_path: Union[str, Path]) -> None:
        """Save records to JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"total_records": len(self.records), "records": self.records}, f, indent=2)

    def compute_condition_aggregates(self) -> Dict[str, Dict[str, float]]:
        """Compute mean and std of alpha_video and alpha_kinematics per condition."""
        by_cond: Dict[str, List[Tuple[float, float]]] = {}
        for r in self.records:
            cond = r["condition"]
            by_cond.setdefault(cond, []).append((r["alpha_video"], r["alpha_kinematics"]))

        summary: Dict[str, Dict[str, float]] = {}
        for cond, pairs in by_cond.items():
            arr = np.array(pairs)
            summary[cond] = {
                "mean_alpha_video": round(float(np.mean(arr[:, 0])), 4),
                "std_alpha_video": round(float(np.std(arr[:, 0])), 4),
                "mean_alpha_kin": round(float(np.mean(arr[:, 1])), 4),
                "std_alpha_kin": round(float(np.std(arr[:, 1])), 4),
                "count": len(arr),
            }
        return summary
