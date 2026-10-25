from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.metrics import (
    compute_classification_metrics,
    to_numpy_labels,
)


def fuse_logits(
    video_logits: Union[np.ndarray, torch.Tensor],
    kinematics_logits: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.5,
) -> Union[np.ndarray, torch.Tensor]:
    """Combine video and kinematics class logits using a convex combination.
    
    Formula:
        fused_logits = alpha * video_logits + (1.0 - alpha) * kinematics_logits
        
    Args:
        video_logits: Logits from video model (shape [B, C] or [B, W, C]).
        kinematics_logits: Logits from kinematics model (same shape).
        alpha: Weight for video modality in [0.0, 1.0].
               alpha = 0.0 -> pure kinematics logits
               alpha = 1.0 -> pure video logits
               alpha = 0.5 -> equal arithmetic weighting
               
    Returns:
        Fused logits with identical type and shape as input.
        
    Raises:
        ValueError: If alpha is outside [0.0, 1.0], shapes mismatch, or values are non-finite.
        TypeError: If input types mismatch or are unsupported.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Fusion weight alpha must be within [0.0, 1.0], got {alpha}")

    is_torch = isinstance(video_logits, torch.Tensor)
    if is_torch != isinstance(kinematics_logits, torch.Tensor):
        raise TypeError(
            f"Input type mismatch: video is {type(video_logits)}, "
            f"kinematics is {type(kinematics_logits)}"
        )

    if is_torch:
        if video_logits.shape != kinematics_logits.shape:
            raise ValueError(
                f"Shape mismatch: video logits {video_logits.shape} != "
                f"kinematics logits {kinematics_logits.shape}"
            )
        if not torch.all(torch.isfinite(video_logits)):
            raise ValueError("video_logits contains non-finite values (NaN or Inf)")
        if not torch.all(torch.isfinite(kinematics_logits)):
            raise ValueError("kinematics_logits contains non-finite values (NaN or Inf)")

        # Exact boundary handling avoids numerical precision artifacts
        if alpha == 0.0:
            return kinematics_logits.clone()
        if alpha == 1.0:
            return video_logits.clone()

        fused = alpha * video_logits + (1.0 - alpha) * kinematics_logits
        return fused
    else:
        v_arr = np.asarray(video_logits, dtype=np.float32)
        k_arr = np.asarray(kinematics_logits, dtype=np.float32)

        if v_arr.shape != k_arr.shape:
            raise ValueError(
                f"Shape mismatch: video logits {v_arr.shape} != "
                f"kinematics logits {k_arr.shape}"
            )
        if not np.all(np.isfinite(v_arr)):
            raise ValueError("video_logits contains non-finite values (NaN or Inf)")
        if not np.all(np.isfinite(k_arr)):
            raise ValueError("kinematics_logits contains non-finite values (NaN or Inf)")

        if alpha == 0.0:
            return k_arr.copy()
        if alpha == 1.0:
            return v_arr.copy()

        fused = alpha * v_arr + (1.0 - alpha) * k_arr
        return fused


def validate_class_mappings(
    vocab_video: Dict[str, int],
    vocab_kinematics: Dict[str, int],
) -> bool:
    """Verify that video and kinematics models share identical class index mappings.
    
    Prevents silent reordering, permutation, or class dimension truncation.
    
    Raises:
        ValueError: If vocabularies differ in size, keys, or integer index assignments.
    """
    if set(vocab_video.keys()) != set(vocab_kinematics.keys()):
        diff = set(vocab_video.keys()) ^ set(vocab_kinematics.keys())
        raise ValueError(f"Vocabulary mismatch between modalities: differing keys {diff}")

    for gesture, vid_idx in vocab_video.items():
        kin_idx = vocab_kinematics[gesture]
        if vid_idx != kin_idx:
            raise ValueError(
                f"Class index mismatch for gesture '{gesture}': "
                f"video index {vid_idx} != kinematics index {kin_idx}"
            )

    return True


def search_optimal_alpha(
    val_video_logits: Union[np.ndarray, torch.Tensor],
    val_kinematics_logits: Union[np.ndarray, torch.Tensor],
    val_targets: Union[np.ndarray, torch.Tensor, List[int]],
    metric: str = "macro_f1",
    alpha_candidates: Optional[List[float]] = None,
    num_classes: Optional[int] = None,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Search for the optimal fusion weight alpha exclusively on validation data.
    
    Zero-leakage guarantee: This function accepts validation data only. Test set
    logits or labels must NEVER be passed to this search function.
    
    Args:
        val_video_logits: Validation logits from video model (shape [N, C]).
        val_kinematics_logits: Validation logits from kinematics model (shape [N, C]).
        val_targets: Ground truth integer labels for validation set.
        metric: Metric to optimize ('macro_f1', 'balanced_accuracy', or 'accuracy').
        alpha_candidates: List of candidate alphas in [0, 1]. Defaults to grid [0.0, 0.1, ..., 1.0].
        num_classes: Total number of classes.
        class_names: Readable class names mapping.
        
    Returns:
        Dict containing optimal_alpha, best_val_score, and detailed candidate scores.
    """
    if alpha_candidates is None:
        alpha_candidates = [round(a, 2) for a in np.linspace(0.0, 1.0, 11).tolist()]

    # Validate candidates
    for a in alpha_candidates:
        if not (0.0 <= a <= 1.0):
            raise ValueError(f"Candidate alpha {a} outside [0.0, 1.0]")

    y_val_true = to_numpy_labels(val_targets)

    # Convert logits to numpy if needed
    if isinstance(val_video_logits, torch.Tensor):
        v_logits_np = val_video_logits.detach().cpu().numpy()
    else:
        v_logits_np = np.asarray(val_video_logits)

    if isinstance(val_kinematics_logits, torch.Tensor):
        k_logits_np = val_kinematics_logits.detach().cpu().numpy()
    else:
        k_logits_np = np.asarray(val_kinematics_logits)

    best_score = -1.0
    best_alpha = 0.5
    search_history: List[Dict[str, Any]] = []

    for a in alpha_candidates:
        fused = fuse_logits(v_logits_np, k_logits_np, alpha=a)
        preds = np.argmax(fused, axis=-1)

        m = compute_classification_metrics(
            y_true=y_val_true,
            y_pred=preds,
            num_classes=num_classes,
            class_names=class_names,
        )

        val_score = float(m.get(metric, m["macro_f1"]))
        search_history.append({
            "alpha": float(a),
            "val_score": val_score,
            "accuracy": float(m["accuracy"]),
            "balanced_accuracy": float(m["balanced_accuracy"]),
            "macro_f1": float(m["macro_f1"]),
        })

        if val_score > best_score:
            best_score = val_score
            best_alpha = float(a)

    return {
        "optimal_alpha": best_alpha,
        "best_val_score": best_score,
        "metric_name": metric,
        "search_history": search_history,
    }


class FixedLateFusion:
    """Fixed late-logit fusion operator combining pre-computed or streaming model logits."""

    def __init__(self, alpha: float = 0.5) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
        self.alpha = float(alpha)

    def forward(
        self,
        video_logits: Union[np.ndarray, torch.Tensor],
        kinematics_logits: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Combine logits with fixed weight alpha."""
        return fuse_logits(video_logits, kinematics_logits, alpha=self.alpha)

    def forward_sequence(
        self,
        video_logits_seq: Union[np.ndarray, torch.Tensor],
        kinematics_logits_seq: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Combine sequence logits across timesteps."""
        return fuse_logits(video_logits_seq, kinematics_logits_seq, alpha=self.alpha)

    def predict(
        self,
        video_logits: Union[np.ndarray, torch.Tensor],
        kinematics_logits: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """Compute fused logits and return predicted class indices."""
        fused = self.forward(video_logits, kinematics_logits)
        if isinstance(fused, torch.Tensor):
            return torch.argmax(fused, dim=-1).cpu().numpy()
        return np.argmax(fused, axis=-1)


class FixedLateFusionClassifier(nn.Module):
    """End-to-end PyTorch module combining trained video and kinematics models via late fusion."""

    def __init__(
        self,
        video_model: nn.Module,
        kinematics_model: nn.Module,
        alpha: float = 0.5,
        vocab_video: Optional[Dict[str, int]] = None,
        vocab_kinematics: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.video_model = video_model
        self.kinematics_model = kinematics_model
        self.alpha = float(alpha)

        if vocab_video is not None and vocab_kinematics is not None:
            validate_class_mappings(vocab_video, vocab_kinematics)

    def forward(
        self,
        video_x: torch.Tensor,
        kinematics_x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute fused logits for causal window classification."""
        v_logits = self.video_model(video_x)
        k_logits = self.kinematics_model(kinematics_x)
        return fuse_logits(v_logits, k_logits, alpha=self.alpha)

    def forward_sequence(
        self,
        video_x: torch.Tensor,
        kinematics_x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute fused logits for full sequence decoding."""
        v_seq = self.video_model.forward_sequence(video_x)
        k_seq = self.kinematics_model.forward_sequence(kinematics_x)
        return fuse_logits(v_seq, k_seq, alpha=self.alpha)
