"""Loss functions for class-imbalance mitigation in surgical gesture recognition.

Provides configurable loss functions for ITER-35:
1. Standard Cross-Entropy ('ce')
2. Class-Weighted Cross-Entropy ('class_weighted')
3. Focal Loss ('focal') (Lin et al., ICCV 2017)
4. Class-Balanced Focal Loss ('cb_focal') (Cui et al., CVPR 2019)

Strict Scientific Rule:
All class frequencies / weights MUST be computed exclusively on training folds.
Never compute weights or calibrate losses using held-out validation or test labels.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_class_weights(
    targets: Union[np.ndarray, torch.Tensor, List[int]],
    num_classes: int,
    mode: str = "inverse",
    beta: float = 0.9999,
) -> torch.Tensor:
    """Compute per-class weights strictly from training fold targets.

    Args:
        targets: 1D array/tensor/list of class indices from training data only.
        num_classes: Total number of classes in the vocabulary.
        mode: Weighting strategy:
            - 'inverse': Standard inverse class frequency w_c = N / (C * n_c).
            - 'effective': Class-balanced effective number of samples (Cui et al., CVPR 2019):
                           E_c = (1 - beta^n_c) / (1 - beta), w_c = 1 / E_c.
            - 'none': Uniform weighting (all 1.0).
        beta: Hyperparameter for effective number of samples (typically 0.99, 0.999, or 0.9999).

    Returns:
        torch.Tensor: 1D float tensor of shape [num_classes] containing class weights.
    """
    if isinstance(targets, torch.Tensor):
        targets_np = targets.detach().cpu().numpy()
    elif isinstance(targets, list):
        targets_np = np.array(targets, dtype=np.int64)
    else:
        targets_np = np.asarray(targets, dtype=np.int64)

    if targets_np.ndim != 1:
        targets_np = targets_np.ravel()

    counts = np.bincount(targets_np, minlength=num_classes).astype(np.float64)
    total_samples = len(targets_np)

    if total_samples == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    if mode == "none":
        weights = np.ones(num_classes, dtype=np.float64)

    elif mode == "inverse":
        # w_c = N / (C * n_c) if n_c > 0 else 0.0 (or 1.0 for absent classes)
        weights = np.zeros(num_classes, dtype=np.float64)
        active_mask = counts > 0
        weights[active_mask] = total_samples / (num_classes * counts[active_mask])
        # For classes with 0 samples in training fold, assign weight 0 so they don't produce gradients
        weights[~active_mask] = 0.0
        # Normalize active weights so their mean is 1.0
        if np.sum(active_mask) > 0:
            weights = weights / np.mean(weights[active_mask])

    elif mode == "effective":
        # Cui et al. 2019: E_n = (1 - beta^n) / (1 - beta), weight = (1 - beta) / (1 - beta^n)
        effective_num = np.zeros(num_classes, dtype=np.float64)
        weights = np.zeros(num_classes, dtype=np.float64)
        active_mask = counts > 0
        effective_num[active_mask] = 1.0 - np.power(beta, counts[active_mask])
        weights[active_mask] = (1.0 - beta) / np.maximum(effective_num[active_mask], 1e-12)
        weights[~active_mask] = 0.0
        # Normalize so sum of weights equals num_classes
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights) * num_classes

    else:
        raise ValueError(f"Unknown class weight mode: '{mode}'. Supported: 'inverse', 'effective', 'none'")

    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Numerically stable multi-class Focal Loss (Lin et al., ICCV 2017).

    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)

    Uses log_softmax for numerical stability and probability clamping to prevent
    infinite loss or gradient explosion.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        eps: float = 1e-7,
    ) -> None:
        """Initialize FocalLoss.

        Args:
            gamma: Focusing parameter (gamma >= 0). When gamma=0, equivalent to CrossEntropy.
            weight: Optional per-class scaling weights [num_classes].
            reduction: 'mean', 'sum', or 'none'.
            eps: Small epsilon for probability clamping.
        """
        super().__init__()
        if gamma < 0.0:
            raise ValueError(f"Focal loss gamma must be non-negative, got {gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction: '{reduction}'. Choose 'mean', 'sum', or 'none'.")

        self.gamma = float(gamma)
        self.reduction = reduction
        self.eps = float(eps)

        if weight is not None:
            self.register_buffer("weight", weight.clone().detach().float())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            logits: [B, C] unnormalized class scores or [B, C, ...]
            targets: [B] class labels (0 <= targets < C) or [B, ...]

        Returns:
            torch.Tensor: Reduced focal loss scalar (or tensor if reduction='none').
        """
        if logits.ndim > 2:
            # Flatten spatial/temporal dimensions if present: [B, C, ...] -> [N, C]
            num_classes = logits.shape[1]
            logits = logits.transpose(1, -1).contiguous().view(-1, num_classes)
            targets = targets.view(-1)

        # 1. Numerically stable log-probabilities via log_softmax
        log_p = F.log_softmax(logits, dim=-1)  # [B, C]
        p = torch.exp(log_p)                   # [B, C]

        # 2. Extract p_t and log(p_t) for target class
        # targets: [B] -> [B, 1]
        targets = targets.view(-1, 1)
        log_pt = log_p.gather(dim=-1, index=targets).squeeze(-1)  # [B]
        pt = p.gather(dim=-1, index=targets).squeeze(-1)          # [B]

        # Clamp pt for guaranteed numerical stability
        pt = torch.clamp(pt, min=self.eps, max=1.0 - self.eps)

        # 3. Compute modulating factor: (1 - pt)^gamma
        focal_weight = torch.pow(1.0 - pt, self.gamma)  # [B]

        # 4. Apply per-class weights if configured
        loss = -focal_weight * log_pt  # [B]
        if self.weight is not None:
            class_weights = self.weight.gather(dim=0, index=targets.squeeze(-1))  # [B]
            loss = loss * class_weights

        # 5. Apply reduction
        if self.reduction == "mean":
            if self.weight is not None:
                # Weighted mean normalization
                return loss.sum() / torch.clamp(class_weights.sum(), min=self.eps)
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ClassBalancedFocalLoss(nn.Module):
    """Class-Balanced Focal Loss (Cui et al., CVPR 2019).

    Combines the effective number of samples weighting:
        w_c = (1 - beta) / (1 - beta^n_c)
    with the focal loss modulating factor:
        FL(p_t) = - w_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        class_counts: Union[np.ndarray, torch.Tensor, List[int]],
        num_classes: int,
        gamma: float = 2.0,
        beta: float = 0.9999,
        reduction: str = "mean",
    ) -> None:
        """Initialize ClassBalancedFocalLoss.

        Args:
            class_counts: Training fold class frequencies [num_classes].
            num_classes: Number of target classes.
            gamma: Focal loss modulating factor.
            beta: Effective number parameter in [0.9, 0.99999].
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        weights = compute_class_weights(
            targets=np.repeat(np.arange(num_classes), np.maximum(np.asarray(class_counts, dtype=np.int64), 0)),
            num_classes=num_classes,
            mode="effective",
            beta=beta,
        )
        self.focal_loss = FocalLoss(gamma=gamma, weight=weights, reduction=reduction)
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal_loss(logits, targets)


def get_loss_function(
    loss_type: str = "ce",
    class_counts: Optional[Union[np.ndarray, List[int], torch.Tensor]] = None,
    num_classes: int = 10,
    gamma: float = 2.0,
    beta: float = 0.9999,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Factory creating loss functions for training.

    Args:
        loss_type: One of 'ce', 'class_weighted', 'focal', 'cb_focal'.
        class_counts: Class counts strictly from training fold. Required for 'class_weighted' and 'cb_focal'.
        num_classes: Number of classes in the vocabulary.
        gamma: Focusing parameter for focal losses (default: 2.0).
        beta: Effective number parameter for cb_focal (default: 0.9999).
        device: Torch device to place class weights on.

    Returns:
        nn.Module: Instantiated PyTorch loss function.
    """
    loss_type = loss_type.lower().strip()

    if loss_type in ("ce", "cross_entropy"):
        logger.info("Instantiated Standard CrossEntropyLoss (unweighted).")
        criterion = nn.CrossEntropyLoss()

    elif loss_type in ("class_weighted", "weighted_ce"):
        if class_counts is None:
            raise ValueError("class_counts from training fold must be provided for 'class_weighted' loss.")
        weights = compute_class_weights(
            targets=np.repeat(np.arange(num_classes), np.maximum(np.asarray(class_counts, dtype=np.int64), 0)),
            num_classes=num_classes,
            mode="inverse",
        )
        if device is not None:
            weights = weights.to(device)
        logger.info(f"Instantiated Class-Weighted CrossEntropyLoss with training-derived weights: {weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weights)

    elif loss_type == "focal":
        logger.info(f"Instantiated FocalLoss(gamma={gamma}, weight=None).")
        criterion = FocalLoss(gamma=gamma, weight=None)

    elif loss_type in ("cb_focal", "class_balanced_focal"):
        if class_counts is None:
            raise ValueError("class_counts from training fold must be provided for 'cb_focal' loss.")
        criterion = ClassBalancedFocalLoss(
            class_counts=class_counts,
            num_classes=num_classes,
            gamma=gamma,
            beta=beta,
        )
        logger.info(f"Instantiated ClassBalancedFocalLoss(gamma={gamma}, beta={beta}).")

    else:
        raise ValueError(
            f"Unsupported loss_type: '{loss_type}'. Supported: 'ce', 'class_weighted', 'focal', 'cb_focal'."
        )

    if device is not None:
        criterion = criterion.to(device)

    return criterion
