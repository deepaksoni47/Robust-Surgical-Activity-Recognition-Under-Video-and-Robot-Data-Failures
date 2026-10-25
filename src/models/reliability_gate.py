"""src/models/reliability_gate.py

Lightweight Reliability-Aware Gating Network & Adaptive Multimodal Fusion.
Implements explicit modality availability masking, entropy-based reliability signals,
and mathematically exact single-modality fallback guarantees under clinical data failures.

Key Equations:
    Raw unnormalized scores:
        [w_raw_vis, w_raw_kin] = Softplus(W_g [h_vis; h_kin; a] + b_g)
    Availability masking & normalization:
        alpha_m = (w_raw_m * a_m) / sum_j (w_raw_j * a_j + eps)
    Fused logits:
        y_fused = alpha_vis * y_vis + alpha_kin * y_kin

Fallback Guarantees:
    - Exactly one modality available (e.g., [1, 0] or [0, 1]):
      The available modality weight is identically 1.0 and missing modality is identically 0.0.
    - Both modalities missing ([0, 0]):
      Fails loudly with BothModalitiesUnavailableError unless safe fallback mode is enabled.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BothModalitiesUnavailableError(ValueError):
    """Raised when all input modalities are flagged as unavailable for a sample."""
    pass


def validate_availability_mask(
    availability_mask: Union[torch.Tensor, np.ndarray],
    expected_batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Validate and convert availability mask to boolean torch.Tensor of shape [B, 2].
    
    Args:
        availability_mask: Binary mask indicating [video_available, kinematics_available].
                          Must have shape [B, 2] or [2] (which broadcasts to [1, 2]).
        expected_batch_size: Expected batch dimension B.
        
    Returns:
        torch.Tensor: Boolean tensor of shape [B, 2].
        
    Raises:
        ValueError: If shape is invalid, values are not binary {0, 1, True, False}, or non-finite.
    """
    if isinstance(availability_mask, np.ndarray):
        if not np.all(np.isfinite(availability_mask)):
            raise ValueError("availability_mask contains NaN or Inf.")
        mask_t = torch.from_numpy(availability_mask)
    elif isinstance(availability_mask, torch.Tensor):
        if not torch.all(torch.isfinite(mask_t := availability_mask)):
            raise ValueError("availability_mask contains NaN or Inf.")
    else:
        raise TypeError(f"availability_mask must be torch.Tensor or np.ndarray, got {type(availability_mask)}")

    if mask_t.ndim == 1:
        if mask_t.shape[0] != 2:
            raise ValueError(f"1D availability_mask must have length 2 [video, kin], got shape {mask_t.shape}")
        mask_t = mask_t.unsqueeze(0)  # [1, 2]

    if mask_t.ndim != 2 or mask_t.shape[1] != 2:
        raise ValueError(f"availability_mask must have shape [B, 2], got {mask_t.shape}")

    if expected_batch_size is not None and mask_t.shape[0] != expected_batch_size:
        if mask_t.shape[0] == 1:
            mask_t = mask_t.expand(expected_batch_size, 2)
        else:
            raise ValueError(
                f"Batch size mismatch: availability_mask has {mask_t.shape[0]}, "
                f"expected {expected_batch_size}"
            )

    # Verify binary domain {0, 1, True, False}
    if mask_t.dtype not in (torch.bool, torch.uint8):
        unique_vals = torch.unique(mask_t)
        for v in unique_vals:
            if v.item() not in (0.0, 1.0):
                raise ValueError(f"availability_mask must contain only binary values (0 or 1), found {v.item()}")

    return mask_t.bool()


def compute_masked_normalized_weights(
    raw_weights: torch.Tensor,
    availability_mask: torch.Tensor,
    eps: float = 1e-8,
    allow_both_missing: bool = False,
    both_missing_fallback: str = "equal",
) -> torch.Tensor:
    """Compute normalized modality weights enforcing exact masking and single-modality fallback.
    
    Args:
        raw_weights: Non-negative raw weights of shape [B, 2] from gating network.
        availability_mask: Boolean tensor of shape [B, 2] [is_video_avail, is_kin_avail].
        eps: Numerical stabilizer.
        allow_both_missing: If True, does not raise error on [0, 0] mask.
        both_missing_fallback: Fallback when both missing: 'equal' ([0.5, 0.5]) or 'zero' ([0.0, 0.0]).
        
    Returns:
        torch.Tensor: Normalized weights of shape [B, 2] summing to 1.0 over available modalities.
    """
    if raw_weights.ndim != 2 or raw_weights.shape[1] != 2:
        raise ValueError(f"raw_weights must have shape [B, 2], got {raw_weights.shape}")
    if not torch.all(torch.isfinite(raw_weights)):
        raise ValueError("raw_weights contains non-finite values (NaN or Inf).")

    mask = validate_availability_mask(availability_mask, expected_batch_size=raw_weights.shape[0])
    mask = mask.to(device=raw_weights.device)

    v_avail = mask[:, 0]
    k_avail = mask[:, 1]
    both_avail = v_avail & k_avail
    v_only = v_avail & (~k_avail)
    k_only = (~v_avail) & k_avail
    both_missing = (~v_avail) & (~k_avail)

    if both_missing.any() and not allow_both_missing:
        n_missing = int(both_missing.sum().item())
        raise BothModalitiesUnavailableError(
            f"Both video and kinematics are unavailable for {n_missing} sample(s) in batch. "
            "Cannot fuse unobserved modalities without an explicit fallback policy."
        )

    # Initialize output weights
    weights = torch.zeros_like(raw_weights)

    # 1. Exact Single-Modality Fallbacks (Guaranteed exact 1.0 / 0.0, zero float artifact)
    if v_only.any():
        weights[v_only, 0] = 1.0
        weights[v_only, 1] = 0.0

    if k_only.any():
        weights[k_only, 0] = 0.0
        weights[k_only, 1] = 1.0

    # 2. Both Available: Mask and Normalize
    if both_avail.any():
        sub_raw = raw_weights[both_avail]
        # Guarantee non-negativity
        sub_raw = torch.clamp(sub_raw, min=0.0)
        denom = sub_raw.sum(dim=-1, keepdim=True) + eps
        weights[both_avail] = sub_raw / denom

    # 3. Both Missing Fallback (if explicitly enabled)
    if both_missing.any() and allow_both_missing:
        if both_missing_fallback == "equal":
            weights[both_missing, :] = 0.5
        elif both_missing_fallback == "zero":
            weights[both_missing, :] = 0.0
        else:
            raise ValueError(f"Unknown both_missing_fallback policy: {both_missing_fallback}")

    return weights


class ReliabilityGate(nn.Module):
    """Lightweight parametric gating module estimating dynamic modality weights.
    
    Accepts video and kinematics representations alongside explicit binary availability signals.
    Produces strictly masked and normalized fusion weights [alpha_vis, alpha_kin].
    """

    def __init__(
        self,
        video_dim: int = 128,
        kinematics_dim: int = 128,
        hidden_dim: int = 32,
        use_linear_direct: bool = True,
        dropout: float = 0.1,
    ) -> None:
        """Initialize reliability gating module.
        
        Args:
            video_dim: Feature or hidden dimension of video modality.
            kinematics_dim: Feature or hidden dimension of kinematics modality.
            hidden_dim: Hidden dimension for two-layer gating network if not direct.
            use_linear_direct: If True, uses single linear projection W_g [h_v; h_k; a] + b
                              directly to 2 positive scores via Softplus (paper eq. 158).
            dropout: Dropout probability.
        """
        super().__init__()
        self.video_dim = int(video_dim)
        self.kinematics_dim = int(kinematics_dim)
        self.use_linear_direct = use_linear_direct

        # Input vector: [h_vis; h_kin; a_vis; a_kin] -> dimension = video_dim + kinematics_dim + 2
        in_dim = self.video_dim + self.kinematics_dim + 2

        if use_linear_direct:
            self.net = nn.Sequential(
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(in_dim, 2),
                nn.Softplus(),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, 2),
                nn.Softplus(),
            )

    def forward(
        self,
        video_repr: torch.Tensor,
        kinematics_repr: torch.Tensor,
        availability_mask: Optional[torch.Tensor] = None,
        allow_both_missing: bool = False,
    ) -> torch.Tensor:
        """Compute masked, normalized fusion weights [alpha_vis, alpha_kin].
        
        Args:
            video_repr: Video feature or logit tensor of shape [B, D_v].
            kinematics_repr: Kinematics feature or logit tensor of shape [B, D_k].
            availability_mask: Optional boolean or binary float mask [B, 2].
                              Defaults to both modalities available [1, 1].
            allow_both_missing: If True, does not raise BothModalitiesUnavailableError.
            
        Returns:
            torch.Tensor: Normalized weights of shape [B, 2] summing to 1.0 over available modalities.
        """
        if not torch.all(torch.isfinite(video_repr)):
            raise ValueError("video_repr contains non-finite values (NaN or Inf).")
        if not torch.all(torch.isfinite(kinematics_repr)):
            raise ValueError("kinematics_repr contains non-finite values (NaN or Inf).")

        if video_repr.ndim != 2 or video_repr.shape[1] != self.video_dim:
            raise ValueError(f"video_repr shape {video_repr.shape} incompatible with video_dim={self.video_dim}")
        if kinematics_repr.ndim != 2 or kinematics_repr.shape[1] != self.kinematics_dim:
            raise ValueError(f"kinematics_repr shape {kinematics_repr.shape} incompatible with kin_dim={self.kinematics_dim}")

        b_size = video_repr.shape[0]
        if kinematics_repr.shape[0] != b_size:
            raise ValueError(f"Batch size mismatch: video has {b_size}, kinematics has {kinematics_repr.shape[0]}")

        if availability_mask is None:
            # Default: both available
            mask = torch.ones((b_size, 2), dtype=torch.bool, device=video_repr.device)
        else:
            mask = validate_availability_mask(availability_mask, expected_batch_size=b_size)
            mask = mask.to(device=video_repr.device)

        # Concatenate inputs alongside float availability indicator
        mask_float = mask.float()
        combined_input = torch.cat([video_repr, kinematics_repr, mask_float], dim=-1)

        # Forward through gate trunk
        raw_scores = self.net(combined_input)

        # Apply availability masking and exact normalization
        alpha = compute_masked_normalized_weights(
            raw_weights=raw_scores,
            availability_mask=mask,
            allow_both_missing=allow_both_missing,
        )
        return alpha


class AdaptiveReliabilityFusion(nn.Module):
    """End-to-end adaptive multimodal classifier combining unimodal heads via reliability gating.
    
    Operates either at logit level or feature level:
        y_fused = alpha_vis * y_vis + alpha_kin * y_kin
    Guarantees seamless fallback to single-modality outputs when optical or robotic telemetry drops.
    """

    def __init__(
        self,
        num_classes: int,
        video_model: Optional[nn.Module] = None,
        kinematics_model: Optional[nn.Module] = None,
        gate: Optional[ReliabilityGate] = None,
        video_dim: int = 128,
        kinematics_dim: int = 128,
    ) -> None:
        """Initialize adaptive fusion model.
        
        Args:
            num_classes: Total gesture classes C.
            video_model: Optional trained video sequence classifier.
            kinematics_model: Optional trained kinematics sequence classifier.
            gate: Optional pre-configured ReliabilityGate. If None, builds a default gate.
            video_dim: Dimension of video representation fed to gate.
            kinematics_dim: Dimension of kinematics representation fed to gate.
        """
        super().__init__()
        self.num_classes = int(num_classes)
        self.video_model = video_model
        self.kinematics_model = kinematics_model

        if gate is not None:
            self.gate = gate
        else:
            self.gate = ReliabilityGate(
                video_dim=video_dim,
                kinematics_dim=kinematics_dim,
                use_linear_direct=True,
            )

    def forward_logits(
        self,
        video_logits: torch.Tensor,
        kinematics_logits: torch.Tensor,
        video_repr: Optional[torch.Tensor] = None,
        kinematics_repr: Optional[torch.Tensor] = None,
        availability_mask: Optional[torch.Tensor] = None,
        allow_both_missing: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse provided pre-computed logits using the reliability gate.
        
        Args:
            video_logits: Logits from video stream [B, C].
            kinematics_logits: Logits from kinematics stream [B, C].
            video_repr: Features fed to gate [B, D_v]. Defaults to video_logits if None.
            kinematics_repr: Features fed to gate [B, D_k]. Defaults to kinematics_logits if None.
            availability_mask: Modality availability mask [B, 2].
            allow_both_missing: If True, does not raise BothModalitiesUnavailableError.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (fused_logits [B, C], gating_weights [B, 2]).
        """
        if not torch.all(torch.isfinite(video_logits)):
            raise ValueError("video_logits contains non-finite values (NaN or Inf).")
        if not torch.all(torch.isfinite(kinematics_logits)):
            raise ValueError("kinematics_logits contains non-finite values (NaN or Inf).")
        if video_logits.shape != kinematics_logits.shape:
            raise ValueError(f"Logits shape mismatch: video {video_logits.shape} != kin {kinematics_logits.shape}")

        v_in = video_repr if video_repr is not None else video_logits
        k_in = kinematics_repr if kinematics_repr is not None else kinematics_logits

        alpha = self.gate(
            video_repr=v_in,
            kinematics_repr=k_in,
            availability_mask=availability_mask,
            allow_both_missing=allow_both_missing,
        )

        alpha_v = alpha[:, 0:1]  # [B, 1]
        alpha_k = alpha[:, 1:2]  # [B, 1]

        fused_logits = alpha_v * video_logits + alpha_k * kinematics_logits
        return fused_logits, alpha

    def forward(
        self,
        video_x: Optional[torch.Tensor] = None,
        kinematics_x: Optional[torch.Tensor] = None,
        availability_mask: Optional[torch.Tensor] = None,
        allow_both_missing: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """End-to-end forward pass through unimodal backbones and adaptive gate.
        
        Optimized execution: skips executing a sub-model if its modality is entirely missing in batch.
        """
        b_size = video_x.shape[0] if video_x is not None else (kinematics_x.shape[0] if kinematics_x is not None else 1)
        if availability_mask is None:
            mask = torch.ones((b_size, 2), dtype=torch.bool)
        else:
            mask = validate_availability_mask(availability_mask, expected_batch_size=b_size)

        v_any = mask[:, 0].any().item()
        k_any = mask[:, 1].any().item()

        if not v_any and not k_any and not allow_both_missing:
            raise BothModalitiesUnavailableError("Both video and kinematics streams are entirely unobserved.")

        # Compute video logits
        if v_any:
            if self.video_model is None:
                raise RuntimeError("video_model is None but video stream is available.")
            v_logits = self.video_model(video_x)
        else:
            # Placeholder zeros for missing modality; weight will be strictly 0.0
            device = kinematics_x.device if kinematics_x is not None else torch.device("cpu")
            v_logits = torch.zeros((b_size, self.num_classes), device=device)

        # Compute kinematics logits
        if k_any:
            if self.kinematics_model is None:
                raise RuntimeError("kinematics_model is None but kinematics stream is available.")
            k_logits = self.kinematics_model(kinematics_x)
        else:
            device = video_x.device if video_x is not None else torch.device("cpu")
            k_logits = torch.zeros((b_size, self.num_classes), device=device)

        return self.forward_logits(
            video_logits=v_logits,
            kinematics_logits=k_logits,
            availability_mask=mask,
            allow_both_missing=allow_both_missing,
        )


def fuse_logits_with_availability(
    video_logits: Union[np.ndarray, torch.Tensor],
    kinematics_logits: Union[np.ndarray, torch.Tensor],
    availability_mask: Union[np.ndarray, torch.Tensor],
    raw_alpha: float = 0.5,
) -> Union[np.ndarray, torch.Tensor]:
    """Pure logit-level fusion with explicit availability masking for PyTorch or NumPy arrays.
    
    Enforces exact single-modality fallback when one modality is unavailable.
    """
    is_torch = isinstance(video_logits, torch.Tensor)
    if is_torch != isinstance(kinematics_logits, torch.Tensor):
        raise TypeError("video_logits and kinematics_logits must have matching types (both Tensor or both ndarray).")

    if is_torch:
        mask = validate_availability_mask(availability_mask, expected_batch_size=video_logits.shape[0])
        mask = mask.to(device=video_logits.device)
        raw_weights = torch.full((video_logits.shape[0], 2), raw_alpha, device=video_logits.device)
        raw_weights[:, 1] = 1.0 - raw_alpha
        weights = compute_masked_normalized_weights(raw_weights, mask)
        return weights[:, 0:1] * video_logits + weights[:, 1:2] * kinematics_logits
    else:
        v_np = np.asarray(video_logits, dtype=np.float32)
        k_np = np.asarray(kinematics_logits, dtype=np.float32)
        m_t = validate_availability_mask(availability_mask, expected_batch_size=v_np.shape[0])
        raw_w_t = torch.full((v_np.shape[0], 2), raw_alpha)
        raw_w_t[:, 1] = 1.0 - raw_alpha
        weights_t = compute_masked_normalized_weights(raw_w_t, m_t)
        w_np = weights_t.numpy()
        return w_np[:, 0:1] * v_np + w_np[:, 1:2] * k_np
