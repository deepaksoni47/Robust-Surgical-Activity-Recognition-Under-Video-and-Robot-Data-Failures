from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class ConcatFusionGRU(nn.Module):
    """Early/intermediate feature concatenation multimodal temporal classifier.
    
    Concatenates synchronized visual feature representations and normalized 76D robot kinematics
    along the feature dimension at each timestep, projects through a shared projection trunk,
    and models cross-modal temporal interactions using a Gated Recurrent Unit (GRU).
    
    Architecture:
        1. Input: [x_video (512-D), x_kinematics (76-D)] -> concatenated (588-D)
        2. Shared projection: Linear(588, proj_dim) -> LayerNorm -> GELU -> Dropout
        3. Recurrent Sequence: GRU(proj_dim, hidden_dim, num_layers=2)
        4. Causal Head: Dropout -> Linear(hidden_dim, num_classes)
        
    Directly tests Hypothesis H3 (whether early feature concatenation fusion outperforms
    the strongest unimodal baseline under identical official LOUO splits).
    """

    def __init__(
        self,
        video_dim: int = 512,
        kinematics_dim: int = 76,
        proj_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 11,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.video_dim = video_dim
        self.kinematics_dim = kinematics_dim
        self.input_dim = video_dim + kinematics_dim
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # 1. Joint multimodal projection trunk
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
        )

        # 2. Recurrent sequence modeling over concatenated representations
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.feature_dim = gru_out_dim

        # 3. Gesture classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(gru_out_dim, num_classes),
        )

    def _prepare_inputs(
        self,
        video_x: torch.Tensor,
        kin_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Validate shapes, finite values, and concatenate modality streams.
        
        Supports either:
            - Two separate tensors: video_x (B, W, D_v) and kin_x (B, W, D_k)
            - A single pre-concatenated tensor: video_x (B, W, D_v + D_k) when kin_x is None
        """
        if kin_x is not None:
            # Two separate modality inputs
            if not isinstance(video_x, torch.Tensor) or not isinstance(kin_x, torch.Tensor):
                raise TypeError(
                    f"Inputs must be torch.Tensor, got video={type(video_x)}, kin={type(kin_x)}"
                )

            # Expand 2D (B, D) to 3D (B, 1, D)
            if video_x.ndim == 2:
                video_x = video_x.unsqueeze(1)
            if kin_x.ndim == 2:
                kin_x = kin_x.unsqueeze(1)

            if video_x.ndim != 3 or kin_x.ndim != 3:
                raise ValueError(
                    f"Expected 2D or 3D tensors, got video shape {video_x.shape}, kin shape {kin_x.shape}"
                )

            # Check batch size and sequence length match
            if video_x.shape[0] != kin_x.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: video batch {video_x.shape[0]} != kin batch {kin_x.shape[0]}"
                )
            if video_x.shape[1] != kin_x.shape[1]:
                raise ValueError(
                    f"Sequence length mismatch: video length {video_x.shape[1]} != kin length {kin_x.shape[1]}"
                )

            # Check feature dimensions
            if video_x.shape[-1] != self.video_dim:
                raise ValueError(
                    f"Video feature dimension mismatch: expected {self.video_dim}, got {video_x.shape[-1]}"
                )
            if kin_x.shape[-1] != self.kinematics_dim:
                raise ValueError(
                    f"Kinematics dimension mismatch: expected {self.kinematics_dim}, got {kin_x.shape[-1]}"
                )

            # Check finiteness
            if not torch.all(torch.isfinite(video_x)):
                raise ValueError("video_x contains non-finite values (NaN or Inf)")
            if not torch.all(torch.isfinite(kin_x)):
                raise ValueError("kin_x contains non-finite values (NaN or Inf)")

            # Concatenate along feature dimension: (B, W, D_v + D_k)
            fused = torch.cat([video_x, kin_x], dim=-1)
            return fused

        else:
            # Single pre-concatenated tensor
            if not isinstance(video_x, torch.Tensor):
                raise TypeError(f"video_x must be torch.Tensor, got {type(video_x)}")

            if video_x.ndim == 2:
                video_x = video_x.unsqueeze(1)
            elif video_x.ndim != 3:
                raise ValueError(f"Expected 2D or 3D tensor, got shape {video_x.shape}")

            if video_x.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Pre-concatenated feature dimension mismatch: expected {self.input_dim}, "
                    f"got {video_x.shape[-1]}"
                )

            if not torch.all(torch.isfinite(video_x)):
                raise ValueError("video_x contains non-finite values (NaN or Inf)")

            return video_x

    def forward(
        self,
        video_x: torch.Tensor,
        kin_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for causal window classification.
        
        Args:
            video_x: Video feature tensor of shape (B, W, 512) or pre-concatenated (B, W, 588).
            kin_x: Kinematics tensor of shape (B, W, 76), or None if pre-concatenated.
            
        Returns:
            Logits tensor of shape (B, num_classes) for the final timestep in the window.
        """
        fused = self._prepare_inputs(video_x, kin_x)

        # Project: (B, W, proj_dim)
        proj = self.input_proj(fused)

        # Recurrent modeling: (B, W, gru_out_dim)
        gru_out, _ = self.gru(proj)

        # Extract representation at causal window end
        last_step_feats = gru_out[:, -1, :]  # (B, gru_out_dim)
        logits = self.classifier(last_step_feats)

        return logits

    def forward_sequence(
        self,
        video_x: torch.Tensor,
        kin_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Produce predictions for all timesteps in the sequence.
        
        Used for full-trial sequential decoding and segmental Edit Score evaluation.
        
        Returns:
            Logits tensor of shape (B, W, num_classes).
        """
        fused = self._prepare_inputs(video_x, kin_x)
        proj = self.input_proj(fused)
        gru_out, _ = self.gru(proj)  # (B, W, gru_out_dim)
        logits_seq = self.classifier(gru_out)  # (B, W, num_classes)

        return logits_seq

    def extract_features(
        self,
        video_x: torch.Tensor,
        kin_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract multimodal temporal representation at the final timestep.
        
        Returns:
            Feature tensor of shape (B, feature_dim).
        """
        fused = self._prepare_inputs(video_x, kin_x)
        proj = self.input_proj(fused)
        gru_out, _ = self.gru(proj)
        return gru_out[:, -1, :]


class ConcatFusionMLP(nn.Module):
    """Frame-independent feature concatenation baseline.
    
    Evaluates fused multimodal features at independent timesteps to test
    Hypothesis H5 (temporal modeling gain over independent timesteps) under early fusion.
    """

    def __init__(
        self,
        video_dim: int = 512,
        kinematics_dim: int = 76,
        hidden_dim: int = 128,
        num_classes: int = 11,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.video_dim = video_dim
        self.kinematics_dim = kinematics_dim
        self.input_dim = video_dim + kinematics_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.feature_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _prepare_inputs(
        self,
        video_x: torch.Tensor,
        kin_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if kin_x is not None:
            if video_x.ndim == 2:
                video_x = video_x.unsqueeze(1)
            if kin_x.ndim == 2:
                kin_x = kin_x.unsqueeze(1)
            if video_x.shape[0] != kin_x.shape[0] or video_x.shape[1] != kin_x.shape[1]:
                raise ValueError("Batch size or sequence length mismatch")
            if video_x.shape[-1] != self.video_dim or kin_x.shape[-1] != self.kinematics_dim:
                raise ValueError("Feature dimension mismatch")
            if not torch.all(torch.isfinite(video_x)) or not torch.all(torch.isfinite(kin_x)):
                raise ValueError("Non-finite values detected")
            return torch.cat([video_x, kin_x], dim=-1)
        else:
            if video_x.ndim == 2:
                video_x = video_x.unsqueeze(1)
            if video_x.shape[-1] != self.input_dim:
                raise ValueError(f"Expected {self.input_dim}, got {video_x.shape[-1]}")
            if not torch.all(torch.isfinite(video_x)):
                raise ValueError("Non-finite values detected")
            return video_x

    def forward(
        self,
        video_x: torch.Tensor,
        kin_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Causal window prediction at final timestep."""
        fused = self._prepare_inputs(video_x, kin_x)
        last_step = fused[:, -1, :]
        feats = self.net(last_step)
        return self.classifier(feats)

    def forward_sequence(
        self,
        video_x: torch.Tensor,
        kin_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-timestep independent classification."""
        fused = self._prepare_inputs(video_x, kin_x)
        b, w, d = fused.shape
        flat = fused.reshape(b * w, d)
        feats = self.net(flat)
        logits_flat = self.classifier(feats)
        return logits_flat.reshape(b, w, self.num_classes)

    def extract_features(
        self,
        video_x: torch.Tensor,
        kin_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        fused = self._prepare_inputs(video_x, kin_x)
        last_step = fused[:, -1, :]
        return self.net(last_step)
