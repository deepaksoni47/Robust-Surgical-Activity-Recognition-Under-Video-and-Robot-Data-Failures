from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class KinematicsGRU(nn.Module):
    """Gated Recurrent Unit (GRU) temporal sequence classifier for 76D da Vinci kinematics.
    
    Models temporal context and inter-frame motion transitions over windowed sequences.
    Directly tests Hypothesis H5 (temporal modeling gain over frame-independent MLP).
    """

    def __init__(
        self,
        input_dim: int = 76,
        proj_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 11,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # 1. Kinematic projection trunk
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
        )

        # 2. Recurrent sequence modeling
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for causal window classification.
        
        Args:
            x: Input kinematics sequence tensor of shape (B, W, 76).
               
        Returns:
            Logits tensor of shape (B, num_classes) for the final timestep in the window.
        """
        if x.ndim == 2:
            # Expand single timestep to sequence of length 1: (B, 1, D)
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

        # Project kinematics: (B, W, proj_dim)
        proj = self.input_proj(x)

        # Process sequence: gru_out has shape (B, W, gru_out_dim)
        gru_out, _ = self.gru(proj)

        # Extract representation at final timestep (causal prediction)
        last_step_features = gru_out[:, -1, :]  # shape: (B, gru_out_dim)
        logits = self.classifier(last_step_features)

        return logits

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Produce predictions for all timesteps in the sequence.
        
        Args:
            x: Input kinematics tensor of shape (B, W, 76).
            
        Returns:
            Logits tensor of shape (B, W, num_classes).
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

        proj = self.input_proj(x)
        gru_out, _ = self.gru(proj)  # (B, W, gru_out_dim)
        logits_seq = self.classifier(gru_out)  # (B, W, num_classes)

        return logits_seq

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract temporal feature representation for downstream multimodal fusion.
        
        Args:
            x: Input kinematics tensor of shape (B, W, 76).
            
        Returns:
            Feature tensor of shape (B, feature_dim) at the final timestep.
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

        proj = self.input_proj(x)
        gru_out, _ = self.gru(proj)
        return gru_out[:, -1, :]
