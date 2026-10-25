from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class VideoGRU(nn.Module):
    """Gated Recurrent Unit (GRU) temporal sequence classifier for cached visual features.
    
    Models temporal context and motion transitions over sequences of frozen visual features.
    Directly tests Hypothesis H1 (frozen pretrained visual representation + small temporal head
    yields a viable video-only baseline) and H5 (temporal context improves sequence continuity).
    """

    def __init__(
        self,
        input_dim: int = 512,
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

        # 1. Visual feature projection trunk
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
        )

        # 2. Recurrent sequence modeling over visual representations
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
            x: Input video feature sequence tensor of shape (B, W, input_dim) or (B, input_dim).
               
        Returns:
            Logits tensor of shape (B, num_classes) for the final timestep in the window.
        """
        if x.ndim == 2:
            # Expand single timestep to sequence of length 1: (B, 1, D)
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.input_dim}, got {x.shape[-1]}"
            )

        # Project features: (B, W, proj_dim)
        proj = self.input_proj(x)

        # Process sequence: gru_out has shape (B, W, gru_out_dim)
        gru_out, _ = self.gru(proj)

        # Extract representation at final timestep (causal prediction)
        last_step_features = gru_out[:, -1, :]  # shape: (B, gru_out_dim)
        logits = self.classifier(last_step_features)

        return logits

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Produce predictions for all timesteps in the sequence.
        
        Used for full-trial sequential decoding and Edit Score evaluation.
        
        Args:
            x: Input video feature sequence tensor of shape (B, W, input_dim) or (B, input_dim).
            
        Returns:
            Logits tensor of shape (B, W, num_classes).
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.input_dim}, got {x.shape[-1]}"
            )

        proj = self.input_proj(x)
        gru_out, _ = self.gru(proj)  # (B, W, gru_out_dim)
        logits_seq = self.classifier(gru_out)  # (B, W, num_classes)

        return logits_seq

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract temporal visual feature representation for downstream multimodal fusion.
        
        Args:
            x: Input video feature tensor of shape (B, W, input_dim) or (B, input_dim).
            
        Returns:
            Feature tensor of shape (B, feature_dim) at the final timestep.
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.input_dim}, got {x.shape[-1]}"
            )

        proj = self.input_proj(x)
        gru_out, _ = self.gru(proj)
        return gru_out[:, -1, :]


class VideoMLP(nn.Module):
    """Frame-independent MLP classifier for cached visual features.
    
    Provides a frame-independent visual comparison model to test Hypothesis H5
    (temporal context gain over independent timesteps) on visual modalities.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        num_classes: int = 11,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.feature_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for causal window classification.
        
        Args:
            x: Input tensor of shape (B, W, input_dim) or (B, input_dim).
               If 3D, the final timestep is evaluated for causal prediction.
        """
        if x.ndim == 3:
            x = x[:, -1, :]
        elif x.ndim != 2:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.input_dim}, got {x.shape[-1]}"
            )

        feats = self.net(x)
        return self.classifier(feats)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Produce predictions for all timesteps independently.
        
        Args:
            x: Input tensor of shape (B, W, input_dim) or (B, input_dim).
            
        Returns:
            Logits tensor of shape (B, W, num_classes).
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.input_dim}, got {x.shape[-1]}"
            )

        b, w, d = x.shape
        x_flat = x.reshape(b * w, d)
        feats = self.net(x_flat)
        logits_flat = self.classifier(feats)
        return logits_flat.reshape(b, w, self.num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract visual representation at the final timestep."""
        if x.ndim == 3:
            x = x[:, -1, :]
        elif x.ndim != 2:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.input_dim}, got {x.shape[-1]}"
            )

        return self.net(x)
