from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


class KinematicsMLP(nn.Module):
    """Multi-Layer Perceptron for frame-independent surgical gesture classification from 76D kinematics.
    
    Acts as the non-temporal unimodal kinematics baseline to test Hypothesis H2 (predictive power)
    and compare against temporal sequence models in H5 (temporal context gain).
    """

    def __init__(
        self,
        input_dim: int = 76,
        num_classes: int = 11,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        activation: str = "gelu",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 64]
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm

        # Select activation function
        act_lower = activation.lower()
        if act_lower == "relu":
            act_cls = nn.ReLU
        elif act_lower == "gelu":
            act_cls = nn.GELU
        elif act_lower == "tanh":
            act_cls = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}. Choose 'relu', 'gelu', or 'tanh'.")

        # Build feature trunk
        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim

        self.trunk = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for causal window classification.
        
        Args:
            x: Input kinematics tensor of shape (B, D) or (B, W, D).
               If shape is (B, W, D), causal prediction uses the final timestep x[:, -1, :].
               
        Returns:
            Logits tensor of shape (B, num_classes).
        """
        if x.ndim == 3:
            # Extract final timestep for causal prediction
            x = x[:, -1, :]
        elif x.ndim != 2:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

        features = self.trunk(x)
        logits = self.classifier(features)
        return logits

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP across all timesteps independently.
        
        Args:
            x: Input kinematics tensor of shape (B, W, D).
            
        Returns:
            Logits tensor of shape (B, W, num_classes).
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor of shape (B, W, D), got {x.shape}")

        batch_size, seq_len, feat_dim = x.shape
        x_flat = x.contiguous().view(batch_size * seq_len, feat_dim)
        features = self.trunk(x_flat)
        logits_flat = self.classifier(features)
        return logits_flat.view(batch_size, seq_len, self.num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate feature embedding for downstream fusion or inspection.
        
        Args:
            x: Input tensor of shape (B, D) or (B, W, D).
            
        Returns:
            Feature embedding of shape (B, hidden_dims[-1]).
        """
        if x.ndim == 3:
            x = x[:, -1, :]
        return self.trunk(x)
