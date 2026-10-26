"""SurgMotion Visual Foundation Model Encoder.

Official architecture based on Meta's V-JEPA 2 (ViT-Large 300M backbone)
pretrained on SurgMotion-15M (3,658 hours of multi-organ surgical video).

Paper: Wu et al., arXiv:2602.05638 (2026).
Repository: https://github.com/CAIR-HKISI/SurgMotion
Model Hub: https://huggingface.co/CAIR-HKISI/SurgMotion
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.visual_encoder import BaseVisualEncoder


# ---------------------------------------------------------------------------
# 3D Positional Embeddings (Exact official V-JEPA 2 / SurgMotion implementation)
# ---------------------------------------------------------------------------

def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate 1D sine-cosine positional embedding for an array of positions."""
    assert embed_dim % 2 == 0, f"embed_dim must be even, got {embed_dim}"
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000.0 ** omega)  # [D/2]

    pos = pos.reshape(-1)  # [M]
    out = np.einsum("m,d->md", pos, omega)  # [M, D/2]

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # [M, D]


def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    grid_depth: int,
    cls_token: bool = False,
    uniform_power: bool = False,
) -> np.ndarray:
    """Generate 3D sine-cosine positional embeddings for spatiotemporal video tokens.
    
    Args:
        embed_dim: Total embedding dimensionality (e.g. 1024 for ViT-L).
        grid_size: Spatial patch grid size along H and W (e.g. 14 for 224x224 with 16x16 patches).
        grid_depth: Temporal tubelet depth (e.g. 32 for 64 frames with 2-frame tubelets).
        cls_token: Whether to prepend a zero CLS token positional embedding.
        uniform_power: If True, distributes dimension evenly across D, H, W.
        
    Returns:
        np.ndarray: Positional embedding of shape [grid_depth * grid_size * grid_size, embed_dim].
    """
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(grid_h, grid_d, grid_w)  # order [d, h, w]

    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim / 6.0) * 2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)

    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed


# ---------------------------------------------------------------------------
# Core Transformer Components (V-JEPA 2 ViT Architecture)
# ---------------------------------------------------------------------------

class PatchEmbed3D(nn.Module):
    """3D Spatiotemporal Convolutional Patch Embedding."""

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 1024,
        patch_size: int = 16,
        tubelet_size: int = 2,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize [B, C, T, H, W] tensor into [B, N, embed_dim] tokens."""
        B, C, T, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, T//tubelet_size, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        return x


class Attention(nn.Module):
    """Multi-Head Self-Attention with Scaled Dot-Product Attention (SDPA)."""

    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 16,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_sdpa = use_sdpa

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        if self.use_sdpa:
            # Scaled Dot-Product Attention: FlashAttention on CUDA, memory-efficient on CPU
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    """Two-layer feed-forward network with GELU activation."""

    def __init__(self, dim: int = 1024, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Transformer Encoder Block."""

    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, use_sdpa=use_sdpa)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SurgMotionViTLBackbone(nn.Module):
    """Vision Transformer Large (ViT-L/16) Video Backbone.
    
    Total parameters: ~303.9M. Matches CAIR-HKISI/SurgMotion-vitl.pt state dict keys exactly.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        num_frames: int = 64,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Grid dimensions
        self.grid_depth = num_frames // tubelet_size  # 32
        self.grid_h = img_size[0] // patch_size       # 14
        self.grid_w = img_size[1] // patch_size       # 14
        self.num_patches = self.grid_depth * self.grid_h * self.grid_w  # 6272

        # 1. 3D Patch embedding
        self.patch_embed = PatchEmbed3D(
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
        )

        # 2. Positional embedding buffer (non-trainable fixed 3D sincos)
        pos_embed = get_3d_sincos_pos_embed(
            embed_dim=embed_dim,
            grid_size=self.grid_h,
            grid_depth=self.grid_depth,
            cls_token=False,
        )
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)

        # 3. 24 Transformer blocks
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, use_sdpa=use_sdpa)
            for _ in range(depth)
        ])

        # 4. Final LayerNorm
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ViT-Large backbone.
        
        Args:
            x: Input video tensor of shape [B, 3, T, H, W] where T=64, H=224, W=224.
            
        Returns:
            torch.Tensor: Spatiotemporal token representations of shape [B, 6272, 1024].
        """
        B, C, T, H, W = x.shape
        x = self.patch_embed(x)  # [B, N, 1024]

        # Add 3D positional embedding
        if self.pos_embed.shape[1] == x.shape[1]:
            x = x + self.pos_embed.to(device=x.device, dtype=x.dtype)
        else:
            # Dynamically compute matching sincos embedding if clip length differs
            grid_depth = T // self.tubelet_size
            grid_size = H // self.patch_size
            dyn_pos = get_3d_sincos_pos_embed(self.embed_dim, grid_size, grid_depth)
            x = x + torch.from_numpy(dyn_pos).float().unsqueeze(0).to(device=x.device, dtype=x.dtype)

        # Pass through all 24 transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Apply final LayerNorm
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# SurgMotion Visual Encoder Interface (Conforms to BaseVisualEncoder)
# ---------------------------------------------------------------------------

class SurgMotionVisualEncoder(BaseVisualEncoder):
    """Frozen visual foundation encoder wrapping CAIR-HKISI SurgMotion ViT-Large.
    
    Adheres strictly to the BaseVisualEncoder interface to ensure seamless compatibility
    with existing downstream sequence classifiers and multimodal fusion modules.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "models/SurgMotion-L/SurgMotion-vitl.pt",
        clip_length: int = 64,
        img_size: Tuple[int, int] = (224, 224),
        pooling_mode: str = "spatiotemporal",
        device: Optional[Union[str, torch.device]] = None,
        use_sdpa: bool = True,
    ) -> None:
        """Initialize and freeze SurgMotion-ViTL encoder.
        
        Args:
            checkpoint_path: Path to the downloaded SurgMotion-vitl.pt checkpoint.
            clip_length: Input clip frame count (default: 64).
            img_size: Spatial (height, width) resolution (default: (224, 224)).
            pooling_mode: Pooling strategy:
                - 'spatiotemporal': Mean over all 6,272 tokens -> returns [B, 1024]
                - 'tubelet': Spatial mean over 196 tokens -> returns [B, 32, 1024]
                - 'tokens': Returns raw [B, 6272, 1024] tokens
            device: Target device (cuda/cpu).
            use_sdpa: Whether to use PyTorch scaled dot-product attention.
        """
        super().__init__(device=device)
        self._encoder_name = "SurgMotion-ViTL"
        self._output_dim = 1024
        self.clip_length = clip_length
        self.img_size = img_size
        self.pooling_mode = pooling_mode.lower().strip()
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"SurgMotion checkpoint not found at: {self.checkpoint_path.resolve()}\n"
                "Please verify that SurgMotion-vitl.pt exists in the expected models directory."
            )

        # 1. Instantiate backbone architecture
        self.backbone = SurgMotionViTLBackbone(
            img_size=img_size,
            num_frames=clip_length,
            embed_dim=self._output_dim,
            use_sdpa=use_sdpa,
        )

        # 2. Load checkpoint weights
        self._load_weights()

        # 3. Move to target device and freeze
        self.to(self.device)
        self.freeze()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def encoder_name(self) -> str:
        return self._encoder_name

    def _load_weights(self) -> None:
        """Load pretrained weights from checkpoint, mapping DDP state_dict keys."""
        # Memory-map the checkpoint on CPU to minimize RAM overhead
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", mmap=True)
        if isinstance(ckpt, dict) and "encoder" in ckpt:
            state_dict = ckpt["encoder"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            raise ValueError(f"Unrecognized checkpoint format in {self.checkpoint_path}")

        # Strip 'module.backbone.' or 'module.' or 'backbone.' prefix
        cleaned_sd = {}
        for k, v in state_dict.items():
            clean_k = k
            if clean_k.startswith("module.backbone."):
                clean_k = clean_k[len("module.backbone."):]
            elif clean_k.startswith("module."):
                clean_k = clean_k[len("module."):]
            elif clean_k.startswith("backbone."):
                clean_k = clean_k[len("backbone."):]
            cleaned_sd[clean_k] = v

        missing, unexpected = self.backbone.load_state_dict(cleaned_sd, strict=True)
        assert len(missing) == 0, f"Missing keys during SurgMotion loading: {missing}"
        assert len(unexpected) == 0, f"Unexpected keys during SurgMotion loading: {unexpected}"

    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract frozen 1024-D representations from video clips.
        
        Args:
            frames: Video tensor of shape:
                - [B, C, T, H, W] (channel-first video format)
                - [B, T, C, H, W] (batch-sequence-channel format, auto-permuted)
                - [C, T, H, W] (single clip, auto-batched)
                
        Returns:
            torch.Tensor: Extracted features:
                - If pooling_mode == 'spatiotemporal': shape [B, 1024]
                - If pooling_mode == 'tubelet': shape [B, 32, 1024]
                - If pooling_mode == 'tokens': shape [B, 6272, 1024]
                
        Raises:
            ValueError: If input contains NaNs/Infs, or shape is invalid.
        """
        if not torch.all(torch.isfinite(frames)):
            raise ValueError("Input frames tensor contains NaN or Inf values.")

        # Handle dimensions
        x = frames
        if x.dim() == 4:
            # [C, T, H, W] -> [1, C, T, H, W]
            x = x.unsqueeze(0)

        if x.dim() != 5:
            raise ValueError(
                f"Expected 5D video tensor [B, C, T, H, W] or [B, T, C, H, W], got shape {tuple(x.shape)}"
            )

        # If [B, T, C, H, W] where C=3 and T=64, permute to [B, C, T, H, W]
        if x.shape[1] == self.clip_length and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4)
        elif x.shape[1] == 3 and x.shape[2] == self.clip_length:
            pass  # Already [B, C, T, H, W]
        else:
            raise ValueError(
                f"Expected 3 channels and {self.clip_length} frames, got shape {tuple(x.shape)}"
            )

        orig_device = frames.device
        x = x.to(self.device)

        with torch.no_grad():
            tokens = self.backbone(x)  # [B, 6272, 1024]

            if self.pooling_mode == "spatiotemporal":
                # Average pool across all 6,272 spatiotemporal tokens
                feats = tokens.mean(dim=1)  # [B, 1024]
            elif self.pooling_mode == "tubelet":
                # Reshape to [B, T_tokens=32, S_tokens=196, 1024] and pool across spatial patches
                B, N, D = tokens.shape
                grid_depth = self.backbone.grid_depth
                grid_spatial = self.backbone.grid_h * self.backbone.grid_w
                tokens_reshaped = tokens.view(B, grid_depth, grid_spatial, D)
                feats = tokens_reshaped.mean(dim=2)  # [B, 32, 1024]
            elif self.pooling_mode == "tokens":
                feats = tokens  # [B, 6272, 1024]
            else:
                raise ValueError(f"Unknown pooling_mode: '{self.pooling_mode}'")

        if not torch.all(torch.isfinite(feats)):
            raise RuntimeError("Extracted SurgMotion features contain NaN or Inf values.")

        return feats.to(orig_device)
