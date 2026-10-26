"""Unit and integration tests for SurgMotion Visual Foundation Encoder."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.surgmotion_encoder import (
    Attention,
    Block,
    MLP,
    PatchEmbed3D,
    SurgMotionViTLBackbone,
    SurgMotionVisualEncoder,
    get_3d_sincos_pos_embed,
)
from scripts.validate_surgmotion_cache import validate_single_cache


def test_get_3d_sincos_pos_embed_shape_and_finite():
    """Verify that 3D sincos positional embedding produces correct shape and finite values."""
    embed_dim = 1024
    grid_size = 14
    grid_depth = 32
    expected_tokens = grid_depth * grid_size * grid_size  # 6272

    pos_embed = get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth, cls_token=False)
    assert pos_embed.shape == (expected_tokens, embed_dim)
    assert np.all(np.isfinite(pos_embed)), "Positional embeddings must be finite"
    assert pos_embed.dtype == np.float64 or pos_embed.dtype == np.float32


def test_patch_embed3d_shape():
    """Verify 3D patch embedding produces correct tokenized shape."""
    embed_dim = 128
    patch_size = 16
    tubelet_size = 2
    in_chans = 3

    pe = PatchEmbed3D(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size, tubelet_size=tubelet_size)
    B, C, T, H, W = 2, 3, 8, 32, 32
    dummy_video = torch.randn(B, C, T, H, W)

    out = pe(dummy_video)
    expected_tokens = (T // tubelet_size) * (H // patch_size) * (W // patch_size)  # 4 * 2 * 2 = 16
    assert out.shape == (B, expected_tokens, embed_dim)
    assert torch.all(torch.isfinite(out))


def test_attention_sdpa():
    """Verify multi-head self-attention module forward pass and finiteness."""
    dim = 128
    num_heads = 4
    attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=True, use_sdpa=True)

    B, N = 2, 16
    x = torch.randn(B, N, dim)
    out = attn(x)

    assert out.shape == (B, N, dim)
    assert torch.all(torch.isfinite(out))


def test_mlp_forward():
    """Verify MLP forward pass shape and finiteness."""
    dim = 128
    mlp = MLP(dim=dim, mlp_ratio=4.0)

    B, N = 2, 16
    x = torch.randn(B, N, dim)
    out = mlp(x)

    assert out.shape == (B, N, dim)
    assert torch.all(torch.isfinite(out))


def test_block_forward():
    """Verify transformer block forward pass with residual connections."""
    dim = 128
    num_heads = 4
    blk = Block(dim=dim, num_heads=num_heads, mlp_ratio=4.0, use_sdpa=True)

    B, N = 2, 16
    x = torch.randn(B, N, dim)
    out = blk(x)

    assert out.shape == (B, N, dim)
    assert torch.all(torch.isfinite(out))


def test_surgmotion_backbone_small_instantiation():
    """Test scaled-down backbone instantiation and forward token shapes."""
    backbone = SurgMotionViTLBackbone(
        img_size=(64, 64),
        num_frames=8,
        patch_size=16,
        tubelet_size=2,
        embed_dim=128,
        depth=2,
        num_heads=4,
        use_sdpa=True,
    )

    B = 1
    dummy_clip = torch.randn(B, 3, 8, 64, 64)
    out = backbone(dummy_clip)

    # 4 temporal * 4 height * 4 width = 64 tokens
    assert out.shape == (B, 64, 128)
    assert torch.all(torch.isfinite(out))


def test_encoder_missing_checkpoint_raises():
    """Verify that initializing SurgMotionVisualEncoder with a nonexistent file fails loudly."""
    with pytest.raises(FileNotFoundError, match="SurgMotion checkpoint not found"):
        SurgMotionVisualEncoder(checkpoint_path="nonexistent_checkpoint.pt")


def test_validate_surgmotion_cache_valid_and_corrupt():
    """Test validate_single_cache on valid and corrupted synthetic .npz caches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Valid cache
        valid_cache = tmp_path / "valid.npz"
        features = np.random.randn(10, 1024).astype(np.float32)
        frame_indices = np.arange(16, 16 + 10 * 16, 16, dtype=np.int32)
        np.savez(
            valid_cache,
            features=features,
            frame_indices=frame_indices,
            clip_center_frames=frame_indices,
            metadata_json='{"encoder_name": "SurgMotion-ViTL", "stride": 16}',
        )

        res = validate_single_cache(valid_cache, expected_dim=1024)
        assert res["status"] == "VALID"
        assert res["num_clips"] == 10
        assert res["feature_dim"] == 1024

        # 2. Corrupt cache with NaNs
        nan_cache = tmp_path / "nan.npz"
        bad_features = features.copy()
        bad_features[2, 5] = np.nan
        np.savez(nan_cache, features=bad_features, frame_indices=frame_indices)

        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_single_cache(nan_cache, expected_dim=1024)

        # 3. Corrupt cache with non-monotonic frames
        nonmono_cache = tmp_path / "nonmono.npz"
        bad_frames = np.array([10, 20, 15, 30], dtype=np.int32)
        np.savez(nonmono_cache, features=features[:4], frame_indices=bad_frames)

        with pytest.raises(ValueError, match="not strictly monotonically increasing"):
            validate_single_cache(nonmono_cache, expected_dim=1024)
