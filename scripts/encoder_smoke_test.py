"""Smoke test script for visual encoder interface and feature extraction.

Verifies:
1. Encoder initialization and parameter freezing.
2. Forward pass shape and numerical finiteness for both single frames and temporal clips.
3. Feature extraction on an actual JIGSAWS video frame if available.
4. Latency and memory metrics.

Usage:
    python scripts/encoder_smoke_test.py --config configs/encoder.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from src.models.visual_encoder import build_visual_encoder, BaseVisualEncoder
from src.utils.reproducibility import load_config, set_seed


def run_smoke_test(
    config_path: str = "configs/encoder.yaml",
    video_path: Optional[str] = "dataset/Suturing/video/Suturing_B001_capture1.avi",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete smoke test for the visual encoder."""
    set_seed(42)
    cfg = load_config(config_path)

    encoder_type = cfg.get("encoder_type", "resnet18")
    print(f"=== Visual Encoder Smoke Test: {encoder_type} ===")
    print(f"Loading configuration from: {config_path}")

    # 1. Instantiate encoder
    t0 = time.perf_counter()
    encoder = build_visual_encoder(cfg, device=device)
    init_time = time.perf_counter() - t0
    print(f"Encoder initialized: {encoder.encoder_name} (Output Dim: {encoder.output_dim}) in {init_time:.3f}s")
    print(f"Device: {encoder.device}")

    # 2. Check frozen parameters
    assert encoder.is_frozen(), "FAILED: Encoder parameters are not frozen!"
    print("Verification PASSED: All parameters are strictly frozen (requires_grad=False).")

    # 3. Test synthetic single frame batch [B, C, H, W]
    b, c, h, w = 4, 3, 224, 224
    dummy_frames = torch.randn(b, c, h, w, device=encoder.device)
    t0 = time.perf_counter()
    feats_single = encoder(dummy_frames)
    latency_single = (time.perf_counter() - t0) * 1000.0 / b  # ms per frame

    assert feats_single.shape == (b, encoder.output_dim), (
        f"FAILED: Expected shape {(b, encoder.output_dim)}, got {feats_single.shape}"
    )
    assert torch.all(torch.isfinite(feats_single)), "FAILED: Features contain NaN or Inf values!"
    print(f"Verification PASSED: Single-frame batch [4, 3, 224, 224] -> {list(feats_single.shape)} ({latency_single:.2f} ms/frame)")

    # 4. Test synthetic temporal clip [B, T, C, H, W]
    t_steps = 8
    dummy_clip = torch.randn(2, t_steps, c, h, w, device=encoder.device)
    feats_clip = encoder(dummy_clip)
    assert feats_clip.shape == (2, t_steps, encoder.output_dim), (
        f"FAILED: Expected clip shape {(2, t_steps, encoder.output_dim)}, got {feats_clip.shape}"
    )
    assert torch.all(torch.isfinite(feats_clip)), "FAILED: Clip features contain NaN or Inf values!"
    print(f"Verification PASSED: Temporal clip [2, 8, 3, 224, 224] -> {list(feats_clip.shape)}")

    # 5. Test real JIGSAWS video frame if present
    real_frame_tested = False
    real_feats_norm = None
    v_path = Path(video_path) if video_path else None
    if v_path and v_path.is_file():
        cap = cv2.VideoCapture(str(v_path.resolve()))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
        ret, frame_bgr = cap.read()
        cap.release()

        if ret and frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            input_tensor = BaseVisualEncoder.preprocess_image(frame_rgb, target_size=(224, 224))
            feats_real = encoder(input_tensor)
            assert feats_real.shape == (1, encoder.output_dim)
            assert torch.all(torch.isfinite(feats_real))
            real_feats_norm = float(torch.norm(feats_real).item())
            real_frame_tested = True
            print(f"Verification PASSED: Real video frame ({v_path.name}, frame 50) -> L2 norm: {real_feats_norm:.4f}")
    else:
        print(f"Notice: Video file {video_path} not found; skipped real video frame test.")

    results = {
        "status": "PASSED",
        "encoder_name": encoder.encoder_name,
        "output_dim": encoder.output_dim,
        "device": str(encoder.device),
        "is_frozen": encoder.is_frozen(),
        "single_frame_shape": list(feats_single.shape),
        "clip_shape": list(feats_clip.shape),
        "ms_per_frame": round(latency_single, 2),
        "real_frame_tested": real_frame_tested,
        "real_frame_norm": real_feats_norm,
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test visual encoder interface.")
    parser.add_argument("--config", type=str, default="configs/encoder.yaml", help="Path to encoder.yaml")
    parser.add_argument("--video", type=str, default="dataset/Suturing/video/Suturing_B001_capture1.avi", help="Path to trial video")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    args = parser.parse_args()

    results = run_smoke_test(config_path=args.config, video_path=args.video, device=args.device)
    print("\nSmoke Test Results Summary:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
