"""Standalone SurgMotion Feature Extraction Pipeline for Remote GPU / Local Execution.

Extracts frozen spatiotemporal representations from JIGSAWS surgical video trials using
the official SurgMotion ViT-Large foundation model (Wu et al., 2026).

Designed for reproducible execution on the mentor's GPU machine as well as local environments.
Caches are saved to an isolated directory ('data/cache/features_surgmotion') to preserve
all existing ResNet-18 benchmark artifacts.

Usage:
    # 1-clip GPU/CPU smoke test (measure throughput, VRAM, and verify outputs):
    python scripts/extract_surgmotion_features.py --smoke-test

    # Extract single pilot trial on GPU (e.g. Suturing_B001):
    python scripts/extract_surgmotion_features.py --task Suturing --trial-id Suturing_B001 --fp16

    # Extract all Suturing trials on GPU with batch size 8:
    python scripts/extract_surgmotion_features.py --task Suturing --batch-size 8 --fp16
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.manifest import build_manifest
from src.models.surgmotion_encoder import SurgMotionVisualEncoder


def detect_attention_backends(device: torch.device) -> Dict[str, bool]:
    """Detect available PyTorch SDPA attention backends on current device."""
    backends = {
        "flash_attention": False,
        "mem_efficient": False,
        "math": True,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        if hasattr(torch.backends.cuda, "flash_sdp_enabled"):
            backends["flash_attention"] = torch.backends.cuda.flash_sdp_enabled()
        if hasattr(torch.backends.cuda, "mem_efficient_sdp_enabled"):
            backends["mem_efficient"] = torch.backends.cuda.mem_efficient_sdp_enabled()
        if hasattr(torch.backends.cuda, "math_sdp_enabled"):
            backends["math"] = torch.backends.cuda.math_sdp_enabled()
    return backends


def compute_file_sha256(filepath: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute SHA256 checksum of a file without reading the whole file into RAM."""
    p = Path(filepath)
    if not p.is_file():
        return "NOT_FOUND"
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_trial_clips(
    video_path: Path,
    encoder: SurgMotionVisualEncoder,
    clip_length: int = 64,
    stride: int = 16,
    batch_size: int = 4,
    fp16: bool = False,
    limit_clips: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Stream video frames and extract windowed features in batches without loading whole video.
    
    Args:
        video_path: Path to .avi video file.
        encoder: Frozen SurgMotionVisualEncoder instance.
        clip_length: Clip temporal length (default: 64).
        stride: Temporal window step between clips (default: 16).
        batch_size: Mini-batch size of clips passed to encoder.
        fp16: Whether to use FP16 mixed precision on CUDA.
        limit_clips: Optional maximum clips to process (for debugging).
        device: Target execution device.
        
    Returns:
        features: [N_clips, D] or [N_clips, T_tokens, D] float32 array.
        center_frames: [N_clips] int array of clip center frame indices (1-indexed).
        start_frames: [N_clips] int array of clip start frame indices (1-indexed).
        end_frames: [N_clips] int array of clip end frame indices (1-indexed).
        elapsed_sec: Time spent in model inference.
    """
    dev = device or encoder.device
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate clip window boundaries (1-indexed to match JIGSAWS kinematics convention)
    starts = list(range(1, max(1, total_frames - clip_length + 2), stride))
    if limit_clips is not None and limit_clips > 0:
        starts = starts[:limit_clips]

    if len(starts) == 0:
        cap.release()
        raise ValueError(f"Video {video_path.name} has only {total_frames} frames, shorter than clip_length={clip_length}")

    # Sliding frame buffer
    frame_buffer: Dict[int, torch.Tensor] = {}
    extracted_features: List[np.ndarray] = []
    clip_starts: List[int] = []
    clip_ends: List[int] = []
    clip_centers: List[int] = []

    current_frame_idx = 1
    total_infer_time = 0.0

    batch_clips: List[torch.Tensor] = []
    batch_meta: List[Tuple[int, int, int]] = []

    for s in starts:
        e = s + clip_length - 1
        c = (s + e) // 2

        # Advance video capture to load needed frames
        while current_frame_idx <= e:
            ret, frame = cap.read()
            if not ret:
                break
            # Preprocess: BGR -> RGB, resize to (224, 224), ImageNet norm
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t_frame = encoder.preprocess_image(frame_rgb, target_size=(224, 224))  # [1, 3, 224, 224]
            frame_buffer[current_frame_idx] = t_frame.squeeze(0)  # [3, 224, 224]
            current_frame_idx += 1

        # Check if full clip is available in buffer
        if all(f_idx in frame_buffer for f_idx in range(s, e + 1)):
            clip_tensor = torch.stack([frame_buffer[f_idx] for f_idx in range(s, e + 1)], dim=0)  # [64, 3, 224, 224]
            batch_clips.append(clip_tensor)
            batch_meta.append((s, e, c))

        # Evict frames from buffer that will never be needed again
        min_needed = s
        keys_to_evict = [k for k in frame_buffer.keys() if k < min_needed]
        for k in keys_to_evict:
            del frame_buffer[k]

        # Execute mini-batch forward pass when batch is full
        if len(batch_clips) >= batch_size:
            b_tensor = torch.stack(batch_clips, dim=0)  # [B, 64, 3, 224, 224]
            t0 = time.time()
            with torch.no_grad():
                if dev.type == "cuda" and fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        feats = encoder.extract_features(b_tensor)
                else:
                    feats = encoder.extract_features(b_tensor)
            total_infer_time += (time.time() - t0)

            extracted_features.append(feats.cpu().float().numpy())
            for ms, me, mc in batch_meta:
                clip_starts.append(ms)
                clip_ends.append(me)
                clip_centers.append(mc)

            batch_clips.clear()
            batch_meta.clear()

    # Process remaining clips in last partial batch
    if len(batch_clips) > 0:
        b_tensor = torch.stack(batch_clips, dim=0)
        t0 = time.time()
        with torch.no_grad():
            if dev.type == "cuda" and fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = encoder.extract_features(b_tensor)
            else:
                feats = encoder.extract_features(b_tensor)
        total_infer_time += (time.time() - t0)

        extracted_features.append(feats.cpu().float().numpy())
        for ms, me, mc in batch_meta:
            clip_starts.append(ms)
            clip_ends.append(me)
            clip_centers.append(mc)

        batch_clips.clear()
        batch_meta.clear()

    cap.release()

    if len(extracted_features) == 0:
        raise RuntimeError(f"No valid clips could be extracted from {video_path}")

    all_feats = np.concatenate(extracted_features, axis=0)
    return (
        all_feats,
        np.array(clip_centers, dtype=np.int32),
        np.array(clip_starts, dtype=np.int32),
        np.array(clip_ends, dtype=np.int32),
        total_infer_time,
    )


def save_surgmotion_cache(
    output_path: Path,
    features: np.ndarray,
    center_frames: np.ndarray,
    start_frames: np.ndarray,
    end_frames: np.ndarray,
    metadata: Dict[str, Any],
) -> None:
    """Atomically save extracted SurgMotion features and metadata to .npz archive."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.npz")

    # Encode metadata dict as JSON string array to store cleanly in npz
    meta_json = json.dumps(metadata, indent=2)

    np.savez_compressed(
        tmp_path,
        features=features.astype(np.float32),
        frame_indices=center_frames,  # Standard key for downstream loader
        clip_center_frames=center_frames,
        clip_start_frames=start_frames,
        clip_end_frames=end_frames,
        metadata_json=np.array(meta_json, dtype=object),
    )

    if output_path.exists():
        output_path.unlink()
    tmp_path.rename(output_path)


def run_extraction(args: argparse.Namespace) -> None:
    """Main extraction driver function."""
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 70)
    print("SurgMotion Foundation Feature Extraction Engine")
    print("=" * 70)
    print(f"Target Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else platform.processor()})")
    print(f"PyTorch Version: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")

    # Detect attention backends
    attn_backends = detect_attention_backends(device)
    print(f"Attention Backends: FlashAttention={attn_backends['flash_attention']}, "
          f"MemEfficient={attn_backends['mem_efficient']}, Math={attn_backends['math']}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path.resolve()}")

    print(f"Loading SurgMotion backbone from: {ckpt_path.resolve()} ...")
    t0_load = time.time()
    encoder = SurgMotionVisualEncoder(
        checkpoint_path=ckpt_path,
        clip_length=args.clip_length,
        pooling_mode=args.pooling_mode,
        device=device,
        use_sdpa=True,
    )
    load_time = time.time() - t0_load
    print(f"Model loaded and frozen in {load_time:.2f}s (Parameters: {sum(p.numel() for p in encoder.parameters()):,})")

    # 1. Smoke-test mode: process 1 single clip and report diagnostics
    if args.smoke_test:
        print("\n--- Running Smoke Test (1 clip) ---")
        dummy_clip = torch.zeros(1, args.clip_length, 3, 224, 224, dtype=torch.float32)
        t0 = time.time()
        with torch.no_grad():
            if device.type == "cuda" and args.fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = encoder.extract_features(dummy_clip)
            else:
                feats = encoder.extract_features(dummy_clip)
        elapsed = time.time() - t0

        print(f"Smoke Test Result: SUCCESS")
        print(f"Input shape: {tuple(dummy_clip.shape)}")
        print(f"Output shape: {tuple(feats.shape)}")
        print(f"Feature dim: {feats.shape[-1]}")
        print(f"Finite check: {torch.all(torch.isfinite(feats)).item()}")
        print(f"Elapsed inference time: {elapsed:.3f}s")
        if device.type == "cuda":
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"Peak GPU VRAM allocated: {vram_mb:.1f} MB")
        print("Smoke test complete. Exiting.")
        return

    # 2. Build or load manifest
    manifest_csv = Path("artifacts/manifest.csv")
    if manifest_csv.is_file():
        df = pd.read_csv(manifest_csv)
    else:
        df = build_manifest()

    if args.task != "all":
        df = df[df["task"] == args.task].reset_index(drop=True)

    if args.trial_id:
        df = df[df["trial_id"] == args.trial_id].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"Trial '{args.trial_id}' not found in manifest.")

    if args.limit_trials and args.limit_trials > 0:
        df = df.head(args.limit_trials).reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nProcessing {len(df)} trials. Caching to: {output_dir.resolve()}")
    print(f"Clip length: {args.clip_length} frames | Stride: {args.stride} frames | Pooling: {args.pooling_mode}")

    summary_records = []

    for idx, row in df.iterrows():
        trial_id = row["trial_id"]
        task = row["task"]
        video_col = f"video_{args.camera}_path"
        rel_video = row.get(video_col, None)
        if not rel_video or not Path(rel_video).is_file():
            print(f"[{idx+1}/{len(df)}] Skipping {trial_id}: video not found at {rel_video}")
            continue

        video_path = Path(rel_video)
        out_filename = f"{trial_id}_{args.camera}.npz"
        cache_file = output_dir / out_filename

        if cache_file.is_file() and not args.force:
            print(f"[{idx+1}/{len(df)}] Cache already exists for {trial_id}: {cache_file.name} (use --force to overwrite)")
            continue

        print(f"\n[{idx+1}/{len(df)}] Extracting features for {trial_id} ({video_path.name}) ...")
        t_start_trial = time.time()

        feats, centers, starts, ends, infer_time = extract_trial_clips(
            video_path=video_path,
            encoder=encoder,
            clip_length=args.clip_length,
            stride=args.stride,
            batch_size=args.batch_size,
            fp16=args.fp16,
            limit_clips=args.limit_clips,
            device=device,
        )
        total_trial_sec = time.time() - t_start_trial

        # Verify finite values
        if not np.all(np.isfinite(feats)):
            raise RuntimeError(f"NaN/Inf detected in extracted features for {trial_id}!")

        # Compile metadata
        meta = {
            "trial_id": trial_id,
            "task": task,
            "camera": args.camera,
            "source_video": str(video_path),
            "encoder_name": encoder.encoder_name,
            "checkpoint_path": str(ckpt_path),
            "clip_length": args.clip_length,
            "stride": args.stride,
            "pooling_mode": args.pooling_mode,
            "feature_dim": int(feats.shape[-1]),
            "num_clips": int(len(centers)),
            "device": str(device),
            "fp16": bool(args.fp16),
            "batch_size": int(args.batch_size),
            "inference_time_sec": float(round(infer_time, 2)),
            "total_time_sec": float(round(total_trial_sec, 2)),
            "sec_per_clip": float(round(infer_time / len(centers), 4)),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "extracted_at": datetime.datetime.now().isoformat(),
        }

        save_surgmotion_cache(
            output_path=cache_file,
            features=feats,
            center_frames=centers,
            start_frames=starts,
            end_frames=ends,
            metadata=meta,
        )

        size_mb = cache_file.stat().st_size / (1024 * 1024)
        print(f"Saved: {cache_file.name} ({len(centers)} clips, shape {feats.shape}, {size_mb:.2f} MB)")
        print(f"Timing: Total={total_trial_sec:.1f}s, Inference={infer_time:.1f}s ({infer_time/len(centers):.3f}s/clip)")

        summary_records.append(meta)

    # Save extraction session summary
    if summary_records:
        summary_path = output_dir / f"extraction_summary_{args.task}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_records, f, indent=2)
        print(f"\nExtraction summary written to: {summary_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SurgMotion features from JIGSAWS videos.")
    parser.add_argument("--checkpoint", type=str, default="models/SurgMotion-L/SurgMotion-vitl.pt",
                        help="Path to SurgMotion-vitl.pt checkpoint")
    parser.add_argument("--task", type=str, default="Suturing",
                        choices=["Suturing", "Knot_Tying", "Needle_Passing", "all"],
                        help="Target surgical task")
    parser.add_argument("--trial-id", type=str, default=None,
                        help="Optional specific trial ID (e.g. 'Suturing_B001')")
    parser.add_argument("--limit-trials", type=int, default=None,
                        help="Optional maximum number of trials to extract")
    parser.add_argument("--limit-clips", type=int, default=None,
                        help="Optional maximum number of clips per trial (for debugging)")
    parser.add_argument("--clip-length", type=int, default=64,
                        help="Temporal window length in frames (default: 64)")
    parser.add_argument("--stride", type=int, default=16,
                        help="Temporal window stride in frames (default: 16)")
    parser.add_argument("--pooling-mode", type=str, default="spatiotemporal",
                        choices=["spatiotemporal", "tubelet", "tokens"],
                        help="Feature pooling mode")
    parser.add_argument("--camera", type=str, default="capture1",
                        choices=["capture1", "capture2"],
                        help="Video camera stream to extract")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size of clips during forward pass")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable FP16 autocast (recommended on CUDA GPUs)")
    parser.add_argument("--device", type=str, default=None,
                        help="Execution device ('cuda', 'cuda:0', 'cpu')")
    parser.add_argument("--output-dir", type=str, default="data/cache/features_surgmotion",
                        help="Directory to store extracted .npz cache files")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run single-clip diagnostic test and exit")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing cache files")
    return parser.parse_args()


if __name__ == "__main__":
    run_extraction(parse_args())
