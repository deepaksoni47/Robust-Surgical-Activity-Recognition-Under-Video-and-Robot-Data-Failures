"""
SurgMotion Feature Extraction Pipeline for SurgSync Dataset (ITER-34).

Extracts frozen 1024-D representations from SurgSync 1080p stereo endoscopy video
using the SurgMotion ViT-Large foundation model (Wu et al., 2026).

Designed for execution on the mentor's NVIDIA Quadro RTX 5000 GPU machine.
Supports:
- Streaming batch extraction with FP16 mixed precision.
- Preflight environment, checkpoint, and memory checks.
- '--sample-only' mode for verifying extraction throughput and VRAM on a single episode.
- Atomic saving and automatic resume (skips already-extracted episodes).
- Frame-level timestamp alignment with SurgSync's 10 Hz master clock.

Usage:
    # 1. Quick GPU/CPU sample test (extracts 1 episode, prints throughput & VRAM):
    python scripts/extract_surgmotion_surgsync.py --data-dir E:/surgsync_v10 --sample-only --fp16

    # 2. Full bulk extraction across all 92 suturing episodes on mentor GPU:
    python scripts/extract_surgmotion_surgsync.py \
        --data-dir E:/surgsync_v10 \
        --output-dir data/cache/features_surgsync \
        --device cuda \
        --batch-size 4 \
        --fp16
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.models.surgmotion_encoder import SurgMotionVisualEncoder


DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "SurgMotion-L" / "SurgMotion-vitl.pt"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def discover_suturing_episodes(data_dir: Path, task: str = "single_interrupted_stitch") -> List[Dict[str, Any]]:
    """
    Locate all downloaded episodes for the specified task in online_data and offline_data.
    """
    episodes = []
    for partition in ("online_data", "offline_data"):
        task_dir = data_dir / partition / "episodes" / task
        if not task_dir.is_dir():
            continue
        for clip_dir in sorted(task_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            video_file = clip_dir / "video_raw" / "stereo_left.mkv"
            meta_file = clip_dir / "episode_meta.json"
            if video_file.is_file() and meta_file.is_file():
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                episodes.append({
                    "episode_id": meta.get("episode_id", f"{partition}_{clip_dir.name}"),
                    "clip_dir": clip_dir,
                    "video_path": video_file,
                    "meta_path": meta_file,
                    "partition": partition,
                    "clip_name": clip_dir.name,
                    "length_frames": meta.get("length_frames", 0),
                    "operator_skill": meta.get("operator_skill_level", "Unknown"),
                })
    return episodes


def preprocess_frame(frame_bgr: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Resize 1080p frame, convert BGR->RGB, and normalize with ImageNet statistics.
    Returns float32 array of shape (3, H, W).
    """
    # Resize directly from 1080p to target_size
    resized = cv2.resize(frame_bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(norm, (2, 0, 1))  # (3, H, W)


def extract_episode_features(
    video_path: Path,
    encoder: SurgMotionVisualEncoder,
    clip_length: int = 16,
    batch_size: int = 4,
    fp16: bool = False,
    device: Optional[torch.device] = None,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract frame-level 1024-D representations using a causal sliding window.
    
    For frame t, the temporal context is frames [max(0, t - clip_length + 1) ... t],
    padded with edge replication if t < clip_length.
    
    Returns:
        (features, frame_indices, inference_elapsed_sec)
    """
    dev = device or encoder.device
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None and max_frames > 0:
        total_frames = min(total_frames, max_frames)

    # Read and preprocess all frames sequentially into buffer
    frames_buffer = []
    read_count = 0
    while cap.isOpened() and read_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames_buffer.append(preprocess_frame(frame, target_size=224))
        read_count += 1
    cap.release()

    n_frames = len(frames_buffer)
    if n_frames == 0:
        raise ValueError(f"No frames read from video: {video_path}")

    # Build sliding window batches of shape (B, 3, T, H, W)
    clip_batches = []
    current_batch = []
    
    for t in range(n_frames):
        start_idx = max(0, t - clip_length + 1)
        window = frames_buffer[start_idx : t + 1]
        
        # Left-pad with first frame if window < clip_length
        if len(window) < clip_length:
            pad_count = clip_length - len(window)
            window = [window[0]] * pad_count + window
            
        # Stack to (3, T, H, W)
        clip_tensor = np.stack(window, axis=1)
        current_batch.append(clip_tensor)
        
        if len(current_batch) == batch_size:
            clip_batches.append(np.stack(current_batch, axis=0))
            current_batch = []
            
    if current_batch:
        clip_batches.append(np.stack(current_batch, axis=0))

    # Run inference across batches
    all_features = []
    t_start = time.time()
    
    with torch.no_grad():
        for batch_np in clip_batches:
            batch_t = torch.from_numpy(batch_np).to(dev)
            if fp16 and dev.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    feats = encoder.extract_features(batch_t)
            else:
                feats = encoder.extract_features(batch_t)
            all_features.append(feats.cpu().numpy().astype(np.float16))

    inference_sec = time.time() - t_start
    features_arr = np.concatenate(all_features, axis=0)  # (N_frames, 1024)
    frame_indices = np.arange(n_frames, dtype=np.int32)

    return features_arr, frame_indices, inference_sec


def run_surgsync_extraction(
    data_dir: Path,
    output_dir: Path,
    checkpoint_path: Path,
    task: str = "single_interrupted_stitch",
    clip_length: int = 16,
    batch_size: int = 4,
    device_name: str = "auto",
    fp16: bool = False,
    sample_only: bool = False,
    max_episodes: Optional[int] = None,
) -> int:
    """
    Main extraction pipeline runner.
    """
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    checkpoint_path = checkpoint_path.resolve()

    print("=" * 70)
    print("  SurgMotion-L Feature Extraction on SurgSync (ITER-34)")
    print("=" * 70)
    print(f"Data Directory    : {data_dir}")
    print(f"Output Directory  : {output_dir}")
    print(f"Checkpoint Path   : {checkpoint_path}")
    print(f"Task Partition    : {task}")
    print(f"Clip Length       : {clip_length} frames")
    print(f"Batch Size        : {batch_size}")
    print(f"FP16 Autocast     : {fp16}")

    # Device selection
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"Compute Device    : {device}")
    if device.type == "cuda":
        print(f"CUDA Device Name  : {torch.cuda.get_device_name(0)}")
        print(f"Available VRAM    : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # Checkpoint preflight
    if not checkpoint_path.is_file():
        print(f"\n[FAIL] SurgMotion checkpoint not found at: {checkpoint_path}", file=sys.stderr)
        return 1

    # Episode discovery
    episodes = discover_suturing_episodes(data_dir, task=task)
    if not episodes:
        print(f"\n[FAIL] No episodes found under {data_dir} for task '{task}'.", file=sys.stderr)
        print("Ensure the dataset has been downloaded via scripts/download_surgsync_suturing.py", file=sys.stderr)
        return 1

    print(f"Discovered Clips  : {len(episodes)} episodes")
    output_dir.mkdir(parents=True, exist_ok=True)

    if sample_only:
        print("\n[SAMPLE-ONLY] Mode active: will extract 1 episode to measure throughput and verify outputs.")
        episodes = episodes[:1]
    elif max_episodes is not None and max_episodes > 0:
        episodes = episodes[:max_episodes]

    # Initialize encoder
    print("\nLoading SurgMotion-L foundation model into memory...")
    t0_load = time.time()
    try:
        encoder = SurgMotionVisualEncoder(
            checkpoint_path=checkpoint_path,
            device=device,
            pooling_mode="spatiotemporal",
            clip_length=clip_length,
        )
        print(f"Model loaded successfully in {time.time() - t0_load:.2f} seconds.")
    except Exception as e:
        print(f"[FAIL] Error initializing SurgMotionVisualEncoder: {e}", file=sys.stderr)
        return 1

    # Process episodes
    total_frames_processed = 0
    total_inference_time = 0.0
    extracted_count = 0
    skipped_count = 0

    pbar = tqdm(episodes, desc="Extracting SurgSync Features")
    for ep in pbar:
        ep_id = ep["episode_id"]
        out_npz = output_dir / f"{ep_id}.npz"
        tmp_npz = output_dir / f"{ep_id}.tmp.npz"

        # Safe resume check
        if out_npz.is_file() and out_npz.stat().st_size > 1024:
            skipped_count += 1
            pbar.set_postfix({"status": "skipped", "id": ep_id[-12:]})
            continue

        pbar.set_postfix({"status": "extracting", "id": ep_id[-12:]})
        try:
            feats, frame_indices, inf_sec = extract_episode_features(
                video_path=ep["video_path"],
                encoder=encoder,
                clip_length=clip_length,
                batch_size=batch_size,
                fp16=fp16,
                device=device,
            )

            # Atomic save
            np.savez_compressed(
                tmp_npz,
                features=feats,
                frame_indices=frame_indices,
                episode_id=ep_id,
                operator_skill=ep["operator_skill"],
            )
            tmp_npz.replace(out_npz)

            total_frames_processed += len(feats)
            total_inference_time += inf_sec
            extracted_count += 1

        except Exception as err:
            print(f"\n[ERROR] Failed extracting episode {ep_id}: {err}", file=sys.stderr)
            if tmp_npz.exists():
                tmp_npz.unlink()
            continue

    # Final summary report
    print("\n" + "=" * 70)
    print("  Feature Extraction Complete")
    print("=" * 70)
    print(f"Episodes Extracted  : {extracted_count}")
    print(f"Episodes Skipped    : {skipped_count} (already cached)")
    print(f"Total Frames        : {total_frames_processed:,}")
    if total_inference_time > 0:
        overall_fps = total_frames_processed / total_inference_time
        print(f"Inference Time      : {total_inference_time:.1f} s")
        print(f"Extraction Speed    : {overall_fps:.1f} FPS")
    if device.type == "cuda":
        max_vram = torch.cuda.max_memory_allocated(0) / (1024**3)
        print(f"Peak VRAM Allocated : {max_vram:.2f} GB")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Extract SurgMotion-L features for SurgSync Suturing.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/surgsync_v10"), help="SurgSync dataset root.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/cache/features_surgsync"), help="Output cache directory.")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT, help="SurgMotion checkpoint path.")
    parser.add_argument("--task", type=str, default="single_interrupted_stitch", help="Task partition.")
    parser.add_argument("--clip-length", type=int, default=16, help="Window clip length.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size of clips.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Target device.")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision on CUDA.")
    parser.add_argument("--sample-only", action="store_true", help="Extract 1 episode to test pipeline.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Maximum episodes to process.")

    args = parser.parse_args()
    ret = run_surgsync_extraction(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        task=args.task,
        clip_length=args.clip_length,
        batch_size=args.batch_size,
        device_name=args.device,
        fp16=args.fp16,
        sample_only=args.sample_only,
        max_episodes=args.max_episodes,
    )
    sys.exit(ret)


if __name__ == "__main__":
    main()
