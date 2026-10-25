from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch

from src.data.kinematics import load_kinematics
from src.data.transcriptions import parse_transcription
from src.data.video import get_video_metadata
from src.models.visual_encoder import BaseVisualEncoder, build_visual_encoder
from src.utils.reproducibility import load_config


def get_cache_path(
    cache_dir: Union[str, Path],
    trial_id: str,
    encoder_name: str,
    camera: str = "capture1",
) -> Path:
    """Return the canonical, deterministic filepath for a trial's cached features.
    
    Format:
        <cache_dir>/<encoder_name>/<trial_id>_<camera>.npz
    """
    clean_encoder = encoder_name.replace(" ", "_").replace("/", "_")
    return Path(cache_dir) / clean_encoder / f"{trial_id}_{camera}.npz"


def validate_feature_cache(
    cache_path: Union[str, Path],
    expected_dim: Optional[int] = None,
    expected_trial_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate integrity, finiteness, shapes, and metadata of a cached feature file.
    
    Args:
        cache_path: Path to the .npz cache file.
        expected_dim: Optional expected feature vector dimension (e.g. 512).
        expected_trial_id: Optional expected trial identifier (e.g. 'Suturing_B001').
        
    Returns:
        Dict[str, Any]: Summary dictionary with cache verification details.
        
    Raises:
        FileNotFoundError: If cache file does not exist.
        ValueError: If file is empty, corrupted, missing required keys, or contains NaNs/Infs.
    """
    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cache file not found: {path.resolve()}")

    if path.stat().st_size == 0:
        raise ValueError(f"Cache file is empty (0 bytes): {path.name}")

    try:
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            required_keys = {"features", "frame_indices"}
            missing_keys = required_keys - keys
            if missing_keys:
                raise ValueError(f"Cache file {path.name} is missing required arrays: {missing_keys}")

            features = data["features"]
            frame_indices = data["frame_indices"]

            # 1. Dimensionality and shape checks
            if features.ndim != 2:
                raise ValueError(f"Expected 2D features array [N, D], got shape {features.shape}")

            n_samples, feat_dim = features.shape
            if n_samples == 0:
                raise ValueError(f"Cache {path.name} contains 0 feature rows.")

            if expected_dim is not None and feat_dim != expected_dim:
                raise ValueError(f"Feature dimension mismatch in {path.name}: expected {expected_dim}, got {feat_dim}")

            # 2. Frame indices consistency
            if frame_indices.ndim != 1:
                raise ValueError(f"Expected 1D frame_indices array [N], got shape {frame_indices.shape}")

            if len(frame_indices) != n_samples:
                raise ValueError(
                    f"Length mismatch in {path.name}: features has {n_samples} rows, "
                    f"but frame_indices has {len(frame_indices)} elements."
                )

            # Monotonicity check
            if not np.all(np.diff(frame_indices) > 0):
                raise ValueError(f"frame_indices in {path.name} are not strictly monotonically increasing.")

            # 3. Finite value assertion
            if not np.all(np.isfinite(features)):
                raise ValueError(f"Features in {path.name} contain non-finite values (NaN or Inf).")

            # 4. Metadata verification
            metadata: Dict[str, Any] = {}
            if "metadata_json" in data:
                meta_str = str(data["metadata_json"])
                try:
                    metadata = json.loads(meta_str)
                except Exception:
                    metadata = {}

            cached_trial = metadata.get("trial_id", data.get("trial_id", None))
            if cached_trial is not None and isinstance(cached_trial, np.ndarray):
                cached_trial = str(cached_trial)

            if expected_trial_id is not None and cached_trial is not None:
                if str(cached_trial) != expected_trial_id:
                    raise ValueError(
                        f"Trial ID mismatch in {path.name}: expected '{expected_trial_id}', got '{cached_trial}'"
                    )

            return {
                "valid": True,
                "cache_path": str(path).replace("\\", "/"),
                "trial_id": cached_trial or path.stem.split("_capture")[0],
                "num_frames": int(n_samples),
                "feature_dim": int(feat_dim),
                "frame_indices_range": [int(frame_indices[0]), int(frame_indices[-1])],
                "file_size_bytes": path.stat().st_size,
                "metadata": metadata,
            }

    except Exception as e:
        if isinstance(e, (ValueError, FileNotFoundError)):
            raise
        raise ValueError(f"Failed to read/decompress cache archive {path.name}: {e}") from e


def save_feature_cache(
    cache_path: Union[str, Path],
    features: np.ndarray,
    frame_indices: np.ndarray,
    metadata: Dict[str, Any],
    overwrite: bool = False,
) -> Path:
    """Atomically save extracted features, frame indices, and metadata to compressed .npz archive.
    
    Uses atomic temporary-file creation and rename (os.replace) to ensure interrupted runs
    never leave corrupt or partially written cache archives.
    
    Args:
        cache_path: Target .npz filepath.
        features: 2D numpy array [N, D] of float32 features.
        frame_indices: 1D numpy array [N] of int32 frame indices.
        metadata: Comprehensive metadata dictionary.
        overwrite: If True, replaces existing file. If False and exists, raises FileExistsError.
        
    Returns:
        Path: Verified path to the saved cache file.
    """
    target = Path(cache_path)
    if target.is_file() and not overwrite:
        raise FileExistsError(f"Cache file already exists and overwrite=False: {target}")

    # Validate inputs before writing
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features array [N, D], got shape {features.shape}")
    if frame_indices.ndim != 1 or len(frame_indices) != len(features):
        raise ValueError(
            f"Shape mismatch: features len {len(features)} != frame_indices len {len(frame_indices)}"
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("Cannot save cache containing NaN or Inf features.")

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.parent / f"{target.stem}.tmp_{os.getpid()}_{np.random.randint(10000, 99999)}.npz"

    metadata_json = json.dumps(metadata, indent=2)

    try:
        np.savez_compressed(
            temp_path,
            features=features.astype(np.float32),
            frame_indices=frame_indices.astype(np.int32),
            metadata_json=metadata_json,
            trial_id=np.array(metadata.get("trial_id", "")),
            encoder_name=np.array(metadata.get("encoder_name", "")),
            output_dim=np.array(features.shape[1]),
        )

        # Verify temporary file integrity before atomic promotion
        validate_feature_cache(temp_path, expected_dim=features.shape[1])

        # Atomic rename
        os.replace(temp_path, target)

    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    return target


def load_feature_cache(
    cache_path: Union[str, Path],
    validate: bool = True,
    expected_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """Load cached features, indices, and metadata from an .npz archive.
    
    Args:
        cache_path: Path to the .npz cache file.
        validate: Whether to validate the cache before returning.
        expected_dim: Optional expected feature dimension.
        
    Returns:
        Dict containing 'features' (np.ndarray), 'frame_indices' (np.ndarray), and 'metadata' (dict).
    """
    path = Path(cache_path)
    if validate:
        validate_feature_cache(path, expected_dim=expected_dim)

    with np.load(path, allow_pickle=False) as data:
        features = data["features"].copy()
        frame_indices = data["frame_indices"].copy()
        metadata: Dict[str, Any] = {}
        if "metadata_json" in data:
            try:
                metadata = json.loads(str(data["metadata_json"]))
            except Exception:
                metadata = {}

    return {
        "features": features,
        "frame_indices": frame_indices,
        "metadata": metadata,
        "trial_id": metadata.get("trial_id", path.stem.split("_capture")[0]),
        "num_frames": features.shape[0],
        "output_dim": features.shape[1],
    }


def extract_frames_in_batches(
    video_path: Union[str, Path],
    target_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    batch_size: int = 32,
    max_frames: Optional[int] = None,
) -> Generator[Tuple[torch.Tensor, List[int]], None, None]:
    """Stream and preprocess video frames sequentially in batches from an AVI video.
    
    Avoids reading the entire video into memory by streaming frame-by-frame and yielding
    normalized PyTorch batches of shape [B, 3, H, W].
    
    Args:
        video_path: Path to the video file.
        target_size: (H, W) spatial resolution for model input.
        mean: Channel means for ImageNet normalization.
        std: Channel stds for ImageNet normalization.
        batch_size: Number of frames per mini-batch.
        max_frames: Optional upper bound on frames to process (useful for smoke tests).
        
    Yields:
        Tuple[torch.Tensor, List[int]]: Preprocessed batch tensor [B, 3, H, W] and list of 0-indexed frame IDs.
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path.resolve()}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV failed to open video: {path.name}")

    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    frame_idx = 0
    current_frames = []
    current_indices = []

    try:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break

            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            # Preprocessing: BGR -> RGB and resize
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if (frame_rgb.shape[0], frame_rgb.shape[1]) != target_size:
                resized = cv2.resize(frame_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
            else:
                resized = frame_rgb

            current_frames.append(resized)
            current_indices.append(frame_idx)
            frame_idx += 1

            if len(current_frames) == batch_size:
                arr = np.stack(current_frames).astype(np.float32) / 255.0
                tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)
                norm_tensor = (tensor - mean_t) / std_t
                yield norm_tensor, current_indices

                current_frames = []
                current_indices = []

        # Flush any remaining frames
        if current_frames:
            arr = np.stack(current_frames).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)
            norm_tensor = (tensor - mean_t) / std_t
            yield norm_tensor, current_indices

    finally:
        cap.release()


def extract_and_cache_trial(
    trial_row: Union[pd.Series, Dict[str, Any]],
    encoder: BaseVisualEncoder,
    cache_dir: Union[str, Path] = "data/cache/features",
    camera: str = "capture1",
    batch_size: int = 32,
    max_frames: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Extract and cache visual features for a single JIGSAWS trial.
    
    Supports resuming by automatically skipping valid existing caches unless force=True.
    
    Args:
        trial_row: Record from manifest (must contain trial_id, task, video paths).
        encoder: Instantiated, frozen BaseVisualEncoder.
        cache_dir: Destination directory for cached features.
        camera: 'capture1' (default left stereo) or 'capture2'.
        batch_size: Mini-batch size for frame streaming.
        max_frames: Optional cap on frames extracted.
        force: If True, re-extracts and overwrites any existing cache.
        
    Returns:
        Dict[str, Any]: Status and execution statistics dictionary.
    """
    trial_id = str(trial_row["trial_id"])
    task = str(trial_row.get("task", "Suturing"))
    target_cache = get_cache_path(cache_dir, trial_id, encoder.encoder_name, camera=camera)

    # 1. Skip if already validly cached and not force
    if target_cache.is_file() and not force:
        try:
            val_info = validate_feature_cache(
                target_cache, expected_dim=encoder.output_dim, expected_trial_id=trial_id
            )
            return {
                "status": "SKIPPED",
                "trial_id": trial_id,
                "cache_path": str(target_cache).replace("\\", "/"),
                "num_frames": val_info["num_frames"],
                "reason": "Valid cache already exists (use force=True to re-extract)",
            }
        except Exception:
            # Corrupted existing cache: proceed to re-extract
            pass

    # 2. Resolve video path from manifest
    vid_col = f"video_{camera}_path"
    if vid_col not in trial_row or not trial_row[vid_col]:
        raise ValueError(f"Missing {vid_col} in trial row for {trial_id}")

    video_path = Path(trial_row[vid_col])
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file missing: {video_path}")

    # Inspect video metadata
    vid_meta = get_video_metadata(video_path)
    total_vid_frames = vid_meta["n_frames"]
    fps = vid_meta["fps"]

    # 3. Check kinematics row count and transcription end frame for alignment metadata
    n_kin_samples: Optional[int] = None
    if "kinematics_path" in trial_row and trial_row["kinematics_path"]:
        kin_path = Path(trial_row["kinematics_path"])
        if kin_path.is_file():
            kin_arr = load_kinematics(kin_path)
            n_kin_samples = int(kin_arr.shape[0])

    n_trans_frames: Optional[int] = None
    if "transcription_path" in trial_row and trial_row["transcription_path"]:
        trans_path = Path(trial_row["transcription_path"])
        if trans_path.is_file():
            segments = parse_transcription(trans_path)
            n_trans_frames = int(segments[-1]["end_frame"]) if segments else 0

    # 4. Stream batches and extract features
    extracted_features_list: List[np.ndarray] = []
    extracted_indices_list: List[int] = []

    target_size = (224, 224)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    batch_generator = extract_frames_in_batches(
        video_path=video_path,
        target_size=target_size,
        mean=mean,
        std=std,
        batch_size=batch_size,
        max_frames=max_frames,
    )

    t_start = datetime.datetime.now(datetime.timezone.utc)

    for batch_tensor, batch_indices in batch_generator:
        with torch.inference_mode():
            batch_feats = encoder(batch_tensor)  # [B, D]

        feats_np = batch_feats.cpu().numpy().astype(np.float32)
        extracted_features_list.append(feats_np)
        extracted_indices_list.extend(batch_indices)

    if not extracted_features_list:
        raise RuntimeError(f"No frames could be extracted from {video_path.name}")

    all_features = np.vstack(extracted_features_list)
    all_indices = np.array(extracted_indices_list, dtype=np.int32)
    n_extracted = all_features.shape[0]

    t_end = datetime.datetime.now(datetime.timezone.utc)
    duration_sec = (t_end - t_start).total_seconds()

    # 5. Compile metadata
    metadata = {
        "trial_id": trial_id,
        "task": task,
        "camera": camera,
        "video_path": str(video_path).replace("\\", "/"),
        "encoder_name": encoder.encoder_name,
        "output_dim": int(encoder.output_dim),
        "num_frames": int(n_extracted),
        "total_video_frames": int(total_vid_frames),
        "n_kinematics_samples": n_kin_samples,
        "n_transcription_frames": n_trans_frames,
        "delta_vid_kin_frames": abs(total_vid_frames - n_kin_samples) if n_kin_samples else None,
        "fps": float(fps),
        "target_size": list(target_size),
        "mean": list(mean),
        "std": list(std),
        "extraction_duration_sec": round(duration_sec, 2),
        "extraction_fps": round(n_extracted / duration_sec, 2) if duration_sec > 0 else 0.0,
        "created_at": t_start.isoformat(),
        "index_convention": "0-indexed contiguous video frame indices matching 30 Hz nominal sampling",
    }

    # 6. Save atomically and validate
    saved_path = save_feature_cache(
        cache_path=target_cache,
        features=all_features,
        frame_indices=all_indices,
        metadata=metadata,
        overwrite=True,
    )

    return {
        "status": "EXTRACTED",
        "trial_id": trial_id,
        "cache_path": str(saved_path).replace("\\", "/"),
        "num_frames": int(n_extracted),
        "output_dim": int(encoder.output_dim),
        "duration_sec": round(duration_sec, 2),
        "file_size_bytes": saved_path.stat().st_size,
    }
