from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.feature_cache import (
    get_cache_path,
    load_feature_cache,
    validate_feature_cache,
)
from src.data.kinematics import load_kinematics
from src.data.splits import assert_no_leakage, get_louo_splits
from src.data.transcriptions import (
    get_gesture_vocabulary,
    parse_transcription,
    transcription_to_framewise_labels,
)


class VideoFeatureNormalizer:
    """Z-score normalizer for visual feature representations.
    
    CRITICAL: Must be fitted strictly on training trials of a fold to prevent
    data leakage into validation and testing splits.
    
    Formula:
        x_norm = (x - mean) / (std + eps)
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, features_list: List[np.ndarray]) -> "VideoFeatureNormalizer":
        """Compute feature-wise mean and std across a list of trial feature arrays.
        
        Args:
            features_list: List of 2D numpy arrays of shape (T_i, D).
            
        Returns:
            self
        """
        if not features_list:
            raise ValueError("features_list cannot be empty.")

        expected_dim = features_list[0].shape[1]
        for i, arr in enumerate(features_list):
            if arr.ndim != 2 or arr.shape[1] != expected_dim:
                raise ValueError(
                    f"Array {i} must have shape (T, {expected_dim}), got {arr.shape}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Array {i} contains non-finite values (NaN or Inf).")

        # Concatenate across time: shape (sum(T_i), D)
        all_samples = np.vstack(features_list).astype(np.float64)

        mean = np.mean(all_samples, axis=0)
        std = np.std(all_samples, axis=0)

        # Protect against constant / zero-variance feature dimensions
        std = np.where(std < self.eps, 1.0, std)

        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.is_fitted = True

        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply z-score normalization using fitted statistics.
        
        Args:
            features: Array of shape (T, D) or (D,).
            
        Returns:
            Normalized array with the same shape as input.
        """
        if not self.is_fitted or self.mean is None or self.std is None:
            raise RuntimeError("VideoFeatureNormalizer must be fitted before calling transform().")

        if features.shape[-1] != self.mean.shape[0]:
            raise ValueError(
                f"Expected last dimension to be {self.mean.shape[0]}, got {features.shape[-1]}"
            )

        normalized = (features.astype(np.float32) - self.mean) / (self.std + self.eps)
        return normalized.astype(np.float32)

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        """Reverse normalization back to original feature scale."""
        if not self.is_fitted or self.mean is None or self.std is None:
            raise RuntimeError("VideoFeatureNormalizer must be fitted before calling inverse_transform().")

        denormalized = (normalized.astype(np.float32) * (self.std + self.eps)) + self.mean
        return denormalized.astype(np.float32)

    def fit_transform(self, features_list: List[np.ndarray]) -> List[np.ndarray]:
        """Fit on provided arrays and return transformed versions."""
        self.fit(features_list)
        return [self.transform(arr) for arr in features_list]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize normalization parameters."""
        if not self.is_fitted or self.mean is None or self.std is None:
            raise RuntimeError("Cannot serialize unfitted VideoFeatureNormalizer.")
        return {
            "eps": self.eps,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "dimension": int(self.mean.shape[0]),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoFeatureNormalizer":
        """Instantiate normalizer from serialized parameter dict."""
        normalizer = cls(eps=data.get("eps", 1e-8))
        normalizer.mean = np.array(data["mean"], dtype=np.float32)
        normalizer.std = np.array(data["std"], dtype=np.float32)
        normalizer.is_fitted = True
        return normalizer


class JIGSAWSVideoWindowDataset(Dataset):
    """PyTorch Dataset generating sliding windows of cached visual features and gesture labels.
    
    Each sample represents a window of length W:
        - 'video': (W, D) float32 tensor of cached visual features
        - 'labels': (W,) int64 sequence labels (framewise)
        - 'target': int64 scalar (causal label at end of window: labels[-1])
        - 'trial_id': string identifier
        - 'start_idx': start frame index in original trial
        - 'end_idx': end frame index in original trial
    """

    def __init__(
        self,
        task: str,
        trials: List[str],
        dataset_dir: Union[str, Path] = "dataset",
        cache_dir: Union[str, Path] = "data/cache/features",
        encoder_name: str = "ResNet18",
        camera: str = "capture1",
        expected_dim: Optional[int] = 512,
        window_size: int = 16,
        stride: int = 8,
        normalizer: Optional[VideoFeatureNormalizer] = None,
        vocab: Optional[Dict[str, int]] = None,
        include_background: bool = True,
        preload: bool = True,
        max_allowed_delta: int = 500,
    ) -> None:
        super().__init__()
        self.task = task
        self.trials = sorted(list(trials))
        self.dataset_dir = Path(dataset_dir)
        self.cache_dir = Path(cache_dir)
        self.encoder_name = encoder_name
        self.camera = camera
        self.expected_dim = expected_dim
        self.window_size = window_size
        self.stride = stride
        self.normalizer = normalizer
        self.include_background = include_background
        self.max_allowed_delta = max_allowed_delta

        if vocab is None:
            self.vocab = get_gesture_vocabulary(task=task, include_background=include_background)
        else:
            self.vocab = vocab
        self.inv_vocab = {idx: g for g, idx in self.vocab.items()}

        self.trial_records: List[Dict[str, Any]] = []
        self.windows: List[Tuple[int, int, int]] = []  # (trial_idx, start_frame, end_frame)

        if preload:
            self._load_and_index()

    def _load_and_index(self) -> None:
        """Pre-load all trial cached features and transcriptions and index sliding windows."""
        task_dir = self.dataset_dir / self.task
        transcriptions_base = task_dir / "transcriptions"
        kinematics_base = task_dir / "kinematics" / "AllGestures"

        for trial_idx, trial_id in enumerate(self.trials):
            cache_file = get_cache_path(
                self.cache_dir, trial_id, self.encoder_name, camera=self.camera
            )
            trans_file = transcriptions_base / f"{trial_id}.txt"

            # 1. Informative error if cache is missing
            if not cache_file.is_file():
                clean_enc = self.encoder_name.replace(" ", "_")
                raise FileNotFoundError(
                    f"Missing visual feature cache for trial '{trial_id}' at '{cache_file}'. "
                    f"To generate the required cache, run:\n"
                    f"    python scripts/extract_visual_features.py --task {self.task} --encoder {clean_enc}"
                )

            if not trans_file.is_file():
                raise FileNotFoundError(
                    f"Transcription file missing for trial '{trial_id}': {trans_file}"
                )

            # 2. Validate cache integrity and encoder compatibility
            cache_info = validate_feature_cache(
                cache_file,
                expected_dim=self.expected_dim,
                expected_trial_id=trial_id,
            )
            cached_data = load_feature_cache(
                cache_file, validate=False, expected_dim=self.expected_dim
            )
            feat_arr = cached_data["features"]
            n_feat_frames = feat_arr.shape[0]

            # Verify encoder identity: strictly prevent mixing ResNet-18 and SurgMotion
            cached_enc = cached_data.get("metadata", {}).get("encoder_name", None)
            if cached_enc and cached_enc.replace(" ", "_") != self.encoder_name.replace(" ", "_"):
                raise ValueError(
                    f"Encoder mismatch in cache {cache_file.name}: "
                    f"requested '{self.encoder_name}', but cache was generated with '{cached_enc}'. "
                    f"Do not mix visual feature encoders."
                )

            # 3. Kinematics alignment check (if kinematics file exists)
            kin_file = kinematics_base / f"{trial_id}.txt"
            effective_frames = n_feat_frames

            if kin_file.is_file():
                kin_arr = load_kinematics(kin_file)
                n_kin_samples = kin_arr.shape[0]
                delta = abs(n_feat_frames - n_kin_samples)

                if delta > self.max_allowed_delta:
                    raise ValueError(
                        f"Unacceptable frame alignment delta for trial {trial_id}: "
                        f"video has {n_feat_frames} frames, kinematics has {n_kin_samples} rows "
                        f"(delta={delta} > {self.max_allowed_delta})."
                    )

                # Alignment rule: 1-to-1 at 30 Hz nominal, tail clipped to min(N_vid, N_kin)
                effective_frames = min(n_feat_frames, n_kin_samples)

            # 4. Transcription alignment and framewise integer labels
            segments = parse_transcription(trans_file)
            str_labels = transcription_to_framewise_labels(
                segments, total_frames=effective_frames, background_label="G0"
            )
            int_labels = np.array([self.vocab[lbl] for lbl in str_labels], dtype=np.int64)

            # Align feature length to effective frames
            feat_arr = feat_arr[:effective_frames].copy()
            if not np.all(np.isfinite(feat_arr)):
                raise ValueError(f"Features for trial {trial_id} contain non-finite values.")

            # 5. Normalize if normalizer is provided
            if self.normalizer is not None:
                feat_arr = self.normalizer.transform(feat_arr)

            self.trial_records.append({
                "trial_id": trial_id,
                "video": feat_arr,
                "labels": int_labels,
                "total_frames": effective_frames,
            })

            # 6. Generate sliding window index entries: [start, start + W)
            if effective_frames < self.window_size:
                continue

            for start in range(0, effective_frames - self.window_size + 1, self.stride):
                end = start + self.window_size
                self.windows.append((trial_idx, start, end))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        trial_idx, start, end = self.windows[idx]
        record = self.trial_records[trial_idx]

        video_window = record["video"][start:end]      # shape (W, D)
        labels_window = record["labels"][start:end]    # shape (W,)
        target = labels_window[-1]                      # Causal label at end of window

        return {
            "video": torch.from_numpy(video_window).float(),
            "labels": torch.from_numpy(labels_window).long(),
            "target": torch.tensor(target, dtype=torch.long),
            "trial_id": record["trial_id"],
            "start_idx": start,
            "end_idx": end,
        }

    def get_class_counts(self) -> Dict[int, int]:
        """Compute window-level target class distribution."""
        counts: Dict[int, int] = {idx: 0 for idx in self.vocab.values()}
        for trial_idx, start, end in self.windows:
            target = int(self.trial_records[trial_idx]["labels"][end - 1])
            counts[target] = counts.get(target, 0) + 1
        return counts

    def compute_class_weights(self) -> torch.Tensor:
        """Compute balanced inverse-frequency class weights for loss weighting."""
        counts = self.get_class_counts()
        num_classes = len(self.vocab)
        total_samples = len(self.windows)

        weights = np.zeros(num_classes, dtype=np.float32)
        for cls_idx, count in counts.items():
            if count > 0:
                weights[cls_idx] = total_samples / (num_classes * count)
            else:
                weights[cls_idx] = 0.0

        return torch.from_numpy(weights)


def get_video_fold_datasets(
    task: str,
    fold_num: int,
    window_size: int = 16,
    train_stride: int = 8,
    eval_stride: int = 8,
    dataset_dir: Union[str, Path] = "dataset",
    cache_dir: Union[str, Path] = "data/cache/features",
    encoder_name: str = "ResNet18",
    camera: str = "capture1",
    expected_dim: Optional[int] = 512,
    splits_root: Union[str, Path] = "dataset/Experimental_setup",
    create_val_split: bool = True,
    normalize: bool = True,
) -> Dict[str, Any]:
    """Build train, val, and test datasets for a specific LOUO fold using cached visual features.
    
    Guarantees:
        1. Strict zero leakage between train, val, and test trials/users.
        2. VideoFeatureNormalizer is fitted STRICTLY on training trials.
    
    Returns:
        Dict with keys: 'train', 'val', 'test', 'normalizer', 'vocab', 'inv_vocab',
        'class_weights', 'fold_info'.
    """
    splits = get_louo_splits(task, splits_root=splits_root, create_val_split=create_val_split)
    if fold_num not in splits:
        raise ValueError(
            f"Fold {fold_num} not found for task '{task}'. Available: {list(splits.keys())}"
        )

    fold_info = splits[fold_num]
    train_trials = fold_info["train_trials"]
    val_trials = fold_info.get("val_trials", [])
    test_trials = fold_info["test_trials"]

    # 1. Assert zero leakage
    assert_no_leakage(train_trials=train_trials, test_trials=test_trials, val_trials=val_trials)

    # 2. Fit normalizer STRICTLY on training trials
    normalizer: Optional[VideoFeatureNormalizer] = None
    if normalize:
        train_feat_list: List[np.ndarray] = []
        for t_id in train_trials:
            c_path = get_cache_path(cache_dir, t_id, encoder_name, camera=camera)
            data = load_feature_cache(c_path, validate=True, expected_dim=expected_dim)
            train_feat_list.append(data["features"])

        normalizer = VideoFeatureNormalizer()
        normalizer.fit(train_feat_list)

    # 3. Gesture vocabulary
    vocab = get_gesture_vocabulary(task=task, include_background=True)
    inv_vocab = {idx: g for g, idx in vocab.items()}

    # 4. Construct datasets with the trained normalizer
    train_dataset = JIGSAWSVideoWindowDataset(
        task=task,
        trials=train_trials,
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        encoder_name=encoder_name,
        camera=camera,
        expected_dim=expected_dim,
        window_size=window_size,
        stride=train_stride,
        normalizer=normalizer,
        vocab=vocab,
        include_background=True,
    )

    val_dataset = (
        JIGSAWSVideoWindowDataset(
            task=task,
            trials=val_trials,
            dataset_dir=dataset_dir,
            cache_dir=cache_dir,
            encoder_name=encoder_name,
            camera=camera,
            expected_dim=expected_dim,
            window_size=window_size,
            stride=eval_stride,
            normalizer=normalizer,
            vocab=vocab,
            include_background=True,
        )
        if val_trials
        else None
    )

    test_dataset = JIGSAWSVideoWindowDataset(
        task=task,
        trials=test_trials,
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        encoder_name=encoder_name,
        camera=camera,
        expected_dim=expected_dim,
        window_size=window_size,
        stride=eval_stride,
        normalizer=normalizer,
        vocab=vocab,
        include_background=True,
    )

    class_weights = train_dataset.compute_class_weights()

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "normalizer": normalizer,
        "vocab": vocab,
        "inv_vocab": inv_vocab,
        "class_weights": class_weights,
        "fold_info": fold_info,
    }


def get_video_fold_dataloaders(
    task: str,
    fold_num: int,
    batch_size: int = 32,
    window_size: int = 16,
    train_stride: int = 8,
    eval_stride: int = 8,
    num_workers: int = 0,
    dataset_dir: Union[str, Path] = "dataset",
    cache_dir: Union[str, Path] = "data/cache/features",
    encoder_name: str = "ResNet18",
    camera: str = "capture1",
    expected_dim: Optional[int] = 512,
    splits_root: Union[str, Path] = "dataset/Experimental_setup",
    create_val_split: bool = True,
    normalize: bool = True,
) -> Dict[str, Any]:
    """Create PyTorch DataLoaders for training, validation, and testing on a given LOUO fold."""
    datasets = get_video_fold_datasets(
        task=task,
        fold_num=fold_num,
        window_size=window_size,
        train_stride=train_stride,
        eval_stride=eval_stride,
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        encoder_name=encoder_name,
        camera=camera,
        expected_dim=expected_dim,
        splits_root=splits_root,
        create_val_split=create_val_split,
        normalize=normalize,
    )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = (
        DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
        if datasets["val"] is not None
        else None
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "datasets": datasets,
        "normalizer": datasets["normalizer"],
        "vocab": datasets["vocab"],
        "class_weights": datasets["class_weights"],
        "fold_info": datasets["fold_info"],
    }
