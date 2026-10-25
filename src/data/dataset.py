import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.kinematics import load_kinematics
from src.data.splits import assert_no_leakage, get_louo_splits
from src.data.transcriptions import (
    get_gesture_vocabulary,
    parse_transcription,
    transcription_to_framewise_labels,
)


class KinematicsNormalizer:
    """Z-score normalizer for 76D da Vinci kinematics.
    
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

    def fit(self, kinematics_list: List[np.ndarray]) -> "KinematicsNormalizer":
        """Compute feature-wise mean and std across a list of trial kinematics arrays.
        
        Args:
            kinematics_list: List of 2D numpy arrays of shape (T_i, D), where D=76.
            
        Returns:
            self
        """
        if not kinematics_list:
            raise ValueError("kinematics_list cannot be empty.")

        for i, arr in enumerate(kinematics_list):
            if arr.ndim != 2 or arr.shape[1] != 76:
                raise ValueError(f"Array {i} must have shape (T, 76), got {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Array {i} contains non-finite values (NaN or Inf).")

        # Concatenate across time: shape (sum(T_i), 76)
        all_samples = np.vstack(kinematics_list).astype(np.float64)

        mean = np.mean(all_samples, axis=0)
        std = np.std(all_samples, axis=0)

        # Protect against constant/zero-variance channels
        std = np.where(std < self.eps, 1.0, std)

        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.is_fitted = True

        return self

    def transform(self, kinematics: np.ndarray) -> np.ndarray:
        """Apply z-score normalization using fitted statistics.
        
        Args:
            kinematics: Array of shape (T, 76) or (76,).
            
        Returns:
            Normalized array with same shape as input.
        """
        if not self.is_fitted or self.mean is None or self.std is None:
            raise RuntimeError("KinematicsNormalizer must be fitted before calling transform().")

        if kinematics.shape[-1] != 76:
            raise ValueError(f"Expected last dimension to be 76, got {kinematics.shape[-1]}")

        normalized = (kinematics.astype(np.float32) - self.mean) / (self.std + self.eps)
        return normalized.astype(np.float32)

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        """Reverse normalization back to original physical units."""
        if not self.is_fitted or self.mean is None or self.std is None:
            raise RuntimeError("KinematicsNormalizer must be fitted before calling inverse_transform().")

        denormalized = (normalized.astype(np.float32) * (self.std + self.eps)) + self.mean
        return denormalized.astype(np.float32)

    def fit_transform(self, kinematics_list: List[np.ndarray]) -> List[np.ndarray]:
        """Fit on provided arrays and return transformed versions."""
        self.fit(kinematics_list)
        return [self.transform(arr) for arr in kinematics_list]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize normalization parameters."""
        if not self.is_fitted or self.mean is None or self.std is None:
            raise RuntimeError("Cannot serialize unfitted KinematicsNormalizer.")
        return {
            "eps": self.eps,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "dimension": int(self.mean.shape[0]),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KinematicsNormalizer":
        """Instantiate normalizer from serialized parameter dict."""
        normalizer = cls(eps=data.get("eps", 1e-8))
        normalizer.mean = np.array(data["mean"], dtype=np.float32)
        normalizer.std = np.array(data["std"], dtype=np.float32)
        normalizer.is_fitted = True
        return normalizer


class JIGSAWSWindowDataset(Dataset):
    """PyTorch Dataset generating sliding windows of synchronized kinematics & gesture labels.
    
    Each sample represents a window of length W:
        - 'kinematics': (W, 76) float32 tensor
        - 'labels': (W,) int64 sequence labels (framewise)
        - 'target': int64 scalar (label at causal window end: labels[-1])
        - 'trial_id': string identifier
        - 'start_idx': start frame index in original trial
        - 'end_idx': end frame index in original trial
    """

    def __init__(
        self,
        task: str,
        trials: List[str],
        dataset_dir: Union[str, Path] = "dataset",
        window_size: int = 16,
        stride: int = 8,
        normalizer: Optional[KinematicsNormalizer] = None,
        vocab: Optional[Dict[str, int]] = None,
        include_background: bool = True,
        preload: bool = True,
    ) -> None:
        super().__init__()
        self.task = task
        self.trials = sorted(list(trials))
        self.dataset_dir = Path(dataset_dir)
        self.window_size = window_size
        self.stride = stride
        self.normalizer = normalizer
        self.include_background = include_background

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
        """Pre-load all trial kinematics and transcriptions and index sliding windows."""
        task_dir = self.dataset_dir / self.task
        kinematics_base = task_dir / "kinematics" / "AllGestures"
        transcriptions_base = task_dir / "transcriptions"

        for trial_idx, trial_id in enumerate(self.trials):
            kin_file = kinematics_base / f"{trial_id}.txt"
            trans_file = transcriptions_base / f"{trial_id}.txt"

            if not kin_file.is_file():
                raise FileNotFoundError(f"Kinematics file missing for trial {trial_id}: {kin_file}")
            if not trans_file.is_file():
                raise FileNotFoundError(f"Transcription file missing for trial {trial_id}: {trans_file}")

            # 1. Load 76D kinematics
            kin_arr = load_kinematics(kin_file)
            total_frames = kin_arr.shape[0]

            # 2. Normalize if normalizer is provided
            if self.normalizer is not None:
                kin_arr = self.normalizer.transform(kin_arr)

            # 3. Load transcription intervals and map to framewise integer labels
            segments = parse_transcription(trans_file)
            str_labels = transcription_to_framewise_labels(
                segments, total_frames=total_frames, background_label="G0"
            )
            int_labels = np.array([self.vocab[lbl] for lbl in str_labels], dtype=np.int64)

            self.trial_records.append({
                "trial_id": trial_id,
                "kinematics": kin_arr,
                "labels": int_labels,
                "total_frames": total_frames,
            })

            # 4. Generate sliding window index entries: [t, t + W)
            if total_frames < self.window_size:
                continue

            for start in range(0, total_frames - self.window_size + 1, self.stride):
                end = start + self.window_size
                self.windows.append((trial_idx, start, end))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        trial_idx, start, end = self.windows[idx]
        record = self.trial_records[trial_idx]

        kinematics_window = record["kinematics"][start:end]  # shape (W, 76)
        labels_window = record["labels"][start:end]          # shape (W,)
        target = labels_window[-1]                            # Causal label at end of window

        return {
            "kinematics": torch.from_numpy(kinematics_window).float(),
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

    def compute_class_weights(self, eps: float = 1e-4) -> torch.Tensor:
        """Compute balanced inverse-frequency class weights for loss weighting."""
        counts = self.get_class_counts()
        num_classes = len(self.vocab)
        total_samples = len(self.windows)

        weights = np.zeros(num_classes, dtype=np.float32)
        for cls_idx, count in counts.items():
            if count > 0:
                # Standard balanced weighting: total / (num_classes * count)
                weights[cls_idx] = total_samples / (num_classes * count)
            else:
                weights[cls_idx] = 0.0

        return torch.from_numpy(weights)


def get_fold_datasets(
    task: str,
    fold_num: int,
    window_size: int = 16,
    train_stride: int = 8,
    eval_stride: int = 8,
    dataset_dir: Union[str, Path] = "dataset",
    splits_root: Union[str, Path] = "dataset/Experimental_setup",
    create_val_split: bool = True,
) -> Dict[str, Any]:
    """Build train, val, and test datasets for a specific LOUO fold.
    
    Guarantees:
        1. Strict zero leakage between train, val, and test trials/users.
        2. KinematicsNormalizer is fitted STRICTLY on training trials.
    
    Returns:
        Dict with keys: 'train', 'val', 'test', 'normalizer', 'vocab', 'inv_vocab',
        'class_weights', 'fold_info'.
    """
    splits = get_louo_splits(task, splits_root=splits_root, create_val_split=create_val_split)
    if fold_num not in splits:
        raise ValueError(f"Fold {fold_num} not found for task '{task}'. Available: {list(splits.keys())}")

    fold_info = splits[fold_num]
    train_trials = fold_info["train_trials"]
    val_trials = fold_info.get("val_trials", [])
    test_trials = fold_info["test_trials"]

    # 1. Assert zero leakage
    assert_no_leakage(train_trials=train_trials, test_trials=test_trials, val_trials=val_trials)

    # 2. Fit normalizer STRICTLY on training trials
    task_dir = Path(dataset_dir) / task
    kin_base = task_dir / "kinematics" / "AllGestures"

    train_kin_list = []
    for t_id in train_trials:
        path = kin_base / f"{t_id}.txt"
        kin_arr = load_kinematics(path)
        train_kin_list.append(kin_arr)

    normalizer = KinematicsNormalizer()
    normalizer.fit(train_kin_list)

    # 3. Gesture vocabulary
    vocab = get_gesture_vocabulary(task=task, include_background=True)
    inv_vocab = {idx: g for g, idx in vocab.items()}

    # 4. Construct datasets with the trained normalizer
    train_dataset = JIGSAWSWindowDataset(
        task=task,
        trials=train_trials,
        dataset_dir=dataset_dir,
        window_size=window_size,
        stride=train_stride,
        normalizer=normalizer,
        vocab=vocab,
        include_background=True,
    )

    val_dataset = JIGSAWSWindowDataset(
        task=task,
        trials=val_trials,
        dataset_dir=dataset_dir,
        window_size=window_size,
        stride=eval_stride,
        normalizer=normalizer,
        vocab=vocab,
        include_background=True,
    ) if val_trials else None

    test_dataset = JIGSAWSWindowDataset(
        task=task,
        trials=test_trials,
        dataset_dir=dataset_dir,
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


def get_fold_dataloaders(
    task: str,
    fold_num: int,
    batch_size: int = 32,
    window_size: int = 16,
    train_stride: int = 8,
    eval_stride: int = 8,
    num_workers: int = 0,
    dataset_dir: Union[str, Path] = "dataset",
    splits_root: Union[str, Path] = "dataset/Experimental_setup",
    create_val_split: bool = True,
) -> Dict[str, Any]:
    """Create PyTorch DataLoaders for training, validation, and testing on a given LOUO fold."""
    datasets = get_fold_datasets(
        task=task,
        fold_num=fold_num,
        window_size=window_size,
        train_stride=train_stride,
        eval_stride=eval_stride,
        dataset_dir=dataset_dir,
        splits_root=splits_root,
        create_val_split=create_val_split,
    )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    ) if datasets["val"] is not None else None

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
