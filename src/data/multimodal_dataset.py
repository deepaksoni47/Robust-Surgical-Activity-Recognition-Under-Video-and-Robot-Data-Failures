from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import KinematicsNormalizer
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
from src.data.video_dataset import VideoFeatureNormalizer


class JIGSAWSMultimodalWindowDataset(Dataset):
    """PyTorch Dataset generating synchronized sliding windows of both video features and kinematics.
    
    Each sample represents a synchronized multimodal window of length W:
        - 'video': (W, D_v) float32 tensor of visual features
        - 'kinematics': (W, 76) float32 tensor of robot kinematics
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
        kin_normalizer: Optional[KinematicsNormalizer] = None,
        video_normalizer: Optional[VideoFeatureNormalizer] = None,
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
        self.kin_normalizer = kin_normalizer
        self.video_normalizer = video_normalizer
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
        """Pre-load and synchronize trial video features, kinematics, and transcriptions."""
        task_dir = self.dataset_dir / self.task
        kin_base = task_dir / "kinematics" / "AllGestures"
        trans_base = task_dir / "transcriptions"

        for trial_idx, trial_id in enumerate(self.trials):
            cache_file = get_cache_path(
                self.cache_dir, trial_id, self.encoder_name, camera=self.camera
            )
            kin_file = kin_base / f"{trial_id}.txt"
            trans_file = trans_base / f"{trial_id}.txt"

            # 1. Verify existence with informative error messages
            if not cache_file.is_file():
                clean_enc = self.encoder_name.replace(" ", "_")
                raise FileNotFoundError(
                    f"Missing visual feature cache for trial '{trial_id}' at '{cache_file}'. "
                    f"To generate the required cache, run:\n"
                    f"    python scripts/extract_visual_features.py --task {self.task} --encoder {clean_enc}"
                )
            if not kin_file.is_file():
                raise FileNotFoundError(f"Kinematics file missing for trial '{trial_id}': {kin_file}")
            if not trans_file.is_file():
                raise FileNotFoundError(f"Transcription file missing for trial '{trial_id}': {trans_file}")

            # 2. Validate cache integrity
            validate_feature_cache(
                cache_file,
                expected_dim=self.expected_dim,
                expected_trial_id=trial_id,
            )
            cached_data = load_feature_cache(
                cache_file, validate=False, expected_dim=self.expected_dim
            )
            feat_arr = cached_data["features"]
            n_feat = feat_arr.shape[0]

            # Verify encoder identity
            cached_enc = cached_data.get("metadata", {}).get("encoder_name", None)
            if cached_enc and cached_enc.replace(" ", "_") != self.encoder_name.replace(" ", "_"):
                raise ValueError(
                    f"Encoder mismatch in cache {cache_file.name}: "
                    f"requested '{self.encoder_name}', got '{cached_enc}'"
                )

            # 3. Load 76D kinematics
            kin_arr = load_kinematics(kin_file)
            n_kin = kin_arr.shape[0]

            # 4. Alignment verification & tail clipping convention
            delta = abs(n_feat - n_kin)
            if delta > self.max_allowed_delta:
                raise ValueError(
                    f"Unacceptable video-kinematics frame delta for trial {trial_id}: "
                    f"video has {n_feat} frames, kinematics has {n_kin} rows "
                    f"(delta={delta} > {self.max_allowed_delta})."
                )

            effective_frames = min(n_feat, n_kin)

            # 5. Load transcription and map to framewise integer labels
            segments = parse_transcription(trans_file)
            str_labels = transcription_to_framewise_labels(
                segments, total_frames=effective_frames, background_label="G0"
            )
            int_labels = np.array([self.vocab[lbl] for lbl in str_labels], dtype=np.int64)

            # Synchronize feature lengths to effective_frames
            feat_arr = feat_arr[:effective_frames].copy()
            kin_arr = kin_arr[:effective_frames].copy()

            # 6. Apply normalizers if provided
            if self.kin_normalizer is not None:
                kin_arr = self.kin_normalizer.transform(kin_arr)
            if self.video_normalizer is not None:
                feat_arr = self.video_normalizer.transform(feat_arr)

            self.trial_records.append({
                "trial_id": trial_id,
                "video": feat_arr,
                "kinematics": kin_arr,
                "labels": int_labels,
                "total_frames": effective_frames,
            })

            # 7. Slicing sliding windows: [start, start + W)
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

        v_window = record["video"][start:end]         # (W, D_v)
        k_window = record["kinematics"][start:end]    # (W, 76)
        lbl_window = record["labels"][start:end]      # (W,)
        target = lbl_window[-1]                        # Causal target at end of window

        return {
            "video": torch.from_numpy(v_window).float(),
            "kinematics": torch.from_numpy(k_window).float(),
            "labels": torch.from_numpy(lbl_window).long(),
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


def get_multimodal_fold_datasets(
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
    """Build train, val, and test datasets for a specific LOUO fold with synchronized modalities.
    
    Guarantees:
        1. Strict zero leakage between train, val, and test trials/users.
        2. Normalizers are fitted STRICTLY on training trials.
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

    # 2. Fit normalizers STRICTLY on training trials
    kin_normalizer: Optional[KinematicsNormalizer] = None
    vid_normalizer: Optional[VideoFeatureNormalizer] = None

    if normalize:
        task_dir = Path(dataset_dir) / task
        kin_base = task_dir / "kinematics" / "AllGestures"

        train_kin_list: List[np.ndarray] = []
        train_vid_list: List[np.ndarray] = []

        for t_id in train_trials:
            # Kinematics
            k_path = kin_base / f"{t_id}.txt"
            train_kin_list.append(load_kinematics(k_path))

            # Video features
            v_path = get_cache_path(cache_dir, t_id, encoder_name, camera=camera)
            v_data = load_feature_cache(v_path, validate=True, expected_dim=expected_dim)
            train_vid_list.append(v_data["features"])

        kin_normalizer = KinematicsNormalizer()
        kin_normalizer.fit(train_kin_list)

        vid_normalizer = VideoFeatureNormalizer()
        vid_normalizer.fit(train_vid_list)

    # 3. Gesture vocabulary
    vocab = get_gesture_vocabulary(task=task, include_background=True)
    inv_vocab = {idx: g for g, idx in vocab.items()}

    # 4. Construct datasets with fitted normalizers
    train_dataset = JIGSAWSMultimodalWindowDataset(
        task=task,
        trials=train_trials,
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        encoder_name=encoder_name,
        camera=camera,
        expected_dim=expected_dim,
        window_size=window_size,
        stride=train_stride,
        kin_normalizer=kin_normalizer,
        video_normalizer=vid_normalizer,
        vocab=vocab,
        include_background=True,
    )

    val_dataset = (
        JIGSAWSMultimodalWindowDataset(
            task=task,
            trials=val_trials,
            dataset_dir=dataset_dir,
            cache_dir=cache_dir,
            encoder_name=encoder_name,
            camera=camera,
            expected_dim=expected_dim,
            window_size=window_size,
            stride=eval_stride,
            kin_normalizer=kin_normalizer,
            video_normalizer=vid_normalizer,
            vocab=vocab,
            include_background=True,
        )
        if val_trials
        else None
    )

    test_dataset = JIGSAWSMultimodalWindowDataset(
        task=task,
        trials=test_trials,
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        encoder_name=encoder_name,
        camera=camera,
        expected_dim=expected_dim,
        window_size=window_size,
        stride=eval_stride,
        kin_normalizer=kin_normalizer,
        video_normalizer=vid_normalizer,
        vocab=vocab,
        include_background=True,
    )

    class_weights = train_dataset.compute_class_weights()

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "kin_normalizer": kin_normalizer,
        "video_normalizer": vid_normalizer,
        "vocab": vocab,
        "inv_vocab": inv_vocab,
        "class_weights": class_weights,
        "fold_info": fold_info,
    }


def get_multimodal_fold_dataloaders(
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
    """Create PyTorch DataLoaders for synchronized multimodal training, validation, and testing."""
    datasets = get_multimodal_fold_datasets(
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
        "kin_normalizer": datasets["kin_normalizer"],
        "video_normalizer": datasets["video_normalizer"],
        "vocab": datasets["vocab"],
        "class_weights": datasets["class_weights"],
        "fold_info": datasets["fold_info"],
    }
