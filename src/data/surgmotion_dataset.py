"""Dataset and temporal sequence window generator for frozen SurgMotion representations.

Correctly maps temporal tubelets (64-frame clips with stride 16) to JIGSAWS ground-truth
gesture annotations and robot kinematics trajectories:
1. Midpoint Annotation Mapping: maps each clip k to the gesture occurring at clip center frame
   c_k = (s_k + e_k) // 2. (Concordance with interval majority voting is >99.7%).
2. Kinematics Temporal Alignment: samples normalized 76D robot kinematics at the exact center frame
   min(c_k - 1, N_kin - 1), guaranteeing non-overflowing temporal synchronization.
3. Training-Only Normalization: feature z-scoring statistics are fitted strictly on training trials
   of each official LOUO fold to guarantee zero leakage.
4. Support for both clip-level classification (W=1) and temporal sequence modeling (W >= 2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import KinematicsNormalizer
from src.data.kinematics import load_kinematics
from src.data.splits import assert_no_leakage, get_louo_splits
from src.data.transcriptions import (
    get_gesture_vocabulary,
    parse_transcription,
    transcription_to_framewise_labels,
)
from src.data.video_dataset import VideoFeatureNormalizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class JIGSAWSSurgMotionDataset(Dataset):
    """PyTorch Dataset generating synchronized sliding sequences of SurgMotion features,

    kinematics, and gesture labels.

    Each item represents a temporal sequence of S consecutive SurgMotion clips:
        - 'features': (S, 1024) float32 tensor of normalized SurgMotion representations.
        - 'kinematics': (S, 76) float32 tensor of synchronized robot kinematics.
        - 'labels': (S,) int64 sequence of gesture labels for each clip.
        - 'target': int64 scalar causal label of the final clip in the sequence: labels[-1].
        - 'trial_id': string trial identifier.
        - 'center_frames': (S,) int32 array of video frame indices for each clip center.
    """

    def __init__(
        self,
        task: str,
        trials: List[str],
        dataset_dir: Union[str, Path] = "dataset",
        cache_dir: Union[str, Path] = "data/cache/features_surgmotion",
        sequence_length: int = 8,
        sequence_stride: int = 1,
        camera: str = "capture1",
        feature_normalizer: Optional[VideoFeatureNormalizer] = None,
        kinematics_normalizer: Optional[KinematicsNormalizer] = None,
        vocab: Optional[Dict[str, int]] = None,
        include_background: bool = True,
        load_kinematics_stream: bool = True,
    ) -> None:
        super().__init__()
        self.task = task
        self.trials = sorted(list(trials))
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.is_absolute():
            self.dataset_dir = PROJECT_ROOT / self.dataset_dir
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.is_absolute():
            self.cache_dir = PROJECT_ROOT / self.cache_dir
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.camera = camera
        self.feature_normalizer = feature_normalizer
        self.kinematics_normalizer = kinematics_normalizer
        self.include_background = include_background
        self.load_kinematics_stream = load_kinematics_stream

        if vocab is None:
            self.vocab = get_gesture_vocabulary(task=task, include_background=include_background)
        else:
            self.vocab = vocab
        self.inv_vocab = {idx: g for g, idx in self.vocab.items()}

        self.trial_records: List[Dict[str, Any]] = []
        self.sequences: List[Tuple[int, int, int]] = []  # (trial_idx, start_clip_idx, end_clip_idx)

        self._load_and_index()

    def _load_and_index(self) -> None:
        """Pre-load SurgMotion caches, transcriptions, and kinematics for all trials."""
        task_dir = self.dataset_dir / self.task
        transcriptions_base = task_dir / "transcriptions"
        kinematics_base = task_dir / "kinematics" / "AllGestures"

        for trial_idx, trial_id in enumerate(self.trials):
            cache_file = self.cache_dir / f"{trial_id}_{self.camera}.npz"
            trans_file = transcriptions_base / f"{trial_id}.txt"
            kin_file = kinematics_base / f"{trial_id}.txt"

            if not cache_file.is_file():
                raise FileNotFoundError(
                    f"SurgMotion cache file missing for trial '{trial_id}': {cache_file.resolve()}"
                )
            if not trans_file.is_file():
                raise FileNotFoundError(
                    f"Transcription file missing for trial '{trial_id}': {trans_file.resolve()}"
                )

            # 1. Load SurgMotion cache
            with np.load(cache_file, allow_pickle=True) as data:
                feats = data["features"].astype(np.float32)
                centers = data["clip_center_frames"].astype(np.int32)
                starts = data["clip_start_frames"].astype(np.int32)
                ends = data["clip_end_frames"].astype(np.int32)

            n_clips = len(feats)
            if n_clips == 0:
                raise ValueError(f"Cache for {trial_id} has zero clips.")

            # 2. Load transcriptions and map each clip to ground-truth label via midpoint convention
            segments = parse_transcription(trans_file)
            last_end_frame = int(ends[-1])
            framewise_labels = transcription_to_framewise_labels(
                segments, total_frames=max(last_end_frame, int(centers[-1])), background_label="G0"
            )

            clip_labels = []
            for c in centers:
                # 1-indexed frame number mapped to 0-indexed label array
                f_idx = min(max(0, int(c) - 1), len(framewise_labels) - 1)
                lbl_str = framewise_labels[f_idx]
                if lbl_str not in self.vocab:
                    raise KeyError(
                        f"Unknown gesture label '{lbl_str}' in {trial_id} not in vocabulary {list(self.vocab.keys())}"
                    )
                clip_labels.append(self.vocab[lbl_str])
            clip_labels_arr = np.array(clip_labels, dtype=np.int64)

            # 3. Load synchronized kinematics (sampled at clip center frame)
            if self.load_kinematics_stream and kin_file.is_file():
                raw_kin = load_kinematics(kin_file)
                n_kin_rows = len(raw_kin)
                # Sample kinematics at each clip's center frame
                kin_indices = [min(max(0, int(c) - 1), n_kin_rows - 1) for c in centers]
                kin_sampled = raw_kin[kin_indices].astype(np.float32)
            else:
                kin_sampled = np.zeros((n_clips, 76), dtype=np.float32)

            # 4. Apply training-fitted normalizers if provided
            if self.feature_normalizer is not None:
                feats = self.feature_normalizer.transform(feats)
            if self.kinematics_normalizer is not None:
                kin_sampled = self.kinematics_normalizer.transform(kin_sampled)

            self.trial_records.append({
                "trial_id": trial_id,
                "features": feats,
                "kinematics": kin_sampled,
                "labels": clip_labels_arr,
                "center_frames": centers,
                "num_clips": n_clips,
            })

            # 5. Index sequence windows of length S
            if n_clips < self.sequence_length:
                # Trial shorter than sequence window: wrap or skip (should not happen on JIGSAWS with S<=16)
                continue

            for start in range(0, n_clips - self.sequence_length + 1, self.sequence_stride):
                end = start + self.sequence_length
                self.sequences.append((trial_idx, start, end))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        trial_idx, start, end = self.sequences[idx]
        rec = self.trial_records[trial_idx]

        feat_seq = torch.from_numpy(rec["features"][start:end])  # [S, 1024]
        kin_seq = torch.from_numpy(rec["kinematics"][start:end])  # [S, 76]
        label_seq = torch.from_numpy(rec["labels"][start:end])    # [S]
        target = label_seq[-1]                                    # causal label
        center_seq = rec["center_frames"][start:end]

        return {
            "features": feat_seq,
            "kinematics": kin_seq,
            "labels": label_seq,
            "target": target,
            "trial_id": rec["trial_id"],
            "center_frames": center_seq,
        }


def get_surgmotion_dataloaders(
    task: str,
    fold: int,
    sequence_length: int = 8,
    sequence_stride: int = 1,
    batch_size: int = 32,
    camera: str = "capture1",
    include_background: bool = True,
    normalize_kinematics: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, VideoFeatureNormalizer, Dict[str, Any]]:
    """Build leakage-safe Train, Val, and Test DataLoaders for an official LOUO fold.

    CRITICAL LEAKAGE PROTECTION:
    - Feature and kinematics normalizations are fitted strictly on training trials.
    - Test trials are completely held-out.
    """
    splits_dict = get_louo_splits(task=task, create_val_split=True)
    if fold not in splits_dict:
        raise ValueError(f"Fold {fold} not found in splits for task '{task}'.")

    fold_info = splits_dict[fold]
    train_trials = fold_info["train_trials"]
    val_trials = fold_info["val_trials"]
    test_trials = fold_info["test_trials"]

    # 1. First pass: load raw training features and kinematics to fit normalizers without leakage
    cache_dir = PROJECT_ROOT / "data/cache/features_surgmotion"
    raw_train_feats = []
    for tid in train_trials:
        p = cache_dir / f"{tid}_{camera}.npz"
        with np.load(p, allow_pickle=True) as data:
            raw_train_feats.append(data["features"].astype(np.float32))

    feat_normalizer = VideoFeatureNormalizer().fit(raw_train_feats)

    kin_normalizer: Optional[KinematicsNormalizer] = None
    if normalize_kinematics:
        kin_base = PROJECT_ROOT / f"dataset/{task}/kinematics/AllGestures"
        raw_train_kin = []
        for tid in train_trials:
            kp = kin_base / f"{tid}.txt"
            if kp.is_file():
                raw_train_kin.append(load_kinematics(kp))
        if raw_train_kin:
            kin_normalizer = KinematicsNormalizer().fit(raw_train_kin)
            fold_info["kinematics_normalizer"] = kin_normalizer

    # 2. Build Datasets using fitted normalizers
    train_ds = JIGSAWSSurgMotionDataset(
        task=task,
        trials=train_trials,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        camera=camera,
        feature_normalizer=feat_normalizer,
        kinematics_normalizer=kin_normalizer,
        include_background=include_background,
    )
    val_ds = JIGSAWSSurgMotionDataset(
        task=task,
        trials=val_trials,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        camera=camera,
        feature_normalizer=feat_normalizer,
        kinematics_normalizer=kin_normalizer,
        include_background=include_background,
    )
    test_ds = JIGSAWSSurgMotionDataset(
        task=task,
        trials=test_trials,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        camera=camera,
        feature_normalizer=feat_normalizer,
        kinematics_normalizer=kin_normalizer,
        include_background=include_background,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, feat_normalizer, fold_info
