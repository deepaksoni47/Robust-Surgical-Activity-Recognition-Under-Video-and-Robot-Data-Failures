"""Data loading, manifest creation, kinematics parsing, windowing, and visual feature caching."""

from src.data.manifest import build_manifest, export_manifest
from src.data.kinematics import load_kinematics
from src.data.transcriptions import (
    get_gesture_vocabulary,
    parse_transcription,
    transcription_to_framewise_labels,
)
from src.data.video import get_video_metadata, read_frame_at_index
from src.data.splits import get_louo_splits, assert_no_leakage
from src.data.dataset import KinematicsNormalizer, JIGSAWSWindowDataset, get_fold_datasets, get_fold_dataloaders
from src.data.video_dataset import (
    VideoFeatureNormalizer,
    JIGSAWSVideoWindowDataset,
    get_video_fold_datasets,
    get_video_fold_dataloaders,
)
from src.data.feature_cache import (
    get_cache_path,
    save_feature_cache,
    load_feature_cache,
    validate_feature_cache,
    extract_frames_in_batches,
    extract_and_cache_trial,
)

__all__ = [
    "build_manifest",
    "export_manifest",
    "load_kinematics",
    "get_gesture_vocabulary",
    "parse_transcription",
    "transcription_to_framewise_labels",
    "get_video_metadata",
    "read_frame_at_index",
    "get_louo_splits",
    "assert_no_leakage",
    "KinematicsNormalizer",
    "JIGSAWSWindowDataset",
    "get_fold_datasets",
    "get_fold_dataloaders",
    "VideoFeatureNormalizer",
    "JIGSAWSVideoWindowDataset",
    "get_video_fold_datasets",
    "get_video_fold_dataloaders",
    "get_cache_path",
    "save_feature_cache",
    "load_feature_cache",
    "validate_feature_cache",
    "extract_frames_in_batches",
    "extract_and_cache_trial",
]

