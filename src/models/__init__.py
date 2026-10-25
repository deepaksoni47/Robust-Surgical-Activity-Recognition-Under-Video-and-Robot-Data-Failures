"""Model architectures: unimodal classifiers, temporal encoders, fixed and adaptive fusion gates."""

from src.models.majority import MajorityClassClassifier
from src.models.kinematics_mlp import KinematicsMLP
from src.models.kinematics_gru import KinematicsGRU
from src.models.visual_encoder import (
    BaseVisualEncoder,
    ResNet18VisualEncoder,
    SurgMotionVisualEncoder,
    build_visual_encoder,
)
from src.models.video_head import VideoGRU, VideoMLP
from src.models.fusion import (
    FixedLateFusion,
    FixedLateFusionClassifier,
    fuse_logits,
    search_optimal_alpha,
    validate_class_mappings,
)

__all__ = [
    "MajorityClassClassifier",
    "KinematicsMLP",
    "KinematicsGRU",
    "BaseVisualEncoder",
    "ResNet18VisualEncoder",
    "SurgMotionVisualEncoder",
    "build_visual_encoder",
    "VideoGRU",
    "VideoMLP",
    "FixedLateFusion",
    "FixedLateFusionClassifier",
    "fuse_logits",
    "search_optimal_alpha",
    "validate_class_mappings",
]


