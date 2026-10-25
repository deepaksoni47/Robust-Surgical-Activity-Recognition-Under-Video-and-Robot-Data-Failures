from pathlib import Path
import numpy as np
import pytest
import torch

from src.data.video import read_frame_at_index
from src.models.visual_encoder import (
    BaseVisualEncoder,
    ResNet18VisualEncoder,
    SurgMotionVisualEncoder,
    build_visual_encoder,
)


def test_base_visual_encoder_abstract() -> None:
    """Verify that BaseVisualEncoder cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseVisualEncoder()  # type: ignore[abstract]


def test_resnet18_encoder_initialization_frozen() -> None:
    """Verify ResNet18VisualEncoder initializes with output_dim=512 and is strictly frozen."""
    encoder = ResNet18VisualEncoder(pretrained=False)
    assert encoder.output_dim == 512
    assert encoder.encoder_name == "ResNet18"
    assert encoder.is_frozen()
    for name, param in encoder.named_parameters():
        assert not param.requires_grad, f"Parameter {name} has requires_grad=True!"


def test_resnet18_encoder_eval_mode_enforced() -> None:
    """Verify that calling .train() on the encoder does not put it in training mode."""
    encoder = ResNet18VisualEncoder(pretrained=False)
    assert not encoder.training
    encoder.train(True)
    assert not encoder.training, "Encoder should remain in eval mode even after train(True) call!"


def test_resnet18_encoder_output_shapes() -> None:
    """Verify feature shapes for both single frame batches [B, C, H, W] and clips [B, T, C, H, W]."""
    encoder = ResNet18VisualEncoder(pretrained=False)

    # 1. Single frame batch: [3, 3, 224, 224] -> [3, 512]
    frames_4d = torch.randn(3, 3, 224, 224)
    feats_4d = encoder(frames_4d)
    assert feats_4d.shape == (3, 512)
    assert torch.all(torch.isfinite(feats_4d))

    # 2. Sequence clip batch: [2, 5, 3, 224, 224] -> [2, 5, 512]
    frames_5d = torch.randn(2, 5, 3, 224, 224)
    feats_5d = encoder(frames_5d)
    assert feats_5d.shape == (2, 5, 512)
    assert torch.all(torch.isfinite(feats_5d))


def test_resnet18_encoder_invalid_dimension() -> None:
    """Verify that 2D or 3D tensor input raises ValueError."""
    encoder = ResNet18VisualEncoder(pretrained=False)
    with pytest.raises(ValueError, match="Expected 4D.*or 5D.*tensor"):
        encoder(torch.randn(3, 224, 224))


def test_resnet18_encoder_nan_inf_rejection() -> None:
    """Verify that input containing NaN or Inf raises ValueError."""
    encoder = ResNet18VisualEncoder(pretrained=False)
    bad_tensor = torch.randn(2, 3, 224, 224)
    bad_tensor[0, 0, 10, 10] = float("nan")
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        encoder(bad_tensor)


def test_resnet18_encoder_determinism() -> None:
    """Verify that exact identical input frames produce identical feature outputs."""
    encoder = ResNet18VisualEncoder(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out1 = encoder(x)
    out2 = encoder(x)
    assert torch.equal(out1, out2), "Encoder output is not deterministic across identical calls!"


def test_preprocess_image_valid() -> None:
    """Verify image preprocessing scales, normalizes, and resizes accurately."""
    # Create synthetic 480x640x3 uint8 image
    img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    tensor = BaseVisualEncoder.preprocess_image(img, target_size=(224, 224))

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.all(torch.isfinite(tensor))


def test_preprocess_image_invalid() -> None:
    """Verify image preprocessing rejects non-3-channel inputs."""
    with pytest.raises(ValueError, match="Expected 3-channel RGB image"):
        BaseVisualEncoder.preprocess_image(np.zeros((100, 100), dtype=np.uint8))


def test_surgmotion_missing_checkpoint_raises_informative_error() -> None:
    """Verify SurgMotion fails loudly with an informative Hugging Face manual gating message when weights are absent."""
    with pytest.raises(FileNotFoundError) as exc_info:
        SurgMotionVisualEncoder(checkpoint_path="checkpoints/non_existent_surgmotion.pt")

    err_text = str(exc_info.value)
    assert "SurgMotion checkpoint not found" in err_text
    assert "CAIR-HKISI/SurgMotion" in err_text
    assert "gated: manual" in err_text


def test_build_visual_encoder_factory_default() -> None:
    """Verify factory loads encoder from configs/encoder.yaml."""
    encoder = build_visual_encoder("configs/encoder.yaml")
    assert isinstance(encoder, ResNet18VisualEncoder)
    assert encoder.output_dim == 512
    assert encoder.is_frozen()


def test_build_visual_encoder_unknown_type() -> None:
    """Verify factory raises ValueError for invalid encoder_type."""
    with pytest.raises(ValueError, match="Unknown encoder_type"):
        build_visual_encoder({"encoder_type": "invalid_model_xyz"})


def test_encoder_on_actual_jigsaws_video_frame() -> None:
    """Integration test: extract frame from real JIGSAWS video and verify representation."""
    video_path = Path("dataset/Suturing/video/Suturing_B001_capture1.avi")
    if not video_path.is_file():
        pytest.skip(f"Video file {video_path} not available.")

    frame_bgr = read_frame_at_index(video_path, frame_idx=50)
    import cv2
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    encoder = ResNet18VisualEncoder(pretrained=False)
    input_tensor = BaseVisualEncoder.preprocess_image(frame_rgb, target_size=(224, 224))
    feats = encoder(input_tensor)

    assert feats.shape == (1, 512)
    assert torch.all(torch.isfinite(feats))
    assert float(torch.norm(feats).item()) > 0.0
