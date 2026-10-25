from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from src.utils.reproducibility import load_config


class BaseVisualEncoder(nn.Module, abc.ABC):
    """Abstract base class for visual feature encoders in surgical activity recognition.
    
    All encoders are frozen by design to extract static representations for downstream
    temporal sequence modeling (Hypothesis H1).
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None) -> None:
        super().__init__()
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """Return the output feature dimensionality (e.g. 512 for ResNet-18, 1024 for SurgMotion-ViTL)."""
        pass

    @property
    @abc.abstractmethod
    def encoder_name(self) -> str:
        """Return the human-readable canonical identifier of the encoder architecture."""
        pass

    @property
    def device(self) -> torch.device:
        """Return the device where encoder parameters reside."""
        return self._device

    def is_frozen(self) -> bool:
        """Verify whether all model parameters have requires_grad set to False."""
        return all(not p.requires_grad for p in self.parameters())

    def freeze(self) -> None:
        """Explicitly freeze all model parameters and set model to eval mode."""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode: bool = True) -> BaseVisualEncoder:
        """Enforce eval mode on frozen backbones to avoid altering BatchNorm or Dropout statistics."""
        # Always stay in eval mode
        return super().train(False)

    @abc.abstractmethod
    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract feature representation from input video frames.
        
        Args:
            frames: Input tensor of shape [B, C, H, W] for single frame or
                    [B, T, C, H, W] for a video sequence of T frames.
                    Pixel values should be scaled to [0, 1] and ImageNet normalized.
                    
        Returns:
            torch.Tensor: Feature tensor of shape [B, D] or [B, T, D].
        """
        pass

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Delegates forward call to extract_features."""
        return self.extract_features(frames)

    @staticmethod
    def preprocess_image(
        image: np.ndarray,
        target_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> torch.Tensor:
        """Preprocess an RGB numpy frame into an ImageNet-standard normalized torch.Tensor.
        
        Args:
            image: HxWxC uint8 or float32 image array (RGB).
            target_size: (height, width) tuple for spatial resizing.
            mean: ImageNet channel means.
            std: ImageNet channel standard deviations.
            
        Returns:
            torch.Tensor: Normalized tensor of shape [1, 3, H, W].
        """
        import cv2

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel RGB image [H, W, 3], got shape {image.shape}")

        if (image.shape[0], image.shape[1]) != target_size:
            resized = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        else:
            resized = image

        # Scale to [0, 1]
        arr = resized.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

        # Transpose to [C, H, W]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float()

        # Normalize with mean and std
        mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        norm_tensor = (tensor - mean_t) / std_t

        return norm_tensor.unsqueeze(0)


class ResNet18VisualEncoder(BaseVisualEncoder):
    """Lightweight frozen visual encoder using ResNet-18 pretrained on ImageNet-1K.
    
    Serves as the verified, deterministic, and CPU-efficient unimodal visual baseline
    and fallback for SurgMotion under local hardware or access constraints.
    """

    def __init__(
        self,
        pretrained: bool = True,
        weights: Optional[ResNet18_Weights] = ResNet18_Weights.IMAGENET1K_V1,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(device=device)
        self._encoder_name = "ResNet18"
        self._output_dim = 512

        # Initialize backbone
        if pretrained:
            weights_to_use = weights if weights is not None else ResNet18_Weights.DEFAULT
            backbone = models.resnet18(weights=weights_to_use)
        else:
            backbone = models.resnet18(weights=None)

        # Retain all layers up to avgpool; strip classification fc layer
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.to(self.device)
        self.freeze()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def encoder_name(self) -> str:
        return self._encoder_name

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Internal forward pass through convolutional stages and pooling."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        return features

    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract 512-D visual features from single frames [B, C, H, W] or sequences [B, T, C, H, W].
        
        Args:
            frames: Tensor of shape [B, C, H, W] or [B, T, C, H, W].
            
        Returns:
            torch.Tensor: Shape [B, 512] or [B, T, 512].
            
        Raises:
            ValueError: If input tensor dimensionality is not 4 or 5, or if values contain NaNs/Infs.
        """
        if not torch.all(torch.isfinite(frames)):
            raise ValueError("Input frames tensor contains NaN or Inf values.")

        orig_device = frames.device
        x = frames.to(self.device)

        with torch.no_grad():
            if x.dim() == 4:
                # [B, C, H, W]
                feats = self._forward_backbone(x)
            elif x.dim() == 5:
                # [B, T, C, H, W]
                b, t, c, h, w = x.shape
                flat_x = x.view(b * t, c, h, w)
                flat_feats = self._forward_backbone(flat_x)
                feats = flat_feats.view(b, t, self._output_dim)
            else:
                raise ValueError(
                    f"Expected 4D [B, C, H, W] or 5D [B, T, C, H, W] tensor, got shape {tuple(x.shape)}"
                )

        if not torch.all(torch.isfinite(feats)):
            raise RuntimeError("Extracted visual features contain NaN or Inf values.")

        return feats.to(orig_device)


class SurgMotionVisualEncoder(BaseVisualEncoder):
    """Visual foundation model interface for SurgMotion (Wu et al., 2026).
    
    Target architecture: V-JEPA 2 (ViT-Large 300M parameter backbone) pre-trained
    on the 3,658-hour SurgMotion-15M multi-organ surgical video corpus.
    
    Official Model Specifications:
        - GitHub: https://github.com/CAIR-HKISI/SurgMotion
        - Hugging Face: https://huggingface.co/CAIR-HKISI/SurgMotion
        - Status: Manually gated on Hugging Face ('gated: manual'). Requires authorized token.
        - Input: 64-frame video clips at 224x224, sampled at 1 FPS.
        - Output: 1024-dimensional spatiotemporally pooled representation.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        clip_length: int = 64,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(device=device)
        self._encoder_name = "SurgMotion-ViTL"
        self._output_dim = 1024
        self.clip_length = clip_length
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        # Verify checkpoint presence
        if self.checkpoint_path is None or not self.checkpoint_path.is_file():
            err_msg = (
                f"SurgMotion checkpoint not found at '{self.checkpoint_path}'.\n"
                "ACCESS NOTICE: The official SurgMotion foundation model weights (CAIR-HKISI/SurgMotion) "
                "are subject to manual gating on Hugging Face ('gated: manual').\n"
                "To use SurgMotion:\n"
                "  1. Register at https://huggingface.co/CAIR-HKISI/SurgMotion and request checkpoint access.\n"
                "  2. Download 'SurgMotion-vitl.pt' into 'checkpoints/SurgMotion-vitl.pt'.\n"
                "  3. Alternatively, use the verified frozen fallback encoder by setting "
                "'encoder_type: resnet18' in 'configs/encoder.yaml'."
            )
            raise FileNotFoundError(err_msg)

        # Load weights and initialize backbone
        self._init_surgmotion_backbone()
        self.to(self.device)
        self.freeze()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def encoder_name(self) -> str:
        return self._encoder_name

    def _init_surgmotion_backbone(self) -> None:
        """Initialize the V-JEPA 2 ViT backbone from verified checkpoint."""
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        # Placeholder for exact V-JEPA 2 state dict mapping
        self.model = ckpt if isinstance(ckpt, nn.Module) else None

    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract 1024-D spatiotemporally pooled features from 64-frame video clips."""
        if not torch.all(torch.isfinite(frames)):
            raise ValueError("Input frames tensor contains NaN or Inf values.")

        x = frames.to(self.device)
        if x.dim() != 5 or x.shape[1] != self.clip_length:
            raise ValueError(
                f"SurgMotion requires 5D clips of shape [B, {self.clip_length}, C, H, W], got {tuple(x.shape)}"
            )

        with torch.no_grad():
            # Run forward pass through V-JEPA 2 encoder
            feats = self.model(x)

        return feats


def build_visual_encoder(
    config: Optional[Union[Dict[str, Any], str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> BaseVisualEncoder:
    """Factory function to build and return the configured visual encoder.
    
    Args:
        config: Either a loaded configuration dictionary, a filepath to encoder.yaml,
                or None (defaults to 'configs/encoder.yaml').
        device: Target execution device. If None, inherits from config or defaults to cuda/cpu.
        
    Returns:
        BaseVisualEncoder: Instantiated and frozen visual encoder.
        
    Raises:
        ValueError: If encoder_type is unknown.
        FileNotFoundError: If SurgMotion is requested but weights are unavailable.
    """
    if config is None:
        cfg = load_config("configs/encoder.yaml")
    elif isinstance(config, (str, Path)):
        cfg = load_config(config)
    else:
        cfg = config

    encoder_type = cfg.get("encoder_type", "resnet18").lower().strip()
    target_device = device or cfg.get("extraction", {}).get("device", None)

    if encoder_type == "resnet18":
        pretrained = cfg.get("resnet18", {}).get("pretrained", True)
        return ResNet18VisualEncoder(pretrained=pretrained, device=target_device)
    elif encoder_type in ("surgmotion", "surgmotion_vitl"):
        ckpt_path = cfg.get("surgmotion", {}).get("checkpoint_path", "checkpoints/SurgMotion-vitl.pt")
        clip_len = cfg.get("surgmotion", {}).get("clip_length", 64)
        return SurgMotionVisualEncoder(checkpoint_path=ckpt_path, clip_length=clip_len, device=target_device)
    else:
        raise ValueError(
            f"Unknown encoder_type: '{encoder_type}'. Expected 'resnet18' or 'surgmotion_vitl'."
        )
