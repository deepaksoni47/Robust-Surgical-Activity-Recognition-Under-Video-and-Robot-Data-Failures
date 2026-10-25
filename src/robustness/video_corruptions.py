"""src/robustness/video_corruptions.py

Deterministic Visual Corruption Suite for Surgical Activity Recognition.
Implements Endo-C6 inspired optical corruptions (pre-encoder, frame-level)
and synthetic feature-space perturbations (post-encoder, cached-feature-level).

Key Principles:
1. Frame-level optical corruptions (e.g. defocus blur, motion blur, shot noise,
   cautery smoke, packet loss) operate on RGB image tensors [H, W, 3] or [T, H, W, 3].
2. Feature-space perturbations (e.g. Gaussian noise, channel dropout, temporal
   frame dropout, scale drift) operate on extracted feature representations [D],
   [T, D], or [B, T, D]. These are explicitly documented as latent perturbations,
   NOT as realistic optical artifacts.
3. Clean Input Identity: When severity=0, the operator returns an exact copy of the input.
4. Determinism: For a fixed seed, outputs are bitwise deterministic.
5. Immutability: Input arrays and source caches are never modified in-place.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Base Visual Corruption Interface
# ---------------------------------------------------------------------------

class BaseVisualCorruption(ABC):
    """Abstract base class for all visual corruptions."""

    def __init__(
        self,
        severity: int = 1,
        seed: Optional[int] = 42,
    ) -> None:
        if not (0 <= severity <= 5):
            raise ValueError(f"Severity must be in range [0, 5], got {severity}")
        self.severity = int(severity)
        self.seed = seed

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the corruption operator."""
        pass

    @property
    @abstractmethod
    def level(self) -> str:
        """Operating level: 'frame' for RGB video or 'feature' for latent features."""
        pass

    @abstractmethod
    def apply(
        self,
        data: Union[np.ndarray, torch.Tensor],
        seed: Optional[int] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply corruption deterministically without modifying the input in-place."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Return serializable metadata describing operator parameters."""
        return {
            "name": self.name,
            "level": self.level,
            "severity": self.severity,
            "seed": self.seed,
            "is_identity": (self.severity == 0),
        }


# ---------------------------------------------------------------------------
# Frame-Level Optical Corruptions (Endo-C6 Inspired)
# ---------------------------------------------------------------------------

class DefocusBlurCorruption(BaseVisualCorruption):
    """Optical defocus blur simulating camera optics out of focus."""

    # Kernel radius and gaussian sigma across severities 1..5
    RADII = [1, 2, 3, 5, 7]
    SIGMAS = [0.8, 1.2, 1.8, 2.5, 3.5]

    @property
    def name(self) -> str:
        return "defocus_blur"

    @property
    def level(self) -> str:
        return "frame"

    def apply(
        self,
        data: Union[np.ndarray, torch.Tensor],
        seed: Optional[int] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        if self.severity == 0:
            return copy.deepcopy(data) if isinstance(data, np.ndarray) else data.clone()

        is_torch = isinstance(data, torch.Tensor)
        img = data.cpu().numpy() if is_torch else np.copy(data)

        if not np.all(np.isfinite(img)):
            raise ValueError("Input image contains non-finite values (NaN or Inf).")

        sigma = self.SIGMAS[self.severity - 1]
        ksize = int(2 * np.ceil(2 * sigma) + 1)
        if ksize % 2 == 0:
            ksize += 1

        # Process single frame (H, W, C) or clip (T, H, W, C)
        orig_shape = img.shape
        if img.ndim == 3:
            blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        elif img.ndim == 4:
            blurred = np.stack(
                [cv2.GaussianBlur(f, (ksize, ksize), sigmaX=sigma, sigmaY=sigma) for f in img],
                axis=0,
            )
        else:
            raise ValueError(f"Expected 3D or 4D image array, got ndim={img.ndim}")

        if is_torch:
            return torch.from_numpy(blurred).to(device=data.device, dtype=data.dtype)
        return blurred.astype(data.dtype)


class MotionBlurCorruption(BaseVisualCorruption):
    """Directional linear motion blur simulating rapid endoscopic camera movement."""

    KERNEL_LENGTHS = [3, 7, 11, 15, 21]

    @property
    def name(self) -> str:
        return "motion_blur"

    @property
    def level(self) -> str:
        return "frame"

    def apply(
        self,
        data: Union[np.ndarray, torch.Tensor],
        seed: Optional[int] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        if self.severity == 0:
            return copy.deepcopy(data) if isinstance(data, np.ndarray) else data.clone()

        active_seed = seed if seed is not None else self.seed
        rng = np.random.RandomState(active_seed)

        is_torch = isinstance(data, torch.Tensor)
        img = data.cpu().numpy() if is_torch else np.copy(data)

        if not np.all(np.isfinite(img)):
            raise ValueError("Input image contains non-finite values (NaN or Inf).")

        length = self.KERNEL_LENGTHS[self.severity - 1]
        angle = rng.uniform(0, 180)

        # Build directional motion kernel
        kernel = np.zeros((length, length), dtype=np.float32)
        center = (length - 1) / 2.0
        cv2.line(
            kernel,
            (int(center - length / 2 * np.cos(np.radians(angle))),
             int(center - length / 2 * np.sin(np.radians(angle)))),
            (int(center + length / 2 * np.cos(np.radians(angle))),
             int(center + length / 2 * np.sin(np.radians(angle)))),
            color=1.0,
            thickness=1,
        )
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel /= kernel_sum
        else:
            kernel[int(center), int(center)] = 1.0

        if img.ndim == 3:
            blurred = cv2.filter2D(img, -1, kernel)
        elif img.ndim == 4:
            blurred = np.stack([cv2.filter2D(f, -1, kernel) for f in img], axis=0)
        else:
            raise ValueError(f"Expected 3D or 4D image array, got ndim={img.ndim}")

        if is_torch:
            return torch.from_numpy(blurred).to(device=data.device, dtype=data.dtype)
        return blurred.astype(data.dtype)


class ShotNoiseCorruption(BaseVisualCorruption):
    """Photon shot noise simulating low-light illumination in endoscopic cavity."""

    # Scale parameter lambda: smaller lambda = higher relative photon noise
    LAMBDA_SCALES = [120.0, 60.0, 30.0, 15.0, 8.0]

    @property
    def name(self) -> str:
        return "shot_noise"

    @property
    def level(self) -> str:
        return "frame"

    def apply(
        self,
        data: Union[np.ndarray, torch.Tensor],
        seed: Optional[int] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        if self.severity == 0:
            return copy.deepcopy(data) if isinstance(data, np.ndarray) else data.clone()

        active_seed = seed if seed is not None else self.seed
        rng = np.random.RandomState(active_seed)

        is_torch = isinstance(data, torch.Tensor)
        img = data.cpu().numpy() if is_torch else np.copy(data)

        if not np.all(np.isfinite(img)):
            raise ValueError("Input image contains non-finite values (NaN or Inf).")

        scale = self.LAMBDA_SCALES[self.severity - 1]
        is_uint8 = np.issubdtype(img.dtype, np.integer)

        # Normalize to [0, 1] for Poisson sampling
        img_norm = img.astype(np.float32) / (255.0 if is_uint8 else 1.0)
        img_norm = np.clip(img_norm, 0.0, 1.0)

        # Poisson shot noise: Poisson(I * scale) / scale
        noisy = rng.poisson(img_norm * scale) / scale
        noisy = np.clip(noisy, 0.0, 1.0)

        if is_uint8:
            noisy = (noisy * 255.0).astype(np.uint8)

        if is_torch:
            return torch.from_numpy(noisy).to(device=data.device, dtype=data.dtype)
        return noisy.astype(data.dtype)


class CauterySmokeCorruption(BaseVisualCorruption):
    """Surgical cautery smoke / fog plume attenuating contrast and visibility."""

    OPACITIES = [0.15, 0.30, 0.45, 0.60, 0.75]

    @property
    def name(self) -> str:
        return "cautery_smoke"

    @property
    def level(self) -> str:
        return "frame"

    def apply(
        self,
        data: Union[np.ndarray, torch.Tensor],
        seed: Optional[int] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        if self.severity == 0:
            return copy.deepcopy(data) if isinstance(data, np.ndarray) else data.clone()

        active_seed = seed if seed is not None else self.seed
        rng = np.random.RandomState(active_seed)

        is_torch = isinstance(data, torch.Tensor)
        img = data.cpu().numpy() if is_torch else np.copy(data)

        if not np.all(np.isfinite(img)):
            raise ValueError("Input image contains non-finite values (NaN or Inf).")

        opacity = self.OPACITIES[self.severity - 1]
        is_uint8 = np.issubdtype(img.dtype, np.integer)

        h, w = img.shape[-3], img.shape[-2]

        # Generate smooth smoke density map using low-frequency noise
        small_h, small_w = max(4, h // 16), max(4, w // 16)
        low_res = rng.uniform(0.6, 1.0, size=(small_h, small_w)).astype(np.float32)
        smoke_mask = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_CUBIC)
        smoke_mask = cv2.GaussianBlur(smoke_mask, (31, 31), 10.0)
        smoke_mask = np.clip(smoke_mask * opacity, 0.0, 1.0)

        # Smoke color: off-white/gray smoke (approx. 220 in uint8 or 0.86 in float)
        smoke_color = 220.0 if is_uint8 else 0.86

        if img.ndim == 3:
            smoke_3d = smoke_mask[..., np.newaxis]
            corrupted = (1.0 - smoke_3d) * img.astype(np.float32) + smoke_3d * smoke_color
        elif img.ndim == 4:
            smoke_4d = smoke_mask[np.newaxis, ..., np.newaxis]
            corrupted = (1.0 - smoke_4d) * img.astype(np.float32) + smoke_4d * smoke_color
        else:
            raise ValueError(f"Expected 3D or 4D image array, got ndim={img.ndim}")

        corrupted = np.clip(corrupted, 0.0, 255.0 if is_uint8 else 1.0)

        if is_torch:
            return torch.from_numpy(corrupted).to(device=data.device, dtype=data.dtype)
        return corrupted.astype(data.dtype)


class PacketLossCorruption(BaseVisualCorruption):
    """Digital transmission packet loss causing rectangular macroblock dropouts."""

    BLOCK_SIZES = [8, 16, 16, 32, 32]
    DROP_PROBABILITIES = [0.02, 0.05, 0.10, 0.18, 0.28]

    @property
    def name(self) -> str:
        return "packet_loss"

    @property
    def level(self) -> str:
        return "frame"

    def apply(
        self,
        data: Union[np.ndarray, torch.Tensor],
        seed: Optional[int] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        if self.severity == 0:
            return copy.deepcopy(data) if isinstance(data, np.ndarray) else data.clone()

        active_seed = seed if seed is not None else self.seed
        rng = np.random.RandomState(active_seed)

        is_torch = isinstance(data, torch.Tensor)
        img = data.cpu().numpy() if is_torch else np.copy(data)

        if not np.all(np.isfinite(img)):
            raise ValueError("Input image contains non-finite values (NaN or Inf).")

        block_size = self.BLOCK_SIZES[self.severity - 1]
        drop_prob = self.DROP_PROBABILITIES[self.severity - 1]

        h, w = img.shape[-3], img.shape[-2]
        n_blocks_y = (h + block_size - 1) // block_size
        n_blocks_x = (w + block_size - 1) // block_size

        # Binary dropout mask per macroblock
        drop_mask = rng.rand(n_blocks_y, n_blocks_x) < drop_prob

        corrupted = np.copy(img)
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                if drop_mask[by, bx]:
                    y0, y1 = by * block_size, min(h, (by + 1) * block_size)
                    x0, x1 = bx * block_size, min(w, (bx + 1) * block_size)
                    if img.ndim == 3:
                        corrupted[y0:y1, x0:x1, :] = 0
                    else:
                        corrupted[:, y0:y1, x0:x1, :] = 0

        if is_torch:
            return torch.from_numpy(corrupted).to(device=data.device, dtype=data.dtype)
        return corrupted.astype(data.dtype)


# ---------------------------------------------------------------------------
# Feature-Space Perturbations (Post-Encoder Latent Corruptions)
# ---------------------------------------------------------------------------

class FeatureGaussianNoiseCorruption(BaseVisualCorruption):
    """Additive Gaussian noise in cached latent feature space.
    
    NOTE: This is a synthetic feature perturbation simulating transmission channel
    noise or sensory encoder noise. It is NOT an optical camera degradation.
    """

    NOISE_SCALES = [0.05, 0.10, 0.20, 0.35, 0.50]

    @property
    def name(self) -> str:
        return "feature_gaussian_noise"

    @property
    def level(self) -> str:
        return "feature"

    def apply(
        self,
        data: Union[np.ndarray, torch.Tensor],
        seed: Optional[int] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        if self.severity == 0:
            return copy.deepcopy(data) if isinstance(data, np.ndarray) else data.clone()

        active_seed = seed if seed is not None else self.seed
        rng = np.random.RandomState(active_seed)

        is_torch = isinstance(data, torch.Tensor)
        feat = data.cpu().numpy() if is_torch else np.copy(data)

        if not np.all(np.isfinite(feat)):
            raise ValueError("Input feature contains non-finite values (NaN or Inf).")

        scale = self.NOISE_SCALES[self.severity - 1]
        feat_std = np.std(feat)
        if feat_std < 1e-6:
            feat_std = 1.0

        noise = rng.normal(loc=0.0, scale=scale * feat_std, size=feat.shape).astype(feat.dtype)
        corrupted = feat + noise

        if is_torch:
            return torch.from_numpy(corrupted).to(device=data.device, dtype=data.dtype)
        return corrupted


class FeatureDropoutCorruption(BaseVisualCorruption):
    """Channel-level feature dropout zeroing latent channels."""

    DROP_RATES = [0.05, 0.10, 0.20, 0.35, 0.50]

    @property
    def name(self) -> str:
        return "feature_dropout"

    @property
    def level(self) -> str:
        return "feature"

    def apply(
        self,
        data: Union[np.ndarray, torch.Tensor],
        seed: Optional[int] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        if self.severity == 0:
            return copy.deepcopy(data) if isinstance(data, np.ndarray) else data.clone()

        active_seed = seed if seed is not None else self.seed
        rng = np.random.RandomState(active_seed)

        is_torch = isinstance(data, torch.Tensor)
        feat = data.cpu().numpy() if is_torch else np.copy(data)

        if not np.all(np.isfinite(feat)):
            raise ValueError("Input feature contains non-finite values (NaN or Inf).")

        p = self.DROP_RATES[self.severity - 1]
        # Mask generated on last feature dimension D
        feat_dim = feat.shape[-1]
        keep_mask = (rng.rand(feat_dim) >= p).astype(feat.dtype)

        corrupted = feat * keep_mask

        if is_torch:
            return torch.from_numpy(corrupted).to(device=data.device, dtype=data.dtype)
        return corrupted


class TemporalFrameDropoutCorruption(BaseVisualCorruption):
    """Temporal frame dropout zeroing out entire frames in sliding windows."""

    FRAME_DROP_RATES = [0.05, 0.10, 0.15, 0.25, 0.40]

    @property
    def name(self) -> str:
        return "temporal_frame_dropout"

    @property
    def level(self) -> str:
        return "feature"

    def apply(
        self,
        data: Union[np.ndarray, torch.Tensor],
        seed: Optional[int] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        if self.severity == 0:
            return copy.deepcopy(data) if isinstance(data, np.ndarray) else data.clone()

        active_seed = seed if seed is not None else self.seed
        rng = np.random.RandomState(active_seed)

        is_torch = isinstance(data, torch.Tensor)
        feat = data.cpu().numpy() if is_torch else np.copy(data)

        if not np.all(np.isfinite(feat)):
            raise ValueError("Input feature contains non-finite values (NaN or Inf).")

        if feat.ndim < 2:
            raise ValueError(f"TemporalFrameDropout requires at least 2 dimensions (T, D), got {feat.shape}")

        p = self.FRAME_DROP_RATES[self.severity - 1]
        # Drop mask along temporal dimension T (second to last)
        t_len = feat.shape[-2]
        keep_t = (rng.rand(t_len) >= p).astype(feat.dtype)

        # Reshape keep_t to broadcast with feature dimension D and optional batch B
        if feat.ndim == 2:
            corrupted = feat * keep_t[:, np.newaxis]
        elif feat.ndim == 3:
            corrupted = feat * keep_t[np.newaxis, :, np.newaxis]
        else:
            corrupted = feat

        if is_torch:
            return torch.from_numpy(corrupted).to(device=data.device, dtype=data.dtype)
        return corrupted


class FeatureScaleDriftCorruption(BaseVisualCorruption):
    """Systematic gain drift and additive shift across latent feature channels."""

    DRIFT_SCALES = [0.05, 0.10, 0.20, 0.35, 0.50]

    @property
    def name(self) -> str:
        return "feature_scale_drift"

    @property
    def level(self) -> str:
        return "feature"

    def apply(
        self,
        data: Union[np.ndarray, torch.Tensor],
        seed: Optional[int] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        if self.severity == 0:
            return copy.deepcopy(data) if isinstance(data, np.ndarray) else data.clone()

        active_seed = seed if seed is not None else self.seed
        rng = np.random.RandomState(active_seed)

        is_torch = isinstance(data, torch.Tensor)
        feat = data.cpu().numpy() if is_torch else np.copy(data)

        if not np.all(np.isfinite(feat)):
            raise ValueError("Input feature contains non-finite values (NaN or Inf).")

        delta = self.DRIFT_SCALES[self.severity - 1]
        feat_dim = feat.shape[-1]

        # Multiplicative scaling factor gamma in [1 - delta, 1 + delta]
        gamma = 1.0 + rng.uniform(-delta, delta, size=feat_dim).astype(feat.dtype)
        # Additive shift beta
        feat_std = np.std(feat)
        if feat_std < 1e-6:
            feat_std = 1.0
        beta = rng.uniform(-delta, delta, size=feat_dim).astype(feat.dtype) * feat_std

        corrupted = feat * gamma + beta

        if is_torch:
            return torch.from_numpy(corrupted).to(device=data.device, dtype=data.dtype)
        return corrupted


# ---------------------------------------------------------------------------
# Corruption Registry & Factory
# ---------------------------------------------------------------------------

FRAME_CORRUPTIONS = {
    "defocus_blur": DefocusBlurCorruption,
    "motion_blur": MotionBlurCorruption,
    "shot_noise": ShotNoiseCorruption,
    "cautery_smoke": CauterySmokeCorruption,
    "packet_loss": PacketLossCorruption,
}

FEATURE_CORRUPTIONS = {
    "feature_gaussian_noise": FeatureGaussianNoiseCorruption,
    "feature_dropout": FeatureDropoutCorruption,
    "temporal_frame_dropout": TemporalFrameDropoutCorruption,
    "feature_scale_drift": FeatureScaleDriftCorruption,
}

ALL_CORRUPTIONS = {**FRAME_CORRUPTIONS, **FEATURE_CORRUPTIONS}


def list_available_corruptions() -> Dict[str, List[str]]:
    """List names of all registered corruption operators by operating level."""
    return {
        "frame_level": list(FRAME_CORRUPTIONS.keys()),
        "feature_level": list(FEATURE_CORRUPTIONS.keys()),
    }


def get_visual_corruption(
    name: str,
    severity: int = 1,
    seed: Optional[int] = 42,
) -> BaseVisualCorruption:
    """Instantiate a visual corruption operator by name.
    
    Args:
        name: Name of registered corruption operator.
        severity: Severity integer in [0, 5]. 0 is clean identity.
        seed: Deterministic random seed.
        
    Returns:
        BaseVisualCorruption instance.
        
    Raises:
        KeyError: If corruption name is unregistered.
    """
    clean_name = name.strip().lower()
    if clean_name not in ALL_CORRUPTIONS:
        available = list(ALL_CORRUPTIONS.keys())
        raise KeyError(f"Unknown corruption '{name}'. Available corruptions: {available}")

    operator_cls = ALL_CORRUPTIONS[clean_name]
    return operator_cls(severity=severity, seed=seed)
