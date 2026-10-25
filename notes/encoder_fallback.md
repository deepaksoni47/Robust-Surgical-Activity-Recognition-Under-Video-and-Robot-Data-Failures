# Technical Specification: Frozen Visual Encoder Fallback (ResNet-18)

**Date:** October 25, 2026  
**Document ID:** `notes/encoder_fallback.md`  
**Related Milestone:** Phase 6 / Milestone 4 (Visual Feature Extraction & Baselines)  
**Tasks:** `ENC-003`, `REC-004`, `ITER-14`  

---

## 1. Rationale for Fallback Selection

Following the verification in `notes/encoder_review.md` that official SurgMotion checkpoints require manual Hugging Face authorization and dedicated high-VRAM GPU hardware, we specify a clean, reproducible, and verifiable fallback visual encoder architecture.

### Requirements for the Fallback Encoder:
1. **Lightweight & CPU-Feasible:** Must execute feature extraction and forward inference cleanly within local memory constraints (Intel Core i5-12450H, 16 GB RAM) while maintaining remote GPU compatibility.
2. **Standard & Verifiable:** Must use standard, public, non-fabricated pre-trained weights (`torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`).
3. **Strictly Frozen:** All backbone parameters must be permanently frozen (`p.requires_grad = False`, `eval()` mode), matching Hypothesis **H1** (frozen visual representations + compact temporal sequence classifier).
4. **Deterministic Feature Representations:** Given identical input video frames, feature outputs must be mathematically identical (`torch.equal`).
5. **Clear Representation in Research Claims:** Downstream results obtained using this encoder must be explicitly and unequivocally documented as "Frozen ResNet-18 Visual Baseline" — never mislabeled as SurgMotion.

---

## 2. Model Specifications

| Parameter | Specification |
|---|---|
| **Architecture** | ResNet-18 (`torchvision.models.resnet18`) |
| **Pretrained Weights** | ImageNet-1K (`ResNet18_Weights.IMAGENET1K_V1`) |
| **Input Shape** | `[B, C, H, W]` or `[B, T, C, H, W]` |
| **Spatial Resolution** | 224 $\times$ 224 pixels |
| **Input Normalization** | ImageNet standard: mean = `[0.485, 0.456, 0.406]`, std = `[0.229, 0.224, 0.225]` |
| **Output Representation** | 512-dimensional vector per frame (from global average pooling layer, prior to final FC layer) |
| **Total Parameters** | 11,176,512 parameters (11.2M), all frozen |
| **Execution Latency** | ~3.8 ms per frame on Intel CPU |

---

## 3. Modular Architecture Design

Both the primary target (`SurgMotionVisualEncoder`) and the fallback (`ResNet18VisualEncoder`) adhere to a unified base class:
```python
class BaseVisualEncoder(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract frozen visual features.
        
        Args:
            frames: Tensor of shape [B, C, H, W] or [B, T, C, H, W]
        Returns:
            Tensor of shape [B, D] (for single frame) or [B, T, D] (for clip)
        """
        pass

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        pass
```

This guarantees seamless drop-in interoperability when SurgMotion weights are unlocked in the future or executed on remote compute.
