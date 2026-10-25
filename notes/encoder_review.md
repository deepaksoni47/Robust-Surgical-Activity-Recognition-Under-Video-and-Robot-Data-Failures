# Technical Audit & Decision Gate: SurgMotion Visual Foundation Model

**Date:** October 25, 2026  
**Document ID:** `notes/encoder_review.md`  
**Related Milestone:** Phase 6 / Milestone 4 (Visual Feature Extraction & Baselines)  
**Tasks:** `ENC-001`, `REC-004`  

---

## 1. Executive Summary & Objective

In accordance with `AGENTS.md` and the Research Integrity Rules, this document audits the official candidate visual foundation model: **SurgMotion** (Wu et al., 2026). Prior to writing feature extraction pipelines or training downstream temporal heads, we verify the availability, official model interface, pre-trained weights, license, input requirements, and computational constraints of SurgMotion to prevent fabricated checkpoints, silent model substitutions, or compute failure.

---

## 2. SurgMotion Model Profile & Technical Audit

| Dimension | Audit Finding | Verification Evidence |
|---|---|---|
| **Full Title** | *SurgMotion: A Video-Native Foundation Model for Universal Understanding of Surgical Videos* | arXiv:2602.05638 (Feb 2026) |
| **Authors & Affiliation** | Jinlin Wu et al., Center for Artificial Intelligence and Robotics (CAIR), Hong Kong Institute of Science and Innovation (HKISI), Chinese Academy of Sciences | Author list & institutional affiliations verified |
| **Official Repository** | `https://github.com/CAIR-HKISI/SurgMotion` | GitHub repo indexed; contains probing adapters, configs, and V-JEPA 2 core |
| **Core Architecture** | Video Joint Embedding Predictive Architecture (V-JEPA 2) with latent motion prediction | ViT-Large (300M params, `SurgMotion-vitl`) and ViT-Giant-xformer (1B params, `SurgMotion-vitg-xformer`) |
| **Pre-training Corpus** | SurgMotion-15M (3,658 hours, 15M+ frames across 13+ anatomical domains) | Verified from paper and README |
| **Input Specifications** | 64-frame video clips (`[B, C, 64, H, W]`), 224$\times$224 or 256$\times$256 spatial resolution, 1 FPS dense sampling | Requires spatiotemporal patch cube tokenization (`[B, F*N, D]`) |
| **Downstream Interface** | Spatiotemporal token sequence pooling $\to$ temporal classification head | Probing adapters in `evals/foundation_phase_probing/modelcustom/` |
| **Hugging Face Hub** | `https://huggingface.co/CAIR-HKISI/SurgMotion` and `CAIR-HKISI/SurgMotion-vitl` | Target files: `SurgMotion-vitl.pt`, `config.json` |

---

## 3. Pretrained Weights Availability & Gating Blocker

An automated inspection of the official Hugging Face Hub API (`https://huggingface.co/api/models/CAIR-HKISI/SurgMotion`) was conducted on October 25, 2026:
```json
{
  "id": "CAIR-HKISI/SurgMotion",
  "gated": "manual",
  "private": false,
  "siblings": [
    {"rfilename": ".gitattributes"},
    {"rfilename": "README.md"},
    {"rfilename": "SurgMotion-vitg-xformer.pt"},
    {"rfilename": "SurgMotion-vitl.pt"},
    {"rfilename": "SurgMotion_vitg_probing-head_cholec80.pt"},
    {"rfilename": "Surgmotion-vitb.pt"},
    {"rfilename": "config.json"}
  ]
}
```

### Critical Blocker: Manual Gating
1. **Access Status (`gated: manual`):** The model repository is subject to manual gating by the authors (`CAIR-HKISI`). Direct, anonymous, or automated download of the weights files (`SurgMotion-vitl.pt` / `SurgMotion-vitg-xformer.pt`) via `huggingface_hub` returns HTTP 401/403 Forbidden. Download requires creating an authenticated Hugging Face account, agreeing to individual terms of use, and awaiting manual approval from the repository maintainers.
2. **Weight File Scale:** `SurgMotion-vitl.pt` is a multi-gigabyte PyTorch checkpoint for 300 million parameters.

---

## 4. Hardware & Computational Viability

1. **Host System Constraints (Primary Machine):**
   - OS: Windows 11 64-bit
   - CPU: Intel 12th Gen Core i5-12450H (8 Cores, 12 Threads)
   - GPU: Intel UHD Graphics (Integrated; 0 MB dedicated VRAM, no local CUDA)
   - System RAM: 16 GB (~15.7 GB usable)
2. **Compute Analysis:**
   - SurgMotion's ViT-Large architecture takes 64-frame clips at 224$\times$224 resolution through 24 transformer blocks with latent spatiotemporal cross-attention.
   - Forward inference on CPU for 39 Suturing trials (~284,000 frames = ~4,400 64-frame clips) would require several days of continuous CPU processing with extreme memory pressure, and is strictly designed for multi-GPU CUDA environments (8$\times$ NVIDIA A100/H100 as cited in the paper).
   - Remote GPU execution (e.g. Google Colab / cloud instance) is feasible for batch feature extraction, but requires approved Hugging Face credentials to download the gated checkpoint.

---

## 5. Architectural Decision & Protocol

Per **Rule 1** (no invented APIs/checkpoints) and **Rule 2** (no silent substitutions):
1. **Documented Blocker:** We formally record that `CAIR-HKISI/SurgMotion` weights are gated (`gated: manual`) and inaccessible without individual user credentials.
2. **Interface Implementation:** We implement `SurgMotionVisualEncoder` matching the exact official specifications, which loads `SurgMotion-vitl.pt` if provided locally or via approved HF token, and fails loudly with an informative error if weights are missing.
3. **Fallback Implementation:** In accordance with task `ENC-003` and `REC-004`, we implement and thoroughly test a lightweight frozen visual fallback encoder (`ResNet18VisualEncoder`, 512-D), explicitly labeled in all logs, code, artifacts, and paper sections as the frozen ResNet-18 visual baseline.
