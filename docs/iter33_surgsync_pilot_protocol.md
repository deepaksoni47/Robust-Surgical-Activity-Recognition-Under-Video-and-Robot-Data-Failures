# ITER-33 — SurgSync Pilot Experiment Protocol

**Date:** October 28, 2026  
**Status:** Prepared & Awaiting Approval  
**Target Dataset:** SurgSync v1.0.0 (`single_interrupted_stitch`, 92 episodes, 90,068 frames)  
**Verification Baseline:** `docs/iter33_surgsync_verification.md`  
**Schema Artifact:** `artifacts/iter33_surgsync_schema.json`  

---

## 1. Research Objective & Hypothesis Formulation

The primary research question of this pilot is:
> Does lightweight temporal fusion of frozen SurgMotion visual representations and synchronized robot kinematics outperform unimodal baselines on modern, high-definition ex-vivo surgical robotic data, and does the ReliabilityGate maintain deterministic single-modality fallback when modalities fail?

### Core Hypotheses on SurgSync
- **H1-SurgSync (Video):** Pretrained SurgMotion-L representations on 1080p modern endoscopy provide a viable standalone video baseline.
- **H2-SurgSync (Kinematics):** A 2-layer GRU on normalized 14D bimanual kinematics outperforms the trivial majority baseline on the 18-class suturing taxonomy.
- **H3-SurgSync (Multimodal Fusion):** Fixed and adaptive multimodal fusion outperform the single strongest unimodal modality under identical splits.
- **H4-SurgSync (Reliability Fallback):** When video or kinematics is corrupted or completely unavailable ($a_v=0$ or $a_k=0$), the `ReliabilityGate` maintains stable performance through deterministic routing to the intact stream.

---

## 2. Dataset Subset & Split Protocol

### 2.1 Task Selection
We select **`single_interrupted_stitch`** (92 episodes, 90,068 frames, 3.71 hours), which encompasses the complete clinical workflow of surgical needle positioning, tissue puncture, needle extraction, suture pulling, instrument wrapping, and knot throw seating.

### 2.2 Split Protocol (4-Fold Skill-Stratified Cross-Validation)
Because the public SurgSync v1.0 release strips operator identifiers for privacy, a strict Leave-One-User-Out (LOUO) split cannot be certified from public files. 

We implement a **4-fold skill-stratified and recorder-stratified cross-validation protocol** via `src.data.surgsync_parser.SurgSyncSplitGenerator`:
- **Fold Size:** Exactly 23 validation episodes and 69 training episodes per fold.
- **Stratification Strata:** Balanced across 6 independent strata:
  1. Expert × Online (16 episodes → 4 per fold)
  2. Expert × Offline (17 episodes → 4–5 per fold)
  3. Intermediate × Online (22 episodes → 5–6 per fold)
  4. Intermediate × Offline (24 episodes → 6 per fold)
  5. Novice × Online (7 episodes → 1–2 per fold)
  6. Novice × Offline (6 episodes → 1–2 per fold)
- **Leakage Protection:** Normalization parameters (mean and standard deviation for 14D kinematics) are fitted **strictly on the 69 training episodes** of each fold and never on validation data.

---

## 3. Modalities, Representations, and Preprocessing

### 3.1 Kinematics Stream (14D Bimanual Cartesian Vector)
For each frame $t \in [1, T]$ at 10 Hz, construct a 14-dimensional feature vector:
$$\mathbf{k}_t = \begin{bmatrix} \mathbf{p}_1(t) \\ \mathbf{v}_1(t) \\ \theta_1(t) \\ \mathbf{p}_2(t) \\ \mathbf{v}_2(t) \\ \theta_2(t) \end{bmatrix} \in \mathbb{R}^{14}$$
Where:
- $\mathbf{p}_i(t) \in \mathbb{R}^3$: PSM$i$ Cartesian position ($x, y, z$ in meters).
- $\mathbf{v}_i(t) \in \mathbb{R}^3$: PSM$i$ Cartesian linear velocity ($v_x, v_y, v_z$ in m/s).
- $\theta_i(t) \in \mathbb{R}$: PSM$i$ gripper jaw angle in radians.

Standardize $\mathbf{k}_t$ via z-score normalization using fold-specific training statistics:
$$\hat{\mathbf{k}}_t = \frac{\mathbf{k}_t - \boldsymbol{\mu}_{\text{train}}}{\boldsymbol{\sigma}_{\text{train}} + \epsilon}$$

### 3.2 Visual Stream (1024-D SurgMotion-L Tubelet Embeddings)
- Source: `video_raw/stereo_left.mkv` (1920×1080 FFV1).
- Preprocessing: Resize to $224 \times 224$, group into 16-frame sliding clips at 10 Hz.
- Encoder: Frozen SurgMotion-L (V-JEPA latent motion foundation model).
- Cache: Save extracted embeddings as 1024-D vectors in float16 compressed `.npz` archives.

### 3.3 Target Annotations
- Primary target: `gesture.PSM1` (dominant operative arm, 18 classes).
- Mapping: Canonical gestures 1–18 mapped to class indices 1–18. Ad-hoc codes (`00`, `01`) and Phase 0 preparatory exposure frames mapped to background class (0). Total target vocabulary: $C = 19$ classes.

---

## 4. Models and Architecture Specifications

All models maintain exact parameter parity with our verified JIGSAWS architecture, adjusting only the input projection dimensions:

1. **Majority-Class Baseline:** Predicts the most frequent training gesture for all frames.
2. **Kinematics GRU:**
   - Input layer: `nn.Linear(14, 128)`
   - Sequence trunk: 2-layer GRU, hidden dimension 128, dropout 0.2.
   - Classification head: `nn.Linear(128, 19)`
3. **Video GRU:**
   - Input layer: `nn.Linear(1024, 128)`
   - Sequence trunk: 2-layer GRU, hidden dimension 128, dropout 0.2.
   - Classification head: `nn.Linear(128, 19)`
4. **Fixed Late-Logit Fusion:**
   $$p_{\text{fixed}} = \beta \cdot \text{Softmax}(z_v) + (1 - \beta) \cdot \text{Softmax}(z_k)$$
   where $\beta \in [0, 1]$ is optimized exclusively on training fold validation sequences.
5. **Adaptive Reliability-Aware Fusion:**
   $$\alpha = \text{ReliabilityGate}([h_v; h_k; a_v; a_k]) \in [0, 1]^2, \quad \alpha_v + \alpha_k = 1$$
   $$z_{\text{fused}} = \alpha_v z_v + \alpha_k z_k$$
   with exact fallback enforced when $a_v=0$ or $a_k=0$.

---

## 5. Robustness and Stress-Testing Suite

Evaluate all models across 5 standardized operating regimes:
1. **Clean Test Set:** Intact video and kinematics ($a = [1, 1]$).
2. **Visual Feature Noise:** Additive Gaussian noise $\mathbf{v}' = \mathbf{v} + \epsilon, \, \epsilon \sim \mathcal{N}(0, \sigma_v^2 \mathbf{I})$ at $\sigma_v \in \{0.5, 1.0, 1.5\}$.
3. **Kinematics Sensor Failure:** Additive Gaussian noise on kinematics at $\sigma_k = 1.0$.
4. **Missing Video Stream:** Video completely unavailable ($a = [0, 1]$).
5. **Missing Kinematics Stream:** Robot kinematics completely unavailable ($a = [1, 0]$).

Metrics reported: Macro-F1, Balanced Accuracy, Top-1 Accuracy, Edit Score, and Relative Degradation ($\Delta_{\text{deg}}$).

---

## 6. Hardware Budget & Two-Machine Execution Plan

```
[Machine 1: Mentor Workstation (Quadro RTX 5000, >450 GB free, Fast Network)]
   ├── Step 1: Download 92 episodes of single_interrupted_stitch (~372 GB).
   ├── Step 2: Run scripts/extract_surgmotion_surgsync.py to extract 1024-D tubelets.
   ├── Step 3: Package extracted feature caches (~184 MB total) into surgsync_features.tar.gz.
   └── Step 4: Push feature caches and parquet tables to repository / cloud drive.

[Machine 2: Primary Development Laptop (16 GB RAM, 22 GB free, Local CPU)]
   ├── Step 5: Pull lightweight feature caches (~184 MB) and parquet tables (~80 MB).
   ├── Step 6: Execute 4-fold cross-validation and robustness evaluation on CPU (~40 min).
   └── Step 7: Audit results, compute paired statistics, generate figures, update paper.
```

---

## 7. Go / No-Go Decision Criteria

Before launching the pilot on the mentor machine, the following gates must be cleared:

| Gate | Criterion | Status | Action Required |
|---|---|---|---|
| **Gate 1: Schema & Alignment** | Parquets, timestamps, and vocabulary verified | **CLEARED** | Unit tests passing in `tests/test_surgsync_schema.py` |
| **Gate 2: Storage Availability** | Mentor machine confirmed to have $\ge 450$ GB free disk space | **PENDING USER CONFIRMATION** | Verify free storage on mentor machine |
| **Gate 3: Network Bandwidth** | Ability to download ~372 GB raw video on mentor machine | **PENDING USER CONFIRMATION** | Estimate download time (e.g. ~4 h at 250 Mbps) |
| **Gate 4: Split Approval** | User approval of 4-fold skill-stratified protocol vs waiting for author operator IDs | **PENDING USER DECISION** | Confirm protocol approval |

**Status:** ALL code, schema parsers, tests, and protocols are verified and committed. Awaiting user approval to proceed with mentor machine download and feature extraction.
