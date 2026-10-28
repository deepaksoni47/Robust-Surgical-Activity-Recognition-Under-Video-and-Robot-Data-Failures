# ITER-34 — SurgSync Pilot Experiment Protocol and Split Resolution

**Document Date:** October 28, 2026  
**Status:** Protocol Version 1.0 (Awaiting User Approval Prior to Execution)  
**Target Dataset:** SurgSync v1.0.0 (`single_interrupted_stitch`, 92 episodes, 90,068 frames)  
**Schema Artifact:** [`artifacts/iter33_surgsync_schema.json`](file:///D:/miniproject-5thsem/artifacts/iter33_surgsync_schema.json)  
**Acquisition Guide:** [`docs/iter34_surgsync_acquisition.md`](file:///D:/miniproject-5thsem/docs/iter34_surgsync_acquisition.md)  

---

## 1. Resolution of the Subject-Split Limitation

### 1.1 Verified Dataset Reality
During the ITER-33 schema audit, empirical inspection of `meta/episodes.jsonl` confirmed that **per-episode surgeon/operator identifiers are withheld in the public SurgSync v1.0 release** for participant anonymity. The official Hugging Face documentation explicitly states:
> *"This release does not ship per-episode operator identity — only the three-bucket skill level (Novice: 37, Intermediate: 94, Expert: 74) — in line with anonymization for the public archive."*

For the 92 suturing episodes (`single_interrupted_stitch`), the operator skill distribution is:
- **Intermediate:** 46 episodes (50.0%)
- **Expert:** 33 episodes (35.9%)
- **Novice:** 13 episodes (14.1%)

### 1.2 Scientific Implications & Non-Claim Boundaries
Because surgeon identities are not present in public metadata:
1. **No Claim of Leave-One-User-Out (LOUO):** We **strictly do not claim** a Leave-One-User-Out cross-validation on SurgSync.
2. **Same-Surgeon Leakage Disclosure:** Any episode-level split risks placing stitches performed by the same surgeon into both the training and validation sets. Consequently, **performance on this split cannot be interpreted as conclusive evidence of generalization to unseen surgeons**.
3. **Formal Experiment Nomenclature:** This experiment is formally designated as:
   $$\text{\textbf{Episode-Level Stratified Cross-Validation (EL-SCV)}}$$
   stratified across operator skill levels (`Novice`, `Intermediate`, `Expert`) and recorder variants (`Online`, `Offline`).

### 1.3 Path to True LOUO (Future Step)
To transition from an episode-level feasibility study to strict surgeon-independent evaluation, we have formulated a formal request to the primary author (`Haoying (Jack) Zhou` at JHU/WPI) to obtain an approved de-anonymized operator key. Until such mapping is provided, all results will be explicitly labeled as EL-SCV.

---

## 2. Dual-Arm Gesture Label Representation

### 2.1 Distinction from JIGSAWS
In JIGSAWS, surgical activity is annotated as a single system-level gesture per frame ($G_1$–$G_{15}$), embedding handedness directly into the class label (e.g., $G_1$ reach right hand, $G_{12}$ reach left hand).

In SurgSync, the da Vinci robot's operative arms are annotated **independently**:
$$\mathbf{y}_t = \big(y_t^{\text{PSM1}}, \, y_t^{\text{PSM2}}\big) \in \{0, 1, \dots, 18\}^2$$
Where:
- $y_t^{\text{PSM1}}$: Gesture of Patient Side Manipulator 1 (primary needle-driving arm).
- $y_t^{\text{PSM2}}$: Gesture of Patient Side Manipulator 2 (tissue-manipulation and counter-traction arm).
- $0$: Background / Idle (unannotated frames during Phase 0 tissue exposure; 2,961 frames, 3.29%).
- $1$–$18$: Canonical suturing gestures (Zhou et al., ICRA 2026).
- Ad-hoc codes `'00'` (pause, 200 frames) and `'01'` (camera move, 132 frames) are mapped to background ($0$) or dedicated motion states.

### 2.2 Chosen Evaluation Targets
To maintain scientific clarity and avoid arbitrary label conversions, we evaluate:
1. **Primary Evaluation Target (PSM1 Dominant Arm):**  
   Predicts the gesture of the primary operative arm ($y_t^{\text{PSM1}}$) across $C=19$ classes (Background + 18 gestures). PSM1 executes needle driving ($g_6$), needle grasping ($g_2$), suture pulling ($g_9$), and throw wrapping ($g_{13}$).
2. **Secondary Multi-Task Target (Bimanual PSM1 + PSM2):**  
   A dual-head classifier predicting both arms concurrently, reporting metrics separately for each arm.

**Under no circumstances will SurgSync labels be artificially forced or mapped into JIGSAWS $G_1$–$G_{15}$ IDs.**

---

## 3. Kinematic Feature Vector Specification

### 3.1 14D Bimanual Cartesian Vector
The input kinematic feature vector $\mathbf{k}_t \in \mathbb{R}^{14}$ at each $10\text{ Hz}$ frame $t$ is defined as:

$$\mathbf{k}_t = \begin{bmatrix}
x_1(t), & y_1(t), & z_1(t), & v_{x1}(t), & v_{y1}(t), & v_{z1}(t), & \theta_1(t), &
x_2(t), & y_2(t), & z_2(t), & v_{x2}(t), & v_{y2}(t), & v_{z2}(t), & \theta_2(t)
\end{bmatrix}^T$$

| Dimension | Component Name | Source Field in Parquet | Coordinate Frame | Units | Description |
|---|---|---|---|---|---|
| $0$–$2$ | $\mathbf{p}_1 = [x_1, y_1, z_1]$ | `PSM1.parquet:measured_cp.position` | dVRK Robot Base Frame | meters (m) | PSM1 Cartesian tooltip position |
| $3$–$5$ | $\mathbf{v}_1 = [v_{x1}, v_{y1}, v_{z1}]$ | `PSM1.parquet:measured_cv.linear` | dVRK Robot Base Frame | m/s | PSM1 Cartesian linear velocity |
| $6$ | $\theta_1$ | `PSM1.parquet:jaw.measured_position` | Joint actuator space | radians (rad) | PSM1 gripper jaw opening angle |
| $7$–$9$ | $\mathbf{p}_2 = [x_2, y_2, z_2]$ | `PSM2.parquet:measured_cp.position` | dVRK Robot Base Frame | meters (m) | PSM2 Cartesian tooltip position |
| $10$–$12$ | $\mathbf{v}_2 = [v_{x2}, v_{y2}, v_{z2}]$ | `PSM2.parquet:measured_cv.linear` | dVRK Robot Base Frame | m/s | PSM2 Cartesian linear velocity |
| $13$ | $\theta_2$ | `PSM2.parquet:jaw.measured_position` | Joint actuator space | radians (rad) | PSM2 gripper jaw opening angle |

### 3.2 Preprocessing & Normalization
- **Sampling Frequency:** Master frame clock at $10\text{ Hz}$ (rebased from $\sim 100\text{ Hz}$ native dVRK stream).
- **Normalization:** Fold-specific z-score normalization:
  $$\hat{\mathbf{k}}_t = \frac{\mathbf{k}_t - \boldsymbol{\mu}_{\text{train}}}{\boldsymbol{\sigma}_{\text{train}} + 10^{-6}}$$
  Parameters $\boldsymbol{\mu}_{\text{train}}, \boldsymbol{\sigma}_{\text{train}}$ are fitted **strictly on the 69 training episodes** of each fold.

---

## 4. Visual Stream & SurgMotion-L Feature Extraction

- **Source:** `video_raw/stereo_left.mkv` ($1920 \times 1080$ FFV1 lossless video).
- **Frame Rate:** $10\text{ Hz}$ master clock.
- **Backbone:** Frozen SurgMotion-L (ViT-Large 300M, 24-layer V-JEPA latent motion transformer).
- **Tubelet Window:** $T=16$ frames causal sliding window ($1.6\text{ s}$ temporal context).
- **Resolution:** Spatial downsampling to $224 \times 224$, ImageNet mean/std normalization.
- **Output:** $1024$-D spatiotemporal pooled representation per frame stored as float16 `.npz`.

---

## 5. Model Architecture & Comparative Suite

All models maintain strict architectural parity with our verified JIGSAWS implementations, adjusting only the input projection layer to match the new modality dimensionalities:

```
[Modality Inputs]
   ├── Kinematics (14D) ────► nn.Linear(14, 128)   ────► 2-layer GRU (h=128) ──► Logits z_k (19D)
   └── SurgMotion (1024D) ──► nn.Linear(1024, 128) ────► 2-layer GRU (h=128) ──► Logits z_v (19D)
                                                                 │
                                                                 ▼
[Adaptive Gating] ─────────► ReliabilityGate([h_v; h_k; a_v; a_k]) ────► z_fused = α_v z_v + α_k z_k
```

### Models Compared:
1. **Majority-Class Baseline:** Predicts the most frequent training gesture for all frames.
2. **Kinematics GRU (14D):** Input projection `nn.Linear(14, 128)`, 2-layer GRU ($d=128$, dropout 0.2), classification head `nn.Linear(128, 19)`.
3. **Video GRU (1024D):** Input projection `nn.Linear(1024, 128)`, 2-layer GRU ($d=128$, dropout 0.2), classification head `nn.Linear(128, 19)`.
4. **Fixed Late Fusion:** Logit combination $z_{\text{fixed}} = \beta z_v + (1 - \beta) z_k$, where $\beta \in [0, 1]$ is optimized strictly on training fold validation data.
5. **Adaptive Reliability Fusion:** Dynamic gating $\boldsymbol{\alpha} = \text{ReliabilityGate}([h_v; h_k; a_v; a_k])$ with exact mathematical fallback when $a_v=0$ or $a_k=0$.

---

## 6. Robustness and Sensor-Failure Evaluation

To evaluate whether multimodal fusion provides resilience on modern surgical hardware, models are evaluated across 5 regimes:
1. **Clean Baseline:** Intact video and kinematics ($a = [1, 1]$).
2. **Missing Video Stream:** Camera feed lost or occluded ($a = [0, 1]$).
3. **Missing Kinematics Stream:** Robot kinematics stream fails or drops ($a = [1, 0]$).
4. **Visual Noise Stress Test:** Gaussian feature noise $\sigma_v \in \{0.5, 1.0, 1.5\}$.
5. **Kinematic Sensor Noise:** Joint/velocity jitter $\sigma_k = 1.0$.

Metrics reported: Macro-F1, Balanced Accuracy, Top-1 Accuracy, Edit Score, and Relative Degradation ($\Delta_{\text{deg}}$).

---

## 7. Go / No-Go Decision Gate

Before launching the bulk download and training on the mentor machine, the following gates must be cleared:

| Gate | Requirement | Status | Verification Evidence |
|---|---|---|---|
| **Gate 1: Pipeline Code & Tests** | Download, parser, and extraction scripts tested | **CLEARED** | 14/14 tests pass (`test_surgsync_schema.py`, `test_surgsync_acquisition.py`) |
| **Gate 2: Storage Safety Margin** | $\ge 450\text{ GB}$ free disk space on mentor machine | **CLEARED** | Mentor machine reported to have $\ge 3\text{ TB}$ free storage |
| **Gate 3: GPU Capability** | Quadro RTX 5000 verified for FP16 inference | **CLEARED** | Validated in ITER-28 & ITER-31 mentor GPU runs |
| **Gate 4: Split Acknowledgment** | Explicit acknowledgment of EL-SCV (not LOUO) | **PENDING USER APPROVAL** | User review of this protocol |

**Status:** ALL code, tests, and protocols are committed. Awaiting user approval to proceed with mentor machine download and feature extraction.
