# AGENTS.md — Multimodal Surgical Gesture Recognition Research Guide

This document defines the agent operating protocol, project context, architecture guidelines, and single-iteration task roadmap for pair-programming on this research project.

---

## 1. Project Mission & Core Research Question

> **Primary Research Question:**  
> On JIGSAWS surgical gesture recognition, does lightweight temporal fusion of pretrained video representations and robot kinematics outperform unimodal baselines, and does reliability-aware fusion reduce performance loss when one modality is degraded or unavailable?

### Core Hypotheses
- **H1 (Video):** A frozen pretrained visual representation + small temporal head yields a viable video-only baseline.
- **H2 (Kinematics):** A lightweight model on normalized 76D kinematics predicts gestures substantially above majority-class trivial baseline.
- **H3 (Multimodal Fusion):** Simple fusion (late-logit or concatenation) outperforms the strongest unimodal baseline under identical splits.
- **H4 (Reliability-Aware Adaptive Fusion):** A small gating network receiving modality representations and explicit availability/reliability signals maintains higher accuracy under controlled missingness or corruptions than fixed fusion.
- **H5 (Temporal Context):** Sequence modeling (GRU/TCN) over windowed representations outperforms frame-independent predictions.

---

## 2. Research Paper Synthesis & Foundations

| Paper | Key Insights | Project Application |
|---|---|---|
| **GRAD** (`research_papers/GRAD_...`) | Multimodal vision+kinematics fusion with TCN/LSTM, adversarial alignment, and corruption calibration on MISAW/CUHK-MRG. | Motivates unimodal vs multimodal comparison and robustness stress testing. Our MVP uses a lightweight gate rather than complex multi-domain graph disentanglement. |
| **SurgMotion** (`research_papers/SurgMotion_...`) | Video-native foundation model using V-JEPA latent motion prediction on 3,658h dataset. | Justifies using frozen pretrained visual representations instead of training from scratch. |
| **SurgSync** (`research_papers/SurgSync_...`) | Synchronization framework for da Vinci dVRK (kinematics, stereo video, contact). | Emphasizes strict timestamp/frame alignment and metadata manifests. |
| **RobustEndoCLIP / Endo-C6** (`research_papers/RobustEndoCLIP_...`) | Systematic visual corruption benchmark (blur, noise, smoke, defocus, packet loss) reporting Clean, Mean-C, and Worst-C. | Provides our evaluation protocol for synthetic corruption testing. |

---

## 3. Dataset Architecture (JIGSAWS)

- **Location:** `dataset/`
  - `Suturing/` (39 trials across 8 subjects B-I: video capture1 & capture2 .avi, kinematics 76D .txt, transcriptions .txt, metadata)
  - `Knot_Tying/` (36 trials, kinematics, video, transcriptions)
  - `Needle_Passing/` (28 trials, kinematics, video, transcriptions)
  - `Experimental_setup/` (Official 8-fold Leave-One-User-Out `UserOut` and Leave-One-Trial-Out `OneTrialOut` cross-validation splits)
- **Kinematics Format (76D):**
  - Master Left (1-19): XYZ tooltip (3), Rotation matrix R (9), Linear velocity (3), Rotational velocity (3), Gripper angle (1)
  - Master Right (20-38): 19 features
  - Slave Left (39-57): 19 features
  - Slave Right (58-76): 19 features
- **Annotations:** `start_frame end_frame gesture_id` (e.g. `80 219 G1`). Unannotated segments are background/idle.
- **Primary Task Focus:** All 3 JIGSAWS tasks processed in parallel from the beginning:
  - **Suturing:** 39 trials, 10 gestures (G1-G6, G8-G11)
  - **Knot Tying:** 36 trials, 6 gestures (G1, G11-G15)
  - **Needle Passing:** 28 trials, 8 gestures (G1-G6, G8, G11)
  - Total: 103 trials across the entire benchmark.
- **Visual Encoder Target:** Strictly target **SurgMotion** (V-JEPA motion-guided foundation model). Video feature extraction scripts will be built with Google Colab / remote GPU support and clean local caching.

---

## 4. Hardware & Environment Profile

- **Member 1 (Primary Machine):** Windows 11, 12th Gen Intel Core i5-12450H (8C/12T), 16 GB RAM, Intel UHD Graphics, ~22.7 GB free on D:, Python 3.10.11 via `.venv`.
- **Member 2 (Collaborator Machine):** Windows 11 Home, Intel Core Ultra 5 125H (14C/18T, up to 4.5 GHz, AI Boost NPU), 16 GB LPDDR5x RAM, Intel Arc Graphics, 512 GB NVMe SSD.
- **Compute Implication:** The local pipeline must be CPU-efficient and memory-conscious. Visual feature extraction is executed via cloud GPU (Colab/remote) with frozen SurgMotion embeddings cached to disk (`.npz`), enabling fast local training and fusion on both machines.

---

## 5. Agent Workflow Rules & Iteration Protocol

### Rule 1: Single-Iteration Tasks
Every task in the roadmap must be executable within a single agent iteration. The agent must:
1. State the task objective and preconditions.
2. Write/edit code and configuration.
3. Write unit/integration tests covering the new functionality.
4. Run the test/verification command and display output.
5. Update `docs/implementation_tracker.md` and `memory.md`.

### Rule 2: Absolute Leakage Protection
- Splits must be at the **trial/user level** (Leave-One-User-Out).
- Normalization parameters (mean, std, min, max) must be fitted **strictly on training trials**.
- Never tune hyper-parameters or select models using the test set.

### Rule 3: Robust Error Handling
- Never silently drop or impute misaligned timestamps or missing kinematics.
- Always check finite values (`np.isfinite`).
- Fail loudly on unknown labels or mismatched lengths.

### Rule 4: Live Research Paper Updates
- Maintain `paper/main.tex` and `paper/references.bib` throughout the project.
- As baselines, models, ablation studies, and robustness experiments complete, synchronize the research paper LaTeX document with empirical results, architectural equations, and analysis.

---

## 6. Atomic Single-Iteration Task Roadmap

```text
Phase 1: Environment & Project Skeleton
├── ITER-01: Git init, .gitignore, virtual environment & base dependencies [P0]
├── ITER-02: Directory skeleton & reproducibility module with tests [P0]

Phase 2: JIGSAWS Dataset Audit & Manifest
├── ITER-03: Trial inventory & metadata manifest builder (Suturing) [P0]
├── ITER-04: Kinematics parser & 76D integrity validator [P0]
├── ITER-05: Transcription parser & gesture label vocabulary mapper [P0]
├── ITER-06: Video metadata inspector & frame-kinematics alignment auditor [P0]
├── ITER-07: Official LOUO (Leave-One-User-Out) split loader & leakage assertion [P0]

Phase 3: Data Pipeline & Windowing
├── ITER-08: Windowed sequence generator with training-only normalization [P0]
├── ITER-09: End-to-end dataset pipeline smoke test & sanity cache [P0]

Phase 4: Evaluation Suite & Trivial Baseline
├── ITER-10: Metric calculation engine (Macro-F1, Balanced Acc, Acc, Confusion Matrix) [P0]
├── ITER-11: Majority-class trivial baseline implementation & benchmark [P0]

Phase 5: Kinematics Baselines
├── ITER-12: Kinematics MLP classifier (independent timestep) [P0]
├── ITER-13: Kinematics GRU/TCN temporal sequence classifier [P0]

Phase 6: Visual Feature Extraction & Video Baselines
├── ITER-14: Lightweight frozen visual encoder interface (ResNet18/MobileNet) [P0]
├── ITER-15: Visual feature caching engine for Suturing trials [P0]
├── ITER-16: Video-only temporal classifier on cached representations [P0]

Phase 7: Fixed Multimodal Fusion
├── ITER-17: Fixed late-logit fusion with validation weight search [P0]
├── ITER-18: Feature concatenation fusion model [P1]
├── ITER-19: Unimodal vs Fixed Fusion comparative evaluation [P0]

Phase 8: Reliability-Aware Adaptive Fusion
├── ITER-20: Reliability & availability gating network architecture [P0]
├── ITER-21: Clean training and single-modality fallback verification [P0]

Phase 9: Robustness & Controlled Stress Testing
├── ITER-22: Video corruption suite (Endo-C6 inspired blur, noise, occlusions) [P1]
├── ITER-23: Kinematics corruption & dropout operators [P1]
├── ITER-24: Robustness evaluation benchmark (Clean, Mean-C, Worst-C, Missing-modality) [P0]

Phase 10: Results Synthesis & Documentation Handoff
├── ITER-25: Final comparison tables, error analysis & paper-ready figures [P0]
├── ITER-26: Final reproduction report & clean handoff verification [P0]
```
