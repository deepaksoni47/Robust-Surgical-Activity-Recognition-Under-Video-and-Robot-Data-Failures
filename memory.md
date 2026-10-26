# memory.md — Persistent Research Project Memory Bank

## 1. Project Overview & Objective
- **Project Title:** Adaptive Multimodal Surgical Gesture Recognition
- **Academic Context:** 5th Semester Mini-Project / Research Paper Blueprint
- **Primary Research Question:**
  > On JIGSAWS surgical gesture recognition, does lightweight temporal fusion of pretrained video representations and robot kinematics outperform unimodal baselines, and does reliability-aware fusion reduce performance loss when one modality is degraded or unavailable?
- **Scope Boundary:**
  - In MVP: JIGSAWS surgical gesture classification, unimodal baselines (kinematics-only, video-only), fixed late fusion, adaptive reliability-aware fusion, and controlled test-time corruption benchmarking (visual degradation + kinematic dropout/noise).
  - Out of MVP: Pretraining video foundation models from scratch, phase recognition (no compatible phase dataset verified), clinical claims, full GRAD graph disentanglement reproduction.

---

## 2. Synthesized Research Foundations

1. **GRAD** (`research_papers/GRAD_Multimodal_Graph_Representation_Learning_Robust_Surgical_Workflow.md`)
   - *Key takeaway:* Multimodal vision-kinematic fusion provides complementarity, but naive fusion is fragile under corruption. Employs TCN/LSTM, adversarial alignment, and calibrated decoding.
   - *Our adaptation:* We test whether a simple availability/reliability gate can provide robustness benefits without the overhead of full graph disentanglement.
2. **SurgMotion** (`research_papers/SurgMotion_Video_Native_Foundation_Model_Universal_Understanding.md`)
   - *Key takeaway:* Motion-centric latent space pretraining on 3,658h of surgical video.
   - *Our adaptation:* Demonstrates the value of frozen, pretrained visual embeddings. Confirms we should freeze visual backbones and cache features.
3. **SurgSync** (`research_papers/SurgSync_Time_Synchronized_Multimodal_Data_Collection_Framework.md`)
   - *Key takeaway:* dVRK multimodal collection and strict offline/online timestamp synchronization across stereo video, kinematics, and contact.
   - *Our adaptation:* Informs our alignment audit: timestamps and frame indices must be strictly paired and verified, never guessed.
4. **Endo-C6 / RobustEndoCLIP** (`research_papers/RobustEndoCLIP_EndoC6_Robustness_Temporal_VLMs_Endoscopy.md`)
   - *Key takeaway:* 6 endoscopy-realistic corruptions (defocus, haze, noise, blur, smoke, packet loss); reports Clean, Mean-C, and Worst-C performance.
   - *Our adaptation:* Inspires our test-time robustness evaluation protocol.
5. **Foundation Model Author Correction** (`research_papers/Large_Scale_Self_Supervised_Video_Foundation_Model_Intelligent_Surgery_Correction.md`)
   - *Key takeaway:* Author correction notice for Yang et al., npj Digital Medicine 2026.

---

## 3. Dataset Audit & File Architecture (JIGSAWS)

- **Location:** `d:/miniproject-5thsem/dataset/`
- **Tasks present:**
  - `Suturing/`: 39 trials across 8 subjects (`B001` to `I005`, missing `H002`), 10 gestures (G1-G6, G8-G11).
  - `Knot_Tying/`: 36 trials, 6 gestures (G1, G11-G15).
  - `Needle_Passing/`: 28 trials, 8 gestures (G1-G6, G8, G11).
  - `Experimental_setup/`: Official cross-validation splits including:
    - `Suturing/unBalanced/GestureClassification/UserOut/` (8 folds: `1_Out` to `8_Out` for Leave-One-User-Out).
- **Modality Details:**
  - **Kinematics:** 76 floating-point values sampled at ~30 Hz.
    - Master Left (1-19): Tooltip XYZ (3), Rotation Matrix R (9), Linear Velocity (3), Angular Velocity (3), Gripper Angle (1)
    - Master Right (20-38): 19 features
    - Slave Left (39-57): 19 features
    - Slave Right (58-76): 19 features
  - **Video:** Stereo `.avi` files (`capture1` = Left camera, `capture2` = Right camera) at ~30 FPS, 640x480 resolution.
  - **Transcriptions:** Segment intervals (`start_frame end_frame gesture_id`), 1-indexed to match kinematics rows.
  - **Metadata:** Skill level (`E` = Expert, `I` = Intermediate, `N` = Novice) and modified Global Rating Scores (GRS).

---

## 4. Hardware & Compute Constraints

### Member 1 (Primary Machine — Acer)
- **Processor:** Intel Core i5-12450H (8 Cores: 4P+4E, 12 Threads, up to 4.4 GHz)
- **Graphics:** Intel UHD Graphics (Integrated; no local CUDA)
- **RAM:** 16 GB (~15.7 GB usable)
- **Storage:** ~22.7 GB free on D: drive, ~92.7 GB free on C: drive
- **OS:** Windows 11 (64-bit)

### Member 2 (Collaborator Machine — HP Pavilion Plus 14)
- **Processor:** Intel Core Ultra 5 125H (14 Cores: 4P+8E+2LP-E, 18 Threads, up to 4.5 GHz, AI Boost NPU)
- **Graphics:** Intel Arc Graphics (Integrated)
- **RAM:** 16 GB LPDDR5x high-speed dual-channel RAM
- **Storage:** 512 GB PCIe Gen4 NVMe M.2 SSD
- **Display:** 14" WQXGA (2560x1600), 48-120Hz
- **OS:** Windows 11 Home (64-bit)

### Compute & Execution Strategy
- **Local Work:** All data preprocessing, 76D kinematics normalization, sequence windowing, temporal sequence training (GRU/TCN), multimodal gating, and evaluations are fully CPU/memory-efficient and run locally on both machines.
- **SurgMotion Feature Extraction:** Executed on cloud GPU (Google Colab / remote server) to extract frozen V-JEPA latent representations, which are cached to disk (`data/cache/features/`) for local training and inference.
- **Research Paper LaTeX:** Maintained in `paper/main.tex` and updated incrementally at each milestone to ensure our research paper PDF is complete and submission-ready.

---

## 5. Architectural & Design Decisions

| Decision | Rationale |
|---|---|
| **All 3 JIGSAWS Tasks in Parallel** | Covers all 103 trials across Suturing, Knot Tying, and Needle Passing from the start for comprehensive benchmark evaluation. |
| **Leave-One-User-Out (LOUO) split** | Rigorous evaluation of surgeon-invariant generalization, completely preventing frame/trial leakage across all 3 tasks. |
| **Strictly Target SurgMotion Encoder** | V-JEPA motion-guided surgical video foundation model. Build Google Colab / remote GPU extraction pipeline with deterministic local caching. |
| **Visual Foundation Model + Frozen Fallback** | Audited SurgMotion (Wu et al., 2026); model weights are manually gated ('gated: manual') on Hugging Face and 300M/1B ViT requires high-VRAM CUDA GPU. Implemented unified BaseVisualEncoder interface with ResNet-18 frozen fallback (512-D) and strict SurgMotion checkpoint validation to guarantee reproducible local execution without misrepresenting experimental claims. |
| **Virtual Environment (.venv)** | Clean, isolated workspace environment with pinned dependencies across both machines. |
| **Live LaTeX Research Paper (`paper/main.tex`)** | Continuously updated LaTeX paper blueprint to track methodology, equations, experimental tables, and findings from day 1. |
| **Independent Timestep vs GRU/TCN** | Explicitly verifies Hypothesis H5 (whether temporal sequence modeling adds value). |
| **Late Fusion + Reliability-Aware Gate** | Directly answers Hypotheses H3 & H4 under clean and degraded conditions. |
| **Endo-C6 test-time corruption protocol** | Standardized, non-cherry-picked evaluation of robustness drops (Mean-C, Worst-C). |
| **Video Temporal Classifier (VideoGRU & VideoMLP)** | Built lightweight recurrent sequence classifier (`VideoGRU`) and frame-independent comparison (`VideoMLP`) over cached frozen visual features (512-D), paired with `VideoFeatureNormalizer` (training-only z-score scaling), strict encoder compatibility assertions, and official LOUO split evaluation. |
| **Fixed Late-Logit Fusion & Alpha Grid Search** | Implemented convex combination logit fusion $\mathbf{y}^{\text{fused}} = \alpha \mathbf{y}^{\text{vis}} + (1-\alpha) \mathbf{y}^{\text{kin}}$ with strict validation-only grid search for $\alpha \in [0, 1]$ (preventing any test leakage), verified class vocabulary matching, and synchronized multimodal windowing with tail-clipping convention. |
| **Feature Concatenation Fusion (ConcatFusionGRU & MLP)** | Implemented joint multimodal representation learning concatenating frozen visual features (512-D) and robot kinematics (76-D) into 588-D fused vectors at each timestep, projected through a shared LayerNorm+GELU trunk ($d=128$) into a 2-layer GRU temporal sequence classifier and causal classification head. |
| **Comparative Benchmark & Readiness Auditor** | Built unified multi-modality readiness auditor across all 103 trials checking LOUO split integrity, 76D kinematics, transcriptions, video feature caches, nominal 30 Hz sync (tail delta $\le 500$ frames), and model checkpoints. Standardized matched comparative evaluation with fold coverage assertions, delta calculation vs strongest unimodal baseline, and clean Markdown/CSV export. Real benchmark marked as BLOCKED pending remote GPU visual feature caching. |
| **Adaptive Fusion Gate Review & Staged Unblocking Strategy** | Completed M6 Checkpoint review in `notes/adaptive_fusion_gate.md`. Analyzed temporal alignment metrics (12.37-frame mean vs 5.0 median, single 470f Knot_Tying_F005 outlier; documented dataset start-time jitter limitation), confirmed cache budgets (~500 MB ResNet-18 vs ~1.0 GB SurgMotion across 284k frames), and audited model readiness vs evaluation status. Recommended Staged Remote GPU Extraction Strategy (Suturing first to unblock real multimodal fusion evaluation, followed by remaining tasks). Adaptive fusion architecture justified; ready for ITER-20. |
| **Reliability-Aware Gating Network & Masked Normalization (ITER-20)** | Implemented `ReliabilityGate` and `AdaptiveReliabilityFusion` in `src/models/reliability_gate.py` with explicit availability masking $\mathbf{a} \in \{0, 1\}^2$. Enforces exact single-modality fallback ($\alpha_m \equiv 1.0, \alpha_{\text{other}} \equiv 0.0$ without numerical drift), sub-model execution skipping for unobserved streams, and loud `BothModalitiesUnavailableError` for unobserved inputs. 15 unit tests pass in `tests/test_reliability_gate.py`. |
| **Adaptive Training & Single-Modality Fallback Integration (ITER-21)** | Implemented clean training pipeline in `scripts/train_adaptive_fusion.py` with `configs/adaptive_fusion.yaml`, validation early-stopping on Macro-F1, simulated missingness training, and strict test-time fallback evaluation. Built comprehensive fold readiness audit across all 3 tasks (0/8 folds ready; 38/39 Suturing, 36/36 Knot, 27/27 Needle missing feature caches; kinematics checkpoints exist for Fold 1). Verified submodel execution skipping (absent modality backbones are never called), exact fallback, and checkpoint reload reproducibility (< 1e-6 numerical tolerance). Full benchmark cleanly reported as BLOCKED pending remote GPU feature caching. |
| **Visual Corruption Suite & Robustness Architecture (ITER-22)** | Implemented Endo-C6 inspired frame-level optical corruptions (defocus blur, motion blur, shot noise, cautery smoke, packet loss) and feature-space perturbations (Gaussian jitter, channel dropout, temporal frame dropout, scale drift) in `src/robustness/video_corruptions.py`. Enforces clean identity (sev=0), bitwise determinism for fixed seeds, configurable severity 1–5, shape/finiteness preservation, and strict immutability of source data and labels. Built evaluation engine in `src/robustness/evaluation.py` and CLI in `scripts/evaluate_robustness.py` computing Clean, Mean-C, Worst-C, and Relative Drop under official LOUO splits. Real benchmark cleanly cataloged as BLOCKED pending remote GPU feature caching. |
| **Real-Data Baseline Execution & Evaluation Gate (ITER-23)** | Fully unblocked and executed 100% real empirical benchmark across all 103 trials and 23 active LOUO folds (Suturing 8 folds, Knot Tying 8 folds, Needle Passing 7 active folds; Subject G has 0 trials and is skipped without fabricating empty folds). Extracted ResNet-18 (512-D) features for all 103 trials (`artifacts/feature_cache_audit.json`). Trained and cross-validated KinematicsMLP, KinematicsGRU, VideoGRU, Fixed Late Fusion, ConcatFusion, and Adaptive Reliability Fusion. Evaluated latent perturbations across all 8 folds Suturing and optical frame-level corruptions on raw RGB frames. Verified all five hypotheses (H1–H5); synchronized `paper/main.tex` Table I & Table II; all 180 unit/integration tests passing cleanly. |
| **Final Empirical Audit, Claim Verification, and Paper Freeze (ITER-24)** | Conducted exhaustive audit of all experiment artifacts, checkpoints, and predictions across 23 active LOUO folds (8 Suturing, 8 Knot Tying, 7 Needle Passing). Confirmed Subject G exclusion on Needle Passing traces strictly to empty official split file (`Test.txt` 0 bytes). Recomputed all metrics from raw predictions; computed trial-level edit scores for adaptive fusion (Suturing: 37.34%, Knot Tying: 33.91%, Needle Passing: 29.16%). Reconciled robustness claims: Feature Concatenation has lower relative drop (3.18%) while Adaptive Gating maintains highest absolute retained performance (49.27% Mean-C F1). Performed paired two-sided t-tests (Suturing p=0.0001 vs Kin GRU, p=0.0356 vs Late Fusion). Explicitly clarified frozen ImageNet-pretrained ResNet-18 features (not SurgMotion) and nominal 30 Hz alignment limitations. Paper frozen with all 180 tests passing. |

---

## 6. Single-Iteration Task Progress Tracker

| Iteration ID | Milestone | Task Name | Status |
|---|---|---|---|
| **ITER-01** | M1 Skeleton | Git init, .gitignore, virtual environment & base requirements | Completed |
| **ITER-02** | M1 Skeleton | Directory skeleton & reproducibility module with tests | Completed |
| **ITER-03** | M2 Data Audit | 103-trial multi-task metadata manifest builder (Suturing, Knot, Needle) | Completed |
| **ITER-04** | M2 Data Audit | Kinematics parser & 76D integrity validator | Completed |
| **ITER-05** | M2 Data Audit | Transcription parser & gesture label vocabulary mapper | Completed |
| **ITER-06** | M2 Data Audit | Video metadata inspector & frame-kinematics alignment auditor | Completed |
| **ITER-07** | M2 Data Audit | Official LOUO split loader & leakage assertion | Completed |
| **ITER-08** | M3 Pipeline | Windowed sequence generator with training-only normalization | Completed |
| **ITER-09** | M3 Pipeline | End-to-end dataset pipeline smoke test & sanity cache | Completed |
| **ITER-10** | M4 Evaluation | Metric calculation engine (Macro-F1, Balanced Acc, Acc, CM) | Completed |
| **ITER-11** | M5 Baselines | Majority-class trivial baseline implementation & benchmark | Completed |
| **ITER-12** | M5 Baselines | Kinematics MLP classifier (independent timestep) | Completed |
| **ITER-13** | M5 Baselines | Kinematics GRU/TCN temporal sequence classifier | Completed |
| **ITER-14** | M6 Visual | Lightweight frozen visual encoder interface | Completed |
| **ITER-15** | M6 Visual | Visual feature caching engine for Suturing trials | Completed |
| **ITER-16** | M6 Visual | Video-only temporal classifier on cached representations | Completed |
| **ITER-17** | M7 Fusion | Fixed late-logit fusion with validation weight search | Completed |
| **ITER-18** | M7 Fusion | Feature concatenation fusion model | Completed |
| **ITER-19** | M7 Fusion | Unimodal vs Fixed Fusion comparative evaluation | Completed |
| **FUSE-004** | M6 Checkpoint | Adaptive fusion gate review & benchmark unblocking strategy (`notes/adaptive_fusion_gate.md`) | Completed |
| **ITER-20** | M8 Adaptive | Reliability & availability gating network architecture | Completed |
| **ITER-21** | M8 Adaptive | Clean training and single-modality fallback verification | Completed |
| **ITER-22** | M9 Robustness | Video corruption suite (Endo-C6 inspired blur, noise, occlusions) | Completed |
| **ITER-23** | M9 Robustness | Real-Data Baseline Execution & Evaluation Gate (103 trials, 23 folds, all models, 180 tests pass) | Completed |
| **ITER-24** | M9 Robustness | Final Empirical Audit, Claim Verification, and Paper Freeze | Completed |
| **ITER-25** | M10 Synthesis | Final comparison tables, error analysis & paper figures | Ready to start |
| **ITER-26** | M10 Handoff | Final reproduction report & clean handoff verification | Not Started |


