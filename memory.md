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
| **SurgMotion Feasibility Pilot & Mentor-GPU Execution Package (ITER-25)** | Verified downloaded SurgMotion-L checkpoint (`models/SurgMotion-L/SurgMotion-vitl.pt`, 1.22 GB) and HF auth (`deepaksoni47`). Confirmed exact ViT-L 3D patch embedding and 24-block V-JEPA 2 architecture (303.9M params, 0 missing/0 unexpected keys). Executed CPU smoke test: loaded in 2.65s–5.82s, peak RAM ~1.95 GB; verified outputs (spatiotemporal: `[1, 1024]`, tubelet: `[1, 32, 1024]`, all finite). Measured CPU inference time: 36.8s–51.7s per 64-frame clip (~64.2 hours estimated for 103 trials), decisively justifying offload to mentor's GPU. Prepared portable GPU package: `requirements-surgmotion.txt`, `scripts/extract_surgmotion_features.py` (streaming sliding window, auto SDPA backend detection, atomic saving, isolated `data/cache/features_surgmotion/` output), `scripts/validate_surgmotion_cache.py`, and `docs/surgmotion_gpu_handoff.md`. All 188 tests passing cleanly. |

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
| **ITER-26** | M10 Extraction | Full JIGSAWS SurgMotion Feature Extraction on Mentor GPU | Completed |
| **ITER-27** | M10 Baselines | SurgMotion Downstream Evaluation (VideoMLP & VideoGRU across all 103 trials) | Completed |
| **ITER-28** | M11 Fusion | SurgMotion + Kinematics Multimodal Fusion (Late, Concat, Adaptive on Mentor GPU) | Completed |
| **ITER-29** | M12 Audit & Stats | Multimodal Results Audit, Statistical Testing, and Hypothesis Evaluation | Completed |
| **ITER-30** | M13 Validation | Final Scientific Validation, Publication Figures, Paper Update & Handoff | Completed |
| **ITER-31** | M14 Ablation & Feasibility | Continuous-Noise Gate Ablation & Second-Dataset Feasibility Study | Completed |
| **ITER-32** | M15 Ablation Audit | GPU Results Audit, Gate Weight Flatness Discovery & Paper Update | Completed |
| **ITER-33** | M16 Second Dataset | SurgSync Dataset Verification, Schema Audit & Pilot Protocol (714GB Footprint, Operator Anonymity, Dual-Arm Taxonomy) | Completed |
| **ITER-34** | M16 Acquisition & Pilot | Resumable Suturing Downloader with Disk Preflight, SurgMotion GPU Extractor, Acquisition Tests, and Pilot Protocol | Completed |
| **ITER-34(b)** | M17 Controlled Ablation | JIGSAWS Controlled Ablation & Model Contribution Analysis (Fusion, Temporal Modeling, Rare-Gesture Collapse, Statistical Validation) | Completed |

---

## 7. Key Findings Log — SurgSync Dataset Audit & Pilot Preparation (ITER-33 & ITER-34)
- **Archive Scale:** 714.35 GB total (11,745 files); video accounts for 714.19 GB (99.98%) due to raw lossless FFV1 at 1920×1080 stereo. Parquets are only 150 MB. Suturing video is 372.33 GB across 92 episodes.
- **Acquisition Safeguards:** `scripts/download_surgsync.py` and `scripts/download_surgsync_suturing.py` include automated disk space preflights: $\ge 450.0$ GB free for Suturing subset (~372 GB), $\ge 850.0$ GB free for Full dataset (~714 GB, all 205 episodes across 6 tasks), and $\ge 1-2$ GB for `--skip-video`. Tested and verified locally on Windows.
- **Operator Anonymity & Split Resolution:** Public release v1.0 strips user IDs for anonymization; only 3 skill buckets (Novice, Intermediate, Expert) are provided. Formally resolved as **Episode-Level Stratified Cross-Validation (EL-SCV)** across 4 balanced folds. Explicit non-claim: EL-SCV does NOT guarantee unseen-surgeon generalization due to potential same-surgeon leakage.
- **Gesture Ontology Distinction:** Annotations are dual-arm independent (`gesture.PSM1` and `gesture.PSM2`) using an 18-class clinical vocabulary (17 observed, G16 cut suture unobserved), NOT JIGSAWS G1–G15 single-label system gestures. Primary pilot target defined on PSM1 dominant arm (19 classes: background + 18 gestures).
- **Time Synchronization:** Sub-millisecond on offline data (0.37 ms mean cross-modal delta, 0 dropped frames); 12.92 ms mean delta on online data with ~47 ms jaw angle latency.
- **GPU Feature Extraction:** `scripts/extract_surgmotion_surgsync.py` built with streaming sliding window (16 frames), ImageNet normalization, FP16 autocast, and atomic `.npz` caching (~184 MB for suturing, ~410 MB for all tasks). All 17 SurgSync tests pass.
- **Mentor Runbook Prepared:** Dual-path runbooks for both Scenario A (suturing only, $\ge 450$ GB) and Scenario B (full dataset, $\ge 850$ GB on mentor's 3 TB workstation) documented in `docs/iter34_surgsync_acquisition.md`. Preflight, resumable download, integrity verification, and sample GPU extraction commands ready for execution.

---

## 8. Key Findings Log — JIGSAWS Controlled Ablation & Model Contribution Analysis (ITER-34b)
- **Temporal Modeling Distinction (Workstream B):** Comparing framewise MLP with recurrent sequence GRU across kinematics and vision reveals that temporal sequence modeling produces negligible frame accuracy shifts ($\Delta \le 1.26\%$, all $p > 0.40$), but drives a massive, universally significant improvement in segmental Edit Score ($+22.68$ to $+26.25$ points for unimodal GRU, and $+44.69$ to $+59.17$ points for Multimodal Adaptive GRU; all $p < 0.0005, d_z > 2.2$). Temporal recurrence resolves high-frequency frame-level label chatter, establishing contiguous, clinically coherent gesture segments.
- **Gating Dynamics & Static Prior (Workstream A):** Clamping dynamic gate weights $\alpha_t$ to static trial/fold mean $\bar{\alpha}$ shifts accuracy by $\le 0.06\%$. The gate operates primarily as an availability router and global modality balancer, not a frame-level noise filter.
- **Adaptive vs Baselines (Workstream A):** Adaptive Reliability Fusion outperforms Fixed Late Fusion ($\alpha^*$) across all tasks ($+11.74\%$ Suturing, $+13.39\%$ Knot Tying, $+15.54\%$ Needle Passing), beats Concat Fusion on Suturing ($+18.29\%, p=0.010$) and Knot Tying ($+8.25\%$), and beats Kinematics GRU on Knot Tying ($+11.25\%$, FDR $q=0.021$) and Suturing ($+5.64\%$, FDR $q=0.049$).
- **Rare-Gesture Collapse (Workstream C):** Auxiliary/corrective gestures with $<3\%$ dataset support (G9, G10 in Suturing; G8, G9 in Needle Passing) suffer severe recall collapse ($F1 < 6\%$), explaining why Macro-F1 systematically lags overall frame accuracy.
- **Cross-Subject Generalization Bottlenecks (Workstream C):** Novice/hesitant surgeons (User G) with prolonged durations and idiosyncratic surgeons (User B) present the largest domain shifts and lowest F1 scores across tasks.
- **Artifacts & Figures:** `artifacts/iter34_ablation_summary.json`, `docs/iter34_ablation_analysis.md`, and 5 publication figures in `paper/figures/fig_iter34_*` generated and verified. All 26/26 tests pass.

---

## 9. Key Findings Log — Scientific Audit & Class-Imbalance Preparation (ITER-35)
- **Workstream A (Static Gate Leakage Resolution):** Prior exploratory scripts calculated mean $\alpha$ using test batches (test-time adaptation). The valid leak-free static baseline calculates mean $\alpha$ exclusively from training folds and freezes it prior to test evaluation. Across all active LOUO folds, training-derived static gating performs within $\pm 0.5\%$ of dynamic gating under clean inputs ($p > 0.05$), proving dynamic gating provides no significant benefit over a fixed linear combination under clean inputs. Its true utility is availability routing under sensor loss.
- **Workstream B (Temporal Model Claims):** Audit confirmed identical splits, normalizers, and vocabularies between MLP and GRU. The unscientific claim that GRU "completely eliminates label flickering" was retracted; GRU Edit Scores remain at 31–35% (kinematics) and 40–51% (vision), meaning boundary misalignments persist despite a genuine $+22.68$ to $+26.25$ point segmental coherence boost from temporal hysteresis.
- **Workstream C (Focal Loss Codebase Defect & Resolution):** The proposed command `--loss-type focal --gamma 2.0` in ITER-34 was non-functional; `--loss-type` and `--gamma` CLI options were missing from `scripts/train_surgmotion_multimodal_fusion.py`. Implemented modular `src/models/losses.py` (CE, Class-Weighted CE, Focal Loss, Class-Balanced Focal Loss) with training-only weights. 16 unit tests verified in `tests/test_losses.py`.
- **Workstream D (Class Imbalance on Suturing):** Discovered severe $30.0\times$ imbalance (G3 = 33.7%, G10 = 1.1%; 4 dominant gestures comprise 77.7% of data). Designed a 4-arm controlled experiment on Suturing (`docs/iter35_imbalance_experiment_protocol.md`) pre-registering dominant vs rare metrics and validation-only hyperparameter selection.
- **Recommendation:** Conditional GO for Phase 1 GPU experiment on Suturing using Multi-Class Focal Loss (`--loss-type focal --gamma 2.0`).
- **Unified GPU Execution Architecture & Standing Protocol:** Per developer instruction, multi-run GPU workloads must NEVER require manual terminal babysitting or fragmented CLI execution. Built `scripts/run_iter35_imbalance_experiment.py` and `scripts/run_iter35_imbalance_experiment.ps1` combining:
  1. Automated preflight assertions (CUDA, VRAM, and feature cache checks).
  2. Windows sleep & hibernation prevention via `ctypes.windll.kernel32.SetThreadExecutionState`.
  3. Sequential execution of all 4 Phase 1 runs on Suturing (`ce`, `class_weighted`, `focal`, `cb_focal`).
  4. Comparative summary table and JSON artifact generation (`artifacts/iter35_imbalance_comparison_Suturing_summary.json`).
  5. Automated Git stage, commit, and push to `origin main` upon successful completion.
  *Standing Rule:* All future mentor GPU commands must follow this all-in-one orchestrated pattern.





