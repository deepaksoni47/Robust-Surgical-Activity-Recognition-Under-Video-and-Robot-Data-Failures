# Milestone 6 Checkpoint: Adaptive Fusion Gate Decision & Benchmark Unblocking Review

**Date:** October 2026  
**Document ID:** `notes/adaptive_fusion_gate.md`  
**Milestone:** M6 Checkpoint (`FUSE-004`)  
**Context:** Pre-ITER-20 Readiness Review under `AGENTS.md`, `memory.md`, and `docs/implementation_tracker.md`

---

## 1. Executive Summary

This review establishes the empirical readiness of the Multimodal Surgical Gesture Recognition project before initiating reliability-aware adaptive fusion (ITER-20/21). 

### Key Findings & Status:
1. **Unimodal Kinematic Baselines are Fully Verified:** Official Leave-One-User-Out (LOUO) evaluations across all 103 trials demonstrate that 76D robot kinematics significantly outperforms trivial majority baselines (Macro-F1: 45.43% vs. 4.55% on Suturing, 53.97% vs. 6.07% on Knot Tying, 30.32% vs. 5.20% on Needle Passing). Sequence modeling via GRU achieves massive $3.4\times$ to $4.3\times$ improvements in segmental Levenshtein Edit Score over independent timesteps, empirically confirming Hypothesis **H5**.
2. **Multimodal Infrastructure is Complete but Real Evaluation is Blocked:** Architectures for Video-Only (`VideoGRU`), Fixed Late-Logit Fusion (`FixedLateFusion`), and Feature Concatenation (`ConcatFusionGRU`) are fully implemented and verified with comprehensive unit and regression tests (140/140 passing). However, real multimodal benchmark evaluation is **BLOCKED** because visual features have been cached for only 1 trial (`Suturing_B001`), leaving 38 Suturing trials and all 64 Knot Tying/Needle Passing trials pending extraction.
3. **Data Alignment Status:** 102 of 103 trials exhibit minor trailing video termination offsets of $\le 17$ frames ($\le 0.57$\,s, median 5 frames). However, legacy JIGSAWS metadata contains no hardware clock pulses; nominal 30 Hz synchronization is an empirical convention with unresolved sub-frame initial jitter. Tail clipping reconciles stream termination but cannot guarantee absolute phase synchronization.
4. **Adaptive Fusion Recommendation:** Adaptive reliability-aware fusion is theoretically justified by clinical failure modes (smoke, lens occlusion, robot packet loss), but empirical claims must not be asserted until multimodal feature caches are extracted. We recommend a **Staged Remote GPU Extraction Strategy**, followed by architectural implementation of the reliability gate (ITER-20) with single-modality fallback verification.

---

## 2. Section A: Data Alignment Audit & Synchronization Analysis

### 2.1 What the 12.37-Frame Mean Discrepancy Actually Measures
The reported mean discrepancy of $12.37$ frames in `artifacts/alignment_audit.json` represents the **arithmetic mean of the absolute difference in stream lengths**:
$$\overline{\Delta}_{\text{vid-kin}} = \frac{1}{103} \sum_{i=1}^{103} |N_{\text{vid}}^{(i)} - N_{\text{kin}}^{(i)}|$$

Crucially, this metric is heavily skewed by a single extreme outlier:
- **`Knot_Tying_F005`:** Video length is $1,379$ frames ($45.97$\,s) while kinematics length is $1,849$ samples ($61.63$\,s), producing a delta of **$470$ frames ($15.67$\,s)**. In this trial, video recording terminated prematurely while kinematic telemetry continued logging.
- **Robust Quantiles across the Remaining 102 Trials:**
  - Minimum delta: $2.0$ frames ($0.07$\,s)
  - 25th percentile ($p25$): $5.0$ frames ($0.17$\,s)
  - Median ($p50$): **$5.0$ frames ($0.17$\,s)**
  - 75th percentile ($p75$): $14.0$ frames ($0.47$\,s)
  - 90th percentile ($p90$): $15.0$ frames ($0.50$\,s)
  - 99th percentile ($p99$): $17.0$ frames ($0.57$\,s)
  - In **102 out of 103 trials**, the video stream is slightly longer than the kinematics stream ($N_{\text{vid}} > N_{\text{kin}}$).

### 2.2 Re-examination of the Suturing B001 Discrepancy
For trial `Suturing_B001`:
- Video frames (`capture1`): $5,640$ frames ($188.00$\,s at $30.0$ FPS)
- Kinematics samples: $5,635$ samples ($187.83$\,s at $30.0$ Hz)
- Transcription segments: 19 gesture segments ending at frame $5,635$.
- Delta: Exactly **$5$ frames ($167$ ms)**.
- **Interpretation:** The 5-frame discrepancy is a clean trailing artifact. The kinematics daemon halted at frame 5,635, while the video capture process recorded for an additional 167 ms before receiving the stop signal. Tail-clipping the video to 5,635 frames discards only trailing idle frames after the surgical task concluded.

### 2.3 Distinction: Stream-Length Differences vs. Synchronization Evidence
To maintain scientific rigor, we distinguish four separate temporal phenomena:

1. **Stream-Length Differences (Trailing Offsets):**
   - *Cause:* Asynchronous software stop signals between the dVRK kinematics logging process and the video frame grabber daemon.
   - *Mitigation:* Tail clipping to $\min(N_{\text{vid}}, N_{\text{kin}})$ safely truncates trailing non-overlapping frames.
2. **Initial Temporal Offsets (Start-Time Jitter):**
   - *Cause:* In JIGSAWS (recorded ~2011–2014), recording was initiated via shell scripts without hardware trigger pulses.
   - *Limitation:* The JIGSAWS distribution contains no global NTP or PTP timestamps in the `.txt` kinematics files or `.avi` headers. Whether frame 0 of the video corresponds precisely to kinematic timestep 0 within $\pm 1$ frame is fundamentally unobservable from open-source metadata alone.
3. **Dropped Frames and Compression Jitter:**
   - *Cause:* Software video encoding at 30.0 FPS vs. da Vinci controller hardware polling at ~30 Hz. Minor clock drift or dropped video frames can accumulate small sub-second phase drift across long trials ($> 180$\,s).
4. **Actual Synchronization Evidence in Literature:**
   - As documented in recent literature (e.g., SurgSync, Zhou et al., 2026), legacy surgical datasets lack hardware-level multimodal synchronization.
   - In JIGSAWS, empirical synchronization is established by transcription segment boundaries: expert annotators labeled gesture start/end frames referencing the kinematics rows, which visually match the tool tip contact events in video within a few frames.

### 2.4 Defensible Limitation Statement
> **Formal Limitation:** In JIGSAWS, video and kinematic streams are assumed nominally synchronized at 30 Hz with 1-to-1 frame pairing. While tail clipping to $\min(N_{\text{vid}}, N_{\text{kin}})$ successfully resolves terminal stream-length differences, potential sub-second initial start-time offsets and minor clock jitter cannot be definitively eliminated from open-source file metadata alone. We explicitly catalog this as an intrinsic dataset limitation, rather than declaring perfect synchronization verified solely because deltas remain below an arbitrary threshold.

---

## 3. Section B: Extraction Feasibility & Resource Budget

### 3.1 CLI Arguments, Configuration Defaults, and Cache Conventions
- **Feature Extraction CLI:** `python scripts/extract_visual_features.py`
  - Required arguments: `--task <Task>` (`"Suturing"`, `"Knot_Tying"`, `"Needle_Passing"`, or `"all"`).
  - Defaults: `--config configs/encoder.yaml`, `--camera capture1`, `--batch-size 16`.
  - Flags: `--verify-only` (readiness audit), `--force` (overwrite existing).
- **Cache Architecture:**
  - Files are stored as compressed `.npz` archives at:
    `data/cache/features/{encoder_name}/{trial_id}_{camera}.npz`
  - Each archive contains: `features` (shape $[N, D_v]$ float32), `frame_indices` ($[N]$ int64), and serialized JSON `metadata` containing trial ID, encoder provenance, weights version, camera view, and extraction timestamp.

### 3.2 Measured & Estimated Cache Storage

| Task | Trials | Video Frames (Capture1) | ResNet-18 Storage (Estimated)* | SurgMotion Storage (Estimated)** |
|---|:---:|:---:|:---:|:---:|
| **Suturing** | 39 | 131,701 | ~231.5 MB (0.23 GB) | ~463.0 MB (0.45 GB) |
| **Knot Tying** | 36 | 61,839 | ~108.7 MB (0.11 GB) | ~217.4 MB (0.21 GB) |
| **Needle Passing** | 28 | 91,104 | ~160.1 MB (0.16 GB) | ~320.2 MB (0.31 GB) |
| **Total (103 trials)** | **103** | **284,644** | **~500.4 MB (0.49 GB)** | **~1,000.7 MB (0.98 GB)** |

*\*Note on ResNet-18 estimates:* Derived from empirically measured trial `Suturing_B001` (5,640 frames = 9.90 MB compressed `.npz` $\approx 1.8$ KB per frame for 512-D float32).  
*\*\*Note on SurgMotion estimates:* Derived assuming 1024-D representation ($2\times$ uncompressed float32 bytes, $\approx 3.6$ KB per frame compressed).

### 3.3 Compute Environment & Extraction Throughput Estimates

| Environment | Device Profile | Estimated Throughput | Suturing (131k frames) | All 103 Trials (284k frames) |
|---|---|:---:|:---:|:---:|
| **Local Machine (Member 1)** | Intel Core i5-12450H CPU (no CUDA GPU) | ~20 FPS | ~1.83 hours | ~3.95 hours |
| **Collaborator (Member 2)** | Intel Core Ultra 5 125H (AI Boost NPU) | ~30 FPS | ~1.22 hours | ~2.64 hours |
| **Cloud GPU (Google Colab T4)** | NVIDIA T4 (16 GB VRAM) | ~400 FPS | ~5.5 minutes | ~11.9 minutes |
| **Cloud GPU (Google Colab A100)**| NVIDIA A100 (40/80 GB VRAM) | ~1,000 FPS | ~2.2 minutes | ~4.7 minutes |

**Device Strategy:** While local CPU extraction is feasible (3.95 hours total), running ResNet-18 extraction on a cloud GPU (Google Colab T4) reduces the entire 103-trial extraction to **under 12 minutes**, producing a single ~500 MB `.zip` archive that can be transferred locally without thermal throttling.

### 3.4 Visual Encoder Provenance
1. **SurgMotion:**
   - *Status:* Formally gated on Hugging Face (`gated: manual` access pending institutional review by authors Wu et al., 2026).
   - *Compute Requirement:* 300M / 1B parameter ViT backbone; requires minimum 16–24 GB CUDA VRAM for batch inference. Local CPU execution is prohibitive.
2. **ResNet-18 Fallback:**
   - *Status:* ImageNet-pretrained torchvision weights (`torchvision.models.resnet18(weights="IMAGENET1K_V1")`), 512-D penultimate pooling representation, completely frozen.
   - *Role:* Verified fallback enabling fully reproducible execution and local test suite integrity. Clearly labeled as ResNet-18 in all logs, configs, and tables.

---

## 4. Section C: Baseline Artifacts Audit

### 4.1 Artifact Status by Model Family

| Model Family | Code & Architecture | Unit Tests Passing | Saved Checkpoints | LOUO 8-Fold Results | Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **Majority Baseline** | `src/models/majority.py` | Yes | N/A (Analytical) | Verified across all 3 tasks (103 trials) | **Complete** |
| **Kinematics MLP** | `src/models/kinematics_mlp.py` | Yes | Fold 1 retained | Verified across all 3 tasks (103 trials) | **Complete** |
| **Kinematics GRU** | `src/models/kinematics_gru.py` | Yes | Fold 1 retained | Verified across all 3 tasks (103 trials) | **Complete** |
| **Video GRU / MLP** | `src/models/video_head.py` | Yes (13/13) | None | Blocked (1/39 Suturing cached) | **Blocked** |
| **Fixed Late Fusion** | `src/models/fusion.py` | Yes (9/9) | None | Blocked (Video predictions missing) | **Blocked** |
| **Concat Fusion GRU** | `src/models/concat_fusion.py` | Yes (7/7) | None | Blocked (Video features missing) | **Blocked** |
| **Benchmark Auditor** | `src/evaluation/benchmark_comparison.py` | Yes (9/9) | N/A | Full audit reports saved | **Complete** |

### 4.2 Leakage & Split Guarantees
- All evaluations enforce official JIGSAWS Leave-One-User-Out (`UserOut`) splits:
  - Suturing: 8 folds (`B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`)
  - Knot Tying: 8 folds (`B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`)
  - Needle Passing: 7 active folds (`B`, `C`, `D`, `E`, `F`, `H`, `I`; User G has no trials).
- In both unimodal and multimodal data loaders, feature normalizers (`KinematicsNormalizer`, `VideoFeatureNormalizer`) and late-fusion weight search (`search_optimal_alpha`) are fit **strictly on training/validation trials**. Test fold labels are never accessed during normalization or hyperparameter tuning.

---

## 5. Section D: Strategic Decision & Experiment Recommendation

### 5.1 Evaluated Options

#### Option 1: Full Local CPU Extraction of ResNet-18 (Monolithic)
- *Procedure:* Run `scripts/extract_visual_features.py --task all` locally on Member 1 CPU.
- *Runtime:* ~4.0 hours continuous processing.
- *Pros:* Self-contained; no external accounts or file transfers needed.
- *Cons:* Heavy CPU utilization and thermal throttling; delays iterative development.

#### Option 2: Wait for SurgMotion HF Approval before Any Multimodal Experiments
- *Procedure:* Halt visual and fusion experiments until Hugging Face manual gating is approved and access to a high-VRAM CUDA GPU is provisioned.
- *Pros:* Avoids running ResNet-18 baseline if SurgMotion is the primary target.
- *Cons:* Unacceptable project bottleneck. Gated manual approval timelines are unpredictable (often weeks). Stalls mini-project milestone deadlines.

#### Option 3: Staged Execution (Recommended)
- **Stage 1 (Primary Task — Suturing):** Extract ResNet-18 features for the 39 Suturing trials first.
  - Can be completed in ~1.8 hours locally (e.g., overnight/background) or in ~5 minutes via Google Colab.
  - Immediately unblocks real 8-fold LOUO evaluation for VideoGRU, Fixed Late Fusion, and Concat Fusion on the primary JIGSAWS task.
- **Stage 2 (Benchmark Extension — Knot Tying & Needle Passing):**
  - Run remaining 64 trials via Colab GPU (~7 minutes) to complete full 103-trial cross-task coverage.
- **Stage 3 (Adaptive Reliability Gating — ITER-20):**
  - Implement the reliability-aware gating network architecture (`src/models/reliability_gate.py`) and single-modality fallback mechanism. Verify behavior with synthetic corruptions and unit tests.
- **Stage 4 (SurgMotion Track):**
  - Keep ResNet-18 and SurgMotion cleanly separated. When SurgMotion access is approved, run parallel extraction to `data/cache/features/SurgMotion/` without invalidating ResNet-18 baselines.

### 5.2 Recommendation & Action Plan
We recommend **Option 3 (Staged Execution)**:
1. Formally record this decision review in `notes/adaptive_fusion_gate.md`.
2. Update the implementation tracker (`FUSE-004`) and `memory.md`.
3. Proceed to **ITER-20** to implement the reliability gating network architecture and single-modality fallback verification, while scheduling the Suturing feature extraction job to unblock real multimodal training.
