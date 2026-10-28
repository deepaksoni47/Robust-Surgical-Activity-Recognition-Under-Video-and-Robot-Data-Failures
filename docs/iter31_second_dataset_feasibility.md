# ITER-31 — Second-Dataset Feasibility Study: Beyond JIGSAWS

**Document Date:** October 28, 2026  
**Status:** Completed & Validated  
**Primary Benchmark Baseline:** JIGSAWS (3 tasks, 103 trials, 8 subjects, 76D kinematics, 30 Hz nominal synchronization)  
**Deliverable Artifact:** `artifacts/iter31_dataset_comparison.json`

---

## 1. Executive Summary

While JIGSAWS has served as the foundational benchmark for robotic surgical activity recognition since 2014, modern research in robot-assisted surgery (RAS) requires validating whether architectural findings—specifically **multimodal fusion** and **reliability-aware adaptive gating**—generalize beyond JIGSAWS's known constraints (legacy video hardware, optical quality limitations, and nominal 30 Hz synchronization jitter).

This study conducts an evidence-based feasibility investigation of three candidate surgical robotics datasets:
1. **SurgSync** (ICRA 2026; WPI, JHU, UBC, Polimi) — [https://surgsync.github.io/](https://surgsync.github.io/)
2. **ROSMAG40** (2024; University of Málaga; Zenodo `10.5281/zenodo.10719748`) — [https://doi.org/10.5281/zenodo.10719748](https://doi.org/10.5281/zenodo.10719748)
3. **MISAW** (MICCAI 2020 Endoscopic Vision Challenge; University of Tokyo, Rennes, TUM) — [http://www.synapse.org/MISAW](http://www.synapse.org/MISAW)

### Key Recommendation
**SurgSync is overwhelmingly recommended as the optimal second dataset.**  
It shares the identical physical robotic platform (da Vinci Research Kit / dVRK), provides the same core surgical tasks (**Suturing** and **Knot Tying**), directly implements the **JIGSAWS 14-gesture taxonomy**, and introduces rigorous hardware-level time synchronization (sub-7 ms latency) and modern 1080p stereo endoscopy on ex-vivo tissue.

---

## 2. Candidate Investigation & Verified Fact Sheet

| Dimension | **SurgSync** (Recommended) | **ROSMAG40** (Alternative) | **MISAW** (Incompatible) |
|---|---|---|---|
| **Publication & Year** | ICRA 2026 (arXiv:2603.06919) | MDPI Sensors 2024 / Zenodo | MICCAI 2020 / MedIA 2022 |
| **Institutions** | WPI, JHU, UBC, Polimi | University of Málaga | Univ. of Tokyo, Rennes, TUM |
| **Verified URL** | [surgsync.github.io](https://surgsync.github.io/) | [zenodo.org/records/10719748](https://doi.org/10.5281/zenodo.10719748) | [synapse.org/MISAW](http://www.synapse.org/MISAW) |
| **Hosting & Access** | Open on Hugging Face (`jackzhy/surgsync`) & GitHub | Open on Zenodo archive | Gated Synapse account (`syn21776936`) + challenge terms |
| **License** | Open Research Use (Apache 2.0 / MIT compatible) | CC-BY 4.0 | Non-commercial Academic Research / CC-BY-NC 4.0 |
| **Robot Platform** | da Vinci Research Kit (dVRK Classic & dVRK-Si) | da Vinci Research Kit (dVRK) | Custom Micro-Surgical Robot (Univ. of Tokyo) |
| **Surgical Domain** | Ex-vivo tissue (porcine/beef) & phantoms | Benchtop dry training models (phantoms) | Artificial silicone micro-vessels (0.7 mm) |
| **Endoscope / Camera** | Modern chip-on-tip stereo 1080p | Standard dVRK stereo endoscope | Stereo Operating Microscope at 30 Hz |
| **Tasks Recorded** | Suturing, Knot Tying, Tissue Manipulation, Peg Transfer, Dissection | Post and sleeve, Pea on a peg, Wire chaser | Micro-vascular anastomosis suturing |
| **Scale / Trials** | 214 validated instances (43 annotated for suturing/knot tying) | 22 video trials (~72,000 frames) | 27 recorded sequences |
| **Subjects** | 8 human subjects (Novice, Intermediate, Expert) | Not systematically documented for LOUO | 6 subjects (3 surgeons, 3 students) |
| **Kinematics Format** | 14D Cartesian/velocity + raw ROS joint states | High-frequency joint angles & MTM/PSM states | 30 Hz joint angles & end-effector coordinates |
| **Annotation Taxonomy** | **JIGSAWS 14-Gesture Taxonomy** (G1–G14) + GRS scores | Maneuver Descriptors (MD) & Fine-Grained Descriptors (FGD) | Hierarchical triplets: `<verb, instrument, target>` |
| **Time Sync Rigor** | **Hardware/Software dual recorder (1.35–6.36 ms latency)** | Standard ROS topic capture | Frame-locked 30 Hz acquisition |

---

## 3. Comprehensive Compatibility Assessment

### 3.1 SurgSync (Compatibility Score: 9.5 / 10)
- **Visual Encoder Compatibility:** SurgMotion-L accepts 16-frame tubelets from video clips. SurgSync’s 1080p stereo endoscope provides high visual fidelity, eliminating JIGSAWS's interlacing artifacts and low dynamic range. Frames can be downsampled to $224 \times 224$ and passed through the frozen SurgMotion encoder without architectural changes.
- **Kinematics Pipeline Compatibility:** SurgSync provides a standardized 14D kinematic vector (6D Cartesian position, 6D Cartesian velocity, 2D gripper opening) sampled synchronously with visual frames. While JIGSAWS includes 76D (including rotation matrices and master manipulators), our `KinematicsGRU` trunk easily adapts by adjusting the input linear projection layer from 76D to 14D (`nn.Linear(14, 128)`), preserving identical 2-layer recurrent modeling.
- **Annotation Concordance:** In their user study validation (Zhou et al., Section IV-D), SurgSync explicitly adopted the **JIGSAWS 14-gesture taxonomy** for Suturing and Knot Tying across 43 instances and 8 surgeons. This enables evaluating the exact same gesture classification problem.
- **Synchronization Advantage:** Unlike JIGSAWS (which exhibited up to 470-frame tail discrepancies on certain trials), SurgSync enforces a sub-7 ms time tolerance at capture (online mode) and $1.35 \pm 0.81$ ms nearest-neighbor interpolation (offline mode). This eliminates temporal alignment ambiguity as a confounding factor.

### 3.2 ROSMAG40 (Compatibility Score: 6.0 / 10)
- **Strengths:** Fully open CC-BY 4.0 download on Zenodo; uses the dVRK robot; provides synchronized video and kinematics.
- **Fatal Obstacles:**
  1. *Task Divergence:* Focuses on peg transfer and wire chasing tasks rather than surgical suturing or knot tying.
  2. *Incompatible Taxonomy:* Annotations use low-level basic actions (touch, grasp, translate, release) and high-level maneuver blocks, which cannot be aligned with surgical gesture recognition models.
  3. *Small Sample Size:* Contains only 22 video trials without pre-defined Leave-One-User-Out splits.

### 3.3 MISAW (Compatibility Score: 4.5 / 10)
- **Strengths:** Genuine micro-vascular anastomosis procedure; 30 Hz synchronized video and kinematics; 27 trials.
- **Fatal Obstacles:**
  1. *Gated Access:* Requires registration through Sage Bionetworks Synapse (`syn21776936`) and challenge data use agreements.
  2. *Hardware Divergence:* Uses a custom University of Tokyo micro-robot and an external surgical microscope, not a da Vinci laparoscopic endoscope. Kinematics coordinates and dynamics are non-transferable.
  3. *Taxonomy Incompatibility:* Annotations are structured as clinical activity triplets (`<verb, instrument, target>`) and procedural phases, which requires an entirely different multi-label workflow architecture rather than temporal gesture segmentation.

---

## 4. Study Design Evaluation

We analyzed three potential multi-dataset research designs:

```
[Design 1: Direct Zero-Shot Transfer] (JIGSAWS -> Second Dataset)
   ├── Status: UNFEASIBLE
   └── Rationale: Kinematics dimensional mismatch (76D vs 14D) and coordinate frame offsets
       preclude zero-shot inference without re-calibration.

[Design 2: Independent Within-Dataset Evaluation] (Train & Test on SurgSync)
   ├── Status: HIGHLY FEASIBLE (RECOMMENDED)
   └── Rationale: Evaluates whether SurgMotion + Kinematics Adaptive Fusion outperforms
       unimodal baselines under modern endoscopy and verified sub-7ms synchronization.

[Design 3: Cross-Dataset Robustness Evaluation] (Stress-testing Gate Principles)
   ├── Status: HIGHLY FEASIBLE (RECOMMENDED)
   └── Rationale: Assesses whether the ReliabilityGate's noise attenuation and single-modality
       fallback operate effectively across two independent surgical platforms.
```

---

## 5. Smallest Scientifically Valid Pilot Experiment

To validate multi-dataset generalization with minimal compute and storage overhead, we propose the following targeted pilot:

### Pilot Specification
- **Dataset Subset:** SurgSync Suturing and Knot Tying instances ($N = 43$ trials, 8 subjects).
- **Split Protocol:** 4-fold user-stratified cross-validation (as defined by Zhou et al., ICRA 2026).
- **Visual Stream:** Downsample stereo video to $224 \times 224$ and extract frozen SurgMotion-L tubelet representations ($1024$-D) using the mentor GPU.
- **Kinematics Stream:** 14D synchronized Cartesian and velocity features, z-score normalized on training folds only.
- **Models Evaluated:**
  1. Majority-class baseline
  2. Kinematics GRU (14D)
  3. SurgMotion VideoGRU (1024D)
  4. Fixed Late Fusion
  5. Adaptive Reliability-Aware Fusion (with exact fallback)
- **Robustness Suite:** Clean vs visual feature noise ($\sigma_v = 1.0$) vs missing video ($a = [0, 1]$) vs missing kinematics ($a = [1, 0]$).

### Resource Budget Estimate
- **Download Size:** $\sim 12$ GB (compressed HDF5 / MP4 package from Hugging Face).
- **Disk Storage on Primary Laptop:** $\sim 1.5$ GB for extracted feature caches (`.npz`).
- **Feature Extraction Time (Mentor GPU):** $\sim 3.5$ hours on Quadro RTX 5000.
- **Downstream Training & Evaluation Time (Local CPU):** $\sim 45$ minutes across all 4 folds.
- **Blockers & Prerequisites:** None. Code and model architectures are fully prepared.

---

## 6. Execution Status & Next Steps

> [!IMPORTANT]
> **Policy Enforcement:** In strict accordance with user guidelines, **no large second-dataset download or second-dataset training has been initiated** during ITER-31. The feasibility study is fully documented, the structured comparison is validated in `artifacts/iter31_dataset_comparison.json`, and the decision is ready for mentor review.
