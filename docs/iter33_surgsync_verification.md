# ITER-33 — SurgSync Dataset Verification and Evidence-Based Audit

**Date:** October 28, 2026  
**Status:** Verification Complete — Critical Caveats Uncovered  
**Audited Release:** SurgSync v1.0.0 (Release Option B)  
**Primary Artifact:** `artifacts/iter33_surgsync_schema.json`  
**Reference Sources:**
- Hugging Face Dataset: [https://huggingface.co/datasets/jackzhy/surgsync](https://huggingface.co/datasets/jackzhy/surgsync)
- Project Website: [https://surgsync.github.io/](https://surgsync.github.io/)
- GitHub Toolkit: [https://github.com/jackzhy96/dvrk_multimodal_data_collection](https://github.com/jackzhy96/dvrk_multimodal_data_collection)
- Publication: Zhou et al., *"SurgSync: Time-Synchronized Multi-Modal Data Collection Framework and Dataset for Surgical Robotics"*, IEEE ICRA 2026 (arXiv:2603.06919).

---

## 1. Executive Summary & Core Verdict

The objective of ITER-33 is to verify the actual structure, scale, accessibility, and annotations of the SurgSync dataset before implementing pilot pipelines against promotional or unverified assumptions.

Following an empirical inspection of the official Hugging Face v1.0 metadata, manifests, and sample partitions, **three critical realities** have been uncovered that fundamentally reshape our research strategy:

1. **Storage Footprint is 714 GB, not 12 GB:**  
   The v1.0 public release ships raw **lossless FFV1 video at 1920×1080** stereo + side camera. Video files comprise **714.19 GB (99.98%)** of the 714.35 GB archive. For the suturing task alone (`single_interrupted_stitch`), raw video is **372.33 GB** across 92 episodes. A full download on the primary laptop is impossible given disk constraints (~22 GB free).
2. **Operator IDs are Withheld for Anonymization:**  
   While the project page references 13 human subjects and the ICRA paper describes an 8-subject user study, the public v1.0 release **explicitly strips operator IDs**. Only a 3-bucket skill level (`Novice`, `Intermediate`, `Expert`) is provided. Consequently, **strict Leave-One-User-Out (LOUO) cross-validation cannot be constructed from public files alone** without author collaboration.
3. **Gesture Taxonomy is Dual-Arm Independent, Not JIGSAWS G1–G15:**  
   In JIGSAWS, surgical gestures are annotated as a single system-level label per frame (G1–G15), embedding handedness directly into the class definition (e.g., G1 right reach, G12 left reach). In SurgSync, gestures are annotated **independently per robotic arm** (`gesture.PSM1` and `gesture.PSM2`) across an 18-class clinical vocabulary. Direct zero-shot cross-dataset evaluation is mathematically invalid without re-architecting the classifier heads.

---

## 2. Directly Verified Dataset Facts vs. Uncertain Claims

| Dimension | Initial Assumption / Claim | Verified Ground Truth (Audited) | Status | Evidence Source |
|---|---|---|---|---|
| **Public Accessibility** | Open access | Fully public, non-gated on Hugging Face (`jackzhy/surgsync`) | **VERIFIED** | Hugging Face Dataset API |
| **License** | Open Research Use | **CC-BY-NC-4.0** (Creative Commons Non-Commercial) | **VERIFIED** | `meta/dataset.json`, HF Dataset card |
| **Total Archive Size** | ~12 GB compressed | **714.35 GB (714,345,996,416 bytes)** across 11,745 files | **VERIFIED (CRITICAL)** | `meta/manifest.json` |
| **Video Format** | Compressed H.264 | **Lossless FFV1 in MKV container**, 1920×1080 stereo + side | **VERIFIED** | `meta/dataset.json`, probe of MKV streams |
| **Derived Modalities** | Bundled depth, flow, heatmaps | **Unbundled in v1.0** (must be generated via toolkit locally) | **VERIFIED** | HF Limitations Note 2 |
| **Total Episodes** | 214 validated instances | **205 episodes** (9 episodes dropped from proposal) | **VERIFIED** | `meta/episodes.jsonl` (205 records) |
| **Suturing Task Scale** | 43 instances | **92 episodes** of `single_interrupted_stitch` (90,068 frames, 3.71 h) | **VERIFIED** | `meta/index.parquet/task=single_interrupted_stitch` |
| **Operator Identities** | 8 or 13 identifiable users | **WITHHELD for public anonymization** (only Novice/Inter/Expert) | **VERIFIED (CRITICAL)** | HF Limitations Note, `episodes.jsonl` |
| **Gesture Taxonomy** | JIGSAWS 14 gestures (G1–G14) | **18-class clinical vocabulary per arm** (`gesture.PSM1/2`) | **VERIFIED (CRITICAL)** | `meta/tasks.jsonl`, `index.parquet` |
| **Gesture Availability** | All tasks annotated | **Only 2 of 6 tasks annotated** (Suturing 18, Dissection 11; others 0) | **VERIFIED** | `meta/tasks.jsonl`, `modalities.json` |
| **Observed Gestures** | All 18 populated | **17 populated** (G16 `Cut suture` has 0 frames; 2 ad-hoc codes '00', '01') | **VERIFIED** | Empirical frequency scan of 90,068 frames |
| **Null / Idle Frames** | 0% missing annotations | **2,961 frames (3.29%)** NULL on gestures (Phase 0 prep exposure) | **VERIFIED** | `index.parquet` null mask query |
| **Kinematics Format** | 14D synchronized Cartesian | **14D bimanual Cartesian** (3 pos + 3 lin vel + 1 jaw per arm) | **VERIFIED** | `PSM1.parquet`, `PSM2.parquet` |
| **Master Clock** | 30 Hz nominal | **10 Hz master clock** (`stereo_left` ROS header stamp) | **VERIFIED** | `meta/dataset.json` conventions |
| **Time Sync Accuracy** | Sub-7 ms latency | Offline: **0.37 ms mean delta**; Online: **12.92 ms mean delta** | **VERIFIED** | `time_sync_stat.json` on sample episodes |
| **Gripper Jaw Latency** | Synchronous | **~47 ms average lag** relative to camera master clock | **VERIFIED** | `time_sync_stat.json` (CAN bus loop) |

---

## 3. Deep-Dive: The Three Critical Architectural Constraints

### 3.1 Operator Anonymity and Split Methodology
In JIGSAWS, every trial filename encodes the subject ID (`Suturing_E001` was performed by Subject E). This enabled our rigorous 8-fold Leave-One-User-Out (LOUO) evaluation across 8 distinct surgeons.

In SurgSync v1.0, the public schema **intentionally omits operator identifiers** (`episode_meta.json` contains `operator_skill_level`, but no `subject_id` or `surgeon_id`). The 92 suturing episodes are distributed across skill levels as follows:
- **Intermediate:** 46 episodes (50.0%)
- **Expert:** 33 episodes (35.9%)
- **Novice:** 13 episodes (14.1%)

**Scientific Consequence:**  
A true Leave-One-User-Out evaluation cannot be certified without risk of user identity leakage between train and test. Any standard random split risks placing Stitch A from Surgeon X into training and Stitch B from Surgeon X into testing.  
**Resolution:** For an independent within-dataset SurgSync study, we must adopt a **4-fold skill-stratified and recorder-stratified cross-validation protocol**, or formally request the de-anonymized operator key from the primary author (`Haoying (Jack) Zhou` at JHU/WPI).

### 3.2 Dual-Arm Independent Gestures vs. JIGSAWS Single-Label System
JIGSAWS maps surgical state to 15 global bimanual gestures:
$$\mathcal{Y}_{\text{JIGSAWS}} \in \{G_1, G_2, \dots, G_{15}\}$$
In contrast, SurgSync treats the da Vinci operative arms independently:
$$\mathcal{Y}_{\text{SurgSync}} = \big(\mathcal{Y}_{\text{PSM1}}, \mathcal{Y}_{\text{PSM2}}\big), \quad \mathcal{Y}_{\text{PSM}i} \in \{g_1, g_2, \dots, g_{18}\} \cup \{\text{ad-hoc}\} \cup \{\emptyset\}$$

Empirical distribution across 90,068 frames reveals pronounced role specialization between arms:
- **Needle Driving ($g_6$):** PSM1 (right arm) accounts for **7,943 frames (8.82%)**, whereas PSM2 accounts for only **1,247 frames (1.38%)**.
- **Tissue Counter-Traction ($g_7$):** PSM2 (left arm) accounts for **8,143 frames (9.04%)**, whereas PSM1 accounts for only **169 frames (0.19%)**.
- **Knot Wrapping ($g_{13}$):** Exactly **20,529 frames (22.79%)** on both arms simultaneously.
- **Throw Seating ($g_{15}$):** Exactly **16,453 frames (18.27%)** on both arms simultaneously.
- **Unobserved Class ($g_{16}$):** *"Cut suture"* has 0 frames because the surgical protocol ended after knot securing.
- **Ad-Hoc Codes:** 332 frames carry literal strings `'00'` (bilateral pause, 200 frames) and `'01'` (move camera, 132 frames).
- **Background ($g_0$):** 2,961 frames (3.29%) occur during initial tissue exposure (Phase 0) where no suturing gesture is active.

**Scientific Consequence:**  
A model trained on JIGSAWS cannot be evaluated "zero-shot" on SurgSync. Any cross-dataset study must be formulated as **within-dataset independent evaluation** (training and testing on SurgSync to see if multimodal fusion and reliability gating principles replicate on modern hardware).

### 3.3 Storage Footprint & Hardware Budget
- Total raw download for Suturing: **372.33 GB**
- Total raw download for full dataset: **714.35 GB**
- Disk space on Primary Laptop: **~22.7 GB free** (Cannot store raw video).
- Mentor Machine Disk Space: Needs **~400 GB free** to hold the raw suturing MKVs and extracted feature caches.
- Once extracted, frozen SurgMotion-L representations (1024-D tubelets at 10 Hz) for all 90,068 frames take only **~184.5 MB** as compressed `.npz`!

---

## 4. Time Synchronization Rigor

SurgSync provides hardware/software timestamping far superior to legacy benchmarks:
1. **Master Clock:** Stereo-left camera ROS header capture stamp, downsampled to **10 Hz**.
2. **Offline Data (47 suturing episodes):**
   - Nearest-neighbor interpolation applied at capture.
   - Mean cross-modal offset: **$0.37 \pm 0.32$ ms** (Maximum: 3.91 ms).
   - Dropped frames: **0** (Contiguity ratio: 1.00).
3. **Online Data (45 suturing episodes):**
   - Real-time hardware/software recorder.
   - Mean cross-modal offset: **$12.92 \pm 17.22$ ms** (Maximum: 113.01 ms).
   - Frame dropping active: Enforces strict temporal tolerance, resulting in ~0.67 contiguity ratio.
   - Gripper jaw angle latency: ~47 ms latency due to lower CAN bus cycle rates.

Every frame in `timestamp.parquet` carries bit-exact nanosecond deltas (`delta_to_master.<modality>_ns`), eliminating temporal jitter as a confounding factor.

---

## 5. Verification Verdict & Next Steps

| Assessment Criteria | Evaluation | Verdict |
|---|---|---|
| **Data Authenticity** | Official Hugging Face release verified down to parquet schema | **PASSED** |
| **Gesture Label Integrity** | 87,107 valid labeled frames across 17 active gestures | **PASSED** |
| **Temporal Alignment** | Sub-millisecond offline sync, verified nanosecond deltas | **PASSED** |
| **LOUO Split Feasibility** | Public release strips user IDs; requires skill-stratified 4-fold | **CONDITIONAL** |
| **Local Storage Feasibility** | 372 GB suturing download exceeds laptop storage | **BLOCKED LOCALLY** |
| **Mentor GPU Feasibility** | Mentor workstation can download 372 GB and extract in ~1.5 h | **FEASIBLE** |

**Final Recommendation:** **GO FOR TARGETED SUTURING PILOT PROTOCOL**, subject to executing raw download and feature extraction on the mentor workstation, and utilizing a verified 4-fold skill-stratified cross-validation protocol.
