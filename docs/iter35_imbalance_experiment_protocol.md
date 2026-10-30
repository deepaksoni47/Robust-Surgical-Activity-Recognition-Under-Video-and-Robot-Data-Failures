# ITER-35 — Experimental Protocol: Class-Imbalance Mitigation for Surgical Gesture Recognition

## 1. Executive Summary & Research Motivation

In surgical gesture recognition on JIGSAWS, gesture distributions exhibit severe class imbalance:
- In **Suturing**, the dominant gesture **G3** ("Pushing needle through tissue") accounts for **33.70%** of all frames (43,815 frames), while rare gesture **G10** ("Loosening more suture") accounts for only **1.12%** (1,459 frames) — an imbalance ratio of **30.0:1**.
- Across the entire benchmark, four dominant gestures (G2, G3, G4, G6) comprise **77.69%** of the Suturing dataset.
- In standard Cross-Entropy training, majority-class gradients overwhelm backpropagation, leading to high frame accuracy (>70%) but severely depressed Macro-F1 (~45–47%) because rare gestures suffer from low recall and boundary suppression.

### Core Research Questions
1. Does addressing class imbalance via loss function reformulation (Class-Weighted CE, Focal Loss, Class-Balanced Focal Loss) significantly improve recognition of rare gestures?
2. Does improving rare-gesture recall incur an unacceptable penalty on dominant-gesture accuracy or temporal sequence quality (Edit Score)?
3. Which formulation provides the Pareto-optimal trade-off between frame-level balanced performance and segmental coherence?

---

## 2. Experimental Design & Controlled Variables

To maintain scientific validity, all conditions are compared under **strictly identical experimental controls**:

| Parameter | Controlled Value | Justification |
|---|---|---|
| **Benchmark Task** | Suturing (Phase 1 focus) | 39 trials, 8 active LOUO folds, 10 gesture classes + background |
| **Model Architecture** | `SurgMotionAdaptiveFusionModel` | 1024D frozen SurgMotion + 76D kinematics + Dynamic Reliability Gate |
| **Cross-Validation Split** | Official 8-fold LOUO | Strict subject isolation; test user never appears in train or validation |
| **Normalization** | `KinematicsNormalizer` (z-score) | Fitted strictly on training fold trials (zero test leakage) |
| **Sequence Length** | 8 frames (center frame prediction) | Standard multimodal temporal window |
| **Batch Size** | 32 | Consistent gradient batching |
| **Epochs** | 25 | Sufficient convergence with early stopping |
| **Optimizer** | AdamW (`lr=1e-3`, `weight_decay=1e-4`) | Standard adaptive optimizer |
| **LR Scheduler** | `ReduceLROnPlateau(factor=0.5, patience=3)` | Guided by validation Macro-F1 |
| **Random Seed** | `42` | Complete deterministic reproducibility |
| **Execution Hardware** | Mentor Machine (Quadro RTX 5000, 16GB) | Single GPU execution |

---

## 3. Evaluated Loss Formulations

### Condition 1: Standard Cross-Entropy Baseline (`ce`)
Unweighted multi-class cross-entropy:
$$\mathcal{L}_{\text{CE}} = - \frac{1}{B} \sum_{i=1}^B \log p_{i, y_i}$$
*Serves as the unweighted empirical control baseline.*

### Condition 2: Class-Weighted Cross-Entropy (`class_weighted`)
Inverse-frequency weighting derived strictly from training fold targets:
$$w_c = \frac{N_{\text{train}}}{C \cdot n_c}, \quad \mathcal{L}_{\text{CW-CE}} = - \frac{1}{\sum_i w_{y_i}} \sum_{i=1}^B w_{y_i} \log p_{i, y_i}$$
*Forces gradient contributions from rare classes to be inversely proportional to their sample frequency.*

### Condition 3: Multi-Class Focal Loss (`focal`) (Lin et al., ICCV 2017)
Dynamically down-weights well-classified easy samples to focus learning on hard, uncertain samples:
$$\mathcal{L}_{\text{FL}} = - \frac{1}{B} \sum_{i=1}^B (1 - p_{i, y_i})^\gamma \log p_{i, y_i}$$
- **Hyperparameter:** Focusing parameter $\gamma = 2.0$.
- **Mechanism:** As prediction confidence on dominant classes increases ($p_t \to 1$), their gradient contribution decays exponentially as $(1 - p_t)^2$, leaving optimizer capacity for rare, ambiguous gestures.

### Condition 4: Class-Balanced Focal Loss (`cb_focal`) (Cui et al., CVPR 2019)
Combines effective-number class weighting with focal modulation:
$$E_{n_c} = \frac{1 - \beta^{n_c}}{1 - \beta}, \quad w_c = \frac{1 - \beta}{1 - \beta^{n_c}}, \quad \mathcal{L}_{\text{CB-FL}} = - \frac{1}{\sum_i w_{y_i}} \sum_{i=1}^B w_{y_i} (1 - p_{i, y_i})^\gamma \log p_{i, y_i}$$
- **Hyperparameters:** $\beta = 0.9999$, $\gamma = 2.0$.
- **Mechanism:** Models the diminishing marginal benefit of additional samples in dominant classes while applying focal hardness modulation.

---

## 4. Class Distribution & Gesture Partitioning (Suturing)

Based on the empirical audit of the 130,026 transcribed frames across all 39 Suturing trials:

| Group | Gestures | Total Frames | Percentage | Gesture Descriptions |
|---|---|:---:|:---:|---|
| **Dominant** (>10%) | G3, G6, G2, G4 | 101,015 | **77.69%** | Pushing needle (33.7%), Pulling suture (18.3%), Positioning tip (13.3%), Transferring needle (12.4%) |
| **Moderate** (3–10%) | G8, G11 | 19,205 | **14.77%** | Orienting needle (8.1%), Dropping suture/needle (6.7%) |
| **Rare** (<3%) | G5, G9, G1, G10, G0 | 9,806 | **7.54%** | Moving needle (2.6%), Right hand tighten (1.7%), Reaching right (1.4%), Loosening suture (1.1%), Idle (0.7%) |

---

## 5. Pre-Registered Hypotheses & Evaluation Metrics

### Evaluation Metrics
Every condition must report all metrics averaged across the 8 LOUO folds ($\mu \pm \sigma$):
1. **Overall Accuracy (%)**: Total correct frames over total frames.
2. **Balanced Accuracy (%)**: Unweighted average of per-class recalls.
3. **Macro-F1 (%)**: Primary metric; unweighted mean of per-class F1-scores.
4. **Segmental Edit Score (%)**: Levenshtein distance on collapsed contiguous gesture sequences (penalizes over-segmentation and label flickering).
5. **Dominant Gesture F1**: Macro-F1 calculated strictly across $\{G2, G3, G4, G6\}$.
6. **Rare Gesture F1**: Macro-F1 calculated strictly across $\{G1, G5, G9, G10\}$.
7. **Per-Class Metrics**: Complete precision, recall, and F1 for each gesture $G0 \dots G11$.

### Pre-Registered Hypotheses
- **H_imb1 (Rare Class Recovery):** Both `class_weighted` and `focal` will increase Rare Gesture F1 by $\ge +4.0$ percentage points over standard `ce`.
- **H_imb2 (Dominant Class Retention):** `focal` loss will preserve Dominant Gesture F1 within $2.0\%$ of standard `ce`, whereas extreme `class_weighted` may degrade dominant precision due to over-prediction of rare classes.
- **H_imb3 (Edit Score Preservation):** Focal loss will maintain segmental Edit Score within $\pm 1.5\%$ of baseline, avoiding excessive label fragmentation.

---

## 6. Validation-Only Protocol (Zero Test Leakage)

To eliminate any risk of test-set overfitting:
1. All class counts are computed **strictly from the training fold** (`train_loader`).
2. Training loss selection and hyperparameter screening ($\gamma \in \{1.0, 2.0, 3.0\}$, $\beta \in \{0.99, 0.999, 0.9999\}$) are conducted **exclusively on validation folds**.
3. Best model checkpoints are selected by peak **validation Macro-F1**.
4. Test folds are evaluated **exactly once** using the frozen model checkpoint.

---

## 7. Recommended Experiment Sequence (Phased Rollout)

To minimize mentor GPU time while ensuring scientific rigor:

```text
Phase 1: Suturing Head-to-Head Comparison (Est. ~25 min on Quadro RTX 5000)
├── Run 1.1: Standard Cross-Entropy Baseline (ce)
├── Run 1.2: Class-Weighted Cross-Entropy (class_weighted)
├── Run 1.3: Focal Loss (focal, gamma=2.0) [RECOMMENDED FIRST CANDIDATE]
└── Run 1.4: Class-Balanced Focal Loss (cb_focal, gamma=2.0, beta=0.9999)

Phase 2: Checkpoint & Metric Analysis
├── Compare Macro-F1, Balanced Acc, Dominant vs Rare F1, and Edit Score
├── Select winning formulation based on decision matrix
└── Verify whether improvement is statistically significant across 8 folds

Phase 3: Benchmark Expansion (Only if Phase 1 is Positive)
├── Run winning loss formulation on Knot Tying (6 classes, 8 folds)
└── Run winning loss formulation on Needle Passing (10 classes, 7 folds)
```

---

## 8. Exact Mentor-Machine Execution Commands

Execute from the repository root `D:\miniproject-5thsem` inside `.venv`:

### Single-Command All-in-One Execution (Recommended for Mentor GPU)

To avoid manual terminal babysitting over multi-hour runs, all 4 Phase 1 conditions are orchestrated sequentially through a single command with automated Windows sleep prevention, preflight checks, comparative summary generation, and automated Git commit & push:

```powershell
# ==============================================================================
# SINGLE-COMMAND ALL-IN-ONE EXECUTION (PHASE 1 SUTURING + AUTO GIT PUSH)
# ==============================================================================

# Option A: Direct PowerShell Script (Windows Mentor GPU)
.\scripts\run_iter35_imbalance_experiment.ps1

# Option B: Direct Python Execution
python scripts/run_iter35_imbalance_experiment.py
```

#### What the All-in-One Script Executes Automatically:
1. **Preflight Assertions:** Checks CUDA availability, VRAM, and confirms all 39 Suturing SurgMotion feature caches exist.
2. **Windows Sleep Prevention:** Uses Windows API `SetThreadExecutionState` to prevent PC sleep/hibernation while training.
3. **Run 1.1:** Standard Cross-Entropy (`ce`) — unweighted baseline.
4. **Run 1.2:** Class-Weighted Cross-Entropy (`class_weighted`) — inverse frequency weighting.
5. **Run 1.3:** Multi-Class Focal Loss (`focal`, $\gamma=2.0$) — primary candidate.
6. **Run 1.4:** Class-Balanced Focal Loss (`cb_focal`, $\gamma=2.0, \beta=0.9999$) — effective number weighting.
7. **Collation & Summary Table:** Computes side-by-side comparison across all 4 formulations and saves `artifacts/iter35_imbalance_comparison_Suturing_summary.json`.
8. **Automated Git Push:** Stages `artifacts/`, `checkpoints/`, `docs/`, `memory.md`, commits with a descriptive message, and runs `git push origin main`.

#### Optional Diagnostic Flags:
```powershell
# Rapid smoke-test verification (1 fold, 1 epoch in ~1 minute without saving artifacts or pushing):
python scripts/run_iter35_imbalance_experiment.py --smoke-test --no-push

# Run full training without pushing to remote repository:
python scripts/run_iter35_imbalance_experiment.py --no-push

# Hardware & cache preflight verification only:
python scripts/run_iter35_imbalance_experiment.py --preflight-only
```

---

### Alternative: Individual Step-by-Step Commands (Manual)

If you ever need to run an isolated condition independently:

```powershell
# Run 1.1: Standard Cross-Entropy Baseline (Unweighted)
python scripts/train_surgmotion_multimodal_fusion.py `
    --task Suturing `
    --fusion-type adaptive `
    --loss-type ce `
    --epochs 25 `
    --batch-size 32 `
    --device cuda

# Run 1.2: Class-Weighted Cross-Entropy (Inverse Frequency)
python scripts/train_surgmotion_multimodal_fusion.py `
    --task Suturing `
    --fusion-type adaptive `
    --loss-type class_weighted `
    --epochs 25 `
    --batch-size 32 `
    --device cuda

# Run 1.3: Multi-Class Focal Loss (gamma = 2.0) [RECOMMENDED FIRST CANDIDATE]
python scripts/train_surgmotion_multimodal_fusion.py `
    --task Suturing `
    --fusion-type adaptive `
    --loss-type focal `
    --gamma 2.0 `
    --epochs 25 `
    --batch-size 32 `
    --device cuda

# Run 1.4: Class-Balanced Focal Loss (gamma = 2.0, beta = 0.9999)
python scripts/train_surgmotion_multimodal_fusion.py `
    --task Suturing `
    --fusion-type adaptive `
    --loss-type cb_focal `
    --gamma 2.0 `
    --beta 0.9999 `
    --epochs 25 `
    --batch-size 32 `
    --device cuda
```

### Artifact Outputs
Each run automatically outputs:
- Results JSON: `artifacts/surgmotion_fusion_adaptive_Suturing_{loss_type}_results.json`
- Model Checkpoints: `checkpoints/surgmotion_fusion_adaptive_Suturing_{loss_type}/model_fold*.pt`
- Preserves all historical baseline files without overwriting.
