# Adaptive Multimodal Surgical Gesture Recognition
## Pretrained Visual Representations & Reliability-Aware Dynamic Gating

**Authors:** Deepak Soni, Aparna Banpela  
**Faculty Mentor / Supervisor:** Dr. Princy Matlani  
**Department of Computer Science & Engineering**  
**Institution:** Mini-Project (5th Semester), October 2026  

---

## Slide 1: Research Problem & Clinical Motivation

### Clinical & Technological Context
- **Robot-Assisted Minimally Invasive Surgery (RMIS):** Systems like the da Vinci Surgical System enhance dexterity, but lack autonomous context awareness.
- **Surgical Gesture Recognition:** Decomposing complex surgical workflows into fine-grained atomic gestures (e.g., reaching needle, positioning, suture pulling) is essential for:
  - Objective surgeon skill assessment & technical credentialing
  - Context-aware robotic assistance & camera automation
  - Autonomous sub-task execution & safety boundary enforcement

### The Complementary Sensing Dilemma
1. **Endoscopic Video (Optical Channel):**
   - *Strengths:* Captures tissue deformation, suture state, and anatomic context.
   - *Vulnerabilities:* Electrocautery smoke, lens fogging, fluid splatter, motion blur, and illumination collapse.
2. **Robot Kinematics (Physical Telemetry):**
   - *Strengths:* Continuous 30 Hz Cartesian coordinates, rotation matrices, gripper angles, velocities; completely invariant to optical smoke/fog.
   - *Vulnerabilities:* Lacks environmental contact context; susceptible to sensor dropout or calibration drift.

### The Problem with Conventional Fusion
- Standard multimodal models use **static concatenation** or **uncalibrated late fusion**, which degrades accuracy when visual noise corrupts confident kinematics predictions.

---

## Slide 2: Research Question & Core Hypotheses

### Primary Research Question
> *On JIGSAWS surgical gesture recognition, does lightweight temporal fusion of pretrained video representations and robot kinematics outperform unimodal baselines, and does reliability-aware dynamic gating reduce performance loss when one modality is degraded or unavailable?*

### The Five Formal Hypotheses
- **H1 (Visual Representation Viability):** Frozen latent features from a pretrained visual model (SurgMotion / ResNet-18) + lightweight temporal classifier yield a viable unimodal baseline above chance.
  - *Verdict: Supported* (Viable Balanced Acc $31.0\%-54.5\%$ vs $10.3\%-14.9\%$ majority baseline).
- **H2 (Kinematic Information):** Lightweight sequence modeling on normalized 76-D kinematics provides strong gesture discrimination.
  - *Verdict: Strongly Supported* ($p < 0.0001, d_z > 3.0$; single most predictive unimodal modality).
- **H3 (Multimodal Complementarity):** Multimodal fusion outperforms the strongest unimodal baseline under strictly identical LOUO splits.
  - *Verdict: Partially Supported* (Statistically significant on Knot Tying $+11.25\%$, $p_{\text{adj}}=0.0286$; strong descriptive trend on Suturing $+5.64\%$, $p_{\text{adj}}=0.0820$).
- **H4 (Reliability-Aware Gating):** Dynamic gating maintains accuracy under corruptions and enforces single-modality fallback.
  - *Verdict: Supported with Documented Nuance* (Acts as an effective availability router and global modality balancer, not an active continuous noise sensor).
- **H5 (Temporal Sequence Modeling):** Temporal context (GRU) significantly reduces oversegmentation chatter compared to framewise models.
  - *Verdict: Strongly Supported for Segmental Coherence* ($+22$ to $+59$ pt Edit Score improvement, $p < 0.0005, d_z > 2.2$).

---

## Slide 3: JIGSAWS Benchmark & Experimental Protocol

### Benchmark Architecture (All 103 Trials)
- **Suturing:** 39 trials, 8 subjects (B–I), 10 gesture classes (G1–G6, G8–G11)
- **Knot Tying:** 36 trials, 8 subjects (B–I), 6 gesture classes (G1, G11–G15)
- **Needle Passing:** 28 trials, 8 subjects (B–I), 8 gesture classes (G1–G6, G8, G11)
- **Total Benchmark:** 103 trials across 15 unique gesture vocabularies.

### 76-D da Vinci Kinematics Formulation
- Four manipulators: Master Left ($19\text{D}$), Master Right ($19\text{D}$), Slave Left ($19\text{D}$), Slave Right ($19\text{D}$).
- Each manipulator encompasses:
  - 3D Tool-tip Position $\mathbf{p} = [x, y, z]^\top$
  - 9D Rotation Matrix $\mathbf{R} \in \mathbb{R}^{3 \times 3}$
  - 3D Linear Translational Velocity $\mathbf{v} = [\dot{x}, \dot{y}, \dot{z}]^\top$
  - 3D Rotational Angular Velocity $\boldsymbol{\omega} \in \mathbb{R}^3$
  - 1D Gripper Jaw Angle $\theta \in \mathbb{R}^1$

### Absolute Leakage Protection Protocol
- **Cross-Validation:** Official Leave-One-User-Out (LOUO). All trials of an unseen surgeon are reserved exclusively for testing.
  - Suturing: 8 folds ($N=8$)
  - Knot Tying: 8 folds ($N=8$)
  - Needle Passing: 7 active folds ($N=7$; User G contains an empty test set in official distribution).
- **Normalization:** Z-score parameters $(\boldsymbol{\mu}_{\text{train}}, \boldsymbol{\sigma}_{\text{train}})$ computed **strictly on training trials**. Zero test leakage.

---

## Slide 4: Proposed Multimodal Architecture

### Design Philosophy: Decoupled & Computationally Lightweight
Instead of end-to-end heavy backbones prone to overfitting on small surgical datasets, we decouple visual representation from temporal sequence modeling:

1. **Frozen Visual Foundation Encoder:**
   - SurgMotion-ViTL ($D_v = 1024$) trained on 3,658 hours of multi-organ surgery via Video-JEPA motion prediction.
   - ResNet-18 ($D_v = 512$) ImageNet baseline.
   - Cached to disk as static embeddings for reproducible, energy-efficient CPU/GPU training.

2. **Modality-Specific Temporal Sequence Encoders:**
   - Kinematics Projection: $76\text{D} \rightarrow 128\text{D}$ (LayerNorm, GELU, Dropout 0.2) + 2-layer GRU.
   - Video Projection: $1024\text{D} \rightarrow 128\text{D}$ (LayerNorm, GELU, Dropout 0.2) + 2-layer GRU.
   - Temporal context window $S=8$ timesteps.

3. **Reliability-Aware Dynamic Gating Network:**
   - Input: Concatenated hidden representations $[\mathbf{h}_t^{\text{vis}}; \mathbf{h}_t^{\text{kin}}]$ and explicit binary availability vector $\mathbf{a}_t = [a_t^{\text{vis}}, a_t^{\text{kin}}]^\top \in \{0, 1\}^2$:
   $$\tilde{\boldsymbol{\alpha}}_t = \text{Softplus}\left( \mathbf{W}_g [\mathbf{h}_t^{\text{vis}}; \mathbf{h}_t^{\text{kin}}; \mathbf{a}_t] + \mathbf{b}_g \right)$$
   $$\alpha_t^m = \frac{\tilde{\alpha}_t^m \, a_t^m}{\sum_{j \in \mathcal{M}} \tilde{\alpha}_t^j \, a_t^j + \epsilon}, \quad \hat{\mathbf{y}}_t = \alpha_t^{\text{vis}} \mathbf{y}_t^{\text{vis}} + \alpha_t^{\text{kin}} \mathbf{y}_t^{\text{kin}}$$
   - **Guaranteed Single-Modality Fallback:** When $a_t^{\text{vis}} = 0$, $\alpha_t^{\text{vis}} \equiv 0$ and $\alpha_t^{\text{kin}} \equiv 1$, enforcing exact unimodal prediction $\hat{\mathbf{y}}_t = \mathbf{y}_t^{\text{kin}}$ without numerical drift.

---

## Slide 5: Benchmark Results: Unimodal vs. Multimodal

### Empirical Cross-Validation Across All 103 Trials

| Task | Model Architecture | Modality | Accuracy (%) | Balanced Acc (%) | Macro-F1 (%) | Edit Score (%) |
|---|---|---|---|---|---|---|
| **Suturing** | Majority Baseline | None | $33.43 \pm 2.34$ | $10.30 \pm 0.69$ | $4.55 \pm 0.24$ | $4.80 \pm 0.39$ |
| ($N=8$) | ResNet-18 VideoGRU | Video | $45.42 \pm 10.96$ | $42.64 \pm 9.03$ | $33.22 \pm 7.50$ | $20.01 \pm 3.88$ |
| | SurgMotion VideoGRU | Video | $39.17 \pm 13.51$ | $34.10 \pm 8.69$ | $22.16 \pm 7.95$ | $39.99 \pm 9.51$ |
| | Kinematics GRU | Kinematics | $66.64 \pm 9.01$ | $57.77 \pm 9.40$ | $\mathbf{45.09 \pm 8.24}$ | $34.69 \pm 9.43$ |
| | Fixed Late Fusion | Multimodal | $60.54 \pm 8.15$ | $50.26 \pm 10.64$ | $37.01 \pm 10.14$ | $60.41 \pm 15.07$ |
| | Feature Concat Fusion | Multimodal | $54.00 \pm 15.45$ | $46.90 \pm 14.82$ | $32.29 \pm 10.45$ | $49.19 \pm 14.12$ |
| | **Adaptive Reliability Fusion (Ours)** | Multimodal | $\mathbf{72.28 \pm 9.57}^\dagger$ | $\mathbf{60.05 \pm 11.44}$ | $44.64 \pm 8.26$ | $\mathbf{68.95 \pm 9.38}^{**}$ |
| **Knot Tying** | Kinematics GRU | Kinematics | $61.09 \pm 11.79$ | $56.53 \pm 12.43$ | $51.75 \pm 15.14$ | $35.48 \pm 13.86$ |
| ($N=8$) | Feature Concat Fusion | Multimodal | $64.09 \pm 11.29$ | $63.37 \pm 9.23$ | $53.97 \pm 9.91$ | $62.45 \pm 11.77$ |
| | **Adaptive Reliability Fusion (Ours)** | Multimodal | $\mathbf{72.34 \pm 9.70}^{*}$ | $\mathbf{66.73 \pm 8.59}^{*}$ | $\mathbf{60.48 \pm 10.20}^{*}$ | $\mathbf{64.66 \pm 12.61}^{**}$ |
| **Needle Passing** | Kinematics GRU | Kinematics | $48.69 \pm 11.62$ | $41.70 \pm 9.97$ | $29.99 \pm 9.35$ | $31.34 \pm 7.32$ |
| ($N=7$) | Feature Concat Fusion | Multimodal | $46.21 \pm 19.07$ | $42.32 \pm 19.34$ | $28.40 \pm 12.37$ | $45.17 \pm 20.64$ |
| | **Adaptive Reliability Fusion (Ours)** | Multimodal | $\mathbf{51.48 \pm 13.94}$ | $\mathbf{46.98 \pm 15.16}$ | $\mathbf{31.03 \pm 11.52}$ | $\mathbf{54.81 \pm 9.40}^{**}$ |

*Statistical significance vs. Kinematics GRU (Holm-Bonferroni): $^{*} p_{\text{adj}} < 0.05$, $^{**} p_{\text{adj}} < 0.001$; $^\dagger$ marginal trend ($p_{\text{adj}} = 0.082$).*

### Key Benchmark Takeaways
1. **Kinematics Dominance:** Kinematics alone achieves $66.64\%$ accuracy on Suturing—substantially higher than frozen video ($39.17\%$).
2. **Failure of Naive Fixed Fusion:** Fixed late fusion drops $-6.10\%$ on Suturing ($60.54\%$ vs $66.64\%$) because static probability weighting dilutes confident kinematics predictions with noisy video representations.
3. **Adaptive Fusion Wins Universally:** Delivers highest accuracy on all three tasks; statistically significant on Knot Tying ($+11.25\%$, $p_{\text{adj}}=0.0286$).

---

## Slide 6: Temporal Modeling & Gate Dynamics Findings

### Finding 1: Sequence Modeling Resolves Segmental Fragmentation (H5)
- **Frame Accuracy vs. Edit Score Paradox:**
  - Framewise Kinematic MLP vs. Recurrent GRU: Frame accuracy difference is negligible ($\Delta \le 1.26\%$, $p > 0.40$).
  - Segmental Edit Score: Recurrent GRU nearly **doubles** sequence coherence:
    - Kinematics GRU: $+22.68$ to $+26.25$ points over MLP ($p < 0.0005, d_z > 2.2$).
    - Adaptive Multimodal GRU: $+34.26$ to $+59.17$ points over framewise models ($p < 0.0001, d_z > 3.0$).
- **Mechanism:** Sequential recurrence enforces temporal regularizing hysteresis, eliminating rapid single-frame label oscillation at gesture transition boundaries.

### Finding 2: Dynamic Gate Operates as Modality Balancer & Availability Router
- Over 4.2 million time-resolved gate weight logs audited across all 23 folds:
  - Kinematics weight heavily dominates: $\bar{\alpha}_k \approx 80\%-95\%$ across all tasks.
  - Video feature weight acts as a global contextual regularizer: $\bar{\alpha}_v \approx 5\%-20\%$.
  - Replacing time-varying $\alpha_t$ with static fold mean $\bar{\alpha}$ shifted clean accuracy by $< 0.06\%$.
  - **Honest Finding:** The gate does not act as a high-frequency continuous optical noise sensor; rather, it provides **global modality balancing** and **discrete availability routing**.

---

## Slide 7: Robustness Stress Testing & Sensor Dropouts

### Evaluation Across 5 Standardized Stress Conditions

| Condition | Availability Vector | Suturing Acc (%) | Knot Tying Acc (%) | Needle Passing Acc (%) | Clinical Significance |
|---|:---:|:---:|:---:|:---:|---|
| **Clean Baseline** | $\mathbf{a} = [1, 1]$ | $72.28\%$ | $72.34\%$ | $51.48\%$ | Standard surgical operation |
| **Video Corrupted** | $\mathbf{a} = [1, 1], \sigma_v=1.0$ | $72.22\%$ ($-0.06\%$) | $72.22\%$ ($-0.12\%$) | $51.48\%$ ($0.00\%$) | Electrocautery smoke / lens fog |
| **Kinematics Corrupted** | $\mathbf{a} = [1, 1], \sigma_k=1.0$ | $64.09\%$ ($-8.19\%$) | $67.37\%$ ($-4.97\%$) | $48.22\%$ ($-3.26\%$) | Sensor calibration drift / tremor |
| **Missing Video** | $\mathbf{a} = [0, 1], \mathbf{v}=0$ | $71.86\%$ ($-0.42\%$) | $70.06\%$ ($-2.28\%$) | $51.21\%$ ($-0.27\%$) | Camera occlusion / frame loss |
| **Missing Kinematics** | $\mathbf{a} = [1, 0], \mathbf{k}=0$ | $30.58\%$ ($-41.70\%$) | $46.29\%$ ($-26.05\%$) | $24.55\%$ ($-26.93\%$) | API disconnection / robotic fault |

### Core Empirical Discoveries
1. **Flawless Video Fallback:** Under complete optical loss ($a=[0, 1]$), the gate routes $100\%$ mass to kinematics, sustaining performance with negligible degradation ($-0.42\%$ on Suturing, $-0.27\%$ on Needle Passing).
2. **Kinematic Degradation Asymmetry:** When kinematics telemetry drops, performance declines severely ($-26\%$ to $-41\%$). This proves kinematics is the foundational substrate for atomic gesture recognition on JIGSAWS.
3. **The Concat Anomaly:** In Feature Concatenation on Suturing, zeroing out video features *increased* accuracy ($+17.83\%$), proving that naive concatenation allows 1024D visual noise to corrupt 76D kinematic signals. Adaptive gating completely prevents this interference.

---

## Slide 8: Class-Imbalance & Rare Gesture Empirical Audit

### 4-Arm Loss Ablation on Suturing (8 LOUO Folds)

| Loss Formulation | Accuracy (%) | Balanced Acc (%) | Macro-F1 (%) | Edit Score (%) | Holm $p_{\text{adj}}$ vs CE |
|---|:---:|:---:|:---:|:---:|:---:|
| **Standard CE** | $\mathbf{74.95 \pm 7.68}$ | $57.73 \pm 12.30$ | $44.29 \pm 6.38$ | $65.37 \pm 8.21$ | *Control* |
| **Class-Weighted CE** | $72.28 \pm 10.24$ | $\mathbf{60.05 \pm 12.23}$ | $\mathbf{44.64 \pm 8.83}$ | $\mathbf{68.95 \pm 10.03}$ | $p_{\text{adj}} > 0.14$ (Non-Sig) |
| **Focal Loss ($\gamma=2.0$)** | $73.67 \pm 9.28$ | $55.32 \pm 9.49$ | $42.19 \pm 7.28$ | $65.84 \pm 7.79$ | $p_{\text{adj}} > 0.32$ (Non-Sig) |
| **CB-Focal ($\gamma=2, \beta=.9999$)** | $71.41 \pm 13.23$ | $59.16 \pm 10.27$ | $42.54 \pm 9.36$ | $64.77 \pm 10.28$ | $p_{\text{adj}} > 0.32$ (Non-Sig) |

### The Rare-Gesture Reality (Audited from 32 Test Runs)
- **Gesture G10 (Pulling suture, Support = 91 frames, 1.1%):** **0.00% Recall across ALL 4 models.** Zero true positive frames predicted.
- **Gesture G9 (Suture orientation, Support = 140 frames, 2.5%):**
  - Standard CE & Focal: $0.00\%$ Recall.
  - Class-Weighted CE: $6.43\%$ Recall (9/140 frames predicted, but with 58 false positives, precision $13.43\%$, F1 $8.70\%$).
- **The True Source of the Balanced Accuracy Gain (+2.32%):**
  - Driven by medium-frequency gestures: G5 recall rose $+21.88\%$ ($33.3\% \rightarrow 55.2\%$) and G1 rose $+6.25\%$.
  - Offset by dominant class loss: G3 (making C-loop, 30.6% of data) recall dropped by **$-6.95\%$** ($85.08\% \rightarrow 78.13\%$), explaining the raw accuracy drop.
- **Conclusion:** Loss reweighting redistributes probability among medium classes, but **cannot fix severe representation starvation** on rare transition gestures.

---

## Slide 9: Limitations & Threats to Scientific Validity

### Transparent Disclosure of Research Constraints
1. **Effective Statistical Sample Size ($N=8$ / $N=7$):**
   - In Leave-One-User-Out cross-validation on 8 subjects, $N=8$ independent test points limits statistical power.
   - Knot Tying achieves family-wise significance ($p_{\text{adj}} \le 0.0457$), but Suturing ($p_{\text{adj}}=0.0820$) and Needle Passing are descriptive trends.
   - Needle Passing evaluates $N=7$ folds because Subject G has an empty test set in the official distribution.
2. **Timestamp Hardware Synchronization:**
   - JIGSAWS video (30 FPS) and kinematics (nominal 30 Hz) were recorded via software commands without hardware genlock.
   - Software recording shutoff created tail lag averaging $12.37$ frames (up to 470 frames).
   - Alignment is a discrete frame-center approximation within $\pm 5$ to $15$ frames ($\sim 167$ to $500$ ms). True microsecond synchronization requires hardware master clocks (e.g., SurgSync).
3. **Synthetic vs. In-Vivo Clinical Disturbances:**
   - Robustness evaluations used controlled Gaussian perturbations and discrete missingness masks.
   - While validating mathematical fallback guarantees, real intraoperative disturbances (blood pooling, lens condensation, robotic cable compliance) exhibit complex spatial-temporal dynamics.
4. **Subject Variance & Novice Generalization:**
   - Novice surgeons (User G) exhibit idiosyncratic kinematic styles that resist generalization regardless of loss function.

---

## Slide 10: Conclusions & Future Research Directions

### Core Research Takeaways
1. **Lightweight Decoupled Architecture is Effective:** Combining frozen surgical foundation features (SurgMotion) with compact GRU encoders delivers strong performance without large-scale GPU retraining.
2. **Kinematics is King on JIGSAWS:** Robot kinematics carries the primary signal for surgical gesture recognition; video serves as boundary stabilizer and contextual regularizer.
3. **Adaptive Gating Prevents Degradation:** Outperforms fixed late fusion and naive concatenation, guaranteeing exact mathematical single-modality fallback under sensor failure.
4. **Temporal Modeling Eliminates Boundary Chatter:** Boosts segmental Edit Score by $+22$ to $+59$ points ($p < 0.0005$).
5. **Class Imbalance Requires Representation Solutions:** Mathematical loss weighting cannot resurrect feature-starved classes like G10.

### Concrete Future Directions
- **Hardware-Synchronized Multimodal Data:** Migration to modern hardware-synchronized frameworks (SurgSync dVRK).
- **Tail-Gesture Representation Learning:** Contrastive few-shot augmentation specifically targeted at transitional gestures (G9, G10).
- **Online Noise-Calibrated Gating:** Integrating uncertainty estimation (Monte Carlo Dropout / Evidential Deep Learning) into dynamic gate weights.

---
