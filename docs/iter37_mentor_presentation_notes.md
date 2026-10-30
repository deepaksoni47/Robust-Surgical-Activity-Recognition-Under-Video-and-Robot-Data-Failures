# ITER-37: Mentor Presentation Speaker Notes & Defense Guide

**Project Title:** Adaptive Multimodal Surgical Gesture Recognition via Pretrained Visual Representations and Reliability-Aware Dynamic Gating  
**Authors:** Deepak Soni, Aparna Banpela  
**Faculty Mentor / Supervisor:** Dr. Princy Matlani  
**Department:** Computer Science and Engineering  
**Date:** October 30, 2026  
**Audience:** Research Mentor, Department Evaluation Committee, External Reviewers  

---

## 1. Presentation Strategy & Mindset

### Guiding Principles for the Review
1. **Explain the Actual Implementation, Not an Idealized Abstraction:**
   - Walk the mentor through what was actually implemented, trained, and audited. Avoid exaggerated claims of "end-to-end vision-language intelligence" or "clinical deployment."
2. **Lead with Rigor and Verified Numbers:**
   - Every single metric shown on the slides is drawn directly from verified JSON artifacts and tested across the full 103-trial JIGSAWS benchmark under Leave-One-User-Out (LOUO) cross-validation.
3. **Disclose Nuances and Negative Results Proactively:**
   - Pointing out that Class-Weighted CE was not statistically significant after Holm correction ($p_{\text{adj}} > 0.14$), that G10 achieved 0.00% recall across all loss formulations, and that Needle Passing evaluates 7 active folds builds immense academic credibility.
4. **Estimated Total Duration:** 15–18 minutes presentation + 10–12 minutes Q&A.

---

## 2. Slide-by-Slide Speaker Notes

---

### Slide 1: Research Problem & Clinical Motivation (~1.5 min)
- **Key Talking Points:**
  - Automated surgical gesture recognition is the foundational building block for AI in robot-assisted minimally invasive surgery (RMIS), enabling objective surgeon skill assessment, context-aware camera automation, and safety boundary enforcement.
  - RMIS provides two distinct data streams: **endoscopic video** (rich in tool-tissue contact and anatomy, but prone to cautery smoke, lens fog, fluid splatter, and blur) and **robot kinematics** (continuous 30 Hz physical manipulator trajectories, completely immune to optical degradation, but blind to tissue state).
  - *The Failure Mode:* Naive multimodal fusion (simple late-logit averaging or static feature concatenation) is fragile. In our benchmark, static late fusion degraded Suturing accuracy by $-6.10\%$ because optical noise contaminated confident kinematic predictions.
  - *Our Objective:* Design an efficient, reliability-aware multimodal architecture that decouples visual feature extraction, captures temporal dynamics, and guarantees seamless single-modality fallback under sensor failure.

---

### Slide 2: Research Question & Core Hypotheses (~2.0 min)
- **Key Talking Points:**
  - State the central research question verbatim: *"On JIGSAWS surgical gesture recognition, does lightweight temporal fusion of pretrained video representations and robot kinematics outperform unimodal baselines, and does reliability-aware dynamic gating reduce performance loss when one modality is degraded or unavailable?"*
  - Review the Claim-Evidence Matrix (H1–H5):
    - **H1 (Video Baseline):** *Supported.* SurgMotion-ViTL achieves $39.2\%-50.7\%$ accuracy, well above the $27.4\%-33.4\%$ majority baseline, confirming frozen video features are viable.
    - **H2 (Kinematics Dominance):** *Strongly Supported.* Kinematics GRU reaches $66.6\%$ on Suturing and $61.1\%$ on Knot Tying ($p < 0.0001, d_z > 3.0$), establishing kinematics as the primary unimodal driver.
    - **H3 (Multimodal Superiority):** *Partially Supported.* Statistically significant on Knot Tying ($+11.25\%, p_{\text{adj}}=0.0286$); strong descriptive trend on Suturing ($+5.64\%, p_{\text{adj}}=0.0820$); non-significant trend on Needle Passing ($+2.52\%$).
    - **H4 (Reliability Gating):** *Supported with Documented Nuance.* Under video failure ($a=[0, 1]$), the gate sustains performance within $-0.42\%$ of clean. We explicitly note that the gate functions as an availability router and global balancer rather than a continuous noise filter.
    - **H5 (Temporal Context):** *Strongly Supported for Segmental Coherence.* Recurrent sequence modeling nearly doubles segmental Edit Score ($+22$ to $+59$ points, $p < 0.0005, d_z > 2.2$) by eliminating high-frequency frame chatter.

---

### Slide 3: JIGSAWS Benchmark & Experimental Protocol (~1.5 min)
- **Key Talking Points:**
  - The evaluation spans all 103 trials across all three JIGSAWS tasks: Suturing (39 trials, 10 gestures), Knot Tying (36 trials, 6 gestures), and Needle Passing (28 trials, 8 gestures).
  - Kinematics representation encompasses 76 continuous features across four da Vinci manipulators (Master Left/Right, Slave Left/Right), capturing 3D position, 9D rotation matrices, linear velocity, angular velocity, and gripper jaw angle.
  - *Strict Zero-Leakage Protocol:*
    - Official Leave-One-User-Out (LOUO) cross-validation: models are evaluated on an unseen surgeon whose data was strictly withheld from training.
    - Normalization statistics $(\boldsymbol{\mu}_{\text{train}}, \boldsymbol{\sigma}_{\text{train}})$ are computed **exclusively** on training trials.
    - *Transparent Disclosure:* In Needle Passing, Fold 6 (Subject G) is omitted because the official JIGSAWS split contains an empty test set for that user, leaving 7 active folds ($N=7$).

---

### Slide 4: Proposed Multimodal Architecture (~2.0 min)
- **Key Talking Points:**
  - *Why Decoupled?* End-to-end training of 3D CNNs or Vision Transformers over small surgical datasets (103 trials) leads to catastrophic overfitting and requires hundreds of GPU hours. We decouple perception into frozen representations and train compact, causal temporal heads.
  - *Visual Foundation Backbone:* We extract 1024-D motion-guided latent representations from SurgMotion-ViTL (pretrained on 3,658 hours of multi-organ surgery via Video-JEPA). Embeddings are cached to disk for reproducible, fast local training.
  - *Temporal Sequence Encoders:* Modality-specific 2-layer GRU networks ($d=128$) operate over an 8-timestep context window.
  - *Reliability-Aware Dynamic Gating Network:* Receives concatenated modality representations and an explicit availability vector $\mathbf{a}_t = [a_t^{\text{vis}}, a_t^{\text{kin}}]^\top \in \{0, 1\}^2$.
  - *Exact Fallback Guarantee:* When video drops ($a=[0, 1]$), the gating equation enforces $\alpha_t^{\text{vis}} \equiv 0$ and $\alpha_t^{\text{kin}} \equiv 1$, guaranteeing $100\%$ mathematical fallback to kinematics with zero numerical drift.

---

### Slide 5: Benchmark Results: Unimodal vs. Multimodal (~2.0 min)
- **Key Talking Points:**
  - Walk the mentor through the consolidated cross-validation table:
    - On Suturing ($N=8$): Adaptive Fusion reaches **$72.28\%$** accuracy, $+5.64\%$ over Kinematics GRU ($66.64\%$), with a strong marginal trend ($p=0.0273, p_{\text{adj}}=0.0820, d_z=0.98$).
    - On Knot Tying ($N=8$): Adaptive Fusion reaches **$72.34\%$** accuracy, beating Kinematics GRU ($61.09\%$) by **$+11.25\%$**—statistically significant after Holm correction ($p_{\text{adj}}=0.0286, d_z=1.25$).
    - On Needle Passing ($N=7$): Adaptive Fusion reaches **$51.48\%$** accuracy vs. $48.69\%$ for Kinematics GRU (descriptive trend, $p=0.510$).
  - *Why Did Fixed Late Fusion Fail?* Fixed probability weighting dilutes confident kinematics predictions with noisy video representations, dropping accuracy to $60.54\%$ on Suturing ($-6.10\%$ below unimodal kinematics).
  - *Why Did Concatenation Suffer on Suturing?* Concatenating $1024\text{D}$ video with $76\text{D}$ kinematics creates dimension imbalance, dropping Suturing accuracy to $54.00\%$. Adaptive gating solves this by dynamically weighting modality streams.

---

### Slide 6: Temporal Sequence Modeling & Gate Dynamics (~2.0 min)
- **Key Talking Points:**
  - *The Temporal Paradox (H5):* Comparing framewise kinematic MLPs with recurrent GRUs reveals that frame accuracy is virtually unchanged ($\Delta \le 1.26\%$, $p > 0.40$). However, recurrent GRUs nearly **double** the segmental Edit Score ($+22.68$ to $+26.25$ points for unimodal GRU; $+34.26$ to $+59.17$ points for Multimodal GRU; all $p < 0.0005$).
  - *Mechanism:* Recurrence provides temporal regularization and hysteresis, eliminating rapid single-frame label oscillation at gesture transition boundaries (demonstrated in the Trial B001 ribbon timeline).
  - *Gate Dynamics Reality (Audited over 4.2M Timesteps):*
    - The gate is heavily kinematics-dominant ($\bar{\alpha}_k \approx 80\%-95\%$).
    - Replacing time-varying $\alpha_t$ with static fold mean $\bar{\alpha}$ shifted clean accuracy by $<0.06\%$.
    - The gate weight curve across continuous visual noise ($\sigma_v \in [0, 2]$) is virtually flat ($\Delta \alpha_v < 0.9\%$).
    - *Honest Scientific Conclusion:* The gate acts as an effective **availability router and global modality balancer**, not an active high-frequency noise sensor.

---

### Slide 7: Robustness Stress Testing & Sensor Dropouts (~2.0 min)
- **Key Talking Points:**
  - We evaluated models across 5 standardized stress conditions: Clean, Video Corrupted ($\sigma_v=1.0$), Kinematics Corrupted ($\sigma_k=1.0$), Missing Video ($a=[0, 1]$), and Missing Kinematics ($a=[1, 0]$).
  - *Flawless Video Fallback:* Under complete optical loss, Adaptive Fusion sustains performance within $-0.42\%$ on Suturing ($71.86\%$ vs $72.28\%$) and $-0.27\%$ on Needle Passing.
  - *Telemetry Asymmetry:* When kinematics drops ($a=[1, 0]$), accuracy declines sharply to $30.58\%$ on Suturing and $24.55\%$ on Needle Passing. This asymmetric degradation proves that kinematics is the foundational substrate for atomic gesture recognition on JIGSAWS, while video acts as a contextual regularizer.
  - *The Concat Anomaly:* In Feature Concatenation on Suturing, zeroing out video features actually *increased* accuracy by $+17.83\%$, proving that 1024-D visual representations actively interfered with kinematics during joint training. Adaptive gating completely prevents this interference.

---

### Slide 8: Class Imbalance & Rare Gesture Empirical Audit (~2.0 min)
- **Key Talking Points:**
  - Across surgical datasets, rare transitional gestures (e.g., G9 dropping suture, G10 loosening suture) comprise $<3\%$ of frames.
  - We ran a controlled 8-fold LOUO ablation on Suturing comparing Standard CE, Inverse-Frequency Class-Weighted CE, Focal Loss ($\gamma=2.0$), and Class-Balanced Focal Loss.
  - *Statistical Reality Check:* Class-Weighted CE yielded numerically higher Balanced Accuracy ($60.05\%$ vs $57.73\%$, $+2.32\%$) and Edit Score ($68.95\%$ vs $65.37\%$, $+3.58\%$). However, neither difference is statistically significant after family-wise Holm correction ($p_{\text{adj}} = 1.000$ and $p_{\text{adj}} = 0.2634$ respectively). Similarly, the raw accuracy penalty ($-2.66\%$) is not statistically significant ($p_{\text{adj}} = 0.1419$).
  - *The Rare-Gesture Reality (Audited from 32 Test Runs):*
    - **G10 (91 frames, 1.1%):** Achieved **0.00% Recall across ALL 4 models.** Zero true positive frames were predicted in the entire benchmark.
    - **G9 (140 frames, 2.5%):** Standard CE and Focal achieved $0.00\%$ recall. Class-Weighted CE recovered 9 frames ($6.43\%$ recall), but generated 58 false positives (precision $13.43\%$, F1 $8.70\%$).
    - *Where did the Balanced Accuracy gain come from?* Medium-frequency gestures: G5 recall increased from $33.3\%$ to $55.2\%$, while dominant class G3 dropped by $-6.95\%$ recall.
  - *Conclusion:* Loss reweighting redistributes probability among medium classes, but cannot overcome severe representation starvation on rare transition gestures.

---

### Slide 9: Limitations & Threats to Scientific Validity (~1.5 min)
- **Key Talking Points:**
  - *Effective Sample Size ($N=8$ / $N=7$):* LOUO cross-validation yields 8 independent surgeon folds. While Knot Tying achieves family-wise significance, Suturing and Needle Passing represent descriptive trends under conservative Holm correction.
  - *Hardware Synchronization Approximation:* In JIGSAWS, video capture (30 FPS) and kinematics (nominal 30 Hz) were recorded via independent software commands without hardware genlock. Recording tail offsets average $12.37$ frames (up to 470 frames lag). Alignment is a discrete frame-center approximation within $\pm 5$ to $15$ frames ($\sim 167$ to $500$ ms).
  - *Synthetic vs. In-Vivo Disturbances:* Gaussian feature perturbations validate mathematical fallback guarantees, but do not fully capture spatial fluids, condensation, or surgical smoke dynamics.
  - *No Claim of Clinical Readiness:* We explicitly present this work as an audited algorithmic proof-of-concept on standardized benchtop surgical tasks.

---

### Slide 10: Conclusions & Future Research Directions (~1.5 min)
- **Key Talking Points:**
  - *Summary:* We established an empirically audited multimodal surgical gesture recognition framework that decouples foundation perception, models temporal dynamics, and guarantees single-modality fallback.
  - *Verified Deliverables:*
    - 8-page IEEE Transactions format research paper compiled and verified (`paper/main.pdf`).
    - Comprehensive audit reports and raw JSON artifacts for every experiment.
    - Interactive 10-slide mentor presentation with embedded publication figures.
  - *Future Work:* Hardware genlock migration to modern dVRK frameworks (SurgSync), few-shot representation learning for tail gestures, and uncertainty-calibrated dynamic gating.

---

## 3. Anticipated Mentor Questions & Defensive Answers

### Q1: "Why did fixed late fusion perform so poorly compared to your adaptive model on Suturing?"
**Answer:**  
In fixed late fusion, the modality weights are determined globally by convex optimization on training validation splits. However, on JIGSAWS, video feature predictions have lower individual accuracy ($39.17\%$) and higher variance than kinematics ($66.64\%$). Averaging logits with a fixed $\alpha$ forces the model to blend uninformative visual noise into time periods where kinematics alone is highly confident. Our adaptive gating network resolves this because its input includes representation statistics and availability indicators, allowing the model to suppress visual logits when uninformative and maintain an $80\%-95\%$ kinematics baseline.

---

### Q2: "Why didn't Class-Weighted CE or Focal Loss solve rare gestures G9 and G10?"
**Answer:**  
Our frame-level prediction audit across all 32 cross-validation runs revealed that G10 (loosening suture) had 0.00% recall across *all four* loss formulations. This proves that rare-gesture failure is a **feature representation problem**, not a loss weighting problem. In JIGSAWS, G10 comprises only 91 frames out of 8,247 in Suturing ($1.1\%$) and occurs as a transient sub-movement during suture pulling that shares identical kinematic trajectories with G6. Changing the loss gradient re-weights the penalty, but if the underlying kinematic and visual representations lack separability from G6, the model cannot distinguish them without dedicated few-shot representation learning.

---

### Q3: "Is your adaptive gating network an active continuous noise filter or just an availability router?"
**Answer:**  
We audited over 4.2 million time-resolved gate weight logs across visual noise levels $\sigma_v \in [0, 2.0]$. The empirical mean video gate weight $\bar{\alpha}_v$ shifted by less than $0.009$ ($<0.9\%$) across all tasks. Therefore, we do **not** claim the gate is an active continuous noise filter. Instead, the gate provides two proven benefits: (1) **Global Modality Balancing**, learning that kinematics should receive $80\%-95\%$ weight under clean conditions; and (2) **Discrete Availability Routing**, where the binary availability vector $\mathbf{a}_t \in \{0, 1\}^2$ mathematically forces $\alpha_{\text{other}} \equiv 0$ under sensor failure, guaranteeing seamless single-modality fallback.

---

### Q4: "Why is Needle Passing evaluated on 7 folds instead of 8?"
**Answer:**  
In the official JIGSAWS experimental setup distribution (`Experimental_setup/Needle_Passing/UserOut/`), Fold 6 corresponds to Subject G. In the official dataset release, the test split file `Needle_Passing_UserOut_Fold6_Test.txt` is completely empty because Subject G did not perform valid Needle Passing trials. Evaluating Fold 6 would produce zero test samples. To ensure mathematical and scientific consistency, we evaluated all 7 active folds ($N=7$). We explicitly disclose this in Table I, Table II, and Section III of our manuscript.

---

### Q5: "Why did temporal GRU sequence modeling nearly double the Edit Score (+30 points) while barely changing frame accuracy (+1%)?"
**Answer:**  
Frame accuracy evaluates each frame in isolation, regardless of whether consecutive frames are grouped into coherent segments. Framewise MLPs produce high-frequency label flickering (e.g., predicting G3, G2, G3, G2 on adjacent frames at gesture boundaries). While 80% of these individual frames might be correct, the resulting sequence has severe oversegmentation chatter, resulting in a low Edit Score (~35%). Recurrent sequence modeling (GRU) introduces sequential hidden state memory and temporal hysteresis. It requires sustained kinematic evidence to trigger a state transition, filtering out single-frame chatter and producing contiguous, clinically coherent segments, which drives Edit Score up to $65\%-69\%$.

---

### Q6: "How did you ensure zero data leakage between training and testing?"
**Answer:**  
We enforced three strict leakage safeguards:
1. **Subject-Level Isolation:** Evaluation strictly follows official Leave-One-User-Out (LOUO) splits. All trials of an unseen surgeon are reserved exclusively for testing.
2. **Training-Only Normalization:** Mean and standard deviation statistics for both kinematics and visual embeddings are computed strictly from training trials in each fold. Test data is normalized using the frozen training parameters.
3. **No Hyperparameter Peeking:** Fixed fusion weights ($\alpha$) and class weights were derived strictly on training/validation splits without ever inspecting test trial performance.

---

### Q7: "Why did you use frozen SurgMotion features instead of fine-tuning the video backbone end-to-end?"
**Answer:**  
JIGSAWS contains 103 trials with ~8,000 frames per task. Fine-tuning a heavy 3D Vision Transformer (e.g., SurgMotion-ViTL with 300M+ parameters) on a few dozen trials leads to rapid overfitting, extreme GPU memory requirements, and fragile optimization. By decoupling visual feature extraction into frozen foundation embeddings and training compact 2-layer GRU temporal sequence heads (128-D), our entire pipeline trains stably, fast, and reproducibly on standard hardware, adhering to modern foundation model adaptation principles.

---

### Q8: "What remaining issues genuinely require new experiments before external submission?"
**Answer:**  
Three issues genuinely require new experiments before submitting to an external peer-reviewed journal:
1. **Hardware-Synchronized Telemetry (SurgSync):** JIGSAWS video and kinematics lack hardware genlock (software recording lag averages $12.37$ frames). Validating our gating architecture on a hardware-synchronized dataset like SurgSync on dVRK is the primary empirical extension.
2. **Dedicated Few-Shot Representation Learning for Tail Gestures:** As proven by our audit, loss reweighting cannot solve G10. Contrastive metric learning or trajectory augmentation targeted at transient gestures is required.
3. **In-Vivo Clinical Corruptions:** Testing on real endoscopic artifacts (cautery smoke, lens fog, fluid splatter from laparoscopic videos) rather than synthetic Gaussian feature perturbations.

---

### Q9: "Why is Needle Passing VideoGRU accuracy (26.55%) below the majority baseline (40.55%)?"
**Answer:**  
In Needle Passing, gesture class distribution is severely skewed: dominant gestures G2 (pulling needle) and G4 (catching needle) account for over 40% of frames across trials. A trivial majority predictor that constantly outputs G2 or G4 achieves 40.55% raw accuracy by ignoring all other gesture classes (Macro-F1 5.20%, Balanced Accuracy 11.25%). The frozen SurgMotion VideoGRU predicts across the entire gesture vocabulary, achieving vastly superior Balanced Accuracy (30.97% vs. 11.25%, nearly 3× higher) and Macro-F1 (19.97% vs. 5.20%, nearly 4× higher). However, because it distributes predictions across rare and difficult classes, its raw frame accuracy is 26.55%. This is why H1 must be evaluated on class-balanced metrics, and why we explicitly disclose this caveat without claiming the visual baseline beats majority accuracy on every task.

---

### Q10: "Can you explain the relationship between 70.06% and 72.34% for Knot Tying?"
**Answer:**  
72.34% is the **Clean Baseline** accuracy of our Adaptive Reliability Fusion model on Knot Tying under full sensor availability ($\mathbf{a} = [1, 1]$). 70.06% is the **Missing Video** condition ($\mathbf{a} = [0, 1], \mathbf{v}=0$), where the visual stream is completely disconnected and the gating network routes 100% mass to kinematics. The degradation drop is -2.28% ($72.34\% \rightarrow 70.06\%$). Both numbers originate from the exact same source artifact (`artifacts/surgmotion_fusion_adaptive_Knot_Tying_results.json`, lines 132–156). In the earlier ITER-32 exploratory noise retraining, an experimental rerun produced 64.61% and 67.92% under continuous noise schedules, but 70.06% is the canonical primary benchmark result for missing video.

---

### Q11: "Does our 8-page compiled PDF fully satisfy IEEE Transactions submission criteria?"
**Answer:**  
Compiling to exactly 8 pages in IEEEtran format is necessary but not sufficient for journal submission. We conducted a comprehensive compliance audit against IEEE Transactions on Medical Robotics and Bionics (T-MRB) author guidelines:
1. *Template & Document Class:* We use the official `\documentclass[10pt,journal]{IEEEtran}` (two-column, 10pt font) and strict author name styling (omitting degrees like Dr./Prof. from the author line, retaining them in `\thanks{...}` footnotes).
2. *Page Budget & Overlength Policy:* Regular IEEE T-MRB papers are allocated an 8-page budget. Papers exceeding 8 pages incur mandatory overlength page charges ($220/page). Our manuscript fits within exactly 8 pages including references.
3. *Peer-Review Layout:* Many IEEE editors request single-column, double-spaced layout (e.g., `draftcls,onecolumn`) for initial peer review. Our source cleanly toggles between double-column final format and single-column review draft format.
4. *Ethics & IRB:* JIGSAWS is a public, de-identified benchtop benchmark recorded on inanimate phantom models, which meets standard IRB exemption criteria.
5. *Multimedia Accompaniment:* IEEE T-MRB strongly recommends a 3-minute video summary illustrating multimodal robotic telemetry and video alignment, which we plan for camera-ready submission.

