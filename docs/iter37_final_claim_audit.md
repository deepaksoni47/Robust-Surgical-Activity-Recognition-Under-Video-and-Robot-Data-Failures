# ITER-37: Final Scientific Integrity Audit and Main Research Claim Verification

**Date:** October 30, 2026  
**Status:** Completed & Empirically Verified  
**Scope:** Exhaustive claim verification, H1–H5 Claim-Evidence Matrix, multiple-comparison audit, per-class rare gesture analysis, and paper correction guidelines.  
**Audited Artifacts:**
- `artifacts/iter36_rare_gesture_audit.json` (Commit `77fe81a`, 32 prediction JSONs in `artifacts/predictions/`)
- `artifacts/iter36_integrated_results.json`
- `artifacts/iter35_imbalance_comparison_Suturing_summary.json`
- `artifacts/iter34_ablation_summary.json`
- `artifacts/iter30_numerical_validation.json`
- `artifacts/iter29_unified_comparison.json`
- `paper/main.tex` & `paper/main.pdf`

---

## 1. Executive Audit Summary

This audit establishes a rigorous, defensive standard of scientific integrity for our mini-project research paper and mentor presentation. Every empirical claim made in the manuscript, documentation, and presentation has been cross-checked against the raw data artifacts. 

### Key Audit Rectifications:
1. **Statistical Significance vs Descriptive Shifts in Loss Ablation:**
   - Under standard unweighted Cross-Entropy (`ce`), the model achieves **$74.95 \pm 7.68\%$** raw frame accuracy.
   - Inverse-frequency Class-Weighted CE (`class_weighted`) achieves **$72.28 \pm 10.24\%$** accuracy ($\Delta = -2.66\%$, unadjusted $p=0.0473$, but Holm-adjusted $p_{\text{adj}} = 0.1419$, **non-significant**).
   - Its balanced accuracy is **$60.05 \pm 12.23\%$** ($\Delta = +2.32\%$, $p_{\text{adj}} = 1.0000$, **non-significant**).
   - Its Edit Score is **$68.95 \pm 10.03\%$** ($\Delta = +3.58\%$, $p = 0.0878$, $p_{\text{adj}} = 0.2634$, **non-significant trend**).
   - *Verdict:* The paper and presentation must strictly characterize these as **numerical/descriptive trends**, never as "statistically significant improvements."
2. **Definitive Rare-Gesture Audit (G9 & G10):**
   - With all 32 prediction JSONs now exported and audited from the mentor GPU execution:
     - **Gesture G10 (Pulling suture, Support = 91 frames, 1.1% of Suturing):** Achieved **0.00% Recall and 0.00% F1 across ALL four loss formulations** (`ce`, `class_weighted`, `focal`, `cb_focal`).
     - **Gesture G9 (Suture orientation, Support = 140 frames, 2.5%):** Achieved **0.00% Recall** under CE and Focal, **6.43% Recall (8.70% F1)** under Class-Weighted CE, and **4.29% Recall (7.27% F1)** under CB-Focal.
     - *Verdict:* Loss re-weighting and focal loss did **not** resolve rare-gesture collapse for G10, and only marginally recovered 9 frames of G9 at the cost of poor precision (13.43%). The balanced accuracy lift (+2.32%) was driven by medium-frequency gestures (G5 recall jumped from 33.3% to 55.2%), not G9/G10.
3. **Subject Shifts (User D and User G):**
   - User D's performance drop under CB-Focal (-14.74% frame accuracy) and User G's persistent collapse (<48% balanced accuracy) are **post-hoc descriptive observations** of kinematic divergence, not experimentally isolated causal mechanisms. Language must reflect hypothesis generation, not proven causation.
4. **GRU Temporal Modeling:**
   - Temporal sequence modeling produces negligible changes in frame accuracy ($\Delta \le 1.26\%$, all $p > 0.40$), but drives an **enormous, universally significant improvement in segmental Edit Score** ($+22.68$ to $+26.25$ points for unimodal GRU, and $+34.26$ to $+59.17$ points for Multimodal Adaptive GRU; all $p < 0.0005, d_z > 2.2$).
   - We claim that recurrent sequence modeling enforces temporal hysteresis that resolves high-frequency frame chatter, but we do **not** claim it "proves transition-noise absorption" without dedicated noise-injection experiments at segment boundaries.

---

## 2. Main Research Question & Claim-Evidence Matrix (H1–H5)

> **Primary Research Question:**  
> On JIGSAWS surgical gesture recognition, does lightweight temporal fusion of pretrained video representations and robot kinematics outperform unimodal baselines, and does reliability-aware fusion reduce performance loss when one modality is degraded or unavailable?

| Hypothesis | Claimed Statement | Supporting Experiments | Empirical Evidence (Mean $\pm$ SD) | Statistical Limitations & Nuances | Verdict |
|---|---|---|---|---|:---:|
| **H1 (Video Baseline)** | Frozen visual features + lightweight recurrent head provide a viable unimodal baseline above chance. | VideoGRU benchmark on frozen ResNet-18 (512D) and SurgMotion (1024D) across all 103 trials. | SurgMotion VideoGRU: Suturing $39.17 \pm 13.51\%$ (BalAcc $34.10\%$, F1 $22.16\%$), Knot Tying $53.60 \pm 14.12\%$ (BalAcc $54.49\%$, F1 $48.74\%$), Needle Passing $26.55 \pm 11.23\%$ (BalAcc $30.97\%$, F1 $19.97\%$). Majority baselines: Suturing $33.43\%$ (BalAcc $10.30\%$), Knot Tying $27.35\%$ (BalAcc $14.93\%$), Needle Passing $40.55\%$ (BalAcc $11.25\%$). | **Critical Caveat:** On Needle Passing, raw accuracy ($26.55\%$) is below the trivial majority baseline ($40.55\%$) due to severe class skew towards dominant gestures G2 and G4. However, visual models achieve $3\times$ higher Balanced Accuracy ($30.97\%$ vs $11.25\%$) and $4\times$ higher Macro-F1 ($19.97\%$ vs $5.20\%$). Therefore, the visual baseline is viable for class-balanced discrimination above chance, but does NOT beat raw majority accuracy on every task. | **Supported (with Caveat)** |
| **H2 (Kinematics Baseline)** | Normalized 76D robot kinematics predict gestures substantially above majority-class trivial baseline. | Kinematics GRU benchmark across all 23 active LOUO folds. | Kinematics GRU: Suturing $66.64 \pm 9.01\%$, Knot Tying $61.09 \pm 8.27\%$, Needle Passing $48.96 \pm 8.01\%$. | Vastly outperforms majority class baseline ($p < 0.0001, d_z > 3.0$). Demonstrates kinematics is the primary driver of fine gesture state. | **Strongly Supported** |
| **H3 (Multimodal Fusion)** | Multimodal fusion outperforms the strongest unimodal baseline under identical LOUO splits. | Fixed Late-Logit Fusion, Feature Concatenation, and Adaptive Reliability Fusion across all 103 trials. | **Knot Tying:** Adaptive ($72.34\%$) vs Kinematics ($61.09\%$): $\Delta = +11.25\%$, $p=0.007$, FDR $q=0.021$.<br>**Suturing:** Adaptive ($72.28\%$) vs Kinematics ($66.64\%$): $\Delta = +5.64\%$, unadjusted $p=0.027$, Holm $p_{\text{adj}}=0.082$.<br>**Needle Passing:** Adaptive ($51.48\%$) vs Kinematics ($48.96\%$): $\Delta = +2.52\%$, $p=0.218$, $p_{\text{adj}}=0.218$. | Multimodal superiority is **statistically established on Knot Tying** ($p_{\text{adj}} < 0.05$). On Suturing, it is a **strong descriptive trend** ($p_{\text{adj}}=0.082$). On Needle Passing, it is a non-significant numerical trend ($p=0.218$). Simple fixed late fusion actually degraded Suturing accuracy ($60.54\%$), proving naive fusion is insufficient. | **Partially Supported** (Statistically on Knot Tying; Descriptive trend on Suturing/Needle Passing) |
| **H4 (Reliability-Aware Adaptive Fusion)** | Gating network with explicit availability signals maintains performance under missingness or corruptions and guarantees fallback. | Stress testing under optical corruption ($\sigma_v=1.0$), kinematics corruption ($\sigma_k=1.0$), video dropout ($a=[0, 1]$), and kinematics dropout ($a=[1, 0]$). | **Video Missing:** Suturing $71.86\%$ ($-0.42\%$ drop from $72.28\%$), Knot Tying $70.06\%$ ($-2.28\%$ drop from clean baseline $72.34\%$; source artifact `artifacts/surgmotion_fusion_adaptive_Knot_Tying_results.json`), Needle Passing $51.21\%$ ($-0.27\%$ drop from $51.48\%$). Concat fusion dropped $-2.9\%$ on Knot Tying.<br>**Single-Modality Fallback:** Exact mathematical identity $\alpha_{\text{other}} \equiv 0$ verified without numerical drift. | **Gate Behavior Nuance:** Gate acts primarily as an **availability router** ($\mathbf{a}_t \in \{0, 1\}^2$) and **global modality balancer** ($\bar{\alpha}_k \approx 80\%-95\%$), not an active continuous noise filter (weight shifts $<0.9\%$ across $\sigma_v \in [0, 2]$). Kinematics loss ($a=[1, 0]$) drops accuracy to $30.6\%-48.6\%$. | **Supported with Documented Nuance** |
| **H5 (Temporal Context)** | Recurrent sequence modeling (GRU) over windowed representations outperforms frame-independent predictions. | Paired comparison of Kinematics MLP vs GRU and Video MLP vs GRU across all 23 active LOUO folds. | **Frame Accuracy:** Shifts are negligible ($\Delta \le 1.26\%$, all $p > 0.40$).<br>**Segmental Edit Score:** Massive, universally significant boost: Kinematics GRU $+22.68$ to $+26.25$ points ($p < 0.0005, d_z > 2.2$). Multimodal GRU $+34.26$ to $+59.17$ points ($p < 0.0001, d_z > 3.0$). | H5 is **not supported for frame-level accuracy**, but is **overwhelmingly supported for segmental sequence coherence** (Edit Score). GRU suppresses high-frequency label chatter. | **Supported for Segmental Coherence** |

---

## 3. Audit of the ITER-35 / ITER-36 Class-Imbalance Experiment

### The 4-Arm Comparison on Suturing (8 LOUO Folds)

```text
========================================================================================================================
CONDITION SUMMARY TABLE (SUTURING, 8 FOLDS)
========================================================================================================================
Condition            | Frame Accuracy (%) | Balanced Acc (%)  | Macro-F1 (%)     | Segmental Edit Score (%)
------------------------------------------------------------------------------------------------------------------------
ce                   | 74.95 ± 7.68       | 57.73 ± 12.30     | 44.29 ± 6.38     | 65.37 ± 8.21
class_weighted       | 72.28 ± 10.24      | 60.05 ± 12.23     | 44.64 ± 8.83     | 68.95 ± 10.03
focal                | 73.67 ± 9.28       | 55.32 ± 9.49      | 42.19 ± 7.28     | 65.84 ± 7.79
cb_focal             | 71.41 ± 13.23      | 59.16 ± 10.27     | 42.54 ± 9.36     | 64.77 ± 10.28
========================================================================================================================
```

### Multiple Comparisons & Statistical Integrity
- **Comparison Family:** $m=3$ treatment arms (`class_weighted`, `focal`, `cb_focal`) compared against the standard unweighted control (`ce`).
- **Correction Method:** Step-down Holm-Bonferroni correction applied independently to each metric family:
  1. *Frame Accuracy:*
     - `class_weighted` vs `ce`: $\Delta = -2.66\%$, unadjusted $p=0.0473 \rightarrow p_{\text{adj}} = 0.1419$ (Not significant at $\alpha=0.05$).
     - `focal` vs `ce`: $\Delta = -1.27\%$, unadjusted $p=0.3285 \rightarrow p_{\text{adj}} = 0.3285$.
     - `cb_focal` vs `ce`: $\Delta = -3.54\%$, unadjusted $p=0.1640 \rightarrow p_{\text{adj}} = 0.3280$.
  2. *Balanced Accuracy:*
     - `class_weighted` vs `ce`: $\Delta = +2.32\%$, unadjusted $p=0.4773 \rightarrow p_{\text{adj}} = 1.0000$.
     - `cb_focal` vs `ce`: $\Delta = +1.43\%$, unadjusted $p=0.6365 \rightarrow p_{\text{adj}} = 1.0000$.
     - `focal` vs `ce`: $\Delta = -2.41\%$, unadjusted $p=0.5964 \rightarrow p_{\text{adj}} = 1.0000$.
  3. *Segmental Edit Score:*
     - `class_weighted` vs `ce`: $\Delta = +3.58\%$, unadjusted $p=0.0878 \rightarrow p_{\text{adj}} = 0.2634$ (Trend, not significant).

---

## 4. Empirical Per-Class Findings from the GPU Prediction Audit

The export of 32 trial-level prediction JSONs from the mentor GPU (`artifacts/predictions/`) reveals the true per-class behavior:

```text
========================================================================================================================
PER-CLASS METRIC AUDIT ACROSS ALL 8 SUTURING FOLDS
========================================================================================================================
Gesture Class      Support  | CE Recall (F1)     | CW Recall (F1)     | Focal Recall (F1)  | CB-Focal Recall (F1)
------------------------------------------------------------------------------------------------------------------------
G0  (Background)        59  | 67.80% (56.34%)    | 55.93% (43.71%)    | 72.88% (58.90%)    | 66.10% (32.23%)
G1  (Reach needle)      16  |  6.25% ( 7.14%)    | 12.50% ( 9.09%)    |  0.00% ( 0.00%)    |  6.25% ( 6.67%)
G2  (Pull needle)     1069  | 69.60% (67.24%)    | 71.56% (65.02%)    | 68.10% (67.91%)    | 66.04% (63.72%)
G3  (Orient needle)   2748  | 85.08% (84.94%)    | 78.13% (82.98%)    | 87.15% (85.98%)    | 80.46% (83.23%)
G4  (Catch needle)    1002  | 76.25% (77.13%)    | 75.75% (73.83%)    | 72.06% (73.49%)    | 74.05% (74.91%)
G5  (Push through)      87  | 33.33% (36.02%)    | 55.17% (47.76%)    | 45.98% (45.98%)    | 51.72% (43.90%)
G6  (Pull suture)     1481  | 83.25% (79.99%)    | 80.76% (76.77%)    | 83.32% (75.96%)    | 78.87% (78.52%)
G8  (Orient suture)    623  | 59.55% (64.52%)    | 51.36% (61.07%)    | 47.35% (57.39%)    | 57.14% (66.42%)
G9  (Assist needle)    140  |  0.00% ( 0.00%)    |  6.43% ( 8.70%)    |  0.00% ( 0.00%)    |  4.29% ( 7.27%)
G10 (Pull loose end)    91  |  0.00% ( 0.00%)    |  0.00% ( 0.00%)    |  0.00% ( 0.00%)    |  0.00% ( 0.00%)
G11 (Drop needle)      508  | 61.42% (55.22%)    | 65.16% (54.94%)    | 53.74% (49.59%)    | 57.48% (43.91%)
========================================================================================================================
```

### Critical Scientific Insights:
1. **G10 Remains Completely Unrecognized (0.00% F1):**
   - No loss formulation predicted a single true-positive frame for G10. The failure is not loss weighting; it is feature under-representation. G10 occurs during transient suture pull-through and lacks distinctive kinematic signatures from G6.
2. **G9 is Marginally Recovered at High Precision Cost:**
   - Class-weighted CE recovered 9 true frames of G9 (6.43% recall), but induced 58 false positives (13.43% precision). CB-Focal recovered 6 frames (4.29% recall, 24.00% precision).
3. **Where Did the Balanced Accuracy Lift Come From?**
   - Balanced accuracy increased from $57.73\%$ to $60.05\%$ because of **G5** (Recall rose from $33.33\%$ to $55.17\%$) and **G1** (Recall rose from $6.25\%$ to $12.50\%$), NOT from resolving G9/G10.
4. **Dominant Gesture Cost:**
   - To achieve this modest gain, G3 recall dropped by **$-6.95\%$** (from $85.08\%$ to $78.13\%$), explaining why overall frame accuracy dropped by $2.66\%$.

---

## 5. Paper Correction Directives for `paper/main.tex`

1. **Avoid Overstating Significance:**
   - In Section IV-D, replace any phrasing implying that class weighting "outperformed" or "significantly improved" the baseline with:
     *"Inverse-frequency class weighting yielded numerically higher balanced accuracy ($60.05\%$ vs. $57.73\%$) and segmental edit score ($68.95\%$ vs. $65.37\%$), though paired differences were not statistically significant after Holm-Bonferroni multiple-comparison correction ($p_{\text{adj}} > 0.14$)."*
2. **Incorporate Per-Class Reality:**
   - Disclose that G10 achieved $0.00\%$ recall across all four formulations and G9 achieved $\le 6.43\%$ recall, proving that loss re-weighting alone cannot resolve severe temporal feature collapse on rare transition gestures.
3. **Qualitative User Disclosures:**
   - Label User D and User G variance as observed surgeon-specific kinematic domain shifts rather than proven causal mechanisms.
4. **Disclose Needle Passing 7/8 Fold Status:**
   - Ensure the note in Table I clearly states that User G (Fold 6) was omitted from Needle Passing due to an empty test set in the official JIGSAWS split, resulting in $N=7$ active folds for that task.

---

## 6. Artifact Provenance Audit: Knot Tying Clean (72.34%) vs. Missing Video (70.06%)

A discrepancies audit was conducted between the ITER-32 report and the final benchmark tables:
- **Source Artifact:** `artifacts/surgmotion_fusion_adaptive_Knot_Tying_results.json` (lines 132–156) & `artifacts/iter29_robustness_comparison.json`.
- **Condition Analysis:**
  - **Clean Baseline ($\mathbf{a} = [1, 1]$):** **$72.34 \pm 9.70\%$** (Balanced Acc $66.73\%$, Macro-F1 $60.48\%$, Edit Score $64.66\%$).
  - **Missing Video ($\mathbf{a} = [0, 1], \mathbf{v}=0$):** **$70.06 \pm 9.73\%$** (Balanced Acc $64.21\%$, Macro-F1 $58.12\%$, Edit Score $62.34\%$).
  - **Degradation Drop:** $\Delta = -2.28\%$.
- **Resolution of Discrepancy:** The figure $72.34\%$ was the **clean multimodal baseline**; $70.06\%$ is the **missing video fallback condition**. In the separate ITER-32 continuous noise retraining experiments (`artifacts/iter32_results_summary.json`), models trained with noise schedules yielded $66.42\%$ clean / $64.61\%$ missing (unaugmented) and $70.07\%$ clean / $67.92\%$ missing (noise-augmented). The canonical benchmark evaluation retains $72.34\%$ for clean and $70.06\%$ for missing video. Both numbers are valid, consistent, and traceable to their respective JSON keys.

---

## 7. IEEE Transactions Compliance and Submission Criteria Audit

Merely compiling to an 8-page PDF does not establish compliance with IEEE Transactions submission rules. We audited the manuscript against IEEE Transactions on Medical Robotics and Bionics (T-MRB) guidelines:

1. **Document Class & Formatting:**
   - Standard IEEE Transactions journals use `\documentclass[10pt,journal]{IEEEtran}` (two-column, 10pt font, 12pt line spacing).
   - Removed `compsoc` option (which is reserved for IEEE Computer Society publications and produces distinct sans-serif headings and margins).
   - Fixed author line to omit titles (`Dr.`, `Prof.`) per IEEE standard practice, moving supervisor title to the author affiliations footnote (`\thanks{...}`).

2. **Page Budget & Overlength Charges:**
   - The compiled manuscript spans **exactly 8 pages** (`paper/main.pdf`, 291 KB, 8 pages).
   - IEEE T-MRB policy permits up to 8 pages for regular research papers without overlength page charges. Pages 9 and beyond incur mandatory overlength charges ($220 per excess page). Our manuscript satisfies the 8-page budget.

3. **Peer-Review Layout vs. Camera-Ready:**
   - Note for mentor submission: While camera-ready publication requires two-column format, some IEEE journal submission portals require or recommend a single-column, double-spaced draft layout (`\documentclass[11pt,draftcls,onecolumn]{IEEEtran}`). The LaTeX source is structured to cleanly switch between these modes by changing the documentclass arguments.

4. **Ethics, Data, & Reproducibility:**
   - JIGSAWS contains de-identified kinematic telemetry and surgical phantom videos recorded on dry-lab benchtop models, meeting standard institutional exemption from human subjects IRB review.
   - All source code, random seeds (`SEED=42`), fold split indices, and model checkpoints are version-controlled with automated deterministic test suites.

