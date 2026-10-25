# Comparative Evaluation of Multimodal and Unimodal Baselines: Suturing

- **Protocol:** Official JIGSAWS Leave-One-User-Out (LOUO) Cross-Validation
- **Folds Evaluated:** 8 (1, 2, 3, 4, 5, 6, 7, 8)
- **Strongest Unimodal Baseline:** Acc=66.84%, BalAcc=56.18%, Macro-F1=45.43%, Edit=34.39%

## Benchmark Summary Table

| Model | Modality | Accuracy (%) | Balanced Acc (%) | Macro-F1 (%) | Edit Score | $\Delta$ Acc | $\Delta$ Macro-F1 |
|---|---|---|---|---|---|---|---|
| **KinematicsMLP** | Kinematics | 66.84 ± 9.07 | 55.88 ± 12.16 | 45.43 ± 9.49 | 10.15 ± 2.22 | — | — |
| **KinematicsGRU** | Kinematics | 66.83 ± 10.93 | 56.18 ± 7.65 | 44.45 ± 8.98 | 34.39 ± 9.50 | — | — |

## Analysis & Hypotheses Verification

- **H1 (Video Baseline):** Frozen visual representation + temporal GRU baseline evaluated under matched LOUO splits.
- **H2 (Kinematics Baseline):** Lightweight temporal GRU / MLP on 76D robot kinematics.
- **H3 (Multimodal Fusion Superiority):** Evaluates whether fixed late-logit fusion and feature concatenation outperform the strongest unimodal baseline.