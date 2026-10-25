# Comparative Evaluation of Multimodal and Unimodal Baselines: Knot_Tying

- **Protocol:** Official JIGSAWS Leave-One-User-Out (LOUO) Cross-Validation
- **Folds Evaluated:** 8 (1, 2, 3, 4, 5, 6, 7, 8)
- **Strongest Unimodal Baseline:** Acc=61.56%, BalAcc=58.60%, Macro-F1=53.97%, Edit=36.62%

## Benchmark Summary Table

| Model | Modality | Accuracy (%) | Balanced Acc (%) | Macro-F1 (%) | Edit Score | $\Delta$ Acc | $\Delta$ Macro-F1 |
|---|---|---|---|---|---|---|---|
| **KinematicsMLP** | Kinematics | 61.50 ± 8.76 | 58.60 ± 9.00 | 53.20 ± 9.85 | 8.88 ± 2.80 | — | — |
| **KinematicsGRU** | Kinematics | 61.56 ± 12.81 | 56.98 ± 13.67 | 53.97 ± 16.49 | 36.62 ± 12.44 | — | — |

## Analysis & Hypotheses Verification

- **H1 (Video Baseline):** Frozen visual representation + temporal GRU baseline evaluated under matched LOUO splits.
- **H2 (Kinematics Baseline):** Lightweight temporal GRU / MLP on 76D robot kinematics.
- **H3 (Multimodal Fusion Superiority):** Evaluates whether fixed late-logit fusion and feature concatenation outperform the strongest unimodal baseline.