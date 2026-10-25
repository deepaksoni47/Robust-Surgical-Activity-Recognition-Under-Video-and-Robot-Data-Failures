# Comparative Evaluation of Multimodal and Unimodal Baselines: Needle_Passing

- **Protocol:** Official JIGSAWS Leave-One-User-Out (LOUO) Cross-Validation
- **Folds Evaluated:** 7 (1, 2, 3, 4, 5, 7, 8)
- **Strongest Unimodal Baseline:** Acc=48.77%, BalAcc=43.01%, Macro-F1=30.32%, Edit=31.05%

## Benchmark Summary Table

| Model | Modality | Accuracy (%) | Balanced Acc (%) | Macro-F1 (%) | Edit Score | $\Delta$ Acc | $\Delta$ Macro-F1 |
|---|---|---|---|---|---|---|---|
| **KinematicsMLP** | Kinematics | 44.16 ± 9.61 | 38.83 ± 7.74 | 27.69 ± 7.64 | 7.14 ± 2.22 | — | — |
| **KinematicsGRU** | Kinematics | 48.77 ± 10.17 | 43.01 ± 10.13 | 30.32 ± 8.81 | 31.05 ± 4.54 | — | — |

## Analysis & Hypotheses Verification

- **H1 (Video Baseline):** Frozen visual representation + temporal GRU baseline evaluated under matched LOUO splits.
- **H2 (Kinematics Baseline):** Lightweight temporal GRU / MLP on 76D robot kinematics.
- **H3 (Multimodal Fusion Superiority):** Evaluates whether fixed late-logit fusion and feature concatenation outperform the strongest unimodal baseline.