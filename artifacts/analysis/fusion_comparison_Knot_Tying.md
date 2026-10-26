# Comparative Evaluation of Multimodal and Unimodal Baselines: Knot_Tying

- **Protocol:** Official JIGSAWS Leave-One-User-Out (LOUO) Cross-Validation
- **Folds Evaluated:** 8 (1, 2, 3, 4, 5, 6, 7, 8)
- **Strongest Unimodal Baseline:** Acc=62.77%, BalAcc=61.59%, Macro-F1=56.01%, Edit=35.48%

## Benchmark Summary Table

| Model | Modality | Accuracy (%) | Balanced Acc (%) | Macro-F1 (%) | Edit Score | $\Delta$ Acc | $\Delta$ Macro-F1 |
|---|---|---|---|---|---|---|---|
| **KinematicsMLP** | Kinematics | 62.77 ± 10.80 | 61.59 ± 9.37 | 56.01 ± 11.23 | 9.23 ± 3.39 | — | — |
| **KinematicsGRU** | Kinematics | 61.09 ± 11.79 | 56.53 ± 12.43 | 51.75 ± 15.14 | 35.48 ± 13.86 | — | — |
| **VideoGRU** | Video | 50.74 ± 5.20 | 50.02 ± 7.18 | 44.34 ± 7.75 | 24.89 ± 5.96 | — | — |
| **FixedLateFusion** | Multimodal (Video + Kinematics) | 64.16 ± 10.86 | 59.28 ± 12.88 | 54.45 ± 14.69 | 34.88 ± 12.37 | +1.39% | -1.55% |
| **ConcatFusionGRU** | Multimodal (Video + Kinematics) | 60.78 ± 6.98 | 57.63 ± 7.27 | 52.39 ± 7.28 | 33.53 ± 12.17 | -1.99% | -3.61% |
| **AdaptiveReliabilityFusion** | Multimodal (Video + Kinematics) | 67.05 ± 9.30 | 63.05 ± 8.33 | 58.77 ± 11.17 | 33.91 ± 11.37 | +4.28% | +2.77% |

## Analysis & Hypotheses Verification

- **H1 (Video Baseline):** Frozen visual representation + temporal GRU baseline evaluated under matched LOUO splits.
- **H2 (Kinematics Baseline):** Lightweight temporal GRU / MLP on 76D robot kinematics.
- **H3 (Multimodal Fusion Superiority):** Evaluates whether fixed late-logit fusion and feature concatenation outperform the strongest unimodal baseline.