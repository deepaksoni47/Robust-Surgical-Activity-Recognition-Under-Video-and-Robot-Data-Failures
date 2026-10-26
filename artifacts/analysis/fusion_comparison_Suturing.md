# Comparative Evaluation of Multimodal and Unimodal Baselines: Suturing

- **Protocol:** Official JIGSAWS Leave-One-User-Out (LOUO) Cross-Validation
- **Folds Evaluated:** 8 (1, 2, 3, 4, 5, 6, 7, 8)
- **Strongest Unimodal Baseline:** Acc=66.64%, BalAcc=57.77%, Macro-F1=45.09%, Edit=34.69%

## Benchmark Summary Table

| Model | Modality | Accuracy (%) | Balanced Acc (%) | Macro-F1 (%) | Edit Score | $\Delta$ Acc | $\Delta$ Macro-F1 |
|---|---|---|---|---|---|---|---|
| **KinematicsMLP** | Kinematics | 65.39 ± 11.06 | 55.42 ± 12.81 | 44.37 ± 9.35 | 9.78 ± 2.31 | — | — |
| **KinematicsGRU** | Kinematics | 66.64 ± 9.01 | 57.77 ± 9.40 | 45.09 ± 8.24 | 34.69 ± 9.43 | — | — |
| **VideoGRU** | Video | 45.42 ± 10.96 | 42.64 ± 9.03 | 33.22 ± 7.50 | 20.01 ± 3.88 | — | — |
| **FixedLateFusion** | Multimodal (Video + Kinematics) | 69.39 ± 8.52 | 60.80 ± 10.28 | 48.76 ± 9.86 | 35.87 ± 8.30 | +2.75% | +3.67% |
| **ConcatFusionGRU** | Multimodal (Video + Kinematics) | 66.90 ± 6.39 | 57.91 ± 7.73 | 46.92 ± 6.07 | 31.64 ± 7.02 | +0.26% | +1.83% |
| **AdaptiveReliabilityFusion** | Multimodal (Video + Kinematics) | 73.54 ± 8.08 | 63.20 ± 8.76 | 52.62 ± 8.56 | 37.34 ± 8.00 | +6.90% | +7.53% |

## Analysis & Hypotheses Verification

- **H1 (Video Baseline):** Frozen visual representation + temporal GRU baseline evaluated under matched LOUO splits.
- **H2 (Kinematics Baseline):** Lightweight temporal GRU / MLP on 76D robot kinematics.
- **H3 (Multimodal Fusion Superiority):** Evaluates whether fixed late-logit fusion and feature concatenation outperform the strongest unimodal baseline.