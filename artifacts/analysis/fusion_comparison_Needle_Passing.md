# Comparative Evaluation of Multimodal and Unimodal Baselines: Needle_Passing

- **Protocol:** Official JIGSAWS Leave-One-User-Out (LOUO) Cross-Validation
- **Folds Evaluated:** 7 (1, 2, 3, 4, 5, 7, 8)
- **Strongest Unimodal Baseline:** Acc=48.69%, BalAcc=41.70%, Macro-F1=29.99%, Edit=31.34%

## Benchmark Summary Table

| Model | Modality | Accuracy (%) | Balanced Acc (%) | Macro-F1 (%) | Edit Score | $\Delta$ Acc | $\Delta$ Macro-F1 |
|---|---|---|---|---|---|---|---|
| **KinematicsMLP** | Kinematics | 48.66 ± 11.70 | 39.45 ± 8.62 | 28.62 ± 9.40 | 8.66 ± 3.97 | — | — |
| **KinematicsGRU** | Kinematics | 48.69 ± 11.62 | 41.70 ± 9.97 | 29.99 ± 9.35 | 31.34 ± 7.32 | — | — |
| **VideoGRU** | Video | 30.22 ± 6.57 | 32.07 ± 6.25 | 19.62 ± 3.27 | 22.86 ± 4.63 | — | — |
| **FixedLateFusion** | Multimodal (Video + Kinematics) | 49.68 ± 11.16 | 41.80 ± 10.60 | 29.71 ± 10.52 | 31.71 ± 7.44 | +0.99% | -0.28% |
| **ConcatFusionGRU** | Multimodal (Video + Kinematics) | 50.41 ± 8.23 | 45.16 ± 6.18 | 33.34 ± 4.15 | 26.60 ± 3.31 | +1.71% | +3.35% |
| **AdaptiveReliabilityFusion** | Multimodal (Video + Kinematics) | 51.14 ± 9.53 | 43.06 ± 7.50 | 31.48 ± 8.34 | 29.16 ± 5.83 | +2.45% | +1.50% |

## Analysis & Hypotheses Verification

- **H1 (Video Baseline):** Frozen visual representation + temporal GRU baseline evaluated under matched LOUO splits.
- **H2 (Kinematics Baseline):** Lightweight temporal GRU / MLP on 76D robot kinematics.
- **H3 (Multimodal Fusion Superiority):** Evaluates whether fixed late-logit fusion and feature concatenation outperform the strongest unimodal baseline.