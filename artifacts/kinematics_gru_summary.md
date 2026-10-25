# JIGSAWS Kinematics-Only GRU Baseline Benchmark

Architecture: 2-Layer GRU [76 -> 128 -> GRU(128) -> C], LayerNorm, Dropout(0.2).
Evaluation: Official 8-fold Leave-One-User-Out (LOUO) cross-validation.
Zero-leakage guarantee: KinematicsNormalizer and model weights fitted strictly on training trials.

| Task | Accuracy (%) | Balanced Acc (%) | Macro-F1 (%) | Edit Score (%) |
|---|---:|---:|---:|---:|
| **Suturing** | 66.83 ± 10.93 | 56.18 ± 7.65 | 44.45 ± 8.98 | 34.39 ± 9.50 |
| **Knot_Tying** | 61.56 ± 12.81 | 56.98 ± 13.67 | 53.97 ± 16.49 | 36.62 ± 12.44 |
| **Needle_Passing** | 48.77 ± 10.17 | 43.01 ± 10.13 | 30.32 ± 8.81 | 31.05 ± 4.54 |
