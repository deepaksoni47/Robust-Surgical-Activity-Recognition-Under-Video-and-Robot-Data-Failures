# JIGSAWS Kinematics-Only MLP Baseline Benchmark

Architecture: 3-Layer MLP [76 -> 128 -> 64 -> C], GELU, LayerNorm, Dropout(0.2).
Evaluation: Official 8-fold Leave-One-User-Out (LOUO) cross-validation.
Zero-leakage guarantee: KinematicsNormalizer and model weights fitted strictly on training trials.

| Task | Accuracy (%) | Balanced Acc (%) | Macro-F1 (%) | Edit Score (%) |
|---|---:|---:|---:|---:|
| **Suturing** | 66.84 ± 9.07 | 55.88 ± 12.16 | 45.43 ± 9.49 | 10.15 ± 2.22 |
| **Knot_Tying** | 61.50 ± 8.76 | 58.60 ± 9.00 | 53.20 ± 9.85 | 8.88 ± 2.80 |
| **Needle_Passing** | 44.16 ± 9.61 | 38.83 ± 7.74 | 27.69 ± 7.64 | 7.14 ± 2.22 |
