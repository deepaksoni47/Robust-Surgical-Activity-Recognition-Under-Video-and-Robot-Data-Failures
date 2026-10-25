# JIGSAWS Kinematics-Only MLP Baseline Benchmark

Architecture: 3-Layer MLP [76 -> 128 -> 64 -> C], GELU, LayerNorm, Dropout(0.2).
Evaluation: Official 8-fold Leave-One-User-Out cross-validation.

| Task | Accuracy (%) | Balanced Acc (%) | Macro-F1 | Edit Score (%) |
|---|---:|---:|---:|---:|
| **Needle_Passing** | 48.66 ± 11.70 | 39.45 ± 8.62 | 0.2862 ± 0.0940 | 8.66 ± 3.97 |
