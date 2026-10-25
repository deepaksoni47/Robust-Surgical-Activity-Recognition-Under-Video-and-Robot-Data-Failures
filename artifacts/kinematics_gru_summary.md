# JIGSAWS Kinematics-Only GRU Baseline Benchmark

Architecture: 2-Layer GRU [76 -> 128 -> GRU(128) -> C], LayerNorm, Dropout(0.2).
Evaluation: Official 8-fold Leave-One-User-Out cross-validation.

| Task | Accuracy (%) | Balanced Acc (%) | Macro-F1 | Edit Score (%) |
|---|---:|---:|---:|---:|
| **Needle_Passing** | 48.69 ± 11.62 | 41.70 ± 9.97 | 0.2999 ± 0.0935 | 31.34 ± 7.32 |
