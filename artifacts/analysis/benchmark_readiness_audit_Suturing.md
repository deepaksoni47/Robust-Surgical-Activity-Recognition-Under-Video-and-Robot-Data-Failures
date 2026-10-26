# Multi-Modality Benchmark Readiness Audit: Suturing

- **Overall Benchmark Readiness:** `READY`
- **Total Trials:** 39
- **Cached Feature Archives:** 39/39 trials verified
- **Missing Feature Caches:** 0 trials
- **Active Visual Encoder:** ResNet18 (ResNet-18 (ImageNet Pretrained, 512-D))
- **SurgMotion Access Status:** SurgMotion is gated on Hugging Face ('gated: manual' pending institutional review)
- **Alignment Exceptions (> 500 frames delta):** 0 trials

## Checkpoint Availability Across Folds

| Model Family | Available Checkpoints | Missing Checkpoints |
|---|---|---|
| **kinematics_mlp** | 8/8 (fold_1, fold_2, fold_3, fold_4, fold_5, fold_6, fold_7, fold_8) | 0/8 |
| **kinematics_gru** | 8/8 (fold_1, fold_2, fold_3, fold_4, fold_5, fold_6, fold_7, fold_8) | 0/8 |
| **video_gru** | 8/8 (fold_1, fold_2, fold_3, fold_4, fold_5, fold_6, fold_7, fold_8) | 0/8 |
| **concat_fusion** | 8/8 (fold_1, fold_2, fold_3, fold_4, fold_5, fold_6, fold_7, fold_8) | 0/8 |

## Required Action to Unblock Full Real Benchmark

All prerequisites are satisfied. Full real benchmark can be executed.
