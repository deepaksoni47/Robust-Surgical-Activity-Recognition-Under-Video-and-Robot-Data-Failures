# Multi-Modality Benchmark Readiness Audit: Needle_Passing

- **Overall Benchmark Readiness:** `READY`
- **Total Trials:** 27
- **Cached Feature Archives:** 27/27 trials verified
- **Missing Feature Caches:** 0 trials
- **Active Visual Encoder:** ResNet18 (ResNet-18 (ImageNet Pretrained, 512-D))
- **SurgMotion Access Status:** SurgMotion is gated on Hugging Face ('gated: manual' pending institutional review)
- **Alignment Exceptions (> 500 frames delta):** 0 trials

## Checkpoint Availability Across Folds

| Model Family | Available Checkpoints | Missing Checkpoints |
|---|---|---|
| **kinematics_mlp** | 7/8 (fold_1, fold_2, fold_3, fold_4, fold_5, fold_7, fold_8) | 1/8 |
| **kinematics_gru** | 7/8 (fold_1, fold_2, fold_3, fold_4, fold_5, fold_7, fold_8) | 1/8 |
| **video_gru** | 7/8 (fold_1, fold_2, fold_3, fold_4, fold_5, fold_7, fold_8) | 1/8 |
| **concat_fusion** | 7/8 (fold_1, fold_2, fold_3, fold_4, fold_5, fold_7, fold_8) | 1/8 |

## Required Action to Unblock Full Real Benchmark

All prerequisites are satisfied. Full real benchmark can be executed.
