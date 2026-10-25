# Multi-Modality Benchmark Readiness Audit: Knot_Tying

- **Overall Benchmark Readiness:** `BLOCKED`
- **Total Trials:** 36
- **Cached Feature Archives:** 0/36 trials verified
- **Missing Feature Caches:** 36 trials
- **Active Visual Encoder:** ResNet18 (ResNet-18 (ImageNet Pretrained, 512-D))
- **SurgMotion Access Status:** Gated on Hugging Face ('gated: manual' pending institutional review)
- **Alignment Exceptions (> 500 frames delta):** 0 trials

## Checkpoint Availability Across Folds

| Model Family | Available Checkpoints | Missing Checkpoints |
|---|---|---|
| **kinematics_mlp** | 1/8 (fold_1) | 7/8 |
| **kinematics_gru** | 1/8 (fold_1) | 7/8 |
| **video_gru** | 0/8 (None) | 8/8 |
| **concat_fusion** | 0/8 (None) | 8/8 |

## Required Action to Unblock Full Real Benchmark

Real multimodal comparative benchmark cannot proceed until visual feature extraction is completed across all trials.

Execute the following sequential commands:

```bash
# 1. Extract visual features for all trials in Knot_Tying
python scripts/extract_visual_features.py --task Knot_Tying --encoder ResNet18

# 2. Train video-only temporal baseline across official folds
python scripts/train_video_baseline.py --task Knot_Tying

# 3. Execute fixed late-logit fusion evaluation
python scripts/run_late_fusion.py --task Knot_Tying

# 4. Train feature concatenation fusion baseline
python scripts/train_concat_fusion.py --task Knot_Tying

# 5. Generate final comparative benchmark report
python scripts/evaluate_comparative_benchmark.py --task Knot_Tying
```
