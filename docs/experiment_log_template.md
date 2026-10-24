# Experiment log template

Copy this file for every experiment. Do not fill results before running the experiment.

## Identification
- Experiment ID:
- Date:
- Owner:
- Git commit:
- Config path / hash:
- Dataset version / archive hashes:
- Split protocol / split file:
- Random seed(s):

## Research question and hypothesis
- Question:
- Hypothesis:
- Primary comparison:
- What result would support the hypothesis?
- What result would fail to support it?

## Model and data
- Model variant:
- Video encoder and checkpoint:
- Encoder preprocessing:
- Kinematics features and normalization:
- Temporal window/clip length and stride:
- Fusion method:
- Reliability/availability inputs:
- Training/validation/test trials:
- Corruption condition and severity, if applicable:

## Training configuration
- Optimizer:
- Learning rate:
- Batch size:
- Epochs / early stopping:
- Loss:
- Class weighting:
- Hardware:
- Runtime:
- Peak memory:
- Feature-cache size:

## Results
- Accuracy:
- Macro-F1:
- Balanced accuracy:
- Per-class metrics:
- Clean-to-corrupted drop:
- Mean condition score:
- Worst condition score:
- Confusion matrix path:
- Prediction artifact path:
- Checkpoint path:

## Observations
- Convergence behavior:
- Class-specific errors:
- Modality gate weights:
- Alignment or data warnings:
- Differences from the planned protocol:

## Decision
- Supported / not supported / inconclusive:
- Limitations:
- Next action:
- Is a rerun needed? Why?
