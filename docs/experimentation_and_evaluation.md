# Experimentation and evaluation protocol

## Evaluation principles
- The project is a planned study; no experiments have been run.
- Use the same split, label mapping, and primary metrics for all model comparisons.
- Report exact split protocol, seeds, encoder/checkpoint, preprocessing, hyperparameters, and compute.
- Do not tune on test performance.
- Treat synthetic corruptions as controlled stress tests, not as evidence about real-world failure frequency.

## Splits
Use a published JIGSAWS protocol that can be reconstructed from the available metadata. Keep all windows from a trial in one split. If a protocol tests generalization to unseen users/surgeons, use it as specified. A validation subset is needed for model selection; if the benchmark protocol does not provide one, carve it from training trials without touching the official test set and document the deviation.

## Metrics
Primary:
- **Macro-F1:** gives each gesture class equal weight.
- **Balanced accuracy:** average recall across classes.
Secondary:
- Accuracy.
- Per-class precision/recall/F1 and support.
- Confusion matrix.
- Clean-to-corrupted performance drop.
- Mean score across corruption conditions and worst-condition score.
- Runtime, peak GPU memory, feature-cache size.

For temporally segmented predictions, specify whether metrics are per frame, per timestep, or per segment. If smoothing or segment-level postprocessing is used, report it separately.

## Planned experiments

| ID | Question / hypothesis | Configuration | Comparison | Split | Metrics | Artifacts | Interpretation |
|---|---|---|---|---|---|---|---|
| E0 | Is the data pipeline trustworthy? | Audit all task files, labels, and alignment | Raw counts and parsed counts | All trials, no training | Integrity counts; invalid-record counts | Audit report, manifest | Any unexplained mismatch blocks modeling |
| E1 | What is the trivial reference score? | Majority-class predictor | Other models | Same official split | Macro-F1, balanced accuracy, accuracy | Prediction CSV and metrics | High accuracy with low macro-F1 signals imbalance |
| E2 | Can kinematics alone predict gestures? | Small MLP; later optional GRU/TCN | Majority baseline | Same split | Primary + per-class metrics | Config/checkpoint/predictions | Establishes information in motion stream |
| E3 | Can pretrained visual features predict gestures? | Frozen encoder + small classifier | Majority and kinematics | Same split | Primary + per-class metrics | Encoder metadata, cache, metrics | Tests visual representation without expensive fine-tuning |
| E4 | Does temporal context help? | Independent-timestep vs GRU/TCN over same features | Same modality and split | Same split | Primary metrics, runtime | Comparison table | Gain supports temporal modeling; no gain suggests simpler head may suffice |
| E5 | Does fixed multimodal fusion help? | Late-logit fusion; optional feature concatenation | Best unimodal model | Same split | Primary metrics | Fusion config and predictions | Improvement supports complementarity; regression triggers alignment/calibration review |
| E6 | Does adaptive fusion help? | Small gate with availability/reliability flags | Fixed fusion and unimodal fallbacks | Same split | Primary + robustness metrics | Gate weights, metrics, predictions | Improvement only on one setting must be described precisely |
| E7 | What happens under video corruption? | Predefined blur/noise/occlusion recipe, test-time only | Clean and fixed/adaptive fusion | Test trials only | Macro-F1, balanced accuracy, drop | Corruption config and outputs | Supports robustness only to the chosen perturbations |
| E8 | What happens under kinematics corruption/missingness? | Noise, dropout windows, modality absent | Clean and fixed/adaptive fusion | Test trials only | Same metrics + worst condition | Condition summary | Shows sensitivity to specified failures |
| E9 | Does adaptive fusion help when one modality is absent? | Video-only, kinematics-only, fixed fusion with masking, adaptive gate | Matching clean and missing conditions | Test trials only | Primary metrics, degradation | Missing-modality table | Gate should fall back to available modality; failure indicates gate design issue |
| E10 | Are results stable? | Repeat selected seeds or folds within budget | Main model variants | Same frozen protocol | Mean ± SD where justified | Aggregated results | Large variability limits strength of claims |
| E11 | What errors remain? | Per-class/confusion/failure case analysis | All main models | Test only after selection frozen | Per-class metrics and examples | Error analysis | Identify short/rare gestures and modality-specific failures |

## Minimum experiment set
E0–E6 are mandatory. E7–E9 are required if the adaptive reliability claim remains central and time permits; if not, narrow the final claim rather than imply robustness was evaluated. E10–E11 are strongly recommended.

## Corruption protocol
- Predefine corruption types, levels, random seeds, and target modality.
- Apply test-time corruption after the clean model is trained, unless explicitly testing corruption augmentation as a separate training intervention.
- Keep labels and split membership unchanged.
- Report clean results alongside every corrupted condition.
- Include modality-absent cases and explicit fallback behavior.
- Do not train on the final corrupted test set.
- Avoid overly strong corruption that changes the semantic label; inspect examples.

## Ablations
At minimum compare:
1. Video-only.
2. Kinematics-only.
3. Fixed fusion.
4. Adaptive fusion.
5. Adaptive fusion without reliability/availability indicators (if feasible).
6. Independent timestep vs temporal model.
7. Missing video vs missing kinematics.
8. One selected corruption level versus clean.

## Statistical caution
JIGSAWS is small relative to modern deep-learning datasets. Report trial/user-aware splits and uncertainty across folds/seeds where the protocol supports it. Avoid claiming clinical generalization or statistical certainty from one split. Do not treat thousands of adjacent frames as independent subjects.

## Result interpretation
- Fusion improves clean metrics and robustness: supports the hypotheses under this dataset/protocol.
- Fusion improves clean metrics but worsens missing-modality performance: complementarity exists but reliability handling is inadequate.
- Adaptive gate improves only degraded conditions: describe this as condition-specific robustness, not universal superiority.
- No fusion gain: a valid negative result; investigate alignment, scale calibration, data size, and model complexity before concluding modalities are unhelpful.
- All models perform poorly: first audit labels, split implementation, class imbalance, and synchronization.
