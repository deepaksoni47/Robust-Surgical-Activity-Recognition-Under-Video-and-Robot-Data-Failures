# Adaptive Multimodal Surgical Gesture Recognition

## Project status
**Planning/documentation only. Model implementation has not started.**

This repository blueprint proposes a reproducible mini-project to test whether frozen pretrained video features, robot kinematics, and lightweight temporal modeling improve surgical gesture recognition, and whether reliability-aware fusion helps when one modality is degraded or missing.

## Research objective
Primary question:

> On JIGSAWS surgical gesture recognition, does lightweight temporal fusion of pretrained video representations and robot kinematics outperform video-only and kinematics-only baselines, and does reliability-aware fusion reduce performance loss when a modality is degraded or unavailable?

JIGSAWS supports video, robot kinematics, and gesture annotations. Its standard labels should **not** be described as general surgical phase labels. Phase recognition is an optional extension requiring a separately verified dataset with compatible labels.

## Scope
Minimum viable project (MVP):
1. Verify the downloaded JIGSAWS files and annotation conventions.
2. Establish leakage-safe, trial/surgeon-aware splits consistent with the selected protocol.
3. Build simple kinematics-only and video-only baselines.
4. Extract frozen features from a pretrained video encoder if its weights and inference interface can be verified; otherwise use a documented lighter fallback.
5. Compare late fusion against a small reliability-aware gate.
6. Test controlled missing/noisy-modality conditions and report macro-F1, balanced accuracy, and robustness drops.

Not in the MVP: training a foundation model from scratch, reproducing GRAD in full, clinical deployment, robot control, or claiming real-world failure prevalence from synthetic corruptions.

## Prerequisites
- Python 3.10 or 3.11 is a reasonable initial target; confirm compatibility against chosen model dependencies before locking versions.
- Git.
- A machine with enough storage for the dataset and cached features.
- A CPU path for data inspection and small tests; a free/remote GPU is preferred for video feature extraction and training.
- Permission to use the dataset and pretrained weights under their applicable terms.

## Setup
Detailed, staged setup is in [`docs/environment_and_setup.md`](docs/environment_and_setup.md). Do not install a large collection of model dependencies before the encoder choice is verified.

## Documentation map
- [Research synthesis](docs/research_synthesis.md)
- [Problem statement](docs/problem_statement.md)
- [Feasibility and dataset analysis](docs/feasibility_and_dataset_analysis.md)
- [System architecture](docs/system_architecture.md)
- [Implementation plan](docs/implementation_plan.md)
- [Implementation tracker](docs/implementation_tracker.md)
- [Data pipeline](docs/data_pipeline.md)
- [Experimentation and evaluation](docs/experimentation_and_evaluation.md)
- [Environment and setup](docs/environment_and_setup.md)
- [Testing and validation](docs/testing_and_validation.md)
- [Risks and open questions](docs/risks_and_open_questions.md)
- [Experiment log template](docs/experiment_log_template.md)
- [Documentation audit](docs/documentation_audit.md)

## Current decision summary
- **Primary dataset:** JIGSAWS, already downloaded by the team, subject to local integrity and access-term checks.
- **Primary task:** frame/short-window surgical gesture recognition.
- **Core modalities:** video and robot kinematics.
- **Temporal context:** a lightweight temporal model over sampled feature sequences.
- **Primary video strategy:** freeze a verified pretrained encoder and cache features.
- **Fusion progression:** unimodal baselines → concatenation/late fusion → small reliability-aware gate.
- **Phase recognition:** deferred unless a compatible phase-labelled dataset is verified.

## First task
Start with `ENV-001` in the implementation tracker: record the actual operating system, Python version, CPU/RAM/storage, and whether a GPU is available. No model code should be written before the dataset and environment gates are passed.
