# Problem statement and research scope

## Final primary research question

> On JIGSAWS surgical gesture recognition, does lightweight temporal fusion of pretrained video representations and robot kinematics outperform unimodal baselines, and does reliability-aware fusion reduce performance loss when one modality is degraded or unavailable?

This is a testable hypothesis, not an established result.

## Motivation
Video shows the surgical scene and instrument–tissue interaction. Robot kinematics describe instrument motion numerically. The signals may complement each other, but fusion can also fail if one modality is noisy, misaligned, or uninformative. The project tests whether simple fusion helps and whether a small reliability-aware gate helps under controlled stress tests.

## Tasks
**MVP task:** surgical gesture/action classification using JIGSAWS labels.

**Not part of the MVP:** general surgical phase recognition. JIGSAWS's standard annotations are gesture labels; do not relabel them as phases. Add phase recognition only after a separate phase-labelled dataset is verified and the schedule permits it.

## Hypotheses
- **H1 (video):** a pretrained video/image representation plus a small temporal classifier provides a meaningful video-only baseline.
- **H2 (kinematics):** a lightweight model on normalized kinematics can predict gestures above a trivial baseline.
- **H3 (fusion):** simple multimodal fusion improves at least one primary metric over the strongest unimodal baseline under the same split.
- **H4 (adaptive fusion):** a small gate that receives modality features and explicit availability/reliability indicators is more robust than fixed fusion under specified synthetic modality failures.
- **H5 (temporal context):** a short temporal model improves over independent-frame/independent-timestep prediction, assessed on identical splits and comparable parameter budgets.

No numerical gain is assumed in advance. Any hypothesis may be unsupported by the results.

## Minimum viable contribution
A reproducible comparison of video-only, kinematics-only, fixed fusion, and a lightweight reliability-aware fusion method on leakage-safe JIGSAWS splits, with clean performance and controlled modality degradation/missingness.

The contribution is an empirical evaluation and careful protocol—not a claim to invent multimodal fusion generally. GRAD already studies multimodal vision–kinematics fusion and visual corruption robustness.

## Essential components
1. Correctly parsed JIGSAWS videos, kinematics, gesture labels, trial/user identifiers.
2. Split protocol with no frame/window leakage across train and test.
3. A verified pretrained visual encoder, initially frozen.
4. A kinematics encoder (small MLP or GRU/TCN).
5. A lightweight temporal prediction head.
6. Fixed fusion baseline.
7. A small reliability-aware gate, introduced only after fixed fusion works.
8. Clean and controlled corruption/missing-modality evaluation.

## Deferred extensions
- Surgical phase recognition on another dataset.
- Cross-dataset generalization.
- Kinematic/video time-shift robustness if JIGSAWS timestamps permit a defensible alignment test.
- Complex graph networks, adversarial modality alignment, VeRA/LoRA, language prompts, contact sensors, skill scoring.
- Fine-tuning the visual backbone; only attempt if frozen-feature baselines and compute budget justify it.

## Non-goals
- Training SurgMotion or another foundation model from scratch.
- Reproducing all of GRAD, SurgSync, or RobustEndoCLIP.
- Clinical claims, real-time robot control, or deployment.
- Claiming synthetic corruption rates represent real-world incidence.
- Claiming novelty before a targeted literature review and exact protocol comparison.

## Proposed contribution statement
“We systematically compare unimodal and lightweight multimodal temporal models for JIGSAWS surgical gesture recognition and evaluate whether reliability-aware fusion improves robustness under controlled modality degradation and missingness.”

This statement must be revised if dataset inspection shows that any required modality or annotation is absent or incompatible.
