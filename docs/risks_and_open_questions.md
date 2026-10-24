# Risks, open questions, and decision gates

## Blocking questions before implementation

| ID | Question | Why it matters | Owner / deadline | Gate |
|---|---|---|---|---|
| Q-01 | What OS/Python/RAM/storage and remote GPU are available? | Determines dependencies and feasible encoder | Team to fill | ENV-001 |
| Q-02 | Are all JIGSAWS archives intact and fully extracted? | Missing files invalidate modality comparisons | Team to fill | DATA-002 |
| Q-03 | What exact kinematics format and frame/sample mapping are present? | Fusion requires defensible alignment | Team to fill | DATA-006 |
| Q-04 | Which JIGSAWS split protocol will be reported? | Prevents leakage and supports comparison | Team + mentor | DATA-008 |
| Q-05 | Is a SurgMotion downstream checkpoint publicly accessible and usable? | Paper description alone does not prove practical checkpoint access | Team to verify official repo | ENC-001 |
| Q-06 | What is the time limit for encoder exploration? | Prevents project being consumed by environment work | Team + mentor | ENV-005 |
| Q-07 | Is phase recognition required by the mentor? | JIGSAWS standard labels are gestures, not general phases | Team + mentor | ENV-004 |
| Q-08 | Are dataset/model terms compatible with intended academic use? | Licensing/access must be respected | Team to verify | ENV-003 / ENC-001 |

## Main risks and fallbacks

### Dataset or modality unavailable
**Risk:** Video, kinematics, and annotations may not be paired as assumed for every task.  
**Response:** Audit one trial per task; exclude only with documented reason, or revise scope before training. Never fabricate modality values.

### Label mismatch
**Risk:** Gesture/action labels are mistaken for phase labels.  
**Response:** Keep gesture recognition as MVP. Add phase recognition only with verified phase annotations and a compatible split.

### Pretrained checkpoint unavailable or too large
**Risk:** SurgMotion repository may not provide an immediately usable checkpoint or may exceed remote GPU limits.  
**Response:** Apply a fixed feasibility time cap and use a smaller pretrained encoder fallback. State exact model identity.

### Small dataset / class imbalance
**Risk:** Deep models overfit and rare gestures have poor recall.  
**Response:** Freeze encoder, use small heads, trial/user-aware evaluation, macro-F1, per-class metrics, and limited model complexity.

### Misalignment
**Risk:** Kinematics and video rows appear aligned by index but are not.  
**Response:** Inspect timestamps and published conventions; run alignment audit before fusion.

### Leakage
**Risk:** Random windows from the same trial appear in both train and test.  
**Response:** Split at trial/user level and enforce automated disjointness tests.

### Adaptive fusion adds complexity without benefit
**Risk:** Gate may overfit or learn to ignore one modality.  
**Response:** Compare to fixed fusion, inspect gate weights, test single-modality fallbacks, and accept a negative result.

### Synthetic corruption is overinterpreted
**Risk:** Artificial blur/dropout is presented as clinical reliability evidence.  
**Response:** Describe perturbation operators exactly and limit claims to the controlled benchmark.

### Scope creep
**Risk:** Full GRAD reproduction, phase recognition, foundation-model fine-tuning, and robust fusion exceed a semester mini-project.  
**Response:** Protect the MVP; optional extensions require all mandatory gates and time budget.

## Novelty caution
GRAD already reports multimodal vision–kinematics fusion and visual corruption robustness. Endo-C6 reports systematic visual corruption testing for temporal vision-language models. The defensible project question is a focused evaluation of lightweight fusion and explicit modality availability/reliability handling on the chosen dataset—not “the first multimodal surgical fusion model.” Conduct a targeted literature review before any novelty claim.

## Stop/go rules
- **Go to modeling** only after DATA-012 and PIPE-010 pass.
- **Go to foundation encoder extraction** only after ENC-001 and ENC-002 pass or fallback ENC-003 is accepted.
- **Go to fusion** only after unimodal baselines have reproducible predictions.
- **Go to adaptive fusion** only after fixed fusion has been evaluated and reliability signals are specified.
- **Go to final test evaluation** only after model selection is frozen.
