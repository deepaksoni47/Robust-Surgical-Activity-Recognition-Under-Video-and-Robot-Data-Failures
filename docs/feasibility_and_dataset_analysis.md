# Feasibility and dataset analysis

## Recommended dataset strategy

### Primary: JIGSAWS
The team reports having downloaded ZIP files for knot tying, needle passing, suturing, and experimental setup. The supplied project context indicates JIGSAWS includes synchronized video, robot kinematics, gesture annotations, and published evaluation protocols. The local archive must still be verified before coding.

**Supports:** gesture/action recognition, video and kinematics experiments, temporal sequence modeling.

**Does not automatically support:** general surgical phase recognition, clinical surgery, broad cross-procedure generalization, or real-world sensor failure prevalence.

**Access/licensing:** verify the official dataset terms and cite the official JIGSAWS page in the eventual paper. Do not redistribute the archives unless permitted. The official page referenced in project notes is `https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/`; verify current access instructions and terms.

### Candidate alternative: SurgSync dataset
The supplied paper describes 214 validated recordings across training tasks, including suturing and knot tying, with video, kinematics, and annotations. It may be relevant if the team later needs richer synchronization metadata or different task coverage. However, it is not necessary for the MVP, and exact download availability, terms, labels, and file structure must be verified from the project site before choosing it.

### Other papers' datasets
- GRAD uses MISAW and CUHK-MRG in the supplied text. These are useful comparisons but require separate access and label/split verification.
- Endo-C6 uses CholecT50, Kvasir, and TEMSET-24K for video/text-prompt classification; the supplied protocol does not establish matching robot kinematics.
- SurgMotion evaluates many datasets and describes a large pretraining corpus, but the corpus includes private institutional sources. Do not assume that all pretraining videos are downloadable.
- The attached author correction does not provide enough information to recommend the original paper's dataset or pipeline.

## Dataset selection decision
Use JIGSAWS first because it is already downloaded and supports the central video–kinematics gesture question. Do not switch datasets simply because a recent paper uses a different one.

## Required local verification
- Archive integrity and extraction success.
- Task folders and trial naming.
- Video codec, resolution, duration, and frame rate.
- Kinematic file format, sampling rate, dimensions, units, and timestamps/index conventions.
- Gesture annotation format, label vocabulary, start/end conventions, and label coverage.
- Mapping from video frames to kinematics samples.
- User/surgeon and trial identifiers.
- Official split protocols and whether they support the planned evaluation.
- Dataset license/access conditions.

## Split and leakage policy
Never randomly split adjacent frames or overlapping windows across train/test. All frames from a trial must remain within one split. Prefer a published JIGSAWS evaluation protocol and report it exactly. If evaluating generalization to unseen surgeons, use the relevant leave-one-user/surgeon-out protocol where supported. Keep validation trials/users separate from final test trials/users. Do not tune on the test set.

## Compute feasibility
The team has no local GPU suitable for training, so design the pipeline to:
- Run archive checks, metadata extraction, annotation parsing, and CPU unit tests locally.
- Use remote/free GPU only for encoder inference or model training when required.
- Freeze the visual encoder first.
- Cache visual features in a compact format and avoid re-extracting them for every experiment.
- Start with short clips, low batch sizes, and a tiny subset.
- Track GPU RAM, runtime, and storage per feature cache.
- Keep a fallback encoder that is small enough to run on available hardware.

The exact GPU, VRAM, storage, and remote runtime limits are not yet confirmed. They are explicit blockers for selecting a final foundation-model checkpoint.

## Foundation-model decision gate
Before adopting SurgMotion, verify:
1. Public repository and exact commit/version.
2. A downloadable downstream checkpoint, not merely pretraining code.
3. Inference example and expected clip length/frame sampling.
4. Model architecture, dependencies, license, and input normalization.
5. GPU memory and runtime on a small sample.
6. Whether it can output features without full fine-tuning.
7. Whether checkpoint use is permitted for the intended academic project.

If any item fails or cannot be resolved in a short feasibility window, choose a documented fallback such as a smaller pretrained video model or image encoder with temporal pooling. Label it accurately; do not call an image encoder a video foundation model.

## Feasibility verdict
The MVP is feasible if JIGSAWS video, kinematics, and gesture labels parse correctly and a pretrained visual encoder can be run or replaced with a smaller fallback. The full original question including both action and phase recognition, a large video foundation model, complex temporal modeling, and adaptive fusion is too broad for a short mini-project. Gesture recognition is the defensible core.
