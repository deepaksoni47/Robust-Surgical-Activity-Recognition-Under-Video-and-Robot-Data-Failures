# Research synthesis: attached-paper analysis

## Scope and evidence policy

The analysis below uses the attached Markdown content surfaced in this project. Claims described as **paper-reported** are attributed to the supplied paper text. Implementation recommendations are our proposed choices, not reported results. Repository contents, pretrained checkpoints, full licenses, and supplementary materials must be checked directly before depending on them.

### Attachment inventory identified
1. **GRAD** — *Multimodal Graph Representation Learning for Robust Surgical Workflow Recognition with Adversarial Feature Disentanglement* (`2505.md`).
2. **SurgMotion** — *SurgMotion: A Video-Native Foundation Model for Universal Understanding of Surgical Videos* (`2602.05638v5.md`).
3. **SurgSync** — *SurgSync: Time-Synchronized Multi-Modal Data Collection Framework and Dataset for Surgical Robotics* (paper Markdown surfaced as `turn0file2`).
4. **On the Robustness of Temporal Vision-Language Models for Surgical Endoscopy Videos** / Endo-C6 / RobustEndoCLIP (paper Markdown surfaced as `turn0file3`).
5. **Author Correction: Large-scale self-supervised video foundation model for intelligent surgery** (`s41746-026-02864-3.md`). This is a correction notice, not the full research paper; the original article is referenced by DOI `10.1038/s41746-026-02403-0` and must be read separately before using its method details.

If additional `.md` files are attached but not surfaced in this inventory, add them to the inventory and analyze them before implementation. The correction notice does not count as a substitute for the original article.

## 1. GRAD

**Paper-reported problem and motivation.** GRAD targets surgical workflow recognition when video or kinematic data are corrupted or shifted. It argues that vision and kinematics provide complementary signals and that naive fusion may not model modality-specific structure or cross-modal relationships adequately.

**Reported method.**
- MDGNet: extracts modality-specific features and models intra-/inter-modal relationships with graph-based representations.
- Visual pathway: feature disentanglement across spatial, wavelet, and Fourier domains; temporal convolutional network (TCN) for sequence information.
- Kinematics pathway: TCN and LSTM temporal feature extraction.
- Vision-Kinematic Adversarial (VKA) learning: aligns the feature distributions across modalities using adversarial training.
- Contextual Calibrated Decoder: combines graph and original modality embeddings to improve calibrated prediction under corruption.
- The supplied text reports a pretrained ResNet18 visual backbone, 320×256 resizing followed by 224×224 crops, graph hidden/output dimensions of 64, dropout 0.5, Adam with learning rate 1e-4 and batch size 64.

**Datasets and labels.** The supplied paper describes MISAW and CUHK-MRG, both with visual data, kinematics, and workflow/gesture annotations. MISAW is described as 27 recordings with synchronized video and kinematics at 30 Hz and six gesture categories; the reported split is 17 training and 10 test cases. CUHK-MRG is described as 24 sequences with five gesture steps and end-effector pose/orientation plus gripper opening, with a three-fold protocol. Exact split details should be checked against the original dataset protocols before reproduction.

**Reported evaluation/results.** The abstract reports 86.87% and 92.38% accuracy on the two datasets. The paper also reports robustness testing over image corruption types and severities. These figures are paper-reported and must not be compared directly with JIGSAWS results because datasets, labels, and splits differ.

**Limitations explicitly reported.** Short-duration classes are hard to recognize and class imbalance can bias predictions. Multimodal training requires time, compute, and accurate alignment. The evaluation is on benchmark datasets rather than clinical surgical scenes.

**Useful for our project.** The paper motivates unimodal baselines, explicit temporal modeling, multimodal comparison, and robustness evaluation. It also warns against assuming multimodal fusion is automatically better.

**Do not reproduce in the MVP.** Full graph disentanglement across multiple frequency/spatial domains, adversarial modality alignment, and the full calibrated decoder add substantial implementation and tuning burden. A small fusion gate can test the main hypothesis more cleanly.

**Evidence boundary.** The supplied excerpt does not establish that GRAD evaluates deliberate robot-kinematics dropout or controlled video–kinematics timestamp offsets. Do not claim those exact tests are absent without checking the full methods and supplement.

## 2. SurgMotion

**Paper-reported problem and motivation.** SurgMotion argues that pixel-level reconstruction can spend capacity on visually variable details that are not central to surgical semantics. It targets motion-centric features, temporal relational consistency, and broader procedure diversity.

**Reported method.**
- Built on Video Joint Embedding Predictive Architecture (V-JEPA).
- Motion-guided latent masked prediction weights learning toward motion-informative regions.
- Spatiotemporal affinity self-distillation uses a student/EMA teacher to encourage relational consistency.
- Spatiotemporal Feature Diversity Regularization (SFDR) aims to prevent low-diversity/collapsed representations in texture-sparse scenes.
- The paper describes SurgMotion-15M as 3,658 hours from 50 sources across 13 organs and reports evaluation across 17 benchmarks.

**Tasks/datasets/results reported.** The supplied text reports workflow recognition results including +14.6 F1 on EgoSurgery and +10.3 F1 on PitVis relative to the best prior method, and 39.54% mAP-IVT on CholecT50 for action-triplet recognition. It also reports skill assessment, segmentation, and depth-estimation experiments. These results are paper-reported, not expected outcomes for our project.

**Limitations/constraints.** Large-scale pretraining is far outside this mini-project's compute and data budget. The paper notes that some procedures and institutions remain underrepresented. A usable downstream checkpoint, code interface, license, and hardware requirements must be verified; the Markdown alone does not prove that a downloadable checkpoint is immediately usable.

**Useful for our project.** The paper supports using a pretrained surgical video representation rather than training a foundation model from scratch. It motivates keeping the encoder frozen initially and caching its features.

**MVP recommendation.** Treat SurgMotion as a preferred candidate only after checkpoint, dependency, and inference-memory checks. If it cannot be run reliably on available remote compute, use a smaller verifiable pretrained video/image encoder and state clearly that it is a fallback rather than a foundation-model reproduction.

## 3. SurgSync

**Paper-reported problem and motivation.** SurgSync addresses acquisition and synchronization of multiple surgical sensing streams. It provides online and offline timestamp matching and a post-collection processing toolbox.

**Reported modalities and system.** Stereo/mono endoscopic video, robot kinematics from patient-side manipulators and endoscope manipulator, measured and setpoint states, gripper/jaw information, optional contact sensing, and event/phase/gesture annotations are described. The setup is based on the da Vinci Research Kit and ROS. The paper reports 214 validated recordings across canonical training tasks, including suturing, knot tying, peg transfer, tissue manipulation, and dissection. It reports 1080p image streams at 30/60 Hz depending on camera and a 1 kHz dVRK framework, while the offline recorder's measured recording frequency is reported around 10 Hz in its evaluation.

**Reported synchronization approach.** Online matching pairs samples within a time tolerance. Offline matching records streams separately and matches timestamps later, with nearest-neighbor selection/interpolation. The paper reports a mean latency of 6.36 ± 4.72 ms for online matching and 1.35 ± 0.81 ms for offline matching in its evaluated subsets, with different recording frequencies and readiness trade-offs.

**Reported downstream model.** Its skill-assessment demonstration combines 14D kinematics, 2048D ResNet101 visual features, and 14D one-hot gesture labels. It reports SROCC means of 0.803 for suturing and 0.765 for knot tying across four folds. This is a skill-assessment task, not the same task as our gesture classifier.

**Useful for our project.** Timestamp alignment is a data requirement, not a cosmetic preprocessing step. Keep timestamps and trial identifiers in the data manifest, inspect alignment explicitly, and avoid assuming synchronization simply because two files share an index.

**Limitations for our project.** SurgSync is a framework and a separate dataset, not evidence that the team's JIGSAWS files have the same format or labels. The paper's full sensor setup and hardware pipeline are not required for an offline JIGSAWS project. Dataset access, license, and exact annotation availability still need verification if considering it as an alternative.

## 4. On the Robustness of Temporal Vision-Language Models for Surgical Endoscopy Videos (Endo-C6 / RobustEndoCLIP)

**Paper-reported problem.** Temporal vision-language models may fail under endoscopy-specific corruptions such as defocus, fog/haze, shot noise, motion blur, packet loss, and smoke.

**Reported method and protocol.**
- Endo-C6 is a six-corruption test benchmark.
- It evaluates temporal VLMs on CholecT50, Kvasir, and TEMSET-24K.
- Clean test clips are treated as in-distribution; corruptions are applied at test time.
- Metrics include clean Top-1 accuracy, Mean-C across corruptions, and Worst-C.
- RobustEndoCLIP uses few-shot parameter-efficient tuning with VeRA. The supplied text describes a SurgVLP/ResNet-50 instantiation, frozen encoders, adaptation of the final FC layer, Adam, learning rate 1e-3, batch size 16, 50 epochs, and 224×224 resolution.

**Reported results.** The supplied tables report substantial variation by dataset and corruption and report RobustEndoCLIP as the strongest method on several Mean-C/Worst-C comparisons. Do not copy those numbers as targets for JIGSAWS; the tasks and datasets differ.

**Limitations and relevance.** The paper studies prompt-based temporal vision-language clip classification and visual corruptions. It does not, in the supplied text, establish multimodal robot-kinematics fusion. Its main value here is the robustness protocol: report clean performance and both average and worst-case degradation, with controlled test-only perturbations.

**MVP adaptation.** Test a small, clearly specified set of video corruptions and kinematics perturbations only after clean baselines work. Keep corruption operators and severities documented. Synthetic corruption tests robustness to those operators, not the prevalence of failures in actual surgery.

## 5. Author Correction: Large-scale self-supervised video foundation model for intelligent surgery

The attached file is a correction notice stating that an author was omitted from the original author list. It is not a method paper and contains no sufficient architecture, dataset, training, or evaluation details for implementation planning.

**Required follow-up:** retrieve and read the original paper at DOI `10.1038/s41746-026-02403-0`, inspect its license, code, checkpoint access, and relevant downstream task protocol. Until then, cite it only as a correction notice and do not infer its method from the title.

## Cross-paper comparison

| Paper | Main contribution in supplied text | Modalities | Most relevant lesson | MVP decision |
|---|---|---|---|---|
| GRAD | Graph-based multimodal fusion, adversarial alignment, calibrated decoding, corruption tests | Video + kinematics | Compare modalities separately; temporal and robustness design matter | Adapt conceptually; do not reproduce full architecture |
| SurgMotion | Surgical video foundation model with motion-aware latent pretraining | Video | Reuse pretrained visual features; do not pretrain from scratch | Candidate encoder after checkpoint/compute verification |
| SurgSync | Synchronized collection and processing framework | Video + kinematics + optional sensors/labels | Timestamp alignment and manifests must be explicit | Use as data-engineering reference, not as assumed data format |
| Endo-C6 / RobustEndoCLIP | Controlled visual corruption benchmark and light adaptation | Video + text prompts | Report clean, mean-corruption, and worst-case performance | Adapt evaluation style; not its exact model |
| Author correction | Corrects author list of a separate foundation-model paper | Not enough method detail | Original paper must be retrieved | No implementation component until verified |

## Dependencies between ideas

1. Dataset integrity and annotation interpretation come before model choice.
2. Timestamp/sequence alignment comes before multimodal fusion.
3. A valid unimodal baseline comes before any claim about fusion.
4. Feature caching comes before repeated temporal/fusion experiments if video inference is expensive.
5. Basic fusion comes before reliability-aware gating.
6. Clean evaluation comes before corruption robustness experiments.
7. Model/checkpoint selection depends on verified code, weight access, license, and memory use.
8. Any phase-recognition extension depends on finding a phase-labelled dataset with compatible modalities; JIGSAWS gesture labels alone do not satisfy this requirement.
