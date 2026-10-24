# Implementation plan and milestones

**No model implementation begins until the documentation audit and decision gates are accepted.** The tracker provides the atomic tasks.

## Milestone 0 — Constraints & Setup Confirmed
- **OS & Hardware:** Windows 11, Intel Core i5-12450H (8C/12T), 16 GB RAM, Intel UHD Graphics (local CPU mode; remote GPU / Google Colab for SurgMotion inference).
- **Disk:** 22.7 GB free on D: drive, 92.7 GB free on C: drive.
- **Python:** 3.10.11 managed via dedicated `.venv`.
- **Dataset Confirmed:** JIGSAWS already extracted in `dataset/` across all 3 tasks (Suturing: 39 trials, Knot Tying: 36 trials, Needle Passing: 28 trials = 103 trials total) with 76D kinematics, stereo videos, annotations, and official LOUO cross-validation splits in `dataset/Experimental_setup/`.
- **Target Encoder:** Strictly targeting **SurgMotion** (V-JEPA motion-guided surgical video foundation model), with a Colab / remote GPU extraction pipeline producing cached embeddings for local temporal & fusion modeling.
- **Evaluation Scope:** All 3 tasks processed in parallel from the start under the official 8-fold Leave-One-User-Out protocol.

## Milestone 1 — Reproducible project skeleton
- Create Git repository, Python environment, source/test/config directories.
- Pin only foundational dependencies first.
- Add run configuration, logging conventions, and documentation.
- Verify clean environment import and deterministic seed utility.

**Exit gate:** a clean checkout can recreate the environment and run smoke tests.

## Milestone 2 — Dataset audit
- Inspect ZIP listing before extraction.
- Extract into a data directory excluded from Git.
- Build a file inventory and manifest.
- Parse one trial from each task.
- Verify annotations, kinematics shape, video metadata, and alignment assumptions.
- Record exact label vocabulary and split protocol.

**Exit gate:** one complete trial can be loaded and checked end to end; all required modality availability is documented.

## Milestone 3 — Data pipeline
- Implement video sampling and deterministic time/frame mapping.
- Implement annotation-to-frame label mapping.
- Implement kinematics loading, missing-value checks, and training-only normalization.
- Build trial-level split files and leakage tests.
- Run the pipeline on a tiny subset.

**Exit gate:** pipeline outputs are reproducible and pass data-integrity tests.

## Milestone 4 — Encoder feasibility
- Verify SurgMotion repository/checkpoint/access and run one small inference if possible.
- Measure memory/runtime and test output shapes.
- If infeasible, test a smaller pretrained fallback.
- Cache features for a small subset and validate feature metadata.

**Exit gate:** one selected encoder is usable within the compute budget and has a documented fallback.

## Milestone 5 — Baselines
- Establish majority-class and/or simple kinematics statistical baseline.
- Train kinematics-only classifier.
- Train video-only classifier using frozen cached features.
- Compare independent-timestep prediction with temporal model.
- Tune using validation only.

**Exit gate:** each baseline has saved config, checkpoint, predictions, and metrics.

## Milestone 6 — Fixed multimodal fusion
- Implement late-logit fusion.
- Implement feature concatenation if time permits.
- Verify modality order and dimensions.
- Compare to best unimodal model on the same split.

**Exit gate:** fusion experiment is reproducible and its result can be interpreted even if it does not improve performance.

## Milestone 7 — Reliability-aware fusion
- Define measurable reliability/availability features.
- Implement a small gate with explicit masking for missing modalities.
- Unit-test weights, normalization, and single-modality fallback.
- Train and compare on clean validation data first.
- Test missing/noisy modality conditions using fixed corruption recipes.

**Exit gate:** gate behavior is verified, and results include fixed-fusion comparison.

## Milestone 8 — Final experiments and analysis
- Run selected seeds/splits according to available time.
- Report macro-F1, balanced accuracy, accuracy, and per-class results.
- Report robustness drop and worst-case condition.
- Analyze confusion matrices and failure examples.
- Include negative results and runtime/resource use.

**Exit gate:** no results are reported without associated split/config/checkpoint and prediction files.

## Milestone 9 — Reproducible handoff
- Re-run the main experiment from a clean environment/config.
- Confirm artifacts and documentation.
- Update tracker status only after acceptance criteria pass.
- Write limitations and future-work section without overstating novelty.
- Prepare a short mentor-facing summary.

**Exit gate:** another student can reproduce the main table from the documented commands and available files.

## Suggested division of labor
Parallel work is possible after Milestone 0:
- Person A: dataset format, annotations, and split protocol.
- Person B: encoder/checkpoint feasibility and environment survey.
Both review the final architecture and evaluation protocol before model implementation.

## Scope guard
If the encoder cannot be verified within the agreed time budget, do not spend the project trying to reproduce a large foundation model. Move to the documented fallback and preserve the core question about multimodal fusion.
