# Testing and validation plan

## Test levels

### 1. Environment tests
- Python and dependency versions can be printed.
- `pip check` reports no unresolved dependency conflicts.
- PyTorch imports and device discovery is accurate.
- Video decoder can open and read one sample frame.

### 2. Data unit tests
- Manifest identifiers are unique.
- Every referenced path exists or is explicitly marked unavailable.
- Video metadata values are plausible and finite.
- Kinematics arrays have expected rank/dimensions and finite-value report.
- Annotation labels belong to the declared vocabulary.
- Segment boundaries map to expected frame indices.
- Alignment does not silently extrapolate.
- Normalization is fitted on training data only.

### 3. Leakage tests
- Trial IDs are disjoint across train/validation/test.
- All windows from a trial inherit the same split.
- Duplicate/overlapping windows do not cross splits.
- Test data do not contribute to feature normalization, class weights, model selection, or threshold tuning.

### 4. Feature tests
- Encoder returns finite features.
- Feature dimensions match config.
- Cached features include trial ID, time/frame index, checkpoint ID, and preprocessing config.
- Different encoder/checkpoint versions cannot silently share the same cache key.

### 5. Model tests
- Input and output batch shapes are correct.
- Logits dimension equals number of gesture classes.
- Loss is finite.
- One tiny training batch produces gradients for trainable parameters.
- Frozen encoder parameters do not receive gradients in the MVP.
- Reliability gate weights sum to one over available modalities.
- Missing modality receives zero gate weight.
- Both modalities missing triggers a documented safe failure or fallback.

### 6. Evaluation tests
- Macro-F1 and balanced accuracy match a hand-computed toy case.
- Prediction files contain trial/time IDs, true labels, predicted labels, and scores.
- Metric calculations are reproducible from saved predictions.
- Corruption operators leave labels and split assignments unchanged.
- Clean and corrupted results are reported side by side.

## Smoke-test order
1. Import project modules.
2. Load one trial.
3. Run data checks.
4. Sample frames and map labels.
5. Load kinematics and verify alignment.
6. Run one encoder batch.
7. Load cached features.
8. Run one forward pass per baseline.
9. Train for a few batches.
10. Run evaluation on a tiny held-out subset.
11. Run a missing-modality gate test.
12. Run the full test suite before major experiments.

## Completion rule
Code being present is not enough. A task is complete only when its acceptance criteria pass and the command/log or test evidence is recorded.
