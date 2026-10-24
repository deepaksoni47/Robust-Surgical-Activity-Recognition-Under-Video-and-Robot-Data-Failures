# Data pipeline specification

## 1. Acquisition and raw-data policy
- Keep original JIGSAWS ZIPs unchanged in `data/raw_archives/`.
- Extract into `data/raw/jigsaws/`.
- Do not commit raw data, extracted videos, checkpoints, or feature caches to Git.
- Record archive names, byte sizes, hashes where practical, access terms, and acquisition date.
- Never infer missing content from an expected filename.

## 2. Manifest
Create one record per trial with:
- `trial_id`, `task`, `user_id`/surgeon ID if encoded,
- video and kinematics paths,
- annotation path,
- video FPS, frame count, duration,
- kinematics sample count/rate and dimensions,
- label vocabulary/version,
- alignment method and known limitations,
- split assignment.

Use stable identifiers from the dataset rather than generated row numbers as the sole join key.

## 3. Annotation handling
- Parse the original annotation format exactly.
- Confirm whether labels are frame-wise, segment-wise, or encoded by start/end frame.
- Test inclusive/exclusive boundaries and any gaps/overlaps.
- Map labels to video frames using verified frame indexing.
- Preserve the raw annotation alongside the derived frame labels.
- Report class counts by trial and split.
- Do not reinterpret gesture labels as surgical phases.

## 4. Video processing
- Inspect codecs and FPS before selecting a decoder.
- Make clip length, stride, frame sampling, resizing, crop, and normalization configurable.
- Save sampled frame indices/timestamps in metadata.
- Use deterministic sampling for validation/test.
- Use the encoder's documented preprocessing; do not assume generic ImageNet normalization is correct for a surgical checkpoint.
- Test one video first, then one trial per task, then a small subset.

## 5. Kinematics processing
- Inspect columns, units, dimensions, timestamps, and missing-value patterns.
- Separate identifiers/timestamps from model features.
- Fit scaling statistics on training trials only; reuse them for validation/test.
- Treat missing rows and non-finite values explicitly.
- If resampling is needed, document interpolation and maximum allowed gap.
- Do not extrapolate outside the recorded time range silently.

## 6. Synchronization
- Prefer explicit timestamps or documented frame/sample mapping.
- If only sequence indices exist, verify the dataset's published convention before index-based pairing.
- Create a per-trial alignment audit with length comparisons, start/end times, and sample examples.
- If timing mismatch is detected, document whether samples are dropped, interpolated, or excluded.
- Never use test labels or predictions to decide alignment parameters.

## 7. Splits and leakage
- Assign complete trials to a single split.
- Follow an established JIGSAWS protocol where feasible and cite it.
- If windows overlap, they inherit the trial split; no overlapping window can cross train/test.
- Fit normalization and class weights on training data only.
- Freeze split files before final evaluation.
- Add an automated test that asserts trial IDs are disjoint across splits.

## 8. Feature extraction and cache
- Run the selected frozen encoder in inference mode.
- Cache feature arrays with `trial_id`, frame/time index, model name/version, checkpoint ID/hash, preprocessing config hash, feature dimension, and extraction date.
- Use safe formats such as NumPy `.npy`/`.npz` or HDF5 only if needed.
- Validate finite values and dimensions.
- Do not mix feature caches from different checkpoints or preprocessing configs.
- Keep cache metadata separate from model training labels when possible.

## 9. Pipeline validation gates
1. Archive listing verified.
2. One trial's video, kinematics, and labels loaded.
3. Alignment rule confirmed.
4. Class vocabulary/counts validated.
5. Split leakage test passes.
6. Two-trial end-to-end smoke test passes.
7. Feature cache validated.
8. Full dataset audit report has no unexplained hard errors.
