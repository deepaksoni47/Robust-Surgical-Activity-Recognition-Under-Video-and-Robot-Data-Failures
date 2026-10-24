# Environment and setup plan

This is a setup specification, not a claim that an environment has already been created.

## Recommended stack
- Python 3.10 or 3.11, chosen after confirming compatibility with the selected encoder.
- PyTorch for model training/inference.
- NumPy and pandas for arrays/manifests.
- scikit-learn for metrics and simple baselines.
- PyYAML for configuration.
- pytest for unit/integration tests.
- OpenCV or PyAV for video decoding, selected after codec inspection.
- matplotlib for diagnostic plots.
- Git for version control.

Do not install every possible library up front. Foundation-model dependencies may impose version constraints; isolate encoder-specific requirements if needed.

## Hardware check
Record:
- CPU model and cores.
- RAM.
- Free disk.
- GPU model and VRAM, if present.
- Driver/CUDA compatibility if using NVIDIA GPU.
- Remote GPU provider and session/storage limits, if used.

Local CPU should be sufficient for archive inspection and unit tests. Video encoder inference and training may require a remote GPU depending on the checkpoint.

## Suggested initial directory structure
```text
surgical-multimodal/
├── README.md
├── pyproject.toml                 # or pinned requirements files
├── .gitignore
├── configs/
│   ├── data.yaml
│   ├── encoder.yaml
│   ├── kinematics_baseline.yaml
│   ├── video_baseline.yaml
│   ├── fusion.yaml
│   ├── robustness.yaml
│   └── splits/
├── data/
│   ├── raw_archives/              # ignored by Git
│   ├── raw/                       # ignored by Git
│   └── cache/                     # ignored by Git
├── src/
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   ├── robustness/
│   └── utils/
├── scripts/
├── tests/
├── artifacts/                     # ignored by Git except small reports if desired
├── notes/
└── docs/
```

## Setup sequence
1. Install Git and a supported Python distribution.
2. Create the repository and `.gitignore`.
3. Create and activate a virtual environment.
4. Install minimal base dependencies.
5. Run `python -m pip check`.
6. Install PyTorch using the current official installation selector for the target machine; do not copy a CUDA command without checking the runtime.
7. Verify:
   ```bash
   python --version
   python -m pip --version
   python -m pip check
   python -c "import numpy, pandas, sklearn, pytest; print('base imports OK')"
   python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
   ```
8. Install the chosen video-decoding library and test one sample video.
9. Install encoder-specific dependencies only after `ENC-001` verifies its requirements.
10. Run `pytest -q` after adding starter tests.

## Reproducibility conventions
- Store hyperparameters in YAML, not only command-line history.
- Each run saves config, Git commit, seed, split ID, encoder/checkpoint identity, metrics, predictions, and resource usage.
- Do not place credentials or tokens in repository files.
- Large dataset/checkpoint files remain outside Git.
