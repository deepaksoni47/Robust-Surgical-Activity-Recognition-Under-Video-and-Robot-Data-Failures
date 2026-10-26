# SurgMotion Mentor-GPU Execution & Handoff Protocol

> **Core Operating Principle:**  
> Our local laptop is the primary development workstation. The mentor's machine is exclusively an execution compute node for heavy GPU forward passes.  
> **Do not clone the repository, re-download models from Hugging Face, or restart development on the mentor's computer.** Execute only the pre-tested scripts and transfer results back immediately.

---

## 1. Machine Division of Responsibilities

| Responsibility | Primary Laptop (Local) | Mentor's GPU Machine (Remote) |
|---|---|---|
| **Code Development & Debugging** | **100% on Laptop** | None (no source editing) |
| **Unit & Integration Testing** | **100% on Laptop** (all 188 tests pass) | None |
| **Model Architecture Verification** | **100% on Laptop** (strict 0-key mismatch verified) | None |
| **Single-Clip Smoke Test** | Completed locally on CPU (51.7s / 68.7s) | 1-clip GPU sanity check (~0.03s) |
| **Heavy Feature Extraction** | Infeasible on CPU (~24.5h Suturing, ~64h benchmark) | **Execute approved pilot trial only** |
| **Cache Validation & Integrity Audit** | Auditor implemented and tested locally | Run validator script after extraction |
| **Downstream Model Training & Pilot Eval** | **100% on Laptop** (VideoGRU, metrics, comparisons) | None |
| **Analysis, Paper Writing, & Version Control** | **100% on Laptop** | None |

---

## 2. Minimum Transfer Package Manifest

Transfer only the following pre-tested files from the laptop to the mentor's machine (via USB flash drive, local network share, or SCP). Total payload is ~1.16 GB, consisting almost entirely of the already-downloaded checkpoint and the 3.5 MB pilot video.

```text
surgmotion_gpu_package/
├── models/
│   └── SurgMotion-L/
│       ├── SurgMotion-vitl.pt          (1,159.34 MB - Pretrained foundation weights)
│       └── config.json                 (165 bytes - Model configuration)
├── dataset/
│   └── Suturing/
│       └── video/
│           └── Suturing_B001_capture1.avi (3.50 MB - Pilot trial video, 5,640 frames)
├── artifacts/
│   └── manifest.csv                    (31.66 KB - Trial metadata index)
├── src/                                (Source code modules)
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── manifest.py
│   │   └── video.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── visual_encoder.py
│   │   └── surgmotion_encoder.py       (16.91 KB - ViT-Large backbone wrapper)
│   └── utils/
│       ├── __init__.py
│       └── reproducibility.py
├── scripts/
│   ├── extract_surgmotion_features.py  (18.57 KB - Streaming extraction engine)
│   └── validate_surgmotion_cache.py    (5.96 KB - Cache integrity auditor)
└── requirements-surgmotion.txt         (347 bytes - Pinned dependency manifest)
```

**Files explicitly EXCLUDED from transfer:**
- Entire remaining JIGSAWS video dataset (39 Suturing, 36 Knot, 28 Needle videos: ~8 GB - NOT needed for pilot).
- `.git/` history and branches.
- Virtual environment (`.venv/`).
- Existing ResNet-18 feature caches (`data/cache/features/ResNet18/` - preserved on laptop).
- Checkpoints and prediction runs from prior baseline experiments.

---

## 3. Execution Protocol on the Mentor's Machine

On the mentor's computer, open a terminal in the transferred `surgmotion_gpu_package/` directory.

### Step 1: Set Up Isolated Python Environment
Install PyTorch matching the machine's installed NVIDIA driver and CUDA toolkit:

```bash
# Create and activate an isolated environment
python -m venv .venv

# On Linux:
source .venv/bin/activate
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# Install CUDA-enabled PyTorch (e.g., CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install the minimal pipeline dependencies:
pip install -r requirements-surgmotion.txt
```

### Step 2: Query GPU Hardware & Driver
Verify device visibility and driver support:
```bash
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB, CUDA: {torch.version.cuda}')"
```
*Acceptance criteria:* Device is CUDA-compatible with $\ge 8\text{ GB}$ VRAM.

### Step 3: Run Diagnostic 1-Clip GPU Smoke Test
Verify model loading, PyTorch SDPA attention backends, and memory safety without processing any real files:
```bash
python scripts/extract_surgmotion_features.py --smoke-test --device cuda --fp16
```
*Expected terminal output:*
```text
Target Device: cuda:0 (NVIDIA ...)
PyTorch Version: 2.x.x+cu121 (CUDA available: True)
Attention Backends: FlashAttention=True (or MemEfficient=True), Math=True
Loading SurgMotion backbone from: models/SurgMotion-L/SurgMotion-vitl.pt ...
Model loaded and frozen in ~2.5s (Parameters: 303,885,312)

--- Running Smoke Test (1 clip) ---
Smoke Test Result: SUCCESS
Input shape: (1, 64, 3, 224, 224)
Output shape: (1, 1024)
Feature dim: 1024
Finite check: True
Elapsed inference time: ~0.035s
Peak GPU VRAM allocated: ~3200 MB
```

### Step 4: Execute Approved Pilot Extraction (`Suturing_B001`)
Extract features for the single pilot trial using sliding-window batching:
```bash
python scripts/extract_surgmotion_features.py \
    --task Suturing \
    --trial-id Suturing_B001 \
    --device cuda \
    --fp16 \
    --batch-size 8 \
    --stride 16 \
    --output-dir data/cache/features_surgmotion
```
*Expected duration:* ~10 to 18 seconds (vs ~4 hours on CPU).

### Step 5: Validate Cache Integrity
Immediately audit the generated cache file on the mentor's computer:
```bash
python scripts/validate_surgmotion_cache.py data/cache/features_surgmotion/Suturing_B001_capture1.npz
```
*Acceptance criteria:* Exit code 0, status `VALID`, 349 clips, shape `(349, 1024)`, zero NaNs/Infs.

---

## 4. Artifact Return & Resuming Work on Laptop

### What to Copy Back to the Laptop:
Only copy the generated output folder:
- `data/cache/features_surgmotion/Suturing_B001_capture1.npz` (~1.4 MB)
- `data/cache/features_surgmotion/extraction_summary_Suturing.json` (~1 KB)

Place these files directly into `D:\miniproject-5thsem\data\cache\features_surgmotion\` on the laptop.

### Resuming Analysis on Laptop:
1. Run local validation:
   ```powershell
   .venv\Scripts\python.exe scripts/validate_surgmotion_cache.py data/cache/features_surgmotion/Suturing_B001_capture1.npz
   ```
2. Inspect measured extraction throughput, VRAM usage, and clip timings from `extraction_summary_Suturing.json`.
3. Proceed to pilot sequence modeling and metric comparison on our laptop under JIGSAWS official Suturing Fold 1.
