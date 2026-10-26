# SurgMotion Mentor-GPU Execution & Handoff Protocol

This guide documents how to execute SurgMotion visual feature extraction on the mentor's GPU machine and transfer cached representations back to our primary development workstation.

---

## 1. Machine Profile & Resource Requirements

| Specification | Primary Workstation (Acer) | Mentor's GPU Machine (Target) |
|---|---|---|
| **CPU** | Intel Core i5-12450H (8C/12T) | Intel / AMD Multi-Core Host |
| **GPU / Accelerator** | Intel UHD Graphics (No CUDA) | NVIDIA CUDA GPU (RTX 3080/4080/4090, A100, V100, A6000) |
| **VRAM Requirement** | N/A (CPU-only) | Minimum 8 GB VRAM (Recommended: 12+ GB) |
| **RAM** | 16 GB (~5.9 GB free) | 16+ GB System RAM |
| **1-Clip Inference Time** | 36.79s (Spatiotemporal) / 68.69s (Tubelet) | ~0.025s – 0.050s (Est. 700x–1000x acceleration) |
| **Full Suturing Extraction** | ~24.5 hours (CPU, single-threaded) | ~5–10 minutes (GPU, batch_size=8, FP16) |

---

## 2. Environment Setup on Mentor's GPU Machine

### Step 1: Clone Repository or Copy Directory
Clone the repository on the GPU machine:
```bash
git clone https://github.com/deepaksoni47/Robust-Surgical-Activity-Recognition-Under-Video-and-Robot-Data-Failures.git
cd Robust-Surgical-Activity-Recognition-Under-Video-and-Robot-Data-Failures
```

### Step 2: Copy Checkpoint & Target Videos
Transfer the downloaded SurgMotion checkpoint to the GPU machine:
- Place checkpoint at: `models/SurgMotion-L/SurgMotion-vitl.pt`
- Place config at: `models/SurgMotion-L/config.json`

For pilot evaluation, copy only the pilot trial video(s) to avoid copying the entire dataset:
- `dataset/Suturing/video/Suturing_B001_capture1.avi` (and any other selected pilot trials)

### Step 3: Install Dependencies
Create an isolated Python environment and install PyTorch matching the target machine's CUDA driver:

**On Linux / Windows (with CUDA 12.1):**
```bash
python -m venv .venv

# Activate environment:
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Install CUDA-enabled PyTorch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install required pipeline dependencies:
pip install -r requirements-surgmotion.txt
```

---

## 3. Step-by-Step Execution Workflow

### Step 1: Run 1-Clip Diagnostic Smoke Test
Verify that CUDA, PyTorch, model loading, and SDPA attention backends operate correctly:
```bash
python scripts/extract_surgmotion_features.py --smoke-test --device cuda --fp16
```

**Expected Smoke Test Output:**
```text
Target Device: cuda:0 (NVIDIA RTX ...)
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

### Step 2: Run Pilot Feature Extraction (Single Trial)
Extract features for the official pilot trial (`Suturing_B001`):
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

### Step 3: Verify Cache Integrity
Validate that the resulting `.npz` archive has correct shapes, strictly monotonic frame indices, zero NaNs/Infs, and complete provenance metadata:
```bash
python scripts/validate_surgmotion_cache.py data/cache/features_surgmotion/Suturing_B001_capture1.npz
```

**Expected Validation Output:**
```text
  [OK] Suturing_B001_capture1.npz: 349 clips, shape (349, 1024), 1.36 MB, frames 32-5600
==================================================
Audit Summary: 1/1 files VALID.
All audited cache files passed integrity and finiteness checks.
```

---

## 4. Artifact Transfer Back to Primary Workstation

Once extraction completes, transfer **only the generated feature cache archives** back to the primary computer:
1. Copy the `.npz` files and JSON summaries from:
   `<remote_gpu_path>/data/cache/features_surgmotion/`
2. Place them into our local directory:
   `D:\miniproject-5thsem\data\cache\features_surgmotion\`
3. Run the local validator to confirm zero corruption during transfer:
   ```powershell
   .venv\Scripts\python.exe scripts/validate_surgmotion_cache.py --dir data/cache/features_surgmotion
   ```

---

## 5. Temporal Alignment & Annotation Mapping Protocol

### Coordinate Systems & Sampling Conventions
- **Raw Video:** 30 FPS (`Suturing_B001_capture1.avi`, 5,640 frames).
- **Robot Kinematics:** 30 Hz (5,640 timesteps, 76 features).
- **Gesture Annotations:** 1-indexed intervals $[s, e, \text{gesture}]$ from `transcriptions/`.
- **SurgMotion Windowing:**
  - Clip length $L = 64$ frames (~2.13 seconds of physical time).
  - Window stride $S = 16$ frames (~0.53 seconds step).
  - For clip $k$ spanning frames $[s_k, e_k]$ where $e_k = s_k + 63$:
    $$\text{center frame } c_k = \left\lfloor \frac{s_k + e_k}{2} \right\rfloor = s_k + 31$$
  - The ground-truth gesture label $y_k$ is assigned strictly based on the center frame:
    $$y_k = \text{GestureLabel}(c_k)$$
  - Kinematics vector $x_k^{\text{kin}}$ is sampled at the identical center frame $c_k$:
    $$x_k^{\text{kin}} = \text{Kinematics}(c_k)$$
- **Zero-Leakage Assurance:**
  - Normalization parameters are computed solely on training trials within each LOUO fold.
  - Test set trials are never used to fit feature scalers or select temporal strides.
