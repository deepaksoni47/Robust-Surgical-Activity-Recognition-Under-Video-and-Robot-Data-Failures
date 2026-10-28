# ITER-34 — SurgSync Acquisition and Extraction Guide

**Document Date:** October 28, 2026  
**Status:** Pipeline Prepared & Verified; Awaiting Mentor Machine Execution  
**Target Dataset:** SurgSync v1.0.0 (`jackzhy/surgsync`)  
**Target Task:** `single_interrupted_stitch` (92 episodes, 90,068 frames, 3.71 h)  
**Scripts:**
- Downloader: [`scripts/download_surgsync_suturing.py`](file:///D:/miniproject-5thsem/scripts/download_surgsync_suturing.py)
- Extractor: [`scripts/extract_surgmotion_surgsync.py`](file:///D:/miniproject-5thsem/scripts/extract_surgmotion_surgsync.py)
- Test Suite: [`tests/test_surgsync_acquisition.py`](file:///D:/miniproject-5thsem/tests/test_surgsync_acquisition.py) (6/6 tests pass)

---

## 1. Executive Summary & Storage Realities

The complete SurgSync v1.0 release is **$714.35\text{ GB}$** on disk across 11,745 files. Because raw video streams are encoded in lossless FFV1 at $1920 \times 1080$, video accounts for **$99.98\%$ ($714.19\text{ GB}$)** of the archive.

To avoid wasting storage and bandwidth on unrelated tasks (dissection, peg transfer, tissue manipulation), our acquisition pipeline selectively downloads **only the Suturing subset (`single_interrupted_stitch`)** and metadata:
- **Total Suturing Download Size:** **$372.41\text{ GB}$** ($372.33\text{ GB}$ raw FFV1 video + $\sim 85\text{ MB}$ parquets and calibration metadata).
- **Required Storage Threshold:** **$\ge 450.0\text{ GB}$ free disk space** (incorporates a $77.6\text{ GB}$ safety margin).
- **Mentor Machine Suitability:** The mentor machine reportedly has $\ge 3\text{ TB}$ of storage, easily satisfying the $450\text{ GB}$ safety threshold.
- **Laptop Suitability:** The development laptop has $\sim 20.6\text{ GB}$ free space and **cannot store the raw video**. However, the laptop can download the parquets and metadata alone via `--skip-video` ($\sim 85\text{ MB}$).

---

## 2. Safety Features & Resumability

The download pipeline [`scripts/download_surgsync_suturing.py`](file:///D:/miniproject-5thsem/scripts/download_surgsync_suturing.py) guarantees:
1. **Automated Free-Space Preflight:** Queries `shutil.disk_usage` on the target volume before initiating network transfers. If free space is less than the required threshold ($450\text{ GB}$ for full, $1\text{ GB}$ for `--skip-video`), it aborts with exit code 1 to protect the host machine from disk exhaustion.
2. **Resumable Transfers:** Leverages `huggingface_hub.snapshot_download` with `resume_download=True`. If network connection drops or the process is halted, re-running the script resumes from the exact byte offset without re-downloading completed files.
3. **Selective Task Ingestion:** Employs `allow_patterns` to restrict ingestion strictly to:
   - `meta/*` (release metadata, manifest, tasks vocabulary)
   - `online_data/episodes/single_interrupted_stitch/*` (45 episodes)
   - `offline_data/episodes/single_interrupted_stitch/*` (47 episodes)
4. **Post-Download Integrity Verification:** Scans every downloaded file against the official `meta/manifest.json`, verifying file existence, byte size, and optional SHA-256 hash.
5. **Git Isolation:** `.gitignore` excludes all `data/`, `*.mkv`, `*.npz`, and `*.parquet` files, guaranteeing that large binaries cannot be accidentally staged into version control.

---

## 3. Step-by-Step Mentor Machine Execution Runbook

### Step 1: Storage & Environment Preflight
Before downloading, verify the destination directory and free space:

```bash
# 1. Pull the approved commit
git pull origin main

# 2. Activate conda / python environment
conda activate dvrk_env

# 3. Run preflight check on target drive (e.g. E:/surgsync_v10 or D:/surgsync_v10)
python scripts/download_surgsync_suturing.py --dest-dir E:/surgsync_v10 --preflight-only
```
*Expected Output:*
```text
======================================================================
  SurgSync Suturing Subset Acquisition Engine (ITER-34)
======================================================================
Target Repository : jackzhy/surgsync (dataset)
Destination Path  : E:\surgsync_v10
Mode              : Full Suturing Subset (~372 GB with raw FFV1)

[Preflight] Free Disk Space    : 2480.50 GB
[Preflight] Required Threshold  : 450.00 GB
[Preflight] Disk Space Check    : PASSED

Preflight-only requested. All storage checks passed. Exiting without downloading.
```

---

### Step 2: Execute Resumable Download
Once preflight passes, initiate the download on the mentor machine:

```bash
python scripts/download_surgsync_suturing.py \
    --dest-dir E:/surgsync_v10 \
    --max-workers 8 \
    --min-free-gb 450.0
```
*Expected Download Time:*
- On a $250\text{ Mbps}$ connection: $\sim 3.5$ to $4.0\text{ hours}$.
- On a $500\text{ Mbps}$ connection: $\sim 1.8$ to $2.2\text{ hours}$.

---

### Step 3: Post-Download Integrity Verification
After download completes, certify that all 92 episodes match the official manifest:

```bash
python scripts/download_surgsync_suturing.py --dest-dir E:/surgsync_v10 --verify-only
```
*Acceptance Criteria:* `Integrity Status: PASSED` (all files present with exact byte size).

---

### Step 4: Small-Sample GPU Feature Extraction Check
Before launching bulk extraction across all 92 episodes, verify the model, VRAM allocation, and throughput on a single episode:

```bash
python scripts/extract_surgmotion_surgsync.py \
    --data-dir E:/surgsync_v10 \
    --output-dir data/cache/features_surgsync \
    --sample-only \
    --fp16 \
    --device cuda
```
*Acceptance Criteria:*
- Model loads in $< 10\text{ s}$ without missing keys.
- Peak VRAM allocation is $< 8\text{ GB}$ (well within Quadro RTX 5000's $16\text{ GB}$ capacity).
- Extraction throughput is $\ge 20\text{ FPS}$.
- Generated `.npz` file contains valid float16 features with shape `(N_frames, 1024)`.

---

### Step 5: Full Bulk GPU Feature Extraction (Only After Approval)
```bash
python scripts/extract_surgmotion_surgsync.py \
    --data-dir E:/surgsync_v10 \
    --output-dir data/cache/features_surgsync \
    --batch-size 4 \
    --fp16 \
    --device cuda
```
*Expected Metrics:*
- Total frames extracted: $90,068\text{ frames}$.
- Extraction duration: $\sim 1.0$ to $1.5\text{ hours}$ on Quadro RTX 5000.
- Cache size on disk: **$\sim 184\text{ MB}$ total** (92 `.npz` files).

---

## 4. Lightweight Development Mode (Laptop Workflow)

If testing data loaders, tokenization, or kinematics models locally on the development laptop without downloading $372\text{ GB}$ of video:

```bash
python scripts/download_surgsync_suturing.py --dest-dir data/surgsync_v10 --skip-video
```
This downloads all 92 episode parquet tables (kinematics, timestamps, contacts, annotations) in $< 30\text{ seconds}$ ($\sim 85\text{ MB}$ total), allowing full kinematics and sequence pipeline development on CPU.
