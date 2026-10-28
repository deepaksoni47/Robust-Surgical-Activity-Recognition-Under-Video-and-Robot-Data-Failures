# ITER-34 — SurgSync Acquisition and Extraction Guide

**Document Date:** October 28, 2026  
**Status:** Pipeline Prepared & Dual-Path Verified; Ready for Mentor Machine Execution  
**Target Dataset:** SurgSync v1.0.0 (`jackzhy/surgsync`)  
**Scripts:**
- Downloader: [`scripts/download_surgsync.py`](file:///D:/miniproject-5thsem/scripts/download_surgsync.py) / [`scripts/download_surgsync_suturing.py`](file:///D:/miniproject-5thsem/scripts/download_surgsync_suturing.py)
- Extractor: [`scripts/extract_surgmotion_surgsync.py`](file:///D:/miniproject-5thsem/scripts/extract_surgmotion_surgsync.py)
- Test Suite: [`tests/test_surgsync_acquisition.py`](file:///D:/miniproject-5thsem/tests/test_surgsync_acquisition.py) (9/9 tests pass)

---

## 1. Executive Summary & Dual Acquisition Paths

The complete SurgSync v1.0 release is **$714.35\text{ GB}$** on disk across 11,745 files (205 episodes across 6 tasks). Because raw video streams are encoded in lossless FFV1 at $1920 \times 1080$, video accounts for **$99.98\%$ ($714.19\text{ GB}$)** of the archive.

Depending on available storage on the mentor workstation tomorrow, the pipeline supports **two execution scenarios**:

| Parameter | **Path A: Suturing Subset Only** | **Path B: Full SurgSync Dataset (All Tasks)** |
|:---|:---|:---|
| **Scope** | `single_interrupted_stitch` (92 episodes, 90,068 frames, 3.71 h) | All 6 tasks: suturing, dissection, peg transfer, tissue manipulation (205 episodes) |
| **Download Size** | **$\sim 372.41\text{ GB}$** ($372.33\text{ GB}$ video + $85\text{ MB}$ parquets) | **$\sim 714.35\text{ GB}$** ($714.19\text{ GB}$ video + $160\text{ MB}$ parquets) |
| **Free Space Preflight** | **$\ge 450.0\text{ GB}$** required ($77.6\text{ GB}$ safety buffer) | **$\ge 850.0\text{ GB}$** required ($135.6\text{ GB}$ safety buffer) |
| **Est. Download Time** | $\sim 1.8\text{ to } 3.5\text{ hours}$ (250–500 Mbps) | $\sim 3.5\text{ to } 6.5\text{ hours}$ (250–500 Mbps) |
| **GPU Extraction Target** | 92 suturing episodes ($\sim 1.0 - 1.5\text{ hours}$ on RTX 5000) | 92 suturing episodes first ($\sim 1.0\text{ h}$) or all 205 episodes ($\sim 3.0\text{ h}$) |
| **Feature Cache Size** | **$\sim 184\text{ MB}$** `.npz` files | **$\sim 410\text{ MB}$** `.npz` files |
| **Mentor Machine Feasibility** | **Easily passes** on $\ge 500\text{ GB}$ drive | **Easily passes** on mentor $\ge 3\text{ TB}$ workstation drive |

---

## 2. Safety Features & Resumability

The acquisition pipeline [`scripts/download_surgsync.py`](file:///D:/miniproject-5thsem/scripts/download_surgsync.py) guarantees:
1. **Automated Free-Space Preflight:** Queries `shutil.disk_usage` on the target volume before initiating network transfers. If free space is less than the required threshold ($450\text{ GB}$ for suturing, $850\text{ GB}$ for all tasks, $1-2\text{ GB}$ for `--skip-video`), it aborts with exit code 1 to protect the host machine from disk exhaustion.
2. **Resumable Transfers:** Leverages `huggingface_hub.snapshot_download` with `resume_download=True`. If network connection drops or the process is halted, re-running the script resumes from the exact byte offset without re-downloading completed files.
3. **Selective Task Ingestion:** When `--task suturing` is passed, `allow_patterns` restricts ingestion strictly to `meta/*` and `single_interrupted_stitch/*`. When `--task all` is passed, the full repository is synchronized.
4. **Post-Download Integrity Verification:** Scans downloaded files against the official `meta/manifest.json`, verifying file existence, byte size, and optional SHA-256 hash.
5. **Git Isolation:** `.gitignore` excludes all `data/`, `*.mkv`, `*.npz`, `*.parquet`, and `*.h5` files, guaranteeing that large binaries cannot be staged into Git.

---

## 3. Mentor Machine Runbook — Scenario A: Suturing Only (~372 GB)

*Use this path if disk space is below $850\text{ GB}$ or if you wish to minimize bandwidth and extract the Suturing pilot immediately.*

```bash
# 1. Pull the approved commit
git pull origin main

# 2. Storage Preflight (verifies >= 450 GB free space on target drive)
python scripts/download_surgsync.py --dest-dir E:/surgsync_v10 --task suturing --preflight-only

# 3. Resumable Suturing Download (~372 GB)
python scripts/download_surgsync.py --dest-dir E:/surgsync_v10 --task suturing --min-free-gb 450.0 --max-workers 8

# 4. Integrity Verification against manifest
python scripts/download_surgsync.py --dest-dir E:/surgsync_v10 --task suturing --verify-only

# 5. Quick GPU Sample Test (1 episode, verifies VRAM < 8 GB & FPS >= 20)
python scripts/extract_surgmotion_surgsync.py --data-dir E:/surgsync_v10 --task single_interrupted_stitch --sample-only --fp16 --device cuda

# 6. Full Suturing Feature Extraction (92 episodes, ~1-1.5 h, creates ~184 MB in cache)
python scripts/extract_surgmotion_surgsync.py --data-dir E:/surgsync_v10 --output-dir data/cache/features_surgsync --task single_interrupted_stitch --batch-size 4 --fp16 --device cuda
```

---

## 4. Mentor Machine Runbook — Scenario B: Full Dataset Execution (~714 GB)

*Use this path if the mentor machine has ample free space ($\ge 850\text{ GB}$ free, e.g. on the 3 TB drive).*

```bash
# 1. Pull the approved commit
git pull origin main

# 2. Storage Preflight (verifies >= 850 GB free space on target drive)
python scripts/download_surgsync.py --dest-dir E:/surgsync_v10 --task all --preflight-only

# 3. Resumable Full Dataset Download (~714 GB, all 205 episodes across 6 tasks)
python scripts/download_surgsync.py --dest-dir E:/surgsync_v10 --task all --min-free-gb 850.0 --max-workers 8

# 4. Integrity Verification of all 11,745 files against manifest
python scripts/download_surgsync.py --dest-dir E:/surgsync_v10 --task all --verify-only

# 5. Quick GPU Sample Test (1 episode, verifies VRAM < 8 GB & FPS >= 20)
python scripts/extract_surgmotion_surgsync.py --data-dir E:/surgsync_v10 --task single_interrupted_stitch --sample-only --fp16 --device cuda

# 6. Feature Extraction:
# Option 6a (Recommended): Extract Suturing first (92 episodes, ~1-1.5 h, produces 184 MB features)
python scripts/extract_surgmotion_surgsync.py --data-dir E:/surgsync_v10 --output-dir data/cache/features_surgsync --task single_interrupted_stitch --batch-size 4 --fp16 --device cuda

# Option 6b: Bulk Extract ALL tasks across the entire dataset (205 episodes, ~3-4 h, produces ~410 MB features)
python scripts/extract_surgmotion_surgsync.py --data-dir E:/surgsync_v10 --output-dir data/cache/features_surgsync --task all --batch-size 4 --fp16 --device cuda
```

---

## 5. Lightweight Development Mode (Laptop Workflow)

If testing data loaders, tokenization, or kinematics models locally on the development laptop without downloading hundreds of gigabytes of video:

```bash
python scripts/download_surgsync.py --dest-dir data/surgsync_v10 --task suturing --skip-video
```
This downloads all 92 episode parquet tables (kinematics, timestamps, contacts, annotations) in $< 30\text{ seconds}$ ($\sim 85\text{ MB}$ total), allowing full kinematics and sequence pipeline development on CPU.

