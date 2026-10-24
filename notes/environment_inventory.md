# Team Hardware & Environment Inventory

This document tracks the hardware specifications, operating systems, compute constraints, and execution strategy for both team members collaborating on the research project.

---

## Member 1 (Primary Machine — Current Workspace)
- **Device Model:** Acer Gaming / High-Performance Laptop
- **Processor:** 12th Gen Intel(R) Core(TM) i5-12450H (8 physical cores: 4 P-cores + 4 E-cores, 12 logical threads, up to 4.40 GHz)
- **Graphics:** Intel(R) UHD Graphics (Integrated, no local NVIDIA CUDA accelerator)
- **RAM:** 16.0 GB (~15.74 GB physical memory usable)
- **Storage:** 
  - `D:` Drive: 138 GB Total (~22.71 GB Free) — Active Project Root (`d:\miniproject-5thsem`)
  - `C:` Drive: 336.6 GB Total (~92.71 GB Free)
- **Operating System:** Windows 11 (64-bit)
- **Python Runtime:** Python 3.10.11 (Managed via workspace `.venv`)

---

## Member 2 (Collaborator Machine)
- **Device Model:** HP Pavilion Plus 14-ew1082tu (AI-Enhanced Thin & Light PC)
- **Processor:** Intel(R) Core(TM) Ultra 5 125H (14 cores / 18 threads: 4 P-cores, 8 E-cores, 2 Low Power E-cores, up to 4.5 GHz boost, dedicated Intel AI Boost NPU)
- **Graphics:** Intel(R) Arc(TM) Graphics (Integrated)
- **RAM:** 16 GB LPDDR5x high-speed dual-channel RAM
- **Storage:** 512 GB PCIe Gen4 NVMe M.2 SSD
- **Display:** 14-inch WQXGA (2560 x 1600 pixels), anti-glare, 300 nits brightness, 48–120Hz variable refresh rate, Eyesafe certified
- **Operating System:** Windows 11 Home (64-bit)
- **Camera & Battery:** 5MP IR camera with Windows Hello support; 4-cell 68 Wh Li-ion battery supporting HP Fast Charge (50% in 30 min)

---

## Hardware Synthesis & Compute Strategy

1. **Local Compute (CPU / Memory):**
   - Both machines run 64-bit Windows 11 with 16 GB high-speed RAM and modern multi-core Intel CPUs (12th Gen i5 and Core Ultra 5 Series 1).
   - Neither machine features a dedicated NVIDIA CUDA discrete GPU.
   - **Local Work:** All data parsing, manifest building, 76D kinematic normalization, temporal GRU/TCN training, fusion models, and evaluation runs are fully optimized for local CPU/memory efficiency and run in seconds to minutes.

2. **Visual Feature Extraction (SurgMotion Foundation Model):**
   - SurgMotion V-JEPA foundation model inference on 103 full videos is computationally heavy and requires CUDA tensor acceleration.
   - **Cloud GPU Execution:** A dedicated Google Colab / remote GPU script (`scripts/extract_surgmotion_colab.py` or `.ipynb`) will execute the frozen visual encoder and export lightweight feature embeddings (`.npz` files).
   - **Feature Caching:** The resulting cached visual representations are stored in `data/cache/features/`, allowing both team members to execute, train, and benchmark temporal fusion heads locally without needing expensive local GPUs.

3. **Cross-Platform Reproducibility:**
   - Both members will share identical split files, random seeds (`seed=42`), config YAML files, and test suites.
   - Dependency files (`requirements.txt` / `pyproject.toml`) and environment scripts will guarantee identical package versions.
