# ITER-31 — Continuous-Noise Reliability-Gate Ablation Protocol

**Document Date:** October 28, 2026  
**Status:** Implementation Ready & Unit Tested  
**Primary Benchmark Baseline:** JIGSAWS with Frozen SurgMotion-L Video Features & 76D Kinematics  
**Target Execution Environment:** Mentor GPU (NVIDIA Quadro RTX 5000, 16 GB VRAM)  
**Deliverable Script:** `scripts/train_surgmotion_noise_ablation.py`

---

## 1. Research Motivation & Core Scientific Question

In ITER-29 and ITER-30, our empirical audit established that while the `ReliabilityGate` successfully performs deterministic fallback under total modality absence ($a = [0, 1] \implies \alpha_v \equiv 0.0, \alpha_k \equiv 1.0$), its response to continuous sensor corruption was nearly flat:
- Under test-time visual feature noise ($\sigma_v = 1.0$), the gate's average weights shifted by less than $0.01$ (e.g., from $\sim 0.50$ to $\sim 0.49$).
- **Root Cause Identified:** The ITER-28 training loop regularized the gate using discrete binary modality dropout ($p=0.10$ zero-drop per modality). The gate was **never exposed to noisy continuous signals during training** and was never trained to associate feature variance or dispersion with degraded reliability.

### The Research Question
> Does training the `ReliabilityGate` with continuous Gaussian noise injected into normalized visual features ($\sigma_v \sim \mathcal{U}(0.0, 1.5)$) teach the gating network to dynamically downweight corrupted visual inputs during test-time inference, and does this continuous attenuation improve robustness compared to discrete binary dropout?

---

## 2. Mathematical Formulation & Architecture

### 2.1 Feature Distribution & Noise Scaling
Because visual representations from SurgMotion-L are z-score normalized by `VideoFeatureNormalizer` fitted strictly on training trials, each dimension has zero mean and unit variance:
$$\mathbb{E}[v] \approx 0, \quad \operatorname{Var}(v) \approx 1$$

We define the noise perturbation operator:
$$\tilde{v} = v + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_v^2 \mathbf{I})$$

Where $\sigma_v$ is sampled uniformly at each training batch:
$$\sigma_v \sim \mathcal{U}(\sigma_{\min}, \sigma_{\max}) = \mathcal{U}(0.0, 1.5)$$

- $\sigma_v = 0.0$: pristine feature representation ($\text{SNR} = \infty$).
- $\sigma_v = 0.5$: mild corruption ($\text{SNR} = 4.0$, or $+6 \text{ dB}$).
- $\sigma_v = 1.0$: matched variance ($\text{SNR} = 1.0$, or $0 \text{ dB}$).
- $\sigma_v = 1.5$: severe corruption ($\text{SNR} = 0.44$, or $-3.5 \text{ dB}$).
- $\sigma_v = 2.0$: out-of-distribution severe corruption (used only during evaluation).

### 2.2 Availability Masking vs Noise Injection
Crucially, when noise is injected into $v$, the availability mask remains:
$$\mathbf{a} = [1, 1]$$
The video modality is **present but noisy**. The gating network must rely on its learned representation projections $h_{\text{vis}}$ and $h_{\text{kin}}$ rather than the availability mask to determine fusion weights:
$$[w_{\text{raw}, v}, w_{\text{raw}, k}] = \operatorname{Softplus}(\mathbf{W}_g [h_{\text{vis}}; h_{\text{kin}}; \mathbf{a}] + \mathbf{b}_g)$$
$$\alpha_v = \frac{w_{\text{raw}, v} \cdot a_v}{w_{\text{raw}, v} \cdot a_v + w_{\text{raw}, k} \cdot a_k + \epsilon}$$

Because the task loss is CrossEntropy on $\hat{\mathbf{y}} = \alpha_v \mathbf{y}_{\text{vis}} + \alpha_k \mathbf{y}_{\text{kin}}$, the gradient $\frac{\partial \mathcal{L}}{\partial \alpha_v}$ will penalize assigning high weight to $\mathbf{y}_{\text{vis}}$ when its predictions are noisy, driving $\alpha_v \to 0$ and $\alpha_k \to 1$.

---

## 3. Experimental Controls

To isolate the contribution of continuous-noise training without confounding factors, we establish three controlled training conditions:

| Condition Name | Modality Dropout | Continuous Visual Noise | Description |
|---|---|---|---|
| **`existing`** | $p = 0.10$ video, $p = 0.10$ kin | None ($\sigma_v = 0.0$) | Exact replication of ITER-28 baseline (binary dropout only). |
| **`continuous_noise`** | $p = 0.10$ video, $p = 0.10$ kin | $\sigma_v \sim \mathcal{U}(0.0, 1.5)$ ($p = 0.50$) | Primary ablation: binary dropout + continuous Gaussian noise. |
| **`clean_only`** | None ($p = 0.0$) | None ($\sigma_v = 0.0$) | Negative control: pristine inputs with zero regularization. |

All three conditions use:
- Identical network architecture: 2-layer `VideoGRU`, 2-layer `KinematicsGRU`, `ReliabilityGate` ($d=128$, hidden $d=64$, 258-D input).
- Identical official JIGSAWS LOUO splits (8 folds Suturing, 8 folds Knot Tying, 7 folds Needle Passing).
- Identical optimizer and schedule: AdamW ($\text{lr} = 10^{-3}$, weight decay $10^{-4}$), `ReduceLROnPlateau` (patience 3, factor 0.5), 25 epochs.
- Identical random seed: 42.

---

## 4. Multi-Severity Test-Time Robustness Suite

Each trained model is evaluated across held-out test trials under 9 standardized conditions:

1. **`clean`**: $\sigma_v = 0.0, \sigma_k = 0.0, \mathbf{a} = [1, 1]$.
2. **`video_noise_0.2`**: Mild visual feature noise ($\sigma_v = 0.2$).
3. **`video_noise_0.5`**: Moderate visual feature noise ($\sigma_v = 0.5$).
4. **`video_noise_1.0`**: Standard benchmark visual noise ($\sigma_v = 1.0$).
5. **`video_noise_1.5`**: Upper training limit visual noise ($\sigma_v = 1.5$).
6. **`video_noise_2.0`**: Out-of-distribution unseen noise severity ($\sigma_v = 2.0$).
7. **`kinematics_corrupted`**: Robot kinematics noise ($\sigma_k = 1.0$).
8. **`missing_video`**: Complete camera failure ($\mathbf{a} = [0, 1], v = \mathbf{0}$).
9. **`missing_kinematics`**: Complete robot telemetry failure ($\mathbf{a} = [1, 0], k = \mathbf{0}$).

For each condition, we report:
- Accuracy, Balanced Accuracy, Macro-F1, and Segmental Edit Score.
- Performance degradation $\Delta = \text{Score}_{\text{corrupt}} - \text{Score}_{\text{clean}}$.
- Mean and standard deviation of gate weights $\bar{\alpha}_v$ and $\bar{\alpha}_k$.

---

## 5. Gate Trajectory Logging Protocol

To fulfill requirement A4, `scripts/train_surgmotion_noise_ablation.py` logs time-resolved gate weights for every clip of every test trial:
```json
{
  "fold": 1,
  "task": "Suturing",
  "trial_id": "Suturing_B001",
  "time_idx": 45,
  "center_frame": 720,
  "condition": "video_noise_1.0",
  "noise_std": 1.0,
  "alpha_video": 0.1824,
  "alpha_kinematics": 0.8176,
  "ground_truth": 1,
  "prediction": 1
}
```

The dedicated visualization script `scripts/plot_gate_dynamics.py` generates:
1. `paper/figures/fig_gate_noise_response.pdf`: Grouped response curves comparing $\bar{\alpha}_v(\sigma)$ for `existing` vs `continuous_noise`.
2. `paper/figures/fig_gate_temporal_dynamics.pdf`: Dynamic timeline $\alpha_v(t), \alpha_k(t)$ showing instantaneous attenuation.

---

## 6. Mentor GPU Execution Guide

### Preflight Verification
On the mentor machine, pull the latest commit and verify environment readiness:
```bash
git pull origin main
python scripts/train_surgmotion_noise_ablation.py --preflight-only
```

### Full GPU Ablation Workload
Execute the ablation across all three JIGSAWS tasks on GPU:

```bash
# 1. Primary Ablation: Continuous-Noise Trained Adaptive Fusion
python scripts/train_surgmotion_noise_ablation.py \
  --task all \
  --condition continuous_noise \
  --noise-min 0.0 \
  --noise-max 1.5 \
  --noise-prob 0.5 \
  --sequence-length 8 \
  --epochs 25 \
  --batch-size 32 \
  --device cuda

# 2. Control Baseline: Existing Discrete Binary Dropout
python scripts/train_surgmotion_noise_ablation.py \
  --task all \
  --condition existing \
  --sequence-length 8 \
  --epochs 25 \
  --batch-size 32 \
  --device cuda
```

### Expected Output Artifacts
The mentor GPU run will produce:
- `artifacts/iter31_noise_ablation/surgmotion_ablation_continuous_noise_{task}_results.json`
- `artifacts/iter31_noise_ablation/gate_trajectories_continuous_noise_{task}.json`
- `artifacts/iter31_noise_ablation/surgmotion_ablation_existing_{task}_results.json`
- `artifacts/iter31_noise_ablation/gate_trajectories_existing_{task}.json`

### Verification Checklist Upon Transfer to Laptop
1. Check that all 6 expected JSON files exist and are non-empty.
2. Confirm 23 active folds evaluated per condition (8 Suturing, 8 Knot Tying, 7 Needle Passing).
3. Run `python scripts/plot_gate_dynamics.py --trajectory-file artifacts/iter31_noise_ablation/gate_trajectories_continuous_noise_Suturing.json`.
4. Perform paired statistical significance tests comparing `continuous_noise` vs `existing`.

---

## 7. Limitations & Scientific Nuance

1. **Synthetic Noise vs Optical Corruption:** Feature-space Gaussian perturbations test the network's resilience to representation variance, but do not capture spatial structured occlusions (e.g. cautery smoke, lens fogging). They serve as controlled stress tests.
2. **Subject Sample Size:** With $N=8$ subjects (7 on Needle Passing), effect sizes must be interpreted alongside confidence intervals.
3. **No Premature Claims:** Until the mentor GPU run finishes and results are verified on the laptop, the paper draft will remain strictly unchanged.
