"""scripts/generate_iter32_figures.py

Generates publication-quality figures from the verified ITER-31 mentor GPU ablation results:
1. paper/figures/fig_iter32_real_gate_noise_response.pdf / .png:
   Empirical gate weight alpha_v across continuous noise levels sigma in [0.0, 2.0].
   Directly demonstrates the empirical finding: gate weights are flat (no active continuous downweighting).
2. paper/figures/fig_iter32_real_gate_temporal_dynamics.pdf / .png:
   Frame-by-frame alpha_v(t) on real held-out trial under Clean vs Noise vs Masked fallback.
3. paper/figures/fig_iter32_real_ablation_performance.pdf / .png:
   Per-task clean performance and robustness retention comparing continuous_noise vs existing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "iter31_noise_ablation"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "figure.autolayout": True,
})

COLORS = {
    "continuous_noise": "#1b9e77", # Emerald teal
    "existing": "#7570b3",         # Purple
    "kinematics": "#d95f02",       # Burnt orange
    "video": "#2b5c8f",            # Steel blue
    "suturing": "#2b5c8f",
    "knot_tying": "#d95f02",
    "needle_passing": "#7570b3",
}


def load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_real_gate_noise_response() -> None:
    """Plot mean alpha_v vs noise severity across all 3 tasks for both models."""
    sigmas = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]
    tasks = ["Suturing", "Knot_Tying", "Needle_Passing"]
    task_labels = ["Suturing", "Knot Tying", "Needle Passing"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, t, t_lbl in zip(axes, tasks, task_labels):
        # Load trajectory summaries from results summary
        summary = load_json(PROJECT_ROOT / "artifacts" / "iter32_results_summary.json")
        rob_t = summary["robustness_comparison"][t]

        cond_keys = ["clean", "video_noise_0.2", "video_noise_0.5", "video_noise_1.0", "video_noise_1.5", "video_noise_2.0"]

        alpha_v_cont = [rob_t[k]["continuous_alpha_v"] for k in cond_keys]
        alpha_v_exist = [rob_t[k]["existing_alpha_v"] for k in cond_keys]

        ax.plot(sigmas, alpha_v_cont, "o-", color=COLORS["continuous_noise"], label="Continuous-Noise Trained")
        ax.plot(sigmas, alpha_v_exist, "s--", color=COLORS["existing"], label="Existing Baseline (Binary Dropout)")

        ax.set_title(f"{t_lbl}")
        ax.set_xlabel(r"Visual Noise Severity $\sigma_v$")
        ax.set_ylim(-0.02, 0.35)
        if ax == axes[0]:
            ax.set_ylabel(r"Mean Video Gate Weight $\bar{\alpha}_v$")
            ax.legend(loc="upper right", framealpha=0.9)

        # Highlight flat dynamic
        delta_c = (alpha_v_cont[-1] - alpha_v_cont[0]) * 100
        delta_e = (alpha_v_exist[-1] - alpha_v_exist[0]) * 100
        ax.text(
            0.05, 0.08,
            f"$\\Delta \\alpha_v$ (Cont): {delta_c:+.2f}%\n$\\Delta \\alpha_v$ (Exist): {delta_e:+.2f}%",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#ccc"),
        )

    fig.suptitle(r"Empirical Gate Weight Response to Continuous Visual Feature Noise $\sigma_v \in [0.0, 2.0]$", y=1.02)
    plt.tight_layout()

    pdf_out = FIGURES_DIR / "fig_iter32_real_gate_noise_response.pdf"
    png_out = FIGURES_DIR / "fig_iter32_real_gate_noise_response.png"
    plt.savefig(pdf_out, dpi=300)
    plt.savefig(png_out, dpi=300)
    plt.close()
    print(f"Saved {pdf_out} and {png_out}")


def plot_real_gate_temporal_dynamics() -> None:
    """Plot real frame-by-frame alpha_v(t) and alpha_k(t) for a held-out test trial."""
    # Load real Suturing trajectory data
    traj_path = ARTIFACTS_DIR / "gate_trajectories_continuous_noise_Suturing.json"
    with open(traj_path, "r", encoding="utf-8") as f:
        tdata = json.load(f)

    records = tdata["records"]
    # Select first trial in fold 1 (User B)
    clean_recs = [r for r in records if r["fold"] == 1 and r["condition"] == "clean"]
    trial_id = clean_recs[0]["trial_id"]
    clean_trial = [r for r in clean_recs if r["trial_id"] == trial_id]

    noise_recs = [r for r in records if r["fold"] == 1 and r["condition"] == "video_noise_1.0" and r["trial_id"] == trial_id]
    mask_recs = [r for r in records if r["fold"] == 1 and r["condition"] == "missing_video" and r["trial_id"] == trial_id]

    frames = [r["center_frame"] for r in clean_trial]
    alpha_v_clean = [r["alpha_video"] for r in clean_trial]
    alpha_k_clean = [r["alpha_kinematics"] for r in clean_trial]

    alpha_v_noise = [r["alpha_video"] for r in noise_recs]
    alpha_k_noise = [r["alpha_kinematics"] for r in noise_recs]

    alpha_v_mask = [r["alpha_video"] for r in mask_recs]
    alpha_k_mask = [r["alpha_kinematics"] for r in mask_recs]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 7), sharex=True, sharey=True)

    # Panel 1: Clean
    ax1.plot(frames, alpha_k_clean, color=COLORS["kinematics"], label=r"Kinematics $\alpha_k$")
    ax1.plot(frames, alpha_v_clean, color=COLORS["video"], label=r"Video $\alpha_v$")
    ax1.set_title(f"Trial {trial_id} (Held-out User B) — Clean Conditions ($\sigma_v = 0.0$)")
    ax1.set_ylabel("Gate Weight")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper right")

    # Panel 2: Continuous Noise sigma = 1.0
    ax2.plot(frames, alpha_k_noise, color=COLORS["kinematics"], label=r"Kinematics $\alpha_k$")
    ax2.plot(frames, alpha_v_noise, color=COLORS["video"], label=r"Video $\alpha_v$")
    ax2.set_title(r"Continuous Noise Condition ($\sigma_v = 1.0$) — Weight trajectory is unchanged ($\alpha_v$ remains small)")
    ax2.set_ylabel("Gate Weight")
    ax2.legend(loc="upper right")

    # Panel 3: Missing Video (Mask = [0, 1])
    ax3.plot(frames, alpha_k_mask, color=COLORS["kinematics"], label=r"Kinematics $\alpha_k$")
    ax3.plot(frames, alpha_v_mask, color=COLORS["video"], label=r"Video $\alpha_v$")
    ax3.set_title(r"Missing Video Condition ($a = [0, 1]$) — Deterministic Fallback ($\alpha_k \equiv 1.0, \alpha_v \equiv 0.0$)")
    ax3.set_xlabel("Video Frame Index")
    ax3.set_ylabel("Gate Weight")
    ax3.legend(loc="upper right")

    plt.tight_layout()
    pdf_out = FIGURES_DIR / "fig_iter32_real_gate_temporal_dynamics.pdf"
    png_out = FIGURES_DIR / "fig_iter32_real_gate_temporal_dynamics.png"
    plt.savefig(pdf_out, dpi=300)
    plt.savefig(png_out, dpi=300)
    plt.close()
    print(f"Saved {pdf_out} and {png_out}")


def plot_real_ablation_performance() -> None:
    """Plot clean performance and degradation comparisons between conditions."""
    summary = load_json(PROJECT_ROOT / "artifacts" / "iter32_results_summary.json")
    clean = summary["clean_comparison"]

    tasks = ["Suturing", "Knot_Tying", "Needle_Passing"]
    task_labels = ["Suturing", "Knot Tying", "Needle Passing"]

    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "edit_score"]
    metric_labels = ["Accuracy (%)", "Balanced Acc (%)", "Macro-F1 (%)", "Edit Score (%)"]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), sharey=True)

    x = np.arange(len(tasks))
    width = 0.35

    for ax, m, m_lbl in zip(axes, metrics, metric_labels):
        means_c = [clean[t][m]["mean_x"] for t in tasks]
        stds_c = [clean[t][m]["std_x"] for t in tasks]
        means_e = [clean[t][m]["mean_y"] for t in tasks]
        stds_e = [clean[t][m]["std_y"] for t in tasks]

        rects1 = ax.bar(x - width / 2, means_c, width, yerr=stds_c, capsize=4, label="Continuous-Noise", color=COLORS["continuous_noise"])
        rects2 = ax.bar(x + width / 2, means_e, width, yerr=stds_e, capsize=4, label="Existing (Dropout)", color=COLORS["existing"])

        ax.set_title(m_lbl)
        ax.set_xticks(x)
        ax.set_xticklabels(task_labels)
        ax.set_ylim(0, 100)

        # Annotate p-values
        for i, t in enumerate(tasks):
            p_val = clean[t][m]["p_value_paired_t"]
            diff = clean[t][m]["diff_mean"]
            sign = "+" if diff >= 0 else ""
            ax.text(
                i, max(means_c[i] + stds_c[i], means_e[i] + stds_e[i]) + 3,
                f"{sign}{diff:.1f}\n$p$={p_val:.2f}",
                ha="center", fontsize=8,
            )

        if ax == axes[0]:
            ax.set_ylabel("Score (%)")
            ax.legend(loc="lower right")

    fig.suptitle("Clean Performance Comparison: Continuous-Noise Trained vs Existing Baseline (Paired t-test)", y=1.02)
    plt.tight_layout()

    pdf_out = FIGURES_DIR / "fig_iter32_real_ablation_performance.pdf"
    png_out = FIGURES_DIR / "fig_iter32_real_ablation_performance.png"
    plt.savefig(pdf_out, dpi=300)
    plt.savefig(png_out, dpi=300)
    plt.close()
    print(f"Saved {pdf_out} and {png_out}")


def main() -> None:
    plot_real_gate_noise_response()
    plot_real_gate_temporal_dynamics()
    plot_real_ablation_performance()


if __name__ == "__main__":
    main()
