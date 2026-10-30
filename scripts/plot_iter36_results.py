"""scripts/plot_iter36_results.py

Publication-Quality Figure Generation for ITER-36.
Generates publication-ready figures for the Suturing 8-fold LOUO loss function ablation:
1. fig_iter36_loss_comparison.png / .pdf (Accuracy, BalAcc, Macro-F1, Edit Score)
2. fig_iter36_subject_variance.png / .pdf (Per-subject trajectory across Users B-I)
3. fig_iter36_robustness_profiles.png / .pdf (Stress test under corruption & missingness)
4. fig_iter36_imbalance_tradeoff.png / .pdf (Accuracy vs Balanced Accuracy tradeoff)
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "paper/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PALETTE = {
    "ce": "#2b5c8f",            # Deep Steel Blue
    "class_weighted": "#2e7d32", # Emerald Green
    "focal": "#d97706",         # Warm Amber
    "cb_focal": "#c0392b",      # Crimson
}

LABELS = {
    "ce": "Standard CE",
    "class_weighted": "Class-Weighted CE",
    "focal": r"Focal ($\gamma=2.0$)",
    "cb_focal": r"CB-Focal ($\gamma=2, \beta=0.9999$)",
}


def load_integrated_results() -> dict:
    path = PROJECT_ROOT / "artifacts/iter36_integrated_results.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_loss_metric_comparison(data: dict) -> None:
    """Figure 1: 4-Panel comparison across Accuracy, Balanced Acc, Macro-F1, and Edit Score."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=False)
    metrics = [
        ("accuracy", "Frame Accuracy (%)", [45, 90]),
        ("balanced_accuracy", "Balanced Accuracy (%)", [35, 80]),
        ("macro_f1", "Macro-F1 (%)", [25, 65]),
        ("edit_score", "Segmental Edit Score (%)", [45, 85]),
    ]
    conds = ["ce", "class_weighted", "focal", "cb_focal"]
    summary = data["iter35_class_imbalance_analysis"]["metrics_summary"]
    breakdown = data["iter35_class_imbalance_analysis"]["fold_breakdown"]

    for ax, (m_key, title, y_lim) in zip(axes, metrics):
        means = [summary[c][m_key]["mean"] for c in conds]
        stds = [summary[c][m_key]["std"] for c in conds]
        x_pos = np.arange(len(conds))

        # Bars for mean with errorbars
        bars = ax.bar(
            x_pos, means, yerr=stds, capsize=4,
            color=[PALETTE[c] for c in conds], alpha=0.85,
            edgecolor="black", linewidth=0.8, width=0.6,
        )

        # Overlay individual fold points with jitter
        for i, c in enumerate(conds):
            vals = [row[f"{c}_{'acc' if m_key == 'accuracy' else 'bal_acc' if m_key == 'balanced_accuracy' else 'f1' if m_key == 'macro_f1' else 'edit'}"] for row in breakdown]
            jitter = np.random.normal(0, 0.04, size=len(vals))
            ax.scatter(
                np.full_like(vals, x_pos[i]) + jitter, vals,
                color="black", alpha=0.6, s=22, zorder=4, edgecolor="white", linewidth=0.5
            )

        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([LABELS[c] for c in conds], rotation=22, ha="right", fontsize=9)
        ax.set_ylim(y_lim)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Highlight best mean
        best_idx = int(np.argmax(means))
        ax.annotate(
            f"Best: {means[best_idx]:.2f}%",
            xy=(best_idx, means[best_idx] + stds[best_idx] + 1.2),
            xytext=(best_idx, means[best_idx] + stds[best_idx] + 4.0),
            ha="center", fontsize=8.5, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        )

    plt.suptitle("Suturing 8-Fold LOUO Benchmark Across Loss Formulations (ITER-35)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_iter36_loss_comparison.png")
    plt.savefig(FIGURES_DIR / "fig_iter36_loss_comparison.pdf")
    plt.close()
    print("Saved fig_iter36_loss_comparison")


def plot_subject_variance(data: dict) -> None:
    """Figure 2: Per-Subject performance trajectories across Users B-I."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    breakdown = data["iter35_class_imbalance_analysis"]["fold_breakdown"]
    users = [row["user"] for row in breakdown]
    conds = ["ce", "class_weighted", "focal", "cb_focal"]
    x = np.arange(len(users))

    # Panel A: Balanced Accuracy by Subject
    for c in conds:
        vals = [row[f"{c}_bal_acc"] for row in breakdown]
        ax1.plot(x, vals, marker="o", label=LABELS[c], color=PALETTE[c], linewidth=2.0, markersize=6)

    ax1.set_title("Balanced Accuracy Across Held-Out Surgeons", fontweight="bold")
    ax1.set_xlabel("Held-Out Subject (LOUO Fold)")
    ax1.set_ylabel("Balanced Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"User {u}" for u in users])
    ax1.axvspan(4.5, 5.5, color="red", alpha=0.1, label="Novice User G (Collapse)")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel B: Segmental Edit Score by Subject
    for c in conds:
        vals = [row[f"{c}_edit"] for row in breakdown]
        ax2.plot(x, vals, marker="s", label=LABELS[c], color=PALETTE[c], linewidth=2.0, markersize=6)

    ax2.set_title("Segmental Edit Score Across Held-Out Surgeons", fontweight="bold")
    ax2.set_xlabel("Held-Out Subject (LOUO Fold)")
    ax2.set_ylabel("Segmental Edit Score (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"User {u}" for u in users])
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Cross-Subject Generalization Dynamics (8 LOUO Folds)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_iter36_subject_variance.png")
    plt.savefig(FIGURES_DIR / "fig_iter36_subject_variance.pdf")
    plt.close()
    print("Saved fig_iter36_subject_variance")


def plot_robustness_profiles(data: dict) -> None:
    """Figure 3: Stress-testing under corruptions and missing modalities across loss functions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    rob = data["iter35_class_imbalance_analysis"]["robustness_comparison"]
    conds = ["ce", "class_weighted", "focal", "cb_focal"]
    conditions_list = [
        ("clean", "Clean"),
        ("video_corrupted", "Video Corrupted"),
        ("kinematics_corrupted", "Kin Corrupted"),
        ("video_missing", "Video Missing"),
        ("kinematics_missing", "Kin Missing"),
    ]

    x = np.arange(len(conditions_list))
    width = 0.18

    # Panel A: Accuracy under Stress
    for i, c in enumerate(conds):
        accs = [rob[c][cond_key]["accuracy"] for cond_key, _ in conditions_list]
        ax1.bar(x + (i - 1.5) * width, accs, width=width, label=LABELS[c], color=PALETTE[c], edgecolor="black", linewidth=0.6)

    ax1.set_title("Frame Accuracy Under Modal Stress", fontweight="bold")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([name for _, name in conditions_list], rotation=15, ha="right")
    ax1.set_ylim([20, 85])
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel B: Macro-F1 under Stress
    for i, c in enumerate(conds):
        f1s = [rob[c][cond_key]["macro_f1"] for cond_key, _ in conditions_list]
        ax2.bar(x + (i - 1.5) * width, f1s, width=width, label=LABELS[c], color=PALETTE[c], edgecolor="black", linewidth=0.6)

    ax2.set_title("Macro-F1 Under Modal Stress", fontweight="bold")
    ax2.set_ylabel("Macro-F1 (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([name for _, name in conditions_list], rotation=15, ha="right")
    ax2.set_ylim([10, 52])
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Multimodal Robustness Profile Across Loss Formulations", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_iter36_robustness_profiles.png")
    plt.savefig(FIGURES_DIR / "fig_iter36_robustness_profiles.pdf")
    plt.close()
    print("Saved fig_iter36_robustness_profiles")


def plot_imbalance_tradeoff(data: dict) -> None:
    """Figure 4: Accuracy vs Balanced Accuracy tradeoff scatter with fold contours."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    breakdown = data["iter35_class_imbalance_analysis"]["fold_breakdown"]
    summary = data["iter35_class_imbalance_analysis"]["metrics_summary"]
    conds = ["ce", "class_weighted", "focal", "cb_focal"]

    for c in conds:
        accs = [row[f"{c}_acc"] for row in breakdown]
        bal_accs = [row[f"{c}_bal_acc"] for row in breakdown]

        # Scatter individual folds
        ax.scatter(accs, bal_accs, color=PALETTE[c], alpha=0.45, s=40, edgecolor="none")

        # Mean point with large marker
        m_acc = summary[c]["accuracy"]["mean"]
        m_bal = summary[c]["balanced_accuracy"]["mean"]
        s_acc = summary[c]["accuracy"]["std"]
        s_bal = summary[c]["balanced_accuracy"]["std"]

        ax.errorbar(
            m_acc, m_bal, xerr=s_acc / np.sqrt(8), yerr=s_bal / np.sqrt(8),
            fmt="o", color=PALETTE[c], ecolor=PALETTE[c], elinewidth=1.5,
            capsize=4, markersize=10, label=f"{LABELS[c]} (Mean ± SE)",
            markeredgecolor="black", markeredgewidth=1.2, zorder=5
        )

    # Reference diagonal (y=x)
    lims = [45, 88]
    ax.plot(lims, lims, "k--", alpha=0.3, label="Balanced Acc = Accuracy (Zero Imbalance Bias)")

    ax.set_title("Frame Accuracy vs. Balanced Accuracy Pareto Trade-Off", fontweight="bold", pad=12)
    ax.set_xlabel("Overall Frame Accuracy (%) [Favors Dominant Classes G2, G3]")
    ax.set_ylabel("Balanced Accuracy (%) [Treats Rare G9, G10 Equally]")
    ax.set_xlim(lims)
    ax.set_ylim([45, 80])
    ax.legend(loc="lower right", framealpha=0.92, fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_iter36_imbalance_tradeoff.png")
    plt.savefig(FIGURES_DIR / "fig_iter36_imbalance_tradeoff.pdf")
    plt.close()
    print("Saved fig_iter36_imbalance_tradeoff")


def main() -> None:
    data = load_integrated_results()
    plot_loss_metric_comparison(data)
    plot_subject_variance(data)
    plot_robustness_profiles(data)
    plot_imbalance_tradeoff(data)
    print("All ITER-36 publication figures generated successfully in paper/figures/")


if __name__ == "__main__":
    main()
