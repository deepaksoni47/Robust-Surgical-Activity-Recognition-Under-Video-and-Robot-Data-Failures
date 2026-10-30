"""scripts/plot_iter34_ablation.py

Publication-grade Visualization Suite for ITER-34b Ablation and Model Contribution Analysis.
Generates vector (PDF) and 300 DPI raster (PNG) figures:
1. fig_iter34_ablation_contributions: Controlled fusion ablations (Unimodal, Late, Concat, Adaptive).
2. fig_iter34_temporal_impact: Frame accuracy vs Sequence Edit Score (MLP vs GRU vs Adaptive).
3. fig_iter34_confusion_matrices: Normalized confusion matrices for Suturing, Knot Tying, Needle Passing.
4. fig_iter34_subject_variance: Per-subject cross-validation performance breakdown (Users B-I).
5. fig_iter34_gesture_imbalance: Gesture support vs Recall/F1 highlighting rare-gesture collapse.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "iter34_ablation_summary.json"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Curated, professional, colorblind-friendly aesthetic
PALETTE = {
    "unimodal_video": "#4A90E2",       # Soft Blue
    "unimodal_kinematics": "#F5A623",  # Amber Orange
    "fixed_late": "#7ED321",           # Vibrant Green
    "feature_concat": "#9013FE",       # Purple
    "adaptive_fusion": "#D0021B",      # Deep Crimson Red
    "mlp": "#9B9B9B",                  # Medium Gray
    "gru": "#417505",                  # Forest Green
}

plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_ablation_summary() -> Dict[str, Any]:
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================================================================
# Figure 1: Controlled Fusion Ablations across Tasks
# ==============================================================================

def plot_fusion_ablation(summary: Dict[str, Any]) -> None:
    data = summary["workstream_a_fusion_ablations"]
    tasks = ["Suturing", "Knot_Tying", "Needle_Passing"]
    metrics = ["accuracy", "macro_f1", "edit_score"]
    metric_labels = ["Accuracy (%)", "Macro-F1 (%)", "Edit Score (%)"]
    
    models = [
        ("unimodal_video", "Video Only (SurgMotion)", PALETTE["unimodal_video"]),
        ("unimodal_kinematics", "Kinematics Only (GRU)", PALETTE["unimodal_kinematics"]),
        ("fixed_late_fusion", "Fixed Late Fusion", PALETTE["fixed_late"]),
        ("feature_concat_fusion", "Concat Fusion", PALETTE["feature_concat"]),
        ("adaptive_reliability_fusion", "Adaptive Fusion (Ours)", PALETTE["adaptive_fusion"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    for i, (m_key, m_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        x = np.arange(len(tasks))
        n_models = len(models)
        width = 0.16

        for j, (m_id, m_name, color) in enumerate(models):
            offsets = x + (j - (n_models - 1) / 2) * width
            vals = [data[t]["models"][m_id][m_key] for t in tasks]
            ax.bar(offsets, vals, width, label=m_name if i == 0 else None, color=color, alpha=0.88, edgecolor="black", linewidth=0.6)

        ax.set_title(m_label, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["Suturing", "Knot Tying", "Needle Passing"])
        ax.set_ylim(0, 85 if m_key != "edit_score" else 80)
        ax.set_ylabel(m_label)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=5, frameon=True)
    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / "fig_iter34_ablation_contributions.pdf")
    fig.savefig(OUTPUT_DIR / "fig_iter34_ablation_contributions.png", dpi=300)
    plt.close(fig)
    print("  [Saved] fig_iter34_ablation_contributions (.pdf / .png)")


# ==============================================================================
# Figure 2: Temporal Modeling Impact (Frame Acc vs Sequence Edit Score)
# ==============================================================================

def plot_temporal_impact(summary: Dict[str, Any]) -> None:
    data = summary["workstream_b_temporal_modeling"]
    tasks = ["Suturing", "Knot_Tying", "Needle_Passing"]
    task_names = ["Suturing", "Knot Tying", "Needle Passing"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    x = np.arange(len(tasks))
    width = 0.22

    # Panel 1: Frame-Level Accuracy (Kinematics MLP vs Kinematics GRU vs Adaptive Fusion)
    mlp_accs = [data[t]["kinematics_gru_vs_mlp"]["accuracy"]["baseline_mean"] for t in tasks]
    gru_accs = [data[t]["kinematics_gru_vs_mlp"]["accuracy"]["treatment_mean"] for t in tasks]
    adapt_accs = [data[t]["adaptive_fusion_vs_framewise_mlp"]["accuracy"]["treatment_mean"] for t in tasks]

    ax1.bar(x - width, mlp_accs, width, label="Framewise MLP (Kinematics)", color=PALETTE["mlp"], edgecolor="black", linewidth=0.6)
    ax1.bar(x, gru_accs, width, label="Sequence GRU (Kinematics)", color=PALETTE["gru"], edgecolor="black", linewidth=0.6)
    ax1.bar(x + width, adapt_accs, width, label="Multimodal Adaptive GRU", color=PALETTE["adaptive_fusion"], edgecolor="black", linewidth=0.6)

    ax1.set_title("Frame-Level Classification (Accuracy)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(task_names)
    ax1.set_ylim(0, 85)
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend(loc="lower left", framealpha=0.9)

    # Panel 2: Sequence Quality (Edit Score)
    mlp_edits = [data[t]["kinematics_gru_vs_mlp"]["edit_score"]["baseline_mean"] for t in tasks]
    gru_edits = [data[t]["kinematics_gru_vs_mlp"]["edit_score"]["treatment_mean"] for t in tasks]
    adapt_edits = [data[t]["adaptive_fusion_vs_framewise_mlp"]["edit_score"]["treatment_mean"] for t in tasks]

    ax2.bar(x - width, mlp_edits, width, label="Framewise MLP", color=PALETTE["mlp"], edgecolor="black", linewidth=0.6)
    ax2.bar(x, gru_edits, width, label="Sequence GRU", color=PALETTE["gru"], edgecolor="black", linewidth=0.6)
    ax2.bar(x + width, adapt_edits, width, label="Multimodal Adaptive GRU", color=PALETTE["adaptive_fusion"], edgecolor="black", linewidth=0.6)

    ax2.set_title("Sequence Quality (Segmental Edit Score)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(task_names)
    ax2.set_ylim(0, 80)
    ax2.set_ylabel("Edit Score (%)")
    ax2.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_iter34_temporal_impact.pdf")
    fig.savefig(OUTPUT_DIR / "fig_iter34_temporal_impact.png", dpi=300)
    plt.close(fig)
    print("  [Saved] fig_iter34_temporal_impact (.pdf / .png)")


# ==============================================================================
# Figure 3: Normalized Task Confusion Matrices
# ==============================================================================

def plot_confusion_matrices(summary: Dict[str, Any]) -> None:
    data = summary["workstream_c_error_and_subject_analysis"]
    tasks = ["Suturing", "Knot_Tying", "Needle_Passing"]
    titles = ["Suturing (11 Classes)", "Knot Tying (7 Classes)", "Needle Passing (11 Classes)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for i, (t, title) in enumerate(zip(tasks, titles)):
        ax = axes[i]
        cm = np.array(data[t]["confusion_matrix"], dtype=np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        norm_cm = np.divide(cm, np.maximum(row_sums, 1), where=row_sums > 0)

        sns.heatmap(
            norm_cm,
            ax=ax,
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            cbar=(i == 2),
            square=True,
            linewidths=0.5,
            linecolor="#E0E0E0",
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Predicted Class")
        if i == 0:
            ax.set_ylabel("Ground Truth Class")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_iter34_confusion_matrices.pdf")
    fig.savefig(OUTPUT_DIR / "fig_iter34_confusion_matrices.png", dpi=300)
    plt.close(fig)
    print("  [Saved] fig_iter34_confusion_matrices (.pdf / .png)")


# ==============================================================================
# Figure 4: Per-Subject Cross-Validation Performance Variance
# ==============================================================================

def plot_subject_variance(summary: Dict[str, Any]) -> None:
    data = summary["workstream_c_error_and_subject_analysis"]
    tasks = ["Suturing", "Knot_Tying", "Needle_Passing"]
    all_users = ["B", "C", "D", "E", "F", "G", "H", "I"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)

    for i, t in enumerate(tasks):
        ax = axes[i]
        sb = data[t]["subject_breakdown"]

        accs = [sb[u]["accuracy"] if u in sb else 0.0 for u in all_users]
        f1s = [sb[u]["macro_f1"] if u in sb else 0.0 for u in all_users]

        x = np.arange(len(all_users))
        width = 0.35

        ax.bar(x - width/2, accs, width, label="Accuracy (%)", color="#4A90E2", alpha=0.9, edgecolor="black", linewidth=0.5)
        ax.bar(x + width/2, f1s, width, label="Macro-F1 (%)", color="#F5A623", alpha=0.9, edgecolor="black", linewidth=0.5)

        ax.set_title(f"{t.replace('_', ' ')}", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(all_users)
        ax.set_xlabel("Held-Out Subject (LOUO Fold)")
        if i == 0:
            ax.set_ylabel("Score (%)")
            ax.legend(loc="lower left")

        # Mark empty folds
        if t == "Needle_Passing":
            ax.text(5, 5, "N/A\n(0 trials)", ha="center", va="bottom", fontsize=8, color="red")

    fig.suptitle("Leave-One-User-Out Generalization Across Held-Out Subjects", y=1.02, fontweight="bold")
    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / "fig_iter34_subject_variance.pdf")
    fig.savefig(OUTPUT_DIR / "fig_iter34_subject_variance.png", dpi=300)
    plt.close(fig)
    print("  [Saved] fig_iter34_subject_variance (.pdf / .png)")


# ==============================================================================
# Figure 5: Gesture Support vs F1-score (Severe Rare-Gesture Collapse)
# ==============================================================================

def plot_gesture_imbalance(summary: Dict[str, Any]) -> None:
    data = summary["workstream_c_error_and_subject_analysis"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    tasks = ["Suturing", "Knot_Tying", "Needle_Passing"]

    for i, t in enumerate(tasks):
        ax = axes[i]
        breakdown = data[t]["per_gesture_breakdown"]

        supports = [d["support"] for d in breakdown.values()]
        f1s = [d["f1_score"] for d in breakdown.values()]
        labels = [g.split()[0] for g in breakdown.keys()]  # Just 'G0', 'G1', etc.
        is_rare = [d["is_rare"] for d in breakdown.values()]

        colors = ["#D0021B" if r else "#4A90E2" for r in is_rare]

        scatter = ax.scatter(supports, f1s, c=colors, s=80, alpha=0.85, edgecolors="black", zorder=3)

        for lx, ly, txt in zip(supports, f1s, labels):
            ax.annotate(txt, (lx, ly), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)

        ax.set_title(t.replace("_", " "), fontweight="bold")
        ax.set_xlabel("Sample Support (Log Scale)")
        ax.set_xscale("symlog")
        ax.set_ylabel("F1 Score (%)")
        ax.set_ylim(-5, 95)

    fig.suptitle("Severe Performance Collapse on Rare Gestures (< 3% Support in Red)", y=1.02, fontweight="bold")
    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / "fig_iter34_gesture_imbalance.pdf")
    fig.savefig(OUTPUT_DIR / "fig_iter34_gesture_imbalance.png", dpi=300)
    plt.close(fig)
    print("  [Saved] fig_iter34_gesture_imbalance (.pdf / .png)")


def main():
    print("Generating publication-ready figures for ITER-34b...")
    summary = load_ablation_summary()
    plot_fusion_ablation(summary)
    plot_temporal_impact(summary)
    plot_confusion_matrices(summary)
    plot_subject_variance(summary)
    plot_gesture_imbalance(summary)
    print("[SUCCESS] All 5 publication figures successfully generated in paper/figures/")


if __name__ == "__main__":
    main()
