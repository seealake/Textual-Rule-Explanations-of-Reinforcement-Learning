#!/usr/bin/env python
"""
Figures for the merge-repair study

Generates six figures from the merge-repair results:
    1. fig_repair_ladder — Bar chart showing Macro-F1 improvement across repair stages
    2. fig_soft_support_heatmap — Heatmap of SSC ablation (λ_B × support_mode × safeguard)
  3. fig_support_comparison — Hard vs soft support comparison
  4. fig_repair_boundary_crossing — Boundary crossing rates before/after repair
  5. fig_repair_geometric_distortion — Geometric distortion before/after repair
  6. fig_repair_main_comparison — Main method comparison bar chart

Usage:
    python figures/fig_merge_repairs.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

# Setup path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "figures"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from _style import apply_style, COLORS, COL1, COL2, ENVS, ENV_TAGS, ENV_SHORT, _HAS_LATEX
    apply_style()
except ImportError:
    sns.set_context("paper")
    ENVS = ["MountainCar-v0", "CartPole-v1", "LunarLander-v3"]
    ENV_TAGS = {e: e.replace("-", "_").lower() for e in ENVS}
    ENV_SHORT = {"MountainCar-v0": "MC", "CartPole-v1": "CP", "LunarLander-v3": "LL"}
    COL1, COL2 = 3.5, 7.16

RESULTS_DIR = ROOT / "experiments" / "results"
FIGURES_DIR = ROOT / "figures"

# Extended colors for B experiment methods
B_COLORS = {
    "default_consensus": "#DC3220",  # red (broken)
    "tuned_merge": "#648FFF",           # blue
    "soft_support": "#35A86B",            # green
    "match_only": "#FE6100",         # orange
    "match_hard_support": "#FFB000", # amber
    "match_aggregation": "#785EF0",  # purple
    "full_default": "#DC3220",       # red
    "v2_soft_support": "#35A86B",    # green
    "hard": "#DC3220",               # red
    "soft": "#35A86B",               # green
}


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save(fig, name):
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    png_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    if _HAS_LATEX:
        try:
            import fitz
            doc = fitz.open(pdf_path)
            pix = doc[0].get_pixmap(dpi=300)
            pix.save(str(png_path))
            doc.close()
        except Exception:
            pass
    else:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.pdf/.png")


# ── Figure 1: Repair Ladder (Failure Decomposition stages) ──────────


def fig_repair_ladder():
    """Bar chart of Macro-F1 across 5 ablation stages for each environment."""
    stages = ["match_only", "match_hard_support", "match_aggregation",
              "full_default", "v2_soft_support"]
    stage_labels = ["Match\nOnly", "Match+\nHard Sup.", "Match+\nAggr.",
                    "Full\nDCM", "SSC"]

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.2), sharey=True)

    for ax, env_name in zip(axes, ENVS):
        env_tag = ENV_TAGS[env_name]
        fd = _load_json(RESULTS_DIR / "merge_stages" / env_tag / "failure_decomposition.json")
        if fd is None:
            ax.set_title(ENV_SHORT.get(env_name, env_name))
            continue

        summary = fd.get("summary", {})
        means = [summary.get(s, {}).get("f1", {}).get("mean", 0) for s in stages]
        stds = [summary.get(s, {}).get("f1", {}).get("std", 0) for s in stages]

        colors = [B_COLORS.get(s, "#999999") for s in stages]
        x = np.arange(len(stages))
        ax.bar(x, means, yerr=stds, color=colors, edgecolor="white",
               linewidth=0.5, capsize=2, width=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(stage_labels, fontsize=9)
        ax.set_title(ENV_SHORT.get(env_name, env_name))
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Macro-F1")
    fig.suptitle("Failure Decomposition: 5-Stage Repair Ladder", fontsize=9, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_repair_ladder")


# ── Figure 2: SSC Ablation Heatmap ──────────────────────────────────


def fig_soft_support_heatmap():
    """Heatmap of SSC ablation: λ_B × (support_mode, safeguard)."""
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.5))

    lambda_bs = [0.0, 0.1, 0.2]
    configs = [(sm, sg) for sm in ["hard", "soft"]
               for sg in ["off", "on"]]
    col_labels = [f"{sm}/{sg}" for sm, sg in configs]
    row_labels = [r"$\lambda_B$=" + f"{lb}" for lb in lambda_bs]

    for ax, env_name in zip(axes, ENVS):
        env_tag = ENV_TAGS[env_name]
        summary_path = RESULTS_DIR / "soft_support_sweep" / env_tag / "summary.json"
        data = _load_json(str(summary_path))
        if data is None:
            ax.set_title(ENV_SHORT.get(env_name, env_name))
            continue

        v2_summary = data.get("v2_summary", {})

        matrix = np.zeros((len(lambda_bs), len(configs)))
        for i, lb in enumerate(lambda_bs):
            for j, (sm, sg) in enumerate(configs):
                key = f"lB{lb}_sm{sm}_sg{sg}"
                cell = v2_summary.get(key, {})
                f1 = cell.get("summary", {}).get("F1", {}).get("mean", 0)
                matrix[i, j] = f1

        xlabels = [f"{sm}\n{'sg' if sg=='on' else '—'}" for sm, sg in configs]
        sns.heatmap(matrix, ax=ax, annot=True, fmt=".3f",
                    xticklabels=xlabels,
                    yticklabels=row_labels,
                    cmap="YlOrRd", vmin=0.3, vmax=0.9,
                    cbar=False, linewidths=0.5)
        ax.set_title(ENV_SHORT.get(env_name, env_name))
        ax.set_xlabel("support / safeguard", fontsize=9)

    fig.suptitle(r"SSC Ablation: Macro-F1 ($\lambda_B$ $\times$ support $\times$ safeguard)", fontsize=9, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_soft_support_heatmap")


# ── Figure 3: Support Comparison ────────────────────────────────────


def fig_support_comparison():
    """Grouped bar chart: hard vs soft support filtering."""
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.0), sharey=True)

    metrics = ["f1", "worst_action_recall"]
    metric_labels = ["Macro-F1", "Worst-Action Recall"]

    for ax, env_name in zip(axes, ENVS):
        env_tag = ENV_TAGS[env_name]
        sc = _load_json(RESULTS_DIR / "merge_stages" / env_tag / "support_comparison.json")
        if sc is None:
            ax.set_title(ENV_SHORT.get(env_name, env_name))
            continue

        x = np.arange(len(metrics))
        width = 0.35

        for i, mode in enumerate(["hard", "soft"]):
            summ = sc.get(f"{mode}_summary", {})
            means = [summ.get(m, {}).get("mean", 0) for m in metrics]
            stds = [summ.get(m, {}).get("std", 0) for m in metrics]
            ax.bar(x + (i - 0.5) * width, means, width,
                   yerr=stds, label=mode.capitalize(),
                   color=B_COLORS[mode], edgecolor="white",
                   linewidth=0.5, capsize=2)

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_title(ENV_SHORT.get(env_name, env_name))
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Score")
    axes[-1].legend(fontsize=9)
    fig.suptitle("Support Filtering: Hard vs Soft", fontsize=9, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_support_comparison")


# ── Figure 4: Boundary Crossing Before/After ────────────────────────


def fig_repair_boundary_crossing():
    """Grouped bar chart: boundary crossing rates across methods."""
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.0), sharey=True)
    methods = ["default_consensus", "tuned_merge", "soft_support"]
    method_labels = ["DCM", "MTC", "SSC"]

    for ax, env_name in zip(axes, ENVS):
        env_tag = ENV_TAGS[env_name]
        bc = _load_json(RESULTS_DIR / "merge_stages" / env_tag / "boundary_crossing.json")
        if bc is None:
            ax.set_title(ENV_SHORT.get(env_name, env_name))
            continue

        means = []
        stds = []
        colors = []
        for m in methods:
            summ = bc.get(f"{m}_summary", {})
            means.append(summ.get("mergeable_crossing_pct", {}).get("mean", 0))
            stds.append(summ.get("mergeable_crossing_pct", {}).get("std", 0))
            colors.append(B_COLORS.get(m, "#999999"))

        x = np.arange(len(methods))
        ax.bar(x, means, yerr=stds, color=colors, edgecolor="white",
               linewidth=0.5, capsize=2, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, fontsize=9)
        ax.set_title(ENV_SHORT.get(env_name, env_name))

    axes[0].set_ylabel("Crossing Rate")
    fig.suptitle("Boundary Crossing: Before vs After Repair", fontsize=9, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_repair_boundary_crossing")


# ── Figure 5: Geometric Distortion Before/After ─────────────────────


def fig_repair_geometric_distortion():
    """Grouped bar chart: geometric distortion metrics across methods."""
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.0), sharey=True)
    methods = ["default_consensus", "tuned_merge", "soft_support"]
    method_labels = ["DCM", "MTC", "SSC"]

    for ax, env_name in zip(axes, ENVS):
        env_tag = ENV_TAGS[env_name]
        gd = _load_json(RESULTS_DIR / "merge_stages" / env_tag / "geometric_distortion.json")
        if gd is None:
            ax.set_title(ENV_SHORT.get(env_name, env_name))
            continue

        means = []
        stds = []
        colors = []
        for m in methods:
            summ = gd.get(f"{m}_summary", {})
            means.append(summ.get("failed_merge_frac", {}).get("mean", 0))
            stds.append(summ.get("failed_merge_frac", {}).get("std", 0))
            colors.append(B_COLORS.get(m, "#999999"))

        x = np.arange(len(methods))
        ax.bar(x, means, yerr=stds, color=colors, edgecolor="white",
               linewidth=0.5, capsize=2, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, fontsize=9)
        ax.set_title(ENV_SHORT.get(env_name, env_name))

    axes[0].set_ylabel("Failed Merge Fraction")
    fig.suptitle("Geometric Distortion: Before vs After Repair", fontsize=9, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_repair_geometric_distortion")


# ── Figure 6: Main Method Comparison ────────────────────────────────


def fig_repair_main_comparison():
    """Grouped bar chart: main comparison across all methods and envs."""
    methods = ["default_consensus", "tuned_merge", "soft_support"]
    method_labels = ["DCM", "MTC", "SSC"]

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.5))

    # Macro-F1 subplot
    ax = axes[0]
    x = np.arange(len(ENVS))
    width = 0.25
    for i, (m, label) in enumerate(zip(methods, method_labels)):
        means = []
        stds = []
        for env_name in ENVS:
            env_tag = ENV_TAGS[env_name]
            rl = _load_json(RESULTS_DIR / "merge_stages" / env_tag / "repair_ladder.json")
            if rl and "summary" in rl and m in rl["summary"]:
                means.append(rl["summary"][m].get("f1", {}).get("mean", 0))
                stds.append(rl["summary"][m].get("f1", {}).get("std", 0))
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x + (i - 1) * width, means, width, yerr=stds,
               label=label, color=B_COLORS.get(m, "#999999"),
               edgecolor="white", linewidth=0.5, capsize=2)

    ax.set_ylabel("Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_SHORT.get(e, e) for e in ENVS])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Fidelity (Macro-F1)")

    # Return subplot
    ax = axes[1]
    for i, (m, label) in enumerate(zip(methods, method_labels)):
        means = []
        stds = []
        for env_name in ENVS:
            env_tag = ENV_TAGS[env_name]
            rl = _load_json(RESULTS_DIR / "merge_stages" / env_tag / "repair_ladder.json")
            if rl and "summary" in rl and m in rl["summary"]:
                means.append(rl["summary"][m].get("E_CR", {}).get("mean", 0))
                stds.append(rl["summary"][m].get("E_CR", {}).get("std", 0))
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x + (i - 1) * width, means, width, yerr=stds,
               label=label, color=B_COLORS.get(m, "#999999"),
               edgecolor="white", linewidth=0.5, capsize=2)

    ax.set_ylabel("Return")
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_SHORT.get(e, e) for e in ENVS])
    ax.set_title("Deployment Return")

    fig.suptitle("Main Method Comparison", fontsize=9, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_repair_main_comparison")


# ── Main ────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  MERGE-REPAIR FIGURES")
    print("=" * 60)

    fig_repair_ladder()
    fig_soft_support_heatmap()
    fig_support_comparison()
    fig_repair_boundary_crossing()
    fig_repair_geometric_distortion()
    fig_repair_main_comparison()

    print("\n" + "=" * 60)
    print("  Figures complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
