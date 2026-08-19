#!/usr/bin/env python
"""External validity: MiniGrid + PPO.

Grouped bar chart comparing 4 methods (CBS, RV, DCM, SSC)
on Macro-F1, BRA, GRS-TA, and worst-action recall for MiniGrid-Dynamic-Obstacles-8x8-v0.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from figures._style import apply_style, savefig, load_results, COL2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

METHOD_COLORS = {
    "CBS":              "#648FFF",  # blue
    "B3-vote":          "#35A86B",  # green
    "Consensus_default": "#9B59B6",  # purple
    "V2_soft_support":  "#FE6100",  # orange
}
METHOD_LABELS = {
    "CBS":              "CBS",
    "B3-vote":          "RV",
    "Consensus_default": "DCM",
    "V2_soft_support":  "SSC",
}
METHOD_ORDER = ["CBS", "B3-vote", "Consensus_default", "V2_soft_support"]

METRICS = [
    ("f1_mean", "Macro-F1"),
    ("BRA", "BRA"),
    ("GRS_TA", "GRS-TA"),
    ("worst_recall_mean", "Worst-Action Recall"),
]


def main():
    apply_style()

    d = load_results("minigrid_dynamic_obstacles_8x8_v0", "external_validity.json")
    if d is None:
        print("  SKIP: no external_validity.json for MiniGrid")
        return

    summary = d["cross_seed_summary"]

    fig, ax = plt.subplots(1, 1, figsize=(COL2 * 0.65, 2.8))

    n_metrics = len(METRICS)
    n_methods = len(METHOD_ORDER)
    bar_width = 0.17
    x_base = np.arange(n_metrics)

    for m_idx, method in enumerate(METHOD_ORDER):
        if method not in summary:
            continue
        s = summary[method]
        means = [s[mk]["mean"] for mk, _ in METRICS]
        stds = [s[mk]["std"] for mk, _ in METRICS]
        offset = (m_idx - (n_methods - 1) / 2) * bar_width

        ax.bar(x_base + offset, means, bar_width,
               yerr=stds,
               color=METHOD_COLORS[method], alpha=0.85,
               edgecolor="white", linewidth=0.5,
               error_kw=dict(lw=0.8, capsize=2, capthick=0.8),
               label=METHOD_LABELS[method])

    ax.set_xticks(x_base)
    ax.set_xticklabels([label for _, label in METRICS], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title("MiniGrid-Dynamic-Obstacles-8x8-v0 (PPO)",
                 fontsize=9, fontweight="bold")

    # Add rule count annotation
    ax.text(0.98, 0.98,
            "Rules: " + ", ".join(
                f"{METHOD_LABELS[m]}={summary[m]['n_rules_mean']['mean']:.0f}"
                for m in METHOD_ORDER if m in summary),
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            color="#666", style="italic")

    ax.legend(loc="upper center", ncol=2, fontsize=9,
              bbox_to_anchor=(0.5, -0.10), framealpha=0.9)

    plt.tight_layout()
    savefig(fig, "fig_minigrid_transfer")


if __name__ == "__main__":
    main()
