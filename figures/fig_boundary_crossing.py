#!/usr/bin/env python
"""Boundary-crossing analysis.

Redesigned: single focused panel showing boundary crossing rate
(the metric cited in the main text) with midpoint diagnostics
as compact secondary annotations.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from figures._style import apply_style, savefig, load_results, COL1
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent

ENVS = [
    ("mountaincar_v0", "MC"),
    ("cartpole_v1", "CP"),
    ("lunarlander_v3", "LL"),
]

MERGE_COLOR = "#648FFF"      # blue
NON_MERGE_COLOR = "#FE6100"  # orange


def main():
    apply_style()

    all_data = {}
    for env_tag, _ in ENVS:
        d = load_results(env_tag, "boundary_crossing.json")
        if d is None:
            print(f"  SKIP: no boundary_crossing.json for {env_tag}")
            continue
        all_data[env_tag] = d

    fig, axes = plt.subplots(2, 1, figsize=(COL1, 5.0),
                              gridspec_kw={"height_ratios": [1.05, 1.0]})

    # ── Panel (a): Boundary crossing rate (the key metric) ──
    ax = axes[0]
    merge_vals, nonmerge_vals, env_labels = [], [], []
    for env_tag, env_short in ENVS:
        summary = all_data[env_tag]["summary"]
        merge_vals.append(summary["mergeable"]["mean_boundary_crossing_rate"])
        nonmerge_vals.append(summary["non_mergeable"]["mean_boundary_crossing_rate"])
        env_labels.append(env_short)

    x = np.arange(len(env_labels))
    bar_width = 0.32

    bars1 = ax.bar(x - bar_width / 2, merge_vals, bar_width,
                    color=MERGE_COLOR, alpha=0.85,
                    edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + bar_width / 2, nonmerge_vals, bar_width,
                    color=NON_MERGE_COLOR, alpha=0.85,
                    edgecolor="white", linewidth=0.5)

    for bar_group, vals in [(bars1, merge_vals), (bars2, nonmerge_vals)]:
        for bar, val in zip(bar_group, vals):
            if val > 0.15:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.0%}", ha="center", va="bottom", fontsize=10,
                        rotation=0, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(env_labels)
    ax.set_ylabel("Boundary crossing rate")
    ax.set_ylim(0, 1.15)
    ax.set_title("(a) Crossing rate", fontweight="bold")

    merge_patch = mpatches.Patch(color=MERGE_COLOR, alpha=0.85, label="Mergeable")
    nonmerge_patch = mpatches.Patch(color=NON_MERGE_COLOR, alpha=0.85, label="Non-mergeable")
    ax.legend(handles=[merge_patch, nonmerge_patch], loc="upper left",
              fontsize=9, framealpha=0.9)

    # ── Panel (b): Midpoint diagnostics (compact dot plot) ──
    ax2 = axes[1]
    metrics = [
        ("mean_midpoint_mismatch", "Midpoint\nmismatch"),
        ("mean_midpoint_low_density", "Low-density\nrate"),
    ]

    x2 = np.arange(len(ENVS))
    offsets = [-0.15, 0.15]
    markers = ["o", "D"]
    colors_m = ["#2166AC", "#7A3A10"]

    for m_idx, (metric_key, metric_label) in enumerate(metrics):
        merge_v = [all_data[et]["summary"]["mergeable"].get(metric_key, 0) for et, _ in ENVS]
        nonmerge_v = [all_data[et]["summary"]["non_mergeable"].get(metric_key, 0) for et, _ in ENVS]

        ax2.scatter(x2 + offsets[m_idx] - 0.04, merge_v, marker=markers[m_idx],
                    s=50, color=colors_m[m_idx], alpha=0.9, edgecolors="white",
                    linewidths=0.5, zorder=3,
                    label=f"{metric_label} (merge)" if m_idx == 0 else None)
        ax2.scatter(x2 + offsets[m_idx] + 0.04, nonmerge_v, marker=markers[m_idx],
                    s=50, color=colors_m[m_idx], alpha=0.35, edgecolors="white",
                    linewidths=0.5, zorder=3)

        # Connect merge/non-merge with thin lines
        for i in range(len(ENVS)):
            ax2.plot([x2[i] + offsets[m_idx] - 0.04, x2[i] + offsets[m_idx] + 0.04],
                     [merge_v[i], nonmerge_v[i]], color=colors_m[m_idx],
                     alpha=0.3, linewidth=0.8, zorder=2)

    ax2.set_xticks(x2)
    ax2.set_xticklabels([es for _, es in ENVS])
    ax2.set_ylabel("Rate")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("(b) Midpoint diagnostics", fontweight="bold")

    # Custom legend for panel b
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors_m[0],
                    markersize=6, label="Midpoint mismatch"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=colors_m[1],
                    markersize=5, label="Low-density rate"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#333",
                    markersize=5, alpha=0.9, label="Mergeable (solid)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#333",
                    markersize=5, alpha=0.35, label="Non-merge (faded)"),
    ]
    ax2.legend(handles=legend_elements, fontsize=7.5, loc="lower right",
               framealpha=0.9, handletextpad=0.3)

    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.08, top=0.95, hspace=0.38)
    savefig(fig, "fig_boundary_crossing", tight=False)


if __name__ == "__main__":
    main()
