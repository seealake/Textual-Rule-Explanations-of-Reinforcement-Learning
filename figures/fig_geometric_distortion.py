#!/usr/bin/env python
"""Geometric distortion: failed vs successful merges.

Two-panel figure:
  (a) Grouped bar chart comparing diagnostic metrics between failed and
      successful merges across 3 environments.
  (b) Per-group scatter: action_mismatch_rate vs low_density_bridge_rate,
      colored by fail/success, for all environments pooled.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from figures._style import apply_style, savefig, load_results, COL1
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ENVS = [
    ("mountaincar_v0", "MC"),
    ("cartpole_v1", "CP"),
    ("lunarlander_v3", "LL"),
]

FAIL_COLOR = "#DC3220"    # red
SUCCESS_COLOR = "#35A86B"  # green


def main():
    apply_style()

    all_data = {}
    for env_tag, env_short in ENVS:
        d = load_results(env_tag, "geometric_distortion.json")
        if d is None:
            print(f"  SKIP: no geometric_distortion.json for {env_tag}")
            continue
        all_data[env_tag] = d

    fig, (ax_bar, ax_scatter) = plt.subplots(
        1,
        2,
        figsize=(COL1 * 1.85, 3.2),
        gridspec_kw={"width_ratios": [1.3, 1]},
    )

    # ── Panel (a): Grouped bar chart — env × metric ──
    metrics = [
        ("mean_action_mismatch", "Act. mismatch"),
        ("mean_bridge_rate", "Bridge rate"),
        ("mean_knn_gap", "kNN gap"),
    ]
    n_metrics = len(metrics)
    n_envs = len(ENVS)
    bar_width = 0.13
    env_gap = 0.08        # gap between env pairs within a metric
    metric_gap = 0.55     # gap between metric groups

    # Compute bar positions
    xtick_positions = []
    xtick_labels = []
    metric_centers = []

    x_cursor = 0.0
    for m_idx, (metric_key, metric_label) in enumerate(metrics):
        env_centers = []
        for e_idx, (env_tag, env_short) in enumerate(ENVS):
            comp = all_data[env_tag]["comparison"]
            fail_val = comp["failed_merges"].get(metric_key, 0)
            succ_val = comp["successful_merges"].get(metric_key, 0)

            x_fail = x_cursor
            x_succ = x_cursor + bar_width

            ax_bar.bar(x_fail, fail_val, bar_width * 0.9,
                       color=FAIL_COLOR, alpha=0.85,
                       edgecolor="white", linewidth=0.5)
            ax_bar.bar(x_succ, succ_val, bar_width * 0.9,
                       color=SUCCESS_COLOR, alpha=0.85,
                       edgecolor="white", linewidth=0.5)

            pair_center = (x_fail + x_succ) / 2
            env_centers.append(pair_center)
            xtick_positions.append(pair_center)
            xtick_labels.append(env_short)
            x_cursor += 2 * bar_width + env_gap

        metric_centers.append(np.mean(env_centers))
        x_cursor += metric_gap - env_gap  # extra gap before next metric

    # Env ticks
    ax_bar.set_xticks(xtick_positions)
    ax_bar.set_xticklabels(xtick_labels, fontsize=9)

    # Metric group labels above x-axis
    trans = ax_bar.get_xaxis_transform()
    for m_idx, (_, metric_label) in enumerate(metrics):
        ax_bar.text(metric_centers[m_idx], -0.18, metric_label,
                    transform=trans, ha="center", va="top",
                    fontsize=9, fontweight="bold")

    ax_bar.set_ylabel("Rate")
    ax_bar.set_title("(a) Diagnostic metrics by environment", fontsize=9,
                     fontweight="bold")
    ax_bar.set_ylim(0, 0.85)

    # Legend
    fail_patch = mpatches.Patch(color=FAIL_COLOR, alpha=0.85, label="Failed merges")
    succ_patch = mpatches.Patch(color=SUCCESS_COLOR, alpha=0.85, label="Successful merges")
    ax_bar.legend(handles=[fail_patch, succ_patch], loc="upper right",
                  fontsize=9, framealpha=0.9)

    # ── Panel (b): Scatter plot ──
    env_markers = {"mountaincar_v0": "o", "cartpole_v1": "s", "lunarlander_v3": "D"}
    env_labels = {"mountaincar_v0": "MC", "cartpole_v1": "CP", "lunarlander_v3": "LL"}

    for env_tag, env_short in ENVS:
        diags = all_data[env_tag]["per_group_diagnostics"]
        marker = env_markers[env_tag]

        for diag in diags:
            mismatch = diag["action_mismatch_rate"]
            bridge = diag.get("low_density_bridge_rate", 0)
            is_fail = mismatch > 0.15

            ax_scatter.scatter(
                mismatch, bridge,
                c=FAIL_COLOR if is_fail else SUCCESS_COLOR,
                marker=marker, s=28, alpha=0.72,
                edgecolors="white", linewidth=0.3,
            )

    # Threshold line
    ax_scatter.axvline(0.15, color="#999", linestyle="--", linewidth=0.8,
                       label="Fail threshold")

    ax_scatter.set_xlabel("Action mismatch rate")
    ax_scatter.set_ylabel("Low-density bridge rate")
    ax_scatter.set_title("(b) Per-group diagnostics", fontsize=9, fontweight="bold")
    ax_scatter.set_xlim(-0.02, 1.0)
    ax_scatter.set_ylim(-0.02, 1.0)

    # Env shape legend
    handles = [
        plt.Line2D([0], [0], marker=m, color="w", markerfacecolor="#888",
                    markersize=5, label=l)
        for m, l in [("o", "MC"), ("s", "CP"), ("D", "LL")]
    ]
    handles.append(plt.Line2D([0], [0], linestyle="--", color="#999",
                              linewidth=0.8, label="Threshold"))
    ax_scatter.legend(handles=handles, loc="upper left", fontsize=9,
                      framealpha=0.9)

    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.22, top=0.88, wspace=0.22)
    savefig(fig, "fig_geometric_distortion", tight=False)


if __name__ == "__main__":
    main()
