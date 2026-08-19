#!/usr/bin/env python
"""Merge-mechanism figure from stored decomposition and boundary data."""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from figures._style import COL2, ROOT, apply_style, load_results, savefig


ENVS = [
    ("mountaincar_v0", "MC", "#2166AC", "o"),
    ("cartpole_v1", "CP", "#B2182B", "s"),
    ("lunarlander_v3", "LL", "#1B7837", "D"),
]

STAGES = [
    ("match_only", "Match"),
    ("match_hard_support", "+ support"),
    ("match_aggregation", "+ aggregate"),
    ("full_default", "DCM"),
    ("v2_soft_support", "+ soft support"),
]


def main():
    apply_style()

    decomposition = {
        env: load_results(env, "failure_decomposition.json")["summary"]
        for env, _, _, _ in ENVS
    }
    boundary = {
        env: load_results(env, "boundary_crossing.json")["summary"]
        for env, _, _, _ in ENVS
    }
    with (Path(ROOT) / "experiments" / "results"
          / "condition_monotonicity_summary.json").open(encoding="utf-8") as file:
        condition = json.load(file)

    fig = plt.figure(figsize=(COL2, 3.05))
    outer = fig.add_gridspec(1, 3, width_ratios=[1.13, 0.90, 0.88])
    ax_stage = fig.add_subplot(outer[0, 0])
    ax_boundary = fig.add_subplot(outer[0, 1])
    pooled = outer[0, 2].subgridspec(2, 1, hspace=0.92)
    ax_density = fig.add_subplot(pooled[0, 0])
    ax_mismatch = fig.add_subplot(pooled[1, 0])

    x = np.arange(len(STAGES))
    for env, label, color, marker in ENVS:
        means = [decomposition[env][key]["f1"]["mean"] for key, _ in STAGES]
        stds = [decomposition[env][key]["f1"]["std"] for key, _ in STAGES]
        ax_stage.errorbar(
            x,
            means,
            yerr=stds,
            color=color,
            marker=marker,
            linewidth=1.4,
            markersize=4.5,
            capsize=2.5,
        )
        ax_stage.text(4.08, means[-1], label, color=color, va="center", fontsize=9)

    ax_stage.set_xticks(x)
    ax_stage.set_xticklabels(
        ["Match", "Hard support", "Interval merge", "Full merge", "Soft support"],
        rotation=24,
        ha="right",
        rotation_mode="anchor",
    )
    ax_stage.tick_params(axis="x", labelsize=9, pad=2)
    ax_stage.set_xlim(-0.2, 4.45)
    ax_stage.set_ylim(0.2, 0.9)
    ax_stage.set_ylabel(r"$E_{F1}$")
    ax_stage.set_title("(a) Merge stages", loc="left", fontweight="bold")

    metric_rows = [
        ("Above threshold\ncrossing", "mergeable", "mean_boundary_crossing_rate"),
        ("Below threshold\ncrossing", "non_mergeable", "mean_boundary_crossing_rate"),
        ("Midpoint\nmismatch", "mergeable", "mean_midpoint_mismatch"),
        ("Low-density\nmidpoint", "mergeable", "mean_midpoint_low_density"),
    ]
    y = np.arange(len(metric_rows))[::-1]
    offsets = [0.16, 0.0, -0.16]
    for offset, (env, label, color, marker) in zip(offsets, ENVS):
        values = [boundary[env][group][metric] for _, group, metric in metric_rows]
        ax_boundary.scatter(
            values,
            y + offset,
            color=color,
            marker=marker,
            s=34,
            linewidths=0.5,
            edgecolors="white",
            label=label,
            zorder=3,
        )

    ax_boundary.set_yticks(y)
    ax_boundary.set_yticklabels([label for label, _, _ in metric_rows])
    ax_boundary.tick_params(axis="y", labelsize=9, pad=2)
    ax_boundary.set_xlim(0, 1)
    ax_boundary.set_xlabel("Rate")
    ax_boundary.set_title("(b) Boundary crossing", loc="left", fontweight="bold")

    density = condition["analysis_a"]["pooled"]["Path low-density frac"]
    density_bins = density["bins"]
    density_x = np.arange(1, len(density_bins) + 1)
    density_y = np.array([row["probability"] for row in density_bins])
    density_lo = density_y - np.array([row["ci_lo"] for row in density_bins])
    density_hi = np.array([row["ci_hi"] for row in density_bins]) - density_y
    ax_density.errorbar(
        density_x,
        density_y,
        yerr=np.vstack([density_lo, density_hi]),
        color="#5B4B8A",
        marker="o",
        linewidth=1.35,
        markersize=4.2,
        capsize=2.2,
    )
    ax_density.set_xticks(density_x)
    ax_density.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax_density.set_ylim(0, 1.08)
    ax_density.set_ylabel("Any crossing")
    ax_density.set_title("(c) Pooled pairs", loc="left", fontweight="bold")
    ax_density.text(
        0.98,
        0.08,
        r"$\rho=.36$",
        transform=ax_density.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    mismatch = condition["analysis_b"]["pooled"]
    mismatch_rows = [mismatch["no_crossing"], mismatch["yes_crossing"]]
    mismatch_y = np.array([row["p_fail"] for row in mismatch_rows])
    mismatch_lo = mismatch_y - np.array([row["ci_lo"] for row in mismatch_rows])
    mismatch_hi = np.array([row["ci_hi"] for row in mismatch_rows]) - mismatch_y
    ax_mismatch.bar(
        [0, 1],
        mismatch_y,
        yerr=np.vstack([mismatch_lo, mismatch_hi]),
        color=["#BDBDBD", "#5B4B8A"],
        width=0.62,
        capsize=2.2,
        edgecolor="white",
        linewidth=0.5,
    )
    ax_mismatch.set_xticks([0, 1])
    ax_mismatch.set_xticklabels(["No cross.", "Cross."])
    ax_mismatch.set_ylim(0, 1.08)
    ax_mismatch.set_ylabel(r"Mismatch $>.5$")
    ax_mismatch.text(
        0.98,
        0.88,
        r"Fisher $p=.003$",
        transform=ax_mismatch.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.23, top=0.91, wspace=0.78)
    savefig(fig, "fig_merge_mechanism", tight=False)


if __name__ == "__main__":
    main()
