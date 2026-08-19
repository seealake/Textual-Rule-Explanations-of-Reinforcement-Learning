#!/usr/bin/env python
"""Merge failure-stage decomposition.

Each environment gets one panel; two lines trace E_F1 and
worst-action recall across the 5 pipeline stages.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from figures._style import apply_style, savefig, load_results, COL1
import matplotlib.pyplot as plt

ENVS = [
    ("mountaincar_v0", "MountainCar"),
    ("cartpole_v1", "CartPole"),
    ("lunarlander_v3", "LunarLander"),
]

STAGES = [
    ("match_only", "Match"),
    ("match_hard_support", "M+HS"),
    ("match_aggregation", "M+Agg"),
    ("full_default", "DCM"),
    ("v2_soft_support", "SSC"),
]

F1_COLOR = "#2166AC"       # strong blue
WR_COLOR = "#B2182B"       # strong red


def main():
    apply_style()

    data = {}
    for env_tag, env_short in ENVS:
        d = load_results(env_tag, "failure_decomposition.json")
        if d is None:
            print(f"  SKIP: no failure_decomposition.json for {env_tag}")
            continue
        data[env_tag] = d["summary"]

    fig, axes = plt.subplots(3, 1, figsize=(COL1, 5.1), sharex=True, sharey=True)

    x = np.arange(len(STAGES))

    for row, (env_tag, env_short) in enumerate(ENVS):
        ax = axes[row]
        summary = data[env_tag]

        f1_means, f1_stds = [], []
        wr_means, wr_stds = [], []

        for stage_key, _ in STAGES:
            s = summary[stage_key]
            f1_means.append(s["f1"]["mean"])
            f1_stds.append(s["f1"]["std"])
            wr_means.append(s["worst_action_recall"]["mean"])
            wr_stds.append(s["worst_action_recall"]["std"])

        # Fidelity line
        ax.errorbar(x, f1_means, yerr=f1_stds, fmt="o-", color=F1_COLOR,
                     linewidth=1.5, markersize=5, capsize=3, capthick=0.8,
                     markeredgecolor="white", markeredgewidth=0.5,
                     label=r"$E_{F1}$" if row == 0 else None)

        # Worst-action recall line
        ax.errorbar(x, wr_means, yerr=wr_stds, fmt="s--", color=WR_COLOR,
                     linewidth=1.5, markersize=4.5, capsize=3, capthick=0.8,
                     markeredgecolor="white", markeredgewidth=0.5,
                     label="Worst-Action Recall" if row == 0 else None)

        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in STAGES], rotation=25, ha="right")
        ax.set_title(env_short, fontweight="bold", loc="left", fontsize=10)
        ax.set_ylim(-0.05, 1.05)

        if row == 1:
            ax.set_ylabel("Score")

    # Legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.995), framealpha=0.9)

    fig.subplots_adjust(bottom=0.12, top=0.91, hspace=0.24)
    savefig(fig, "fig_merge_stage_decomposition")


if __name__ == "__main__":
    main()
