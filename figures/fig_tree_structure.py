#!/usr/bin/env python
"""
Decision-tree structural diagnostic

Two panels:
  (a) Structure Variability: violin/box plots of depth, leaf count,
      rule count across 21 runs per environment.
  (b) Shallow-Tree Ablation: grouped bar chart showing F1, GRS, BRA
      for max_depth ∈ {3, 5, 7, None} per environment.

Also generates:
  (c) Structure–Stability Scatter: leaf count vs GRS-TA per run.

Data: experiments/results/<env>/tree_depth_ablation.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _style import (
    apply_style, savefig, load_json, RESULTS_DIR,
    ENVS, ENV_TAGS, ENV_SHORT, COLORS, COL2,
)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_diagnostic(env_tag):
    path = RESULTS_DIR / env_tag / "tree_depth_ablation.json"
    if path.exists():
        return load_json(path)
    return None


def structure_variability():
    """Violin/box plots of tree structural metrics across 21 runs."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.4))

    metrics = ["depth", "n_leaves", "n_rules"]
    metric_labels = ["Tree Depth", "Leaf Count", "Rule Count"]

    for col_idx, env in enumerate(ENVS):
        ax = axes[col_idx]
        tag = ENV_TAGS[env]
        diag = load_diagnostic(tag)
        if diag is None:
            ax.set_title(ENV_SHORT[env])
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        per_run = diag["structural_analysis"]["per_run"]
        data = {m: [] for m in metrics}
        for run_key, run_info in per_run.items():
            for m in metrics:
                data[m].append(run_info[m])

        positions = [1, 2, 3]
        parts = ax.violinplot(
            [data[m] for m in metrics],
            positions=positions,
            showmeans=True,
            showextrema=True,
            showmedians=True,
        )

        # Style violins
        for pc in parts["bodies"]:
            pc.set_facecolor(COLORS["DT"])
            pc.set_alpha(0.3)
        for key in ["cmeans", "cmedians", "cmins", "cmaxes", "cbars"]:
            if key in parts:
                parts[key].set_color(COLORS["DT"])

        ax.set_xticks(positions)
        ax.set_xticklabels(metric_labels, rotation=25, ha="right")
        ax.set_title(ENV_SHORT[env])
        ax.set_ylabel("Count" if col_idx == 0 else "")

        # Annotate mean ± std
        for i, m in enumerate(metrics):
            arr = np.array(data[m])
            ax.annotate(
                f"{arr.mean():.0f}$\\pm${arr.std():.0f}",
                xy=(positions[i], arr.max()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=9,
            )

    fig.suptitle("DT Structural Variability (21 Perturbation Runs)",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig_tree_structure_variability")


def depth_ablation():
    """Grouped bar chart: F1, GRS-TA, BRA vs max_depth per env."""
    apply_style()

    # Check which envs have ablation data
    envs_with_data = []
    all_data = {}
    for env in ENVS:
        tag = ENV_TAGS[env]
        diag = load_diagnostic(tag)
        if diag and diag.get("shallow_tree_ablation"):
            ablation = diag["shallow_tree_ablation"]
            if ablation:
                envs_with_data.append(env)
                all_data[env] = ablation

    if not envs_with_data:
        print("  SKIP depth_ablation: no ablation data found")
        return

    n_envs = len(envs_with_data)
    fig, axes = plt.subplots(1, n_envs, figsize=(COL2, 2.6),
                              squeeze=False)

    depth_labels = ["3", "5", "7", "None"]
    metric_keys = ["f1_mean", "GRS_ta", "BRA"]
    metric_labels = ["F1", "GRS-TA", "BRA"]
    bar_colors = ["#648FFF", "#DC3220", "#35A86B"]

    for col_idx, env in enumerate(envs_with_data):
        ax = axes[0, col_idx]
        ablation = all_data[env]

        x = np.arange(len(depth_labels))
        width = 0.22
        offsets = [-width, 0, width]

        for mi, (mk, ml, bc) in enumerate(zip(metric_keys, metric_labels, bar_colors)):
            values = []
            for dk in depth_labels:
                if dk in ablation:
                    values.append(ablation[dk].get(mk, 0))
                else:
                    values.append(0)
            bars = ax.bar(x + offsets[mi], values, width, label=ml, color=bc,
                          alpha=0.85, edgecolor="white", linewidth=0.5)

            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                        f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([f"d={d}" for d in depth_labels])
        ax.set_xlabel("max\\_depth")
        ax.set_ylabel("Score" if col_idx == 0 else "")
        ax.set_ylim(0, 1.15)
        ax.set_title(ENV_SHORT[env])

        if col_idx == 0:
            ax.legend(loc="upper left", fontsize=9, ncol=1)

    fig.suptitle("Shallow-Tree Ablation: Depth vs. Fidelity/Stability",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig_tree_depth_ablation")


def structure_stability_scatter():
    """Scatter: leaf count vs GRS-TA per run, colored by env."""
    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(COL2 * 0.5, 2.8))

    env_colors = {
        "MountainCar-v0": "#648FFF",
        "CartPole-v1": "#FE6100",
        "LunarLander-v3": "#35A86B",
    }

    has_data = False
    for env in ENVS:
        tag = ENV_TAGS[env]
        diag = load_diagnostic(tag)
        if diag is None:
            continue
        per_run = diag["structural_analysis"]["per_run"]

        leaves = []
        grs_ta = []
        for run_key, run_info in per_run.items():
            leaves.append(run_info["n_leaves"])
            if "grs_ta" in run_info:
                grs_ta.append(run_info["grs_ta"])
            else:
                grs_ta.append(np.nan)

        leaves = np.array(leaves)
        grs_ta = np.array(grs_ta)
        mask = ~np.isnan(grs_ta)
        if mask.sum() > 0:
            has_data = True
            ax.scatter(leaves[mask], grs_ta[mask],
                       c=env_colors[env], label=ENV_SHORT[env],
                       alpha=0.7, s=20, edgecolors="white", linewidths=0.3)

            # Trend line
            if mask.sum() >= 3:
                from scipy import stats
                slope, intercept, r, p, _ = stats.linregress(
                    leaves[mask], grs_ta[mask])
                x_line = np.linspace(leaves[mask].min(), leaves[mask].max(), 50)
                ax.plot(x_line, slope * x_line + intercept,
                        color=env_colors[env], linestyle="--", alpha=0.5,
                        linewidth=0.8)
                ax.annotate(f"$r$={r:.2f}",
                            xy=(leaves[mask].max(), slope * leaves[mask].max() + intercept),
                            fontsize=9, color=env_colors[env])

    if not has_data:
        print("  SKIP structure_stability_scatter: no stability data")
        plt.close(fig)
        return

    ax.set_xlabel("Leaf Count")
    ax.set_ylabel("GRS-TA (per-run proxy)")
    ax.set_title("Tree Complexity vs. Structural Stability")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, "fig_tree_structure_vs_stability")


def main():
    print("Generating DT Structural Diagnostic figures...")
    structure_variability()
    depth_ablation()
    structure_stability_scatter()
    print("Done.")


if __name__ == "__main__":
    main()
