#!/usr/bin/env python
"""Predicate drift across replay seeds.

Three compact panels for CartPole CBS across 5 seed shifts:
  Left: Rule count per action per seed (grouped bars)
  Center/right: Mean threshold midpoint per action, feature, and seed.

This replaces the unreadable text-box version.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from figures._style import apply_style, savefig, load_json, COL2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"

FEAT_NAMES = ["cart pos", "cart vel", "pole angle", "pole ang vel"]
ACTION_NAMES = {0: "Push Left", 1: "Push Right"}
N_SEEDS = 5


def main():
    apply_style()

    st = load_json(RESULTS / "cartpole_v1" / "stress_test_results.json")
    seeds = [f"seed_shift_s{i}" for i in range(N_SEEDS)]

    # ── Collect data ────────────────────────────────────────────────
    rule_counts = {a: [] for a in [0, 1]}       # action → [count_per_seed]
    # threshold_mid[action][feat_idx][seed_idx] = list of midpoints
    threshold_mid = {a: {f: [[] for _ in range(N_SEEDS)]
                         for f in range(4)} for a in [0, 1]}

    for si, sk in enumerate(seeds):
        rules = st["cbs"]["per_run"][sk]["rules"]
        for a in [0, 1]:
            a_rules = [r for r in rules if r["action"] == a]
            rule_counts[a].append(len(a_rules))
            for r in a_rules:
                for p in r["predicates"]:
                    fi = p["feature_idx"]
                    lb = p.get("lower_bound")
                    ub = p.get("upper_bound")
                    if lb is not None and ub is not None and fi < 4:
                        threshold_mid[a][fi][si].append((lb + ub) / 2)

    # Build both heatmaps before plotting so they share one color scale.
    heatmaps = {}
    for action in [0, 1]:
        mat = np.full((4, N_SEEDS), np.nan)
        for fi in range(4):
            for si in range(N_SEEDS):
                vals = threshold_mid[action][fi][si]
                if vals:
                    mat[fi, si] = np.mean(vals)
        heatmaps[action] = mat

    vmax = max(np.nanmax(np.abs(mat)) for mat in heatmaps.values())
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

    # Match the height of the mechanism figure and keep all panels in one row.
    fig = plt.figure(figsize=(COL2, 2.75))
    gs = fig.add_gridspec(
        1, 5, width_ratios=[0.84, 0.18, 1.27, 1.27, 0.06], wspace=0.25,
    )

    # Panel A: Rule count bars
    ax_bar = fig.add_subplot(gs[0, 0])
    x = np.arange(N_SEEDS)
    w = 0.35
    bars0 = ax_bar.bar(x - w/2, rule_counts[0], w, label="Push Left",
                        color="#648FFF", alpha=0.85, edgecolor="white", lw=0.5)
    bars1 = ax_bar.bar(x + w/2, rule_counts[1], w, label="Push Right",
                        color="#FE6100", alpha=0.85, edgecolor="white", lw=0.5)
    ax_bar.bar_label(
        bars0, padding=-12, fontsize=9, color="white", fontweight="bold"
    )
    ax_bar.bar_label(
        bars1, padding=-12, fontsize=9, color="white", fontweight="bold"
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([str(i) for i in range(N_SEEDS)])
    ax_bar.set_ylabel("Rule count")
    ax_bar.set_xlabel("Replay seed")
    ax_bar.set_title("(a) Rule counts", fontsize=9, fontweight="bold")
    ax_bar.legend(fontsize=9, loc="upper right")
    ax_bar.set_ylim(0, max(max(rule_counts[0]), max(rule_counts[1])) + 3)

    # Panels B and C: threshold midpoint heatmaps per action.
    heat_axes = []
    im = None
    for panel_idx, action in enumerate([0, 1]):
        ax = fig.add_subplot(gs[0, panel_idx + 2])
        heat_axes.append(ax)
        mat = heatmaps[action]

        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm,
                        interpolation="nearest")

        # Heatmap panel should not use global dashed grid from style.
        ax.grid(False)

        # Show every stored mean. Hatched N/U cells mean the feature was not
        # used by any rule for that action and seed.
        for fi in range(4):
            for si in range(N_SEEDS):
                if not np.isnan(mat[fi, si]):
                    v = mat[fi, si]
                    color = "white" if abs(v) > vmax * 0.55 else "black"
                    bbox_fc = (0, 0, 0, 0.18) if color == "white" else (1, 1, 1, 0.58)
                    ax.text(
                        si,
                        fi,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=color,
                        bbox=dict(boxstyle="round,pad=0.06", facecolor=bbox_fc,
                                  edgecolor="none"),
                    )
                else:
                    ax.add_patch(
                        Rectangle(
                            (si - 0.5, fi - 0.5),
                            1,
                            1,
                            facecolor="#F2F2F2",
                            edgecolor="#888888",
                            linewidth=0.5,
                            hatch="///",
                        )
                    )
                    ax.text(
                        si,
                        fi,
                        "not\nused",
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color="#555555",
                    )

        ax.set_xticks(range(N_SEEDS))
        ax.set_xticklabels([str(i) for i in range(N_SEEDS)])
        ax.set_yticks(range(4))
        if panel_idx == 0:
            ax.set_yticklabels(FEAT_NAMES, fontsize=9)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Replay seed")
        aname = ACTION_NAMES[action]
        panel_letter = chr(98 + panel_idx)  # 'b' or 'c'
        ax.set_title(f"({panel_letter}) Midpoints: {aname}",
                     fontsize=9, fontweight="bold")

    cax = fig.add_subplot(gs[0, 4])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=9)

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.89)

    savefig(fig, "fig_predicate_drift")


if __name__ == "__main__":
    main()
