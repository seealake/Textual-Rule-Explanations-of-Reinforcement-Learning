#!/usr/bin/env python
"""Conceptual overview.

Figure 1 in the main paper.
  (a) Fidelity-stability scatter averaged over the main environments.
  (b) Illustration of an interval merge crossing a policy boundary.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["MPLBACKEND_NO_LATEX"] = "1"

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from figures._style import apply_style, savefig, COL2, COLORS, MARKERS

apply_style()
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

C_A     = "#2166AC"
C_B     = "#B2182B"
C_ALERT = "#CC3333"


def _load_main_points():
    """Load the aggregated means instead of hard-coding plotted constants."""
    root = Path(__file__).resolve().parents[1]
    with (root / "experiments/results/main_results.json").open(
            "r", encoding="utf-8") as f:
        main = json.load(f)["environments"]

    envs = ("mountaincar_v0", "cartpole_v1", "lunarlander_v3")
    methods = ("CBS", "FT-CBS", "DCM", "DT", "BDR")
    points = {}
    for method in methods:
        rows = [main[env][method] for env in envs]
        points[method] = (
            float(np.mean([row["E_F1"] for row in rows])),
            float(np.mean([row["GRS_TA"] for row in rows])),
        )
    return points


def _draw_decoupling(ax):
    ax.set_title("(a)  Fidelity and stability differ",
                 fontsize=10, fontweight="bold", loc="left")
    ax.set_xlabel(r"Fidelity ($E_{F1}$)", fontsize=10)
    ax.set_ylabel("Near-match rule overlap", fontsize=10)

    pts = _load_main_points()
    display_names = {
        "BDR": "Boolean + policy",
        "DCM": "Default merge",
        "CBS": "Clustering",
        "FT-CBS": "Tuned clustering",
        "DT": "Decision tree",
    }

    for n, (f, g) in pts.items():
        ax.scatter(f, g, s=130, c=COLORS[n], marker=MARKERS[n],
                   edgecolors="white", linewidths=0.8, zorder=5,
                   label=display_names[n])

    ax.legend(
        loc="upper right",
        fontsize=9,
        frameon=True,
        borderpad=0.35,
        handletextpad=0.4,
        labelspacing=0.3,
    )

    # CBS to FT-CBS degradation arrow.
    cbs_f, cbs_g = pts["CBS"]
    ft_f, ft_g = pts["FT-CBS"]
    ax.annotate("", xy=(ft_f, ft_g + 0.01), xytext=(cbs_f, cbs_g - 0.01),
                arrowprops=dict(arrowstyle="-|>", color=C_ALERT, lw=2.0),
                zorder=4)
    ax.text(0.76, 0.14, "Fidelity tuning\nlowers stability",
            fontsize=9, color=C_ALERT, style="italic", ha="center",
            zorder=5)

    ax.set_xlim(0.38, 0.95)
    ax.set_ylim(0.10, 0.85)


def _draw_merge_region(ax):
    ax.set_title("(b)  A merge can cross a policy boundary",
                 fontsize=10, fontweight="bold", loc="left")
    ax.set_xlabel("Feature $f_1$", fontsize=10)
    ax.set_ylabel("Feature $f_2$", fontsize=10)
    ax.grid(False)

    # Geometry fills the visible area.
    ri = (0.08, 0.06, 0.42, 0.32)     # R_i
    rj = (0.30, 0.26, 0.48, 0.44)     # R_j
    # DCM aggregates lower and upper bounds coordinate-wise by their medians.
    mx0 = np.median([ri[0], rj[0]])
    my0 = np.median([ri[1], rj[1]])
    mx1 = np.median([ri[0] + ri[2], rj[0] + rj[2]])
    my1 = np.median([ri[1] + ri[3], rj[1] + rj[3]])

    # Tight axes around content
    ax.set_xlim(0, 0.88)
    ax.set_ylim(0, 0.82)

    # Policy boundary
    x = np.linspace(0, 0.88, 300)
    bnd = 0.72 - 0.50 * x
    ax.plot(x, bnd, "k-", lw=2.0, alpha=0.5, zorder=4)

    # Background fills (no text labels — colors + caption are enough)
    ax.fill_between(x, 0, np.clip(bnd, 0, 0.82), alpha=0.04, color=C_A)
    ax.fill_between(x, np.clip(bnd, 0, 0.82), 0.82, alpha=0.04, color=C_B)

    # R_i
    ax.add_patch(Rectangle((ri[0], ri[1]), ri[2], ri[3],
                           lw=2.0, ec=C_A, fc=C_A, alpha=0.22, zorder=3))
    ax.text(ri[0]+ri[2]/2, ri[1]+ri[3]/2,
            "$R_i$", fontsize=11, color=C_A,
            ha="center", va="center", fontweight="bold", zorder=4)

    # R_j
    ax.add_patch(Rectangle((rj[0], rj[1]), rj[2], rj[3],
                           lw=2.0, ec=C_A, fc=C_A, alpha=0.12,
                           ls="--", zorder=3))
    ax.text(rj[0]+rj[2]/2, rj[1]+rj[3]*0.22,
            "$R_j$", fontsize=11, color=C_A,
            ha="center", va="center", fontweight="bold", zorder=4)

    # Merged rectangle
    ax.add_patch(Rectangle((mx0, my0), mx1-mx0, my1-my0,
                           lw=2.5, ec=C_ALERT, fc="none", zorder=5))

    # Mismatch zone
    x_fill = np.linspace(mx0, mx1, 400)
    bnd_fill = 0.72 - 0.50 * x_fill
    y_lo = np.clip(bnd_fill, my0, my1)
    y_hi = np.full_like(x_fill, my1)
    ax.fill_between(x_fill, y_lo, y_hi, where=(y_lo < y_hi),
                    alpha=0.25, color=C_ALERT, zorder=2)

    # Labels — only essential ones
    ax.text(0.53, 0.50, "Mismatch", fontsize=9,
            color="white", fontweight="bold",
            ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", fc=C_ALERT,
                      ec="none", alpha=0.85))

    ax.text(0.04, 0.68, "Policy boundary", fontsize=9,
            color="#555555", style="italic", va="top", zorder=5)

    ax.text(mx1+0.02, my0+0.01, "Merged", fontsize=9,
            color=C_ALERT, fontweight="bold", va="bottom", zorder=5)

    ax.text(0.44, 0.10, "Sim $\\geq\\rho$ $\\rightarrow$ merge",
            fontsize=9, color="#555555", ha="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec="#CCCCCC", lw=0.5, alpha=0.9),
            zorder=6)


def main():
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(COL2, COL2 * 0.42),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.30})

    _draw_decoupling(ax_a)
    _draw_merge_region(ax_b)

    # Leave room for titles and axis labels.
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.82)

    savefig(fig, "fig_overview")


if __name__ == "__main__":
    main()
