#!/usr/bin/env python
"""Conditional trend analysis of rule conditions.

Main-text version:
- pooled across environments only
- Analysis A uses mergeable pairs only
- per-environment breakdown is deferred to the appendix tables
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figures._style import COL1, ROOT, apply_style, savefig

RESULTS = ROOT / "experiments" / "results"
SUMMARY_JSON = RESULTS / "condition_monotonicity_summary.json"

PREDICTOR_ORDER = [
    "Path low-density frac",
    "Midpoint low-density rate",
    "Rule dissimilarity (1−sim)",
]
PREDICTOR_TITLES = {
    "Path low-density frac": "Path low-density",
    "Midpoint low-density rate": "Midpoint low-density",
    "Rule dissimilarity (1−sim)": "Rule dissimilarity",
}
PREDICTOR_COLORS = {
    "Path low-density frac": "#1f4e79",
    "Midpoint low-density rate": "#2f7d32",
    "Rule dissimilarity (1−sim)": "#9c6b00",
}
FAIL_COLORS = ["#888888", "#DC3220"]

TITLE_FONTSIZE = 7.7
LABEL_FONTSIZE = 8.5
TICK_FONTSIZE = 8
PANEL_LABEL_FONTSIZE = 9


def main():
    apply_style()

    if not SUMMARY_JSON.exists():
        print("ERROR: run experiments/analyze_condition_monotonicity.py first")
        sys.exit(1)

    with open(SUMMARY_JSON) as f:
        summary = json.load(f)

    pooled_a = summary["analysis_a"]["pooled"]
    pooled_b = summary["analysis_b"]["pooled"]

    # 2x2 layout: three predictors and one midpoint diagnostic.
    fig, axes = plt.subplots(2, 2, figsize=(COL1, 3.65))
    axes = axes.flatten()  # [top-left, top-right, bottom-left, bottom-right]

    for idx, pred_label in enumerate(PREDICTOR_ORDER):
        ax = axes[idx]
        pred = pooled_a[pred_label]
        bins = pred["bins"]
        x = np.arange(1, len(bins) + 1)
        y = [b["probability"] for b in bins]
        ci_lo = [b["ci_lo"] for b in bins]
        ci_hi = [b["ci_hi"] for b in bins]
        err_lo = [p - lo for p, lo in zip(y, ci_lo)]
        err_hi = [hi - p for p, hi in zip(y, ci_hi)]
        n_vals = [b["n"] for b in bins]

        color = PREDICTOR_COLORS[pred_label]
        ax.errorbar(x, y, yerr=[err_lo, err_hi], fmt="o-", color=color,
                    capsize=3, linewidth=1.5, markersize=4.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        ax.set_ylim(0.0, 1.10)
        ax.set_title(f"({chr(97 + idx)}) {PREDICTOR_TITLES[pred_label]}",
                     fontsize=TITLE_FONTSIZE, pad=7)
        ax.set_xlabel("Quantile bin", fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        if idx == 0:
            ax.set_ylabel("Crossing rate", fontsize=LABEL_FONTSIZE)

    ax = axes[3]  # bottom-right panel
    x = np.array([0, 1])
    y = [pooled_b["no_crossing"]["p_fail"], pooled_b["yes_crossing"]["p_fail"]]
    ci_lo = [pooled_b["no_crossing"]["ci_lo"], pooled_b["yes_crossing"]["ci_lo"]]
    ci_hi = [pooled_b["no_crossing"]["ci_hi"], pooled_b["yes_crossing"]["ci_hi"]]
    err_lo = [p - lo for p, lo in zip(y, ci_lo)]
    err_hi = [hi - p for p, hi in zip(y, ci_hi)]
    n_vals = [pooled_b["no_crossing"]["n"], pooled_b["yes_crossing"]["n"]]

    ax.bar(x, y, width=0.56, color=FAIL_COLORS, edgecolor="white", linewidth=0.6)
    ax.errorbar(x, y, yerr=[err_lo, err_hi], fmt="none", ecolor="black", capsize=3, linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(["No crossing", "Crossing"], fontsize=LABEL_FONTSIZE)
    ax.set_ylim(0.0, 1.08)
    ax.set_title("(d) Midpoint disagreement", fontsize=TITLE_FONTSIZE, pad=7)
    ax.set_ylabel("Disagreement rate", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    fig.subplots_adjust(hspace=0.52, wspace=0.44, left=0.18, right=0.98,
                        bottom=0.10, top=0.93)

    savefig(fig, "fig_condition_monotonicity", tight=False)
    print("Done.")


if __name__ == "__main__":
    main()
