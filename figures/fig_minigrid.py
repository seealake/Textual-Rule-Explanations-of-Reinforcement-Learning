#!/usr/bin/env python
"""MiniGrid main results: CBS, DT and RV stability.

Grouped bar chart showing GRS_wj, GRS_ta, TD, BRA for each method
on MiniGrid-Dynamic-Obstacles-8x8-v0.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from figures._style import apply_style, savefig, load_json, COLORS, COL2, RESULTS_DIR


def main():
    apply_style()

    path = RESULTS_DIR / "minigrid_dynamic_obstacles_8x8_v0" / "stress_test_results.json"
    data = load_json(str(path))
    methods_data = data["methods"]

    method_keys = ["CBS", "DT", "B3-vote"]
    display_names = {"CBS": "CBS", "DT": "DT", "B3-vote": "RV"}
    metrics = ["GRS_weighted_jaccard", "GRS_threshold_aware", "TD", "BRA"]
    metric_labels = ["GRS", "GRS-TA", "TD", "BRA"]

    fig, ax = plt.subplots(1, 1, figsize=(COL2 * 0.65, COL2 * 0.4))

    x = np.arange(len(metrics))
    width = 0.22
    offsets = {"CBS": -width, "DT": 0, "B3-vote": width}
    colors = {"CBS": COLORS["CBS"], "DT": COLORS["DT"],
              "B3-vote": COLORS["RV"]}

    for method_key in method_keys:
        stab = methods_data[method_key]["stability"]
        vals = []
        for m in metrics:
            v = stab.get(m, np.nan)
            vals.append(float(v) if v is not None else np.nan)
        vals = np.array(vals)
        mask = ~np.isnan(vals)
        ax.bar(x[mask] + offsets[method_key], vals[mask], width,
               label=display_names[method_key], color=colors[method_key],
               edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("MiniGrid-Dynamic-Obstacles-8x8-v0 (PPO)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.axhline(y=1.0, color="#cccccc", linestyle=":", linewidth=0.5, zorder=1)

    savefig(fig, "fig_minigrid")
    print("  Generated fig_minigrid")


if __name__ == "__main__":
    main()
