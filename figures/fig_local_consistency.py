#!/usr/bin/env python
"""Local prediction consistency curves.

Consistency score vs perturbation radius, one subplot per environment.
Uses noise_severity_results.json, which has dense 7-point curves.
Falls back to lec_results.json (3 epsilon levels) if needed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from figures._style import (apply_style, savefig, load_results, ENVS,
                             ENV_TAGS, ENV_SHORT, COLORS, MARKERS, COL2)
import matplotlib.pyplot as plt

METHODS = [("cbs_pred", "CBS"), ("b3_vote", "RV"), ("dt", "DT")]


def main():
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(COL2, COL2 * 0.35), sharey=True)

    for col, env in enumerate(ENVS):
        ax = axes[col]
        tag = ENV_TAGS[env]

        # Try dense curves from noise_severity first
        ns = load_results(tag, "noise_severity_results.json")
        if ns and "lec_curves" in ns:
            epsilons = ns.get("epsilons", [])
            lec = ns["lec_curves"]
            for mk, label in METHODS:
                curve = lec.get(mk, {})
                if isinstance(curve, dict) and epsilons:
                    scores = []
                    for ep in epsilons:
                        ep_key = f"{float(ep):.4f}" if f"{float(ep):.4f}" in curve else f"{float(ep):.3f}"
                        val = curve.get(ep_key, curve.get(str(ep), {}))
                        if isinstance(val, dict):
                            scores.append(val.get("lec", val.get("mean_lec", np.nan)))
                        elif isinstance(val, (int, float)):
                            scores.append(val)
                        else:
                            scores.append(np.nan)
                    c, m = COLORS[label], MARKERS[label]
                    mask = ~np.isnan(np.array(scores, dtype=float))
                    if mask.any():
                        eps_arr = np.array(epsilons, dtype=float)
                        ax.plot(eps_arr[mask], np.array(scores, dtype=float)[mask],
                                marker=m, color=c, label=label,
                                markeredgecolor="white", markeredgewidth=0.3)
        else:
            # Fallback: lec_results.json with 3 epsilon levels
            lec_data = load_results(tag, "lec_results.json")
            if lec_data:
                epsilons = lec_data.get("epsilons", [0.01, 0.03, 0.05])
                for mk, label in [("cbs", "CBS"), ("cbs_maxf1", "FT-CBS")]:
                    mdata = lec_data.get(mk, {})
                    if mdata:
                        scores = [mdata.get(str(ep), {}).get("lec", np.nan)
                                  for ep in epsilons]
                        c, m = COLORS.get(label, "#333"), MARKERS.get(label, "o")
                        ax.plot(epsilons, scores, marker=m, color=c, label=label,
                                markeredgecolor="white", markeredgewidth=0.3)

        ax.set_title(ENV_SHORT[env])
        if col == 1:
            ax.set_xlabel(r"Perturbation radius ($\varepsilon$)")
        ax.set_ylim(0.35, 1.05)
        ax.axhline(0.8, ls=":", color="#cc0000", lw=0.5, alpha=0.4,
                   label="_nolegend_")

    axes[0].set_ylabel("LPC")

    handles, labels = {}, []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in handles:
                handles[l] = h
                labels.append(l)
    if handles:
        fig.legend(handles.values(), labels, loc="lower center",
                   ncol=len(labels), bbox_to_anchor=(0.5, -0.02))

    fig.subplots_adjust(wspace=0.06, bottom=0.28)
    savefig(fig, "fig_local_consistency")


if __name__ == "__main__":
    main()
