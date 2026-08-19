#!/usr/bin/env python
"""Environment perturbation: return drop vs agreement degradation.

Scatter plot showing correlation between policy performance degradation
under environment changes and explanation stability.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from figures._style import (apply_style, savefig, load_results,
                             ENV_TAGS, ENV_SHORT, COLORS, MARKERS, COL2)
import matplotlib.pyplot as plt

ENVS_EP = ["MountainCar-v0", "LunarLander-v3"]
METHODS = [("cbs", "CBS"), ("b3_vote", "RV"), ("dt", "DT")]


def main():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(COL2 * 0.72, COL2 * 0.35))

    for col, env in enumerate(ENVS_EP):
        ax = axes[col]
        data = load_results(ENV_TAGS[env], "env_perturbation_results.json")
        if data is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            ax.set_title(ENV_SHORT[env])
            continue

        clean = data.get("clean_baseline", {})
        clean_ret = clean.get("policy_return", {}).get("mean_return", 0)
        perturbations = data.get("perturbations", {})

        for mk, label in METHODS:
            rets, bras = [], []
            for pk, pv in perturbations.items():
                md = pv.get(mk, {})
                ret_info = pv.get("policy_return", {})
                ret_delta = ret_info.get("relative_drop", 0)
                bra = md.get("mean_BRA")
                if bra is not None:
                    rets.append(-ret_delta)   # negative = worse
                    bras.append(bra)

            if rets:
                c, m = COLORS[label], MARKERS[label]
                ax.scatter(rets, bras, c=c, marker=m, s=28, alpha=0.8,
                           edgecolors="#333333", linewidths=0.4, label=label)

        ax.axhline(1.0, ls=":", color="#aaa", lw=0.4)
        ax.set_title(ENV_SHORT[env])
        ax.set_xlabel("Return drop (relative)")
        if col == 0:
            ax.set_ylabel("BRA under perturbation")

    handles, labels = {}, []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in handles:
                handles[l] = h
                labels.append(l)
    if handles:
        fig.legend(handles.values(), labels, loc="lower center",
                   ncol=len(labels), bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(wspace=0.22, bottom=0.25)
    savefig(fig, "fig_env_perturbation")


if __name__ == "__main__":
    main()
