#!/usr/bin/env python
"""Fidelity and prediction agreement vs replay noise level.

2×3 grid: top row = Macro-F1 degradation, bottom row = BRA degradation.
Columns = environments.  Lines = methods with ±1σ error bands.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from figures._style import (apply_style, savefig, load_results, ENVS,
                             ENV_TAGS, ENV_SHORT, COLORS, MARKERS, COL2)
import matplotlib.pyplot as plt

METHODS = [("cbs", "CBS"), ("b3_vote", "RV"), ("dt", "DT")]


def main():
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(COL2, COL2 * 0.52),
                              sharex="col", sharey="row")

    for col, env in enumerate(ENVS):
        data = load_results(ENV_TAGS[env], "noise_severity_results.json")
        if data is None:
            continue

        noise_levels = np.array([float(x) for x in data["noise_levels"]])
        ax_f1, ax_bra = axes[0, col], axes[1, col]

        for mk, label in METHODS:
            nc = data["noise_curves"].get(mk)
            if nc is None:
                continue

            f1, f1_s, bra, bra_s = [], [], [], []
            for lvl in data["noise_levels"]:
                e = nc.get(f"{float(lvl):.3f}", {})
                f1.append(e.get("mean_f1"))
                f1_s.append(e.get("std_f1", 0))
                bra.append(e.get("mean_BRA"))
                bra_s.append(e.get("std_BRA", 0))

            f1 = np.array([v if v is not None else np.nan for v in f1])
            f1_s = np.array([v if v is not None else 0 for v in f1_s])
            bra = np.array([v if v is not None else np.nan for v in bra])
            bra_s = np.array([v if v is not None else 0 for v in bra_s])

            c, m = COLORS[label], MARKERS[label]

            mask = ~np.isnan(f1)
            if mask.any():
                x = noise_levels[mask]
                ax_f1.plot(x, f1[mask], marker=m, color=c, label=label,
                           markeredgecolor="white", markeredgewidth=0.3)
                ax_f1.fill_between(x, f1[mask] - f1_s[mask],
                                   f1[mask] + f1_s[mask], alpha=0.12, color=c)

            mask = ~np.isnan(bra)
            if mask.any():
                x = noise_levels[mask]
                ax_bra.plot(x, bra[mask], marker=m, color=c, label=label,
                            markeredgecolor="white", markeredgewidth=0.3)
                ax_bra.fill_between(x, bra[mask] - bra_s[mask],
                                    bra[mask] + bra_s[mask], alpha=0.12, color=c)

        ax_f1.set_title(ENV_SHORT[env])
        ax_bra.set_xlabel(r"Replay noise level ($\sigma$)")

        if col == 0:
            ax_f1.set_ylabel("Macro-F1")
            ax_bra.set_ylabel("BRA")

        for ax in [ax_f1, ax_bra]:
            ax.set_ylim(0.25, 1.05)
            ax.set_xlim(noise_levels[0] - 0.002, noise_levels[-1] + 0.002)

    # Single shared legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METHODS),
               bbox_to_anchor=(0.5, -0.01))

    fig.align_ylabels(axes[:, 0])
    fig.subplots_adjust(hspace=0.15, wspace=0.08, bottom=0.13)
    savefig(fig, "fig_noise_severity", tight=True)


if __name__ == "__main__":
    main()
