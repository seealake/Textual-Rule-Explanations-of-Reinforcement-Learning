#!/usr/bin/env python
"""Fidelity vs prediction-agreement scatter.

1x3 panels (one per env). Each point = one perturbation run.
Uses smaller markers, jitter, and per-method mean+CI ellipses
to reduce visual clutter while preserving the decoupling message.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from figures._style import (apply_style, savefig, load_results, ENVS,
                             ENV_TAGS, ENV_SHORT, COLORS, MARKERS, COL2)
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


METHOD_FILES = [
    ("stress_test_results.json", {"cbs": "CBS", "cbs_maxf1": "FT-CBS"}),
    ("decision_tree_results.json", {"b4_dt": "DT"}),
    ("consensus_merge_results.json", {"consensus_vote": "RV",
                                     "consensus_cbs": "DCM"}),
]

# Ordered for consistent legend
METHOD_ORDER = ["CBS", "FT-CBS", "RV", "DCM", "DT"]


def extract_points(data, method_key):
    if method_key not in data:
        return [], []
    per_run = data[method_key].get("per_run", {})
    f1s, bras = [], []
    for rv in per_run.values():
        f1 = rv.get("fidelity_heldout", {}).get("f1")
        sp = rv.get("stability_proxy_family",
                     rv.get("stability_proxy_global", {}))
        bra = sp.get("BRA") if sp else rv.get("BRA")
        if f1 is not None and bra is not None:
            f1s.append(f1)
            bras.append(bra)
    return f1s, bras


def add_ellipse(ax, xs, ys, color, n_std=1.5):
    """Add a confidence ellipse around a cloud of points."""
    if len(xs) < 3:
        return
    mx, my = np.mean(xs), np.mean(ys)
    cov = np.cov(xs, ys)
    if np.any(np.isnan(cov)) or cov.shape != (2, 2):
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-10)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=(mx, my), width=w, height=h, angle=angle,
                  facecolor=color, alpha=0.08, edgecolor=color,
                  linewidth=0.8, linestyle="--")
    ax.add_patch(ell)


def main():
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(COL2, COL2 * 0.34), sharey=True)

    all_handles = {}
    for col, env in enumerate(ENVS):
        ax = axes[col]
        tag = ENV_TAGS[env]

        for fname, mmap in METHOD_FILES:
            data = load_results(tag, fname)
            if data is None:
                continue
            for mk, label in mmap.items():
                f1s, bras = extract_points(data, mk)
                if not f1s:
                    continue
                c, m = COLORS[label], MARKERS[label]

                # Small semi-transparent scatter
                ax.scatter(f1s, bras, c=c, marker=m, s=12, alpha=0.45,
                           edgecolors="none", zorder=3, label=label)

                # Mean marker (larger, with border)
                ax.scatter([np.mean(f1s)], [np.mean(bras)], c=c, marker=m,
                           s=55, edgecolors="#333333", linewidths=0.6,
                           zorder=6)

                # Confidence ellipse
                add_ellipse(ax, f1s, bras, c, n_std=1.2)

        ax.set_title(ENV_SHORT[env])
        ax.set_xlabel("Macro-F1")
        ax.set_xlim(0.15, 1.07)
        ax.set_ylim(0.12, 1.07)
        ax.plot([0, 1.1], [0, 1.1], ls=":", color="#bbbbbb", lw=0.5, zorder=1)

    axes[0].set_ylabel("BRA")

    # Deduplicated legend in method order
    handles_map = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in handles_map:
                handles_map[l] = h
    ordered_handles = [(handles_map[m], m) for m in METHOD_ORDER if m in handles_map]
    fig.legend([h for h, _ in ordered_handles],
               [l for _, l in ordered_handles],
               loc="lower center", ncol=len(ordered_handles),
               bbox_to_anchor=(0.5, -0.05), markerscale=1.3)

    # Remove per-axis legends
    for ax in axes:
        leg = ax.get_legend()
        if leg:
            leg.remove()

    fig.subplots_adjust(wspace=0.06, bottom=0.23)
    savefig(fig, "fig_fidelity_vs_stability")


if __name__ == "__main__":
    main()
