#!/usr/bin/env python
"""Pareto frontier: near-match rule overlap vs fidelity.

1×3 panels.  Mean markers with small individual-run dots.
Pareto front highlighted.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from figures._style import (apply_style, savefig, load_results, ENVS,
                             ENV_TAGS, ENV_SHORT, COLORS, MARKERS, COL2)
import matplotlib.pyplot as plt


METHOD_FILES = [
    ("stress_test_results.json", {"cbs": "CBS", "cbs_maxf1": "FT-CBS"}),
    ("decision_tree_results.json", {"b4_dt": "DT"}),
    ("consensus_merge_results.json", {"consensus_vote": "RV",
                                     "consensus_cbs": "DCM"}),
]


def extract_f1_grs(data, mk):
    if mk not in data:
        return [], []
    per_run = data[mk].get("per_run", {})
    f1s, grs = [], []
    for rv in per_run.values():
        f1 = rv.get("fidelity_heldout", {}).get("f1")
        sp = rv.get("stability_proxy_family",
                     rv.get("stability_proxy_global", {}))
        g = sp.get("GRS_ta") if sp else rv.get("GRS_ta")
        if f1 is not None and g is not None:
            f1s.append(f1)
            grs.append(g)
    return f1s, grs


def pareto_front(pts):
    """pts: list of (f1, grs) — maximize both."""
    pts = sorted(pts, key=lambda p: -p[0])
    front = []
    best_grs = -1
    for p in pts:
        if p[1] > best_grs:
            front.append(p)
            best_grs = p[1]
    return sorted(front, key=lambda p: p[0])


def main():
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(COL2, COL2 * 0.38), sharey=True)

    for col, env in enumerate(ENVS):
        ax = axes[col]
        tag = ENV_TAGS[env]
        means = []

        for fname, mmap in METHOD_FILES:
            data = load_results(tag, fname)
            if data is None:
                continue
            for mk, label in mmap.items():
                f1s, grs = extract_f1_grs(data, mk)
                if not f1s:
                    continue
                c, m = COLORS[label], MARKERS[label]
                mf1, mgrs = np.mean(f1s), np.mean(grs)

                # Ghost individual runs
                ax.scatter(f1s, grs, c=c, marker=m, s=8, alpha=0.20,
                           edgecolors="none", zorder=2)
                # Mean marker
                ax.scatter([mf1], [mgrs], c=c, marker=m, s=60,
                           edgecolors="#333333", linewidths=0.7,
                           label=label, zorder=5)
                means.append((mf1, mgrs))

        # Pareto front
        if len(means) >= 2:
            pf = pareto_front(means)
            if len(pf) >= 2:
                ax.plot([p[0] for p in pf], [p[1] for p in pf],
                        ls="--", color="#888888", lw=0.8, zorder=1)

        ax.set_title(ENV_SHORT[env])
        if col == 1:
            ax.set_xlabel("Macro-F1")
        ax.set_xlim(0.2, 1.05)
        ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel("GRS-TA")

    handles, labels = {}, []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in handles:
                handles[l] = h
                labels.append(l)
    fig.legend(handles.values(), labels, loc="lower center",
               ncol=len(labels), bbox_to_anchor=(0.5, -0.02))

    fig.subplots_adjust(wspace=0.06, bottom=0.25)
    savefig(fig, "fig_pareto_frontier")


if __name__ == "__main__":
    main()
