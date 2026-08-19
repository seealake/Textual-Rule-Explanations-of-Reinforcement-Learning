#!/usr/bin/env python
"""SoftSupport Consensus (SSC) Pareto frontier.

Shows fidelity vs near-match rule overlap for the SSC sweep configurations and
the baseline methods.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from figures._style import (
    COLORS,
    COL2,
    ENVS,
    ENV_SHORT,
    ENV_TAGS,
    KEY_TO_LABEL,
    apply_style,
    load_json,
    savefig,
)


ROOT = Path(__file__).resolve().parent.parent
SOFT_SUPPORT_DIR = ROOT / "experiments" / "results" / "soft_support_merge" / "raw"

LAMBDA_COLORS = {
    0.0: "#A6CEE3",
    0.1: "#1F78B4",
    0.2: "#08306B",
}
SUPPORT_MARKERS = {"hard": "v", "soft": "^"}
BASELINE_KEY_TO_LABEL = {
    "CBS": "CBS",
    "B3_consensus": "DCM",
    "B3_vote": "RV",
}


def pareto_front(points):
    ordered = sorted(points, key=lambda item: (-item[0], -item[1]))
    frontier = []
    best_grs = -1.0
    for f1, grs, label in ordered:
        if grs > best_grs:
            frontier.append((f1, grs, label))
            best_grs = grs
    return sorted(frontier, key=lambda item: item[0])


def mean_f1(per_run):
    values = [entry.get("f1") for entry in per_run if entry.get("f1") is not None]
    return float(np.mean(values)) if values else None


def load_env_data(env):
    path = SOFT_SUPPORT_DIR / f"{ENV_TAGS[env]}_soft_support_results.json"
    if not path.exists():
        return None
    return load_json(path)


def short_config_label(lambda_b, support_mode, safeguard_enabled):
    safeguard = "on" if safeguard_enabled else "off"
    return f"$\\lambda_B$={lambda_b:.1f}, {support_mode}, sg {safeguard}"


def padded_limits(values, floor=0.0, ceiling=1.0, min_span=0.10, pad_frac=0.18):
    low = min(values)
    high = max(values)
    span = max(high - low, min_span)
    pad = span * pad_frac
    low = max(floor, low - pad)
    high = min(ceiling, high + pad)
    if high - low < min_span:
        center = 0.5 * (high + low)
        half = 0.5 * min_span
        low = max(floor, center - half)
        high = min(ceiling, center + half)
    return low, high


def main():
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(COL2, COL2 * 0.42), sharey=False)

    for axis, env in zip(axes, ENVS):
        data = load_env_data(env)
        if data is None:
            axis.set_title(f"{ENV_SHORT[env]}\n(no data)")
            continue

        baseline_points = []
        for base_key, base_entry in data.get("baselines", {}).items():
            label = BASELINE_KEY_TO_LABEL.get(base_key, KEY_TO_LABEL.get(base_key.lower(), base_key))
            f1 = mean_f1(base_entry.get("per_run", []))
            grs_ta = base_entry.get("stability", {}).get("GRS_ta")
            if f1 is None or grs_ta is None:
                continue
            baseline_points.append((f1, grs_ta, label))
            axis.scatter(
                [f1],
                [grs_ta],
                c=COLORS.get(label, "#444444"),
                marker={"CBS": "o", "DCM": "p", "RV": "^"}.get(label, "o"),
                s=90,
                edgecolors="#222222",
                linewidths=0.9,
                zorder=6,
            )

        sweep_points = []
        for entry in data.get("v2_sweep", {}).values():
            config = entry.get("config", {})
            lambda_b = round(float(config.get("lambda_B", 0.0)), 1)
            support_mode = config.get("support_mode", "hard")
            safeguard_enabled = bool(config.get("safeguard_enabled", False))
            f1 = mean_f1(entry.get("per_run", []))
            grs_ta = entry.get("stability", {}).get("GRS_ta")
            if f1 is None or grs_ta is None:
                continue
            label = short_config_label(lambda_b, support_mode, safeguard_enabled)
            sweep_points.append((f1, grs_ta, label, lambda_b, support_mode, safeguard_enabled))
            axis.scatter(
                [f1],
                [grs_ta],
                c=LAMBDA_COLORS.get(lambda_b, "#666666"),
                marker=SUPPORT_MARKERS.get(support_mode, "o"),
                s=44,
                edgecolors=("#111111" if safeguard_enabled else "white"),
                linewidths=(1.0 if safeguard_enabled else 0.6),
                alpha=0.95,
                zorder=3,
            )

        frontier = pareto_front([(f1, grs_ta, label) for f1, grs_ta, label, *_ in sweep_points])
        if len(frontier) >= 2:
            axis.plot(
                [point[0] for point in frontier],
                [point[1] for point in frontier],
                ls="--",
                lw=0.9,
                color="#666666",
                zorder=2,
            )

        for f1, grs_ta, _ in frontier:
            axis.scatter(
                [f1],
                [grs_ta],
                s=110,
                facecolors="none",
                edgecolors="#222222",
                linewidths=1.1,
                zorder=5,
            )

        all_x = [item[0] for item in baseline_points] + [item[0] for item in sweep_points]
        all_y = [item[1] for item in baseline_points] + [item[1] for item in sweep_points]
        if all_x and all_y:
            axis.set_xlim(*padded_limits(all_x, min_span=0.10))
            axis.set_ylim(*padded_limits(all_y, min_span=0.12))

        axis.text(
            0.03,
            0.97,
            f"Pareto configs: {len(frontier)}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#444444",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#dddddd", alpha=0.92),
        )

        axis.set_title(ENV_SHORT[env])
        axis.set_xlabel("Macro-F1")

    axes[0].set_ylabel("GRS-TA")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["CBS"], markeredgecolor="#222222", markersize=6, label="CBS baseline"),
        Line2D([0], [0], marker="p", color="none", markerfacecolor=COLORS["DCM"], markeredgecolor="#222222", markersize=6, label="DCM baseline"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=COLORS["RV"], markeredgecolor="#222222", markersize=6, label="RV baseline"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor="#1F78B4", markeredgecolor="white", markersize=5, label="SSC hard support"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#1F78B4", markeredgecolor="white", markersize=5, label="SSC soft support"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LAMBDA_COLORS[0.0], markeredgecolor="white", markersize=5, label="$\\lambda_B$=0.0"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LAMBDA_COLORS[0.1], markeredgecolor="white", markersize=5, label="$\\lambda_B$=0.1"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LAMBDA_COLORS[0.2], markeredgecolor="white", markersize=5, label="$\\lambda_B$=0.2"),
        Line2D([0], [0], marker="o", color="#666666", markerfacecolor="none", markeredgecolor="#111111", markersize=5, label="black edge = safeguard on"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#222222", markersize=7, label="Pareto-optimal SSC"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.10),
        columnspacing=0.9,
        handletextpad=0.4,
    )
    fig.subplots_adjust(wspace=0.14, bottom=0.31)
    savefig(fig, "fig_soft_support_pareto")


if __name__ == "__main__":
    main()