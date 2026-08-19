#!/usr/bin/env python
"""Multi-axis scaling and extrapolation across environments."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from experiments.generate_suite_summary import build_complexity_summary
from figures._style import apply_style, savefig, COL2


CMAP = sns.color_palette("YlGnBu", as_cmap=True)


def _normalize(values):
    values = np.asarray(values, dtype=float)
    span = values.max() - values.min()
    if span == 0:
        return np.full_like(values, 0.5)
    return (values - values.min()) / span


def _point_size(action_count):
    return 80 + 45 * action_count


def main():
    apply_style()
    rows = sorted(build_complexity_summary(), key=lambda row: row["obs_features"])

    env_labels = [row["short"] for row in rows]
    obs_features = np.array([row["obs_features"] for row in rows], dtype=float)
    action_counts = np.array([row["action_count"] for row in rows], dtype=float)
    sharpness = np.array([row["policy_sharpness_proxy"] for row in rows], dtype=float)
    grs_ta = np.array([row["cbs_grs_ta"] for row in rows], dtype=float)
    bra = np.array([row["cbs_bra"] for row in rows], dtype=float)

    heatmap_values = np.column_stack([
        _normalize(obs_features),
        _normalize(action_counts),
        _normalize(sharpness),
    ])
    annotations = np.array([
        [f"{int(row['obs_features'])}", f"{int(row['action_count'])}", f"{row['policy_sharpness_proxy']:.2f}"]
        for row in rows
    ])

    fig = plt.figure(figsize=(COL2, COL2 * 0.49))
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.0, 1.02, 1.02, 0.10],
        left=0.06,
        right=0.96,
        bottom=0.18,
        top=0.80,
        wspace=0.48,
    )

    ax0 = fig.add_subplot(grid[0, 0])
    sns.heatmap(
        heatmap_values,
        annot=annotations,
        fmt="",
        cmap=CMAP,
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        xticklabels=["Obs.\nfeat.", "Actions", "Sharp.\nproxy"],
        yticklabels=env_labels,
        ax=ax0,
    )
    ax0.set_title("(a) Complexity Profile", fontsize=9)
    ax0.tick_params(axis="x", rotation=0)
    ax0.tick_params(axis="y", rotation=0)

    scatter_kwargs = {
        "c": sharpness,
        "cmap": CMAP,
        "vmin": 0.0,
        "vmax": max(0.35, float(sharpness.max())),
        "edgecolors": "#2F2F2F",
        "linewidths": 0.6,
        "zorder": 3,
    }

    ax1 = fig.add_subplot(grid[0, 1])
    ax1.plot(obs_features, grs_ta, linestyle="--", color="#B0B0B0", linewidth=0.9, zorder=1)
    sc = ax1.scatter(obs_features, grs_ta, s=[_point_size(v) for v in action_counts], **scatter_kwargs)
    for row, x, y in zip(rows, obs_features, grs_ta):
        ax1.annotate(row["short"], (x, y), xytext=(4, 5), textcoords="offset points", fontsize=9)
    ax1.set_title("(b) Structural Stability", fontsize=9)
    ax1.set_xlabel("Observation features")
    ax1.set_ylabel("CBS GRS-TA")
    ax1.set_xticks(obs_features)
    ax1.set_ylim(0.18, 0.63)

    ax2 = fig.add_subplot(grid[0, 2])
    ax2.plot(obs_features, bra, linestyle="--", color="#B0B0B0", linewidth=0.9, zorder=1)
    ax2.scatter(obs_features, bra, s=[_point_size(v) for v in action_counts], **scatter_kwargs)
    for row, x, y in zip(rows, obs_features, bra):
        ax2.annotate(row["short"], (x, y), xytext=(4, 5), textcoords="offset points", fontsize=9)
    ax2.set_title("(c) Behavioral Stability", fontsize=9)
    ax2.set_xlabel("Observation features")
    ax2.set_ylabel("CBS BRA")
    ax2.set_xticks(obs_features)
    ax2.set_ylim(0.62, 1.02)
    cax = fig.add_subplot(grid[0, 3])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Policy sharpness proxy\n$1 - H(a)/\\log |A|$", fontsize=9)

    fig.suptitle("Scaling / Extrapolation: Stability Depends on Multiple Complexity Axes", fontsize=9.0, y=0.95)
    fig.text(
        0.5,
        0.865,
        "Bubble size = action count; color = policy sharpness. MiniGrid is the contrast case: 14 features but higher stability than CartPole or LunarLander.",
        ha="center",
        fontsize=9,
        color="#555555",
    )

    savefig(fig, "fig_complexity")


if __name__ == "__main__":
    main()
