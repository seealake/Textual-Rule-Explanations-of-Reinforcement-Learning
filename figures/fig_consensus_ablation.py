#!/usr/bin/env python
"""Ensemble size x support threshold ablation (default merge, MountainCar).

Shows held-out Macro-F1 and GRS across B∈{3,5,10} × τ∈{0.5,0.7,0.9}
using the stored ablation cells in consensus_merge_results.json.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from figures._style import apply_style, savefig, load_json, COL2
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"

BS = [3, 5, 10]
TAUS = [0.5, 0.7, 0.9]


def _extract_grids(data):
    grid = data.get("ablations", {}).get("B_tau_grid", {})
    if not grid:
        return None, None

    f1_grid = np.full((len(BS), len(TAUS)), np.nan, dtype=float)
    grs_grid = np.full((len(BS), len(TAUS)), np.nan, dtype=float)

    for b_idx, b_val in enumerate(BS):
        for tau_idx, tau_val in enumerate(TAUS):
            key = f"B{b_val}_tau{tau_val}"
            cell = grid.get(key)
            if cell is None:
                continue
            f1_grid[b_idx, tau_idx] = cell["fidelity"]["mean_f1"]
            grs_grid[b_idx, tau_idx] = cell["stability"]["GRS_wj"]

    if np.isnan(f1_grid).any() or np.isnan(grs_grid).any():
        return None, None

    return f1_grid, grs_grid


def main():
    apply_style()

    # Load consensus results for MC — ablation data is stored in the main file.
    path = RESULTS / "mountaincar_v0" / "consensus_merge_results.json"
    if not path.exists():
        print("  SKIP fig5: no consensus results for MountainCar")
        return

    data = load_json(path)

    f1_grid, grs_grid = _extract_grids(data)
    if f1_grid is None or grs_grid is None:
        print("  SKIP fig5: incomplete B×tau ablation data in consensus results")
        return

    f1_vmin = min(0.48, float(np.nanmin(f1_grid)))
    f1_vmax = max(0.65, float(np.nanmax(f1_grid)))
    grs_vmin = min(0.5, float(np.nanmin(grs_grid)))
    grs_vmax = max(0.85, float(np.nanmax(grs_grid)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2 * 0.75, COL2 * 0.3))

    # F1 heatmap
    sns.heatmap(f1_grid, ax=ax1, annot=True, fmt=".3f", cmap="YlGnBu",
                xticklabels=[r"$\tau$=0.5", r"$\tau$=0.7", r"$\tau$=0.9"],
                yticklabels=[f"B={b}" for b in BS],
                vmin=f1_vmin, vmax=f1_vmax, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 7}, linewidths=0.5, linecolor="white")
    ax1.set_title("Macro-F1", fontsize=9)

    # GRS heatmap
    sns.heatmap(grs_grid, ax=ax2, annot=True, fmt=".3f", cmap="YlOrRd_r",
                xticklabels=[r"$\tau$=0.5", r"$\tau$=0.7", r"$\tau$=0.9"],
                yticklabels=[f"B={b}" for b in BS],
                vmin=grs_vmin, vmax=grs_vmax, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 7}, linewidths=0.5, linecolor="white")
    ax2.set_title("GRS", fontsize=9)

    fig.suptitle(r"DCM --- $B \times \tau$ Ablation (MountainCar)", fontsize=9, y=1.02)
    fig.subplots_adjust(wspace=0.35)
    savefig(fig, "fig_consensus_ablation")


if __name__ == "__main__":
    main()
