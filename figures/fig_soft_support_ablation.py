#!/usr/bin/env python
"""SoftSupport Consensus (SSC) ablation heatmap.

Shows Macro-F1 and BRA across lambda_B x support_mode configurations,
one row per environment.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from figures._style import apply_style, savefig, load_json, ENVS, ENV_TAGS, ENV_SHORT, COL2
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"


def main():
    apply_style()

    v2_dir = RESULTS / "soft_support_merge" / "raw"
    envs_found, all_data = [], {}
    for env in ENVS:
        tag = ENV_TAGS[env]
        path = v2_dir / f"{tag}_soft_support_results.json"
        if path.exists():
            all_data[env] = load_json(path)
            envs_found.append(env)

    if not envs_found:
        print("  SKIP: no SSC appendix results found")
        return

    lambda_bs = [0.0, 0.1, 0.2]
    configs = [("hard", False), ("hard", True), ("soft", False), ("soft", True)]
    config_labels = ["hard\nsg_off", "hard\nsg_on", "soft\nsg_off", "soft\nsg_on"]
    row_labels = [f"$\\lambda_B$={lb}" for lb in lambda_bs]

    n_envs = len(envs_found)
    fig, axes = plt.subplots(n_envs, 2, figsize=(COL2 * 0.78, 1.6 * n_envs + 0.5),
                              squeeze=False)

    for row, env in enumerate(envs_found):
        d = all_data[env]
        sweep = d.get("v2_sweep", {})
        if not sweep:
            print(f"  SKIP: missing canonical v2_sweep block for {env}")
            continue

        f1_mat = np.full((len(lambda_bs), len(configs)), np.nan)
        bra_mat = np.full((len(lambda_bs), len(configs)), np.nan)

        for ri, lb in enumerate(lambda_bs):
            for ci, (sm, sg) in enumerate(configs):
                sg_str = "sgon" if sg else "sgoff"
                key = f"lB{lb:.1f}_sm{sm}_{sg_str}"
                entry = sweep.get(key)
                if entry is not None:
                    runs = entry.get("per_run", entry.get("runs", []))
                    if runs and isinstance(runs, list):
                        f1_mat[ri, ci] = np.mean([r.get("f1", np.nan) for r in runs])
                    bra_val = entry.get("stability", {}).get("BRA")
                    if bra_val is not None:
                        bra_mat[ri, ci] = bra_val

        ax_f1, ax_bra = axes[row, 0], axes[row, 1]

        valid_f1 = f1_mat[~np.isnan(f1_mat)]
        if len(valid_f1) > 0:
            sns.heatmap(f1_mat, ax=ax_f1, annot=True, fmt=".3f",
                        cmap="YlGnBu", linewidths=0.8, linecolor="white",
                        xticklabels=config_labels, yticklabels=row_labels,
                        vmin=max(0.35, valid_f1.min() - 0.03),
                        vmax=min(1.0, valid_f1.max() + 0.03),
                        cbar_kws={"shrink": 0.8, "pad": 0.02},
                        annot_kws={"size": 9})
        ax_f1.set_title(f"{ENV_SHORT[env]} -- Macro-F1", fontsize=9)
        ax_f1.tick_params(axis="both", labelsize=9)

        valid_bra = bra_mat[~np.isnan(bra_mat)]
        if len(valid_bra) > 0:
            sns.heatmap(bra_mat, ax=ax_bra, annot=True, fmt=".3f",
                        cmap="YlOrRd", linewidths=0.8, linecolor="white",
                        xticklabels=config_labels, yticklabels=row_labels,
                        vmin=max(0.0, valid_bra.min() - 0.05),
                        vmax=min(1.0, valid_bra.max() + 0.05),
                        cbar_kws={"shrink": 0.8, "pad": 0.02},
                        annot_kws={"size": 9})
        ax_bra.set_title(f"{ENV_SHORT[env]} -- BRA", fontsize=9)
        ax_bra.tick_params(axis="both", labelsize=9)

    fig.subplots_adjust(hspace=0.55, wspace=0.4)
    savefig(fig, "fig_soft_support_ablation")


if __name__ == "__main__":
    main()
