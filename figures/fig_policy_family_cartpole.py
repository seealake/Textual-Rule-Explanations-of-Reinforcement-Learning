#!/usr/bin/env python
"""PPO vs DQN on CartPole-v1.

Grouped bar chart comparing CBS/DT/RV stability metrics
between PPO and DQN on the same environment.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from figures._style import apply_style, savefig, load_json, COLORS, COL2, RESULTS_DIR


def main():
    apply_style()

    # Load PPO results
    ppo_path = RESULTS_DIR / "cross_algo_comparison" / "ppo_cartpole_stress.json"
    ppo_data = load_json(str(ppo_path))

    # Load DQN results (old format)
    dqn_path = RESULTS_DIR / "cartpole_v1" / "stress_test_results.json"
    dqn_data = load_json(str(dqn_path))

    # Extract CBS stability for both
    ppo_cbs = ppo_data.get("CBS", {}).get("stability", {})
    dqn_cbs = dqn_data.get("cbs", {}).get("stability", {})

    # PPO also has DT and RV
    ppo_dt = ppo_data.get("DT", {}).get("stability", {})
    ppo_rv = ppo_data.get("B3-vote", {}).get("stability", {})

    fig, axes = plt.subplots(1, 2, figsize=(COL2 * 0.75, COL2 * 0.35))

    # Panel A: CBS PPO vs DQN
    ax = axes[0]
    metrics = ["GRS_weighted_jaccard", "GRS_threshold_aware", "BRA"]
    metric_short = ["GRS", "GRS-TA", "BRA"]
    x = np.arange(len(metrics))
    width = 0.3

    dqn_vals = [float(dqn_cbs.get(m, 0)) for m in metrics]
    ppo_vals = [float(ppo_cbs.get(m, 0)) for m in metrics]

    ax.bar(x - width / 2, dqn_vals, width, label="DQN", color="#648FFF",
           edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x + width / 2, ppo_vals, width, label="PPO", color="#FE6100",
           edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_short)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("(a) CBS: PPO vs DQN", fontsize=9)
    ax.legend(loc="upper right", fontsize=9)

    # Add value labels
    for i, (dv, pv) in enumerate(zip(dqn_vals, ppo_vals)):
        ax.text(i - width / 2, dv + 0.02, f"{dv:.2f}", ha="center",
                va="bottom", fontsize=9)
        ax.text(i + width / 2, pv + 0.02, f"{pv:.2f}", ha="center",
                va="bottom", fontsize=9)

    # Panel B: PPO methods comparison (CBS vs DT vs RV)
    ax = axes[1]
    methods = ["CBS", "DT", "RV"]
    method_stabs = [ppo_cbs, ppo_dt, ppo_rv]
    method_colors = [COLORS["CBS"], COLORS["DT"],
                     COLORS["RV"]]

    bra_vals = [float(s.get("BRA", 0)) for s in method_stabs]

    ax.bar(np.arange(len(methods)), bra_vals, 0.5,
           color=method_colors, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_ylabel("BRA")
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) PPO Methods (CartPole)", fontsize=9)

    for i, v in enumerate(bra_vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    savefig(fig, "fig_policy_family_cartpole")
    print("  Generated fig_policy_family_cartpole")


if __name__ == "__main__":
    main()
