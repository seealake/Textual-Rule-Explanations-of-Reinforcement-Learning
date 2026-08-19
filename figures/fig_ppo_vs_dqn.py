#!/usr/bin/env python
"""Cross-environment PPO vs DQN summary.

Side-by-side comparison of CBS stability (PPO vs DQN) across CartPole and
LunarLander, showing that explanation stability is algorithm-agnostic.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from figures._style import apply_style, savefig, load_json, COLORS, COL2, RESULTS_DIR


def main():
    apply_style()

    # ── Load all data ────────────────────────────────────────────────
    # CartPole
    cp_ppo = load_json(str(RESULTS_DIR / "cross_algo_comparison" / "ppo_cartpole_stress.json"))
    cp_dqn = load_json(str(RESULTS_DIR / "cartpole_v1" / "stress_test_results.json"))
    cp_ppo_cbs = cp_ppo.get("CBS", {}).get("stability", {})
    cp_dqn_raw = cp_dqn.get("cbs", {})
    cp_dqn_cbs = cp_dqn_raw.get("stability", cp_dqn_raw)

    # LunarLander
    ll_ppo = load_json(str(RESULTS_DIR / "cross_algo_comparison" / "ppo_lunarlander_stress.json"))
    ll_dqn = load_json(str(RESULTS_DIR / "lunarlander_v3" / "stress_test_results.json"))
    ll_ppo_cbs = ll_ppo.get("CBS", {}).get("stability", {})
    ll_dqn_raw = ll_dqn.get("cbs", {})
    ll_dqn_cbs = ll_dqn_raw.get("stability", ll_dqn_raw)

    metrics = ["GRS_weighted_jaccard", "GRS_threshold_aware", "TD", "BRA"]
    metric_short = ["GRS", "GRS-TA", "TD", "BRA"]

    fig, axes = plt.subplots(1, 2, figsize=(COL2 * 0.8, COL2 * 0.32), sharey=True)

    width = 0.3
    x = np.arange(len(metrics))

    # Panel A: CartPole
    ax = axes[0]
    dqn_vals = [float(cp_dqn_cbs.get(m, 0)) for m in metrics]
    ppo_vals = [float(cp_ppo_cbs.get(m, 0)) for m in metrics]
    ax.bar(x - width / 2, dqn_vals, width, label="DQN", color="#648FFF",
           edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x + width / 2, ppo_vals, width, label="PPO", color="#FE6100",
           edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_short)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title("(a) CartPole-v1", fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    for i, (dv, pv) in enumerate(zip(dqn_vals, ppo_vals)):
        ax.text(i - width / 2, dv + 0.02, f"{dv:.2f}", ha="center",
                va="bottom", fontsize=9)
        ax.text(i + width / 2, pv + 0.02, f"{pv:.2f}", ha="center",
                va="bottom", fontsize=9)

    # Panel B: LunarLander
    ax = axes[1]
    dqn_vals = [float(ll_dqn_cbs.get(m, 0)) for m in metrics]
    ppo_vals = [float(ll_ppo_cbs.get(m, 0)) for m in metrics]
    ax.bar(x - width / 2, dqn_vals, width, label="DQN", color="#648FFF",
           edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x + width / 2, ppo_vals, width, label="PPO", color="#FE6100",
           edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_short)
    ax.set_title("(b) LunarLander-v3", fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    for i, (dv, pv) in enumerate(zip(dqn_vals, ppo_vals)):
        ax.text(i - width / 2, dv + 0.02, f"{dv:.2f}", ha="center",
                va="bottom", fontsize=9)
        ax.text(i + width / 2, pv + 0.02, f"{pv:.2f}", ha="center",
                va="bottom", fontsize=9)

    fig.suptitle("CBS Stability: PPO vs DQN (Algorithm-Agnostic)", fontsize=9, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig_ppo_vs_dqn")
    print("  Generated fig_ppo_vs_dqn")


if __name__ == "__main__":
    main()
