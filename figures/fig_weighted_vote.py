#!/usr/bin/env python
"""
Weighted RuleVote comparison figures.

Generates:
  fig_weighted_vote_comparison.{pdf,png}  — Main 4-env comparison (F1, worst-R, BRA)
  fig_vote_size_sensitivity.{pdf,png}        — B sensitivity for CP + LL

Usage:
    python figures/fig_weighted_vote.py
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from figures._style import apply_style, savefig, COL2

# ── Configuration ────────────────────────────────────────────────────

ENVS = ["MountainCar-v0", "CartPole-v1", "LunarLander-v3",
        "MiniGrid-Dynamic-Obstacles-8x8-v0"]
ENV_SHORT = {"MountainCar-v0": "MC", "CartPole-v1": "CP",
             "LunarLander-v3": "LL",
             "MiniGrid-Dynamic-Obstacles-8x8-v0": "MiniGrid"}
ENV_TAGS = {"MountainCar-v0": "mountaincar_v0",
            "CartPole-v1": "cartpole_v1",
            "LunarLander-v3": "lunarlander_v3",
            "MiniGrid-Dynamic-Obstacles-8x8-v0": "minigrid_dynamic_obstacles_8x8_v0"}

RESULTS_ROOT = os.path.join("experiments", "results", "weighted_vote")

# Colors for methods
C_CBS = "#648FFF"
C_VANILLA = "#35A86B"
C_WEIGHTED = "#DC3220"

METHODS = ["CBS", "B3_vote", "best_weighted"]
METHOD_COLORS = {"CBS": C_CBS, "B3_vote": C_VANILLA, "best_weighted": C_WEIGHTED}
METHOD_LABELS = {"CBS": "CBS", "B3_vote": "RV", "best_weighted": "W-RV"}


def load_main(env):
    tag = ENV_TAGS[env]
    path = os.path.join(RESULTS_ROOT, tag, "main_comparison.json")
    with open(path) as f:
        return json.load(f)


def load_bsens(env):
    tag = ENV_TAGS[env]
    path = os.path.join(RESULTS_ROOT, tag, "b_sensitivity.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def select_best_weighted(results):
    """Select best weighted config by F1 delta with BRA constraint."""
    vanilla_f1 = np.mean([m["fidelity"]["f1"]
                          for m in results["B3_vote"]["per_run"]])
    vanilla_bra = results["B3_vote"]["stability"]["BRA"]
    best_tag = None
    best_delta = -1e9
    for key, val in results.items():
        if not key.startswith("weighted_"):
            continue
        tag = key[len("weighted_"):]
        f1 = np.mean([m["fidelity"]["f1"] for m in val["per_run"]])
        bra = val["stability"]["BRA"]
        if bra < vanilla_bra - 0.005:
            continue
        delta = f1 - vanilla_f1
        if delta > best_delta:
            best_delta = delta
            best_tag = tag
    return best_tag or "f1_b1"


def extract(per_run, metric):
    vals = []
    for r in per_run:
        if metric in ("f1", "worst_action_recall"):
            vals.append(r["fidelity"][metric])
        elif metric == "E_CR":
            vals.append(r["deployment"]["E_CR"])
    return np.array(vals)


# ── Figure 1: Main Comparison ────────────────────────────────────────

def fig_main_comparison():
    apply_style()

    metrics = [("f1", "Macro-F1"), ("worst_action_recall", "Worst-Action Recall")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(COL2, 2.4))

    for col, (metric, ylabel) in enumerate(metrics):
        ax = axes[col]
        x_pos = np.arange(len(ENVS))
        bar_w = 0.25

        for i, method in enumerate(METHODS):
            means, stds = [], []
            for env in ENVS:
                data = load_main(env)
                results = data["results"]
                if method == "best_weighted":
                    best_tag = select_best_weighted(results)
                    key = f"weighted_{best_tag}"
                else:
                    key = method
                vals = extract(results[key]["per_run"], metric)
                means.append(vals.mean())
                stds.append(vals.std())

            ax.bar(x_pos + (i - 1) * bar_w, means, bar_w,
                   yerr=stds, color=METHOD_COLORS[method],
                   edgecolor="white", linewidth=0.5, capsize=2,
                   label=METHOD_LABELS[method] if col == 0 else None)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([ENV_SHORT[e] for e in ENVS])
        ax.set_ylabel(ylabel)

    axes[0].legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    savefig(fig, "fig_weighted_vote_comparison")
    plt.close(fig)
    print("  Saved fig_weighted_vote_comparison")


# ── Figure 2: BRA Comparison ────────────────────────────────────────

def fig_bra_comparison():
    apply_style()

    fig, ax = plt.subplots(1, 1, figsize=(COL2 * 0.55, 2.4))
    x_pos = np.arange(len(ENVS))
    bar_w = 0.25

    for i, method in enumerate(METHODS):
        bras = []
        for env in ENVS:
            data = load_main(env)
            results = data["results"]
            if method == "best_weighted":
                best_tag = select_best_weighted(results)
                key = f"weighted_{best_tag}"
            else:
                key = method
            bras.append(results[key]["stability"]["BRA"])

        ax.bar(x_pos + (i - 1) * bar_w, bras, bar_w,
               color=METHOD_COLORS[method], edgecolor="white",
               linewidth=0.5, label=METHOD_LABELS[method])

    ax.set_xticks(x_pos)
    ax.set_xticklabels([ENV_SHORT[e] for e in ENVS])
    ax.set_ylabel("BRA")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, "fig_agreement_comparison")
    plt.close(fig)
    print("  Saved fig_agreement_comparison")


# ── Figure 3: B Sensitivity ────────────────────────────────────────

def fig_b_sensitivity():
    apply_style()

    b_envs = ["CartPole-v1", "LunarLander-v3"]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.4))

    for col, env in enumerate(b_envs):
        ax = axes[col]
        bsens = load_bsens(env)
        if bsens is None:
            ax.set_title(ENV_SHORT[env])
            continue

        b_vals = bsens["B_values"]
        best_tag = bsens["best_weighted_tag"]

        van_f1, van_wr, van_bra = [], [], []
        w_f1, w_wr, w_bra = [], [], []
        for bv in b_vals:
            bkey = f"B{bv}"
            vr = bsens["results"][bkey]["vanilla"]["per_run"]
            van_f1.append(np.mean([r["fidelity"]["f1"] for r in vr]))
            van_wr.append(np.mean([r["fidelity"]["worst_action_recall"] for r in vr]))
            van_bra.append(bsens["results"][bkey]["vanilla"]["stability"]["BRA"])

            wkey = f"weighted_{best_tag}"
            wr = bsens["results"][bkey][wkey]["per_run"]
            w_f1.append(np.mean([r["fidelity"]["f1"] for r in wr]))
            w_wr.append(np.mean([r["fidelity"]["worst_action_recall"] for r in wr]))
            w_bra.append(bsens["results"][bkey][wkey]["stability"]["BRA"])

        x = np.arange(len(b_vals))
        bar_w = 0.35
        ax.bar(x - bar_w / 2, van_f1, bar_w, color=C_VANILLA,
             edgecolor="white", linewidth=0.5, label="RV")
        ax.bar(x + bar_w / 2, w_f1, bar_w, color=C_WEIGHTED,
             edgecolor="white", linewidth=0.5, label="W-RV")

        # Plot BRA as line on twin axis
        ax2 = ax.twinx()
        ax2.plot(x, van_bra, "o--", color=C_VANILLA, markersize=4, alpha=0.7)
        ax2.plot(x, w_bra, "s--", color=C_WEIGHTED, markersize=4, alpha=0.7)
        ax2.set_ylabel("BRA", fontsize=9)
        ax2.set_ylim(0.7, 1.0)

        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in b_vals])
        ax.set_xlabel("B (ensemble size)")
        ax.set_ylabel("Macro-F1")
        ax.set_title(ENV_SHORT[env])
        if col == 0:
            ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    savefig(fig, "fig_vote_size_sensitivity")
    plt.close(fig)
    print("  Saved fig_vote_size_sensitivity")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    fig_main_comparison()
    fig_bra_comparison()
    fig_b_sensitivity()
    print("\n  All Experiment A figures generated.")


if __name__ == "__main__":
    main()
