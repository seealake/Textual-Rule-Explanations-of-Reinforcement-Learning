#!/usr/bin/env python
"""Boundary-crossing case study (CartPole).

Publication-quality 2D scatter showing rule state regions, interpolation
paths, and boundary-crossing markers.

Generates the visualization by re-running the analysis on CartPole-v1.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from itertools import combinations
from pathlib import Path
from figures._style import apply_style, savefig, COL2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent

# Feature names for CartPole
FEAT_NAMES = ["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"]

# Colors
RULE_A_COLOR = "#648FFF"   # blue
RULE_B_COLOR = "#DC3220"   # red
PATH_COLORS = ["#35A86B", "#FE6100", "#785EF0"]  # green, orange, purple


def collect_data():
    """Collect replay, fit CBS, find mergeable pairs with good visual spread."""
    from reproduction.collect_replay import collect_replay
    from experiments.rule_matching import (
        canonicalize_rules, rule_similarity_threshold_aware,
    )
    from experiments.consensus_merge import run_cbs_on_data
    from stable_baselines3 import DQN

    env_name = "CartPole-v1"
    model_path = str(ROOT / "reproduction" / "models" / "dqn_cartpole_v1.zip")
    dqn = DQN.load(model_path)

    data = collect_replay(env_name, model_path=model_path,
                          num_transitions=10000, seed=0)
    states = data["states"]
    actions = data["actions"]

    rng = np.random.RandomState(42)
    n = len(states)
    all_cbs = []
    all_rules = []
    for _ in range(5):
        idx = rng.choice(n, size=int(n * 0.8), replace=False)
        cbs, rules = run_cbs_on_data(states[idx], actions[idx], env_name)
        all_cbs.append(cbs)
        all_rules.append(rules)

    # Find mergeable pairs across runs
    pairs = []
    for i in range(len(all_rules)):
        for j in range(i + 1, len(all_rules)):
            for ri in all_rules[i]:
                for rj in all_rules[j]:
                    if ri.action != rj.action:
                        continue
                    sim = rule_similarity_threshold_aware(
                        ri, rj, lambda1=0.6, lambda2=0.4)
                    if 0.70 <= sim <= 0.95:
                        pairs.append((i, j, ri, rj, sim))

    if not pairs:
        print("  No suitable pairs found")
        return None

    # Helper: find states matching a rule (bounds-only, no CBS predict)
    def matching_states(rule, all_states):
        mask = np.ones(len(all_states), dtype=bool)
        for pred in rule.predicates:
            fidx = pred.feature_idx
            if pred.lower_bound is not None:
                mask &= all_states[:, fidx] >= pred.lower_bound
            if pred.upper_bound is not None:
                mask &= all_states[:, fidx] <= pred.upper_bound
        return np.where(mask)[0]

    # Score each pair by visual spread across all 2D projections
    feat_pairs = list(combinations(range(4), 2))
    best_score = -1
    best_result = None

    for run_i, run_j, ri, rj, sim in pairs:
        idx_a = matching_states(ri, states)
        idx_b = matching_states(rj, states)
        if len(idx_a) < 30 or len(idx_b) < 30:
            continue

        # Test boundary crossings
        n_test = min(15, len(idx_a), len(idx_b))
        crossings = 0
        for k in range(n_test):
            pa = states[idx_a[rng.randint(len(idx_a))]]
            pb = states[idx_b[rng.randint(len(idx_b))]]
            alphas = np.linspace(0, 1, 21)
            path_pts = np.array([(1 - a) * pa + a * pb for a in alphas])
            path_acts, _ = dqn.predict(path_pts, deterministic=True)
            if any(path_acts[t] != path_acts[t - 1] for t in range(1, len(path_acts))):
                crossings += 1
        rate = crossings / n_test
        if rate < 0.2:
            continue

        # Find best 2D projection
        for dim_x, dim_y in feat_pairs:
            centroid_a = states[idx_a][:, [dim_x, dim_y]].mean(axis=0)
            centroid_b = states[idx_b][:, [dim_x, dim_y]].mean(axis=0)
            dist = np.linalg.norm(centroid_a - centroid_b)
            # Penalize if one cluster has very low spread
            spread_a = states[idx_a][:, [dim_x, dim_y]].std()
            spread_b = states[idx_b][:, [dim_x, dim_y]].std()
            min_spread = min(spread_a, spread_b)
            score = dist * min_spread * rate

            if score > best_score:
                best_score = score
                best_result = (idx_a, idx_b, ri, rj, sim, rate,
                               dim_x, dim_y)

    if best_result is None:
        print("  No good pair found for visualization")
        return None

    idx_a, idx_b, ri, rj, sim, rate, dim_x, dim_y = best_result
    return states, dqn, idx_a, idx_b, ri, rj, sim, rate, dim_x, dim_y, rng


def main():
    apply_style()

    result = collect_data()
    if result is None:
        return

    states, dqn, idx_a, idx_b, ri, rj, sim, rate, dim_x, dim_y, rng = result
    print(f"  Best pair: sim={sim:.2f}, crossing={rate:.0%}, "
          f"dims=({FEAT_NAMES[dim_x]}, {FEAT_NAMES[dim_y]}), "
          f"|A|={len(idx_a)}, |B|={len(idx_b)}")

    fig, ax = plt.subplots(1, 1, figsize=(COL2 * 0.5, COL2 * 0.42))

    # Background: DQN decision regions
    dqn_preds, _ = dqn.predict(states, deterministic=True)
    unique_actions = np.unique(dqn_preds)
    bg_palette = ["#E8E8E8", "#D0D0D0"]  # neutral grays for background
    for a_idx, action in enumerate(unique_actions):
        mask = dqn_preds == action
        bg_idx = np.where(mask)[0]
        if len(bg_idx) > 700:
            bg_idx = rng.choice(bg_idx, 700, replace=False)
        ax.scatter(states[bg_idx, dim_x], states[bg_idx, dim_y],
                   c=bg_palette[a_idx % len(bg_palette)],
                   s=3.2, alpha=0.16, zorder=1, rasterized=True,
                   label=f"DQN action {action}" if a_idx < 2 else None)

    # Rule A states
    plot_a = idx_a if len(idx_a) <= 90 else rng.choice(idx_a, 90, replace=False)
    ax.scatter(states[plot_a, dim_x], states[plot_a, dim_y],
               c=RULE_A_COLOR, s=28, alpha=0.66, zorder=3,
               edgecolors="white", linewidth=0.4,
               label="Rule A")

    # Rule B states
    plot_b = idx_b if len(idx_b) <= 90 else rng.choice(idx_b, 90, replace=False)
    ax.scatter(states[plot_b, dim_x], states[plot_b, dim_y],
               c=RULE_B_COLOR, s=28, alpha=0.66, zorder=3,
               edgecolors="white", linewidth=0.4,
               label="Rule B")

    # Draw 3 interpolation paths with well-separated endpoints
    # Pick endpoints from opposite edges of each cluster for better visibility
    centroid_a = states[idx_a][:, [dim_x, dim_y]].mean(axis=0)
    centroid_b = states[idx_b][:, [dim_x, dim_y]].mean(axis=0)
    direction = centroid_b - centroid_a
    direction /= (np.linalg.norm(direction) + 1e-8)

    # Project rule A states onto direction → pick from spread
    proj_a = states[idx_a][:, [dim_x, dim_y]] @ direction
    proj_b = states[idx_b][:, [dim_x, dim_y]] @ direction

    path_handles = []
    for p_idx in range(3):
        # Stratified sampling: pick from different parts of each cluster
        frac = [0.2, 0.5, 0.8][p_idx]
        a_sorted = np.argsort(proj_a)
        b_sorted = np.argsort(proj_b)
        pa = states[idx_a[a_sorted[int(frac * len(a_sorted))]]]
        pb = states[idx_b[b_sorted[int((1 - frac) * len(b_sorted))]]]

        alphas_interp = np.linspace(0, 1, 25)
        path = np.array([(1 - a) * pa + a * pb for a in alphas_interp])
        path_acts, _ = dqn.predict(path, deterministic=True)

        color = PATH_COLORS[p_idx]
        ax.plot(path[:, dim_x], path[:, dim_y],
                color=color, linewidth=2.0, alpha=0.9, zorder=4)

        # Arrowhead at midpoint
        mid = len(path) // 2
        ax.annotate("", xy=(path[mid + 1, dim_x], path[mid + 1, dim_y]),
                    xytext=(path[mid, dim_x], path[mid, dim_y]),
                    arrowprops=dict(arrowstyle="->", color=color,
                                   lw=1.8, mutation_scale=12),
                    zorder=5)

        # Start/end markers
        ax.scatter(path[0, dim_x], path[0, dim_y],
                   marker="o", c=color, s=40, zorder=5,
                   edgecolors="black", linewidth=0.6)
        ax.scatter(path[-1, dim_x], path[-1, dim_y],
                   marker="s", c=color, s=40, zorder=5,
                   edgecolors="black", linewidth=0.6)

        # Mark boundary crossings with bold X
        n_crosses = 0
        for t in range(1, len(path_acts)):
            if path_acts[t] != path_acts[t - 1]:
                ax.scatter(path[t, dim_x], path[t, dim_y],
                           marker="X", c="black", s=50, zorder=6,
                           edgecolors="yellow", linewidth=0.6)
                n_crosses += 1

        path_handles.append(
            Line2D([0], [0], color=color, linewidth=1.8,
                   label=f"Path {p_idx + 1}" +
                         (f" ({n_crosses} cross)" if n_crosses else "")))

    ax.set_xlabel(FEAT_NAMES[dim_x])
    ax.set_ylabel(FEAT_NAMES[dim_y])
    ax.set_title(
        f"CartPole: mergeable pair (sim={sim:.2f}, crossing rate={rate:.0%})",
        fontsize=9)

    # Compact legend
    legend_handles = [
        mpatches.Patch(color=RULE_A_COLOR, alpha=0.75, label="Rule A states"),
        mpatches.Patch(color=RULE_B_COLOR, alpha=0.75, label="Rule B states"),
        *path_handles,
        Line2D([0], [0], marker="X", color="w", markerfacecolor="black",
               markersize=7, markeredgecolor="yellow", markeredgewidth=0.5,
               label="Boundary crossing"),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=9,
              framealpha=0.92, ncol=2, handletextpad=0.4)

    plt.tight_layout()
    savefig(fig, "fig_boundary_case_study")


if __name__ == "__main__":
    main()
