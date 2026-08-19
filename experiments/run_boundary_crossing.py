#!/usr/bin/env python
"""
Boundary-crossing diagnosis

Proves that merged "consensus rules" produced by median/mean aggregation
create new rule boundaries that cross the DQN's actual decision boundary.

Protocol:
  For each matched rule pair:
    1. Identify states covered by each member rule.
    2. Sample representative points from each rule's state region.
    3. Linearly interpolate between rule regions.
    4. Query the DQN along the interpolation path.
    5. Detect action flips (boundary crossings) and low-density zones.
  
  Compare:
    - Pairs that WOULD be merged at default ρ=0.8 ("mergeable")
    - Pairs that would NOT be merged at ρ=0.8 ("non-mergeable")

Reports:
  - boundary_crossing_rate: fraction of interpolation paths with action flip
  - midpoint_action_mismatch: merged rule action ≠ DQN action at midpoint
  - low_density_bridge_rate: fraction of midpoints in low-density replay region

Usage:
    python experiments/run_boundary_crossing.py --env MountainCar-v0
    python experiments/run_boundary_crossing.py --env CartPole-v1
    python experiments/run_boundary_crossing.py --env LunarLander-v3
    python experiments/run_boundary_crossing.py --env all
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.neighbors import NearestNeighbors

from reproduction.cbs import CBSPipeline
from reproduction.collect_replay import collect_replay, ENV_FEATURE_NAMES
from experiments.rule_matching import (
    canonicalize_rules,
    rule_similarity_threshold_aware,
)
from experiments.consensus_merge import (
    run_cbs_on_data,
    _match_rules_across_runs,
    merge_rule_group,
    aggregate_thresholds,
    make_consensus_pipeline,
)
from experiments.run_stress_test import _serialize

# ── Constants ─────────────────────────────────────────────────────────
ENVS = ["MountainCar-v0", "CartPole-v1", "LunarLander-v3"]
N_BOOTSTRAP = 5
SUBSAMPLE_FRACTION = 0.8
DEFAULT_RHO = 0.8
CONSERVATIVE_RHO = 0.95  # more conservative threshold for "non-mergeable"
DEFAULT_LAMBDA1 = 0.6
DEFAULT_LAMBDA2 = 0.4
N_INTERPOLATION_STEPS = 21  # 0.0, 0.05, ..., 1.0
N_SAMPLE_PAIRS = 30  # number of representative point pairs per rule pair
LOW_DENSITY_PERCENTILE = 15  # top 15% of kNN distances = low density

FEATURE_NAMES_MAP = {
    "MountainCar-v0": ["position", "velocity"],
    "CartPole-v1": ["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"],
    "LunarLander-v3": ["x_pos", "y_pos", "x_vel", "y_vel",
                        "angle", "ang_vel", "left_leg", "right_leg"],
}


def get_model_path(env_name):
    tag = env_name.replace("-", "_").lower()
    return f"reproduction/models/dqn_{tag}.zip"


def _states_matching_rule(rule, cbs_pipeline, replay_states):
    """Return indices of replay states matching all predicates of a rule."""
    encoded = cbs_pipeline._encode_states(replay_states)
    mask = np.ones(len(replay_states), dtype=bool)
    for pred in rule.predicates:
        f = pred.feature_idx
        mask &= np.abs(encoded[:, f] - pred.level) < 0.01
    return np.where(mask)[0]


def _build_density_model(replay_states, k=10):
    """Build a kNN density model for low-density detection."""
    nn = NearestNeighbors(n_neighbors=min(k, len(replay_states) - 1))
    nn.fit(replay_states)
    distances, _ = nn.kneighbors(replay_states)
    density_threshold = np.percentile(distances[:, -1],
                                      100 - LOW_DENSITY_PERCENTILE)
    return nn, density_threshold


def analyze_rule_pair(rule_a, rule_b, run_a, run_b, all_cbs,
                      replay_states, dqn_model, nn_model, density_threshold,
                      merged_rule=None, rng=None):
    """Analyze a single pair of rules for boundary crossing.
    
    Returns dict with crossing diagnostics.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    cbs_a = all_cbs[run_a]
    cbs_b = all_cbs[run_b]

    # Get states covered by each rule
    idx_a = _states_matching_rule(rule_a, cbs_a, replay_states)
    idx_b = _states_matching_rule(rule_b, cbs_b, replay_states)

    states_a = replay_states[idx_a] if len(idx_a) > 0 else None
    states_b = replay_states[idx_b] if len(idx_b) > 0 else None

    if states_a is None or states_b is None or len(states_a) < 2 or len(states_b) < 2:
        return None

    # Sample representative point pairs
    n_pairs = min(N_SAMPLE_PAIRS, len(states_a), len(states_b))
    sample_a = states_a[rng.choice(len(states_a), n_pairs, replace=True)]
    sample_b = states_b[rng.choice(len(states_b), n_pairs, replace=True)]

    # Interpolation alphas
    alphas = np.linspace(0, 1, N_INTERPOLATION_STEPS)

    # Per-path analysis
    n_paths_with_crossing = 0
    midpoint_mismatches = 0
    midpoint_low_density = 0
    total_crossings = 0
    total_interpolation_points = 0
    low_density_points = 0

    for i in range(n_pairs):
        p_a = sample_a[i]
        p_b = sample_b[i]

        # Interpolation path
        path = np.array([(1 - a) * p_a + a * p_b for a in alphas])

        # Query DQN along path
        path_actions, _ = dqn_model.predict(path, deterministic=True)

        # Count boundary crossings (action flips along path)
        flips = np.sum(np.diff(path_actions) != 0)
        if flips > 0:
            n_paths_with_crossing += 1
        total_crossings += flips

        # Midpoint analysis (alpha=0.5)
        mid_idx = N_INTERPOLATION_STEPS // 2
        mid_action = int(path_actions[mid_idx])
        if merged_rule is not None:
            if mid_action != merged_rule.action:
                midpoint_mismatches += 1

        # Low-density check along path
        path_dists, _ = nn_model.kneighbors(path)
        path_low = np.sum(path_dists[:, -1] > density_threshold)
        low_density_points += path_low
        total_interpolation_points += len(path)

        # Midpoint low-density
        if path_dists[mid_idx, -1] > density_threshold:
            midpoint_low_density += 1

    return {
        "n_states_a": len(states_a),
        "n_states_b": len(states_b),
        "n_pairs": n_pairs,
        "boundary_crossing_rate": round(n_paths_with_crossing / n_pairs, 4),
        "mean_crossings_per_path": round(total_crossings / n_pairs, 4),
        "midpoint_action_mismatch_rate": round(midpoint_mismatches / n_pairs, 4),
        "midpoint_low_density_rate": round(midpoint_low_density / n_pairs, 4),
        "path_low_density_frac": round(
            low_density_points / total_interpolation_points, 4)
            if total_interpolation_points > 0 else 0.0,
        "action_a": int(rule_a.action),
        "action_b": int(rule_b.action),
    }


def generate_path_plots(pair_results, replay_states, dqn_model,
                        all_cbs, env_name, out_dir):
    """Generate interpolation path visualization for low-dim environments."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("    matplotlib not available, skipping path plots")
        return []

    n_features = replay_states.shape[1]
    if n_features > 4:
        print("    Skipping path plots (>4 features)")
        return []

    feat_names = FEATURE_NAMES_MAP.get(env_name,
                                       [f"f{i}" for i in range(n_features)])

    if env_name == "MountainCar-v0":
        plot_dims = [(0, 1)]
    elif env_name == "CartPole-v1":
        plot_dims = [(0, 2)]
    else:
        plot_dims = [(0, 1)]

    # Select up to 4 interesting pairs: 2 mergeable with crossings, 2 non-mergeable
    mergeable_crossing = [p for p in pair_results
                          if p["category"] == "mergeable"
                          and p["boundary_crossing_rate"] > 0.3]
    non_mergeable = [p for p in pair_results
                     if p["category"] == "non_mergeable"]

    case_studies = []
    for p in mergeable_crossing[:2]:
        case_studies.append(p)
    for p in non_mergeable[:2]:
        case_studies.append(p)

    if not case_studies:
        return []

    plot_paths = []
    rng = np.random.RandomState(42)

    for c_idx, pair in enumerate(case_studies):
        rule_a = pair["_rule_a"]
        rule_b = pair["_rule_b"]
        run_a = pair["_run_a"]
        run_b = pair["_run_b"]

        cbs_a = all_cbs[run_a]
        cbs_b = all_cbs[run_b]

        idx_a = _states_matching_rule(rule_a, cbs_a, replay_states)
        idx_b = _states_matching_rule(rule_b, cbs_b, replay_states)

        for dim_x, dim_y in plot_dims:
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))

            # Background: DQN decision regions (sampled)
            dqn_preds, _ = dqn_model.predict(replay_states, deterministic=True)
            scatter = ax.scatter(replay_states[:, dim_x],
                                 replay_states[:, dim_y],
                                 c=dqn_preds, cmap="Set3", s=1, alpha=0.15)

            # Rule A states
            if len(idx_a) > 0:
                ax.scatter(replay_states[idx_a, dim_x],
                           replay_states[idx_a, dim_y],
                           c="blue", s=10, alpha=0.5, label="Rule A states",
                           edgecolors="navy", linewidth=0.3)

            # Rule B states
            if len(idx_b) > 0:
                ax.scatter(replay_states[idx_b, dim_x],
                           replay_states[idx_b, dim_y],
                           c="red", s=10, alpha=0.5, label="Rule B states",
                           edgecolors="darkred", linewidth=0.3)

            # Draw interpolation paths (3 examples)
            n_draw = min(3, len(idx_a), len(idx_b))
            colors_path = ["green", "orange", "purple"]
            for p_idx in range(n_draw):
                p_a = replay_states[idx_a[rng.randint(len(idx_a))]]
                p_b = replay_states[idx_b[rng.randint(len(idx_b))]]
                alphas = np.linspace(0, 1, N_INTERPOLATION_STEPS)
                path = np.array([(1 - a) * p_a + a * p_b for a in alphas])
                path_actions, _ = dqn_model.predict(path, deterministic=True)

                ax.plot(path[:, dim_x], path[:, dim_y],
                        color=colors_path[p_idx % len(colors_path)],
                        linewidth=1.5, alpha=0.8,
                        label=f"path {p_idx}")

                # Mark action flips
                for j in range(1, len(path_actions)):
                    if path_actions[j] != path_actions[j - 1]:
                        ax.scatter(path[j, dim_x], path[j, dim_y],
                                   marker="x", c="black", s=50, zorder=5)

            ax.set_xlabel(feat_names[dim_x])
            ax.set_ylabel(feat_names[dim_y])
            cat = pair["category"]
            bcr = pair["boundary_crossing_rate"]
            ax.set_title(
                f"{cat} pair — crossing_rate={bcr:.2f}\n"
                f"action={pair['action_a']}, sim={pair.get('similarity', 0):.2f}"
            )
            ax.legend(fontsize=7, loc="best")

            fname = (f"boundary_crossing_{env_name.lower().replace('-','_')}"
                     f"_case{c_idx}_{cat}_{dim_x}v{dim_y}.png")
            fpath = os.path.join(out_dir, fname)
            fig.tight_layout()
            fig.savefig(fpath, dpi=200)
            plt.close(fig)
            plot_paths.append(fpath)
            print(f"    Saved plot: {fname}")

    return plot_paths


def run_boundary_crossing(env_name):
    """Run boundary crossing analysis for one environment."""
    print(f"\n{'='*70}")
    print(f"  Boundary Crossing Diagnosis: {env_name}")
    print(f"{'='*70}")

    env_tag = env_name.replace("-", "_").lower()
    model_path = get_model_path(env_name)

    from stable_baselines3 import DQN
    dqn_model = DQN.load(model_path)

    # Collect replay
    print("  Collecting replay (seed=0)...")
    data = collect_replay(
        env_name=env_name, model_path=model_path,
        num_transitions=10000, seed=0, deterministic=True,
    )
    replay_states = data["states"]
    replay_actions = data["actions"]
    print(f"  Replay: {len(replay_states)} transitions")

    # Build density model
    nn_model, density_threshold = _build_density_model(replay_states)

    # Build CBS subsamples
    print(f"  Building {N_BOOTSTRAP} CBS subsamples...")
    rng = np.random.RandomState(42)
    n_total = len(replay_states)

    all_cbs = []
    all_rules = []

    for i in range(N_BOOTSTRAP):
        idx = rng.choice(n_total, size=int(n_total * SUBSAMPLE_FRACTION),
                         replace=False)
        s = replay_states[idx]
        a = replay_actions[idx]
        cbs, rules = run_cbs_on_data(s, a, env_name)
        all_cbs.append(cbs)
        all_rules.append(rules)

    # Collect rule pairs — categorized as mergeable vs non-mergeable
    print("  Computing pairwise similarities...")
    pair_analyses = []
    analysis_rng = np.random.RandomState(123)

    level_values = all_cbs[0].level_values_
    level_labels = all_cbs[0].level_labels_

    # Build all cross-run same-action pairs
    for run_i in range(N_BOOTSTRAP):
        for run_j in range(run_i + 1, N_BOOTSTRAP):
            for rule_a in all_rules[run_i]:
                for rule_b in all_rules[run_j]:
                    if rule_a.action != rule_b.action:
                        continue

                    sim = rule_similarity_threshold_aware(
                        rule_a, rule_b,
                        lambda1=DEFAULT_LAMBDA1,
                        lambda2=DEFAULT_LAMBDA2,
                    )

                    if sim >= DEFAULT_RHO:
                        category = "mergeable"
                    elif sim < CONSERVATIVE_RHO and sim >= 0.5:
                        category = "non_mergeable"
                    else:
                        continue  # skip very dissimilar pairs

                    # Create merged rule for midpoint action check
                    merged = merge_rule_group(
                        [rule_a, rule_b], level_values, level_labels)

                    result = analyze_rule_pair(
                        rule_a, rule_b, run_i, run_j,
                        all_cbs, replay_states, dqn_model,
                        nn_model, density_threshold,
                        merged_rule=merged, rng=analysis_rng,
                    )

                    if result is not None:
                        result["category"] = category
                        result["similarity"] = round(sim, 4)
                        result["run_a"] = run_i
                        result["run_b"] = run_j
                        # Store rule refs for plotting (not serialized)
                        result["_rule_a"] = rule_a
                        result["_rule_b"] = rule_b
                        result["_run_a"] = run_i
                        result["_run_b"] = run_j
                        pair_analyses.append(result)

    # Cap the number of pairs to analyze (prioritize diversity)
    mergeable = [p for p in pair_analyses if p["category"] == "mergeable"]
    non_mergeable = [p for p in pair_analyses if p["category"] == "non_mergeable"]

    MAX_PER_CAT = 50
    if len(mergeable) > MAX_PER_CAT:
        analysis_rng.shuffle(mergeable)
        mergeable = mergeable[:MAX_PER_CAT]
    if len(non_mergeable) > MAX_PER_CAT:
        analysis_rng.shuffle(non_mergeable)
        non_mergeable = non_mergeable[:MAX_PER_CAT]

    all_pairs = mergeable + non_mergeable
    print(f"  Analyzed pairs: {len(mergeable)} mergeable, "
          f"{len(non_mergeable)} non-mergeable")

    # Compute summary statistics per category
    def _summarize(pairs):
        if not pairs:
            return {}
        return {
            "count": len(pairs),
            "mean_boundary_crossing_rate": round(
                float(np.mean([p["boundary_crossing_rate"] for p in pairs])), 4),
            "mean_crossings_per_path": round(
                float(np.mean([p["mean_crossings_per_path"] for p in pairs])), 4),
            "mean_midpoint_mismatch": round(
                float(np.mean([p["midpoint_action_mismatch_rate"] for p in pairs])), 4),
            "mean_midpoint_low_density": round(
                float(np.mean([p["midpoint_low_density_rate"] for p in pairs])), 4),
            "mean_path_low_density": round(
                float(np.mean([p["path_low_density_frac"] for p in pairs])), 4),
        }

    summary_mergeable = _summarize(mergeable)
    summary_non_mergeable = _summarize(non_mergeable)

    # Print comparison table
    print(f"\n  {'='*70}")
    print(f"  Boundary Crossing Results:")
    print(f"  {'Metric':<35} {'Mergeable':>15} {'Non-Mergeable':>15}")
    print(f"  {'-'*65}")
    if summary_mergeable and summary_non_mergeable:
        for metric in ["count", "mean_boundary_crossing_rate",
                        "mean_crossings_per_path", "mean_midpoint_mismatch",
                        "mean_midpoint_low_density", "mean_path_low_density"]:
            m_val = summary_mergeable.get(metric, "N/A")
            n_val = summary_non_mergeable.get(metric, "N/A")
            print(f"  {metric:<35} {str(m_val):>15} {str(n_val):>15}")
    elif summary_mergeable:
        for metric, val in summary_mergeable.items():
            print(f"  {metric:<35} {str(val):>15} {'N/A':>15}")

    # Generate plots
    out_dir = f"experiments/results/{env_tag}"
    os.makedirs(out_dir, exist_ok=True)
    plot_paths = generate_path_plots(
        all_pairs, replay_states, dqn_model,
        all_cbs, env_name, out_dir)

    # Serialize for JSON (remove internal references)
    serializable_pairs = []
    for p in all_pairs:
        sp = {k: v for k, v in p.items() if not k.startswith("_")}
        serializable_pairs.append(sp)

    # Save results
    output = {
        "schema_version": "boundary_crossing_v1",
        "env": env_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "n_bootstrap": N_BOOTSTRAP,
            "default_rho": DEFAULT_RHO,
            "conservative_rho": CONSERVATIVE_RHO,
            "n_interpolation_steps": N_INTERPOLATION_STEPS,
            "n_sample_pairs": N_SAMPLE_PAIRS,
            "low_density_percentile": LOW_DENSITY_PERCENTILE,
        },
        "summary": {
            "mergeable": summary_mergeable,
            "non_mergeable": summary_non_mergeable,
        },
        "n_pairs_analyzed": len(all_pairs),
        "pair_details": serializable_pairs,
        "plot_files": [os.path.basename(p) for p in plot_paths],
    }

    out_path = os.path.join(out_dir, "boundary_crossing.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Boundary Crossing Diagnosis")
    parser.add_argument("--env", default="all",
                        choices=ENVS + ["all"],
                        help="Environment to test")
    args = parser.parse_args()

    envs = ENVS if args.env == "all" else [args.env]
    for env in envs:
        run_boundary_crossing(env)


if __name__ == "__main__":
    main()
