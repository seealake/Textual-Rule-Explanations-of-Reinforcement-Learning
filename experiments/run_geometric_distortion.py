#!/usr/bin/env python
"""
Geometric distortion diagnosis

Directly measures whether consensus merge collapses because it merges rules
that cover physically disjoint state-space regions.

For each merged rule group:
  1. Collect all replay states satisfying each member rule.
  2. Compute geometric diagnostics:
     - Modality: n_peaks from 1D kernel density per feature
     - Connectivity: DBSCAN cluster count (n_connected)
     - Density gap: max kNN gap ratio within merged region
     - Action consistency: merged rule vs DQN on member states
     - Low-density bridge: fraction of midpoints in low-density zone
  3. Classify groups as "successful" vs "failed" merges based on
     action mismatch rate (> 0.15 → failed).
  4. Generate 2D scatter plots for low-dim environments (MountainCar/CartPole).

Usage:
    python experiments/run_geometric_distortion.py --env MountainCar-v0
    python experiments/run_geometric_distortion.py --env CartPole-v1
    python experiments/run_geometric_distortion.py --env LunarLander-v3
    python experiments/run_geometric_distortion.py --env all
"""
import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from reproduction.cbs import CBSPipeline
from reproduction.collect_replay import collect_replay, ENV_FEATURE_NAMES
from experiments.perturbations import load_replay_npz, generate_subsamples
from experiments.rule_matching import (
    CanonicalRule,
    CanonicalPredicate,
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
from experiments.run_stress_test import (
    EVAL_SEEDS,
    SUCCESS_THRESHOLDS,
    HELDOUT_SEED,
    _serialize,
)

# ── Constants ─────────────────────────────────────────────────────────
ENVS = ["MountainCar-v0", "CartPole-v1", "LunarLander-v3"]
N_BOOTSTRAP = 5
SUBSAMPLE_FRACTION = 0.8
DEFAULT_RHO = 0.8
DEFAULT_LAMBDA1 = 0.6
DEFAULT_LAMBDA2 = 0.4
DEFAULT_TAU = 0.7
ACTION_MISMATCH_THRESHOLD = 0.15  # above this → "failed merge"

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
    """Return indices of replay states that match all predicates in a rule.
    
    A state matches a predicate if its encoded value for that feature
    equals the predicate's level. We use the CBS pipeline's encoding.
    """
    encoded = cbs_pipeline._encode_states(replay_states)
    mask = np.ones(len(replay_states), dtype=bool)

    for pred in rule.predicates:
        f = pred.feature_idx
        # Match if encoded level is close enough (float comparison)
        encoded_col = encoded[:, f]
        mask &= np.abs(encoded_col - pred.level) < 0.01

    return np.where(mask)[0]


def _states_in_bounds(rule, replay_states):
    """Return indices of states within the continuous bounds of a rule.
    
    Uses lower_bound/upper_bound if available, otherwise falls back to
    level-based matching.
    """
    mask = np.ones(len(replay_states), dtype=bool)

    for pred in rule.predicates:
        f = pred.feature_idx
        vals = replay_states[:, f]
        if pred.lower_bound is not None and pred.upper_bound is not None:
            mask &= (vals >= pred.lower_bound) & (vals <= pred.upper_bound)
        # If no bounds, this predicate doesn't filter

    return np.where(mask)[0]


def count_modes_1d(values, bw_method="silverman"):
    """Count number of modes in a 1D distribution via KDE peak detection."""
    if len(values) < 5:
        return 1

    try:
        kde = gaussian_kde(values, bw_method=bw_method)
        x_grid = np.linspace(values.min(), values.max(), 200)
        density = kde(x_grid)
        peaks, _ = find_peaks(density, height=density.max() * 0.05)
        return max(1, len(peaks))
    except (np.linalg.LinAlgError, ValueError):
        return 1


def compute_dbscan_components(states, eps_quantile=0.3):
    """Count connected components via DBSCAN.
    
    eps is set to the median of the k-nearest-neighbor distances
    to be adaptive to data density.
    """
    if len(states) < 3:
        return 1

    k = min(5, len(states) - 1)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(states)
    distances, _ = nn.kneighbors(states)
    eps = np.quantile(distances[:, -1], eps_quantile)
    if eps < 1e-10:
        eps = 0.1

    db = DBSCAN(eps=eps, min_samples=max(2, k // 2))
    labels = db.fit_predict(states)
    n_clusters = len(set(labels) - {-1})
    return max(1, n_clusters)


def compute_knn_gap_ratio(states, k=5):
    """Compute max kNN distance gap ratio as a measure of fragmentation."""
    if len(states) < k + 1:
        return 0.0

    nn = NearestNeighbors(n_neighbors=min(k, len(states) - 1))
    nn.fit(states)
    distances, _ = nn.kneighbors(states)
    mean_knn = distances[:, -1]

    if len(mean_knn) < 2:
        return 0.0

    sorted_dists = np.sort(mean_knn)
    gaps = np.diff(sorted_dists)
    if sorted_dists[-1] < 1e-10:
        return 0.0

    max_gap_ratio = float(gaps.max() / sorted_dists[-1])
    return max_gap_ratio


def compute_low_density_bridge_rate(group_states_list, replay_states,
                                    n_interpolations=20, density_percentile=10):
    """Check if midpoints between member rule regions fall in low-density zones.
    
    For each pair of member rule state sets, sample midpoints via linear
    interpolation and check if they fall in low-density areas of the
    replay distribution.
    """
    if len(group_states_list) < 2:
        return 0.0

    # Compute density threshold from full replay
    k = min(10, len(replay_states) - 1)
    nn_full = NearestNeighbors(n_neighbors=k)
    nn_full.fit(replay_states)
    full_dists, _ = nn_full.kneighbors(replay_states)
    # Low density = kNN distance above the Nth percentile (inverted density)
    density_threshold = np.percentile(full_dists[:, -1],
                                      100 - density_percentile)

    low_density_count = 0
    total_midpoints = 0

    for i in range(len(group_states_list)):
        for j in range(i + 1, len(group_states_list)):
            s_i = group_states_list[i]
            s_j = group_states_list[j]
            if len(s_i) == 0 or len(s_j) == 0:
                continue

            # Sample midpoints
            n_pairs = min(n_interpolations, len(s_i), len(s_j))
            rng = np.random.RandomState(42)
            idx_i = rng.choice(len(s_i), n_pairs, replace=True)
            idx_j = rng.choice(len(s_j), n_pairs, replace=True)

            for alpha in [0.5]:
                midpoints = (1 - alpha) * s_i[idx_i] + alpha * s_j[idx_j]
                mid_dists, _ = nn_full.kneighbors(midpoints)
                low_density_count += np.sum(mid_dists[:, -1] > density_threshold)
                total_midpoints += len(midpoints)

    if total_midpoints == 0:
        return 0.0
    return float(low_density_count / total_midpoints)


def compute_action_consistency(merged_rule, member_states, dqn_model):
    """Compute action agreement between merged rule and DQN on member states."""
    if len(member_states) == 0:
        return 1.0

    dqn_actions, _ = dqn_model.predict(member_states, deterministic=True)
    merged_action = merged_rule.action
    agreement = float(np.mean(dqn_actions == merged_action))
    return agreement


def analyze_group(group, group_idx, all_cbs, replay_states, replay_actions,
                  dqn_model, env_name, n_bootstrap):
    """Analyze geometric properties of one matched rule group."""
    rules = [rule for _, rule in group]
    action = rules[0].action

    # Collect states matching each member rule
    member_states_list = []
    all_member_indices = set()

    for run_idx, rule in group:
        # Use the CBS pipeline from that run for encoding
        cbs = all_cbs[run_idx]
        indices = _states_matching_rule(rule, cbs, replay_states)
        member_states_list.append(replay_states[indices] if len(indices) > 0
                                  else np.empty((0, replay_states.shape[1])))
        all_member_indices.update(indices)

    # Also get states via continuous bounds (for merged rule reference)
    merged_level_values = all_cbs[0].level_values_
    merged_level_labels = all_cbs[0].level_labels_
    merged_rule = merge_rule_group(rules, merged_level_values, merged_level_labels)

    merged_indices = _states_in_bounds(merged_rule, replay_states)
    merged_states = replay_states[merged_indices] if len(merged_indices) > 0 \
        else np.empty((0, replay_states.shape[1]))

    all_member_states = replay_states[list(all_member_indices)] \
        if all_member_indices else np.empty((0, replay_states.shape[1]))

    # ------- Geometric diagnostics -------
    diagnostics = {
        "group_idx": group_idx,
        "action": int(action),
        "n_rules": len(rules),
        "n_distinct_runs": len(set(ri for ri, _ in group)),
    }

    # 1) Modality per feature
    n_features = replay_states.shape[1]
    feature_names = FEATURE_NAMES_MAP.get(env_name,
                                          [f"f{i}" for i in range(n_features)])
    modality = {}
    for f_idx in range(n_features):
        if len(all_member_states) > 5:
            n_modes = count_modes_1d(all_member_states[:, f_idx])
        else:
            n_modes = 0
        modality[feature_names[f_idx]] = n_modes

    diagnostics["modality_per_feature"] = modality
    diagnostics["max_modes"] = max(modality.values()) if modality else 0
    diagnostics["is_multimodal"] = diagnostics["max_modes"] > 1

    # 2) Connectivity (DBSCAN)
    if len(all_member_states) >= 3:
        # Normalize for DBSCAN
        state_min = replay_states.min(axis=0)
        state_range = replay_states.max(axis=0) - state_min
        state_range[state_range < 1e-10] = 1.0
        norm_states = (all_member_states - state_min) / state_range
        n_connected = compute_dbscan_components(norm_states)
    else:
        n_connected = len(all_member_states)

    diagnostics["n_connected_components"] = n_connected
    diagnostics["is_fragmented"] = n_connected > 1

    # 3) kNN gap ratio
    if len(all_member_states) >= 5:
        state_min = replay_states.min(axis=0)
        state_range = replay_states.max(axis=0) - state_min
        state_range[state_range < 1e-10] = 1.0
        norm_states = (all_member_states - state_min) / state_range
        gap_ratio = compute_knn_gap_ratio(norm_states)
    else:
        gap_ratio = 0.0
    diagnostics["knn_gap_ratio"] = round(gap_ratio, 4)

    # 4) Action consistency with DQN
    action_consistency = compute_action_consistency(
        merged_rule, all_member_states, dqn_model)
    diagnostics["action_consistency"] = round(action_consistency, 4)
    diagnostics["action_mismatch_rate"] = round(1.0 - action_consistency, 4)

    # Also check on merged region
    merged_consistency = compute_action_consistency(
        merged_rule, merged_states, dqn_model)
    diagnostics["merged_region_consistency"] = round(merged_consistency, 4)

    # 5) Low-density bridge
    non_empty = [s for s in member_states_list if len(s) > 0]
    if len(non_empty) >= 2:
        bridge_rate = compute_low_density_bridge_rate(
            non_empty, replay_states)
    else:
        bridge_rate = 0.0
    diagnostics["low_density_bridge_rate"] = round(bridge_rate, 4)

    # 6) State counts
    diagnostics["n_member_states"] = len(all_member_states)
    diagnostics["n_merged_region_states"] = len(merged_states)
    diagnostics["per_rule_state_counts"] = [len(s) for s in member_states_list]

    # Classify as successful or failed merge
    diagnostics["is_failed_merge"] = (
        diagnostics["action_mismatch_rate"] > ACTION_MISMATCH_THRESHOLD
    )

    return diagnostics


def generate_scatter_plots(group_diagnostics, replay_states, replay_actions,
                           all_cbs, all_rules, all_groups, dqn_model,
                           env_name, out_dir):
    """Generate 2D scatter plots for low-dimensional environments."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    matplotlib not available, skipping scatter plots")
        return []

    n_features = replay_states.shape[1]
    if n_features > 4:
        print("    Skipping scatter plots (>4 features)")
        return []

    # Select up to 4 case-study groups: 2 successful + 2 failed merges
    failed = [g for g in group_diagnostics if g["is_failed_merge"]]
    success = [g for g in group_diagnostics if not g["is_failed_merge"]]

    # Sort by action_mismatch_rate
    failed.sort(key=lambda g: g["action_mismatch_rate"], reverse=True)
    success.sort(key=lambda g: g["n_member_states"], reverse=True)

    case_studies = []
    for g in failed[:2]:
        case_studies.append(("failed", g))
    for g in success[:2]:
        case_studies.append(("success", g))

    if not case_studies:
        return []

    plot_paths = []
    feat_names = FEATURE_NAMES_MAP.get(env_name,
                                       [f"f{i}" for i in range(n_features)])

    # For MountainCar: plot dimensions 0,1
    # For CartPole: plot pole_angle (2) vs cart_pos (0)
    if env_name == "MountainCar-v0":
        plot_dims = [(0, 1)]
    elif env_name == "CartPole-v1":
        plot_dims = [(0, 2), (2, 3)]
    else:
        plot_dims = [(0, 1)]

    for case_label, diag in case_studies:
        group_idx = diag["group_idx"]
        group = all_groups[group_idx]
        rules = [rule for _, rule in group]
        merged_level_values = all_cbs[0].level_values_
        merged_level_labels = all_cbs[0].level_labels_
        merged_rule = merge_rule_group(rules, merged_level_values,
                                       merged_level_labels)

        for dim_x, dim_y in plot_dims:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))

            # Background: all replay states (light gray)
            ax.scatter(replay_states[:, dim_x], replay_states[:, dim_y],
                       c="lightgray", s=1, alpha=0.3, label="all replay")

            # DQN action coloring for context
            dqn_preds, _ = dqn_model.predict(replay_states, deterministic=True)

            # Member states per rule (different colors)
            colors = plt.cm.Set2(np.linspace(0, 1, max(len(group), 3)))
            for i, (run_idx, rule) in enumerate(group):
                cbs = all_cbs[run_idx]
                indices = _states_matching_rule(rule, cbs, replay_states)
                if len(indices) > 0:
                    states = replay_states[indices]
                    ax.scatter(states[:, dim_x], states[:, dim_y],
                               c=[colors[i]], s=15, alpha=0.7,
                               label=f"rule {i} (run {run_idx})",
                               edgecolors="k", linewidth=0.3)

            # Merged region bounds (rectangle)
            for pred in merged_rule.predicates:
                if pred.feature_idx == dim_x and pred.lower_bound is not None:
                    ax.axvline(pred.lower_bound, color="red", linestyle="--",
                               alpha=0.5, linewidth=1)
                    ax.axvline(pred.upper_bound, color="red", linestyle="--",
                               alpha=0.5, linewidth=1)
                if pred.feature_idx == dim_y and pred.lower_bound is not None:
                    ax.axhline(pred.lower_bound, color="red", linestyle="--",
                               alpha=0.5, linewidth=1)
                    ax.axhline(pred.upper_bound, color="red", linestyle="--",
                               alpha=0.5, linewidth=1)

            ax.set_xlabel(feat_names[dim_x])
            ax.set_ylabel(feat_names[dim_y])
            ax.set_title(
                f"Group {group_idx} ({case_label}) — action={diag['action']}\n"
                f"mismatch={diag['action_mismatch_rate']:.2f}, "
                f"components={diag['n_connected_components']}, "
                f"modes={diag['max_modes']}"
            )
            ax.legend(fontsize=7, loc="best")

            fname = f"geometric_{env_name.lower().replace('-','_')}_group{group_idx}_{case_label}_{dim_x}v{dim_y}.png"
            fpath = os.path.join(out_dir, fname)
            fig.tight_layout()
            fig.savefig(fpath, dpi=200)
            plt.close(fig)
            plot_paths.append(fpath)
            print(f"    Saved plot: {fname}")

    return plot_paths


def run_geometric_distortion(env_name):
    """Run geometric distortion analysis for one environment."""
    print(f"\n{'='*70}")
    print(f"  Geometric Distortion Diagnosis: {env_name}")
    print(f"{'='*70}")

    env_tag = env_name.replace("-", "_").lower()
    model_path = get_model_path(env_name)

    # Load DQN model for action consistency
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

    # Build internal subsamples
    print(f"  Building {N_BOOTSTRAP} CBS subsamples...")
    rng = np.random.RandomState(42)
    n_total = len(replay_states)

    all_cbs = []
    all_rules = []
    all_thresholds = []

    for i in range(N_BOOTSTRAP):
        idx = rng.choice(n_total, size=int(n_total * SUBSAMPLE_FRACTION),
                         replace=False)
        s = replay_states[idx]
        a = replay_actions[idx]
        cbs, rules = run_cbs_on_data(s, a, env_name)
        all_cbs.append(cbs)
        all_rules.append(rules)
        all_thresholds.append(cbs.get_thresholds())

    # Match rules across runs
    actions_set = sorted(set(r.action for rules in all_rules for r in rules))
    all_groups = []
    for action in actions_set:
        per_run = [[r for r in rules if r.action == action] for rules in all_rules]
        groups = _match_rules_across_runs(
            per_run, rho=DEFAULT_RHO, lambda1=DEFAULT_LAMBDA1,
            lambda2=DEFAULT_LAMBDA2)
        all_groups.extend(groups)

    print(f"  Matched groups: {len(all_groups)}")

    # Analyze each group
    print("  Analyzing geometric properties...")
    group_diagnostics = []
    for g_idx, group in enumerate(all_groups):
        diag = analyze_group(
            group, g_idx, all_cbs, replay_states, replay_actions,
            dqn_model, env_name, N_BOOTSTRAP)
        group_diagnostics.append(diag)

    # Classify
    n_failed = sum(1 for g in group_diagnostics if g["is_failed_merge"])
    n_success = sum(1 for g in group_diagnostics if not g["is_failed_merge"])
    print(f"  Merge classification: {n_success} successful, {n_failed} failed")

    # Compute summary statistics for failed vs successful
    def _compute_stats(groups):
        if not groups:
            return {}
        return {
            "count": len(groups),
            "mean_modes": round(float(np.mean([g["max_modes"] for g in groups])), 2),
            "mean_components": round(float(np.mean([g["n_connected_components"] for g in groups])), 2),
            "mean_knn_gap": round(float(np.mean([g["knn_gap_ratio"] for g in groups])), 4),
            "mean_action_mismatch": round(float(np.mean([g["action_mismatch_rate"] for g in groups])), 4),
            "mean_bridge_rate": round(float(np.mean([g["low_density_bridge_rate"] for g in groups])), 4),
            "mean_n_states": round(float(np.mean([g["n_member_states"] for g in groups])), 1),
            "multimodal_frac": round(float(np.mean([g["is_multimodal"] for g in groups])), 3),
            "fragmented_frac": round(float(np.mean([g["is_fragmented"] for g in groups])), 3),
        }

    failed_groups = [g for g in group_diagnostics if g["is_failed_merge"]]
    success_groups = [g for g in group_diagnostics if not g["is_failed_merge"]]

    comparison = {
        "failed_merges": _compute_stats(failed_groups),
        "successful_merges": _compute_stats(success_groups),
    }

    # Print comparison table
    print(f"\n  {'='*70}")
    print(f"  Failed vs Successful Merges:")
    print(f"  {'Metric':<30} {'Failed':>15} {'Successful':>15}")
    print(f"  {'-'*60}")
    if comparison["failed_merges"] and comparison["successful_merges"]:
        for metric in ["count", "mean_modes", "mean_components",
                        "mean_knn_gap", "mean_action_mismatch",
                        "mean_bridge_rate", "mean_n_states",
                        "multimodal_frac", "fragmented_frac"]:
            f_val = comparison["failed_merges"].get(metric, "N/A")
            s_val = comparison["successful_merges"].get(metric, "N/A")
            print(f"  {metric:<30} {str(f_val):>15} {str(s_val):>15}")

    # Generate scatter plots
    out_dir = f"experiments/results/{env_tag}"
    os.makedirs(out_dir, exist_ok=True)
    plot_paths = generate_scatter_plots(
        group_diagnostics, replay_states, replay_actions,
        all_cbs, all_rules, all_groups, dqn_model,
        env_name, out_dir)

    # Save results
    output = {
        "schema_version": "geometric_distortion_v1",
        "env": env_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "n_bootstrap": N_BOOTSTRAP,
            "rho": DEFAULT_RHO,
            "tau": DEFAULT_TAU,
            "action_mismatch_threshold": ACTION_MISMATCH_THRESHOLD,
        },
        "n_groups": len(all_groups),
        "n_failed_merges": n_failed,
        "n_successful_merges": n_success,
        "comparison": comparison,
        "per_group_diagnostics": group_diagnostics,
        "plot_files": [os.path.basename(p) for p in plot_paths],
    }

    out_path = os.path.join(out_dir, "geometric_distortion.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Geometric Distortion Diagnosis")
    parser.add_argument("--env", default="all",
                        choices=ENVS + ["all"],
                        help="Environment to test")
    args = parser.parse_args()

    envs = ENVS if args.env == "all" else [args.env]
    for env in envs:
        run_geometric_distortion(env)


if __name__ == "__main__":
    main()
