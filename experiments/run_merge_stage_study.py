#!/usr/bin/env python
"""
Merge-stage diagnostic chain

Builds a continuous evidence chain from "default merge is broken" to
"the tuned and soft-support variants repair it", using:

  1. Failure decomposition (5 ablation stages)
  2. Stage-to-stage metric deltas
  3. Support hard vs soft comparison
  4. Aggregation method comparison
  5. Boundary crossing before/after repair
  6. Geometric distortion before/after repair

For each environment, compares:
  - default_consensus (baseline broken)
  - tuned_merge (numeric-merge repair)
  - soft_support (soft-support repair)

Output:
    experiments/results/merge_stages/{env}/failure_decomposition.json
    experiments/results/merge_stages/{env}/support_comparison.json
    experiments/results/merge_stages/{env}/aggregation_comparison.json
    experiments/results/merge_stages/{env}/boundary_crossing.json
    experiments/results/merge_stages/{env}/geometric_distortion.json
    experiments/results/merge_stages/{env}/repair_ladder.json
    experiments/results/merge_stages/{env}/stage_deltas.json

Usage:
    python experiments/run_merge_stage_study.py --env MountainCar-v0
    python experiments/run_merge_stage_study.py --env all
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from reproduction.cbs import CBSPipeline, Rule, Condition, Predicate
from reproduction.collect_replay import collect_replay, ENV_FEATURE_NAMES
from experiments.perturbations import (
    load_replay_npz, compute_feature_ranges, generate_subsamples,
)
from experiments.consensus_merge import (
    build_consensus_ruleset,
    run_cbs_on_data,
    _match_rules_across_runs,
    merge_rule_group,
    aggregate_thresholds,
    make_consensus_pipeline,
    _canonical_to_rule,
)
from experiments.soft_support_merge import SoftSupportConfig, build_soft_support_consensus
from experiments.rule_matching import (
    canonicalize_rules,
    serialize_canonical_rules,
    mean_pairwise_jaccard,
    mean_pairwise_soft_jaccard,
    mean_pairwise_threshold_drift,
    rule_similarity_threshold_aware,
)
from experiments.run_stress_test import (
    evaluate_single_run,
    compute_bra_from_predictions,
    EVAL_SEEDS,
    SUCCESS_THRESHOLDS,
    HELDOUT_SEED,
)

# ── Configuration ────────────────────────────────────────────────────

ENVS = ["MountainCar-v0", "CartPole-v1", "LunarLander-v3"]
N_OUTER_REPEATS = 5
OUTER_SEEDS = list(range(N_OUTER_REPEATS))
N_BOOTSTRAP = 5
SUBSAMPLE_FRACTION = 0.8
DEFAULT_TAU = 0.7
DEFAULT_RHO = 0.8
TUNED_RHO = 0.9
LAMBDA1 = 0.6
LAMBDA2 = 0.4
OUT_ROOT = "experiments/results/merge_stages"

# Best tuned-merge configuration per environment
TUNED_MERGE_CONFIGS = {
    "MountainCar-v0": {"B": 5, "tau": 0.7, "rho": 0.9, "lambda1": 0.6, "lambda2": 0.4},
    "CartPole-v1": {"B": 10, "tau": 0.5, "rho": 0.8, "lambda1": 0.6, "lambda2": 0.4},
    "LunarLander-v3": {"B": 5, "tau": 0.7, "rho": 0.9, "lambda1": 0.5, "lambda2": 0.5},
}

# Best soft-support configuration per environment
SOFT_SUPPORT_CONFIGS = {
    "MountainCar-v0": SoftSupportConfig(
        n_bootstrap=5, consensus_threshold=0.7, similarity_cutoff=0.9,
        lambda_P=0.35, lambda_I=0.45, lambda_B=0.1,
        support_mode="soft", safeguard_enabled=False),
    "CartPole-v1": SoftSupportConfig(
        n_bootstrap=5, consensus_threshold=0.7, similarity_cutoff=0.9,
        lambda_P=0.35, lambda_I=0.45, lambda_B=0.0,
        support_mode="soft", safeguard_enabled=False),
    "LunarLander-v3": SoftSupportConfig(
        n_bootstrap=5, consensus_threshold=0.7, similarity_cutoff=0.9,
        lambda_P=0.35, lambda_I=0.45, lambda_B=0.2,
        support_mode="soft", safeguard_enabled=False),
}


def get_model_path(env_name):
    tag = env_name.replace("-", "_").lower()
    return f"reproduction/models/dqn_{tag}.zip"


def _build_subsamples(data, env_name):
    """Build B internal CBS subsamples and return pipelines, rules, thresholds."""
    subsets = generate_subsamples(data, n_subsets=N_BOOTSTRAP,
                                  fraction=SUBSAMPLE_FRACTION, seed=42)
    all_cbs = []
    all_rules = []
    all_thresholds = []

    for i, sub in enumerate(subsets):
        cbs, rules = run_cbs_on_data(sub["states"], sub["actions"],
                                      env_name, kmeans_seed=i)
        thresholds = {int(k): [float(v) for v in vs]
                      for k, vs in cbs.get_thresholds().items()}
        all_cbs.append(cbs)
        all_rules.append(rules)
        all_thresholds.append(thresholds)

    return all_cbs, all_rules, all_thresholds


def _evaluate_pipeline(pipeline, rules, heldout_s, heldout_a, env_name):
    """Evaluate a pipeline returning uniform metrics dict."""
    res = evaluate_single_run(pipeline, rules, heldout_s, heldout_a, env_name)
    pa = res["fidelity_per_action"]["per_action"]
    recalls = [pa[a]["recall"] for a in pa]
    worst_recall = min(recalls) if recalls else 0.0

    return {
        "f1": res["fidelity_heldout"]["f1"],
        "accuracy": res["fidelity_heldout"]["accuracy"],
        "recall": res["fidelity_heldout"].get("recall", 0.0),
        "worst_action_recall": worst_recall,
        "E_CR": res["deployment"]["E_CR"],
        "E_CR_std": res["deployment"]["E_CR_std"],
        "success_rate": res["deployment"]["success_rate"],
        "n_rules": len(rules),
    }


# ── Module 1: Failure Decomposition ─────────────────────────────────


def run_failure_decomposition(env_name, data, heldout_s, heldout_a):
    """Run 5-stage failure decomposition for one outer repeat."""
    all_cbs, all_rules, all_thresholds = _build_subsamples(data, env_name)

    # Match rules across runs (per action)
    actions = set()
    for rules in all_rules:
        for r in rules:
            actions.add(r.action)

    all_groups = {}
    for action in sorted(actions):
        per_run = []
        for rules in all_rules:
            per_run.append([r for r in rules if r.action == action])
        groups = _match_rules_across_runs(per_run, DEFAULT_RHO, LAMBDA1, LAMBDA2)
        all_groups[action] = groups

    results = {}

    # Stage 1: match_only - all groups, first rule, no aggregation
    stage_rules = []
    for action, groups in all_groups.items():
        for group in groups:
            if group:
                stage_rules.append(group[0][1])  # first rule from first member
    if stage_rules:
        ref_cbs = all_cbs[0]
        pipeline = make_consensus_pipeline(
            ref_cbs, stage_rules, aggregate_thresholds(all_thresholds))
        results["match_only"] = _evaluate_pipeline(
            pipeline, stage_rules, heldout_s, heldout_a, env_name)
        results["match_only"]["surviving_groups"] = sum(
            len(g) for g in all_groups.values())
        results["match_only"]["filtered_groups"] = 0
    else:
        results["match_only"] = {"f1": 0, "n_rules": 0, "surviving_groups": 0,
                                  "filtered_groups": 0}

    # Stage 2: match_hard_support - hard support filter
    min_support = int(np.ceil(DEFAULT_TAU * N_BOOTSTRAP))
    stage_rules = []
    n_kept = 0
    n_filtered = 0
    for action, groups in all_groups.items():
        for group in groups:
            distinct_runs = len(set(r[0] for r in group))
            if distinct_runs >= min_support:
                stage_rules.append(group[0][1])
                n_kept += 1
            else:
                n_filtered += 1
    if stage_rules:
        pipeline = make_consensus_pipeline(
            all_cbs[0], stage_rules, aggregate_thresholds(all_thresholds))
        results["match_hard_support"] = _evaluate_pipeline(
            pipeline, stage_rules, heldout_s, heldout_a, env_name)
    else:
        results["match_hard_support"] = {"f1": 0, "n_rules": 0}
    results["match_hard_support"]["surviving_groups"] = n_kept
    results["match_hard_support"]["filtered_groups"] = n_filtered

    # Stage 3: match_aggregation - merge without support filter
    stage_rules = []
    ref_cbs = all_cbs[0]
    for action, groups in all_groups.items():
        for group in groups:
            group_rules = [r[1] for r in group]
            if len(group_rules) >= 2:
                lvs = getattr(ref_cbs, 'level_values_', None)
                lls = getattr(ref_cbs, 'level_labels_', None)
                merged = merge_rule_group(group_rules, lvs, lls)
                stage_rules.append(merged)
            else:
                stage_rules.append(group_rules[0])
    if stage_rules:
        pipeline = make_consensus_pipeline(
            ref_cbs, stage_rules, aggregate_thresholds(all_thresholds))
        results["match_aggregation"] = _evaluate_pipeline(
            pipeline, stage_rules, heldout_s, heldout_a, env_name)
    else:
        results["match_aggregation"] = {"f1": 0, "n_rules": 0}
    results["match_aggregation"]["surviving_groups"] = sum(
        len(g) for g in all_groups.values())
    results["match_aggregation"]["filtered_groups"] = 0

    # Stage 4: full_default - standard consensus
    pipeline, rules, info = build_consensus_ruleset(
        data, env_name,
        n_bootstrap=N_BOOTSTRAP,
        consensus_threshold=DEFAULT_TAU,
        similarity_cutoff=DEFAULT_RHO,
        lambda1=LAMBDA1, lambda2=LAMBDA2,
    )
    results["full_default"] = _evaluate_pipeline(
        pipeline, rules, heldout_s, heldout_a, env_name)
    results["full_default"]["surviving_groups"] = info.get("n_kept_groups", 0)
    results["full_default"]["filtered_groups"] = info.get("n_filtered_groups", 0)

    # Stage 5: v2_soft_support
    soft_support_cfg = SOFT_SUPPORT_CONFIGS[env_name]
    pipeline, rules, info = build_soft_support_consensus(data, env_name, soft_support_cfg)
    results["v2_soft_support"] = _evaluate_pipeline(
        pipeline, rules, heldout_s, heldout_a, env_name)
    results["v2_soft_support"]["surviving_groups"] = info.get("n_kept_groups", 0)
    results["v2_soft_support"]["filtered_groups"] = info.get("n_filtered_groups", 0)

    return results


# ── Module 2: Support Hard vs Soft ──────────────────────────────────


def run_support_comparison(env_name, data, heldout_s, heldout_a):
    """Compare hard vs soft support filtering on same matched groups."""
    results = {}

    # Hard support
    pipeline_hard, rules_hard, info_hard = build_consensus_ruleset(
        data, env_name,
        n_bootstrap=N_BOOTSTRAP,
        consensus_threshold=DEFAULT_TAU,
        similarity_cutoff=TUNED_RHO,
        lambda1=LAMBDA1, lambda2=LAMBDA2,
    )
    hard_metrics = _evaluate_pipeline(
        pipeline_hard, rules_hard, heldout_s, heldout_a, env_name)
    hard_metrics["kept_groups"] = info_hard.get("n_kept_groups", 0)
    hard_metrics["dropped_groups"] = info_hard.get("n_filtered_groups", 0)
    hard_metrics["total_groups"] = info_hard.get("n_groups_total",
        hard_metrics["kept_groups"] + hard_metrics["dropped_groups"])
    results["hard"] = hard_metrics

    # Soft support (v2 with lambda_B=0 to isolate soft-support effect)
    soft_cfg = SoftSupportConfig(
        n_bootstrap=N_BOOTSTRAP,
        consensus_threshold=DEFAULT_TAU,
        similarity_cutoff=TUNED_RHO,
        lambda_P=0.35, lambda_I=0.45, lambda_B=0.0,
        support_mode="soft",
        safeguard_enabled=False,
    )
    pipeline_soft, rules_soft, info_soft = build_soft_support_consensus(
        data, env_name, soft_cfg)
    soft_metrics = _evaluate_pipeline(
        pipeline_soft, rules_soft, heldout_s, heldout_a, env_name)
    soft_metrics["kept_groups"] = info_soft.get("n_kept_groups", 0)
    soft_metrics["dropped_groups"] = info_soft.get("n_filtered_groups", 0)
    soft_metrics["total_groups"] = info_soft.get("n_groups_total",
        soft_metrics["kept_groups"] + soft_metrics["dropped_groups"])
    results["soft"] = soft_metrics

    # Delta
    results["delta_hard_to_soft"] = {
        "f1": soft_metrics["f1"] - hard_metrics["f1"],
        "worst_action_recall": soft_metrics["worst_action_recall"] - hard_metrics["worst_action_recall"],
        "n_rules": soft_metrics["n_rules"] - hard_metrics["n_rules"],
        "E_CR": soft_metrics["E_CR"] - hard_metrics["E_CR"],
        "kept_groups": soft_metrics["kept_groups"] - hard_metrics["kept_groups"],
    }

    return results


# ── Module 3: Aggregation Comparison ────────────────────────────────


def run_aggregation_comparison(env_name, data, heldout_s, heldout_a):
    """Compare median vs support-weighted aggregation."""
    results = {}

    # Median (default)
    pipeline_m, rules_m, info_m = build_consensus_ruleset(
        data, env_name,
        n_bootstrap=N_BOOTSTRAP,
        consensus_threshold=DEFAULT_TAU,
        similarity_cutoff=TUNED_RHO,
        lambda1=LAMBDA1, lambda2=LAMBDA2,
    )
    results["median"] = _evaluate_pipeline(
        pipeline_m, rules_m, heldout_s, heldout_a, env_name)

    # Support-weighted: use v2 with soft support
    sw_cfg = SoftSupportConfig(
        n_bootstrap=N_BOOTSTRAP,
        consensus_threshold=DEFAULT_TAU,
        similarity_cutoff=TUNED_RHO,
        lambda_P=0.35, lambda_I=0.45, lambda_B=0.0,
        support_mode="soft",
        safeguard_enabled=False,
    )
    pipeline_sw, rules_sw, info_sw = build_soft_support_consensus(
        data, env_name, sw_cfg)
    results["support_weighted"] = _evaluate_pipeline(
        pipeline_sw, rules_sw, heldout_s, heldout_a, env_name)

    return results


# ── Module 4: Boundary Crossing Before/After ────────────────────────


def run_boundary_crossing_comparison(env_name, data, heldout_s, heldout_a):
    """Compare boundary crossing rates for default, tuned v1, v2 best."""
    from stable_baselines3 import DQN
    from sklearn.neighbors import NearestNeighbors

    model_path = get_model_path(env_name)
    dqn_model = DQN.load(model_path)

    # Build density model
    k = 10
    nn_model = NearestNeighbors(n_neighbors=k)
    nn_model.fit(data["states"])
    dists, _ = nn_model.kneighbors(data["states"])
    mean_dists = dists.mean(axis=1)
    density_threshold = np.percentile(mean_dists, 85)

    # Build subsamples for rule matching
    all_cbs, all_rules, all_thresholds = _build_subsamples(data, env_name)

    results = {}

    # For each method, build consensus and get the merged rules
    methods = {
        "default_consensus": {
            "B": N_BOOTSTRAP, "tau": DEFAULT_TAU, "rho": DEFAULT_RHO,
            "lambda1": LAMBDA1, "lambda2": LAMBDA2,
        },
        "tuned_merge": TUNED_MERGE_CONFIGS[env_name],
    }

    for method_name, params in methods.items():
        pipeline, rules, info = build_consensus_ruleset(
            data, env_name,
            n_bootstrap=params.get("B", N_BOOTSTRAP),
            consensus_threshold=params.get("tau", DEFAULT_TAU),
            similarity_cutoff=params.get("rho", DEFAULT_RHO),
            lambda1=params.get("lambda1", LAMBDA1),
            lambda2=params.get("lambda2", LAMBDA2),
        )

        # Analyse crossing: sample pairs from consensus rules
        crossings = _analyze_crossing_for_rules(
            rules, data["states"], dqn_model, nn_model, density_threshold)
        results[method_name] = crossings

    # V2 best
    soft_support_cfg = SOFT_SUPPORT_CONFIGS[env_name]
    pipeline_soft_support, rules_soft_support, info_soft_support = build_soft_support_consensus(
        data, env_name, soft_support_cfg)
    crossings_soft_support = _analyze_crossing_for_rules(
        rules_soft_support, data["states"], dqn_model, nn_model, density_threshold)
    results["soft_support"] = crossings_soft_support

    return results


def _analyze_crossing_for_rules(rules, replay_states, dqn_model,
                                 nn_model, density_threshold):
    """Analyze boundary crossing for a set of consensus rules."""
    if len(rules) < 2:
        return {
            "n_rules": len(rules),
            "n_pairs_analyzed": 0,
            "mergeable_crossing_pct": 0.0,
            "midpoint_mismatch_pct": 0.0,
            "midpoint_low_density_pct": 0.0,
        }

    n_steps = 21
    alphas = np.linspace(0.0, 1.0, n_steps)

    crossing_rates = []
    mismatch_rates = []
    low_density_rates = []
    n_pairs = 0

    # For each pair of same-action rules
    for i in range(len(rules)):
        for j in range(i + 1, len(rules)):
            if rules[i].action != rules[j].action:
                continue

            # Get representative states for each rule
            states_i = _get_rule_states(rules[i], replay_states)
            states_j = _get_rule_states(rules[j], replay_states)

            if len(states_i) < 2 or len(states_j) < 2:
                continue

            n_pairs += 1
            if n_pairs > 30:  # cap at 30 pairs per method
                break

            # Sample pairs and interpolate
            n_sample = min(10, len(states_i), len(states_j))
            rng = np.random.default_rng(42 + i * 100 + j)
            idx_i = rng.choice(len(states_i), size=n_sample, replace=True)
            idx_j = rng.choice(len(states_j), size=n_sample, replace=True)

            crossings_this = 0
            mismatches_this = 0
            low_density_this = 0
            total_paths = 0

            for si, sj in zip(idx_i, idx_j):
                s_a = states_i[si]
                s_b = states_j[sj]

                # Interpolate
                path = np.array([s_a + alpha * (s_b - s_a) for alpha in alphas])

                # Get DQN actions along path
                dqn_actions = []
                for pt in path:
                    obs = pt.reshape(1, -1).astype(np.float32)
                    action, _ = dqn_model.predict(obs, deterministic=True)
                    dqn_actions.append(int(np.asarray(action).item()))

                # Check for boundary crossing
                flips = sum(1 for k in range(len(dqn_actions)-1)
                           if dqn_actions[k] != dqn_actions[k+1])
                if flips > 0:
                    crossings_this += 1

                # Check midpoint
                mid_idx = n_steps // 2
                if dqn_actions[mid_idx] != rules[i].action:
                    mismatches_this += 1

                # Check midpoint density
                mid_pt = path[mid_idx].reshape(1, -1)
                mid_dist = nn_model.kneighbors(mid_pt)[0].mean()
                if mid_dist > density_threshold:
                    low_density_this += 1

                total_paths += 1

            if total_paths > 0:
                crossing_rates.append(crossings_this / total_paths)
                mismatch_rates.append(mismatches_this / total_paths)
                low_density_rates.append(low_density_this / total_paths)

        if n_pairs > 30:
            break

    return {
        "n_rules": len(rules),
        "n_pairs_analyzed": n_pairs,
        "mergeable_crossing_pct": float(np.mean(crossing_rates)) if crossing_rates else 0.0,
        "midpoint_mismatch_pct": float(np.mean(mismatch_rates)) if mismatch_rates else 0.0,
        "midpoint_low_density_pct": float(np.mean(low_density_rates)) if low_density_rates else 0.0,
    }


def _get_rule_states(rule, replay_states):
    """Get states from replay that match a rule's predicates (approximate)."""
    mask = np.ones(len(replay_states), dtype=bool)
    for pred in rule.predicates:
        feat_vals = replay_states[:, pred.feature_idx]
        if hasattr(pred, 'lower_bound') and pred.lower_bound is not None:
            mask &= (feat_vals >= pred.lower_bound)
        if hasattr(pred, 'upper_bound') and pred.upper_bound is not None:
            mask &= (feat_vals <= pred.upper_bound)
    return replay_states[mask]


# ── Module 5: Geometric Distortion Before/After ─────────────────────


def run_geometric_distortion_comparison(env_name, data, heldout_s, heldout_a):
    """Compare geometric distortion metrics for default, tuned v1, v2 best."""
    from stable_baselines3 import DQN

    model_path = get_model_path(env_name)
    dqn_model = DQN.load(model_path)

    results = {}

    methods = {
        "default_consensus": build_consensus_ruleset(
            data, env_name,
            n_bootstrap=N_BOOTSTRAP,
            consensus_threshold=DEFAULT_TAU,
            similarity_cutoff=DEFAULT_RHO,
            lambda1=LAMBDA1, lambda2=LAMBDA2,
        ),
        "tuned_merge": build_consensus_ruleset(
            data, env_name,
            n_bootstrap=TUNED_MERGE_CONFIGS[env_name].get("B", N_BOOTSTRAP),
            consensus_threshold=TUNED_MERGE_CONFIGS[env_name].get("tau", DEFAULT_TAU),
            similarity_cutoff=TUNED_MERGE_CONFIGS[env_name].get("rho", DEFAULT_RHO),
            lambda1=TUNED_MERGE_CONFIGS[env_name].get("lambda1", LAMBDA1),
            lambda2=TUNED_MERGE_CONFIGS[env_name].get("lambda2", LAMBDA2),
        ),
    }

    # V2 best
    soft_support_cfg = SOFT_SUPPORT_CONFIGS[env_name]
    methods["soft_support"] = build_soft_support_consensus(data, env_name, soft_support_cfg)

    for method_name, (pipeline, rules, info) in methods.items():
        metrics = _compute_distortion_metrics(
            rules, pipeline, data["states"], data["actions"], dqn_model)
        results[method_name] = metrics

    return results


def _compute_distortion_metrics(rules, pipeline, replay_states, replay_actions,
                                 dqn_model):
    """Compute geometric distortion metrics for a set of rules."""
    from sklearn.neighbors import NearestNeighbors

    if not rules:
        return {
            "n_rules": 0,
            "n_failed_merges": 0,
            "mean_action_mismatch": 0.0,
            "mean_knn_gap": 0.0,
            "n_multimodal": 0,
            "n_fragmented": 0,
        }

    action_mismatches = []
    knn_gaps = []
    n_multimodal = 0
    n_fragmented = 0
    n_failed = 0

    for rule in rules:
        states = _get_rule_states(rule, replay_states)
        if len(states) < 5:
            continue

        # Action consistency with DQN
        dqn_actions = []
        for s in states:
            obs = s.reshape(1, -1).astype(np.float32)
            action, _ = dqn_model.predict(obs, deterministic=True)
            dqn_actions.append(int(np.asarray(action).item()))
        dqn_actions = np.array(dqn_actions)
        mismatch = float(np.mean(dqn_actions != rule.action))
        action_mismatches.append(mismatch)
        if mismatch > 0.15:
            n_failed += 1

        # KNN gap ratio
        if len(states) >= 5:
            nn = NearestNeighbors(n_neighbors=min(5, len(states)))
            nn.fit(states)
            dists, _ = nn.kneighbors(states)
            mean_dists = np.sort(dists.mean(axis=1))
            if len(mean_dists) > 1:
                gaps = np.diff(mean_dists)
                gap_ratio = float(gaps.max() / (mean_dists.max() - mean_dists.min() + 1e-10))
                knn_gaps.append(gap_ratio)

        # Multimodality check (per feature)
        max_modes = 1
        for f in range(states.shape[1]):
            feat_vals = states[:, f]
            if feat_vals.std() < 1e-8:
                continue
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(feat_vals, bw_method="silverman")
                x_grid = np.linspace(feat_vals.min(), feat_vals.max(), 200)
                density = kde(x_grid)
                # Count peaks
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(density)
                n_modes = max(1, len(peaks))
                max_modes = max(max_modes, n_modes)
            except Exception:
                pass
        if max_modes > 1:
            n_multimodal += 1

        # Fragmentation check (DBSCAN)
        if len(states) >= 5:
            try:
                from sklearn.cluster import DBSCAN
                nn_temp = NearestNeighbors(n_neighbors=min(5, len(states)))
                nn_temp.fit(states)
                dists_temp, _ = nn_temp.kneighbors(states)
                eps = np.percentile(dists_temp[:, -1], 30)
                if eps < 1e-10:
                    eps = 0.1
                db = DBSCAN(eps=eps, min_samples=2).fit(states)
                n_components = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
                if n_components > 1:
                    n_fragmented += 1
            except Exception:
                pass

    return {
        "n_rules": len(rules),
        "n_failed_merges": n_failed,
        "mean_action_mismatch": float(np.mean(action_mismatches)) if action_mismatches else 0.0,
        "max_action_mismatch": float(np.max(action_mismatches)) if action_mismatches else 0.0,
        "mean_knn_gap": float(np.mean(knn_gaps)) if knn_gaps else 0.0,
        "n_multimodal": n_multimodal,
        "n_fragmented": n_fragmented,
        "multimodal_frac": n_multimodal / len(rules) if rules else 0.0,
        "fragmented_frac": n_fragmented / len(rules) if rules else 0.0,
        "failed_merge_frac": n_failed / len(rules) if rules else 0.0,
    }


# ── Stage Deltas ────────────────────────────────────────────────────


def compute_stage_deltas(fd_results):
    """Compute stage-to-stage metric deltas from failure decomposition."""
    stages = ["match_only", "match_hard_support", "match_aggregation",
              "full_default", "v2_soft_support"]
    deltas = {}

    pairs = [
        ("match_only", "full_default"),
        ("full_default", "v2_soft_support"),
    ]

    for s_from, s_to in pairs:
        if s_from in fd_results and s_to in fd_results:
            r_from = fd_results[s_from]
            r_to = fd_results[s_to]
            delta_key = f"{s_from}_to_{s_to}"
            deltas[delta_key] = {}
            for metric in ["f1", "worst_action_recall", "n_rules", "E_CR"]:
                if metric in r_from and metric in r_to:
                    deltas[delta_key][metric] = r_to[metric] - r_from[metric]

    return deltas


# ── Repair Ladder ───────────────────────────────────────────────────


def build_repair_ladder(env_name, data, heldout_s, heldout_a):
    """Build the complete repair ladder: default → tuned v1 → v2 best."""
    ladder = {}

    # Default consensus
    pipeline, rules, info = build_consensus_ruleset(
        data, env_name,
        n_bootstrap=N_BOOTSTRAP,
        consensus_threshold=DEFAULT_TAU,
        similarity_cutoff=DEFAULT_RHO,
        lambda1=LAMBDA1, lambda2=LAMBDA2,
    )
    ladder["default_consensus"] = _evaluate_pipeline(
        pipeline, rules, heldout_s, heldout_a, env_name)
    ladder["default_consensus"]["build_info"] = {
        "n_kept": info.get("n_kept_groups", 0),
        "n_filtered": info.get("n_filtered_groups", 0),
    }

    # Tuned v1
    cfg = TUNED_MERGE_CONFIGS[env_name]
    pipeline, rules, info = build_consensus_ruleset(
        data, env_name,
        n_bootstrap=cfg["B"],
        consensus_threshold=cfg["tau"],
        similarity_cutoff=cfg["rho"],
        lambda1=cfg["lambda1"], lambda2=cfg["lambda2"],
    )
    ladder["tuned_merge"] = _evaluate_pipeline(
        pipeline, rules, heldout_s, heldout_a, env_name)
    ladder["tuned_merge"]["build_info"] = {
        "n_kept": info.get("n_kept_groups", 0),
        "n_filtered": info.get("n_filtered_groups", 0),
    }

    # V2 best
    soft_support_cfg = SOFT_SUPPORT_CONFIGS[env_name]
    pipeline, rules, info = build_soft_support_consensus(data, env_name, soft_support_cfg)
    ladder["soft_support"] = _evaluate_pipeline(
        pipeline, rules, heldout_s, heldout_a, env_name)
    ladder["soft_support"]["build_info"] = {
        "n_kept": info.get("n_kept_groups", 0),
        "n_filtered": info.get("n_filtered_groups", 0),
    }

    # Compute deltas
    ladder["delta_default_to_tuned_merge"] = {
        m: ladder["tuned_merge"].get(m, 0) - ladder["default_consensus"].get(m, 0)
        for m in ["f1", "worst_action_recall", "n_rules", "E_CR"]
    }
    ladder["delta_tuned_merge_to_soft_support"] = {
        m: ladder["soft_support"].get(m, 0) - ladder["tuned_merge"].get(m, 0)
        for m in ["f1", "worst_action_recall", "n_rules", "E_CR"]
    }
    ladder["delta_default_to_soft_support"] = {
        m: ladder["soft_support"].get(m, 0) - ladder["default_consensus"].get(m, 0)
        for m in ["f1", "worst_action_recall", "n_rules", "E_CR"]
    }

    return ladder


# ── Main ────────────────────────────────────────────────────────────


def run_env(env_name):
    """Run the complete merge-stage study for one environment."""
    print(f"\n{'='*70}")
    print(f"  Merge-stage study: {env_name}")
    print(f"{'='*70}")

    model_path = get_model_path(env_name)
    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found at {model_path}. Skipping.")
        return None

    # Collect held-out replay
    print(f"  Collecting held-out replay...")
    heldout = collect_replay(
        env_name=env_name, model_path=model_path,
        num_transitions=5000, seed=HELDOUT_SEED, deterministic=True,
    )
    heldout_s, heldout_a = heldout["states"], heldout["actions"]

    env_tag = env_name.replace("-", "_").lower()
    out_dir = os.path.join(OUT_ROOT, env_tag)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()

    # Collect outer datasets
    outer_datasets = []
    for seed in OUTER_SEEDS:
        data = collect_replay(
            env_name=env_name, model_path=model_path,
            num_transitions=10000, seed=seed, deterministic=True,
        )
        outer_datasets.append(data)

    # ── 1. Failure Decomposition ──
    print(f"\n  [1/6] Failure Decomposition ({N_OUTER_REPEATS} repeats)...")
    fd_per_seed = {}
    for i, (seed, data) in enumerate(zip(OUTER_SEEDS, outer_datasets)):
        print(f"    seed={seed}...", end=" ", flush=True)
        fd = run_failure_decomposition(env_name, data, heldout_s, heldout_a)
        fd_per_seed[str(seed)] = fd
        print(f"done (full_default F1={fd['full_default'].get('f1', 0):.3f})")

    # Summarize FD
    fd_summary = {}
    stages = ["match_only", "match_hard_support", "match_aggregation",
              "full_default", "v2_soft_support"]
    for stage in stages:
        vals = {}
        for metric in ["f1", "worst_action_recall", "n_rules", "E_CR"]:
            v = [fd_per_seed[s][stage].get(metric, 0) for s in fd_per_seed
                 if stage in fd_per_seed[s]]
            vals[metric] = {"mean": float(np.mean(v)), "std": float(np.std(v))} if v else {}
        sg = [fd_per_seed[s][stage].get("surviving_groups", 0) for s in fd_per_seed
              if stage in fd_per_seed[s]]
        vals["surviving_groups"] = {"mean": float(np.mean(sg))} if sg else {}
        fd_summary[stage] = vals

    fd_output = {
        "env": env_name, "n_outer_repeats": N_OUTER_REPEATS,
        "per_seed": fd_per_seed, "summary": fd_summary,
    }
    with open(os.path.join(out_dir, "failure_decomposition.json"), "w") as f:
        json.dump(fd_output, f, indent=2, default=str)
    print(f"    Saved failure_decomposition.json")

    # Compute stage deltas
    avg_fd = {}
    for stage in stages:
        avg_fd[stage] = {
            m: fd_summary[stage].get(m, {}).get("mean", 0)
            for m in ["f1", "worst_action_recall", "n_rules", "E_CR"]
        }
    stage_deltas = compute_stage_deltas(avg_fd)
    with open(os.path.join(out_dir, "stage_deltas.json"), "w") as f:
        json.dump(stage_deltas, f, indent=2, default=str)
    print(f"    Saved stage_deltas.json")

    # ── 2. Support Comparison ──
    print(f"\n  [2/6] Support Hard vs Soft ({N_OUTER_REPEATS} repeats)...")
    sc_per_seed = {}
    for i, (seed, data) in enumerate(zip(OUTER_SEEDS, outer_datasets)):
        print(f"    seed={seed}...", end=" ", flush=True)
        sc = run_support_comparison(env_name, data, heldout_s, heldout_a)
        sc_per_seed[str(seed)] = sc
        print(f"hard F1={sc['hard']['f1']:.3f}, soft F1={sc['soft']['f1']:.3f}")

    sc_output = {"env": env_name, "per_seed": sc_per_seed}
    # Summary
    for mode in ["hard", "soft"]:
        vals = {m: [sc_per_seed[s][mode][m] for s in sc_per_seed]
                for m in ["f1", "worst_action_recall", "n_rules", "E_CR",
                          "kept_groups", "dropped_groups"]}
        sc_output[f"{mode}_summary"] = {
            m: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for m, v in vals.items()
        }
    with open(os.path.join(out_dir, "support_comparison.json"), "w") as f:
        json.dump(sc_output, f, indent=2, default=str)
    print(f"    Saved support_comparison.json")

    # ── 3. Aggregation Comparison ──
    print(f"\n  [3/6] Aggregation Comparison ({N_OUTER_REPEATS} repeats)...")
    ac_per_seed = {}
    for i, (seed, data) in enumerate(zip(OUTER_SEEDS, outer_datasets)):
        print(f"    seed={seed}...", end=" ", flush=True)
        ac = run_aggregation_comparison(env_name, data, heldout_s, heldout_a)
        ac_per_seed[str(seed)] = ac
        print(f"median F1={ac['median']['f1']:.3f}, "
              f"sw F1={ac['support_weighted']['f1']:.3f}")

    ac_output = {"env": env_name, "per_seed": ac_per_seed}
    for agg in ["median", "support_weighted"]:
        vals = {m: [ac_per_seed[s][agg][m] for s in ac_per_seed]
                for m in ["f1", "worst_action_recall", "n_rules", "E_CR"]}
        ac_output[f"{agg}_summary"] = {
            m: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for m, v in vals.items()
        }
    with open(os.path.join(out_dir, "aggregation_comparison.json"), "w") as f:
        json.dump(ac_output, f, indent=2, default=str)
    print(f"    Saved aggregation_comparison.json")

    # ── 4. Boundary Crossing Before/After ──
    print(f"\n  [4/6] Boundary Crossing Comparison...")
    bc_per_seed = {}
    for i, (seed, data) in enumerate(zip(OUTER_SEEDS, outer_datasets)):
        print(f"    seed={seed}...", end=" ", flush=True)
        bc = run_boundary_crossing_comparison(env_name, data, heldout_s, heldout_a)
        bc_per_seed[str(seed)] = bc
        dc_cross = bc.get("default_consensus", {}).get("mergeable_crossing_pct", 0)
        soft_support_cross = bc.get("soft_support", {}).get("mergeable_crossing_pct", 0)
        print(f"default={dc_cross:.3f}, v2={soft_support_cross:.3f}")

    bc_output = {"env": env_name, "per_seed": bc_per_seed}
    for method in ["default_consensus", "tuned_merge", "soft_support"]:
        vals = {m: [bc_per_seed[s].get(method, {}).get(m, 0) for s in bc_per_seed]
                for m in ["mergeable_crossing_pct", "midpoint_mismatch_pct",
                          "midpoint_low_density_pct"]}
        bc_output[f"{method}_summary"] = {
            m: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for m, v in vals.items()
        }
    with open(os.path.join(out_dir, "boundary_crossing.json"), "w") as f:
        json.dump(bc_output, f, indent=2, default=str)
    print(f"    Saved boundary_crossing.json")

    # ── 5. Geometric Distortion Before/After ──
    print(f"\n  [5/6] Geometric Distortion Comparison...")
    gd_per_seed = {}
    for i, (seed, data) in enumerate(zip(OUTER_SEEDS, outer_datasets)):
        print(f"    seed={seed}...", end=" ", flush=True)
        gd = run_geometric_distortion_comparison(env_name, data, heldout_s, heldout_a)
        gd_per_seed[str(seed)] = gd
        dc_fail = gd.get("default_consensus", {}).get("failed_merge_frac", 0)
        soft_support_fail = gd.get("soft_support", {}).get("failed_merge_frac", 0)
        print(f"default fail={dc_fail:.3f}, v2 fail={soft_support_fail:.3f}")

    gd_output = {"env": env_name, "per_seed": gd_per_seed}
    for method in ["default_consensus", "tuned_merge", "soft_support"]:
        vals = {m: [gd_per_seed[s].get(method, {}).get(m, 0) for s in gd_per_seed]
                for m in ["mean_action_mismatch", "failed_merge_frac",
                          "multimodal_frac", "fragmented_frac", "mean_knn_gap"]}
        gd_output[f"{method}_summary"] = {
            m: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for m, v in vals.items()
        }
    with open(os.path.join(out_dir, "geometric_distortion.json"), "w") as f:
        json.dump(gd_output, f, indent=2, default=str)
    print(f"    Saved geometric_distortion.json")

    # ── 6. Repair Ladder ──
    print(f"\n  [6/6] Repair Ladder ({N_OUTER_REPEATS} repeats)...")
    rl_per_seed = {}
    for i, (seed, data) in enumerate(zip(OUTER_SEEDS, outer_datasets)):
        print(f"    seed={seed}...", end=" ", flush=True)
        rl = build_repair_ladder(env_name, data, heldout_s, heldout_a)
        rl_per_seed[str(seed)] = rl
        dc_f1 = rl["default_consensus"]["f1"]
        tv1_f1 = rl["tuned_merge"]["f1"]
        soft_support_f1 = rl["soft_support"]["f1"]
        print(f"default={dc_f1:.3f} → tuned={tv1_f1:.3f} → v2={soft_support_f1:.3f}")

    # Summary
    rl_summary = {}
    for method in ["default_consensus", "tuned_merge", "soft_support"]:
        vals = {m: [rl_per_seed[s][method][m] for s in rl_per_seed]
                for m in ["f1", "worst_action_recall", "n_rules", "E_CR"]}
        rl_summary[method] = {
            m: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for m, v in vals.items()
        }

    rl_output = {"env": env_name, "per_seed": rl_per_seed, "summary": rl_summary}
    with open(os.path.join(out_dir, "repair_ladder.json"), "w") as f:
        json.dump(rl_output, f, indent=2, default=str)
    print(f"    Saved repair_ladder.json")

    elapsed = time.time() - t0
    print(f"\n  {env_name} complete in {elapsed:.1f}s")

    # Print repair ladder summary
    print(f"\n  ── REPAIR LADDER SUMMARY ({env_name}) ──")
    print(f"  {'Method':<25} {'F1':>6} {'worst-R':>8} {'rules':>6} {'E_CR':>7}")
    print(f"  {'-'*55}")
    for method in ["default_consensus", "tuned_merge", "soft_support"]:
        s = rl_summary[method]
        print(f"  {method:<25} {s['f1']['mean']:>6.3f} "
              f"{s['worst_action_recall']['mean']:>8.3f} "
              f"{s['n_rules']['mean']:>6.1f} {s['E_CR']['mean']:>7.1f}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Merge-stage diagnostic chain")
    parser.add_argument("--env", default="all",
                        choices=ENVS + ["all"])
    args = parser.parse_args()

    envs = ENVS if args.env == "all" else [args.env]

    print(f"\n{'#'*70}")
    print(f"  MERGE-STAGE DIAGNOSTIC CHAIN")
    print(f"  Environments: {envs}")
    print(f"  Outer repeats: {N_OUTER_REPEATS}")
    print(f"{'#'*70}")

    t_total = time.time()

    for env_name in envs:
        run_env(env_name)

    t_elapsed = time.time() - t_total
    print(f"\n{'#'*70}")
    print(f"  COMPLETE — Total elapsed: {t_elapsed:.1f}s")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
