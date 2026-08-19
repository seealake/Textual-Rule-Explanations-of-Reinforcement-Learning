#!/usr/bin/env python
"""
Consensus CBS Core Algorithm

Implements the consensus merge:
  Given ONE replay dataset, generate B internal subsamples, run CBS on each,
  match rules across runs via order-invariant graph clustering, filter by
  support threshold τ, aggregate thresholds in interval space, and
  reconstruct a consensus rule set.

Also provides:
  - Naive voting baseline (rule-set voting) for internal comparison
  - Policy-aware reweighting (importance-weighted voting variant) weight computation
  - Utility: canonical↔CBS rule conversion, threshold aggregation
"""
import copy
import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reproduction.cbs import CBSPipeline, Predicate, Condition, Rule
from reproduction.collect_replay import ENV_FEATURE_NAMES
from experiments.perturbations import generate_subsamples
from experiments.rule_matching import (
    CanonicalRule,
    CanonicalPredicate,
    canonicalize_rules,
    rule_similarity_threshold_aware,
)


# ── Utilities ────────────────────────────────────────────────────────


def _canonical_to_rule(cr: CanonicalRule) -> Rule:
    """Convert CanonicalRule back to CBS Rule for pipeline compatibility."""
    predicates = [
        Predicate(
            feature_idx=p.feature_idx,
            level=p.level,
            level_label=p.level_label,
            lower_bound=p.lower_bound,
            upper_bound=p.upper_bound,
        )
        for p in cr.predicates
    ]
    condition = Condition(predicates=predicates, n_instances=cr.n_instances)
    return Rule(action=cr.action, condition=condition, weight=cr.weight)


def aggregate_thresholds(
    all_thresholds: list[dict],
    aggregation: str = "median",
) -> dict:
    """Aggregate thresholds across B CBS runs (per feature, per index).

    Parameters
    ----------
    all_thresholds : list of dict[int, list[float]]
        Each dict maps feature_idx → sorted threshold list.
    aggregation : "median" or "mean"

    Returns
    -------
    dict[int, list[float]] — aggregated thresholds.
    """
    features = sorted(all_thresholds[0].keys())
    result = {}
    agg_fn = np.median if aggregation == "median" else np.mean
    for f in features:
        n_thresh = len(all_thresholds[0][f])
        agg = []
        for k in range(n_thresh):
            values = [t[f][k] for t in all_thresholds if f in t and k < len(t[f])]
            agg.append(float(agg_fn(values)))
        result[f] = agg
    return result


# ── Order-Invariant Rule Matching ────────────────────────────────────


def _match_rules_across_runs(
    per_run_rules: list[list[CanonicalRule]],
    rho: float,
    lambda1: float = 0.6,
    lambda2: float = 0.4,
) -> list[list[tuple[int, CanonicalRule]]]:
    """Match rules across B runs for one action using connected components.

    Order-invariant: result does not depend on run ordering or rule ordering
    within runs.

    Parameters
    ----------
    per_run_rules : list of B lists of CanonicalRule (all same action)
    rho : similarity cutoff for matching
    lambda1, lambda2 : weights for rule_similarity_threshold_aware

    Returns
    -------
    list of groups, each group = list of (run_idx, CanonicalRule)
    """
    # Collect all (run_idx, rule_idx_within_run, rule) triples
    items = []
    for run_idx, rules in enumerate(per_run_rules):
        for rule in rules:
            items.append((run_idx, rule))

    n = len(items)
    if n == 0:
        return []

    # Build adjacency: edge between cross-run pairs with sim >= rho
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            # Only match rules from DIFFERENT runs
            if items[i][0] == items[j][0]:
                continue
            sim = rule_similarity_threshold_aware(
                items[i][1], items[j][1],
                lambda1=lambda1, lambda2=lambda2,
            )
            if sim >= rho:
                adj[i].append(j)
                adj[j].append(i)

    # Find connected components via BFS
    visited = [False] * n
    groups = []
    for start in range(n):
        if visited[start]:
            continue
        # BFS
        component = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            component.append(items[node])
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        groups.append(component)

    return groups


# ── Interval-Space Rule Aggregation ──────────────────────────────────


def merge_rule_group(
    rules: list[CanonicalRule],
    level_values: np.ndarray,
    level_labels: list[str],
    total_instances_per_run: dict = None,
) -> CanonicalRule:
    """Merge a group of matched rules into one consensus rule.

    Aggregation in interval (continuous) space:
        - Predicates: median of lower/upper bounds; level = closest
            encoded level among the matched member rules
        - Include a feature only if present in >= 50% of rules in the group
        - Weight: mean of per-run normalized support
        - n_instances: sum

    Parameters
    ----------
    rules : list of CanonicalRule (same action, matched across runs)
    level_values : canonical level values from CBSPipeline
    level_labels : canonical level labels
    total_instances_per_run : dict[run_idx -> total_instances], for normalized support.
        If None, uses simple mean of weights.
    """
    assert len(rules) > 0
    action = rules[0].action
    n_rules = len(rules)
    threshold_50pct = n_rules / 2.0

    # Collect per-feature data
    feature_data = {}
    for r in rules:
        for p in r.predicates:
            f = p.feature_idx
            if f not in feature_data:
                feature_data[f] = {"levels": [], "lbs": [], "ubs": [],
                                   "labels": []}
            feature_data[f]["levels"].append(p.level)
            if p.lower_bound is not None:
                feature_data[f]["lbs"].append(p.lower_bound)
            if p.upper_bound is not None:
                feature_data[f]["ubs"].append(p.upper_bound)
            feature_data[f]["labels"].append(p.level_label)

    # Build consensus predicates
    preds = []
    for f in sorted(feature_data.keys()):
        data = feature_data[f]
        if len(data["levels"]) < threshold_50pct:
            continue  # feature not present in >= 50% of group

        # Aggregate raw interval bounds for threshold-aware matching and
        # diagnostics, but keep the representative rule level in encoded space.
        # Condition.matches() operates on encoded categorical levels, so mapping
        # a raw feature-space midpoint directly onto [-1, 1] is not meaningful.
        if data["lbs"] and data["ubs"]:
            med_lb = float(np.median(data["lbs"]))
            med_ub = float(np.median(data["ubs"]))
        else:
            med_lb = None
            med_ub = None
        encoded_midpoint = float(np.median(data["levels"]))

        # Choose the representative level using the member rules' encoded bins.
        level_idx = int(np.argmin(np.abs(level_values - encoded_midpoint)))
        level_val = float(level_values[level_idx])
        label = level_labels[level_idx]

        preds.append(CanonicalPredicate(
            feature_idx=f,
            level=level_val,
            level_label=label,
            lower_bound=med_lb,
            upper_bound=med_ub,
        ))

    if not preds:
        # Degenerate: no features survived filtering
        preds = [rules[0].predicates[0]] if rules[0].predicates else []

    # Weight: mean of weights (or normalized support)
    weight = float(np.mean([r.weight for r in rules]))
    n_inst = sum(r.n_instances for r in rules)

    return CanonicalRule(
        action=action,
        predicates=tuple(sorted(preds, key=lambda p: p.feature_idx)),
        weight=weight,
        n_instances=n_inst,
    )


# ── Main Consensus Builder ───────────────────────────────────────────


def run_cbs_on_data(states, actions, env_name, kmeans_seed=0, delta=0,
                    use_maxf1=False, sample_weight=None):
    """Fit CBS, optionally refine with MaxF1, return pipeline and canonical rules."""
    feature_names = ENV_FEATURE_NAMES.get(env_name)
    cbs = CBSPipeline(
        n_categories=5,
        inclusion_threshold=0.70,
        kmeans_seed=kmeans_seed,
        cluster_count_delta=delta,
        feature_names=feature_names,
    )
    cbs.fit(states, actions, sample_weight=sample_weight)
    if use_maxf1:
        cbs.refine_max_f1(states, actions)
    rules = canonicalize_rules(cbs.get_rules())
    return cbs, rules


def build_consensus_ruleset(
    base_data: dict,
    env_name: str,
    n_bootstrap: int = 5,
    consensus_threshold: float = 0.7,
    similarity_cutoff: float = 0.8,
    lambda1: float = 0.6,
    lambda2: float = 0.4,
    use_maxf1: bool = False,
    sample_weight: np.ndarray = None,
    subsample_fraction: float = 0.8,
    subsample_seed: int = 42,
) -> tuple:
    """Build a Consensus CBS rule set from a single replay dataset.

    Parameters
    ----------
    base_data : ReplayData dict with "states", "actions" keys
    env_name : environment name
    n_bootstrap : B — number of internal subsamples
    consensus_threshold : τ — minimum support fraction to keep a rule group
    similarity_cutoff : ρ — minimum similarity to match rules
    lambda1, lambda2 : weights for threshold-aware similarity
    use_maxf1 : apply MaxF1 refinement to each internal CBS run
    sample_weight : per-sample weights for importance-weighted voting variant
    subsample_fraction : fraction of data per subsample
    subsample_seed : random seed for subsampling

    Returns
    -------
    (consensus_pipeline, consensus_rules, build_info)
    """
    # 1. Generate B subsamples
    subsamples = generate_subsamples(
        base_data, n_bootstrap, subsample_fraction, seed=subsample_seed)

    # If sample_weight provided, subsample the weights too
    # (generate_subsamples uses indices internally — we need to track them)
    # Simpler: regenerate subsamples with index tracking
    rng = np.random.RandomState(subsample_seed)
    n_total = len(base_data["states"])
    subsample_indices = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_total, size=int(n_total * subsample_fraction),
                         replace=False)
        subsample_indices.append(idx)

    # 2. Run CBS on each subsample
    all_cbs = []
    all_rules = []
    all_thresholds = []

    for i, idx in enumerate(subsample_indices):
        s = base_data["states"][idx]
        a = base_data["actions"][idx]
        sw = sample_weight[idx] if sample_weight is not None else None

        cbs, rules = run_cbs_on_data(s, a, env_name,
                                      use_maxf1=use_maxf1,
                                      sample_weight=sw)
        all_cbs.append(cbs)
        all_rules.append(rules)
        all_thresholds.append(cbs.get_thresholds())

    # 3. Group rules by action and match across runs
    actions_set = sorted(set(r.action for rules in all_rules for r in rules))
    all_groups = []
    n_input_rules = sum(len(rs) for rs in all_rules)

    for action in actions_set:
        per_run = []
        for rules in all_rules:
            per_run.append([r for r in rules if r.action == action])

        groups = _match_rules_across_runs(
            per_run, rho=similarity_cutoff,
            lambda1=lambda1, lambda2=lambda2)
        all_groups.extend(groups)

    # 4. Filter by support >= τ
    min_support = int(np.ceil(consensus_threshold * n_bootstrap))
    kept_groups = []
    filtered_groups = []

    for group in all_groups:
        distinct_runs = len(set(run_idx for run_idx, _ in group))
        if distinct_runs >= min_support:
            kept_groups.append(group)
        else:
            filtered_groups.append(group)

    # 5. Aggregate each kept group into a consensus rule
    level_values = all_cbs[0].level_values_
    level_labels = all_cbs[0].level_labels_

    consensus_rules = []
    for group in kept_groups:
        rules_in_group = [rule for _, rule in group]
        cr = merge_rule_group(rules_in_group, level_values, level_labels)
        consensus_rules.append(cr)

    # Sort consensus rules by (action, signature) for determinism
    consensus_rules.sort(key=lambda r: (r.action, r.signature))

    # 6. Aggregate thresholds
    agg_thresholds = aggregate_thresholds(all_thresholds, "median")

    # 7. Construct consensus pipeline
    consensus_pipeline = make_consensus_pipeline(
        all_cbs[0], consensus_rules, agg_thresholds)

    # 8. Build info for diagnostics
    per_action_counts = {}
    for a in actions_set:
        per_action_counts[int(a)] = sum(
            1 for r in consensus_rules if r.action == a)
    actions_lost = [a for a, c in per_action_counts.items() if c == 0]

    build_info = {
        "n_bootstrap": n_bootstrap,
        "consensus_threshold": consensus_threshold,
        "similarity_cutoff": similarity_cutoff,
        "n_input_rules": n_input_rules,
        "n_matched_groups": len(all_groups),
        "n_kept_groups": len(kept_groups),
        "n_filtered_groups": len(filtered_groups),
        "n_consensus_rules": len(consensus_rules),
        "per_action_rule_counts": per_action_counts,
        "actions_lost": actions_lost,
    }

    return consensus_pipeline, consensus_rules, build_info


def make_consensus_pipeline(
    reference_cbs: CBSPipeline,
    consensus_rules: list[CanonicalRule],
    aggregated_thresholds: dict,
) -> CBSPipeline:
    """Create a CBSPipeline with consensus rules and aggregated thresholds.

    The result is fully compatible with evaluate_single_run().
    """
    cbs = copy.deepcopy(reference_cbs)

    # Replace thresholds with median-aggregated values
    cbs.thresholds_ = {int(k): v for k, v in aggregated_thresholds.items()}

    # Update feature min/max to be consistent
    # (keep reference's — they cover the same data distribution)

    # Convert canonical rules to CBS Rules
    cbs.rules_ = [_canonical_to_rule(cr) for cr in consensus_rules]

    # Recompute condition centers for approximation
    cbs._precompute_condition_centers()

    return cbs


# ── Naive Voting Baseline (rule-set voting) ─────────────────────────────────


def build_voting_ensemble(
    base_data: dict,
    env_name: str,
    n_bootstrap: int = 5,
    use_maxf1: bool = False,
    subsample_fraction: float = 0.8,
    subsample_seed: int = 42,
) -> list[CBSPipeline]:
    """Build B independent CBS pipelines for majority-vote prediction.

    Returns list of fitted CBSPipeline objects.
    """
    rng = np.random.RandomState(subsample_seed)
    n_total = len(base_data["states"])
    pipelines = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n_total, size=int(n_total * subsample_fraction),
                         replace=False)
        s = base_data["states"][idx]
        a = base_data["actions"][idx]
        cbs, _ = run_cbs_on_data(s, a, env_name, use_maxf1=use_maxf1)
        pipelines.append(cbs)

    return pipelines


def voting_predict(pipelines: list[CBSPipeline], states: np.ndarray) -> np.ndarray:
    """Majority-vote prediction across B pipelines."""
    all_preds = np.array([p.predict(states) for p in pipelines])  # (B, N)
    # Majority vote per state
    result = np.zeros(len(states), dtype=int)
    for i in range(len(states)):
        votes = all_preds[:, i]
        counts = np.bincount(votes)
        result[i] = int(np.argmax(counts))
    return result


# ── Policy-Aware Reweighting (importance-weighted voting variant) ─────────────────────────


def compute_policy_aware_weights(
    states: np.ndarray,
    actions: np.ndarray,
    model_path: str,
    clip_ratio: float = 5.0,
    temperature: float = 1.0,
) -> np.ndarray:
    """Compute policy-aware reweighting: w = π̃(a|s) / μ(a).

    π̃(a|s) is softmax of DQN Q-values (surrogate, not true greedy policy).
    μ(a) is global empirical action frequency in the replay.

    Parameters
    ----------
    states : (N, obs_dim)
    actions : (N,)
    model_path : path to stable-baselines3 DQN model
    clip_ratio : clip w to [1/clip_ratio, clip_ratio]
    temperature : softmax temperature

    Returns
    -------
    weights : (N,) array, clipped and normalized to mean=1
    """
    import torch
    from stable_baselines3 import DQN

    model = DQN.load(model_path)
    n_actions = model.action_space.n

    # Compute μ(a) = empirical marginal action frequency
    mu = np.bincount(actions.astype(int), minlength=n_actions).astype(float)
    mu /= mu.sum()
    mu = np.maximum(mu, 1e-8)  # avoid division by zero

    # Compute π̃(a|s) via softmax of Q-values
    weights = np.zeros(len(states))
    batch_size = 512

    for start in range(0, len(states), batch_size):
        end = min(start + batch_size, len(states))
        obs = states[start:end]
        obs_t, _ = model.policy.obs_to_tensor(obs)

        with torch.no_grad():
            q_values = model.policy.q_net(obs_t).cpu().numpy()

        # Softmax
        q_values = q_values / temperature
        q_values = q_values - q_values.max(axis=1, keepdims=True)
        exp_q = np.exp(q_values)
        pi = exp_q / exp_q.sum(axis=1, keepdims=True)  # (batch, n_actions)

        for i in range(end - start):
            a = int(actions[start + i])
            weights[start + i] = pi[i, a] / mu[a]

    # Clip and normalize
    weights = np.clip(weights, 1.0 / clip_ratio, clip_ratio)
    weights = weights / weights.mean()

    return weights
