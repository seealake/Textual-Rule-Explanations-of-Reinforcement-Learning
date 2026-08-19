#!/usr/bin/env python
"""
Median vs Support-Weighted Aggregation Comparison

Compares two threshold/bound aggregation strategies in Consensus CBS:
  1. Median (current default): median of lower/upper bounds per feature
  2. Support-weighted: weighted average by n_instances

Runs on all 3 environments with 5 outer seed-shift repeats, B=5, τ=0.7, ρ=0.8.

Usage:
    python experiments/run_aggregation_comparison.py
    python experiments/run_aggregation_comparison.py --env CartPole-v1

Output:
    experiments/results/<env>/aggregation_comparison.json
    experiments/results/aggregation_comparison_combined.json
"""
import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from reproduction.cbs import CBSPipeline
from reproduction.collect_replay import collect_replay, ENV_FEATURE_NAMES
from experiments.perturbations import load_replay_npz
from experiments.rule_matching import (
    CanonicalRule,
    CanonicalPredicate,
    canonicalize_rules,
    serialize_canonical_rules,
    ruleset_weighted_jaccard,
    ruleset_soft_jaccard,
    threshold_drift,
)
from experiments.consensus_merge import (
    run_cbs_on_data,
    _match_rules_across_runs,
    merge_rule_group,
    aggregate_thresholds,
    make_consensus_pipeline,
    _canonical_to_rule,
)
from experiments.run_stress_test import (
    evaluate_single_run,
    compute_bra_from_predictions,
    get_model_path,
    get_replay_path,
    collect_heldout,
    EVAL_SEEDS,
    SUCCESS_THRESHOLDS,
)

# ── Constants ─────────────────────────────────────────────────────────

ENVS = ["MountainCar-v0", "CartPole-v1", "LunarLander-v3"]
SEED_SHIFTS = [0, 1, 2, 3, 4]
DEFAULT_B = 5
DEFAULT_TAU = 0.7
DEFAULT_RHO = 0.8
DEFAULT_LAMBDA1 = 0.6
DEFAULT_LAMBDA2 = 0.4


# ── Support-Weighted Variants ────────────────────────────────────────

def merge_rule_group_weighted(
    rules: list[CanonicalRule],
    level_values: np.ndarray,
    level_labels: list[str],
) -> CanonicalRule:
    """Merge rules with support-weighted bounds (instead of median).

    Each rule's bounds contribute proportionally to its n_instances.
    """
    assert len(rules) > 0
    action = rules[0].action
    n_rules = len(rules)
    threshold_50pct = n_rules / 2.0

    feature_data = {}
    for r in rules:
        for p in r.predicates:
            f = p.feature_idx
            if f not in feature_data:
                feature_data[f] = {"levels": [], "lbs": [], "ubs": [],
                                   "labels": [], "weights": []}
            feature_data[f]["levels"].append(p.level)
            feature_data[f]["weights"].append(max(r.n_instances, 1))
            if p.lower_bound is not None:
                feature_data[f]["lbs"].append(p.lower_bound)
            if p.upper_bound is not None:
                feature_data[f]["ubs"].append(p.upper_bound)
            feature_data[f]["labels"].append(p.level_label)

    preds = []
    for f in sorted(feature_data.keys()):
        data = feature_data[f]
        if len(data["levels"]) < threshold_50pct:
            continue

        n_lbs = len(data["lbs"])
        n_weights = len(data["weights"])
        if n_weights < n_lbs:
            raise ValueError(
                f"Inconsistent lengths for weights and lower bounds in feature {f}: "
                f"{n_weights} weights for {n_lbs} lower bounds."
            )
        weights = np.array(data["weights"][:n_lbs], dtype=float)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_lbs) / n_lbs

        if data["lbs"] and data["ubs"]:
            w_lb = float(np.average(data["lbs"], weights=weights))
            w_ub = float(np.average(data["ubs"], weights=weights))
        else:
            w_lb = None
            w_ub = None

        level_idx = int(np.argmin(np.abs(level_values - np.median(data["levels"]))))
        level_val = float(level_values[level_idx])
        label = level_labels[level_idx]

        preds.append(CanonicalPredicate(
            feature_idx=f, level=level_val, level_label=label,
            lower_bound=w_lb, upper_bound=w_ub,
        ))

    if not preds:
        preds = [rules[0].predicates[0]] if rules[0].predicates else []

    weight = float(np.mean([r.weight for r in rules]))
    n_inst = sum(r.n_instances for r in rules)

    return CanonicalRule(
        action=action,
        predicates=tuple(sorted(preds, key=lambda p: p.feature_idx)),
        weight=weight,
        n_instances=n_inst,
    )


def aggregate_thresholds_weighted(
    all_thresholds: list[dict],
    all_n_instances: list[int],
) -> dict:
    """Aggregate thresholds using support-weighted average."""
    features = sorted(all_thresholds[0].keys())
    total_inst = sum(all_n_instances)
    weights = np.array(all_n_instances, dtype=float) / total_inst if total_inst > 0 \
        else np.ones(len(all_n_instances)) / len(all_n_instances)

    result = {}
    for f in features:
        n_thresh = len(all_thresholds[0][f])
        agg = []
        for k in range(n_thresh):
            values = []
            w_vals = []
            for bi, t in enumerate(all_thresholds):
                if f in t and k < len(t[f]):
                    values.append(t[f][k])
                    w_vals.append(weights[bi])
            w_arr = np.array(w_vals)
            w_arr = w_arr / w_arr.sum() if w_arr.sum() > 0 else w_arr
            agg.append(float(np.average(values, weights=w_arr)))
        result[f] = agg
    return result


# ── Consensus Build with Configurable Aggregation ────────────────────

def build_consensus_with_aggregation(
    base_data: dict,
    env_name: str,
    aggregation: str = "median",  # "median" or "support_weighted"
    n_bootstrap: int = DEFAULT_B,
    consensus_threshold: float = DEFAULT_TAU,
    similarity_cutoff: float = DEFAULT_RHO,
    lambda1: float = DEFAULT_LAMBDA1,
    lambda2: float = DEFAULT_LAMBDA2,
    subsample_seed: int = 42,
) -> tuple:
    """Build consensus CBS with configurable aggregation method."""
    from experiments.perturbations import generate_subsamples

    rng = np.random.RandomState(subsample_seed)
    n_total = len(base_data["states"])
    subsample_indices = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_total, size=int(n_total * 0.8), replace=False)
        subsample_indices.append(idx)

    all_cbs = []
    all_rules = []
    all_thresholds = []
    all_n_instances_per_run = []

    for idx in subsample_indices:
        s = base_data["states"][idx]
        a = base_data["actions"][idx]
        cbs, rules = run_cbs_on_data(s, a, env_name)
        all_cbs.append(cbs)
        all_rules.append(rules)
        all_thresholds.append(cbs.get_thresholds())
        all_n_instances_per_run.append(len(s))

    # Match rules
    actions_set = sorted(set(r.action for rules in all_rules for r in rules))
    all_groups = []
    for action in actions_set:
        per_run = [[r for r in rules if r.action == action] for rules in all_rules]
        groups = _match_rules_across_runs(per_run, rho=similarity_cutoff,
                                           lambda1=lambda1, lambda2=lambda2)
        all_groups.extend(groups)

    min_support = int(np.ceil(consensus_threshold * n_bootstrap))
    kept_groups = [g for g in all_groups
                   if len(set(ri for ri, _ in g)) >= min_support]

    level_values = all_cbs[0].level_values_
    level_labels = all_cbs[0].level_labels_

    # Merge with selected aggregation
    consensus_rules = []
    for group in kept_groups:
        rules_in_group = [rule for _, rule in group]
        if aggregation == "support_weighted":
            cr = merge_rule_group_weighted(rules_in_group, level_values, level_labels)
        else:
            cr = merge_rule_group(rules_in_group, level_values, level_labels)
        consensus_rules.append(cr)

    consensus_rules.sort(key=lambda r: (r.action, r.signature))

    # Aggregate thresholds
    if aggregation == "support_weighted":
        agg_thresholds = aggregate_thresholds_weighted(
            all_thresholds, all_n_instances_per_run)
    else:
        agg_thresholds = aggregate_thresholds(all_thresholds, "median")

    consensus_pipeline = make_consensus_pipeline(
        all_cbs[0], consensus_rules, agg_thresholds)

    return consensus_pipeline, consensus_rules


# ── Runner ────────────────────────────────────────────────────────────

def run_comparison(env_name: str):
    """Run median vs support-weighted comparison for one environment."""
    print(f"\n{'='*60}")
    print(f"Aggregation Comparison: {env_name}")
    print(f"{'='*60}")

    tag = env_name.replace("-", "_").lower()
    model_path = get_model_path(env_name)

    # Collect held-out data
    h_states, h_actions = collect_heldout(env_name, model_path)

    results = {
        "env": env_name,
        "n_outer_repeats": len(SEED_SHIFTS),
        "n_bootstrap": DEFAULT_B,
        "tau": DEFAULT_TAU,
        "rho": DEFAULT_RHO,
        "median": {"per_run": []},
        "support_weighted": {"per_run": []},
    }

    for seed in SEED_SHIFTS:
        print(f"  Seed shift {seed} ...")
        replay_path = get_replay_path(env_name, seed=seed)
        if not os.path.exists(replay_path):
            # Collect replay for this seed
            data = collect_replay(
                env_name=env_name, model_path=model_path,
                num_transitions=10000, seed=seed, deterministic=True,
            )
            base_data = {"states": data["states"], "actions": data["actions"]}
        else:
            base_data = load_replay_npz(replay_path)

        for agg_method in ["median", "support_weighted"]:
            pipeline, rules = build_consensus_with_aggregation(
                base_data, env_name, aggregation=agg_method,
                subsample_seed=42 + seed,
            )
            ev = evaluate_single_run(pipeline, rules, h_states, h_actions, env_name)
            fid = ev["fidelity_heldout"]
            dep = ev["deployment"]

            run_result = {
                "seed": seed,
                "f1": fid["f1"],
                "accuracy": fid["accuracy"],
                "recall": fid["recall"],
                "E_CR": dep["E_CR"],
                "E_CR_std": dep["E_CR_std"],
                "success_rate": dep["success_rate"],
                "n_rules": ev["n_rules"],
            }

            # Per-action fidelity
            fpa = ev.get("fidelity_per_action", {})
            if "per_action" in fpa:
                worst_recall = min(
                    v["recall"] for v in fpa["per_action"].values()
                ) if fpa["per_action"] else 0
                run_result["worst_action_recall"] = worst_recall

            results[agg_method]["per_run"].append(run_result)

    # Compute summaries
    for agg_method in ["median", "support_weighted"]:
        runs = results[agg_method]["per_run"]
        results[agg_method]["summary"] = {
            "mean_f1": float(np.mean([r["f1"] for r in runs])),
            "std_f1": float(np.std([r["f1"] for r in runs])),
            "mean_E_CR": float(np.mean([r["E_CR"] for r in runs])),
            "std_E_CR": float(np.std([r["E_CR"] for r in runs])),
            "mean_n_rules": float(np.mean([r["n_rules"] for r in runs])),
            "mean_success_rate": float(np.mean([r["success_rate"] for r in runs])),
        }
        if runs and "worst_action_recall" in runs[0]:
            results[agg_method]["summary"]["mean_worst_recall"] = float(
                np.mean([r["worst_action_recall"] for r in runs]))

    # Save
    out_dir = os.path.join("experiments", "results", tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "aggregation_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {out_path}")

    # Print summary table
    print(f"\n  {'Method':<20} {'F1':>8} {'E_CR':>10} {'Rules':>6} {'Worst-R':>8}")
    print(f"  {'-'*54}")
    for m in ["median", "support_weighted"]:
        s = results[m]["summary"]
        wr = s.get("mean_worst_recall", 0)
        print(f"  {m:<20} {s['mean_f1']:.4f}  {s['mean_E_CR']:>8.1f}  "
              f"{s['mean_n_rules']:>5.1f}  {wr:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="all")
    args = parser.parse_args()

    envs = ENVS if args.env == "all" else [args.env]
    t0 = time.time()
    all_results = {}

    for env in envs:
        all_results[env] = run_comparison(env)

    # Save combined
    out_path = os.path.join("experiments", "results",
                            "aggregation_comparison_combined.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results → {out_path}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
