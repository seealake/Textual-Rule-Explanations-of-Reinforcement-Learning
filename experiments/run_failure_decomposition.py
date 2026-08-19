#!/usr/bin/env python
"""
Merge failure-stage decomposition

Decomposes the "default merge is broken" finding into separable causes:
  1. match_only:        rule grouping only (no filtering, no aggregation)
  2. match_hard_support: match + hard support pruning (no aggregation)
  3. match_aggregation:  match + aggregation (no support pruning, τ=0)
  4. full_default:       original default consensus CBS (hard τ=0.7)
  5. v2_soft_support:    v2 soft support variant (repair reference)

For each ablation we report:
  - F1 (held-out)
  - BRA (prediction agreement with DQN on held-out)
  - worst-action recall
  - surviving groups / final rule count
  - filtered group count

Usage:
    python experiments/run_failure_decomposition.py --env MountainCar-v0
    python experiments/run_failure_decomposition.py --env CartPole-v1
    python experiments/run_failure_decomposition.py --env LunarLander-v3
    python experiments/run_failure_decomposition.py --env all
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from reproduction.cbs import CBSPipeline
from reproduction.collect_replay import collect_replay, ENV_FEATURE_NAMES

from experiments.rule_matching import (
    CanonicalRule,
    CanonicalPredicate,
    canonicalize_rules,
    serialize_canonical_rules,
    rule_similarity_threshold_aware,
    mean_pairwise_jaccard,
    mean_pairwise_soft_jaccard,
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
    EVAL_SEEDS,
    SUCCESS_THRESHOLDS,
    HELDOUT_SEED,
    _serialize,
)

# ── Constants ─────────────────────────────────────────────────────────
ENVS = ["MountainCar-v0", "CartPole-v1", "LunarLander-v3"]
N_OUTER_REPEATS = 5  # different outer replay seeds for stability
OUTER_SEEDS = [0, 1, 2, 3, 4]
N_BOOTSTRAP = 5
SUBSAMPLE_FRACTION = 0.8
DEFAULT_TAU = 0.7
DEFAULT_RHO = 0.8
DEFAULT_LAMBDA1 = 0.6
DEFAULT_LAMBDA2 = 0.4


def get_model_path(env_name):
    tag = env_name.replace("-", "_").lower()
    return f"reproduction/models/dqn_{tag}.zip"


def collect_heldout(env_name, model_path, n_transitions=5000):
    data = collect_replay(
        env_name=env_name, model_path=model_path,
        num_transitions=n_transitions, seed=HELDOUT_SEED,
        deterministic=True,
    )
    return data["states"], data["actions"]


def _build_internal_subsamples(data, env_name, n_bootstrap=N_BOOTSTRAP,
                                subsample_seed=42):
    """Run CBS on B subsamples and return all intermediate objects."""
    rng = np.random.RandomState(subsample_seed)
    n_total = len(data["states"])

    all_cbs = []
    all_rules = []
    all_thresholds = []

    for i in range(n_bootstrap):
        idx = rng.choice(n_total, size=int(n_total * SUBSAMPLE_FRACTION),
                         replace=False)
        s = data["states"][idx]
        a = data["actions"][idx]
        cbs, rules = run_cbs_on_data(s, a, env_name)
        all_cbs.append(cbs)
        all_rules.append(rules)
        all_thresholds.append(cbs.get_thresholds())

    return all_cbs, all_rules, all_thresholds


def _match_all_groups(all_rules, rho=DEFAULT_RHO, lambda1=DEFAULT_LAMBDA1,
                      lambda2=DEFAULT_LAMBDA2):
    """Match rules across runs, return all groups per action."""
    actions_set = sorted(set(r.action for rules in all_rules for r in rules))
    all_groups = []

    for action in actions_set:
        per_run = [[r for r in rules if r.action == action] for rules in all_rules]
        groups = _match_rules_across_runs(
            per_run, rho=rho, lambda1=lambda1, lambda2=lambda2)
        all_groups.extend(groups)

    return all_groups


def _evaluate_pipeline(pipeline, heldout_s, heldout_a, env_name, n_rules):
    """Evaluate a pipeline and return standardized metrics."""
    fid = pipeline.evaluate_fidelity(heldout_s, heldout_a)
    fid_pa = pipeline.evaluate_fidelity_per_action(heldout_s, heldout_a)
    deploy = pipeline.evaluate_in_env(
        env_name, eval_seeds=EVAL_SEEDS,
        success_threshold=SUCCESS_THRESHOLDS.get(env_name),
    )

    worst_recall = min(
        (v["recall"] for v in fid_pa["per_action"].values()),
        default=0.0,
    )

    return {
        "f1": fid["f1"],
        "accuracy": fid["accuracy"],
        "recall": fid["recall"],
        "worst_action_recall": worst_recall,
        "E_CR": deploy["E_CR"],
        "E_CR_std": deploy["E_CR_std"],
        "success_rate": deploy["success_rate"],
        "n_rules": n_rules,
        "fidelity_per_action": fid_pa,
    }


def run_ablation_match_only(all_cbs, all_rules, all_thresholds, all_groups,
                            heldout_s, heldout_a, env_name):
    """Ablation 1: match only — use ALL matched groups, no filtering, no aggregation.
    
    Each group is represented by its first rule (arbitrarily chosen).
    This isolates the matching step.
    """
    # Take all groups, use first rule from each group as representative
    rules = []
    for group in all_groups:
        # Use the first rule in the group (no aggregation)
        _, first_rule = group[0]
        rules.append(first_rule)

    n_rules = len(rules)
    agg_thresholds = aggregate_thresholds(all_thresholds, "median")
    pipeline = make_consensus_pipeline(all_cbs[0], rules, agg_thresholds)
    metrics = _evaluate_pipeline(pipeline, heldout_s, heldout_a, env_name, n_rules)
    metrics["surviving_groups"] = len(all_groups)
    metrics["filtered_groups"] = 0
    return metrics


def run_ablation_match_hard_support(all_cbs, all_rules, all_thresholds, all_groups,
                                    heldout_s, heldout_a, env_name,
                                    tau=DEFAULT_TAU, n_bootstrap=N_BOOTSTRAP):
    """Ablation 2: match + hard support pruning (no aggregation).
    
    Filter groups by support >= ceil(τ * B), then use first rule as representative.
    This isolates the impact of support pruning.
    """
    min_support = int(np.ceil(tau * n_bootstrap))
    kept_groups = []
    filtered_groups = []

    for group in all_groups:
        distinct_runs = len(set(run_idx for run_idx, _ in group))
        if distinct_runs >= min_support:
            kept_groups.append(group)
        else:
            filtered_groups.append(group)

    # Use first rule from each kept group (no aggregation)
    rules = []
    for group in kept_groups:
        _, first_rule = group[0]
        rules.append(first_rule)

    n_rules = len(rules)
    agg_thresholds = aggregate_thresholds(all_thresholds, "median")
    pipeline = make_consensus_pipeline(all_cbs[0], rules, agg_thresholds)
    metrics = _evaluate_pipeline(pipeline, heldout_s, heldout_a, env_name, n_rules)
    metrics["surviving_groups"] = len(kept_groups)
    metrics["filtered_groups"] = len(filtered_groups)
    return metrics


def run_ablation_match_aggregation(all_cbs, all_rules, all_thresholds, all_groups,
                                   heldout_s, heldout_a, env_name):
    """Ablation 3: match + aggregation (no support pruning).
    
    All matched groups are aggregated (merge_rule_group), no filtering by τ.
    This isolates the impact of aggregation.
    """
    level_values = all_cbs[0].level_values_
    level_labels = all_cbs[0].level_labels_

    rules = []
    for group in all_groups:
        rules_in_group = [rule for _, rule in group]
        cr = merge_rule_group(rules_in_group, level_values, level_labels)
        rules.append(cr)

    n_rules = len(rules)
    agg_thresholds = aggregate_thresholds(all_thresholds, "median")
    pipeline = make_consensus_pipeline(all_cbs[0], rules, agg_thresholds)
    metrics = _evaluate_pipeline(pipeline, heldout_s, heldout_a, env_name, n_rules)
    metrics["surviving_groups"] = len(all_groups)
    metrics["filtered_groups"] = 0
    return metrics


def run_ablation_full_default(all_cbs, all_rules, all_thresholds, all_groups,
                              heldout_s, heldout_a, env_name,
                              tau=DEFAULT_TAU, n_bootstrap=N_BOOTSTRAP):
    """Ablation 4: full default consensus merge.
    
    Hard support pruning + aggregation — the standard consensus CBS.
    """
    min_support = int(np.ceil(tau * n_bootstrap))
    level_values = all_cbs[0].level_values_
    level_labels = all_cbs[0].level_labels_

    kept_groups = []
    filtered_groups = []
    for group in all_groups:
        distinct_runs = len(set(run_idx for run_idx, _ in group))
        if distinct_runs >= min_support:
            kept_groups.append(group)
        else:
            filtered_groups.append(group)

    rules = []
    for group in kept_groups:
        rules_in_group = [rule for _, rule in group]
        cr = merge_rule_group(rules_in_group, level_values, level_labels)
        rules.append(cr)

    n_rules = len(rules)
    agg_thresholds = aggregate_thresholds(all_thresholds, "median")
    pipeline = make_consensus_pipeline(all_cbs[0], rules, agg_thresholds)
    metrics = _evaluate_pipeline(pipeline, heldout_s, heldout_a, env_name, n_rules)
    metrics["surviving_groups"] = len(kept_groups)
    metrics["filtered_groups"] = len(filtered_groups)
    return metrics


def run_ablation_v2_soft(all_cbs, all_rules, all_thresholds, all_groups,
                         heldout_s, heldout_a, env_name,
                         tau=DEFAULT_TAU, n_bootstrap=N_BOOTSTRAP):
    """Ablation 5: v2 soft support (repair reference).
    
    Soft support: support = (1/B) * sum_run max_sim, filter if >= τ.
    Otherwise same matching and aggregation.
    """
    level_values = all_cbs[0].level_values_
    level_labels = all_cbs[0].level_labels_

    kept_groups = []
    filtered_groups = []

    for group in all_groups:
        # Compute soft support: which runs contribute to this group
        run_rules_map = {}
        for run_idx, rule in group:
            if run_idx not in run_rules_map:
                run_rules_map[run_idx] = []
            run_rules_map[run_idx].append(rule)

        # For each run, compute max similarity to any rule in the group
        # If run has a rule in the group, max_sim = 1.0
        # Otherwise, check if any of that run's rules is similar enough
        group_rules = [rule for _, rule in group]
        soft_support = 0.0
        for run_idx in range(n_bootstrap):
            if run_idx in run_rules_map:
                soft_support += 1.0
            else:
                # Check all rules from this run against group rules
                run_all_rules = all_rules[run_idx]
                max_sim = 0.0
                for r1 in run_all_rules:
                    for r2 in group_rules:
                        if r1.action != r2.action:
                            continue
                        sim = rule_similarity_threshold_aware(
                            r1, r2,
                            lambda1=DEFAULT_LAMBDA1,
                            lambda2=DEFAULT_LAMBDA2,
                        )
                        max_sim = max(max_sim, sim)
                soft_support += max_sim

        soft_support /= n_bootstrap

        if soft_support >= tau:
            kept_groups.append(group)
        else:
            filtered_groups.append(group)

    rules = []
    for group in kept_groups:
        rules_in_group = [rule for _, rule in group]
        cr = merge_rule_group(rules_in_group, level_values, level_labels)
        rules.append(cr)

    n_rules = len(rules)
    agg_thresholds = aggregate_thresholds(all_thresholds, "median")
    pipeline = make_consensus_pipeline(all_cbs[0], rules, agg_thresholds)
    metrics = _evaluate_pipeline(pipeline, heldout_s, heldout_a, env_name, n_rules)
    metrics["surviving_groups"] = len(kept_groups)
    metrics["filtered_groups"] = len(filtered_groups)
    return metrics


def run_decomposition_experiment(env_name):
    """Run the full failure mechanism decomposition for one environment."""
    print(f"\n{'='*70}")
    print(f"  Failure Mechanism Decomposition: {env_name}")
    print(f"{'='*70}")

    env_tag = env_name.replace("-", "_").lower()
    model_path = get_model_path(env_name)

    # Collect held-out replay
    print("  Collecting held-out replay...")
    heldout_s, heldout_a = collect_heldout(env_name, model_path)
    print(f"  Held-out: {len(heldout_s)} transitions")

    all_results = {
        "schema_version": "failure_decomposition_v1",
        "env": env_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "n_outer_repeats": N_OUTER_REPEATS,
            "outer_seeds": OUTER_SEEDS,
            "n_bootstrap": N_BOOTSTRAP,
            "subsample_fraction": SUBSAMPLE_FRACTION,
            "tau": DEFAULT_TAU,
            "rho": DEFAULT_RHO,
            "lambda1": DEFAULT_LAMBDA1,
            "lambda2": DEFAULT_LAMBDA2,
        },
        "per_seed": {},
    }

    ablation_names = [
        "match_only",
        "match_hard_support",
        "match_aggregation",
        "full_default",
        "v2_soft_support",
    ]

    # Aggregate accumulators
    agg = {name: {k: [] for k in [
        "f1", "worst_action_recall", "E_CR", "n_rules",
        "surviving_groups", "filtered_groups"
    ]} for name in ablation_names}

    for seed in OUTER_SEEDS:
        print(f"\n  Outer seed {seed}...")

        # Collect replay for this seed
        data = collect_replay(
            env_name=env_name, model_path=model_path,
            num_transitions=10000, seed=seed, deterministic=True,
        )
        print(f"    Replay: {len(data['states'])} transitions")

        # Build internal subsamples
        all_cbs, all_rules, all_thresholds = _build_internal_subsamples(
            data, env_name, subsample_seed=seed * 100 + 42)

        # Match rules
        all_groups = _match_all_groups(all_rules)
        n_input_rules = sum(len(rs) for rs in all_rules)
        print(f"    Input rules: {n_input_rules}, Matched groups: {len(all_groups)}")

        seed_results = {}

        # Run 5 ablations
        ablation_fns = [
            ("match_only", lambda: run_ablation_match_only(
                all_cbs, all_rules, all_thresholds, all_groups,
                heldout_s, heldout_a, env_name)),
            ("match_hard_support", lambda: run_ablation_match_hard_support(
                all_cbs, all_rules, all_thresholds, all_groups,
                heldout_s, heldout_a, env_name)),
            ("match_aggregation", lambda: run_ablation_match_aggregation(
                all_cbs, all_rules, all_thresholds, all_groups,
                heldout_s, heldout_a, env_name)),
            ("full_default", lambda: run_ablation_full_default(
                all_cbs, all_rules, all_thresholds, all_groups,
                heldout_s, heldout_a, env_name)),
            ("v2_soft_support", lambda: run_ablation_v2_soft(
                all_cbs, all_rules, all_thresholds, all_groups,
                heldout_s, heldout_a, env_name)),
        ]

        for name, fn in ablation_fns:
            t0 = time.time()
            metrics = fn()
            elapsed = time.time() - t0
            metrics["elapsed_s"] = round(elapsed, 1)

            # Sanitize for JSON
            sanitized = {}
            for k, v in metrics.items():
                if k == "fidelity_per_action":
                    sanitized[k] = _serialize(v)
                elif isinstance(v, (np.integer, np.floating)):
                    sanitized[k] = float(v)
                else:
                    sanitized[k] = v

            seed_results[name] = sanitized

            # Accumulate
            for k in agg[name]:
                if k in sanitized:
                    agg[name][k].append(sanitized[k])

            print(f"    {name}: F1={metrics['f1']:.3f}, "
                  f"worst_recall={metrics['worst_action_recall']:.3f}, "
                  f"rules={metrics['n_rules']}, "
                  f"kept={metrics['surviving_groups']}, "
                  f"filtered={metrics['filtered_groups']}")

        all_results["per_seed"][str(seed)] = seed_results

    # Compute summary statistics
    summary = {}
    for name in ablation_names:
        summary[name] = {}
        for k in agg[name]:
            vals = agg[name][k]
            if vals:
                summary[name][k] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "min": round(float(np.min(vals)), 4),
                    "max": round(float(np.max(vals)), 4),
                }

    all_results["summary"] = summary

    # Print summary table
    print(f"\n  {'='*80}")
    print(f"  Summary (mean ± std across {N_OUTER_REPEATS} seeds):")
    print(f"  {'Ablation':<25} {'F1':>10} {'W-Recall':>10} "
          f"{'E_CR':>10} {'Rules':>8} {'Kept':>8} {'Filtered':>8}")
    print(f"  {'-'*80}")
    for name in ablation_names:
        s = summary[name]
        f1 = f"{s['f1']['mean']:.3f}±{s['f1']['std']:.3f}"
        wr = f"{s['worst_action_recall']['mean']:.3f}±{s['worst_action_recall']['std']:.3f}"
        ecr = f"{s['E_CR']['mean']:.1f}±{s['E_CR']['std']:.1f}"
        nr = f"{s['n_rules']['mean']:.1f}"
        kg = f"{s['surviving_groups']['mean']:.1f}"
        fg = f"{s['filtered_groups']['mean']:.1f}"
        print(f"  {name:<25} {f1:>10} {wr:>10} {ecr:>10} {nr:>8} {kg:>8} {fg:>8}")

    # Save results
    out_dir = f"experiments/results/{env_tag}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "failure_decomposition.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Failure Mechanism Decomposition")
    parser.add_argument("--env", default="all",
                        choices=ENVS + ["all"],
                        help="Environment to test")
    args = parser.parse_args()

    envs = ENVS if args.env == "all" else [args.env]
    for env in envs:
        run_decomposition_experiment(env)


if __name__ == "__main__":
    main()
