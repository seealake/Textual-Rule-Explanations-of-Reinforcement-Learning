#!/usr/bin/env python
"""
Matching/Canonicalization Robustness (Cross-Environment)

Verify that ρ/λ sweep conclusions from MountainCar replicate on CartPole and LunarLander.

Protocol:
  - ρ ∈ {0.7, 0.8, 0.9}, λ ∈ {(0.5,0.5), (0.6,0.4), (0.7,0.3)}
  - 5 outer repeats (seed-shift replays) per cell
  - Check: method ranking stable across environments

Usage:
    python experiments/run_matching_robustness.py --env CartPole-v1
    python experiments/run_matching_robustness.py --env LunarLander-v3
    python experiments/run_matching_robustness.py --env all

Output:
    experiments/results/<env>/matching_robustness_results.json
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
from experiments.consensus_merge import build_consensus_ruleset, make_consensus_pipeline
from experiments.rule_matching import (
    canonicalize_rules,
    mean_pairwise_jaccard,
    mean_pairwise_soft_jaccard,
    mean_pairwise_threshold_drift,
)
from experiments.perturbations import compute_feature_ranges

# ── Constants ─────────────────────────────────────────────────────────
ENVS = ["CartPole-v1", "LunarLander-v3"]  # MC already done
RHO_VALUES = [0.7, 0.8, 0.9]
LAMBDA_PAIRS = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]
N_OUTER_REPEATS = 5
OUTER_SEEDS = [0, 1, 2, 3, 4]
HELDOUT_SEED = 99
RESULTS_DIR = "experiments/results"


def get_model_path(env_name):
    tag = env_name.replace("-", "_").lower()
    return f"reproduction/models/dqn_{tag}.zip"


def get_ncat(env_name):
    return 6 if env_name == "LunarLander-v3" else 5


def run_single_consensus(env_name, model_path, replay_seed, heldout_s, heldout_a,
                          rho=0.8, lambda1=0.6, lambda2=0.4,
                          n_bootstrap=5, tau=0.7):
    """Run one consensus CBS and return metrics."""
    data = collect_replay(env_name, model_path, num_transitions=10000, seed=replay_seed)
    ncat = get_ncat(env_name)

    base_data = {
        "states": data["states"],
        "actions": data["actions"],
        "rewards": np.zeros(len(data["actions"])),
        "dones": np.zeros(len(data["actions"]), dtype=bool),
        "episode_ids": np.zeros(len(data["actions"]), dtype=int),
    }

    pipeline, rules, build_info = build_consensus_ruleset(
        base_data, env_name,
        n_bootstrap=n_bootstrap,
        consensus_threshold=tau,
        similarity_cutoff=rho,
        lambda1=lambda1,
        lambda2=lambda2,
    )

    # build_consensus_ruleset already returns a fully-built consensus pipeline
    consensus_pipe = pipeline

    # Evaluate
    fidelity = consensus_pipe.evaluate_fidelity(heldout_s, heldout_a)
    canonical = canonicalize_rules(consensus_pipe.rules_)
    thresholds = {int(k): [float(v) for v in vals] for k, vals in consensus_pipe.thresholds_.items()}
    preds = consensus_pipe.predict(heldout_s)

    return {
        "f1": float(fidelity["f1"]),
        "n_rules": len(consensus_pipe.rules_),
        "rules": canonical,
        "thresholds": thresholds,
        "preds": preds,
    }


def compute_stability_across_repeats(repeat_results, feature_ranges):
    """Compute stability metrics across N outer repeats."""
    rule_sets = [r["rules"] for r in repeat_results]
    threshold_sets = [r["thresholds"] for r in repeat_results]
    pred_sets = [r["preds"] for r in repeat_results]

    grs_wj = mean_pairwise_jaccard(rule_sets, weighted=True)
    grs_ta = mean_pairwise_soft_jaccard(rule_sets, threshold_aware=True)
    td = mean_pairwise_threshold_drift(threshold_sets, feature_ranges)

    # BRA computed from prediction arrays directly
    n = len(pred_sets)
    bra_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            agree = float(np.mean(pred_sets[i] == pred_sets[j]))
            bra_pairs.append(agree)
    bra = float(np.mean(bra_pairs)) if bra_pairs else 1.0

    f1_vals = [r["f1"] for r in repeat_results]
    n_rules_vals = [r["n_rules"] for r in repeat_results]

    return {
        "GRS_wj": float(grs_wj),
        "GRS_ta": float(grs_ta),
        "TD": float(td),
        "BRA": float(bra),
        "mean_f1": float(np.mean(f1_vals)),
        "std_f1": float(np.std(f1_vals)),
        "mean_n_rules": float(np.mean(n_rules_vals)),
    }


def run_matching_robustness(env_name):
    """Run matching robustness experiment for one environment."""
    print(f"\n{'='*60}")
    print(f"Matching/Canonicalization Robustness: {env_name}")
    print(f"{'='*60}")
    t0 = time.time()

    model_path = get_model_path(env_name)
    feature_names = ENV_FEATURE_NAMES[env_name]

    # Collect held-out
    print("  Collecting held-out replay...")
    heldout = collect_replay(env_name, model_path, num_transitions=5000, seed=HELDOUT_SEED)
    heldout_s, heldout_a = heldout["states"], heldout["actions"]

    # Feature ranges for TD
    ref = collect_replay(env_name, model_path, num_transitions=10000, seed=42)
    feature_ranges_arr = compute_feature_ranges({"states": ref["states"]})
    feature_ranges = {i: float(feature_ranges_arr[i]) for i in range(len(feature_ranges_arr))}

    # ρ sweep
    print(f"\n  rho sweep (lambda1=0.6, lambda2=0.4, tau=0.7, B=5):")
    rho_results = {}
    for rho in RHO_VALUES:
        print(f"    rho={rho}...", end=" ")
        repeats = []
        for seed in OUTER_SEEDS:
            r = run_single_consensus(env_name, model_path, seed, heldout_s, heldout_a,
                                      rho=rho, lambda1=0.6, lambda2=0.4)
            repeats.append(r)
        stability = compute_stability_across_repeats(repeats, feature_ranges)
        rho_results[f"rho_{rho}"] = stability
        print(f"F1={stability['mean_f1']:.3f}, GRS={stability['GRS_wj']:.3f}, BRA={stability['BRA']:.3f}")

    # λ sweep
    print(f"\n  lambda sweep (rho=0.8, tau=0.7, B=5):")
    lambda_results = {}
    for l1, l2 in LAMBDA_PAIRS:
        print(f"    lambda=({l1},{l2})...", end=" ")
        repeats = []
        for seed in OUTER_SEEDS:
            r = run_single_consensus(env_name, model_path, seed, heldout_s, heldout_a,
                                      rho=0.8, lambda1=l1, lambda2=l2)
            repeats.append(r)
        stability = compute_stability_across_repeats(repeats, feature_ranges)
        lambda_results[f"l{l1}_{l2}"] = stability
        print(f"F1={stability['mean_f1']:.3f}, GRS={stability['GRS_wj']:.3f}, BRA={stability['BRA']:.3f}")

    elapsed = time.time() - t0
    tag = env_name.replace("-", "_").lower()

    output = {
        "schema_version": "matching_robustness_v1",
        "env": env_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "n_outer_repeats": N_OUTER_REPEATS,
        "rho_sweep": rho_results,
        "lambda_sweep": lambda_results,
    }

    out_dir = os.path.join(RESULTS_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "matching_robustness_results.json")

    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return str(obj)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=_serialize)

    print(f"\nResults saved to {out_path}")
    print(f"Elapsed: {elapsed:.1f}s")
    return output


def main():
    parser = argparse.ArgumentParser(description="Matching robustness sweep")
    parser.add_argument("--env", default="all",
                        choices=["CartPole-v1", "LunarLander-v3", "all"])
    args = parser.parse_args()

    envs = ENVS if args.env == "all" else [args.env]
    for env_name in envs:
        run_matching_robustness(env_name)


if __name__ == "__main__":
    main()
