#!/usr/bin/env python
"""
Tuned consensus merge: 10-repeat results

Runs the tuned numeric consensus merge at the best known operating
point for each of three environments, with 10 independent outer repeats.

Environments & tuned configurations:
  MountainCar-v0:  B=5, tau=0.7, rho=0.9, lambda1=0.6, lambda2=0.4
  CartPole-v1:
    Group A: B=10, tau=0.5, rho=0.8, lambda1=0.6, lambda2=0.4
    Group B: B=5,  tau=0.7, rho=0.9, lambda1=0.5, lambda2=0.5
  LunarLander-v3:
    Group A: B=5,  tau=0.7, rho=0.9, lambda1=0.6, lambda2=0.4
    Group B: B=5,  tau=0.7, rho=0.9, lambda1=0.5, lambda2=0.5

Each repeat:
  1. Collect independent replay (10K transitions)
  2. Build consensus from B internal subsamples
  3. Evaluate on held-out replay (5K) and deployment (50 episodes)
  4. Record all per-run metrics

Output:
    experiments/results/tuned_merge/{env}/raw_runs.json
    experiments/results/tuned_merge/{env}/summary.json
    experiments/results/tuned_merge/{env}/tables.csv

Usage:
    python experiments/run_tuned_merge.py --env MountainCar-v0
    python experiments/run_tuned_merge.py --env CartPole-v1
    python experiments/run_tuned_merge.py --env LunarLander-v3
    python experiments/run_tuned_merge.py --env all
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from reproduction.collect_replay import collect_replay, ENV_FEATURE_NAMES
from experiments.perturbations import load_replay_npz, compute_feature_ranges
from experiments.consensus_merge import build_consensus_ruleset
from experiments.rule_matching import (
    canonicalize_rules,
    serialize_canonical_rules,
    mean_pairwise_jaccard,
    mean_pairwise_soft_jaccard,
    mean_pairwise_threshold_drift,
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
N_OUTER_REPEATS = 10
OUTER_SEEDS = list(range(N_OUTER_REPEATS))
OUT_ROOT = "experiments/results/tuned_merge"

# Tuned configurations per environment
# Each entry is a dict with a label and consensus parameters
ENV_CONFIGS = {
    "MountainCar-v0": [
        {
            "label": "tuned_merge",
            "B": 5, "tau": 0.7, "rho": 0.9,
            "lambda1": 0.6, "lambda2": 0.4,
        },
    ],
    "CartPole-v1": [
        {
            "label": "tuned_merge_A",
            "B": 10, "tau": 0.5, "rho": 0.8,
            "lambda1": 0.6, "lambda2": 0.4,
        },
        {
            "label": "tuned_merge_B",
            "B": 5, "tau": 0.7, "rho": 0.9,
            "lambda1": 0.5, "lambda2": 0.5,
        },
    ],
    "LunarLander-v3": [
        {
            "label": "tuned_merge_A",
            "B": 5, "tau": 0.7, "rho": 0.9,
            "lambda1": 0.6, "lambda2": 0.4,
        },
        {
            "label": "tuned_merge_B",
            "B": 5, "tau": 0.7, "rho": 0.9,
            "lambda1": 0.5, "lambda2": 0.5,
        },
    ],
}


def get_model_path(env_name):
    tag = env_name.replace("-", "_").lower()
    return f"reproduction/models/dqn_{tag}.zip"


def ci95(arr):
    a = np.array(arr, dtype=float)
    if len(a) < 2:
        return (float(a[0]), float(a[0])) if len(a) == 1 else (0.0, 0.0)
    se = a.std(ddof=1) / np.sqrt(len(a))
    return (float(a.mean() - 1.96 * se), float(a.mean() + 1.96 * se))


def run_single_config(env_name, config, heldout_s, heldout_a, model_path):
    """Run one tuned v1 configuration across all outer repeats."""
    label = config["label"]
    B = config["B"]
    tau = config["tau"]
    rho = config["rho"]
    lam1 = config["lambda1"]
    lam2 = config["lambda2"]

    print(f"\n  Config: {label}  B={B}, tau={tau}, rho={rho}, "
          f"lambda=({lam1},{lam2})")
    print(f"  Running {N_OUTER_REPEATS} outer repeats...")

    # Collect outer-repeat datasets
    outer_datasets = []
    for seed in OUTER_SEEDS:
        data = collect_replay(
            env_name=env_name, model_path=model_path,
            num_transitions=10000, seed=seed, deterministic=True,
        )
        outer_datasets.append(data)

    feature_ranges = compute_feature_ranges(outer_datasets[0])
    fr = {i: float(feature_ranges[i]) for i in range(len(feature_ranges))}

    all_rules = []
    all_thresholds = []
    all_preds = []
    per_repeat = {}

    for i, (seed, data) in enumerate(zip(OUTER_SEEDS, outer_datasets)):
        print(f"    [{i+1}/{N_OUTER_REPEATS}] seed={seed}...", end=" ")
        pipeline, rules, build_info = build_consensus_ruleset(
            data, env_name,
            n_bootstrap=B,
            consensus_threshold=tau,
            similarity_cutoff=rho,
            lambda1=lam1, lambda2=lam2,
        )
        res = evaluate_single_run(pipeline, rules, heldout_s, heldout_a, env_name)
        preds = pipeline.predict(heldout_s)
        thresholds = {int(k): [float(v) for v in vs]
                      for k, vs in res["thresholds"].items()}

        all_rules.append(rules)
        all_thresholds.append(thresholds)
        all_preds.append(preds)

        # Per-action recall
        pa = res["fidelity_per_action"]["per_action"]
        recalls = [pa[a]["recall"] for a in pa]
        worst_recall = min(recalls) if recalls else 0.0

        per_repeat[f"seed_{seed}"] = {
            "fidelity_heldout": res["fidelity_heldout"],
            "fidelity_per_action": res["fidelity_per_action"],
            "deployment": res["deployment"],
            "n_rules": len(rules),
            "worst_action_recall": worst_recall,
            "build_info": build_info,
            "rules": serialize_canonical_rules(rules),
        }
        f1 = res["fidelity_heldout"]["f1"]
        ecr = res["deployment"]["E_CR"]
        print(f"F1={f1:.3f}, E_CR={ecr:.1f}, rules={len(rules)}, "
              f"war={worst_recall:.3f}")

    # Stability across all repeats
    grs_wj = mean_pairwise_jaccard(all_rules, weighted=True)
    grs_ta = mean_pairwise_soft_jaccard(all_rules, threshold_aware=True)
    td = mean_pairwise_threshold_drift(all_thresholds, feature_ranges=fr)
    bra = compute_bra_from_predictions(all_preds)

    # Aggregate metrics
    f1_vals = [r["fidelity_heldout"]["f1"] for r in per_repeat.values()]
    acc_vals = [r["fidelity_heldout"]["accuracy"] for r in per_repeat.values()]
    recall_vals = [r["fidelity_heldout"].get("recall", 0.0) for r in per_repeat.values()]
    war_vals = [r["worst_action_recall"] for r in per_repeat.values()]
    ecr_vals = [r["deployment"]["E_CR"] for r in per_repeat.values()]
    n_rules_vals = [r["n_rules"] for r in per_repeat.values()]

    summary = {
        "F1": {"mean": float(np.mean(f1_vals)), "std": float(np.std(f1_vals)),
               "ci95": ci95(f1_vals)},
        "accuracy": {"mean": float(np.mean(acc_vals)), "std": float(np.std(acc_vals)),
                     "ci95": ci95(acc_vals)},
        "recall": {"mean": float(np.mean(recall_vals)), "std": float(np.std(recall_vals)),
                   "ci95": ci95(recall_vals)},
        "worst_action_recall": {"mean": float(np.mean(war_vals)), "std": float(np.std(war_vals)),
                                "ci95": ci95(war_vals)},
        "E_CR": {"mean": float(np.mean(ecr_vals)), "std": float(np.std(ecr_vals)),
                 "ci95": ci95(ecr_vals)},
        "rules": {"mean": float(np.mean(n_rules_vals)), "std": float(np.std(n_rules_vals))},
    }

    stability = {
        "GRS_wj": float(grs_wj),
        "GRS_ta": float(grs_ta),
        "BRA": float(bra),
        "TD": float(td),
        "n_repeats": N_OUTER_REPEATS,
        "n_pairs": N_OUTER_REPEATS * (N_OUTER_REPEATS - 1) // 2,
    }

    return {
        "config": config,
        "summary": summary,
        "stability": stability,
        "per_repeat": per_repeat,
    }


def run_env(env_name):
    """Run all tuned v1 configurations for one environment."""
    print(f"\n{'='*70}")
    print(f"  Tuned consensus merge: {env_name}")
    print(f"{'='*70}")

    model_path = get_model_path(env_name)
    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found at {model_path}. Skipping.")
        return None

    # Collect held-out replay
    print(f"  Collecting held-out replay (seed={HELDOUT_SEED})...")
    heldout = collect_replay(
        env_name=env_name, model_path=model_path,
        num_transitions=5000, seed=HELDOUT_SEED, deterministic=True,
    )
    heldout_s, heldout_a = heldout["states"], heldout["actions"]
    print(f"  Held-out: {len(heldout_s)} transitions")

    t0 = time.time()
    configs = ENV_CONFIGS[env_name]
    results = {}

    for cfg in configs:
        result = run_single_config(
            env_name, cfg, heldout_s, heldout_a, model_path)
        results[cfg["label"]] = result

    elapsed = time.time() - t0

    # Save results
    env_tag = env_name.replace("-", "_").lower()
    out_dir = os.path.join(OUT_ROOT, env_tag)
    os.makedirs(out_dir, exist_ok=True)

    output = {
        "schema_version": "tuned_merge_v1",
        "env": env_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "n_outer_repeats": N_OUTER_REPEATS,
        "outer_seeds": OUTER_SEEDS,
        "heldout_seed": HELDOUT_SEED,
        "heldout_size": len(heldout_s),
        "eval_episodes": len(EVAL_SEEDS),
        "results": {},
    }

    for label, res in results.items():
        output["results"][label] = {
            "config": res["config"],
            "summary": res["summary"],
            "stability": res["stability"],
            "per_repeat": res["per_repeat"],
        }

    # Save raw runs (full detail)
    raw_path = os.path.join(out_dir, "raw_runs.json")
    with open(raw_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Raw results saved to {raw_path}")

    # Save compact summary
    summary_output = {
        "env": env_name,
        "timestamp": output["timestamp"],
        "n_outer_repeats": N_OUTER_REPEATS,
    }
    for label, res in results.items():
        summary_output[label] = {
            "config": res["config"],
            "summary": res["summary"],
            "stability": res["stability"],
        }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_output, f, indent=2, default=str)
    print(f"  Summary saved to {summary_path}")

    # Save tables CSV
    rows = []
    for label, res in results.items():
        s = res["summary"]
        st = res["stability"]
        rows.append({
            "env": env_name,
            "method": label,
            "B": res["config"]["B"],
            "tau": res["config"]["tau"],
            "rho": res["config"]["rho"],
            "lambda1": res["config"]["lambda1"],
            "lambda2": res["config"]["lambda2"],
            "F1_mean": s["F1"]["mean"],
            "F1_std": s["F1"]["std"],
            "accuracy_mean": s["accuracy"]["mean"],
            "recall_mean": s["recall"]["mean"],
            "worst_R_mean": s["worst_action_recall"]["mean"],
            "worst_R_std": s["worst_action_recall"]["std"],
            "GRS_wj": st["GRS_wj"],
            "GRS_ta": st["GRS_ta"],
            "BRA": st["BRA"],
            "TD": st["TD"],
            "rules_mean": s["rules"]["mean"],
            "E_CR_mean": s["E_CR"]["mean"],
            "E_CR_std": s["E_CR"]["std"],
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "tables.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Tables saved to {csv_path}")

    # Print summary
    for label, res in results.items():
        s = res["summary"]
        st = res["stability"]
        print(f"\n  ── {label} ──")
        print(f"  F1:    {s['F1']['mean']:.3f} ± {s['F1']['std']:.3f}  "
              f"CI95 {s['F1']['ci95']}")
        print(f"  E_CR:  {s['E_CR']['mean']:.1f} ± {s['E_CR']['std']:.1f}")
        print(f"  WAR:   {s['worst_action_recall']['mean']:.3f} ± "
              f"{s['worst_action_recall']['std']:.3f}")
        print(f"  Rules: {s['rules']['mean']:.1f} ± {s['rules']['std']:.1f}")
        print(f"  GRS_wj: {st['GRS_wj']:.4f}  GRS_ta: {st['GRS_ta']:.4f}")
        print(f"  BRA: {st['BRA']:.4f}  TD: {st['TD']:.4f}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Tuned consensus merge results")
    parser.add_argument("--env", default="all",
                        choices=ENVS + ["all"],
                        help="Environment to run (default: all)")
    args = parser.parse_args()

    envs = ENVS if args.env == "all" else [args.env]

    print(f"\n{'#'*70}")
    print(f"  TUNED CONSENSUS MERGE")
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
