#!/usr/bin/env python
"""
Confirmatory rerun: Consensus CBS with rho=0.9 on MountainCar-v0.

Goal: Verify whether the operating point (B=5, tau=0.7, rho=0.9)
Pareto-dominates CBS with additional outer repeats (10 instead of 5).

Usage:
    python experiments/run_match_threshold_check.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reproduction.collect_replay import collect_replay
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
ENV_NAME = "MountainCar-v0"
MODEL_PATH = "reproduction/models/dqn_mountaincar_v0.zip"
REPLAY_PATH = "reproduction/data/replay_mountaincar_v0_seed42.npz"

# Operating point under test
B = 5
TAU = 0.7
RHO = 0.9
LAMBDA1 = 0.6
LAMBDA2 = 0.4

# 10 outer-repeat seeds (superset of original 5)
OUTER_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def main():
    t0 = time.time()
    print(f"{'='*60}")
    print(f"  Confirmatory Rerun: rho=0.9 on {ENV_NAME}")
    print(f"  B={B}, tau={TAU}, rho={RHO}, lambda=({LAMBDA1},{LAMBDA2})")
    print(f"  Outer repeats: {len(OUTER_SEEDS)} seeds")
    print(f"{'='*60}")

    # Collect held-out replay
    print(f"\n  Collecting held-out replay (seed={HELDOUT_SEED})...")
    heldout = collect_replay(
        env_name=ENV_NAME, model_path=MODEL_PATH,
        num_transitions=5000, seed=HELDOUT_SEED, deterministic=True,
    )
    heldout_s, heldout_a = heldout["states"], heldout["actions"]
    print(f"  Held-out: {len(heldout_s)} transitions")

    # Collect outer-repeat datasets
    print(f"\n  Collecting {len(OUTER_SEEDS)} outer-repeat replays...")
    outer_datasets = []
    for seed in OUTER_SEEDS:
        data = collect_replay(
            env_name=ENV_NAME, model_path=MODEL_PATH,
            num_transitions=10000, seed=seed, deterministic=True,
        )
        outer_datasets.append(data)
        print(f"    seed={seed}: {len(data['states'])} transitions")

    feature_ranges = compute_feature_ranges(outer_datasets[0])
    fr = {i: float(feature_ranges[i]) for i in range(len(feature_ranges))}

    # Run consensus for each outer repeat
    print(f"\n  Running {len(OUTER_SEEDS)} consensus builds...")
    all_rules = []
    all_thresholds = []
    all_preds = []
    per_repeat_results = {}

    for i, (seed, data) in enumerate(zip(OUTER_SEEDS, outer_datasets)):
        print(f"    [{i+1}/{len(OUTER_SEEDS)}] seed={seed}...", end=" ")
        pipeline, rules, build_info = build_consensus_ruleset(
            data, ENV_NAME,
            n_bootstrap=B,
            consensus_threshold=TAU,
            similarity_cutoff=RHO,
            lambda1=LAMBDA1, lambda2=LAMBDA2,
        )
        res = evaluate_single_run(pipeline, rules, heldout_s, heldout_a, ENV_NAME)
        preds = pipeline.predict(heldout_s)
        thresholds = {int(k): v for k, v in res["thresholds"].items()}

        all_rules.append(rules)
        all_thresholds.append(thresholds)
        all_preds.append(preds)

        # Per-action recall
        pa = res["fidelity_per_action"]["per_action"]
        recalls = [pa[a]["recall"] for a in pa]
        worst_recall = min(recalls) if recalls else 0.0

        per_repeat_results[f"seed_{seed}"] = {
            "fidelity_heldout": res["fidelity_heldout"],
            "fidelity_per_action": res["fidelity_per_action"],
            "deployment": res["deployment"],
            "n_rules": len(rules),
            "worst_action_recall": worst_recall,
            "build_info": build_info,
            "rules": serialize_canonical_rules(rules),
        }
        print(f"F1={res['fidelity_heldout']['f1']:.3f}, "
              f"E_CR={res['deployment']['E_CR']:.1f}, "
              f"rules={len(rules)}, war={worst_recall:.3f}")

    # Compute stability across ALL outer repeats
    print(f"\n  Computing stability across {len(OUTER_SEEDS)} repeats...")
    grs_wj = mean_pairwise_jaccard(all_rules, weighted=True)
    grs_ta = mean_pairwise_soft_jaccard(all_rules, threshold_aware=True)
    td = mean_pairwise_threshold_drift(all_thresholds, feature_ranges=fr)
    bra = compute_bra_from_predictions(all_preds)

    # Also compute for first-5 subset (for direct comparison with original)
    grs_wj_5 = mean_pairwise_jaccard(all_rules[:5], weighted=True)
    grs_ta_5 = mean_pairwise_soft_jaccard(all_rules[:5], threshold_aware=True)
    td_5 = mean_pairwise_threshold_drift(all_thresholds[:5], feature_ranges=fr)
    bra_5 = compute_bra_from_predictions(all_preds[:5])

    # Aggregate fidelity & deployment metrics
    f1_vals = [r["fidelity_heldout"]["f1"] for r in per_repeat_results.values()]
    ecr_vals = [r["deployment"]["E_CR"] for r in per_repeat_results.values()]
    war_vals = [r["worst_action_recall"] for r in per_repeat_results.values()]
    n_rules_vals = [r["n_rules"] for r in per_repeat_results.values()]

    def ci95(arr):
        a = np.array(arr)
        se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
        return float(a.mean() - 1.96 * se), float(a.mean() + 1.96 * se)

    elapsed = time.time() - t0

    # Build output
    output = {
        "schema_version": "match_threshold_check_v1",
        "env": ENV_NAME,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "purpose": "Confirmatory rerun: does rho=0.9 Pareto-dominate CBS?",
        "config": {
            "B": B, "tau": TAU, "rho": RHO,
            "lambda1": LAMBDA1, "lambda2": LAMBDA2,
            "n_outer_repeats": len(OUTER_SEEDS),
            "outer_seeds": OUTER_SEEDS,
            "heldout_seed": HELDOUT_SEED,
            "heldout_size": len(heldout_s),
            "deployment_episodes": len(EVAL_SEEDS),
        },
        "summary": {
            "F1": {
                "mean": float(np.mean(f1_vals)),
                "std": float(np.std(f1_vals)),
                "ci95": ci95(f1_vals),
            },
            "E_CR": {
                "mean": float(np.mean(ecr_vals)),
                "std": float(np.std(ecr_vals)),
                "ci95": ci95(ecr_vals),
            },
            "worst_action_recall": {
                "mean": float(np.mean(war_vals)),
                "std": float(np.std(war_vals)),
                "ci95": ci95(war_vals),
            },
            "rules": {
                "mean": float(np.mean(n_rules_vals)),
                "std": float(np.std(n_rules_vals)),
            },
        },
        "stability_10_repeats": {
            "GRS_wj": float(grs_wj),
            "GRS_ta": float(grs_ta),
            "BRA": float(bra),
            "TD": float(td),
            "n_repeats": len(OUTER_SEEDS),
            "n_pairs": len(OUTER_SEEDS) * (len(OUTER_SEEDS) - 1) // 2,
        },
        "stability_first5_repeats": {
            "GRS_wj": float(grs_wj_5),
            "GRS_ta": float(grs_ta_5),
            "BRA": float(bra_5),
            "TD": float(td_5),
            "n_repeats": 5,
            "n_pairs": 10,
            "note": "Direct comparison with original 5-repeat ablation",
        },
        "per_repeat": per_repeat_results,
    }

    # Save
    out_dir = "experiments/results/mountaincar_v0"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "match_threshold_check.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    # Print summary
    print(f"\n  {'='*60}")
    print(f"  CONFIRMATORY RESULTS: rho=0.9 ({len(OUTER_SEEDS)} repeats)")
    print(f"  {'='*60}")
    print(f"  F1:    {np.mean(f1_vals):.3f} +/- {np.std(f1_vals):.3f}  "
          f"95%CI {ci95(f1_vals)}")
    print(f"  E_CR:  {np.mean(ecr_vals):.1f} +/- {np.std(ecr_vals):.1f}  "
          f"95%CI {ci95(ecr_vals)}")
    print(f"  WAR:   {np.mean(war_vals):.3f} +/- {np.std(war_vals):.3f}")
    print(f"  rules: {np.mean(n_rules_vals):.1f} +/- {np.std(n_rules_vals):.1f}")
    print(f"\n  Stability (10 repeats / 45 pairs):")
    print(f"    GRS_wj: {grs_wj:.4f}")
    print(f"    GRS-TA: {grs_ta:.4f}")
    print(f"    BRA:    {bra:.4f}")
    print(f"    TD:     {td:.4f}")
    print(f"\n  Stability (first 5 repeats, for comparison):")
    print(f"    GRS_wj: {grs_wj_5:.4f}")
    print(f"    GRS-TA: {grs_ta_5:.4f}")
    print(f"    BRA:    {bra_5:.4f}")
    print(f"    TD:     {td_5:.4f}")

    # Load CBS baseline for comparison
    st = json.load(open("experiments/results/mountaincar_v0/stress_test_results.json"))
    cbs_st = st["cbs"]["stability"]
    cbs_runs = st["cbs"]["per_run"]
    cbs_f1s = [r["fidelity_heldout"]["f1"] for r in cbs_runs.values()]
    cbs_ecrs = [r["deployment"]["E_CR"] for r in cbs_runs.values()]

    print(f"\n  CBS baseline (21 runs):")
    print(f"    F1:    {np.mean(cbs_f1s):.3f} +/- {np.std(cbs_f1s):.3f}")
    print(f"    E_CR:  {np.mean(cbs_ecrs):.1f} +/- {np.std(cbs_ecrs):.1f}")
    print(f"    GRS_wj: {cbs_st['GRS_weighted_jaccard']:.4f}")
    print(f"    GRS-TA: {cbs_st['GRS_threshold_aware']:.4f}")
    print(f"    BRA:    {cbs_st['BRA']:.4f}")
    print(f"    TD:     {cbs_st['TD']:.4f}")

    # Pareto dominance check
    rho09_better_f1 = np.mean(f1_vals) >= np.mean(cbs_f1s)
    rho09_better_grs = grs_wj >= cbs_st["GRS_weighted_jaccard"]
    rho09_better_ecr = np.mean(ecr_vals) >= np.mean(cbs_ecrs)
    pareto = rho09_better_f1 and rho09_better_grs

    print(f"\n  PARETO DOMINANCE CHECK:")
    print(f"    F1:  rho=0.9 ({np.mean(f1_vals):.3f}) "
          f"{'≥' if rho09_better_f1 else '<'} CBS ({np.mean(cbs_f1s):.3f})")
    print(f"    GRS: rho=0.9 ({grs_wj:.4f}) "
          f"{'≥' if rho09_better_grs else '<'} CBS ({cbs_st['GRS_weighted_jaccard']:.4f})")
    print(f"    E_CR: rho=0.9 ({np.mean(ecr_vals):.1f}) "
          f"{'≥' if rho09_better_ecr else '<'} CBS ({np.mean(cbs_ecrs):.1f})")
    print(f"    => Pareto-dominates CBS on (F1, GRS): "
          f"{'YES' if pareto else 'NO'}")
    print(f"\n  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
