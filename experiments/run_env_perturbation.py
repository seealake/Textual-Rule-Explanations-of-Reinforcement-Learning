#!/usr/bin/env python
"""
Environment perturbation

Tests whether explanation instability correlates with policy robustness
degradation under physics perturbations (gravity, force, wind, turbulence).

For each perturbation level:
  1. Evaluate policy return in perturbed env
  2. Collect perturbed replay (5 seeds)
  3. Fit CBS / rule-set voting / DT → extract rules
  4. Compare to clean anchor: ΔF1, ΔGRS, ΔBRA
  5. Scatter: Δreturn vs Δstability, Spearman correlation

Usage:
    python experiments/run_env_perturbation.py --env MountainCar-v0
    python experiments/run_env_perturbation.py --env LunarLander-v3
    python experiments/run_env_perturbation.py --env all

Output:
    experiments/results/<env>/env_perturbation_results.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats as scipy_stats

from reproduction.cbs import CBSPipeline
from reproduction.collect_replay import collect_replay, ENV_FEATURE_NAMES
from experiments.env_perturbation import (
    ENV_PERTURBATIONS,
    collect_replay_perturbed,
    evaluate_policy_return,
)
from experiments.rule_matching import (
    canonicalize_rules,
    serialize_canonical_rules,
    ruleset_weighted_jaccard,
    ruleset_soft_jaccard,
    threshold_drift,
)
from experiments.perturbations import compute_feature_ranges
from experiments.decision_tree_surrogate import DecisionTreeSurrogate, find_best_depth
from experiments.consensus_merge import build_voting_ensemble, voting_predict

# ── Constants ─────────────────────────────────────────────────────────
ENVS = ["MountainCar-v0", "LunarLander-v3"]
EVAL_SEEDS = list(range(1000, 1050))  # 50 fixed deployment seeds
REPLAY_SEEDS = [0, 1, 2, 3, 4]  # 5 seeds per perturbation level
HELDOUT_SEED = 99
SUCCESS_THRESHOLDS = {"MountainCar-v0": -150.0, "LunarLander-v3": 200.0}
RESULTS_DIR = "experiments/results"


def get_model_path(env_name):
    tag = env_name.replace("-", "_").lower()
    return f"reproduction/models/dqn_{tag}.zip"


def get_ncat(env_name):
    return 6 if env_name == "LunarLander-v3" else 5


def fit_cbs(states, actions, env_name, kmeans_seed=0):
    """Fit CBS and return (pipeline, canonical_rules, thresholds)."""
    ncat = get_ncat(env_name)
    cbs = CBSPipeline(n_categories=ncat, inclusion_threshold=0.70, kmeans_seed=kmeans_seed)
    cbs.fit(states, actions)
    rules = canonicalize_rules(cbs.rules_)
    thresholds = {int(k): [float(v) for v in vals] for k, vals in cbs.thresholds_.items()}
    return cbs, rules, thresholds


def fit_dt(states, actions, env_name):
    """Fit DT surrogate and return (pipeline, canonical_rules, thresholds)."""
    from experiments.decision_tree_surrogate import canonicalize_dt_rules
    best_depth, _ = find_best_depth(states, actions)
    dt = DecisionTreeSurrogate(max_depth=best_depth)
    dt.fit(states, actions)
    rules = canonicalize_dt_rules(dt.get_rules())
    thresholds = dt.get_thresholds()
    thresholds = {int(k): [float(v) for v in vals] for k, vals in thresholds.items()}
    return dt, rules, thresholds


def fit_b3_vote(states, actions, env_name):
    """Fit rule-set voting ensemble and return (pipelines, canonical_rules_list, avg_thresholds)."""
    from experiments.perturbations import generate_subsamples
    data = {"states": states, "actions": actions,
            "rewards": np.zeros(len(actions)), "dones": np.zeros(len(actions), dtype=bool),
            "episode_ids": np.zeros(len(actions), dtype=int)}
    subsamples = generate_subsamples(data, n_subsets=5, fraction=0.8, seed=42)
    pipelines = []
    all_rules = []
    for ss in subsamples:
        cbs, rules, _ = fit_cbs(ss["states"], ss["actions"], env_name)
        pipelines.append(cbs)
        all_rules.extend(rules)
    # For comparison purposes, aggregate canonical rules from all sub-pipelines
    return pipelines, all_rules, {}


def compute_pairwise_metrics(rules_a, rules_b, thresh_a, thresh_b, feature_ranges,
                             preds_a=None, preds_b=None):
    """Compute stability metrics between two rule sets."""
    grs_wj = ruleset_weighted_jaccard(rules_a, rules_b)
    grs_ta = ruleset_soft_jaccard(rules_a, rules_b, threshold_aware=True)
    td = threshold_drift(thresh_a, thresh_b, feature_ranges) if thresh_a and thresh_b else None
    bra = float(np.mean(preds_a == preds_b)) if preds_a is not None and preds_b is not None else None
    return {"GRS_wj": grs_wj, "GRS_ta": grs_ta, "TD": td, "BRA": bra}


def run_env_perturbation(env_name):
    """Run environment perturbation experiment for one environment."""
    print(f"\n{'='*60}")
    print(f"Environment Perturbation: {env_name}")
    print(f"{'='*60}")
    t0 = time.time()

    model_path = get_model_path(env_name)
    perturbation_configs = ENV_PERTURBATIONS[env_name]
    feature_names = ENV_FEATURE_NAMES[env_name]

    # ── Step 1: Collect clean held-out replay for evaluation ──────────
    print("\nCollecting clean held-out replay (seed=99)...")
    heldout = collect_replay(env_name, model_path, num_transitions=5000, seed=HELDOUT_SEED)
    heldout_s = heldout["states"]
    heldout_a = heldout["actions"]

    # ── Step 2: Collect clean reference replay & fit anchors ──────────
    print("Collecting clean reference replay (seed=42)...")
    clean_replay = collect_replay(env_name, model_path, num_transitions=10000, seed=42)
    clean_s = clean_replay["states"]
    clean_a = clean_replay["actions"]
    feature_ranges_arr = compute_feature_ranges({"states": clean_s})
    # threshold_drift expects dict {feature_idx: range}, not numpy array
    feature_ranges = {i: float(feature_ranges_arr[i]) for i in range(len(feature_ranges_arr))}

    print("Fitting clean anchors (CBS, rule-set voting, DT)...")
    cbs_anchor, cbs_anchor_rules, cbs_anchor_thresh = fit_cbs(clean_s, clean_a, env_name)
    cbs_anchor_preds = cbs_anchor.predict(heldout_s)
    cbs_anchor_f1 = cbs_anchor.evaluate_fidelity(heldout_s, heldout_a)["f1"]

    dt_anchor, dt_anchor_rules, dt_anchor_thresh = fit_dt(clean_s, clean_a, env_name)
    dt_anchor_preds = dt_anchor.predict(heldout_s)
    dt_anchor_f1 = dt_anchor.evaluate_fidelity(heldout_s, heldout_a)["f1"]

    vote_pipelines, vote_anchor_rules, _ = fit_b3_vote(clean_s, clean_a, env_name)
    vote_anchor_preds = voting_predict(vote_pipelines, heldout_s)
    vote_anchor_f1 = float(np.mean(vote_anchor_preds == heldout_a))

    # ── Step 3: Evaluate clean policy return ──────────────────────────
    print("Evaluating clean policy return...")
    clean_return = evaluate_policy_return(model_path, env_name, {}, EVAL_SEEDS[:20])
    print(f"  Clean policy return: {clean_return['mean_return']:.1f} ± {clean_return['std_return']:.1f}")

    # ── Step 4: For each perturbation level ───────────────────────────
    results_per_perturbation = {}

    for pconfig in perturbation_configs:
        pname = pconfig["name"]
        pparams = pconfig["params"]
        ptype = pconfig["type"]
        severity = pconfig["severity"]

        if ptype == "clean":
            continue  # skip, we already have clean

        print(f"\n--- Perturbation: {pname} (type={ptype}, severity={severity}) ---")

        # Evaluate perturbed policy return
        perturbed_return = evaluate_policy_return(model_path, env_name, pparams, EVAL_SEEDS[:20])
        delta_return = perturbed_return["mean_return"] - clean_return["mean_return"]
        relative_return_drop = (clean_return["mean_return"] - perturbed_return["mean_return"]) / abs(clean_return["mean_return"]) if clean_return["mean_return"] != 0 else 0
        print(f"  Policy return: {perturbed_return['mean_return']:.1f} (Δ={delta_return:+.1f}, "
              f"relative drop={relative_return_drop:.1%})")

        # Collect perturbed replay across seeds and fit methods
        seed_results = {"cbs": [], "dt": [], "b3_vote": []}

        for seed in REPLAY_SEEDS:
            # Collect perturbed replay
            prep = collect_replay_perturbed(env_name, model_path, pparams,
                                           num_transitions=10000, seed=seed)
            ps, pa = prep["states"], prep["actions"]

            # CBS
            try:
                cbs_p, cbs_p_rules, cbs_p_thresh = fit_cbs(ps, pa, env_name)
                cbs_p_preds = cbs_p.predict(heldout_s)
                cbs_p_f1 = cbs_p.evaluate_fidelity(heldout_s, heldout_a)["f1"]
                cbs_metrics = compute_pairwise_metrics(
                    cbs_anchor_rules, cbs_p_rules, cbs_anchor_thresh, cbs_p_thresh,
                    feature_ranges, cbs_anchor_preds, cbs_p_preds)
                cbs_metrics["f1"] = cbs_p_f1
                cbs_metrics["delta_f1"] = cbs_p_f1 - cbs_anchor_f1
                seed_results["cbs"].append(cbs_metrics)
            except Exception as e:
                print(f"    CBS seed {seed} failed: {e}")

            # DT
            try:
                dt_p, dt_p_rules, dt_p_thresh = fit_dt(ps, pa, env_name)
                dt_p_preds = dt_p.predict(heldout_s)
                dt_p_f1 = dt_p.evaluate_fidelity(heldout_s, heldout_a)["f1"]
                dt_metrics = compute_pairwise_metrics(
                    dt_anchor_rules, dt_p_rules, dt_anchor_thresh, dt_p_thresh,
                    feature_ranges, dt_anchor_preds, dt_p_preds)
                dt_metrics["f1"] = dt_p_f1
                dt_metrics["delta_f1"] = dt_p_f1 - dt_anchor_f1
                seed_results["dt"].append(dt_metrics)
            except Exception as e:
                print(f"    DT seed {seed} failed: {e}")

            # rule-set voting
            try:
                vote_ps, vote_p_rules, _ = fit_b3_vote(ps, pa, env_name)
                vote_p_preds = voting_predict(vote_ps, heldout_s)
                vote_p_f1 = float(np.mean(vote_p_preds == heldout_a))
                vote_bra = float(np.mean(vote_anchor_preds == vote_p_preds))
                seed_results["b3_vote"].append({
                    "f1": vote_p_f1, "delta_f1": vote_p_f1 - vote_anchor_f1,
                    "BRA": vote_bra, "GRS_wj": None, "GRS_ta": None, "TD": None,
                })
            except Exception as e:
                print(f"    rule-set voting seed {seed} failed: {e}")

        # Aggregate across seeds
        def aggregate_method(method_results):
            if not method_results:
                return None
            agg = {}
            for key in method_results[0]:
                vals = [r[key] for r in method_results if r.get(key) is not None]
                if vals:
                    agg[f"mean_{key}"] = float(np.mean(vals))
                    agg[f"std_{key}"] = float(np.std(vals))
            return agg

        perturbation_result = {
            "name": pname,
            "type": ptype,
            "params": {k: float(v) if isinstance(v, (int, float)) else v for k, v in pparams.items()},
            "severity": severity,
            "policy_return": {
                "mean": perturbed_return["mean_return"],
                "std": perturbed_return["std_return"],
                "delta": delta_return,
                "relative_drop": relative_return_drop,
            },
            "cbs": aggregate_method(seed_results["cbs"]),
            "dt": aggregate_method(seed_results["dt"]),
            "b3_vote": aggregate_method(seed_results["b3_vote"]),
        }

        for method in ["cbs", "dt", "b3_vote"]:
            agg = perturbation_result[method]
            if agg:
                bra_key = "mean_BRA"
                f1_key = "mean_delta_f1"
                if bra_key in agg:
                    print(f"  {method:10s}: ΔF1={agg.get(f1_key, 0):+.3f}, "
                          f"BRA={agg.get(bra_key, 0):.3f}, "
                          f"GRS_wj={agg.get('mean_GRS_wj', 'N/A')}")

        results_per_perturbation[pname] = perturbation_result

    # ── Step 5: Correlation analysis ──────────────────────────────────
    print("\n--- Correlation: Δreturn vs Δstability ---")
    correlations = {}

    for method in ["cbs", "dt", "b3_vote"]:
        x_vals = []  # relative return drop
        y_grs = []   # GRS_wj (1 - GRS = instability)
        y_bra = []   # BRA drop

        for pname, presult in results_per_perturbation.items():
            method_agg = presult.get(method)
            if method_agg is None:
                continue

            ret_drop = presult["policy_return"]["relative_drop"]
            grs = method_agg.get("mean_GRS_wj")
            bra = method_agg.get("mean_BRA")

            if bra is not None:
                x_vals.append(ret_drop)
                y_bra.append(1.0 - bra)  # instability = 1 - BRA

            if grs is not None:
                y_grs.append(1.0 - grs)  # instability = 1 - GRS

        method_corr = {}
        if len(x_vals) >= 4 and len(y_bra) >= 4:
            rho, p = scipy_stats.spearmanr(x_vals, y_bra)
            method_corr["return_drop_vs_bra_drop"] = {
                "spearman_rho": float(rho), "p_value": float(p), "n": len(x_vals)
            }
            print(f"  {method:10s} Δreturn vs (1-BRA): ρ={rho:.3f}, p={p:.4f}, n={len(x_vals)}")

        if len(x_vals) >= 4 and len(y_grs) == len(x_vals):
            rho, p = scipy_stats.spearmanr(x_vals, y_grs)
            method_corr["return_drop_vs_grs_drop"] = {
                "spearman_rho": float(rho), "p_value": float(p), "n": len(x_vals)
            }
            print(f"  {method:10s} Δreturn vs (1-GRS): ρ={rho:.3f}, p={p:.4f}, n={len(x_vals)}")

        correlations[method] = method_corr

    # ── Save results ──────────────────────────────────────────────────
    elapsed = time.time() - t0
    tag = env_name.replace("-", "_").lower()
    output = {
        "schema_version": "env_perturbation_v1",
        "env": env_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "clean_baseline": {
            "policy_return": clean_return,
            "cbs_f1": cbs_anchor_f1,
            "dt_f1": dt_anchor_f1,
            "b3_vote_f1": vote_anchor_f1,
        },
        "perturbations": results_per_perturbation,
        "correlations": correlations,
        "protocol": {
            "n_replay_seeds": len(REPLAY_SEEDS),
            "n_transitions_per_seed": 10000,
            "n_heldout": 5000,
            "n_eval_episodes_policy": 20,
            "eval_seeds": EVAL_SEEDS[:20],
        },
    }

    out_dir = os.path.join(RESULTS_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "env_perturbation_results.json")

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
    parser = argparse.ArgumentParser(description="Environment perturbation experiment")
    parser.add_argument("--env", default="all", choices=["MountainCar-v0", "LunarLander-v3", "all"])
    args = parser.parse_args()

    envs = ENVS if args.env == "all" else [args.env]

    for env_name in envs:
        run_env_perturbation(env_name)


if __name__ == "__main__":
    main()
