#!/usr/bin/env python
"""
Perturbation Severity Sweep — Dense Noise Curves

Two sub-experiments:
  A. Dense LEC curve (local, no refit): CBS, rule-set voting, DT at 7 epsilon levels
  B. Replay noise curve (global, refit): 7 noise levels × 5 seeds × 3 methods

Also computes:
  - critical_ε: smallest ε where LEC < 0.8 or BRA < 0.9 × clean
  - AUC-degradation: area under the degradation curve
  - Mean relative drop: average (clean - perturbed) / clean across all ε > 0

Usage:
    python experiments/run_noise_severity_sweep.py --env MountainCar-v0
    python experiments/run_noise_severity_sweep.py --env all

Output:
    experiments/results/<env>/noise_severity_results.json
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
from experiments.lec import (
    compute_lec,
    compute_lec_prediction_based,
    normalize_states,
)
from experiments.perturbations import add_feature_noise, compute_feature_ranges
from experiments.rule_matching import (
    canonicalize_rules,
    ruleset_weighted_jaccard,
    ruleset_soft_jaccard,
    threshold_drift,
)
from experiments.decision_tree_surrogate import DecisionTreeSurrogate, find_best_depth
from experiments.consensus_merge import build_voting_ensemble
from experiments.run_env_perturbation import fit_cbs, fit_dt, fit_b3_vote, compute_pairwise_metrics

# ── Constants ─────────────────────────────────────────────────────────
ENVS = ["MountainCar-v0", "CartPole-v1", "LunarLander-v3"]
DENSE_EPSILONS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
NOISE_LEVELS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
N_SEEDS = 5
HELDOUT_SEED = 99
RESULTS_DIR = "experiments/results"


def get_model_path(env_name):
    tag = env_name.replace("-", "_").lower()
    return f"reproduction/models/dqn_{tag}.zip"


def find_critical_epsilon(epsilons, values, threshold, direction="below"):
    """Find smallest epsilon where value crosses threshold.

    direction='below': find ε where value < threshold (for LEC)
    direction='above': find ε where value > threshold (for TD)
    """
    for eps, val in zip(epsilons, values):
        if eps == 0:
            continue
        if direction == "below" and val < threshold:
            return float(eps)
        elif direction == "above" and val > threshold:
            return float(eps)
    return None  # threshold never crossed


def compute_auc_degradation(epsilons, values, clean_value):
    """Compute area under degradation curve, normalized by epsilon range.

    Degradation = (clean - value) / clean for each epsilon.
    AUC computed via trapezoidal rule, normalized by total epsilon range.
    """
    if clean_value == 0:
        return 0.0

    degradation = [(clean_value - v) / abs(clean_value) for v in values]
    # Trapezoidal AUC
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    auc = float(trapz_fn(degradation, epsilons))
    # Normalize by epsilon range
    eps_range = max(epsilons) - min(epsilons)
    return auc / eps_range if eps_range > 0 else 0.0


def mean_relative_drop(values, clean_value):
    """Mean relative drop across all non-zero-epsilon values."""
    if clean_value == 0:
        return 0.0
    drops = [(clean_value - v) / abs(clean_value) for v in values if v != clean_value]
    return float(np.mean(drops)) if drops else 0.0


def run_lec_curves(env_name):
    """Sub-experiment A: Dense LEC curves for CBS, rule-set voting, DT."""
    print(f"\n  Sub-experiment A: Dense LEC curves")
    model_path = get_model_path(env_name)

    # Collect data
    clean = collect_replay(env_name, model_path, num_transitions=10000, seed=42)
    heldout = collect_replay(env_name, model_path, num_transitions=5000, seed=HELDOUT_SEED)
    clean_s, clean_a = clean["states"], clean["actions"]
    heldout_s = heldout["states"]

    feature_mins = np.min(clean_s, axis=0)
    feature_maxs = np.max(clean_s, axis=0)

    results = {}

    # CBS
    print("    CBS LEC...")
    cbs, _, _ = fit_cbs(clean_s, clean_a, env_name)
    lec_cbs = compute_lec(cbs, heldout_s, feature_mins, feature_maxs,
                          epsilons=DENSE_EPSILONS, n_perturbations=50, seed=42)
    results["cbs"] = {str(eps): lec_cbs[eps] for eps in DENSE_EPSILONS}

    # CBS prediction-based (for comparison)
    lec_cbs_pred = compute_lec_prediction_based(
        cbs, heldout_s, feature_mins, feature_maxs,
        epsilons=DENSE_EPSILONS, n_perturbations=50, seed=42)
    results["cbs_pred"] = {str(eps): lec_cbs_pred[eps] for eps in DENSE_EPSILONS}

    # DT surrogate
    print("    DT LEC (prediction-based)...")
    dt, _, _ = fit_dt(clean_s, clean_a, env_name)
    lec_dt = compute_lec_prediction_based(
        dt, heldout_s, feature_mins, feature_maxs,
        epsilons=DENSE_EPSILONS, n_perturbations=50, seed=42)
    results["dt"] = {str(eps): lec_dt[eps] for eps in DENSE_EPSILONS}

    # rule-set voting (prediction-based via voting)
    print("    rule-set voting LEC (prediction-based)...")

    class VotingPredictor:
        def __init__(self, pipelines):
            self.pipelines = pipelines
        def predict(self, states):
            from experiments.consensus_merge import voting_predict
            return voting_predict(self.pipelines, states)

    vote_pipelines, _, _ = fit_b3_vote(clean_s, clean_a, env_name)
    vote_predictor = VotingPredictor(vote_pipelines)
    lec_vote = compute_lec_prediction_based(
        vote_predictor, heldout_s, feature_mins, feature_maxs,
        epsilons=DENSE_EPSILONS, n_perturbations=50, seed=42)
    results["b3_vote"] = {str(eps): lec_vote[eps] for eps in DENSE_EPSILONS}

    # Compute summary statistics
    for method in ["cbs", "dt", "b3_vote"]:
        key = "cbs" if method == "cbs" else method
        lec_vals = [results[key][str(eps)]["lec"] for eps in DENSE_EPSILONS]
        clean_lec = lec_vals[0] if lec_vals[0] > 0 else 1.0

        results[f"{method}_summary"] = {
            "critical_eps_lec_08": find_critical_epsilon(DENSE_EPSILONS, lec_vals, 0.8, "below"),
            "auc_degradation": compute_auc_degradation(DENSE_EPSILONS, lec_vals, clean_lec),
            "mean_relative_drop": mean_relative_drop(lec_vals[1:], clean_lec),
        }
        crit = results[f"{method}_summary"]["critical_eps_lec_08"]
        print(f"    {method:10s}: LEC curve = {[f'{v:.3f}' for v in lec_vals]}, critical_ε(0.8)={crit}")

    return results


def run_noise_curves(env_name):
    """Sub-experiment B: Replay noise curves — refit pipeline at each noise level."""
    print(f"\n  Sub-experiment B: Replay noise curves")
    model_path = get_model_path(env_name)

    # Collect data
    clean = collect_replay(env_name, model_path, num_transitions=10000, seed=42)
    heldout = collect_replay(env_name, model_path, num_transitions=5000, seed=HELDOUT_SEED)
    clean_s, clean_a = clean["states"], clean["actions"]
    heldout_s, heldout_a = heldout["states"], heldout["actions"]

    feature_ranges_arr = compute_feature_ranges({"states": clean_s})
    feature_ranges = {i: float(feature_ranges_arr[i]) for i in range(len(feature_ranges_arr))}

    # Fit clean anchors
    print("    Fitting clean anchors...")
    cbs_anchor, cbs_anchor_rules, cbs_anchor_thresh = fit_cbs(clean_s, clean_a, env_name)
    cbs_anchor_preds = cbs_anchor.predict(heldout_s)
    cbs_anchor_f1 = cbs_anchor.evaluate_fidelity(heldout_s, heldout_a)["f1"]

    dt_anchor, dt_anchor_rules, dt_anchor_thresh = fit_dt(clean_s, clean_a, env_name)
    dt_anchor_preds = dt_anchor.predict(heldout_s)
    dt_anchor_f1 = dt_anchor.evaluate_fidelity(heldout_s, heldout_a)["f1"]

    from experiments.consensus_merge import voting_predict
    vote_pipelines, vote_anchor_rules, _ = fit_b3_vote(clean_s, clean_a, env_name)
    vote_anchor_preds = voting_predict(vote_pipelines, heldout_s)
    vote_anchor_f1 = float(np.mean(vote_anchor_preds == heldout_a))

    results = {method: {} for method in ["cbs", "dt", "b3_vote"]}
    clean_data = {"states": clean_s, "actions": clean_a,
                  "rewards": np.zeros(len(clean_a)), "dones": np.zeros(len(clean_a), dtype=bool),
                  "episode_ids": np.zeros(len(clean_a), dtype=int)}

    for noise_level in NOISE_LEVELS:
        noise_key = f"{noise_level:.3f}"
        print(f"    Noise level {noise_level}...")

        if noise_level == 0.0:
            # Clean baseline
            for method in ["cbs", "dt", "b3_vote"]:
                anchor_f1 = {"cbs": cbs_anchor_f1, "dt": dt_anchor_f1, "b3_vote": vote_anchor_f1}[method]
                results[method][noise_key] = {
                    "mean_f1": anchor_f1, "std_f1": 0.0,
                    "mean_BRA": 1.0, "std_BRA": 0.0,
                    "mean_GRS_wj": 1.0, "std_GRS_wj": 0.0,
                }
            continue

        seed_metrics = {m: [] for m in ["cbs", "dt", "b3_vote"]}

        for seed in range(N_SEEDS):
            noisy = add_feature_noise(clean_data, noise_level=noise_level, seed=seed)
            ns, na = noisy["states"], noisy["actions"]

            # CBS
            try:
                cbs_p, cbs_p_rules, cbs_p_thresh = fit_cbs(ns, na, env_name)
                cbs_p_preds = cbs_p.predict(heldout_s)
                cbs_p_f1 = cbs_p.evaluate_fidelity(heldout_s, heldout_a)["f1"]
                metrics = compute_pairwise_metrics(
                    cbs_anchor_rules, cbs_p_rules, cbs_anchor_thresh, cbs_p_thresh,
                    feature_ranges, cbs_anchor_preds, cbs_p_preds)
                metrics["f1"] = cbs_p_f1
                seed_metrics["cbs"].append(metrics)
            except Exception:
                pass

            # DT
            try:
                dt_p, dt_p_rules, dt_p_thresh = fit_dt(ns, na, env_name)
                dt_p_preds = dt_p.predict(heldout_s)
                dt_p_f1 = dt_p.evaluate_fidelity(heldout_s, heldout_a)["f1"]
                metrics = compute_pairwise_metrics(
                    dt_anchor_rules, dt_p_rules, dt_anchor_thresh, dt_p_thresh,
                    feature_ranges, dt_anchor_preds, dt_p_preds)
                metrics["f1"] = dt_p_f1
                seed_metrics["dt"].append(metrics)
            except Exception:
                pass

            # rule-set voting
            try:
                vote_ps, _, _ = fit_b3_vote(ns, na, env_name)
                vote_p_preds = voting_predict(vote_ps, heldout_s)
                vote_f1 = float(np.mean(vote_p_preds == heldout_a))
                vote_bra = float(np.mean(vote_anchor_preds == vote_p_preds))
                seed_metrics["b3_vote"].append({"f1": vote_f1, "BRA": vote_bra, "GRS_wj": None})
            except Exception:
                pass

        # Aggregate
        for method in ["cbs", "dt", "b3_vote"]:
            sm = seed_metrics[method]
            if sm:
                f1s = [m["f1"] for m in sm]
                bras = [m["BRA"] for m in sm if m.get("BRA") is not None]
                grss = [m["GRS_wj"] for m in sm if m.get("GRS_wj") is not None]
                results[method][noise_key] = {
                    "mean_f1": float(np.mean(f1s)), "std_f1": float(np.std(f1s)),
                    "mean_BRA": float(np.mean(bras)) if bras else None,
                    "std_BRA": float(np.std(bras)) if bras else None,
                    "mean_GRS_wj": float(np.mean(grss)) if grss else None,
                    "std_GRS_wj": float(np.std(grss)) if grss else None,
                }

    # Compute summary statistics
    summaries = {}
    for method in ["cbs", "dt", "b3_vote"]:
        f1_curve = [results[method].get(f"{nl:.3f}", {}).get("mean_f1", None) for nl in NOISE_LEVELS]
        bra_curve = [results[method].get(f"{nl:.3f}", {}).get("mean_BRA", None) for nl in NOISE_LEVELS]

        f1_vals = [v for v in f1_curve if v is not None]
        bra_vals = [v for v in bra_curve if v is not None]

        clean_f1 = f1_vals[0] if f1_vals else 1.0
        clean_bra = bra_vals[0] if bra_vals else 1.0

        summaries[method] = {
            "critical_eps_bra_90pct": find_critical_epsilon(
                NOISE_LEVELS, bra_vals, 0.9 * clean_bra, "below") if len(bra_vals) == len(NOISE_LEVELS) else None,
            "auc_f1_degradation": compute_auc_degradation(NOISE_LEVELS, f1_vals, clean_f1) if len(f1_vals) == len(NOISE_LEVELS) else None,
            "auc_bra_degradation": compute_auc_degradation(NOISE_LEVELS, bra_vals, clean_bra) if len(bra_vals) == len(NOISE_LEVELS) else None,
            "mean_relative_f1_drop": mean_relative_drop(f1_vals[1:], clean_f1),
            "mean_relative_bra_drop": mean_relative_drop(bra_vals[1:], clean_bra) if bra_vals else None,
        }
        f1_str = [f"{v:.3f}" for v in f1_vals]
        bra_str = [f"{v:.3f}" for v in bra_vals]
        print(f"    {method:10s}: F1={f1_str}, BRA={bra_str}")
        print(f"    {'':10s}  AUC-F1-deg={summaries[method]['auc_f1_degradation']:.4f}, "
              f"critical_ε(BRA)={summaries[method]['critical_eps_bra_90pct']}")

    return results, summaries


def run_noise_severity_sweep(env_name):
    """Run full noise severity sweep for one environment."""
    print(f"\n{'='*60}")
    print(f"Noise Severity Sweep: {env_name}")
    print(f"{'='*60}")
    t0 = time.time()

    # Sub-experiment A: LEC curves
    lec_results = run_lec_curves(env_name)

    # Sub-experiment B: Noise curves
    noise_results, noise_summaries = run_noise_curves(env_name)

    elapsed = time.time() - t0
    tag = env_name.replace("-", "_").lower()

    output = {
        "schema_version": "noise_severity_v1",
        "env": env_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "epsilons": DENSE_EPSILONS,
        "noise_levels": NOISE_LEVELS,
        "n_seeds": N_SEEDS,
        "lec_curves": lec_results,
        "noise_curves": noise_results,
        "noise_summaries": noise_summaries,
    }

    out_dir = os.path.join(RESULTS_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "noise_severity_results.json")

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
    parser = argparse.ArgumentParser(description="Noise severity sweep")
    parser.add_argument("--env", default="all",
                        choices=["MountainCar-v0", "CartPole-v1", "LunarLander-v3", "all"])
    args = parser.parse_args()

    envs = ENVS if args.env == "all" else [args.env]
    for env_name in envs:
        run_noise_severity_sweep(env_name)


if __name__ == "__main__":
    main()
