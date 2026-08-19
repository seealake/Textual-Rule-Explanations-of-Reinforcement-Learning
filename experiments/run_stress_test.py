#!/usr/bin/env python
"""
Unified stress-test runner

Runs CBS and CBS+MaxF1 across all perturbation families on both pilot
environments (MountainCar-v0, CartPole-v1), evaluates on held-out replay
and fixed deployment seeds, and computes real stability metrics
(GRS, GRS-TA, BRA, TD).

Usage:
    python experiments/run_stress_test.py --env MountainCar-v0
    python experiments/run_stress_test.py --env CartPole-v1
    python experiments/run_stress_test.py --env all

Output:
    experiments/results/<env>/stress_test_results.json
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from reproduction.cbs import CBSPipeline
from reproduction.collect_replay import collect_replay, ENV_FEATURE_NAMES
from experiments.perturbations import (
    load_replay_npz,
    generate_subsamples,
    generate_stratified_subsamples,
    add_feature_noise,
    compute_feature_ranges,
)
from experiments.rule_matching import (
    canonicalize_rules,
    serialize_canonical_rules,
    mean_pairwise_jaccard,
    mean_pairwise_soft_jaccard,
    mean_pairwise_threshold_drift,
    mean_pairwise_bra,
    ruleset_weighted_jaccard,
    ruleset_soft_jaccard,
    threshold_drift,
)

# ── Constants ─────────────────────────────────────────────────────────
ENVS = ["MountainCar-v0", "CartPole-v1", "LunarLander-v3"]
EVAL_SEEDS = list(range(1000, 1050))  # 50 fixed seeds for deployment
SUCCESS_THRESHOLDS = {"MountainCar-v0": -150.0, "CartPole-v1": 475.0, "LunarLander-v3": 200.0}
SEED_SHIFT_SEEDS = [0, 1, 2, 3, 4]
N_SUBSAMPLES = 5
SUBSAMPLE_FRACTION = 0.8
CLUSTER_DELTAS = [-1, 0, 1]
NOISE_LEVELS = [0.01, 0.03, 0.05]
HELDOUT_SEED = 99  # seed for collecting held-out replay


def get_model_path(env_name):
    tag = env_name.replace("-", "_").lower()
    return f"reproduction/models/dqn_{tag}.zip"


def get_replay_path(env_name, seed=42):
    tag = env_name.replace("-", "_").lower()
    return f"reproduction/data/replay_{tag}_seed{seed}.npz"


def run_cbs_on_data(states, actions, env_name, kmeans_seed=0, delta=0,
                    use_maxf1=False):
    """Fit CBS (optionally with MaxF1 refinement), return pipeline and canonical rules."""
    feature_names = ENV_FEATURE_NAMES.get(env_name)
    cbs = CBSPipeline(
        n_categories=5,
        inclusion_threshold=0.70,
        kmeans_seed=kmeans_seed,
        cluster_count_delta=delta,
        feature_names=feature_names,
    )
    cbs.fit(states, actions)
    if use_maxf1:
        cbs.refine_max_f1(states, actions)
    rules = canonicalize_rules(cbs.get_rules())
    return cbs, rules


def evaluate_single_run(cbs, rules, heldout_states, heldout_actions,
                        env_name):
    """Evaluate a single CBS run: held-out fidelity + deployment."""
    fid = cbs.evaluate_fidelity(heldout_states, heldout_actions)
    fid_pa = cbs.evaluate_fidelity_per_action(heldout_states, heldout_actions)
    deploy = cbs.evaluate_in_env(
        env_name, eval_seeds=EVAL_SEEDS,
        success_threshold=SUCCESS_THRESHOLDS.get(env_name),
    )
    props = cbs.evaluate_properties()
    cov = cbs.evaluate_coverage(heldout_states)
    return {
        "fidelity_heldout": fid,
        "fidelity_per_action": fid_pa,
        "deployment": {
            "E_CR": deploy["E_CR"],
            "E_CR_std": deploy["E_CR_std"],
            "E_TS": deploy["E_TS"],
            "success_rate": deploy["success_rate"],
        },
        "properties": props,
        "coverage": cov,
        "n_rules": len(rules),
        "thresholds": {int(k): [float(v) for v in vs]
                       for k, vs in cbs.get_thresholds().items()},
    }


def collect_heldout(env_name, model_path, n_transitions=5000):
    """Collect independent held-out replay for fidelity evaluation."""
    data = collect_replay(
        env_name=env_name, model_path=model_path,
        num_transitions=n_transitions, seed=HELDOUT_SEED,
        deterministic=True,
    )
    return data["states"], data["actions"]


def compute_bra_from_predictions(pred_list):
    """Compute BRA as mean pairwise agreement on prediction vectors.

    This avoids encoding-mismatch issues across runs with different thresholds
    by using each CBS pipeline's predict() directly on raw states.
    """
    n = len(pred_list)
    if n < 2:
        return 1.0
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            agree = np.mean(pred_list[i] == pred_list[j])
            total += agree
            count += 1
    return total / count


def compute_per_run_stability_proxies(run_entries, feature_ranges):
    """Compute leave-one-out pairwise stability for each run.

    For each run i, computes its average pairwise similarity with all other
    runs (global proxy) and with same-family runs only (family proxy).

    Parameters
    ----------
    run_entries : list[dict]
        Each entry has keys: "key", "family", "rules", "thresholds", "preds".
    feature_ranges : dict[int, float]
        Per-feature range for TD normalisation.

    Returns
    -------
    dict[str, dict] — run_key -> {"global": {...}, "family": {...} or None}
    """
    n = len(run_entries)
    proxies = {}

    for i in range(n):
        key_i = run_entries[i]["key"]
        family_i = run_entries[i]["family"]

        g_grs_wj, g_grs_ta, g_bra, g_td = [], [], [], []
        f_grs_wj, f_grs_ta, f_bra, f_td = [], [], [], []

        for j in range(n):
            if i == j:
                continue

            grs_wj = ruleset_weighted_jaccard(
                run_entries[i]["rules"], run_entries[j]["rules"])
            grs_ta = ruleset_soft_jaccard(
                run_entries[i]["rules"], run_entries[j]["rules"],
                threshold_aware=True)
            bra = float(np.mean(
                run_entries[i]["preds"] == run_entries[j]["preds"]))
            td_val = threshold_drift(
                run_entries[i]["thresholds"], run_entries[j]["thresholds"],
                feature_ranges)

            g_grs_wj.append(grs_wj)
            g_grs_ta.append(grs_ta)
            g_bra.append(bra)
            g_td.append(td_val)

            if run_entries[j]["family"] == family_i:
                f_grs_wj.append(grs_wj)
                f_grs_ta.append(grs_ta)
                f_bra.append(bra)
                f_td.append(td_val)

        proxies[key_i] = {
            "global": {
                "GRS_wj": float(np.mean(g_grs_wj)),
                "GRS_ta": float(np.mean(g_grs_ta)),
                "BRA": float(np.mean(g_bra)),
                "TD": float(np.mean(g_td)),
            },
            "family": {
                "GRS_wj": float(np.mean(f_grs_wj)),
                "GRS_ta": float(np.mean(f_grs_ta)),
                "BRA": float(np.mean(f_bra)),
                "TD": float(np.mean(f_td)),
            } if f_grs_wj else None,
        }

    return proxies


def run_perturbation_suite(env_name, ref_data, heldout_s, heldout_a,
                           model_path, use_maxf1=False):
    """Run all perturbation families for one environment, one method.

    Returns (results, stability, method_params) where results contains
    per-run stability proxies at both global and family level.
    """
    method_tag = "CBS+MaxF1" if use_maxf1 else "CBS"
    method_key = "cbs_maxf1" if use_maxf1 else "cbs"
    env_tag = env_name.replace("-", "_").lower()
    method_params = {
        "n_categories": 5,
        "inclusion_threshold": 0.70,
        "maxf1": use_maxf1,
    }

    results = {}
    run_entries = []  # parallel tracking for proxy computation

    feature_ranges = compute_feature_ranges(ref_data)

    def _record_run(run_key, family, params, cbs, rules, res, n_replay):
        """Helper: store result and collect data for proxy computation."""
        preds = cbs.predict(heldout_s)
        thresholds = {int(k): v for k, v in res["thresholds"].items()}

        # Enrich per-run result with metadata
        res["run_id"] = f"{env_tag}_{method_key}_{run_key}"
        res["group_id"] = f"{env_tag}_{method_key}"
        res["method_params"] = method_params
        res["perturbation_family"] = family
        res["perturbation_id"] = run_key
        res["perturbation_params"] = params
        res["n_replay"] = n_replay
        res["n_heldout"] = len(heldout_s)
        res["n_eval_episodes"] = len(EVAL_SEEDS)

        res["rules"] = serialize_canonical_rules(rules)
        results[run_key] = res
        run_entries.append({
            "key": run_key,
            "family": family,
            "rules": rules,
            "thresholds": thresholds,
            "preds": preds,
        })

    # --- 1. Seed shift ---
    print(f"  [{method_tag}][seed_shift] Collecting {len(SEED_SHIFT_SEEDS)} replays...")
    for seed in SEED_SHIFT_SEEDS:
        data = collect_replay(
            env_name=env_name, model_path=model_path,
            num_transitions=10000, seed=seed, deterministic=True,
        )
        cbs, rules = run_cbs_on_data(data["states"], data["actions"],
                                      env_name, use_maxf1=use_maxf1)
        res = evaluate_single_run(cbs, rules, heldout_s, heldout_a, env_name)
        _record_run(f"seed_shift_s{seed}", "seed_shift",
                    {"replay_seed": seed}, cbs, rules, res,
                    len(data["states"]))

    # --- 2. Subsampling ---
    print(f"  [{method_tag}][subsampling] {N_SUBSAMPLES} uniform subsamples...")
    subsamples = generate_subsamples(ref_data, N_SUBSAMPLES,
                                      SUBSAMPLE_FRACTION, seed=42)
    for i, subset in enumerate(subsamples):
        cbs, rules = run_cbs_on_data(subset["states"], subset["actions"],
                                      env_name, use_maxf1=use_maxf1)
        res = evaluate_single_run(cbs, rules, heldout_s, heldout_a, env_name)
        _record_run(f"subsample_{i}", "subsample",
                    {"subsample_idx": i, "fraction": SUBSAMPLE_FRACTION},
                    cbs, rules, res, len(subset["states"]))

    # --- 3. Stratified subsampling ---
    print(f"  [{method_tag}][stratified] {N_SUBSAMPLES} stratified subsamples...")
    strat = generate_stratified_subsamples(ref_data, N_SUBSAMPLES,
                                            SUBSAMPLE_FRACTION, seed=42)
    for i, subset in enumerate(strat):
        cbs, rules = run_cbs_on_data(subset["states"], subset["actions"],
                                      env_name, use_maxf1=use_maxf1)
        res = evaluate_single_run(cbs, rules, heldout_s, heldout_a, env_name)
        _record_run(f"stratified_{i}", "stratified",
                    {"subsample_idx": i, "fraction": SUBSAMPLE_FRACTION,
                     "stratified": True},
                    cbs, rules, res, len(subset["states"]))

    # --- 4. Cluster count variation ---
    print(f"  [{method_tag}][cluster_count] deltas={CLUSTER_DELTAS}...")
    for delta in CLUSTER_DELTAS:
        cbs, rules = run_cbs_on_data(
            ref_data["states"], ref_data["actions"], env_name, delta=delta,
            use_maxf1=use_maxf1)
        res = evaluate_single_run(cbs, rules, heldout_s, heldout_a, env_name)
        _record_run(f"cluster_delta_{delta:+d}", "cluster_count",
                    {"cluster_delta": delta},
                    cbs, rules, res, len(ref_data["states"]))

    # --- 5. Feature noise ---
    print(f"  [{method_tag}][feature_noise] levels={NOISE_LEVELS}...")
    for nl in NOISE_LEVELS:
        noisy = add_feature_noise(ref_data, nl, seed=42,
                                   feature_ranges=feature_ranges)
        cbs, rules = run_cbs_on_data(noisy["states"], noisy["actions"],
                                      env_name, use_maxf1=use_maxf1)
        res = evaluate_single_run(cbs, rules, heldout_s, heldout_a, env_name)
        _record_run(f"noise_{nl:.3f}", "feature_noise",
                    {"noise_level": nl},
                    cbs, rules, res, len(noisy["states"]))

    # --- Anchor set validation for BRA ---
    all_preds = [e["preds"] for e in run_entries]
    expected_shape = all_preds[0].shape
    assert all(p.shape == expected_shape for p in all_preds), \
        f"BRA anchor mismatch: expected shape {expected_shape}, " \
        f"got {[p.shape for p in all_preds if p.shape != expected_shape]}"

    # --- Compute global stability metrics (backward compat) ---
    print(f"  [{method_tag}][metrics] Computing GRS, GRS-TA, TD, BRA...")
    all_rule_sets = [e["rules"] for e in run_entries]
    all_threshold_sets = [e["thresholds"] for e in run_entries]

    grs_wj = mean_pairwise_jaccard(all_rule_sets, weighted=True)
    grs_plain = mean_pairwise_jaccard(all_rule_sets, weighted=False)
    grs_ta = mean_pairwise_soft_jaccard(all_rule_sets, threshold_aware=True)

    td_sets = [{int(k): v for k, v in ts.items()} for ts in all_threshold_sets]
    fr = {i: float(feature_ranges[i]) for i in range(len(feature_ranges))}
    td = mean_pairwise_threshold_drift(td_sets, feature_ranges=fr)

    bra = compute_bra_from_predictions(all_preds)

    stability = {
        "GRS_weighted_jaccard": float(grs_wj),
        "GRS_plain_jaccard": float(grs_plain),
        "GRS_threshold_aware": float(grs_ta),
        "TD": float(td),
        "BRA": float(bra),
        "n_perturbation_runs": len(all_rule_sets),
    }

    # --- Compute per-run stability proxies (global + family) ---
    print(f"  [{method_tag}][metrics] Computing per-run stability proxies...")
    proxies = compute_per_run_stability_proxies(run_entries, fr)
    for key, proxy in proxies.items():
        results[key]["stability_proxy_global"] = proxy["global"]
        results[key]["stability_proxy_family"] = proxy["family"]

    return results, stability, method_params


def run_env(env_name):
    """Run full stress test for one environment."""
    print(f"\n{'='*60}")
    print(f"  Stress Test: {env_name}")
    print(f"{'='*60}")

    model_path = get_model_path(env_name)
    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found at {model_path}. Skipping.")
        return

    # Load reference replay
    ref_path = get_replay_path(env_name)
    if not os.path.exists(ref_path):
        print(f"  ERROR: Reference replay not found at {ref_path}. Skipping.")
        return
    ref_data = load_replay_npz(ref_path)
    print(f"  Reference replay: {len(ref_data['states'])} transitions")

    # Collect held-out replay
    print(f"  Collecting held-out replay (seed={HELDOUT_SEED})...")
    heldout_s, heldout_a = collect_heldout(env_name, model_path)
    print(f"  Held-out replay: {len(heldout_s)} transitions")

    t0 = time.time()

    # Anchor set audit info
    anchor_hash = hashlib.sha256(heldout_s.tobytes()).hexdigest()
    anchor_shape = list(heldout_s.shape)

    # --- CBS ---
    print(f"\n  --- Method: CBS ---")
    cbs_results, cbs_stability, cbs_params = run_perturbation_suite(
        env_name, ref_data, heldout_s, heldout_a, model_path,
        use_maxf1=False)

    # --- CBS + MaxF1 ---
    print(f"\n  --- Method: CBS + MaxF1 ---")
    maxf1_results, maxf1_stability, maxf1_params = run_perturbation_suite(
        env_name, ref_data, heldout_s, heldout_a, model_path,
        use_maxf1=True)

    elapsed = time.time() - t0

    # --- Save results ---
    out_dir = f"experiments/results/{env_name.replace('-', '_').lower()}"
    os.makedirs(out_dir, exist_ok=True)

    output = {
        "schema_version": "stress_test_v2",
        "env": env_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "protocol": {
            "extraction_transitions": 10000,
            "heldout_transitions": len(heldout_s),
            "deployment_episodes": len(EVAL_SEEDS),
            "eval_seeds": EVAL_SEEDS[:5],  # save first 5 for reference
            "anchor": {
                "seed": HELDOUT_SEED,
                "size": len(heldout_s),
                "shape": anchor_shape,
                "sha256": anchor_hash,
            },
        },
        "cbs": {
            "method_params": cbs_params,
            "per_run": _serialize(cbs_results),
            "stability": cbs_stability,
        },
        "cbs_maxf1": {
            "method_params": maxf1_params,
            "per_run": _serialize(maxf1_results),
            "stability": maxf1_stability,
        },
    }

    out_path = os.path.join(out_dir, "stress_test_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print(f"  Elapsed: {elapsed:.1f}s")

    # --- Print summary ---
    print(f"\n  {'='*50}")
    print(f"  CBS Stability Summary:")
    for k, v in cbs_stability.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")
    print(f"\n  CBS+MaxF1 Stability Summary:")
    for k, v in maxf1_stability.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")
    print(f"  {'='*50}")


def _serialize(d):
    """Make dict JSON-serializable."""
    if isinstance(d, dict):
        return {str(k): _serialize(v) for k, v in d.items()}
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, (np.floating,)):
        return float(d)
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, list):
        return [_serialize(x) for x in d]
    return d


def main():
    parser = argparse.ArgumentParser(description="Run stability stress tests")
    parser.add_argument("--env", type=str, default="all",
                        help="Environment name or 'all'")
    args = parser.parse_args()

    envs = ENVS if args.env == "all" else [args.env]
    for env_name in envs:
        run_env(env_name)

    print("\nAll stress tests complete!")


if __name__ == "__main__":
    main()
