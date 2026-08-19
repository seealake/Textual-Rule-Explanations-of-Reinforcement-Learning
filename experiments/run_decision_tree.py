#!/usr/bin/env python
"""
Decision Tree surrogate (DT) — stress-test runner

Runs the decision-tree surrogate through the same perturbation
suite as CBS/MaxF1/Consensus, reusing existing infrastructure.

Usage:
    python experiments/run_decision_tree.py --env MountainCar-v0
    python experiments/run_decision_tree.py --env CartPole-v1
    python experiments/run_decision_tree.py --env all
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from reproduction.collect_replay import collect_replay, ENV_FEATURE_NAMES
from experiments.decision_tree_surrogate import (
    DecisionTreeSurrogate,
    canonicalize_dt_rules,
    find_best_depth,
)
from experiments.perturbations import (
    load_replay_npz,
    generate_subsamples,
    generate_stratified_subsamples,
    add_feature_noise,
    compute_feature_ranges,
)
from experiments.rule_matching import (
    serialize_canonical_rules,
    mean_pairwise_jaccard,
    mean_pairwise_soft_jaccard,
    mean_pairwise_threshold_drift,
)
from experiments.run_stress_test import (
    compute_bra_from_predictions,
    compute_per_run_stability_proxies,
    _serialize,
    EVAL_SEEDS,
    SUCCESS_THRESHOLDS,
    HELDOUT_SEED,
    SEED_SHIFT_SEEDS,
    N_SUBSAMPLES,
    SUBSAMPLE_FRACTION,
    CLUSTER_DELTAS,
    NOISE_LEVELS,
)


ENVS = ["MountainCar-v0", "CartPole-v1", "LunarLander-v3"]


def get_model_path(env_name):
    tag = env_name.replace("-", "_").lower()
    return f"reproduction/models/dqn_{tag}.zip"


def get_replay_path(env_name, seed=42):
    tag = env_name.replace("-", "_").lower()
    return f"reproduction/data/replay_{tag}_seed{seed}.npz"


def collect_heldout(env_name, model_path, n_transitions=5000):
    data = collect_replay(
        env_name=env_name, model_path=model_path,
        num_transitions=n_transitions, seed=HELDOUT_SEED,
        deterministic=True,
    )
    return data["states"], data["actions"]


def run_dt_on_data(states, actions, env_name, max_depth=None,
                   min_samples_leaf=5, random_state=42):
    """Fit DT surrogate, return pipeline and canonical rules."""
    feature_names = ENV_FEATURE_NAMES.get(env_name)
    dt = DecisionTreeSurrogate(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        feature_names=feature_names,
    )
    dt.fit(states, actions)
    rules = canonicalize_dt_rules(dt.get_rules())
    return dt, rules


def evaluate_single_run(dt, rules, heldout_states, heldout_actions, env_name):
    """Evaluate a single DT run (same interface as CBS evaluate_single_run)."""
    fid = dt.evaluate_fidelity(heldout_states, heldout_actions)
    fid_pa = dt.evaluate_fidelity_per_action(heldout_states, heldout_actions)
    deploy = dt.evaluate_in_env(
        env_name, eval_seeds=EVAL_SEEDS,
        success_threshold=SUCCESS_THRESHOLDS.get(env_name),
    )
    props = dt.evaluate_properties()
    cov = dt.evaluate_coverage(heldout_states)
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
                       for k, vs in dt.get_thresholds().items()},
    }


def run_decision_tree_suite(
    env_name, ref_data, heldout_s, heldout_a, model_path,
    max_depth=None, min_samples_leaf=5,
):
    """Run the DT baseline through all perturbation families."""
    env_tag = env_name.replace("-", "_").lower()
    method_key = "b4_dt"
    method_params = {
        "type": "decision_tree_surrogate",
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
    }

    results = {}
    run_entries = []
    feature_ranges = compute_feature_ranges(ref_data)

    def _run_and_record(run_key, family, params, data):
        dt, rules = run_dt_on_data(
            data["states"], data["actions"], env_name,
            max_depth=max_depth, min_samples_leaf=min_samples_leaf)

        res = evaluate_single_run(dt, rules, heldout_s, heldout_a, env_name)
        res["run_id"] = f"{env_tag}_{method_key}_{run_key}"
        res["group_id"] = f"{env_tag}_{method_key}"
        res["method_params"] = method_params
        res["perturbation_family"] = family
        res["perturbation_id"] = run_key
        res["perturbation_params"] = params
        res["n_replay"] = len(data["states"])
        res["n_heldout"] = len(heldout_s)
        res["n_eval_episodes"] = len(EVAL_SEEDS)
        res["rules"] = serialize_canonical_rules(rules)

        preds = dt.predict(heldout_s)
        thresholds = {int(k): v for k, v in res["thresholds"].items()}

        results[run_key] = res
        run_entries.append({
            "key": run_key,
            "family": family,
            "rules": rules,
            "thresholds": thresholds,
            "preds": preds,
        })

    # --- 1. Seed shift ---
    print(f"  [DT][seed_shift] {len(SEED_SHIFT_SEEDS)} replays...")
    for seed in SEED_SHIFT_SEEDS:
        data = collect_replay(
            env_name=env_name, model_path=model_path,
            num_transitions=10000, seed=seed, deterministic=True,
        )
        _run_and_record(f"seed_shift_s{seed}", "seed_shift",
                        {"replay_seed": seed}, data)

    # --- 2. Subsampling ---
    print(f"  [DT][subsampling] {N_SUBSAMPLES} uniform subsamples...")
    subsamples = generate_subsamples(ref_data, N_SUBSAMPLES,
                                     SUBSAMPLE_FRACTION, seed=42)
    for i, subset in enumerate(subsamples):
        _run_and_record(f"subsample_{i}", "subsample",
                        {"subsample_idx": i, "fraction": SUBSAMPLE_FRACTION}, subset)

    # --- 3. Stratified subsampling ---
    print(f"  [DT][stratified] {N_SUBSAMPLES} stratified subsamples...")
    strat = generate_stratified_subsamples(ref_data, N_SUBSAMPLES,
                                           SUBSAMPLE_FRACTION, seed=42)
    for i, subset in enumerate(strat):
        _run_and_record(f"stratified_{i}", "stratified",
                        {"subsample_idx": i, "fraction": SUBSAMPLE_FRACTION,
                         "stratified": True}, subset)

    # --- 4. Cluster count variation (N/A for DT — use depth ±1 instead) ---
    print(f"  [DT][depth_variation] depth={max_depth} ± 1...")
    depths_to_test = []
    if max_depth is not None:
        for delta in [-1, 0, 1]:
            d = max(1, max_depth + delta)
            depths_to_test.append((delta, d))
    else:
        # Use fixed depths around a reasonable default
        depths_to_test = [(-1, 4), (0, 5), (1, 6)]

    for delta, d in depths_to_test:
        dt_var, rules_var = run_dt_on_data(
            ref_data["states"], ref_data["actions"], env_name,
            max_depth=d, min_samples_leaf=min_samples_leaf)
        res = evaluate_single_run(dt_var, rules_var, heldout_s, heldout_a, env_name)
        run_key = f"depth_delta_{delta:+d}"
        res["run_id"] = f"{env_tag}_{method_key}_{run_key}"
        res["group_id"] = f"{env_tag}_{method_key}"
        res["method_params"] = {**method_params, "actual_depth": d}
        res["perturbation_family"] = "depth_variation"
        res["perturbation_id"] = run_key
        res["perturbation_params"] = {"depth_delta": delta, "actual_depth": d}
        res["n_replay"] = len(ref_data["states"])
        res["n_heldout"] = len(heldout_s)
        res["n_eval_episodes"] = len(EVAL_SEEDS)

        preds = dt_var.predict(heldout_s)
        thresholds = {int(k): v for k, v in res["thresholds"].items()}
        results[run_key] = res
        run_entries.append({
            "key": run_key,
            "family": "depth_variation",
            "rules": rules_var,
            "thresholds": thresholds,
            "preds": preds,
        })

    # --- 5. Feature noise ---
    print(f"  [DT][feature_noise] levels={NOISE_LEVELS}...")
    for nl in NOISE_LEVELS:
        noisy = add_feature_noise(ref_data, nl, seed=42,
                                  feature_ranges=feature_ranges)
        _run_and_record(f"noise_{nl:.3f}", "feature_noise",
                        {"noise_level": nl}, noisy)

    # --- Stability computation ---
    all_preds = [e["preds"] for e in run_entries]
    assert all(p.shape == all_preds[0].shape for p in all_preds), "BRA anchor mismatch"

    print(f"  [DT][metrics] Computing stability...")
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

    # --- Per-run proxies ---
    print(f"  [DT][metrics] Computing per-run proxies...")
    proxies = compute_per_run_stability_proxies(run_entries, fr)
    for key, proxy in proxies.items():
        results[key]["stability_proxy_global"] = proxy["global"]
        results[key]["stability_proxy_family"] = proxy["family"]

    return results, stability, method_params


def run_env(env_name):
    print(f"\n{'='*60}")
    print(f"  Decision Tree baseline: {env_name}")
    print(f"{'='*60}")

    model_path = get_model_path(env_name)
    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found at {model_path}. Skipping.")
        return

    ref_path = get_replay_path(env_name)
    if not os.path.exists(ref_path):
        print(f"  ERROR: Replay not found at {ref_path}. Skipping.")
        return
    ref_data = load_replay_npz(ref_path)
    print(f"  Reference replay: {len(ref_data['states'])} transitions")

    print(f"  Collecting held-out replay (seed={HELDOUT_SEED})...")
    heldout_s, heldout_a = collect_heldout(env_name, model_path)
    print(f"  Held-out replay: {len(heldout_s)} transitions")

    anchor_hash = hashlib.sha256(heldout_s.tobytes()).hexdigest()
    anchor_shape = list(heldout_s.shape)

    # Find best depth via cross-validation
    print(f"  Finding best tree depth via 3-fold CV...")
    best_depth, cv_f1 = find_best_depth(
        ref_data["states"], ref_data["actions"],
        max_depths=(3, 4, 5, 6, 7, 8, 10, None),
        min_samples_leaf=5,
    )
    print(f"  Best depth: {best_depth} (CV macro-F1={cv_f1:.3f})")

    t0 = time.time()

    # Run perturbation suite
    b4_results, b4_stability, b4_params = run_decision_tree_suite(
        env_name, ref_data, heldout_s, heldout_a, model_path,
        max_depth=best_depth,
    )
    b4_params["best_depth_cv_f1"] = cv_f1

    elapsed = time.time() - t0

    # Save results
    out_dir = f"experiments/results/{env_name.replace('-', '_').lower()}"
    os.makedirs(out_dir, exist_ok=True)

    output = {
        "schema_version": "b4_dt_v1",
        "env": env_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "protocol": {
            "extraction_transitions": 10000,
            "heldout_transitions": len(heldout_s),
            "deployment_episodes": len(EVAL_SEEDS),
            "eval_seeds": EVAL_SEEDS[:5],
            "anchor": {
                "seed": HELDOUT_SEED,
                "size": len(heldout_s),
                "shape": anchor_shape,
                "sha256": anchor_hash,
            },
        },
        "b4_dt": {
            "method_params": b4_params,
            "per_run": _serialize(b4_results),
            "stability": b4_stability,
        },
    }

    out_path = os.path.join(out_dir, "decision_tree_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    # Print summary
    print(f"\n  {'='*50}")
    print(f"  Decision Tree summary ({env_name}):")
    print(f"    Best depth: {best_depth}")

    f1s = [r["fidelity_heldout"]["f1"] for r in b4_results.values()]
    ecrs = [r["deployment"]["E_CR"] for r in b4_results.values()]
    n_rules = [r["n_rules"] for r in b4_results.values()]
    print(f"    F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"    E_CR: {np.mean(ecrs):.1f} ± {np.std(ecrs):.1f}")
    print(f"    Rules: {np.mean(n_rules):.1f} ± {np.std(n_rules):.1f}")

    for k, v in b4_stability.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    print(f"  {'='*50}")
    print(f"  Elapsed: {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Run Decision Tree baseline stress tests")
    parser.add_argument("--env", type=str, default="all",
                        help="Environment name or 'all'")
    args = parser.parse_args()

    envs = ENVS if args.env == "all" else [args.env]
    for env_name in envs:
        run_env(env_name)

    print("\nAll decision-tree experiments complete!")


if __name__ == "__main__":
    main()
