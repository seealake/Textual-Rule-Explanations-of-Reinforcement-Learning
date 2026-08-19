#!/usr/bin/env python
"""
Boolean Decision Rules (BDR) — stress-test runner

Runs the BDR surrogate through the same perturbation suite as
CBS/MaxF1/Consensus/DT, reusing existing infrastructure.

Usage:
    python experiments/run_boolean_rules.py --env MountainCar-v0
    python experiments/run_boolean_rules.py --env CartPole-v1
    python experiments/run_boolean_rules.py --env all

Output:
    experiments/results/<env>/boolean_rule_results.json
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
from experiments.boolean_rules import (
    BDRSurrogate,
    canonicalize_bdr_rules,
    find_best_bdr_params,
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


def run_bdr_on_data(states, actions, env_name, max_literals=3,
                    max_rules_per_action=8, min_support_frac=0.01,
                    random_state=42, fallback_policy_path=None):
    """Fit BDR surrogate, return pipeline and canonical rules."""
    feature_names = ENV_FEATURE_NAMES.get(env_name)
    bdr = BDRSurrogate(
        n_quantile_thresholds=4,
        max_literals=max_literals,
        max_rules_per_action=max_rules_per_action,
        min_support_frac=min_support_frac,
        random_state=random_state,
        feature_names=feature_names,
        fallback_policy_path=fallback_policy_path,
        env_name=env_name,
    )
    bdr.fit(states, actions)
    rules = canonicalize_bdr_rules(bdr.get_rules())
    return bdr, rules


def evaluate_single_run(bdr, rules, heldout_states, heldout_actions, env_name):
    """Evaluate a single BDR run (same interface as CBS/DT evaluate_single_run)."""
    fid = bdr.evaluate_fidelity(heldout_states, heldout_actions)
    fid_pa = bdr.evaluate_fidelity_per_action(heldout_states, heldout_actions)
    deploy = bdr.evaluate_in_env(
        env_name, eval_seeds=EVAL_SEEDS,
        success_threshold=SUCCESS_THRESHOLDS.get(env_name),
    )
    props = bdr.evaluate_properties()
    cov = bdr.evaluate_coverage(heldout_states)
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
                       for k, vs in bdr.get_thresholds().items()},
    }


def run_boolean_rule_suite(
    env_name, ref_data, heldout_s, heldout_a, model_path,
    best_params=None,
):
    """Run the BDR baseline through all perturbation families."""
    if best_params is None:
        best_params = {
            "max_literals": 3,
            "max_rules_per_action": 8,
            "min_support_frac": 0.01,
        }

    env_tag = env_name.replace("-", "_").lower()
    method_key = "b5_bdr"
    method_params = {
        "type": "boolean_rule_summarizer",
        "n_quantile_thresholds": 4,
        **best_params,
    }

    results = {}
    run_entries = []
    feature_ranges = compute_feature_ranges(ref_data)

    def _run_and_record(run_key, family, params, data):
        bdr, rules = run_bdr_on_data(
            data["states"], data["actions"], env_name,
            max_literals=best_params["max_literals"],
            max_rules_per_action=best_params["max_rules_per_action"],
            min_support_frac=best_params["min_support_frac"],
            fallback_policy_path=model_path,
        )

        res = evaluate_single_run(bdr, rules, heldout_s, heldout_a, env_name)
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

        preds = bdr.predict(heldout_s)
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
    print(f"  [BDR][seed_shift] {len(SEED_SHIFT_SEEDS)} replays...")
    for seed in SEED_SHIFT_SEEDS:
        data = collect_replay(
            env_name=env_name, model_path=model_path,
            num_transitions=10000, seed=seed, deterministic=True,
        )
        _run_and_record(f"seed_shift_s{seed}", "seed_shift",
                        {"replay_seed": seed}, data)

    # --- 2. Subsampling ---
    print(f"  [BDR][subsampling] {N_SUBSAMPLES} uniform subsamples...")
    subsamples = generate_subsamples(ref_data, N_SUBSAMPLES,
                                     SUBSAMPLE_FRACTION, seed=42)
    for i, subset in enumerate(subsamples):
        _run_and_record(f"subsample_{i}", "subsample",
                        {"subsample_idx": i, "fraction": SUBSAMPLE_FRACTION},
                        subset)

    # --- 3. Stratified subsampling ---
    print(f"  [BDR][stratified] {N_SUBSAMPLES} stratified subsamples...")
    strat = generate_stratified_subsamples(ref_data, N_SUBSAMPLES,
                                           SUBSAMPLE_FRACTION, seed=42)
    for i, subset in enumerate(strat):
        _run_and_record(f"stratified_{i}", "stratified",
                        {"subsample_idx": i, "fraction": SUBSAMPLE_FRACTION,
                         "stratified": True}, subset)

    # --- 4. Algorithmic variation (BDR-specific: max_literals, max_rules, min_support) ---
    print(f"  [BDR][algo_variation] BDR hyperparameter variation...")
    algo_variations = [
        # max_literals variation
        {"max_literals": max(1, best_params["max_literals"] - 1),
         "max_rules_per_action": best_params["max_rules_per_action"],
         "min_support_frac": best_params["min_support_frac"],
         "label": "max_literals_minus1"},
        {"max_literals": best_params["max_literals"] + 1,
         "max_rules_per_action": best_params["max_rules_per_action"],
         "min_support_frac": best_params["min_support_frac"],
         "label": "max_literals_plus1"},
        # max_rules variation
        {"max_literals": best_params["max_literals"],
         "max_rules_per_action": max(2, best_params["max_rules_per_action"] - 2),
         "min_support_frac": best_params["min_support_frac"],
         "label": "max_rules_minus2"},
        {"max_literals": best_params["max_literals"],
         "max_rules_per_action": best_params["max_rules_per_action"] + 2,
         "min_support_frac": best_params["min_support_frac"],
         "label": "max_rules_plus2"},
        # min_support variation
        {"max_literals": best_params["max_literals"],
         "max_rules_per_action": best_params["max_rules_per_action"],
         "min_support_frac": max(0.005, best_params["min_support_frac"] * 0.5),
         "label": "min_support_half"},
        {"max_literals": best_params["max_literals"],
         "max_rules_per_action": best_params["max_rules_per_action"],
         "min_support_frac": min(0.10, best_params["min_support_frac"] * 2.0),
         "label": "min_support_double"},
    ]

    for var in algo_variations:
        label = var.pop("label")
        bdr_var, rules_var = run_bdr_on_data(
            ref_data["states"], ref_data["actions"], env_name,
            max_literals=var["max_literals"],
            max_rules_per_action=var["max_rules_per_action"],
            min_support_frac=var["min_support_frac"],
            fallback_policy_path=model_path,
        )
        res = evaluate_single_run(bdr_var, rules_var, heldout_s, heldout_a,
                                  env_name)
        run_key = f"algo_{label}"
        res["run_id"] = f"{env_tag}_{method_key}_{run_key}"
        res["group_id"] = f"{env_tag}_{method_key}"
        var_params = {**method_params, **var}
        res["method_params"] = var_params
        res["perturbation_family"] = "algo_variation"
        res["perturbation_id"] = run_key
        res["perturbation_params"] = {"variation": label, **var}
        res["n_replay"] = len(ref_data["states"])
        res["n_heldout"] = len(heldout_s)
        res["n_eval_episodes"] = len(EVAL_SEEDS)
        res["rules"] = serialize_canonical_rules(rules_var)

        preds = bdr_var.predict(heldout_s)
        thresholds = {int(k): v for k, v in res["thresholds"].items()}
        results[run_key] = res
        run_entries.append({
            "key": run_key,
            "family": "algo_variation",
            "rules": rules_var,
            "thresholds": thresholds,
            "preds": preds,
        })

    # --- 5. Feature noise ---
    print(f"  [BDR][feature_noise] levels={NOISE_LEVELS}...")
    for nl in NOISE_LEVELS:
        noisy = add_feature_noise(ref_data, nl, seed=42,
                                  feature_ranges=feature_ranges)
        _run_and_record(f"noise_{nl:.3f}", "feature_noise",
                        {"noise_level": nl}, noisy)

    # --- Stability computation ---
    all_preds = [e["preds"] for e in run_entries]
    assert all(p.shape == all_preds[0].shape for p in all_preds), \
        "BRA anchor mismatch"

    print(f"  [BDR][metrics] Computing stability...")
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
    print(f"  [BDR][metrics] Computing per-run proxies...")
    proxies = compute_per_run_stability_proxies(run_entries, fr)
    for key, proxy in proxies.items():
        results[key]["stability_proxy_global"] = proxy["global"]
        results[key]["stability_proxy_family"] = proxy["family"]

    return results, stability, method_params


def run_env(env_name):
    print(f"\n{'='*60}")
    print(f"  Boolean Decision Rules (BDR): {env_name}")
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

    # Find best BDR hyperparams via cross-validation
    print(f"  Finding best BDR hyperparams via 3-fold CV...")
    best_params, cv_f1 = find_best_bdr_params(
        ref_data["states"], ref_data["actions"],
        param_grid={
            "max_literals": [2, 3],
            "max_rules_per_action": [4, 8],
            "min_support_frac": [0.01, 0.03],
        },
    )
    print(f"  Best params: {best_params} (CV macro-F1={cv_f1:.3f})")

    t0 = time.time()

    # Run perturbation suite
    b5_results, b5_stability, b5_params = run_boolean_rule_suite(
        env_name, ref_data, heldout_s, heldout_a, model_path,
        best_params=best_params,
    )
    b5_params["best_cv_f1"] = cv_f1

    elapsed = time.time() - t0

    # Save results
    out_dir = f"experiments/results/{env_name.replace('-', '_').lower()}"
    os.makedirs(out_dir, exist_ok=True)

    output = {
        "schema_version": "b5_bdr_v1",
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
        "b5_bdr": {
            "method_params": b5_params,
            "per_run": _serialize(b5_results),
            "stability": b5_stability,
        },
    }

    out_path = os.path.join(out_dir, "boolean_rule_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    # Print summary
    print(f"\n  {'='*50}")
    print(f"  BDR summary ({env_name}):")
    print(f"    Best params: {best_params}")

    f1s = [r["fidelity_heldout"]["f1"] for r in b5_results.values()]
    ecrs = [r["deployment"]["E_CR"] for r in b5_results.values()]
    n_rules = [r["n_rules"] for r in b5_results.values()]
    print(f"    F1: {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
    print(f"    E_CR: {np.mean(ecrs):.1f} +/- {np.std(ecrs):.1f}")
    print(f"    Rules: {np.mean(n_rules):.1f} +/- {np.std(n_rules):.1f}")

    for k, v in b5_stability.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    print(f"  {'='*50}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run Boolean Decision Rules (BDR) stress tests")
    parser.add_argument("--env", type=str, default="all",
                        help="Environment name or 'all'")
    args = parser.parse_args()

    envs = ENVS if args.env == "all" else [args.env]
    all_outputs = {}
    for env_name in envs:
        result = run_env(env_name)
        if result:
            all_outputs[env_name] = result

    print("\nAll BDR experiments complete!")


if __name__ == "__main__":
    main()
