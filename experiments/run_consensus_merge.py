#!/usr/bin/env python
"""
Consensus CBS Runner

Runs the consensus merge, the naive voting baseline (rule-set voting), and
policy-aware reweighting variant (importance-weighted voting) across the same perturbation
suite used for the clustering baselines, then runs ablation sweeps.

Usage:
    python experiments/run_consensus_merge.py --env MountainCar-v0
    python experiments/run_consensus_merge.py --env CartPole-v1
    python experiments/run_consensus_merge.py --env all
    python experiments/run_consensus_merge.py --env all --skip-ablations
"""
import argparse
import hashlib
import json
import os
import sys
import time

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
    ruleset_weighted_jaccard,
    ruleset_soft_jaccard,
    threshold_drift,
)
from experiments.consensus_merge import (
    build_consensus_ruleset,
    build_voting_ensemble,
    voting_predict,
    compute_policy_aware_weights,
)
from experiments.run_stress_test import (
    evaluate_single_run,
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

# ── Constants ─────────────────────────────────────────────────────────
ENVS = ["MountainCar-v0", "CartPole-v1", "LunarLander-v3"]

# Pre-registered defaults (D8)
DEFAULT_B = 5
DEFAULT_TAU = 0.7
DEFAULT_RHO = 0.8
DEFAULT_LAMBDA1 = 0.6
DEFAULT_LAMBDA2 = 0.4

# Ablation sweeps
B_VALUES = [3, 5, 10]
TAU_VALUES = [0.5, 0.7, 0.9]
RHO_VALUES = [0.7, 0.8, 0.9]
LAMBDA_PAIRS = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]


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


# ── Consensus Perturbation Suite ─────────────────────────────────────

def run_consensus_perturbation_suite(
    env_name, ref_data, heldout_s, heldout_a, model_path,
    n_bootstrap=DEFAULT_B, consensus_threshold=DEFAULT_TAU,
    similarity_cutoff=DEFAULT_RHO, lambda1=DEFAULT_LAMBDA1,
    lambda2=DEFAULT_LAMBDA2, use_maxf1=False,
    sample_weight_base=None, method_tag="Consensus CBS",
):
    """Run consensus CBS through all perturbation families.

    For each perturbation condition: build consensus from B internal
    subsamples of the perturbed data, then evaluate.

    Returns (results, stability, method_params, all_build_infos)
    """
    method_key = method_tag.lower().replace(" ", "_").replace("+", "")
    env_tag = env_name.replace("-", "_").lower()
    method_params = {
        "n_categories": 5,
        "inclusion_threshold": 0.70,
        "maxf1": use_maxf1,
        "n_bootstrap": n_bootstrap,
        "consensus_threshold": consensus_threshold,
        "similarity_cutoff": similarity_cutoff,
        "lambda1": lambda1,
        "lambda2": lambda2,
    }

    results = {}
    run_entries = []
    all_build_infos = {}
    feature_ranges = compute_feature_ranges(ref_data)
    use_is = sample_weight_base is not None  # Flag: compute fresh IS weights per dataset

    def _run_consensus_and_record(run_key, family, params, data, sw=None):
        """Build consensus from data, evaluate, record."""
        # For importance-weighted voting: compute fresh IS weights for this specific dataset
        if use_is and sw is None:
            sw = compute_policy_aware_weights(
                data["states"], data["actions"], model_path)
        pipeline, rules, build_info = build_consensus_ruleset(
            data, env_name,
            n_bootstrap=n_bootstrap,
            consensus_threshold=consensus_threshold,
            similarity_cutoff=similarity_cutoff,
            lambda1=lambda1, lambda2=lambda2,
            use_maxf1=use_maxf1,
            sample_weight=sw,
        )
        res = evaluate_single_run(pipeline, rules, heldout_s, heldout_a, env_name)

        # Enrich with metadata
        res["run_id"] = f"{env_tag}_{method_key}_{run_key}"
        res["group_id"] = f"{env_tag}_{method_key}"
        res["method_params"] = method_params
        res["perturbation_family"] = family
        res["perturbation_id"] = run_key
        res["perturbation_params"] = params
        res["n_replay"] = len(data["states"])
        res["n_heldout"] = len(heldout_s)
        res["n_eval_episodes"] = len(EVAL_SEEDS)
        res["build_info"] = build_info
        res["rules"] = serialize_canonical_rules(rules)

        preds = pipeline.predict(heldout_s)
        thresholds = {int(k): v for k, v in res["thresholds"].items()}

        results[run_key] = res
        run_entries.append({
            "key": run_key,
            "family": family,
            "rules": rules,
            "thresholds": thresholds,
            "preds": preds,
        })
        all_build_infos[run_key] = build_info

    # --- 1. Seed shift ---
    print(f"  [{method_tag}][seed_shift] {len(SEED_SHIFT_SEEDS)} replays...")
    for seed in SEED_SHIFT_SEEDS:
        data = collect_replay(
            env_name=env_name, model_path=model_path,
            num_transitions=10000, seed=seed, deterministic=True,
        )
        _run_consensus_and_record(
            f"seed_shift_s{seed}", "seed_shift",
            {"replay_seed": seed}, data)

    # --- 2. Subsampling ---
    print(f"  [{method_tag}][subsampling] {N_SUBSAMPLES} uniform subsamples...")
    subsamples = generate_subsamples(ref_data, N_SUBSAMPLES,
                                      SUBSAMPLE_FRACTION, seed=42)
    for i, subset in enumerate(subsamples):
        _run_consensus_and_record(
            f"subsample_{i}", "subsample",
            {"subsample_idx": i, "fraction": SUBSAMPLE_FRACTION}, subset)

    # --- 3. Stratified subsampling ---
    print(f"  [{method_tag}][stratified] {N_SUBSAMPLES} stratified subsamples...")
    strat = generate_stratified_subsamples(ref_data, N_SUBSAMPLES,
                                            SUBSAMPLE_FRACTION, seed=42)
    for i, subset in enumerate(strat):
        _run_consensus_and_record(
            f"stratified_{i}", "stratified",
            {"subsample_idx": i, "fraction": SUBSAMPLE_FRACTION,
             "stratified": True}, subset)

    # --- 4. Cluster count variation ---
    print(f"  [{method_tag}][cluster_count] deltas={CLUSTER_DELTAS}...")
    for delta in CLUSTER_DELTAS:
        _run_consensus_and_record(
            f"cluster_delta_{delta:+d}", "cluster_count",
            {"cluster_delta": delta}, ref_data)

    # --- 5. Feature noise ---
    print(f"  [{method_tag}][feature_noise] levels={NOISE_LEVELS}...")
    for nl in NOISE_LEVELS:
        noisy = add_feature_noise(ref_data, nl, seed=42,
                                   feature_ranges=feature_ranges)
        _run_consensus_and_record(
            f"noise_{nl:.3f}", "feature_noise",
            {"noise_level": nl}, noisy)

    # --- Anchor validation ---
    all_preds = [e["preds"] for e in run_entries]
    assert all(p.shape == all_preds[0].shape for p in all_preds), \
        "BRA anchor mismatch"

    # --- Global stability ---
    print(f"  [{method_tag}][metrics] Computing stability...")
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
    print(f"  [{method_tag}][metrics] Computing per-run proxies...")
    proxies = compute_per_run_stability_proxies(run_entries, fr)
    for key, proxy in proxies.items():
        results[key]["stability_proxy_global"] = proxy["global"]
        results[key]["stability_proxy_family"] = proxy["family"]

    return results, stability, method_params, all_build_infos


# ── Ablation Sweeps ──────────────────────────────────────────────────

def run_ablation_sweeps(env_name, model_path, heldout_s, heldout_a):
    """Run B×τ, ρ, and λ sweeps with 5 outer repeats per cell.

    Uses seed-shift replays as outer repeats for true cross-run stability.
    """
    print(f"\n  --- Ablation Sweeps ---")
    feature_names = ENV_FEATURE_NAMES.get(env_name)

    # Collect 5 outer-repeat replays (seed shift)
    outer_datasets = []
    for seed in SEED_SHIFT_SEEDS:
        data = collect_replay(
            env_name=env_name, model_path=model_path,
            num_transitions=10000, seed=seed, deterministic=True,
        )
        outer_datasets.append(data)

    feature_ranges = compute_feature_ranges(outer_datasets[0])
    fr = {i: float(feature_ranges[i]) for i in range(len(feature_ranges))}

    def _run_ablation_cell(B, tau, rho, lam1, lam2, label):
        """Run one ablation cell with 5 outer repeats."""
        all_rules = []
        all_thresholds = []
        all_preds = []
        fidelities = []
        deployments = []
        build_infos = []
        per_action_recalls = []

        for data in outer_datasets:
            pipeline, rules, info = build_consensus_ruleset(
                data, env_name,
                n_bootstrap=B,
                consensus_threshold=tau,
                similarity_cutoff=rho,
                lambda1=lam1, lambda2=lam2,
            )
            res = evaluate_single_run(
                pipeline, rules, heldout_s, heldout_a, env_name)
            fidelities.append(res["fidelity_heldout"])
            deployments.append(res["deployment"])
            build_infos.append(info)
            all_rules.append(rules)
            all_thresholds.append(
                {int(k): v for k, v in res["thresholds"].items()})
            all_preds.append(pipeline.predict(heldout_s))

            # Worst-action recall
            pa = res["fidelity_per_action"]["per_action"]
            recalls = [pa[a]["recall"] for a in pa]
            per_action_recalls.append(min(recalls) if recalls else 0.0)

        # Compute stability across 5 outer repeats
        grs_wj = mean_pairwise_jaccard(all_rules, weighted=True)
        grs_ta = mean_pairwise_soft_jaccard(all_rules, threshold_aware=True)
        td = mean_pairwise_threshold_drift(all_thresholds, feature_ranges=fr)
        bra = compute_bra_from_predictions(all_preds)

        f1_vals = [f["f1"] for f in fidelities]
        ecr_vals = [d["E_CR"] for d in deployments]
        n_rules_vals = [i["n_consensus_rules"] for i in build_infos]

        return {
            "fidelity": {
                "mean_f1": float(np.mean(f1_vals)),
                "std_f1": float(np.std(f1_vals)),
                "mean_worst_action_recall": float(np.mean(per_action_recalls)),
            },
            "deployment": {
                "mean_ecr": float(np.mean(ecr_vals)),
                "std_ecr": float(np.std(ecr_vals)),
            },
            "stability": {
                "GRS_wj": float(grs_wj),
                "GRS_ta": float(grs_ta),
                "BRA": float(bra),
                "TD": float(td),
            },
            "n_outer_repeats": len(outer_datasets),
            "mean_consensus_rules": float(np.mean(n_rules_vals)),
            "per_action_rule_counts": build_infos[0]["per_action_rule_counts"],
            "actions_lost": build_infos[0]["actions_lost"],
        }

    # B × τ grid
    print(f"  [ablation] B×τ grid ({len(B_VALUES)}×{len(TAU_VALUES)})...")
    b_tau_grid = {}
    for B in B_VALUES:
        for tau in TAU_VALUES:
            label = f"B{B}_tau{tau}"
            print(f"    {label}...")
            b_tau_grid[label] = _run_ablation_cell(
                B, tau, DEFAULT_RHO, DEFAULT_LAMBDA1, DEFAULT_LAMBDA2, label)

    # ρ sweep
    print(f"  [ablation] ρ sweep ({len(RHO_VALUES)} values)...")
    rho_sweep = {}
    for rho in RHO_VALUES:
        label = f"rho_{rho}"
        print(f"    {label}...")
        rho_sweep[label] = _run_ablation_cell(
            DEFAULT_B, DEFAULT_TAU, rho, DEFAULT_LAMBDA1, DEFAULT_LAMBDA2, label)

    # λ sweep
    print(f"  [ablation] λ sweep ({len(LAMBDA_PAIRS)} pairs)...")
    lambda_sweep = {}
    for l1, l2 in LAMBDA_PAIRS:
        label = f"l{l1}_{l2}"
        print(f"    {label}...")
        lambda_sweep[label] = _run_ablation_cell(
            DEFAULT_B, DEFAULT_TAU, DEFAULT_RHO, l1, l2, label)

    return {
        "B_tau_grid": b_tau_grid,
        "rho_sweep": rho_sweep,
        "lambda_sweep": lambda_sweep,
    }


# ── Main Runner ──────────────────────────────────────────────────────

def run_env(env_name, skip_ablations=False):
    print(f"\n{'='*60}")
    print(f"  Consensus CBS: {env_name}")
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

    t0 = time.time()

    # --- Consensus merge ---
    print(f"\n  --- Method: Consensus CBS ---")
    b3_results, b3_stability, b3_params, b3_infos = \
        run_consensus_perturbation_suite(
            env_name, ref_data, heldout_s, heldout_a, model_path,
            method_tag="Consensus CBS")

    # --- rule-set voting: Naive voting baseline ---
    print(f"\n  --- Method: Voting Baseline (rule-set voting) ---")
    vote_results, vote_stability, vote_params, vote_infos = \
        run_consensus_voting_suite(
            env_name, ref_data, heldout_s, heldout_a, model_path)

    elapsed_main = time.time() - t0

    # --- Ablations ---
    ablations = {}
    if not skip_ablations:
        t_abl = time.time()
        ablations = run_ablation_sweeps(
            env_name, model_path, heldout_s, heldout_a)
        elapsed_abl = time.time() - t_abl
    else:
        elapsed_abl = 0

    elapsed = time.time() - t0

    # --- importance-weighted voting: Policy-aware reweighting variant ---
    print(f"\n  --- Method: Policy-Aware Reweighting (importance-weighted voting variant) ---")
    t_is = time.time()
    is_weights = compute_policy_aware_weights(
        ref_data["states"], ref_data["actions"], model_path)
    print(f"  IS weights: mean={is_weights.mean():.3f}, "
          f"std={is_weights.std():.3f}, "
          f"min={is_weights.min():.3f}, max={is_weights.max():.3f}")

    is_results, is_stability, is_params, is_infos = \
        run_consensus_perturbation_suite(
            env_name, ref_data, heldout_s, heldout_a, model_path,
            sample_weight_base=is_weights,
            method_tag="Policy Reweight")
    elapsed_is = time.time() - t_is
    elapsed = time.time() - t0

    # --- Save results ---
    out_dir = f"experiments/results/{env_name.replace('-', '_').lower()}"
    os.makedirs(out_dir, exist_ok=True)

    output = {
        "schema_version": "consensus_cbs_v1",
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
        "consensus_cbs": {
            "method_params": b3_params,
            "per_run": _serialize(b3_results),
            "stability": b3_stability,
        },
        "consensus_vote": {
            "method_params": vote_params,
            "per_run": _serialize(vote_results),
            "stability": vote_stability,
        },
        "ablations": ablations,
        "policy_reweight": {
            "method_params": is_params,
            "per_run": _serialize(is_results),
            "stability": is_stability,
            "weight_stats": {
                "mean": float(is_weights.mean()),
                "std": float(is_weights.std()),
                "min": float(is_weights.min()),
                "max": float(is_weights.max()),
            },
        },
    }

    out_path = os.path.join(out_dir, "consensus_merge_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    # --- Print summary ---
    print(f"\n  {'='*50}")
    print(f"  Consensus CBS Stability:")
    for k, v in b3_stability.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    print(f"\n  Voting Baseline Stability:")
    for k, v in vote_stability.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    print(f"\n  Policy Reweight (importance-weighted voting) Stability:")
    for k, v in is_stability.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    print(f"  {'='*50}")
    print(f"  Total elapsed: {elapsed:.1f}s")


def _deploy_voting_ensemble(pipelines, env_name, eval_seeds, success_threshold):
    """Deploy a voting ensemble as a policy in the environment."""
    import gymnasium as gym

    env = gym.make(env_name)
    episode_rewards = []
    episode_lengths = []

    for ep_seed in eval_seeds:
        obs, info = env.reset(seed=ep_seed)
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action = int(voting_predict(pipelines, obs.reshape(1, -1))[0])
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    env.close()

    rewards_arr = np.array(episode_rewards)
    lengths_arr = np.array(episode_lengths)
    n_success = sum(1 for r in episode_rewards if r >= success_threshold) if success_threshold else None
    return {
        "E_CR": float(rewards_arr.mean()),
        "E_CR_std": float(rewards_arr.std()),
        "E_TS": float(lengths_arr.mean()),
        "success_rate": n_success / len(episode_rewards) if n_success is not None else None,
    }


def run_consensus_voting_suite(
    env_name, ref_data, heldout_s, heldout_a, model_path,
):
    """Run naive voting baseline through all perturbation families.

    For each perturbation: build B=5 CBS pipelines, predict by majority vote.
    """
    method_tag = "Voting Baseline"
    method_key = "consensus_vote"
    env_tag = env_name.replace("-", "_").lower()
    method_params = {
        "n_categories": 5,
        "inclusion_threshold": 0.70,
        "n_bootstrap": DEFAULT_B,
        "type": "majority_vote",
    }

    results = {}
    run_entries = []
    feature_ranges = compute_feature_ranges(ref_data)

    def _run_vote_and_record(run_key, family, params, data):
        pipelines = build_voting_ensemble(
            data, env_name, n_bootstrap=DEFAULT_B)
        preds = voting_predict(pipelines, heldout_s)

        # Compute fidelity manually (since we don't have a single pipeline)
        accuracy = float(np.mean(preds == heldout_a))
        # Per-action recall + precision + F1
        actions_set = sorted(np.unique(heldout_a))
        recalls, precisions = [], []
        per_action = {}
        for a in actions_set:
            true_mask = heldout_a == a
            pred_mask = preds == a
            support = int(true_mask.sum())
            tp = int((true_mask & pred_mask).sum())
            prec = tp / pred_mask.sum() if pred_mask.sum() > 0 else 0.0
            rec = tp / true_mask.sum() if true_mask.sum() > 0 else 0.0
            pa_f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            # Count rules for this action across all pipelines (use first for structural)
            rule_count = sum(1 for r in pipelines[0].get_rules() if r.action == a)
            per_action[int(a)] = {
                "precision": float(prec), "recall": float(rec),
                "f1": float(pa_f1), "support": support,
                "rule_count": rule_count,
            }
            if support > 0:
                precisions.append(prec)
                recalls.append(rec)
        recall = float(np.mean(recalls)) if recalls else 0.0
        f1 = 2 * accuracy * recall / (accuracy + recall) if (accuracy + recall) > 0 else 0.0
        macro_prec = float(np.mean(precisions)) if precisions else 0.0
        macro_rec = recall
        macro_f1 = (2 * macro_prec * macro_rec / (macro_prec + macro_rec)
                     if (macro_prec + macro_rec) > 0 else 0.0)

        # Deploy with voting across all pipelines
        deploy = _deploy_voting_ensemble(
            pipelines, env_name, EVAL_SEEDS,
            SUCCESS_THRESHOLDS.get(env_name))

        # Use first pipeline for rules/thresholds (for structural metrics)
        rules = canonicalize_rules(pipelines[0].get_rules())
        thresholds = {int(k): [float(v) for v in vs]
                      for k, vs in pipelines[0].get_thresholds().items()}

        res = {
            "fidelity_heldout": {"accuracy": accuracy, "recall": recall, "f1": f1},
            "fidelity_per_action": {
                "per_action": per_action,
                "macro_precision": macro_prec,
                "macro_recall": macro_rec,
                "macro_f1": macro_f1,
            },
            "deployment": {
                "E_CR": deploy["E_CR"],
                "E_CR_std": deploy["E_CR_std"],
                "E_TS": deploy["E_TS"],
                "success_rate": deploy["success_rate"],
            },
            "n_rules": len(rules),
            "thresholds": thresholds,
            "run_id": f"{env_tag}_{method_key}_{run_key}",
            "group_id": f"{env_tag}_{method_key}",
            "method_params": method_params,
            "perturbation_family": family,
            "perturbation_id": run_key,
            "perturbation_params": params,
            "n_replay": len(data["states"]),
            "n_heldout": len(heldout_s),
            "n_eval_episodes": len(EVAL_SEEDS),
            "rules": serialize_canonical_rules(rules),
        }

        results[run_key] = res
        run_entries.append({
            "key": run_key,
            "family": family,
            "rules": rules,
            "thresholds": thresholds,
            "preds": preds,
        })

    # Same perturbation families as the consensus merge
    print(f"  [{method_tag}][seed_shift]...")
    for seed in SEED_SHIFT_SEEDS:
        data = collect_replay(env_name=env_name, model_path=model_path,
                              num_transitions=10000, seed=seed, deterministic=True)
        _run_vote_and_record(f"seed_shift_s{seed}", "seed_shift",
                             {"replay_seed": seed}, data)

    print(f"  [{method_tag}][subsampling]...")
    subs = generate_subsamples(ref_data, N_SUBSAMPLES, SUBSAMPLE_FRACTION, seed=42)
    for i, subset in enumerate(subs):
        _run_vote_and_record(f"subsample_{i}", "subsample",
                             {"subsample_idx": i}, subset)

    print(f"  [{method_tag}][stratified]...")
    strats = generate_stratified_subsamples(ref_data, N_SUBSAMPLES, SUBSAMPLE_FRACTION, seed=42)
    for i, subset in enumerate(strats):
        _run_vote_and_record(f"stratified_{i}", "stratified",
                             {"subsample_idx": i}, subset)

    print(f"  [{method_tag}][cluster_count]...")
    for delta in CLUSTER_DELTAS:
        _run_vote_and_record(f"cluster_delta_{delta:+d}", "cluster_count",
                             {"cluster_delta": delta}, ref_data)

    print(f"  [{method_tag}][feature_noise]...")
    for nl in NOISE_LEVELS:
        noisy = add_feature_noise(ref_data, nl, seed=42,
                                   feature_ranges=feature_ranges)
        _run_vote_and_record(f"noise_{nl:.3f}", "feature_noise",
                             {"noise_level": nl}, noisy)

    # Stability
    all_preds = [e["preds"] for e in run_entries]
    all_rule_sets = [e["rules"] for e in run_entries]
    all_threshold_sets = [e["thresholds"] for e in run_entries]
    fr = {i: float(feature_ranges[i]) for i in range(len(feature_ranges))}

    stability = {
        "GRS_weighted_jaccard": float(mean_pairwise_jaccard(all_rule_sets, weighted=True)),
        "GRS_plain_jaccard": float(mean_pairwise_jaccard(all_rule_sets, weighted=False)),
        "GRS_threshold_aware": float(mean_pairwise_soft_jaccard(all_rule_sets, threshold_aware=True)),
        "TD": float(mean_pairwise_threshold_drift(
            [{int(k): v for k, v in ts.items()} for ts in all_threshold_sets],
            feature_ranges=fr)),
        "BRA": float(compute_bra_from_predictions(all_preds)),
        "n_perturbation_runs": len(all_rule_sets),
    }

    # Per-run proxies
    proxies = compute_per_run_stability_proxies(run_entries, fr)
    for key, proxy in proxies.items():
        results[key]["stability_proxy_global"] = proxy["global"]
        results[key]["stability_proxy_family"] = proxy["family"]

    return results, stability, method_params, {}


def main():
    parser = argparse.ArgumentParser(description="Run Consensus CBS experiments")
    parser.add_argument("--env", type=str, default="all",
                        help="Environment name or 'all'")
    parser.add_argument("--skip-ablations", action="store_true",
                        help="Skip ablation sweeps")
    args = parser.parse_args()

    envs = ENVS if args.env == "all" else [args.env]
    for env_name in envs:
        run_env(env_name, skip_ablations=args.skip_ablations)

    print("\nAll Consensus CBS experiments complete!")


if __name__ == "__main__":
    main()
