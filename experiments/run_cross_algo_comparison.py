#!/usr/bin/env python
"""
Cross-Algorithm (PPO vs DQN) Explanation Stability Comparison on CartPole-v1.

Collects PPO replay on CartPole-v1, runs the same CBS / DT / rule-set voting
stress protocol, and compares against existing DQN results.

Usage:
    python experiments/run_cross_algo_comparison.py

Output:
    experiments/results/cross_algo_comparison/
        ppo_cartpole_stress.json
        ppo_vs_dqn_comparison.json
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from reproduction.cbs import CBSPipeline
from reproduction.collect_replay import (
    collect_replay, save_replay, print_summary,
    ENV_FEATURE_NAMES, ENV_ACTION_NAMES,
)
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
)
from experiments.decision_tree_surrogate import DecisionTreeSurrogate
from experiments.consensus_merge import build_voting_ensemble, voting_predict

# ── Constants ─────────────────────────────────────────────────────────
ENV_NAME = "CartPole-v1"
ENV_TAG = "cartpole_v1"
RESULTS_DIR = "experiments/results/cross_algo_comparison"
EVAL_SEEDS = list(range(1000, 1050))
SUCCESS_THRESHOLD = 475.0
HELDOUT_SEED = 99
SEED_SHIFT_SEEDS = [0, 1, 2, 3, 4]
N_SUBSAMPLES = 5
SUBSAMPLE_FRACTION = 0.8
CLUSTER_DELTAS = [-1, 0, 1]
NOISE_LEVELS = [0.01, 0.03, 0.05]
FEATURE_NAMES = ENV_FEATURE_NAMES[ENV_NAME]

PPO_MODEL = "reproduction/models/ppo_cartpole_v1.zip"
DQN_MODEL = "reproduction/models/dqn_cartpole_v1.zip"
DQN_STRESS_PATH = "experiments/results/cartpole_v1/stress_test_results.json"


def _load_model(model_path, algo):
    if algo == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(model_path)
    else:
        from stable_baselines3 import DQN
        return DQN.load(model_path)


def collect_replay_cartpole(model_path, n_transitions, seed, algo):
    """Collect replay from CartPole using given algo."""
    import gymnasium as gym
    env = gym.make(ENV_NAME)
    model = _load_model(model_path, algo)

    states, actions, rewards, dones, episode_ids = [], [], [], [], []
    episode_count = 0
    obs, _ = env.reset(seed=seed)

    while len(states) < n_transitions:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        states.append(obs.copy())
        actions.append(action)
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        rewards.append(reward)
        dones.append(done)
        episode_ids.append(episode_count)
        if done:
            episode_count += 1
            obs, _ = env.reset(seed=seed + episode_count)
        else:
            obs = next_obs

    env.close()
    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int32),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=bool),
        "episode_ids": np.array(episode_ids, dtype=np.int32),
    }


def run_cbs_on_data(states, actions, kmeans_seed=0, delta=0):
    cbs = CBSPipeline(
        n_categories=5,
        inclusion_threshold=0.70,
        kmeans_seed=kmeans_seed,
        cluster_count_delta=delta,
        feature_names=FEATURE_NAMES,
    )
    cbs.fit(states, actions)
    rules = canonicalize_rules(cbs.get_rules())
    return cbs, rules


def evaluate_cbs_in_env(cbs):
    import gymnasium as gym
    env = gym.make(ENV_NAME)
    rewards_list, lengths = [], []
    for seed in EVAL_SEEDS[:20]:
        obs, _ = env.reset(seed=seed)
        done, ep_r, ep_l = False, 0.0, 0
        while not done:
            a = cbs.predict(obs.reshape(1, -1))[0]
            obs, r, term, trunc, _ = env.step(int(a))
            ep_r += r; ep_l += 1; done = term or trunc
        rewards_list.append(ep_r)
        lengths.append(ep_l)
    env.close()
    return {
        "E_CR": float(np.mean(rewards_list)),
        "E_CR_std": float(np.std(rewards_list)),
        "success_rate": float(np.mean([r >= SUCCESS_THRESHOLD for r in rewards_list])),
    }


def evaluate_dt_in_env(dt):
    import gymnasium as gym
    env = gym.make(ENV_NAME)
    rewards_list, lengths = [], []
    for seed in EVAL_SEEDS[:20]:
        obs, _ = env.reset(seed=seed)
        done, ep_r, ep_l = False, 0.0, 0
        while not done:
            a = dt.predict(obs.reshape(1, -1))[0]
            obs, r, term, trunc, _ = env.step(int(a))
            ep_r += r; ep_l += 1; done = term or trunc
        rewards_list.append(ep_r)
        lengths.append(ep_l)
    env.close()
    return {
        "E_CR": float(np.mean(rewards_list)),
        "E_CR_std": float(np.std(rewards_list)),
        "success_rate": float(np.mean([r >= SUCCESS_THRESHOLD for r in rewards_list])),
    }


def compute_bra_from_predictions(pred_list):
    """BRA across multiple prediction arrays."""
    if len(pred_list) < 2:
        return 1.0
    agreements = []
    for i in range(len(pred_list)):
        for j in range(i + 1, len(pred_list)):
            minlen = min(len(pred_list[i]), len(pred_list[j]))
            agreements.append(np.mean(pred_list[i][:minlen] == pred_list[j][:minlen]))
    return float(np.mean(agreements))


def add_feature_noise_masked(data, noise_level, seed=42, feature_ranges=None):
    """Add Gaussian noise to continuous features only (all for CartPole)."""
    rng = np.random.default_rng(seed)
    noisy = data.copy()
    noisy["states"] = data["states"].copy()
    n_features = data["states"].shape[1]
    for j in range(n_features):
        rng_feat = feature_ranges.get(j, 1.0) if feature_ranges else 1.0
        noise = rng.normal(0, noise_level * rng_feat, size=len(data["states"]))
        noisy["states"][:, j] += noise.astype(np.float32)
    return noisy


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Cross-Algorithm Comparison: PPO vs DQN on CartPole-v1")
    print("=" * 60)

    algo = "ppo"
    model_path = PPO_MODEL

    # Collect reference replay
    replay_path = f"reproduction/data/replay_{ENV_TAG}_{algo}_seed42.npz"
    if os.path.exists(replay_path):
        ref_data = load_replay_npz(replay_path)
        print(f"  Loaded existing replay: {len(ref_data['states'])} transitions")
    else:
        print("  Collecting PPO reference replay...")
        ref_data = collect_replay_cartpole(model_path, 10000, seed=42, algo=algo)
        np.savez_compressed(replay_path,
                            states=ref_data["states"],
                            actions=ref_data["actions"],
                            rewards=ref_data["rewards"],
                            dones=ref_data["dones"],
                            episode_ids=ref_data["episode_ids"])
        print(f"  Saved replay: {len(ref_data['states'])} transitions")

    # Collect held-out
    print("  Collecting held-out replay...")
    heldout_data = collect_replay_cartpole(model_path, 5000, seed=HELDOUT_SEED, algo=algo)
    heldout_s = heldout_data["states"]
    heldout_a = heldout_data["actions"]
    print(f"  Held-out: {len(heldout_s)} transitions")

    feature_ranges = compute_feature_ranges(ref_data)
    fr_dict = {i: float(feature_ranges[i]) for i in range(len(feature_ranges))}

    t0 = time.time()
    all_results = {}

    # === CBS Stress ===
    print("\n  === Method: CBS (PPO) ===")
    cbs_entries = []
    cbs_results = {}

    def cbs_record(run_key, family, params, states, actions, n_replay,
                   kmeans_seed=0, delta=0):
        p, rules = run_cbs_on_data(states, actions, kmeans_seed=kmeans_seed,
                                    delta=delta)
        preds = p.predict(heldout_s)
        f1 = f1_score(heldout_a, preds, average="macro")
        acc = accuracy_score(heldout_a, preds)
        deploy = evaluate_cbs_in_env(p)
        res = {
            "fidelity_heldout": {"f1": float(f1), "accuracy": float(acc)},
            "deployment": deploy,
            "n_rules": len(rules),
            "run_id": f"{ENV_TAG}_ppo_cbs_{run_key}",
            "method": "CBS",
            "algo": "ppo",
            "perturbation_family": family,
            "perturbation_id": run_key,
            "perturbation_params": params,
            "n_replay": n_replay,
        }
        cbs_results[run_key] = res
        thresholds = {}
        if hasattr(p, 'get_thresholds'):
            thresholds = {int(k): [float(v) for v in vs]
                          for k, vs in p.get_thresholds().items()}
        cbs_entries.append({
            "key": run_key, "family": family, "rules": rules, "preds": preds,
            "thresholds": thresholds,
        })

    print("  [CBS] seed_shift ×5...")
    for seed in SEED_SHIFT_SEEDS:
        data = collect_replay_cartpole(model_path, 10000, seed=seed, algo=algo)
        cbs_record(f"seed_shift_s{seed}", "seed_shift", {"seed": seed},
                   data["states"], data["actions"], len(data["states"]))

    print("  [CBS] stratified_subsample ×5...")
    strat = generate_stratified_subsamples(ref_data, N_SUBSAMPLES,
                                            SUBSAMPLE_FRACTION, seed=42)
    for i, ss in enumerate(strat):
        cbs_record(f"stratified_{i}", "stratified_subsample", {"idx": i},
                   ss["states"], ss["actions"], len(ss["states"]))

    print("  [CBS] cluster_count ×3...")
    for delta in CLUSTER_DELTAS:
        cbs_record(f"cluster_delta_{delta:+d}", "cluster_count", {"delta": delta},
                   ref_data["states"], ref_data["actions"],
                   len(ref_data["states"]), delta=delta)

    print("  [CBS] feature_noise ×3...")
    for nl in NOISE_LEVELS:
        noisy = add_feature_noise_masked(ref_data, nl, seed=42,
                                          feature_ranges=fr_dict)
        cbs_record(f"noise_{nl:.3f}", "feature_noise", {"noise_level": nl},
                   noisy["states"], noisy["actions"], len(noisy["states"]))

    # CBS stability
    cbs_rules_list = [e["rules"] for e in cbs_entries]
    grs_wj = mean_pairwise_jaccard(cbs_rules_list, weighted=True)
    grs_ta = mean_pairwise_soft_jaccard(cbs_rules_list, threshold_aware=True)
    cbs_thresh_list = [e["thresholds"] for e in cbs_entries]
    td_val = mean_pairwise_threshold_drift(cbs_thresh_list, feature_ranges=fr_dict)
    cbs_preds = [e["preds"] for e in cbs_entries]
    bra = compute_bra_from_predictions(cbs_preds)
    all_results["CBS"] = {
        "runs": cbs_results,
        "stability": {
            "GRS_weighted_jaccard": float(grs_wj),
            "GRS_threshold_aware": float(grs_ta),
            "TD": float(td_val),
            "BRA": float(bra),
            "n_runs": len(cbs_entries),
        },
    }
    print(f"    GRS_wj={grs_wj:.3f}, GRS_ta={grs_ta:.3f}, TD={td_val:.3f}, BRA={bra:.3f}")

    # === DT Stress ===
    print("\n  === Method: DT (PPO) ===")
    dt_entries = []
    dt_results = {}

    def dt_record(run_key, family, params, states, actions, n_replay, max_depth=None):
        dt = DecisionTreeSurrogate(max_depth=max_depth, feature_names=FEATURE_NAMES)
        dt.fit(states, actions)
        rules = canonicalize_rules(dt.get_rules())
        preds = dt.predict(heldout_s)
        f1 = f1_score(heldout_a, preds, average="macro")
        acc = accuracy_score(heldout_a, preds)
        deploy = evaluate_dt_in_env(dt)
        res = {
            "fidelity_heldout": {"f1": float(f1), "accuracy": float(acc)},
            "deployment": deploy,
            "n_rules": len(rules),
            "run_id": f"{ENV_TAG}_ppo_dt_{run_key}",
            "method": "DT",
            "algo": "ppo",
            "perturbation_family": family,
            "perturbation_id": run_key,
            "perturbation_params": params,
            "n_replay": n_replay,
        }
        dt_results[run_key] = res
        dt_entries.append({
            "key": run_key, "family": family, "rules": rules, "preds": preds,
        })

    print("  [DT] seed_shift ×5...")
    for seed in SEED_SHIFT_SEEDS:
        data = collect_replay_cartpole(model_path, 10000, seed=seed, algo=algo)
        dt_record(f"seed_shift_s{seed}", "seed_shift", {"seed": seed},
                  data["states"], data["actions"], len(data["states"]))

    print("  [DT] stratified_subsample ×5...")
    for i, ss in enumerate(strat):
        dt_record(f"stratified_{i}", "stratified_subsample", {"idx": i},
                  ss["states"], ss["actions"], len(ss["states"]))

    print("  [DT] depth_variation ×3...")
    for depth in [3, None, 7]:
        dt_record(f"depth_{depth}", "depth_variation", {"max_depth": depth},
                  ref_data["states"], ref_data["actions"],
                  len(ref_data["states"]), max_depth=depth)

    print("  [DT] feature_noise ×3...")
    for nl in NOISE_LEVELS:
        noisy = add_feature_noise_masked(ref_data, nl, seed=42,
                                          feature_ranges=fr_dict)
        dt_record(f"noise_{nl:.3f}", "feature_noise", {"noise_level": nl},
                  noisy["states"], noisy["actions"], len(noisy["states"]))

    dt_rules_list = [e["rules"] for e in dt_entries]
    grs_wj_dt = mean_pairwise_jaccard(dt_rules_list, weighted=True)
    grs_ta_dt = mean_pairwise_soft_jaccard(dt_rules_list, threshold_aware=True)
    dt_thresh_list = [{} for _ in dt_entries]
    td_dt = mean_pairwise_threshold_drift(dt_thresh_list, feature_ranges=fr_dict)
    dt_preds = [e["preds"] for e in dt_entries]
    bra_dt = compute_bra_from_predictions(dt_preds)
    all_results["DT"] = {
        "runs": dt_results,
        "stability": {
            "GRS_weighted_jaccard": float(grs_wj_dt),
            "GRS_threshold_aware": float(grs_ta_dt),
            "TD": float(td_dt),
            "BRA": float(bra_dt),
            "n_runs": len(dt_entries),
        },
    }
    print(f"    GRS_wj={grs_wj_dt:.3f}, GRS_ta={grs_ta_dt:.3f}, TD={td_dt:.3f}, BRA={bra_dt:.3f}")

    # === rule-set voting Stress ===
    print("\n  === Method: rule-set voting (PPO) ===")
    b3_entries = []
    b3_results = {}

    def b3_record(run_key, family, params, replay_data, n_replay):
        full_data = dict(replay_data)
        n = len(full_data["states"])
        if "rewards" not in full_data:
            full_data["rewards"] = np.zeros(n, dtype=np.float32)
        if "dones" not in full_data:
            full_data["dones"] = np.zeros(n, dtype=bool)
        if "episode_ids" not in full_data:
            full_data["episode_ids"] = np.zeros(n, dtype=np.int32)
        subsamples = generate_subsamples(full_data, 5, 0.8, seed=42)
        pipelines = []
        for ss in subsamples:
            p, _ = run_cbs_on_data(ss["states"], ss["actions"])
            pipelines.append(p)
        preds = voting_predict(pipelines, heldout_s)
        f1 = f1_score(heldout_a, preds, average="macro")
        acc = accuracy_score(heldout_a, preds)
        import gymnasium as gym
        env = gym.make(ENV_NAME)
        deploy_r = []
        for seed in EVAL_SEEDS[:20]:
            obs, _ = env.reset(seed=seed)
            done, ep_r = False, 0.0
            while not done:
                a = voting_predict(pipelines, obs.reshape(1, -1))[0]
                obs, r, term, trunc, _ = env.step(int(a))
                ep_r += r; done = term or trunc
            deploy_r.append(ep_r)
        env.close()
        res = {
            "fidelity_heldout": {"f1": float(f1), "accuracy": float(acc)},
            "deployment": {
                "E_CR": float(np.mean(deploy_r)),
                "E_CR_std": float(np.std(deploy_r)),
                "success_rate": float(np.mean([r >= SUCCESS_THRESHOLD for r in deploy_r])),
            },
            "n_rules": sum(len(p.get_rules()) for p in pipelines),
            "run_id": f"{ENV_TAG}_ppo_b3vote_{run_key}",
            "method": "B3-vote",
            "algo": "ppo",
            "perturbation_family": family,
            "perturbation_id": run_key,
            "perturbation_params": params,
            "n_replay": n_replay,
        }
        b3_results[run_key] = res
        b3_entries.append({
            "key": run_key, "family": family, "preds": preds,
        })

    print("  [RV] seed_shift ×5...")
    for seed in SEED_SHIFT_SEEDS:
        data = collect_replay_cartpole(model_path, 10000, seed=seed, algo=algo)
        b3_record(f"seed_shift_s{seed}", "seed_shift", {"seed": seed},
                  data, len(data["states"]))

    print("  [RV] stratified ×5...")
    for i, ss in enumerate(strat):
        b3_record(f"stratified_{i}", "stratified_subsample", {"idx": i},
                  ss, len(ss["states"]))

    print("  [RV] cluster_count ×3...")
    for delta in CLUSTER_DELTAS:
        b3_record(f"cluster_delta_{delta:+d}", "cluster_count",
                  {"delta": delta}, ref_data, len(ref_data["states"]))

    print("  [RV] feature_noise ×3...")
    for nl in NOISE_LEVELS:
        noisy = add_feature_noise_masked(ref_data, nl, seed=42,
                                          feature_ranges=fr_dict)
        b3_record(f"noise_{nl:.3f}", "feature_noise", {"noise_level": nl},
                  noisy, len(noisy["states"]))

    b3_preds = [e["preds"] for e in b3_entries]
    b3_bra = compute_bra_from_predictions(b3_preds)
    all_results["B3-vote"] = {
        "runs": b3_results,
        "stability": {"BRA": float(b3_bra), "n_runs": len(b3_entries)},
    }
    print(f"    rule-set voting BRA={b3_bra:.3f}")

    elapsed = time.time() - t0

    # Save PPO stress results
    output = {
        "schema_version": "external_validity_v1",
        "env": ENV_NAME,
        "algo": "ppo",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        **all_results,
    }
    ppo_path = os.path.join(RESULTS_DIR, "ppo_cartpole_stress.json")
    with open(ppo_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  PPO stress results saved to {ppo_path}")

    # === Load DQN results and build comparison ===
    print("\n  === Building PPO vs DQN comparison ===")
    comparison = {
        "env": ENV_NAME,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ppo": {},
        "dqn": {},
        "delta": {},
    }

    # PPO summary
    for method in ["CBS", "DT", "B3-vote"]:
        if method in all_results:
            comparison["ppo"][method] = all_results[method]["stability"]

    # Load DQN stress results
    if os.path.exists(DQN_STRESS_PATH):
        with open(DQN_STRESS_PATH) as f:
            dqn_data = json.load(f)
        # DQN results use different keys (cbs, cbs_maxf1)
        if "cbs" in dqn_data:
            comparison["dqn"]["CBS"] = dqn_data["cbs"]["stability"]
        # DQN results may not have DT/rule-set voting from original stress test
        for method in ["DT", "B3-vote"]:
            if method in dqn_data:
                comparison["dqn"][method] = dqn_data[method]["stability"]

        # Compute deltas
        for method in comparison["ppo"]:
            if method in comparison["dqn"]:
                delta = {}
                for metric in comparison["ppo"][method]:
                    if metric in comparison["dqn"][method] and isinstance(
                            comparison["ppo"][method][metric], (int, float)):
                        delta[metric] = round(
                            comparison["ppo"][method][metric] -
                            comparison["dqn"][method][metric], 4)
                comparison["delta"][method] = delta
    else:
        comparison["dqn"]["note"] = f"DQN stress results not found at {DQN_STRESS_PATH}"

    comp_path = os.path.join(RESULTS_DIR, "ppo_vs_dqn_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"  Comparison saved to {comp_path}")

    # Print comparison table
    print("\n  ┌────────────┬───────────────────────────────┬───────────────────────────────┐")
    print("  │ Metric     │ DQN                           │ PPO                           │")
    print("  ├────────────┼───────────────────────────────┼───────────────────────────────┤")
    if "CBS" in comparison["dqn"] and "CBS" in comparison["ppo"]:
        for metric in ["GRS_weighted_jaccard", "GRS_threshold_aware", "TD", "BRA"]:
            dv = comparison["dqn"]["CBS"].get(metric, "N/A")
            pv = comparison["ppo"]["CBS"].get(metric, "N/A")
            dv_s = f"{dv:.3f}" if isinstance(dv, float) else str(dv)
            pv_s = f"{pv:.3f}" if isinstance(pv, float) else str(pv)
            print(f"  │ CBS {metric:6s} │ {dv_s:29s} │ {pv_s:29s} │")
    print("  └────────────┴───────────────────────────────┴───────────────────────────────┘")

    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
