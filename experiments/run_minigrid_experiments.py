#!/usr/bin/env python
"""
MiniGrid External Validity Experiment Runner.

Runs the full CBS / rule-set voting / DT surrogate stability protocol on
MiniGrid-Dynamic-Obstacles-8x8-v0 with PPO policy.

Steps:
  1. Feasibility check  (CBS, rule-set voting, DT on reference replay)
  2. Compressed stress suite  (seed_shift×5, subsample×5, cluster_count×3, noise×3)
  3. PPO vs DQN comparison  (if DQN model available, or note infeasibility)

Usage:
    python experiments/run_minigrid_experiments.py                  # full suite
    python experiments/run_minigrid_experiments.py --step feasibility  # step A only
    python experiments/run_minigrid_experiments.py --step stress       # step B only

Output:
    experiments/results/minigrid_dynamic_obstacles_8x8_v0/
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

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
    ruleset_weighted_jaccard,
    ruleset_soft_jaccard,
    threshold_drift,
)
from experiments.decision_tree_surrogate import DecisionTreeSurrogate

# ── Constants ─────────────────────────────────────────────────────────
ENV_NAME = "MiniGrid-Dynamic-Obstacles-8x8-v0"
ENV_TAG = "minigrid_dynamic_obstacles_8x8_v0"
RESULTS_DIR = f"experiments/results/{ENV_TAG}"
EVAL_SEEDS = list(range(1000, 1050))
SUCCESS_THRESHOLD = 0.5  # reward > 0 counts as success
HELDOUT_SEED = 99
SEED_SHIFT_SEEDS = [0, 1, 2, 3, 4]
N_SUBSAMPLES = 5
SUBSAMPLE_FRACTION = 0.8
CLUSTER_DELTAS = [-1, 0, 1]
NOISE_LEVELS = [0.01, 0.03, 0.05]
# Noise mask: only continuous features (skip binary flags at indices 6,7,8,9,13)
NOISE_MASK = [0, 1, 2, 3, 4, 5, 10, 11, 12]

FEATURE_NAMES = ENV_FEATURE_NAMES[ENV_NAME]


def get_model_path(algo="ppo", seed=42):
    return f"reproduction/models/{algo}_{ENV_TAG}.zip"


def get_replay_path(algo="ppo", seed=42):
    return f"reproduction/data/replay_{ENV_TAG}_{algo}_seed{seed}.npz"


def _make_env():
    from reproduction.minigrid_feature_wrapper import make_minigrid_env
    return make_minigrid_env(ENV_NAME)


def _load_model(model_path, algo="ppo"):
    if algo == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(model_path)
    else:
        from stable_baselines3 import DQN
        return DQN.load(model_path)


def collect_heldout(model_path, algo="ppo", n_transitions=5000):
    """Collect held-out replay for fidelity evaluation."""
    data = collect_replay_minigrid(
        model_path, n_transitions=n_transitions, seed=HELDOUT_SEED, algo=algo,
    )
    return data["states"], data["actions"]


def collect_replay_minigrid(model_path, n_transitions=10000, seed=42, algo="ppo"):
    """Collect replay data from a trained MiniGrid policy."""
    env = _make_env()
    model = _load_model(model_path, algo)

    states, actions, rewards, dones, episode_ids = [], [], [], [], []
    episode_count = 0
    total_reward = 0.0
    collected = 0
    successes = 0

    obs, info = env.reset(seed=seed)

    while collected < n_transitions:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        states.append(obs.copy())
        actions.append(action)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rewards.append(reward)
        dones.append(done)
        episode_ids.append(episode_count)
        total_reward += reward
        collected += 1

        if reward > 0:
            successes += 1

        if done:
            episode_count += 1
            obs, info = env.reset(seed=seed + episode_count)
        else:
            obs = next_obs

    env.close()

    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int32),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=bool),
        "episode_ids": np.array(episode_ids, dtype=np.int32),
        "num_episodes": episode_count + (1 if not dones[-1] else 0),
        "total_reward": total_reward,
        "successes": successes,
    }


def run_cbs_on_data(states, actions, kmeans_seed=0, delta=0):
    """Fit CBS and return pipeline + canonical rules."""
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


def evaluate_cbs_in_env(cbs, n_episodes=50, seeds=None):
    """Deploy CBS rules in the MiniGrid environment."""
    if seeds is None:
        seeds = EVAL_SEEDS
    env = _make_env()

    rewards_list = []
    lengths = []
    successes = 0

    for ep_seed in seeds:
        obs, _ = env.reset(seed=ep_seed)
        done = False
        ep_reward = 0.0
        ep_len = 0
        while not done:
            action = cbs.predict(obs.reshape(1, -1))[0]
            obs, reward, terminated, truncated, _ = env.step(int(action))
            ep_reward += reward
            ep_len += 1
            done = terminated or truncated
        rewards_list.append(ep_reward)
        lengths.append(ep_len)
        if ep_reward > 0:
            successes += 1

    env.close()

    return {
        "E_CR": float(np.mean(rewards_list)),
        "E_CR_std": float(np.std(rewards_list)),
        "E_TS": float(np.mean(lengths)),
        "success_rate": successes / len(seeds),
    }


def evaluate_dt_in_env(dt, n_episodes=50, seeds=None):
    """Deploy DT surrogate in the MiniGrid environment."""
    if seeds is None:
        seeds = EVAL_SEEDS
    env = _make_env()

    rewards_list = []
    lengths = []
    successes = 0

    for ep_seed in seeds:
        obs, _ = env.reset(seed=ep_seed)
        done = False
        ep_reward = 0.0
        ep_len = 0
        while not done:
            action = dt.predict(obs.reshape(1, -1))[0]
            obs, reward, terminated, truncated, _ = env.step(int(action))
            ep_reward += reward
            ep_len += 1
            done = terminated or truncated
        rewards_list.append(ep_reward)
        lengths.append(ep_len)
        if ep_reward > 0:
            successes += 1

    env.close()

    return {
        "E_CR": float(np.mean(rewards_list)),
        "E_CR_std": float(np.std(rewards_list)),
        "E_TS": float(np.mean(lengths)),
        "success_rate": successes / len(seeds),
    }


def evaluate_single_run(pipeline, rules, heldout_s, heldout_a, is_dt=False):
    """Evaluate a CBS or DT run."""
    fid = pipeline.evaluate_fidelity(heldout_s, heldout_a)
    fid_pa = pipeline.evaluate_fidelity_per_action(heldout_s, heldout_a)

    if is_dt:
        deploy = evaluate_dt_in_env(pipeline)
    else:
        deploy = evaluate_cbs_in_env(pipeline)

    props = pipeline.evaluate_properties()
    cov = pipeline.evaluate_coverage(heldout_s)

    result = {
        "fidelity_heldout": fid,
        "fidelity_per_action": fid_pa,
        "deployment": deploy,
        "properties": props,
        "coverage": cov,
        "n_rules": len(rules),
    }

    if hasattr(pipeline, 'get_thresholds'):
        result["thresholds"] = {int(k): [float(v) for v in vs]
                                for k, vs in pipeline.get_thresholds().items()}
    return result


def compute_bra_from_predictions(pred_list):
    n = len(pred_list)
    if n < 2:
        return 1.0
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.mean(pred_list[i] == pred_list[j])
            count += 1
    return total / count


def add_feature_noise_masked(data, noise_level, seed=42, feature_ranges=None):
    """Add Gaussian noise only to continuous features (respecting noise mask)."""
    rng = np.random.RandomState(seed)
    noisy_states = data["states"].copy()

    for fidx in NOISE_MASK:
        if feature_ranges is not None and fidx in feature_ranges:
            sigma = noise_level * feature_ranges[fidx]
        else:
            col = noisy_states[:, fidx]
            sigma = noise_level * (col.max() - col.min() + 1e-8)
        noise = rng.normal(0, sigma, size=noisy_states.shape[0])
        noisy_states[:, fidx] += noise

    return {
        "states": noisy_states,
        "actions": data["actions"],
    }


# ═════════════════════════════════════════════════════════════════════
# Step A: Feasibility Check
# ═════════════════════════════════════════════════════════════════════

def run_feasibility_check(algo="ppo"):
    """Run CBS, rule-set voting, DT on reference replay to verify pipeline works."""
    print(f"\n{'='*60}")
    print(f"  Feasibility Check: {ENV_NAME} ({algo.upper()})")
    print(f"{'='*60}")

    model_path = get_model_path(algo)
    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found at {model_path}")
        return None

    # Load or collect reference replay
    replay_path = get_replay_path(algo)
    if os.path.exists(replay_path):
        print(f"  Loading existing replay: {replay_path}")
        ref_data = load_replay_npz(replay_path)
    else:
        print(f"  Collecting reference replay (10k transitions)...")
        ref_data = collect_replay_minigrid(model_path, n_transitions=10000,
                                            seed=42, algo=algo)
        # Save
        os.makedirs("reproduction/data", exist_ok=True)
        np.savez_compressed(
            replay_path,
            states=ref_data["states"],
            actions=ref_data["actions"],
            rewards=ref_data["rewards"],
            dones=ref_data["dones"],
            episode_ids=ref_data["episode_ids"],
        )
        print_summary(ref_data, ENV_NAME, 42)

    # Collect held-out
    print(f"  Collecting held-out replay (5k transitions)...")
    heldout_s, heldout_a = collect_heldout(model_path, algo)

    feasibility = {"env": ENV_NAME, "algo": algo, "methods": {}}

    # --- CBS ---
    print(f"\n  --- CBS ---")
    try:
        cbs, cbs_rules = run_cbs_on_data(ref_data["states"], ref_data["actions"])
        cbs_res = evaluate_single_run(cbs, cbs_rules, heldout_s, heldout_a)
        feasibility["methods"]["CBS"] = {
            "status": "OK",
            "F1": cbs_res["fidelity_heldout"]["f1"],
            "accuracy": cbs_res["fidelity_heldout"]["accuracy"],
            "n_rules": cbs_res["n_rules"],
            "deployment": cbs_res["deployment"],
            "coverage": cbs_res["coverage"],
            "fidelity_per_action": cbs_res["fidelity_per_action"],
        }
        print(f"    F1={cbs_res['fidelity_heldout']['f1']:.3f}, "
              f"rules={cbs_res['n_rules']}, "
              f"E_CR={cbs_res['deployment']['E_CR']:.3f}, "
              f"success_rate={cbs_res['deployment']['success_rate']:.2%}")
    except Exception as e:
        feasibility["methods"]["CBS"] = {"status": "FAILED", "error": str(e)}
        print(f"    FAILED: {e}")

    # --- rule-set voting ---
    print(f"\n  --- rule-set voting ---")
    try:
        from experiments.consensus_merge import build_voting_ensemble, voting_predict
        subsamples = generate_subsamples(ref_data, 5, 0.8, seed=42)
        pipelines = []
        for ss in subsamples:
            p, _ = run_cbs_on_data(ss["states"], ss["actions"])
            pipelines.append(p)
        # Evaluate rule-set voting fidelity
        vote_preds = voting_predict(pipelines, heldout_s)
        from sklearn.metrics import accuracy_score, f1_score
        vote_f1 = f1_score(heldout_a, vote_preds, average="macro")
        vote_acc = accuracy_score(heldout_a, vote_preds)
        feasibility["methods"]["B3-vote"] = {
            "status": "OK",
            "F1": float(vote_f1),
            "accuracy": float(vote_acc),
            "n_pipelines": len(pipelines),
        }
        print(f"    F1={vote_f1:.3f}, accuracy={vote_acc:.3f}")
    except Exception as e:
        feasibility["methods"]["B3-vote"] = {"status": "FAILED", "error": str(e)}
        print(f"    FAILED: {e}")

    # --- DT Surrogate ---
    print(f"\n  --- DT Surrogate ---")
    try:
        dt = DecisionTreeSurrogate(feature_names=FEATURE_NAMES)
        dt.fit(ref_data["states"], ref_data["actions"])
        dt_rules = canonicalize_rules(dt.get_rules())
        dt_res = evaluate_single_run(dt, dt_rules, heldout_s, heldout_a, is_dt=True)
        feasibility["methods"]["DT Surrogate"] = {
            "status": "OK",
            "F1": dt_res["fidelity_heldout"]["f1"],
            "accuracy": dt_res["fidelity_heldout"]["accuracy"],
            "n_rules": dt_res["n_rules"],
            "deployment": dt_res["deployment"],
        }
        print(f"    F1={dt_res['fidelity_heldout']['f1']:.3f}, "
              f"rules={dt_res['n_rules']}, "
              f"E_CR={dt_res['deployment']['E_CR']:.3f}")
    except Exception as e:
        feasibility["methods"]["DT Surrogate"] = {"status": "FAILED", "error": str(e)}
        print(f"    FAILED: {e}")

    # Save feasibility results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    feas_path = os.path.join(RESULTS_DIR, "feasibility_check.json")
    with open(feas_path, "w") as f:
        json.dump(feasibility, f, indent=2)
    print(f"\n  Feasibility results saved to {feas_path}")

    return feasibility


# ═════════════════════════════════════════════════════════════════════
# Step B: Compressed Stress Suite
# ═════════════════════════════════════════════════════════════════════

def run_stress_suite(algo="ppo"):
    """Run compressed stress suite on MiniGrid."""
    print(f"\n{'='*60}")
    print(f"  Stress Suite: {ENV_NAME} ({algo.upper()})")
    print(f"{'='*60}")

    model_path = get_model_path(algo)
    replay_path = get_replay_path(algo)

    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found at {model_path}")
        return None
    if not os.path.exists(replay_path):
        print(f"  ERROR: Replay not found at {replay_path}")
        return None

    ref_data = load_replay_npz(replay_path)
    print(f"  Reference replay: {len(ref_data['states'])} transitions")

    print(f"  Collecting held-out replay...")
    heldout_s, heldout_a = collect_heldout(model_path, algo)
    print(f"  Held-out replay: {len(heldout_s)} transitions")

    anchor_hash = hashlib.sha256(heldout_s.tobytes()).hexdigest()
    feature_ranges = compute_feature_ranges(ref_data)
    fr_dict = {i: float(feature_ranges[i]) for i in range(len(feature_ranges))}

    t0 = time.time()

    all_results = {}

    for method_name, run_method in [("CBS", "cbs"), ("DT", "dt")]:
        print(f"\n  === Method: {method_name} ===")
        run_entries = []
        results = {}

        def _record(run_key, family, params, pipeline, rules, n_replay, is_dt=False):
            res = evaluate_single_run(pipeline, rules, heldout_s, heldout_a,
                                       is_dt=is_dt)
            preds = pipeline.predict(heldout_s)
            thresholds = {}
            if hasattr(pipeline, 'get_thresholds'):
                thresholds = {int(k): [float(v) for v in vs]
                              for k, vs in pipeline.get_thresholds().items()}

            res["run_id"] = f"{ENV_TAG}_{run_method}_{run_key}"
            res["group_id"] = f"{ENV_TAG}_{run_method}"
            res["method"] = method_name
            res["algo"] = algo
            res["perturbation_family"] = family
            res["perturbation_id"] = run_key
            res["perturbation_params"] = params
            res["n_replay"] = n_replay
            res["n_heldout"] = len(heldout_s)
            res["rules"] = serialize_canonical_rules(rules)
            results[run_key] = res
            run_entries.append({
                "key": run_key,
                "family": family,
                "rules": rules,
                "thresholds": thresholds,
                "preds": preds,
            })

        # Seed shift
        print(f"  [{method_name}] seed_shift ×{len(SEED_SHIFT_SEEDS)}...")
        for seed in SEED_SHIFT_SEEDS:
            data = collect_replay_minigrid(model_path, n_transitions=10000,
                                            seed=seed, algo=algo)
            if run_method == "cbs":
                p, r = run_cbs_on_data(data["states"], data["actions"])
                _record(f"seed_shift_s{seed}", "seed_shift",
                        {"seed": seed}, p, r, len(data["states"]))
            else:
                dt = DecisionTreeSurrogate(feature_names=FEATURE_NAMES)
                dt.fit(data["states"], data["actions"])
                r = canonicalize_rules(dt.get_rules())
                _record(f"seed_shift_s{seed}", "seed_shift",
                        {"seed": seed}, dt, r, len(data["states"]), is_dt=True)

        # Subsampling (stratified)
        print(f"  [{method_name}] stratified_subsample ×{N_SUBSAMPLES}...")
        strat = generate_stratified_subsamples(ref_data, N_SUBSAMPLES,
                                                SUBSAMPLE_FRACTION, seed=42)
        for i, ss in enumerate(strat):
            if run_method == "cbs":
                p, r = run_cbs_on_data(ss["states"], ss["actions"])
                _record(f"stratified_{i}", "stratified_subsample",
                        {"idx": i}, p, r, len(ss["states"]))
            else:
                dt = DecisionTreeSurrogate(feature_names=FEATURE_NAMES)
                dt.fit(ss["states"], ss["actions"])
                r = canonicalize_rules(dt.get_rules())
                _record(f"stratified_{i}", "stratified_subsample",
                        {"idx": i}, dt, r, len(ss["states"]), is_dt=True)

        # Cluster count (CBS only) / Depth variation (DT only)
        if run_method == "cbs":
            print(f"  [{method_name}] cluster_count ×{len(CLUSTER_DELTAS)}...")
            for delta in CLUSTER_DELTAS:
                p, r = run_cbs_on_data(ref_data["states"], ref_data["actions"],
                                        delta=delta)
                _record(f"cluster_delta_{delta:+d}", "cluster_count",
                        {"delta": delta}, p, r, len(ref_data["states"]))
        else:
            print(f"  [{method_name}] depth_variation ×3...")
            for depth in [3, 5, None]:
                dt = DecisionTreeSurrogate(
                    max_depth=depth, feature_names=FEATURE_NAMES)
                dt.fit(ref_data["states"], ref_data["actions"])
                r = canonicalize_rules(dt.get_rules())
                _record(f"depth_{depth}", "depth_variation",
                        {"max_depth": depth}, dt, r,
                        len(ref_data["states"]), is_dt=True)

        # Feature noise (masked)
        print(f"  [{method_name}] feature_noise ×{len(NOISE_LEVELS)}...")
        for nl in NOISE_LEVELS:
            noisy = add_feature_noise_masked(ref_data, nl, seed=42,
                                              feature_ranges=fr_dict)
            if run_method == "cbs":
                p, r = run_cbs_on_data(noisy["states"], noisy["actions"])
                _record(f"noise_{nl:.3f}", "feature_noise",
                        {"noise_level": nl}, p, r, len(noisy["states"]))
            else:
                dt = DecisionTreeSurrogate(feature_names=FEATURE_NAMES)
                dt.fit(noisy["states"], noisy["actions"])
                r = canonicalize_rules(dt.get_rules())
                _record(f"noise_{nl:.3f}", "feature_noise",
                        {"noise_level": nl}, dt, r,
                        len(noisy["states"]), is_dt=True)

        # Compute stability metrics
        print(f"  [{method_name}] Computing stability metrics...")
        all_rule_sets = [e["rules"] for e in run_entries]
        all_preds = [e["preds"] for e in run_entries]

        grs_wj = mean_pairwise_jaccard(all_rule_sets, weighted=True)
        grs_ta = mean_pairwise_soft_jaccard(all_rule_sets, threshold_aware=True)
        td_sets = [e["thresholds"] for e in run_entries]
        td_val = mean_pairwise_threshold_drift(td_sets, feature_ranges=fr_dict)
        bra = compute_bra_from_predictions(all_preds)

        stability = {
            "GRS_weighted_jaccard": float(grs_wj),
            "GRS_threshold_aware": float(grs_ta),
            "TD": float(td_val),
            "BRA": float(bra),
            "n_runs": len(run_entries),
        }

        all_results[method_name] = {
            "runs": results,
            "stability": stability,
        }

        print(f"    GRS_wj={grs_wj:.3f}, GRS_ta={grs_ta:.3f}, "
              f"TD={td_val:.3f}, BRA={bra:.3f}")

    # --- rule-set voting stress suite ---
    print(f"\n  === Method: rule-set voting ===")
    from experiments.consensus_merge import build_voting_ensemble, voting_predict
    from sklearn.metrics import f1_score, accuracy_score

    b3_run_entries = []
    b3_results = {}

    def _run_vote(run_key, family, params, replay_data, n_replay):
        # Ensure the data dict has all keys expected by generate_subsamples
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
        # Deployment
        deploy_r, deploy_l, deploy_succ = [], [], 0
        env = _make_env()
        for ep_seed in EVAL_SEEDS[:10]:  # Reduced for speed
            obs, _ = env.reset(seed=ep_seed)
            done = False
            ep_reward, ep_len = 0.0, 0
            while not done:
                a = voting_predict(pipelines, obs.reshape(1, -1))[0]
                obs, reward, term, trunc, _ = env.step(int(a))
                ep_reward += reward
                ep_len += 1
                done = term or trunc
            deploy_r.append(ep_reward)
            deploy_l.append(ep_len)
            if ep_reward > 0:
                deploy_succ += 1
        env.close()

        res = {
            "fidelity_heldout": {"f1": float(f1), "accuracy": float(acc)},
            "deployment": {
                "E_CR": float(np.mean(deploy_r)),
                "E_CR_std": float(np.std(deploy_r)),
                "success_rate": deploy_succ / len(EVAL_SEEDS[:10]),
            },
            "n_rules": sum(len(p.get_rules()) for p in pipelines),
            "run_id": f"{ENV_TAG}_b3vote_{run_key}",
            "method": "B3-vote",
            "algo": algo,
            "perturbation_family": family,
            "perturbation_id": run_key,
            "perturbation_params": params,
            "n_replay": n_replay,
        }
        # Collect all individual pipeline rules for stability
        all_indiv_rules = []
        for p in pipelines:
            all_indiv_rules.extend(canonicalize_rules(p.get_rules()))

        b3_results[run_key] = res
        b3_run_entries.append({
            "key": run_key,
            "family": family,
            "rules": all_indiv_rules,
            "thresholds": {},
            "preds": preds,
        })

    # Seed shift
    for seed in SEED_SHIFT_SEEDS:
        data = collect_replay_minigrid(model_path, n_transitions=10000,
                                        seed=seed, algo=algo)
        _run_vote(f"seed_shift_s{seed}", "seed_shift",
                    {"seed": seed}, data, len(data["states"]))

    # Stratified subsample
    strat = generate_stratified_subsamples(ref_data, N_SUBSAMPLES,
                                            SUBSAMPLE_FRACTION, seed=42)
    for i, ss in enumerate(strat):
        _run_vote(f"stratified_{i}", "stratified_subsample",
                    {"idx": i}, ss, len(ss["states"]))

    # Cluster count - use different seeds for CBS inside rule-set voting
    for delta in CLUSTER_DELTAS:
        _run_vote(f"cluster_delta_{delta:+d}", "cluster_count",
                    {"delta": delta}, ref_data, len(ref_data["states"]))

    # Feature noise
    for nl in NOISE_LEVELS:
        noisy = add_feature_noise_masked(ref_data, nl, seed=42,
                                          feature_ranges=fr_dict)
        _run_vote(f"noise_{nl:.3f}", "feature_noise",
                    {"noise_level": nl}, noisy, len(noisy["states"]))

    # rule-set voting stability
    b3_preds = [e["preds"] for e in b3_run_entries]
    b3_bra = compute_bra_from_predictions(b3_preds)
    all_results["B3-vote"] = {
        "runs": b3_results,
        "stability": {
            "BRA": float(b3_bra),
            "n_runs": len(b3_run_entries),
        },
    }
    print(f"    rule-set voting BRA={b3_bra:.3f}")

    elapsed = time.time() - t0

    # Save all results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "env": ENV_NAME,
        "algo": algo,
        "anchor_hash": anchor_hash,
        "elapsed_seconds": elapsed,
        "methods": all_results,
    }
    out_path = os.path.join(RESULTS_DIR, "stress_test_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print(f"  Total time: {elapsed:.1f}s")

    return output


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MiniGrid experiments")
    parser.add_argument("--step", choices=["feasibility", "stress", "all"],
                        default="all")
    parser.add_argument("--algo", default="ppo")
    args = parser.parse_args()

    if args.step in ("feasibility", "all"):
        run_feasibility_check(args.algo)

    if args.step in ("stress", "all"):
        run_stress_suite(args.algo)


if __name__ == "__main__":
    main()
