#!/usr/bin/env python
"""
Cross-Policy Replication (Minimal)

Train 2 additional DQN policies per environment (seeds 1, 2) for MC + LL,
then run compressed experiment suite to verify qualitative patterns hold.

Usage:
    python experiments/run_cross_policy.py --env MountainCar-v0
    python experiments/run_cross_policy.py --env LunarLander-v3
    python experiments/run_cross_policy.py --env all
    python experiments/run_cross_policy.py --train-only  # just train models

Output:
    experiments/results/<env>/cross_policy_results.json
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
from experiments.rule_matching import (
    canonicalize_rules,
    mean_pairwise_jaccard,
)
from experiments.perturbations import compute_feature_ranges
from experiments.decision_tree_surrogate import DecisionTreeSurrogate, find_best_depth
from experiments.consensus_merge import voting_predict

# ── Constants ─────────────────────────────────────────────────────────
ENVS = ["MountainCar-v0", "LunarLander-v3", "CartPole-v1"]
POLICY_SEEDS = [1, 2]  # additional seeds beyond existing seed=42
ALL_POLICY_SEEDS = [42, 1, 2]
SEED_SHIFT_SEEDS = [0, 1, 2, 3, 4]
HELDOUT_SEED = 99
EVAL_SEEDS = list(range(1000, 1050))
SUCCESS_THRESHOLDS = {"MountainCar-v0": -150.0, "LunarLander-v3": 200.0, "CartPole-v1": 475.0}
RESULTS_DIR = "experiments/results"


def get_model_path(env_name, policy_seed=42):
    tag = env_name.replace("-", "_").lower()
    if policy_seed == 42:
        return f"reproduction/models/dqn_{tag}.zip"
    return f"reproduction/models/dqn_{tag}_seed{policy_seed}.zip"


def get_ncat(env_name):
    return 6 if env_name == "LunarLander-v3" else 5


def train_policy(env_name, seed):
    """Train a DQN policy with given seed."""
    from reproduction.train_dqn import train_dqn, DEFAULT_TIMESTEPS
    model_path = get_model_path(env_name, seed)
    if os.path.exists(model_path):
        print(f"    Model already exists: {model_path}")
        return model_path

    timesteps = DEFAULT_TIMESTEPS.get(env_name, 300000)
    print(f"    Training {env_name} seed={seed} ({timesteps} steps)...")
    # train_dqn saves to dqn_{tag}.zip by default; we need seed-specific names
    tag = env_name.replace("-", "_").lower()
    default_save_path = os.path.join("reproduction/models", f"dqn_{tag}.zip")
    # Back up existing model if it exists
    backup_path = default_save_path + ".backup"
    if os.path.exists(default_save_path):
        import shutil
        shutil.copy2(default_save_path, backup_path)
    try:
        train_dqn(env_name, total_timesteps=timesteps, seed=seed,
                  model_dir="reproduction/models", verbose=0)
        # Copy to seed-specific name
        if os.path.exists(default_save_path):
            import shutil
            shutil.copy2(default_save_path, model_path)
            print(f"    Saved to {model_path}")
    finally:
        # Restore original model
        if os.path.exists(backup_path):
            import shutil
            shutil.copy2(backup_path, default_save_path)
            os.remove(backup_path)
    return model_path


def run_compressed_suite(env_name, model_path, policy_seed, heldout_s, heldout_a):
    """Run compressed experiment suite for one policy.

    Methods: CBS, CBS+MaxF1, rule-set voting, DT
    Perturbation: seed_shift (5 seeds)
    """
    ncat = get_ncat(env_name)
    results = {"cbs": [], "cbs_maxf1": [], "b3_vote": [], "dt": []}

    for replay_seed in SEED_SHIFT_SEEDS:
        data = collect_replay(env_name, model_path, num_transitions=10000, seed=replay_seed)
        states, actions = data["states"], data["actions"]

        # CBS
        cbs = CBSPipeline(n_categories=ncat, inclusion_threshold=0.70, kmeans_seed=0)
        cbs.fit(states, actions)
        fid = cbs.evaluate_fidelity(heldout_s, heldout_a)
        preds = cbs.predict(heldout_s)
        rules = canonicalize_rules(cbs.rules_)
        results["cbs"].append({"f1": float(fid["f1"]), "rules": rules, "preds": preds})

        # CBS + MaxF1
        cbs_mf = CBSPipeline(n_categories=ncat, inclusion_threshold=0.70, kmeans_seed=0)
        cbs_mf.fit(states, actions)
        cbs_mf.refine_max_f1(states, actions)
        fid_mf = cbs_mf.evaluate_fidelity(heldout_s, heldout_a)
        preds_mf = cbs_mf.predict(heldout_s)
        rules_mf = canonicalize_rules(cbs_mf.rules_)
        results["cbs_maxf1"].append({"f1": float(fid_mf["f1"]), "rules": rules_mf, "preds": preds_mf})

        # rule-set voting
        from experiments.perturbations import generate_subsamples
        sub_data = {"states": states, "actions": actions,
                    "rewards": np.zeros(len(actions)), "dones": np.zeros(len(actions), dtype=bool),
                    "episode_ids": np.zeros(len(actions), dtype=int)}
        subsamples = generate_subsamples(sub_data, n_subsets=5, fraction=0.8, seed=42)
        pipelines = []
        for ss in subsamples:
            p = CBSPipeline(n_categories=ncat, inclusion_threshold=0.70, kmeans_seed=0)
            p.fit(ss["states"], ss["actions"])
            pipelines.append(p)
        vote_preds = voting_predict(pipelines, heldout_s)
        vote_f1 = float(np.mean(vote_preds == heldout_a))
        results["b3_vote"].append({"f1": vote_f1, "preds": vote_preds})

        # DT
        best_depth, _ = find_best_depth(states, actions)
        dt = DecisionTreeSurrogate(max_depth=best_depth)
        dt.fit(states, actions)
        fid_dt = dt.evaluate_fidelity(heldout_s, heldout_a)
        preds_dt = dt.predict(heldout_s)
        from experiments.decision_tree_surrogate import canonicalize_dt_rules
        rules_dt = canonicalize_dt_rules(dt.get_rules())
        results["dt"].append({"f1": float(fid_dt["f1"]), "rules": rules_dt, "preds": preds_dt})

    # Compute stability across seed_shift runs
    summary = {}
    for method in ["cbs", "cbs_maxf1", "b3_vote", "dt"]:
        f1_vals = [r["f1"] for r in results[method]]
        pred_sets = [r["preds"] for r in results[method]]

        # BRA from prediction arrays
        n = len(pred_sets)
        bra_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                bra_pairs.append(float(np.mean(pred_sets[i] == pred_sets[j])))
        bra = float(np.mean(bra_pairs)) if bra_pairs else 1.0

        method_summary = {
            "mean_f1": float(np.mean(f1_vals)),
            "std_f1": float(np.std(f1_vals)),
            "BRA": float(bra),
        }

        if "rules" in results[method][0]:
            rule_sets = [r["rules"] for r in results[method]]
            grs = mean_pairwise_jaccard(rule_sets, weighted=True)
            method_summary["GRS_wj"] = float(grs)

        summary[method] = method_summary

    return summary


def check_pattern_replication(per_policy_summaries):
    """Check if qualitative patterns replicate across policies."""
    patterns = {}

    # Pattern 1: MaxF1 hurts stability (CBS GRS > MaxF1 GRS)
    checks = []
    for ps, data in per_policy_summaries.items():
        cbs_grs = data.get("cbs", {}).get("GRS_wj", 0)
        mf1_grs = data.get("cbs_maxf1", {}).get("GRS_wj", 0)
        checks.append(cbs_grs > mf1_grs)
    patterns["maxf1_hurts_stability"] = {"pass": all(checks), "checks": checks}

    # Pattern 2: DT has low GRS but high BRA
    checks_grs = []
    checks_bra = []
    for ps, data in per_policy_summaries.items():
        dt_grs = data.get("dt", {}).get("GRS_wj", 1)
        cbs_grs = data.get("cbs", {}).get("GRS_wj", 0)
        dt_bra = data.get("dt", {}).get("BRA", 0)
        cbs_bra = data.get("cbs", {}).get("BRA", 1)
        checks_grs.append(dt_grs < cbs_grs)
        checks_bra.append(dt_bra > cbs_bra)
    patterns["dt_low_grs_high_bra"] = {
        "pass": all(checks_grs) and all(checks_bra),
        "grs_checks": checks_grs, "bra_checks": checks_bra
    }

    # Pattern 3: rule-set voting improves BRA
    checks = []
    for ps, data in per_policy_summaries.items():
        vote_bra = data.get("b3_vote", {}).get("BRA", 0)
        cbs_bra = data.get("cbs", {}).get("BRA", 0)
        checks.append(vote_bra >= cbs_bra)
    patterns["b3vote_improves_bra"] = {"pass": all(checks), "checks": checks}

    return patterns


def run_cross_policy(env_name, skip_training=False):
    """Run cross-policy replication for one environment."""
    print(f"\n{'='*60}")
    print(f"Cross-Policy Replication: {env_name}")
    print(f"{'='*60}")
    t0 = time.time()

    # Step 1: Ensure all policies exist
    if not skip_training:
        print("\n  Step 1: Ensure policies trained")
        for seed in POLICY_SEEDS:
            try:
                train_policy(env_name, seed)
            except Exception as e:
                print(f"    WARNING: Failed to train seed {seed}: {e}")

    # Step 2: Run compressed suite per policy
    per_policy = {}
    for policy_seed in ALL_POLICY_SEEDS:
        model_path = get_model_path(env_name, policy_seed)
        if not os.path.exists(model_path):
            print(f"  Skipping seed {policy_seed} — model not found")
            continue

        print(f"\n  Policy seed={policy_seed}")

        # Collect held-out (same for all policies — uses held-out seed)
        heldout = collect_replay(env_name, model_path, num_transitions=5000, seed=HELDOUT_SEED)
        heldout_s, heldout_a = heldout["states"], heldout["actions"]

        # Evaluate policy quality
        from stable_baselines3 import DQN
        import gymnasium as gym
        model = DQN.load(model_path)
        env = gym.make(env_name)
        from stable_baselines3.common.evaluation import evaluate_policy
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
        env.close()
        print(f"    Policy return: {mean_reward:.1f} +/- {std_reward:.1f}")

        threshold = SUCCESS_THRESHOLDS.get(env_name, -999)
        if mean_reward < threshold:
            print(f"    WARNING: Policy below solve threshold ({threshold})")

        # Run compressed suite
        summary = run_compressed_suite(env_name, model_path, policy_seed, heldout_s, heldout_a)
        per_policy[str(policy_seed)] = {
            "policy_return": float(mean_reward),
            "policy_std": float(std_reward),
            "methods": summary,
        }

        for method, ms in summary.items():
            print(f"    {method:12s}: F1={ms['mean_f1']:.3f}, "
                  f"BRA={ms['BRA']:.3f}"
                  + (f", GRS={ms['GRS_wj']:.3f}" if 'GRS_wj' in ms else ""))

    # Step 3: Check pattern replication
    print(f"\n  Pattern Replication Check:")
    all_summaries = {k: v["methods"] for k, v in per_policy.items()}
    patterns = check_pattern_replication(all_summaries)
    for pname, pcheck in patterns.items():
        status = "PASS" if pcheck["pass"] else "FAIL"
        print(f"    [{status}] {pname}")

    elapsed = time.time() - t0
    tag = env_name.replace("-", "_").lower()

    output = {
        "schema_version": "cross_policy_v1",
        "env": env_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "policy_seeds": ALL_POLICY_SEEDS,
        "per_policy": per_policy,
        "pattern_replication": {k: {"pass": v["pass"]} for k, v in patterns.items()},
    }

    out_dir = os.path.join(RESULTS_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cross_policy_results.json")

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


def main():
    parser = argparse.ArgumentParser(description="Cross-policy replication")
    parser.add_argument("--env", default="all",
                        choices=["MountainCar-v0", "LunarLander-v3", "CartPole-v1", "all"])
    parser.add_argument("--train-only", action="store_true",
                        help="Only train models, don't run experiments")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, use existing models only")
    args = parser.parse_args()

    envs = ENVS if args.env == "all" else [args.env]

    if args.train_only:
        for env_name in envs:
            for seed in POLICY_SEEDS:
                train_policy(env_name, seed)
        return

    for env_name in envs:
        run_cross_policy(env_name, skip_training=args.skip_training)


if __name__ == "__main__":
    main()
