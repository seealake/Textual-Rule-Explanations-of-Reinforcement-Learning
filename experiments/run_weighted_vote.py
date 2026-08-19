#!/usr/bin/env python
"""
Weighted rule-set voting: main comparison and ensemble-size sensitivity

Main comparison (4 envs: MC, CP, LL, MiniGrid)
    Methods: CBS / rule-set voting / weighted rule-set voting variants / soft support or tuned merge
    Metrics: F1, BRA, worst-R, LEC (max radius), n_rules, active-voter cost
    10 outer repeats, paired bootstrap tests

Ensemble-size sensitivity (CP + LL only)
    B ∈ {3, 5, 10}, vanilla vs best weighted
    Output: BRA-vs-B and worst-R-vs-B

Weighted variants (6 configs):
    weight_type ∈ {f1, worst_r, hybrid_05, hybrid_07}
    beta ∈ {1, 3}
    → but hybrid has two alpha values, so:
      f1/β1, f1/β3, worst_r/β1, worst_r/β3, hybrid_05/β1, hybrid_05/β3,
      hybrid_07/β1, hybrid_07/β3
    Total: 8 weighted configs (3 types × 2 betas, hybrid counted twice for alpha)

Calibration: separate replay (seed=77) for weight computation.
Evaluation: held-out (seed=99).

Usage:
    python experiments/run_weighted_vote.py --env MountainCar-v0
    python experiments/run_weighted_vote.py --env CartPole-v1
    python experiments/run_weighted_vote.py --env LunarLander-v3
    python experiments/run_weighted_vote.py --env MiniGrid-Dynamic-Obstacles-8x8-v0
    python experiments/run_weighted_vote.py --env all
    python experiments/run_weighted_vote.py --env CartPole-v1 --b-sensitivity
    python experiments/run_weighted_vote.py --env all --b-sensitivity
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from reproduction.collect_replay import collect_replay, ENV_FEATURE_NAMES
from experiments.perturbations import compute_feature_ranges
from experiments.consensus_merge import (
    build_voting_ensemble,
    voting_predict,
)
from experiments.weighted_voting import (
    compute_voter_weights,
    weighted_voting_predict,
    topk_weighted_voting_predict,
    compute_voter_cost_metrics,
)
from experiments.lec import compute_lec_prediction_based
from experiments.run_stress_test import (
    run_cbs_on_data,
    evaluate_single_run,
    compute_bra_from_predictions,
    EVAL_SEEDS,
    SUCCESS_THRESHOLDS,
    HELDOUT_SEED,
)

# ── Configuration ────────────────────────────────────────────────────

MAIN_ENVS = ["MountainCar-v0", "CartPole-v1", "LunarLander-v3",
             "MiniGrid-Dynamic-Obstacles-8x8-v0"]
B_SENS_ENVS = ["CartPole-v1", "LunarLander-v3"]

N_OUTER_REPEATS = 10
OUTER_SEEDS = list(range(N_OUTER_REPEATS))
CALIBRATION_SEED = 77  # separate from training (42) and held-out (99)
DEFAULT_B = 5
LEC_EPSILONS = (0.01, 0.03, 0.05)

OUT_ROOT = "experiments/results/weighted_vote"
LEC_N_STATES = 500  # subsample held-out for LEC to keep runtime reasonable

# Weighted configs: (tag, weight_type, alpha, beta)
WEIGHTED_CONFIGS = [
    ("f1_b1",      "f1",      None, 1.0),
    ("f1_b3",      "f1",      None, 3.0),
    ("worstR_b1",  "worst_r", None, 1.0),
    ("worstR_b3",  "worst_r", None, 3.0),
    ("hybrid05_b1","hybrid",  0.5,  1.0),
    ("hybrid05_b3","hybrid",  0.5,  3.0),
    ("hybrid07_b1","hybrid",  0.7,  1.0),
    ("hybrid07_b3","hybrid",  0.7,  3.0),
]

# B sensitivity grid
B_VALUES = [3, 5, 10]


def get_model_path(env_name):
    if env_name == "MiniGrid-Dynamic-Obstacles-8x8-v0":
        return "reproduction/models/ppo_minigrid_dynamic_obstacles_8x8_v0.zip"
    tag = env_name.replace("-", "_").lower()
    return f"reproduction/models/dqn_{tag}.zip"


def ci95(arr):
    a = np.array(arr, dtype=float)
    if len(a) < 2:
        return (float(a[0]), float(a[0])) if len(a) == 1 else (0.0, 0.0)
    se = a.std(ddof=1) / np.sqrt(len(a))
    return (float(a.mean() - 1.96 * se), float(a.mean() + 1.96 * se))


def _subsample_lec_states(states, n=LEC_N_STATES, seed=42):
    """Subsample states for LEC computation to keep runtime feasible."""
    if len(states) <= n:
        return states
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(states), size=n, replace=False)
    return states[idx]


def _deploy_voting_predict(pipelines, weights, env_name, eval_seeds,
                           success_threshold, weighted=True):
    """Deploy weighted or vanilla voting ensemble in environment."""
    import gymnasium as gym

    if env_name == "MiniGrid-Dynamic-Obstacles-8x8-v0":
        import minigrid  # noqa: F401 — registers envs
        from reproduction.minigrid_feature_wrapper import MiniGridFeatureWrapper
        env = MiniGridFeatureWrapper(gym.make(env_name))
    else:
        env = gym.make(env_name)

    episode_rewards = []
    for ep_seed in eval_seeds:
        obs, info = env.reset(seed=ep_seed)
        total_reward = 0.0
        done = False
        while not done:
            obs_2d = np.array(obs, dtype=np.float32).reshape(1, -1)
            if weighted and weights is not None:
                action = int(weighted_voting_predict(
                    pipelines, obs_2d, weights)[0])
            else:
                action = int(voting_predict(pipelines, obs_2d)[0])
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        episode_rewards.append(total_reward)
    env.close()

    rewards_arr = np.array(episode_rewards)
    n_success = (sum(1 for r in episode_rewards if r >= success_threshold)
                 if success_threshold else None)
    return {
        "E_CR": float(rewards_arr.mean()),
        "E_CR_std": float(rewards_arr.std()),
        "success_rate": (n_success / len(episode_rewards)
                         if n_success is not None else None),
    }


def _eval_fidelity(preds, heldout_a):
    """Compute macro-F1, accuracy, per-action metrics, worst-R."""
    acc = float(np.mean(preds == heldout_a))
    actions_set = sorted(np.unique(heldout_a))
    per_action = {}
    recalls = []
    f1s = []
    for a in actions_set:
        true_mask = heldout_a == a
        pred_mask = preds == a
        tp = int((true_mask & pred_mask).sum())
        prec = tp / pred_mask.sum() if pred_mask.sum() > 0 else 0.0
        rec = tp / true_mask.sum() if true_mask.sum() > 0 else 0.0
        pa_f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_action[int(a)] = {"precision": prec, "recall": rec, "f1": pa_f1}
        recalls.append(rec)
        f1s.append(pa_f1)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    worst_r = min(recalls) if recalls else 0.0
    return {
        "f1": macro_f1,
        "accuracy": acc,
        "per_action": per_action,
        "worst_action_recall": worst_r,
    }


def _collect_data(env_name, model_path, seed, n_transitions=10000):
    """Collect replay data. Handles MiniGrid specially."""
    if env_name == "MiniGrid-Dynamic-Obstacles-8x8-v0":
        import minigrid  # noqa: F401 — registers envs
        from reproduction.minigrid_feature_wrapper import MiniGridFeatureWrapper
        import gymnasium as gym
        env = MiniGridFeatureWrapper(gym.make(env_name))
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        states, actions, rewards, dones = [], [], [], []
        obs, info = env.reset(seed=seed)
        n = 0
        while n < n_transitions:
            action, _ = model.predict(obs, deterministic=True)
            action = int(np.asarray(action).item())
            states.append(obs.copy())
            actions.append(action)
            obs_next, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            dones.append(terminated or truncated)
            obs = obs_next
            n += 1
            if terminated or truncated:
                obs, info = env.reset()
        env.close()
        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int32),
        }
    else:
        return collect_replay(
            env_name=env_name, model_path=model_path,
            num_transitions=n_transitions, seed=seed, deterministic=True,
        )


class _VotingPredictor:
    """Wrapper to make voting ensemble compatible with LEC prediction API."""
    def __init__(self, pipelines, weights=None):
        self.pipelines = pipelines
        self.weights = weights

    def predict(self, states):
        if self.weights is not None:
            return weighted_voting_predict(self.pipelines, states, self.weights)
        return voting_predict(self.pipelines, states)


# ── Main comparison ─────────────────────────────────────────────


def run_main_comparison(env_name):
    """Run the main comparison for one environment."""
    print(f"\n{'='*60}")
    print(f"  Experiment A — Main Comparison: {env_name}")
    print(f"{'='*60}")
    t0 = time.time()

    model_path = get_model_path(env_name)
    env_tag = env_name.replace("-", "_").lower()
    out_dir = os.path.join(OUT_ROOT, env_tag)
    os.makedirs(out_dir, exist_ok=True)

    # ── Collect data ──
    print(f"  Collecting outer datasets (seeds 0–{N_OUTER_REPEATS-1})...")
    outer_datasets = []
    for seed in OUTER_SEEDS:
        data = _collect_data(env_name, model_path, seed)
        outer_datasets.append(data)

    print(f"  Collecting calibration data (seed={CALIBRATION_SEED})...")
    cal_data = _collect_data(env_name, model_path, CALIBRATION_SEED)
    cal_s, cal_a = cal_data["states"], cal_data["actions"]

    print(f"  Collecting held-out data (seed={HELDOUT_SEED})...")
    heldout_data = _collect_data(env_name, model_path, HELDOUT_SEED)
    heldout_s, heldout_a = heldout_data["states"], heldout_data["actions"]

    feature_ranges = compute_feature_ranges(outer_datasets[0])
    feature_mins = outer_datasets[0]["states"].min(axis=0)
    feature_maxs = outer_datasets[0]["states"].max(axis=0)

    # Subsample held-out for LEC to keep runtime feasible
    lec_states = _subsample_lec_states(heldout_s)

    success_thresh = SUCCESS_THRESHOLDS.get(env_name)

    results = {}

    # ── CBS baseline ──
    print(f"\n  --- CBS baseline ({N_OUTER_REPEATS} repeats) ---")
    cbs_metrics = []
    cbs_all_preds = []
    for data in outer_datasets:
        cbs, rules = run_cbs_on_data(data["states"], data["actions"], env_name)
        preds = cbs.predict(heldout_s)
        fid = _eval_fidelity(preds, heldout_a)
        if env_name == "MiniGrid-Dynamic-Obstacles-8x8-v0":
            # MiniGrid needs feature wrapper; use _deploy with single-pipeline list
            deploy = _deploy_voting_predict(
                [cbs], None, env_name, EVAL_SEEDS, success_thresh, weighted=False)
        else:
            deploy_raw = cbs.evaluate_in_env(
                env_name, eval_seeds=EVAL_SEEDS,
                success_threshold=success_thresh,
            )
            deploy = {"E_CR": deploy_raw["E_CR"], "E_CR_std": deploy_raw["E_CR_std"],
                       "success_rate": deploy_raw.get("success_rate")}
        lec = compute_lec_prediction_based(
            cbs, lec_states, feature_mins, feature_maxs,
            epsilons=LEC_EPSILONS, n_perturbations=30, seed=42,
        )
        cbs_metrics.append({
            "fidelity": fid,
            "deployment": deploy,
            "n_rules": len(rules),
            "lec": {str(k): v for k, v in lec.items()},
        })
        cbs_all_preds.append(preds)

    cbs_bra = compute_bra_from_predictions(cbs_all_preds)
    results["CBS"] = {
        "per_run": cbs_metrics,
        "stability": {"BRA": float(cbs_bra)},
    }
    _print_summary("CBS", cbs_metrics, cbs_bra)

    # ── Vanilla rule-set voting ──
    print(f"\n  --- Vanilla rule-set voting ({N_OUTER_REPEATS} repeats) ---")
    vote_metrics = []
    vote_all_preds = []
    for data in outer_datasets:
        pipelines = build_voting_ensemble(data, env_name, n_bootstrap=DEFAULT_B)
        preds = voting_predict(pipelines, heldout_s)
        fid = _eval_fidelity(preds, heldout_a)
        deploy = _deploy_voting_predict(
            pipelines, None, env_name, EVAL_SEEDS, success_thresh, weighted=False)
        predictor = _VotingPredictor(pipelines)
        lec = compute_lec_prediction_based(
            predictor, lec_states, feature_mins, feature_maxs,
            epsilons=LEC_EPSILONS, n_perturbations=30, seed=42,
        )
        cost = compute_voter_cost_metrics(pipelines, heldout_s[:200])
        vote_metrics.append({
            "fidelity": fid,
            "deployment": deploy,
            "n_rules": sum(len(p.get_rules()) for p in pipelines),
            "lec": {str(k): v for k, v in lec.items()},
            "cost": cost,
        })
        vote_all_preds.append(preds)

    vote_bra = compute_bra_from_predictions(vote_all_preds)
    results["B3_vote"] = {
        "per_run": vote_metrics,
        "stability": {"BRA": float(vote_bra)},
    }
    _print_summary("B3-vote", vote_metrics, vote_bra)

    # ── Weighted rule-set voting variants ──
    for tag, wt, alpha, beta in WEIGHTED_CONFIGS:
        print(f"\n  --- Weighted rule-set voting [{tag}] ({N_OUTER_REPEATS} repeats) ---")
        w_metrics = []
        w_all_preds = []

        for data in outer_datasets:
            pipelines = build_voting_ensemble(data, env_name, n_bootstrap=DEFAULT_B)
            # Compute weights on calibration data
            weights = compute_voter_weights(
                pipelines, cal_s, cal_a,
                weight_type=wt,
                alpha=alpha if alpha is not None else 0.5,
                beta=beta,
            )
            preds = weighted_voting_predict(pipelines, heldout_s, weights)
            fid = _eval_fidelity(preds, heldout_a)
            deploy = _deploy_voting_predict(
                pipelines, weights, env_name, EVAL_SEEDS, success_thresh)
            predictor = _VotingPredictor(pipelines, weights)
            lec = compute_lec_prediction_based(
                predictor, lec_states, feature_mins, feature_maxs,
                epsilons=LEC_EPSILONS, n_perturbations=30, seed=42,
            )
            cost = compute_voter_cost_metrics(pipelines, heldout_s[:200])
            w_metrics.append({
                "fidelity": fid,
                "deployment": deploy,
                "n_rules": sum(len(p.get_rules()) for p in pipelines),
                "lec": {str(k): v for k, v in lec.items()},
                "cost": cost,
                "weights": weights.tolist(),
                "config": {"weight_type": wt, "alpha": alpha, "beta": beta},
            })
            w_all_preds.append(preds)

        w_bra = compute_bra_from_predictions(w_all_preds)
        results[f"weighted_{tag}"] = {
            "per_run": w_metrics,
            "stability": {"BRA": float(w_bra)},
        }
        _print_summary(f"W-{tag}", w_metrics, w_bra)

    # ── Save results ──
    elapsed = time.time() - t0
    output = {
        "env_name": env_name,
        "n_outer_repeats": N_OUTER_REPEATS,
        "calibration_seed": CALIBRATION_SEED,
        "heldout_seed": HELDOUT_SEED,
        "default_B": DEFAULT_B,
        "elapsed_seconds": elapsed,
        "results": results,
    }
    out_path = os.path.join(out_dir, "main_comparison.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved → {out_path}  ({elapsed:.0f}s)")

    return output


def _print_summary(label, metrics, bra):
    """Print one-line summary of a method's results."""
    f1s = [m["fidelity"]["f1"] for m in metrics]
    wrs = [m["fidelity"]["worst_action_recall"] for m in metrics]
    print(f"    {label:20s}  F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}  "
          f"worst-R={np.mean(wrs):.3f}  BRA={bra:.3f}")


# ── Ensemble-size sensitivity ──────────────────────────────────────────────


def run_b_sensitivity(env_name, best_weighted_tag=None):
    """Run B sensitivity: B ∈ {3,5,10} for vanilla vs best weighted.

    best_weighted_tag: if None, will be auto-detected from main_comparison results.
    """
    print(f"\n{'='*60}")
    print(f"  Experiment A — B Sensitivity: {env_name}")
    print(f"{'='*60}")
    t0 = time.time()

    model_path = get_model_path(env_name)
    env_tag = env_name.replace("-", "_").lower()
    out_dir = os.path.join(OUT_ROOT, env_tag)
    os.makedirs(out_dir, exist_ok=True)

    # Auto-detect best weighted config
    if best_weighted_tag is None:
        main_path = os.path.join(out_dir, "main_comparison.json")
        if os.path.exists(main_path):
            with open(main_path) as f:
                main_res = json.load(f)
            best_weighted_tag = _select_best_weighted(main_res["results"])
            print(f"  Auto-selected best weighted: {best_weighted_tag}")
        else:
            best_weighted_tag = "f1_b1"
            print(f"  No main_comparison.json found, defaulting to {best_weighted_tag}")

    # Find the config
    best_cfg = None
    for tag, wt, alpha, beta in WEIGHTED_CONFIGS:
        if tag == best_weighted_tag:
            best_cfg = (tag, wt, alpha, beta)
            break
    if best_cfg is None:
        print(f"  ERROR: config '{best_weighted_tag}' not found")
        return None

    # Collect data
    print(f"  Collecting outer datasets...")
    outer_datasets = []
    for seed in OUTER_SEEDS:
        data = _collect_data(env_name, model_path, seed)
        outer_datasets.append(data)

    cal_data = _collect_data(env_name, model_path, CALIBRATION_SEED)
    cal_s, cal_a = cal_data["states"], cal_data["actions"]

    heldout_data = _collect_data(env_name, model_path, HELDOUT_SEED)
    heldout_s, heldout_a = heldout_data["states"], heldout_data["actions"]

    success_thresh = SUCCESS_THRESHOLDS.get(env_name)

    results = {}
    for B in B_VALUES:
        print(f"\n  --- B={B} ---")

        # Vanilla
        v_metrics = []
        v_all_preds = []
        for data in outer_datasets:
            pipelines = build_voting_ensemble(data, env_name, n_bootstrap=B)
            preds = voting_predict(pipelines, heldout_s)
            fid = _eval_fidelity(preds, heldout_a)
            v_metrics.append({"fidelity": fid})
            v_all_preds.append(preds)
        v_bra = compute_bra_from_predictions(v_all_preds)
        _print_summary(f"vanilla B={B}", v_metrics, v_bra)

        # Weighted (best config)
        w_tag, wt, alpha, beta = best_cfg
        w_metrics = []
        w_all_preds = []
        for data in outer_datasets:
            pipelines = build_voting_ensemble(data, env_name, n_bootstrap=B)
            weights = compute_voter_weights(
                pipelines, cal_s, cal_a,
                weight_type=wt,
                alpha=alpha if alpha is not None else 0.5,
                beta=beta,
            )
            preds = weighted_voting_predict(pipelines, heldout_s, weights)
            fid = _eval_fidelity(preds, heldout_a)
            w_metrics.append({"fidelity": fid, "weights": weights.tolist()})
            w_all_preds.append(preds)
        w_bra = compute_bra_from_predictions(w_all_preds)
        _print_summary(f"weighted B={B}", w_metrics, w_bra)

        results[f"B{B}"] = {
            "vanilla": {
                "per_run": v_metrics,
                "stability": {"BRA": float(v_bra)},
            },
            f"weighted_{w_tag}": {
                "per_run": w_metrics,
                "stability": {"BRA": float(w_bra)},
            },
        }

    elapsed = time.time() - t0
    output = {
        "env_name": env_name,
        "best_weighted_tag": best_weighted_tag,
        "B_values": B_VALUES,
        "n_outer_repeats": N_OUTER_REPEATS,
        "elapsed_seconds": elapsed,
        "results": results,
    }
    out_path = os.path.join(out_dir, "b_sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved → {out_path}  ({elapsed:.0f}s)")

    return output


def _select_best_weighted(results):
    """Select the best weighted config by F1 improvement over vanilla with BRA ≥ vanilla."""
    vanilla_f1 = np.mean([m["fidelity"]["f1"]
                          for m in results["B3_vote"]["per_run"]])
    vanilla_bra = results["B3_vote"]["stability"]["BRA"]

    best_tag = None
    best_delta = -1e9
    for key, val in results.items():
        if not key.startswith("weighted_"):
            continue
        tag = key[len("weighted_"):]
        f1 = np.mean([m["fidelity"]["f1"] for m in val["per_run"]])
        bra = val["stability"]["BRA"]
        wr = np.mean([m["fidelity"]["worst_action_recall"]
                      for m in val["per_run"]])
        # Must not degrade BRA
        if bra < vanilla_bra - 0.005:
            continue
        delta = f1 - vanilla_f1
        if delta > best_delta:
            best_delta = delta
            best_tag = tag
    return best_tag or "f1_b1"


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Weighted rule-set voting")
    parser.add_argument("--env", type=str, default="all",
                        help="Environment name or 'all'")
    parser.add_argument("--b-sensitivity", action="store_true",
                        help="Run the ensemble-size sensitivity study instead of the main comparison")
    args = parser.parse_args()

    if args.b_sensitivity:
        envs = B_SENS_ENVS if args.env == "all" else [args.env]
        for env in envs:
            run_b_sensitivity(env)
    else:
        envs = MAIN_ENVS if args.env == "all" else [args.env]
        for env in envs:
            run_main_comparison(env)


if __name__ == "__main__":
    main()
