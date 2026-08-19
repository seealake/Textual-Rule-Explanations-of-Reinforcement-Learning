#!/usr/bin/env python
"""
Rare-Action Support Sweep

Two layers:
  Layer 1 (Natural Scaling): Vary total replay size, keep natural action proportions
  Layer 2 (Diagnostic Quota Resampling): Fixed N=10k, control rare-action proportion

Tests whether CBS instability is driven by minority-action data scarcity.

Usage:
    python experiments/run_rare_action_sweep.py --env MountainCar-v0
    python experiments/run_rare_action_sweep.py --env LunarLander-v3
    python experiments/run_rare_action_sweep.py --env all

Output:
    experiments/results/<env>/rare_action_sweep_results.json
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

# ── Constants ─────────────────────────────────────────────────────────
ENVS = ["MountainCar-v0", "LunarLander-v3"]
HELDOUT_SEED = 99
RESULTS_DIR = "experiments/results"

MC_CONFIG = {
    "rare_action": 1,  # no_push, ~0.9% natural
    "natural_sizes": [5000, 10000, 20000, 50000],
    "resampled_supports": [0.02, 0.05, 0.10, 0.15],
    "target_size": 10000,
    "master_pool_size": 50000,
}

LL_CONFIG = {
    "rare_action": 3,  # fire_right, ~6.7% natural
    "natural_sizes": [5000, 10000, 20000, 50000],
    "resampled_supports": [0.02, 0.05, 0.08, 0.12, 0.15],
    "target_size": 10000,
    "master_pool_size": 50000,
}

ENV_CONFIGS = {
    "MountainCar-v0": MC_CONFIG,
    "LunarLander-v3": LL_CONFIG,
}


def get_model_path(env_name):
    tag = env_name.replace("-", "_").lower()
    return f"reproduction/models/dqn_{tag}.zip"


def get_ncat(env_name):
    return 6 if env_name == "LunarLander-v3" else 5


def construct_controlled_dataset(master_states, master_actions, target_size,
                                  rare_action, rare_support, seed=42):
    """Construct a fixed-size dataset with controlled rare-action proportion.

    Returns (states, actions, info_dict) where info_dict has:
      n_rare, n_other, actual_support, replace_used, n_unique_rare
    """
    rng = np.random.default_rng(seed)

    rare_mask = master_actions == rare_action
    other_mask = ~rare_mask
    rare_indices = np.where(rare_mask)[0]
    other_indices = np.where(other_mask)[0]

    n_rare_target = int(target_size * rare_support)
    n_other_target = target_size - n_rare_target

    # Sample rare-action transitions
    replace_rare = n_rare_target > len(rare_indices)
    rare_sample = rng.choice(rare_indices, size=n_rare_target, replace=replace_rare)
    n_unique_rare = len(np.unique(rare_sample))

    # Sample other transitions
    replace_other = n_other_target > len(other_indices)
    other_sample = rng.choice(other_indices, size=n_other_target, replace=replace_other)

    # Combine and shuffle
    all_indices = np.concatenate([rare_sample, other_sample])
    rng.shuffle(all_indices)

    return (
        master_states[all_indices],
        master_actions[all_indices],
        {
            "n_rare": n_rare_target,
            "n_other": n_other_target,
            "actual_support": n_rare_target / target_size,
            "replace_used_rare": bool(replace_rare),
            "n_unique_rare": int(n_unique_rare),
            "n_available_rare": int(len(rare_indices)),
        }
    )


def evaluate_cbs(states, actions, heldout_s, heldout_a, env_name, rare_action):
    """Fit CBS and return the metric bundle."""
    ncat = get_ncat(env_name)
    cbs = CBSPipeline(n_categories=ncat, inclusion_threshold=0.70, kmeans_seed=0)
    cbs.fit(states, actions)

    # Overall fidelity
    fidelity = cbs.evaluate_fidelity(heldout_s, heldout_a)

    # Per-action fidelity
    preds = cbs.predict(heldout_s)
    n_actions = max(heldout_a.max(), actions.max()) + 1

    per_action = {}
    worst_recall = 1.0
    action_lost_rules = []

    for a in range(n_actions):
        a_mask = heldout_a == a
        if a_mask.sum() == 0:
            continue
        a_preds = preds[a_mask]
        a_true = heldout_a[a_mask]
        tp = int(np.sum((a_preds == a) & (a_true == a)))
        support = int(a_mask.sum())
        recall = tp / support if support > 0 else 0.0
        per_action[int(a)] = {
            "recall": recall,
            "support": support,
            "n_correct": tp,
        }
        worst_recall = min(worst_recall, recall)

        # Check if action has any rules
        action_rules = [r for r in cbs.rules_ if r.action == a]
        if len(action_rules) == 0:
            action_lost_rules.append(int(a))
        per_action[int(a)]["n_rules"] = len(action_rules)

    # Count rules per action in training data
    n_rules_total = len(cbs.rules_)
    n_rules_rare = sum(1 for r in cbs.rules_ if r.action == rare_action)

    return {
        "f1": float(fidelity["f1"]),
        "accuracy": float(fidelity["accuracy"]),
        "recall": float(fidelity["recall"]),
        "n_rules": n_rules_total,
        "n_rules_rare": n_rules_rare,
        "worst_action_recall": float(worst_recall),
        "per_action": per_action,
        "actions_lost_rules": action_lost_rules,
    }


def run_natural_scaling(env_name, model_path, config, heldout_s, heldout_a):
    """Layer 1: Natural Scaling — vary total replay size."""
    print(f"\n  Layer 1: Natural Scaling")
    results = []
    rare_action = config["rare_action"]

    for n_total in config["natural_sizes"]:
        print(f"    N={n_total}...", end=" ")

        # Collect replay of this size
        data = collect_replay(env_name, model_path, num_transitions=n_total, seed=42)
        states, actions = data["states"], data["actions"]

        # Natural rare-action count
        n_rare_natural = int(np.sum(actions == rare_action))
        natural_support = n_rare_natural / n_total

        # Evaluate
        metrics = evaluate_cbs(states, actions, heldout_s, heldout_a, env_name, rare_action)
        metrics["n_total"] = n_total
        metrics["n_rare_natural"] = n_rare_natural
        metrics["natural_support"] = float(natural_support)

        print(f"rare_support={natural_support:.3f} ({n_rare_natural}/{n_total}), "
              f"worst_R={metrics['worst_action_recall']:.3f}, F1={metrics['f1']:.3f}, "
              f"rules_rare={metrics['n_rules_rare']}")

        results.append(metrics)

    return results


def run_diagnostic_resampling(env_name, model_path, config, heldout_s, heldout_a):
    """Layer 2: Diagnostic Quota Resampling — control rare-action proportion."""
    print(f"\n  Layer 2: Diagnostic Quota Resampling")
    rare_action = config["rare_action"]
    target_size = config["target_size"]

    # Collect master pool
    print(f"    Collecting master pool ({config['master_pool_size']} transitions)...")
    master = collect_replay(env_name, model_path,
                           num_transitions=config["master_pool_size"], seed=42)
    master_s, master_a = master["states"], master["actions"]

    n_rare_available = int(np.sum(master_a == rare_action))
    natural_prop = n_rare_available / len(master_a)
    print(f"    Master pool: {len(master_a)} transitions, "
          f"rare action {rare_action}: {n_rare_available} ({natural_prop:.3f})")

    results = []

    for support in config["resampled_supports"]:
        print(f"    support={support:.2f}...", end=" ")

        # Construct controlled dataset (average over 3 seeds for robustness)
        seed_metrics = []
        for seed in range(3):
            states, actions, info = construct_controlled_dataset(
                master_s, master_a, target_size, rare_action, support, seed=seed)

            metrics = evaluate_cbs(states, actions, heldout_s, heldout_a, env_name, rare_action)
            seed_metrics.append(metrics)

        # Average across seeds
        avg_metrics = {}
        for key in ["f1", "worst_action_recall", "n_rules", "n_rules_rare"]:
            vals = [m[key] for m in seed_metrics]
            avg_metrics[f"mean_{key}"] = float(np.mean(vals))
            avg_metrics[f"std_{key}"] = float(np.std(vals))

        avg_metrics["target_support"] = support
        avg_metrics["target_size"] = target_size
        avg_metrics["n_rare_available"] = n_rare_available
        avg_metrics["replace_used"] = seed_metrics[0].get("replace_used_rare",
                                        info.get("replace_used_rare", False))
        avg_metrics["per_action_example"] = seed_metrics[0]["per_action"]
        avg_metrics["actions_lost_rules_any"] = list(set(
            a for m in seed_metrics for a in m.get("actions_lost_rules", [])))

        print(f"worst_R={avg_metrics['mean_worst_action_recall']:.3f}±{avg_metrics['std_worst_action_recall']:.3f}, "
              f"F1={avg_metrics['mean_f1']:.3f}, "
              f"rules_rare={avg_metrics['mean_n_rules_rare']:.1f}, "
              f"replace={avg_metrics['replace_used']}")

        results.append(avg_metrics)

    return results


def run_rare_action_sweep(env_name):
    """Run full rare-action support sweep for one environment."""
    print(f"\n{'='*60}")
    print(f"Rare-Action Support Sweep: {env_name}")
    print(f"{'='*60}")
    t0 = time.time()

    model_path = get_model_path(env_name)
    config = ENV_CONFIGS[env_name]

    # Collect held-out replay
    print("  Collecting held-out replay...")
    heldout = collect_replay(env_name, model_path, num_transitions=5000, seed=HELDOUT_SEED)
    heldout_s, heldout_a = heldout["states"], heldout["actions"]

    # Layer 1
    natural_results = run_natural_scaling(env_name, model_path, config, heldout_s, heldout_a)

    # Layer 2
    diagnostic_results = run_diagnostic_resampling(env_name, model_path, config, heldout_s, heldout_a)

    elapsed = time.time() - t0
    tag = env_name.replace("-", "_").lower()

    output = {
        "schema_version": "rare_action_v1",
        "env": env_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "rare_action": config["rare_action"],
        "natural_scaling": natural_results,
        "diagnostic_resampling": diagnostic_results,
    }

    out_dir = os.path.join(RESULTS_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "rare_action_sweep_results.json")

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
    parser = argparse.ArgumentParser(description="Rare-action support sweep")
    parser.add_argument("--env", default="all",
                        choices=["MountainCar-v0", "LunarLander-v3", "all"])
    args = parser.parse_args()

    envs = ENVS if args.env == "all" else [args.env]
    for env_name in envs:
        run_rare_action_sweep(env_name)


if __name__ == "__main__":
    main()
