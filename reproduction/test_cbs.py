#!/usr/bin/env python
"""
Test CBS pipeline on collected replay data.

Usage:
    python reproduction/test_cbs.py
    python reproduction/test_cbs.py --env CartPole-v1
    python reproduction/test_cbs.py --env MountainCar-v0 --n-categories 5
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from reproduction.cbs import CBSPipeline

# Feature names per environment
ENV_FEATURE_NAMES = {
    "MountainCar-v0": ["car_position", "car_velocity"],
    "CartPole-v1": ["cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity"],
}

ENV_ACTION_NAMES = {
    "MountainCar-v0": {0: "push_left", 1: "no_push", 2: "push_right"},
    "CartPole-v1": {0: "push_left", 1: "push_right"},
}


def load_replay(env_name: str, seed: int = 42, data_dir: str = "reproduction/data"):
    """Load replay data from .npz file."""
    env_tag = env_name.replace("-", "_").lower()
    path = os.path.join(data_dir, f"replay_{env_tag}_seed{seed}.npz")
    if not os.path.exists(path):
        print(f"ERROR: Replay not found at {path}")
        sys.exit(1)
    data = np.load(path)
    return data["states"], data["actions"]


def run_cbs_test(env_name: str, n_categories: int = 5, seed: int = 42,
                 run_env_deploy: bool = False, run_maxf1: bool = False):
    """Run CBS on a single environment and print results."""
    print(f"\n{'='*70}")
    print(f"  CBS Test: {env_name} (ncat={n_categories})")
    print(f"{'='*70}")

    # Load data
    states, actions = load_replay(env_name, seed)
    feature_names = ENV_FEATURE_NAMES.get(env_name)
    action_names = ENV_ACTION_NAMES.get(env_name, {})

    print(f"\n  Data: {len(states)} transitions, {states.shape[1]} features")
    print(f"  Actions: {np.unique(actions)}")

    # Fit CBS
    print(f"\n  Fitting CBS pipeline...")
    t0 = time.time()

    cbs = CBSPipeline(
        n_categories=n_categories,
        inclusion_threshold=0.70,
        max_clusters=40,
        kmeans_seed=42,
        feature_names=feature_names,
    )
    cbs.fit(states, actions)

    fit_time = time.time() - t0
    print(f"  Fit time: {fit_time:.2f}s")

    # Print thresholds
    print(f"\n  Predicate Thresholds:")
    cbs.print_thresholds()

    # Print rules
    print(f"\n  Extracted Rules:")
    cbs.print_rules()

    # Evaluate properties
    props = cbs.evaluate_properties()
    print(f"\n  Explanation Properties:")
    print(f"    Total conditions (E_len):    {props['n_conditions']}")
    print(f"    Duplicated conditions:       {props['n_duplicated']}")
    print(f"    Unique signatures:           {props['n_unique_signatures']}")

    # Evaluate coverage
    coverage = cbs.evaluate_coverage(states)
    print(f"\n  Coverage:")
    print(f"    States needing approximation: {coverage['n_approximated']}/{coverage['n_total']} "
          f"({coverage['approx_rate']*100:.1f}%)")

    # Evaluate fidelity
    fidelity = cbs.evaluate_fidelity(states, actions)
    print(f"\n  Fidelity (on training replay):")
    print(f"    Accuracy (E_acc):  {fidelity['accuracy']*100:.1f}%")
    print(f"    Recall (E_rec):    {fidelity['recall']*100:.1f}%")
    print(f"    F1 (E_F1):         {fidelity['f1']*100:.1f}%")

    # Environment deployment
    env_results = None
    if run_env_deploy:
        print(f"\n  Deploying rules as policy in {env_name}...")
        env_results = cbs.evaluate_in_env(env_name, n_episodes=10, seed=0)
        print(f"    E_CR (cumulative reward):   {env_results['E_CR']:.1f} ± {env_results['E_CR_std']:.1f}")
        print(f"    E_TS (total steps):         {env_results['E_TS']:.1f} ± {env_results['E_TS_std']:.1f}")
        print(f"    E_AR (avg reward/step):     {env_results['E_AR']:.4f}")
        print(f"    Episode rewards: {[f'{r:.0f}' for r in env_results['episode_rewards']]}")

    # MaxF1 refinement
    maxf1_results = None
    if run_maxf1:
        print(f"\n  Running MaxF1 Refinement (Algorithm 6)...")
        maxf1_results = cbs.refine_max_f1(states, actions, alpha=0.5, max_iterations=10, verbose=True)
        print(f"\n  MaxF1 Results:")
        print(f"    F1 before: {maxf1_results['f1_before']*100:.1f}%")
        print(f"    F1 after:  {maxf1_results['f1_after']*100:.1f}%")
        print(f"    Improvement: {maxf1_results['f1_improvement']*100:+.1f}pp")

        # Re-evaluate fidelity after refinement
        fidelity_after = cbs.evaluate_fidelity(states, actions)
        print(f"\n  Post-refinement fidelity:")
        print(f"    Accuracy (E_acc):  {fidelity_after['accuracy']*100:.1f}%")
        print(f"    Recall (E_rec):    {fidelity_after['recall']*100:.1f}%")
        print(f"    F1 (E_F1):         {fidelity_after['f1']*100:.1f}%")

        # Updated thresholds
        print(f"\n  Updated Predicate Thresholds:")
        cbs.print_thresholds()

        # Re-deploy in environment after refinement
        if run_env_deploy:
            print(f"\n  Re-deploying refined rules in {env_name}...")
            env_after = cbs.evaluate_in_env(env_name, n_episodes=10, seed=0)
            print(f"    E_CR: {env_after['E_CR']:.1f} ± {env_after['E_CR_std']:.1f}")
            print(f"    E_TS: {env_after['E_TS']:.1f} ± {env_after['E_TS_std']:.1f}")
            print(f"    E_AR: {env_after['E_AR']:.4f}")

    print(f"\n{'='*70}\n")

    return cbs, fidelity, props, coverage


def main():
    parser = argparse.ArgumentParser(description="Test CBS pipeline")
    parser.add_argument("--env", type=str, default=None,
                        help="Environment name (default: test both)")
    parser.add_argument("--n-categories", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-deploy", action="store_true",
                        help="Deploy rules as policy in environment (E_CR, E_TS, E_AR)")
    parser.add_argument("--maxf1", action="store_true",
                        help="Run MaxF1 threshold refinement (Algorithm 6)")
    args = parser.parse_args()

    envs = [args.env] if args.env else ["MountainCar-v0", "CartPole-v1"]

    for env in envs:
        run_cbs_test(env, args.n_categories, args.seed,
                     run_env_deploy=args.env_deploy, run_maxf1=args.maxf1)


if __name__ == "__main__":
    main()
