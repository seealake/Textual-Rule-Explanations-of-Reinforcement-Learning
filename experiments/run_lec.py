#!/usr/bin/env python
"""
LEC Runner — Local Explanation Consistency Analysis

Runs the LEC analysis for CBS and tuned CBS.

Usage:
    python experiments/run_lec.py --env MountainCar-v0
    python experiments/run_lec.py --env CartPole-v1
    python experiments/run_lec.py --env all

Output:
    experiments/results/<env>/lec_results.json
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
from experiments.perturbations import load_replay_npz
from experiments.lec import compute_lec

ENVS = ["MountainCar-v0", "CartPole-v1", "LunarLander-v3"]
EPSILONS = [0.01, 0.03, 0.05]
N_PERTURBATIONS = 50
HELDOUT_SEED = 99


def get_model_path(env_name):
    tag = env_name.replace("-", "_").lower()
    return f"reproduction/models/dqn_{tag}.zip"


def get_replay_path(env_name, seed=42):
    tag = env_name.replace("-", "_").lower()
    return f"reproduction/data/replay_{tag}_seed{seed}.npz"


def run_lec_for_env(env_name):
    """Run LEC analysis for one environment."""
    print(f"\n{'='*60}")
    print(f"  LEC Analysis: {env_name}")
    print(f"{'='*60}")

    model_path = get_model_path(env_name)
    ref_path = get_replay_path(env_name)

    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found at {model_path}. Skipping.")
        return
    if not os.path.exists(ref_path):
        print(f"  ERROR: Replay not found at {ref_path}. Skipping.")
        return

    # Load reference replay
    ref_data = load_replay_npz(ref_path)
    print(f"  Reference replay: {len(ref_data['states'])} transitions")

    # Collect held-out replay
    print(f"  Collecting held-out replay (seed={HELDOUT_SEED})...")
    heldout = collect_replay(
        env_name=env_name, model_path=model_path,
        num_transitions=5000, seed=HELDOUT_SEED,
        deterministic=True,
    )
    heldout_s = heldout["states"]
    print(f"  Held-out replay: {len(heldout_s)} transitions")

    # Compute feature ranges from reference data
    feature_mins = ref_data["states"].min(axis=0)
    feature_maxs = ref_data["states"].max(axis=0)
    feature_names = ENV_FEATURE_NAMES.get(env_name, [])
    print(f"  Feature ranges:")
    for i, (mn, mx) in enumerate(zip(feature_mins, feature_maxs)):
        name = feature_names[i] if i < len(feature_names) else f"f{i}"
        print(f"    {name}: [{mn:.4f}, {mx:.4f}]")

    results = {}

    # --- CBS ---
    print(f"\n  --- Method: CBS ---")
    t0 = time.time()
    cbs = CBSPipeline(
        n_categories=5, inclusion_threshold=0.70,
        kmeans_seed=0, feature_names=feature_names,
    )
    cbs.fit(ref_data["states"], ref_data["actions"])
    fid = cbs.evaluate_fidelity(heldout_s, heldout["actions"])
    print(f"  CBS fidelity (held-out): F1={fid['f1']:.4f}")
    print(f"  CBS rules: {len(cbs.get_rules())}")

    print(f"  Computing LEC at epsilon={EPSILONS}...")
    lec_cbs = compute_lec(
        cbs, heldout_s, feature_mins, feature_maxs,
        epsilons=EPSILONS, n_perturbations=N_PERTURBATIONS, seed=42,
    )
    elapsed_cbs = time.time() - t0
    print(f"  CBS LEC results ({elapsed_cbs:.1f}s):")
    for eps, data in sorted(lec_cbs.items()):
        print(f"    epsilon={eps}: LEC={data['lec']:.4f} "
              f"(std={data['lec_std']:.4f}, null_orig={data['n_null_original']})")
    results["cbs"] = {str(eps): data for eps, data in lec_cbs.items()}

    # --- CBS + MaxF1 ---
    print(f"\n  --- Method: CBS + MaxF1 ---")
    t1 = time.time()
    cbs_mf = CBSPipeline(
        n_categories=5, inclusion_threshold=0.70,
        kmeans_seed=0, feature_names=feature_names,
    )
    cbs_mf.fit(ref_data["states"], ref_data["actions"])
    cbs_mf.refine_max_f1(ref_data["states"], ref_data["actions"])
    fid_mf = cbs_mf.evaluate_fidelity(heldout_s, heldout["actions"])
    print(f"  CBS+MaxF1 fidelity (held-out): F1={fid_mf['f1']:.4f}")
    print(f"  CBS+MaxF1 rules: {len(cbs_mf.get_rules())}")

    print(f"  Computing LEC at epsilon={EPSILONS}...")
    lec_maxf1 = compute_lec(
        cbs_mf, heldout_s, feature_mins, feature_maxs,
        epsilons=EPSILONS, n_perturbations=N_PERTURBATIONS, seed=42,
    )
    elapsed_mf = time.time() - t1
    print(f"  CBS+MaxF1 LEC results ({elapsed_mf:.1f}s):")
    for eps, data in sorted(lec_maxf1.items()):
        print(f"    epsilon={eps}: LEC={data['lec']:.4f} "
              f"(std={data['lec_std']:.4f}, null_orig={data['n_null_original']})")
    results["cbs_maxf1"] = {str(eps): data for eps, data in lec_maxf1.items()}

    # --- Save ---
    out_dir = f"experiments/results/{env_name.replace('-', '_').lower()}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "lec_results.json")
    output = {
        "env": env_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epsilons": EPSILONS,
        "n_perturbations": N_PERTURBATIONS,
        "n_heldout_states": len(heldout_s),
        "feature_ranges": {
            "mins": feature_mins.tolist(),
            "maxs": feature_maxs.tolist(),
        },
        "cbs": results["cbs"],
        "cbs_maxf1": results["cbs_maxf1"],
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # --- Print summary table ---
    print(f"\n  {'='*50}")
    print(f"  LEC Summary: {env_name}")
    print(f"  {'epsilon':<10} {'CBS':<15} {'CBS+MaxF1':<15}")
    print(f"  {'-'*40}")
    for eps in EPSILONS:
        lec_c = lec_cbs[eps]['lec']
        lec_m = lec_maxf1[eps]['lec']
        print(f"  {eps:<10} {lec_c:<15.4f} {lec_m:<15.4f}")
    print(f"  {'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Run LEC analysis")
    parser.add_argument("--env", type=str, default="all",
                        help="Environment name or 'all'")
    args = parser.parse_args()

    envs = ENVS if args.env == "all" else [args.env]
    for env_name in envs:
        run_lec_for_env(env_name)

    print("\nAll LEC analyses complete!")


if __name__ == "__main__":
    main()
