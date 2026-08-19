#!/usr/bin/env python
"""
SoftSupport merge: statistical tests

Bootstrap 95% CIs and paired bootstrap tests for the SoftSupport merge against the baselines.

Usage:
    python experiments/run_soft_support_statistics.py

Output:
    experiments/results/soft_support_merge/statistical_tests.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

SOFT_SUPPORT_DIR = "experiments/results/soft_support_merge/raw"
OUT_PATH = "experiments/results/soft_support_merge/statistical_tests.json"

ENVS = {
    "CartPole-v1": "cartpole_v1",
    "LunarLander-v3": "lunarlander_v3",
    "MountainCar-v0": "mountaincar_v0",
}

N_BOOTSTRAP = 1000
ALPHA = 0.05


def bootstrap_ci(vals, n_boot=N_BOOTSTRAP, alpha=ALPHA):
    """Bootstrap 95% CI for the mean."""
    arr = np.array(vals)
    if len(arr) < 2:
        return {"mean": float(arr.mean()), "ci_lo": float(arr.mean()),
                "ci_hi": float(arr.mean()), "std": 0.0}
    rng = np.random.RandomState(42)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(sample.mean())
    # Use numpy's percentile function for robust CI estimation
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(arr.mean()), "ci_lo": float(lo),
            "ci_hi": float(hi), "std": float(arr.std())}


def paired_bootstrap_test(vals_a, vals_b, n_boot=N_BOOTSTRAP):
    """Paired bootstrap test: H0: mean(A) >= mean(B).

    Returns p-value for one-sided test that A < B (i.e., B is better).
    If p < 0.05, we conclude B > A significantly.
    """
    a = np.array(vals_a)
    b = np.array(vals_b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    obs_diff = b.mean() - a.mean()

    rng = np.random.RandomState(42)
    count = 0
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if (b[idx].mean() - a[idx].mean()) <= 0:
            count += 1
    p = count / n_boot
    return {"obs_diff": float(obs_diff), "p_value": float(p),
            "significant": p < ALPHA}


def run_tests():
    print("Running V2 Statistical Tests ...\n")
    t0 = time.time()
    results = {}

    for env_name, tag in ENVS.items():
        path = os.path.join(SOFT_SUPPORT_DIR, f"{tag}_soft_support_results.json")
        if not os.path.exists(path):
            print(f"  SKIP {env_name}: no v2 results")
            continue

        with open(path) as f:
            data = json.load(f)

        print(f"=== {env_name} ===")
        env_results = {"baselines": {}, "v2_configs": {}, "comparisons": []}

        # Extract baseline per-run metrics
        baselines = data.get("baselines", {})
        baseline_f1 = {}
        for bname, bdata in baselines.items():
            runs = bdata.get("per_run", [])
            f1_vals = [r["f1"] for r in runs]
            ecr_vals = [r["E_CR"] for r in runs]
            wr_vals = [r.get("worst_action_recall", 0) for r in runs]

            ci_f1 = bootstrap_ci(f1_vals)
            ci_ecr = bootstrap_ci(ecr_vals)
            ci_wr = bootstrap_ci(wr_vals)

            env_results["baselines"][bname] = {
                "f1": ci_f1, "E_CR": ci_ecr, "worst_recall": ci_wr,
                "n_runs": len(runs),
            }
            baseline_f1[bname] = f1_vals
            print(f"  {bname}: F1={ci_f1['mean']:.3f} [{ci_f1['ci_lo']:.3f}, {ci_f1['ci_hi']:.3f}]")

        # Extract v2 sweep per-run metrics
        sweep_key = "v2_sweep" if "v2_sweep" in data else "sweep"
        sweep = data.get(sweep_key, {})
        best_soft_name = None
        best_soft_f1 = -1

        for config_name, config_data in sweep.items():
            if not isinstance(config_data, dict):
                continue
            runs = config_data.get("per_run", [])
            if not runs:
                continue

            f1_vals = [r["f1"] for r in runs]
            ecr_vals = [r["E_CR"] for r in runs]
            wr_vals = [r.get("worst_action_recall", 0) for r in runs]

            ci_f1 = bootstrap_ci(f1_vals)
            ci_ecr = bootstrap_ci(ecr_vals)
            ci_wr = bootstrap_ci(wr_vals)

            env_results["v2_configs"][config_name] = {
                "f1": ci_f1, "E_CR": ci_ecr, "worst_recall": ci_wr,
                "n_runs": len(runs),
            }

            if ci_f1["mean"] > best_soft_f1:
                best_soft_f1 = ci_f1["mean"]
                best_soft_name = config_name

        if best_soft_name:
            print(f"  Best v2: {best_soft_name} (F1={best_soft_f1:.3f})")

        # Paired comparisons: best v2 vs each baseline
        if best_soft_name and best_soft_name in sweep:
            v2_runs = sweep[best_soft_name].get("per_run", [])
            soft_support_f1 = [r["f1"] for r in v2_runs]
            v2_wr = [r.get("worst_action_recall", 0) for r in v2_runs]

            for bname, bf1 in baseline_f1.items():
                # Test: v2 F1 > baseline F1
                test_f1 = paired_bootstrap_test(bf1, soft_support_f1)
                b_wr = [r.get("worst_action_recall", 0) for r in baselines[bname]["per_run"]]
                test_wr = paired_bootstrap_test(b_wr, v2_wr)

                comparison = {
                    "baseline": bname,
                    "v2_config": best_soft_name,
                    "f1_test": test_f1,
                    "worst_recall_test": test_wr,
                }
                env_results["comparisons"].append(comparison)

                sig_f1 = "✓" if test_f1["significant"] else "✗"
                sig_wr = "✓" if test_wr["significant"] else "✗"
                print(f"    {best_soft_name} vs {bname}: "
                      f"ΔF1={test_f1['obs_diff']:+.4f} (p={test_f1['p_value']:.3f} {sig_f1}), "
                      f"ΔWR={test_wr['obs_diff']:+.4f} (p={test_wr['p_value']:.3f} {sig_wr})")

        # Also test: soft vs hard (holding lambda_B and safeguard constant)
        for lb in ["0.0", "0.1", "0.2"]:
            hard_key = f"lB{lb}_smhard_sgoff"
            soft_key = f"lB{lb}_smsoft_sgoff"
            if hard_key in sweep and soft_key in sweep:
                hard_runs = sweep[hard_key].get("per_run", [])
                soft_runs = sweep[soft_key].get("per_run", [])
                if hard_runs and soft_runs:
                    hard_f1 = [r["f1"] for r in hard_runs]
                    soft_f1 = [r["f1"] for r in soft_runs]
                    test = paired_bootstrap_test(hard_f1, soft_f1)
                    sig = "✓" if test["significant"] else "✗"
                    print(f"    soft vs hard (λ_B={lb}): "
                          f"ΔF1={test['obs_diff']:+.4f} (p={test['p_value']:.3f} {sig})")
                    env_results["comparisons"].append({
                        "test": f"soft_vs_hard_lB{lb}",
                        "obs_diff_f1": test["obs_diff"],
                        "p_value_f1": test["p_value"],
                        "significant": test["significant"],
                    })

        results[env_name] = env_results

    # Save
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    run_tests()
