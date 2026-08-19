#!/usr/bin/env python
"""
Weighted voting: paired bootstrap tests and summary tables.

Compares weighted rule-set voting (best config per env) vs vanilla rule-set voting and CBS.
Generates paired_bootstrap.json and summary CSV tables.

Usage:
    python experiments/run_weighted_vote_statistics.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Configuration ────────────────────────────────────────────────────

ENVS = ["MountainCar-v0", "CartPole-v1", "LunarLander-v3",
        "MiniGrid-Dynamic-Obstacles-8x8-v0"]
ENV_TAGS = {
    "MountainCar-v0": "mountaincar_v0",
    "CartPole-v1": "cartpole_v1",
    "LunarLander-v3": "lunarlander_v3",
    "MiniGrid-Dynamic-Obstacles-8x8-v0": "minigrid_dynamic_obstacles_8x8_v0",
}
ENV_SHORT = {
    "MountainCar-v0": "MC",
    "CartPole-v1": "CP",
    "LunarLander-v3": "LL",
    "MiniGrid-Dynamic-Obstacles-8x8-v0": "MG",
}

RESULTS_ROOT = "experiments/results/weighted_vote"
OUT_DIR = "experiments/results/weighted_vote_statistics"
N_BOOT = 10000
ALPHA = 0.05
METRICS = ["f1", "worst_action_recall", "E_CR"]


def load_main_comparison(env_name):
    env_tag = ENV_TAGS[env_name]
    path = os.path.join(RESULTS_ROOT, env_tag, "main_comparison.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_b_sensitivity(env_name):
    env_tag = ENV_TAGS[env_name]
    path = os.path.join(RESULTS_ROOT, env_tag, "b_sensitivity.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract_metric(per_run, metric):
    """Extract a metric vector from per_run list."""
    vals = []
    for run in per_run:
        if metric in ("f1", "worst_action_recall", "accuracy"):
            vals.append(run["fidelity"][metric])
        elif metric == "E_CR":
            vals.append(run["deployment"]["E_CR"])
        elif metric == "n_rules":
            vals.append(run["n_rules"])
        elif metric.startswith("lec_"):
            eps = metric.split("_")[1]
            vals.append(run["lec"][eps]["lec"])
        else:
            vals.append(run.get(metric, 0.0))
    return np.array(vals, dtype=float)


def paired_bootstrap_test(vals_a, vals_b, n_boot=N_BOOT, seed=42):
    """Paired bootstrap test (two-sided).

    Tests H0: mean(b) - mean(a) = 0.
    Returns dict with obs_diff, ci_lo, ci_hi, p_value, significant.
    """
    a = np.asarray(vals_a, dtype=float)
    b = np.asarray(vals_b, dtype=float)
    n = len(a)
    diffs = b - a
    obs_diff = float(diffs.mean())

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        boot_diffs[i] = diffs[idx].mean()

    ci_lo = float(np.percentile(boot_diffs, 100 * ALPHA / 2))
    ci_hi = float(np.percentile(boot_diffs, 100 * (1 - ALPHA / 2)))

    if obs_diff >= 0:
        p = float(2 * np.mean(boot_diffs <= 0))
    else:
        p = float(2 * np.mean(boot_diffs >= 0))
    p = min(p, 1.0)

    return {
        "obs_diff": obs_diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p,
        "significant": p < ALPHA,
    }


def select_best_weighted(results):
    """Select best weighted config by F1 delta with BRA constraint."""
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
        if bra < vanilla_bra - 0.005:
            continue
        delta = f1 - vanilla_f1
        if delta > best_delta:
            best_delta = delta
            best_tag = tag
    return best_tag or "f1_b1"


# ── Main ─────────────────────────────────────────────────────────────

def run_all_statistics():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_tests = {}
    summary_rows = []

    for env in ENVS:
        data = load_main_comparison(env)
        if data is None:
            print(f"  Skipping {env} — no main_comparison.json")
            continue

        results = data["results"]
        best_tag = select_best_weighted(results)
        best_key = f"weighted_{best_tag}"
        env_short = ENV_SHORT[env]

        print(f"\n  {env} — best weighted: {best_tag}")

        env_tests = {}

        # Method pairs to test
        pairs = [
            (f"weighted_{best_tag}", "B3_vote", f"W-{best_tag} vs Vanilla"),
            (f"weighted_{best_tag}", "CBS", f"W-{best_tag} vs CBS"),
        ]

        for key_b, key_a, pair_label in pairs:
            if key_a not in results or key_b not in results:
                continue
            pair_tests = {}
            for metric in METRICS:
                vals_a = extract_metric(results[key_a]["per_run"], metric)
                vals_b = extract_metric(results[key_b]["per_run"], metric)
                test = paired_bootstrap_test(vals_a, vals_b)
                pair_tests[metric] = test
                sig = "***" if test["significant"] else ""
                print(f"    {pair_label} | {metric:>20s}: "
                      f"Δ={test['obs_diff']:+.4f}  "
                      f"CI=[{test['ci_lo']:+.4f}, {test['ci_hi']:+.4f}]  "
                      f"p={test['p_value']:.4f} {sig}")
            env_tests[pair_label] = pair_tests

        all_tests[env] = {
            "best_weighted": best_tag,
            "tests": env_tests,
        }

        # Summary row
        cbs_f1 = np.mean(extract_metric(results["CBS"]["per_run"], "f1"))
        van_f1 = np.mean(extract_metric(results["B3_vote"]["per_run"], "f1"))
        van_wr = np.mean(extract_metric(results["B3_vote"]["per_run"], "worst_action_recall"))
        van_bra = results["B3_vote"]["stability"]["BRA"]
        w_f1 = np.mean(extract_metric(results[best_key]["per_run"], "f1"))
        w_wr = np.mean(extract_metric(results[best_key]["per_run"], "worst_action_recall"))
        w_bra = results[best_key]["stability"]["BRA"]

        summary_rows.append({
            "env": env_short,
            "best_config": best_tag,
            "CBS_F1": f"{cbs_f1:.3f}",
            "vanilla_F1": f"{van_f1:.3f}",
            "vanilla_worstR": f"{van_wr:.3f}",
            "vanilla_BRA": f"{van_bra:.3f}",
            "weighted_F1": f"{w_f1:.3f}",
            "weighted_worstR": f"{w_wr:.3f}",
            "weighted_BRA": f"{w_bra:.3f}",
            "delta_F1": f"{w_f1 - van_f1:+.3f}",
            "delta_worstR": f"{w_wr - van_wr:+.3f}",
            "delta_BRA": f"{w_bra - van_bra:+.3f}",
        })

    # Save paired bootstrap results
    out_path = os.path.join(OUT_DIR, "paired_bootstrap.json")
    with open(out_path, "w") as f:
        json.dump(all_tests, f, indent=2, default=str)
    print(f"\n  Saved → {out_path}")

    # Save summary CSV
    if summary_rows:
        import csv
        csv_path = os.path.join(OUT_DIR, "summary_table.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  Saved → {csv_path}")

    return all_tests


if __name__ == "__main__":
    run_all_statistics()
