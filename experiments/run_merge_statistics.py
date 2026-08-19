#!/usr/bin/env python
"""
Statistical tests and summary tables

Reads the tuned-merge, soft-support sweep, and merge-stage results,
runs paired bootstrap tests, and generates summary tables/CSVs.

Comparisons (6 pairs × 7 metrics):
  1. CBS vs default_consensus
  2. CBS vs tuned_merge
  3. CBS vs soft_support
  4. default_consensus vs tuned_merge
  5. default_consensus vs soft_support
  6. tuned_merge vs soft_support

Metrics: F1, accuracy, worst_action_recall, n_rules, E_CR, GRS_ta, BRA

Output:
    experiments/results/merge_statistics/paired_bootstrap.json
    experiments/results/merge_statistics/main_results_table.csv
    experiments/results/merge_statistics/repair_ladder_table.csv
    experiments/results/merge_statistics/crossing_distortion_table.csv
"""
import json
import os
import sys
import csv

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ENVS = ["MountainCar-v0", "CartPole-v1", "LunarLander-v3"]
ENV_TAGS = {e: e.replace("-", "_").lower() for e in ENVS}
N_BOOT = 1000
ALPHA = 0.05
OUT_DIR = "experiments/results/merge_statistics"


def paired_bootstrap_test(vals_a, vals_b, n_boot=N_BOOT, seed=42):
    """Paired bootstrap test: is B better than A?
    Returns dict with obs_diff, ci_lo, ci_hi, p_value, significant."""
    a = np.array(vals_a)
    b = np.array(vals_b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    diffs = b - a
    obs_diff = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_diffs.append(float(np.mean(diffs[idx])))
    boot_diffs = np.array(boot_diffs)

    ci_lo = float(np.percentile(boot_diffs, 100 * ALPHA / 2))
    ci_hi = float(np.percentile(boot_diffs, 100 * (1 - ALPHA / 2)))

    # Two-sided p-value
    if obs_diff >= 0:
        p = 2 * np.mean(boot_diffs <= 0)
    else:
        p = 2 * np.mean(boot_diffs >= 0)
    p = min(float(p), 1.0)

    return {
        "obs_diff": obs_diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p,
        "significant": p < ALPHA,
    }


def load_b2_raw(env_tag):
    """Load the soft-support sweep raw_runs.json and extract per-seed metrics for each method."""
    path = f"experiments/results/soft_support_sweep/{env_tag}/raw_runs.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)

    # Organize: method_name -> list of per-seed metric dicts
    methods = {}
    for run in data:
        name = run.get("method") or run.get("config_label", "unknown")
        if name not in methods:
            methods[name] = []
        methods[name].append(run)

    return methods


def load_b3_repair_ladder(env_tag):
    """Load the repair-ladder per-seed data."""
    path = f"experiments/results/merge_stages/{env_tag}/repair_ladder.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data


def load_b3_support_comparison(env_tag):
    """Load the support-comparison data."""
    path = f"experiments/results/merge_stages/{env_tag}/support_comparison.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_b3_boundary_crossing(env_tag):
    path = f"experiments/results/merge_stages/{env_tag}/boundary_crossing.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_b3_geometric_distortion(env_tag):
    path = f"experiments/results/merge_stages/{env_tag}/geometric_distortion.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_b3_failure_decomposition(env_tag):
    path = f"experiments/results/merge_stages/{env_tag}/failure_decomposition.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract_method_metrics(b2_data, method_key, metric_name):
    """Extract a list of per-seed metric values from the soft-support sweep."""
    if b2_data is None or method_key not in b2_data:
        return []
    runs = b2_data[method_key]
    vals = []
    for r in runs:
        v = r.get(metric_name)
        if v is None:
            # Try nested
            for sub in ["metrics", "fidelity", "deployment"]:
                if sub in r and metric_name in r[sub]:
                    v = r[sub][metric_name]
                    break
        if v is not None:
            vals.append(float(v))
    return vals


def extract_ladder_metrics(ladder_data, method_name, metric_name):
    """Extract per-seed values from repair ladder data."""
    if ladder_data is None:
        return []
    per_seed = ladder_data.get("per_seed", {})
    vals = []
    for seed_key in sorted(per_seed.keys()):
        seed_data = per_seed[seed_key]
        if method_name in seed_data and metric_name in seed_data[method_name]:
            vals.append(float(seed_data[method_name][metric_name]))
    return vals


def run_all_statistics():
    """Run paired bootstrap tests across all envs and method pairs."""
    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = {}

    metrics = ["f1", "worst_action_recall", "n_rules", "E_CR"]
    method_pairs = [
        ("default_consensus", "tuned_merge"),
        ("default_consensus", "soft_support"),
        ("tuned_merge", "soft_support"),
    ]

    for env_name in ENVS:
        env_tag = ENV_TAGS[env_name]
        print(f"\n=== {env_name} ===")

        ladder = load_b3_repair_ladder(env_tag)
        if ladder is None:
            print(f"  WARNING: No repair ladder data for {env_name}")
            continue

        env_results = {}

        for m_a, m_b in method_pairs:
            pair_key = f"{m_a}_vs_{m_b}"
            env_results[pair_key] = {}

            for metric in metrics:
                vals_a = extract_ladder_metrics(ladder, m_a, metric)
                vals_b = extract_ladder_metrics(ladder, m_b, metric)

                if len(vals_a) < 2 or len(vals_b) < 2:
                    env_results[pair_key][metric] = {
                        "error": "insufficient data",
                        "n_a": len(vals_a), "n_b": len(vals_b),
                    }
                    continue

                test = paired_bootstrap_test(vals_a, vals_b)
                env_results[pair_key][metric] = test
                sig = "*" if test["significant"] else ""
                print(f"  {pair_key} | {metric}: "
                      f"diff={test['obs_diff']:+.4f} "
                      f"CI=[{test['ci_lo']:.4f},{test['ci_hi']:.4f}] "
                      f"p={test['p_value']:.4f}{sig}")

        all_results[env_name] = env_results

    # Save
    with open(os.path.join(OUT_DIR, "paired_bootstrap.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved paired_bootstrap.json")

    return all_results


def generate_main_results_table():
    """Generate the main results table CSV from the repair-ladder data."""
    rows = []
    header = ["Environment", "Method", "F1_mean", "F1_std",
              "worst_R_mean", "worst_R_std",
              "n_rules_mean", "n_rules_std",
              "E_CR_mean", "E_CR_std"]

    for env_name in ENVS:
        env_tag = ENV_TAGS[env_name]
        ladder = load_b3_repair_ladder(env_tag)
        if ladder is None:
            continue

        summary = ladder.get("summary", {})
        for method in ["default_consensus", "tuned_merge", "soft_support"]:
            s = summary.get(method, {})
            rows.append([
                env_name, method,
                f"{s.get('f1', {}).get('mean', 0):.4f}",
                f"{s.get('f1', {}).get('std', 0):.4f}",
                f"{s.get('worst_action_recall', {}).get('mean', 0):.4f}",
                f"{s.get('worst_action_recall', {}).get('std', 0):.4f}",
                f"{s.get('n_rules', {}).get('mean', 0):.1f}",
                f"{s.get('n_rules', {}).get('std', 0):.1f}",
                f"{s.get('E_CR', {}).get('mean', 0):.1f}",
                f"{s.get('E_CR', {}).get('std', 0):.1f}",
            ])

    path = os.path.join(OUT_DIR, "main_results_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Saved main_results_table.csv ({len(rows)} rows)")


def generate_repair_ladder_table():
    """Generate the repair-ladder table from the failure decomposition."""
    rows = []
    header = ["Environment", "Stage", "F1_mean", "F1_std",
              "worst_R_mean", "n_rules_mean", "surviving_groups_mean"]

    stages = ["match_only", "match_hard_support", "match_aggregation",
              "full_default", "v2_soft_support"]

    for env_name in ENVS:
        env_tag = ENV_TAGS[env_name]
        fd = load_b3_failure_decomposition(env_tag)
        if fd is None:
            continue

        summary = fd.get("summary", {})
        for stage in stages:
            s = summary.get(stage, {})
            rows.append([
                env_name, stage,
                f"{s.get('f1', {}).get('mean', 0):.4f}",
                f"{s.get('f1', {}).get('std', 0):.4f}",
                f"{s.get('worst_action_recall', {}).get('mean', 0):.4f}",
                f"{s.get('n_rules', {}).get('mean', 0):.1f}",
                f"{s.get('surviving_groups', {}).get('mean', 0):.1f}",
            ])

    path = os.path.join(OUT_DIR, "repair_ladder_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Saved repair_ladder_table.csv ({len(rows)} rows)")


def generate_crossing_distortion_table():
    """Generate crossing+distortion summary table."""
    rows = []
    header = ["Environment", "Method",
              "crossing_pct_mean", "crossing_pct_std",
              "midpoint_mismatch_mean",
              "failed_merge_frac_mean", "failed_merge_frac_std",
              "multimodal_frac_mean", "fragmented_frac_mean"]

    for env_name in ENVS:
        env_tag = ENV_TAGS[env_name]
        bc = load_b3_boundary_crossing(env_tag)
        gd = load_b3_geometric_distortion(env_tag)

        for method in ["default_consensus", "tuned_merge", "soft_support"]:
            bc_summ = bc.get(f"{method}_summary", {}) if bc else {}
            gd_summ = gd.get(f"{method}_summary", {}) if gd else {}

            rows.append([
                env_name, method,
                f"{bc_summ.get('mergeable_crossing_pct', {}).get('mean', 0):.4f}",
                f"{bc_summ.get('mergeable_crossing_pct', {}).get('std', 0):.4f}",
                f"{bc_summ.get('midpoint_mismatch_pct', {}).get('mean', 0):.4f}",
                f"{gd_summ.get('failed_merge_frac', {}).get('mean', 0):.4f}",
                f"{gd_summ.get('failed_merge_frac', {}).get('std', 0):.4f}",
                f"{gd_summ.get('multimodal_frac', {}).get('mean', 0):.4f}",
                f"{gd_summ.get('fragmented_frac', {}).get('mean', 0):.4f}",
            ])

    path = os.path.join(OUT_DIR, "crossing_distortion_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Saved crossing_distortion_table.csv ({len(rows)} rows)")


def generate_soft_support_ablation_table():
    """Generate the soft-support ablation table."""
    rows = []
    header = ["Environment", "Config", "F1_mean", "F1_std",
              "worst_R_mean", "n_rules_mean", "E_CR_mean"]

    for env_name in ENVS:
        env_tag = ENV_TAGS[env_name]
        summary_path = f"experiments/results/soft_support_sweep/{env_tag}/summary.json"
        if not os.path.exists(summary_path):
            continue

        with open(summary_path) as f:
            data = json.load(f)

        # V2 cells
        v2_summary = data.get("v2_summary", {})
        for cell_name, cell_data in sorted(v2_summary.items()):
            f1 = cell_data.get("F1", {})
            rows.append([
                env_name, cell_name,
                f"{f1.get('mean', 0):.4f}",
                f"{f1.get('std', 0):.4f}",
                f"{cell_data.get('worst_action_recall', {}).get('mean', 0):.4f}",
                f"{cell_data.get('n_rules', {}).get('mean', 0):.1f}",
                f"{cell_data.get('E_CR', {}).get('mean', 0):.1f}",
            ])

    path = os.path.join(OUT_DIR, "soft_support_ablation_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Saved soft_support_ablation_table.csv ({len(rows)} rows)")


def generate_support_comparison_table():
    """Generate support hard vs soft comparison table."""
    rows = []
    header = ["Environment", "Mode", "F1_mean", "F1_std",
              "worst_R_mean", "n_rules_mean",
              "kept_groups_mean", "dropped_groups_mean"]

    for env_name in ENVS:
        env_tag = ENV_TAGS[env_name]
        sc = load_b3_support_comparison(env_tag)
        if sc is None:
            continue

        for mode in ["hard", "soft"]:
            summ = sc.get(f"{mode}_summary", {})
            rows.append([
                env_name, mode,
                f"{summ.get('f1', {}).get('mean', 0):.4f}",
                f"{summ.get('f1', {}).get('std', 0):.4f}",
                f"{summ.get('worst_action_recall', {}).get('mean', 0):.4f}",
                f"{summ.get('n_rules', {}).get('mean', 0):.1f}",
                f"{summ.get('kept_groups', {}).get('mean', 0):.1f}",
                f"{summ.get('dropped_groups', {}).get('mean', 0):.1f}",
            ])

    path = os.path.join(OUT_DIR, "support_comparison_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Saved support_comparison_table.csv ({len(rows)} rows)")


def main():
    print("=" * 60)
    print("  STATISTICS & TABLES")
    print("=" * 60)

    # Run paired bootstrap tests
    run_all_statistics()

    # Generate tables
    print("\n--- Generating tables ---")
    generate_main_results_table()
    generate_repair_ladder_table()
    generate_crossing_distortion_table()
    generate_soft_support_ablation_table()
    generate_support_comparison_table()

    print("\n" + "=" * 60)
    print("  Statistics & tables complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
