#!/usr/bin/env python
"""
Formal Statistical Confirmation

Bootstrap 95% CIs for all main-table metrics per method x environment,
plus paired bootstrap tests for core method comparisons.

Usage:
    python experiments/run_statistical_tests.py
    python experiments/run_statistical_tests.py --n-bootstrap 5000

Output:
    experiments/results/statistical_tests.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────
ENVS = ["MountainCar-v0", "CartPole-v1", "LunarLander-v3"]
ENV_TAGS = {
    "MountainCar-v0": "mountaincar_v0",
    "CartPole-v1": "cartpole_v1",
    "LunarLander-v3": "lunarlander_v3",
}
RESULTS_DIR = "experiments/results"


# ── Data loading ──────────────────────────────────────────────────────

def load_results(env_name):
    """Load all result JSONs for an environment."""
    tag = ENV_TAGS[env_name]
    base = os.path.join(RESULTS_DIR, tag)
    results = {}
    for fname, key in [
        ("stress_test_results.json", "stress_test"),
        ("consensus_merge_results.json", "consensus"),
        ("decision_tree_results.json", "b4"),
    ]:
        path = os.path.join(base, fname)
        if os.path.exists(path):
            with open(path) as f:
                results[key] = json.load(f)
    return results


def extract_per_run_f1(per_run_data):
    """Extract F1 values from per-run data dict."""
    return np.array([run["fidelity_heldout"]["f1"] for run in per_run_data.values()])


def extract_per_run_ecr(per_run_data):
    """Extract E_CR values from per-run data dict."""
    return np.array([run["deployment"]["E_CR"] for run in per_run_data.values()])


def extract_per_run_metric(per_run_data, metric_path):
    """Extract arbitrary metric from per-run data.
    metric_path: tuple of keys, e.g. ('fidelity_heldout', 'f1')
    """
    values = []
    for run in per_run_data.values():
        v = run
        for k in metric_path:
            v = v[k]
        values.append(v)
    return np.array(values)


def get_method_per_run(results, method_key):
    """Get per-run data for a method from loaded results."""
    if method_key == "cbs":
        return results["stress_test"]["cbs"]["per_run"]
    elif method_key == "cbs_maxf1":
        return results["stress_test"]["cbs_maxf1"]["per_run"]
    elif method_key == "b3_vote":
        return results["consensus"]["consensus_vote"]["per_run"]
    elif method_key == "consensus_cbs":
        return results["consensus"]["consensus_cbs"]["per_run"]
    elif method_key == "dt":
        return results["b4"]["b4_dt"]["per_run"]
    else:
        raise ValueError(f"Unknown method: {method_key}")


def get_method_stability(results, method_key):
    """Get group-level stability metrics for a method."""
    if method_key == "cbs":
        return results["stress_test"]["cbs"]["stability"]
    elif method_key == "cbs_maxf1":
        return results["stress_test"]["cbs_maxf1"]["stability"]
    elif method_key == "b3_vote":
        return results["consensus"]["consensus_vote"]["stability"]
    elif method_key == "consensus_cbs":
        return results["consensus"]["consensus_cbs"]["stability"]
    elif method_key == "dt":
        return results["b4"]["b4_dt"]["stability"]
    else:
        raise ValueError(f"Unknown method: {method_key}")


# ── Bootstrap utilities ───────────────────────────────────────────────

def bootstrap_ci(values, n_bootstrap=1000, ci=0.95, statistic=np.mean, seed=42):
    """Compute bootstrap confidence interval for a statistic.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    point = float(statistic(values))

    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_stats[i] = statistic(values[idx])

    alpha = (1 - ci) / 2
    ci_lower = float(np.percentile(boot_stats, 100 * alpha))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha)))
    return point, ci_lower, ci_upper


def paired_bootstrap_test(values_a, values_b, n_bootstrap=1000, seed=42):
    """Paired bootstrap test for difference in means.

    Tests H0: mean(A) == mean(B).
    Returns (mean_diff, ci_lower, ci_upper, p_value).
    The sign convention: diff = mean(A) - mean(B).
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(values_a)
    b = np.asarray(values_b)
    assert len(a) == len(b), "Paired test requires equal lengths"
    n = len(a)

    diffs = a - b
    observed_diff = float(np.mean(diffs))

    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = np.mean(diffs[idx])

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    # Two-sided p-value: fraction of bootstrap diffs on opposite side of zero
    if observed_diff >= 0:
        p_value = float(np.mean(boot_diffs <= 0)) * 2
    else:
        p_value = float(np.mean(boot_diffs >= 0)) * 2
    p_value = min(p_value, 1.0)

    return observed_diff, ci_lower, ci_upper, p_value


def bootstrap_stability_ci(per_run_data, metric_key, n_bootstrap=1000, seed=42):
    """Bootstrap CI for a group-level stability metric.

    Since stability metrics are computed from pairwise comparisons,
    we bootstrap over runs: resample the set of runs, then recompute
    the pairwise metric on the resampled set.

    For simplicity, we bootstrap the per-run stability proxies (leave-one-out
    averages) which are already per-run scalars.
    """
    proxy_key = "stability_proxy_global"
    values = []
    for run in per_run_data.values():
        if proxy_key in run and run[proxy_key] is not None:
            val = run[proxy_key].get(metric_key)
            if val is not None:
                values.append(val)
    if len(values) < 3:
        return None, None, None
    return bootstrap_ci(values, n_bootstrap=n_bootstrap, seed=seed)


# ── Main analysis ─────────────────────────────────────────────────────

def compute_per_method_cis(results, env_name, methods, n_bootstrap):
    """Compute bootstrap CIs for all metrics per method."""
    out = {}
    for method in methods:
        try:
            per_run = get_method_per_run(results, method)
        except (KeyError, TypeError):
            continue

        f1_vals = extract_per_run_f1(per_run)
        ecr_vals = extract_per_run_ecr(per_run)

        method_cis = {}
        # F1
        mean, lo, hi = bootstrap_ci(f1_vals, n_bootstrap)
        method_cis["f1"] = {"mean": mean, "ci_lower": lo, "ci_upper": hi,
                            "std": float(np.std(f1_vals)), "n": len(f1_vals)}
        # E_CR
        mean, lo, hi = bootstrap_ci(ecr_vals, n_bootstrap)
        method_cis["e_cr"] = {"mean": mean, "ci_lower": lo, "ci_upper": hi,
                              "std": float(np.std(ecr_vals)), "n": len(ecr_vals)}
        # N rules
        n_rules = np.array([run.get("n_rules", 0) for run in per_run.values()])
        mean, lo, hi = bootstrap_ci(n_rules, n_bootstrap)
        method_cis["n_rules"] = {"mean": mean, "ci_lower": lo, "ci_upper": hi}

        # Stability proxies (bootstrap over per-run leave-one-out proxies)
        for stab_key, stab_label in [
            ("GRS_wj", "grs_wj"), ("GRS_ta", "grs_ta"),
            ("BRA", "bra"), ("TD", "td"),
        ]:
            result = bootstrap_stability_ci(per_run, stab_key, n_bootstrap)
            if result[0] is not None:
                method_cis[stab_label] = {
                    "mean": result[0], "ci_lower": result[1], "ci_upper": result[2]
                }

        # Also report group-level stability as point estimates
        try:
            stab = get_method_stability(results, method)
            method_cis["group_stability"] = {
                "GRS_wj": stab.get("GRS_weighted_jaccard"),
                "GRS_ta": stab.get("GRS_threshold_aware"),
                "BRA": stab.get("BRA"),
                "TD": stab.get("TD"),
            }
        except (KeyError, TypeError):
            pass

        out[method] = method_cis

    return out


def compute_method_comparisons(results, env_name, n_bootstrap):
    """Compute paired bootstrap tests for core method comparisons."""
    comparisons = []

    # Define comparisons: (method_a, method_b, metric, direction_hypothesis)
    comparison_specs = [
        # H4: MaxF1 hurts stability
        ("cbs", "cbs_maxf1", "stability_proxy_global.GRS_wj",
         "CBS GRS > FT-CBS GRS (FT-CBS hurts structural stability)"),
        ("cbs", "cbs_maxf1", "stability_proxy_global.BRA",
         "CBS BRA > FT-CBS BRA (FT-CBS hurts behavioral stability)"),
        # RuleVote improves BRA
        ("b3_vote", "cbs", "stability_proxy_global.BRA",
         "RV BRA > CBS BRA (RuleVote improves behavioral stability)"),
        # RuleVote fidelity
        ("b3_vote", "cbs", "fidelity_heldout.f1",
         "RV Macro-F1 vs CBS Macro-F1"),
        # DT vs CBS
        ("dt", "cbs", "fidelity_heldout.f1",
         "DT Macro-F1 > CBS Macro-F1 (DT has higher fidelity)"),
        ("cbs", "dt", "stability_proxy_global.GRS_wj",
         "CBS GRS > DT GRS (DT has lower structural stability)"),
        ("dt", "cbs", "stability_proxy_global.BRA",
         "DT BRA > CBS BRA (DT has higher behavioral stability)"),
    ]

    for method_a, method_b, metric_path_str, hypothesis in comparison_specs:
        try:
            per_run_a = get_method_per_run(results, method_a)
            per_run_b = get_method_per_run(results, method_b)
        except (KeyError, TypeError):
            continue

        # Extract metric values aligned by perturbation run
        # Both methods use the same perturbation IDs, so we align by sorted keys
        keys_a = sorted(per_run_a.keys())
        keys_b = sorted(per_run_b.keys())

        # If different number of runs, use the smaller set
        n = min(len(keys_a), len(keys_b))
        keys_a = keys_a[:n]
        keys_b = keys_b[:n]

        path_parts = metric_path_str.split(".")

        def extract_val(run_data, parts):
            v = run_data
            for p in parts:
                if v is None:
                    return None
                v = v.get(p) if isinstance(v, dict) else None
            return v

        vals_a = []
        vals_b = []
        for ka, kb in zip(keys_a, keys_b):
            va = extract_val(per_run_a[ka], path_parts)
            vb = extract_val(per_run_b[kb], path_parts)
            if va is not None and vb is not None:
                vals_a.append(va)
                vals_b.append(vb)

        if len(vals_a) < 5:
            continue

        vals_a = np.array(vals_a)
        vals_b = np.array(vals_b)

        diff, ci_lo, ci_hi, p_val = paired_bootstrap_test(vals_a, vals_b, n_bootstrap)

        comparisons.append({
            "method_a": method_a,
            "method_b": method_b,
            "metric": metric_path_str,
            "hypothesis": hypothesis,
            "mean_a": float(np.mean(vals_a)),
            "mean_b": float(np.mean(vals_b)),
            "mean_diff": diff,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "p_value": p_val,
            "significant_005": p_val < 0.05,
            "significant_001": p_val < 0.01,
            "n_paired": len(vals_a),
        })

    return comparisons


def summarize_core_claims(all_comparisons):
    """Summarize the statistical evidence for each core paper claim."""
    claims = []

    claim_map = {
        "FT-CBS decreases structural stability": [
            ("CBS GRS > FT-CBS GRS", "stability_proxy_global.GRS_wj"),
        ],
        "FT-CBS decreases behavioral stability": [
            ("CBS BRA > FT-CBS BRA", "stability_proxy_global.BRA"),
        ],
        "RuleVote (RV) improves behavioral stability over CBS": [
            ("RV BRA > CBS BRA", "stability_proxy_global.BRA"),
        ],
        "DT has highest Macro-F1": [
            ("DT Macro-F1 > CBS Macro-F1", "fidelity_heldout.f1"),
        ],
        "DT has lowest structural stability": [
            ("CBS GRS > DT GRS", "stability_proxy_global.GRS_wj"),
        ],
        "DT has highest behavioral stability": [
            ("DT BRA > CBS BRA", "stability_proxy_global.BRA"),
        ],
    }

    for claim_text, specs in claim_map.items():
        evidence = []
        for hyp_prefix, metric in specs:
            # Find matching comparisons across all envs
            for env_name, env_comps in all_comparisons.items():
                for comp in env_comps:
                    if comp["metric"] == metric and hyp_prefix in comp["hypothesis"]:
                        evidence.append({
                            "env": env_name,
                            "diff": comp["mean_diff"],
                            "ci": [comp["ci_lower"], comp["ci_upper"]],
                            "p": comp["p_value"],
                            "sig_005": comp["significant_005"],
                        })

        n_sig = sum(1 for e in evidence if e["sig_005"])
        n_total = len(evidence)
        claims.append({
            "claim": claim_text,
            "n_environments_significant": n_sig,
            "n_environments_tested": n_total,
            "supported": n_sig >= max(1, n_total // 2),
            "evidence": evidence,
        })

    return claims


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Formal statistical confirmation")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    n_boot = args.n_bootstrap
    print(f"=== Formal Statistical Confirmation ===")
    print(f"Bootstrap resamples: {n_boot}")
    t0 = time.time()

    methods = ["cbs", "cbs_maxf1", "b3_vote", "consensus_cbs", "dt"]

    all_cis = {}
    all_comparisons = {}

    for env_name in ENVS:
        print(f"\n--- {env_name} ---")
        results = load_results(env_name)
        if not results:
            print(f"  No results found, skipping")
            continue

        # Per-method CIs
        cis = compute_per_method_cis(results, env_name, methods, n_boot)
        all_cis[env_name] = cis

        for method, method_cis in cis.items():
            if "f1" in method_cis:
                f1 = method_cis["f1"]
                print(f"  {method:15s} F1={f1['mean']:.3f} "
                      f"[{f1['ci_lower']:.3f}, {f1['ci_upper']:.3f}]", end="")
            if "grs_wj" in method_cis:
                grs = method_cis["grs_wj"]
                print(f"  GRS={grs['mean']:.3f} "
                      f"[{grs['ci_lower']:.3f}, {grs['ci_upper']:.3f}]", end="")
            if "bra" in method_cis:
                bra = method_cis["bra"]
                print(f"  BRA={bra['mean']:.3f} "
                      f"[{bra['ci_lower']:.3f}, {bra['ci_upper']:.3f}]", end="")
            print()

        # Method comparisons
        comps = compute_method_comparisons(results, env_name, n_boot)
        all_comparisons[env_name] = comps

        print(f"\n  Paired bootstrap tests (n_bootstrap={n_boot}):")
        for comp in comps:
            sig_marker = "***" if comp["significant_001"] else ("**" if comp["significant_005"] else "ns")
            print(f"    {comp['hypothesis'][:65]:65s} "
                  f"Δ={comp['mean_diff']:+.4f} "
                  f"[{comp['ci_lower']:+.4f}, {comp['ci_upper']:+.4f}] "
                  f"p={comp['p_value']:.4f} {sig_marker}")

    # Summarize core claims
    print("\n=== Core Claims Summary ===")
    claims = summarize_core_claims(all_comparisons)
    for claim in claims:
        status = "SUPPORTED" if claim["supported"] else "NOT SUPPORTED"
        print(f"  [{status}] {claim['claim']} "
              f"({claim['n_environments_significant']}/{claim['n_environments_tested']} envs significant)")
        for ev in claim["evidence"]:
            sig = "*" if ev["sig_005"] else ""
            print(f"    {ev['env']:20s} Δ={ev['diff']:+.4f} "
                  f"CI=[{ev['ci'][0]:+.4f}, {ev['ci'][1]:+.4f}] p={ev['p']:.4f}{sig}")

    # Save results
    elapsed = time.time() - t0
    output = {
        "schema_version": "statistical_v1",
        "n_bootstrap": n_boot,
        "confidence_level": 0.95,
        "elapsed_seconds": round(elapsed, 1),
        "per_method_cis": all_cis,
        "method_comparisons": {k: v for k, v in all_comparisons.items()},
        "core_claims": claims,
    }

    out_path = os.path.join(RESULTS_DIR, "statistical_tests.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
