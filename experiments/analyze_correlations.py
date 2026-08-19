#!/usr/bin/env python
"""
Fidelity-Stability Correlation Analysis

Loads stress test results, computes Spearman correlations between
fidelity and stability metrics at run level, generates scatter plots
and Pareto frontier visualizations.

Supports three stability source modes:
  - group:      old behavior (shared per method x env)
  - run_global: per-run leave-one-out across all runs in group
  - run_family: per-run leave-one-out within perturbation family

Usage:
    python experiments/analyze_correlations.py
    python experiments/analyze_correlations.py --stability-source run_family
    python experiments/analyze_correlations.py --stability-source group

Output:
    experiments/results/correlation_analysis.json
    experiments/results/correlation_plots/stability_vs_fidelity.png
    experiments/results/correlation_plots/pareto_frontier.png
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Configuration ─────────────────────────────────────────────────────

ENVS = ["MountainCar-v0", "CartPole-v1", "LunarLander-v3"]
METHODS = ["cbs", "cbs_maxf1"]
CONSENSUS_METHODS = ["consensus_cbs", "consensus_vote"]
METHOD_LABELS = {
    "cbs": "CBS", "cbs_maxf1": "CBS+MaxF1",
    "consensus_cbs": "Consensus CBS", "consensus_vote": "B3-vote",
}
ENV_LABELS = {
    "MountainCar-v0": "MountainCar",
    "CartPole-v1": "CartPole",
    "LunarLander-v3": "LunarLander",
}

# Perturbation families and their run key prefixes
FAMILIES = {
    "seed_shift": "seed_shift_s",
    "subsample": "subsample_",
    "stratified": "stratified_",
    "cluster_count": "cluster_delta_",
    "feature_noise": "noise_",
}

RESULTS_DIR = "experiments/results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "correlation_plots")


def load_stress_test(env_name):
    """Load stress test results JSON for one environment."""
    tag = env_name.replace("-", "_").lower()
    path = os.path.join(RESULTS_DIR, tag, "stress_test_results.json")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping {env_name}")
        return None
    with open(path) as f:
        return json.load(f)


def load_consensus_results(env_name):
    """Load consensus CBS results JSON for one environment."""
    tag = env_name.replace("-", "_").lower()
    path = os.path.join(RESULTS_DIR, tag, "consensus_merge_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract_per_family_data(data, method, stability_source="run_family"):
    """Extract per-perturbation-family data points from stress test results.

    stability_source controls where stability metrics come from:
      - "group":      shared per method x env (v1 behavior)
      - "run_global":  per-run proxy across all runs in group
      - "run_family":  per-run proxy within perturbation family
    """
    per_run = data[method]["per_run"]
    group_stability = data[method].get("stability", {})
    families = {}

    for family_name, prefix in FAMILIES.items():
        runs = []
        for key, run_data in per_run.items():
            if key.startswith(prefix):
                # Resolve stability metrics based on source
                if stability_source == "group":
                    stab = group_stability
                    stab_grs_wj = stab.get("GRS_weighted_jaccard")
                    stab_grs_ta = stab.get("GRS_threshold_aware")
                    stab_bra = stab.get("BRA")
                    stab_td = stab.get("TD")
                elif stability_source == "run_global":
                    proxy = run_data.get("stability_proxy_global", {})
                    if not proxy:
                        proxy = group_stability
                        stab_grs_wj = proxy.get("GRS_weighted_jaccard")
                        stab_grs_ta = proxy.get("GRS_threshold_aware")
                    else:
                        stab_grs_wj = proxy.get("GRS_wj")
                        stab_grs_ta = proxy.get("GRS_ta")
                    stab_bra = proxy.get("BRA")
                    stab_td = proxy.get("TD")
                else:  # run_family
                    proxy = run_data.get("stability_proxy_family")
                    if not proxy:
                        proxy = run_data.get("stability_proxy_global", {})
                    if not proxy:
                        proxy = group_stability
                        stab_grs_wj = proxy.get("GRS_weighted_jaccard")
                        stab_grs_ta = proxy.get("GRS_threshold_aware")
                    else:
                        stab_grs_wj = proxy.get("GRS_wj")
                        stab_grs_ta = proxy.get("GRS_ta")
                    stab_bra = proxy.get("BRA")
                    stab_td = proxy.get("TD")

                runs.append({
                    "key": key,
                    "f1": run_data["fidelity_heldout"]["f1"],
                    "ecr": run_data["deployment"]["E_CR"],
                    "success_rate": run_data["deployment"].get("success_rate"),
                    "n_rules": run_data["n_rules"],
                    "grs_wj": stab_grs_wj,
                    "grs_ta": stab_grs_ta,
                    "bra": stab_bra,
                    "td": stab_td,
                })
        if runs:
            families[family_name] = runs

    return families


def compute_family_summary(families):
    """Compute mean F1 and E_CR per perturbation family."""
    summaries = []
    for family_name, runs in families.items():
        f1_vals = [r["f1"] for r in runs]
        ecr_vals = [r["ecr"] for r in runs]
        summaries.append({
            "family": family_name,
            "n_runs": len(runs),
            "mean_f1": float(np.mean(f1_vals)),
            "std_f1": float(np.std(f1_vals)),
            "mean_ecr": float(np.mean(ecr_vals)),
            "std_ecr": float(np.std(ecr_vals)),
        })
    return summaries


def compute_correlations(all_points):
    """Compute Spearman correlations between metric pairs.

    all_points: list of dicts with keys like 'f1', 'ecr', 'grs', 'bra', etc.
    """
    results = {}
    metric_pairs = [
        ("f1", "grs_wj", "F1 vs GRS (weighted Jaccard)"),
        ("f1", "grs_ta", "F1 vs GRS-TA (threshold-aware)"),
        ("f1", "bra", "F1 vs BRA"),
        ("ecr", "grs_wj", "E_CR vs GRS (weighted Jaccard)"),
        ("ecr", "bra", "E_CR vs BRA"),
        ("f1", "td", "F1 vs TD (threshold drift)"),
    ]

    for x_key, y_key, label in metric_pairs:
        x_vals = [p[x_key] for p in all_points if p.get(x_key) is not None and p.get(y_key) is not None]
        y_vals = [p[y_key] for p in all_points if p.get(x_key) is not None and p.get(y_key) is not None]

        if len(x_vals) < 3:
            results[label] = {"rho": None, "p_value": None, "n": len(x_vals),
                              "note": "insufficient data points"}
            continue

        rho, p_val = stats.spearmanr(x_vals, y_vals)
        results[label] = {
            "rho": float(rho),
            "p_value": float(p_val),
            "n": len(x_vals),
        }

    return results


def compute_pareto_front(points_x, points_y, maximize_x=True, maximize_y=True):
    """Compute Pareto-optimal indices (both objectives to maximize by default)."""
    n = len(points_x)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is at least as good in both and strictly better in one
            if maximize_x and maximize_y:
                dominates = (points_x[j] >= points_x[i] and points_y[j] >= points_y[i]
                             and (points_x[j] > points_x[i] or points_y[j] > points_y[i]))
            elif maximize_x and not maximize_y:
                dominates = (points_x[j] >= points_x[i] and points_y[j] <= points_y[i]
                             and (points_x[j] > points_x[i] or points_y[j] < points_y[i]))
            else:
                dominates = False
            if dominates:
                is_pareto[i] = False
                break

    return is_pareto


def plot_stability_vs_fidelity(all_points, correlations, out_path):
    """Generate scatter plots of stability vs fidelity metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    method_colors = {"CBS": "#2196F3", "CBS+MaxF1": "#FF5722",
                     "Consensus CBS": "#4CAF50", "B3-vote": "#9C27B0"}
    env_markers = {"MountainCar": "o", "CartPole": "s", "LunarLander": "D"}

    plot_configs = [
        ("f1", "grs_wj", "F1 (Held-out)", "GRS (Weighted Jaccard)",
         "F1 vs GRS (weighted Jaccard)"),
        ("f1", "grs_ta", "F1 (Held-out)", "GRS-TA (Threshold-Aware)",
         "F1 vs GRS-TA (threshold-aware)"),
        ("f1", "bra", "F1 (Held-out)", "BRA",
         "F1 vs BRA"),
    ]

    for ax, (x_key, y_key, x_label, y_label, corr_key) in zip(axes, plot_configs):
        for point in all_points:
            if point.get(x_key) is None or point.get(y_key) is None:
                continue
            color = method_colors.get(point["method"], "gray")
            marker = env_markers.get(point["env_label"], "o")
            ax.scatter(point[x_key], point[y_key], c=color, marker=marker,
                       s=60, alpha=0.7, edgecolors="white", linewidth=0.5)

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)

        # Add correlation annotation
        corr = correlations.get(corr_key, {})
        if corr.get("rho") is not None:
            rho = corr["rho"]
            p = corr["p_value"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            ax.set_title(f"Spearman rho={rho:.3f}{sig}", fontsize=10)
        else:
            ax.set_title("Insufficient data", fontsize=10)

        ax.grid(True, alpha=0.3)

    # Legend
    handles = []
    for method, color in method_colors.items():
        handles.append(mpatches.Patch(color=color, label=method))
    for env, marker in env_markers.items():
        handles.append(plt.Line2D([0], [0], marker=marker, color="gray",
                                   label=env, markersize=8, linestyle="None"))
    axes[-1].legend(handles=handles, loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved scatter plot: {out_path}")


def plot_pareto_frontier(all_points, out_path):
    """Generate Pareto frontier plot: F1 vs GRS."""
    n_envs = len(ENV_LABELS)
    fig, axes = plt.subplots(1, n_envs, figsize=(6 * n_envs, 5), squeeze=False)
    axes = axes[0]

    method_colors = {"CBS": "#2196F3", "CBS+MaxF1": "#FF5722",
                     "Consensus CBS": "#4CAF50", "B3-vote": "#9C27B0"}

    for ax_idx, (env_name, env_label) in enumerate(ENV_LABELS.items()):
        ax = axes[ax_idx]
        env_points = [p for p in all_points if p["env"] == env_name]

        if not env_points:
            ax.set_title(f"{env_label} — No data")
            continue

        for method, color in method_colors.items():
            mp = [p for p in env_points if p["method"] == method]
            if not mp:
                continue

            f1_vals = np.array([p["f1"] for p in mp])
            grs_vals = np.array([p["grs_wj"] for p in mp])

            ax.scatter(f1_vals, grs_vals, c=color, s=60, alpha=0.7,
                       edgecolors="white", linewidth=0.5, label=method, zorder=3)

            # Pareto front
            is_pareto = compute_pareto_front(f1_vals, grs_vals)
            if np.any(is_pareto):
                pareto_f1 = f1_vals[is_pareto]
                pareto_grs = grs_vals[is_pareto]
                sort_idx = np.argsort(pareto_f1)
                ax.plot(pareto_f1[sort_idx], pareto_grs[sort_idx],
                        color=color, linewidth=2, alpha=0.5, linestyle="--",
                        zorder=2)

        ax.set_xlabel("F1 (Held-out Fidelity)", fontsize=11)
        ax.set_ylabel("GRS (Weighted Jaccard)", fontsize=11)
        ax.set_title(f"Pareto Frontier — {env_label}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved Pareto plot: {out_path}")


def _print_correlations(correlations, header=""):
    """Print a correlation table."""
    if header:
        print(f"\n  --- {header} ---")
    print(f"  {'Metric Pair':<40} {'rho':>8} {'p-value':>10} {'n':>5}")
    print(f"  {'-'*65}")
    for label, corr in correlations.items():
        if corr.get("rho") is not None:
            sig = "***" if corr["p_value"] < 0.001 else "**" if corr["p_value"] < 0.01 else "*" if corr["p_value"] < 0.05 else ""
            print(f"  {label:<40} {corr['rho']:>8.4f} {corr['p_value']:>10.4f} {corr['n']:>5} {sig}")
        else:
            print(f"  {label:<40} {'N/A':>8} {'N/A':>10} {corr['n']:>5}")


def collect_all_points(stability_source="run_family"):
    """Collect all per-run data points across envs and methods."""
    all_points = []

    for env_name in ENVS:
        env_label = ENV_LABELS.get(env_name, env_name)

        # Load the clustering baselines from stress_test_results.json
        data = load_stress_test(env_name)
        if data is not None:
            for method in METHODS:
                if method not in data:
                    continue
                method_label = METHOD_LABELS[method]
                families = extract_per_family_data(data, method, stability_source)
                for family_name, runs in families.items():
                    for run in runs:
                        all_points.append({
                            "env": env_name, "env_label": env_label,
                            "method": method_label, "family": family_name,
                            "run_key": run["key"], "f1": run["f1"],
                            "ecr": run["ecr"], "n_rules": run["n_rules"],
                            "grs_wj": run["grs_wj"], "grs_ta": run["grs_ta"],
                            "bra": run["bra"], "td": run["td"],
                        })

        # Load the consensus merge and rule-set voting results
        cdata = load_consensus_results(env_name)
        if cdata is not None:
            for method in CONSENSUS_METHODS:
                if method not in cdata:
                    continue
                method_label = METHOD_LABELS[method]
                families = extract_per_family_data(cdata, method, stability_source)
                for family_name, runs in families.items():
                    for run in runs:
                        all_points.append({
                            "env": env_name, "env_label": env_label,
                            "method": method_label, "family": family_name,
                            "run_key": run["key"], "f1": run["f1"],
                            "ecr": run["ecr"], "n_rules": run["n_rules"],
                            "grs_wj": run["grs_wj"], "grs_ta": run["grs_ta"],
                            "bra": run["bra"], "td": run["td"],
                        })

    return all_points


def main():
    parser = argparse.ArgumentParser(
        description="Fidelity-Stability Correlation Analysis")
    parser.add_argument(
        "--stability-source", type=str, default="run_family",
        choices=["group", "run_global", "run_family"],
        help="Source of stability metrics: "
             "group (shared per method x env), "
             "run_global (per-run, all runs), "
             "run_family (per-run, within family)")
    args = parser.parse_args()

    stability_source = args.stability_source

    print("=" * 60)
    print("  Fidelity-Stability Correlation Analysis")
    print(f"  Stability source: {stability_source}")
    print("=" * 60)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Collect all per-run data points
    all_points = collect_all_points(stability_source)

    if not all_points:
        print("  ERROR: No data points found. Run stress tests first.")
        return

    print(f"\n  Total data points: {len(all_points)}")
    print(f"  Environments: {set(p['env'] for p in all_points)}")
    print(f"  Methods: {set(p['method'] for p in all_points)}")

    # ── Pooled Spearman correlations ─────────────────────────────────
    print("\n  Computing Spearman correlations...")
    correlations = compute_correlations(all_points)
    _print_correlations(correlations,
                        f"Pooled correlations (source={stability_source})")

    # ── Per-family Spearman breakdown ────────────────────────────────
    family_names = sorted(set(p["family"] for p in all_points))
    per_family_correlations = {}
    for fam in family_names:
        fam_points = [p for p in all_points if p["family"] == fam]
        if len(fam_points) >= 3:
            fam_corr = compute_correlations(fam_points)
            per_family_correlations[fam] = fam_corr
            _print_correlations(fam_corr,
                                f"Per-family: {fam} (n={len(fam_points)})")

    # ── Side-by-side: group vs current source ────────────────────────
    if stability_source != "group":
        print("\n  === Side-by-side comparison: group vs "
              f"{stability_source} ===")
        group_points = collect_all_points("group")
        group_corr = compute_correlations(group_points)

        print(f"\n  {'Metric Pair':<40} "
              f"{'group rho':>10} {f'{stability_source} rho':>14}")
        print(f"  {'-'*68}")
        for label in correlations:
            g = group_corr.get(label, {})
            r = correlations.get(label, {})
            g_rho = f"{g['rho']:.4f}" if g.get("rho") is not None else "N/A"
            r_rho = f"{r['rho']:.4f}" if r.get("rho") is not None else "N/A"
            print(f"  {label:<40} {g_rho:>10} {r_rho:>14}")

    # ── Generate plots ───────────────────────────────────────────────
    print("\n  Generating plots...")
    suffix = f"_{stability_source}" if stability_source != "run_family" else ""
    plot_stability_vs_fidelity(
        all_points, correlations,
        os.path.join(PLOTS_DIR,
                     f"stability_vs_fidelity{suffix}.png"))
    plot_pareto_frontier(
        all_points,
        os.path.join(PLOTS_DIR,
                     f"pareto_frontier{suffix}.png"))

    # ── Save results ─────────────────────────────────────────────────
    output = {
        "stability_source": stability_source,
        "n_data_points": len(all_points),
        "envs": list(set(p["env"] for p in all_points)),
        "methods": list(set(p["method"] for p in all_points)),
        "correlations": correlations,
        "per_family_correlations": per_family_correlations,
        "per_run_summary": {
            "mean_f1": float(np.mean([p["f1"] for p in all_points])),
            "std_f1": float(np.std([p["f1"] for p in all_points])),
            "mean_ecr": float(np.mean([p["ecr"] for p in all_points])),
        },
    }

    out_path = os.path.join(RESULTS_DIR, "correlation_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
