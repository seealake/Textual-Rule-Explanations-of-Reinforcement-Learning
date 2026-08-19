#!/usr/bin/env python
"""Condition-characterization monotonicity analysis.

Analyses implemented
--------------------
**Analysis A  (geometry → boundary crossing)**
    Unit: mergeable rule pair.
    Predictors (geometric conditions from boundary-crossing pair details):
      1. path_low_density_frac  – fraction of interpolation path in low-density
      2. midpoint_low_density_rate – low-density rate at interpolation midpoints
      3. 1 − similarity          – rule dissimilarity (kNN-gap proxy)
    Outcome: boundary_crossing_rate > 0  (binary indicator)

**Analysis B  (boundary crossing → merge failure)**
    Unit: mergeable rule pair.
    Predictor: boundary_crossing_rate > 0  (binary)
    Outcome:  midpoint_action_mismatch_rate > 0.5  (post-merge disagreement)

    This is non-circular because:
    • ``is_failed_merge`` was defined by group-level action_mismatch > 0.15
      on *member* states — a different metric, unit, and threshold.
    • Here the predictor (boundary-crossing) measures DQN self-consistency
      along interpolation paths, while the outcome measures DQN vs merged-rule
      agreement at a single midpoint — conceptually and operationally distinct.

Outputs
-------
- experiments/results/condition_monotonicity_summary.csv
- experiments/results/condition_monotonicity_summary.json

Usage
-----
    python experiments/analyze_condition_monotonicity.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
OUT_CSV = RESULTS / "condition_monotonicity_summary.csv"
OUT_JSON = RESULTS / "condition_monotonicity_summary.json"

ENVS = {
    "MountainCar-v0": "mountaincar_v0",
    "CartPole-v1":    "cartpole_v1",
    "LunarLander-v3": "lunarlander_v3",
}


# ── Wilson score interval ────────────────────────────────────────────
def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score 95 % confidence interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# ── Load boundary-crossing pair data ────────────────────────────────────────────────
def load_a3_pairs() -> pd.DataFrame:
    """Return one DataFrame with all boundary-crossing pair details across environments."""
    rows = []
    for env_name, env_tag in ENVS.items():
        path = RESULTS / env_tag / "boundary_crossing.json"
        if not path.exists():
            print(f"  SKIP: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        for rec in data.get("pair_details", []):
            rec["env"] = env_name
            rows.append(rec)
    if not rows:
        print("ERROR: no boundary_crossing data found"); sys.exit(1)
    df = pd.DataFrame(rows)
    # derived columns
    df["rule_dissimilarity"] = 1.0 - df["similarity"]
    df["has_crossing"] = (df["boundary_crossing_rate"] > 0).astype(int)
    df["post_merge_fail"] = (df["midpoint_action_mismatch_rate"] > 0.5).astype(int)
    return df


# ── Binned monotonicity for one predictor ────────────────────────────
def binned_analysis(df: pd.DataFrame, predictor: str, outcome: str,
                    n_bins: int = 4, label: str | None = None):
    """Compute binned probabilities, Wilson CIs, and trend statistics."""
    label = label or predictor
    x = df[predictor].values
    y = df[outcome].values

    # quantile bins — fall back to fewer bins if ties dominate
    for nb in range(n_bins, 1, -1):
        try:
            df["_bin"] = pd.qcut(x, nb, duplicates="drop")
            if df["_bin"].nunique() >= 2:
                break
        except ValueError:
            continue
    else:
        return None  # cannot bin

    bins = sorted(df["_bin"].unique(), key=lambda iv: iv.left)
    rows = []
    for biv in bins:
        mask = df["_bin"] == biv
        n_total = int(mask.sum())
        n_pos = int(y[mask].sum())
        prob = n_pos / n_total if n_total else np.nan
        lo, hi = wilson_ci(n_pos, n_total)
        rows.append({
            "predictor": label,
            "bin": str(biv),
            "n": n_total,
            "n_positive": n_pos,
            "probability": round(prob, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
        })
    df.drop(columns=["_bin"], inplace=True)

    # trend diagnostics
    valid = np.isfinite(x) & np.isfinite(y)
    sp_rho, sp_p = stats.spearmanr(x[valid], y[valid])
    kt_tau, kt_p = stats.kendalltau(x[valid], y[valid])

    return {
        "predictor": label,
        "bins": rows,
        "spearman_rho": round(float(sp_rho), 4),
        "spearman_p": float(sp_p),
        "kendall_tau": round(float(kt_tau), 4),
        "kendall_p": float(kt_p),
        "n_total": int(valid.sum()),
    }


# ── Analysis B: 2×2 contingency ─────────────────────────────────────
def analysis_b(df_mergeable: pd.DataFrame):
    """Boundary crossing (binary) → post-merge failure (binary)."""
    a = df_mergeable["has_crossing"].values
    b = df_mergeable["post_merge_fail"].values

    table = np.zeros((2, 2), dtype=int)
    table[0, 0] = int(((a == 0) & (b == 0)).sum())
    table[0, 1] = int(((a == 0) & (b == 1)).sum())
    table[1, 0] = int(((a == 1) & (b == 0)).sum())
    table[1, 1] = int(((a == 1) & (b == 1)).sum())

    odds_ratio_res = stats.fisher_exact(table)
    if isinstance(odds_ratio_res, tuple):
        oddsratio, p_fisher = odds_ratio_res
    else:
        oddsratio = float(odds_ratio_res.statistic)
        p_fisher = float(odds_ratio_res.pvalue)
    # JSON-safe: inf → "inf"
    if np.isinf(oddsratio):
        oddsratio = float("inf")

    # probabilities by crossing status
    no_cross = df_mergeable[df_mergeable["has_crossing"] == 0]
    yes_cross = df_mergeable[df_mergeable["has_crossing"] == 1]

    def _summary(sub):
        n = len(sub)
        k = int(sub["post_merge_fail"].sum())
        p = k / n if n else np.nan
        lo, hi = wilson_ci(k, n)
        return {"n": n, "n_fail": k, "p_fail": round(p, 4),
                "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}

    return {
        "contingency_table": table.tolist(),
        "odds_ratio": "inf" if np.isinf(oddsratio) else round(float(oddsratio), 4),
        "fisher_p": float(p_fisher),
        "no_crossing": _summary(no_cross),
        "yes_crossing": _summary(yes_cross),
        "n_total": len(df_mergeable),
    }


# ── Per-environment breakdown ────────────────────────────────────────
def per_env_analysis(df: pd.DataFrame, predictors, outcome):
    """Run binned analysis per environment."""
    results = {}
    for env_name in df["env"].unique():
        sub = df[df["env"] == env_name].copy()
        env_res = {}
        for pred_col, pred_label in predictors:
            res = binned_analysis(sub, pred_col, outcome, n_bins=4, label=pred_label)
            if res is not None:
                env_res[pred_label] = res
        results[env_name] = env_res
    return results


# ── Main ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Condition-Characterization Monotonicity Analysis")
    print("=" * 60)

    df = load_a3_pairs()
    print(f"\nLoaded {len(df)} rule pairs across {df['env'].nunique()} environments")
    print(f"  Mergeable: {(df['category'] == 'mergeable').sum()}")
    print(f"  Non-mergeable: {(df['category'] == 'non_mergeable').sum()}")
    df_m = df[df["category"] == "mergeable"].copy()

    predictors_a = [
        ("path_low_density_frac",     "Path low-density frac"),
        ("midpoint_low_density_rate",  "Midpoint low-density rate"),
        ("rule_dissimilarity",         "Rule dissimilarity (1−sim)"),
    ]

    # ── Analysis A: pooled across environments ────────────────────
    print("\n── Analysis A: Geometry → Boundary Crossing (mergeable pairs only) ──")
    analysis_a_pooled = {}
    for pred_col, pred_label in predictors_a:
        res = binned_analysis(df_m, pred_col, "has_crossing",
                              n_bins=4, label=pred_label)
        if res is None:
            print(f"  {pred_label}: could not bin"); continue
        analysis_a_pooled[pred_label] = res
        print(f"\n  {pred_label}:")
        print(f"    Spearman ρ = {res['spearman_rho']:.3f}  (p = {res['spearman_p']:.2e})")
        print(f"    Kendall  τ = {res['kendall_tau']:.3f}  (p = {res['kendall_p']:.2e})")
        for b in res["bins"]:
            print(f"    {b['bin']:>24s}  n={b['n']:3d}  P={b['probability']:.3f}"
                  f"  [{b['ci_lo']:.3f}, {b['ci_hi']:.3f}]")

    # ── Analysis A: per environment ───────────────────────────────
    print("\n── Analysis A: per-environment breakdown ──")
    analysis_a_per_env = per_env_analysis(df_m, predictors_a, "has_crossing")
    for env_name, env_res in analysis_a_per_env.items():
        print(f"\n  {env_name}:")
        for pred_label, res in env_res.items():
            print(f"    {pred_label}: ρ={res['spearman_rho']:.3f}, "
                  f"τ={res['kendall_tau']:.3f}")

    # ── Analysis B: Boundary crossing → merge failure (mergeable only) ──
    print("\n── Analysis B: Boundary Crossing → Post-Merge Failure ──")
    ab_pooled = analysis_b(df_m)
    print(f"  N = {ab_pooled['n_total']}")
    nc = ab_pooled["no_crossing"]
    yc = ab_pooled["yes_crossing"]
    print(f"  No crossing:  P(fail) = {nc['p_fail']:.3f}  "
          f"[{nc['ci_lo']:.3f}, {nc['ci_hi']:.3f}]  n={nc['n']}")
    print(f"  Yes crossing: P(fail) = {yc['p_fail']:.3f}  "
          f"[{yc['ci_lo']:.3f}, {yc['ci_hi']:.3f}]  n={yc['n']}")
    print(f"  Odds ratio = {ab_pooled['odds_ratio']}")
    print(f"  Fisher p   = {ab_pooled['fisher_p']:.4f}")

    # ── Analysis B: per environment ──
    ab_per_env = {}
    for env_name in df_m["env"].unique():
        sub = df_m[df_m["env"] == env_name].copy()
        ab_per_env[env_name] = analysis_b(sub)
        res = ab_per_env[env_name]
        nc_e = res["no_crossing"]
        yc_e = res["yes_crossing"]
        print(f"  {env_name}: P(fail|no cross)={nc_e['p_fail']:.3f}, "
              f"P(fail|cross)={yc_e['p_fail']:.3f}, Fisher p={res['fisher_p']:.4f}")

    # ── Assemble output ──────────────────────────────────────────────
    summary = {
        "analysis": "condition_characterization_monotonicity",
        "unit_of_analysis": "rule pair (cross-run interpolation pair)",
        "n_pairs_total": len(df),
        "n_mergeable": int((df["category"] == "mergeable").sum()),
        "analysis_a": {
            "description": "Geometric conditions → boundary-crossing probability among mergeable pairs",
            "unit": "mergeable rule pair",
            "n_pairs": len(df_m),
            "predictors": {
                "path_low_density_frac": "Fraction of interpolation-path points "
                                         "falling in low-density regions",
                "midpoint_low_density_rate": "Fraction of midpoints (α=0.5) "
                                             "falling in low-density regions",
                "rule_dissimilarity": "1 − rule_similarity (threshold-aware); "
                                      "proxy for kNN-gap / geometric separation",
            },
            "outcome": "has_crossing: boundary_crossing_rate > 0",
            "schema_note": (
                "Distortion groups and boundary-crossing pairs do not share an explicit join key. "
                "This analysis therefore uses the strongest non-circular pair-level "
                "fallback supported directly by the boundary-crossing outputs."
            ),
            "pooled": analysis_a_pooled,
            "per_env": analysis_a_per_env,
        },
        "analysis_b": {
            "description": "Boundary crossing → post-merge failure probability",
            "unit": "mergeable rule pair",
            "predictor": "has_crossing (binary)",
            "outcome": "midpoint_action_mismatch_rate > 0.5 (post-merge "
                        "disagreement on interpolated midpoints)",
            "non_circularity_note": (
                "A2's is_failed_merge used group-level action_mismatch > 0.15 "
                "on member states. Here the predictor (boundary crossing) "
                "measures DQN policy self-consistency along interpolation "
                "paths, and the outcome measures DQN vs merged-rule agreement "
                "at a single midpoint — operationally and conceptually distinct."
            ),
            "pooled": ab_pooled,
            "per_env": ab_per_env,
        },
    }

    # ── Save JSON ─────────────────────────────────────────────────
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved → {OUT_JSON.relative_to(ROOT)}")

    # ── Save CSV (binned rows from Analysis A, pooled) ────────────
    csv_rows = []
    for pred_label, res in analysis_a_pooled.items():
        for b in res["bins"]:
            csv_rows.append({
                "scope": "pooled",
                "analysis": "A",
                "predictor": pred_label,
                "bin": b["bin"],
                "n": b["n"],
                "n_positive": b["n_positive"],
                "probability": b["probability"],
                "ci_lo": b["ci_lo"],
                "ci_hi": b["ci_hi"],
                "spearman_rho": res["spearman_rho"],
                "kendall_tau": res["kendall_tau"],
            })
    # add per-env rows
    for env_name, env_res in analysis_a_per_env.items():
        for pred_label, res in env_res.items():
            for b in res["bins"]:
                csv_rows.append({
                    "scope": env_name,
                    "analysis": "A",
                    "predictor": pred_label,
                    "bin": b["bin"],
                    "n": b["n"],
                    "n_positive": b["n_positive"],
                    "probability": b["probability"],
                    "ci_lo": b["ci_lo"],
                    "ci_hi": b["ci_hi"],
                    "spearman_rho": res["spearman_rho"],
                    "kendall_tau": res["kendall_tau"],
                })
    # add Analysis B rows
    for scope, res in [("pooled", ab_pooled)] + list(ab_per_env.items()):
        for label_key, tag in [("no_crossing", 0), ("yes_crossing", 1)]:
            r = res[label_key]
            csv_rows.append({
                "scope": scope,
                "analysis": "B",
                "predictor": "has_crossing",
                "bin": str(tag),
                "n": r["n"],
                "n_positive": r["n_fail"],
                "probability": r["p_fail"],
                "ci_lo": r["ci_lo"],
                "ci_hi": r["ci_hi"],
                "spearman_rho": None,
                "kendall_tau": None,
            })
    pd.DataFrame(csv_rows).to_csv(OUT_CSV, index=False)
    print(f"  Saved → {OUT_CSV.relative_to(ROOT)}")

    print("\n" + "=" * 60)
    print("Done.")
    return summary


if __name__ == "__main__":
    main()
