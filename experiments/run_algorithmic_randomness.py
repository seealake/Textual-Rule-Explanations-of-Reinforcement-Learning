#!/usr/bin/env python
"""
Algorithmic Randomness Perturbation for CBS.

This script tests how sensitive CBS rule extraction is to two sources
of algorithmic randomness (input data held fixed):

  1. **K-means seed variation**  — Run CBS M times with different
      ``kmeans_seed`` values on the *same* replay dataset.
  2. **Cluster count ±δ**        — Run CBS with K_default-1, K_default,
      K_default+1 clusters (delta applied per-action to the Elbow-
      selected K).

For each variant the script:
  • Fits CBS
  • Prints the extracted rules
  • Reports fidelity metrics (Accuracy, Recall, F1)
  • Reports the per-action cluster counts chosen
  • Saves extracted rules as JSON for downstream stability analysis

Usage
-----
    # Run both perturbation types for MountainCar-v0
    python experiments/run_algorithmic_randomness.py --env MountainCar-v0

    # Run both for CartPole-v1
    python experiments/run_algorithmic_randomness.py --env CartPole-v1

    # Run only K-means seed variation
    python experiments/run_algorithmic_randomness.py --env MountainCar-v0 --only kmeans_seeds

    # Run only cluster count ±1
    python experiments/run_algorithmic_randomness.py --env CartPole-v1 --only cluster_count

    # Custom K-means seeds
    python experiments/run_algorithmic_randomness.py --env MountainCar-v0 --kmeans-seeds 0 1 2 3 4

    # Custom cluster delta
    python experiments/run_algorithmic_randomness.py --env CartPole-v1 --cluster-delta 2

Output
------
    experiments/results/algorithmic_randomness/<env_tag>/
        kmeans_seed_<s>/
            rules.json
            metrics.json
        cluster_delta_<d>/
            rules.json
            metrics.json
        summary.json          # aggregated comparison across all variants
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from experiments.config_loader import load_config
from experiments.perturbations import load_replay_npz
from reproduction.cbs import CBSPipeline
from reproduction.collect_replay import ENV_FEATURE_NAMES


# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────

def _env_tag(env_name: str) -> str:
    return env_name.replace("-", "_").lower()


def _rules_to_serializable(rules) -> list:
    """Convert Rule objects to JSON-serializable dicts."""
    result = []
    for r in rules:
        result.append({
            "action": int(r.action),
            "weight": float(r.weight),
            "n_instances": int(r.condition.n_instances),
            "cluster_id": int(r.condition.cluster_id),
            "predicates": [
                {
                    "feature_idx": int(p.feature_idx),
                    "level": float(p.level),
                    "level_label": p.level_label,
                }
                for p in r.condition.predicates
            ],
        })
    return result


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)


def _run_cbs_variant(
    states: np.ndarray,
    actions: np.ndarray,
    feature_names: list[str],
    n_categories: int,
    inclusion_threshold: float,
    kmeans_seed: int | None,
    cluster_count_delta: int,
    label: str,
    output_dir: str,
) -> dict:
    """Run one CBS variant, print results, save JSON, return summary."""
    print(f"\n  ── {label} ──")

    cbs = CBSPipeline(
        n_categories=n_categories,
        inclusion_threshold=inclusion_threshold,
        kmeans_seed=kmeans_seed,
        cluster_count_delta=cluster_count_delta,
        feature_names=feature_names,
    )
    cbs.fit(states, actions)

    # Fidelity
    fidelity = cbs.evaluate_fidelity(states, actions)
    props = cbs.evaluate_properties()
    coverage = cbs.evaluate_coverage(states)

    print(f"    Cluster counts per action: {cbs.cluster_counts_}")
    print(f"    Rules extracted: {len(cbs.get_rules())}")
    print(f"    Fidelity — Acc={fidelity['accuracy']*100:.1f}%  "
          f"Rec={fidelity['recall']*100:.1f}%  F1={fidelity['f1']*100:.1f}%")
    print(f"    Coverage — approx_rate={coverage['approx_rate']*100:.1f}%")

    # Print rules compactly
    cbs.print_rules()

    # Prepare serializable result
    rules_data = _rules_to_serializable(cbs.get_rules())
    metrics_data = {
        "label": label,
        "kmeans_seed": kmeans_seed,
        "cluster_count_delta": cluster_count_delta,
        "cluster_counts": {str(k): v for k, v in cbs.cluster_counts_.items()},
        "n_rules": len(rules_data),
        "fidelity": {k: float(v) for k, v in fidelity.items()},
        "properties": {k: int(v) if isinstance(v, (int, np.integer)) else v
                       for k, v in props.items()},
        "coverage": {k: float(v) if isinstance(v, float) else int(v)
                     for k, v in coverage.items()},
    }

    # Save
    os.makedirs(output_dir, exist_ok=True)
    _save_json(rules_data, os.path.join(output_dir, "rules.json"))
    _save_json(metrics_data, os.path.join(output_dir, "metrics.json"))
    print(f"    ✓ Saved to {output_dir}/")

    return metrics_data


# ───────────────────────────────────────────────────────────────────────
# K-means Seed Variation
# ───────────────────────────────────────────────────────────────────────

def run_kmeans_seed_variation(
    states, actions, feature_names, cfg, kmeans_seeds, base_output_dir,
) -> list[dict]:
    """Run CBS with M different K-means seeds on the same data."""
    print(f"\n{'─'*60}")
    print(f"  K-means Seed Variation  (M={len(kmeans_seeds)} seeds)")
    print(f"{'─'*60}")

    results = []
    for seed in kmeans_seeds:
        out_dir = os.path.join(base_output_dir, f"kmeans_seed_{seed}")
        r = _run_cbs_variant(
            states=states,
            actions=actions,
            feature_names=feature_names,
            n_categories=cfg["cbs"]["n_categories"],
            inclusion_threshold=cfg["cbs"]["inclusion_threshold"],
            kmeans_seed=seed,
            cluster_count_delta=0,
            label=f"K-means seed={seed}",
            output_dir=out_dir,
        )
        results.append(r)

    return results


# ───────────────────────────────────────────────────────────────────────
# Cluster Count ±δ
# ───────────────────────────────────────────────────────────────────────

def run_cluster_count_variation(
    states, actions, feature_names, cfg, delta, base_output_dir,
) -> list[dict]:
    """Run CBS with cluster_count_delta in {-delta, 0, +delta}."""
    deltas = list(range(-delta, delta + 1))
    print(f"\n{'─'*60}")
    print(f"  Cluster Count Variation  (delta ∈ {deltas})")
    print(f"{'─'*60}")

    # Use a fixed K-means seed for fair comparison
    fixed_seed = cfg["global"]["random_seed"]

    results = []
    for d in deltas:
        sign_str = f"+{d}" if d >= 0 else str(d)
        out_dir = os.path.join(base_output_dir, f"cluster_delta_{sign_str}")
        r = _run_cbs_variant(
            states=states,
            actions=actions,
            feature_names=feature_names,
            n_categories=cfg["cbs"]["n_categories"],
            inclusion_threshold=cfg["cbs"]["inclusion_threshold"],
            kmeans_seed=fixed_seed,
            cluster_count_delta=d,
            label=f"Cluster count delta={sign_str}",
            output_dir=out_dir,
        )
        results.append(r)

    return results


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Algorithmic Randomness perturbation for CBS"
    )
    parser.add_argument(
        "--env", type=str, default="MountainCar-v0",
        help="Gymnasium environment name",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        choices=["kmeans_seeds", "cluster_count"],
        help="Run only one perturbation type (default: both)",
    )
    parser.add_argument(
        "--kmeans-seeds", type=int, nargs="+", default=None,
        help="List of K-means seeds to test (default: from config)",
    )
    parser.add_argument(
        "--cluster-delta", type=int, default=None,
        help="Cluster count delta (default: from config, typically 1)",
    )
    parser.add_argument(
        "--ref-seed", type=int, default=42,
        help="Seed of the reference replay dataset",
    )
    parser.add_argument(
        "--output-base", type=str,
        default="experiments/results/algorithmic_randomness",
        help="Base output directory",
    )
    args = parser.parse_args()

    # ── Load config ──
    cfg = load_config(env_override=args.env)
    tag = _env_tag(args.env)
    pert_cfg = cfg["perturbations"]["algorithmic"]

    # ── Resolve parameters ──
    kmeans_seeds = args.kmeans_seeds or list(range(
        pert_cfg["kmeans_seeds"]["n_seeds"]
    ))
    cluster_delta = (args.cluster_delta if args.cluster_delta is not None
                     else pert_cfg["cluster_count_variation"]["delta"])

    # ── Load reference replay ──
    ref_path = os.path.join(
        cfg["replay"]["data_dir"], f"replay_{tag}_seed{args.ref_seed}.npz"
    )
    if not os.path.exists(ref_path):
        print(f"ERROR: Reference replay not found at {ref_path}")
        print(f"Collect first: python reproduction/collect_replay.py --env {args.env}")
        sys.exit(1)

    data = load_replay_npz(ref_path)
    states = data["states"]
    actions = data["actions"]
    feature_names = ENV_FEATURE_NAMES.get(args.env, [f"f{i}" for i in range(states.shape[1])])

    # ── Print plan ──
    run_all = args.only is None
    base_dir = os.path.join(args.output_base, tag)
    print(f"\n{'='*64}")
    print(f"  Algorithmic Randomness")
    print(f"{'='*64}")
    print(f"  Environment:       {args.env}")
    print(f"  Reference replay:  {ref_path}")
    print(f"  Data:              {len(states):,} transitions, "
          f"{states.shape[1]} features")
    print(f"  Output:            {base_dir}/")
    if run_all or args.only == "kmeans_seeds":
        print(f"  K-means seeds:    {kmeans_seeds}")
    if run_all or args.only == "cluster_count":
        print(f"  Cluster delta:    ±{cluster_delta}")
    print(f"{'='*64}")

    all_results = []

    # ── K-means seed variation ──
    if run_all or args.only == "kmeans_seeds":
        results = run_kmeans_seed_variation(
            states, actions, feature_names, cfg, kmeans_seeds, base_dir,
        )
        all_results.extend(results)

    # ── Cluster count ±δ ──
    if run_all or args.only == "cluster_count":
        results = run_cluster_count_variation(
            states, actions, feature_names, cfg, cluster_delta, base_dir,
        )
        all_results.extend(results)

    # ── Summary ──
    print(f"\n{'='*64}")
    print(f"  Summary — Algorithmic Randomness Results")
    print(f"{'='*64}")
    print(f"  {'Label':<30s} {'Rules':>6s} {'Acc%':>7s} {'Rec%':>7s} {'F1%':>7s} {'Approx%':>8s}")
    print(f"  {'─'*30} {'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*8}")
    for r in all_results:
        f = r["fidelity"]
        c = r["coverage"]
        print(f"  {r['label']:<30s} {r['n_rules']:>6d} "
              f"{f['accuracy']*100:>7.1f} {f['recall']*100:>7.1f} "
              f"{f['f1']*100:>7.1f} {c['approx_rate']*100:>8.1f}")

    # F1 spread
    f1_values = [r["fidelity"]["f1"] for r in all_results]
    if len(f1_values) > 1:
        print(f"\n  F1 range: {min(f1_values)*100:.1f}% – {max(f1_values)*100:.1f}%  "
              f"(spread = {(max(f1_values)-min(f1_values))*100:.1f}pp)")

    # Save summary
    summary_path = os.path.join(base_dir, "summary.json")
    _save_json(all_results, summary_path)
    print(f"\n  ✓ Summary saved to {summary_path}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
