#!/usr/bin/env python
"""
SoftSupport consensus: 12-cell sweep

Runs the SoftSupport protocol (behavior-aware matching, soft support,
rare-action safeguard) for all three main environments with a fixed
12-cell grid.

Sweep grid (12 cells):
  lambda_B ∈ {0.0, 0.1, 0.2}
  support_mode ∈ {hard, soft}
  safeguard ∈ {off, on}

Fixed base parameters:
  B=5, rho=0.9, tau=0.7, lambda_P=0.35, lambda_I=0.45

Each cell uses 10 outer repeats.

Also produces:
  - Comparison table: CBS / default consensus / tuned v1 / v2 best / rule-set voting
  - Per-environment best v2 cell selection

Output:
    experiments/results/soft_support_sweep/{env}/raw_runs.json
    experiments/results/soft_support_sweep/{env}/summary.json
    experiments/results/soft_support_sweep/{env}/tables.csv
    experiments/results/soft_support_sweep/comparison_table.csv

Usage:
    python experiments/run_soft_support_sweep.py --env MountainCar-v0
    python experiments/run_soft_support_sweep.py --env CartPole-v1
    python experiments/run_soft_support_sweep.py --env LunarLander-v3
    python experiments/run_soft_support_sweep.py --env all
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from reproduction.collect_replay import collect_replay, ENV_FEATURE_NAMES
from experiments.perturbations import load_replay_npz, compute_feature_ranges
from experiments.consensus_merge import (
    build_consensus_ruleset,
    build_voting_ensemble,
    voting_predict,
)
from experiments.soft_support_merge import SoftSupportConfig, build_soft_support_consensus
from experiments.rule_matching import (
    canonicalize_rules,
    serialize_canonical_rules,
    mean_pairwise_jaccard,
    mean_pairwise_soft_jaccard,
    mean_pairwise_threshold_drift,
)
from experiments.run_stress_test import (
    run_cbs_on_data,
    evaluate_single_run,
    compute_bra_from_predictions,
    EVAL_SEEDS,
    SUCCESS_THRESHOLDS,
    HELDOUT_SEED,
)

# ── Configuration ────────────────────────────────────────────────────

ENVS = ["MountainCar-v0", "CartPole-v1", "LunarLander-v3"]
N_OUTER_REPEATS = 10
OUTER_SEEDS = list(range(N_OUTER_REPEATS))
OUT_ROOT = "experiments/results/soft_support_sweep"

# V2 sweep grid
LAMBDA_B_VALUES = [0.0, 0.1, 0.2]
SUPPORT_MODES = ["hard", "soft"]
SAFEGUARD_OPTIONS = [False, True]

# Fixed base parameters
BASE_B = 5
BASE_RHO = 0.9
BASE_TAU = 0.7
BASE_LAMBDA_P = 0.35
BASE_LAMBDA_I = 0.45

# Known best v2 cells from prior experiments
BEST_V2_HINTS = {
    "CartPole-v1": {"lambda_B": 0.1, "support_mode": "soft", "safeguard": False},
    "LunarLander-v3": {"lambda_B": 0.2, "support_mode": "soft", "safeguard": False},
    "MountainCar-v0": None,  # will select from sweep
}


def get_model_path(env_name):
    tag = env_name.replace("-", "_").lower()
    return f"reproduction/models/dqn_{tag}.zip"


def ci95(arr):
    a = np.array(arr, dtype=float)
    if len(a) < 2:
        return (float(a[0]), float(a[0])) if len(a) == 1 else (0.0, 0.0)
    se = a.std(ddof=1) / np.sqrt(len(a))
    return (float(a.mean() - 1.96 * se), float(a.mean() + 1.96 * se))


def _deploy_voting_ensemble(pipelines, env_name, eval_seeds, success_threshold):
    """Deploy voting ensemble as policy."""
    import gymnasium as gym
    env = gym.make(env_name)
    episode_rewards = []
    for ep_seed in eval_seeds:
        obs, info = env.reset(seed=ep_seed)
        total_reward = 0.0
        done = False
        while not done:
            action = int(voting_predict(pipelines, obs.reshape(1, -1))[0])
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        episode_rewards.append(total_reward)
    env.close()
    rewards_arr = np.array(episode_rewards)
    n_success = sum(1 for r in episode_rewards if r >= success_threshold) \
        if success_threshold else None
    return {
        "E_CR": float(rewards_arr.mean()),
        "E_CR_std": float(rewards_arr.std()),
        "success_rate": n_success / len(episode_rewards) if n_success is not None else None,
    }


def run_baselines(env_name, heldout_s, heldout_a, model_path):
    """Run CBS, default consensus, rule-set voting baselines with 10 outer repeats."""
    print(f"\n  --- Baselines ({env_name}) ---")

    outer_datasets = []
    for seed in OUTER_SEEDS:
        data = collect_replay(
            env_name=env_name, model_path=model_path,
            num_transitions=10000, seed=seed, deterministic=True,
        )
        outer_datasets.append(data)

    feature_ranges = compute_feature_ranges(outer_datasets[0])
    fr = {i: float(feature_ranges[i]) for i in range(len(feature_ranges))}

    results = {}

    # ── CBS ──
    print(f"    CBS (10 repeats)...")
    cbs_metrics = []
    cbs_all_preds = []
    cbs_all_rules = []
    cbs_all_thresholds = []

    for i, data in enumerate(outer_datasets):
        cbs, rules = run_cbs_on_data(data["states"], data["actions"], env_name)
        res = evaluate_single_run(cbs, rules, heldout_s, heldout_a, env_name)
        preds = cbs.predict(heldout_s)
        thresholds = {int(k): [float(v) for v in vs]
                      for k, vs in cbs.get_thresholds().items()}

        pa = res["fidelity_per_action"]["per_action"]
        recalls = [pa[a]["recall"] for a in pa]
        worst_recall = min(recalls) if recalls else 0.0

        cbs_metrics.append({
            "fidelity_heldout": res["fidelity_heldout"],
            "fidelity_per_action": res["fidelity_per_action"],
            "deployment": res["deployment"],
            "n_rules": len(rules),
            "worst_action_recall": worst_recall,
        })
        cbs_all_preds.append(preds)
        cbs_all_rules.append(rules)
        cbs_all_thresholds.append(thresholds)

    grs_wj = mean_pairwise_jaccard(cbs_all_rules, weighted=True)
    grs_ta = mean_pairwise_soft_jaccard(cbs_all_rules, threshold_aware=True)
    td = mean_pairwise_threshold_drift(cbs_all_thresholds, feature_ranges=fr)
    bra = compute_bra_from_predictions(cbs_all_preds)

    results["CBS"] = {
        "per_run": cbs_metrics,
        "stability": {"GRS_wj": float(grs_wj), "GRS_ta": float(grs_ta),
                       "BRA": float(bra), "TD": float(td)},
    }

    # ── Default consensus (v1 defaults) ──
    print(f"    Default consensus (10 repeats)...")
    dc_metrics = []
    dc_all_preds = []
    dc_all_rules = []
    dc_all_thresholds = []

    for data in outer_datasets:
        pipeline, rules, build_info = build_consensus_ruleset(
            data, env_name,
            n_bootstrap=5, consensus_threshold=0.7,
            similarity_cutoff=0.8, lambda1=0.6, lambda2=0.4,
        )
        res = evaluate_single_run(pipeline, rules, heldout_s, heldout_a, env_name)
        preds = pipeline.predict(heldout_s)
        thresholds = {int(k): [float(v) for v in vs]
                      for k, vs in res["thresholds"].items()}

        pa = res["fidelity_per_action"]["per_action"]
        recalls = [pa[a]["recall"] for a in pa]
        worst_recall = min(recalls) if recalls else 0.0

        dc_metrics.append({
            "fidelity_heldout": res["fidelity_heldout"],
            "fidelity_per_action": res["fidelity_per_action"],
            "deployment": res["deployment"],
            "n_rules": len(rules),
            "worst_action_recall": worst_recall,
            "build_info": build_info,
        })
        dc_all_preds.append(preds)
        dc_all_rules.append(rules)
        dc_all_thresholds.append(thresholds)

    grs_wj = mean_pairwise_jaccard(dc_all_rules, weighted=True)
    grs_ta = mean_pairwise_soft_jaccard(dc_all_rules, threshold_aware=True)
    td = mean_pairwise_threshold_drift(dc_all_thresholds, feature_ranges=fr)
    bra = compute_bra_from_predictions(dc_all_preds)

    results["default_consensus"] = {
        "per_run": dc_metrics,
        "stability": {"GRS_wj": float(grs_wj), "GRS_ta": float(grs_ta),
                       "BRA": float(bra), "TD": float(td)},
    }

    # ── rule-set voting ──
    print(f"    rule-set voting (10 repeats)...")
    vote_metrics = []
    vote_all_preds = []

    for data in outer_datasets:
        pipelines = build_voting_ensemble(data, env_name, n_bootstrap=5)
        preds = voting_predict(pipelines, heldout_s)

        acc = float(np.mean(preds == heldout_a))
        actions_set = sorted(np.unique(heldout_a))
        per_action = {}
        for a in actions_set:
            true_mask = heldout_a == a
            pred_mask = preds == a
            tp = int((true_mask & pred_mask).sum())
            prec = tp / pred_mask.sum() if pred_mask.sum() > 0 else 0.0
            rec = tp / true_mask.sum() if true_mask.sum() > 0 else 0.0
            pa_f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            per_action[int(a)] = {"precision": prec, "recall": rec, "f1": pa_f1}

        recalls = [per_action[a]["recall"] for a in per_action]
        macro_rec = float(np.mean(recalls))
        macro_f1 = 2 * acc * macro_rec / (acc + macro_rec) if (acc + macro_rec) > 0 else 0.0

        deploy = _deploy_voting_ensemble(
            pipelines, env_name, EVAL_SEEDS,
            SUCCESS_THRESHOLDS.get(env_name))

        worst_r = min(recalls) if recalls else 0.0
        n_rules = len(pipelines[0].get_rules())

        vote_metrics.append({
            "fidelity_heldout": {"f1": macro_f1, "accuracy": acc},
            "deployment": deploy,
            "n_rules": n_rules,
            "worst_action_recall": worst_r,
        })
        vote_all_preds.append(preds)

    bra = compute_bra_from_predictions(vote_all_preds)
    results["B3_vote"] = {
        "per_run": vote_metrics,
        "stability": {"GRS_wj": None, "GRS_ta": None,
                       "BRA": float(bra), "TD": None},
    }

    return results, outer_datasets, fr


def run_v2_sweep(env_name, outer_datasets, heldout_s, heldout_a, fr):
    """Run full 12-cell v2 sweep."""
    print(f"\n  --- V2 Sweep ({env_name}) ---")

    results = {}
    total_cells = len(LAMBDA_B_VALUES) * len(SUPPORT_MODES) * len(SAFEGUARD_OPTIONS)
    cell_idx = 0

    for lb in LAMBDA_B_VALUES:
        for sm in SUPPORT_MODES:
            for sg in SAFEGUARD_OPTIONS:
                cell_idx += 1
                tag = f"lB{lb}_sm{sm}_sg{'on' if sg else 'off'}"
                print(f"    [{cell_idx}/{total_cells}] {tag}...")

                cfg = SoftSupportConfig(
                    n_bootstrap=BASE_B,
                    consensus_threshold=BASE_TAU,
                    similarity_cutoff=BASE_RHO,
                    lambda_P=BASE_LAMBDA_P,
                    lambda_I=BASE_LAMBDA_I,
                    lambda_B=lb,
                    support_mode=sm,
                    safeguard_enabled=sg,
                )

                cell_metrics = []
                cell_preds = []
                cell_rules = []
                cell_thresholds = []

                for data in outer_datasets:
                    pipeline, rules, info = build_soft_support_consensus(
                        data, env_name, cfg)
                    res = evaluate_single_run(
                        pipeline, rules, heldout_s, heldout_a, env_name)
                    preds = pipeline.predict(heldout_s)
                    thresholds = {int(k): [float(v) for v in vs]
                                  for k, vs in res["thresholds"].items()}

                    pa = res["fidelity_per_action"]["per_action"]
                    recalls = [pa[a]["recall"] for a in pa]
                    worst_recall = min(recalls) if recalls else 0.0

                    cell_metrics.append({
                        "fidelity_heldout": res["fidelity_heldout"],
                        "fidelity_per_action": res["fidelity_per_action"],
                        "deployment": res["deployment"],
                        "n_rules": len(rules),
                        "worst_action_recall": worst_recall,
                        "build_info": info,
                        "rules": serialize_canonical_rules(rules),
                    })
                    cell_preds.append(preds)
                    cell_rules.append(rules)
                    cell_thresholds.append(thresholds)

                grs_wj = mean_pairwise_jaccard(cell_rules, weighted=True)
                grs_ta = mean_pairwise_soft_jaccard(cell_rules, threshold_aware=True)
                td = mean_pairwise_threshold_drift(cell_thresholds, feature_ranges=fr)
                bra = compute_bra_from_predictions(cell_preds)

                # Summaries
                f1_vals = [m["fidelity_heldout"]["f1"] for m in cell_metrics]
                ecr_vals = [m["deployment"]["E_CR"] for m in cell_metrics]
                war_vals = [m["worst_action_recall"] for m in cell_metrics]
                nr_vals = [m["n_rules"] for m in cell_metrics]

                results[tag] = {
                    "config": cfg.to_dict(),
                    "per_run": cell_metrics,
                    "stability": {
                        "GRS_wj": float(grs_wj),
                        "GRS_ta": float(grs_ta),
                        "BRA": float(bra),
                        "TD": float(td),
                    },
                    "summary": {
                        "F1": {"mean": float(np.mean(f1_vals)),
                               "std": float(np.std(f1_vals))},
                        "E_CR": {"mean": float(np.mean(ecr_vals)),
                                 "std": float(np.std(ecr_vals))},
                        "worst_R": {"mean": float(np.mean(war_vals)),
                                    "std": float(np.std(war_vals))},
                        "rules": {"mean": float(np.mean(nr_vals)),
                                  "std": float(np.std(nr_vals))},
                    },
                }

                print(f"      F1={np.mean(f1_vals):.3f} ± {np.std(f1_vals):.3f}, "
                      f"GRS_ta={grs_ta:.4f}, BRA={bra:.4f}")

    return results


def select_best_soft_support(soft_support_results, env_name):
    """Select best v2 cell by composite score (F1 + GRS_ta + BRA)."""
    best_tag = None
    best_score = -1.0

    for tag, data in soft_support_results.items():
        s = data["summary"]
        st = data["stability"]
        # Composite: F1 + GRS_ta + BRA (all [0,1])
        score = s["F1"]["mean"] + st["GRS_ta"] + st["BRA"]
        if score > best_score:
            best_score = score
            best_tag = tag

    return best_tag


def run_env(env_name):
    """Run the complete soft-support sweep for one environment."""
    print(f"\n{'='*70}")
    print(f"  SoftSupport consensus sweep: {env_name}")
    print(f"{'='*70}")

    model_path = get_model_path(env_name)
    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found at {model_path}. Skipping.")
        return None

    print(f"  Collecting held-out replay (seed={HELDOUT_SEED})...")
    heldout = collect_replay(
        env_name=env_name, model_path=model_path,
        num_transitions=5000, seed=HELDOUT_SEED, deterministic=True,
    )
    heldout_s, heldout_a = heldout["states"], heldout["actions"]
    print(f"  Held-out: {len(heldout_s)} transitions")

    t0 = time.time()

    # Run baselines and get outer datasets
    baseline_results, outer_datasets, fr = run_baselines(
        env_name, heldout_s, heldout_a, model_path)

    # Run v2 sweep using same outer datasets
    soft_support_results = run_v2_sweep(
        env_name, outer_datasets, heldout_s, heldout_a, fr)

    # Select best v2 cell
    best_soft_tag = select_best_soft_support(soft_support_results, env_name)

    elapsed = time.time() - t0

    # Load tuned v1 results if available
    env_tag = env_name.replace("-", "_").lower()
    tuned_merge_path = f"experiments/results/tuned_merge/{env_tag}/summary.json"
    tuned_merge_data = None
    if os.path.exists(tuned_merge_path):
        with open(tuned_merge_path) as f:
            tuned_merge_data = json.load(f)

    # Save results
    out_dir = os.path.join(OUT_ROOT, env_tag)
    os.makedirs(out_dir, exist_ok=True)

    output = {
        "schema_version": "b2_v2_sweep_v1",
        "env": env_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "n_outer_repeats": N_OUTER_REPEATS,
        "outer_seeds": OUTER_SEEDS,
        "heldout_seed": HELDOUT_SEED,
        "heldout_size": len(heldout_s),
        "eval_episodes": len(EVAL_SEEDS),
        "best_v2_cell": best_soft_tag,
        "baselines": baseline_results,
        "v2_sweep": soft_support_results,
    }

    raw_path = os.path.join(out_dir, "raw_runs.json")
    with open(raw_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Raw results saved to {raw_path}")

    # Build comparison table
    rows = []

    def _add_method_row(method_name, metrics_list, stability, config_str=""):
        f1_vals = [m["fidelity_heldout"]["f1"] for m in metrics_list]
        ecr_vals = [m["deployment"]["E_CR"] for m in metrics_list]
        war_vals = [m["worst_action_recall"] for m in metrics_list]
        nr_vals = [m["n_rules"] for m in metrics_list]
        rows.append({
            "env": env_name,
            "method": method_name,
            "config": config_str,
            "F1_mean": float(np.mean(f1_vals)),
            "F1_std": float(np.std(f1_vals)),
            "worst_R_mean": float(np.mean(war_vals)),
            "GRS_wj": stability.get("GRS_wj"),
            "GRS_ta": stability.get("GRS_ta"),
            "BRA": stability.get("BRA"),
            "TD": stability.get("TD"),
            "rules_mean": float(np.mean(nr_vals)),
            "E_CR_mean": float(np.mean(ecr_vals)),
        })

    # Add baselines
    for method, data in baseline_results.items():
        _add_method_row(method, data["per_run"], data["stability"])

    # Add tuned v1 if available
    if tuned_merge_data:
        for label in tuned_merge_data:
            if label in ("env", "timestamp", "n_outer_repeats"):
                continue
            v1d = tuned_merge_data[label]
            s = v1d["summary"]
            st = v1d["stability"]
            rows.append({
                "env": env_name,
                "method": f"tuned_merge_{label}",
                "config": str(v1d["config"]),
                "F1_mean": s["F1"]["mean"],
                "F1_std": s["F1"]["std"],
                "worst_R_mean": s["worst_action_recall"]["mean"],
                "GRS_wj": st["GRS_wj"],
                "GRS_ta": st["GRS_ta"],
                "BRA": st["BRA"],
                "TD": st["TD"],
                "rules_mean": s["rules"]["mean"],
                "E_CR_mean": s["E_CR"]["mean"],
            })

    # Add best v2
    if best_soft_tag and best_soft_tag in soft_support_results:
        v2d = soft_support_results[best_soft_tag]
        _add_method_row(f"soft_support ({best_soft_tag})",
                        v2d["per_run"], v2d["stability"],
                        best_soft_tag)

    # Add all v2 cells
    for tag, v2d in soft_support_results.items():
        _add_method_row(f"v2_{tag}", v2d["per_run"], v2d["stability"], tag)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "tables.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Tables saved to {csv_path}")

    # Save summary
    summary = {
        "env": env_name,
        "timestamp": output["timestamp"],
        "best_v2_cell": best_soft_tag,
        "baselines_summary": {},
        "v2_summary": {},
    }
    for method, data in baseline_results.items():
        f1_vals = [m["fidelity_heldout"]["f1"] for m in data["per_run"]]
        ecr_vals = [m["deployment"]["E_CR"] for m in data["per_run"]]
        summary["baselines_summary"][method] = {
            "F1": {"mean": float(np.mean(f1_vals)), "std": float(np.std(f1_vals))},
            "E_CR": {"mean": float(np.mean(ecr_vals))},
            "stability": data["stability"],
        }
    for tag, data in soft_support_results.items():
        summary["v2_summary"][tag] = {
            "summary": data["summary"],
            "stability": data["stability"],
        }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary saved to {summary_path}")

    # Print comparison
    print(f"\n  ── COMPARISON TABLE ({env_name}) ──")
    print(f"  {'Method':<35} {'F1':>6} {'worst-R':>8} {'GRS_ta':>7} "
          f"{'BRA':>6} {'rules':>6} {'E_CR':>7}")
    print(f"  {'-'*80}")
    for _, row in df.iterrows():
        method = row["method"]
        if method.startswith("v2_lB"):  # skip individual v2 cells in printout
            continue
        grs_ta = f"{row['GRS_ta']:.3f}" if row['GRS_ta'] is not None else "  N/A"
        bra = f"{row['BRA']:.3f}" if row['BRA'] is not None else "  N/A"
        td = f"{row['TD']:.4f}" if row.get('TD') is not None else " N/A"
        print(f"  {method:<35} {row['F1_mean']:>6.3f} {row['worst_R_mean']:>8.3f} "
              f"{grs_ta:>7} {bra:>6} {row['rules_mean']:>6.1f} "
              f"{row['E_CR_mean']:>7.1f}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="SoftSupport consensus sweep")
    parser.add_argument("--env", default="all",
                        choices=ENVS + ["all"])
    args = parser.parse_args()

    envs = ENVS if args.env == "all" else [args.env]

    print(f"\n{'#'*70}")
    print(f"  SOFTSUPPORT CONSENSUS SWEEP")
    print(f"  Environments: {envs}")
    print(f"  Grid: {len(LAMBDA_B_VALUES)} × {len(SUPPORT_MODES)} × "
          f"{len(SAFEGUARD_OPTIONS)} = "
          f"{len(LAMBDA_B_VALUES)*len(SUPPORT_MODES)*len(SAFEGUARD_OPTIONS)} cells")
    print(f"  Outer repeats: {N_OUTER_REPEATS}")
    print(f"{'#'*70}")

    t_total = time.time()

    all_results = {}
    for env_name in envs:
        result = run_env(env_name)
        if result:
            all_results[env_name] = result

    # Build combined comparison table
    if len(all_results) > 1:
        combined_rows = []
        for env_name, result in all_results.items():
            env_tag = env_name.replace("-", "_").lower()
            csv_path = os.path.join(OUT_ROOT, env_tag, "tables.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                combined_rows.append(df)

        if combined_rows:
            combined = pd.concat(combined_rows, ignore_index=True)
            combined_path = os.path.join(OUT_ROOT, "comparison_table.csv")
            combined.to_csv(combined_path, index=False)
            print(f"\n  Combined comparison table saved to {combined_path}")

    t_elapsed = time.time() - t_total
    print(f"\n{'#'*70}")
    print(f"  COMPLETE — Total elapsed: {t_elapsed:.1f}s")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
