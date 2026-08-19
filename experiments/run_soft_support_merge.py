#!/usr/bin/env python
"""
SoftSupport merge: compressed evaluation runner

Runs a compressed appendix evaluation comparing:
  CBS, default consensus merge, rule-set voting, soft_support_merge

on CartPole-v1 and LunarLander-v3, with sweeps over:
  lambda_B ∈ {0.0, 0.1, 0.2}
  support_mode ∈ {hard, soft}
  safeguard ∈ {off, on}

Outputs:
  experiments/results/soft_support_merge/raw/
  experiments/results/soft_support_merge/tables/
  experiments/results/soft_support_merge/logs/

Usage:
    python experiments/run_soft_support_merge.py
    python experiments/run_soft_support_merge.py --env CartPole-v1
    python experiments/run_soft_support_merge.py --env LunarLander-v3
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
from experiments.perturbations import (
    load_replay_npz,
    compute_feature_ranges,
)
from experiments.rule_matching import (
    canonicalize_rules,
    serialize_canonical_rules,
    mean_pairwise_jaccard,
    mean_pairwise_soft_jaccard,
    mean_pairwise_threshold_drift,
)
from experiments.consensus_merge import (
    build_consensus_ruleset,
    build_voting_ensemble,
    voting_predict,
)
from experiments.soft_support_merge import (
    SoftSupportConfig,
    build_soft_support_consensus,
)
from experiments.run_stress_test import (
    run_cbs_on_data,
    evaluate_single_run,
    compute_bra_from_predictions,
    EVAL_SEEDS,
    SUCCESS_THRESHOLDS,
    HELDOUT_SEED,
    SEED_SHIFT_SEEDS,
)

# ── Constants ────────────────────────────────────────────────────────
ENVS = ["CartPole-v1", "LunarLander-v3"]
OUT_ROOT = "experiments/results/soft_support_merge"

# Experiment defaults
DEFAULT_B = 5
DEFAULT_RHO = 0.9
DEFAULT_TAU = 0.7

# Sweep grid
LAMBDA_B_VALUES = [0.0, 0.1, 0.2]
SUPPORT_MODES = ["hard", "soft"]
SAFEGUARD_OPTIONS = [False, True]


def get_model_path(env_name):
    tag = env_name.replace("-", "_").lower()
    return f"reproduction/models/dqn_{tag}.zip"


def get_replay_path(env_name, seed=42):
    tag = env_name.replace("-", "_").lower()
    return f"reproduction/data/replay_{tag}_seed{seed}.npz"


def collect_heldout(env_name, model_path, n_transitions=5000):
    data = collect_replay(
        env_name=env_name, model_path=model_path,
        num_transitions=n_transitions, seed=HELDOUT_SEED,
        deterministic=True,
    )
    return data["states"], data["actions"]


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


def _extract_metrics(res, rules, n_rules=None):
    """Extract standardised metrics from evaluation result dict."""
    fid = res["fidelity_heldout"]
    dep = res["deployment"]
    pa = res.get("fidelity_per_action", {}).get("per_action", {})

    worst_recall = 1.0
    rules_per_action = {}
    per_action_recalls = {}
    for a, info in pa.items():
        r = info.get("recall", 0.0)
        per_action_recalls[a] = r
        if r < worst_recall:
            worst_recall = r
        rules_per_action[a] = info.get("rule_count", 0)

    return {
        "f1": fid.get("f1", 0.0),
        "accuracy": fid.get("accuracy", 0.0),
        "E_CR": dep.get("E_CR", 0.0),
        "E_CR_std": dep.get("E_CR_std", 0.0),
        "success_rate": dep.get("success_rate"),
        "n_rules": n_rules if n_rules is not None else res.get("n_rules", 0),
        "worst_action_recall": worst_recall,
        "rules_per_action": rules_per_action,
        "per_action_recalls": per_action_recalls,
    }


def run_baselines(env_name, ref_data, heldout_s, heldout_a, model_path):
    """Run CBS, default consensus, and rule-set voting baselines with 5 outer seed-shift repeats."""
    print(f"\n  --- Baselines ({env_name}) ---")

    outer_datasets = []
    for seed in SEED_SHIFT_SEEDS:
        data = collect_replay(
            env_name=env_name, model_path=model_path,
            num_transitions=10000, seed=seed, deterministic=True,
        )
        outer_datasets.append(data)

    feature_ranges = compute_feature_ranges(ref_data)
    fr = {i: float(feature_ranges[i]) for i in range(len(feature_ranges))}

    results = {}

    # ── CBS ──
    print(f"    CBS...")
    cbs_metrics = []
    cbs_all_preds = []
    cbs_all_rules = []
    for data in outer_datasets:
        cbs, rules = run_cbs_on_data(data["states"], data["actions"], env_name)
        res = evaluate_single_run(cbs, rules, heldout_s, heldout_a, env_name)
        m = _extract_metrics(res, rules, len(rules))
        cbs_metrics.append(m)
        cbs_all_preds.append(cbs.predict(heldout_s))
        cbs_all_rules.append(rules)

    grs_wj = mean_pairwise_jaccard(cbs_all_rules, weighted=True)
    grs_ta = mean_pairwise_soft_jaccard(cbs_all_rules)
    bra = compute_bra_from_predictions(cbs_all_preds)

    results["CBS"] = {
        "per_run": cbs_metrics,
        "stability": {"GRS_wj": grs_wj, "GRS_ta": grs_ta, "BRA": bra},
    }

    # ── Default consensus merge ──
    print(f"    default consensus...")
    b3_metrics = []
    b3_all_preds = []
    b3_all_rules = []
    for data in outer_datasets:
        pipeline, rules, info = build_consensus_ruleset(
            data, env_name,
            n_bootstrap=DEFAULT_B,
            consensus_threshold=DEFAULT_TAU,
            similarity_cutoff=DEFAULT_RHO,
        )
        res = evaluate_single_run(pipeline, rules, heldout_s, heldout_a, env_name)
        m = _extract_metrics(res, rules, len(rules))
        m["build_info"] = {
            "n_consensus_rules": info["n_consensus_rules"],
            "n_kept_groups": info["n_kept_groups"],
            "per_action_rule_counts": info["per_action_rule_counts"],
        }
        b3_metrics.append(m)
        b3_all_preds.append(pipeline.predict(heldout_s))
        b3_all_rules.append(rules)

    grs_wj = mean_pairwise_jaccard(b3_all_rules, weighted=True)
    grs_ta = mean_pairwise_soft_jaccard(b3_all_rules)
    bra = compute_bra_from_predictions(b3_all_preds)

    results["B3_consensus"] = {
        "per_run": b3_metrics,
        "stability": {"GRS_wj": grs_wj, "GRS_ta": grs_ta, "BRA": bra},
    }

    # ── rule-set voting ──
    print(f"    rule-set voting...")
    vote_metrics = []
    vote_all_preds = []
    for i, data in enumerate(outer_datasets):
        pipelines = build_voting_ensemble(
            data, env_name, n_bootstrap=DEFAULT_B)
        preds = voting_predict(pipelines, heldout_s)

        # Compute fidelity
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
            rule_count = sum(1 for r in pipelines[0].get_rules() if r.action == a)
            per_action[int(a)] = {
                "precision": prec, "recall": rec, "f1": pa_f1,
                "support": int(true_mask.sum()), "rule_count": rule_count,
            }

        recalls = [per_action[a]["recall"] for a in per_action]
        macro_rec = float(np.mean(recalls))
        macro_f1 = 2 * acc * macro_rec / (acc + macro_rec) if (acc + macro_rec) > 0 else 0.0

        deploy = _deploy_voting_ensemble(
            pipelines, env_name, EVAL_SEEDS,
            SUCCESS_THRESHOLDS.get(env_name))

        n_rules = len(pipelines[0].get_rules())
        worst_r = min(recalls) if recalls else 0.0

        m = {
            "f1": macro_f1, "accuracy": acc,
            "E_CR": deploy["E_CR"], "E_CR_std": deploy["E_CR_std"],
            "success_rate": deploy["success_rate"],
            "n_rules": n_rules,
            "worst_action_recall": worst_r,
            "rules_per_action": {a: per_action[a]["rule_count"] for a in per_action},
            "per_action_recalls": {a: per_action[a]["recall"] for a in per_action},
        }
        vote_metrics.append(m)
        vote_all_preds.append(preds)

    bra = compute_bra_from_predictions(vote_all_preds)
    results["B3_vote"] = {
        "per_run": vote_metrics,
        "stability": {"GRS_wj": None, "GRS_ta": None, "BRA": bra},
    }

    return results


def run_v2_sweep(env_name, ref_data, heldout_s, heldout_a, model_path):
    """Run soft_support_merge with all sweep configurations."""
    print(f"\n  --- Consensus CBS v2 Sweep ({env_name}) ---")

    outer_datasets = []
    for seed in SEED_SHIFT_SEEDS:
        data = collect_replay(
            env_name=env_name, model_path=model_path,
            num_transitions=10000, seed=seed, deterministic=True,
        )
        outer_datasets.append(data)

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
                    n_bootstrap=DEFAULT_B,
                    consensus_threshold=DEFAULT_TAU,
                    similarity_cutoff=DEFAULT_RHO,
                    lambda_P=0.35,
                    lambda_I=0.45,
                    lambda_B=lb,
                    support_mode=sm,
                    safeguard_enabled=sg,
                    safeguard_floor=0.10,
                    safeguard_topk=2,
                    calibration_n=2000,
                    calibration_seed=123,
                )

                cell_metrics = []
                cell_preds = []
                cell_rules = []
                cell_build_infos = []

                for data in outer_datasets:
                    pipeline, rules, info = build_soft_support_consensus(
                        data, env_name, cfg)
                    res = evaluate_single_run(
                        pipeline, rules, heldout_s, heldout_a, env_name)
                    m = _extract_metrics(res, rules, len(rules))
                    m["build_info"] = {
                        "n_consensus_rules": info["n_consensus_rules"],
                        "n_kept_groups": info["n_kept_groups"],
                        "per_action_rule_counts": info["per_action_rule_counts"],
                        "safeguard_rescued": info["safeguard"]["rescued_groups"],
                    }
                    cell_metrics.append(m)
                    cell_preds.append(pipeline.predict(heldout_s))
                    cell_rules.append(rules)
                    cell_build_infos.append(info)

                grs_wj = mean_pairwise_jaccard(cell_rules, weighted=True)
                grs_ta = mean_pairwise_soft_jaccard(cell_rules)
                bra = compute_bra_from_predictions(cell_preds)

                # group size stats
                all_group_sizes = []
                for info in cell_build_infos:
                    for d in info.get("group_diagnostics", []):
                        all_group_sizes.append(d.get("group_size", 0))

                results[tag] = {
                    "config": cfg.to_dict(),
                    "per_run": cell_metrics,
                    "stability": {
                        "GRS_wj": float(grs_wj),
                        "GRS_ta": float(grs_ta),
                        "BRA": float(bra),
                    },
                    "group_size_stats": {
                        "mean": float(np.mean(all_group_sizes)) if all_group_sizes else 0,
                        "std": float(np.std(all_group_sizes)) if all_group_sizes else 0,
                        "min": int(np.min(all_group_sizes)) if all_group_sizes else 0,
                        "max": int(np.max(all_group_sizes)) if all_group_sizes else 0,
                    },
                }

    return results


def build_summary_table(baseline_results, soft_support_results, env_name):
    """Build summary table rows."""
    rows = []

    def _avg(metrics_list, key):
        vals = [m[key] for m in metrics_list if m.get(key) is not None]
        if not vals:
            return 0.0, 0.0
        return float(np.mean(vals)), float(np.std(vals))

    # Baselines
    for method_name, data in baseline_results.items():
        f1_m, f1_s = _avg(data["per_run"], "f1")
        ecr_m, ecr_s = _avg(data["per_run"], "E_CR")
        war_m, war_s = _avg(data["per_run"], "worst_action_recall")
        nr_m, nr_s = _avg(data["per_run"], "n_rules")
        stab = data["stability"]

        rows.append({
            "env": env_name,
            "method": method_name,
            "config": "baseline",
            "f1_mean": f1_m, "f1_std": f1_s,
            "E_CR_mean": ecr_m, "E_CR_std": ecr_s,
            "GRS_wj": stab.get("GRS_wj"),
            "GRS_ta": stab.get("GRS_ta"),
            "BRA": stab.get("BRA"),
            "worst_action_recall_mean": war_m,
            "worst_action_recall_std": war_s,
            "n_rules_mean": nr_m, "n_rules_std": nr_s,
            "group_size_mean": None,
        })

    # V2 variants
    for tag, data in soft_support_results.items():
        f1_m, f1_s = _avg(data["per_run"], "f1")
        ecr_m, ecr_s = _avg(data["per_run"], "E_CR")
        war_m, war_s = _avg(data["per_run"], "worst_action_recall")
        nr_m, nr_s = _avg(data["per_run"], "n_rules")
        stab = data["stability"]
        gs = data.get("group_size_stats", {})

        rows.append({
            "env": env_name,
            "method": "soft_support",
            "config": tag,
            "f1_mean": f1_m, "f1_std": f1_s,
            "E_CR_mean": ecr_m, "E_CR_std": ecr_s,
            "GRS_wj": stab.get("GRS_wj"),
            "GRS_ta": stab.get("GRS_ta"),
            "BRA": stab.get("BRA"),
            "worst_action_recall_mean": war_m,
            "worst_action_recall_std": war_s,
            "n_rules_mean": nr_m, "n_rules_std": nr_s,
            "group_size_mean": gs.get("mean"),
        })

    return rows


def save_markdown_table(df, path, title=""):
    """Save a dataframe as Markdown table."""
    with open(path, "w") as f:
        if title:
            f.write(f"# {title}\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n")


def run_env(env_name):
    print(f"\n{'='*60}")
    print(f"  Appendix Consensus CBS v2: {env_name}")
    print(f"{'='*60}")

    model_path = get_model_path(env_name)
    ref_path = get_replay_path(env_name)

    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found at {model_path}. Skipping.")
        return None
    if not os.path.exists(ref_path):
        print(f"  ERROR: Replay not found at {ref_path}. Skipping.")
        return None

    ref_data = load_replay_npz(ref_path)
    print(f"  Reference replay: {len(ref_data['states'])} transitions")

    print(f"  Collecting held-out replay (seed={HELDOUT_SEED})...")
    heldout_s, heldout_a = collect_heldout(env_name, model_path)
    print(f"  Held-out replay: {len(heldout_s)} transitions")

    t0 = time.time()

    baseline_results = run_baselines(
        env_name, ref_data, heldout_s, heldout_a, model_path)
    soft_support_results = run_v2_sweep(
        env_name, ref_data, heldout_s, heldout_a, model_path)

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    return {
        "env": env_name,
        "baselines": baseline_results,
        "v2_sweep": soft_support_results,
        "elapsed_seconds": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="all", choices=ENVS + ["all"])
    args = parser.parse_args()

    envs = ENVS if args.env == "all" else [args.env]

    # Create output directories
    for sub in ["raw", "tables", "figures", "logs"]:
        os.makedirs(os.path.join(OUT_ROOT, sub), exist_ok=True)

    all_results = {}
    all_rows = []

    for env_name in envs:
        result = run_env(env_name)
        if result is None:
            continue
        all_results[env_name] = result

        # Save raw per-env
        env_tag = env_name.replace("-", "_").lower()
        raw_path = os.path.join(OUT_ROOT, "raw", f"{env_tag}_soft_support_results.json")
        with open(raw_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Raw results saved to {raw_path}")

        rows = build_summary_table(
            result["baselines"], result["v2_sweep"], env_name)
        all_rows.extend(rows)

    if not all_rows:
        print("No results to summarise.")
        return

    # Build summary dataframe
    df = pd.DataFrame(all_rows)

    # Save summary CSV
    csv_path = os.path.join(OUT_ROOT, "tables", "summary.csv")
    df.to_csv(csv_path, index=False)

    # ── Main table: best v2 config vs baselines per env ──
    main_rows = []
    for env_name in all_results:
        env_df = df[df["env"] == env_name]
        baselines = env_df[env_df["config"] == "baseline"]
        v2_cells = env_df[env_df["config"] != "baseline"]

        for _, row in baselines.iterrows():
            main_rows.append(row.to_dict())

        # Pick best v2 by F1
        if len(v2_cells) > 0:
            best = v2_cells.loc[v2_cells["f1_mean"].idxmax()]
            main_rows.append(best.to_dict())

    main_df = pd.DataFrame(main_rows)
    cols = ["env", "method", "config", "f1_mean", "f1_std",
            "GRS_wj", "GRS_ta", "BRA",
            "worst_action_recall_mean", "n_rules_mean"]
    main_table = main_df[[c for c in cols if c in main_df.columns]]

    main_csv = os.path.join(OUT_ROOT, "tables", "main_table.csv")
    main_table.to_csv(main_csv, index=False)
    save_markdown_table(main_table,
                        os.path.join(OUT_ROOT, "tables", "main_table.md"),
                        "Main comparison: CBS vs default merge vs rule-set voting vs soft support")

    # ── Ablation table: all v2 configs ──
    ablation_df = df[df["config"] != "baseline"]
    abl_cols = ["env", "config", "f1_mean", "f1_std",
                "GRS_wj", "GRS_ta", "BRA",
                "worst_action_recall_mean", "n_rules_mean", "group_size_mean"]
    ablation_table = ablation_df[[c for c in abl_cols if c in ablation_df.columns]]

    abl_csv = os.path.join(OUT_ROOT, "tables", "ablation_table.csv")
    ablation_table.to_csv(abl_csv, index=False)
    save_markdown_table(ablation_table,
                        os.path.join(OUT_ROOT, "tables", "ablation_table.md"),
                        "Ablation: Consensus CBS v2 Configuration Sweep")

    # Save config snapshot
    config_snapshot = {
        "B": DEFAULT_B,
        "rho": DEFAULT_RHO,
        "tau": DEFAULT_TAU,
        "lambda_B_sweep": LAMBDA_B_VALUES,
        "support_modes": SUPPORT_MODES,
        "safeguard_options": SAFEGUARD_OPTIONS,
        "outer_repeats": len(SEED_SHIFT_SEEDS),
        "seed_shift_seeds": SEED_SHIFT_SEEDS,
        "eval_seeds": EVAL_SEEDS[:5],
        "heldout_seed": HELDOUT_SEED,
    }
    config_path = os.path.join(OUT_ROOT, "logs", "config_snapshot.json")
    with open(config_path, "w") as f:
        json.dump(config_snapshot, f, indent=2)

    # Save README
    readme_path = os.path.join(OUT_ROOT, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Appendix: Consensus CBS v2 Evaluation\n\n")
        f.write("## How to reproduce\n\n")
        f.write("```bash\n")
        f.write("# Activate virtual environment\n")
        f.write("# Run full evaluation:\n")
        f.write("python experiments/run_soft_support_merge.py --env all\n")
        f.write("# Or per environment:\n")
        f.write("python experiments/run_soft_support_merge.py --env CartPole-v1\n")
        f.write("python experiments/run_soft_support_merge.py --env LunarLander-v3\n")
        f.write("```\n\n")
        f.write("## Output structure\n\n")
        f.write("- `raw/` — per-environment JSON with all run details\n")
        f.write("- `tables/` — summary CSV + Markdown tables\n")
        f.write("- `figures/` — plots (if generated)\n")
        f.write("- `logs/` — config snapshot\n\n")
        f.write("## Key parameters\n\n")
        f.write(f"- B = {DEFAULT_B} internal subsamples\n")
        f.write(f"- ρ = {DEFAULT_RHO} (similarity cutoff)\n")
        f.write(f"- τ = {DEFAULT_TAU} (consensus threshold)\n")
        f.write(f"- λ_B sweep: {LAMBDA_B_VALUES}\n")
        f.write(f"- support_mode: {SUPPORT_MODES}\n")
        f.write(f"- safeguard: {SAFEGUARD_OPTIONS}\n")
        f.write(f"- {len(SEED_SHIFT_SEEDS)} outer repeats (seed shift)\n")

    print(f"\n{'='*60}")
    print(f"  All results saved to {OUT_ROOT}/")
    print(f"  Main table: {main_csv}")
    print(f"  Ablation table: {abl_csv}")
    print(f"  Summary CSV: {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
