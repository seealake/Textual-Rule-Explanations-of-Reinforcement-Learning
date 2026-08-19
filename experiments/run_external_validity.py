#!/usr/bin/env python
"""
External validity: MiniGrid + PPO

Validates that the instability phenomena and Consensus CBS findings
generalize beyond DQN + low-dim classic control environments.

Runs a compact stability comparison on MiniGrid-Dynamic-Obstacles-8x8-v0
with PPO policy:
  - CBS
  - rule-set voting (majority voting ensemble)
  - Default Consensus CBS (merge)
  - V2 soft-support Consensus

Core metrics:
  - F1 (held-out)
  - BRA (prediction agreement)
  - GRS-TA (threshold-aware set similarity)
  - worst-action recall

Protocol:
  5 outer seed-shift repeats, each with:
    seed_shift ×5, subsample ×5, cluster_count ×3, noise ×3
  = 16 perturbation runs per outer seed

Usage:
    python experiments/run_external_validity.py           # full suite
    python experiments/run_external_validity.py --quick   # 3 outer seeds only
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from reproduction.cbs import CBSPipeline
from reproduction.collect_replay import ENV_FEATURE_NAMES
from experiments.perturbations import (
    load_replay_npz,
    generate_subsamples,
    generate_stratified_subsamples,
    add_feature_noise,
    compute_feature_ranges,
)
from experiments.rule_matching import (
    canonicalize_rules,
    serialize_canonical_rules,
    mean_pairwise_soft_jaccard,
    rule_similarity_threshold_aware,
)
from experiments.consensus_merge import (
    build_consensus_ruleset,
    build_voting_ensemble,
    voting_predict,
    run_cbs_on_data as consensus_run_cbs,
    _match_rules_across_runs,
    merge_rule_group,
    aggregate_thresholds,
    make_consensus_pipeline,
)

# ── Constants ─────────────────────────────────────────────────────────
ENV_NAME = "MiniGrid-Dynamic-Obstacles-8x8-v0"
ENV_TAG = "minigrid_dynamic_obstacles_8x8_v0"
RESULTS_DIR = f"experiments/results/{ENV_TAG}"
EVAL_SEEDS = list(range(1000, 1050))
SUCCESS_THRESHOLD = 0.0  # reward > 0 = success
HELDOUT_SEED = 99
OUTER_SEEDS = [0, 1, 2, 3, 4]
SEED_SHIFT_SEEDS = [10, 11, 12]
N_SUBSAMPLES = 3
SUBSAMPLE_FRACTION = 0.8
CLUSTER_DELTAS = [-1, 0, 1]
NOISE_LEVELS = [0.01, 0.03, 0.05]
NOISE_MASK = [0, 1, 2, 3, 4, 5, 10, 11, 12]  # continuous features only

DEFAULT_B = 5
DEFAULT_TAU = 0.7
DEFAULT_RHO = 0.8
DEFAULT_LAMBDA1 = 0.6
DEFAULT_LAMBDA2 = 0.4

FEATURE_NAMES = ENV_FEATURE_NAMES.get(ENV_NAME, [f"f{i}" for i in range(14)])


def get_model_path():
    return f"reproduction/models/ppo_{ENV_TAG}.zip"


def _make_env():
    from reproduction.minigrid_feature_wrapper import make_minigrid_env
    return make_minigrid_env(ENV_NAME)


def _load_model():
    from stable_baselines3 import PPO
    return PPO.load(get_model_path())


def collect_replay_minigrid(n_transitions=10000, seed=42):
    """Collect replay from PPO policy."""
    env = _make_env()
    model = _load_model()

    states, actions, rewards, dones, episode_ids = [], [], [], [], []
    episode_count = 0
    obs, _ = env.reset(seed=seed)

    while len(states) < n_transitions:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        states.append(obs.copy())
        actions.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        dones.append(terminated or truncated)
        episode_ids.append(episode_count)
        if terminated or truncated:
            episode_count += 1
            obs, _ = env.reset(seed=seed + episode_count)

    env.close()
    n = n_transitions
    return {
        "states": np.array(states[:n], dtype=np.float32),
        "actions": np.array(actions[:n], dtype=np.int32),
        "rewards": np.array(rewards[:n], dtype=np.float32),
        "dones": np.array(dones[:n], dtype=bool),
        "episode_ids": np.array(episode_ids[:n], dtype=np.int32),
    }


def run_cbs_on_data(states, actions, kmeans_seed=0, delta=0):
    """Fit CBS on MiniGrid data."""
    cbs = CBSPipeline(
        n_categories=5,
        inclusion_threshold=0.70,
        kmeans_seed=kmeans_seed,
        cluster_count_delta=delta,
        feature_names=FEATURE_NAMES,
    )
    cbs.fit(states, actions)
    rules = canonicalize_rules(cbs.get_rules())
    return cbs, rules


def add_feature_noise_masked(data, noise_level, seed=42, feature_ranges=None):
    """Add noise only to continuous features."""
    rng = np.random.RandomState(seed)
    noisy = data["states"].copy()
    for fidx in NOISE_MASK:
        col = noisy[:, fidx]
        sigma = noise_level * (col.max() - col.min() + 1e-8)
        noisy[:, fidx] += rng.normal(0, sigma, size=len(noisy))
    result = {"states": noisy, "actions": data["actions"]}
    # Copy optional fields needed by generate_subsamples etc.
    for key in ("rewards", "dones", "episode_ids"):
        if key in data:
            result[key] = data[key]
    return result


def evaluate_cbs_in_env(cbs, seeds=None):
    """Deploy CBS in MiniGrid env."""
    if seeds is None:
        seeds = EVAL_SEEDS
    env = _make_env()
    rewards, lengths = [], []
    successes = 0

    for ep_seed in seeds:
        obs, _ = env.reset(seed=ep_seed)
        done = False
        ep_r, ep_l = 0.0, 0
        while not done:
            action = cbs.predict(obs.reshape(1, -1))[0]
            obs, reward, terminated, truncated, _ = env.step(int(action))
            ep_r += reward
            ep_l += 1
            done = terminated or truncated
        rewards.append(ep_r)
        lengths.append(ep_l)
        if ep_r > 0:
            successes += 1

    env.close()
    return {
        "E_CR": float(np.mean(rewards)),
        "E_CR_std": float(np.std(rewards)),
        "success_rate": successes / len(seeds),
    }


def evaluate_single(cbs, rules, heldout_s, heldout_a):
    """Evaluate one CBS pipeline (fidelity only, no deployment for speed)."""
    fid = cbs.evaluate_fidelity(heldout_s, heldout_a)
    fid_pa = cbs.evaluate_fidelity_per_action(heldout_s, heldout_a)
    worst_recall = min(
        (v["recall"] for v in fid_pa["per_action"].values()),
        default=0.0)
    return {
        "f1": round(fid["f1"], 4),
        "accuracy": round(fid["accuracy"], 4),
        "worst_action_recall": round(worst_recall, 4),
        "n_rules": len(rules),
    }


def compute_bra_from_predictions(pred_list):
    n = len(pred_list)
    if n < 2:
        return 1.0
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.mean(pred_list[i] == pred_list[j])
            count += 1
    return total / count


def build_v2_soft_consensus(data, env_name_unused, all_cbs, all_rules,
                             all_thresholds, tau=DEFAULT_TAU,
                             n_bootstrap=DEFAULT_B,
                             rho=DEFAULT_RHO):
    """Build consensus with soft support (v2-style)."""
    actions_set = sorted(set(r.action for rules in all_rules for r in rules))
    all_groups = []
    for action in actions_set:
        per_run = [[r for r in rules if r.action == action] for rules in all_rules]
        groups = _match_rules_across_runs(
            per_run, rho=rho, lambda1=DEFAULT_LAMBDA1, lambda2=DEFAULT_LAMBDA2)
        all_groups.extend(groups)

    level_values = all_cbs[0].level_values_
    level_labels = all_cbs[0].level_labels_

    kept_groups = []
    for group in all_groups:
        group_rules = [rule for _, rule in group]
        run_rules_map = {}
        for run_idx, rule in group:
            if run_idx not in run_rules_map:
                run_rules_map[run_idx] = []
            run_rules_map[run_idx].append(rule)

        soft_support = 0.0
        for run_idx in range(n_bootstrap):
            if run_idx in run_rules_map:
                soft_support += 1.0
            else:
                run_all_rules = all_rules[run_idx]
                max_sim = 0.0
                for r1 in run_all_rules:
                    for r2 in group_rules:
                        if r1.action != r2.action:
                            continue
                        sim = rule_similarity_threshold_aware(
                            r1, r2, lambda1=DEFAULT_LAMBDA1,
                            lambda2=DEFAULT_LAMBDA2)
                        max_sim = max(max_sim, sim)
                soft_support += max_sim

        soft_support /= n_bootstrap
        if soft_support >= tau:
            kept_groups.append(group)

    rules = []
    for group in kept_groups:
        rules_in_group = [rule for _, rule in group]
        cr = merge_rule_group(rules_in_group, level_values, level_labels)
        rules.append(cr)

    agg_thresh = aggregate_thresholds(all_thresholds, "median")
    pipeline = make_consensus_pipeline(all_cbs[0], rules, agg_thresh)
    return pipeline, rules


def run_perturbation_suite_for_method(method_name, method_fn, ref_data,
                                       heldout_s, heldout_a,
                                       seed_shift_data=None):
    """Run all perturbation families for one method.
    
    method_fn(data) -> (pipeline or list_of_pipelines, rules, is_voting)
    seed_shift_data: pre-collected list of (seed, data) tuples to avoid re-collecting
    """
    feature_ranges = compute_feature_ranges(ref_data)
    all_run_results = []
    all_predictions = []
    all_rulesets = []

    def _record_run(run_key, family, data):
        try:
            result = method_fn(data)
            if result is None:
                return

            if len(result) == 3:
                pipeline, rules, is_voting = result
            else:
                pipeline, rules = result
                is_voting = False

            if is_voting:
                # rule-set voting: evaluate via voting
                preds = voting_predict(pipeline, heldout_s)
                f1 = f1_score(heldout_a, preds, average="macro")
                acc = accuracy_score(heldout_a, preds)
                # Per-action recall
                from sklearn.metrics import recall_score
                per_action_recall = recall_score(
                    heldout_a, preds, average=None, zero_division=0)
                worst_recall = float(min(per_action_recall)) if len(per_action_recall) > 0 else 0.0

                run_result = {
                    "run_key": run_key,
                    "family": family,
                    "f1": round(float(f1), 4),
                    "accuracy": round(float(acc), 4),
                    "worst_action_recall": round(worst_recall, 4),
                    "n_rules": sum(len(canonicalize_rules(p.get_rules())) for p in pipeline),
                }
                all_predictions.append(preds)
            else:
                metrics = evaluate_single(pipeline, rules, heldout_s, heldout_a)
                run_result = {
                    "run_key": run_key,
                    "family": family,
                    **metrics,
                }
                preds = pipeline.predict(heldout_s)
                all_predictions.append(preds)
                all_rulesets.append(rules)

            all_run_results.append(run_result)
        except Exception as e:
            print(f"      WARN: {run_key} failed: {e}")

    # Seed shift (use pre-collected data)
    if seed_shift_data:
        for seed, data in seed_shift_data:
            _record_run(f"seed_shift_{seed}", "seed_shift", data)
    else:
        for seed in SEED_SHIFT_SEEDS:
            data = collect_replay_minigrid(n_transitions=10000, seed=seed)
            _record_run(f"seed_shift_{seed}", "seed_shift", data)

    # Subsampling
    subsamples = generate_subsamples(ref_data, N_SUBSAMPLES, SUBSAMPLE_FRACTION, seed=42)
    for i, ss in enumerate(subsamples):
        _record_run(f"subsample_{i}", "subsample", ss)

    # Cluster count
    # Need a wrapper that changes delta
    for delta in CLUSTER_DELTAS:
        _record_run(f"cluster_d{delta:+d}", "cluster_count",
                     {"data": ref_data, "delta": delta})

    # Feature noise
    for nl in NOISE_LEVELS:
        noisy = add_feature_noise_masked(ref_data, nl, seed=42)
        _record_run(f"noise_{nl}", "feature_noise", noisy)

    # Compute stability metrics
    bra = compute_bra_from_predictions(all_predictions) if len(all_predictions) >= 2 else 1.0

    grs_ta = 0.0
    if len(all_rulesets) >= 2:
        grs_ta = mean_pairwise_soft_jaccard(all_rulesets, threshold_aware=True)

    # Aggregate fidelity metrics
    f1_vals = [r["f1"] for r in all_run_results]
    wr_vals = [r["worst_action_recall"] for r in all_run_results]
    n_rules_vals = [r["n_rules"] for r in all_run_results]

    summary = {
        "method": method_name,
        "n_runs": len(all_run_results),
        "f1_mean": round(float(np.mean(f1_vals)), 4) if f1_vals else 0,
        "f1_std": round(float(np.std(f1_vals)), 4) if f1_vals else 0,
        "worst_recall_mean": round(float(np.mean(wr_vals)), 4) if wr_vals else 0,
        "worst_recall_std": round(float(np.std(wr_vals)), 4) if wr_vals else 0,
        "BRA": round(bra, 4),
        "GRS_TA": round(grs_ta, 4),
        "n_rules_mean": round(float(np.mean(n_rules_vals)), 1) if n_rules_vals else 0,
    }

    return summary, all_run_results


def run_external_validity(quick=False):
    """Run the full external validity experiment."""
    print(f"\n{'='*70}")
    print(f"  External Validity: {ENV_NAME} + PPO")
    print(f"{'='*70}")

    model_path = get_model_path()
    if not os.path.exists(model_path):
        print(f"  ERROR: PPO model not found at {model_path}")
        return None

    outer_seeds = OUTER_SEEDS[:3] if quick else OUTER_SEEDS

    # Collect reference replay and held-out
    print("  Collecting reference replay (10k)...")
    ref_data = collect_replay_minigrid(n_transitions=10000, seed=42)
    print(f"  Replay: {len(ref_data['states'])} transitions, "
          f"actions: {np.bincount(ref_data['actions'])}")

    print("  Collecting held-out replay (5k)...")
    ho_data = collect_replay_minigrid(n_transitions=5000, seed=HELDOUT_SEED)
    heldout_s = ho_data["states"]
    heldout_a = ho_data["actions"]

    all_outer_results = {}
    method_aggregates = {}

    for outer_seed in outer_seeds:
        print(f"\n  === Outer seed {outer_seed} ===")

        # Collect replay for this outer seed
        outer_data = collect_replay_minigrid(n_transitions=10000, seed=outer_seed)

        # Pre-collect seed-shift replays once (shared by all methods)
        print("    Pre-collecting seed-shift replays...")
        seed_shift_data = []
        for seed in SEED_SHIFT_SEEDS:
            sd = collect_replay_minigrid(n_transitions=10000, seed=seed)
            seed_shift_data.append((seed, sd))

        # Pre-build CBS subsamples for consensus methods
        rng = np.random.RandomState(outer_seed * 100 + 42)
        n_total = len(outer_data["states"])
        sub_cbs_list = []
        sub_rules_list = []
        sub_thresh_list = []

        for i in range(DEFAULT_B):
            idx = rng.choice(n_total, size=int(n_total * SUBSAMPLE_FRACTION),
                             replace=False)
            cbs, rules = run_cbs_on_data(outer_data["states"][idx],
                                          outer_data["actions"][idx])
            sub_cbs_list.append(cbs)
            sub_rules_list.append(rules)
            sub_thresh_list.append(cbs.get_thresholds())

        # Method 1: CBS
        def cbs_method(data):
            if isinstance(data, dict) and "delta" in data:
                p, r = run_cbs_on_data(data["data"]["states"],
                                        data["data"]["actions"],
                                        delta=data["delta"])
            else:
                p, r = run_cbs_on_data(data["states"], data["actions"])
            return p, r, False

        # Method 2: rule-set voting
        def vote_method(data):
            if isinstance(data, dict) and "delta" in data:
                actual_data = data["data"]
            else:
                actual_data = data
            pipelines = build_voting_ensemble(
                actual_data, ENV_NAME, n_bootstrap=DEFAULT_B,
                subsample_fraction=SUBSAMPLE_FRACTION, subsample_seed=42)
            all_rules = []
            for p in pipelines:
                all_rules.extend(canonicalize_rules(p.get_rules()))
            return pipelines, all_rules, True

        # Method 3: Default Consensus (hard support)
        def consensus_method(data):
            if isinstance(data, dict) and "delta" in data:
                actual_data = data["data"]
            else:
                actual_data = data
            pipeline, rules, info = build_consensus_ruleset(
                actual_data, ENV_NAME,
                n_bootstrap=DEFAULT_B,
                consensus_threshold=DEFAULT_TAU,
                similarity_cutoff=DEFAULT_RHO,
                lambda1=DEFAULT_LAMBDA1,
                lambda2=DEFAULT_LAMBDA2,
            )
            return pipeline, rules, False

        # Method 4: V2 soft-support consensus
        def v2_soft_method(data):
            if isinstance(data, dict) and "delta" in data:
                actual_data = data["data"]
            else:
                actual_data = data
            # Build fresh subsamples
            local_rng = np.random.RandomState(42)
            n = len(actual_data["states"])
            local_cbs, local_rules, local_thresh = [], [], []
            for _ in range(DEFAULT_B):
                idx = local_rng.choice(n, size=int(n * SUBSAMPLE_FRACTION),
                                        replace=False)
                c, r = run_cbs_on_data(actual_data["states"][idx],
                                        actual_data["actions"][idx])
                local_cbs.append(c)
                local_rules.append(r)
                local_thresh.append(c.get_thresholds())

            pipeline, rules = build_v2_soft_consensus(
                actual_data, ENV_NAME, local_cbs, local_rules, local_thresh)
            return pipeline, rules, False

        methods = [
            ("CBS", cbs_method),
            ("B3-vote", vote_method),
            ("Consensus_default", consensus_method),
            ("V2_soft_support", v2_soft_method),
        ]

        seed_results = {}
        for method_name, method_fn in methods:
            print(f"    Running {method_name}...")
            t0 = time.time()
            summary, runs = run_perturbation_suite_for_method(
                method_name, method_fn, outer_data, heldout_s, heldout_a,
                seed_shift_data=seed_shift_data)
            
            # Run single deployment evaluation on baseline config
            try:
                result = method_fn(outer_data)
                if result is not None and len(result) >= 2:
                    pipeline = result[0]
                    is_voting = len(result) == 3 and result[2]
                    if not is_voting:
                        deploy = evaluate_cbs_in_env(pipeline, seeds=EVAL_SEEDS[:10])
                        summary["E_CR"] = round(deploy["E_CR"], 4)
                        summary["success_rate"] = round(deploy["success_rate"], 4)
            except Exception:
                pass
            
            elapsed = time.time() - t0
            summary["elapsed_s"] = round(elapsed, 1)
            seed_results[method_name] = {"summary": summary, "runs": runs}

            print(f"      F1={summary['f1_mean']:.3f}±{summary['f1_std']:.3f}, "
                  f"BRA={summary['BRA']:.3f}, "
                  f"GRS-TA={summary['GRS_TA']:.3f}, "
                  f"worst_recall={summary['worst_recall_mean']:.3f}, "
                  f"rules={summary['n_rules_mean']:.0f} "
                  f"({elapsed:.0f}s)")

            # Accumulate
            if method_name not in method_aggregates:
                method_aggregates[method_name] = {
                    k: [] for k in ["f1_mean", "BRA", "GRS_TA",
                                    "worst_recall_mean", "n_rules_mean"]
                }
            for k in method_aggregates[method_name]:
                method_aggregates[method_name][k].append(summary[k])

        all_outer_results[str(outer_seed)] = seed_results

    # Compute cross-seed summary
    cross_seed_summary = {}
    for method_name, agg in method_aggregates.items():
        cross_seed_summary[method_name] = {
            k: {
                "mean": round(float(np.mean(v)), 4),
                "std": round(float(np.std(v)), 4),
            }
            for k, v in agg.items()
        }

    # Print final table
    print(f"\n{'='*80}")
    print(f"  Cross-Seed Summary ({len(outer_seeds)} outer seeds):")
    print(f"  {'Method':<22} {'F1':>12} {'BRA':>10} {'GRS-TA':>10} "
          f"{'W-Recall':>10} {'Rules':>8}")
    print(f"  {'-'*72}")
    for method in ["CBS", "B3-vote", "Consensus_default", "V2_soft_support"]:
        if method in cross_seed_summary:
            s = cross_seed_summary[method]
            f1 = f"{s['f1_mean']['mean']:.3f}±{s['f1_mean']['std']:.3f}"
            bra = f"{s['BRA']['mean']:.3f}±{s['BRA']['std']:.3f}"
            grs = f"{s['GRS_TA']['mean']:.3f}±{s['GRS_TA']['std']:.3f}"
            wr = f"{s['worst_recall_mean']['mean']:.3f}±{s['worst_recall_mean']['std']:.3f}"
            nr = f"{s['n_rules_mean']['mean']:.0f}"
            print(f"  {method:<22} {f1:>12} {bra:>10} {grs:>10} {wr:>10} {nr:>8}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "schema_version": "external_validity_v1",
        "env": ENV_NAME,
        "policy": "PPO",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "n_outer_seeds": len(outer_seeds),
            "outer_seeds": outer_seeds,
            "n_bootstrap": DEFAULT_B,
            "tau": DEFAULT_TAU,
            "rho": DEFAULT_RHO,
        },
        "cross_seed_summary": cross_seed_summary,
        "per_outer_seed": all_outer_results,
    }

    out_path = os.path.join(RESULTS_DIR, "external_validity.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="External Validity: MiniGrid + PPO")
    parser.add_argument("--quick", action="store_true",
                        help="Use only 3 outer seeds instead of 5")
    args = parser.parse_args()

    run_external_validity(quick=args.quick)


if __name__ == "__main__":
    main()
