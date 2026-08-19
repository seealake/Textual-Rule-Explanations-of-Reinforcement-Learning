#!/usr/bin/env python
"""
Decision-tree structural diagnostic

Motivated by CGXplain: decision-tree-based surrogate explanations are
structurally unstable because tree induction produces different splits
on minor data perturbations.  This script:

  1. Reuses the stored 21-run DT results to extract structural
     complexity indicators (depth, leaf count, internal nodes, rule
     count per action, top-level split feature distribution).

  2. (Optional) Runs a shallow-tree ablation with max_depth ∈ {3,5,7,None}
     and evaluates F1, BRA, GRS on 5 seed-shift outer repeats each.

Outputs:
  experiments/results/<env>/tree_depth_ablation.json

Usage:
    python experiments/run_tree_depth_ablation.py --env MountainCar-v0
    python experiments/run_tree_depth_ablation.py --env all
    python experiments/run_tree_depth_ablation.py --env all --skip-ablation
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from reproduction.collect_replay import collect_replay, ENV_FEATURE_NAMES
from experiments.decision_tree_surrogate import (
    DecisionTreeSurrogate,
    canonicalize_dt_rules,
)
from experiments.perturbations import (
    load_replay_npz,
    generate_subsamples,
    generate_stratified_subsamples,
    add_feature_noise,
    compute_feature_ranges,
)
from experiments.rule_matching import (
    serialize_canonical_rules,
    mean_pairwise_jaccard,
    mean_pairwise_soft_jaccard,
    mean_pairwise_threshold_drift,
)
from experiments.run_stress_test import (
    compute_bra_from_predictions,
    EVAL_SEEDS,
    SUCCESS_THRESHOLDS,
    HELDOUT_SEED,
    SEED_SHIFT_SEEDS,
    _serialize,
)

ENVS = ["MountainCar-v0", "CartPole-v1", "LunarLander-v3"]
ABLATION_DEPTHS = [3, 5, 7, None]
N_ABLATION_SEEDS = 5


# ── Tree structure extraction ────────────────────────────────────────

def extract_tree_structure(dt: DecisionTreeSurrogate) -> dict:
    """Extract structural complexity indicators from a fitted DT.

    Returns dict with: depth, n_leaves, n_internal_nodes, n_rules,
    rules_per_action, top_split_feature, per_level_split_features.
    """
    tree = dt.tree_.tree_

    depth = int(tree.max_depth)
    n_nodes = int(tree.node_count)

    # Count leaves and internal nodes
    n_leaves = 0
    n_internal = 0
    for nid in range(n_nodes):
        if tree.children_left[nid] == tree.children_right[nid]:
            n_leaves += 1
        else:
            n_internal += 1

    # Rules per action
    rules = dt.get_rules()
    actions = sorted(set(r.action for r in rules))
    rules_per_action = {int(a): sum(1 for r in rules if r.action == a)
                        for a in actions}

    # Top-level split feature (root node)
    top_split_feature = int(tree.feature[0]) if n_internal > 0 else -1

    # Per-level dominant split features (BFS)
    per_level_splits = {}
    queue = [(0, 0)]  # (node_id, level)
    while queue:
        nid, level = queue.pop(0)
        if tree.children_left[nid] == tree.children_right[nid]:
            continue  # leaf
        feat = int(tree.feature[nid])
        per_level_splits.setdefault(level, []).append(feat)
        queue.append((int(tree.children_left[nid]), level + 1))
        queue.append((int(tree.children_right[nid]), level + 1))

    # Summarize: most common feature per level
    level_dominant = {}
    for level, feats in sorted(per_level_splits.items()):
        counts = {}
        for f in feats:
            counts[f] = counts.get(f, 0) + 1
        dominant = max(counts, key=counts.get)
        level_dominant[level] = {
            "dominant_feature": dominant,
            "dominant_count": counts[dominant],
            "total_nodes": len(feats),
            "all_features": dict(sorted(counts.items())),
        }

    # Feature importance from sklearn
    importances = dt.tree_.feature_importances_.tolist()

    return {
        "depth": depth,
        "n_leaves": n_leaves,
        "n_internal_nodes": n_internal,
        "n_rules": len(rules),
        "rules_per_action": rules_per_action,
        "top_split_feature": top_split_feature,
        "per_level_splits": {int(k): v for k, v in level_dominant.items()},
        "feature_importances": importances,
    }


def extract_from_existing_results(env_name: str) -> dict:
    """Re-fit DT for each existing run to extract structural info.

    We re-fit because the original decision_tree_results.json doesn't store
    tree structure.  Uses the same data and parameters.
    """
    env_tag = env_name.replace("-", "_").lower()
    results_path = f"experiments/results/{env_tag}/decision_tree_results.json"

    if not os.path.exists(results_path):
        print(f"  WARNING: {results_path} not found, skipping.")
        return {}

    with open(results_path) as f:
        data = json.load(f)

    method_params = data["b4_dt"]["method_params"]
    max_depth = method_params.get("max_depth")
    min_samples_leaf = method_params.get("min_samples_leaf", 5)

    model_path = _get_model_path(env_name)
    ref_path = _get_replay_path(env_name)
    ref_data = load_replay_npz(ref_path)
    feature_names = ENV_FEATURE_NAMES.get(env_name)
    feature_ranges = compute_feature_ranges(ref_data)

    per_run = data["b4_dt"]["per_run"]
    structural_results = {}

    print(f"  Re-fitting {len(per_run)} DT runs to extract structure...")

    for run_key, run_data in per_run.items():
        family = run_data["perturbation_family"]
        params = run_data["perturbation_params"]

        # Reconstruct training data for this run
        states, actions = _reconstruct_training_data(
            env_name, model_path, ref_data, family, params, run_key)
        if states is None:
            continue

        # Determine depth for this run
        run_depth = max_depth
        if family == "depth_variation":
            actual_depth = run_data.get("method_params", {}).get("actual_depth")
            if actual_depth is not None:
                run_depth = actual_depth
            else:
                delta = params.get("depth_delta", 0)
                if max_depth is not None:
                    run_depth = max(1, max_depth + delta)
                else:
                    run_depth = max(1, 5 + delta)

        dt = DecisionTreeSurrogate(
            max_depth=run_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            feature_names=feature_names,
        )
        dt.fit(states, actions)

        struct = extract_tree_structure(dt)
        struct["run_key"] = run_key
        struct["perturbation_family"] = family
        struct["f1"] = run_data["fidelity_heldout"]["f1"]

        # Add per-run stability proxies if available
        if "stability_proxy_global" in run_data:
            struct["grs_wj"] = run_data["stability_proxy_global"]["GRS_wj"]
            struct["grs_ta"] = run_data["stability_proxy_global"]["GRS_ta"]
            struct["bra"] = run_data["stability_proxy_global"]["BRA"]
            struct["td"] = run_data["stability_proxy_global"]["TD"]

        structural_results[run_key] = struct

    return structural_results


def _get_model_path(env_name):
    tag = env_name.replace("-", "_").lower()
    return f"reproduction/models/dqn_{tag}.zip"


def _get_replay_path(env_name, seed=42):
    tag = env_name.replace("-", "_").lower()
    return f"reproduction/data/replay_{tag}_seed{seed}.npz"


def _reconstruct_training_data(env_name, model_path, ref_data, family, params, run_key):
    """Reconstruct the training data for a given perturbation run."""
    if family == "seed_shift":
        seed = params.get("replay_seed")
        if seed is None:
            # Parse from run_key
            seed = int(run_key.split("_s")[-1])
        data = collect_replay(
            env_name=env_name, model_path=model_path,
            num_transitions=10000, seed=seed, deterministic=True,
        )
        return data["states"], data["actions"]

    elif family == "subsample":
        idx = params.get("subsample_idx", 0)
        fraction = params.get("fraction", 0.8)
        subs = generate_subsamples(ref_data, idx + 1, fraction, seed=42)
        return subs[idx]["states"], subs[idx]["actions"]

    elif family == "stratified":
        idx = params.get("subsample_idx", 0)
        fraction = params.get("fraction", 0.8)
        subs = generate_stratified_subsamples(ref_data, idx + 1, fraction, seed=42)
        return subs[idx]["states"], subs[idx]["actions"]

    elif family == "depth_variation":
        return ref_data["states"], ref_data["actions"]

    elif family == "feature_noise":
        nl = params.get("noise_level")
        if nl is None:
            # Parse from run_key
            nl = float(run_key.split("noise_")[-1])
        feature_ranges = compute_feature_ranges(ref_data)
        noisy = add_feature_noise(ref_data, nl, seed=42,
                                  feature_ranges=feature_ranges)
        return noisy["states"], noisy["actions"]

    else:
        print(f"  WARNING: Unknown family '{family}' for {run_key}")
        return None, None


# ── Shallow-tree ablation ────────────────────────────────────────────

def run_shallow_tree_ablation(env_name: str) -> dict:
    """Run DT with max_depth ∈ {3, 5, 7, None} × 5 seed-shift repeats.

    For each depth setting, collect F1, BRA, GRS, tree structure.
    """
    model_path = _get_model_path(env_name)
    ref_path = _get_replay_path(env_name)
    ref_data = load_replay_npz(ref_path)
    feature_names = ENV_FEATURE_NAMES.get(env_name)
    feature_ranges = compute_feature_ranges(ref_data)

    # Collect held-out
    heldout_data = collect_replay(
        env_name=env_name, model_path=model_path,
        num_transitions=5000, seed=HELDOUT_SEED, deterministic=True,
    )
    heldout_s = heldout_data["states"]
    heldout_a = heldout_data["actions"]

    ablation_results = {}

    for depth in ABLATION_DEPTHS:
        depth_key = str(depth) if depth is not None else "None"
        print(f"  [ablation] depth={depth_key}, {N_ABLATION_SEEDS} seed repeats...")

        run_data_list = []
        all_rule_sets = []
        all_preds = []
        all_threshold_sets = []
        structures = []

        for seed in SEED_SHIFT_SEEDS[:N_ABLATION_SEEDS]:
            # Collect replay with this seed
            data = collect_replay(
                env_name=env_name, model_path=model_path,
                num_transitions=10000, seed=seed, deterministic=True,
            )

            dt = DecisionTreeSurrogate(
                max_depth=depth,
                min_samples_leaf=5,
                random_state=42,
                feature_names=feature_names,
            )
            dt.fit(data["states"], data["actions"])
            rules = canonicalize_dt_rules(dt.get_rules())

            # Fidelity on held-out
            fid = dt.evaluate_fidelity(heldout_s, heldout_a)
            fid_pa = dt.evaluate_fidelity_per_action(heldout_s, heldout_a)

            # Structure
            struct = extract_tree_structure(dt)

            # Predictions on held-out (for BRA)
            preds = dt.predict(heldout_s)

            # Thresholds (for TD)
            thresholds = {int(k): v for k, v in dt.get_thresholds().items()}

            run_entry = {
                "seed": seed,
                "f1": fid["f1"],
                "accuracy": fid["accuracy"],
                "fidelity_per_action": fid_pa,
                "structure": struct,
            }
            run_data_list.append(run_entry)
            all_rule_sets.append(rules)
            all_preds.append(preds)
            all_threshold_sets.append(thresholds)
            structures.append(struct)

        # Compute stability across seed repeats
        grs_wj = mean_pairwise_jaccard(all_rule_sets, weighted=True)
        grs_ta = mean_pairwise_soft_jaccard(all_rule_sets, threshold_aware=True)
        fr = {i: float(feature_ranges[i]) for i in range(len(feature_ranges))}
        td = mean_pairwise_threshold_drift(all_threshold_sets, feature_ranges=fr)
        bra = compute_bra_from_predictions(all_preds)

        # Worst-action recall
        worst_recalls = []
        for rd in run_data_list:
            pa = rd["fidelity_per_action"]["per_action"]
            recalls = [v["recall"] for v in pa.values()]
            worst_recalls.append(min(recalls) if recalls else 0.0)

        # Aggregate structure stats
        depths_arr = [s["depth"] for s in structures]
        leaves_arr = [s["n_leaves"] for s in structures]
        rules_arr = [s["n_rules"] for s in structures]
        internal_arr = [s["n_internal_nodes"] for s in structures]

        ablation_results[depth_key] = {
            "max_depth_setting": depth,
            "n_repeats": N_ABLATION_SEEDS,
            "f1_mean": float(np.mean([r["f1"] for r in run_data_list])),
            "f1_std": float(np.std([r["f1"] for r in run_data_list])),
            "GRS_wj": float(grs_wj),
            "GRS_ta": float(grs_ta),
            "BRA": float(bra),
            "TD": float(td),
            "worst_action_recall_mean": float(np.mean(worst_recalls)),
            "structure_stats": {
                "depth_mean": float(np.mean(depths_arr)),
                "depth_std": float(np.std(depths_arr)),
                "depth_range": [int(min(depths_arr)), int(max(depths_arr))],
                "leaves_mean": float(np.mean(leaves_arr)),
                "leaves_std": float(np.std(leaves_arr)),
                "leaves_range": [int(min(leaves_arr)), int(max(leaves_arr))],
                "rules_mean": float(np.mean(rules_arr)),
                "rules_std": float(np.std(rules_arr)),
                "internal_mean": float(np.mean(internal_arr)),
                "internal_std": float(np.std(internal_arr)),
            },
            "per_run": run_data_list,
        }

    return ablation_results


# ── Main ─────────────────────────────────────────────────────────────

def run_env(env_name, skip_ablation=False):
    print(f"\n{'='*60}")
    print(f"  DT Structural Diagnostic: {env_name}")
    print(f"{'='*60}")

    env_tag = env_name.replace("-", "_").lower()
    out_dir = f"experiments/results/{env_tag}"
    os.makedirs(out_dir, exist_ok=True)

    model_path = _get_model_path(env_name)
    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found at {model_path}. Skipping.")
        return

    t0 = time.time()

    # Part 1: Extract structure from existing 21-run results
    print(f"\n  [Part 1] Extracting tree structure from existing runs...")
    structural = extract_from_existing_results(env_name)

    if structural:
        # Summary statistics
        depths = [s["depth"] for s in structural.values()]
        leaves = [s["n_leaves"] for s in structural.values()]
        rules = [s["n_rules"] for s in structural.values()]
        internals = [s["n_internal_nodes"] for s in structural.values()]

        # Top split feature distribution
        top_splits = [s["top_split_feature"] for s in structural.values()]
        split_counts = {}
        for f in top_splits:
            split_counts[f] = split_counts.get(f, 0) + 1

        print(f"    Depth: {np.mean(depths):.1f} ± {np.std(depths):.1f} "
              f"[{min(depths)}, {max(depths)}]")
        print(f"    Leaves: {np.mean(leaves):.1f} ± {np.std(leaves):.1f} "
              f"[{min(leaves)}, {max(leaves)}]")
        print(f"    Rules: {np.mean(rules):.1f} ± {np.std(rules):.1f} "
              f"[{min(rules)}, {max(rules)}]")
        print(f"    Internal nodes: {np.mean(internals):.1f} ± {np.std(internals):.1f}")
        print(f"    Top split feat distribution: {split_counts}")

        summary = {
            "n_runs": len(structural),
            "depth": {"mean": float(np.mean(depths)),
                      "std": float(np.std(depths)),
                      "min": int(min(depths)),
                      "max": int(max(depths))},
            "n_leaves": {"mean": float(np.mean(leaves)),
                         "std": float(np.std(leaves)),
                         "min": int(min(leaves)),
                         "max": int(max(leaves))},
            "n_rules": {"mean": float(np.mean(rules)),
                        "std": float(np.std(rules)),
                        "min": int(min(rules)),
                        "max": int(max(rules))},
            "n_internal_nodes": {"mean": float(np.mean(internals)),
                                 "std": float(np.std(internals))},
            "top_split_feature_distribution": {str(k): v
                                               for k, v in split_counts.items()},
        }
    else:
        summary = {}

    # Part 2: Shallow-tree ablation
    ablation = {}
    if not skip_ablation:
        print(f"\n  [Part 2] Shallow-tree ablation (depths={ABLATION_DEPTHS})...")
        ablation = run_shallow_tree_ablation(env_name)

        print(f"\n    Ablation summary:")
        print(f"    {'Depth':<8} {'F1':>8} {'GRS_wj':>8} {'GRS_ta':>8} "
              f"{'BRA':>8} {'TD':>8} {'Leaves':>8} {'Rules':>8}")
        print(f"    {'-'*72}")
        for dk, dv in ablation.items():
            ss = dv["structure_stats"]
            print(f"    {dk:<8} {dv['f1_mean']:>8.3f} {dv['GRS_wj']:>8.3f} "
                  f"{dv['GRS_ta']:>8.3f} {dv['BRA']:>8.3f} {dv['TD']:>8.3f} "
                  f"{ss['leaves_mean']:>8.1f} {ss['rules_mean']:>8.1f}")

    elapsed = time.time() - t0

    # Save
    output = {
        "schema_version": "dt_diagnostic_v1",
        "env": env_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "structural_analysis": {
            "summary": summary,
            "per_run": _serialize(structural),
        },
        "shallow_tree_ablation": _serialize(ablation),
    }

    out_path = os.path.join(out_dir, "tree_depth_ablation.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print(f"  Elapsed: {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="DT Structural Diagnostic (Appendix)")
    parser.add_argument("--env", default="all",
                        help="Environment name or 'all'")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip the shallow-tree ablation")
    args = parser.parse_args()

    envs = ENVS if args.env == "all" else [args.env]
    for env in envs:
        run_env(env, skip_ablation=args.skip_ablation)


if __name__ == "__main__":
    main()
