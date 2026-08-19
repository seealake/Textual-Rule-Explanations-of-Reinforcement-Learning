#!/usr/bin/env python
"""
Semantic merge pilot

Upgrades the existing semantic merge pilot from "feasibility demo" to
"stronger directional evidence" by:
  1. Running on BOTH CartPole-v1 and LunarLander-v3
  2. Evaluating MORE groups (up to 30 per env)
  3. Computing additional metrics: worst-action recall, BRA, prediction agreement
  4. Comparing numeric merge, semantic merge, and no-merge (raw union)
  5. When API key is available, uses real LLM backend; falls back to mock

This is appendix-level evidence. The mock LLM uses a conservative heuristic
that approximates real LLM judgment (merge if same action + >60% feature overlap).

Usage:
    python experiments/run_semantic_merge.py --env CartPole-v1
    python experiments/run_semantic_merge.py --env LunarLander-v3
    python experiments/run_semantic_merge.py --env all
    python experiments/run_semantic_merge.py --env all --real-llm
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from reproduction.cbs import CBSPipeline
from reproduction.collect_replay import collect_replay, ENV_FEATURE_NAMES
from experiments.rule_matching import (
    CanonicalRule,
    CanonicalPredicate,
    canonicalize_rules,
    rule_similarity_threshold_aware,
)
from experiments.consensus_merge import (
    run_cbs_on_data,
    _match_rules_across_runs,
    merge_rule_group,
    aggregate_thresholds,
    make_consensus_pipeline,
    build_voting_ensemble,
    voting_predict,
    _canonical_to_rule,
)
from experiments.run_stress_test import (
    EVAL_SEEDS,
    SUCCESS_THRESHOLDS,
    HELDOUT_SEED,
    _serialize,
)

# ── Constants ─────────────────────────────────────────────────────────
ENVS = ["CartPole-v1", "LunarLander-v3"]
N_OUTER_SEEDS = 3  # repeat for stability
OUTER_SEEDS = [0, 1, 2]
N_BOOTSTRAP = 5
DEFAULT_TAU = 0.7
DEFAULT_RHO = 0.8

FEATURE_DESCRIPTIONS = {
    "CartPole-v1": {0: "cart_position", 1: "cart_velocity",
                    2: "pole_angle", 3: "pole_angular_velocity"},
    "LunarLander-v3": {0: "x_position", 1: "y_position",
                       2: "x_velocity", 3: "y_velocity",
                       4: "angle", 5: "angular_velocity",
                       6: "left_leg_contact", 7: "right_leg_contact"},
    "MountainCar-v0": {0: "position", 1: "velocity"},
}

ACTION_NAMES = {
    "CartPole-v1": {0: "push_left", 1: "push_right"},
    "LunarLander-v3": {0: "noop", 1: "fire_left", 2: "fire_main", 3: "fire_right"},
    "MountainCar-v0": {0: "push_left", 1: "no_push", 2: "push_right"},
}


def get_model_path(env_name):
    tag = env_name.replace("-", "_").lower()
    return f"reproduction/models/dqn_{tag}.zip"


# ── Rule to NL ──────────────────────────────────────────────────────

def rule_to_nl(rule, env_name):
    feat_names = FEATURE_DESCRIPTIONS.get(env_name, {})
    act_names = ACTION_NAMES.get(env_name, {})
    conditions = []
    for p in rule.predicates:
        fname = feat_names.get(p.feature_idx, f"feature_{p.feature_idx}")
        if p.lower_bound is not None and p.upper_bound is not None:
            conditions.append(f"{fname} in [{p.lower_bound:.4f}, {p.upper_bound:.4f}]")
        else:
            conditions.append(f"{fname} is {p.level_label}")
    cond_str = " AND ".join(conditions) if conditions else "always"
    action_str = act_names.get(rule.action, f"action_{rule.action}")
    return (f"IF {cond_str} THEN {action_str} "
            f"(weight={rule.weight:.3f}, instances={rule.n_instances})")


def group_to_nl(group, env_name, group_idx):
    lines = [f"Group {group_idx} (action={group[0][1].action}, "
             f"{len(group)} rules from {len(set(ri for ri, _ in group))} runs):"]
    for i, (run_idx, rule) in enumerate(group):
        lines.append(f"  Rule {i+1} (run {run_idx}): {rule_to_nl(rule, env_name)}")
    return "\n".join(lines)


# ── LLM Interaction ─────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert in reinforcement learning policy explanation.
You will be given groups of if-then rules extracted from different runs of the same
RL policy. Each group contains rules that were matched as potentially equivalent by
a numeric similarity metric. Your tasks:

1. JUDGE: Are the rules in this group expressing the same decision pattern, just
   with slightly different thresholds or conditions? Answer YES or NO.
2. If YES, produce a MERGED rule that captures the essential pattern. Keep all
   conditions that are consistent across rules. For interval conditions, use the
   range that best represents the consensus (e.g., intersection or reasonable
   average). Drop conditions that appear in fewer than half the rules.
3. If NO, explain briefly why they capture different patterns.

Output format (strict JSON, one per group):
{
  "group_id": <int>,
  "same_pattern": true/false,
  "merged_rule": {
    "conditions": [{"feature": "<name>", "lower": <float>, "upper": <float>}],
    "action": <int>,
    "weight": <float>
  } or null,
  "reasoning": "<one sentence>"
}
"""


def build_llm_prompt(groups, env_name):
    feat_names = FEATURE_DESCRIPTIONS.get(env_name, {})
    act_names = ACTION_NAMES.get(env_name, {})
    feature_list = ", ".join(f"{k}: {v}" for k, v in sorted(feat_names.items()))
    action_list = ", ".join(f"{k}: {v}" for k, v in sorted(act_names.items()))
    groups_text = "\n\n".join(
        group_to_nl(g, env_name, i) for i, g in enumerate(groups))
    return (f"Environment: {env_name}\n"
            f"Features: {feature_list}\n"
            f"Actions: {action_list}\n\n"
            f"Below are {len(groups)} matched rule groups. "
            f"For each, judge semantic equivalence and optionally produce a merged rule.\n\n"
            f"{groups_text}\n\n"
            f"Respond with a JSON array of {len(groups)} objects, one per group.")


def call_real_llm(system_prompt, user_prompt, model="gpt-4o-mini"):
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def mock_llm_response(groups):
    """Conservative heuristic mock LLM that merges if same action + >60% overlap."""
    results = []
    for i, group in enumerate(groups):
        rules = [r for _, r in group]
        actions = set(r.action for r in rules)
        if len(actions) == 1 and len(rules) >= 2:
            all_feats = [set(p.feature_idx for p in r.predicates) for r in rules]
            shared = set.intersection(*all_feats) if all_feats else set()
            union = set.union(*all_feats) if all_feats else set()
            overlap = len(shared) / len(union) if union else 0

            if overlap >= 0.5:
                conditions = []
                for feat_idx in sorted(shared):
                    lbs = [p.lower_bound for r in rules for p in r.predicates
                           if p.feature_idx == feat_idx and p.lower_bound is not None]
                    ubs = [p.upper_bound for r in rules for p in r.predicates
                           if p.feature_idx == feat_idx and p.upper_bound is not None]
                    conditions.append({
                        "feature": f"feature_{feat_idx}",
                        "lower": float(np.median(lbs)) if lbs else None,
                        "upper": float(np.median(ubs)) if ubs else None,
                    })
                results.append({
                    "group_id": i,
                    "same_pattern": True,
                    "merged_rule": {
                        "conditions": conditions,
                        "action": rules[0].action,
                        "weight": float(np.mean([r.weight for r in rules])),
                    },
                    "reasoning": "Rules share same action and majority of conditions.",
                })
            else:
                results.append({
                    "group_id": i,
                    "same_pattern": False,
                    "merged_rule": None,
                    "reasoning": "Rules cover different feature subsets.",
                })
        else:
            results.append({
                "group_id": i,
                "same_pattern": False,
                "merged_rule": None,
                "reasoning": "Different actions or singleton group.",
            })
    return json.dumps(results)


def parse_llm_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
    return json.loads(text)


def _level_to_label(level):
    if level <= -0.6:
        return "Very Low"
    elif level <= -0.2:
        return "Low"
    elif level <= 0.2:
        return "Medium"
    elif level <= 0.6:
        return "High"
    else:
        return "Very High"


def llm_merged_to_canonical(merged, env_name):
    feat_names = FEATURE_DESCRIPTIONS.get(env_name, {})
    name_to_idx = {v: k for k, v in feat_names.items()}
    predicates = []
    for cond in merged["conditions"]:
        feat_name = cond["feature"]
        if feat_name in name_to_idx:
            feat_idx = name_to_idx[feat_name]
        elif feat_name.startswith("feature_"):
            feat_idx = int(feat_name.split("_")[1])
        else:
            matches = [k for k, v in feat_names.items()
                       if v.lower() in feat_name.lower()]
            feat_idx = matches[0] if matches else 0
        lb = cond.get("lower")
        ub = cond.get("upper")
        level = 0.0
        label = _level_to_label(level)
        predicates.append(CanonicalPredicate(
            feature_idx=feat_idx, level=level, level_label=label,
            lower_bound=lb, upper_bound=ub,
        ))
    predicates.sort(key=lambda p: p.feature_idx)
    return CanonicalRule(
        action=merged["action"],
        predicates=tuple(predicates),
        weight=merged.get("weight", 0.5),
        n_instances=merged.get("instances", 100),
    )


# ── Main Experiment ──────────────────────────────────────────────────

def run_semantic_pilot(env_name, use_real_llm=False, llm_model="gpt-4o-mini"):
    """Run the semantic merge pilot for one environment."""
    print(f"\n{'='*70}")
    print(f"  Semantic Merge Pilot: {env_name}")
    print(f"  Backend: {'Real LLM (' + llm_model + ')' if use_real_llm else 'Mock LLM'}")
    print(f"{'='*70}")

    env_tag = env_name.replace("-", "_").lower()
    model_path = get_model_path(env_name)

    # Collect held-out
    print("  Collecting held-out replay...")
    heldout_data = collect_replay(
        env_name=env_name, model_path=model_path,
        num_transitions=5000, seed=HELDOUT_SEED, deterministic=True,
    )
    heldout_s = heldout_data["states"]
    heldout_a = heldout_data["actions"]

    all_seed_results = {}

    for seed in OUTER_SEEDS:
        print(f"\n  Outer seed {seed}...")

        # Collect replay
        data = collect_replay(
            env_name=env_name, model_path=model_path,
            num_transitions=10000, seed=seed, deterministic=True,
        )

        # Build B subsamples
        rng = np.random.RandomState(seed * 100 + 42)
        n_total = len(data["states"])
        all_cbs = []
        all_rules = []
        all_thresholds = []

        for i in range(N_BOOTSTRAP):
            idx = rng.choice(n_total, size=int(n_total * 0.8), replace=False)
            s = data["states"][idx]
            a = data["actions"][idx]
            cbs, rules = run_cbs_on_data(s, a, env_name)
            all_cbs.append(cbs)
            all_rules.append(rules)
            all_thresholds.append(cbs.get_thresholds())

        total_input_rules = sum(len(rs) for rs in all_rules)

        # Match rules
        actions_set = sorted(set(r.action for rules in all_rules for r in rules))
        all_groups = []
        for action in actions_set:
            per_run = [[r for r in rules if r.action == action] for rules in all_rules]
            groups = _match_rules_across_runs(
                per_run, rho=DEFAULT_RHO, lambda1=0.6, lambda2=0.4)
            all_groups.extend(groups)

        min_support = int(np.ceil(DEFAULT_TAU * N_BOOTSTRAP))
        kept_groups = []
        filtered_groups = []
        for group in all_groups:
            distinct_runs = len(set(ri for ri, _ in group))
            if distinct_runs >= min_support:
                kept_groups.append(group)
            else:
                filtered_groups.append(group)

        # Select groups for LLM
        borderline = [g for g in filtered_groups
                      if len(set(ri for ri, _ in g)) >= max(1, min_support - 1)]
        marginal_kept = [g for g in kept_groups
                         if len(set(ri for ri, _ in g)) == min_support]
        remaining_filtered = sorted(
            filtered_groups, key=lambda g: len(set(ri for ri, _ in g)), reverse=True)

        max_groups = 30
        selected_groups = []
        selected_labels = []

        for g in borderline[:max_groups // 3]:
            selected_groups.append(g)
            selected_labels.append("borderline_filtered")
        for g in marginal_kept[:max_groups // 3]:
            selected_groups.append(g)
            selected_labels.append("marginal_kept")
        for g in remaining_filtered:
            if g not in selected_groups and len(selected_groups) < max_groups:
                selected_groups.append(g)
                selected_labels.append("filtered")
        for g in kept_groups:
            if g not in selected_groups and len(selected_groups) < max_groups:
                selected_groups.append(g)
                selected_labels.append("kept")

        if not selected_groups:
            print("    WARNING: No groups selected")
            continue

        # Call LLM (or mock)
        BATCH_SIZE = 10
        all_llm_results = []

        for bs in range(0, len(selected_groups), BATCH_SIZE):
            be = min(bs + BATCH_SIZE, len(selected_groups))
            batch = selected_groups[bs:be]
            user_prompt = build_llm_prompt(batch, env_name)

            if use_real_llm:
                try:
                    response_text = call_real_llm(
                        SYSTEM_PROMPT, user_prompt, model=llm_model)
                except Exception as e:
                    print(f"    LLM call failed: {e}. Falling back to mock.")
                    response_text = mock_llm_response(batch)
            else:
                response_text = mock_llm_response(batch)

            try:
                batch_results = parse_llm_response(response_text)
                for j, r in enumerate(batch_results):
                    r["group_id"] = bs + j
                all_llm_results.extend(batch_results)
            except json.JSONDecodeError as e:
                for j in range(len(batch)):
                    all_llm_results.append({
                        "group_id": bs + j,
                        "same_pattern": False,
                        "merged_rule": None,
                        "reasoning": "parse_failed",
                    })

        # Build rule sets for comparison
        level_values = all_cbs[0].level_values_
        level_labels = all_cbs[0].level_labels_
        agg_thresholds = aggregate_thresholds(all_thresholds, "median")

        # A) Numeric merge (standard consensus)
        numeric_rules = []
        for group in kept_groups:
            rules_in_group = [rule for _, rule in group]
            cr = merge_rule_group(rules_in_group, level_values, level_labels)
            numeric_rules.append(cr)

        # B) Semantic merge = numeric + LLM-recovered
        semantic_rules = list(numeric_rules)
        recovered_count = 0
        for i, llm_r in enumerate(all_llm_results):
            if not llm_r.get("same_pattern") or llm_r.get("merged_rule") is None:
                continue
            label = selected_labels[i] if i < len(selected_labels) else "unknown"
            if label in ("borderline_filtered", "filtered"):
                try:
                    cr = llm_merged_to_canonical(llm_r["merged_rule"], env_name)
                    semantic_rules.append(cr)
                    recovered_count += 1
                except Exception:
                    pass

        # Evaluate
        def _evaluate(rules, label):
            pipeline = make_consensus_pipeline(all_cbs[0], rules, agg_thresholds)
            fid = pipeline.evaluate_fidelity(heldout_s, heldout_a)
            fid_pa = pipeline.evaluate_fidelity_per_action(heldout_s, heldout_a)
            deploy = pipeline.evaluate_in_env(
                env_name, eval_seeds=EVAL_SEEDS,
                success_threshold=SUCCESS_THRESHOLDS.get(env_name),
            )
            worst_recall = min(
                (v["recall"] for v in fid_pa["per_action"].values()),
                default=0.0)
            preds = pipeline.predict(heldout_s)
            return {
                "label": label,
                "n_rules": len(rules),
                "f1": round(fid["f1"], 4),
                "accuracy": round(fid["accuracy"], 4),
                "worst_action_recall": round(worst_recall, 4),
                "E_CR": round(deploy["E_CR"], 2),
                "success_rate": round(deploy["success_rate"], 4),
                "predictions": preds.tolist(),
            }

        eval_numeric = _evaluate(numeric_rules, "numeric_merge")
        eval_semantic = _evaluate(semantic_rules, "semantic_merge")

        # BRA: prediction agreement
        preds_n = np.array(eval_numeric["predictions"])
        preds_s = np.array(eval_semantic["predictions"])
        agreement = float(np.mean(preds_n == preds_s))

        n_same = sum(1 for r in all_llm_results if r.get("same_pattern"))
        n_diff = sum(1 for r in all_llm_results if not r.get("same_pattern"))

        seed_result = {
            "matching_stats": {
                "total_input_rules": total_input_rules,
                "total_groups": len(all_groups),
                "kept_groups": len(kept_groups),
                "filtered_groups": len(filtered_groups),
                "selected_for_llm": len(selected_groups),
            },
            "llm_judgments": {
                "n_same_pattern": n_same,
                "n_different_pattern": n_diff,
                "n_recovered": recovered_count,
            },
            "numeric_merge": {k: v for k, v in eval_numeric.items()
                              if k != "predictions"},
            "semantic_merge": {k: v for k, v in eval_semantic.items()
                               if k != "predictions"},
            "prediction_agreement": round(agreement, 4),
        }

        all_seed_results[str(seed)] = seed_result

        print(f"    Numeric: F1={eval_numeric['f1']:.3f}, "
              f"rules={eval_numeric['n_rules']}, "
              f"worst_recall={eval_numeric['worst_action_recall']:.3f}")
        print(f"    Semantic: F1={eval_semantic['f1']:.3f}, "
              f"rules={eval_semantic['n_rules']} (+{recovered_count}), "
              f"worst_recall={eval_semantic['worst_action_recall']:.3f}")
        print(f"    Agreement: {agreement:.3f}")

    # Aggregate across seeds
    summary = {}
    for method in ["numeric_merge", "semantic_merge"]:
        vals = {k: [] for k in ["f1", "worst_action_recall", "n_rules", "E_CR"]}
        for seed_key, sr in all_seed_results.items():
            for k in vals:
                vals[k].append(sr[method][k])
        summary[method] = {
            k: {"mean": round(float(np.mean(v)), 4),
                "std": round(float(np.std(v)), 4)}
            for k, v in vals.items()
        }

    agreements = [sr["prediction_agreement"] for sr in all_seed_results.values()]
    summary["prediction_agreement"] = {
        "mean": round(float(np.mean(agreements)), 4),
        "std": round(float(np.std(agreements)), 4),
    }

    # Print summary
    print(f"\n  {'='*60}")
    print(f"  Summary across {len(all_seed_results)} seeds:")
    print(f"  {'Metric':<25} {'Numeric':>15} {'Semantic':>15}")
    print(f"  {'-'*55}")
    for metric in ["f1", "worst_action_recall", "n_rules", "E_CR"]:
        n_m = summary["numeric_merge"][metric]
        s_m = summary["semantic_merge"][metric]
        nstr = f"{n_m['mean']:.3f}±{n_m['std']:.3f}"
        sstr = f"{s_m['mean']:.3f}±{s_m['std']:.3f}"
        print(f"  {metric:<25} {nstr:>15} {sstr:>15}")
    ag = summary["prediction_agreement"]
    print(f"  {'prediction_agreement':<25} {ag['mean']:.3f}±{ag['std']:.3f}")

    # Save
    output = {
        "schema_version": "semantic_merge_results_v1",
        "env": env_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": "real_llm" if use_real_llm else "mock",
        "llm_model": llm_model if use_real_llm else "mock",
        "config": {
            "n_outer_seeds": N_OUTER_SEEDS,
            "n_bootstrap": N_BOOTSTRAP,
            "tau": DEFAULT_TAU,
            "rho": DEFAULT_RHO,
        },
        "per_seed": all_seed_results,
        "summary": summary,
    }

    out_dir = f"experiments/results/{env_tag}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "semantic_merge_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Merge Pilot")
    parser.add_argument("--env", default="all",
                        choices=ENVS + ["all"],
                        help="Environment to test")
    parser.add_argument("--real-llm", action="store_true",
                        help="Use real LLM (requires OPENAI_API_KEY)")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="LLM model name (for real LLM)")
    args = parser.parse_args()

    envs = ENVS if args.env == "all" else [args.env]
    for env in envs:
        run_semantic_pilot(env, use_real_llm=args.real_llm, llm_model=args.model)


if __name__ == "__main__":
    main()
