#!/usr/bin/env python
"""
Semantic merge pilot

Motivated by AutoRule: tests whether LLM-based semantic rule consolidation
can improve upon the purely numeric merge in Consensus CBS. This is a
small-scale proof-of-concept on selected matched groups, NOT a full
production system.

Protocol:
  1. Pick one environment (default: CartPole-v1) and one outer seed.
  2. Build internal CBS subsamples (B=5), extract matched groups.
  3. Select ~15–20 "interesting" groups: borderline-filtered (support
     just below τ) and poorly-merged groups.
  4. Convert each group's rules to natural-language templates.
  5. Query an LLM to judge semantic equivalence and produce merged rules.
  6. Convert LLM-merged rules back to executable CanonicalRule objects.
  7. Evaluate numeric-merge vs semantic-merge vs no-merge on BRA/F1.

Reports three quantities:
  - Average rule count reduction (semantic vs numeric).
  - Recovery rate: fraction of numeric-filtered rules recovered by semantic merge.
  - BRA/F1 comparison.

Requires: OPENAI_API_KEY environment variable (or --mock for offline testing).

Usage:
    python experiments/run_semantic_merge_pilot.py --env CartPole-v1
    python experiments/run_semantic_merge_pilot.py --env CartPole-v1 --mock
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from reproduction.cbs import CBSPipeline, Predicate, Condition, Rule
from reproduction.collect_replay import collect_replay, ENV_FEATURE_NAMES
from experiments.perturbations import (
    load_replay_npz,
    generate_subsamples,
    compute_feature_ranges,
)
from experiments.rule_matching import (
    CanonicalRule,
    CanonicalPredicate,
    canonicalize_rules,
    serialize_canonical_rules,
    rule_similarity_threshold_aware,
)
from experiments.consensus_merge import (
    run_cbs_on_data,
    _match_rules_across_runs,
    merge_rule_group,
    aggregate_thresholds,
    make_consensus_pipeline,
    _canonical_to_rule,
)
from experiments.run_stress_test import (
    EVAL_SEEDS,
    SUCCESS_THRESHOLDS,
    HELDOUT_SEED,
    _serialize,
)


# ── Feature name lookup ─────────────────────────────────────────────

FEATURE_DESCRIPTIONS = {
    "CartPole-v1": {
        0: "cart_position",
        1: "cart_velocity",
        2: "pole_angle",
        3: "pole_angular_velocity",
    },
    "LunarLander-v3": {
        0: "x_position",
        1: "y_position",
        2: "x_velocity",
        3: "y_velocity",
        4: "angle",
        5: "angular_velocity",
        6: "left_leg_contact",
        7: "right_leg_contact",
    },
    "MountainCar-v0": {
        0: "position",
        1: "velocity",
    },
}

ACTION_NAMES = {
    "CartPole-v1": {0: "push_left", 1: "push_right"},
    "LunarLander-v3": {0: "noop", 1: "fire_left", 2: "fire_main", 3: "fire_right"},
    "MountainCar-v0": {0: "push_left", 1: "no_push", 2: "push_right"},
}


# ── Rule to natural language ────────────────────────────────────────

def rule_to_nl(rule: CanonicalRule, env_name: str) -> str:
    """Convert a CanonicalRule to a natural language description."""
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


def group_to_nl(group: list[tuple], env_name: str,
                group_idx: int) -> str:
    """Convert a matched group to NL for LLM input."""
    lines = [f"Group {group_idx} (action={group[0][1].action}, "
             f"{len(group)} rules from {len(set(ri for ri, _ in group))} runs):"]
    for i, (run_idx, rule) in enumerate(group):
        lines.append(f"  Rule {i+1} (run {run_idx}): {rule_to_nl(rule, env_name)}")
    return "\n".join(lines)


# ── LLM interaction ─────────────────────────────────────────────────

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

USER_PROMPT_TEMPLATE = """Environment: {env_name}
Features: {feature_list}
Actions: {action_list}

Below are {n_groups} matched rule groups. For each, judge semantic equivalence
and optionally produce a merged rule.

{groups_text}

Respond with a JSON array of {n_groups} objects, one per group.
"""


def build_llm_prompt(groups: list, env_name: str) -> str:
    """Build the user prompt for the LLM."""
    feat_names = FEATURE_DESCRIPTIONS.get(env_name, {})
    act_names = ACTION_NAMES.get(env_name, {})

    feature_list = ", ".join(f"{k}: {v}" for k, v in sorted(feat_names.items()))
    action_list = ", ".join(f"{k}: {v}" for k, v in sorted(act_names.items()))

    groups_text = "\n\n".join(
        group_to_nl(g, env_name, i) for i, g in enumerate(groups))

    return USER_PROMPT_TEMPLATE.format(
        env_name=env_name,
        feature_list=feature_list,
        action_list=action_list,
        n_groups=len(groups),
        groups_text=groups_text,
    )


def call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI API. Requires OPENAI_API_KEY env var."""
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


def mock_llm_response(groups: list) -> str:
    """Generate plausible mock LLM responses for offline testing."""
    results = []
    for i, group in enumerate(groups):
        rules = [r for _, r in group]
        # Heuristic: if all rules share same action and >60% feature overlap, merge
        actions = set(r.action for r in rules)
        if len(actions) == 1:
            # Check feature consistency
            all_feats = [set(p.feature_idx for p in r.predicates) for r in rules]
            shared = set.intersection(*all_feats) if all_feats else set()
            union = set.union(*all_feats) if all_feats else set()
            overlap = len(shared) / len(union) if union else 0

            if overlap >= 0.5 and len(rules) >= 2:
                # Merge: use median bounds of shared features
                conditions = []
                for feat_idx in sorted(shared):
                    lbs = [p.lower_bound for r in rules for p in r.predicates
                           if p.feature_idx == feat_idx and p.lower_bound is not None]
                    ubs = [p.upper_bound for r in rules for p in r.predicates
                           if p.feature_idx == feat_idx and p.upper_bound is not None]
                    feat_name = f"feature_{feat_idx}"
                    conditions.append({
                        "feature": feat_name,
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
                    "reasoning": "Rules share same action and majority of conditions with similar thresholds.",
                })
            else:
                results.append({
                    "group_id": i,
                    "same_pattern": False,
                    "merged_rule": None,
                    "reasoning": "Rules cover different feature subsets despite same action.",
                })
        else:
            results.append({
                "group_id": i,
                "same_pattern": False,
                "merged_rule": None,
                "reasoning": "Rules have different actions.",
            })
    return json.dumps(results)


def parse_llm_response(response_text: str) -> list[dict]:
    """Parse LLM JSON response, handling markdown code blocks."""
    text = response_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
    return json.loads(text)


# ── Convert LLM merged rule back to executable ──────────────────────

def llm_merged_to_canonical(merged: dict, env_name: str) -> CanonicalRule:
    """Convert an LLM-produced merged rule back to CanonicalRule."""
    feat_names = FEATURE_DESCRIPTIONS.get(env_name, {})
    name_to_idx = {v: k for k, v in feat_names.items()}

    predicates = []
    for cond in merged["conditions"]:
        feat_name = cond["feature"]
        # Resolve feature index
        if feat_name in name_to_idx:
            feat_idx = name_to_idx[feat_name]
        elif feat_name.startswith("feature_"):
            feat_idx = int(feat_name.split("_")[1])
        else:
            # Try fuzzy match
            matches = [k for k, v in feat_names.items()
                       if v.lower() in feat_name.lower()
                       or feat_name.lower() in v.lower()]
            feat_idx = matches[0] if matches else 0

        lb = cond.get("lower")
        ub = cond.get("upper")

        # Compute level from midpoint
        if lb is not None and ub is not None:
            mid = (lb + ub) / 2.0
            level = 0.0  # default; will be overwritten if we can normalize
        else:
            mid = 0.0
            level = 0.0

        label = _level_to_label(level)

        predicates.append(CanonicalPredicate(
            feature_idx=feat_idx,
            level=level,
            level_label=label,
            lower_bound=lb,
            upper_bound=ub,
        ))

    predicates.sort(key=lambda p: p.feature_idx)

    return CanonicalRule(
        action=merged["action"],
        predicates=tuple(predicates),
        weight=merged.get("weight", 0.5),
        n_instances=merged.get("instances", 100),
    )


def _level_to_label(level: float) -> str:
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


# ── Main experiment ──────────────────────────────────────────────────

def run_semantic_merge_pilot(
    env_name: str,
    replay_seed: int = 0,
    n_bootstrap: int = 5,
    consensus_threshold: float = 0.7,
    similarity_cutoff: float = 0.8,
    use_mock: bool = False,
    llm_model: str = "gpt-4o-mini",
    max_groups: int = 20,
):
    """Run the semantic merge pilot experiment."""
    print(f"\n{'='*60}")
    print(f"  Semantic Merge Pilot: {env_name}")
    print(f"  {'Mock LLM' if use_mock else 'LLM model: ' + llm_model}")
    print(f"{'='*60}")

    env_tag = env_name.replace("-", "_").lower()
    model_path = f"reproduction/models/dqn_{env_tag}.zip"
    feature_names_list = ENV_FEATURE_NAMES.get(env_name)

    # 1. Load reference replay
    ref_path = f"reproduction/data/replay_{env_tag}_seed42.npz"
    ref_data = load_replay_npz(ref_path)
    print(f"  Reference replay: {len(ref_data['states'])} transitions")

    # 2. Collect replay with specified seed
    print(f"  Collecting replay (seed={replay_seed})...")
    data = collect_replay(
        env_name=env_name, model_path=model_path,
        num_transitions=10000, seed=replay_seed, deterministic=True,
    )

    # 3. Collect held-out for evaluation
    print(f"  Collecting held-out (seed={HELDOUT_SEED})...")
    heldout_data = collect_replay(
        env_name=env_name, model_path=model_path,
        num_transitions=5000, seed=HELDOUT_SEED, deterministic=True,
    )
    heldout_s = heldout_data["states"]
    heldout_a = heldout_data["actions"]

    # 4. Generate B subsamples and run CBS on each
    print(f"  Running {n_bootstrap} internal CBS subsamples...")
    rng = np.random.RandomState(42)
    n_total = len(data["states"])

    all_cbs = []
    all_rules = []
    all_thresholds = []

    for i in range(n_bootstrap):
        idx = rng.choice(n_total, size=int(n_total * 0.8), replace=False)
        s = data["states"][idx]
        a = data["actions"][idx]
        cbs, rules = run_cbs_on_data(s, a, env_name)
        all_cbs.append(cbs)
        all_rules.append(rules)
        all_thresholds.append(cbs.get_thresholds())

    total_input_rules = sum(len(rs) for rs in all_rules)
    print(f"  Total input rules: {total_input_rules}")

    # 5. Match rules across runs (per action)
    actions_set = sorted(set(r.action for rules in all_rules for r in rules))
    all_groups = []
    for action in actions_set:
        per_run = [[r for r in rules if r.action == action] for rules in all_rules]
        groups = _match_rules_across_runs(
            per_run, rho=similarity_cutoff, lambda1=0.6, lambda2=0.4)
        all_groups.extend(groups)

    min_support = int(np.ceil(consensus_threshold * n_bootstrap))
    kept_groups = []
    filtered_groups = []
    for group in all_groups:
        distinct_runs = len(set(run_idx for run_idx, _ in group))
        if distinct_runs >= min_support:
            kept_groups.append(group)
        else:
            filtered_groups.append(group)

    print(f"  Matched groups: {len(all_groups)} total, "
          f"{len(kept_groups)} kept (support >= {min_support}), "
          f"{len(filtered_groups)} filtered")

    # 6. Select interesting groups for LLM evaluation
    # a) Borderline filtered groups (support = min_support - 1)
    borderline = [g for g in filtered_groups
                  if len(set(ri for ri, _ in g)) == max(1, min_support - 1)]
    # b) Small kept groups (support = min_support exactly)
    marginal_kept = [g for g in kept_groups
                     if len(set(ri for ri, _ in g)) == min_support]
    # c) All filtered groups sorted by support desc
    filtered_by_support = sorted(
        filtered_groups,
        key=lambda g: len(set(ri for ri, _ in g)),
        reverse=True,
    )

    # Build selection: borderline first, then marginal kept, then top filtered
    selected_groups = []
    selected_labels = []  # "borderline_filtered", "marginal_kept", "filtered"

    for g in borderline[:max_groups // 3]:
        selected_groups.append(g)
        selected_labels.append("borderline_filtered")

    for g in marginal_kept[:max_groups // 3]:
        selected_groups.append(g)
        selected_labels.append("marginal_kept")

    remaining = max_groups - len(selected_groups)
    for g in filtered_by_support:
        if g not in selected_groups and remaining > 0:
            selected_groups.append(g)
            selected_labels.append("filtered")
            remaining -= 1

    # If still not enough, add remaining kept groups
    for g in kept_groups:
        if g not in selected_groups and len(selected_groups) < max_groups:
            selected_groups.append(g)
            selected_labels.append("kept")

    print(f"  Selected {len(selected_groups)} groups for LLM evaluation")
    label_counts = {}
    for l in selected_labels:
        label_counts[l] = label_counts.get(l, 0) + 1
    print(f"    Categories: {label_counts}")

    if not selected_groups:
        print("  WARNING: No groups selected. Aborting.")
        return {}

    # 7. Build LLM prompt and call
    # Process in batches to avoid token limits
    BATCH_SIZE = 10
    all_llm_results = []

    for batch_start in range(0, len(selected_groups), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(selected_groups))
        batch_groups = selected_groups[batch_start:batch_end]

        user_prompt = build_llm_prompt(batch_groups, env_name)
        print(f"  Calling LLM on groups {batch_start}–{batch_end-1} "
              f"(prompt length: {len(user_prompt)} chars)...")

        if use_mock:
            response_text = mock_llm_response(batch_groups)
        else:
            response_text = call_llm(SYSTEM_PROMPT, user_prompt, model=llm_model)

        try:
            batch_results = parse_llm_response(response_text)
            # Re-index group_ids to be globally unique
            for j, r in enumerate(batch_results):
                r["group_id"] = batch_start + j
            all_llm_results.extend(batch_results)
        except json.JSONDecodeError as e:
            print(f"  WARNING: Failed to parse LLM response: {e}")
            print(f"  Raw response: {response_text[:500]}")
            # Try to salvage partial results
            for j in range(len(batch_groups)):
                all_llm_results.append({
                    "group_id": batch_start + j,
                    "same_pattern": False,
                    "merged_rule": None,
                    "reasoning": "LLM response parsing failed",
                })

    # 8. Analyze LLM judgments
    n_same = sum(1 for r in all_llm_results if r.get("same_pattern"))
    n_diff = sum(1 for r in all_llm_results if not r.get("same_pattern"))
    n_merged = sum(1 for r in all_llm_results
                   if r.get("same_pattern") and r.get("merged_rule"))

    print(f"\n  LLM Judgments:")
    print(f"    Same pattern: {n_same}/{len(all_llm_results)}")
    print(f"    Different pattern: {n_diff}/{len(all_llm_results)}")
    print(f"    Merged rules produced: {n_merged}")

    # 9. Build three rule sets for comparison:
    #    A) numeric_merge: standard consensus (kept groups only)
    #    B) semantic_merge: kept + LLM-recovered filtered groups
    #    C) all_rules_union: union of all B subsample rules (no merge, rule-set voting style)

    level_values = all_cbs[0].level_values_
    level_labels = all_cbs[0].level_labels_

    # A) Numeric merge (baseline consensus)
    numeric_rules = []
    for group in kept_groups:
        rules_in_group = [rule for _, rule in group]
        cr = merge_rule_group(rules_in_group, level_values, level_labels)
        numeric_rules.append(cr)

    # B) Semantic merge: start with numeric, add LLM-recovered
    semantic_rules = list(numeric_rules)  # copy
    recovered_count = 0
    for i, llm_result in enumerate(all_llm_results):
        if not llm_result.get("same_pattern"):
            continue
        if llm_result.get("merged_rule") is None:
            continue

        label = selected_labels[i] if i < len(selected_labels) else "unknown"
        if label in ("borderline_filtered", "filtered"):
            # This was filtered by numeric merge — LLM wants to recover it
            try:
                cr = llm_merged_to_canonical(llm_result["merged_rule"], env_name)
                semantic_rules.append(cr)
                recovered_count += 1
            except Exception as e:
                print(f"  WARNING: Failed to convert LLM rule {i}: {e}")
        elif label == "marginal_kept":
            # Already in numeric; replace with LLM version if different
            try:
                cr = llm_merged_to_canonical(llm_result["merged_rule"], env_name)
                # Find and replace the corresponding numeric rule
                action = cr.action
                # Don't replace, just note — to keep comparison clean
            except Exception:
                pass

    print(f"\n  Rule set sizes:")
    print(f"    Numeric merge: {len(numeric_rules)} rules")
    print(f"    Semantic merge: {len(semantic_rules)} rules "
          f"(+{recovered_count} recovered)")
    print(f"    All sub-rules: {total_input_rules} rules")

    # 10. Evaluate all three rule sets
    agg_thresholds = aggregate_thresholds(all_thresholds, "median")

    def evaluate_ruleset(rules, label):
        """Build a CBS pipeline from rules and evaluate."""
        pipeline = make_consensus_pipeline(all_cbs[0], rules, agg_thresholds)
        fid = pipeline.evaluate_fidelity(heldout_s, heldout_a)
        fid_pa = pipeline.evaluate_fidelity_per_action(heldout_s, heldout_a)
        deploy = pipeline.evaluate_in_env(
            env_name, eval_seeds=EVAL_SEEDS,
            success_threshold=SUCCESS_THRESHOLDS.get(env_name),
        )
        preds = pipeline.predict(heldout_s)

        worst_recall = min(
            (v["recall"] for v in fid_pa["per_action"].values()),
            default=0.0,
        )

        return {
            "label": label,
            "n_rules": len(rules),
            "f1": fid["f1"],
            "accuracy": fid["accuracy"],
            "worst_action_recall": worst_recall,
            "fidelity_per_action": fid_pa,
            "E_CR": deploy["E_CR"],
            "E_CR_std": deploy["E_CR_std"],
            "success_rate": deploy["success_rate"],
            "predictions": preds.tolist(),
        }

    print(f"\n  Evaluating rule sets...")
    eval_numeric = evaluate_ruleset(numeric_rules, "numeric_merge")
    eval_semantic = evaluate_ruleset(semantic_rules, "semantic_merge")

    # Compute BRA between numeric and semantic predictions
    preds_n = np.array(eval_numeric["predictions"])
    preds_s = np.array(eval_semantic["predictions"])
    agreement = float(np.mean(preds_n == preds_s))

    print(f"\n  {'='*50}")
    print(f"  Results Comparison:")
    print(f"  {'Metric':<25} {'Numeric':>10} {'Semantic':>10} {'Delta':>10}")
    print(f"  {'-'*55}")
    for metric in ["n_rules", "f1", "worst_action_recall", "E_CR"]:
        vn = eval_numeric[metric]
        vs = eval_semantic[metric]
        delta = vs - vn if isinstance(vn, (int, float)) else "N/A"
        fmt = ".3f" if isinstance(vn, float) else "d"
        print(f"  {metric:<25} {vn:>10{fmt}} {vs:>10{fmt}} "
              f"{delta:>+10{fmt}}" if isinstance(delta, (int, float))
              else f"  {metric:<25} {vn:>10} {vs:>10}")
    print(f"  {'prediction_agreement':<25} {agreement:>10.3f}")
    print(f"  {'='*50}")

    # 11. Compile output
    output = {
        "schema_version": "semantic_merge_pilot_v1",
        "env": env_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "replay_seed": replay_seed,
            "n_bootstrap": n_bootstrap,
            "consensus_threshold": consensus_threshold,
            "similarity_cutoff": similarity_cutoff,
            "llm_model": llm_model if not use_mock else "mock",
            "max_groups": max_groups,
        },
        "matching_stats": {
            "total_input_rules": total_input_rules,
            "total_matched_groups": len(all_groups),
            "kept_groups": len(kept_groups),
            "filtered_groups": len(filtered_groups),
            "selected_for_llm": len(selected_groups),
            "selection_categories": label_counts,
        },
        "llm_judgments": {
            "n_same_pattern": n_same,
            "n_different_pattern": n_diff,
            "n_merged_produced": n_merged,
            "per_group": [
                {
                    "group_id": r["group_id"],
                    "category": selected_labels[r["group_id"]]
                    if r["group_id"] < len(selected_labels) else "unknown",
                    "same_pattern": r.get("same_pattern", False),
                    "has_merged_rule": r.get("merged_rule") is not None,
                    "reasoning": r.get("reasoning", ""),
                }
                for r in all_llm_results
            ],
        },
        "comparison": {
            "numeric_merge": {
                k: v for k, v in eval_numeric.items() if k != "predictions"
            },
            "semantic_merge": {
                k: v for k, v in eval_semantic.items() if k != "predictions"
            },
            "recovered_rules": recovered_count,
            "prediction_agreement": agreement,
        },
    }

    # Save
    out_dir = f"experiments/results/{env_tag}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "semantic_merge_pilot.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Merge Pilot (Appendix)")
    parser.add_argument("--env", default="CartPole-v1",
                        help="Environment name")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock LLM responses for offline testing")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="LLM model name (OpenAI)")
    parser.add_argument("--max-groups", type=int, default=20,
                        help="Max groups to send to LLM")
    parser.add_argument("--seed", type=int, default=0,
                        help="Replay collection seed")
    args = parser.parse_args()

    run_semantic_merge_pilot(
        env_name=args.env,
        replay_seed=args.seed,
        use_mock=args.mock,
        llm_model=args.model,
        max_groups=args.max_groups,
    )


if __name__ == "__main__":
    main()
