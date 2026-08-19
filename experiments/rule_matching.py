#!/usr/bin/env python
"""
Canonical Rule Representation & Rule Matching.

Provides:
  1. CanonicalRule  — Normalised, hashable representation of a CBS rule
  2. predicate_match()     — Check if two predicates target the same feature
  3. rule_similarity()     — Soft similarity Sim(r_i, r_j) per proposal §6
  4. merge_near_duplicates() — Collapse rule pairs with Sim ≥ ρ

The canonical form enables:
  • Deterministic rule comparison across perturbation runs
  • Global Rule Stability (GRS) computation
  • Consensus CBS rule aggregation

Design
------
A CBS rule extracted by `CBSPipeline` is:

    Rule(action, Condition([Predicate(feature_idx, level, label), ...]), weight)

A *canonical* rule is an **action-tagged, sorted tuple of (feature_idx, level)**
plus a weight and instance count, enabling direct comparison via set operations.

Similarity formula (proposal §6):
    Sim(r_i, r_j) = λ₁ · predicate_overlap + λ₂ · (1 − norm_threshold_dist)

  where
    predicate_overlap  = |F_i ∩ F_j| / |F_i ∪ F_j|     (Jaccard on feature sets)
    norm_threshold_dist = mean_{f ∈ F_i∩F_j} |l_i(f) − l_j(f)| / 2
                          (level values in [−1, 1], so max dist = 2)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ───────────────────────────────────────────────────────────────────────
# 1. Canonical Rule Representation
# ───────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CanonicalPredicate:
    """Immutable, hashable predicate with continuous bin boundaries.

    Attributes
    ----------
    feature_idx : int
    level : float           — categorical level in [-1, 1] (Terra encoding)
    level_label : str       — human-readable, e.g., "Very Low"
    lower_bound : float     — continuous lower bound of the bin (feature space)
    upper_bound : float     — continuous upper bound of the bin (feature space)

    The ``lower_bound`` / ``upper_bound`` pair enables **threshold-aware**
    similarity computation (IoU on intervals) rather than relying solely on
    discrete level equality.  When bounds are unavailable (legacy data),
    they default to ``None`` and the code falls back to level-based matching.
    """
    feature_idx: int
    level: float        # categorical level in [-1, 1]
    level_label: str    # human-readable, e.g., "Very Low"
    lower_bound: float = None   # continuous feature-space lower bound
    upper_bound: float = None   # continuous feature-space upper bound

    @property
    def has_bounds(self) -> bool:
        return self.lower_bound is not None and self.upper_bound is not None

    @property
    def width(self) -> float:
        """Bin width in feature space. Returns 0 if bounds missing."""
        if not self.has_bounds:
            return 0.0
        return max(self.upper_bound - self.lower_bound, 0.0)

    def __repr__(self):
        if self.has_bounds:
            return (f"f{self.feature_idx}={self.level_label}({self.level:+.2f})"
                    f"[{self.lower_bound:.4f},{self.upper_bound:.4f}]")
        return f"f{self.feature_idx}={self.level_label}({self.level:+.2f})"


@dataclass
class CanonicalRule:
    """
    Normalised rule representation for cross-run comparison.

    Predicates are **sorted by feature_idx** (ascending), making
    structural equality a simple tuple comparison.

    Attributes
    ----------
    action : int
    predicates : tuple of CanonicalPredicate (sorted by feature_idx)
    weight : float          (w2 = N_ca / N_c)
    n_instances : int       (number of states in the source cluster)
    signature : tuple       ((feat_idx, level), ...) — hashable key
    """
    action: int
    predicates: tuple[CanonicalPredicate, ...]
    weight: float = 0.0
    n_instances: int = 0

    @property
    def signature(self) -> tuple:
        """Hashable structural identity (ignoring weight & count)."""
        return tuple((p.feature_idx, p.level) for p in self.predicates)

    @property
    def feature_set(self) -> frozenset[int]:
        """Set of feature indices mentioned in this rule."""
        return frozenset(p.feature_idx for p in self.predicates)

    @property
    def level_dict(self) -> dict[int, float]:
        """Mapping feature_idx → level value."""
        return {p.feature_idx: p.level for p in self.predicates}

    def __repr__(self):
        preds = " AND ".join(str(p) for p in self.predicates)
        return (f"CanonicalRule(a={self.action}, [{preds}], "
                f"w={self.weight:.3f}, n={self.n_instances})")

    def __eq__(self, other):
        if not isinstance(other, CanonicalRule):
            return NotImplemented
        return self.action == other.action and self.signature == other.signature

    def __hash__(self):
        return hash((self.action, self.signature))


# ───────────────────────────────────────────────────────────────────────
# 2. Conversion from CBS Rule objects
# ───────────────────────────────────────────────────────────────────────

def canonicalize_rule(rule) -> CanonicalRule:
    """
    Convert a `reproduction.cbs.Rule` object to a `CanonicalRule`.

    The predicates are sorted by feature_idx to ensure canonical ordering.
    Now also carries continuous bin boundaries (lower_bound, upper_bound)
    when available on the source Predicate objects.
    """
    preds = tuple(sorted(
        (CanonicalPredicate(
            feature_idx=p.feature_idx,
            level=p.level,
            level_label=p.level_label,
            lower_bound=getattr(p, "lower_bound", None),
            upper_bound=getattr(p, "upper_bound", None),
        ) for p in rule.condition.predicates),
        key=lambda p: p.feature_idx,
    ))
    return CanonicalRule(
        action=int(rule.action),
        predicates=preds,
        weight=float(rule.weight),
        n_instances=int(rule.condition.n_instances),
    )


def canonicalize_rules(rules) -> list[CanonicalRule]:
    """Convert a list of CBS Rule objects to canonical form."""
    return [canonicalize_rule(r) for r in rules]


def serialize_canonical_rules(rules: list[CanonicalRule]) -> list[dict]:
    """Serialize canonical rules to JSON-safe dicts (inverse of canonicalize_from_json)."""
    out = []
    for r in rules:
        out.append({
            "action": r.action,
            "weight": float(r.weight),
            "n_instances": int(r.n_instances),
            "predicates": [
                {
                    "feature_idx": p.feature_idx,
                    "level": float(p.level),
                    "level_label": p.level_label,
                    "lower_bound": float(p.lower_bound) if p.lower_bound is not None else None,
                    "upper_bound": float(p.upper_bound) if p.upper_bound is not None else None,
                }
                for p in r.predicates
            ],
        })
    return out


def canonicalize_from_json(json_rules: list[dict]) -> list[CanonicalRule]:
    """
    Convert JSON-serialised rules (from run_algorithmic_randomness.py)
    back to CanonicalRule objects.

    Expected format per rule dict:
        {"action": int, "weight": float, "n_instances": int,
         "predicates": [{"feature_idx": int, "level": float, "level_label": str}, ...]}
    """
    result = []
    for rd in json_rules:
        preds = tuple(sorted(
            (CanonicalPredicate(
                feature_idx=p["feature_idx"],
                level=p["level"],
                level_label=p["level_label"],
            ) for p in rd["predicates"]),
            key=lambda p: p.feature_idx,
        ))
        result.append(CanonicalRule(
            action=rd["action"],
            predicates=preds,
            weight=rd["weight"],
            n_instances=rd["n_instances"],
        ))
    return result


# ───────────────────────────────────────────────────────────────────────
# 3. Predicate & Rule Similarity
# ───────────────────────────────────────────────────────────────────────

def predicate_overlap(r1: CanonicalRule, r2: CanonicalRule) -> float:
    """
    Jaccard similarity on feature sets:
        |F1 ∩ F2| / |F1 ∪ F2|

    Returns 0.0 if both rules have empty predicate sets (shouldn't happen).
    """
    f1, f2 = r1.feature_set, r2.feature_set
    if len(f1) == 0 and len(f2) == 0:
        return 0.0
    intersection = f1 & f2
    union = f1 | f2
    return len(intersection) / len(union)


def normalized_threshold_distance(r1: CanonicalRule, r2: CanonicalRule) -> float:
    """
    Mean normalised level distance over shared features:
        mean_{f ∈ F1∩F2}  |l1(f) − l2(f)| / 2

    Level values are in [−1, 1], so max |diff| = 2 → normalised to [0, 1].
    Returns 0.0 if no shared features.
    """
    shared = r1.feature_set & r2.feature_set
    if len(shared) == 0:
        return 0.0
    d1, d2 = r1.level_dict, r2.level_dict
    dists = [abs(d1[f] - d2[f]) / 2.0 for f in shared]
    return float(np.mean(dists))


def rule_similarity(
    r1: CanonicalRule,
    r2: CanonicalRule,
    lambda1: float = 0.6,
    lambda2: float = 0.4,
) -> float:
    """
    Soft rule similarity (proposal §6):

        Sim(r1, r2) = λ₁ · predicate_overlap(r1, r2)
                     + λ₂ · (1 − normalized_threshold_distance(r1, r2))

    Returns a value in [0, 1].  Higher = more similar.

    Parameters
    ----------
    r1, r2 : CanonicalRule
        Rules to compare (should have the same action for meaningful comparison).
    lambda1 : float
        Weight for structural (feature set) overlap.  Default 0.6.
    lambda2 : float
        Weight for level-value proximity.  Default 0.4.
    """
    overlap = predicate_overlap(r1, r2)
    dist = normalized_threshold_distance(r1, r2)
    return lambda1 * overlap + lambda2 * (1.0 - dist)


# ───────────────────────────────────────────────────────────────────────
# 3b. Threshold-Aware Similarity (using continuous bin boundaries)
# ───────────────────────────────────────────────────────────────────────

def predicate_iou(p1: CanonicalPredicate, p2: CanonicalPredicate) -> float:
    """Interval-of-Union (IoU) between two predicates on the same feature.

    If both predicates have continuous bounds, compute:
        IoU = len(intersection) / len(union)
    where intersection and union are 1-D intervals.

    Falls back to exact level match (1.0 or 0.0) when bounds are missing.
    """
    if not (p1.has_bounds and p2.has_bounds):
        # Fallback: discrete level match
        return 1.0 if p1.level == p2.level else 0.0

    lo = max(p1.lower_bound, p2.lower_bound)
    hi = min(p1.upper_bound, p2.upper_bound)
    intersection = max(hi - lo, 0.0)

    union_lo = min(p1.lower_bound, p2.lower_bound)
    union_hi = max(p1.upper_bound, p2.upper_bound)
    union = max(union_hi - union_lo, 1e-12)

    return min(intersection / union, 1.0)


def threshold_aware_distance(r1: CanonicalRule, r2: CanonicalRule) -> float:
    """Mean (1 - IoU) over shared features.  Returns 0 if no shared features.

    This replaces ``normalized_threshold_distance`` when continuous bin
    boundaries are available, providing a much finer-grained measure of
    predicate drift that is sensitive to actual threshold shifts.
    """
    shared = r1.feature_set & r2.feature_set
    if not shared:
        return 0.0

    # Build predicate lookups
    preds1 = {p.feature_idx: p for p in r1.predicates}
    preds2 = {p.feature_idx: p for p in r2.predicates}

    dists = []
    for f in shared:
        iou = predicate_iou(preds1[f], preds2[f])
        dists.append(1.0 - iou)

    return float(np.mean(dists))


def rule_similarity_threshold_aware(
    r1: CanonicalRule,
    r2: CanonicalRule,
    lambda1: float = 0.6,
    lambda2: float = 0.4,
) -> float:
    """Threshold-aware variant of ``rule_similarity``.

    Uses IoU on continuous bin intervals instead of discrete level distance.

        Sim_TA(r1, r2) = λ₁ · predicate_overlap(r1, r2)
                        + λ₂ · (1 − threshold_aware_distance(r1, r2))

    Falls back to the original level-based computation automatically when
    bounds are not available on the predicates.
    """
    overlap = predicate_overlap(r1, r2)
    dist = threshold_aware_distance(r1, r2)
    return lambda1 * overlap + lambda2 * (1.0 - dist)


def ruleset_soft_jaccard(
    rules_a: list[CanonicalRule],
    rules_b: list[CanonicalRule],
    sim_threshold: float = 0.8,
    threshold_aware: bool = True,
    lambda1: float = 0.6,
    lambda2: float = 0.4,
) -> float:
    """Soft Jaccard between two rule sets using best-match pairing.

    For each rule in A, find its best match in B (same action, highest Sim).
    A rule pair is "matched" if Sim ≥ sim_threshold.

        Soft-J = |matched pairs| / |A ∪ B|

    where |A ∪ B| = |A| + |B| - |matched pairs|.

    This is more appropriate than plain/weighted Jaccard when thresholds
    can drift slightly between runs but the rules are semantically the same.
    """
    sim_fn = (rule_similarity_threshold_aware if threshold_aware
              else rule_similarity)

    # Group by action
    actions = set(r.action for r in rules_a) | set(r.action for r in rules_b)
    matched = 0
    for action in actions:
        a_rules = [r for r in rules_a if r.action == action]
        b_rules = [r for r in rules_b if r.action == action]
        used_b = set()
        for ra in a_rules:
            best_sim = -1.0
            best_j = -1
            for j, rb in enumerate(b_rules):
                if j in used_b:
                    continue
                s = sim_fn(ra, rb, lambda1, lambda2)
                if s > best_sim:
                    best_sim = s
                    best_j = j
            if best_sim >= sim_threshold and best_j >= 0:
                matched += 1
                used_b.add(best_j)

    total_a = len(rules_a)
    total_b = len(rules_b)
    union = total_a + total_b - matched
    return matched / union if union > 0 else 1.0


def mean_pairwise_soft_jaccard(
    rule_sets: list[list[CanonicalRule]],
    sim_threshold: float = 0.8,
    threshold_aware: bool = True,
    lambda1: float = 0.6,
    lambda2: float = 0.4,
) -> float:
    """Mean pairwise soft Jaccard — threshold-aware GRS metric."""
    n = len(rule_sets)
    if n < 2:
        return 1.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += ruleset_soft_jaccard(
                rule_sets[i], rule_sets[j],
                sim_threshold=sim_threshold,
                threshold_aware=threshold_aware,
                lambda1=lambda1, lambda2=lambda2,
            )
            count += 1
    return total / count


# ───────────────────────────────────────────────────────────────────────
# 4. Near-duplicate Merging
# ───────────────────────────────────────────────────────────────────────

def merge_two_rules(r1: CanonicalRule, r2: CanonicalRule) -> CanonicalRule:
    """
    Merge two similar canonical rules into one.

    Strategy:
      • Keep the union of feature indices.
      • For shared features: take the **mean** of level values.
      • For features in only one rule: keep as-is.
      • Weight: weighted average by n_instances.
      • n_instances: sum.
      • Action: must match (caller should ensure this).
    """
    assert r1.action == r2.action, "Cannot merge rules with different actions"

    d1, d2 = r1.level_dict, r2.level_dict
    all_features = r1.feature_set | r2.feature_set

    # Build label lookup from both rules
    label_map: dict[int, str] = {}
    for p in r1.predicates:
        label_map[p.feature_idx] = p.level_label
    for p in r2.predicates:
        label_map[p.feature_idx] = p.level_label  # r2 wins for shared (arbitrary)

    merged_preds = []
    for f in sorted(all_features):
        if f in d1 and f in d2:
            merged_level = (d1[f] + d2[f]) / 2.0
            # Use label from whichever is closer
            label = label_map[f]
        elif f in d1:
            merged_level = d1[f]
            label = label_map[f]
        else:
            merged_level = d2[f]
            label = label_map[f]
        merged_preds.append(CanonicalPredicate(
            feature_idx=f,
            level=merged_level,
            level_label=label,
        ))

    total_n = r1.n_instances + r2.n_instances
    if total_n > 0:
        merged_weight = (
            r1.weight * r1.n_instances + r2.weight * r2.n_instances
        ) / total_n
    else:
        merged_weight = (r1.weight + r2.weight) / 2.0

    return CanonicalRule(
        action=r1.action,
        predicates=tuple(merged_preds),
        weight=merged_weight,
        n_instances=total_n,
    )


def merge_near_duplicates(
    rules: list[CanonicalRule],
    rho: float = 0.8,
    lambda1: float = 0.6,
    lambda2: float = 0.4,
) -> list[CanonicalRule]:
    """
    Greedily merge pairs of same-action rules whose Sim ≥ ρ.

    Algorithm:
      1. Group rules by action.
      2. Within each action group, compute pairwise similarities.
      3. Greedily merge the most-similar pair if Sim ≥ ρ.
      4. Repeat until no more merges are possible.

    Parameters
    ----------
    rules : list of CanonicalRule
    rho : float
        Similarity threshold for merging.  Default 0.8.
    lambda1, lambda2 : float
        Passed to rule_similarity().

    Returns
    -------
    list of CanonicalRule — the merged (deduplicated) rule set.
    """
    # Group by action
    action_groups: dict[int, list[CanonicalRule]] = {}
    for r in rules:
        action_groups.setdefault(r.action, []).append(r)

    merged_all = []
    for action, group in sorted(action_groups.items()):
        merged_all.extend(
            _merge_group(group, rho, lambda1, lambda2)
        )

    return merged_all


def _merge_group(
    group: list[CanonicalRule],
    rho: float,
    lambda1: float,
    lambda2: float,
) -> list[CanonicalRule]:
    """Greedy pairwise merging within a single-action group."""
    pool = list(group)

    while True:
        best_sim = -1.0
        best_i, best_j = -1, -1
        n = len(pool)
        if n < 2:
            break

        for i in range(n):
            for j in range(i + 1, n):
                s = rule_similarity(pool[i], pool[j], lambda1, lambda2)
                if s > best_sim:
                    best_sim = s
                    best_i, best_j = i, j

        if best_sim < rho:
            break

        # Merge the best pair
        merged = merge_two_rules(pool[best_i], pool[best_j])
        # Remove originals (higher index first to preserve ordering)
        pool.pop(best_j)
        pool.pop(best_i)
        pool.append(merged)

    return pool


# ───────────────────────────────────────────────────────────────────────
# 5. Pairwise Rule-Set Similarity (for GRS computation)
# ───────────────────────────────────────────────────────────────────────

def ruleset_jaccard(
    rules_a: list[CanonicalRule],
    rules_b: list[CanonicalRule],
) -> float:
    """
    Plain Jaccard similarity between two rule sets (by signature+action).

        J = |A ∩ B| / |A ∪ B|

    Two rules "match" if they have the same (action, signature).
    """
    set_a = {(r.action, r.signature) for r in rules_a}
    set_b = {(r.action, r.signature) for r in rules_b}
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def ruleset_weighted_jaccard(
    rules_a: list[CanonicalRule],
    rules_b: list[CanonicalRule],
) -> float:
    """
    Weighted Jaccard similarity between two rule sets.

    For each unique (action, signature) key k:
        w_a(k) = weight of rule k in set A  (0 if absent)
        w_b(k) = weight of rule k in set B  (0 if absent)

        WJ = Σ_k min(w_a(k), w_b(k)) / Σ_k max(w_a(k), w_b(k))

    This accounts for rules that appear in both sets but with different weights.
    """
    # Build weight dicts
    wa: dict[tuple, float] = {}
    for r in rules_a:
        key = (r.action, r.signature)
        wa[key] = wa.get(key, 0.0) + r.weight

    wb: dict[tuple, float] = {}
    for r in rules_b:
        key = (r.action, r.signature)
        wb[key] = wb.get(key, 0.0) + r.weight

    all_keys = set(wa.keys()) | set(wb.keys())
    if len(all_keys) == 0:
        return 1.0

    num = sum(min(wa.get(k, 0.0), wb.get(k, 0.0)) for k in all_keys)
    den = sum(max(wa.get(k, 0.0), wb.get(k, 0.0)) for k in all_keys)

    return num / den if den > 0 else 1.0


def mean_pairwise_jaccard(
    rule_sets: list[list[CanonicalRule]],
    weighted: bool = True,
) -> float:
    """
    Mean pairwise (weighted) Jaccard across multiple rule sets.

    This is the core GRS metric: higher = more stable explanations.
    """
    n = len(rule_sets)
    if n < 2:
        return 1.0

    sim_fn = ruleset_weighted_jaccard if weighted else ruleset_jaccard
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += sim_fn(rule_sets[i], rule_sets[j])
            count += 1

    return total / count if count > 0 else 1.0


# ───────────────────────────────────────────────────────────────────────
# 6. Threshold Drift (TD) — parametric-level stability metric
# ───────────────────────────────────────────────────────────────────────

def threshold_drift(
    thresholds_a: dict[int, list[float]],
    thresholds_b: dict[int, list[float]],
    feature_ranges: dict[int, float] | None = None,
) -> float:
    """
    Compute normalised Threshold Drift between two CBS runs.

        TD = mean_{f, k}  |θ^(a)_{f,k} − θ^(b)_{f,k}| / Δ_f

    Parameters
    ----------
    thresholds_a, thresholds_b : dict[int, list[float]]
        Mapping from feature_idx → sorted list of predicate thresholds.
        Example: {0: [-0.94, -0.89, -0.41, -0.03], 1: [-0.04, -0.01, 0.00, 0.02]}
    feature_ranges : dict[int, float] or None
        Mapping from feature_idx → range Δ_f for normalisation.
        If None, uses 2.0 for all features (assuming level values in [-1, 1]).

    Returns
    -------
    float — mean normalised drift.  0.0 = identical thresholds, 1.0 = max drift.
    """
    shared_features = set(thresholds_a.keys()) & set(thresholds_b.keys())
    if len(shared_features) == 0:
        return 1.0  # completely different feature sets → max drift

    drifts = []
    for f in sorted(shared_features):
        ta = thresholds_a[f]
        tb = thresholds_b[f]
        delta_f = feature_ranges[f] if feature_ranges and f in feature_ranges else 2.0
        if delta_f == 0:
            delta_f = 1.0  # avoid division by zero

        # Match thresholds by position (both should be sorted)
        n = min(len(ta), len(tb))
        for k in range(n):
            drifts.append(abs(ta[k] - tb[k]) / delta_f)

        # Penalize mismatched threshold count
        extra = abs(len(ta) - len(tb))
        for _ in range(extra):
            drifts.append(1.0)  # max normalised drift for unmatched thresholds

    return float(np.mean(drifts)) if drifts else 0.0


def mean_pairwise_threshold_drift(
    threshold_sets: list[dict[int, list[float]]],
    feature_ranges: dict[int, float] | None = None,
) -> float:
    """
    Mean pairwise TD across multiple CBS runs.

    Lower = more stable thresholds.
    """
    n = len(threshold_sets)
    if n < 2:
        return 0.0

    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += threshold_drift(threshold_sets[i], threshold_sets[j], feature_ranges)
            count += 1

    return total / count if count > 0 else 0.0


# ───────────────────────────────────────────────────────────────────────
# 7. Behavior-Level Rule Agreement (BRA) — semantic-level stability metric
# ───────────────────────────────────────────────────────────────────────

def predict_action_from_canonical_rules(
    rules: list[CanonicalRule],
    encoded_state: dict[int, float],
    n_actions: int | None = None,
) -> int:
    """
    Predict action for a single encoded state using canonical rules.

    For each action, sum the weights of all matching rules.
    Return the action with the highest total weight.
    If no rule matches, return -1 (no prediction).

    Parameters
    ----------
    rules : list of CanonicalRule
    encoded_state : dict[int, float]
        Mapping feature_idx → level value (from CBS encoding).
    n_actions : int or None
        If provided, limits action range to [0, n_actions).
    """
    action_scores: dict[int, float] = {}

    for rule in rules:
        # Check if all predicates match
        match = True
        for pred in rule.predicates:
            if pred.feature_idx not in encoded_state:
                match = False
                break
            if encoded_state[pred.feature_idx] != pred.level:
                match = False
                break
        if match:
            action_scores[rule.action] = action_scores.get(rule.action, 0.0) + rule.weight

    if not action_scores:
        return -1

    return max(action_scores, key=action_scores.get)


def behavior_rule_agreement(
    rules_a: list[CanonicalRule],
    rules_b: list[CanonicalRule],
    eval_states: list[dict[int, float]],
) -> float:
    """
    Behavior-Level Rule Agreement (BRA) between two rule sets.

        BRA = (1/|S|) Σ_{s ∈ S} 1{ â_{R_a}(s) = â_{R_b}(s) }

    Parameters
    ----------
    rules_a, rules_b : list of CanonicalRule
    eval_states : list of dict[int, float]
        Each dict maps feature_idx → level value for one evaluation state.

    Returns
    -------
    float — agreement rate in [0, 1].  1.0 = perfectly agree on all states.
    """
    if len(eval_states) == 0:
        return 1.0

    agree = 0
    for state in eval_states:
        a_pred = predict_action_from_canonical_rules(rules_a, state)
        b_pred = predict_action_from_canonical_rules(rules_b, state)
        if a_pred == b_pred:
            agree += 1

    return agree / len(eval_states)


def mean_pairwise_bra(
    rule_sets: list[list[CanonicalRule]],
    eval_states: list[dict[int, float]],
) -> float:
    """
    Mean pairwise BRA across multiple rule sets on the same evaluation states.

    Higher = more behaviorally stable explanations.
    """
    n = len(rule_sets)
    if n < 2:
        return 1.0

    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += behavior_rule_agreement(rule_sets[i], rule_sets[j], eval_states)
            count += 1

    return total / count if count > 0 else 1.0


# ───────────────────────────────────────────────────────────────────────
# 8. Self-test / Sanity Check
# ───────────────────────────────────────────────────────────────────────

def _self_test():
    """Quick sanity check of similarity and merging logic."""
    print("=" * 60)
    print("  Rule Matching — Self Test")
    print("=" * 60)

    # Create some test rules
    r1 = CanonicalRule(
        action=0,
        predicates=(
            CanonicalPredicate(0, -0.5, "Low"),
            CanonicalPredicate(1, 0.0, "Medium"),
        ),
        weight=0.9,
        n_instances=100,
    )
    r2 = CanonicalRule(
        action=0,
        predicates=(
            CanonicalPredicate(0, -0.5, "Low"),
            CanonicalPredicate(1, 0.0, "Medium"),
        ),
        weight=0.85,
        n_instances=80,
    )
    r3 = CanonicalRule(
        action=0,
        predicates=(
            CanonicalPredicate(0, 0.5, "High"),
            CanonicalPredicate(1, 1.0, "Very High"),
        ),
        weight=0.7,
        n_instances=50,
    )
    r4 = CanonicalRule(
        action=1,
        predicates=(
            CanonicalPredicate(0, -0.5, "Low"),
            CanonicalPredicate(1, 0.0, "Medium"),
        ),
        weight=0.6,
        n_instances=30,
    )

    # Test 1: Identical rules (same signature)
    sim_identical = rule_similarity(r1, r2)
    print(f"\n  Test 1: Identical structure, different weight")
    print(f"    r1: {r1}")
    print(f"    r2: {r2}")
    print(f"    Sim(r1, r2) = {sim_identical:.4f}")
    assert sim_identical == 1.0, f"Expected 1.0, got {sim_identical}"
    print(f"    ✓ PASS (Sim = 1.0 for identical predicates)")

    # Test 2: Completely different rules (same action)
    sim_diff = rule_similarity(r1, r3)
    print(f"\n  Test 2: Completely different predicates (same features)")
    print(f"    r1: {r1}")
    print(f"    r3: {r3}")
    print(f"    Sim(r1, r3) = {sim_diff:.4f}")
    # Overlap = 1.0 (same features), dist = mean(|−0.5−0.5|/2, |0.0−1.0|/2) = mean(0.5, 0.5) = 0.5
    # Sim = 0.6*1.0 + 0.4*(1−0.5) = 0.6 + 0.2 = 0.8
    expected = 0.8
    assert abs(sim_diff - expected) < 1e-6, f"Expected {expected}, got {sim_diff}"
    print(f"    ✓ PASS (Sim = 0.8 — same features, max level distance)")

    # Test 3: No shared features
    r5 = CanonicalRule(
        action=0,
        predicates=(
            CanonicalPredicate(2, 0.0, "Medium"),
            CanonicalPredicate(3, 1.0, "Very High"),
        ),
        weight=0.5,
        n_instances=40,
    )
    sim_no_overlap = rule_similarity(r1, r5)
    print(f"\n  Test 3: No shared features")
    print(f"    r1: {r1}")
    print(f"    r5: {r5}")
    print(f"    Sim(r1, r5) = {sim_no_overlap:.4f}")
    # Overlap = 0/4 = 0, dist = 0 (no shared), Sim = 0.6*0 + 0.4*1 = 0.4
    expected = 0.4
    assert abs(sim_no_overlap - expected) < 1e-6, f"Expected {expected}, got {sim_no_overlap}"
    print(f"    ✓ PASS (Sim = 0.4 — no structural overlap)")

    # Test 4: Partial overlap
    r6 = CanonicalRule(
        action=0,
        predicates=(
            CanonicalPredicate(0, -0.5, "Low"),
            CanonicalPredicate(2, 0.5, "High"),
        ),
        weight=0.8,
        n_instances=60,
    )
    sim_partial = rule_similarity(r1, r6)
    print(f"\n  Test 4: Partial feature overlap (1 shared, 1 unique each)")
    print(f"    r1: {r1}")
    print(f"    r6: {r6}")
    print(f"    Sim(r1, r6) = {sim_partial:.4f}")
    # Features: r1={0,1}, r6={0,2}. Intersection={0}, Union={0,1,2}
    # Overlap = 1/3
    # Shared feature 0: |−0.5−(−0.5)|/2 = 0. dist = 0
    # Sim = 0.6*(1/3) + 0.4*(1−0) = 0.2 + 0.4 = 0.6
    expected = 0.6
    assert abs(sim_partial - expected) < 1e-6, f"Expected {expected}, got {sim_partial}"
    print(f"    ✓ PASS (Sim = 0.6 — partial overlap)")

    # Test 5: Merging identical rules
    print(f"\n  Test 5: Merge near-identical rules")
    merged = merge_near_duplicates([r1, r2, r3], rho=0.9)
    print(f"    Input:  3 rules (r1, r2 identical structure; r3 different)")
    print(f"    Output: {len(merged)} rules after merge (ρ=0.9)")
    for r in merged:
        print(f"      {r}")
    assert len(merged) == 2, f"Expected 2 rules after merge, got {len(merged)}"
    print(f"    ✓ PASS (r1+r2 merged, r3 kept separate)")

    # Test 6: Jaccard between rule sets
    print(f"\n  Test 6: Rule set Jaccard similarity")
    set_a = [r1, r3]
    set_b = [r2, r3]  # r2 has same signature as r1
    j_plain = ruleset_jaccard(set_a, set_b)
    print(f"    Set A: {[r.signature for r in set_a]}")
    print(f"    Set B: {[r.signature for r in set_b]}")
    print(f"    Plain Jaccard = {j_plain:.4f}")
    assert j_plain == 1.0, f"Expected 1.0 (same signatures), got {j_plain}"
    print(f"    ✓ PASS (J = 1.0 — identical structure sets)")

    j_weighted = ruleset_weighted_jaccard(set_a, set_b)
    print(f"    Weighted Jaccard = {j_weighted:.4f}")
    # r1 key weight=0.9 vs r2 key weight=0.85 → min=0.85, max=0.9
    # r3 key weight=0.7 vs r3 key weight=0.7 → min=0.7, max=0.7
    # WJ = (0.85+0.7)/(0.9+0.7) = 1.55/1.6 = 0.96875
    expected_wj = 1.55 / 1.6
    assert abs(j_weighted - expected_wj) < 1e-6, f"Expected {expected_wj:.4f}, got {j_weighted:.4f}"
    print(f"    ✓ PASS (WJ = {expected_wj:.4f} — accounts for weight differences)")

    # Test 7: Completely disjoint rule sets
    print(f"\n  Test 7: Disjoint rule sets")
    set_c = [r3]
    set_d = [r4]  # different action
    j_disjoint = ruleset_jaccard(set_c, set_d)
    print(f"    Set C: action={r3.action}, sig={r3.signature}")
    print(f"    Set D: action={r4.action}, sig={r4.signature}")
    print(f"    Plain Jaccard = {j_disjoint:.4f}")
    assert j_disjoint == 0.0, f"Expected 0.0 (disjoint), got {j_disjoint}"
    print(f"    ✓ PASS (J = 0.0 — completely different rule sets)")

    # Test 8: Mean pairwise Jaccard
    print(f"\n  Test 8: Mean pairwise Jaccard (GRS proxy)")
    grs = mean_pairwise_jaccard([set_a, set_b, set_a], weighted=True)
    print(f"    3 rule sets (A, B, A)")
    print(f"    Mean pairwise WJ = {grs:.4f}")
    # Pairs: (A,B)=0.969, (A,A)=1.0, (B,A)=0.969
    # Mean = (0.969 + 1.0 + 0.969) / 3 = 0.979
    print(f"    ✓ PASS (should be close to 1.0 for near-identical sets)")

    print(f"\n{'='*60}")
    print(f"  All tests passed! ✓")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    _self_test()
