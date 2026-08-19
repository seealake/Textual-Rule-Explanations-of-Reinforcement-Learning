#!/usr/bin/env python
"""
SoftSupport consensus merge

Three extensions over the default consensus merge:

  1. **Behavior-aware matching** — adds a behavior similarity term to
     rule matching using a separate calibration state pool.
  2. **Soft-support retention** — replaces hard support count with soft
     max-similarity sum for group filtering.
  3. **Rare-action coverage safeguard** — detects low per-action recall
     on a validation replay and relaxes retention for under-covered
     actions.

All features are **off by default** (lambda_B=0, support_mode="hard",
safeguard_enabled=False).  When all flags are at defaults the output
is identical to the original ``build_consensus_ruleset()``.

Usage:
    from experiments.soft_support_merge import build_soft_support_consensus
    pipeline, rules, info = build_soft_support_consensus(base_data, env_name, soft_support_cfg)
"""
from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reproduction.cbs import CBSPipeline
from reproduction.collect_replay import ENV_FEATURE_NAMES
from experiments.perturbations import generate_subsamples
from experiments.rule_matching import (
    CanonicalRule,
    CanonicalPredicate,
    canonicalize_rules,
    predicate_overlap,
    rule_similarity_threshold_aware,
)
from experiments.consensus_merge import (
    run_cbs_on_data,
    merge_rule_group,
    aggregate_thresholds,
    make_consensus_pipeline,
    _canonical_to_rule,
)


# ── V2 Configuration ────────────────────────────────────────────────


@dataclass
class SoftSupportConfig:
    """All v2 parameters in one place; defaults = original behaviour."""

    # ── base consensus params (mirrored from v1) ──
    n_bootstrap: int = 5
    consensus_threshold: float = 0.7       # τ
    similarity_cutoff: float = 0.9         # ρ  (paper default for v2)
    lambda_P: float = 0.35                 # predicate overlap weight
    lambda_I: float = 0.45                 # interval similarity weight
    lambda_B: float = 0.0                  # behavior similarity weight (0 → v1)
    subsample_fraction: float = 0.8
    subsample_seed: int = 42
    use_maxf1: bool = False

    # ── soft support ──
    support_mode: str = "hard"             # "hard" or "soft"

    # ── rare-action safeguard ──
    safeguard_enabled: bool = False
    safeguard_floor: float = 0.10          # min per-action recall
    safeguard_topk: int = 2               # keep top-k groups for starved action
    safeguard_threshold_relax: float = 0.5 # relaxed τ for starved action
    safeguard_epsilon: float = 1e-3        # for inverse-freq weighting

    # ── calibration pool (for behavior-aware matching) ──
    calibration_n: int = 2000              # size of calibration state pool
    calibration_seed: int = 123            # seed separate from eval

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ── Behavior Signatures ─────────────────────────────────────────────


def _build_behavior_signature(
    rule: CanonicalRule,
    calibration_states_encoded: np.ndarray,
    n_actions: int,
) -> np.ndarray:
    """Build a behavior signature vector for a rule.

    For each calibration state, check coverage; if covered, record
    one-hot action output, else zero.  Flatten and L2-normalise.

    Parameters
    ----------
    rule : CanonicalRule
    calibration_states_encoded : (M, n_features) — encoded (level) values
    n_actions : int

    Returns
    -------
    np.ndarray of shape (M * n_actions,)  — L2-normalised
    """
    M = len(calibration_states_encoded)
    sig = np.zeros(M * n_actions, dtype=np.float32)

    for i, state in enumerate(calibration_states_encoded):
        covered = True
        for p in rule.predicates:
            if state[p.feature_idx] != p.level:
                covered = False
                break
        if covered:
            sig[i * n_actions + rule.action] = 1.0

    norm = np.linalg.norm(sig)
    if norm > 0:
        sig /= norm
    return sig


def _compute_behavior_similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Cosine similarity between two behaviour signatures."""
    dot = float(np.dot(sig_a, sig_b))
    return max(0.0, min(1.0, dot))  # clamp due to float precision


class BehaviorSignatureCache:
    """Cache behaviour signatures to avoid recomputation."""

    def __init__(self, calibration_states_encoded: np.ndarray, n_actions: int):
        self._cal_states = calibration_states_encoded
        self._n_actions = n_actions
        self._cache: dict[tuple, np.ndarray] = {}

    def get(self, rule: CanonicalRule) -> np.ndarray:
        key = (rule.action, rule.signature)
        if key not in self._cache:
            self._cache[key] = _build_behavior_signature(
                rule, self._cal_states, self._n_actions)
        return self._cache[key]

    def similarity(self, r1: CanonicalRule, r2: CanonicalRule) -> float:
        return _compute_behavior_similarity(self.get(r1), self.get(r2))


# ── V2 Similarity Function ──────────────────────────────────────────


def soft_rule_similarity(
    r1: CanonicalRule,
    r2: CanonicalRule,
    lambda_P: float = 0.35,
    lambda_I: float = 0.45,
    lambda_B: float = 0.20,
    sig_cache: Optional[BehaviorSignatureCache] = None,
) -> float:
    """V2 similarity: predicate overlap + interval IoU + behavior.

    When lambda_B == 0 this degrades exactly to the v1 threshold-aware
    similarity (with rescaled lambda1/lambda2 weights).
    """
    # Predicate overlap (Jaccard on feature sets)
    pred_sim = predicate_overlap(r1, r2)

    # Interval similarity (1 - threshold_aware_distance)
    # Reuse internal logic from rule_similarity_threshold_aware
    from experiments.rule_matching import threshold_aware_distance
    interval_sim = 1.0 - threshold_aware_distance(r1, r2)

    # Behavior similarity
    if lambda_B > 0 and sig_cache is not None:
        behav_sim = sig_cache.similarity(r1, r2)
    else:
        behav_sim = 0.0

    total_weight = lambda_P + lambda_I + lambda_B
    if total_weight == 0:
        return 0.0

    sim = (lambda_P * pred_sim + lambda_I * interval_sim
           + lambda_B * behav_sim) / total_weight
    return float(max(0.0, min(1.0, sim)))


# ── V2 Matching ──────────────────────────────────────────────────────


def _match_rules_across_runs_soft(
    per_run_rules: list[list[CanonicalRule]],
    rho: float,
    lambda_P: float,
    lambda_I: float,
    lambda_B: float,
    sig_cache: Optional[BehaviorSignatureCache],
) -> list[list[tuple[int, CanonicalRule]]]:
    """Order-invariant matching with v2 similarity (connected components)."""
    items = []
    for run_idx, rules in enumerate(per_run_rules):
        for rule in rules:
            items.append((run_idx, rule))

    n = len(items)
    if n == 0:
        return []

    # Build adjacency
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if items[i][0] == items[j][0]:
                continue
            sim = soft_rule_similarity(
                items[i][1], items[j][1],
                lambda_P=lambda_P, lambda_I=lambda_I,
                lambda_B=lambda_B, sig_cache=sig_cache,
            )
            if sim >= rho:
                adj[i].append(j)
                adj[j].append(i)

    # BFS connected components
    visited = [False] * n
    groups = []
    for start in range(n):
        if visited[start]:
            continue
        component = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            component.append(items[node])
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        groups.append(component)

    return groups


# ── Soft Support ─────────────────────────────────────────────────────


def _compute_soft_support(
    group: list[tuple[int, CanonicalRule]],
    representative: CanonicalRule,
    all_run_rules: list[list[CanonicalRule]],
    n_bootstrap: int,
    lambda_P: float,
    lambda_I: float,
    lambda_B: float,
    sig_cache: Optional[BehaviorSignatureCache],
) -> float:
    """Soft support: (1/B) * Σ_b max_{r ∈ R^b} Sim(r̄_g, r).

    Returns value in [0, 1].
    """
    total = 0.0
    for run_rules in all_run_rules:
        same_action = [r for r in run_rules if r.action == representative.action]
        if not same_action:
            continue
        best_sim = max(
            soft_rule_similarity(representative, r, lambda_P, lambda_I,
                               lambda_B, sig_cache)
            for r in same_action
        )
        total += best_sim
    return total / n_bootstrap


# ── Rare-Action Safeguard ────────────────────────────────────────────


def _estimate_per_action_recall(
    rules: list[CanonicalRule],
    pipeline: CBSPipeline,
    val_states: np.ndarray,
    val_actions: np.ndarray,
) -> dict[int, float]:
    """Estimate per-action recall on validation data."""
    preds = pipeline.predict(val_states)
    actions_set = sorted(set(int(a) for a in val_actions))
    recall = {}
    for a in actions_set:
        mask = val_actions == a
        n_true = int(mask.sum())
        if n_true == 0:
            recall[a] = 1.0
            continue
        n_correct = int(((preds == a) & mask).sum())
        recall[a] = n_correct / n_true
    return recall


def _apply_rare_action_safeguard(
    kept_groups: list,
    filtered_groups: list,
    group_diagnostics: list[dict],
    all_group_diagnostics: list[dict],
    pipeline: CBSPipeline,
    val_states: np.ndarray,
    val_actions: np.ndarray,
    cfg: SoftSupportConfig,
    level_values: np.ndarray,
    level_labels: list[str],
) -> tuple[list, list[dict]]:
    """Safeguard: rescue groups for under-represented actions.

    Strategy: for actions below floor recall, keep top-k filtered groups
    (by soft support or hard support) with relaxed threshold.
    """
    per_action_recall = _estimate_per_action_recall(
        # Build temporary rules from kept groups
        [merge_rule_group([r for _, r in g], level_values, level_labels)
         for g in kept_groups],
        pipeline, val_states, val_actions,
    )

    rescued = []
    rescued_diags = []
    for a, rec in per_action_recall.items():
        if rec >= cfg.safeguard_floor:
            continue
        # Find filtered groups for this action, sorted by support desc
        candidates = []
        for i, (fg, diag) in enumerate(zip(filtered_groups, all_group_diagnostics)):
            rules_in_group = [r for _, r in fg]
            if not rules_in_group or rules_in_group[0].action != a:
                continue
            # Use soft support if available, else hard
            sort_key = diag.get("soft_support", diag.get("hard_support", 0))
            candidates.append((sort_key, fg, diag))

        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, fg, diag in candidates[:cfg.safeguard_topk]:
            rescued.append(fg)
            diag_copy = dict(diag)
            diag_copy["rescued"] = True
            diag_copy["rescue_reason"] = f"action_{a}_recall_{rec:.3f}"
            rescued_diags.append(diag_copy)

    return rescued, rescued_diags


# ── Main Builder ─────────────────────────────────────────────────────


def build_soft_support_consensus(
    base_data: dict,
    env_name: str,
    cfg: Optional[SoftSupportConfig] = None,
    sample_weight: Optional[np.ndarray] = None,
    validation_data: Optional[dict] = None,
) -> tuple:
    """Build a Consensus CBS v2 rule set.

    Parameters
    ----------
    base_data : dict with "states", "actions" keys
    env_name : environment name
    cfg : v2 configuration (defaults = v1 behaviour)
    sample_weight : per-sample weights (for importance-weighted voting variant)
    validation_data : dict with "states", "actions" for safeguard evaluation.
                      If None and safeguard is enabled, uses a held-out
                      portion of base_data.

    Returns
    -------
    (consensus_pipeline, consensus_rules, build_info)
    """
    if cfg is None:
        cfg = SoftSupportConfig()

    n_total = len(base_data["states"])
    n_actions = len(np.unique(base_data["actions"]))

    # ── 1. Generate B subsamples ──
    rng = np.random.RandomState(cfg.subsample_seed)
    subsample_indices = []
    for _ in range(cfg.n_bootstrap):
        idx = rng.choice(n_total, size=int(n_total * cfg.subsample_fraction),
                         replace=False)
        subsample_indices.append(idx)

    # ── 2. Build calibration pool (separate from evaluation) ──
    sig_cache = None
    if cfg.lambda_B > 0:
        cal_rng = np.random.RandomState(cfg.calibration_seed)
        cal_size = min(cfg.calibration_n, n_total)
        cal_idx = cal_rng.choice(n_total, size=cal_size, replace=False)
        # We need encoded calibration states — fit a temporary CBS for encoding
        # or reuse the first subsample's CBS after fitting.
        # We'll encode after getting the first CBS pipeline.
        _cal_raw_states = base_data["states"][cal_idx]
    else:
        _cal_raw_states = None

    # ── 3. Run CBS on each subsample ──
    all_cbs = []
    all_rules = []
    all_thresholds = []

    for i, idx in enumerate(subsample_indices):
        s = base_data["states"][idx]
        a = base_data["actions"][idx]
        sw = sample_weight[idx] if sample_weight is not None else None
        cbs, rules = run_cbs_on_data(s, a, env_name,
                                      use_maxf1=cfg.use_maxf1,
                                      sample_weight=sw)
        all_cbs.append(cbs)
        all_rules.append(rules)
        all_thresholds.append(cbs.get_thresholds())

    # ── 4. Build behavior signature cache from calibration pool ──
    if cfg.lambda_B > 0 and _cal_raw_states is not None:
        # Encode calibration states using the first CBS pipeline.
        # NOTE: We intentionally call the internal CBSPipeline method
        # `_encode_states` here. At the time of writing there is no
        # public encoding API on CBSPipeline that allows reuse of the
        # fitted feature encoder without re-fitting a new pipeline.
        # This experiment code therefore relies on `_encode_states` as a
        # de facto stable internal API in order to:
        #   (a) obtain feature-space representations consistent with the
        #       fitted pipeline, and
        #   (b) avoid duplicating or re-training the encoding logic.
        # If CBSPipeline later exposes a public encoding method, this
        # call should be updated to use it instead.
        ref_cbs = all_cbs[0]
        cal_encoded = ref_cbs._encode_states(_cal_raw_states)  # (M, n_features)
        sig_cache = BehaviorSignatureCache(cal_encoded, n_actions)

    # ── 5. Group rules by action and match across runs ──
    actions_set = sorted(set(r.action for rules in all_rules for r in rules))
    all_groups = []
    n_input_rules = sum(len(rs) for rs in all_rules)

    for action in actions_set:
        per_run = []
        for rules in all_rules:
            per_run.append([r for r in rules if r.action == action])

        groups = _match_rules_across_runs_soft(
            per_run, rho=cfg.similarity_cutoff,
            lambda_P=cfg.lambda_P, lambda_I=cfg.lambda_I,
            lambda_B=cfg.lambda_B, sig_cache=sig_cache,
        )
        all_groups.extend(groups)

    # ── 6. Compute diagnostics and filter by support ──
    min_support_hard = int(np.ceil(cfg.consensus_threshold * cfg.n_bootstrap))

    level_values = all_cbs[0].level_values_
    level_labels = all_cbs[0].level_labels_

    kept_groups = []
    filtered_groups = []
    group_diagnostics = []        # for kept groups
    all_group_diagnostics = []    # for ALL groups (including filtered)

    for gidx, group in enumerate(all_groups):
        distinct_runs = len(set(run_idx for run_idx, _ in group))
        rules_in_group = [r for _, r in group]
        representative = merge_rule_group(rules_in_group, level_values,
                                          level_labels)

        diag = {
            "group_idx": gidx,
            "action": representative.action,
            "hard_support": distinct_runs,
            "group_size": len(group),
            "representative_signature": str(representative.signature),
        }

        # Compute soft support if enabled
        if cfg.support_mode == "soft":
            ss = _compute_soft_support(
                group, representative, all_rules, cfg.n_bootstrap,
                cfg.lambda_P, cfg.lambda_I, cfg.lambda_B, sig_cache,
            )
            diag["soft_support"] = float(ss)

        all_group_diagnostics.append(diag)

        # Filter decision
        if cfg.support_mode == "hard":
            keep = distinct_runs >= min_support_hard
        else:
            # Soft mode: use soft_support >= consensus_threshold
            keep = diag.get("soft_support", 0.0) >= cfg.consensus_threshold

        if keep:
            kept_groups.append(group)
            group_diagnostics.append(diag)
        else:
            filtered_groups.append(group)

    # ── 7. Aggregate each kept group ──
    consensus_rules = []
    for group in kept_groups:
        rules_in_group = [r for _, r in group]
        cr = merge_rule_group(rules_in_group, level_values, level_labels)
        consensus_rules.append(cr)

    consensus_rules.sort(key=lambda r: (r.action, r.signature))

    # ── 8. Rare-action safeguard ──
    safeguard_info = {"enabled": cfg.safeguard_enabled, "rescued_groups": 0}
    if cfg.safeguard_enabled:
        # Get validation data
        if validation_data is not None:
            val_s = validation_data["states"]
            val_a = validation_data["actions"]
        else:
            # Use a small held-out portion of base_data
            val_rng = np.random.RandomState(cfg.calibration_seed + 7)
            val_size = min(2000, n_total // 5)
            val_idx = val_rng.choice(n_total, size=val_size, replace=False)
            val_s = base_data["states"][val_idx]
            val_a = base_data["actions"][val_idx]

        # Build temporary pipeline to evaluate
        agg_thresholds = aggregate_thresholds(all_thresholds, "median")
        temp_pipeline = make_consensus_pipeline(
            all_cbs[0], consensus_rules, agg_thresholds)

        rescued, rescued_diags = _apply_rare_action_safeguard(
            kept_groups, filtered_groups,
            group_diagnostics, all_group_diagnostics,
            temp_pipeline, val_s, val_a, cfg,
            level_values, level_labels,
        )

        if rescued:
            for rg in rescued:
                rules_in_group = [r for _, r in rg]
                cr = merge_rule_group(rules_in_group, level_values, level_labels)
                consensus_rules.append(cr)
            group_diagnostics.extend(rescued_diags)
            consensus_rules.sort(key=lambda r: (r.action, r.signature))
            safeguard_info["rescued_groups"] = len(rescued)
            safeguard_info["rescued_details"] = rescued_diags

    # ── 9. Build pipeline ──
    agg_thresholds = aggregate_thresholds(all_thresholds, "median")
    consensus_pipeline = make_consensus_pipeline(
        all_cbs[0], consensus_rules, agg_thresholds)

    # ── 10. Build info ──
    per_action_counts = {}
    for a in actions_set:
        per_action_counts[int(a)] = sum(
            1 for r in consensus_rules if r.action == a)
    actions_lost = [a for a, c in per_action_counts.items() if c == 0]

    build_info = {
        "method": "soft_support_merge",
        "config": cfg.to_dict(),
        "n_bootstrap": cfg.n_bootstrap,
        "consensus_threshold": cfg.consensus_threshold,
        "similarity_cutoff": cfg.similarity_cutoff,
        "support_mode": cfg.support_mode,
        "lambda_P": cfg.lambda_P,
        "lambda_I": cfg.lambda_I,
        "lambda_B": cfg.lambda_B,
        "n_input_rules": n_input_rules,
        "n_matched_groups": len(all_groups),
        "n_kept_groups": len(kept_groups) + safeguard_info.get("rescued_groups", 0),
        "n_filtered_groups": len(filtered_groups) - safeguard_info.get("rescued_groups", 0),
        "n_consensus_rules": len(consensus_rules),
        "per_action_rule_counts": per_action_counts,
        "actions_lost": actions_lost,
        "group_diagnostics": group_diagnostics,
        "safeguard": safeguard_info,
    }

    return consensus_pipeline, consensus_rules, build_info
