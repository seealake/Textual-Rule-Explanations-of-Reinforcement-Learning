#!/usr/bin/env python
"""
Stability Metric Sanity Checks.

Verifies all four stability metrics (GRS, TD, BRA, and rule_similarity)
on controlled toy examples with known expected outcomes.

Run:
    python experiments/sanity_check_metrics.py
"""

from __future__ import annotations
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.rule_matching import (
    CanonicalPredicate,
    CanonicalRule,
    rule_similarity,
    predicate_overlap,
    normalized_threshold_distance,
    merge_near_duplicates,
    ruleset_jaccard,
    ruleset_weighted_jaccard,
    mean_pairwise_jaccard,
    threshold_drift,
    mean_pairwise_threshold_drift,
    predict_action_from_canonical_rules,
    behavior_rule_agreement,
    mean_pairwise_bra,
)

PASS_COUNT = 0
FAIL_COUNT = 0


def check(name: str, actual, expected, tol: float = 1e-6):
    """Assert approximate equality, print result."""
    global PASS_COUNT, FAIL_COUNT
    ok = abs(actual - expected) < tol
    symbol = "✓" if ok else "✗"
    print(f"    {symbol} {name}: got {actual:.6f}, expected {expected:.6f}")
    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
        print(f"      *** FAILED! Difference = {abs(actual - expected):.2e} ***")


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ═══════════════════════════════════════════════════════════════
# Helper: build toy rules
# ═══════════════════════════════════════════════════════════════

def make_rule(action, preds, weight=0.9, n=100):
    """Shortcut to build a CanonicalRule from (feat, level, label) tuples."""
    return CanonicalRule(
        action=action,
        predicates=tuple(
            CanonicalPredicate(f, l, lab) for f, l, lab in preds
        ),
        weight=weight,
        n_instances=n,
    )


# ═══════════════════════════════════════════════════════════════
# SECTION 1: GRS — Global Rule Stability (Jaccard-based)
# ═══════════════════════════════════════════════════════════════

def test_grs():
    section("GRS (Global Rule Stability) Sanity Checks")

    # ── 1a: Identical rule sets → GRS = 1.0 ──
    rules_A = [
        make_rule(0, [(0, -0.5, "Low"), (1, 0.0, "Medium")], weight=0.9),
        make_rule(0, [(0, 0.5, "High"), (1, 1.0, "Very High")], weight=0.7),
        make_rule(1, [(0, 0.0, "Medium")], weight=0.8),
    ]
    rules_B = [
        make_rule(0, [(0, -0.5, "Low"), (1, 0.0, "Medium")], weight=0.9),
        make_rule(0, [(0, 0.5, "High"), (1, 1.0, "Very High")], weight=0.7),
        make_rule(1, [(0, 0.0, "Medium")], weight=0.8),
    ]
    print("\n  1a. Identical rule sets")
    check("Plain Jaccard", ruleset_jaccard(rules_A, rules_B), 1.0)
    check("Weighted Jaccard", ruleset_weighted_jaccard(rules_A, rules_B), 1.0)
    check("GRS (mean pairwise, 2 identical sets)", mean_pairwise_jaccard([rules_A, rules_B]), 1.0)

    # ── 1b: Completely different rule sets → GRS = 0.0 ──
    rules_C = [
        make_rule(0, [(0, -1.0, "Very Low"), (1, -1.0, "Very Low")], weight=0.5),
        make_rule(2, [(0, 1.0, "Very High")], weight=0.3),
    ]
    rules_D = [
        make_rule(1, [(0, 0.0, "Medium"), (1, 0.5, "High")], weight=0.6),
        make_rule(0, [(0, 0.0, "Medium"), (1, -0.5, "Low")], weight=0.4),
    ]
    print("\n  1b. Completely disjoint rule sets (different actions or signatures)")
    check("Plain Jaccard", ruleset_jaccard(rules_C, rules_D), 0.0)
    check("Weighted Jaccard", ruleset_weighted_jaccard(rules_C, rules_D), 0.0)
    check("GRS (mean pairwise, 2 disjoint sets)", mean_pairwise_jaccard([rules_C, rules_D]), 0.0)

    # ── 1c: Partial overlap → 0 < GRS < 1 ──
    # A has 3 rules, E shares 2 of them
    rules_E = [
        make_rule(0, [(0, -0.5, "Low"), (1, 0.0, "Medium")], weight=0.9),  # same as A
        make_rule(0, [(0, 0.5, "High"), (1, 1.0, "Very High")], weight=0.7),  # same as A
        make_rule(1, [(0, 1.0, "Very High")], weight=0.5),  # DIFFERENT from A
    ]
    print("\n  1c. Partial overlap (2/4 unique rules shared)")
    j_plain = ruleset_jaccard(rules_A, rules_E)
    j_weighted = ruleset_weighted_jaccard(rules_A, rules_E)
    # A keys: {(0, sig1), (0, sig2), (1, sig3)}  E keys: {(0, sig1), (0, sig2), (1, sig4)}
    # Intersection: {(0, sig1), (0, sig2)} = 2
    # Union: {(0,sig1), (0,sig2), (1,sig3), (1,sig4)} = 4
    # Plain J = 2/4 = 0.5
    check("Plain Jaccard (2/4)", j_plain, 0.5)
    assert 0.0 < j_weighted < 1.0, f"Weighted Jaccard should be in (0,1), got {j_weighted}"
    print(f"    ✓ Weighted Jaccard = {j_weighted:.4f} (in (0, 1) as expected)")
    global PASS_COUNT
    PASS_COUNT += 1

    # ── 1d: Same structure, different weights → WJ < 1.0 but plain J = 1.0 ──
    rules_F = [
        make_rule(0, [(0, -0.5, "Low"), (1, 0.0, "Medium")], weight=0.5),  # weight differs
        make_rule(0, [(0, 0.5, "High"), (1, 1.0, "Very High")], weight=0.3),  # weight differs
        make_rule(1, [(0, 0.0, "Medium")], weight=0.2),  # weight differs
    ]
    print("\n  1d. Same structure, different weights")
    check("Plain Jaccard (structure only)", ruleset_jaccard(rules_A, rules_F), 1.0)
    wj = ruleset_weighted_jaccard(rules_A, rules_F)
    assert 0.0 < wj < 1.0, f"WJ should be < 1.0 when weights differ, got {wj}"
    # min(0.9,0.5)+min(0.7,0.3)+min(0.8,0.2) = 0.5+0.3+0.2 = 1.0
    # max(0.9,0.5)+max(0.7,0.3)+max(0.8,0.2) = 0.9+0.7+0.8 = 2.4
    # WJ = 1.0/2.4 ≈ 0.4167
    check("Weighted Jaccard (weights differ)", wj, 1.0 / 2.4)


# ═══════════════════════════════════════════════════════════════
# SECTION 2: TD — Threshold Drift
# ═══════════════════════════════════════════════════════════════

def test_td():
    section("TD (Threshold Drift) Sanity Checks")

    # ── 2a: Identical thresholds → TD = 0.0 ──
    ta = {0: [-0.94, -0.41, -0.03], 1: [-0.04, -0.01, 0.00, 0.02]}
    tb = {0: [-0.94, -0.41, -0.03], 1: [-0.04, -0.01, 0.00, 0.02]}
    print("\n  2a. Identical thresholds")
    check("TD (identical)", threshold_drift(ta, tb), 0.0)

    # ── 2b: Known shift → exact TD ──
    # Feature 0: range = 1.8 (position range [-1.2, 0.6])
    # Feature 1: range = 0.14 (velocity range [-0.07, 0.07])
    tc = {0: [-0.94, -0.41, -0.03], 1: [-0.04, -0.01, 0.00, 0.02]}
    td_shifted = {0: [-0.94, -0.20, -0.03], 1: [-0.04, -0.01, 0.00, 0.02]}
    # Only feature 0, threshold[1] differs: |-0.41 - (-0.20)| = 0.21
    # With feature_ranges: {0: 1.8, 1: 0.14}
    # Normalised drifts: f0: [0, 0.21/1.8, 0] = [0, 0.1167, 0]
    #                    f1: [0, 0, 0, 0]
    # Mean = (0 + 0.1167 + 0 + 0 + 0 + 0 + 0) / 7 = 0.01667
    ranges = {0: 1.8, 1: 0.14}
    print("\n  2b. Known single-threshold shift (feature 0, threshold[1]: -0.41 → -0.20)")
    td_val = threshold_drift(tc, td_shifted, feature_ranges=ranges)
    expected_td = 0.21 / 1.8 / 7  # one drift value out of 7 total
    check("TD (single shift)", td_val, expected_td)

    # ── 2c: All thresholds shifted uniformly → TD = shift/range ──
    te = {0: [0.0, 0.5, 1.0]}
    tf = {0: [0.1, 0.6, 1.1]}
    # All shift by 0.1, with default range 2.0
    # TD = mean(0.1/2.0, 0.1/2.0, 0.1/2.0) = 0.05
    print("\n  2c. Uniform shift of 0.1 (default range=2.0)")
    check("TD (uniform shift)", threshold_drift(te, tf), 0.05)

    # ── 2d: Different number of thresholds → penalty ──
    tg = {0: [0.0, 0.5]}      # 2 thresholds
    th = {0: [0.0, 0.5, 1.0]} # 3 thresholds
    # Matched: [|0-0|/2, |0.5-0.5|/2] = [0, 0]
    # Unmatched: 1 extra → penalty 1.0
    # TD = mean(0, 0, 1.0) = 1/3
    print("\n  2d. Different threshold counts (2 vs 3)")
    check("TD (mismatched count)", threshold_drift(tg, th), 1.0 / 3.0)

    # ── 2e: Mean pairwise TD across 3 identical sets → 0.0 ──
    print("\n  2e. Mean pairwise TD across 3 identical threshold sets")
    check("Mean pairwise TD (identical)", mean_pairwise_threshold_drift([ta, ta, ta]), 0.0)

    # ── 2f: Max drift (opposite extremes) ──
    ti = {0: [-1.0]}
    tj = {0: [1.0]}
    # Drift = |−1 − 1| / 2.0 = 1.0
    print("\n  2f. Max drift (−1.0 vs +1.0, range=2.0)")
    check("TD (max drift)", threshold_drift(ti, tj), 1.0)


# ═══════════════════════════════════════════════════════════════
# SECTION 3: BRA — Behavior-Level Rule Agreement
# ═══════════════════════════════════════════════════════════════

def test_bra():
    section("BRA (Behavior-Level Rule Agreement) Sanity Checks")

    # Build two simple rule sets for a 2-feature, 3-action "MountainCar-like" scenario
    # Rule set A:
    #   a=0 IF f0=Low AND f1=Medium  (w=0.9)
    #   a=2 IF f0=High AND f1=High   (w=0.8)
    rules_A = [
        make_rule(0, [(0, -0.5, "Low"), (1, 0.0, "Medium")], weight=0.9),
        make_rule(2, [(0, 0.5, "High"), (1, 0.5, "High")], weight=0.8),
    ]
    # Rule set B (identical to A):
    rules_B = [
        make_rule(0, [(0, -0.5, "Low"), (1, 0.0, "Medium")], weight=0.9),
        make_rule(2, [(0, 0.5, "High"), (1, 0.5, "High")], weight=0.8),
    ]

    # Evaluation states (encoded as feature_idx → level)
    eval_states = [
        {0: -0.5, 1: 0.0},   # matches a=0 rule
        {0: 0.5, 1: 0.5},    # matches a=2 rule
        {0: 0.0, 1: 0.0},    # no match → -1
        {0: -0.5, 1: 0.5},   # no match → -1
        {0: 0.5, 1: 0.0},    # no match → -1
    ]

    # ── 3a: Identical rule sets → BRA = 1.0 ──
    print("\n  3a. Identical rule sets")
    bra = behavior_rule_agreement(rules_A, rules_B, eval_states)
    check("BRA (identical rules)", bra, 1.0)

    # ── 3b: Completely different rule sets ──
    rules_C = [
        make_rule(1, [(0, -0.5, "Low"), (1, 0.0, "Medium")], weight=0.9),  # same condition, different action
        make_rule(0, [(0, 0.5, "High"), (1, 0.5, "High")], weight=0.8),    # same condition, different action
    ]
    print("\n  3b. Same conditions, swapped actions")
    bra_diff = behavior_rule_agreement(rules_A, rules_C, eval_states)
    # State 0: A→0, C→1 (disagree)
    # State 1: A→2, C→0 (disagree)
    # State 2: A→-1, C→-1 (agree! both no match)
    # State 3: A→-1, C→-1 (agree)
    # State 4: A→-1, C→-1 (agree)
    # BRA = 3/5 = 0.6
    check("BRA (swapped actions, 3/5 agree via no-match)", bra_diff, 3.0 / 5.0)

    # ── 3c: One rule set is empty → all predictions are -1 ──
    rules_empty = []
    print("\n  3c. One empty rule set")
    bra_empty = behavior_rule_agreement(rules_A, rules_empty, eval_states)
    # A matches 2/5 states (0,2). For those, A gives action, empty gives -1 → disagree
    # For 3/5 unmatched: both give -1 → agree
    # BRA = 3/5 = 0.6
    check("BRA (one empty, 3/5 agree via no-match)", bra_empty, 3.0 / 5.0)

    # ── 3d: Both empty → BRA = 1.0 (vacuously agree) ──
    print("\n  3d. Both rule sets empty")
    bra_both_empty = behavior_rule_agreement([], [], eval_states)
    check("BRA (both empty)", bra_both_empty, 1.0)

    # ── 3e: Perfect disagreement (every state matched, always different action) ──
    # All states match and predict different actions
    rules_D = [
        make_rule(0, [(0, -0.5, "Low")], weight=0.9),
        make_rule(0, [(0, 0.0, "Medium")], weight=0.9),
        make_rule(0, [(0, 0.5, "High")], weight=0.9),
    ]
    rules_E = [
        make_rule(1, [(0, -0.5, "Low")], weight=0.9),
        make_rule(1, [(0, 0.0, "Medium")], weight=0.9),
        make_rule(1, [(0, 0.5, "High")], weight=0.9),
    ]
    states_for_de = [
        {0: -0.5, 1: 0.0},
        {0: 0.0, 1: 0.0},
        {0: 0.5, 1: 0.5},
    ]
    print("\n  3e. Perfect disagreement (all matched, always different action)")
    bra_full_disagree = behavior_rule_agreement(rules_D, rules_E, states_for_de)
    check("BRA (complete disagreement)", bra_full_disagree, 0.0)

    # ── 3f: Verify predict_action_from_canonical_rules weight summation ──
    # Two rules for same action with same condition → weights add up
    rules_weighted = [
        make_rule(0, [(0, 0.0, "Medium")], weight=0.3),
        make_rule(0, [(0, 0.0, "Medium")], weight=0.4),  # same condition → total 0.7
        make_rule(1, [(0, 0.0, "Medium")], weight=0.6),   # competing action, w=0.6 < 0.7
    ]
    print("\n  3f. Weight summation (two rules for a=0 sum to 0.7 > a=1's 0.6)")
    pred = predict_action_from_canonical_rules(rules_weighted, {0: 0.0})
    assert pred == 0, f"Expected action 0 (total weight 0.7 > 0.6), got {pred}"
    print(f"    ✓ Predicted action = {pred} (correct: a=0 wins with w=0.7 vs w=0.6)")
    global PASS_COUNT
    PASS_COUNT += 1

    # ── 3g: Mean pairwise BRA ──
    print("\n  3g. Mean pairwise BRA across 3 rule sets")
    mpbra = mean_pairwise_bra([rules_A, rules_A, rules_A], eval_states)
    check("Mean pairwise BRA (3 identical)", mpbra, 1.0)


# ═══════════════════════════════════════════════════════════════
# SECTION 4: Cross-metric consistency checks
# ═══════════════════════════════════════════════════════════════

def test_cross_metric():
    section("Cross-Metric Consistency Checks")

    # Build a scenario where we know all metrics simultaneously
    # "Stable" pair: identical rules and thresholds
    rules_stable_1 = [
        make_rule(0, [(0, -0.5, "Low"), (1, 0.0, "Medium")], weight=0.9),
        make_rule(2, [(0, 0.5, "High"), (1, 0.5, "High")], weight=0.8),
    ]
    rules_stable_2 = list(rules_stable_1)  # same
    thresholds_stable_1 = {0: [-0.94, -0.41, -0.03], 1: [-0.04, -0.01, 0.00, 0.02]}
    thresholds_stable_2 = dict(thresholds_stable_1)

    eval_states = [
        {0: -0.5, 1: 0.0},
        {0: 0.5, 1: 0.5},
        {0: 0.0, 1: 1.0},
    ]

    print("\n  4a. Fully stable pair → all metrics optimal")
    grs = ruleset_weighted_jaccard(rules_stable_1, rules_stable_2)
    td = threshold_drift(thresholds_stable_1, thresholds_stable_2)
    bra = behavior_rule_agreement(rules_stable_1, rules_stable_2, eval_states)
    check("GRS (WJ)", grs, 1.0)
    check("TD", td, 0.0)
    check("BRA", bra, 1.0)

    # "Unstable" pair: completely different
    rules_unstable = [
        make_rule(1, [(0, 0.0, "Medium"), (1, -1.0, "Very Low")], weight=0.4),
        make_rule(0, [(0, -1.0, "Very Low")], weight=0.3),
    ]
    thresholds_unstable = {0: [0.1, 0.5, 0.9], 1: [0.1, 0.3, 0.6, 0.8]}

    print("\n  4b. Fully unstable pair → all metrics worst-case")
    grs_bad = ruleset_weighted_jaccard(rules_stable_1, rules_unstable)
    td_bad = threshold_drift(thresholds_stable_1, thresholds_unstable, feature_ranges={0: 2.0, 1: 2.0})
    bra_bad = behavior_rule_agreement(rules_stable_1, rules_unstable, eval_states)

    check("GRS (WJ) — disjoint", grs_bad, 0.0)
    assert td_bad > 0.0, f"TD should be > 0 for different thresholds, got {td_bad}"
    print(f"    ✓ TD = {td_bad:.4f} > 0 (thresholds differ)")
    global PASS_COUNT
    PASS_COUNT += 1
    # BRA: rules_stable matches state[0]→0, state[1]→2, state[2]→-1
    #       rules_unstable matches state[0]→-1, state[1]→-1, state[2]→-1
    # Agreements: state[2] both -1 → agree. state[0]: 0 vs -1 → disagree. state[1]: 2 vs -1 → disagree.
    check("BRA — mostly disagreeing", bra_bad, 1.0 / 3.0)

    # ── 4c: Verify monotonicity: more perturbation → worse metrics ──
    print("\n  4c. Monotonicity: adding perturbation worsens metrics")
    # Slight perturbation: same rules but one weight changes
    rules_slight = [
        make_rule(0, [(0, -0.5, "Low"), (1, 0.0, "Medium")], weight=0.85),  # slightly different
        make_rule(2, [(0, 0.5, "High"), (1, 0.5, "High")], weight=0.8),
    ]
    grs_slight = ruleset_weighted_jaccard(rules_stable_1, rules_slight)
    assert 0.0 < grs_slight < 1.0, f"GRS should be in (0,1) for slight perturbation, got {grs_slight}"
    assert grs_slight > grs_bad, f"GRS should be higher for slight than severe perturbation"
    print(f"    ✓ GRS: identical={grs:.2f} > slight={grs_slight:.4f} > severe={grs_bad:.2f}")
    PASS_COUNT += 1


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Stability Metric Sanity Checks")
    print("=" * 60)

    test_grs()
    test_td()
    test_bra()
    test_cross_metric()

    print(f"\n{'='*60}")
    if FAIL_COUNT == 0:
        print(f"  ALL {PASS_COUNT} CHECKS PASSED ✓")
    else:
        print(f"  {PASS_COUNT} passed, {FAIL_COUNT} FAILED ✗")
    print(f"{'='*60}\n")

    return 0 if FAIL_COUNT == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
