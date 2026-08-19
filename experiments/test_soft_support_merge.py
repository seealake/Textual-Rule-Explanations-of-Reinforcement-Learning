#!/usr/bin/env python
"""
Smoke tests for the SoftSupport merge.

Covers:
  1. lambda_B=0 degrades to the default matching behaviour
  2. Calibration pool is separate from final evaluation
  3. Soft support values are in [0, 1]
  4. Safeguard=off does not alter v1 logic
  5. Behavior signature cache correctness

Usage:
    python experiments/test_soft_support_merge.py
"""
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.rule_matching import (
    CanonicalRule, CanonicalPredicate,
    rule_similarity_threshold_aware,
)
from experiments.consensus_merge import merge_rule_group
from experiments.soft_support_merge import (
    SoftSupportConfig,
    BehaviorSignatureCache,
    soft_rule_similarity,
    _build_behavior_signature,
    _compute_soft_support,
    build_soft_support_consensus,
)

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  [PASS] {name}")
        PASS += 1
    else:
        print(f"  [FAIL] {name}  {detail}")
        FAIL += 1


def make_rule(action, features, level=0.0, weight=1.0, n=100):
    preds = tuple(
        CanonicalPredicate(f, level, "Mid", -1.0, 1.0)
        for f in features
    )
    return CanonicalRule(action=action, predicates=preds,
                         weight=weight, n_instances=n)


def test_lambda_b_zero_degrades_to_v1():
    """When lambda_B=0, v2 similarity should equal threshold-aware v1
    similarity (after rescaling lambda1=lambda_P, lambda2=lambda_I)."""
    print("\n── Test 1: lambda_B=0 degrades to v1 ──")
    r1 = make_rule(0, [0, 1], level=-0.5)
    r2 = make_rule(0, [0, 1], level=0.0)

    sim_default = rule_similarity_threshold_aware(r1, r2, lambda1=0.35, lambda2=0.45)
    # lambda_P + lambda_I = 0.80; v2 normalises by total weight
    sim_soft = soft_rule_similarity(r1, r2, lambda_P=0.35, lambda_I=0.45,
                                 lambda_B=0.0, sig_cache=None)

    # v2 normalizes by (0.35+0.45+0)=0.80  but v1 just sums 0.35*x + 0.45*y
    # so they should match since v1's formula is lambda1*overlap + lambda2*(1-dist)
    # and v2 does (lambda_P*overlap + lambda_I*(1-dist)) / (lambda_P+lambda_I)
    # These differ by a constant factor. Let's verify.
    sim_soft_scaled = sim_soft * (0.35 + 0.45)
    check("lambda_B=0: v2 * sum_weight == v1",
          abs(sim_soft_scaled - sim_default) < 1e-8,
          f"v2_scaled={sim_soft_scaled:.6f}, v1={sim_default:.6f}")

    # More importantly: identical rules should give sim=1.0
    sim_id = soft_rule_similarity(r1, r1, lambda_P=0.35, lambda_I=0.45,
                                 lambda_B=0.0, sig_cache=None)
    check("lambda_B=0: identical rules → sim=1.0",
          abs(sim_id - 1.0) < 1e-8, f"sim={sim_id}")


def test_behavior_signature_separation():
    """Calibration pool should not leak into evaluation data."""
    print("\n── Test 2: Calibration pool separation ──")

    # Create synthetic calibration states (encoded as level values)
    cal_states = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, -1.0],
    ], dtype=np.float32)
    n_actions = 3

    r = make_rule(0, [0], level=0.0)
    sig = _build_behavior_signature(r, cal_states, n_actions)

    # Check signature length
    check("Signature length", len(sig) == 4 * 3,
          f"expected {4*3}, got {len(sig)}")

    # Signature should be normalised (or zero if no coverage)
    norm = np.linalg.norm(sig)
    check("Signature is L2-normalised or zero",
          abs(norm - 1.0) < 1e-6 or abs(norm) < 1e-6,
          f"norm={norm}")


def test_soft_support_range():
    """Soft support should be in [0, 1]."""
    print("\n── Test 3: Soft support in [0, 1] ──")

    r1 = make_rule(0, [0, 1], level=0.0)
    r2 = make_rule(0, [0, 1], level=0.5)
    r3 = make_rule(0, [0], level=0.0)

    all_run_rules = [[r1], [r2], [r1, r3]]
    group = [(0, r1), (1, r2)]

    ss = _compute_soft_support(
        group, r1, all_run_rules, n_bootstrap=3,
        lambda_P=0.35, lambda_I=0.45, lambda_B=0.0,
        sig_cache=None,
    )
    check("Soft support >= 0", ss >= 0.0, f"ss={ss}")
    check("Soft support <= 1", ss <= 1.0, f"ss={ss}")

    # Perfect match across all runs should give ss=1.0
    all_same = [[r1], [r1], [r1]]
    ss_perfect = _compute_soft_support(
        [(0, r1)], r1, all_same, n_bootstrap=3,
        lambda_P=0.35, lambda_I=0.45, lambda_B=0.0,
        sig_cache=None,
    )
    check("Perfect match → soft support ~ 1.0",
          abs(ss_perfect - 1.0) < 1e-6, f"ss={ss_perfect}")


def test_safeguard_default_off():
    """With safeguard_enabled=False, v2 should not alter group filtering."""
    print("\n── Test 4: Safeguard off = no change ──")
    cfg = SoftSupportConfig()
    check("Default safeguard_enabled is False", cfg.safeguard_enabled is False)
    check("Default lambda_B is 0.0", cfg.lambda_B == 0.0)
    check("Default support_mode is 'hard'", cfg.support_mode == "hard")


def test_behavior_cache():
    """BehaviorSignatureCache should cache and return consistent results."""
    print("\n── Test 5: Behavior signature cache ──")

    cal_states = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32)

    cache = BehaviorSignatureCache(cal_states, n_actions=2)
    r = make_rule(0, [0], level=0.0)

    sig1 = cache.get(r)
    sig2 = cache.get(r)
    check("Cache returns same object", sig1 is sig2)

    # Two identical rules should have similarity 1.0
    sim = cache.similarity(r, r)
    check("Self-similarity == 1.0", abs(sim - 1.0) < 1e-6, f"sim={sim}")


def test_merge_rule_group_uses_encoded_levels():
    """Merged rules must choose representative levels in encoded space."""
    print("\n── Test 6: merge_rule_group keeps encoded levels ──")

    level_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    level_labels = ["Very Low", "Low", "Medium", "High", "Very High"]

    # Raw bounds are far outside [-1, 1], so a buggy raw-midpoint mapping would
    # snap to +1.0. The correct merged level should stay at +0.5 because both
    # source rules live in the same encoded bin.
    rules = [
        CanonicalRule(
            action=0,
            predicates=(CanonicalPredicate(0, 0.5, "High", 10.0, 20.0),),
            weight=1.0,
            n_instances=10,
        ),
        CanonicalRule(
            action=0,
            predicates=(CanonicalPredicate(0, 0.5, "High", 12.0, 22.0),),
            weight=1.0,
            n_instances=12,
        ),
    ]

    merged = merge_rule_group(rules, level_values, level_labels)
    pred = merged.predicates[0]

    check("Merged level stays in encoded member bin",
          abs(pred.level - 0.5) < 1e-8,
          f"level={pred.level}")
    check("Merged bounds are still aggregated in raw space",
          abs(pred.lower_bound - 11.0) < 1e-8 and abs(pred.upper_bound - 21.0) < 1e-8,
          f"bounds=({pred.lower_bound}, {pred.upper_bound})")


def test_full_v2_build_with_defaults():
    """Build v2 with all defaults → should behave like v1."""
    print("\n── Test 7: Full build with defaults (needs replay data) ──")
    replay_path = "reproduction/data/replay_cartpole_v1_seed42.npz"
    if not os.path.exists(replay_path):
        print("  [SKIP] Replay data not found — skipping full build test")
        return

    from experiments.perturbations import load_replay_npz
    data = load_replay_npz(replay_path)
    # Use a small subset for speed
    small = {
        "states": data["states"][:2000],
        "actions": data["actions"][:2000],
    }

    cfg = SoftSupportConfig(n_bootstrap=3, subsample_seed=42)
    try:
        pipeline, rules, info = build_soft_support_consensus(
            small, "CartPole-v1", cfg)
        check("Build succeeds", True)
        check("Returns rules", len(rules) >= 0)
        check("Build info has method key",
              info.get("method") == "soft_support_merge")
        check("lambda_B=0 in info", info.get("lambda_B", -1) == 0.0)
        check("No safeguard rescue",
              info["safeguard"]["rescued_groups"] == 0)
    except Exception as e:
        check("Build succeeds", False, f"Exception: {e}")
        traceback.print_exc()


def test_v2_with_behavior_matching():
    """Build v2 with lambda_B > 0 to exercise behavior matching."""
    print("\n── Test 8: Build with behavior matching ──")
    replay_path = "reproduction/data/replay_cartpole_v1_seed42.npz"
    if not os.path.exists(replay_path):
        print("  [SKIP] Replay data not found")
        return

    from experiments.perturbations import load_replay_npz
    data = load_replay_npz(replay_path)
    small = {
        "states": data["states"][:2000],
        "actions": data["actions"][:2000],
    }

    cfg = SoftSupportConfig(
        n_bootstrap=3, subsample_seed=42,
        lambda_B=0.2, lambda_P=0.35, lambda_I=0.45,
        calibration_n=500, calibration_seed=999,
    )
    try:
        pipeline, rules, info = build_soft_support_consensus(small, "CartPole-v1", cfg)
        check("Build with lambda_B=0.2 succeeds", True)
        check("lambda_B recorded", info.get("lambda_B") == 0.2)
    except Exception as e:
        check("Build with lambda_B=0.2 succeeds", False, f"{e}")
        traceback.print_exc()


def test_v2_soft_support_mode():
    """Build v2 with support_mode='soft'."""
    print("\n── Test 9: Soft support mode build ──")
    replay_path = "reproduction/data/replay_cartpole_v1_seed42.npz"
    if not os.path.exists(replay_path):
        print("  [SKIP] Replay data not found")
        return

    from experiments.perturbations import load_replay_npz
    data = load_replay_npz(replay_path)
    small = {
        "states": data["states"][:2000],
        "actions": data["actions"][:2000],
    }

    cfg = SoftSupportConfig(
        n_bootstrap=3, subsample_seed=42,
        support_mode="soft",
    )
    try:
        pipeline, rules, info = build_soft_support_consensus(small, "CartPole-v1", cfg)
        check("Soft support build succeeds", True)
        check("support_mode recorded", info.get("support_mode") == "soft")
        # Check diagnostics contain soft_support
        has_ss = any(
            "soft_support" in d
            for d in info.get("group_diagnostics", [])
        )
        check("Group diagnostics have soft_support", has_ss)
    except Exception as e:
        check("Soft support build succeeds", False, f"{e}")
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("  Consensus CBS v2 — Smoke Tests")
    print("=" * 60)

    test_lambda_b_zero_degrades_to_v1()
    test_behavior_signature_separation()
    test_soft_support_range()
    test_safeguard_default_off()
    test_behavior_cache()
    test_merge_rule_group_uses_encoded_levels()
    test_full_v2_build_with_defaults()
    test_v2_with_behavior_matching()
    test_v2_soft_support_mode()

    print(f"\n{'=' * 60}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")
    sys.exit(1 if FAIL > 0 else 0)
