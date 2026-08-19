#!/usr/bin/env python
"""
Local Explanation Consistency (LEC)

Measures how stable the active explanation is under small local
perturbations of a single state:

    LEC(epsilon) = E_{s ~ S}[ Pr( E(s) = E(s+delta) | ||delta||_inf <= epsilon ) ]

Where E(s) is the active explanation (highest-weight matching rule) for state s.
"""
import numpy as np
from reproduction.cbs import CBSPipeline


def normalize_states(states, feature_mins, feature_maxs):
    """Normalize states to [0, 1] per feature."""
    ranges = feature_maxs - feature_mins
    ranges[ranges == 0] = 1.0  # avoid division by zero for constant features
    return (states - feature_mins) / ranges


def denormalize_states(normalized_states, feature_mins, feature_maxs):
    """Convert normalized [0, 1] states back to original feature space."""
    ranges = feature_maxs - feature_mins
    ranges[ranges == 0] = 1.0
    return normalized_states * ranges + feature_mins


def generate_local_perturbations(normalized_state, epsilon, n_samples, rng):
    """Generate n_samples perturbed states within L-inf ball of radius epsilon.

    Parameters
    ----------
    normalized_state : np.ndarray, shape (n_features,)
        State in [0, 1] normalized space.
    epsilon : float
        L-inf ball radius.
    n_samples : int
        Number of perturbations to generate.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray, shape (n_samples, n_features)
        Perturbed states, clamped to [0, 1].
    """
    n_features = len(normalized_state)
    # Uniform noise in [-epsilon, +epsilon] per dimension
    delta = rng.uniform(-epsilon, epsilon, size=(n_samples, n_features))
    perturbed = normalized_state + delta
    # Clamp to [0, 1]
    np.clip(perturbed, 0.0, 1.0, out=perturbed)
    return perturbed


def get_active_explanation(cbs, raw_state):
    """Return the signature of the highest-weight matching rule for a state.

    Parameters
    ----------
    cbs : CBSPipeline
        Fitted CBS pipeline.
    raw_state : np.ndarray, shape (n_features,)
        State in original feature space.

    Returns
    -------
    tuple or None
        (action, signature_tuple) of the best-matching rule, or None if no match.
        signature_tuple is a tuple of (feature_idx, level) pairs.
    """
    # Encode state using the CBS pipeline's thresholds
    encoded = cbs._encode_single(raw_state)

    best_rule = None
    best_weight = -1.0

    for rule in cbs.rules_:
        # Check if all predicates match
        matches = True
        for pred in rule.condition.predicates:
            if encoded[pred.feature_idx] != pred.level:
                matches = False
                break
        if matches and rule.weight > best_weight:
            best_weight = rule.weight
            best_rule = rule

    if best_rule is None:
        return None

    # Return (action, tuple of (feature_idx, level) pairs) as signature
    sig = tuple(
        (p.feature_idx, p.level)
        for p in sorted(best_rule.condition.predicates,
                        key=lambda p: p.feature_idx)
    )
    return (best_rule.action, sig)


def compute_lec(cbs, held_out_states, feature_mins, feature_maxs,
                epsilons=(0.01, 0.03, 0.05), n_perturbations=50, seed=42):
    """Compute LEC(epsilon) for each epsilon level.

    Parameters
    ----------
    cbs : CBSPipeline
        Fitted CBS pipeline.
    held_out_states : np.ndarray, shape (N, n_features)
        Held-out states in original feature space.
    feature_mins : np.ndarray, shape (n_features,)
    feature_maxs : np.ndarray, shape (n_features,)
    epsilons : tuple of float
        L-inf ball radii in normalized [0,1] space.
    n_perturbations : int
        Number of perturbations per state per epsilon.
    seed : int
        Random seed.

    Returns
    -------
    dict
        {epsilon: {"lec": float, "n_states": int, "n_null_original": int}}
    """
    rng = np.random.default_rng(seed)
    results = {}

    # Normalize held-out states
    norm_states = normalize_states(held_out_states, feature_mins, feature_maxs)

    for eps in epsilons:
        consistencies = []
        n_null_original = 0

        for i in range(len(held_out_states)):
            # Get active explanation for original state
            explanation_s = get_active_explanation(cbs, held_out_states[i])
            if explanation_s is None:
                n_null_original += 1

            # Generate perturbations in normalized space
            perturbed_norm = generate_local_perturbations(
                norm_states[i], eps, n_perturbations, rng)
            # Denormalize back to original space
            perturbed_raw = denormalize_states(
                perturbed_norm, feature_mins, feature_maxs)

            # Check consistency
            n_consistent = 0
            for j in range(n_perturbations):
                explanation_delta = get_active_explanation(cbs, perturbed_raw[j])
                if explanation_s == explanation_delta:
                    n_consistent += 1

            consistencies.append(n_consistent / n_perturbations)

        lec_score = float(np.mean(consistencies))
        results[eps] = {
            "lec": lec_score,
            "lec_std": float(np.std(consistencies)),
            "n_states": len(held_out_states),
            "n_null_original": n_null_original,
        }

    return results


def compute_lec_prediction_based(model, held_out_states, feature_mins, feature_maxs,
                                  epsilons=(0.01, 0.03, 0.05), n_perturbations=50, seed=42):
    """Compute prediction-based LEC — works with any model that has predict().

    Instead of comparing rule signatures, compares predicted actions.
    This is compatible with DecisionTreeSurrogate, CBSPipeline, and any
    model with a predict(states) method returning action arrays.

    LEC_pred(epsilon) = E_s[ Pr( predict(s) == predict(s+delta) | ||delta||_inf <= epsilon ) ]
    """
    rng = np.random.default_rng(seed)
    results = {}

    norm_states = normalize_states(held_out_states, feature_mins, feature_maxs)

    # Get predictions for all original states
    original_preds = model.predict(held_out_states)

    for eps in epsilons:
        consistencies = []

        for i in range(len(held_out_states)):
            # Generate perturbations in normalized space
            perturbed_norm = generate_local_perturbations(
                norm_states[i], eps, n_perturbations, rng)
            perturbed_raw = denormalize_states(
                perturbed_norm, feature_mins, feature_maxs)

            # Predict on all perturbations at once (batch)
            perturbed_preds = model.predict(perturbed_raw)
            n_consistent = int(np.sum(perturbed_preds == original_preds[i]))
            consistencies.append(n_consistent / n_perturbations)

        lec_score = float(np.mean(consistencies))
        results[eps] = {
            "lec": lec_score,
            "lec_std": float(np.std(consistencies)),
            "n_states": len(held_out_states),
        }

    return results
