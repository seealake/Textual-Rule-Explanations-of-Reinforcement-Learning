#!/usr/bin/env python
"""
Weighted rule-set voting: per-member scalar weights for voting ensembles.

Weight types:
  - f1:      w_b = macro-F1 of member b on calibration data
  - worst_r: w_b = worst per-action recall on calibration data
  - hybrid:  w_b = alpha * F1_b + (1 - alpha) * worst_R_b

Normalisation: softmax only
  w_b = exp(beta * s_b) / sum_j exp(beta * s_j)

Cost metrics:
  - active_voter_cost: avg # voters queried per state (always B for dense)
  - rules_triggered_per_state: avg rules triggered across all voters
"""
import numpy as np
from reproduction.cbs import CBSPipeline


# ── Weight computation ────────────────────────────────────────────────


def _member_f1(pipeline: CBSPipeline, cal_states: np.ndarray,
               cal_actions: np.ndarray) -> float:
    """Compute macro-F1 of a single CBS member on calibration data."""
    preds = pipeline.predict(cal_states)
    actions_set = sorted(set(cal_actions.tolist()) | set(preds.tolist()))
    f1s = []
    for a in actions_set:
        tp = int(((preds == a) & (cal_actions == a)).sum())
        fp = int(((preds == a) & (cal_actions != a)).sum())
        fn = int(((preds != a) & (cal_actions == a)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def _member_worst_recall(pipeline: CBSPipeline, cal_states: np.ndarray,
                         cal_actions: np.ndarray) -> float:
    """Compute worst per-action recall of a single CBS member."""
    preds = pipeline.predict(cal_states)
    actions_set = sorted(set(cal_actions.tolist()))
    recalls = []
    for a in actions_set:
        mask = cal_actions == a
        n_true = int(mask.sum())
        if n_true == 0:
            continue
        tp = int(((preds == a) & mask).sum())
        recalls.append(tp / n_true)
    return min(recalls) if recalls else 0.0


def compute_voter_weights(
    pipelines: list[CBSPipeline],
    cal_states: np.ndarray,
    cal_actions: np.ndarray,
    weight_type: str = "f1",
    alpha: float = 0.5,
    beta: float = 1.0,
) -> np.ndarray:
    """Compute softmax-normalised per-member weights.

    Parameters
    ----------
    pipelines : list of CBSPipeline
    cal_states, cal_actions : calibration split
    weight_type : one of {"f1", "worst_r", "hybrid"}
    alpha : blend coefficient for hybrid (alpha*F1 + (1-alpha)*worst_R)
    beta : softmax temperature

    Returns
    -------
    weights : np.ndarray of shape (B,), softmax-normalised
    """
    B = len(pipelines)
    raw = np.zeros(B)

    for b, p in enumerate(pipelines):
        f1_b = _member_f1(p, cal_states, cal_actions)
        wr_b = _member_worst_recall(p, cal_states, cal_actions)

        if weight_type == "f1":
            raw[b] = f1_b
        elif weight_type == "worst_r":
            raw[b] = wr_b
        elif weight_type == "hybrid":
            raw[b] = alpha * f1_b + (1.0 - alpha) * wr_b
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")

    # Softmax normalisation
    logits = beta * raw
    logits -= logits.max()  # numerical stability
    exp_logits = np.exp(logits)
    weights = exp_logits / exp_logits.sum()

    return weights


# ── Weighted prediction ──────────────────────────────────────────────


def weighted_voting_predict(
    pipelines: list[CBSPipeline],
    states: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted majority vote: â(s) = argmax_a sum_b w_b * 1[â_b(s) = a].

    Parameters
    ----------
    pipelines : list of B CBSPipeline
    states : (N, D)
    weights : (B,)

    Returns
    -------
    predictions : (N,) int
    """
    B = len(pipelines)
    N = len(states)
    all_preds = np.array([p.predict(states) for p in pipelines])  # (B, N)

    n_actions = int(all_preds.max()) + 1
    result = np.zeros(N, dtype=int)

    for i in range(N):
        weighted_counts = np.zeros(n_actions)
        for b in range(B):
            weighted_counts[all_preds[b, i]] += weights[b]
        result[i] = int(np.argmax(weighted_counts))

    return result


def topk_weighted_voting_predict(
    pipelines: list[CBSPipeline],
    states: np.ndarray,
    weights: np.ndarray,
    k: int = 3,
) -> tuple[np.ndarray, float]:
    """Top-k sparse weighted vote — only the k highest-weight voters participate.

    Returns
    -------
    predictions : (N,) int
    active_voter_cost : float — avg voters queried per state (= k)
    """
    topk_idx = np.argsort(weights)[-k:]
    sub_pipelines = [pipelines[i] for i in topk_idx]
    sub_weights = weights[topk_idx]
    sub_weights = sub_weights / sub_weights.sum()  # re-normalise

    preds = weighted_voting_predict(sub_pipelines, states, sub_weights)
    return preds, float(k)


# ── Cost metrics ─────────────────────────────────────────────────────


def compute_voter_cost_metrics(
    pipelines: list[CBSPipeline],
    states: np.ndarray,
) -> dict:
    """Compute per-state cost metrics for the voting ensemble.

    Returns
    -------
    dict with:
      active_voters : float — avg voters queried (= B for dense)
      rules_per_state : float — avg total rules triggered across all voters
    """
    B = len(pipelines)
    N = len(states)

    total_rules_triggered = 0
    for p in pipelines:
        for i in range(N):
            s = states[i:i+1]
            # Count rules that match this state
            encoded = p._encode_single(states[i])
            n_match = 0
            for rule in p.rules_:
                matches = True
                for pred in rule.condition.predicates:
                    if encoded[pred.feature_idx] != pred.level:
                        matches = False
                        break
                if matches:
                    n_match += 1
            total_rules_triggered += n_match

    return {
        "active_voters": float(B),
        "rules_per_state": total_rules_triggered / (N * B),
        "total_rules_per_state": total_rules_triggered / N,
    }
