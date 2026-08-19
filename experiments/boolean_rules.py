#!/usr/bin/env python
"""
Boolean Decision Rules (BDR) baseline

Implements a Boolean Rule Summarizer (BDR) that:
  1. Discretizes continuous state features into boolean predicates via quantile thresholds
  2. Learns one-vs-rest sparse boolean rule sets (DNF) for each action
  3. Predicts actions via rule matching with precision/support tie-breaking
  4. Falls back to black-box policy when no rule matches

Follows the same external API as DecisionTreeSurrogate / CBSPipeline for
drop-in usage with the existing evaluation infrastructure.

Reference: McCarthy et al. (2022), "Boolean Decision Rules for RL Policy Summarisation"
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from itertools import combinations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reproduction.cbs import Predicate, Condition, Rule
from experiments.rule_matching import CanonicalPredicate, CanonicalRule


# ── Boolean Predicate Representation ─────────────────────────────────

@dataclass
class BooleanPredicate:
    """A single boolean predicate: feature_idx <= threshold."""
    feature_idx: int
    threshold: float
    feature_name: str = ""

    def evaluate(self, states: np.ndarray) -> np.ndarray:
        """Evaluate predicate on states, returns bool array."""
        return states[:, self.feature_idx] <= self.threshold

    def __repr__(self):
        name = self.feature_name or f"f{self.feature_idx}"
        return f"{name} <= {self.threshold:.4f}"


@dataclass
class BooleanRule:
    """A conjunction of boolean predicates predicting a specific action."""
    action: int
    predicate_indices: tuple  # indices into the predicate pool
    precision: float = 0.0
    support: float = 0.0    # fraction of data covered
    n_covered: int = 0      # number of training samples covered


class BDRSurrogate:
    """Boolean Decision Rules policy surrogate.

    Wraps a greedy boolean rule learner with the same external API as
    CBSPipeline / DecisionTreeSurrogate so that existing evaluation
    helpers work unchanged.

    Parameters
    ----------
    n_quantile_thresholds : int
        Number of quantile thresholds per continuous feature (default 4: 20/40/60/80%).
    max_literals : int
        Maximum number of literals (predicates) per rule conjunction.
    max_rules_per_action : int
        Maximum number of rules per action class.
    min_support_frac : float
        Minimum fraction of training data a rule must cover.
    random_state : int
        Random seed for reproducibility.
    feature_names : list[str] or None
        Human-readable feature names.
    fallback_policy_path : str or None
        Path to the black-box policy model for fallback predictions.
    env_name : str or None
        Environment name (used for loading fallback policy).
    """

    def __init__(
        self,
        n_quantile_thresholds: int = 4,
        max_literals: int = 3,
        max_rules_per_action: int = 8,
        min_support_frac: float = 0.01,
        random_state: int = 42,
        feature_names: Optional[list] = None,
        fallback_policy_path: Optional[str] = None,
        env_name: Optional[str] = None,
    ):
        self.n_quantile_thresholds = n_quantile_thresholds
        self.max_literals = max_literals
        self.max_rules_per_action = max_rules_per_action
        self.min_support_frac = min_support_frac
        self.random_state = random_state
        self.feature_names = feature_names
        self.fallback_policy_path = fallback_policy_path
        self.env_name = env_name

        self.predicates_: Optional[list] = None   # list[BooleanPredicate]
        self.rules_: Optional[list] = None         # list[Rule] (CBS-compatible)
        self.boolean_rules_: Optional[list] = None # list[BooleanRule] (internal)
        self.actions_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.thresholds_: Optional[dict] = None
        self.feature_mins_: Optional[dict] = None
        self.feature_maxs_: Optional[dict] = None
        self.majority_action_: Optional[int] = None
        self._fallback_model = None
        self._predicate_matrix: Optional[np.ndarray] = None  # cached

        # Binary/contact features that should not be further discretized
        self._binary_features: set = set()

    # ── Fitting ───────────────────────────────────────────────────

    def fit(self, states: np.ndarray, actions: np.ndarray,
            sample_weight: np.ndarray = None):
        """Train BDR: build predicates, learn rules per action."""
        self.n_features_ = states.shape[1]
        self.actions_ = np.sort(np.unique(actions))
        self.majority_action_ = int(np.bincount(actions.astype(int)).argmax())
        self.feature_mins_ = {f: float(states[:, f].min())
                              for f in range(self.n_features_)}
        self.feature_maxs_ = {f: float(states[:, f].max())
                              for f in range(self.n_features_)}

        # Detect binary features
        self._binary_features = set()
        for f in range(self.n_features_):
            unique_vals = np.unique(states[:, f])
            if len(unique_vals) <= 2:
                self._binary_features.add(f)

        # Step 1: Build boolean predicates
        self.predicates_ = self._build_predicates(states)

        # Step 2: Compute predicate truth matrix
        self._predicate_matrix = self._compute_predicate_matrix(states)

        # Step 3: Learn one-vs-rest rules for each action
        self.boolean_rules_ = []
        for action in self.actions_:
            action_rules = self._learn_rules_for_action(
                action, actions, self._predicate_matrix)
            self.boolean_rules_.extend(action_rules)

        # Step 4: Convert to CBS-compatible rule objects
        self.rules_ = self._convert_to_cbs_rules(states)
        self.thresholds_ = self._extract_thresholds()

        # Load fallback policy if provided
        if self.fallback_policy_path and os.path.exists(self.fallback_policy_path):
            self._load_fallback_policy()

    def _build_predicates(self, states: np.ndarray) -> list:
        """Build boolean predicates from state features using quantile thresholds."""
        predicates = []
        quantiles = np.linspace(0, 1, self.n_quantile_thresholds + 2)[1:-1]
        # quantiles for n=4: [0.2, 0.4, 0.6, 0.8]

        for f in range(self.n_features_):
            fname = (self.feature_names[f]
                     if self.feature_names and f < len(self.feature_names)
                     else f"f{f}")

            if f in self._binary_features:
                # Binary feature: use median as single threshold
                median_val = float(np.median(states[:, f]))
                predicates.append(BooleanPredicate(
                    feature_idx=f, threshold=median_val, feature_name=fname))
            else:
                # Continuous feature: quantile thresholds
                for q in quantiles:
                    threshold = float(np.quantile(states[:, f], q))
                    predicates.append(BooleanPredicate(
                        feature_idx=f, threshold=threshold, feature_name=fname))

        return predicates

    def _compute_predicate_matrix(self, states: np.ndarray) -> np.ndarray:
        """Evaluate all predicates on all states, return (N, P) bool matrix."""
        N = len(states)
        P = len(self.predicates_)
        matrix = np.zeros((N, P), dtype=bool)
        for j, pred in enumerate(self.predicates_):
            matrix[:, j] = pred.evaluate(states)
        return matrix

    def _learn_rules_for_action(self, action: int, actions: np.ndarray,
                                pred_matrix: np.ndarray) -> list:
        """Learn boolean rules for one action using greedy forward selection.

        For action `a`, the target is y = (actions == a).
        We greedily search for conjunctions of up to max_literals predicates
        that maximize precision on the positive class, subject to a minimum
        support constraint. After finding a rule, we "cover" those positives
        and search for more rules on the remaining uncovered positives.
        """
        N = len(actions)
        y = (actions == action).astype(bool)
        n_pos_total = y.sum()
        if n_pos_total == 0:
            return []

        min_support = max(1, int(np.ceil(self.min_support_frac * N)))
        rng = np.random.RandomState(self.random_state)

        rules = []
        covered = np.zeros(N, dtype=bool)  # which positives are covered

        for _ in range(self.max_rules_per_action):
            uncovered_pos = y & ~covered
            if uncovered_pos.sum() == 0:
                break

            best_rule = self._find_best_conjunction(
                pred_matrix, y, uncovered_pos, min_support)

            if best_rule is None:
                break

            best_rule.action = action

            # Mark covered positives
            conj_mask = self._evaluate_conjunction(
                pred_matrix, best_rule.predicate_indices)
            covered |= (conj_mask & y)

            rules.append(best_rule)

        return rules

    def _find_best_conjunction(self, pred_matrix: np.ndarray, y: np.ndarray,
                               target_mask: np.ndarray,
                               min_support: int) -> Optional[BooleanRule]:
        """Find the best conjunction of predicates via beam search.

        Uses a breadth-first beam search over conjunction sizes 1..max_literals.
        At each level, keeps the top-K candidates by F1 score (precision × recall
        on target_mask).
        """
        P = pred_matrix.shape[1]
        N = len(y)
        beam_width = 20

        # Initialize with single predicates
        candidates = []
        for j in range(P):
            mask = pred_matrix[:, j]
            n_covered = mask.sum()
            if n_covered < min_support:
                continue
            tp = (mask & y).sum()
            precision = tp / n_covered if n_covered > 0 else 0
            target_recall = (mask & target_mask).sum() / target_mask.sum() if target_mask.sum() > 0 else 0
            # Score: F1-like combination of precision and target recall
            score = (2 * precision * target_recall / (precision + target_recall)
                     if (precision + target_recall) > 0 else 0)
            if precision > 0:
                candidates.append({
                    "indices": (j,),
                    "precision": precision,
                    "target_recall": target_recall,
                    "score": score,
                    "n_covered": int(n_covered),
                    "tp": int(tp),
                })

        if not candidates:
            return None

        # Sort by score, keep top beam_width
        candidates.sort(key=lambda c: -c["score"])
        candidates = candidates[:beam_width]
        best_overall = candidates[0]

        # Expand conjunctions
        for literal_count in range(2, self.max_literals + 1):
            next_candidates = []
            for cand in candidates:
                last_pred_idx = cand["indices"][-1]
                for j in range(last_pred_idx + 1, P):
                    new_indices = cand["indices"] + (j,)
                    mask = self._evaluate_conjunction(pred_matrix, new_indices)
                    n_covered = mask.sum()
                    if n_covered < min_support:
                        continue
                    tp = (mask & y).sum()
                    precision = tp / n_covered if n_covered > 0 else 0
                    target_recall = ((mask & target_mask).sum() / target_mask.sum()
                                     if target_mask.sum() > 0 else 0)
                    score = (2 * precision * target_recall / (precision + target_recall)
                             if (precision + target_recall) > 0 else 0)
                    if precision > 0 and score > 0:
                        next_candidates.append({
                            "indices": new_indices,
                            "precision": precision,
                            "target_recall": target_recall,
                            "score": score,
                            "n_covered": int(n_covered),
                            "tp": int(tp),
                        })

            if next_candidates:
                next_candidates.sort(key=lambda c: -c["score"])
                next_candidates = next_candidates[:beam_width]
                # Track best overall
                if next_candidates[0]["score"] > best_overall["score"]:
                    best_overall = next_candidates[0]
                candidates = next_candidates
            else:
                break

        if best_overall["precision"] <= 0:
            return None

        return BooleanRule(
            action=0,  # will be set by caller
            predicate_indices=best_overall["indices"],
            precision=best_overall["precision"],
            support=best_overall["n_covered"] / N,
            n_covered=best_overall["n_covered"],
        )

    def _evaluate_conjunction(self, pred_matrix: np.ndarray,
                              indices: tuple) -> np.ndarray:
        """Evaluate a conjunction (AND) of predicates."""
        mask = np.ones(pred_matrix.shape[0], dtype=bool)
        for idx in indices:
            mask &= pred_matrix[:, idx]
        return mask

    def _convert_to_cbs_rules(self, states: np.ndarray) -> list:
        """Convert internal BooleanRules to CBS-compatible Rule objects."""
        rules = []
        N = len(states)

        for brule in self.boolean_rules_:
            predicates = []
            for pidx in brule.predicate_indices:
                bp = self.predicates_[pidx]
                f = bp.feature_idx
                feat_range = self.feature_maxs_[f] - self.feature_mins_[f]

                # The predicate is "feature <= threshold"
                # lower bound is feature min, upper bound is threshold
                lo = self.feature_mins_[f]
                hi = bp.threshold

                if feat_range > 0:
                    normalized = ((lo + hi) / 2 - self.feature_mins_[f]) / feat_range
                    level = 2.0 * normalized - 1.0
                else:
                    level = 0.0
                level = float(np.clip(level, -1.0, 1.0))
                label = _level_to_label(level)

                predicates.append(Predicate(
                    feature_idx=f,
                    level=level,
                    level_label=label,
                    lower_bound=lo,
                    upper_bound=hi,
                ))

            condition = Condition(
                predicates=predicates,
                n_instances=brule.n_covered,
            )
            weight = brule.support
            rules.append(Rule(
                action=brule.action,
                condition=condition,
                weight=weight,
            ))

        return rules

    def _extract_thresholds(self) -> dict:
        """Extract per-feature threshold sets for TD metric compatibility."""
        thresholds = {f: set() for f in range(self.n_features_)}
        for pred in (self.predicates_ or []):
            thresholds[pred.feature_idx].add(pred.threshold)
        return {f: sorted(list(ts)) for f, ts in thresholds.items()}

    def _load_fallback_policy(self):
        """Load the black-box policy for fallback predictions."""
        from stable_baselines3 import DQN, PPO
        path = self.fallback_policy_path
        if "ppo" in path.lower():
            self._fallback_model = PPO.load(path)
        else:
            self._fallback_model = DQN.load(path)

    # ── Prediction ────────────────────────────────────────────────

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Predict actions using learned boolean rules.

        For each state:
        1. Evaluate all predicates
        2. Find all matching rules
        3. Pick action by: highest precision, then highest support
        4. If no rule matches: fallback to black-box policy (or majority action)
        """
        N = states.shape[0]
        predictions = np.full(N, -1, dtype=int)

        # Evaluate predicates on input states
        pred_matrix = self._compute_predicate_matrix(states)

        for i in range(N):
            matching_rules = []
            for brule in (self.boolean_rules_ or []):
                # Check if conjunction is satisfied
                match = True
                for pidx in brule.predicate_indices:
                    if not pred_matrix[i, pidx]:
                        match = False
                        break
                if match:
                    matching_rules.append(brule)

            if matching_rules:
                # Sort by (precision desc, support desc, action asc for tie-break)
                matching_rules.sort(
                    key=lambda r: (-r.precision, -r.support, r.action))
                predictions[i] = matching_rules[0].action
            else:
                # Fallback to black-box policy
                if self._fallback_model is not None:
                    action, _ = self._fallback_model.predict(
                        states[i:i+1], deterministic=True)
                    predictions[i] = int(action[0])
                else:
                    predictions[i] = self.majority_action_

        return predictions

    # ── Metrics (same interface as CBSPipeline) ───────────────────

    def evaluate_fidelity(self, states: np.ndarray, actions: np.ndarray) -> dict:
        pred = self.predict(states)
        accuracy = float(np.mean(pred == actions))
        recalls = []
        for a in self.actions_:
            mask = actions == a
            if mask.sum() > 0:
                recalls.append(float(np.mean(pred[mask] == a)))
        recall = float(np.mean(recalls)) if recalls else 0.0
        f1 = 2 * accuracy * recall / (accuracy + recall) if (accuracy + recall) > 0 else 0.0
        return {"accuracy": accuracy, "recall": recall, "f1": f1}

    def evaluate_fidelity_per_action(self, states: np.ndarray,
                                     actions: np.ndarray) -> dict:
        pred = self.predict(states)
        result_per_action = {}
        precisions, recalls = [], []
        for a in self.actions_:
            true_mask = (actions == a)
            pred_mask = (pred == a)
            support = int(true_mask.sum())
            tp = int((true_mask & pred_mask).sum())
            prec = tp / pred_mask.sum() if pred_mask.sum() > 0 else 0.0
            rec = tp / true_mask.sum() if true_mask.sum() > 0 else 0.0
            pa_f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            rule_count = sum(1 for r in (self.boolean_rules_ or [])
                             if r.action == a)
            result_per_action[int(a)] = {
                "precision": float(prec), "recall": float(rec),
                "f1": float(pa_f1), "support": support,
                "rule_count": rule_count,
            }
            if support > 0:
                precisions.append(prec)
                recalls.append(rec)
        macro_prec = float(np.mean(precisions)) if precisions else 0.0
        macro_rec = float(np.mean(recalls)) if recalls else 0.0
        macro_f1 = (2 * macro_prec * macro_rec / (macro_prec + macro_rec)
                     if (macro_prec + macro_rec) > 0 else 0.0)
        return {
            "per_action": result_per_action,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1,
        }

    def evaluate_properties(self) -> dict:
        n_conditions = len(self.boolean_rules_) if self.boolean_rules_ else 0
        cond_sigs = {}
        for brule in (self.boolean_rules_ or []):
            sig = brule.predicate_indices
            cond_sigs.setdefault(sig, set()).add(brule.action)
        n_dup = sum(1 for acts in cond_sigs.values() if len(acts) > 1)
        return {
            "n_conditions": n_conditions,
            "n_duplicated": n_dup,
            "n_unique_signatures": len(cond_sigs),
        }

    def evaluate_coverage(self, states: np.ndarray) -> dict:
        """Evaluate how many states are covered by rules vs fallback."""
        pred_matrix = self._compute_predicate_matrix(states)
        n_covered = 0
        for i in range(len(states)):
            for brule in (self.boolean_rules_ or []):
                match = True
                for pidx in brule.predicate_indices:
                    if not pred_matrix[i, pidx]:
                        match = False
                        break
                if match:
                    n_covered += 1
                    break
        n_approx = len(states) - n_covered
        return {
            "n_total": len(states),
            "n_approximated": n_approx,
            "approx_rate": n_approx / len(states) if len(states) > 0 else 0.0,
        }

    def evaluate_in_env(self, env_name: str, n_episodes: int = 50,
                        seed: int = 0, eval_seeds: list = None,
                        success_threshold: float = None) -> dict:
        import gymnasium as gym
        if eval_seeds is None:
            eval_seeds = list(range(seed, seed + n_episodes))

        if "MiniGrid" in env_name:
            from reproduction.minigrid_feature_wrapper import make_minigrid_env
            env = make_minigrid_env(env_name)
        else:
            env = gym.make(env_name)

        episode_rewards, episode_lengths = [], []
        for ep_seed in eval_seeds:
            obs, info = env.reset(seed=ep_seed)
            total_reward, steps, done = 0.0, 0, False
            while not done:
                action = int(self.predict(obs.reshape(1, -1))[0])
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
        env.close()

        rewards_arr = np.array(episode_rewards)
        lengths_arr = np.array(episode_lengths)
        ecr = float(rewards_arr.mean())
        ets = float(lengths_arr.mean())
        n_success = (sum(1 for r in episode_rewards if r >= success_threshold)
                     if success_threshold is not None else None)
        return {
            "E_CR": ecr,
            "E_CR_std": float(rewards_arr.std()),
            "E_TS": ets,
            "E_TS_std": float(lengths_arr.std()),
            "E_AR": ecr / ets if ets > 0 else 0.0,
            "success_rate": n_success / len(episode_rewards) if n_success is not None else None,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "eval_seeds_used": eval_seeds,
        }

    def get_rules(self) -> list:
        return self.rules_ or []

    def get_thresholds(self) -> dict:
        return self.thresholds_ or {}

    def print_rules(self):
        if not self.boolean_rules_:
            print("No rules extracted.")
            return
        for action in self.actions_:
            action_rules = [r for r in self.boolean_rules_ if r.action == action]
            action_rules.sort(key=lambda r: -r.precision)
            print(f"\n  Action {action} ({len(action_rules)} rules):")
            for r in action_rules:
                parts = []
                for pidx in r.predicate_indices:
                    bp = self.predicates_[pidx]
                    parts.append(str(bp))
                cond_str = " AND ".join(parts)
                print(f"    IF {cond_str} → action={action} "
                      f"(prec={r.precision:.3f}, supp={r.support:.3f}, "
                      f"n={r.n_covered})")


def _level_to_label(level: float) -> str:
    """Map continuous level in [-1, 1] to nearest 5-category label."""
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


def canonicalize_bdr_rules(rules: list) -> list:
    """Convert BDR Rule objects to CanonicalRule for stability metrics."""
    canonical = []
    for rule in rules:
        preds = tuple(sorted([
            CanonicalPredicate(
                feature_idx=p.feature_idx,
                level=round(p.level, 4),
                level_label=p.level_label,
                lower_bound=p.lower_bound,
                upper_bound=p.upper_bound,
            )
            for p in rule.condition.predicates
        ], key=lambda p: p.feature_idx))
        canonical.append(CanonicalRule(
            action=rule.action,
            predicates=preds,
            weight=rule.weight,
            n_instances=rule.condition.n_instances,
        ))
    return canonical


def find_best_bdr_params(states, actions, param_grid=None,
                         cv_folds=3, random_state=42):
    """Find best BDR hyperparams via cross-validated macro-F1.

    Parameters
    ----------
    param_grid : dict or None
        Grid to search. Default:
          max_literals: [2, 3]
          max_rules_per_action: [4, 8]
          min_support_frac: [0.01, 0.03]

    Returns
    -------
    best_params : dict
    best_f1 : float
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score

    if param_grid is None:
        param_grid = {
            "max_literals": [2, 3],
            "max_rules_per_action": [4, 8],
            "min_support_frac": [0.01, 0.03],
        }

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                          random_state=random_state)

    best_f1 = -1.0
    best_params = {
        "max_literals": 3,
        "max_rules_per_action": 8,
        "min_support_frac": 0.01,
    }

    for ml in param_grid["max_literals"]:
        for mr in param_grid["max_rules_per_action"]:
            for ms in param_grid["min_support_frac"]:
                f1_scores = []
                for train_idx, val_idx in skf.split(states, actions):
                    bdr = BDRSurrogate(
                        max_literals=ml,
                        max_rules_per_action=mr,
                        min_support_frac=ms,
                        random_state=random_state,
                    )
                    bdr.fit(states[train_idx], actions[train_idx])
                    preds = bdr.predict(states[val_idx])
                    f1_scores.append(
                        f1_score(actions[val_idx], preds, average="macro"))
                mean_f1 = float(np.mean(f1_scores))
                if mean_f1 > best_f1:
                    best_f1 = mean_f1
                    best_params = {
                        "max_literals": ml,
                        "max_rules_per_action": mr,
                        "min_support_frac": ms,
                    }

    return best_params, best_f1
