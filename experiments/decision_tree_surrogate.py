#!/usr/bin/env python
"""
Decision Tree surrogate (DT) baseline

Fits a DecisionTreeClassifier on (state, action) replay data,
converts tree paths into Rule / CanonicalRule objects compatible
with the existing CBS evaluation pipeline, and provides predict()
and evaluate_in_env() for drop-in usage alongside CBS methods.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from dataclasses import dataclass
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reproduction.cbs import Predicate, Condition, Rule
from experiments.rule_matching import CanonicalPredicate, CanonicalRule


class DecisionTreeSurrogate:
    """Decision-tree policy surrogate.

    Wraps sklearn DecisionTreeClassifier with the same external API as
    CBSPipeline so that existing evaluation helpers work unchanged.

    Parameters
    ----------
    max_depth : int or None
        Maximum tree depth.  None = unlimited (will overfit).
    min_samples_leaf : int
        Minimum samples per leaf.
    random_state : int
        Deterministic tree building.
    feature_names : list[str] or None
        Human-readable feature names.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 5,
        random_state: int = 42,
        feature_names: Optional[list] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.feature_names = feature_names

        self.tree_: Optional[DecisionTreeClassifier] = None
        self.rules_: Optional[list] = None  # list[Rule]
        self.actions_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.thresholds_: Optional[dict] = None  # feat_idx -> sorted thresholds
        self.feature_mins_: Optional[dict] = None
        self.feature_maxs_: Optional[dict] = None

    # ── Fitting ───────────────────────────────────────────────────

    def fit(self, states: np.ndarray, actions: np.ndarray,
            sample_weight: np.ndarray = None):
        """Train the decision tree and extract rules."""
        self.n_features_ = states.shape[1]
        self.actions_ = np.sort(np.unique(actions))
        self.feature_mins_ = {f: float(states[:, f].min())
                              for f in range(self.n_features_)}
        self.feature_maxs_ = {f: float(states[:, f].max())
                              for f in range(self.n_features_)}

        self.tree_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.tree_.fit(states, actions, sample_weight=sample_weight)
        self.rules_ = self._extract_rules()
        self.thresholds_ = self._extract_thresholds()

    # ── Rule Extraction ───────────────────────────────────────────

    def _extract_rules(self) -> list:
        """Convert each leaf path in the decision tree into a Rule object."""
        tree = self.tree_.tree_
        rules = []
        n_total = tree.n_node_samples[0]

        def _recurse(node_id, constraints):
            """Walk the tree; at each leaf, emit a Rule."""
            if tree.children_left[node_id] == tree.children_right[node_id]:
                # Leaf node
                action = int(np.argmax(tree.value[node_id]))
                n_samples = int(tree.n_node_samples[node_id])

                # Build intervals per feature from constraints
                feat_intervals = {}  # feat_idx -> (lower, upper)
                for feat, direction, threshold in constraints:
                    lo, hi = feat_intervals.get(
                        feat, (self.feature_mins_[feat], self.feature_maxs_[feat]))
                    if direction == "<=":
                        hi = min(hi, threshold)
                    else:  # ">"
                        lo = max(lo, threshold)
                    feat_intervals[feat] = (lo, hi)

                # Create predicates from intervals
                predicates = []
                for feat in sorted(feat_intervals.keys()):
                    lo, hi = feat_intervals[feat]
                    mid = (lo + hi) / 2.0
                    # Map to a pseudo-level in [-1, 1] based on feature range
                    feat_range = self.feature_maxs_[feat] - self.feature_mins_[feat]
                    if feat_range > 0:
                        normalized = (mid - self.feature_mins_[feat]) / feat_range
                        level = 2.0 * normalized - 1.0  # map to [-1, 1]
                    else:
                        level = 0.0
                    level = float(np.clip(level, -1.0, 1.0))
                    label = _level_to_label(level)

                    predicates.append(Predicate(
                        feature_idx=feat,
                        level=level,
                        level_label=label,
                        lower_bound=lo,
                        upper_bound=hi,
                    ))

                if not predicates:
                    # Root-only tree (no splits) — use full feature range
                    for feat in range(self.n_features_):
                        predicates.append(Predicate(
                            feature_idx=feat,
                            level=0.0,
                            level_label="Medium",
                            lower_bound=self.feature_mins_[feat],
                            upper_bound=self.feature_maxs_[feat],
                        ))

                condition = Condition(predicates=predicates, n_instances=n_samples)
                weight = n_samples / n_total if n_total > 0 else 0.0
                rules.append(Rule(action=action, condition=condition, weight=weight))
                return

            # Internal node — recurse
            feat = int(tree.feature[node_id])
            threshold = float(tree.threshold[node_id])

            # Left child: feat <= threshold
            _recurse(tree.children_left[node_id],
                     constraints + [(feat, "<=", threshold)])
            # Right child: feat > threshold
            _recurse(tree.children_right[node_id],
                     constraints + [(feat, ">", threshold)])

        _recurse(0, [])
        return rules

    def _extract_thresholds(self) -> dict:
        """Extract per-feature threshold lists from the tree for TD metric."""
        tree = self.tree_.tree_
        thresholds = {f: set() for f in range(self.n_features_)}

        def _collect(node_id):
            if tree.children_left[node_id] == tree.children_right[node_id]:
                return  # leaf
            feat = int(tree.feature[node_id])
            thresholds[feat].add(float(tree.threshold[node_id]))
            _collect(tree.children_left[node_id])
            _collect(tree.children_right[node_id])

        _collect(0)
        return {f: sorted(list(ts)) for f, ts in thresholds.items()}

    # ── Prediction ────────────────────────────────────────────────

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Predict actions using the fitted decision tree."""
        return self.tree_.predict(states).astype(int)

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

    def evaluate_fidelity_per_action(self, states: np.ndarray, actions: np.ndarray) -> dict:
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
            rule_count = sum(1 for r in (self.rules_ or []) if r.action == a)
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
        n_conditions = len(self.rules_) if self.rules_ else 0
        cond_sigs = {}
        for rule in (self.rules_ or []):
            sig = tuple((p.feature_idx, round(p.level, 4)) for p in rule.condition.predicates)
            cond_sigs.setdefault(sig, set()).add(rule.action)
        n_dup = sum(1 for acts in cond_sigs.values() if len(acts) > 1)
        return {
            "n_conditions": n_conditions,
            "n_duplicated": n_dup,
            "n_unique_signatures": len(cond_sigs),
        }

    def evaluate_coverage(self, states: np.ndarray) -> dict:
        # DT always covers all states (no approximation needed)
        return {
            "n_total": len(states),
            "n_approximated": 0,
            "approx_rate": 0.0,
        }

    def evaluate_in_env(self, env_name: str, n_episodes: int = 50,
                        seed: int = 0, eval_seeds: list = None,
                        success_threshold: float = None) -> dict:
        import gymnasium as gym
        if eval_seeds is None:
            eval_seeds = list(range(seed, seed + n_episodes))
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
        if not self.rules_:
            print("No rules extracted.")
            return
        for action in self.actions_:
            action_rules = [r for r in self.rules_ if r.action == action]
            action_rules.sort(key=lambda r: -r.weight)
            print(f"\n  Action {action}:")
            for r in action_rules:
                parts = []
                for p in r.condition.predicates:
                    fname = (self.feature_names[p.feature_idx]
                             if self.feature_names else f"f{p.feature_idx}")
                    parts.append(f"{fname} in [{p.lower_bound:.4f}, {p.upper_bound:.4f}]")
                cond_str = " AND ".join(parts)
                print(f"    IF {cond_str} → action={action} "
                      f"(w={r.weight:.3f}, n={r.condition.n_instances})")


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


def canonicalize_dt_rules(rules: list) -> list:
    """Convert DT Rule objects to CanonicalRule for stability metrics."""
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


def find_best_depth(states, actions, max_depths=(3, 4, 5, 6, 7, 8, 10, None),
                    min_samples_leaf=5, cv_folds=3, random_state=42):
    """Find the best max_depth via cross-validated F1 score."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score

    best_depth = 5
    best_f1 = -1.0

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    for depth in max_depths:
        f1_scores = []
        for train_idx, val_idx in skf.split(states, actions):
            dt = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            )
            dt.fit(states[train_idx], actions[train_idx])
            preds = dt.predict(states[val_idx])
            f1_scores.append(f1_score(actions[val_idx], preds, average="macro"))
        mean_f1 = np.mean(f1_scores)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_depth = depth

    return best_depth, best_f1
