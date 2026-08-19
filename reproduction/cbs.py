#!/usr/bin/env python
"""
CBS (Clustering-Based Summarizer) — Core Implementation.

Reproduces the CBS pipeline from Terra et al. (2026):
  1. Gini-based predicate generation (feature discretization)
  2. Predicate level encoding (continuous → categorical)
  3. K-means clustering per action
  4. Condition extraction from clusters
  5. Rule construction with w2 weighting
  6. Rule application (action prediction) with approximation fallback

Usage:
    from reproduction.cbs import CBSPipeline
    cbs = CBSPipeline(n_categories=5, inclusion_threshold=0.70)
    cbs.fit(states, actions)
    predicted_actions = cbs.predict(new_states)
    rules = cbs.get_rules()
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from kneed import KneeLocator
from dataclasses import dataclass, field
from typing import Optional


# ── Data Structures ───────────────────────────────────────────────────

@dataclass
class Predicate:
    """A single predicate: feature_index, comparator, threshold.

    Attributes
    ----------
    feature_idx : int
        Index of the state feature this predicate refers to.
    level : float
        Categorical level value in [-1, 1] (Terra encoding).
    level_label : str
        Human-readable label, e.g. "Very Low", "Low", …
    lower_bound : float or None
        Continuous lower bound of the bin in the original feature space
        (inclusive for the first bin, exclusive otherwise).  ``None`` if
        not yet computed (backward-compatible).
    upper_bound : float or None
        Continuous upper bound of the bin in the original feature space
        (inclusive).  ``None`` if not yet computed.
    """
    feature_idx: int
    level: float          # categorical level value in [-1, 1]
    level_label: str      # e.g., "Very Low", "Low", ...
    lower_bound: float = None   # continuous feature-space lower bound
    upper_bound: float = None   # continuous feature-space upper bound

    def __repr__(self):
        if self.lower_bound is not None and self.upper_bound is not None:
            return (f"f{self.feature_idx}={self.level_label}({self.level:+.2f})"
                    f"[{self.lower_bound:.4f},{self.upper_bound:.4f}]")
        return f"f{self.feature_idx}={self.level_label}({self.level:.2f})"


@dataclass
class Condition:
    """A conjunction of predicates describing a cluster."""
    predicates: list       # list of Predicate objects
    cluster_id: int = -1
    n_instances: int = 0   # how many states in this cluster

    def matches(self, encoded_state: np.ndarray) -> bool:
        """Check if an encoded state matches all predicates."""
        for p in self.predicates:
            if encoded_state[p.feature_idx] != p.level:
                return False
        return True

    def __repr__(self):
        preds = " AND ".join(str(p) for p in self.predicates)
        return f"[{preds}] (n={self.n_instances})"


@dataclass
class Rule:
    """A rule: action + condition + occurrence weight."""
    action: int
    condition: Condition
    weight: float          # w2 = N_ca / N_c

    def __repr__(self):
        return f"IF {self.condition} THEN action={self.action} (w={self.weight:.3f})"


# ── Level Labels ──────────────────────────────────────────────────────

def get_level_labels(n_categories: int) -> list:
    """Generate human-readable labels for categorical levels."""
    if n_categories == 2:
        return ["Low", "High"]
    elif n_categories == 3:
        return ["Low", "Medium", "High"]
    elif n_categories == 4:
        return ["Low", "Medium-Low", "Medium-High", "High"]
    elif n_categories == 5:
        return ["Very Low", "Low", "Medium", "High", "Very High"]
    elif n_categories == 6:
        return ["Very Low", "Low", "Medium-Low", "Medium-High", "High", "Very High"]
    elif n_categories == 7:
        return ["Very Low", "Low", "Medium-Low", "Medium", "Medium-High", "High", "Very High"]
    else:
        return [f"Level_{i}" for i in range(n_categories)]


def get_level_values(n_categories: int) -> np.ndarray:
    """Compute categorical level values in [-1, 1] per Terra Eq. (1)."""
    values = np.zeros(n_categories)
    values[0] = -1.0
    for i in range(1, n_categories - 1):
        values[i] = 2 * i / (n_categories - 1) - 1
    values[-1] = 1.0
    return values


# ── CBS Pipeline ──────────────────────────────────────────────────────

class CBSPipeline:
    """
    Clustering-Based Summarizer pipeline.

    Parameters
    ----------
    n_categories : int
        Number of discrete categories per feature (ncat). Default 5.
    inclusion_threshold : float
        Fraction of cluster instances that must be covered by
        frequent unique states. Default 0.70.
    max_clusters : int
        Maximum number of K-means clusters per action. Default 40.
    kmeans_seed : int or None
        Random seed for K-means. None = random.
    n_clusters_override : int or None
        If set, skip elbow method and use this many clusters.
    min_support : float
        Minimum fraction of action's states a condition must cover. Default 0.0.
    feature_names : list of str or None
        Human-readable feature names.
    """

    def __init__(
        self,
        n_categories: int = 5,
        inclusion_threshold: float = 0.70,
        max_clusters: int = 40,
        kmeans_seed: Optional[int] = None,
        n_clusters_override: Optional[int] = None,
        cluster_count_delta: int = 0,
        min_support: float = 0.0,
        feature_names: Optional[list] = None,
    ):
        self.n_categories = n_categories
        self.inclusion_threshold = inclusion_threshold
        self.max_clusters = max_clusters
        self.kmeans_seed = kmeans_seed
        self.n_clusters_override = n_clusters_override
        self.cluster_count_delta = cluster_count_delta
        self.min_support = min_support
        self.feature_names = feature_names

        # Computed during fit
        self.thresholds_ = None      # dict: feature_idx -> list of thresholds
        self.level_values_ = None    # array of categorical level values
        self.level_labels_ = None    # list of level label strings
        self.rules_ = None           # list of Rule objects
        self.actions_ = None         # sorted unique actions
        self.n_features_ = None
        self.encoded_conditions_ = None  # for approximation: list of (condition, encoded_center)
        self.cluster_counts_ = None  # dict: action -> n_clusters used (recorded during fit)

    # ── Step 1: Gini-based Predicate Generation ───────────────────

    def _generate_predicates(self, states: np.ndarray, actions: np.ndarray,
                             sample_weight: np.ndarray = None):
        """Algorithm 1: Train per-feature decision trees to find thresholds."""
        n_features = states.shape[1]
        self.thresholds_ = {}
        # Store observed feature ranges for bin-boundary computation
        self.feature_mins_ = {}
        self.feature_maxs_ = {}

        for f in range(n_features):
            self.feature_mins_[f] = float(states[:, f].min())
            self.feature_maxs_[f] = float(states[:, f].max())

            X_f = states[:, f].reshape(-1, 1)
            dt = DecisionTreeClassifier(
                max_leaf_nodes=self.n_categories,
                random_state=42,
            )
            dt.fit(X_f, actions, sample_weight=sample_weight)

            # Extract splitting thresholds from the tree
            tree = dt.tree_
            thresholds = sorted(set(
                tree.threshold[i]
                for i in range(tree.node_count)
                if tree.feature[i] != -2  # -2 means leaf node
            ))

            # Ensure we have exactly n_categories - 1 thresholds
            # If tree has fewer splits, add evenly spaced thresholds
            if len(thresholds) < self.n_categories - 1:
                f_min, f_max = states[:, f].min(), states[:, f].max()
                n_needed = self.n_categories - 1 - len(thresholds)
                extra = np.linspace(f_min, f_max, n_needed + 2)[1:-1]
                thresholds = sorted(set(thresholds) | set(extra.tolist()))

            # Take only the first n_categories - 1 thresholds
            thresholds = thresholds[: self.n_categories - 1]
            self.thresholds_[f] = thresholds

    def get_bin_boundaries(self, feature_idx: int, level_idx: int) -> tuple:
        """Return (lower_bound, upper_bound) for a given feature bin.

        Bin 0:            [feature_min, thresholds[0]]
        Bin i (0<i<ncat-1): (thresholds[i-1], thresholds[i]]
        Bin ncat-1:       (thresholds[-1], feature_max]
        """
        thresholds = self.thresholds_[feature_idx]
        f_min = self.feature_mins_[feature_idx]
        f_max = self.feature_maxs_[feature_idx]

        if level_idx == 0:
            return (f_min, thresholds[0])
        elif level_idx >= len(thresholds):
            return (thresholds[-1], f_max)
        else:
            return (thresholds[level_idx - 1], thresholds[level_idx])

    # ── Step 2: Predicate Level Encoding ──────────────────────────

    def _encode_states(self, states: np.ndarray) -> np.ndarray:
        """Encode continuous states into categorical levels per Eq. (1)."""
        n_samples, n_features = states.shape
        encoded = np.zeros((n_samples, n_features), dtype=np.float64)

        for f in range(n_features):
            thresholds = self.thresholds_[f]
            col = states[:, f]

            # Assign level index based on thresholds
            # Level 0: val <= thresholds[0]
            # Level i: thresholds[i-1] < val <= thresholds[i]
            # Level n_cat-1: val > thresholds[-1]
            level_indices = np.digitize(col, thresholds, right=True)
            # np.digitize with right=True: bin[i-1] < x <= bin[i]
            # Returns 0 for x <= thresholds[0], len(thresholds) for x > thresholds[-1]

            # Map indices to level values
            encoded[:, f] = self.level_values_[level_indices]

        return encoded

    def _encode_single(self, state: np.ndarray) -> np.ndarray:
        """Encode a single state (1D array)."""
        return self._encode_states(state.reshape(1, -1))[0]

    # ── Step 3: Clustering-Based Summarization ────────────────────

    def _find_n_clusters(self, data: np.ndarray) -> int:
        """Use elbow method to find optimal number of clusters."""
        if self.n_clusters_override is not None:
            return self.n_clusters_override

        n_samples = len(data)
        if n_samples < 3:
            return 1

        # Cap at the number of unique rows to avoid KMeans warnings
        n_unique = len(np.unique(data, axis=0))
        max_k = min(self.max_clusters, n_samples - 1, n_unique, 20)
        if max_k < 2:
            return 1

        k_range = range(2, max_k + 1)
        inertias = []
        for k in k_range:
            km = KMeans(
                n_clusters=k, init="k-means++", n_init=10,
                max_iter=300, random_state=self.kmeans_seed,
            )
            km.fit(data)
            inertias.append(km.inertia_)

        try:
            kl = KneeLocator(
                list(k_range), inertias,
                curve="convex", direction="decreasing",
            )
            if kl.knee is not None:
                return kl.knee
        except Exception:
            pass

        # Fallback: use k=5 or half of max
        return min(5, max_k)

    def _extract_conditions(
        self, encoded_data: np.ndarray, action: int,
        action_weights: np.ndarray = None,
    ) -> list:
        """Algorithm 2: Cluster encoded data and extract conditions."""
        n_samples = len(encoded_data)
        if n_samples == 0:
            return []

        # Find optimal number of clusters
        n_clusters = self._find_n_clusters(encoded_data)

        # Apply cluster_count_delta (algorithmic randomness)
        if self.cluster_count_delta != 0 and self.n_clusters_override is None:
            n_clusters = max(1, n_clusters + self.cluster_count_delta)

        n_clusters = min(n_clusters, n_samples)

        # Record cluster count per action (for inspection)
        if self.cluster_counts_ is not None:
            self.cluster_counts_[action] = n_clusters

        if n_clusters <= 1:
            # Single cluster: extract condition from all data
            return self._condition_from_group(
                encoded_data, 0, action,
                weights=action_weights)

        km = KMeans(
            n_clusters=n_clusters, init="k-means++", n_init=10,
            max_iter=300, random_state=self.kmeans_seed,
        )
        labels = km.fit_predict(encoded_data)

        conditions = []
        for k in range(n_clusters):
            mask = labels == k
            cluster_data = encoded_data[mask]
            cluster_weights = action_weights[mask] if action_weights is not None else None
            conds = self._condition_from_group(
                cluster_data, k, action, weights=cluster_weights)
            conditions.extend(conds)

        return conditions

    def _condition_from_group(
        self, cluster_data: np.ndarray, cluster_id: int, action: int,
        weights: np.ndarray = None,
    ) -> list:
        """Extract condition from a cluster using inclusion threshold.

        When weights is provided (importance-weighted voting), uses weighted frequency counts
        instead of raw counts. Backward-compatible: weights=None uses raw counts.
        """
        n = len(cluster_data)
        if n == 0:
            return []

        # Count frequency of each unique encoded state (optionally weighted)
        unique_states, inverse = np.unique(
            cluster_data, axis=0, return_inverse=True
        )
        if weights is not None:
            weighted_counts = np.zeros(len(unique_states))
            np.add.at(weighted_counts, inverse, weights)
            total = weights.sum()
        else:
            weighted_counts = np.bincount(inverse, minlength=len(unique_states)).astype(float)
            total = float(n)

        # Sort by frequency descending
        order = np.argsort(-weighted_counts)
        unique_states = unique_states[order]
        weighted_counts = weighted_counts[order]

        # Find minimum m such that cumsum(counts[0:m]) >= theta * total
        cumsum = np.cumsum(weighted_counts)
        target = self.inclusion_threshold * total
        m = np.searchsorted(cumsum, target, side="left") + 1
        m = min(m, len(weighted_counts))

        # Top-m frequent unique states
        top_states = unique_states[:m]

        # Extract predicates where ALL top states agree
        predicates = []
        for f in range(self.n_features_):
            values = top_states[:, f]
            if np.all(values == values[0]):
                level_val = values[0]
                # Find level index
                level_idx = np.argmin(np.abs(self.level_values_ - level_val))
                # Compute continuous bin boundaries
                lb, ub = self.get_bin_boundaries(f, level_idx)
                predicates.append(Predicate(
                    feature_idx=f,
                    level=self.level_values_[level_idx],
                    level_label=self.level_labels_[level_idx],
                    lower_bound=lb,
                    upper_bound=ub,
                ))

        if len(predicates) == 0:
            return []

        condition = Condition(
            predicates=predicates,
            cluster_id=cluster_id,
            n_instances=n,
        )
        return [condition]

    # ── Step 4: Rule Construction with w2 Weighting ───────────────

    def _build_rules(
        self, states: np.ndarray, actions: np.ndarray,
        encoded_states: np.ndarray,
        conditions_per_action: dict,
    ):
        """Build rules with w2 weighting: N_ca / N_c.

        When self.sample_weight_ is set, uses weighted counts.
        """
        rules = []
        sw = self.sample_weight_

        for action in self.actions_:
            conditions = conditions_per_action.get(action, [])
            for cond in conditions:
                # Count N_c: weighted states matching this condition (any action)
                n_c = sum(sw[i] for i in range(len(encoded_states))
                          if cond.matches(encoded_states[i]))
                # Count N_ca: weighted states matching AND having this action
                n_ca = sum(sw[i] for i in range(len(encoded_states))
                           if cond.matches(encoded_states[i])
                           and actions[i] == action)

                if n_c == 0:
                    weight = 0.0
                else:
                    weight = n_ca / n_c  # w2

                rules.append(Rule(
                    action=action,
                    condition=cond,
                    weight=weight,
                ))

        self.rules_ = rules

    # ── Main API ──────────────────────────────────────────────────

    def fit(self, states: np.ndarray, actions: np.ndarray,
            sample_weight: np.ndarray = None):
        """
        Fit the CBS pipeline.

        Parameters
        ----------
        states : ndarray of shape (N, n_features)
        actions : ndarray of shape (N,) with integer action labels
        sample_weight : ndarray of shape (N,), optional
            Per-sample weights (e.g., importance sampling). If None, uniform.
        """
        self.n_features_ = states.shape[1]
        self.actions_ = np.sort(np.unique(actions))
        self.level_values_ = get_level_values(self.n_categories)
        self.level_labels_ = get_level_labels(self.n_categories)
        self.sample_weight_ = (sample_weight if sample_weight is not None
                               else np.ones(len(states)))

        if self.feature_names is None:
            self.feature_names = [f"f{i}" for i in range(self.n_features_)]

        # Step 1: Generate predicates
        self._generate_predicates(states, actions, sample_weight=sample_weight)

        # Step 2: Encode states
        encoded = self._encode_states(states)

        # Step 3: Cluster and extract conditions per action
        self.cluster_counts_ = {}  # will be filled by _extract_conditions
        conditions_per_action = {}
        for action in self.actions_:
            mask = actions == action
            action_encoded = encoded[mask]
            action_weights = self.sample_weight_[mask]
            conditions = self._extract_conditions(
                action_encoded, action, action_weights=action_weights)
            conditions_per_action[action] = conditions

        # Step 4: Build rules with w2 weighting
        self._build_rules(states, actions, encoded, conditions_per_action)

        # Precompute encoded condition centers for approximation
        self._precompute_condition_centers()

        return self

    def _precompute_condition_centers(self):
        """Precompute encoded centers for approximation (Algorithm 4)."""
        self.encoded_conditions_ = []
        for rule in self.rules_:
            center = np.zeros(self.n_features_)
            specified = set()
            for p in rule.condition.predicates:
                center[p.feature_idx] = p.level
                specified.add(p.feature_idx)
            # For unspecified features, use 0.0 (middle)
            self.encoded_conditions_.append((rule, center, specified))

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Predict actions for states using extracted rules."""
        encoded = self._encode_states(states)
        predictions = np.zeros(len(states), dtype=int)

        for i in range(len(states)):
            predictions[i] = self._predict_single(encoded[i])

        return predictions

    def _predict_single(self, encoded_state: np.ndarray) -> int:
        """Predict action for a single encoded state (Eq. 2)."""
        # Sum weights per action for matching rules
        action_scores = {a: 0.0 for a in self.actions_}

        any_match = False
        for rule in self.rules_:
            if rule.condition.matches(encoded_state):
                action_scores[rule.action] += rule.weight
                any_match = True

        if any_match:
            return max(action_scores, key=action_scores.get)

        # Fallback: approximation (Algorithm 4)
        return self._approximate(encoded_state)

    def _approximate(self, encoded_state: np.ndarray) -> int:
        """Algorithm 4: Find closest condition by distance."""
        best_dist = float("inf")
        best_action = self.actions_[0]

        for rule, center, specified in self.encoded_conditions_:
            # Distance only on specified features
            if len(specified) == 0:
                continue
            dist = sum(
                (encoded_state[f] - center[f]) ** 2
                for f in specified
            )
            if dist < best_dist:
                best_dist = dist
                best_action = rule.action

        return best_action

    # ── Metrics ───────────────────────────────────────────────────

    def evaluate_fidelity(
        self, states: np.ndarray, actions: np.ndarray
    ) -> dict:
        """Compute fidelity metrics: accuracy, per-action recall, F1.

        NOTE: F1 here is Terra et al.'s E_F1 = 2·acc·recall/(acc+recall),
        i.e. the harmonic mean of overall accuracy and macro-recall.
        This differs from standard macro-F1 (harmonic mean of macro-precision
        and macro-recall).  The per-action method evaluate_fidelity_per_action()
        computes standard macro-F1.  The highway experiments use sklearn's
        f1_score(average='macro') which is also standard macro-F1.
        """
        pred = self.predict(states)

        # Overall accuracy
        accuracy = np.mean(pred == actions)

        # Per-action recall
        recalls = []
        for a in self.actions_:
            mask = actions == a
            if mask.sum() == 0:
                continue
            recalls.append(np.mean(pred[mask] == a))
        recall = np.mean(recalls) if recalls else 0.0

        # F1 (macro)
        if accuracy + recall > 0:
            f1 = 2 * accuracy * recall / (accuracy + recall)
        else:
            f1 = 0.0

        return {
            "accuracy": accuracy,
            "recall": recall,
            "f1": f1,
        }

    def evaluate_properties(self) -> dict:
        """Compute explanation property metrics."""
        if self.rules_ is None:
            return {}

        # E_len: total number of conditions
        n_conditions = len(self.rules_)

        # E_duplicate: conditions appearing in multiple actions
        # A condition is "duplicated" if its predicate signature appears
        # for more than one action
        cond_signatures = {}
        for rule in self.rules_:
            sig = tuple(
                (p.feature_idx, p.level) for p in rule.condition.predicates
            )
            cond_signatures.setdefault(sig, set()).add(rule.action)

        n_duplicated = sum(
            1 for actions in cond_signatures.values() if len(actions) > 1
        )

        return {
            "n_conditions": n_conditions,
            "n_duplicated": n_duplicated,
            "n_unique_signatures": len(cond_signatures),
        }

    def evaluate_coverage(
        self, states: np.ndarray
    ) -> dict:
        """Compute coverage metric: fraction of states needing approximation."""
        encoded = self._encode_states(states)
        n_approx = 0

        for i in range(len(encoded)):
            any_match = False
            for rule in self.rules_:
                if rule.condition.matches(encoded[i]):
                    any_match = True
                    break
            if not any_match:
                n_approx += 1

        return {
            "n_total": len(states),
            "n_approximated": n_approx,
            "approx_rate": n_approx / len(states) if len(states) > 0 else 0.0,
        }

    def evaluate_fidelity_per_action(
        self, states: np.ndarray, actions: np.ndarray
    ) -> dict:
        """Compute per-action precision, recall, support, F1, and rule count.

        This is essential for diagnosing rare-action instability (e.g.
        MountainCar action 1 ``no_push`` which has <1% support).

        Returns
        -------
        dict with keys:
            "per_action" : dict[int, dict]
                For each action: precision, recall, f1, support, rule_count
            "macro_precision", "macro_recall", "macro_f1" : float
        """
        pred = self.predict(states)
        result_per_action = {}

        precisions, recalls = [], []
        for a in self.actions_:
            true_mask = (actions == a)
            pred_mask = (pred == a)
            support = int(true_mask.sum())

            tp = int((true_mask & pred_mask).sum())
            # Precision = TP / (TP + FP)
            prec = tp / pred_mask.sum() if pred_mask.sum() > 0 else 0.0
            # Recall = TP / (TP + FN)
            rec = tp / true_mask.sum() if true_mask.sum() > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            # Count rules for this action
            rule_count = sum(1 for r in (self.rules_ or []) if r.action == a)

            result_per_action[int(a)] = {
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "support": support,
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

    # ── Display ───────────────────────────────────────────────────

    def get_rules(self) -> list:
        """Return list of Rule objects."""
        return self.rules_ or []

    def print_rules(self):
        """Pretty-print all rules grouped by action."""
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
                    fname = self.feature_names[p.feature_idx]
                    parts.append(f"{fname} is {p.level_label}")
                cond_str = " AND ".join(parts)
                print(f"    IF {cond_str} → w={r.weight:.3f} (n={r.condition.n_instances})")

    # ── Environment Deployment ────────────────────────────────────

    def evaluate_in_env(
        self, env_name: str, n_episodes: int = 50, seed: int = 0,
        eval_seeds: list = None,
        success_threshold: float = None,
    ) -> dict:
        """Deploy the extracted rules as a policy in the environment.

        Parameters
        ----------
        env_name : str
            Gymnasium environment id.
        n_episodes : int
            Number of evaluation episodes (ignored if *eval_seeds* given).
        seed : int
            Base seed.  Episode *i* uses ``seed + i`` (ignored if
            *eval_seeds* given).
        eval_seeds : list[int] or None
            If provided, use exactly these seeds for deterministic,
            reproducible evaluation across methods.  Length determines
            the number of episodes.
        success_threshold : float or None
            Environment-specific threshold above which an episode is
            considered "successful".  If None, success_rate is not
            computed.  Suggested defaults:
              - CartPole-v1: 475.0  (near-perfect)
              - MountainCar-v0: -150.0  (reached the goal)

        Returns
        -------
        dict with keys:
            E_CR, E_CR_std   — mean / std cumulative return
            E_TS, E_TS_std   — mean / std episode length
            E_AR             — average reward per step
            success_rate     — fraction of episodes ≥ success_threshold
                               (None if threshold not given)
            episode_rewards  — list of per-episode returns
            episode_lengths  — list of per-episode lengths
            eval_seeds_used  — list of seeds actually used
        """
        import gymnasium as gym

        if eval_seeds is None:
            eval_seeds = list(range(seed, seed + n_episodes))

        env = gym.make(env_name)
        episode_rewards = []
        episode_lengths = []

        for ep_seed in eval_seeds:
            obs, info = env.reset(seed=ep_seed)
            total_reward = 0.0
            steps = 0
            done = False

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
        ear = ecr / ets if ets > 0 else 0.0

        result = {
            "E_CR": ecr,
            "E_CR_std": float(rewards_arr.std()),
            "E_TS": ets,
            "E_TS_std": float(lengths_arr.std()),
            "E_AR": ear,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "eval_seeds_used": eval_seeds,
        }

        if success_threshold is not None:
            n_success = int((rewards_arr >= success_threshold).sum())
            result["success_rate"] = n_success / len(eval_seeds)
            result["success_threshold"] = success_threshold
        else:
            result["success_rate"] = None

        return result

    # ── MaxF1 Refinement (Algorithm 6) ────────────────────────────

    def refine_max_f1(
        self, states: np.ndarray, actions: np.ndarray,
        alpha: float = 0.5, max_iterations: int = 10,
        verbose: bool = False,
    ) -> dict:
        """
        Terra Algorithm 6: Maximize F1 by tuning predicate thresholds.

        For each threshold, try shifting left/right by k*alpha*(bin_width)
        and keep the shift that improves F1.

        Returns dict with before/after F1.
        """
        import copy

        before_f1 = self.evaluate_fidelity(states, actions)["f1"]

        for iteration in range(max_iterations):
            improved = False

            for f in range(self.n_features_):
                for t_idx in range(len(self.thresholds_[f])):
                    original_val = self.thresholds_[f][t_idx]

                    # Compute bin boundaries for this threshold
                    if t_idx == 0:
                        lower_bound = states[:, f].min()
                    else:
                        lower_bound = self.thresholds_[f][t_idx - 1]
                    if t_idx == len(self.thresholds_[f]) - 1:
                        upper_bound = states[:, f].max()
                    else:
                        upper_bound = self.thresholds_[f][t_idx + 1]

                    step = alpha * (upper_bound - lower_bound) / 2
                    if step <= 1e-12:
                        continue

                    best_f1 = self.evaluate_fidelity(states, actions)["f1"]
                    best_val = original_val

                    # Try shifting left and right
                    for direction in [-1, 1]:
                        new_val = original_val + direction * step
                        # Strict monotonicity: must stay within bounds
                        if new_val <= lower_bound or new_val >= upper_bound:
                            continue

                        # Temporarily update threshold
                        self.thresholds_[f][t_idx] = new_val
                        trial_f1 = self.evaluate_fidelity(states, actions)["f1"]

                        if trial_f1 > best_f1:
                            best_f1 = trial_f1
                            best_val = new_val
                            improved = True

                        # Restore
                        self.thresholds_[f][t_idx] = original_val

                    # Apply best shift (guaranteed monotonic)
                    self.thresholds_[f][t_idx] = best_val

            if verbose:
                cur_f1 = self.evaluate_fidelity(states, actions)["f1"]
                print(f"    MaxF1 iteration {iteration+1}: F1={cur_f1*100:.1f}%")

            if not improved:
                break

        # Rebuild rules with updated thresholds
        encoded = self._encode_states(states)
        conditions_per_action = {}
        for action in self.actions_:
            mask = actions == action
            action_encoded = encoded[mask]
            conditions = self._extract_conditions(action_encoded, action)
            conditions_per_action[action] = conditions
        self._build_rules(states, actions, encoded, conditions_per_action)
        self._precompute_condition_centers()

        after_f1 = self.evaluate_fidelity(states, actions)["f1"]

        return {
            "f1_before": before_f1,
            "f1_after": after_f1,
            "f1_improvement": after_f1 - before_f1,
        }

    # ── Display ───────────────────────────────────────────────────

    def get_rules(self) -> list:
        """Return list of Rule objects."""
        return self.rules_ or []

    def get_thresholds(self) -> dict:
        """Return predicate thresholds per feature."""
        return self.thresholds_

    def print_rules(self):
        """Pretty-print all rules grouped by action."""
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
                    fname = self.feature_names[p.feature_idx]
                    parts.append(f"{fname} is {p.level_label}")
                cond_str = " AND ".join(parts)
                print(f"    IF {cond_str} → w={r.weight:.3f} (n={r.condition.n_instances})")

    def print_thresholds(self):
        """Pretty-print predicate thresholds."""
        if not self.thresholds_:
            print("No thresholds computed.")
            return
        for f in range(self.n_features_):
            fname = self.feature_names[f]
            thresh = self.thresholds_[f]
            print(f"  {fname}: {[f'{t:.4f}' for t in thresh]}")
