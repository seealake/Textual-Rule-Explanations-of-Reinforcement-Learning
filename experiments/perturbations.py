"""
Replay Perturbation Utilities

Three perturbation families for stress-testing CBS rule explanations:

1. **Seed shift** — collect K replay datasets with different env seeds
   (same pretrained policy).  Handled by calling `collect_replay` from
   `reproduction/collect_replay.py` with varying seeds.

2. **Bootstrap resampling** — draw B subsets (default 80 % without
   replacement) from a single reference replay dataset.

3. **Feature noise** — inject Gaussian noise N(0, (σ·range_f)²) into
   each state feature, where σ is a fraction of the per-feature range
   observed in the reference dataset.

All functions accept and return the same dict format used by
`reproduction/collect_replay.py`:

    {
        "states":      np.ndarray  (N, obs_dim),
        "actions":     np.ndarray  (N,),
        "rewards":     np.ndarray  (N,),
        "dones":       np.ndarray  (N,),
        "episode_ids": np.ndarray  (N,),
    }

Usage examples
--------------
>>> from experiments.perturbations import generate_bootstrap_subsets, add_feature_noise
>>> subsets = generate_bootstrap_subsets(data, n_subsets=5, fraction=0.8, seed=42)
>>> noisy  = add_feature_noise(data, noise_level=0.03, seed=0)
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import numpy as np

# Type alias for a replay dataset dict.
ReplayData = Dict[str, Any]


# ───────────────────────────────────────────────────────────────────────
# Helper: load .npz into the standard dict
# ───────────────────────────────────────────────────────────────────────

def load_replay_npz(path: str) -> ReplayData:
    """Load a replay dataset saved as .npz and return a standard dict."""
    with np.load(path) as f:
        data = {
            "states": f["states"],
            "actions": f["actions"],
            "rewards": f["rewards"],
            "dones": f["dones"],
            "episode_ids": f["episode_ids"],
        }
    return data


# ───────────────────────────────────────────────────────────────────────
# 2.  Subsampling (formerly "bootstrap resampling")
#     NOTE: This is sampling **without replacement**, which is technically
#     subsampling / subagging, not the statistical bootstrap.
# ───────────────────────────────────────────────────────────────────────

def generate_subsamples(
    data: ReplayData,
    n_subsets: int = 5,
    fraction: float = 0.80,
    seed: int = 42,
) -> List[ReplayData]:
    """Draw *n_subsets* sub-samples **without replacement**.

    Each sub-sample contains ``int(fraction * N)`` transitions drawn
    uniformly at random (without replacement) from *data*.

    Parameters
    ----------
    data : ReplayData
        Reference replay dataset.
    n_subsets : int
        Number of subsets to generate (B).
    fraction : float
        Fraction of transitions to keep per subset (0 < fraction ≤ 1).
    seed : int
        Base random seed; subset *i* uses ``seed + i``.

    Returns
    -------
    list of ReplayData
    """
    n_total = len(data["states"])
    n_sample = int(fraction * n_total)
    if n_sample <= 0 or n_sample > n_total:
        raise ValueError(
            f"Invalid sample size {n_sample} (fraction={fraction}, "
            f"total={n_total})"
        )

    subsets: List[ReplayData] = []
    for i in range(n_subsets):
        rng = np.random.default_rng(seed + i)
        indices = rng.choice(n_total, size=n_sample, replace=False)
        indices.sort()

        subset: ReplayData = {
            "states": data["states"][indices].copy(),
            "actions": data["actions"][indices].copy(),
            "rewards": data["rewards"][indices].copy(),
            "dones": data["dones"][indices].copy(),
            "episode_ids": data["episode_ids"][indices].copy(),
        }
        subsets.append(subset)

    return subsets


# Backward-compatible alias
generate_bootstrap_subsets = generate_subsamples


def generate_stratified_subsamples(
    data: ReplayData,
    n_subsets: int = 5,
    fraction: float = 0.80,
    seed: int = 42,
) -> List[ReplayData]:
    """Draw *n_subsets* **action-stratified** sub-samples without replacement.

    Within each action class the same *fraction* of transitions is sampled,
    so the class distribution is preserved even for rare actions.
    This is critical for MountainCar where action 1 (no_push) ≈ 0.9 %
    and uniform subsampling can dilute it further, confounding instability
    measurements with class imbalance artifacts.

    Parameters
    ----------
    data : ReplayData
    n_subsets : int
    fraction : float
    seed : int

    Returns
    -------
    list of ReplayData
    """
    actions = data["actions"]
    unique_actions = np.unique(actions)

    subsets: List[ReplayData] = []
    for i in range(n_subsets):
        rng = np.random.default_rng(seed + i)
        all_indices = []
        for a in unique_actions:
            a_indices = np.where(actions == a)[0]
            n_a = max(1, int(fraction * len(a_indices)))
            chosen = rng.choice(a_indices, size=n_a, replace=False)
            all_indices.append(chosen)
        indices = np.sort(np.concatenate(all_indices))

        subset: ReplayData = {
            "states": data["states"][indices].copy(),
            "actions": data["actions"][indices].copy(),
            "rewards": data["rewards"][indices].copy(),
            "dones": data["dones"][indices].copy(),
            "episode_ids": data["episode_ids"][indices].copy(),
        }
        subsets.append(subset)

    return subsets


# ───────────────────────────────────────────────────────────────────────
# 3.  Feature noise injection
# ───────────────────────────────────────────────────────────────────────

def compute_feature_ranges(data: ReplayData) -> np.ndarray:
    """Return per-feature range (max − min) from the state matrix.

    Returns
    -------
    np.ndarray of shape (obs_dim,)
        Each entry is ``max(feature_i) − min(feature_i)`` over all
        transitions.  If a feature is constant, its range is set to 1.0
        to avoid zero-division.
    """
    states = data["states"]
    f_min = states.min(axis=0)
    f_max = states.max(axis=0)
    ranges = f_max - f_min
    # Guard against constant features (range == 0)
    ranges = np.where(ranges > 0, ranges, 1.0)
    return ranges


def add_feature_noise(
    data: ReplayData,
    noise_level: float,
    seed: int = 0,
    feature_ranges: np.ndarray | None = None,
) -> ReplayData:
    """Return a copy of *data* with Gaussian noise added to state features.

    For each feature *f*, we add ``N(0, (noise_level * range_f)²)``
    where ``range_f = max(f) − min(f)`` computed from the dataset.

    Parameters
    ----------
    data : ReplayData
        Original replay dataset.
    noise_level : float
        Standard-deviation multiplier expressed as a fraction of the
        feature range (e.g. 0.01 for ~1 %, 0.03 for ~3 %, 0.05 for ~5 %).
    seed : int
        Random seed for reproducibility.
    feature_ranges : np.ndarray, optional
        Pre-computed per-feature ranges.  If ``None``, computed from
        *data* automatically.

    Returns
    -------
    ReplayData
        New dict with noisy ``"states"``; all other arrays are copied
        unchanged.
    """
    if feature_ranges is None:
        feature_ranges = compute_feature_ranges(data)

    rng = np.random.default_rng(seed)
    states = data["states"].copy().astype(np.float64)
    # sigma per feature: noise_level * range_f
    sigma = noise_level * feature_ranges  # shape (obs_dim,)
    noise = rng.normal(loc=0.0, scale=sigma, size=states.shape)
    states += noise
    states = states.astype(np.float32)

    noisy: ReplayData = {
        "states": states,
        "actions": data["actions"].copy(),
        "rewards": data["rewards"].copy(),
        "dones": data["dones"].copy(),
        "episode_ids": data["episode_ids"].copy(),
    }
    return noisy


# ───────────────────────────────────────────────────────────────────────
# Quick self-test
# ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Tiny synthetic dataset for smoke-testing
    np.random.seed(0)
    N, D = 100, 4
    fake_data: ReplayData = {
        "states": np.random.randn(N, D).astype(np.float32),
        "actions": np.random.randint(0, 2, size=N).astype(np.int32),
        "rewards": np.ones(N, dtype=np.float32),
        "dones": np.zeros(N, dtype=bool),
        "episode_ids": np.zeros(N, dtype=np.int32),
    }

    # Bootstrap
    subsets = generate_bootstrap_subsets(fake_data, n_subsets=3, fraction=0.8, seed=0)
    print(f"Bootstrap: generated {len(subsets)} subsets, sizes = "
          f"{[len(s['states']) for s in subsets]}")

    # Feature noise
    ranges = compute_feature_ranges(fake_data)
    print(f"Feature ranges: {ranges}")
    for lvl in [0.01, 0.03, 0.05]:
        noisy = add_feature_noise(fake_data, noise_level=lvl, seed=0)
        diff = np.abs(noisy["states"] - fake_data["states"])
        print(f"  noise_level={lvl:.2f}  mean |Δ|={diff.mean():.6f}  "
              f"max |Δ|={diff.max():.6f}")

    print("\n✓ perturbations.py self-test passed")
