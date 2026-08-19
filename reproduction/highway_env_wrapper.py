#!/usr/bin/env python
"""
Highway-env feature wrapper and environment utilities.

Provides a unified observation→feature pipeline for highway-env environments
(merge-v0, intersection-v0). Flattens the (vehicles_count, 7) Kinematics
observation into a 1-D feature vector with raw + derived features, matching
the specification in the experiment plan.

Usage:
    from reproduction.highway_env_wrapper import (
        make_highway_env, HighwayFeatureWrapper,
        HIGHWAY_OBS_CONFIG, HIGHWAY_ACTION_NAMES,
        RAW_FEATURE_NAMES, DERIVED_FEATURE_NAMES, ALL_FEATURE_NAMES,
    )
    env = make_highway_env("merge-v0", vehicles_count=6, flatten=True)
"""

import gymnasium as gym
import numpy as np

# Register highway-env environments
import highway_env  # noqa: F401

# ── Observation schema (shared across all highway experiments) ────────
HIGHWAY_OBS_FEATURES = ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]
N_OBS_FEATURES = len(HIGHWAY_OBS_FEATURES)  # 7

HIGHWAY_ACTION_NAMES = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}

N_ACTIONS = len(HIGHWAY_ACTION_NAMES)  # 5


def get_obs_config(vehicles_count: int = 6) -> dict:
    """Return the unified Kinematics observation config."""
    return {
        "type": "Kinematics",
        "vehicles_count": vehicles_count,
        "features": list(HIGHWAY_OBS_FEATURES),
        "absolute": False,
        "normalize": False,
        "order": "sorted",
    }


def get_env_config(vehicles_count: int = 6) -> dict:
    """Return complete env config with observation + action settings."""
    return {
        "observation": get_obs_config(vehicles_count),
        "action": {"type": "DiscreteMetaAction"},
    }


# ── Feature name generation ──────────────────────────────────────────

def _raw_feature_names(vehicles_count: int = 6) -> list:
    """Generate raw feature names: veh{i}_{feat} for each vehicle."""
    names = []
    for i in range(vehicles_count):
        for f in HIGHWAY_OBS_FEATURES:
            names.append(f"veh{i}_{f}")
    return names


def _derived_feature_names(vehicles_count: int = 6) -> list:
    """Generate derived feature names (ego-centric + relational + summary)."""
    names = []
    # Ego features
    names.extend(["ego_speed", "ego_lane_y"])
    # Per-neighbor relative features (vehicle 1..vehicles_count-1)
    for i in range(1, vehicles_count):
        names.extend([
            f"rel_dx_{i}", f"rel_dy_{i}",
            f"rel_dvx_{i}", f"rel_dvy_{i}",
            f"distance_{i}",
        ])
    # Summary features
    names.extend([
        "front_exists", "front_rel_dx", "front_rel_dvx",
        "left_exists", "left_rel_dx", "left_rel_dy", "left_rel_dvx",
        "right_exists", "right_rel_dx", "right_rel_dy", "right_rel_dvx",
        "closest_distance", "num_present_vehicles",
    ])
    return names


def raw_feature_names(vehicles_count: int = 6) -> list:
    return _raw_feature_names(vehicles_count)


def derived_feature_names(vehicles_count: int = 6) -> list:
    return _derived_feature_names(vehicles_count)


def all_feature_names(vehicles_count: int = 6) -> list:
    return _raw_feature_names(vehicles_count) + _derived_feature_names(vehicles_count)


# Canonical names for vehicles_count=6
RAW_FEATURE_NAMES = _raw_feature_names(6)
DERIVED_FEATURE_NAMES = _derived_feature_names(6)
ALL_FEATURE_NAMES = RAW_FEATURE_NAMES + DERIVED_FEATURE_NAMES


def compute_derived_features(obs_2d: np.ndarray) -> np.ndarray:
    """Compute derived features from (vehicles_count, 7) observation matrix.

    Args:
        obs_2d: Shape (vehicles_count, 7) with columns
                [presence, x, y, vx, vy, cos_h, sin_h].

    Returns:
        1-D array of derived features.
    """
    vehicles_count = obs_2d.shape[0]
    ego = obs_2d[0]  # ego vehicle is always row 0
    derived = []

    # Ego features
    ego_speed = np.sqrt(ego[3] ** 2 + ego[4] ** 2)  # sqrt(vx^2 + vy^2)
    ego_lane_y = ego[2]  # y position
    derived.extend([ego_speed, ego_lane_y])

    # Per-neighbor relative features
    for i in range(1, vehicles_count):
        veh = obs_2d[i]
        present = veh[0]
        if present > 0.5:
            rel_dx = veh[1] - ego[1]
            rel_dy = veh[2] - ego[2]
            rel_dvx = veh[3] - ego[3]
            rel_dvy = veh[4] - ego[4]
            distance = np.sqrt(rel_dx ** 2 + rel_dy ** 2)
        else:
            rel_dx = 0.0
            rel_dy = 0.0
            rel_dvx = 0.0
            rel_dvy = 0.0
            distance = 999.0
        derived.extend([rel_dx, rel_dy, rel_dvx, rel_dvy, distance])

    # Summary features: find front, left, right, closest vehicles
    front_idx = -1
    front_dx = float("inf")
    left_idx = -1
    left_dx = float("inf")
    right_idx = -1
    right_dx = float("inf")
    closest_idx = -1
    closest_dist = float("inf")
    num_present = 0

    for i in range(1, vehicles_count):
        veh = obs_2d[i]
        if veh[0] < 0.5:
            continue
        num_present += 1
        rel_dx = veh[1] - ego[1]
        rel_dy = veh[2] - ego[2]
        dist = np.sqrt(rel_dx ** 2 + rel_dy ** 2)
        if dist < closest_dist:
            closest_dist = dist
            closest_idx = i

        # Front: same lane (|dy| < 0.1) and dx > 0
        if abs(rel_dy) < 0.1 and rel_dx > 0 and rel_dx < front_dx:
            front_dx = rel_dx
            front_idx = i
        # Left: dy < -0.05 and |dx| small-ish
        if rel_dy < -0.05 and abs(rel_dx) < left_dx:
            left_dx = abs(rel_dx)
            left_idx = i
        # Right: dy > 0.05
        if rel_dy > 0.05 and abs(rel_dx) < right_dx:
            right_dx = abs(rel_dx)
            right_idx = i

    # Front summary
    if front_idx >= 0:
        fv = obs_2d[front_idx]
        derived.extend([1.0, fv[1] - ego[1], fv[3] - ego[3]])
    else:
        derived.extend([0.0, 0.0, 0.0])

    # Left summary
    if left_idx >= 0:
        lv = obs_2d[left_idx]
        derived.extend([1.0, lv[1] - ego[1], lv[2] - ego[2], lv[3] - ego[3]])
    else:
        derived.extend([0.0, 0.0, 0.0, 0.0])

    # Right summary
    if right_idx >= 0:
        rv = obs_2d[right_idx]
        derived.extend([1.0, rv[1] - ego[1], rv[2] - ego[2], rv[3] - ego[3]])
    else:
        derived.extend([0.0, 0.0, 0.0, 0.0])

    # Closest distance + num present
    derived.extend([
        closest_dist if closest_dist < 900 else 0.0,
        float(num_present),
    ])

    return np.array(derived, dtype=np.float32)


def flatten_obs_raw(obs_2d: np.ndarray) -> np.ndarray:
    """Flatten (vehicles_count, 7) to 1-D raw feature vector."""
    return obs_2d.flatten().astype(np.float32)


def flatten_obs_full(obs_2d: np.ndarray) -> np.ndarray:
    """Flatten (vehicles_count, 7) to raw + derived feature vector."""
    raw = flatten_obs_raw(obs_2d)
    derived = compute_derived_features(obs_2d)
    return np.concatenate([raw, derived])


class HighwayFeatureWrapper(gym.ObservationWrapper):
    """Wraps highway-env to produce flat 1-D feature vectors.

    Args:
        env: A highway-env gymnasium environment.
        feature_mode: 'raw' for raw features only,
                      'raw_derived' for raw + derived features.
    """

    def __init__(self, env, feature_mode: str = "raw_derived"):
        super().__init__(env)
        self.feature_mode = feature_mode
        # Determine output dimension from a dummy observation
        dummy = np.zeros((env.observation_space.shape[0],
                          env.observation_space.shape[1]), dtype=np.float32)
        if feature_mode == "raw":
            out = flatten_obs_raw(dummy)
        else:
            out = flatten_obs_full(dummy)
        self._n_features = len(out)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._n_features,),
            dtype=np.float32,
        )

    def observation(self, obs):
        if self.feature_mode == "raw":
            return flatten_obs_raw(obs)
        return flatten_obs_full(obs)

    @property
    def n_features(self):
        return self._n_features


def make_highway_env(
    env_name: str,
    vehicles_count: int = 6,
    feature_mode: str = "raw_derived",
    flatten: bool = True,
) -> gym.Env:
    """Create a highway-env environment with unified config.

    Args:
        env_name: 'merge-v0' or 'intersection-v0'.
        vehicles_count: Number of observed vehicles.
        feature_mode: 'raw' or 'raw_derived'.
        flatten: If True, wrap with HighwayFeatureWrapper.

    Returns:
        Gymnasium environment with either (vc, 7) or flat observation space.
    """
    config = get_env_config(vehicles_count)
    env = gym.make(env_name, config=config)
    if flatten:
        env = HighwayFeatureWrapper(env, feature_mode=feature_mode)
    return env


def get_feature_names(vehicles_count: int = 6,
                      feature_mode: str = "raw_derived") -> list:
    """Return feature names matching the flattened observation vector."""
    if feature_mode == "raw":
        return raw_feature_names(vehicles_count)
    return all_feature_names(vehicles_count)
