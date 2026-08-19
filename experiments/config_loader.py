"""
Configuration loader for experiment configs.

Supports:
  - Loading a default YAML config
  - Merging environment-specific overrides on top
  - Deep-merging nested dicts so partial overrides work correctly

Usage:
    from experiments.config_loader import load_config

    # Load default only
    cfg = load_config()

    # Load default + environment override
    cfg = load_config(env_override="cartpole")

    # Load from a specific path
    cfg = load_config(config_path="experiments/configs/default.yaml")
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "experiments" / "configs"
_DEFAULT_CONFIG = _CONFIG_DIR / "default.yaml"

# Map short names → override files
_ENV_OVERRIDE_MAP = {
    "mountaincar": _CONFIG_DIR / "mountaincar.yaml",
    "MountainCar-v0": _CONFIG_DIR / "mountaincar.yaml",
    "cartpole": _CONFIG_DIR / "cartpole.yaml",
    "CartPole-v1": _CONFIG_DIR / "cartpole.yaml",
    "lunarlander": _CONFIG_DIR / "lunarlander.yaml",
    "LunarLander-v3": _CONFIG_DIR / "lunarlander.yaml",
    "minigrid_dynamic_obstacles": _CONFIG_DIR / "minigrid_dynamic_obstacles.yaml",
    "MiniGrid-Dynamic-Obstacles-8x8-v0": _CONFIG_DIR / "minigrid_dynamic_obstacles.yaml",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*.

    - Dict values are merged recursively.
    - All other types in *override* replace the value in *base*.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _load_yaml(path: Path | str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(
    config_path: str | Path | None = None,
    env_override: str | None = None,
) -> dict[str, Any]:
    """Load experiment configuration.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to a base YAML config.  Defaults to ``experiments/configs/default.yaml``.
    env_override : str, optional
        Environment short-name (e.g. ``"cartpole"``, ``"MountainCar-v0"``).
        The corresponding override file will be deep-merged on top of the base.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    base_path = Path(config_path) if config_path else _DEFAULT_CONFIG
    cfg = _load_yaml(base_path)

    if env_override is not None:
        override_path = _ENV_OVERRIDE_MAP.get(env_override)
        if override_path is None:
            raise ValueError(
                f"Unknown environment override '{env_override}'. "
                f"Available: {list(_ENV_OVERRIDE_MAP.keys())}"
            )
        override_cfg = _load_yaml(override_path)
        cfg = deep_merge(cfg, override_cfg)

    return cfg


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=== Default config ===")
    cfg = load_config()
    print(json.dumps(cfg, indent=2, default=str))

    print("\n=== CartPole override ===")
    cfg = load_config(env_override="cartpole")
    print(f"env.name = {cfg['env']['name']}")
    print(f"policy.train.total_timesteps = {cfg['policy']['train']['total_timesteps']}")

    print("\n=== MountainCar override ===")
    cfg = load_config(env_override="MountainCar-v0")
    print(f"env.name = {cfg['env']['name']}")
