#!/usr/bin/env python
"""
Generate all replay perturbation datasets.

This script produces three families of perturbed replay data for a given
environment, ready for downstream CBS stability experiments:

  1. **Seed shift**        — K new replay datasets collected with different
                              environment seeds (same pretrained policy).
  2. **Bootstrap resample** — B subsets drawn without replacement (80 %)
                              from the reference replay dataset.
  3. **Feature noise**     — Gaussian noise injected at 3 severity levels
                              (≈1 %, 3 %, 5 % of per-feature range).

All outputs are saved as both .npz (fast loading) and .csv (inspection)
in a structured directory tree under ``reproduction/data/perturbations/``.

Usage
-----
    # Generate all perturbations for MountainCar-v0 (uses default config)
    python experiments/generate_perturbations.py --env MountainCar-v0

    # Generate all perturbations for CartPole-v1
    python experiments/generate_perturbations.py --env CartPole-v1

    # Generate only seed-shift perturbations
    python experiments/generate_perturbations.py --env MountainCar-v0 --only seed_shift

    # Generate only bootstrap perturbations
    python experiments/generate_perturbations.py --env CartPole-v1 --only bootstrap

    # Generate only feature-noise perturbations
    python experiments/generate_perturbations.py --env CartPole-v1 --only feature_noise

    # Custom parameters (override config defaults)
    python experiments/generate_perturbations.py --env MountainCar-v0 \\
        --n-seeds 10 --n-bootstraps 10 --noise-levels 0.01 0.02 0.05 0.10

Directory layout produced
-------------------------
    reproduction/data/perturbations/<env_tag>/
        seed_shift/
            replay_<env_tag>_seed0.{npz,csv}
            replay_<env_tag>_seed1.{npz,csv}
            ...
        bootstrap/
            replay_<env_tag>_bootstrap0.{npz,csv}
            replay_<env_tag>_bootstrap1.{npz,csv}
            ...
        feature_noise/
            replay_<env_tag>_noise0.01.{npz,csv}
            replay_<env_tag>_noise0.03.{npz,csv}
            replay_<env_tag>_noise0.05.{npz,csv}
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to path so we can import from reproduction/ and experiments/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from experiments.config_loader import load_config
from experiments.perturbations import (
    add_feature_noise,
    compute_feature_ranges,
    generate_bootstrap_subsets,
    load_replay_npz,
)
from reproduction.collect_replay import (
    ENV_FEATURE_NAMES,
    collect_replay,
    save_replay,
    print_summary,
)


# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────

def _env_tag(env_name: str) -> str:
    """Convert 'MountainCar-v0' → 'mountaincar_v0'."""
    return env_name.replace("-", "_").lower()


def _save_perturbed(
    data: dict,
    env_name: str,
    output_dir: str,
    base_name: str,
) -> tuple[str, str]:
    """Save a perturbed replay dataset as .npz and .csv.

    Returns (npz_path, csv_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    # .npz
    npz_path = os.path.join(output_dir, base_name + ".npz")
    np.savez_compressed(
        npz_path,
        states=data["states"],
        actions=data["actions"],
        rewards=data["rewards"],
        dones=data["dones"],
        episode_ids=data["episode_ids"],
    )

    # .csv
    feature_names = ENV_FEATURE_NAMES.get(env_name)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(data["states"].shape[1])]
    df = pd.DataFrame(data["states"], columns=feature_names)
    df["action"] = data["actions"]
    df["reward"] = data["rewards"]
    df["done"] = data["dones"]
    df["episode_id"] = data["episode_ids"]
    csv_path = os.path.join(output_dir, base_name + ".csv")
    df.to_csv(csv_path, index=False)

    return npz_path, csv_path


def _print_data_summary(data: dict, label: str) -> None:
    """Print a one-line summary of a dataset."""
    n = len(data["states"])
    n_actions = len(np.unique(data["actions"]))
    print(f"    {label}: {n:,} transitions, {n_actions} unique actions, "
          f"state shape {data['states'].shape}")


# ───────────────────────────────────────────────────────────────────────
# 1. Seed Shift
# ───────────────────────────────────────────────────────────────────────

def generate_seed_shift(
    env_name: str,
    model_path: str,
    n_transitions: int,
    seeds: list[int],
    output_dir: str,
) -> list[str]:
    """Collect K replay datasets with different env seeds.

    Returns list of .npz paths created.
    """
    tag = _env_tag(env_name)
    created: list[str] = []

    for seed in seeds:
        print(f"  [seed_shift] Collecting replay with seed={seed} ...")
        data = collect_replay(
            env_name=env_name,
            model_path=model_path,
            num_transitions=n_transitions,
            seed=seed,
            deterministic=True,
        )
        base_name = f"replay_{tag}_seed{seed}"
        npz_path, csv_path = _save_perturbed(data, env_name, output_dir, base_name)
        _print_data_summary(data, f"seed={seed}")
        created.append(npz_path)

    return created


# ───────────────────────────────────────────────────────────────────────
# 2. Bootstrap Resampling
# ───────────────────────────────────────────────────────────────────────

def generate_bootstrap(
    ref_data: dict,
    env_name: str,
    n_subsets: int,
    fraction: float,
    seed: int,
    output_dir: str,
) -> list[str]:
    """Generate B bootstrap subsets from the reference replay.

    Returns list of .npz paths created.
    """
    tag = _env_tag(env_name)
    subsets = generate_bootstrap_subsets(
        ref_data, n_subsets=n_subsets, fraction=fraction, seed=seed,
    )
    created: list[str] = []

    for i, subset in enumerate(subsets):
        base_name = f"replay_{tag}_bootstrap{i}"
        npz_path, csv_path = _save_perturbed(subset, env_name, output_dir, base_name)
        _print_data_summary(subset, f"bootstrap[{i}]")
        created.append(npz_path)

    return created


# ───────────────────────────────────────────────────────────────────────
# 3. Feature Noise
# ───────────────────────────────────────────────────────────────────────

def generate_feature_noise(
    ref_data: dict,
    env_name: str,
    noise_levels: list[float],
    seed: int,
    output_dir: str,
) -> list[str]:
    """Generate noisy replay datasets for each noise level.

    Returns list of .npz paths created.
    """
    tag = _env_tag(env_name)
    ranges = compute_feature_ranges(ref_data)
    created: list[str] = []

    for lvl in noise_levels:
        noisy = add_feature_noise(ref_data, noise_level=lvl, seed=seed,
                                  feature_ranges=ranges)
        base_name = f"replay_{tag}_noise{lvl:.2f}"
        npz_path, csv_path = _save_perturbed(noisy, env_name, output_dir, base_name)

        # Report noise magnitude
        diff = np.abs(noisy["states"] - ref_data["states"])
        _print_data_summary(noisy, f"noise={lvl:.2f}  (mean|Δ|={diff.mean():.6f})")
        created.append(npz_path)

    return created


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate all replay perturbation datasets"
    )
    parser.add_argument(
        "--env", type=str, default="MountainCar-v0",
        help="Gymnasium environment name (default: MountainCar-v0)",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        choices=["seed_shift", "bootstrap", "feature_noise"],
        help="Generate only one perturbation family (default: all)",
    )
    # Override parameters (take precedence over YAML config)
    parser.add_argument("--n-seeds", type=int, default=None,
                        help="Number of seed-shift seeds (K)")
    parser.add_argument("--seed-list", type=int, nargs="+", default=None,
                        help="Explicit list of seeds for seed shift")
    parser.add_argument("--n-bootstraps", type=int, default=None,
                        help="Number of bootstrap subsets (B)")
    parser.add_argument("--bootstrap-fraction", type=float, default=None,
                        help="Fraction of data per bootstrap subset")
    parser.add_argument("--noise-levels", type=float, nargs="+", default=None,
                        help="Feature noise levels (as fractions of range)")
    parser.add_argument("--n-transitions", type=int, default=None,
                        help="Transitions per replay dataset")
    parser.add_argument("--ref-seed", type=int, default=42,
                        help="Seed of the reference replay dataset (default: 42)")
    parser.add_argument("--output-base", type=str,
                        default="reproduction/data/perturbations",
                        help="Base output directory")
    args = parser.parse_args()

    # ── Load config ──
    cfg = load_config(env_override=args.env)
    pert_cfg = cfg["perturbations"]["replay"]
    tag = _env_tag(args.env)

    # ── Resolve parameters (CLI > YAML > hardcoded defaults) ──
    n_transitions = args.n_transitions or cfg["replay"]["n_transitions"]

    # Seed shift
    if args.seed_list is not None:
        seed_shift_seeds = args.seed_list
    else:
        n_seeds = args.n_seeds or pert_cfg["seed_shift"]["n_seeds"]
        seed_shift_seeds = list(range(n_seeds))

    # Bootstrap
    n_bootstraps = args.n_bootstraps or pert_cfg["bootstrap"]["n_resamples"]
    bootstrap_fraction = (args.bootstrap_fraction
                          or pert_cfg["bootstrap"]["sample_fraction"])

    # Feature noise
    noise_levels = args.noise_levels or pert_cfg["feature_noise"]["noise_levels"]

    # Paths
    model_dir = cfg["policy"]["model_dir"]
    model_path = os.path.join(model_dir, f"dqn_{tag}.zip")
    ref_replay_path = os.path.join(
        cfg["replay"]["data_dir"], f"replay_{tag}_seed{args.ref_seed}.npz"
    )

    # ── Print plan ──
    run_all = args.only is None
    print(f"\n{'='*64}")
    print(f"  Replay Perturbation Generation")
    print(f"{'='*64}")
    print(f"  Environment:       {args.env}")
    print(f"  Model:             {model_path}")
    print(f"  Reference replay:  {ref_replay_path}")
    print(f"  Transitions/set:   {n_transitions:,}")
    print(f"  Output base:       {args.output_base}/{tag}/")
    print(f"{'─'*64}")
    if run_all or args.only == "seed_shift":
        print(f"  Seed shift:     seeds = {seed_shift_seeds}")
    if run_all or args.only == "bootstrap":
        print(f"  Bootstrap:      B = {n_bootstraps}, fraction = {bootstrap_fraction}")
    if run_all or args.only == "feature_noise":
        print(f"  Feature noise:  levels = {noise_levels}")
    print(f"{'='*64}\n")

    # ── Validate prerequisites ──
    if (run_all or args.only == "seed_shift") and not os.path.exists(model_path):
        print(f"ERROR: Pretrained model not found at {model_path}")
        print(f"Train first: python reproduction/train_dqn.py --env {args.env}")
        sys.exit(1)

    needs_ref = (run_all or args.only in ("bootstrap", "feature_noise"))
    if needs_ref and not os.path.exists(ref_replay_path):
        print(f"ERROR: Reference replay not found at {ref_replay_path}")
        print(f"Collect first: python reproduction/collect_replay.py --env {args.env} "
              f"--seed {args.ref_seed}")
        sys.exit(1)

    # ── Load reference replay (needed for bootstrap & noise) ──
    ref_data = None
    if needs_ref:
        print(f"  Loading reference replay from {ref_replay_path} ...")
        ref_data = load_replay_npz(ref_replay_path)
        _print_data_summary(ref_data, "reference")
        print()

    all_created: list[str] = []

    # ── Seed shift ──
    if run_all or args.only == "seed_shift":
        print(f"  ▸ Seed Shift ({len(seed_shift_seeds)} seeds)")
        out_dir = os.path.join(args.output_base, tag, "seed_shift")
        paths = generate_seed_shift(
            env_name=args.env,
            model_path=model_path,
            n_transitions=n_transitions,
            seeds=seed_shift_seeds,
            output_dir=out_dir,
        )
        all_created.extend(paths)
        print(f"    ✓ {len(paths)} seed-shift datasets saved to {out_dir}/\n")

    # ── Bootstrap resampling ──
    if run_all or args.only == "bootstrap":
        print(f"  ▸ Bootstrap Resampling (B={n_bootstraps}, "
              f"fraction={bootstrap_fraction})")
        out_dir = os.path.join(args.output_base, tag, "bootstrap")
        paths = generate_bootstrap(
            ref_data=ref_data,
            env_name=args.env,
            n_subsets=n_bootstraps,
            fraction=bootstrap_fraction,
            seed=cfg["global"]["random_seed"],
            output_dir=out_dir,
        )
        all_created.extend(paths)
        print(f"    ✓ {len(paths)} bootstrap datasets saved to {out_dir}/\n")

    # ── Feature noise ──
    if run_all or args.only == "feature_noise":
        print(f"  ▸ Feature Noise (levels={noise_levels})")
        out_dir = os.path.join(args.output_base, tag, "feature_noise")
        paths = generate_feature_noise(
            ref_data=ref_data,
            env_name=args.env,
            noise_levels=noise_levels,
            seed=cfg["global"]["random_seed"],
            output_dir=out_dir,
        )
        all_created.extend(paths)
        print(f"    ✓ {len(paths)} noisy datasets saved to {out_dir}/\n")

    # ── Summary ──
    print(f"{'='*64}")
    print(f"  ✓ All done!  {len(all_created)} perturbed dataset(s) generated.")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
