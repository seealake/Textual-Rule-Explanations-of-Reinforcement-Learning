#!/usr/bin/env python
"""
Collect replay data from pretrained DQN agents.

Usage:
    # Collect replay for MountainCar-v0 (default)
    python reproduction/collect_replay.py

    # Collect replay for CartPole-v1
    python reproduction/collect_replay.py --env CartPole-v1

    # Collect with custom number of transitions
    python reproduction/collect_replay.py --env MountainCar-v0 --num-transitions 10000

    # Collect with a specific seed
    python reproduction/collect_replay.py --env CartPole-v1 --seed 42

    # Collect multiple replay datasets with different seeds (seed shift)
    python reproduction/collect_replay.py --env MountainCar-v0 --seeds 0 1 2 3 4

The script:
  1. Loads a pretrained DQN model from reproduction/models/.
  2. Rolls out the policy in the environment, collecting (state, action) pairs.
  3. Saves the replay dataset as both .npz (for fast loading) and .csv (for inspection).
  4. Prints summary statistics for verification.
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO


# ── Feature names for each environment ────────────────────────────────
# These are used as column headers in the CSV output, making the data
# self-documenting for the CBS pipeline.
ENV_FEATURE_NAMES = {
    "MountainCar-v0": ["car_position", "car_velocity"],
    "CartPole-v1": ["cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity"],
    "LunarLander-v3": [
        "x_position", "y_position", "x_velocity", "y_velocity",
        "angle", "angular_velocity", "left_leg_contact", "right_leg_contact",
    ],
    "MiniGrid-Dynamic-Obstacles-8x8-v0": [
        "agent_x", "agent_y", "agent_dir", "goal_dx", "goal_dy", "dist_to_goal",
        "front_is_free", "front_is_obstacle", "left_free", "right_free",
        "nearest_obs_dist", "nearest_obs_dx", "nearest_obs_dy", "num_obstacles_visible",
    ],
}

# Action names for each environment (for human-readable output)
ENV_ACTION_NAMES = {
    "MountainCar-v0": {0: "push_left", 1: "no_push", 2: "push_right"},
    "CartPole-v1": {0: "push_left", 1: "push_right"},
    "LunarLander-v3": {0: "noop", 1: "fire_left", 2: "fire_main", 3: "fire_right"},
    "MiniGrid-Dynamic-Obstacles-8x8-v0": {0: "turn_left", 1: "turn_right", 2: "forward"},
}


def collect_replay(
    env_name: str,
    model_path: str,
    num_transitions: int = 10000,
    seed: int = 42,
    deterministic: bool = True,
) -> dict:
    """
    Roll out a pretrained DQN and collect (state, action) pairs.

    Args:
        env_name: Gymnasium environment name.
        model_path: Path to the saved DQN model (.zip).
        num_transitions: Target number of transitions to collect.
        seed: Environment seed for reproducibility.
        deterministic: If True, use greedy actions (no exploration).

    Returns:
        Dictionary with keys:
            - 'states': np.ndarray of shape (N, obs_dim)
            - 'actions': np.ndarray of shape (N,), dtype int
            - 'rewards': np.ndarray of shape (N,)
            - 'dones': np.ndarray of shape (N,), dtype bool
            - 'episode_ids': np.ndarray of shape (N,), dtype int
            - 'num_episodes': int
            - 'total_reward': float (sum of rewards across all episodes)
    """
    # Create env (with wrapper if needed)
    if "MiniGrid" in env_name:
        from reproduction.minigrid_feature_wrapper import make_minigrid_env
        env = make_minigrid_env(env_name)
    else:
        env = gym.make(env_name)

    # Load model (detect algo from filename)
    if "ppo_" in os.path.basename(model_path).lower():
        model = PPO.load(model_path)
    else:
        model = DQN.load(model_path)

    states = []
    actions = []
    rewards = []
    dones = []
    episode_ids = []

    episode_count = 0
    total_reward = 0.0
    collected = 0

    obs, info = env.reset(seed=seed)

    while collected < num_transitions:
        # Get action from pretrained policy
        action, _ = model.predict(obs, deterministic=deterministic)
        action = int(action)

        # Record this transition
        states.append(obs.copy())
        actions.append(action)

        # Step the environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rewards.append(reward)
        dones.append(done)
        episode_ids.append(episode_count)
        total_reward += reward
        collected += 1

        if done:
            episode_count += 1
            # Reset with a new seed derived from the base seed + episode count
            # This ensures different episodes see slightly different initial states
            # while remaining reproducible
            obs, info = env.reset(seed=seed + episode_count)
        else:
            obs = next_obs

    env.close()

    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int32),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=bool),
        "episode_ids": np.array(episode_ids, dtype=np.int32),
        "num_episodes": episode_count + (1 if not dones[-1] else 0),
        "total_reward": total_reward,
    }


def save_replay(
    data: dict,
    env_name: str,
    seed: int,
    output_dir: str = "reproduction/data",
    algo: str = None,
) -> tuple:
    """
    Save replay data in both .npz (fast loading) and .csv (human-readable) formats.

    Returns:
        (npz_path, csv_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    # File naming: e.g., replay_mountaincar_v0_seed42 or replay_minigrid_..._ppo_seed42
    env_tag = env_name.replace("-", "_").lower()
    if algo:
        base_name = f"replay_{env_tag}_{algo}_seed{seed}"
    else:
        base_name = f"replay_{env_tag}_seed{seed}"

    # ── Save as .npz ──
    npz_path = os.path.join(output_dir, base_name + ".npz")
    np.savez_compressed(
        npz_path,
        states=data["states"],
        actions=data["actions"],
        rewards=data["rewards"],
        dones=data["dones"],
        episode_ids=data["episode_ids"],
    )

    # ── Save as .csv ──
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


def print_summary(data: dict, env_name: str, seed: int):
    """Print a summary of the collected replay data."""
    n = len(data["actions"])
    n_episodes = data["num_episodes"]
    obs_dim = data["states"].shape[1]

    feature_names = ENV_FEATURE_NAMES.get(env_name, [f"f{i}" for i in range(obs_dim)])
    action_names = ENV_ACTION_NAMES.get(env_name, {})

    print(f"\n{'─'*60}")
    print(f"  Replay Summary: {env_name} (seed={seed})")
    print(f"{'─'*60}")
    print(f"  Transitions collected: {n:,}")
    print(f"  Episodes completed:    {n_episodes}")
    print(f"  Avg episode length:    {n / max(n_episodes, 1):.1f}")
    print(f"  Total reward:          {data['total_reward']:.1f}")
    print(f"  Avg reward/episode:    {data['total_reward'] / max(n_episodes, 1):.1f}")

    # Action distribution
    print(f"\n  Action distribution:")
    unique, counts = np.unique(data["actions"], return_counts=True)
    for a, c in zip(unique, counts):
        name = action_names.get(a, f"action_{a}")
        print(f"    {name} (a={a}): {c:,} ({100*c/n:.1f}%)")

    # Feature statistics
    print(f"\n  Feature statistics:")
    print(f"    {'Feature':<30s} {'Min':>10s} {'Max':>10s} {'Mean':>10s} {'Std':>10s}")
    for i, fname in enumerate(feature_names):
        col = data["states"][:, i]
        print(f"    {fname:<30s} {col.min():10.4f} {col.max():10.4f} "
              f"{col.mean():10.4f} {col.std():10.4f}")

    print(f"{'─'*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Collect replay data from pretrained DQN agents"
    )
    parser.add_argument(
        "--env", type=str, default="MountainCar-v0",
        help="Gymnasium environment name",
    )
    parser.add_argument(
        "--num-transitions", type=int, default=10000,
        help="Number of transitions to collect per dataset (default: 10000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Environment seed for single-seed collection (ignored if --seeds is used)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Collect multiple datasets with these seeds (e.g., --seeds 0 1 2 3 4)",
    )
    parser.add_argument(
        "--model-dir", type=str, default="reproduction/models",
        help="Directory containing pretrained models",
    )
    parser.add_argument(
        "--output-dir", type=str, default="reproduction/data",
        help="Directory to save replay datasets",
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic (epsilon-greedy) actions instead of deterministic",
    )
    parser.add_argument(
        "--algo", type=str, default=None,
        help="Algorithm name (dqn or ppo). Auto-detected if not specified.",
    )
    args = parser.parse_args()

    # Locate model — try algo-specific name first, then fallback
    env_tag = args.env.replace("-", "_").lower()
    algo = args.algo
    if algo is None:
        # Auto-detect: check if ppo model exists, else dqn
        ppo_path = os.path.join(args.model_dir, f"ppo_{env_tag}.zip")
        dqn_path = os.path.join(args.model_dir, f"dqn_{env_tag}.zip")
        if os.path.exists(ppo_path):
            algo = "ppo"
        elif os.path.exists(dqn_path):
            algo = "dqn"
        else:
            print(f"ERROR: No model found for {args.env} in {args.model_dir}")
            sys.exit(1)
    model_path = os.path.join(args.model_dir, f"{algo}_{env_tag}.zip")
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    # Determine seeds to collect
    seeds = args.seeds if args.seeds is not None else [args.seed]

    print(f"\n{'='*60}")
    print(f"  Replay Data Collection")
    print(f"  Environment:  {args.env}")
    print(f"  Model:        {model_path}")
    print(f"  Transitions:  {args.num_transitions:,} per dataset")
    print(f"  Seeds:        {seeds}")
    print(f"  Deterministic: {not args.stochastic}")
    print(f"  Output:       {args.output_dir}/")
    print(f"{'='*60}")

    for seed in seeds:
        print(f"\n  Collecting with seed={seed}...")

        data = collect_replay(
            env_name=args.env,
            model_path=model_path,
            num_transitions=args.num_transitions,
            seed=seed,
            deterministic=not args.stochastic,
        )

        npz_path, csv_path = save_replay(
            data=data,
            env_name=args.env,
            seed=seed,
            output_dir=args.output_dir,
            algo=algo,
        )

        print_summary(data, args.env, seed)
        print(f"  ✓ Saved: {npz_path}")
        print(f"  ✓ Saved: {csv_path}")

    print(f"\n{'='*60}")
    print(f"  ✓ All done! {len(seeds)} dataset(s) collected for {args.env}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
