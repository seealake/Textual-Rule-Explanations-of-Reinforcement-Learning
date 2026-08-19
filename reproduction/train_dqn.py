#!/usr/bin/env python
"""
Train DQN agents using Stable-Baselines3.

Usage:
    # Train MountainCar-v0 (default)
    python reproduction/train_dqn.py

    # Train CartPole-v1
    python reproduction/train_dqn.py --env CartPole-v1

    # Train with custom timesteps
    python reproduction/train_dqn.py --env MountainCar-v0 --timesteps 200000

The script:
  1. Reads hyperparameters from experiment configs (with env-specific overrides).
  2. Trains a DQN agent with Stable-Baselines3.
  3. Saves the model to reproduction/models/<env_name>.zip
  4. Runs a quick evaluation (10 episodes) to verify performance.
"""

import argparse
import os
import sys

import torch

# Add project root to path so we can import experiments.config_loader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback


# ── Environment-specific DQN hyperparameters ──────────────────────────
# MountainCar-v0 is notoriously hard for vanilla DQN due to sparse rewards.
# We use tuned hyperparameters from the RL Baselines3 Zoo / known good configs.
ENV_HYPERPARAMS = {
    "MountainCar-v0": dict(
        learning_rate=1e-3,
        buffer_size=50000,
        batch_size=128,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        train_freq=4,
        gradient_steps=8,
        target_update_interval=500,
        learning_starts=2000,
        policy_kwargs=dict(net_arch=[256, 256]),
    ),
    "CartPole-v1": dict(
        learning_rate=2.3e-3,
        buffer_size=100000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.16,
        exploration_final_eps=0.04,
        train_freq=256,
        gradient_steps=128,
        target_update_interval=10,
        learning_starts=1000,
        policy_kwargs=dict(net_arch=[256, 256]),
    ),
    "LunarLander-v3": dict(
        learning_rate=6.3e-4,
        buffer_size=50000,
        batch_size=128,
        gamma=0.99,
        exploration_fraction=0.12,
        exploration_final_eps=0.1,
        train_freq=4,
        gradient_steps=4,
        target_update_interval=250,
        learning_starts=0,
        policy_kwargs=dict(net_arch=[256, 256]),
    ),
}

# Default timesteps per environment
DEFAULT_TIMESTEPS = {
    "MountainCar-v0": 300000,
    "CartPole-v1": 100000,
    "LunarLander-v3": 200000,
}

# "Solved" thresholds — what average reward indicates success
SOLVE_THRESHOLDS = {
    "MountainCar-v0": -150.0,   # reaching goal in <150 steps on average
    "CartPole-v1": 475.0,        # near max of 500
    "LunarLander-v3": 200.0,     # standard solved threshold
}


def train_dqn(env_name: str, total_timesteps: int, seed: int = 42,
              model_dir: str = "reproduction/models",
              tb_log: str = "reproduction/tb_logs",
              verbose: int = 1,
              device: str = "auto") -> str:
    """
    Train a DQN agent and save the model.

    Returns:
        Path to the saved model file.
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"dqn_{env_name.replace('-', '_').lower()}")

    print(f"\n{'='*60}")
    print(f"  Training DQN on {env_name}")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"  Seed: {seed}")
    print(f"  Output: {model_path}.zip")
    print(f"  Device: {device}")
    if tb_log:
        print(f"  TensorBoard: tensorboard --logdir {tb_log}")
    print(f"{'='*60}\n")

    # Create training environment
    env = gym.make(env_name)

    # Get environment-specific hyperparameters (fall back to empty dict)
    hp = ENV_HYPERPARAMS.get(env_name, {})

    # Create DQN model
    model = DQN(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        verbose=verbose,
        device=device,
        tensorboard_log=tb_log if tb_log else None,
        **hp,
    )

    # Optional: create eval callback to monitor progress
    eval_env = gym.make(env_name)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,  # we save manually
        eval_freq=max(total_timesteps // 20, 1000),
        n_eval_episodes=5,
        verbose=verbose,
    )

    # Train!
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    # Save model
    model.save(model_path)
    print(f"\nModel saved to {model_path}.zip")

    # Clean up
    env.close()
    eval_env.close()

    return model_path + ".zip"


def evaluate_model(model_path: str, env_name: str, n_episodes: int = 20,
                   seed: int = 42,
                   device: str = "auto") -> tuple:
    """
    Load and evaluate a trained DQN model.

    Returns:
        (mean_reward, std_reward)
    """
    print(f"\n{'='*60}")
    print(f"  Evaluating {model_path}")
    print(f"  Environment: {env_name}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    env = gym.make(env_name)
    model = DQN.load(model_path, env=env, device=device)

    # Evaluate
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_episodes, deterministic=True
    )

    # Check against solved threshold
    threshold = SOLVE_THRESHOLDS.get(env_name)
    solved = False
    if threshold is not None:
        if env_name == "MountainCar-v0":
            # MountainCar: higher (less negative) is better
            solved = mean_reward >= threshold
        else:
            solved = mean_reward >= threshold

    print(f"\n  Results:")
    print(f"    Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    if threshold is not None:
        status = "✓ SOLVED" if solved else "✗ NOT SOLVED"
        print(f"    Threshold:   {threshold:.1f}")
        print(f"    Status:      {status}")

    # Also print episode-level details
    print(f"\n  Running {n_episodes} individual episodes for detailed view...")
    episode_rewards = []
    episode_lengths = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)
        total_reward = 0
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    print(f"    Episode rewards: min={min(episode_rewards):.0f}, "
          f"max={max(episode_rewards):.0f}, "
          f"mean={np.mean(episode_rewards):.1f}")
    print(f"    Episode lengths: min={min(episode_lengths)}, "
          f"max={max(episode_lengths)}, "
          f"mean={np.mean(episode_lengths):.1f}")

    env.close()
    return mean_reward, std_reward


def main():
    parser = argparse.ArgumentParser(description="Train DQN agent for RL environment")
    parser.add_argument("--env", type=str, default="MountainCar-v0",
                        help="Gymnasium environment name")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total training timesteps (default: env-specific)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--model-dir", type=str, default="reproduction/models",
                        help="Directory to save trained models")
    parser.add_argument("--tb-log", type=str, default="reproduction/tb_logs",
                        help="TensorBoard log directory (set to '' to disable)")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="Number of episodes for evaluation")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate existing model")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Training/eval device: auto prefers CUDA when available")
    args = parser.parse_args()

    # Auto policy: prefer CUDA when available, otherwise CPU.
    if args.device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        selected_device = args.device

    if selected_device == "cuda" and not torch.cuda.is_available():
        print("ERROR: --device cuda was requested, but CUDA is not available in this environment.")
        print("       Use --device auto or --device cpu, or install CUDA-enabled PyTorch.")
        sys.exit(1)

    print(f"Selected device: {selected_device}")
    print(f"PyTorch: {torch.__version__} (CUDA runtime: {torch.version.cuda})")
    if selected_device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Determine timesteps
    timesteps = args.timesteps or DEFAULT_TIMESTEPS.get(args.env, 100000)

    model_path = os.path.join(
        args.model_dir,
        f"dqn_{args.env.replace('-', '_').lower()}.zip"
    )

    if not args.eval_only:
        # Train
        model_path = train_dqn(
            env_name=args.env,
            total_timesteps=timesteps,
            seed=args.seed,
            model_dir=args.model_dir,
            tb_log=args.tb_log,
            device=selected_device,
        )
    else:
        if not os.path.exists(model_path):
            print(f"ERROR: Model not found at {model_path}. Train first (remove --eval-only).")
            sys.exit(1)

    # Evaluate
    mean_reward, std_reward = evaluate_model(
        model_path=model_path,
        env_name=args.env,
        n_episodes=args.eval_episodes,
        seed=args.seed,
        device=selected_device,
    )

    # Final verdict
    threshold = SOLVE_THRESHOLDS.get(args.env)
    if threshold is not None:
        if args.env == "MountainCar-v0":
            solved = mean_reward >= threshold
        else:
            solved = mean_reward >= threshold

        if solved:
            print(f"\n{'='*60}")
            print(f"  ✓ {args.env} is SOLVED!")
            print(f"  Mean reward {mean_reward:.1f} >= threshold {threshold:.1f}")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"  ✗ {args.env} is NOT solved yet.")
            print(f"  Mean reward {mean_reward:.1f} < threshold {threshold:.1f}")
            print(f"  Consider increasing timesteps or tuning hyperparameters.")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()
