#!/usr/bin/env python
"""
Train RL agents using Stable-Baselines3 (DQN or PPO).

Usage:
    # Train PPO on MiniGrid
    python reproduction/train_policy.py --env MiniGrid-Dynamic-Obstacles-8x8-v0 --algo ppo

    # Train DQN on CartPole (same as train_dqn.py)
    python reproduction/train_policy.py --env CartPole-v1 --algo dqn

    # Train PPO on CartPole (for PPO vs DQN comparison)
    python reproduction/train_policy.py --env CartPole-v1 --algo ppo

    # Custom timesteps and seed
    python reproduction/train_policy.py --env MiniGrid-Dynamic-Obstacles-8x8-v0 --algo ppo --timesteps 500000 --seed 0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback


# ── PPO hyperparameters per environment ───────────────────────────────
PPO_HYPERPARAMS = {
    "MiniGrid-Dynamic-Obstacles-8x8-v0": dict(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        n_epochs=10,
        policy_kwargs=dict(net_arch=[128, 128]),
    ),
    "CartPole-v1": dict(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        n_epochs=10,
        policy_kwargs=dict(net_arch=[128, 128]),
    ),
    "LunarLander-v3": dict(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        n_epochs=10,
        policy_kwargs=dict(net_arch=[256, 256]),
    ),
}

PPO_DEFAULT_TIMESTEPS = {
    "MiniGrid-Dynamic-Obstacles-8x8-v0": 1_000_000,
    "CartPole-v1": 100_000,
    "LunarLander-v3": 500_000,
}

# ── DQN hyperparameters — imported from train_dqn.py ──────────────────
# We re-import at call time to avoid circular dependencies.

SOLVE_THRESHOLDS = {
    "MountainCar-v0": -150.0,
    "CartPole-v1": 475.0,
    "LunarLander-v3": 200.0,
    "MiniGrid-Dynamic-Obstacles-8x8-v0": 0.5,  # success rate proxy
}


def _is_minigrid(env_name):
    return "MiniGrid" in env_name


def _make_env(env_name):
    """Create environment, applying feature wrapper if needed."""
    if _is_minigrid(env_name):
        from reproduction.minigrid_feature_wrapper import make_minigrid_env
        return make_minigrid_env(env_name)
    else:
        return gym.make(env_name)


def train_policy(env_name: str, algo: str = "ppo",
                 total_timesteps: int = None, seed: int = 42,
                 model_dir: str = "reproduction/models",
                 tb_log: str = "reproduction/tb_logs",
                 verbose: int = 1, device: str = "auto") -> str:
    """Train a policy (DQN or PPO) and save the model."""
    os.makedirs(model_dir, exist_ok=True)
    algo_lower = algo.lower()
    env_tag = env_name.replace("-", "_").lower()
    model_path = os.path.join(model_dir, f"{algo_lower}_{env_tag}")

    # Default timesteps
    if total_timesteps is None:
        if algo_lower == "ppo":
            total_timesteps = PPO_DEFAULT_TIMESTEPS.get(env_name, 500_000)
        else:
            from reproduction.train_dqn import DEFAULT_TIMESTEPS
            total_timesteps = DEFAULT_TIMESTEPS.get(env_name, 100_000)

    print(f"\n{'='*60}")
    print(f"  Training {algo.upper()} on {env_name}")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"  Seed: {seed}")
    print(f"  Output: {model_path}.zip")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    env = _make_env(env_name)
    eval_env = _make_env(env_name)

    if algo_lower == "ppo":
        hp = PPO_HYPERPARAMS.get(env_name, {})
        model = PPO(
            policy="MlpPolicy", env=env, seed=seed,
            verbose=verbose, device=device,
            tensorboard_log=tb_log if tb_log else None,
            **hp,
        )
    elif algo_lower == "dqn":
        from reproduction.train_dqn import ENV_HYPERPARAMS
        hp = ENV_HYPERPARAMS.get(env_name, {})
        model = DQN(
            policy="MlpPolicy", env=env, seed=seed,
            verbose=verbose, device=device,
            tensorboard_log=tb_log if tb_log else None,
            **hp,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=None,
        eval_freq=max(total_timesteps // 20, 1000),
        n_eval_episodes=10, verbose=verbose,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    model.save(model_path)
    print(f"\nModel saved to {model_path}.zip")

    env.close()
    eval_env.close()
    return model_path + ".zip"


def evaluate_policy_model(model_path: str, env_name: str, algo: str = "ppo",
                          n_episodes: int = 100, seed: int = 42,
                          device: str = "auto") -> dict:
    """Evaluate a trained model."""
    env = _make_env(env_name)

    AlgoClass = PPO if algo.lower() == "ppo" else DQN
    model = AlgoClass.load(model_path, env=env, device=device)

    rewards = []
    lengths = []
    successes = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        ep_len = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            ep_len += 1
            done = terminated or truncated
        rewards.append(ep_reward)
        lengths.append(ep_len)
        if ep_reward > 0:
            successes += 1

    env.close()

    result = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": successes / n_episodes,
        "n_episodes": n_episodes,
    }

    print(f"\n{'='*60}")
    print(f"  Evaluation: {algo.upper()} on {env_name}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Mean reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
    print(f"  Min/Max: {result['min_reward']:.2f} / {result['max_reward']:.2f}")
    print(f"  Mean length: {result['mean_length']:.1f}")
    print(f"  Success rate: {result['success_rate']:.2%}")
    print(f"{'='*60}\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="Train RL policy (DQN/PPO)")
    parser.add_argument("--env", type=str, default="MiniGrid-Dynamic-Obstacles-8x8-v0")
    parser.add_argument("--algo", type=str, default="ppo", choices=["dqn", "ppo"])
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model-dir", type=str, default="reproduction/models")
    args = parser.parse_args()

    model_path = train_policy(
        env_name=args.env, algo=args.algo,
        total_timesteps=args.timesteps, seed=args.seed,
        model_dir=args.model_dir, device=args.device,
    )

    result = evaluate_policy_model(
        model_path=model_path, env_name=args.env, algo=args.algo,
        n_episodes=args.eval_episodes, seed=args.seed, device=args.device,
    )

    # Save training result
    import json
    os.makedirs("experiments/results", exist_ok=True)
    env_tag = args.env.replace("-", "_").lower()
    result_path = f"experiments/results/{env_tag}_training_{args.algo}.json"
    result["env"] = args.env
    result["algo"] = args.algo
    result["seed"] = args.seed
    result["timesteps"] = args.timesteps or PPO_DEFAULT_TIMESTEPS.get(args.env, 500_000)
    result["model_path"] = model_path
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Training results saved to {result_path}")


if __name__ == "__main__":
    main()
