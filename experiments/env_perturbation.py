#!/usr/bin/env python
"""
Environment perturbation helpers

Creates Gymnasium environments with modified physics parameters and
collects replay data from fixed DQN policies in those environments.

MountainCar-v0: gravity and force as instance attributes on env.unwrapped
LunarLander-v3: gravity/wind/turbulence as constructor kwargs to gym.make()
"""
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN


# ── Perturbation configurations ──────────────────────────────────────

MC_DEFAULTS = {"gravity": 0.0025, "force": 0.001}

MC_PERTURBATIONS = [
    {"name": "clean",           "params": {},                           "type": "clean",   "severity": 0.0},
    # Gravity perturbations (stronger gravity = harder to climb)
    {"name": "gravity_small",   "params": {"gravity": 0.00275},         "type": "gravity", "severity": 0.10},
    {"name": "gravity_medium",  "params": {"gravity": 0.003125},        "type": "gravity", "severity": 0.25},
    {"name": "gravity_large",   "params": {"gravity": 0.00375},         "type": "gravity", "severity": 0.50},
    # Force perturbations (weaker force = harder to accelerate)
    {"name": "force_small",     "params": {"force": 0.0009},            "type": "force",   "severity": 0.10},
    {"name": "force_medium",    "params": {"force": 0.00075},           "type": "force",   "severity": 0.25},
    {"name": "force_large",     "params": {"force": 0.0006},            "type": "force",   "severity": 0.40},
]

LL_DEFAULTS = {"gravity": -10.0, "enable_wind": False, "wind_power": 15.0, "turbulence_power": 1.5}

LL_PERTURBATIONS = [
    {"name": "clean",           "params": {},                                                              "type": "clean",      "severity": 0.0},
    # Gravity perturbations (stronger = harder landing; gymnasium constraint: -12.0 < gravity < 0.0)
    {"name": "gravity_small",   "params": {"gravity": -10.5},                                              "type": "gravity",    "severity": 0.05},
    {"name": "gravity_medium",  "params": {"gravity": -11.0},                                              "type": "gravity",    "severity": 0.10},
    {"name": "gravity_large",   "params": {"gravity": -11.9},                                              "type": "gravity",    "severity": 0.19},
    # Wind perturbations
    {"name": "wind_small",      "params": {"enable_wind": True, "wind_power": 3.0},                       "type": "wind",       "severity": 0.15},
    {"name": "wind_medium",     "params": {"enable_wind": True, "wind_power": 7.0},                       "type": "wind",       "severity": 0.35},
    {"name": "wind_large",      "params": {"enable_wind": True, "wind_power": 12.0},                      "type": "wind",       "severity": 0.60},
    # Turbulence perturbations (with light wind baseline)
    {"name": "turb_small",      "params": {"enable_wind": True, "wind_power": 5.0, "turbulence_power": 0.5},  "type": "turbulence", "severity": 0.10},
    {"name": "turb_medium",     "params": {"enable_wind": True, "wind_power": 5.0, "turbulence_power": 1.5},  "type": "turbulence", "severity": 0.25},
    {"name": "turb_large",      "params": {"enable_wind": True, "wind_power": 5.0, "turbulence_power": 3.0},  "type": "turbulence", "severity": 0.50},
]

ENV_PERTURBATIONS = {
    "MountainCar-v0": MC_PERTURBATIONS,
    "LunarLander-v3": LL_PERTURBATIONS,
}


def make_perturbed_env(env_name, perturbation_params):
    """Create a Gymnasium environment with physics perturbations.

    For MountainCar: sets instance attributes after construction.
    For LunarLander: passes kwargs to gym.make().
    """
    if env_name == "MountainCar-v0":
        env = gym.make(env_name)
        inner = env.unwrapped
        if "gravity" in perturbation_params:
            inner.gravity = perturbation_params["gravity"]
        if "force" in perturbation_params:
            inner.force = perturbation_params["force"]
        return env

    elif env_name == "LunarLander-v3":
        kwargs = {}
        for k in ["gravity", "enable_wind", "wind_power", "turbulence_power"]:
            if k in perturbation_params:
                kwargs[k] = perturbation_params[k]
        return gym.make(env_name, **kwargs)

    else:
        raise ValueError(f"Environment perturbation not defined for {env_name}")


def collect_replay_perturbed(env_name, model_path, perturbation_params,
                             num_transitions=10000, seed=42, deterministic=True):
    """Collect replay data from a perturbed environment using the fixed DQN policy.

    Returns dict with keys: states, actions, rewards, dones, episode_ids,
    num_episodes, total_reward, mean_episode_reward.
    """
    env = make_perturbed_env(env_name, perturbation_params)
    model = DQN.load(model_path)

    states, actions, rewards, dones, episode_ids = [], [], [], [], []
    episode_count = 0
    total_reward = 0.0
    episode_rewards = []
    current_episode_reward = 0.0
    collected = 0

    obs, info = env.reset(seed=seed)

    while collected < num_transitions:
        action, _ = model.predict(obs, deterministic=deterministic)
        action = int(action)

        states.append(obs.copy())
        actions.append(action)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rewards.append(reward)
        dones.append(done)
        episode_ids.append(episode_count)
        total_reward += reward
        current_episode_reward += reward
        collected += 1

        if done:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0.0
            episode_count += 1
            obs, info = env.reset(seed=seed + episode_count)
        else:
            obs = next_obs

    # Handle incomplete last episode
    if current_episode_reward != 0.0:
        episode_rewards.append(current_episode_reward)

    env.close()

    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int32),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=bool),
        "episode_ids": np.array(episode_ids, dtype=np.int32),
        "num_episodes": episode_count + (1 if not dones[-1] else 0),
        "total_reward": total_reward,
        "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "episode_rewards": episode_rewards,
    }


def evaluate_policy_return(model_path, env_name, perturbation_params,
                           eval_seeds, deterministic=True):
    """Evaluate DQN policy return in a perturbed environment.

    Returns dict with mean_return, std_return, per_episode_returns.
    """
    model = DQN.load(model_path)
    episode_returns = []

    for ep_seed in eval_seeds:
        env = make_perturbed_env(env_name, perturbation_params)
        obs, info = env.reset(seed=ep_seed)
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or truncated

        episode_returns.append(total_reward)
        env.close()

    returns_arr = np.array(episode_returns)
    return {
        "mean_return": float(np.mean(returns_arr)),
        "std_return": float(np.std(returns_arr)),
        "per_episode_returns": [float(r) for r in returns_arr],
        "n_episodes": len(eval_seeds),
    }
