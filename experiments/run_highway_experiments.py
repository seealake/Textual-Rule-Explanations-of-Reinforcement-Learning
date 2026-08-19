#!/usr/bin/env python
"""
Highway-Env Experiment Pipeline
================================
Complete experiment pipeline for highway-env environments (merge-v0, intersection-v0).

Covers:
    1. Policy training (DQN + PPO, 3 seeds each)
  2. Replay data collection (100 episodes per seed)
    3. Explanation methods (CBS, rule-set voting, DT, BDR) x 10 explainer seeds
  4. Noise severity sweep
  5. vehicles_count ablation (4, 6, 8)
  6. Feature mode ablation (raw-only vs raw+derived)
  7. Behavioral evaluation (rule-policy rollout)
  8. Statistical analysis
  9. Table and figure generation

Usage:
    python experiments/run_highway_experiments.py --phase train
    python experiments/run_highway_experiments.py --phase train --train-workers 2 --worker-torch-threads 12
    python experiments/run_highway_experiments.py --phase replay
    python experiments/run_highway_experiments.py --phase explain
    python experiments/run_highway_experiments.py --phase noise
    python experiments/run_highway_experiments.py --phase vehicles_ablation
    python experiments/run_highway_experiments.py --phase feature_ablation
    python experiments/run_highway_experiments.py --phase behavior_eval
    python experiments/run_highway_experiments.py --phase statistics
    python experiments/run_highway_experiments.py --phase tables
    python experiments/run_highway_experiments.py --phase figures
    python experiments/run_highway_experiments.py --phase all
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from reproduction.highway_env_wrapper import (
    make_highway_env,
    get_feature_names,
    flatten_obs_raw,
    flatten_obs_full,
    get_env_config,
    HIGHWAY_ACTION_NAMES,
    N_ACTIONS,
)

# ── Constants ─────────────────────────────────────────────────────────
ENVS = ["merge-v0", "intersection-v0"]
ALGOS = ["dqn", "ppo"]
POLICY_SEEDS = list(range(3))  # 0..2 for the current highway study
EXPLAINER_SEEDS = list(range(10))  # 0..9
N_ROLLOUT_EPISODES = 100
EVAL_EPISODES = 50
EVAL_SEEDS = list(range(2000, 2050))
VEHICLES_COUNT_DEFAULT = 6
VEHICLES_COUNTS_ABLATION = [4, 6, 8]
FEATURE_MODES = ["raw", "raw_derived"]
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10]
REPLAY_SPLITS = ("train_explainer", "val_explainer", "test_explainer")

# Explanation methods
METHODS = ["cbs", "b3_vote", "dt", "b5_bdr"]

# Training hyperparameters — step counts per user specification:
#   merge-v0:        DQN 500k,  PPO 1M
#   intersection-v0: DQN 700k,  PPO 1M
TRAIN_CONFIG = {
    "merge-v0": {
        "dqn": dict(
            total_timesteps=500_000,
            learning_rate=5e-4,
            buffer_size=50_000,
            batch_size=128,
            gamma=0.99,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=500,
            learning_starts=1_000,
            policy_kwargs=dict(net_arch=[256, 256]),
        ),
        "ppo": dict(
            total_timesteps=1_000_000,
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
    },
    "intersection-v0": {
        "dqn": dict(
            total_timesteps=700_000,
            learning_rate=5e-4,
            buffer_size=50_000,
            batch_size=128,
            gamma=0.99,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=500,
            learning_starts=1_000,
            policy_kwargs=dict(net_arch=[256, 256]),
        ),
        "ppo": dict(
            total_timesteps=1_000_000,
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
    },
}


def _artifact_dir() -> Path:
    root = Path(__file__).resolve().parent.parent / "artifacts"
    return root


def _env_algo_dir(env_name: str, algo: str) -> Path:
    env_tag = env_name.replace("-", "_")
    d = _artifact_dir() / f"highway_{env_tag}" / algo
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp_path, path)


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def _torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _sb3_device(algo: str) -> str:
    """Choose a Stable-Baselines3 device for the requested algorithm.

    Defaults:
      - DQN: use CUDA when available.
      - PPO: stay on CPU unless HIGHWAY_FORCE_PPO_CUDA=1, because SB3 MLP PPO
        is often environment-bound and can be slower on GPU.

    Environment overrides:
      - HIGHWAY_USE_CUDA=0 disables CUDA globally.
      - HIGHWAY_FORCE_PPO_CUDA=1 forces PPO onto CUDA when available.
    """
    if os.environ.get("HIGHWAY_USE_CUDA", "1") == "0":
        return "cpu"
    if not _torch_cuda_available():
        return "cpu"
    if algo == "ppo" and os.environ.get("HIGHWAY_FORCE_PPO_CUDA", "0") != "1":
        return "cpu"
    return "cuda"


def _configure_runtime_threads():
    """Optionally cap Torch thread usage for parallel multi-process runs."""
    raw = os.environ.get("HIGHWAY_TORCH_THREADS")
    if not raw:
        return

    try:
        n_threads = max(1, int(raw))
    except ValueError:
        return

    try:
        import torch

        torch.set_num_threads(n_threads)
        interop_threads = max(1, min(4, n_threads))
        torch.set_num_interop_threads(interop_threads)
    except Exception:
        pass


def _cpu_quota_cores() -> int:
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.exists():
        try:
            quota_text, period_text = cpu_max.read_text().strip().split()
            if quota_text != "max":
                quota = int(quota_text)
                period = int(period_text)
                if quota > 0 and period > 0:
                    return max(1, quota // period)
        except Exception:
            pass
    return os.cpu_count() or 1


def _env_tag(env_name: str) -> str:
    return env_name.replace("-", "_")


def _replay_base_name(env_name: str, algo: str, policy_seed: int) -> str:
    return f"{_env_tag(env_name)}_{algo}_policy_seed{policy_seed}"


def _replay_file(env_name: str, algo: str, policy_seed: int) -> Path:
    replay_dir = _env_algo_dir(env_name, algo) / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    return replay_dir / f"{_replay_base_name(env_name, algo, policy_seed)}_replay.npz"


def _replay_metadata_file(env_name: str, algo: str, policy_seed: int) -> Path:
    replay_dir = _env_algo_dir(env_name, algo) / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    return replay_dir / f"{_replay_base_name(env_name, algo, policy_seed)}_metadata.json"


def _replay_split_file(env_name: str, algo: str, policy_seed: int, split_name: str) -> Path:
    replay_dir = _env_algo_dir(env_name, algo) / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    return replay_dir / f"{_replay_base_name(env_name, algo, policy_seed)}_split_{split_name}.npz"


def _save_npz(path: Path, arrays: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **arrays)


def _slice_replay_bundle(bundle: dict, mask: np.ndarray) -> dict:
    sliced = {}
    for key, value in bundle.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == mask.shape:
            sliced[key] = value[mask]
        else:
            sliced[key] = value
    return sliced


def _split_episode_sets(episode_ids, train_frac=0.7, val_frac=0.1, seed=42):
    rng = np.random.RandomState(seed)
    unique_eps = np.unique(episode_ids)
    rng.shuffle(unique_eps)
    n_total = len(unique_eps)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    return {
        "train_explainer": set(unique_eps[:n_train]),
        "val_explainer": set(unique_eps[n_train:n_train + n_val]),
        "test_explainer": set(unique_eps[n_train + n_val:]),
    }


def _save_replay_splits(bundle: dict, env_name: str, algo: str, policy_seed: int):
    split_sets = _split_episode_sets(bundle["episode_ids"])
    for split_name, episode_set in split_sets.items():
        mask = np.isin(bundle["episode_ids"], list(episode_set))
        split_bundle = _slice_replay_bundle(bundle, mask)
        _save_npz(_replay_split_file(env_name, algo, policy_seed, split_name), split_bundle)


def _load_replay_npz(path: Path) -> dict:
    with np.load(str(path)) as data:
        return {key: data[key] for key in data.files}


def _legacy_replay_file(env_name: str, algo: str, policy_seed: int) -> Path:
    replay_dir = _env_algo_dir(env_name, algo) / "replay"
    return replay_dir / f"replay_seed{policy_seed}.npz"


def _training_key(env_name: str, algo: str, policy_seed: int) -> str:
    return f"{env_name}_{algo}_seed{policy_seed}"


def _training_combo_file(env_name: str, algo: str) -> Path:
    return _artifact_dir() / f"training_results_{_env_tag(env_name)}_{algo}.json"


def _policy_model_path(env_name: str, algo: str, policy_seed: int) -> Path:
    return _env_algo_dir(env_name, algo) / "policies" / f"policy_seed{policy_seed}.zip"


def _training_worker_log_file(env_name: str, algo: str, policy_seed: int) -> Path:
    log_dir = _artifact_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"parallel_train_{_env_tag(env_name)}_{algo}_seed{policy_seed}.log"


def _parse_training_key(key: str):
    prefix, seed_text = key.rsplit("_seed", 1)
    env_name, algo = prefix.rsplit("_", 1)
    return env_name, algo, int(seed_text)


def _load_training_seed_result(env_name: str, algo: str, policy_seed: int):
    path = _training_result_seed_file(env_name, algo, policy_seed)
    if not path.exists():
        return None
    return _load_json(path).get(_training_key(env_name, algo, policy_seed))


def _bootstrap_training_seed_files():
    artifact_dir = _artifact_dir()
    legacy_paths = [artifact_dir / "training_results.json"]
    legacy_paths.extend(sorted(artifact_dir.glob("training_results_*.json")))

    seen_paths = set()
    for path in legacy_paths:
        if path in seen_paths or not path.exists() or "_seed" in path.stem:
            continue
        seen_paths.add(path)

        try:
            payload = _load_json(path)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        for key, value in payload.items():
            if not isinstance(value, dict):
                continue

            env_name = value.get("env")
            algo = value.get("algo")
            policy_seed = value.get("seed")

            if env_name is None or algo is None or policy_seed is None:
                try:
                    env_name, algo, policy_seed = _parse_training_key(key)
                except Exception:
                    continue

            seed_path = _training_result_seed_file(env_name, algo, int(policy_seed))
            if seed_path.exists():
                continue

            normalized = dict(value)
            normalized.setdefault("env", env_name)
            normalized.setdefault("algo", algo)
            normalized.setdefault("seed", int(policy_seed))
            _save_json({key: normalized}, seed_path)


def _training_result_seed_file(env_name: str, algo: str, policy_seed: int) -> Path:
    return _artifact_dir() / f"training_results_{_env_tag(env_name)}_{algo}_seed{policy_seed}.json"


def _merge_training_result_files():
    merged = {}
    combo_results = {}
    for path in sorted(_artifact_dir().glob("training_results_*_seed*.json")):
        seed_payload = _load_json(path)
        merged.update(seed_payload)

        for key, value in seed_payload.items():
            env_name = value.get("env")
            algo = value.get("algo")

            if env_name is None or algo is None:
                try:
                    env_name, algo, _ = _parse_training_key(key)
                except Exception:
                    continue

            combo_results.setdefault((env_name, algo), {})[key] = value

    for (env_name, algo), payload in sorted(combo_results.items()):
        _save_json(payload, _training_combo_file(env_name, algo))

    _save_json(merged, _artifact_dir() / "training_results.json")
    return merged


def _evaluate_existing_model_result(env_name: str, algo: str, seed: int, model_path: Path):
    from stable_baselines3 import DQN, PPO

    AlgoClass = DQN if algo == "dqn" else PPO
    model = AlgoClass.load(str(model_path), device=_sb3_device(algo))
    rewards, lengths = _evaluate_model(model, env_name, n_episodes=20, seed=seed)
    return {
        "status": "success",
        "env": env_name,
        "algo": algo,
        "seed": seed,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
        "model_path": str(model_path),
    }


def _train_single_policy(env_name: str, algo: str, seed: int, persist_combined: bool = True):
    key = _training_key(env_name, algo, seed)
    model_path = _policy_model_path(env_name, algo, seed)
    seed_result_path = _training_result_seed_file(env_name, algo, seed)
    existing_seed_result = _load_training_seed_result(env_name, algo, seed)

    if model_path.exists():
        if existing_seed_result is not None:
            print(f"  [SKIP] {key} — already exists")
            if persist_combined:
                _merge_training_result_files()
            return key, existing_seed_result

        print(f"  [RECOVER] {key} — existing model, regenerating evaluation summary")
        try:
            result = _evaluate_existing_model_result(env_name, algo, seed, model_path)
        except Exception as e:
            print(f"  [FAIL] {key}: {e}")
            traceback.print_exc()
            result = {
                "status": "failed",
                "env": env_name,
                "algo": algo,
                "seed": seed,
                "error": str(e),
                "model_path": str(model_path),
            }
        _save_json({key: result}, seed_result_path)
        if persist_combined:
            _merge_training_result_files()
        return key, result

    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import EvalCallback

    print(f"\n{'='*60}")
    print(f"  Training {algo.upper()} on {env_name} seed={seed}")
    print(f"{'='*60}")

    env = None
    eval_env = None
    try:
        out_dir = _env_algo_dir(env_name, algo) / "policies"
        out_dir.mkdir(parents=True, exist_ok=True)
        model_prefix = out_dir / f"policy_seed{seed}"

        env = make_highway_env(env_name, VEHICLES_COUNT_DEFAULT, "raw_derived")
        eval_env = make_highway_env(env_name, VEHICLES_COUNT_DEFAULT, "raw_derived")

        hp = TRAIN_CONFIG[env_name][algo].copy()
        timesteps = hp.pop("total_timesteps")
        AlgoClass = DQN if algo == "dqn" else PPO

        device = _sb3_device(algo)
        model = AlgoClass(
            policy="MlpPolicy", env=env, seed=seed,
            verbose=0, device=device, **hp,
        )

        eval_cb = EvalCallback(
            eval_env, eval_freq=max(timesteps // 20, 1000),
            n_eval_episodes=10, verbose=0,
        )

        model.learn(total_timesteps=timesteps, callback=eval_cb,
                    progress_bar=True)
        model.save(str(model_prefix))

        rewards, lengths = _evaluate_model(model, env_name, n_episodes=20, seed=seed)
        result = {
            "status": "success",
            "env": env_name,
            "algo": algo,
            "seed": seed,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "mean_length": float(np.mean(lengths)),
            "model_path": str(model_prefix) + ".zip",
        }
        print(f"  Result: reward={result['mean_reward']:.2f}±{result['std_reward']:.2f}")
    except Exception as e:
        print(f"  [FAIL] {key}: {e}")
        traceback.print_exc()
        result = {
            "status": "failed",
            "env": env_name,
            "algo": algo,
            "seed": seed,
            "error": str(e),
        }
    finally:
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()

    _save_json({key: result}, seed_result_path)
    if persist_combined:
        _merge_training_result_files()
    return key, result


def _run_parallel_training_child(env_name: str, algo: str, seed: int,
                                 torch_threads_per_worker: int | None = None):
    script_path = Path(__file__).resolve()
    workspace_root = script_path.parent.parent
    log_path = _training_worker_log_file(env_name, algo, seed)
    child_env = os.environ.copy()

    if torch_threads_per_worker is not None:
        thread_cap = str(max(1, int(torch_threads_per_worker)))
        for name in [
            "HIGHWAY_TORCH_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ]:
            child_env[name] = thread_cap

    command = [
        sys.executable,
        "-u",
        str(script_path),
        "--phase", "train",
        "--envs", env_name,
        "--algos", algo,
        "--seeds", str(seed),
        "--train-child",
    ]

    with open(log_path, "w", buffering=1) as log_file:
        log_file.write("COMMAND: " + " ".join(command) + "\n\n")
        completed = subprocess.run(
            command,
            cwd=str(workspace_root),
            env=child_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    key = _training_key(env_name, algo, seed)
    result = _load_training_seed_result(env_name, algo, seed)
    if result is None:
        result = {
            "status": "failed",
            "env": env_name,
            "algo": algo,
            "seed": seed,
            "error": (
                f"parallel worker exited with code {completed.returncode} "
                f"without writing { _training_result_seed_file(env_name, algo, seed).name }"
            ),
        }

    result["worker_log_path"] = str(log_path)
    result["worker_exit_code"] = int(completed.returncode)
    _save_json({key: result}, _training_result_seed_file(env_name, algo, seed))
    return key, result


def launch_parallel_training(envs=None, algos=None, seeds=None,
                             max_workers: int = 2,
                             torch_threads_per_worker: int | None = None):
    """Train seeds in parallel with bounded worker and thread counts."""
    envs = envs or ENVS
    algos = algos or ALGOS
    seeds = seeds or POLICY_SEEDS

    _bootstrap_training_seed_files()

    max_workers = max(1, int(max_workers))
    if torch_threads_per_worker is None:
        torch_threads_per_worker = max(1, _cpu_quota_cores() // max_workers)

    print(
        f"  Parallel train mode: max_workers={max_workers}, "
        f"worker_threads={torch_threads_per_worker}, cpu_quota={_cpu_quota_cores()}"
    )

    all_results = {}
    for env_name in envs:
        for algo in algos:
            combo_pending = []
            for seed in seeds:
                key = _training_key(env_name, algo, seed)
                model_path = _policy_model_path(env_name, algo, seed)

                if model_path.exists():
                    existing_result = _load_training_seed_result(env_name, algo, seed)
                    if existing_result is None:
                        print(f"  [RECOVER] {key} — existing model, regenerating evaluation summary")
                        try:
                            existing_result = _evaluate_existing_model_result(
                                env_name, algo, seed, model_path,
                            )
                        except Exception as e:
                            existing_result = {
                                "status": "failed",
                                "env": env_name,
                                "algo": algo,
                                "seed": seed,
                                "error": str(e),
                                "model_path": str(model_path),
                            }
                        _save_json({key: existing_result}, _training_result_seed_file(env_name, algo, seed))
                    all_results[key] = existing_result
                    print(f"  [SKIP] {key} — already exists")
                    continue

                combo_pending.append(seed)

            if not combo_pending:
                continue

            combo_workers = min(max_workers, len(combo_pending))
            print(
                f"  Launching {len(combo_pending)} seed workers for {env_name} {algo} "
                f"with max_workers={combo_workers}"
            )

            with ThreadPoolExecutor(max_workers=combo_workers) as executor:
                future_to_seed = {
                    executor.submit(
                        _run_parallel_training_child,
                        env_name,
                        algo,
                        seed,
                        torch_threads_per_worker,
                    ): seed
                    for seed in combo_pending
                }

                for future in as_completed(future_to_seed):
                    seed = future_to_seed[future]
                    key = _training_key(env_name, algo, seed)
                    try:
                        _, result = future.result()
                    except Exception as e:
                        result = {
                            "status": "failed",
                            "env": env_name,
                            "algo": algo,
                            "seed": seed,
                            "error": str(e),
                        }
                        _save_json({key: result}, _training_result_seed_file(env_name, algo, seed))

                    all_results[key] = result
                    status = result.get("status", "unknown")
                    print(f"  [{status.upper()}] {key}")
                    _merge_training_result_files()

    _merge_training_result_files()
    return all_results


# =====================================================================
#  TRAINING
# =====================================================================

def train_all_policies(envs=None, algos=None, seeds=None, persist_combined: bool = True):
    """Train DQN and PPO policies for all env×algo×seed combinations."""
    _configure_runtime_threads()
    _bootstrap_training_seed_files()

    envs = envs or ENVS
    algos = algos or ALGOS
    seeds = seeds or POLICY_SEEDS

    results = {}
    for env_name in envs:
        for algo in algos:
            for seed in seeds:
                key, result = _train_single_policy(
                    env_name, algo, seed, persist_combined=persist_combined,
                )
                results[key] = result

    merged = _merge_training_result_files() if persist_combined else results
    print(f"\nTraining complete. {sum(1 for v in results.values() if v.get('status')=='success')} succeeded, "
          f"{sum(1 for v in results.values() if v.get('status')=='failed')} failed.")
    return results


def _evaluate_model(model, env_name, n_episodes=20, seed=42):
    """Evaluate a trained model in the highway environment."""
    env = make_highway_env(env_name, VEHICLES_COUNT_DEFAULT, "raw_derived")
    rewards, lengths = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep * 100)
        done = False
        ep_reward, ep_len = 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            ep_reward += reward
            ep_len += 1
            done = terminated or truncated
        rewards.append(ep_reward)
        lengths.append(ep_len)
    env.close()
    return rewards, lengths


# =====================================================================
#  REPLAY COLLECTION
# =====================================================================

def collect_all_replay(envs=None, algos=None, seeds=None):
    """Collect replay data for all trained policies."""
    from stable_baselines3 import DQN, PPO

    envs = envs or ENVS
    algos = algos or ALGOS
    seeds = seeds or POLICY_SEEDS

    results = {}
    for env_name in envs:
        for algo in algos:
            for seed in seeds:
                key = f"{env_name}_{algo}_seed{seed}"
                policy_dir = _env_algo_dir(env_name, algo) / "policies"
                model_path = policy_dir / f"policy_seed{seed}.zip"
                out_path = _replay_file(env_name, algo, seed)
                metadata_path = _replay_metadata_file(env_name, algo, seed)

                if not model_path.exists():
                    print(f"  [SKIP] {key} — no trained model")
                    results[key] = {"status": "no_model"}
                    continue

                if out_path.exists() and all(
                    _replay_split_file(env_name, algo, seed, split_name).exists()
                    for split_name in REPLAY_SPLITS
                ):
                    print(f"  [SKIP] {key} — replay exists")
                    existing = {"status": "skipped", "replay_path": str(out_path)}
                    if metadata_path.exists():
                        with open(metadata_path) as fh:
                            existing.update(json.load(fh))
                    results[key] = existing
                    continue

                print(f"  Collecting replay for {key}...")
                try:
                    env = make_highway_env(env_name, VEHICLES_COUNT_DEFAULT, "raw_derived")
                    AlgoClass = DQN if algo == "dqn" else PPO
                    model = AlgoClass.load(str(model_path), device=_sb3_device(algo))

                    state_raw_list, state_features_list = [], []
                    actions, rewards_list = [], []
                    dones, episode_ids = [], []
                    timesteps = []
                    ep_rewards, ep_lengths = [], []
                    episode_count = 0

                    for ep in range(N_ROLLOUT_EPISODES):
                        obs, _ = env.reset(seed=seed * 1000 + ep)
                        done = False
                        ep_r, ep_l = 0.0, 0
                        timestep = 0
                        while not done:
                            raw_obs = env.unwrapped.observation_type.observe().copy()
                            action, _ = model.predict(obs, deterministic=True)
                            action = int(action)
                            state_raw_list.append(flatten_obs_raw(raw_obs))
                            state_features_list.append(obs.copy())
                            actions.append(action)
                            timesteps.append(timestep)
                            obs, reward, terminated, truncated, _ = env.step(action)
                            done = terminated or truncated
                            rewards_list.append(reward)
                            dones.append(done)
                            episode_ids.append(episode_count)
                            ep_r += reward
                            ep_l += 1
                            timestep += 1
                        ep_rewards.append(ep_r)
                        ep_lengths.append(ep_l)
                        episode_count += 1

                    env.close()

                    state_raw_arr = np.array(state_raw_list, dtype=np.float32)
                    state_features_arr = np.array(state_features_list, dtype=np.float32)
                    actions_arr = np.array(actions, dtype=np.int32)
                    rewards_arr = np.array(rewards_list, dtype=np.float32)
                    dones_arr = np.array(dones, dtype=bool)
                    episode_ids_arr = np.array(episode_ids, dtype=np.int32)
                    timesteps_arr = np.array(timesteps, dtype=np.int32)
                    policy_seed_arr = np.full(len(actions_arr), seed, dtype=np.int32)

                    replay_bundle = {
                        "state_raw": state_raw_arr,
                        "state_features": state_features_arr,
                        "states": state_features_arr,
                        "actions": actions_arr,
                        "rewards": rewards_arr,
                        "dones": dones_arr,
                        "episode_ids": episode_ids_arr,
                        "timesteps": timesteps_arr,
                        "policy_seed": policy_seed_arr,
                    }

                    _save_npz(out_path, replay_bundle)
                    _save_replay_splits(replay_bundle, env_name, algo, seed)

                    # Action distribution
                    action_dist = {
                        HIGHWAY_ACTION_NAMES[a]: int(np.sum(actions_arr == a))
                        for a in range(N_ACTIONS)
                    }
                    action_frac = {
                        name: count / max(len(actions_arr), 1)
                        for name, count in action_dist.items()
                    }
                    min_support = min(action_frac.values()) if action_frac else 0.0

                    metadata = {
                        "status": "success",
                        "env": env_name,
                        "algo": algo,
                        "policy_seed": seed,
                        "n_transitions": len(state_features_arr),
                        "n_episodes": episode_count,
                        "mean_reward": float(np.mean(ep_rewards)),
                        "std_reward": float(np.std(ep_rewards)),
                        "mean_length": float(np.mean(ep_lengths)),
                        "action_distribution": action_dist,
                        "action_fraction": action_frac,
                        "min_action_support": float(min_support),
                        "rare_action_warning": min_support < 0.02,
                        "replay_path": str(out_path),
                        "split_paths": {
                            split_name: str(_replay_split_file(env_name, algo, seed, split_name))
                            for split_name in REPLAY_SPLITS
                        },
                    }
                    _save_json(metadata, metadata_path)

                    results[key] = metadata
                    print(
                        f"    {len(state_features_arr)} transitions, "
                        f"reward={np.mean(ep_rewards):.2f}+/-{np.std(ep_rewards):.2f}"
                    )

                except Exception as e:
                    print(f"  [FAIL] {key}: {e}")
                    traceback.print_exc()
                    results[key] = {"status": "failed", "error": str(e)}

    _save_json(results, _artifact_dir() / "replay_collection_results.json")
    print(f"\nReplay collection complete.")
    return results


def _split_replay_by_episode(states, actions, rewards, dones, episode_ids,
                             train_frac=0.7, val_frac=0.1, seed=42):
    """Split replay data by episode into train/val/test."""
    rng = np.random.RandomState(seed)
    unique_eps = np.unique(episode_ids)
    rng.shuffle(unique_eps)
    n = len(unique_eps)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_eps = set(unique_eps[:n_train])
    val_eps = set(unique_eps[n_train:n_train + n_val])
    test_eps = set(unique_eps[n_train + n_val:])

    def _mask(ep_set):
        return np.array([eid in ep_set for eid in episode_ids])

    splits = {}
    for split_name, ep_set in [("train", train_eps), ("val", val_eps), ("test", test_eps)]:
        m = _mask(ep_set)
        splits[split_name] = {
            "states": states[m],
            "actions": actions[m],
            "rewards": rewards[m],
            "dones": dones[m],
            "episode_ids": episode_ids[m],
        }
    return splits


def _load_replay_bundle(env_name: str, algo: str, policy_seed: int):
    replay_path = _replay_file(env_name, algo, policy_seed)
    if replay_path.exists():
        return _load_replay_npz(replay_path)

    legacy_path = _legacy_replay_file(env_name, algo, policy_seed)
    if legacy_path.exists():
        legacy = _load_replay_npz(legacy_path)
        if "state_features" not in legacy and "states" in legacy:
            legacy["state_features"] = legacy["states"]
        if "policy_seed" not in legacy:
            legacy["policy_seed"] = np.full(len(legacy["actions"]), policy_seed, dtype=np.int32)
        if "timesteps" not in legacy:
            timesteps = np.zeros(len(legacy["episode_ids"]), dtype=np.int32)
            current_ep = None
            current_t = 0
            for idx, episode_id in enumerate(legacy["episode_ids"]):
                if current_ep != episode_id:
                    current_ep = episode_id
                    current_t = 0
                timesteps[idx] = current_t
                current_t += 1
            legacy["timesteps"] = timesteps
        return legacy

    return None


def _explanation_run_path(env_name: str, algo: str, policy_seed: int,
                          method: str, explainer_seed: int, feature_mode: str,
                          vehicles_count: int, split_name: str,
                          suffix: str = "") -> Path:
    run_dir = _env_algo_dir(env_name, algo) / "explanations" / feature_mode / method
    run_dir.mkdir(parents=True, exist_ok=True)
    base = (
        f"{_env_tag(env_name)}_{algo}_policy_seed{policy_seed}_"
        f"explainer_{method}_explainer_seed{explainer_seed}_"
        f"vc{vehicles_count}_split_{split_name}"
    )
    if suffix:
        base = f"{base}_{suffix}"
    return run_dir / f"{base}.json"


def _run_method_reruns(method, train_s, train_a, val_s, val_a,
                       test_s, test_a, feature_names,
                       env_name, algo, policy_seed, explainer_seeds,
                       vehicles_count, feature_mode,
                       split_name="test_explainer", suffix=""):
    method_runs = []
    for explainer_seed in explainer_seeds:
        run_result = _run_single_explanation(
            method, train_s, train_a, val_s, val_a,
            test_s, test_a, feature_names,
            env_name, algo, policy_seed, explainer_seed,
            vehicles_count, feature_mode,
        )
        run_result["evaluation_split"] = split_name
        run_result["result_suffix"] = suffix
        _save_json(
            run_result,
            _explanation_run_path(
                env_name, algo, policy_seed, method, explainer_seed,
                feature_mode, vehicles_count, split_name, suffix,
            ),
        )
        method_runs.append(run_result)

    return method_runs, _compute_group_stability(method_runs)


# =====================================================================
#  EXPLANATION EXPERIMENTS
# =====================================================================

def run_explanation_experiments(envs=None, algos=None, policy_seeds=None,
                                explainer_seeds=None, methods=None,
                                feature_mode="raw_derived",
                                vehicles_count=6):
    """Run explanation methods across all combinations."""
    from reproduction.cbs import CBSPipeline
    from experiments.consensus_merge import build_voting_ensemble, voting_predict
    from experiments.decision_tree_surrogate import DecisionTreeSurrogate
    from experiments.rule_matching import (
        canonicalize_rules, serialize_canonical_rules,
        mean_pairwise_soft_jaccard, mean_pairwise_bra,
    )

    envs = envs or ENVS
    algos = algos or ALGOS
    policy_seeds = policy_seeds or POLICY_SEEDS
    explainer_seeds = explainer_seeds or EXPLAINER_SEEDS
    methods = methods or METHODS

    feature_names = get_feature_names(vehicles_count, feature_mode)
    state_key = "state_features" if feature_mode == "raw_derived" else "state_raw"

    root_out_path = _artifact_dir() / f"explanation_results_{feature_mode}.json"
    if root_out_path.exists():
        all_results = _load_json(root_out_path)
    else:
        all_results = {}

    for env_name in envs:
        for algo in algos:
            combo_key = f"{env_name}_{algo}"
            out_path = (
                _env_algo_dir(env_name, algo)
                / "explanations"
                / f"{_env_tag(env_name)}_{algo}_explanation_results_{feature_mode}.json"
            )
            if out_path.exists():
                combo_results = _load_json(out_path)
            else:
                combo_results = dict(all_results.get(combo_key, {}))

            for ps in policy_seeds:
                data = _load_replay_bundle(env_name, algo, ps)
                if data is None:
                    print(f"  [SKIP] {combo_key} seed{ps} — no replay")
                    continue

                states = data.get(state_key, data.get("states"))
                actions = data["actions"]
                rewards = data["rewards"]
                dones = data["dones"]
                episode_ids = data["episode_ids"]

                if len(states) == 0 or len(actions) == 0:
                    print(f"  [SKIP] {combo_key} seed{ps} — empty replay data")
                    continue

                # Split by episode
                splits = _split_replay_by_episode(
                    states, actions, rewards, dones, episode_ids)
                train_s, train_a = splits["train"]["states"], splits["train"]["actions"]
                val_s, val_a = splits["val"]["states"], splits["val"]["actions"]
                test_s, test_a = splits["test"]["states"], splits["test"]["actions"]

                for method in methods:
                    print(f"  [{combo_key}] ps{ps}_{method} reruns={len(explainer_seeds)}")
                    try:
                        method_runs, group_stability = _run_method_reruns(
                            method, train_s, train_a, val_s, val_a,
                            test_s, test_a, feature_names,
                            env_name, algo, ps, explainer_seeds,
                            vehicles_count, feature_mode,
                        )
                    except Exception as e:
                        print(f"  [{combo_key}] ps{ps}_{method} FAIL: {e}")
                        method_runs = [{
                            "status": "failed",
                            "error": str(e),
                            "policy_seed": ps,
                            "method": method,
                        }]
                        group_stability = {"GRS": None, "BRA": None, "n_runs": 0}

                    combo_results[f"ps{ps}_{method}"] = {
                        "runs": method_runs,
                        "group_stability": group_stability,
                    }

            all_results[combo_key] = combo_results
            _save_json(combo_results, out_path)

    _save_json(all_results, root_out_path)
    return all_results


def _run_single_explanation(method, train_s, train_a, val_s, val_a,
                            test_s, test_a, feature_names,
                            env_name, algo, policy_seed, explainer_seed,
                            vehicles_count, feature_mode):
    """Run a single explanation method and evaluate."""
    from reproduction.cbs import CBSPipeline
    from experiments.decision_tree_surrogate import DecisionTreeSurrogate
    from experiments.rule_matching import canonicalize_rules, serialize_canonical_rules
    from sklearn.metrics import f1_score, precision_score, recall_score

    result = {
        "env": env_name, "algo": algo,
        "policy_seed": policy_seed, "explainer_seed": explainer_seed,
        "method": method, "feature_mode": feature_mode,
        "vehicles_count": vehicles_count,
        "n_train": len(train_s), "n_val": len(val_s), "n_test": len(test_s),
    }

    if method == "cbs":
        cbs = CBSPipeline(
            n_categories=5, inclusion_threshold=0.70,
            kmeans_seed=explainer_seed, feature_names=feature_names,
        )
        cbs.fit(train_s, train_a)
        preds_test = cbs.predict(test_s)
        rules = canonicalize_rules(cbs.get_rules())
        result["rules"] = serialize_canonical_rules(rules)
        result["n_rules"] = len(rules)
        result["thresholds"] = {int(k): [float(v) for v in vs]
                                for k, vs in cbs.get_thresholds().items()}
        result["preds_test"] = preds_test.tolist()

    elif method == "b3_vote":
        from experiments.consensus_merge import build_voting_ensemble, voting_predict
        pipelines = []
        for b in range(5):
            rng = np.random.RandomState(explainer_seed * 100 + b)
            n = len(train_s)
            idx = rng.choice(n, size=int(n * 0.8), replace=False)
            sub_s, sub_a = train_s[idx], train_a[idx]
            cbs = CBSPipeline(
                n_categories=5, inclusion_threshold=0.70,
                kmeans_seed=explainer_seed * 10 + b,
                feature_names=feature_names,
            )
            cbs.fit(sub_s, sub_a)
            pipelines.append(cbs)
        preds_test = voting_predict(pipelines, test_s)
        # Canonicalize from first pipeline for rule structure
        rules = canonicalize_rules(pipelines[0].get_rules())
        all_rules = []
        for p in pipelines:
            all_rules.extend(canonicalize_rules(p.get_rules()))
        result["rules"] = serialize_canonical_rules(rules)
        result["n_rules"] = len(all_rules)
        result["preds_test"] = preds_test.tolist()

    elif method == "dt":
        dt = DecisionTreeSurrogate(
            max_depth=None, min_samples_leaf=5,
            random_state=explainer_seed,
            feature_names=feature_names,
        )
        dt.fit(train_s, train_a)
        preds_test = dt.predict(test_s)
        rules = canonicalize_rules(dt.get_rules())
        result["rules"] = serialize_canonical_rules(rules)
        result["n_rules"] = len(rules)
        result["preds_test"] = preds_test.tolist()

    elif method == "b5_bdr":
        from experiments.boolean_rules import BDRSurrogate
        bdr = BDRSurrogate(
            n_quantile_thresholds=4, max_rules_per_action=8,
            min_support_frac=0.01, max_literals=3,
            random_state=explainer_seed,
            feature_names=feature_names,
        )
        bdr.fit(train_s, train_a)
        preds_test = bdr.predict(test_s)
        rules = canonicalize_rules(bdr.get_rules())
        result["rules"] = serialize_canonical_rules(rules)
        result["n_rules"] = len(rules)
        result["preds_test"] = preds_test.tolist()

    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute fidelity metrics
    preds_test = np.array(result["preds_test"])
    result["macro_f1"] = float(f1_score(test_a, preds_test, average="macro", zero_division=0))
    result["weighted_f1"] = float(f1_score(test_a, preds_test, average="weighted", zero_division=0))

    # Per-action F1
    per_action_f1 = {}
    for a in range(N_ACTIONS):
        mask = test_a == a
        if mask.sum() > 0:
            a_preds = (preds_test == a).astype(int)
            a_true = (test_a == a).astype(int)
            per_action_f1[HIGHWAY_ACTION_NAMES[a]] = float(
                f1_score(a_true, a_preds, zero_division=0))
    result["per_action_f1"] = per_action_f1

    # Rule complexity
    if "rules" in result and isinstance(result["rules"], list):
        rule_lengths = []
        features_used = set()
        per_action_rule_count = {}
        for r in result["rules"]:
            if isinstance(r, dict):
                preds = r.get("predicates", [])
                rule_lengths.append(len(preds))
                for p in preds:
                    features_used.add(p.get("feature_idx", -1))
                a = r.get("action")
                per_action_rule_count[a] = per_action_rule_count.get(a, 0) + 1
        result["mean_rule_length"] = float(np.mean(rule_lengths)) if rule_lengths else 0
        result["n_features_used"] = len(features_used)
        result["per_action_rule_count"] = per_action_rule_count

    # Action coverage
    unique_pred_actions = set(int(p) for p in preds_test)
    unique_true_actions = set(int(a) for a in test_a)
    result["action_coverage"] = len(unique_pred_actions & unique_true_actions) / max(len(unique_true_actions), 1)

    # Rare-action support
    unique_a, counts_a = np.unique(test_a, return_counts=True)
    total = len(test_a)
    result["rare_action_support"] = {
        HIGHWAY_ACTION_NAMES.get(int(a), f"a{a}"): int(c) / total
        for a, c in zip(unique_a, counts_a)
    }

    result["status"] = "success"
    return result


def _compute_group_stability(runs):
    """Compute group-level stability across explainer seeds."""
    from experiments.rule_matching import (
        canonicalize_rules, mean_pairwise_soft_jaccard,
    )

    successful = [r for r in runs if r.get("status") == "success"]
    if len(successful) < 2:
        return {"GRS": None, "BRA": None, "n_runs": len(successful)}

    # BRA: pairwise prediction agreement
    pred_arrays = []
    for r in successful:
        if "preds_test" in r:
            pred_arrays.append(np.array(r["preds_test"]))

    bra = None
    if len(pred_arrays) >= 2:
        agreements = []
        for i in range(len(pred_arrays)):
            for j in range(i + 1, len(pred_arrays)):
                if len(pred_arrays[i]) == len(pred_arrays[j]):
                    agreements.append(float(np.mean(pred_arrays[i] == pred_arrays[j])))
        bra = float(np.mean(agreements)) if agreements else None

    # GRS: pairwise rule similarity
    rule_sets = []
    for ri, r in enumerate(successful):
        if "rules" in r and isinstance(r["rules"], list):
            from experiments.rule_matching import canonicalize_from_json
            try:
                cr = canonicalize_from_json(r["rules"])
                rule_sets.append(cr)
            except Exception as e:
                print(f"  [WARN] rule canonicalization failed (run {ri}/{len(successful)}): {e}")

    grs = None
    if len(rule_sets) >= 2:
        try:
            grs = float(mean_pairwise_soft_jaccard(rule_sets))
        except Exception as e:
            print(f"  [WARN] GRS computation failed: {e}")

    # Aggregate run-level metrics
    f1s = [r["macro_f1"] for r in successful if "macro_f1" in r]
    n_rules_list = [r["n_rules"] for r in successful if "n_rules" in r]

    return {
        "GRS": grs,
        "BRA": bra,
        "n_runs": len(successful),
        "mean_f1": float(np.mean(f1s)) if f1s else None,
        "std_f1": float(np.std(f1s)) if f1s else None,
        "mean_n_rules": float(np.mean(n_rules_list)) if n_rules_list else None,
    }


def _aggregate_group_metric(group, metric_name):
    stab = group.get("group_stability", {})
    if metric_name == "GRS":
        return stab.get("GRS")
    if metric_name == "BRA":
        return stab.get("BRA")

    runs = [r for r in group.get("runs", []) if r.get("status") == "success"]
    if not runs:
        return None

    if metric_name == "macro_f1":
        vals = [r.get("macro_f1") for r in runs if r.get("macro_f1") is not None]
        return float(np.mean(vals)) if vals else None
    if metric_name == "weighted_f1":
        vals = [r.get("weighted_f1") for r in runs if r.get("weighted_f1") is not None]
        return float(np.mean(vals)) if vals else None
    if metric_name == "n_rules":
        vals = [r.get("n_rules") for r in runs if r.get("n_rules") is not None]
        return float(np.mean(vals)) if vals else None
    if metric_name == "mean_rule_length":
        vals = [r.get("mean_rule_length") for r in runs if r.get("mean_rule_length") is not None]
        return float(np.mean(vals)) if vals else None
    return None


def _highway_noise_mask(feature_names):
    mask = []
    for name in feature_names:
        is_indicator = (
            name.endswith("presence")
            or name.endswith("_exists")
            or name == "num_present_vehicles"
        )
        mask.append(not is_indicator)
    return np.array(mask, dtype=bool)


def _add_highway_feature_noise(states, feature_names, noise_level, seed):
    if noise_level <= 0:
        return states.copy()

    noisy = states.copy().astype(np.float32)
    mask = _highway_noise_mask(feature_names)
    if mask.sum() == 0:
        return noisy

    rng = np.random.default_rng(seed)
    sigma = noisy[:, mask].std(axis=0)
    sigma = np.where(sigma > 0, sigma, 1e-6)
    noise = rng.normal(loc=0.0, scale=noise_level * sigma,
                       size=(noisy.shape[0], int(mask.sum()))).astype(np.float32)
    noisy[:, mask] = noisy[:, mask] + noise
    return noisy


# =====================================================================
#  NOISE SEVERITY
# =====================================================================

def run_noise_severity(envs=None, algos=None, policy_seeds=None,
                       explainer_seeds=None, methods=None,
                       feature_mode="raw_derived"):
    """Run noise severity experiments across noise levels."""
    envs = envs or ENVS
    algos = algos or ALGOS
    policy_seeds = policy_seeds or POLICY_SEEDS
    explainer_seeds = explainer_seeds or EXPLAINER_SEEDS
    methods = methods or METHODS

    feature_names = get_feature_names(VEHICLES_COUNT_DEFAULT, feature_mode)
    state_key = "state_features" if feature_mode == "raw_derived" else "state_raw"
    root_out_path = _artifact_dir() / "noise_severity_results.json"
    if root_out_path.exists():
        all_results = _load_json(root_out_path)
    else:
        all_results = {}

    for env_name in envs:
        for algo in algos:
            combo_key = f"{env_name}_{algo}"
            out_path = (
                _env_algo_dir(env_name, algo)
                / "metrics"
                / f"{_env_tag(env_name)}_{algo}_noise_severity_results.json"
            )
            if out_path.exists():
                noise_results = _load_json(out_path)
            else:
                noise_results = dict(all_results.get(combo_key, {}))

            for ps in policy_seeds:
                data = _load_replay_bundle(env_name, algo, ps)
                if data is None:
                    continue

                splits = _split_replay_by_episode(
                    data.get(state_key, data["states"]), data["actions"], data["rewards"],
                    data["dones"], data["episode_ids"])
                train_s, train_a = splits["train"]["states"], splits["train"]["actions"]
                val_s, val_a = splits["val"]["states"], splits["val"]["actions"]
                test_s, test_a = splits["test"]["states"], splits["test"]["actions"]

                for method in methods:
                    for noise_level in NOISE_LEVELS:
                        nk = f"ps{ps}_{method}_noise{noise_level}"
                        print(f"  [{combo_key}] {nk} reruns={len(explainer_seeds)}")

                        try:
                            ns = _add_highway_feature_noise(
                                train_s, feature_names, noise_level,
                                seed=ps * 1000 + int(noise_level * 1000),
                            )
                            runs, group_stability = _run_method_reruns(
                                method, ns, train_a, val_s, val_a,
                                test_s, test_a, feature_names,
                                env_name, algo, ps, explainer_seeds,
                                VEHICLES_COUNT_DEFAULT, feature_mode,
                                suffix=f"noise_{str(noise_level).replace('.', 'p')}",
                            )
                            noise_results[nk] = {
                                "noise_level": noise_level,
                                "runs": runs,
                                "group_stability": group_stability,
                            }
                        except Exception as e:
                            print(f"FAIL: {e}")
                            noise_results[nk] = {
                                "status": "failed", "noise_level": noise_level,
                                "error": str(e),
                            }

            all_results[combo_key] = noise_results
            _save_json(noise_results, out_path)

    _save_json(all_results, root_out_path)
    return all_results


# =====================================================================
#  VEHICLES COUNT ABLATION
# =====================================================================

def run_vehicles_count_ablation(envs=None, algos=None, policy_seeds=None,
                                 explainer_seeds=None, methods=None):
    """Run explanation experiments at different vehicles_count settings."""
    envs = envs or ENVS
    algos = algos or ALGOS
    policy_seeds = policy_seeds or POLICY_SEEDS
    explainer_seeds = explainer_seeds or EXPLAINER_SEEDS
    methods = methods or METHODS

    root_out_path = _artifact_dir() / "vehicles_count_ablation_results.json"
    if root_out_path.exists():
        all_results = _load_json(root_out_path)
    else:
        all_results = {}

    for vc in VEHICLES_COUNTS_ABLATION:
        print(f"\n{'='*60}")
        print(f"  Vehicles Count Ablation: vehicles_count = {vc}")
        print(f"{'='*60}")

        # Need to re-collect replay with different vehicles_count
        for env_name in envs:
            for algo in algos:
                combo_key = f"{env_name}_{algo}_vc{vc}"
                vc_results = dict(all_results.get(combo_key, {}))

                for ps in policy_seeds:
                    # Collect replay with this vehicles_count
                    replay = _collect_replay_with_vc(env_name, algo, ps, vc)
                    if replay is None:
                        continue

                    splits = _split_replay_by_episode(
                        replay["states"], replay["actions"],
                        replay["rewards"], replay["dones"],
                        replay["episode_ids"])
                    train_s, train_a = splits["train"]["states"], splits["train"]["actions"]
                    test_s, test_a = splits["test"]["states"], splits["test"]["actions"]

                    feat_names = get_feature_names(vc, "raw_derived")
                    val_s, val_a = test_s, test_a

                    for method in methods:
                        nk = f"ps{ps}_{method}"
                        print(f"  [{combo_key}] {nk} reruns={len(explainer_seeds)}")
                        try:
                            runs, group_stability = _run_method_reruns(
                                method, train_s, train_a, val_s, val_a,
                                test_s, test_a, feat_names,
                                env_name, algo, ps, explainer_seeds,
                                vc, "raw_derived",
                                suffix=f"vc{vc}",
                            )
                            vc_results[nk] = {
                                "vehicles_count": vc,
                                "runs": runs,
                                "group_stability": group_stability,
                            }
                        except Exception as e:
                            print(f"FAIL: {e}")
                            vc_results[nk] = {
                                "vehicles_count": vc,
                                "status": "failed",
                                "error": str(e),
                            }

                all_results[combo_key] = vc_results

    _save_json(all_results, root_out_path)
    return all_results


def _collect_replay_with_vc(env_name, algo, policy_seed, vehicles_count):
    """Collect replay with a specific vehicles_count (re-rollout)."""
    from stable_baselines3 import DQN, PPO

    policy_dir = _env_algo_dir(env_name, algo) / "policies"
    model_path = policy_dir / f"policy_seed{policy_seed}.zip"
    if not model_path.exists():
        return None

    # We need to use the model trained with vc=6 but observe with vc=vehicles_count
    # The model uses raw_derived features (82-dim for vc=6).
    # For different vc, we need to handle shape mismatch.
    # Solution: always use vc=6 model, but re-extract observations at new vc
    # Actually, the model was trained with vc=6 features. For ablation,
    # we re-collect with the same model but different vc observation.
    # The model's policy still works (it takes flat features as input) — 
    # but the input dim changes. We need a workaround.
    #
    # The correct approach for vc ablation: use the model to get ACTIONS,
    # but observe the environment at the new vc for feature extraction.
    # We'll use a dual-env approach: one for action, one for observation.
    #
    # Simpler approach: Run the model in vc=6 env, but also observe
    # the world at vc=4 or vc=8. Since highway-env is deterministic
    # for the same seed, we can collect both simultaneously.
    #
    # Simplest valid approach: re-rollout with vc=6 model, extract obs at target vc.

    env_action = make_highway_env(env_name, 6, "raw_derived")
    AlgoClass = DQN if algo == "dqn" else PPO
    model = AlgoClass.load(str(model_path), device=_sb3_device(algo))

    states, actions_list, rewards_list = [], [], []
    dones, episode_ids = [], []
    episode_count = 0
    feat_names = get_feature_names(vehicles_count, "raw_derived")

    for ep in range(50):  # Fewer episodes for ablation
        obs_action, _ = env_action.reset(seed=policy_seed * 1000 + ep)
        # Get raw 2D obs from wrapped env's underlying env
        raw_2d = env_action.env.unwrapped.observation_type.observe()

        done = False
        while not done:
            action, _ = model.predict(obs_action, deterministic=True)
            action = int(action)

            # Extract features at target vc
            raw_2d = env_action.env.unwrapped.observation_type.observe()
            if vehicles_count <= raw_2d.shape[0]:
                obs_vc = raw_2d[:vehicles_count]
            else:
                obs_vc = np.zeros((vehicles_count, raw_2d.shape[1]), dtype=np.float32)
                obs_vc[:raw_2d.shape[0]] = raw_2d

            flat = flatten_obs_full(obs_vc)
            states.append(flat)
            actions_list.append(action)

            obs_action, reward, terminated, truncated, _ = env_action.step(action)
            done = terminated or truncated
            rewards_list.append(reward)
            dones.append(done)
            episode_ids.append(episode_count)

        episode_count += 1

    env_action.close()

    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions_list, dtype=np.int32),
        "rewards": np.array(rewards_list, dtype=np.float32),
        "dones": np.array(dones, dtype=bool),
        "episode_ids": np.array(episode_ids, dtype=np.int32),
    }


# =====================================================================
#  FEATURE ABLATION
# =====================================================================

def run_feature_ablation(envs=None, algos=None, policy_seeds=None,
                         explainer_seeds=None, methods=None):
    """Compare raw-only vs raw+derived features."""
    envs = envs or ENVS
    algos = algos or ALGOS
    policy_seeds = policy_seeds or POLICY_SEEDS
    explainer_seeds = explainer_seeds or EXPLAINER_SEEDS
    methods = methods or METHODS

    root_out_path = _artifact_dir() / "feature_ablation_results.json"
    if root_out_path.exists():
        all_results = _load_json(root_out_path)
    else:
        all_results = {}

    for fm in FEATURE_MODES:
        print(f"\n{'='*60}")
        print(f"  Feature Ablation: mode = {fm}")
        print(f"{'='*60}")

        for env_name in envs:
            for algo in algos:
                combo_key = f"{env_name}_{algo}_{fm}"
                fm_results = dict(all_results.get(combo_key, {}))

                for ps in policy_seeds:
                    # Re-collect replay with the specified feature mode
                    replay = _collect_replay_feature_mode(env_name, algo, ps, fm)
                    if replay is None:
                        continue

                    splits = _split_replay_by_episode(
                        replay["states"], replay["actions"],
                        replay["rewards"], replay["dones"],
                        replay["episode_ids"])
                    train_s, train_a = splits["train"]["states"], splits["train"]["actions"]
                    test_s, test_a = splits["test"]["states"], splits["test"]["actions"]

                    feat_names = get_feature_names(VEHICLES_COUNT_DEFAULT, fm)
                    val_s, val_a = test_s, test_a

                    for method in methods:
                        nk = f"ps{ps}_{method}"
                        print(f"  [{combo_key}] {nk} reruns={len(explainer_seeds)}")
                        try:
                            runs, group_stability = _run_method_reruns(
                                method, train_s, train_a, val_s, val_a,
                                test_s, test_a, feat_names,
                                env_name, algo, ps, explainer_seeds,
                                VEHICLES_COUNT_DEFAULT, fm,
                                suffix=fm,
                            )
                            fm_results[nk] = {
                                "feature_mode": fm,
                                "runs": runs,
                                "group_stability": group_stability,
                            }
                        except Exception as e:
                            print(f"FAIL: {e}")
                            fm_results[nk] = {
                                "feature_mode": fm,
                                "status": "failed",
                                "error": str(e),
                            }

                all_results[combo_key] = fm_results

    _save_json(all_results, root_out_path)
    return all_results


def _collect_replay_feature_mode(env_name, algo, policy_seed, feature_mode):
    """Collect replay with a specific feature mode."""
    from stable_baselines3 import DQN, PPO

    # For raw_derived, we already have data. For raw, need to re-extract.
    replay_path = _replay_file(env_name, algo, policy_seed)

    if feature_mode == "raw_derived" and replay_path.exists():
        return _load_replay_npz(replay_path)

    # For raw mode, re-collect
    if feature_mode == "raw":
        policy_dir = _env_algo_dir(env_name, algo) / "policies"
        model_path = policy_dir / f"policy_seed{policy_seed}.zip"
        if not model_path.exists():
            return None

        env = make_highway_env(env_name, VEHICLES_COUNT_DEFAULT, "raw")
        # Load with the training env features (raw_derived) for prediction
        env_pred = make_highway_env(env_name, VEHICLES_COUNT_DEFAULT, "raw_derived")
        AlgoClass = DQN if algo == "dqn" else PPO
        model = AlgoClass.load(str(model_path), device=_sb3_device(algo))

        states, actions_list, rewards_list = [], [], []
        dones_list, episode_ids = [], []
        episode_count = 0

        for ep in range(50):
            obs_pred, _ = env_pred.reset(seed=policy_seed * 1000 + ep)
            obs_raw, _ = env.reset(seed=policy_seed * 1000 + ep)
            done = False
            while not done:
                action, _ = model.predict(obs_pred, deterministic=True)
                action = int(action)
                states.append(obs_raw.copy())
                actions_list.append(action)
                obs_pred, reward, terminated, truncated, _ = env_pred.step(action)
                obs_raw, _, _, _, _ = env.step(action)
                done = terminated or truncated
                rewards_list.append(reward)
                dones_list.append(done)
                episode_ids.append(episode_count)
            episode_count += 1

        env.close()
        env_pred.close()

        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions_list, dtype=np.int32),
            "rewards": np.array(rewards_list, dtype=np.float32),
            "dones": np.array(dones_list, dtype=bool),
            "episode_ids": np.array(episode_ids, dtype=np.int32),
        }

    return None


# =====================================================================
#  BEHAVIORAL EVALUATION
# =====================================================================

def run_behavioral_evaluation(envs=None, algos=None, policy_seeds=None,
                              methods=None):
    """Evaluate rule-based policies in the environment."""
    from reproduction.cbs import CBSPipeline
    from experiments.decision_tree_surrogate import DecisionTreeSurrogate

    envs = envs or ENVS
    algos = algos or ALGOS
    policy_seeds = policy_seeds or POLICY_SEEDS
    methods = methods or METHODS

    feature_names = get_feature_names(VEHICLES_COUNT_DEFAULT, "raw_derived")
    root_out_path = _artifact_dir() / "behavioral_evaluation_results.json"
    if root_out_path.exists():
        all_results = _load_json(root_out_path)
    else:
        all_results = {}

    for env_name in envs:
        for algo in algos:
            combo_key = f"{env_name}_{algo}"
            out_path = (
                _env_algo_dir(env_name, algo)
                / "metrics"
                / f"{_env_tag(env_name)}_{algo}_behavioral_evaluation.json"
            )
            if out_path.exists():
                eval_results = _load_json(out_path)
            else:
                eval_results = dict(all_results.get(combo_key, {}))

            for ps in policy_seeds:
                data = _load_replay_bundle(env_name, algo, ps)
                if data is None:
                    continue

                train_s, train_a = data["states"], data["actions"]

                for method in methods:
                    nk = f"ps{ps}_{method}"
                    print(f"  [{combo_key}] behavior eval {nk}...", end=" ", flush=True)

                    try:
                        # Build the explainer
                        predictor = _build_predictor(
                            method, train_s, train_a, feature_names, 0)
                        if predictor is None:
                            print("SKIP")
                            continue

                        # Rollout
                        stats = _rollout_rule_policy(
                            predictor, env_name, EVAL_SEEDS[:EVAL_EPISODES])
                        eval_results[nk] = stats
                        print(f"return={stats['mean_return']:.2f}, "
                              f"collision={stats.get('collision_rate', 0):.2f}")

                    except Exception as e:
                        print(f"FAIL: {e}")
                        eval_results[nk] = {"status": "failed", "error": str(e)}

            all_results[combo_key] = eval_results
            _save_json(eval_results, out_path)

    _save_json(all_results, root_out_path)
    return all_results


def _build_predictor(method, train_s, train_a, feature_names, seed):
    """Build a trained predictor for rollout evaluation."""
    from reproduction.cbs import CBSPipeline
    from experiments.decision_tree_surrogate import DecisionTreeSurrogate

    if method == "cbs":
        cbs = CBSPipeline(
            n_categories=5, inclusion_threshold=0.70,
            kmeans_seed=seed, feature_names=feature_names,
        )
        cbs.fit(train_s, train_a)
        return cbs

    elif method == "b3_vote":
        from experiments.consensus_merge import build_voting_ensemble, voting_predict
        pipelines = []
        for b in range(5):
            rng = np.random.RandomState(seed * 100 + b)
            n = len(train_s)
            idx = rng.choice(n, size=int(n * 0.8), replace=False)
            cbs = CBSPipeline(
                n_categories=5, inclusion_threshold=0.70,
                kmeans_seed=seed * 10 + b,
                feature_names=feature_names,
            )
            cbs.fit(train_s[idx], train_a[idx])
            pipelines.append(cbs)

        class VotingPredictor:
            def __init__(self, pipelines):
                self.pipelines = pipelines
            def predict(self, states):
                return voting_predict(self.pipelines, states)

        return VotingPredictor(pipelines)

    elif method == "dt":
        dt = DecisionTreeSurrogate(
            max_depth=None, min_samples_leaf=5,
            random_state=seed, feature_names=feature_names,
        )
        dt.fit(train_s, train_a)
        return dt

    elif method == "b5_bdr":
        from experiments.boolean_rules import BDRSurrogate

        bdr = BDRSurrogate(
            n_quantile_thresholds=4,
            max_rules_per_action=8,
            min_support_frac=0.01,
            max_literals=3,
            random_state=seed,
            feature_names=feature_names,
        )
        bdr.fit(train_s, train_a)
        return bdr

    return None


def _rollout_rule_policy(predictor, env_name, eval_seeds):
    """Rollout a rule-based policy in the environment."""
    rewards, lengths, collisions, successes = [], [], [], []

    for seed in eval_seeds:
        env = make_highway_env(env_name, VEHICLES_COUNT_DEFAULT, "raw_derived")
        obs, _ = env.reset(seed=seed)
        done = False
        ep_r, ep_l = 0.0, 0
        crashed = False

        while not done:
            pred = predictor.predict(obs.reshape(1, -1))
            action = int(pred[0]) if hasattr(pred, '__len__') else int(pred)
            action = max(0, min(action, N_ACTIONS - 1))
            obs, reward, terminated, truncated, info = env.step(action)
            ep_r += reward
            ep_l += 1
            done = terminated or truncated
            if info.get("crashed", False):
                crashed = True

        rewards.append(ep_r)
        lengths.append(ep_l)
        collisions.append(1.0 if crashed else 0.0)
        successes.append(1.0 if not crashed else 0.0)
        env.close()

    return {
        "mean_return": float(np.mean(rewards)),
        "std_return": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "collision_rate": float(np.mean(collisions)),
        "success_rate": float(np.mean(successes)),
        "n_episodes": len(eval_seeds),
    }


# =====================================================================
#  STATISTICAL ANALYSIS
# =====================================================================

def run_statistical_analysis():
    """Run statistical tests comparing methods."""
    from scipy import stats as sp_stats

    results_path = _artifact_dir() / "explanation_results_raw_derived.json"
    behavior_path = _artifact_dir() / "behavioral_evaluation_results.json"
    if not results_path.exists():
        print("No explanation results found — run explain phase first.")
        return {}

    with open(results_path) as f:
        all_results = json.load(f)

    behavior_data = {}
    if behavior_path.exists():
        with open(behavior_path) as f:
            behavior_data = json.load(f)

    comparisons = [
        ("cbs", "b3_vote"),
        ("cbs", "dt"),
        ("b3_vote", "dt"),
        ("cbs", "b5_bdr"),
        ("b3_vote", "b5_bdr"),
    ]

    stat_results = {}

    def _bootstrap_ci(values):
        rng = np.random.RandomState(42)
        arr = np.asarray(values, dtype=float)
        samples = []
        for _ in range(1000):
            idx = rng.choice(len(arr), size=len(arr), replace=True)
            samples.append(float(np.mean(arr[idx])))
        return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]

    def _group_metric(group, metric_name):
        runs = [r for r in group.get("runs", []) if r.get("status") == "success"]
        stab = group.get("group_stability", {})

        if metric_name == "GRS":
            return stab.get("GRS")
        if metric_name == "BRA":
            return stab.get("BRA")
        if not runs:
            return None
        if metric_name == "macro_f1":
            vals = [r.get("macro_f1") for r in runs if r.get("macro_f1") is not None]
            return float(np.mean(vals)) if vals else None
        if metric_name == "weighted_f1":
            vals = [r.get("weighted_f1") for r in runs if r.get("weighted_f1") is not None]
            return float(np.mean(vals)) if vals else None
        if metric_name == "n_rules":
            vals = [r.get("n_rules") for r in runs if r.get("n_rules") is not None]
            return float(np.mean(vals)) if vals else None
        if metric_name == "mean_rule_length":
            vals = [r.get("mean_rule_length") for r in runs if r.get("mean_rule_length") is not None]
            return float(np.mean(vals)) if vals else None
        return None

    for combo_key, combo_data in all_results.items():
        combo_stats = {"summaries": {}, "comparisons": {}}
        combo_behavior = behavior_data.get(combo_key, {})
        per_method = {}

        for method in METHODS:
            per_seed = {}
            for ps in POLICY_SEEDS:
                pk = f"ps{ps}_{method}"
                if pk not in combo_data:
                    continue
                group = combo_data[pk]
                metric_row = {
                    "macro_f1": _group_metric(group, "macro_f1"),
                    "weighted_f1": _group_metric(group, "weighted_f1"),
                    "GRS": _group_metric(group, "GRS"),
                    "BRA": _group_metric(group, "BRA"),
                    "n_rules": _group_metric(group, "n_rules"),
                    "mean_rule_length": _group_metric(group, "mean_rule_length"),
                    "mean_return": combo_behavior.get(pk, {}).get("mean_return"),
                }
                if any(value is not None for value in metric_row.values()):
                    per_seed[ps] = metric_row

            if not per_seed:
                continue

            per_method[method] = per_seed
            method_summary = {}
            for metric_name in [
                "macro_f1", "weighted_f1", "GRS", "BRA",
                "mean_return", "n_rules", "mean_rule_length",
            ]:
                values = [row[metric_name] for row in per_seed.values() if row.get(metric_name) is not None]
                if not values:
                    continue
                method_summary[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "ci_95": _bootstrap_ci(values),
                    "n": len(values),
                }
            combo_stats["summaries"][method] = method_summary

        for m1, m2 in comparisons:
            if m1 not in per_method or m2 not in per_method:
                continue

            common_policy_seeds = sorted(set(per_method[m1]) & set(per_method[m2]))
            if len(common_policy_seeds) < 3:
                continue

            pair_metrics = {}
            for metric_name in ["macro_f1", "GRS", "BRA", "mean_return"]:
                x, y = [], []
                for ps in common_policy_seeds:
                    v1 = per_method[m1][ps].get(metric_name)
                    v2 = per_method[m2][ps].get(metric_name)
                    if v1 is None or v2 is None:
                        continue
                    x.append(float(v1))
                    y.append(float)

                if len(x) < 3:
                    continue

                diff = np.asarray(x) - np.asarray(y)
                try:
                    w_stat, w_p = sp_stats.wilcoxon(diff)
                except Exception as e:
                    print(f"  [WARN] Wilcoxon test failed for {m1} vs {m2} / {metric_name} (n={len(diff)}): {e}")
                    w_stat, w_p = float("nan"), float("nan")

                pair_metrics[metric_name] = {
                    "mean_diff": float(np.mean(diff)),
                    "std_diff": float(np.std(diff)),
                    "ci_95": _bootstrap_ci(diff),
                    "wilcoxon_W": float(w_stat),
                    "wilcoxon_p": float(w_p),
                    "n_pairs": len(diff),
                }

            if pair_metrics:
                combo_stats["comparisons"][f"{m1}_vs_{m2}"] = pair_metrics

        stat_results[combo_key] = combo_stats

    _save_json(stat_results, _artifact_dir() / "statistical_analysis.json")
    print(f"Statistical analysis complete: {len(stat_results)} combo comparisons.")
    return stat_results


# =====================================================================
#  TABLE & FIGURE GENERATION
# =====================================================================

def generate_main_table():
    """Generate the main results table (Table 1)."""
    results_path = _artifact_dir() / "explanation_results_raw_derived.json"
    behavior_path = _artifact_dir() / "behavioral_evaluation_results.json"

    if not results_path.exists():
        print("No results for table generation.")
        return None

    with open(results_path) as f:
        all_results = json.load(f)

    behavior_data = {}
    if behavior_path.exists():
        with open(behavior_path) as f:
            behavior_data = json.load(f)

    rows = []
    for env_name in ENVS:
        for algo in ALGOS:
            combo_key = f"{env_name}_{algo}"
            combo_data = all_results.get(combo_key, {})

            for method in METHODS:
                f1s, grs_list, bra_list = [], [], []
                n_rules_list, rule_lengths = [], []
                returns = []

                for ps in POLICY_SEEDS:
                    pk = f"ps{ps}_{method}"
                    if pk not in combo_data:
                        continue
                    group = combo_data[pk]
                    stab = group.get("group_stability", {})
                    runs = group.get("runs", [])
                    ok = [r for r in runs if r.get("status") == "success"]

                    for r in ok:
                        f1s.append(r.get("macro_f1", 0))
                        n_rules_list.append(r.get("n_rules", 0))
                        rule_lengths.append(r.get("mean_rule_length", 0))

                    if stab.get("GRS") is not None:
                        grs_list.append(stab["GRS"])
                    if stab.get("BRA") is not None:
                        bra_list.append(stab["BRA"])

                    # Behavioral
                    bk = f"ps{ps}_{method}"
                    be = behavior_data.get(combo_key, {}).get(bk, {})
                    if "mean_return" in be:
                        returns.append(be["mean_return"])

                if not f1s:
                    continue

                METHOD_DISPLAY = {
                    "cbs": "CBS", "b3_vote": "RV",
                    "dt": "DT", "b5_bdr": "BDR",
                }

                rows.append({
                    "Environment": env_name,
                    "Algorithm": algo.upper(),
                    "Method": METHOD_DISPLAY.get(method, method),
                    "Macro-F1": f"{np.mean(f1s):.3f}±{np.std(f1s):.3f}",
                    "GRS": f"{np.mean(grs_list):.3f}" if grs_list else "—",
                    "BRA": f"{np.mean(bra_list):.3f}" if bra_list else "—",
                    "Rule Count": f"{np.mean(n_rules_list):.1f}" if n_rules_list else "—",
                    "Mean Rule Len": f"{np.mean(rule_lengths):.1f}" if rule_lengths else "—",
                    "Return": f"{np.mean(returns):.2f}" if returns else "—",
                })

    df = pd.DataFrame(rows)
    table_path = _artifact_dir() / "tables" / "main_table.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_path, index=False)

    # Also save as markdown
    md_path = _artifact_dir() / "tables" / "main_table.md"
    try:
        df.to_markdown(md_path, index=False)
    except ImportError:
        print("Skipping markdown table export because tabulate is not installed.")

    print(f"\nMain table saved to {table_path}")
    print(df.to_string(index=False))
    return df


def generate_figures():
    """Generate all highway experiment figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures_dir = _artifact_dir() / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Display names for highway figures
    global _HW_LABELS
    _HW_LABELS = {
        "cbs": "CBS", "b3_vote": "RV",
        "dt": "DT", "b5_bdr": "BDR",
    }

    # Figure 1: F1 vs GRS scatter
    _fig_f1_vs_grs(figures_dir)
    # Figure 2: GRS vs BRA scatter
    _fig_grs_vs_bra(figures_dir)
    # Figure 3: Complexity extrapolation
    _fig_complexity_extrapolation(figures_dir)
    # Figure 4: Noise severity curves
    _fig_noise_severity(figures_dir)
    # Figure 5: Vehicles count ablation
    _fig_vehicles_ablation(figures_dir)

    print(f"\nAll figures saved to {figures_dir}")


def _fig_f1_vs_grs(figures_dir):
    """Figure 1: F1 vs GRS scatter plot."""
    import matplotlib.pyplot as plt

    results_path = _artifact_dir() / "explanation_results_raw_derived.json"
    if not results_path.exists():
        print("  [SKIP] fig_f1_vs_grs — no data")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    METHOD_COLORS = {
        "cbs": "#648FFF", "b3_vote": "#35A86B",
        "dt": "#DC3220", "b5_bdr": "#FE6100",
    }
    METHOD_MARKERS = {
        "cbs": "o", "b3_vote": "^", "dt": "D", "b5_bdr": "s",
    }
    METHOD_LABELS = {
        "cbs": "CBS", "b3_vote": "RV",
        "dt": "DT", "b5_bdr": "BDR",
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    for combo_key, combo_data in all_results.items():
        env_name, algo = combo_key.rsplit("_", 1)

        for method in METHODS:
            f1s_m, grs_m = [], []
            for ps in POLICY_SEEDS:
                pk = f"ps{ps}_{method}"
                if pk not in combo_data:
                    continue
                group = combo_data[pk]
                stab = group.get("group_stability", {})
                if stab.get("mean_f1") is not None and stab.get("GRS") is not None:
                    f1s_m.append(stab["mean_f1"])
                    grs_m.append(stab["GRS"])

            if f1s_m:
                label = f"{METHOD_LABELS.get(method, method)} ({env_name}/{algo})"
                ax.scatter(f1s_m, grs_m,
                          c=METHOD_COLORS.get(method, "gray"),
                          marker=METHOD_MARKERS.get(method, "o"),
                          label=label, alpha=0.7, s=40)

    ax.set_xlabel("Macro F1")
    ax.set_ylabel("GRS")
    ax.set_title("F1 vs GRS — Highway Environments")
    ax.legend(fontsize=6, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)

    fig.savefig(figures_dir / "fig_hw1_f1_vs_grs.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "fig_hw1_f1_vs_grs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  OK fig_hw1_f1_vs_grs")


def _fig_grs_vs_bra(figures_dir):
    """Figure 2: GRS vs BRA scatter plot."""
    import matplotlib.pyplot as plt

    results_path = _artifact_dir() / "explanation_results_raw_derived.json"
    if not results_path.exists():
        print("  [SKIP] fig_grs_vs_bra — no data")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    METHOD_COLORS = {
        "cbs": "#648FFF", "b3_vote": "#35A86B",
        "dt": "#DC3220", "b5_bdr": "#FE6100",
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    for combo_key, combo_data in all_results.items():
        for method in METHODS:
            grs_m, bra_m = [], []
            for ps in POLICY_SEEDS:
                pk = f"ps{ps}_{method}"
                if pk not in combo_data:
                    continue
                stab = combo_data[pk].get("group_stability", {})
                if stab.get("GRS") is not None and stab.get("BRA") is not None:
                    grs_m.append(stab["GRS"])
                    bra_m.append(stab["BRA"])

            if grs_m:
                ax.scatter(grs_m, bra_m,
                          c=METHOD_COLORS.get(method, "gray"),
                          label=f"{_HW_LABELS.get(method, method)} ({combo_key})", alpha=0.7, s=40)

    ax.set_xlabel("GRS")
    ax.set_ylabel("BRA")
    ax.set_title("GRS vs BRA — Structural vs Behavioral Stability")
    ax.legend(fontsize=6, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)

    fig.savefig(figures_dir / "fig_hw2_grs_vs_bra.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "fig_hw2_grs_vs_bra.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  OK fig_hw2_grs_vs_bra")


def _fig_complexity_extrapolation(figures_dir):
    """Figure 3: Complexity extrapolation across environments."""
    import matplotlib.pyplot as plt

    # Load existing results + highway results
    from figures._style import RESULTS_DIR

    env_order = ["MountainCar-v0", "CartPole-v1", "LunarLander-v3",
                 "MiniGrid", "merge-v0", "intersection-v0"]

    # Highway results
    results_path = _artifact_dir() / "explanation_results_raw_derived.json"
    if not results_path.exists():
        print("  [SKIP] fig_complexity_extrapolation — no data")
        return

    with open(results_path) as f:
        hw_results = json.load(f)

    # Aggregate highway results by env
    env_metrics = {}
    for combo_key, combo_data in hw_results.items():
        parts = combo_key.rsplit("_", 1)
        env_name = parts[0] if len(parts) == 2 else combo_key
        if env_name not in env_metrics:
            env_metrics[env_name] = {"f1": [], "grs": [], "bra": []}
        for pk, group in combo_data.items():
            stab = group.get("group_stability", {})
            if stab.get("mean_f1") is not None:
                env_metrics[env_name]["f1"].append(stab["mean_f1"])
            if stab.get("GRS") is not None:
                env_metrics[env_name]["grs"].append(stab["GRS"])
            if stab.get("BRA") is not None:
                env_metrics[env_name]["bra"].append(stab["BRA"])

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = ["f1", "grs", "bra"]
    titles = ["Macro F1", "GRS", "BRA"]

    hw_envs = [e for e in env_order if e in env_metrics]
    for ax, metric, title in zip(axes, metrics, titles):
        vals = [np.mean(env_metrics[e][metric]) if env_metrics[e][metric] else 0
                for e in hw_envs]
        errs = [np.std(env_metrics[e][metric]) if env_metrics[e][metric] else 0
                for e in hw_envs]
        ax.bar(range(len(hw_envs)), vals, yerr=errs, capsize=3,
               color=["#648FFF", "#35A86B"][:len(hw_envs)], alpha=0.8)
        ax.set_xticks(range(len(hw_envs)))
        ax.set_xticklabels(hw_envs, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Complexity Extrapolation — Highway Environments", fontsize=11)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_hw3_complexity_extrapolation.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "fig_hw3_complexity_extrapolation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  OK fig_hw3_complexity_extrapolation")


def _fig_noise_severity(figures_dir):
    """Figure 4: Noise severity curves."""
    import matplotlib.pyplot as plt

    noise_path = _artifact_dir() / "noise_severity_results.json"
    if not noise_path.exists():
        print("  [SKIP] fig_noise_severity — no data")
        return

    with open(noise_path) as f:
        all_results = json.load(f)

    METHOD_COLORS = {
        "cbs": "#648FFF", "b3_vote": "#35A86B",
        "dt": "#DC3220", "b5_bdr": "#FE6100",
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metric_keys = ["macro_f1", "GRS", "BRA"]
    metric_titles = ["Macro F1", "GRS", "BRA"]

    for ax, metric_key, metric_title in zip(axes, metric_keys, metric_titles):
        for method in METHODS:
            noise_values = {}
            for combo_data in all_results.values():
                for nk, result in combo_data.items():
                    if method not in nk or not isinstance(result, dict):
                        continue
                    noise_level = result.get("noise_level")
                    if noise_level is None:
                        continue
                    metric_value = _aggregate_group_metric(result, metric_key)
                    if metric_value is None:
                        continue
                    noise_values.setdefault(noise_level, []).append(metric_value)

            if noise_values:
                xs = sorted(noise_values.keys())
                ys = [np.mean(noise_values[x]) for x in xs]
                es = [np.std(noise_values[x]) for x in xs]
                ax.errorbar(
                    xs, ys, yerr=es,
                    label=_HW_LABELS.get(method, method),
                    color=METHOD_COLORS.get(method, "gray"),
                    marker="o", capsize=3,
                )

        ax.set_xlabel("Noise Level")
        ax.set_ylabel(metric_title)
        ax.set_title(metric_title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Noise Severity Curves", fontsize=11)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_hw4_noise_severity.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "fig_hw4_noise_severity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  OK fig_hw4_noise_severity")


def _fig_vehicles_ablation(figures_dir):
    """Figure 5: Vehicles count ablation."""
    import matplotlib.pyplot as plt

    vc_path = _artifact_dir() / "vehicles_count_ablation_results.json"
    if not vc_path.exists():
        print("  [SKIP] fig_vehicles_ablation — no data")
        return

    with open(vc_path) as f:
        all_results = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = ["macro_f1", "GRS", "BRA"]
    titles = ["Macro F1", "GRS", "BRA"]

    METHOD_COLORS = {
        "cbs": "#648FFF", "b3_vote": "#35A86B",
        "dt": "#DC3220", "b5_bdr": "#FE6100",
    }

    for ax, metric, title in zip(axes, metrics, titles):
        for method in METHODS:
            vc_vals = {}
            for vc_key, vc_data in all_results.items():
                # Parse vehicles_count from key (e.g., "merge-v0_dqn_vc4")
                vc = int(vc_key.split("_vc")[-1]) if "_vc" in vc_key else 6
                vals = []
                for nk, result in vc_data.items():
                    if method in nk and isinstance(result, dict):
                        v = _aggregate_group_metric(result, metric)
                        if v is not None:
                            vals.append(v)
                if vals:
                    vc_vals.setdefault(vc, []).extend(vals)

            if vc_vals:
                xs = sorted(vc_vals.keys())
                ys = [np.mean(vc_vals[x]) for x in xs]
                ax.plot(xs, ys, marker="o", label=_HW_LABELS.get(method, method),
                       color=METHOD_COLORS.get(method, "gray"))

        ax.set_xlabel("vehicles_count")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(VEHICLES_COUNTS_ABLATION)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Vehicles Count Ablation", fontsize=11)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_hw5_vehicles_ablation.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "fig_hw5_vehicles_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  OK fig_hw5_vehicles_ablation")


# =====================================================================
#  SUPPLEMENTARY ANALYSIS
# =====================================================================

def run_supplementary_analysis():
    """Generate supplementary analysis tables and data."""
    results_path = _artifact_dir() / "explanation_results_raw_derived.json"
    if not results_path.exists():
        print("No results for supplementary analysis.")
        return {}

    with open(results_path) as f:
        all_results = json.load(f)

    supp = {}

    # 16.1 Rare action analysis
    supp["rare_action"] = _supp_rare_action(all_results)

    # 16.2 Feature usage
    supp["feature_usage"] = _supp_feature_usage(all_results)

    # 16.3 Failure cases
    supp["failure_cases"] = _supp_failure_cases(all_results)

    # 16.4 Environment-level tables
    supp["env_tables"] = _supp_env_tables(all_results)

    # 16.5 Algorithm-level tables
    supp["algo_tables"] = _supp_algo_tables(all_results)

    _save_json(supp, _artifact_dir() / "supplementary_analysis.json")
    print("Supplementary analysis complete.")
    return supp


def _supp_rare_action(all_results):
    """Rare action analysis per method."""
    analysis = {}
    for combo_key, combo_data in all_results.items():
        combo_analysis = {}
        for pk, group in combo_data.items():
            runs = group.get("runs", [])
            for r in runs:
                if r.get("status") != "success":
                    continue
                ras = r.get("rare_action_support", {})
                paf1 = r.get("per_action_f1", {})
                pac = r.get("per_action_rule_count", {})
                combo_analysis[f"{pk}_es{r.get('explainer_seed', '?')}"] = {
                    "action_support": ras,
                    "per_action_f1": paf1,
                    "per_action_rule_count": pac,
                }
        analysis[combo_key] = combo_analysis
    return analysis


def _supp_feature_usage(all_results):
    """Feature usage frequency analysis."""
    analysis = {}
    for combo_key, combo_data in all_results.items():
        feature_counts = {}
        for pk, group in combo_data.items():
            for r in group.get("runs", []):
                if r.get("status") != "success" or "rules" not in r:
                    continue
                for rule in r["rules"]:
                    if isinstance(rule, dict):
                        for pred in rule.get("predicates", []):
                            fidx = pred.get("feature_idx", -1)
                            feature_counts[fidx] = feature_counts.get(fidx, 0) + 1

        # Sort by frequency
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        analysis[combo_key] = {
            "top_features": sorted_features[:20],
            "total_features_used": len(feature_counts),
        }
    return analysis


def _supp_failure_cases(all_results):
    """Identify failure cases: high F1 low GRS, low GRS high BRA, etc."""
    cases = {"high_f1_low_grs": [], "low_grs_high_bra": [], "cbs_collapse_b3_repair": []}

    for combo_key, combo_data in all_results.items():
        for pk, group in combo_data.items():
            stab = group.get("group_stability", {})
            mean_f1 = stab.get("mean_f1")
            grs = stab.get("GRS")
            bra = stab.get("BRA")

            if mean_f1 is not None and grs is not None:
                if mean_f1 > 0.7 and grs < 0.3:
                    cases["high_f1_low_grs"].append({
                        "combo": combo_key, "key": pk,
                        "f1": mean_f1, "grs": grs,
                    })

            if grs is not None and bra is not None:
                if grs < 0.3 and bra > 0.7:
                    cases["low_grs_high_bra"].append({
                        "combo": combo_key, "key": pk,
                        "grs": grs, "bra": bra,
                    })

    return cases


def _supp_env_tables(all_results):
    """Generate per-environment detailed tables."""
    tables = {}
    for env_name in ENVS:
        rows = []
        for algo in ALGOS:
            combo_key = f"{env_name}_{algo}"
            combo_data = all_results.get(combo_key, {})
            for pk, group in combo_data.items():
                stab = group.get("group_stability", {})
                rows.append({
                    "key": pk,
                    "algo": algo,
                    "mean_f1": stab.get("mean_f1"),
                    "std_f1": stab.get("std_f1"),
                    "GRS": stab.get("GRS"),
                    "BRA": stab.get("BRA"),
                    "n_runs": stab.get("n_runs"),
                    "mean_n_rules": stab.get("mean_n_rules"),
                })
        tables[env_name] = rows
    return tables


def _supp_algo_tables(all_results):
    """Generate per-algorithm detailed tables."""
    tables = {}
    for algo in ALGOS:
        rows = []
        for env_name in ENVS:
            combo_key = f"{env_name}_{algo}"
            combo_data = all_results.get(combo_key, {})
            for pk, group in combo_data.items():
                stab = group.get("group_stability", {})
                rows.append({
                    "key": pk,
                    "env": env_name,
                    "mean_f1": stab.get("mean_f1"),
                    "std_f1": stab.get("std_f1"),
                    "GRS": stab.get("GRS"),
                    "BRA": stab.get("BRA"),
                    "n_runs": stab.get("n_runs"),
                })
        tables[algo] = rows
    return tables


# =====================================================================
#  MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Highway-env experiment pipeline")
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["train", "replay", "explain", "noise",
                 "vehicles_ablation", "feature_ablation",
                 "behavior_eval", "statistics", "tables",
                 "figures", "supplementary", "all"],
        help="Which phase to run",
    )
    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument("--algos", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--explainer-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument(
        "--train-workers", type=int, default=1,
        help="Max concurrent seed workers for phase=train",
    )
    parser.add_argument(
        "--worker-torch-threads", type=int, default=None,
        help="Per-worker Torch/BLAS thread cap when --train-workers > 1",
    )
    parser.add_argument("--train-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    phase = args.phase
    t0 = time.time()

    phases_to_run = []
    if phase == "all":
        phases_to_run = [
            "train", "replay", "explain", "noise",
            "vehicles_ablation", "feature_ablation",
            "behavior_eval", "statistics", "tables",
            "figures", "supplementary",
        ]
    else:
        phases_to_run = [phase]

    for p in phases_to_run:
        print(f"\n{'#'*60}")
        print(f"  PHASE: {p.upper()}")
        print(f"{'#'*60}\n")

        if p == "train":
            if args.train_workers > 1 and not args.train_child:
                launch_parallel_training(
                    args.envs,
                    args.algos,
                    args.seeds,
                    max_workers=args.train_workers,
                    torch_threads_per_worker=args.worker_torch_threads,
                )
            else:
                train_all_policies(
                    args.envs,
                    args.algos,
                    args.seeds,
                    persist_combined=not args.train_child,
                )
        elif p == "replay":
            collect_all_replay(args.envs, args.algos, args.seeds)
        elif p == "explain":
            run_explanation_experiments(
                args.envs, args.algos, args.seeds,
                args.explainer_seeds, args.methods,
            )
        elif p == "noise":
            run_noise_severity(
                args.envs, args.algos, args.seeds,
                args.explainer_seeds, args.methods,
            )
        elif p == "vehicles_ablation":
            run_vehicles_count_ablation(
                args.envs, args.algos, args.seeds,
                args.explainer_seeds, args.methods,
            )
        elif p == "feature_ablation":
            run_feature_ablation(
                args.envs, args.algos, args.seeds,
                args.explainer_seeds, args.methods,
            )
        elif p == "behavior_eval":
            run_behavioral_evaluation(args.envs, args.algos, args.seeds, args.methods)
        elif p == "statistics":
            run_statistical_analysis()
        elif p == "tables":
            generate_main_table()
        elif p == "figures":
            generate_figures()
        elif p == "supplementary":
            run_supplementary_analysis()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
