#!/usr/bin/env python3
"""Estimate remaining wall-clock time for the reduced 3-seed highway training run."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
LOG_DQN = ARTIFACTS_DIR / "logs" / "highway_train_dqn.log"
LOG_PPO = ARTIFACTS_DIR / "logs" / "highway_train_ppo.log"
TRAINING_RESULTS = ARTIFACTS_DIR / "training_results.json"

TARGET_SEEDS = [0, 1, 2]
TARGET_ENVS = ["merge-v0", "intersection-v0"]
TARGET_ALGOS = ["dqn", "ppo"]

TRAINING_PROCESS_PATTERNS = {
    "dqn": r"run_highway_experiments\.py --phase train .* --algos dqn .*",
    "ppo_parent": r"run_highway_experiments\.py --phase train .* --algos ppo .* --train-workers 2 .*",
    "ppo_child": r"run_highway_experiments\.py --phase train .* --algos ppo .* --train-child",
}

TIME_PATTERNS = {
    ("merge-v0", "dqn"): re.compile(r"100% .*?\[\s*([0-9:]+)\s*<"),
    ("merge-v0", "ppo"): re.compile(r"100% .*?\[\s*([0-9:]+)\s*<"),
    ("intersection-v0", "dqn"): re.compile(r"100% .*?\[\s*([0-9:]+)\s*<"),
}


@dataclass
class ProcessInfo:
    pid: int
    etime_seconds: int
    command: str


def _run(command: str) -> str:
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        shell=True,
        check=False,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _parse_ps_etime(raw: str) -> int:
    raw = raw.strip()
    if not raw:
        return 0

    days = 0
    if "-" in raw:
        day_text, raw = raw.split("-", 1)
        days = int(day_text)

    parts = [int(p) for p in raw.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    else:
        hours = 0
        minutes = 0
        seconds = parts[0]

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _list_processes(pattern: str) -> list[ProcessInfo]:
    output = _run(f"pgrep -af '{pattern}'")
    processes = []
    for line in output.splitlines():
        if not line.strip():
            continue
        pid_text, command = line.split(" ", 1)
        etime_text = _run(f"ps -o etime= -p {pid_text}")
        processes.append(
            ProcessInfo(
                pid=int(pid_text),
                etime_seconds=_parse_ps_etime(etime_text),
                command=command.strip(),
            )
        )
    return processes


def _load_training_results() -> dict:
    if not TRAINING_RESULTS.exists():
        return {}
    with open(TRAINING_RESULTS) as handle:
        return json.load(handle)


def _completed_required_seeds(results: dict, env_name: str, algo: str) -> list[int]:
    done = []
    for seed in TARGET_SEEDS:
        key = f"{env_name}_{algo}_seed{seed}"
        payload = results.get(key)
        if (
            isinstance(payload, dict)
            and payload.get("status") == "success"
        ) or _policy_exists(env_name, algo, seed):
            done.append(seed)
    return done


def _policy_exists(env_name: str, algo: str, seed: int) -> bool:
    env_tag = env_name.replace("-", "_")
    model_path = ARTIFACTS_DIR / f"highway_{env_tag}" / algo / "policies" / f"policy_seed{seed}.zip"
    return model_path.exists()


def _duration_to_seconds(time_text: str) -> int:
    parts = [int(part) for part in time_text.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    else:
        hours = 0
        minutes = 0
        seconds = parts[0]
    return hours * 3600 + minutes * 60 + seconds


def _extract_completed_durations(log_path: Path, env_name: str) -> list[int]:
    if not log_path.exists():
        return []

    durations = []
    lines = log_path.read_text().splitlines()
    active_env = None
    for line in lines:
        match = re.search(r"Training (DQN|PPO) on ([^ ]+) seed=(\d+)", line)
        if match:
            active_env = match.group(2)
            continue

        if active_env != env_name:
            continue

        duration_match = re.search(r"100% .*?\[\s*([0-9:]+)\s*<", line)
        if duration_match:
            durations.append(_duration_to_seconds(duration_match.group(1)))

    return durations


def _format_seconds(total_seconds: float | int | None) -> str:
    if total_seconds is None:
        return "unknown"
    total_seconds = max(0, int(round(total_seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _estimate_dqn(results: dict) -> tuple[float, dict]:
    merge_completed = _completed_required_seeds(results, "merge-v0", "dqn")
    intersection_completed = _completed_required_seeds(results, "intersection-v0", "dqn")
    running_processes = _list_processes(TRAINING_PROCESS_PATTERNS["dqn"])

    merge_remaining = [seed for seed in TARGET_SEEDS if seed not in merge_completed]
    intersection_remaining = [seed for seed in TARGET_SEEDS if seed not in intersection_completed]

    merge_durations = _extract_completed_durations(LOG_DQN, "merge-v0")
    intersection_durations = _extract_completed_durations(LOG_DQN, "intersection-v0")
    avg_merge = sum(merge_durations[: len(merge_completed)]) / max(len(merge_completed), 1) if merge_completed else 0
    avg_intersection = (
        sum(intersection_durations[: len(intersection_completed)]) / max(len(intersection_completed), 1)
        if intersection_completed
        else (intersection_durations[0] if intersection_durations else 6 * 3600 + 20 * 60)
    )

    remaining_seconds = 0.0
    current_seed_elapsed = None

    if merge_remaining:
        remaining_seconds += avg_merge * len(merge_remaining)

    if intersection_remaining:
        if running_processes:
            completed_prior_seconds = sum(merge_durations) + sum(intersection_durations[: len(intersection_completed)])
            current_seed_elapsed = max(0, running_processes[0].etime_seconds - completed_prior_seconds)
            remaining_seconds += max(avg_intersection - current_seed_elapsed, 0)
            remaining_seconds += avg_intersection * max(len(intersection_remaining) - 1, 0)
        else:
            remaining_seconds += avg_intersection * len(intersection_remaining)

    details = {
        "completed": len(merge_completed) + len(intersection_completed),
        "target": 6,
        "merge_completed": merge_completed,
        "intersection_completed": intersection_completed,
        "current_seed_elapsed": _format_seconds(current_seed_elapsed),
        "avg_intersection_seed": _format_seconds(avg_intersection),
    }
    return remaining_seconds, details


def _estimate_ppo(results: dict) -> tuple[float, dict]:
    merge_completed = _completed_required_seeds(results, "merge-v0", "ppo")
    intersection_completed = [seed for seed in TARGET_SEEDS if _policy_exists("intersection-v0", "ppo", seed)]
    child_processes = _list_processes(TRAINING_PROCESS_PATTERNS["ppo_child"])

    merge_remaining = [seed for seed in TARGET_SEEDS if seed not in merge_completed]
    intersection_remaining = [seed for seed in TARGET_SEEDS if seed not in intersection_completed]

    merge_durations = _extract_completed_durations(LOG_PPO, "merge-v0")
    avg_ppo = (
        sum(merge_durations[: len(merge_completed)]) / max(len(merge_completed), 1)
        if merge_completed
        else (3 * 3600 + 28 * 60)
    )

    remaining_seconds = 0.0
    current_elapsed = sorted((proc.etime_seconds for proc in child_processes), reverse=True)

    if merge_remaining:
        remaining_seconds += avg_ppo * len(merge_remaining)

    if intersection_remaining:
        if child_processes:
            running_now = min(len(child_processes), len(intersection_remaining))
            current_batch_remaining = [max(avg_ppo - elapsed, 0) for elapsed in current_elapsed[:running_now]]
            remaining_seconds += max(current_batch_remaining) if current_batch_remaining else 0
            remaining_after_batch = len(intersection_remaining) - running_now
            if remaining_after_batch > 0:
                batch_width = max(1, len(child_processes))
                full_batches, partial = divmod(remaining_after_batch, batch_width)
                remaining_seconds += full_batches * avg_ppo
                if partial:
                    remaining_seconds += avg_ppo
        else:
            remaining_seconds += avg_ppo * len(intersection_remaining)

    details = {
        "completed": len(merge_completed) + len(intersection_completed),
        "target": 6,
        "merge_completed": merge_completed,
        "intersection_completed": intersection_completed,
        "running_children": [proc.pid for proc in child_processes],
        "avg_seed": _format_seconds(avg_ppo),
    }
    return remaining_seconds, details


def main():
    results = _load_training_results()
    dqn_remaining, dqn_details = _estimate_dqn(results)
    ppo_remaining, ppo_details = _estimate_ppo(results)
    total_remaining = max(dqn_remaining, ppo_remaining)
    eta_dt = datetime.now() + timedelta(seconds=total_remaining)

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "remaining_seconds": int(round(total_remaining)),
        "remaining_hms": _format_seconds(total_remaining),
        "eta_local": eta_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "dqn": {
            "remaining_seconds": int(round(dqn_remaining)),
            "remaining_hms": _format_seconds(dqn_remaining),
            **dqn_details,
        },
        "ppo": {
            "remaining_seconds": int(round(ppo_remaining)),
            "remaining_hms": _format_seconds(ppo_remaining),
            **ppo_details,
        },
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()