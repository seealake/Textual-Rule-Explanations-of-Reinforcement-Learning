#!/usr/bin/env python
"""Generate SHA256 traceability manifest for transfer artifacts."""
import hashlib
import os
import json
import datetime

FILES = [
    "experiments/results/minigrid_dynamic_obstacles_8x8_v0/feasibility_check.json",
    "experiments/results/minigrid_dynamic_obstacles_8x8_v0/stress_test_results.json",
    "experiments/results/cross_algo_comparison/ppo_cartpole_stress.json",
    "experiments/results/cross_algo_comparison/ppo_vs_dqn_comparison.json",
    "experiments/results/minigrid_dynamic_obstacles_8x8_v0_training_ppo.json",
    "experiments/results/cartpole_v1_training_ppo.json",
    "reproduction/models/ppo_minigrid_dynamic_obstacles_8x8_v0.zip",
    "reproduction/models/ppo_cartpole_v1.zip",
    "reproduction/data/replay_minigrid_dynamic_obstacles_8x8_v0_ppo_seed42.npz",
    "reproduction/data/replay_cartpole_v1_ppo_seed42.npz",
    "figures/fig_minigrid.pdf",
    "figures/fig_minigrid.png",
    "figures/fig_complexity.pdf",
    "figures/fig_complexity.png",
    "figures/fig_policy_family_cartpole.pdf",
    "figures/fig_policy_family_cartpole.png",
]

manifest = {
    "schema_version": "traceability_v1",
    "generated": datetime.datetime.now().isoformat(),
    "phase": "External Validity Extension",
    "reproduction_commands": [
        "python reproduction/train_policy.py --env MiniGrid-Dynamic-Obstacles-8x8-v0 --algo ppo --timesteps 1000000 --seed 42",
        "python reproduction/train_policy.py --env CartPole-v1 --algo ppo --timesteps 200000 --seed 42",
        "python experiments/run_minigrid_experiments.py --step feasibility --algo ppo",
        "python experiments/run_minigrid_experiments.py --step stress --algo ppo",
        "python experiments/run_cross_algo_comparison.py",
        "python figures/fig_minigrid.py",
        "python figures/fig_complexity.py",
        "python figures/fig_policy_family_cartpole.py",
    ],
    "artifacts": [],
}

for fp in FILES:
    if os.path.exists(fp):
        with open(fp, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        size = os.path.getsize(fp)
        manifest["artifacts"].append({
            "path": fp.replace(os.sep, "/"),
            "sha256": sha,
            "size_bytes": size,
        })
        print(f"  {sha[:16]}  {size:>10d}  {fp}")
    else:
        print(f"  MISSING: {fp}")

out = "experiments/results/transfer_manifest.json"
with open(out, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nManifest saved to {out} ({len(manifest['artifacts'])} artifacts)")
