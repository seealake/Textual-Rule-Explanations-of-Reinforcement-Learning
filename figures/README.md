# Figures

One script per figure. Each writes `{name}.pdf` (vector) and `{name}.png`
(300 dpi preview) next to the script.

```sh
python -m figures.generate_all            # everything
python -m figures.generate_all --list     # available names
python -m figures.generate_all overview drift
```

Rendering uses the PGF backend with pdflatex when a LaTeX installation is
available, and falls back to matplotlib mathtext otherwise. Shared styling
(colours, sizes, method labels) lives in [_style.py](_style.py); the palette is
the IBM colorblind-safe set.

| Script | Shows | Reads |
|---|---|---|
| `fig_overview.py` | Conceptual overview: measures disagree, and a median box can cross a boundary | `results/main_results.json` |
| `fig_merge_mechanism.py` | Merge stages, boundary crossing, and midpoint mismatch | `failure_decomposition.json`, `boundary_crossing.json`, `condition_monotonicity_summary.json` |
| `fig_predicate_drift.py` | Rule counts and predicate midpoints across replay seeds | `<env>/stress_test_results.json` |
| `fig_noise_severity.py` | Fidelity and agreement as noise increases | `<env>/noise_severity_results.json` |
| `fig_fidelity_vs_stability.py` | Fidelity vs agreement scatter per run | `stress_test_results.json`, `consensus_merge_results.json`, `decision_tree_results.json` |
| `fig_pareto_frontier.py` | Near-match overlap vs fidelity | same as above |
| `fig_consensus_ablation.py` | Ensemble size B crossed with support threshold tau | `<env>/consensus_merge_results.json` |
| `fig_env_perturbation.py` | Return vs agreement under environment perturbation | `<env>/env_perturbation_results.json` |
| `fig_local_consistency.py` | Local explanation consistency curves | `lec_results.json`, `noise_severity_results.json` |
| `fig_minigrid.py` | MiniGrid main comparison | `minigrid_.../stress_test_results.json` |
| `fig_minigrid_transfer.py` | MiniGrid + PPO transfer of the merge failure | `minigrid_.../external_validity.json` |
| `fig_complexity.py` | Stability against rule complexity across environments | `results/suite_summary/` |
| `fig_policy_family_cartpole.py` | PPO vs DQN on CartPole | `cross_algo_comparison/ppo_cartpole_stress.json` |
| `fig_policy_family_lunarlander.py` | PPO vs DQN on LunarLander | `cross_algo_comparison/ppo_lunarlander_stress.json` |
| `fig_ppo_vs_dqn.py` | Cross-environment PPO vs DQN summary | both of the above |
| `fig_suite_summary.py` | Compact summary of the whole robustness suite | `results/suite_summary/` |
| `fig_merge_stage_decomposition.py` | Which merge stage loses fidelity | `<env>/failure_decomposition.json` |
| `fig_merge_stage_by_env.py` | Failure modes broken down by environment | `failure_decomposition.json`, `geometric_distortion.json`, `boundary_crossing.json`, `external_validity.json` |
| `fig_geometric_distortion.py` | Distortion of merged boxes, failed vs successful | `<env>/geometric_distortion.json` |
| `fig_condition_monotonicity.py` | Monotonicity of rule conditions | `condition_monotonicity_summary.json` |
| `fig_boundary_crossing.py` | Boundary-crossing rates by similarity band | `<env>/boundary_crossing.json` |
| `fig_boundary_case_study.py` | Individual crossing rule pairs | `<env>/boundary_crossing.json` |
| `fig_tree_structure.py` | Decision-tree size, depth and stability (3 outputs) | `<env>/tree_depth_ablation.json` |
| `fig_soft_support_ablation.py` | SoftSupport ablation heatmap | `results/soft_support_merge/raw/` |
| `fig_soft_support_pareto.py` | SoftSupport Pareto frontier | `results/soft_support_merge/raw/` |
| `fig_merge_repairs.py` | Merge-repair study (6 outputs) | `results/merge_stages/`, `results/soft_support_sweep/` |
| `fig_weighted_vote.py` | Weighted voting comparison (3 outputs) | `results/weighted_vote/` |

Paths are relative to `experiments/results/`; `<env>` is one of
`mountaincar_v0`, `cartpole_v1`, `lunarlander_v3`.
