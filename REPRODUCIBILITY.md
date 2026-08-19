# Reproducibility

Every command below is run from the repository root with the virtual environment
active. All stored results are committed, so any single step can be run on its own
without re-running the ones before it.

Seeds are pinned: policies use training seed 42, extraction settings are derived
deterministically in [experiments/generate_perturbations.py](experiments/generate_perturbations.py),
and held-out evaluation uses a separate seed-99 replay.

Small numerical differences (about 0.005 on the overlap scores) can appear across
hardware because of floating-point differences inside K-means. Use
`experiments/results/statistical_tests.json` for the exact intervals.

## 0. Clone and fetch large files

```sh
git lfs install
git lfs pull
```

## 1. Policies and replay data

```sh
python reproduction/train_dqn.py --env MountainCar-v0 --seed 42
python reproduction/train_dqn.py --env CartPole-v1 --seed 42
python reproduction/train_dqn.py --env LunarLander-v3 --seed 42

python reproduction/collect_replay.py --env MountainCar-v0 --seed 42 --episodes 100
python reproduction/collect_replay.py --env CartPole-v1 --seed 42 --episodes 20
python reproduction/collect_replay.py --env LunarLander-v3 --seed 42 --episodes 100
```

## 2. Main comparison

Each run covers 21 extraction settings per method per environment: 5 replay seeds,
5 uniform subsamples, 5 stratified subsamples, 3 cluster counts, 3 noise levels.
The main tables use the 18 settings that all methods share.

```sh
# Clustering and tuned clustering
python experiments/run_stress_test.py --env MountainCar-v0
python experiments/run_stress_test.py --env CartPole-v1
python experiments/run_stress_test.py --env LunarLander-v3

# Default merge and rule-set voting
python experiments/run_consensus_merge.py --env MountainCar-v0
python experiments/run_consensus_merge.py --env CartPole-v1
python experiments/run_consensus_merge.py --env LunarLander-v3

# Decision-tree surrogate
python experiments/run_decision_tree.py --env MountainCar-v0
python experiments/run_decision_tree.py --env CartPole-v1
python experiments/run_decision_tree.py --env LunarLander-v3

# Boolean decision rules
python experiments/run_boolean_rules.py --env MountainCar-v0
python experiments/run_boolean_rules.py --env CartPole-v1
python experiments/run_boolean_rules.py --env LunarLander-v3
```

Outputs, per environment tag (`mountaincar_v0`, `cartpole_v1`, `lunarlander_v3`):

- `experiments/results/<env>/stress_test_results.json`
- `experiments/results/<env>/consensus_merge_results.json`
- `experiments/results/<env>/decision_tree_results.json`
- `experiments/results/<env>/boolean_rule_results.json`

Aggregate them into the main table:

```sh
python experiments/generate_main_tables.py
```

Output: `experiments/results/main_results.json`.

## 3. Where the merge loses information

```sh
python experiments/run_failure_decomposition.py     # merge stages
python experiments/run_geometric_distortion.py      # distortion of merged boxes
python experiments/run_boundary_crossing.py         # policy queries along rule paths
python experiments/run_merge_stage_study.py         # combined stage study per environment
```

Outputs: `experiments/results/<env>/{failure_decomposition,geometric_distortion,boundary_crossing}.json`
and `experiments/results/merge_stages/<env>/*.json`.

## 4. Merge variants

```sh
python experiments/run_tuned_merge.py               # tuned merge, 10 repeats
python experiments/run_soft_support_sweep.py        # soft support, 12-cell sweep
python experiments/run_merge_statistics.py          # paired bootstrap + summary tables
python experiments/run_soft_support_merge.py --env all
python experiments/run_soft_support_statistics.py
```

Outputs: `experiments/results/tuned_merge/<env>/`,
`experiments/results/soft_support_sweep/<env>/`,
`experiments/results/merge_statistics/`, and
`experiments/results/soft_support_merge/`.

## 5. Weighted voting

```sh
python experiments/run_weighted_vote.py
python experiments/run_weighted_vote_statistics.py
```

Outputs: `experiments/results/weighted_vote/<env>/main_comparison.json`,
`experiments/results/weighted_vote/{cartpole_v1,lunarlander_v3}/b_sensitivity.json`,
`experiments/results/weighted_vote_statistics/`.

The ensemble-size sweep is skipped for MountainCar: weighted voting improved
fidelity there by less than 0.01, so the sweep would not be informative.

## 6. MiniGrid and cross-algorithm checks

```sh
python experiments/run_minigrid_experiments.py
python experiments/run_external_validity.py
python experiments/run_cross_algo_comparison.py       # PPO vs DQN on CartPole
python experiments/run_cross_algo_comparison_ll.py    # PPO vs DQN on LunarLander
python experiments/generate_transfer_manifest.py
```

## 7. Additional analyses

```sh
python experiments/run_algorithmic_randomness.py --env MountainCar-v0   # K-means seed and cluster count
python experiments/run_cross_policy.py --env all                        # other policy-training seeds
python experiments/run_rare_action_sweep.py --env LunarLander-v3        # rare-action quota sampling
python experiments/run_tree_depth_ablation.py --env all                 # decision-tree depth limits
python experiments/run_lec.py --env all                                 # local explanation consistency
python experiments/run_noise_severity_sweep.py --env all
python experiments/run_env_perturbation.py --env MountainCar-v0
python experiments/run_matching_robustness.py --env CartPole-v1
python experiments/run_match_threshold_check.py
python experiments/run_semantic_merge.py
python experiments/run_aggregation_comparison.py
python experiments/analyze_correlations.py --stability-source run_family
python experiments/analyze_condition_monotonicity.py
python experiments/sanity_check_metrics.py
```

## 8. Highway study

The highway study lives under `artifacts/` and covers `merge-v0` and
`intersection-v0` with DQN and PPO, policy seeds 0-2, and the methods
`cbs`, `b3_vote`, `dt`, `b5_bdr`.

Policies and replay data for seeds 0-2 are committed, so a fresh clone can start
from `--phase explain`:

- `artifacts/highway_*/{dqn,ppo}/policies/policy_seed[0-2].zip`
- `artifacts/highway_*/{dqn,ppo}/replay/*policy_seed[0-2]_replay.npz`
- `artifacts/highway_*/{dqn,ppo}/replay/*policy_seed[0-2]_metadata.json`

```sh
ENVS="merge-v0 intersection-v0"
ALGOS="dqn ppo"
METHODS="cbs b3_vote dt b5_bdr"
XSEEDS="0 1 2 3 4 5 6 7 8 9"

python experiments/run_highway_experiments.py --phase train   --envs $ENVS --algos $ALGOS --seeds 0 1 2
python experiments/run_highway_experiments.py --phase replay  --envs $ENVS --algos $ALGOS --seeds 0 1 2
python experiments/run_highway_experiments.py --phase explain --envs $ENVS --algos $ALGOS --seeds 0 1 2 --methods $METHODS --explainer-seeds $XSEEDS
python experiments/run_highway_experiments.py --phase behavior_eval     --envs $ENVS --algos $ALGOS --seeds 0 1 2 --methods $METHODS
python experiments/run_highway_experiments.py --phase noise             --envs $ENVS --algos $ALGOS --seeds 0 1 2 --methods $METHODS --explainer-seeds $XSEEDS
python experiments/run_highway_experiments.py --phase vehicles_ablation --envs $ENVS --algos $ALGOS --seeds 0 1 2 --methods $METHODS --explainer-seeds $XSEEDS
python experiments/run_highway_experiments.py --phase feature_ablation  --envs $ENVS --algos $ALGOS --seeds 0 1 2 --methods $METHODS --explainer-seeds $XSEEDS
python experiments/run_highway_experiments.py --phase statistics
python experiments/run_highway_experiments.py --phase tables
python experiments/run_highway_experiments.py --phase figures
python experiments/run_highway_experiments.py --phase supplementary
```

Outputs: `artifacts/highway_traceability_manifest.json`,
`artifacts/tables/main_table.csv`, `artifacts/behavioral_evaluation_results.json`,
`artifacts/statistical_analysis.json`.

## 9. Statistics, tables and figures

```sh
python experiments/run_statistical_tests.py       # -> experiments/results/statistical_tests.json
python experiments/generate_summary_tables.py     # -> experiments/results/summary_tables.{json,txt}
python experiments/generate_suite_summary.py      # -> experiments/results/suite_summary/
python -m figures.generate_all                    # -> figures/*.pdf and figures/*.png
```

`python -m figures.generate_all --list` prints the available figure names; pass
one or more names to regenerate only those.

The `B5_tau*` keys in `summary_tables.json` are the ensemble size B crossed with
support threshold tau; they are unrelated to the Boolean decision rules baseline,
whose results live in `experiments/results/<env>/boolean_rule_results.json`.
