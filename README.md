# [Textual Rule Explanations of Reinforcement Learning Policies](https://drive.google.com/file/d/16vTKRqLzp4pDMtUG_kBP3fhTTi8TI6mL/view)

Code and results for a study of **how much textual if-then rule explanations of a
fixed RL policy change when you re-run the extraction**.

A rule extractor is run repeatedly on the same frozen policy while the replay data
and random choices are varied. Three things are measured separately:

| What is measured | Question it answers |
|---|---|
| **Fidelity** | Do the rules pick the same action as the policy? |
| **Rule overlap** | Do repeated runs produce the *same rules*? |
| **Prediction agreement** | Do repeated runs *predict* the same actions? |

The main finding is that these three disagree. A decision tree can match the policy
closely while rewriting its rules every run; voting can make predictions repeatable
without producing a single rule set. The repository also isolates *where* the
default consensus merge loses information (hard support filtering and median
interval merging) and shows that merged interval bounds can cover states where the
frozen policy chooses a different action.

## Quick start

```sh
git clone <repo-url>
cd Textual-Rule-Explanations-of-Reinforcement-Learning

# Large artifacts (highway policies, replays, big result files) use Git LFS
git lfs install && git lfs pull

# Windows
setup.bat
# Linux / macOS
./setup.sh
```

`setup.sh` / `setup.bat` create a `.venv`, install `requirements.txt`, and pick a
CPU or CUDA build of PyTorch. To install manually:

```sh
python -m venv .venv
.venv/Scripts/activate      # Windows
source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

Python 3.10+ is required. The stored results were produced with Python 3.12,
PyTorch 2.10, Stable-Baselines3 2.7.1, Gymnasium 1.2.3 and scikit-learn 1.8.0.

### Smallest end-to-end run

```sh
python reproduction/train_dqn.py --env CartPole-v1 --seed 42
python reproduction/collect_replay.py --env CartPole-v1 --seed 42 --episodes 20
python experiments/run_stress_test.py --env CartPole-v1
```

This trains a policy, collects a replay dataset, and runs the 21 extraction
settings for the clustering extractor. Results land in
`experiments/results/cartpole_v1/stress_test_results.json`.

All stored results are already committed, so you can also skip straight to the
tables and figures:

```sh
python experiments/generate_main_tables.py   # -> experiments/results/main_results.json
python -m figures.generate_all               # -> figures/*.pdf and figures/*.png
```

Full command list: [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Repository layout

```
reproduction/          policy training, replay collection, the clustering rule extractor
experiments/           extraction methods, perturbations, metrics, and run scripts
experiments/configs/   per-environment YAML settings
experiments/results/   committed JSON/CSV results
figures/               one script per figure, plus the generated PDF/PNG
artifacts/             the separate highway study (training, replay, metrics, tables)
```

### Core modules

| File | Contents |
|---|---|
| [reproduction/cbs.py](reproduction/cbs.py) | Clustering-based rule extractor (`CBSPipeline`): per-feature discretization, per-action K-means, weighted interval rules |
| [experiments/consensus_merge.py](experiments/consensus_merge.py) | Default consensus merge, plus the voting and importance-weighted variants |
| [experiments/soft_support_merge.py](experiments/soft_support_merge.py) | SoftSupport merge: partial group support instead of a hard yes/no vote |
| [experiments/decision_tree_surrogate.py](experiments/decision_tree_surrogate.py) | Decision-tree surrogate baseline |
| [experiments/boolean_rules.py](experiments/boolean_rules.py) | Boolean decision rules baseline, with optional policy fallback |
| [experiments/rule_matching.py](experiments/rule_matching.py) | Canonical rule form, rule similarity, exact and near-match overlap |
| [experiments/perturbations.py](experiments/perturbations.py) | Seed shift, uniform and stratified subsampling, cluster count, feature noise |

### Method names

The prose and the code use slightly different names. Result files keep short keys
so that stored JSON stays stable:

| Method | Key in result files | Script |
|---|---|---|
| Clustering | `cbs` | [experiments/run_stress_test.py](experiments/run_stress_test.py) |
| Tuned clustering | `cbs_maxf1` | [experiments/run_stress_test.py](experiments/run_stress_test.py) |
| Default merge | `consensus_cbs` | [experiments/run_consensus_merge.py](experiments/run_consensus_merge.py) |
| Rule-set voting | `b3_vote` | [experiments/run_consensus_merge.py](experiments/run_consensus_merge.py) |
| Decision tree | `b4_dt` | [experiments/run_decision_tree.py](experiments/run_decision_tree.py) |
| Boolean + policy | `b5_bdr` | [experiments/run_boolean_rules.py](experiments/run_boolean_rules.py) |
| Tuned merge | `tuned_merge` | [experiments/run_tuned_merge.py](experiments/run_tuned_merge.py) |
| Soft support | `soft_support` | [experiments/run_soft_support_sweep.py](experiments/run_soft_support_sweep.py) |
| Weighted voting | `weighted_vote` | [experiments/run_weighted_vote.py](experiments/run_weighted_vote.py) |

Metric keys inside the JSON files: `E_F1` is fidelity, `GRS`
(`GRS_weighted_jaccard`) is exact rule overlap, `GRS_TA`
(`GRS_threshold_aware`) is near-match rule overlap, `TD` is threshold drift,
and `BRA` is prediction agreement.

## Experimental design

Four frozen policies, all trained with seed 42:

| Environment | Algorithm | Features | Actions | Steps |
|---|---|---|---|---|
| MountainCar-v0 | DQN | 2 | 3 | 300k |
| CartPole-v1 | DQN | 4 | 2 | 100k |
| LunarLander-v3 | DQN | 8 | 4 | 500k |
| MiniGrid-Dynamic-Obstacles-8x8-v0 | PPO | 14 | 3 | 1M |

Each classic-control extraction is repeated over 18 shared settings: five replay
seeds, five uniform subsamples of 8,000 states, five action-stratified subsamples,
and feature noise at 0.01, 0.03 and 0.05 of each feature's replay range. Base
replays hold 10,000 transitions; a separate 5,000-state replay from seed 99 is
used for held-out evaluation.

## Main results

Averages over the 18 shared settings, from
`experiments/results/main_results.json`. Each cell reads
*fidelity / exact rule overlap / prediction agreement*:

| Method | MountainCar | CartPole | LunarLander |
|---|---|---|---|
| Clustering | .743 / **.504** / .910 | .794 / **.154** / .773 | .542 / .128 / .658 |
| Tuned clustering | .745 / .438 / .889 | .776 / .093 / .741 | .545 / .080 / .616 |
| Default merge | .548 / .239 / .569 | .624 / .065 / .533 | .410 / .085 / .431 |
| Rule-set voting | .755 / — / .929 | .810 / — / .818 | .562 / — / .672 |
| Decision tree | **.947** / .049 / **.980** | **.922** / .004 / .914 | **.721** / .003 / **.767** |
| Boolean + policy | .470 / .012 / .970 | .691 / .007 / **.992** | .314 / **.163** / .589 |

Rule-set voting keeps all component rule sets, so exact rule overlap is undefined
for it. No method wins on every column: the decision tree has the highest fidelity
and the lowest rule overlap, and the Boolean rules keep predictions stable mostly
through their policy fallback.

Paired bootstrap intervals (1,000 resamples) for these comparisons are in
`experiments/results/statistical_tests.json`. Not every fidelity contrast is
significant.

## Where the default merge loses information

[experiments/run_merge_stage_study.py](experiments/run_merge_stage_study.py)
splits the merge into five outputs from the same matched groups: matching only,
hard support, median interval bounds, the full default merge, and soft support.
Both hard support and median bounds lower fidelity, and the effect differs by
environment.

[experiments/run_boundary_crossing.py](experiments/run_boundary_crossing.py)
samples same-action rule pairs, draws states covered by each rule, and queries the
frozen policy along the straight path between them. Some pairs above the
similarity threshold still cross a policy boundary in every environment, and paths
through sparse replay regions cross more often. This is why a coordinate-wise
median box can cover states that no member rule covered.

## License

[MIT](LICENSE)
