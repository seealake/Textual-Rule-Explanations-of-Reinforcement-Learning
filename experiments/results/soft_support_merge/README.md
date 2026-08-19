# SoftSupport merge sweep

Produced by:

```sh
python experiments/run_soft_support_merge.py --env all
python experiments/run_soft_support_merge.py --env CartPole-v1
```

Layout:

- `raw/` — per-environment JSON with all run details
- `tables/` — summary CSV and Markdown tables
- `logs/` — configuration snapshot
- `statistical_tests.json` — paired bootstrap output

Settings swept:

- B = 5 internal subsamples, 5 outer repeats (seed shift)
- matching threshold rho = 0.9, support threshold tau = 0.7
- lambda_B in {0.0, 0.1, 0.2}
- support mode in {hard, soft}
- rare-action safeguard in {off, on}
