# Condition-Characterization Monotonicity Analysis

**Date**: 2026-04-02  
**Status**: Complete

## Unit of Analysis

- **Analysis A** uses A3 **mergeable rule pairs** from `boundary_crossing.json`.
- **Analysis B** also uses A3 **mergeable rule pairs**.

Total available boundary-crossing pairs = 300 (100 per environment). Final analysis unit for both analyses = **150 mergeable pairs** (50 per environment).

## Predictor Definitions

| Predictor | Source field | Definition |
|-----------|-------------|------------|
| Path low-density fraction | `path_low_density_frac` | Fraction of all interpolation points along sampled pairwise paths that fall in low-density replay regions |
| Midpoint low-density rate | `midpoint_low_density_rate` | Fraction of sampled paths whose midpoint (α = 0.5) lies in a low-density replay region |
| Rule dissimilarity | `1 − similarity` | One minus the threshold-aware rule similarity score; a weak proxy for geometric mismatch / separation |

**Schema note**: Distortion groups and boundary-crossing pairs do not share an explicit join key. We therefore use the strongest non-circular fallback directly supported by A3 pair-level outputs. The low-density predictors still capture the same geometric phenomenon emphasized in A2: sparse bridging between regions that the merge heuristic treats as compatible.

## Outcome Definitions

| Analysis | Outcome | Definition | Threshold |
|----------|---------|------------|-----------|
| A | Boundary crossing | Whether at least one frozen-policy action flip occurs along the interpolation path between a mergeable pair | `boundary_crossing_rate > 0` |
| B | Post-merge failure | Whether the merged rule disagrees with the frozen policy at the interpolation midpoint on more than half of sampled paths | `midpoint_action_mismatch_rate > 0.5` |

## Non-Circularity Statement

This analysis is fully non-circular:

1. A2's `is_failed_merge` label is never used as an outcome.
2. Analysis A predicts **boundary crossing**, not the A2 failed/successful merge label.
3. Analysis B uses an **independent** failure definition based on midpoint rule--policy disagreement, again distinct from A2's group-level action-mismatch threshold.

## Main Findings

> *Paper-ready wording (cautious):*  
> We further convert the A2/A3 mechanism diagnosis into a conditional trend analysis over mergeable pairs. Larger low-density exposure is associated with higher probabilities of crossing the frozen policy's decision boundary: pooled boundary-crossing probability rises from 0.74 in the lowest quartile of path low-density exposure to 1.00 in the top two quartiles (Spearman ρ = 0.36, p < 10⁻⁵), with a similar pattern for midpoint low-density rate (ρ = 0.34, p < 10⁻⁴). By contrast, raw rule dissimilarity is weak and not significant (ρ = 0.04, p = 0.60). In turn, boundary-crossing pairs are substantially more likely to yield poor post-merge outcomes: among mergeable pairs, the post-merge failure rate is 0/12 for non-crossing pairs versus 42.7% for crossing pairs (Fisher's exact p = 0.003). These results strengthen the interpretation that DCM failure is linked not merely to threshold choice, but to a mismatch between the merge heuristic and the policy's local decision geometry.

Additional detail:

- **Analysis A**: low-density signals are the dominant predictors.
	- Path low-density fraction: ρ = 0.3605, p = 5.9 × 10⁻⁶; quartile trend 0.74 → 0.95 → 1.00 → 1.00.
	- Midpoint low-density rate: ρ = 0.3392, p = 2.2 × 10⁻⁵; quartile trend 0.74 → 0.98 → 0.97 → 1.00.
	- Rule dissimilarity: ρ = 0.0434, p = 0.598; no reliable monotone trend.
- **Analysis B**: pooled failure probability is 0.00 [0.00, 0.24] without boundary crossing versus 0.43 [0.35, 0.51] with boundary crossing.
- Per-environment Analysis B is directionally consistent across all three environments and reaches significance on LunarLander (`p = 0.005`), but MountainCar and CartPole are underpowered because the no-crossing subset is very small.

## Caveats

- Near-ceiling boundary-crossing rates in the upper low-density bins limit dynamic range.
- The no-crossing subgroup in Analysis B is small (`n = 12` pooled).
- Rule dissimilarity is a much weaker signal than the density-based predictors and should not be overinterpreted.

## Output Files

| File | Content |
|------|---------|
| `experiments/results/condition_monotonicity_summary.json` | Full analysis results (pooled + per-environment) |
| `experiments/results/condition_monotonicity_summary.csv` | Binned probability table |
| `figures/fig_condition_monotonicity.py` | Main-text figure script |
| `figures/fig_condition_monotonicity.pdf` | Vector figure for paper |
| `figures/fig_condition_monotonicity.png` | Raster preview |
| `experiments/results/condition_monotonicity_note.md` | This note |
