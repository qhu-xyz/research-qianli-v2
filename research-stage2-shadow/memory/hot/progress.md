## Status: ITER 1 PLANNED — feat-eng-3-20260304-102111
**Champion**: v0007 (reg_lambda=1.0, mcw=25) — from batch ralph-v2
**Batch constraint**: Feature engineering / selection ONLY (no HP changes)

### Iteration 1 Plan
- **Hypothesis A**: Add 3 distributional shape features (density_skewness, density_kurtosis, density_cv) → 37 features
- **Hypothesis B**: Add 3 distributional + season_hist_da_3 + prob_below_85 → 39 features
- **Screen months**: 2022-06 (weak), 2022-12 (strong)
- **Direction file**: `memory/direction_iter1.md`
- **Status**: Awaiting worker execution
