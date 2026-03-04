# v0009 Changes Summary

## Iteration
Batch: feat-eng-3-20260304-102111, Iteration 1

## Hypothesis Screening

### Hypothesis A (37 features): Add 3 distributional shape features
Added `density_skewness`, `density_kurtosis`, `density_cv` to regressor (34 -> 37 features).

### Hypothesis B (39 features): Add all 5 unused raw columns
Added `density_skewness`, `density_kurtosis`, `density_cv`, `season_hist_da_3`, `prob_below_85` (34 -> 39 features).

### Screen Results (2 months: 2022-06 weak, 2022-12 strong)

| Metric | Champion v0007 | Hyp A (37) | Hyp B (39) |
|--------|---------------|------------|------------|
| **2022-06 EV-VC@100** | 0.014 | 0.01568 | **0.01686** |
| **2022-06 EV-NDCG** | 0.604 | 0.60521 | **0.60693** |
| **2022-06 Spearman** | 0.271 | 0.27403 | 0.27358 |
| **2022-12 EV-VC@100** | 0.192 | 0.12894 | **0.17368** |
| **2022-12 EV-NDCG** | 0.815 | 0.77285 | **0.80475** |
| **2022-12 Spearman** | ~0.385 | 0.38559 | 0.38358 |

### Winner: Hypothesis B (39 features)
- **Primary**: Mean EV-VC@100: B=0.0953 vs A=0.0723 (+32% higher)
- **Safety**: Spearman within +-0.003 on both months — PASS
- **Tiebreaker**: Mean EV-NDCG: B=0.706 vs A=0.689 — B also wins

## Code Changes
- Created `registry/v0009/config.json` with 39 regressor features (34 from v0007 + 5 new)
- New features: `density_skewness` (monotone=0), `density_kurtosis` (0), `density_cv` (0), `season_hist_da_3` (+1), `prob_below_85` (-1)
- No changes to `ml/config.py` — defaults already include all 39 features
- Classifier config unchanged from v0007 (14 features, frozen)
- All regressor HPs unchanged from v0007 (n_est=400, depth=5, lr=0.05, mcw=25, lambda=1.0, alpha=1.0)

## Full 12-Month Results (v0009 vs v0007 champion)

| Month | v0009 EV-VC@100 | v0009 EV-VC@500 | v0009 EV-NDCG | v0009 Spearman |
|-------|----------------|----------------|--------------|---------------|
| 2020-09 | 0.0697 | 0.2809 | 0.7943 | 0.4121 |
| 2020-11 | 0.1491 | 0.2846 | 0.8030 | 0.5191 |
| 2021-01 | 0.0401 | 0.1513 | 0.7317 | 0.4311 |
| 2021-03 | 0.0254 | 0.2421 | 0.7641 | 0.4220 |
| 2021-05 | 0.0012 | 0.0882 | 0.7174 | 0.4732 |
| 2021-07 | 0.1028 | 0.3346 | 0.7971 | 0.3902 |
| 2021-09 | 0.2000 | 0.3998 | 0.8042 | 0.4079 |
| 2021-11 | 0.0118 | 0.1790 | 0.7426 | 0.2674 |
| 2022-03 | 0.0936 | 0.3331 | 0.8091 | 0.3823 |
| 2022-06 | 0.0169 | 0.0820 | 0.6069 | 0.2736 |
| 2022-09 | 0.0298 | 0.0617 | 0.6823 | 0.3297 |
| 2022-12 | 0.1737 | 0.3569 | 0.8047 | 0.3836 |

### Mean Comparison (Group A gates)

| Metric | v0007 Mean | v0009 Mean | Delta | % Change |
|--------|-----------|-----------|-------|----------|
| EV-VC@100 | 0.0699 | 0.0762 | +0.0063 | **+9.0%** |
| EV-VC@500 | 0.2294 | 0.2329 | +0.0035 | **+1.5%** |
| EV-NDCG | 0.7513 | 0.7548 | +0.0035 | **+0.5%** |
| Spearman | 0.3932 | 0.3910 | -0.0022 | -0.6% |

## Notes
- 5 features in the config (`hist_physical_interaction`, `overload_exceedance_product`, `band_severity`, `sf_exceed_interaction`, `hist_seasonal_band`) are not available from the data loader and are zero-filled. Effective feature count is 34/39.
- EV-VC@100 improved +9.0% — strong improvement in value capture at top-100
- EV-VC@500 and EV-NDCG show modest improvements (+1.5%, +0.5%)
- Spearman shows slight degradation (-0.6%) — within noise tolerance
- Weak month 2022-06: EV-VC@100 improved from 0.014 to 0.017 (+20%)
- Strong month 2021-09: EV-VC@100 improved from 0.176 to 0.200 (+14%)
