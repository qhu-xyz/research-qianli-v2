# v0008 Changes Summary

## Screening Results

### Hypothesis A (37 features: +density_skewness, density_kurtosis, density_cv)
| Month | EV-VC@100 | EV-VC@500 | EV-NDCG | Spearman |
|-------|-----------|-----------|---------|----------|
| 2022-06 | 0.01568 | 0.06549 | 0.60521 | 0.27403 |
| 2022-12 | 0.12894 | 0.35683 | 0.77285 | 0.38559 |

### Hypothesis B (39 features: +density_skewness, density_kurtosis, density_cv, season_hist_da_3, prob_below_85)
| Month | EV-VC@100 | EV-VC@500 | EV-NDCG | Spearman |
|-------|-----------|-----------|---------|----------|
| 2022-06 | 0.01686 | 0.08196 | 0.60693 | 0.27358 |
| 2022-12 | 0.17368 | 0.35693 | 0.80475 | 0.38358 |

### Winner: Hypothesis B

**Reason**: Hypothesis A failed the EV-NDCG safety check on 2022-12 (dropped 0.042 > 0.03 threshold vs champion's 0.815). Hypothesis B passed all safety checks and had higher mean EV-VC@100 (0.0953 vs 0.0723).

## Code Changes

- `ml/config.py`: Added 5 features to `_ALL_REGRESSOR_FEATURES` (34 -> 39):
  - `density_skewness` (monotone=0)
  - `density_kurtosis` (monotone=0)
  - `density_cv` (monotone=0)
  - `season_hist_da_3` (monotone=+1)
  - `prob_below_85` (monotone=-1)
- `ml/tests/test_config.py`: Updated feature count assertion (34 -> 39)
- `ml/tests/test_data_loader.py`: Updated feature count assertion (34 -> 39)

No changes to `ml/features.py`, `ml/train.py`, `ml/pipeline.py`, or `ml/evaluate.py` -- all added features are raw MisoDataLoader columns.

Note: 5 of 39 regressor features are consistently unavailable in the data loader (`hist_physical_interaction`, `overload_exceedance_product`, `band_severity`, `sf_exceed_interaction`, `hist_seasonal_band`). These are filled with zeros by the pipeline. This is a pre-existing condition from v0007.

## Full 12-Month Benchmark Results (v0008)

| Metric | Mean | Champion v0007 Mean | Delta |
|--------|------|---------------------|-------|
| EV-VC@100 | 0.0762 | 0.0699 | +9.0% |
| EV-VC@500 | 0.2329 | 0.2294 | +1.5% |
| EV-NDCG | 0.7548 | 0.7513 | +0.5% |
| Spearman | 0.3910 | 0.3932 | -0.6% |

### Per-Month Key Metrics
| Month | EV-VC@100 | EV-VC@500 | EV-NDCG | Spearman |
|-------|-----------|-----------|---------|----------|
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
