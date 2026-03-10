# v0011 Changes Summary — Prune 5 Zero-Filled Features (39→34)

## Screening Winner: Hypothesis A (prune only)

### Hypotheses Tested
- **A (winner)**: Remove 5 features that are always zero (`hist_physical_interaction`, `overload_exceedance_product`, `band_severity`, `sf_exceed_interaction`, `hist_seasonal_band`). Reduces regressor from 39 → 34 features.
- **B**: Same pruning + add `flow_direction`. Would go from 39 → 35 features.

### Screen Results (2 months: 2022-09 weak, 2021-09 strong)

| Metric | Month | Champion (v0009) | Hyp A (34 feat) | Hyp B (35 feat) |
|--------|-------|-----------------|------------------|------------------|
| EV-VC@100 | 2022-09 | 0.0298 | 0.0280 (-6.1%) | 0.0276 (-7.2%) |
| EV-VC@100 | 2021-09 | 0.2000 | 0.2352 (+17.6%) | 0.2199 (+10.0%) |
| EV-VC@500 | 2022-09 | 0.0617 | 0.0527 (-14.6%) | 0.0602 (-2.5%) |
| EV-VC@500 | 2021-09 | 0.3998 | 0.3964 (-0.9%) | 0.3965 (-0.8%) |
| Spearman | 2022-09 | 0.3297 | 0.3284 (-0.4%) | 0.3316 (+0.6%) |
| Spearman | 2021-09 | 0.4079 | 0.4156 (+1.9%) | 0.4089 (+0.2%) |

**Winner selection**: Hyp A mean EV-VC@100 = 0.1316 vs Hyp B = 0.1238 (+6.3% difference, outside ±5% threshold). No Spearman veto for either hypothesis. Hypothesis A selected.

## Code Changes

**File: `ml/config.py`**
- Added `_DEAD_FEATURES` set containing the 5 always-zero features
- Created `_V1_CLF_FOR_REGRESSOR` and `_V1_CLF_MONO_FOR_REGRESSOR` that filter out dead features from `_V1_CLF_FEATURES`
- `_ALL_REGRESSOR_FEATURES` now derives from filtered list (34 features total)
- Classifier features (`_V1_CLF_FEATURES`, `ClassifierConfig`) remain untouched

**File: `ml/tests/test_config.py`**
- Updated feature count assertion: 39 → 34

**File: `ml/tests/test_data_loader.py`**
- Updated feature count assertion: 39 → 34

## Full 12-Month Results (v0011 vs v0009 champion)

| Metric | v0011 | v0009 | Delta | Delta% |
|--------|-------|-------|-------|--------|
| EV-VC@100 (mean) | 0.0801 | 0.0762 | +0.0039 | +5.2% |
| EV-VC@500 (mean) | 0.2270 | 0.2329 | -0.0059 | -2.5% |
| EV-NDCG (mean) | 0.7499 | 0.7548 | -0.0048 | -0.6% |
| Spearman (mean) | 0.3925 | 0.3910 | +0.0015 | +0.4% |
| C-RMSE (mean) | 2866.6 | 2827.4 | +39.2 | +1.4% |
| C-MAE (mean) | 1142.5 | 1136.7 | +5.8 | +0.5% |

### Analysis
- **EV-VC@100 +5.2%**: Strongest improvement at top-100 value capture, driven primarily by 2021-09 (+17.6%) and 2020-09 (+29.1%).
- **Spearman +0.4%**: Slight positive shift, consistent with learning #15 (pruning dead features improves effective sampling).
- **EV-VC@500 -2.5%**: Minor degradation at top-500. Some months (2021-05, 2021-11) show value redistribution from broader to narrower top-K.
- **EV-NDCG -0.6%**: Slight ranking quality decrease, within noise range.
- **C-RMSE/C-MAE slightly worse**: Marginal calibration degradation, non-blocking (Group B).

### Risk: EV-VC@500 degradation
The -2.5% EV-VC@500 decrease is notable. 5 months improved, 7 months degraded. The pruning appears to sharpen top-100 predictions at the expense of broader top-500 coverage. This trades breadth for precision at the top of the ranking.
