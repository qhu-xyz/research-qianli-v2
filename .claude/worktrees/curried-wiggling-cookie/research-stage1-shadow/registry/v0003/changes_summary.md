# v0003 — Training Window Expansion (10→14 months)

## Hypothesis
H5: Expanding the training window from 10 to 14 months addresses the late-2022 distribution shift by providing 40% more training examples with greater seasonal diversity.

## Changes Made

### 1. `ml/config.py`
- **Reverted FeatureConfig** to v0 baseline (14 features). Removed 3 interaction features (`exceed_severity_ratio`, `hist_physical_interaction`, `overload_exceedance_product`) added in v0002 to isolate the training window effect.
- **Changed `PipelineConfig.train_months`** default from 10 to 14.
- Updated `features` property docstring to remove hardcoded "14 items" count.

### 2. `ml/features.py`
- **Guarded interaction feature computation**: interaction columns are only computed if requested by the current `FeatureConfig`. Previously they were unconditionally computed.
- **Added schema guard**: raises `ValueError` if requested feature columns are missing from the DataFrame.
- Updated docstrings to remove hardcoded feature counts.

### 3. `ml/benchmark.py` (BUG FIX)
- **Fixed train_months/val_months plumbing**: Previously `_eval_single_month()` and `run_benchmark()` did not accept or pass `train_months`/`val_months` parameters, causing `PipelineConfig` to use whatever default was set, and the eval_config metadata was hardcoded to `train_months: 10, val_months: 2`.
- Added `train_months` and `val_months` parameters to both functions.
- Override extraction now captures `train_months` and `val_months` from `PipelineConfig`.
- `eval_config` in output now reflects actual values used.

### 4. `ml/data_loader.py`
- Updated docstring to remove hardcoded "14 feature columns".

### 5. Tests updated
- `test_config.py`: Updated feature count assertions from 17→14, monotone constraint string, feature name list, and `train_months` default from 10→14.
- `test_features.py`: Updated shape assertion from 17→14, simplified test data construction.

## Results (12 months, f0, onpeak, real data)

### Group A (Hard Gates) — Mean Aggregates

| Metric | v0 | v0003 | Delta | W/L/T |
|--------|-----|-------|-------|-------|
| S1-AUC | 0.8348 | 0.8361 | +0.0013 | 7W/4L/1T |
| S1-AP | 0.3936 | 0.3948 | +0.0012 | 8W/4L |
| S1-NDCG | 0.7333 | 0.7352 | +0.0019 | 7W/4L/1T |
| S1-VCAP@100 | 0.0149 | 0.0183 | +0.0034 | 9W/3L |

### Group B (Monitor)

| Metric | v0 | v0003 | Delta |
|--------|-----|-------|-------|
| S1-BRIER | 0.1503 | 0.1514 | +0.0011 (slight regression) |
| S1-REC | 0.4192 | 0.4130 | -0.0062 |

### Bottom-2 (Tail Safety)

| Metric | v0 | v0003 | Delta |
|--------|-----|-------|-------|
| S1-AUC | 0.8105 | 0.8162 | +0.0057 |
| S1-AP | 0.3322 | 0.3277 | -0.0045 |
| S1-NDCG | 0.6716 | 0.6657 | -0.0059 |
| S1-VCAP@100 | 0.0014 | 0.0016 | +0.0002 |

### Late-2022 Target Months

| Month | v0 AUC | v0003 AUC | Delta | v0 AP | v0003 AP | Delta |
|-------|--------|-----------|-------|-------|----------|-------|
| 2022-09 | 0.8334 | 0.8334 | +0.0000 | 0.3150 | 0.3059 | -0.0091 |
| 2022-12 | 0.8088 | 0.8186 | +0.0098 | 0.3623 | 0.3765 | +0.0142 |

### Success Criteria Assessment
- **AUC improvement in ≥7/12 months**: YES (7W/4L/1T)
- **Mean AUC > 0.835**: YES (0.8361)
- **Both criteria met.**

## Interpretation

The training window expansion shows a **small but real improvement** across Group A metrics. Key observations:

1. **AUC ceiling slightly broken**: Mean AUC improved from 0.8348 to 0.8361, with 7/12 months improving — a directionally stronger result than HP tuning (0W/11L) or interactions (5W/6L/1T).

2. **2022-12 biggest beneficiary**: AUC +0.0098, AP +0.0142 — the weakest month improved the most, supporting the seasonal diversity hypothesis.

3. **2022-09 unchanged on AUC, regressed on AP**: The window expansion didn't help this month, suggesting the distribution shift there may require different features rather than more historical data.

4. **VCAP@100 improved meaningfully** (9W/3L, +0.0034): Better top-100 value capture suggests improved tail discrimination with more training examples.

5. **Brier and Recall slightly regressed**: More training data made the model marginally less well-calibrated and more conservative (pred_binding_rate 0.0754→0.0733), consistent with the larger training set regularizing predictions.

6. **Bottom-2 mixed**: AUC tail improved (+0.0057) but AP and NDCG tails slightly regressed, indicating the worst months are still fragile.
