# v0007 Changes Summary — Shift Factor + Constraint Metadata Features

## Hypothesis

**H9**: Adding 6 new features from entirely new signal categories (network topology via shift factors + constraint structural metadata) will break the AUC ceiling at ~0.836 because the model is feature-starved, not complexity-starved.

## Changes Made

### 1. `ml/config.py` — FeatureConfig
- Added 6 new features to `step1_features` (13 → 19 total):
  - `sf_max_abs` (monotone=1): peak node sensitivity from shift factors
  - `sf_mean_abs` (monotone=1): average node sensitivity
  - `sf_std` (monotone=0): sensitivity spread across nodes
  - `sf_nonzero_frac` (monotone=0): fraction of nodes with non-zero shift factors (constraint reach)
  - `is_interface` (monotone=0): binary flag for flowgate vs line constraint
  - `constraint_limit` (monotone=0): log-transformed MW limit of the constraint

### 2. `ml/data_loader.py`
- Added diagnostic print to verify new columns are present in source loader output
- Updated `_load_smoke()` to generate realistic synthetic data for the 6 new features (positive values for SF features, binary for `is_interface`, log-transformed for `constraint_limit`)

### 3. `ml/features.py`
- Added explicit verification that source-loader features exist in the DataFrame before feature preparation (raises `ValueError` with clear message if missing)

### 4. `ml/tests/conftest.py`
- Updated `synthetic_features` fixture to dynamically use `len(FeatureConfig().features)` instead of hardcoded 13

### 5. `ml/tests/test_config.py`
- Updated feature count assertion: 13 → 19
- Updated monotone constraints string assertion
- Updated expected feature names list

## What Was NOT Changed
- `train_months`: 14 (kept)
- `threshold_beta`: 0.7 (kept)
- All HPs: v0 defaults (kept)
- `ml/evaluate.py`: NOT modified (HUMAN-WRITE-ONLY)
- `registry/gates.json`: NOT modified (HUMAN-WRITE-ONLY)

## Results (12-month real-data benchmark, f0, onpeak)

### Aggregate Metrics (v0007 vs v0 baseline)

| Metric | v0007 | v0 | Delta | Direction |
|--------|-------|-----|-------|-----------|
| **S1-AUC** | **0.8485** | 0.8348 | **+0.0137** | Massive improvement |
| **S1-AP** | **0.4391** | 0.3936 | **+0.0455** | Largest AP improvement ever |
| **S1-VCAP@100** | **0.0247** | 0.0149 | **+0.0098** | Top-100 value capture up |
| **S1-NDCG** | 0.7333 | 0.7333 | +0.0000 | Neutral |
| S1-BRIER | 0.1395 | 0.1503 | **-0.0108** | Unexpected improvement |
| S1-REC | 0.4318 | 0.4190 | +0.0128 | Modest improvement |

### Bottom-2 Mean (tail safety)

| Metric | v0007 bot2 | v0 bot2 | Delta |
|--------|-----------|---------|-------|
| S1-AUC | 0.8188 | 0.811 | +0.0078 |
| S1-AP | 0.3685 | 0.332 | +0.0365 |
| S1-NDCG | 0.6562 | 0.672 | -0.0158 |
| S1-VCAP@100 | 0.0094 | 0.001 | +0.0084 |

### Per-Month AUC

| Month | v0007 | v0 | Delta |
|-------|-------|-----|-------|
| 2020-09 | 0.854 | 0.833 | +0.021 |
| 2020-11 | 0.841 | 0.841 | +0.000 |
| 2021-01 | 0.864 | 0.850 | +0.014 |
| 2021-04 | 0.844 | 0.836 | +0.008 |
| 2021-06 | 0.846 | 0.832 | +0.014 |
| 2021-08 | 0.872 | 0.840 | +0.032 |
| 2021-10 | 0.865 | 0.852 | +0.013 |
| 2021-12 | 0.826 | 0.826 | +0.000 |
| 2022-03 | 0.854 | 0.841 | +0.013 |
| 2022-06 | 0.851 | 0.838 | +0.013 |
| 2022-09 | 0.853 | 0.833 | +0.020 |
| 2022-12 | 0.812 | 0.809 | +0.003 |

### Feature Importance (6 new features)

| Feature | Mean Gain (%) | Rank (/19) |
|---------|-------------|------------|
| sf_max_abs | 1.20% | #11 |
| sf_std | 1.05% | #14 |
| constraint_limit | 0.98% | #15 |
| sf_mean_abs | 0.60% | #17 |
| sf_nonzero_frac | 0.54% | #18 |
| is_interface | 0.29% | #19 |

Combined new feature contribution: ~4.66% of total gain. Individual features are modest, but the aggregate AUC improvement (+0.0137) is the largest single-experiment improvement across all 7 real-data experiments.

### Success Criteria Assessment

- **AUC > 0.840**: YES (0.8485) — **promotion-worthy threshold exceeded**
- **AP > 0.400**: YES (0.4391) — well above threshold
- **8+/12 wins**: Likely (10+ of 12 months show AUC improvement)
- **Overall**: **PROMOTION-WORTHY** per direction criteria

## Key Observations

1. **AUC ceiling broken**: The previous ceiling across 6 experiments was [0.832, 0.836] (0.004 span). v0007 at 0.8485 is +0.012 above the ceiling — a fundamentally new operating range.
2. **AP breakthrough**: +0.0455 is 3x larger than any previous AP delta. The shift factor features help the model rank positives much better.
3. **2022-09 improved**: AUC 0.853 vs v0's 0.833 (+0.020). Previously considered "structurally broken" after 5 failed interventions. AP at 0.347 (vs v0's 0.315) is also improved.
4. **BRIER improved unexpectedly**: -0.0108 vs v0. More features usually degrade calibration, but topology features apparently improve probability estimates.
5. **NDCG neutral**: The one Group A metric that didn't benefit. May need investigation — the new features improve broad ranking (AUC, AP) but not position-weighted ranking (NDCG).
6. **NDCG bot2 regressed**: 0.6562 vs v0's 0.672 — needs monitoring against gates.
