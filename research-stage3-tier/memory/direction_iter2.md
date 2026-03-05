# Direction — Iteration 2 (batch: tier-fe-2-20260305-001606)

## Constraint

**FE only** — only `features`, `monotone_constraints` in config.py and feature computation in features.py may change. All hyperparams, class weights, bins, midpoints FROZEN.

## Current State

- Champion: v0 (34 features)
- Best candidate: v0005 (37 features, iter1) — NOT promoted, Tier-VC@100 fails L1 by 0.0004
- v0005 added: overload_x_hist, prob110_x_recent_hist, tail_x_hist
- **Gap to close**: Tier-VC@100 mean 0.0746 vs floor 0.0750 — need +0.0004 (+0.5%)
- All other Group A gates pass all 3 layers
- Value-QWK barely passing (0.3918 vs 0.3914) — DO NOT regress this

## Single Hypothesis: Add 4 features to v0005 base (37 → 41)

**No A/B screening.** The gap is tiny (0.0004) and these features are additive (low risk of regression). Go directly to full 12-month benchmark.

### New Features

1. **`log1p_hist_da`** = `log1p(hist_da)` — compress long-tailed price distribution; should help discriminate at high end. Monotone: **+1**.

2. **`log1p_expected_overload`** = `log1p(expected_overload)` — compress long-tailed overload distribution. Monotone: **+1**.

3. **`overload_x_recent_hist`** = `expected_overload * recent_hist_da` — overload × recent price signal. Since recent_hist_da is the #1 importance feature (21.1%) and overload_x_hist (with hist_da) already provided marginal improvement, this adds the same interaction using the stronger parent. Monotone: **+1**.

4. **`prob_range_high`** = `prob_exceed_100 - prob_exceed_110` — probability mass in the 100-110% overload band. Captures constraints that are moderately overloaded (critical transition zone between tier 2 and tier 1). Monotone: **0** (not clearly monotone — higher could indicate either direction).

### Why These 4

- **Log transforms** are the highest-signal candidates remaining. Claude reviewer specifically recommended them. They compress long-tailed distributions (hist_da ranges 0-10000+) into a scale where the model can make finer splits at the high end, exactly where VC@100 discrimination happens.
- **overload_x_recent_hist** extends the successful interaction pattern from iter1. It uses the #1 importance feature (recent_hist_da) instead of the #2 (hist_da) that overload_x_hist already uses.
- **prob_range_high** captures the physical transition zone (100-110% of limit) that separates moderate from severe binding. This is a unique signal not captured by any existing feature.

## Code Changes Required

### 1. `ml/features.py` — Add to `compute_interaction_features()`

Add 4 new feature computations:
```python
# Log transforms
(pl.col("hist_da").map_elements(lambda x: math.log1p(x), return_dtype=pl.Float64))
    .alias("log1p_hist_da"),
(pl.col("expected_overload").map_elements(lambda x: math.log1p(x), return_dtype=pl.Float64))
    .alias("log1p_expected_overload"),
# Interactions
(pl.col("expected_overload") * pl.col("recent_hist_da"))
    .alias("overload_x_recent_hist"),
# Derived
(pl.col("prob_exceed_100") - pl.col("prob_exceed_110"))
    .alias("prob_range_high"),
```

**Note**: For log1p, prefer vectorized polars operations over `map_elements` if available. Use `pl.col("hist_da").log1p()` if polars supports it, otherwise use the expression `(pl.col("hist_da") + 1).log()`.

### 2. `ml/config.py` — Append to feature lists

Append to `_ALL_TIER_FEATURES`:
```python
"log1p_hist_da",
"log1p_expected_overload",
"overload_x_recent_hist",
"prob_range_high",
```

Append to `_ALL_TIER_MONOTONE`:
```python
1, 1, 1, 0,  # log1p_hist_da, log1p_expected_overload, overload_x_recent_hist, prob_range_high
```

### 3. `ml/tests/` — Update feature count assertions

Update any test asserting `len(features) == 37` to `41`.

## Overrides (full feature list for 41 features)

```json
{"tier": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_exceed_90", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_variance", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_skewness", "density_kurtosis", "density_cv", "season_hist_da_3", "prob_below_85", "overload_x_hist", "prob110_x_recent_hist", "tail_x_hist", "log1p_hist_da", "log1p_expected_overload", "overload_x_recent_hist", "prob_range_high"], "monotone_constraints": [1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, -1, 1, 1, 1, 1, 1, 1, 0]}}
```

## Expected Impact

- **Tier-VC@100**: Need +0.5% (0.0004). Log transforms should help discriminate at the top of the ranking. Combined effect of 4 features should provide enough marginal improvement.
- **Value-QWK**: Monitor closely — barely passing. Log transforms should not hurt ordinal consistency.
- **Tier-Recall@1**: Still structural. Do not expect improvement from FE.

## Risk Assessment

- **Very low risk**: All 4 features are additive. XGBoost ignores unhelpful features via low split gain.
- **No pruning**: Confirmed in iter1 that pruning hurts weak months.
- **Main risk**: If these 4 features don't close the 0.0004 gap, FE alone may be insufficient for this pipeline. Iter3 would be the last attempt.

## Success Criteria

- Tier-VC@100 mean >= 0.0750 (L1 pass)
- All other Group A gates continue to pass all 3 layers
- Value-QWK mean >= 0.3914 (don't regress)
- No catastrophic regressions on any metric
