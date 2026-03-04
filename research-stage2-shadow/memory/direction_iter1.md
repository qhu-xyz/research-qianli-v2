# Direction — Iteration 1 (batch feat-eng-3-20260304-091135)

## Batch Constraint: FEATURE ENGINEERING ONLY
No hyperparameter changes. Only `features`, `monotone_constraints`, and feature computation in `ml/features.py` may be modified.

## Analysis

**Champion**: v0007 (34 features, reg_lambda=1.0, mcw=25)
- Mean: EV-VC@100=0.0699, EV-VC@500=0.2294, EV-NDCG=0.7513, Spearman=0.3932
- Key weakness: 3 distributional features (density_skewness, density_kurtosis, density_cv) are used by the v0 classifier (14 features) but were **dropped** when the regressor inherited the v1 classifier feature set (29 features). These features measure flow distribution shape — directly relevant to predicting shadow price magnitude.

**Opportunity**: The v0→v1 classifier transition reorganized features but the 3 distributional shape features were lost in translation. The regressor has `density_mean`, `density_variance`, `density_entropy` but NOT `density_skewness`, `density_kurtosis`, `density_cv`. Skewness and kurtosis capture tail behavior of the flow distribution, which is critical for predicting how large a shadow price will be (heavy-tailed overloads → larger shadow prices).

## Screen Months

- **Weak**: `2022-06` — worst EV-NDCG (0.604), 2nd worst EV-VC@100 (0.014), 2nd worst Spearman (0.271). This month likely has unusual flow distributions where distributional shape features should help most.
- **Strong**: `2022-12` — best EV-VC@100 (0.192), strong EV-NDCG (0.815). Should not regress with additional features.

**Rationale**: 2022-06 is the most informative weak month (bad across all metrics without being catastrophically zero like 2021-05's EV-VC@100=0.0002). 2022-12 is a strong month that tests for regression from feature noise.

---

## Hypothesis A (primary): Add 3 Distributional Shape Features

**What**: Add `density_skewness`, `density_kurtosis`, `density_cv` to the regressor feature set (34 → 37 features).

**Why**: These features describe the shape of the flow probability distribution:
- **density_skewness**: Right-skewed distributions indicate more extreme overloads → larger shadow prices. Directly informative for regression magnitude.
- **density_kurtosis**: Heavy-tailed distributions have more extreme events → captures the "how bad can it get" signal.
- **density_cv**: Coefficient of variation normalizes variability by mean — high CV means volatile flows relative to average, often associated with larger shadow prices.

These are raw MisoDataLoader columns (no feature engineering needed). They're proven useful in the v0 classifier and their omission from the regressor appears to be an oversight from the v1 feature set migration.

**Monotone constraints**: All 0 (unconstrained) — same as in the classifier. The relationship between distribution shape and shadow price magnitude isn't necessarily monotonic (e.g., very negative skewness could mean flow is concentrated well below the limit → low shadow price, but the sign of skewness relative to the binding threshold is ambiguous).

Hypothesis A overrides:
```json
{"regressor": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_exceed_90", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "hist_physical_interaction", "overload_exceedance_product", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_variance", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "band_severity", "sf_exceed_interaction", "hist_seasonal_band", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_skewness", "density_kurtosis", "density_cv"], "monotone_constraints": [1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]}}
```

---

## Hypothesis B (alternative): Add 3 Distributional + 2 Additional Raw Features

**What**: Add `density_skewness`, `density_kurtosis`, `density_cv` PLUS `season_hist_da_3` and `prob_below_85` to the regressor (34 → 39 features).

**Why**: Same distributional shape rationale as Hypothesis A, plus:
- **season_hist_da_3**: Third seasonal DA component. The regressor already uses `season_hist_da_1` and `season_hist_da_2` — adding the 3rd component captures finer seasonal patterns. Available from MisoDataLoader.
- **prob_below_85**: Probability flow is below 85% of limit. Complements `prob_below_90/95/100` — provides a deeper "how far from binding" signal at the bottom of the distribution. Available from MisoDataLoader.

This tests whether broader feature expansion helps or whether additional features dilute signal (colsample_bytree=0.8 means each tree samples 80% of features; more features = each tree sees proportionally less of each signal).

**Monotone constraints**: density_skewness=0, density_kurtosis=0, density_cv=0, season_hist_da_3=+1 (higher seasonal component → higher shadow price), prob_below_85=-1 (higher probability of being far below limit → lower shadow price).

Hypothesis B overrides:
```json
{"regressor": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_exceed_90", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "hist_physical_interaction", "overload_exceedance_product", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_variance", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "band_severity", "sf_exceed_interaction", "hist_seasonal_band", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_skewness", "density_kurtosis", "density_cv", "season_hist_da_3", "prob_below_85"], "monotone_constraints": [1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, -1]}}
```

---

## Winner Criteria

Pick the hypothesis with **higher mean EV-VC@100 across the 2 screen months**, unless:
1. Spearman drops > 0.02 vs champion on either screen month → disqualify
2. EV-NDCG drops > 0.03 vs champion on either screen month → disqualify
3. If both pass safety checks and EV-VC@100 is within 0.005, prefer the one with better EV-VC@500

If both hypotheses fail safety checks, fall back to Hypothesis A (smaller change, lower risk).

---

## Code Changes for Winner

The winning hypothesis requires **NO code changes to `ml/features.py`** — all added features are raw MisoDataLoader columns that are already available in the DataFrame. The override mechanism handles feature/monotone list changes.

For the **full 12-month benchmark**, the worker should:
1. Update `ml/config.py` to make the winning feature set the new default:
   - Modify `_ALL_REGRESSOR_FEATURES` to append the new features after the existing 5 extras
   - Modify `_ALL_REGRESSOR_MONOTONE` to append the corresponding monotone constraints
2. Verify the column count matches: `len(features) == len(monotone_constraints)`
3. Update `ml/tests/test_config.py` to reflect the new feature count

**If Hypothesis A wins** (37 features):
- In `ml/config.py`, change `_ALL_REGRESSOR_FEATURES` from `_V1_CLF_FEATURES + [5 extras]` to `_V1_CLF_FEATURES + [5 extras] + ["density_skewness", "density_kurtosis", "density_cv"]`
- Append `0, 0, 0` to `_ALL_REGRESSOR_MONOTONE`

**If Hypothesis B wins** (39 features):
- Same as A, plus append `"season_hist_da_3", "prob_below_85"` to features
- Append `1, -1` to monotone constraints

---

## Expected Impact

| Gate | Expected Change | Reasoning |
|------|----------------|-----------|
| EV-VC@100 | +2-5% | Distributional shape features help identify high-value constraints on weak months |
| EV-VC@500 | +1-3% | Broader ranking improvement from better prediction of magnitude |
| EV-NDCG | +0.5-1% | Ranking quality improvement from more informative features |
| Spearman | neutral to +0.5% | More features → better regression fit → better rank correlation, but risk of overfitting |
| C-RMSE | -1-3% improvement | Better prediction of shadow price magnitude |

**Primary target**: Improve weak-month EV-VC@100 (currently 0.014 on 2022-06) by providing more signal about distribution shape.

## Risk Assessment

1. **Feature dilution**: Adding 3-5 features to 34 may dilute per-feature signal with colsample_bytree=0.8. Mitigated by the features being demonstrably useful in the classifier.
2. **Overfitting on distributional features**: density_skewness/kurtosis can be noisy on small samples. min_child_weight=25 provides protection.
3. **Column availability**: These are raw MisoDataLoader columns — the main risk is they're missing from the DataFrame for some months. The screen will catch this immediately (crash or NaN explosion).
4. **Monotone constraint mismatch**: All 3 distributional features are unconstrained (0), which is correct — their effect on shadow price magnitude is non-monotonic.
