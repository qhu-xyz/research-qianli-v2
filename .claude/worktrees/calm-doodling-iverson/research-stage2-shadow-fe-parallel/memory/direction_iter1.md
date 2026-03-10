# Direction — Iteration 1 (FE-Parallel Batch)

## Batch Constraint
**FEATURE ENGINEERING ONLY** — do NOT change any hyperparameters. Only modify regressor feature list and monotone_constraints via `--overrides`.

## Current State
- Champion: v0007 (34 regressor features, mcw=25, reg_lambda=1.0)
- 5 features available but NOT in v0007: `density_skewness`, `density_kurtosis`, `density_cv`, `season_hist_da_3`, `prob_below_85`
- Weakest months: 2022-06 (EV-NDCG=0.604, Spearman=0.271), 2021-05 (EV-VC@100≈0)
- Strongest months: 2022-12 (EV-VC@100=0.192), 2021-09 (EV-VC@100=0.122, EV-VC@500=0.395)

## Screen Months
- **Weak month: 2022-06** — worst EV-NDCG (0.604), worst Spearman (0.271 tied with 2021-11). If FE changes help, they should improve ranking quality here.
- **Strong month: 2021-09** — strong EV-VC@100 (0.122), top EV-VC@500 (0.395). Changes must not regress here.

Rationale: 2022-06 is the most consistently weak month across ALL ranking metrics (not just one). 2021-09 is strong across the board — a robust safety check.

---

## Hypothesis A (Primary): Feature Pruning — Remove 7 Low-Signal Features

**Rationale**: With 34 features and mcw=25, trees may split on noisy features that dilute signal on weak months. Pruning features with low expected discriminative power for shadow price *magnitude* should improve signal-to-noise ratio. The model already has strong exceedance probability features (110, 105, 100, 95, 90) — lower thresholds (85, 80) add noise without adding binding-relevant signal. Similarly, `is_interface` (binary, low variance), `sf_nonzero_frac` (redundant with sf_max_abs), and `density_mean`/`density_variance` (subsumed by entropy + exceedance probs) are likely low-value.

**Features to REMOVE (7)**:
1. `prob_exceed_80` — far below binding threshold, low signal for shadow price magnitude
2. `prob_exceed_85` — same reasoning, marginal over prob_exceed_90
3. `is_interface` — binary feature with low variance, weak discriminative power
4. `sf_nonzero_frac` — redundant with sf_max_abs (which captures magnitude)
5. `density_mean` — mean flow level subsumed by exceedance probability features
6. `density_variance` — subsumed by density_entropy + tail_concentration
7. `prob_band_95_100` — narrow band, noisy; prob_exceed_95 and prob_exceed_100 already bracket this

**Resulting feature set: 27 features** (34 - 7)

Hypothesis A overrides:
```json
{"regressor": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_exceed_90", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "hist_physical_interaction", "overload_exceedance_product", "sf_max_abs", "sf_mean_abs", "sf_std", "constraint_limit", "density_entropy", "tail_concentration", "prob_band_100_105", "hist_da_max_season", "band_severity", "sf_exceed_interaction", "hist_seasonal_band", "recent_hist_da", "season_hist_da_1", "season_hist_da_2"], "monotone_constraints": [1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1]}}
```

---

## Hypothesis B (Alternative): Expand to Full 39-Feature Set

**Rationale**: v0007 uses 34 of 39 available features. The 5 missing features carry distributional shape information that may help the regressor on weak months:
- `density_skewness` / `density_kurtosis` — higher-order distribution moments capture tail behavior differently from entropy
- `density_cv` — coefficient of variation normalizes spread by level, useful for comparing constraints with different flow magnitudes
- `season_hist_da_3` — additional seasonal harmonic component may capture pricing patterns not in components 1-2
- `prob_below_85` — captures low-flow scenarios, useful for constraints that oscillate between binding and non-binding

**Resulting feature set: 39 features** (34 + 5)

Hypothesis B overrides:
```json
{"regressor": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_exceed_90", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "hist_physical_interaction", "overload_exceedance_product", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_variance", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "band_severity", "sf_exceed_interaction", "hist_seasonal_band", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_skewness", "density_kurtosis", "density_cv", "season_hist_da_3", "prob_below_85"], "monotone_constraints": [1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, -1]}}
```

---

## Winner Criteria
1. **Primary**: Higher mean EV-VC@100 across the 2 screen months
2. **Safety check**: Spearman must not drop > 0.03 on either month vs champion
3. **Tiebreaker**: Higher EV-NDCG on the weak month (2022-06)

If both hypotheses degrade vs champion on the weak month, pick the one with smaller degradation and consider it a negative signal (pruning/expansion not helpful at this granularity — future iterations should try interaction features instead).

## Code Changes for Winner

### If Hypothesis A wins (pruning):
- **File**: `ml/config.py`
- **Change**: Update `RegressorConfig` defaults to use the 27-feature list. Update `_ALL_REGRESSOR_FEATURES` or override in `RegressorConfig.__init__` defaults.
- Alternatively, just ensure the version's config.json records the 27-feature set (the override will handle this).
- No changes to `ml/features.py` needed.

### If Hypothesis B wins (expansion):
- **File**: `ml/config.py`
- **Change**: v0007 already uses a subset; the override expands to the full `_ALL_REGRESSOR_FEATURES`. No code changes needed — the full set is already the `RegressorConfig` default.
- No changes to `ml/features.py` needed.

### In either case:
- The `--overrides` JSON handles the feature list entirely. The winner's config gets recorded in `registry/{version}/config.json` automatically.

## Expected Impact
- **Hypothesis A (pruning)**: Expect +2-5% EV-VC@100 on weak months from reduced noise. Spearman neutral or slightly positive. Risk of losing signal if pruned features matter for specific constraint types.
- **Hypothesis B (expansion)**: Expect modest improvement (+1-3%) from additional distributional information. Risk of more noise if extra features are uninformative.

## Risk Assessment
- **Low risk overall**: Both hypotheses only change feature selection, which is easily reversible. No hyperparameter or architecture changes.
- **Hypothesis A risk**: Pruning too aggressively could remove features that matter for specific months. 7 features (20% reduction) is moderate.
- **Hypothesis B risk**: Adding 5 features to an already large set (34→39) with colsample_bytree=0.8 means each tree sees ~31 features — diminishing returns likely.
- **Mitigation**: Screen on 2 months catches both directions (help on weak, no regress on strong).
