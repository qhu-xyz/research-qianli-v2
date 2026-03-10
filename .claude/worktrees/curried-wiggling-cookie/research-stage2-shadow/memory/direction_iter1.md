# Direction — Iteration 1 (feat-eng-3-20260304-121042)

## Champion: v0009 (39 features, 34 effective)

Mean EV-VC@100=0.0762, EV-VC@500=0.2329, EV-NDCG=0.7548, Spearman=0.3910

## Batch Constraint

**Feature engineering / selection ONLY.** No HP changes. No training mode changes. Only features and monotone_constraints may change.

---

## Hypothesis A (Primary): Prune 5 Zero-Filled Features (39→34)

**What**: Remove 5 features that are always zero because the data loader doesn't provide them: `hist_physical_interaction`, `overload_exceedance_product`, `band_severity`, `sf_exceed_interaction`, `hist_seasonal_band`.

**Why**: Both Claude and Codex reviewers identified these as dead features. With `colsample_bytree=0.8`, each tree samples ~31 of 39 features — ~5 zero slots waste sampling capacity. After pruning, each tree samples ~27 of 34 features, all carrying real signal. This should improve effective feature utilization per tree without any signal loss.

**Hypothesis A overrides**:
```json
{"regressor": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_exceed_90", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_variance", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_skewness", "density_kurtosis", "density_cv", "season_hist_da_3", "prob_below_85"], "monotone_constraints": [1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, -1]}}
```

---

## Hypothesis B (Alternative): Prune 5 Zero-Filled + Add flow_direction (39→35)

**What**: Same pruning as Hypothesis A, plus add `flow_direction` (integer indicating constraint flow direction). Monotone constraint = 0 (unconstrained — direction is categorical, no monotonic relationship expected).

**Why**: `flow_direction` is available from MisoDataLoader but never included in the regressor. It could differentiate constraint behavior where the same line binds in opposite directions (forward vs reverse congestion). If forward-binding constraints have systematically different shadow price magnitudes than reverse-binding, this feature provides that signal. Combined with pruning, we get cleaner features + a genuinely new signal axis.

**Hypothesis B overrides**:
```json
{"regressor": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_exceed_90", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_variance", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_skewness", "density_kurtosis", "density_cv", "season_hist_da_3", "prob_below_85", "flow_direction"], "monotone_constraints": [1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, -1, 0]}}
```

---

## Screen Months

- **Weak month: 2022-09** — Worst EV-VC@500 (0.062), low EV-VC@100 (0.030), low EV-NDCG (0.682), Spearman (0.330). Tests whether pruning/new signal helps the worst value-capture month. Not screened in previous batch.
- **Strong month: 2021-09** — Best EV-VC@100 (0.200), best EV-VC@500 (0.400), strong NDCG (0.804). Tests that we don't regress on our best performing month. Not screened in previous batch.

**Rationale**: Both months are fresh (not used in prior batch screening). The weak month (2022-09) is poor across ALL Group A metrics, making it diagnostic for general improvement. The strong month (2021-09) has the highest EV-VC values, making it the regression sentinel.

---

## Winner Criteria

Pick the hypothesis with **higher mean EV-VC@100 across the 2 screen months**, with these tiebreakers/vetos:
1. If both are within ±5% of each other on EV-VC@100, prefer the one with higher EV-VC@500.
2. **Veto**: If a hypothesis drops Spearman > 0.02 on either screen month vs champion, disqualify it (Spearman is the binding gate constraint at 4.7% margin).
3. If both pass or both fail the Spearman check, use EV-VC@100 as the primary selector.

---

## Code Changes for Winner

### If Hypothesis A wins (prune only):

**File: `ml/config.py`**
- In `_ADDITIONAL_FEATURES`: Remove `"hist_physical_interaction"`, `"overload_exceedance_product"`, `"band_severity"`, `"sf_exceed_interaction"`, `"hist_seasonal_band"` from the list.
- In `_ADDITIONAL_MONOTONE`: Remove the corresponding monotone constraint values (positions matching the removed features — they are all `0`).
- `_REGRESSOR_FEATURES` and `_REGRESSOR_MONOTONE` are derived from classifier + additional, so they will auto-update.

**File: `ml/tests/`**
- Update any test that asserts feature count (e.g., `len(REGRESSOR_FEATURES) == 39` → `== 34`).

### If Hypothesis B wins (prune + flow_direction):

All changes from Hypothesis A, PLUS:

**File: `ml/config.py`**
- In `_ADDITIONAL_FEATURES`: Add `"flow_direction"` to the list.
- In `_ADDITIONAL_MONOTONE`: Add `0` at the corresponding position.

**File: `ml/features.py`**
- Verify `flow_direction` is passed through from the data loader to the feature matrix. If not, add it as a passthrough feature (no computation needed — it's a raw column).

**File: `ml/tests/`**
- Update feature count assertion to `== 35`.

---

## Expected Impact

| Metric | Hyp A (prune) | Hyp B (prune + flow_direction) |
|--------|---------------|-------------------------------|
| EV-VC@100 | +0-2% (cleaner sampling) | +1-4% (new signal + cleaner) |
| EV-VC@500 | +0-1% | +1-3% |
| EV-NDCG | Neutral | +0-1% |
| Spearman | Neutral (no signal change) | +0-1% (if directional) |
| C-RMSE | Neutral to slight improvement | Neutral to slight improvement |

Conservative estimates. The pruning itself may have minimal impact since XGBoost doesn't split on constant features, but the colsample_bytree sampling improvement is real. flow_direction's impact depends on whether binding direction correlates with shadow price magnitude.

---

## Risk Assessment

1. **Low risk (both)**: Pruning zero-filled features cannot hurt model quality — these features carry zero information. Worst case is neutral.
2. **Medium risk (Hyp B)**: `flow_direction` might not be populated in all eval months or might be a constant. If so, it's equivalent to Hyp A. Worker should verify non-null, non-constant values for flow_direction before proceeding.
3. **Spearman sensitivity**: Any feature change could shift Spearman slightly. With 4.7% margin to floor, even a -1% change is safe. But monitor closely on screen.
4. **Data loader compatibility**: Confirm that removing features from config doesn't break the data loader pipeline. The loader provides columns regardless; config just selects which ones to use.
