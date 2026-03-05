# Direction — Iteration 2 (batch: tier-fe-1-20260304-182037)

## Situation Analysis

**Champion**: v0 (34 features, baseline)
**Batch constraint**: Feature engineering / selection ONLY (no hyperparams, no class weights, no bins/midpoints)
**Previous iteration**: Iter 1 FAILED — worker produced no artifacts. Hypotheses untested.

**Key problems** (unchanged from v0):
1. Tier-VC@100 = 0.071 (below floor 0.075) — the only Group A gate currently failing Layer 1
2. Tier-Recall@1 = 0.047 — model almost never predicts tier 1
3. High monthly variance — 2021-11 disaster (VC@100=0.003, QWK=0.244), 2022-06 weak (NDCG=0.643)

**Feature importance** (mean gain, bottom 6):
- density_skewness (1.09%), prob_exceed_90 (1.13%), density_cv (1.13%), density_variance (1.17%), prob_below_90 (1.19%), prob_exceed_95 (1.20%)

**Available dead interaction features** (computed by `compute_interaction_features()`, exist as DataFrame columns but excluded from model by `_DEAD_FEATURES`):
- `hist_physical_interaction` = hist_da × prob_exceed_100
- `overload_exceedance_product` = expected_overload × prob_exceed_105
- `band_severity` = prob_band_95_100 × expected_overload
- `sf_exceed_interaction` = sf_max_abs × prob_exceed_100
- `hist_seasonal_band` = hist_da_max_season × prob_band_100_105

All 5 are already computed as DataFrame columns on every pipeline run. They can be added via `--overrides` without any code changes.

**Strategic rationale**: The model relies heavily on hist_da (13.3%) and recent_hist_da (21.1%) individually but lacks cross-signal interactions combining price history with physical flow exceedance. Pre-computed interactions should help XGBoost detect compound severity at depth 5. We pick the 3 most domain-relevant interactions:
- `hist_physical_interaction`: Directly captures "high price AND high flow" — the signature of tier 0/1 constraints
- `overload_exceedance_product`: Overload magnitude × extreme exceedance probability — severe binding indicator
- `hist_seasonal_band`: Seasonal price peak × critical flow band — captures seasonal binding patterns

---

## Hypothesis A (primary): Add 3 interaction features, keep all 34 existing → 37 features

**Rationale**: Conservative approach — don't remove anything, just add 3 interaction features from `_DEAD_FEATURES`. This avoids the risk of removing features that matter for specific months while providing new compound severity signals for tier 0/1 discrimination.

Hypothesis A overrides:
```json
{"tier": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_exceed_90", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_variance", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_skewness", "density_kurtosis", "density_cv", "season_hist_da_3", "prob_below_85", "hist_physical_interaction", "overload_exceedance_product", "hist_seasonal_band"], "monotone_constraints": [1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, -1, 0, 0, 0]}}
```

---

## Hypothesis B (alternative): Prune 6 low-importance features + add 3 interactions → 31 features

**Rationale**: Stage-2 showed pruning dead features improved EV-VC@100 by +5.2%. Removing the 6 lowest-importance features (all ~1.1-1.2% gain) frees sampling efficiency and forces the model to concentrate splits on high-signal features. Combined with 3 interaction features, the model gets better signals with less noise.

**Remove** (6 lowest by mean gain):
- `density_skewness` (1.09%), `prob_exceed_90` (1.13%), `density_cv` (1.13%), `density_variance` (1.17%), `prob_below_90` (1.19%), `prob_exceed_95` (1.20%)

**Add** (same 3 interactions as Hypothesis A):
- `hist_physical_interaction`, `overload_exceedance_product`, `hist_seasonal_band`

**Net**: 34 - 6 + 3 = 31 features

Hypothesis B overrides:
```json
{"tier": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_below_100", "prob_below_95", "expected_overload", "hist_da", "hist_da_trend", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_kurtosis", "season_hist_da_3", "prob_below_85", "hist_physical_interaction", "overload_exceedance_product", "hist_seasonal_band"], "monotone_constraints": [1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, -1, 0, 0, 0]}}
```

---

## Screen Months

| Role | Month | Rationale |
|------|-------|-----------|
| **Weak** | 2021-11 | Worst VC@100 (0.003), worst QWK (0.244), Recall@1=0.000. If interactions help compound severity detection here, tail safety improves. |
| **Strong** | 2021-09 | Best VC@100 (0.248), strong QWK (0.449). Changes must not regress here. |

---

## Winner Criteria

1. **Primary**: Higher mean Tier-VC@100 across the 2 screen months
2. **Safety**: QWK must not drop > 0.03 on the strong month (2021-09) compared to v0's 0.449
3. **Tiebreak**: Higher mean QWK across screen months
4. **Both regress**: If both hypotheses regress on VC@100 vs v0 on BOTH screen months, pick the one with less regression

---

## Code Changes for Winner

### If Hypothesis A wins (add 3 interactions, 34→37):

1. **`ml/config.py`** — remove `hist_physical_interaction`, `overload_exceedance_product`, `hist_seasonal_band` from `_DEAD_FEATURES` set (lines 17-23). The set should have 2 remaining entries: `sf_exceed_interaction`, `band_severity` — wait, we're keeping `band_severity` dead? No: we selected `hist_seasonal_band` not `band_severity`. Correction: remove `hist_physical_interaction`, `overload_exceedance_product`, `hist_seasonal_band` from `_DEAD_FEATURES`. Remaining dead: `band_severity`, `sf_exceed_interaction`.

2. **`ml/tests/`** — update any tests that assert feature count = 34 to 37

3. No changes to `ml/features.py` — `compute_interaction_features()` already computes all 5 features including the 3 we're reintroducing.

### If Hypothesis B wins (prune 6 + add 3 interactions, 34→31):

1. **`ml/config.py`** — same `_DEAD_FEATURES` change as Hypothesis A

2. **`ml/config.py`** — remove from `_V1_CLF_FEATURES`: `prob_exceed_95`, `prob_exceed_90`, `prob_below_90`, `density_variance` and their corresponding monotone entries from `_V1_CLF_MONOTONE`

3. **`ml/config.py`** — remove from `_ALL_TIER_FEATURES` append list: `density_skewness`, `density_cv` and their corresponding monotone entries from `_ALL_TIER_MONOTONE`

4. **`ml/tests/`** — update any tests that assert feature count = 34 to 31

5. No changes to `ml/features.py`.

---

## Expected Impact

| Gate | Current (v0 mean) | Floor | Expected Direction |
|------|-------------------|-------|--------------------|
| Tier-VC@100 | 0.071 | 0.075 | ↑ Interaction features provide compound severity signal for top-ranking |
| Tier-VC@500 | 0.230 | 0.217 | → Neutral to slight improvement |
| Tier-NDCG | 0.771 | 0.767 | → Neutral |
| QWK | 0.370 | 0.359 | ↑ Better tier 0/1 discrimination improves ordinal consistency |

**Critical gate**: Tier-VC@100 is the only Group A gate failing Layer 1 (0.071 < 0.075). The interaction features directly address this by combining price history (the model's strongest signal) with flow exceedance (the physical cause of binding).

**Monitor**: Tier-Recall@1 (0.047). `hist_physical_interaction` combines the two features most relevant to distinguishing tier 1 from tier 2/3.

---

## Risk Assessment

1. **Interaction features were dead in stage-2 regression**: But multi-class classification has different optimization dynamics — tier boundary discrimination may benefit from pre-computed interactions that regression did not. Mitigated by 2-month screening.

2. **Hypothesis B pruning may remove monthly-critical features**: Some pruned features spike in specific months (prob_exceed_90 = 1.66% in 2021-03 vs 0.69% in 2020-11). Mitigated by Hypothesis A (no pruning) serving as the safe fallback.

3. **37 features (Hyp A) increases collinearity**: Interactions are products of existing features. With colsample_bytree=0.8, XGBoost samples feature subsets per tree, mitigating collinearity.

4. **Worker execution risk**: Iter1 failed with complex direction. This direction is simpler: both hypotheses use ONLY existing DataFrame columns for screening (no code changes needed). Code changes only happen after the winner is picked.
