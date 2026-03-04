# Direction — Iteration 1 (batch: tier-fe-1-20260304-182037)

## Situation Analysis

**Champion**: v0 (34 features, baseline)
**Batch constraint**: Feature engineering / selection ONLY (no hyperparams, no class weights, no bins/midpoints)
**Key problems**:
1. Tier-Recall@1 = 0.047 — model almost never predicts tier 1 ([1000, 3000))
2. Tier-VC@100 = 0.071 — top-100 ranking quality very poor
3. High monthly variance: 2021-11 is a disaster (VC@100=0.003, QWK=0.244, Recall@1=0.000)

**Feature importance insights** (mean gain across 12 months):
- Top 5: recent_hist_da (21.1%), hist_da (13.3%), prob_band_95_100 (6.8%), prob_band_100_105 (6.5%), hist_da_trend (3.8%)
- Bottom 6: density_skewness (1.1%), prob_exceed_90 (1.1%), density_cv (1.1%), density_variance (1.2%), prob_below_90 (1.2%), prob_exceed_95 (1.2%)
- Model heavily relies on price history but lacks cross-signal interactions

**Key insight**: The 5 "dead" interaction features (pruned in stage-2 regression) are still computed by `compute_interaction_features()` and exist as DataFrame columns. They can be reintroduced via `--overrides` for screening without code changes. In stage-2 regression they were not useful, but for multi-class tier *classification* they might help boundary discrimination — especially interactions involving hist_da (13.3% importance) and flow exceedance signals.

---

## Hypothesis A (primary): Reintroduce interaction features + light pruning

**Rationale**: The model needs to detect compound severity — constraints with BOTH high historical prices AND frequent flow violations. Individual features provide these signals separately, but the model must learn interactions via tree splits. Providing pre-computed interactions should make tier 0/1 discrimination easier with XGBoost's limited tree depth.

**Changes**: Drop 3 lowest-importance features, add 3 existing (but unused) interaction features:
- **Remove**: `density_skewness` (gain 0.011), `density_cv` (gain 0.011), `prob_exceed_90` (gain 0.011)
- **Add**: `hist_physical_interaction` (hist_da × prob_exceed_100), `overload_exceedance_product` (expected_overload × prob_exceed_105), `band_severity` (prob_band_95_100 × expected_overload)
- **Net**: 34 → 34 features (swap 3 for 3)

**Overrides**:
```json
{"tier": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_variance", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_kurtosis", "season_hist_da_3", "prob_below_85", "hist_physical_interaction", "overload_exceedance_product", "band_severity"], "monotone_constraints": [1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, -1, 1, 1, 0]}}
```

---

## Hypothesis B (alternative): Aggressive feature pruning

**Rationale**: Stage-2 showed pruning 5 dead features improved EV-VC@100 by +5.2%. The current 34 features include many low-importance, redundant flow probability thresholds (prob_exceed_85/90/95 overlap with 80/100/105/110). Removing noise features improves sampling efficiency and forces the model to concentrate splits on high-signal features.

**Changes**: Remove 6 lowest-importance features:
- **Remove**: `density_skewness` (0.011), `prob_exceed_90` (0.011), `density_cv` (0.011), `density_variance` (0.012), `prob_below_90` (0.012), `prob_exceed_95` (0.012)
- **Net**: 34 → 28 features

**Overrides**:
```json
{"tier": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_below_100", "prob_below_95", "expected_overload", "hist_da", "hist_da_trend", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_kurtosis", "season_hist_da_3", "prob_below_85"], "monotone_constraints": [1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, -1]}}
```

---

## Screen Months

| Role | Month | Rationale |
|------|-------|-----------|
| **Weak** | 2021-11 | Worst VC@100 (0.003), worst QWK (0.244), Recall@1 = 0.000. If changes help here, tail safety improves. |
| **Strong** | 2021-09 | Best VC@100 (0.248), strong QWK (0.449). Changes must not regress here. |

---

## Winner Criteria

1. **Primary**: Higher mean Tier-VC@100 across the 2 screen months
2. **Safety**: QWK must not drop > 0.03 on the strong month (2021-09) compared to v0's 0.449
3. **Tiebreak**: Higher mean QWK across screen months

If both hypotheses regress on Tier-VC@100 vs v0 on BOTH screen months, pick the one with less regression. If both improve, pick the one with larger improvement.

---

## Code Changes for Winner

### If Hypothesis A wins (interactions + light pruning):

1. **`ml/config.py`** — update `_ALL_TIER_FEATURES`:
   - Remove: `density_skewness`, `density_cv`, `prob_exceed_90`
   - Add at end: `hist_physical_interaction`, `overload_exceedance_product`, `band_severity`

2. **`ml/config.py`** — update `_ALL_TIER_MONOTONE`:
   - Remove corresponding entries (all were 0 or 1)
   - Add at end: `1, 1, 0` (monotone for the 3 interaction features)

3. **`ml/config.py`** — remove `hist_physical_interaction`, `overload_exceedance_product`, `band_severity` from `_DEAD_FEATURES` set (they're no longer dead)

4. **`ml/features.py`** — add 3 NEW interaction features to `compute_interaction_features()`:
   ```python
   (pl.col("expected_overload") * pl.col("hist_da")).alias("overload_x_hist"),
   (pl.col("prob_exceed_110") * pl.col("hist_da")).alias("prob110_x_hist"),
   (pl.col("tail_concentration") * pl.col("hist_da")).alias("tail_x_hist"),
   ```

5. **`ml/config.py`** — add the 3 new interaction features to `_ALL_TIER_FEATURES`:
   - Append: `overload_x_hist`, `prob110_x_hist`, `tail_x_hist`
   - Append to `_ALL_TIER_MONOTONE`: `1, 1, 1`

6. **Net for full run**: 34 - 3 pruned + 3 reintroduced + 3 new = 37 features

7. **`ml/tests/`** — update any tests that assert feature count = 34 to 37

### If Hypothesis B wins (aggressive pruning):

1. **`ml/config.py`** — remove from `_ALL_TIER_FEATURES`:
   `density_skewness`, `density_cv`, `density_variance`, `prob_exceed_90`, `prob_below_90`, `prob_exceed_95`

2. **`ml/config.py`** — remove corresponding entries from `_ALL_TIER_MONOTONE`

3. **`ml/features.py`** — add 2 conservative NEW interaction features to `compute_interaction_features()`:
   ```python
   (pl.col("expected_overload") * pl.col("hist_da")).alias("overload_x_hist"),
   (pl.col("prob_exceed_110") * pl.col("hist_da")).alias("prob110_x_hist"),
   ```

4. **`ml/config.py`** — add 2 new features to `_ALL_TIER_FEATURES`:
   - Append: `overload_x_hist`, `prob110_x_hist`
   - Append to `_ALL_TIER_MONOTONE`: `1, 1`

5. **Net for full run**: 34 - 6 pruned + 2 new = 30 features

6. **`ml/tests/`** — update any tests that assert feature count = 34 to 30

---

## Expected Impact

| Gate | Current (v0 mean) | Floor | Expected Direction |
|------|-------------------|-------|--------------------|
| Tier-VC@100 | 0.071 | 0.075 | ↑ Interactions should improve top-ranking for tier 0/1 |
| Tier-VC@500 | 0.230 | 0.217 | ↑ Slight improvement from better tier discrimination |
| Tier-NDCG | 0.771 | 0.767 | → Neutral to slight improvement |
| QWK | 0.370 | 0.359 | ↑ Better tier 1 recall should improve ordinal consistency |

**Most impactful gate**: Tier-VC@100 (currently below floor at 0.071 vs 0.075). Interaction features directly address the top-ranking problem by providing the model with pre-computed compound severity signals.

**Secondary impact**: Tier-Recall@1 (monitor, 0.047). Interactions between price history and flow exceedance create features that discriminate tier 1 from tier 2/3.

---

## Risk Assessment

1. **Interaction features were dead in stage-2**: They might also be unhelpful for tier classification. Mitigated by: screening on 2 months first; the stage-2 context was regression, not multi-class classification.

2. **Pruning could remove useful signal**: The pruned features have low mean importance but might matter in specific months. Mitigated by: keeping flow features at the extremes (80, 85, 100, 105, 110) while pruning only the middle ones (90, 95).

3. **Overfitting to screen months**: Screening on only 2 months may not generalize. Mitigated by: choosing 1 weak + 1 strong month; the full 12-month run is the real test.

4. **Feature count increase (Hyp A winner path)**: Going from 34 to 37 features increases model complexity. With min_child_weight=25, this is manageable but worth monitoring.
