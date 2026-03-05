# Direction -- Iteration 2 (batch: tier-fe-2)

## Context

Champion is v0 with 34 features. This is an FE-only batch -- only features, monotone constraints, and `features.py` may change. All hyperparameters, class weights, bins, and midpoints are FROZEN.

**3 consecutive worker failures** across 2 batches. No hypotheses have been tested yet. The interaction feature hypothesis remains the correct first experiment.

**Gate status (Group A blocking):**

| Gate | v0 Mean | Floor | Status |
|------|---------|-------|--------|
| Tier-VC@100 | 0.071 | 0.075 | **FAILING Layer 1** |
| Tier-VC@500 | 0.230 | 0.217 | passing |
| Tier0-AP | 0.306 | 0.306 | barely passing |
| Tier01-AP | 0.311 | 0.311 | barely passing |

**Key weakness**: The model lacks explicit compound severity signals. Top features (recent_hist_da 21.1%, hist_da 13.3%) are used independently. Pre-computing interaction products should help tier 0/1 discrimination and push Tier-VC@100 above 0.075.

## REQUIRED CODE CHANGE (do this BEFORE screening)

Both hypotheses need 3 new interaction features computed in `ml/features.py`. Add them to `compute_interaction_features()`:

```python
def compute_interaction_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute derived interaction features from raw columns."""
    return df.with_columns([
        # Existing 5 dead features (keep for backward compat)
        (pl.col("hist_da") * pl.col("prob_exceed_100"))
            .alias("hist_physical_interaction"),
        (pl.col("expected_overload") * pl.col("prob_exceed_105"))
            .alias("overload_exceedance_product"),
        (pl.col("prob_band_95_100") * pl.col("expected_overload"))
            .alias("band_severity"),
        (pl.col("sf_max_abs") * pl.col("prob_exceed_100"))
            .alias("sf_exceed_interaction"),
        (pl.col("hist_da_max_season") * pl.col("prob_band_100_105"))
            .alias("hist_seasonal_band"),
        # NEW interaction features (3)
        (pl.col("expected_overload") * pl.col("hist_da"))
            .alias("overload_x_hist"),
        (pl.col("prob_exceed_110") * pl.col("recent_hist_da"))
            .alias("prob110_x_recent_hist"),
        (pl.col("tail_concentration") * pl.col("hist_da"))
            .alias("tail_x_hist"),
    ])
```

## Hypothesis A (primary): Add 3 interaction features (34 -> 37)

**What**: Add overload_x_hist, prob110_x_recent_hist, tail_x_hist to the existing 34 features.

**Rationale**: These combine the two highest-importance features (recent_hist_da 21.1%, hist_da 13.3%) with physical flow signals (expected_overload, prob_exceed_110, tail_concentration). The products create explicit compound severity signals that should help distinguish tier 0/1 constraints from lower tiers. At max_depth=5, XGBoost may struggle to learn these interactions implicitly.

**Monotone constraints for new features**: all +1 (both factors in each product are +1 monotone).

Hypothesis A overrides:
```json
{"tier": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_exceed_90", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_variance", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_skewness", "density_kurtosis", "density_cv", "season_hist_da_3", "prob_below_85", "overload_x_hist", "prob110_x_recent_hist", "tail_x_hist"], "monotone_constraints": [1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, -1, 1, 1, 1]}}
```

## Hypothesis B (alternative): Add 3 interactions + prune 4 low-importance (34 -> 33)

**What**: Add the same 3 interaction features, but also remove the 4 lowest-importance features: density_skewness (rank 34, 1.09%), prob_exceed_90 (rank 33, 1.13%), density_cv (rank 32, 1.13%), density_variance (rank 31, 1.17%).

**Rationale**: These 4 features each contribute <1.2% importance by gain and may add noise. Removing them concentrates tree splits on higher-signal features. Net: 34 - 4 + 3 = 33 features.

Hypothesis B overrides:
```json
{"tier": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_kurtosis", "season_hist_da_3", "prob_below_85", "overload_x_hist", "prob110_x_recent_hist", "tail_x_hist"], "monotone_constraints": [1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, -1, 1, 1, 1]}}
```

## Screen Months

- **Weak month: 2022-06** -- worst Tier0-AP (0.114), worst Tier-NDCG (0.643), worst Tier-VC@500 (0.045), low VC@100 (0.025). Comprehensive worst month. Interaction features should help here if they help at all.
- **Strong month: 2021-09** -- best Tier-VC@100 (0.248), best Tier-VC@500 (0.381), strong NDCG (0.855). Must not regress here.

**Rationale**: 2022-06 tests improvement potential on the hardest month. 2021-09 tests that we don't hurt the best month. Together they bracket the performance range.

## Winner Criteria

Pick the hypothesis with **higher mean Tier-VC@100 across the 2 screen months**, unless:
- QWK drops > 0.05 vs v0 on either screen month -> disqualify
- Tier0-AP drops below v0's tail_floor (0.114) on either month -> disqualify

If both tied within 0.005 on Tier-VC@100, prefer Hypothesis A (simpler, no information loss from pruning).

## Code Changes for Winner

**If Hypothesis A wins** (add 3 features, 34 -> 37):
1. `ml/features.py` -- already modified above (keep all 3 new columns in `compute_interaction_features`)
2. `ml/config.py` -- append to `_ALL_TIER_FEATURES`: `"overload_x_hist", "prob110_x_recent_hist", "tail_x_hist"` and to `_ALL_TIER_MONOTONE`: `1, 1, 1`
3. `ml/tests/` -- update expected feature count from 34 to 37

**If Hypothesis B wins** (add 3 + prune 4, 34 -> 33):
1. `ml/features.py` -- already modified above
2. `ml/config.py` -- append 3 new features and monotone values as above, THEN add `density_skewness`, `prob_exceed_90`, `density_cv`, `density_variance` to `_DEAD_FEATURES` set
3. `ml/tests/` -- update expected feature count from 34 to 33

## Expected Impact

- **Tier-VC@100**: +0.005 to +0.015 (from 0.071 to ~0.076-0.086). The only failing Group A gate; interaction features should improve top-of-ranking quality.
- **Tier0-AP**: +0.01 to +0.03. Compound severity signals should improve tier 0 precision-recall.
- **Tier01-AP**: +0.005 to +0.02. Similar improvement for combined tier 0+1 detection.
- **QWK**: Neutral to +0.01. Better tier 0/1 detection shouldn't hurt ordinal consistency.

## Risk Assessment

- **Low risk**: Interaction features are products of positively-correlated factors. Monotone constraint (+1) is correct. Worst case is no improvement (neutral).
- **Pruning risk (Hyp B only)**: Removing density_variance/cv/skewness loses distribution shape information. If these help in specific months, Hyp B could regress on tail months. This is why Hyp A is primary.
- **Overfitting risk**: 37 features with max_depth=5 and 400 trees is well within capacity. Not a concern.
