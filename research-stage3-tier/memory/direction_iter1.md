# Direction — Iteration 1 (batch: tier-fe-2)

## Context

Champion is v0 with 34 features. This is an FE-only batch — only features, monotone constraints, and `features.py` may change. All hyperparameters, class weights, bins, and midpoints are FROZEN.

**Gate status** (Group A blocking):
- Tier-VC@100: mean=0.071, floor=0.075 — **FAILING Layer 1** (only failing gate)
- Tier-VC@500: mean=0.230, floor=0.217 — passing
- Tier0-AP: mean=0.306, floor=0.306 — barely passing
- Tier01-AP: mean=0.311, floor=0.311 — barely passing

**Key weakness**: The model lacks explicit compound severity signals. Top features (recent_hist_da 21.1%, hist_da 13.3%) are used independently. XGBoost at max_depth=5 may struggle to learn interactions between flow exceedance and price history. Pre-computing these interactions should help tier 0/1 discrimination.

## REQUIRED CODE CHANGE (do this BEFORE screening)

Both hypotheses need new interaction features computed. Modify `ml/features.py` — add 3 new columns in `compute_interaction_features()`:

```python
def compute_interaction_features(df: pl.DataFrame) -> pl.DataFrame:
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
        # NEW interaction features
        (pl.col("expected_overload") * pl.col("hist_da"))
            .alias("overload_x_hist"),
        (pl.col("prob_exceed_110") * pl.col("recent_hist_da"))
            .alias("prob110_x_recent_hist"),
        (pl.col("tail_concentration") * pl.col("hist_da"))
            .alias("tail_x_hist"),
    ])
```

Make this code change FIRST, then screen both hypotheses with `--overrides`.

## Hypothesis A (primary): Add 3 interaction features (34 → 37)

**What**: Add overload_x_hist, prob110_x_recent_hist, tail_x_hist to the existing 34 features.

**Rationale**: These combine the two highest-importance features (recent_hist_da 21.1%, hist_da 13.3%) with physical flow signals (expected_overload, prob_exceed_110, tail_concentration). The products create explicit compound severity signals that should help distinguish high-tier constraints (where BOTH flow and historical price are elevated) from lower tiers.

**Monotone constraints for new features**: all +1 (both factors are +1 monotone).

Hypothesis A overrides:
```json
{"tier": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_exceed_90", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_variance", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_skewness", "density_kurtosis", "density_cv", "season_hist_da_3", "prob_below_85", "overload_x_hist", "prob110_x_recent_hist", "tail_x_hist"], "monotone_constraints": [1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, -1, 1, 1, 1]}}
```

## Hypothesis B (alternative): Add 3 interactions + prune 4 low-importance (34 → 33)

**What**: Add the same 3 interaction features, but also remove the 4 lowest-importance features: density_skewness (rank 34, 1.09%), prob_exceed_90 (rank 33, 1.13%), density_cv (rank 32, 1.13%), density_variance (rank 31, 1.17%).

**Rationale**: These 4 features contribute <1.2% importance each and may add noise, diluting tree splits. Removing them concentrates sampling on higher-signal features while adding the 3 interaction features provides more discriminative power. Net effect: slightly smaller feature space with higher average signal.

Hypothesis B overrides:
```json
{"tier": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_kurtosis", "season_hist_da_3", "prob_below_85", "overload_x_hist", "prob110_x_recent_hist", "tail_x_hist"], "monotone_constraints": [1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, -1, 1, 1, 1]}}
```

## Screen Months

- **Weak month: 2022-06** — worst Tier0-AP (0.114), worst Tier-NDCG (0.643), worst Tier-VC@500 (0.045), low VC@100 (0.025). Comprehensive disaster month across all metrics. Interaction features should help here if they improve tier 0/1 detection at all.
- **Strong month: 2021-09** — best Tier-VC@100 (0.248), best Tier-VC@500 (0.381), strong NDCG (0.855). Must not regress here — this is the champion's best month.

**Rationale**: 2022-06 tests whether interaction features help in the hardest conditions. 2021-09 tests whether they don't hurt in the best conditions. Together they bracket the performance range.

## Winner Criteria

Pick the hypothesis with **higher mean Tier-VC@100 across the 2 screen months**, unless:
- QWK drops > 0.05 vs v0 on either screen month → disqualify
- Tier0-AP drops below v0's tail_floor (0.114) on either month → disqualify

If both hypotheses are disqualified or tied within 0.005 on Tier-VC@100, prefer Hypothesis A (simpler, more features = less information loss risk).

## Code Changes for Winner

**If Hypothesis A wins** (add 3 features):
1. `ml/features.py` — already modified above (keep all 3 new columns)
2. `ml/config.py` — add to `_ALL_TIER_FEATURES`: `"overload_x_hist", "prob110_x_recent_hist", "tail_x_hist"` and to `_ALL_TIER_MONOTONE`: `1, 1, 1`
3. `ml/tests/` — update expected feature count from 34 to 37

**If Hypothesis B wins** (add 3, prune 4):
1. `ml/features.py` — already modified above
2. `ml/config.py` — add 3 new features to `_ALL_TIER_FEATURES` and `_ALL_TIER_MONOTONE` as above, THEN remove `density_skewness`, `prob_exceed_90`, `density_cv`, `density_variance` from both lists. Also add these 4 to `_DEAD_FEATURES` set.
3. `ml/tests/` — update expected feature count from 34 to 33

## Expected Impact

- **Tier-VC@100**: +0.005 to +0.015 (from 0.071 to ~0.076-0.086). Interaction features should improve top-of-ranking quality by making high-tier constraints more distinguishable.
- **Tier0-AP**: +0.01 to +0.03. Compound severity signals should improve tier 0 precision-recall.
- **Tier01-AP**: +0.005 to +0.02. Similar improvement for combined tier 0+1 detection.
- **QWK**: Neutral to +0.01. Better tier 0/1 detection shouldn't hurt ordinal consistency.

## Risk Assessment

- **Low risk**: Interaction features are products of positively-correlated factors. Monotone constraint (+1) is correct. Worst case is no improvement.
- **Pruning risk (Hyp B only)**: Removing density_variance/cv/skewness loses distribution shape information. If these features help in specific months, Hyp B could regress on tail months. This is why Hyp A is primary.
- **Overfitting risk**: 37 features with max_depth=5 and 400 trees is well within capacity. Not a concern.
- **Previous batch failures**: Were worker execution bugs, not hypothesis issues. The same hypotheses (interaction features) remain untested and are the right first experiment for FE-only.
