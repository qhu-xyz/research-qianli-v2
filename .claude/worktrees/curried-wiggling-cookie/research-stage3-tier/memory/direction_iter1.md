# Direction — Iteration 1 (batch: tier-fe-2-20260305-001606)

## Constraint

**FE only** — only `features`, `monotone_constraints` in config.py and feature computation in features.py may change. All hyperparams, class weights, bins, midpoints FROZEN.

## Current State

- Champion: v0 (34 features, no interactions)
- Tier-VC@100 mean=0.0708 vs floor=0.075 — **FAILING Layer 1** (only Group A gate below floor)
- Tier-VC@500 mean=0.2296, Tier0-AP mean=0.306, Tier01-AP mean=0.311 — barely passing
- Tier-Recall@1 catastrophically low (0.047) — missing most tier 1 constraints
- Feature importance: recent_hist_da (21.1%), hist_da (13.3%) dominate; bottom 5 features all ~1.1%

## Screen Months

- **Weak: 2022-06** — worst multi-metric month: VC@500=0.045, Tier0-AP=0.114, NDCG=0.643, QWK=0.273
- **Strong: 2021-09** — best VC@100 (0.248), best VC@500 (0.381), NDCG=0.855

**Rationale**: 2022-06 is the single worst month across 4+ metrics — improvements should show here. 2021-09 is already strong — we must not regress.

## PREREQUISITE CODE CHANGE (required for BOTH hypotheses)

Before screening either hypothesis, add 3 new interaction feature computations to `features.py`. Modify `compute_interaction_features()` to add these columns:

```python
def compute_interaction_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        # Existing dead features (keep for backward compat)
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

This code change is shared by both hypotheses and MUST be applied before any screening.

## Hypothesis A (primary): Add 3 interaction features (34 → 37)

**What**: Add `overload_x_hist`, `prob110_x_recent_hist`, `tail_x_hist` to the existing 34 features.

**Why**: The top-2 features by importance (recent_hist_da 21.1%, hist_da 13.3%) capture historical price signal, and physical flow features (expected_overload, prob_exceed_110, tail_concentration) capture binding severity. Their products create explicit compound signals that XGBoost would otherwise need 2+ tree levels to learn. This should improve tier 0/1 discrimination — constraints with BOTH high historical price AND high flow exceedance are more likely to be severely binding.

**Overrides** (after prerequisite code change):
```json
{"tier": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_exceed_90", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_variance", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_skewness", "density_kurtosis", "density_cv", "season_hist_da_3", "prob_below_85", "overload_x_hist", "prob110_x_recent_hist", "tail_x_hist"], "monotone_constraints": [1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, -1, 1, 1, 1]}}
```

## Hypothesis B (alternative): Add 3 interactions + prune 5 low-importance (34 - 5 + 3 = 32)

**What**: Add the same 3 interaction features but also remove the 5 lowest-importance features: `density_skewness`, `density_cv`, `prob_exceed_90`, `density_variance`, `prob_below_90` (all ~1.1% importance).

**Why**: Pruning low-signal features reduces noise in tree splits and concentrates sampling on discriminative features. Combined with the new interactions, this creates a tighter, more focused feature set. Stage-2 learnings showed pruning 5 dead features improved EV-VC@100 by 5.2%.

**Overrides** (after prerequisite code change):
```json
{"tier": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_kurtosis", "season_hist_da_3", "prob_below_85", "overload_x_hist", "prob110_x_recent_hist", "tail_x_hist"], "monotone_constraints": [1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, -1, 1, 1, 1]}}
```

Note: `prob_below_90` is kept (monotone -1); the removed features are `density_skewness`, `density_cv`, `prob_exceed_90`, `density_variance`, `prob_below_90`. Wait — correction: remove `prob_below_90` too (it has ~1.1% importance). The list above reflects removal of all 5.

## Winner Criteria

Compare screen results on the 2 months (2022-06, 2021-09):

1. **Primary**: Higher mean Tier-VC@100 across the 2 screen months
2. **Safety**: If winner's Tier-VC@100 is higher but QWK drops > 0.05 on either screen month vs v0, pick the other
3. **Tiebreaker**: Higher mean Tier0-AP across screen months

## Code Changes for Winner

After screening picks a winner, apply these permanent code changes:

### 1. `ml/features.py` — `compute_interaction_features()`
Already modified in prerequisite. No additional changes needed.

### 2. `ml/config.py` — Feature lists
If **Hypothesis A wins**: Append 3 features/monotone values to `_ALL_TIER_FEATURES` and `_ALL_TIER_MONOTONE`:
```python
_ALL_TIER_FEATURES: list[str] = _V1_CLF_FOR_TIER + [
    "prob_exceed_85",
    "prob_exceed_80",
    "recent_hist_da",
    "season_hist_da_1",
    "season_hist_da_2",
    "density_skewness",
    "density_kurtosis",
    "density_cv",
    "season_hist_da_3",
    "prob_below_85",
    "overload_x_hist",
    "prob110_x_recent_hist",
    "tail_x_hist",
]

_ALL_TIER_MONOTONE: list[int] = _V1_CLF_MONO_FOR_TIER + [
    1, 1,     # prob_exceed_85, prob_exceed_80
    1,        # recent_hist_da
    1, 1,     # season_hist_da_1, season_hist_da_2
    0, 0, 0,  # density_skewness, density_kurtosis, density_cv
    1,        # season_hist_da_3
    -1,       # prob_below_85
    1, 1, 1,  # overload_x_hist, prob110_x_recent_hist, tail_x_hist
]
```

If **Hypothesis B wins**: Same as A but also remove 5 features. Set `_ALL_TIER_FEATURES` and `_ALL_TIER_MONOTONE` to match the B overrides list above.

### 3. `ml/tests/` — Update feature count assertions
Update any test that asserts `len(features) == 34` to match the new count (37 for A, 32 for B).

## Expected Impact

- **Tier-VC@100**: +5-15% improvement expected. Interaction features should help rank tier 0/1 constraints higher by providing explicit compound severity signal.
- **Tier0-AP**: Modest improvement expected from better tier 0 discrimination.
- **Tier-VC@500**: Should improve proportionally.
- **QWK**: Neutral to slight improvement — interactions don't change ordinal structure much.
- **Tier-Recall@1**: May not improve significantly — this is more of a class weight / threshold issue (blocked in FE-only batch).

## Risk Assessment

- **Low risk**: Adding features is additive — XGBoost can ignore unhelpful features via low split gain. Worst case is no improvement, not regression.
- **Medium risk (B only)**: Pruning removes some information. If the pruned features are used for edge cases in certain months, we could see tail regression.
- **Mitigation**: The 2-month screen catches regression on the strong month (2021-09).
