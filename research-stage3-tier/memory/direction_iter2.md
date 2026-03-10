# Direction — Iteration 2 (batch: tier-fe-2-20260305-001606)

## Constraint

**FE only** — only `features`, `monotone_constraints` in config.py and feature computation in features.py may change. All hyperparams, class weights, bins, midpoints FROZEN.

## Current State

- Champion: v0 (34 features)
- v0005 (iter1): 37 features (34 + overload_x_hist, prob110_x_recent_hist, tail_x_hist) — NOT promoted
- v0005 Tier-VC@100 = 0.0746 vs floor 0.0750 — fails L1 by 0.0004 (0.6% relative)
- All other Group A gates PASS all 3 layers
- Code already reflects v0005's 37 features (committed)
- Value-QWK barely passing (0.3918 vs 0.3914 floor) — fragile, monitor closely

## Analysis

**What worked in iter1**: Interaction features (price × flow) provided consistent +5.4% VC@100 improvement, strongest on weak months (bottom_2_mean +64%). XGBoost benefits from pre-computed products because they reduce the tree depth needed to capture compound severity signals.

**What reviewers recommended**: (1) Log transforms to compress long-tailed price/overload distributions — helps XGBoost split more effectively at the high end, (2) prob_range_high to capture flow concentration, (3) overload_x_recent_hist for a recent-price variant of the overload interaction.

**Strategy**: The gap is tiny (0.0004). We need broad, consistent improvement rather than one home-run month. Two approaches:
- **A (conservative)**: Add 4 diverse FE features — 2 log transforms + 2 interactions
- **B (aggressive)**: Add 7 features — A's 4 plus 3 more exceedance×price interactions

## Screen Months

- **Weak: 2021-11** — v0005's worst VC@100 month (0.0082), worst QWK (0.252), worst Value-QWK (0.184). This month drives bottom_2_mean. Improving it is critical for L3 gates.
- **Strong: 2021-09** — v0005's best VC@100 month (0.2489), strong NDCG (0.852), strong QWK (0.452). Must not regress.

**Rationale**: Different weak month from iter1 screening (was 2022-06) for diversity. 2021-11 is the absolute floor — it determines bottom_2_mean for VC@100 and QWK. If new features help here, we likely close the gap. 2021-09 stays as strongest month to guard against regression.

## PREREQUISITE CODE CHANGE (required for BOTH hypotheses)

Before screening, add new feature computations to `ml/features.py` in `compute_interaction_features()`. Add these expressions inside the `df.with_columns([...])` block:

```python
# Log transforms (compress long-tailed distributions)
(pl.col("hist_da") + 1).log().alias("log1p_hist_da"),
(pl.col("expected_overload") + 1).log().alias("log1p_expected_overload"),
# Recent price × overload interaction
(pl.col("expected_overload") * pl.col("recent_hist_da"))
    .alias("overload_x_recent_hist"),
# Flow concentration in near-binding zone
(pl.col("prob_exceed_90") - pl.col("prob_exceed_110"))
    .alias("prob_range_high"),
# Exceedance × long-term price interactions (for hypothesis B)
(pl.col("prob_exceed_110") * pl.col("hist_da"))
    .alias("prob110_x_hist"),
(pl.col("prob_exceed_105") * pl.col("hist_da"))
    .alias("prob105_x_hist"),
(pl.col("prob_exceed_100") * pl.col("hist_da"))
    .alias("prob100_x_hist"),
```

Note: Use `(col + 1).log()` for log1p — this is vectorized polars, no `map_elements` needed. All 7 computations should be added for both hypotheses (unused features computed but not selected are harmless). All source columns already exist in the raw data.

## Hypothesis A (primary): Add 4 diverse FE features (37 → 41)

**What**: Add log transforms + 2 new interactions:
1. `log1p_hist_da` (monotone +1): Compresses long-tailed DA price distribution. Helps XGBoost discriminate at the high end — linear splits on log-scale separate $50 from $5000 better than raw scale where most values cluster near 0.
2. `log1p_expected_overload` (monotone +1): Same compression for overload MW. Large overloads (100+ MW) become more separable from moderate ones (1-10 MW).
3. `overload_x_recent_hist` (monotone +1): Expected overload × recent DA price. Extends the overload_x_hist pattern to recent (shorter window) price signal, which is the #1 feature by importance (21.1%).
4. `prob_range_high` (monotone 0): P(flow 90-110%) = prob_exceed_90 - prob_exceed_110. Captures probability mass in the "near-binding" zone — constraints that frequently approach limits.

**Why A is primary**: Diverse signal types (compression + interaction + distribution shape). Log transforms are orthogonal to existing features — they provide a fundamentally different representation. Lower feature count means less dilution of colsample_bytree sampling.

**Overrides**:
```json
{"tier": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_exceed_90", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_variance", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_skewness", "density_kurtosis", "density_cv", "season_hist_da_3", "prob_below_85", "overload_x_hist", "prob110_x_recent_hist", "tail_x_hist", "log1p_hist_da", "log1p_expected_overload", "overload_x_recent_hist", "prob_range_high"], "monotone_constraints": [1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, -1, 1, 1, 1, 1, 1, 1, 0]}}
```

## Hypothesis B (alternative): Add 7 FE features (37 → 44)

**What**: All 4 from A plus 3 exceedance×price interactions:
5. `prob110_x_hist` (monotone +1): P(flow > 110%) × hist_da. Extreme exceedance × long-term price — strongest compound severity signal for tier 0.
6. `prob105_x_hist` (monotone +1): P(flow > 105%) × hist_da. Medium exceedance × price.
7. `prob100_x_hist` (monotone +1): P(flow > 100%) × hist_da. At-limit exceedance × price.

**Why**: Saturates the price×exceedance interaction space. Iter1 showed price×flow interactions helped — this extends to all exceedance thresholds crossed with long-term price. These 3 features specifically target tier 0/1 discrimination: constraints with both high exceedance probability AND high historical prices are the strongest tier 0 candidates.

**Risk**: 7 new features (44 total) may dilute colsample_bytree=0.8 sampling — each tree samples fewer important features. Redundancy between prob100/105/110_x_hist could waste splits on correlated features.

**Overrides**:
```json
{"tier": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_exceed_90", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_variance", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_skewness", "density_kurtosis", "density_cv", "season_hist_da_3", "prob_below_85", "overload_x_hist", "prob110_x_recent_hist", "tail_x_hist", "log1p_hist_da", "log1p_expected_overload", "overload_x_recent_hist", "prob_range_high", "prob110_x_hist", "prob105_x_hist", "prob100_x_hist"], "monotone_constraints": [1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, -1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]}}
```

## Winner Criteria

Compare screen results on 2 months (2021-11 weak, 2021-09 strong):

1. **Primary**: Higher mean Tier-VC@100 across the 2 screen months
2. **Safety**: If winner's QWK drops > 0.05 on either screen month vs v0005 baseline, pick the other
3. **Secondary safety**: If winner's Value-QWK drops > 0.03 on 2021-11 (the floor month), pick the other
4. **Tiebreaker**: Higher mean Tier0-AP across screen months

## Code Changes for Winner

After screening picks a winner:

### 1. `ml/features.py` — `compute_interaction_features()`
Already modified in prerequisite. Keep all 7 computations in the function regardless of winner (unused computed columns are harmless; only the config list controls model input).

### 2. `ml/config.py` — Feature lists

If **Hypothesis A wins** (41 features): Append to `_ALL_TIER_FEATURES` and `_ALL_TIER_MONOTONE`:
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
    "log1p_hist_da",
    "log1p_expected_overload",
    "overload_x_recent_hist",
    "prob_range_high",
]

_ALL_TIER_MONOTONE: list[int] = _V1_CLF_MONO_FOR_TIER + [
    1, 1,     # prob_exceed_85, prob_exceed_80
    1,        # recent_hist_da
    1, 1,     # season_hist_da_1, season_hist_da_2
    0, 0, 0,  # density_skewness, density_kurtosis, density_cv
    1,        # season_hist_da_3
    -1,       # prob_below_85
    1, 1, 1,  # overload_x_hist, prob110_x_recent_hist, tail_x_hist
    1, 1,     # log1p_hist_da, log1p_expected_overload
    1,        # overload_x_recent_hist
    0,        # prob_range_high
]
```

If **Hypothesis B wins** (44 features): Same as A, then also append:
```python
    "prob110_x_hist",
    "prob105_x_hist",
    "prob100_x_hist",
]
# monotone (append):
    1, 1, 1,  # prob110_x_hist, prob105_x_hist, prob100_x_hist
]
```

### 3. `ml/tests/` — Update feature count assertions
Update any test asserting `len(features) == 37` to match the winner (41 for A, 44 for B).

## Expected Impact

- **Tier-VC@100**: +3-8% improvement expected. Log transforms compress long tails, helping XGBoost make finer splits at the high end where top-100 ranking matters. The 0.0004 gap (0.6%) should be closeable.
- **Tier0-AP / Tier01-AP**: Modest improvement from better high-value constraint discrimination.
- **Tier-VC@500**: Should improve proportionally.
- **QWK**: Neutral to slight improvement.
- **Value-QWK**: MONITOR — currently passing by 0.0004. Any regression flips this gate.
- **Tier-Recall@1**: Not expected to improve — structural class weight issue, blocked by FE-only constraint.

## Risk Assessment

- **Low risk (A)**: 4 additive features, diverse signal types. Log transforms are well-established for long-tailed distributions. XGBoost ignores unhelpful features via low split gain.
- **Medium risk (B)**: 7 new features (44 total) risk diluting colsample_bytree sampling. Three correlated exceedance×price features may reduce efficiency.
- **Fragile metric**: Value-QWK passes by 0.0004. Both hypotheses could flip it. 2021-11 screen month is the Value-QWK floor (0.184) — watch this closely.
- **Last effective FE iteration**: If this doesn't close the gap, iter3 is the final attempt. We may exhaust FE-only improvements.
