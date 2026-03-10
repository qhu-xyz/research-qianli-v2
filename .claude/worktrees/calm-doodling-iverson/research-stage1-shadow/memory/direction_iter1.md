# Direction — Iter 1 (feat-eng-3-20260303-104101)

## Hypothesis H10: Distribution Shape + Near-Boundary Band + Seasonal Historical Features

**Core idea**: v0007's shift factors broke the AUC ceiling by adding network topology signal, but NDCG remained flat (5W/7L, bot2 regressed -0.0154). The new features help separate binders from non-binders (AUC) but don't help rank AMONG binders. Distribution shape and near-boundary features should improve ranking quality (NDCG) because they capture HOW strongly a constraint is likely to bind, not just WHETHER it will bind.

## Specific Changes

### Add 7 new source-loader features (19 → 26 features)

All 7 features are confirmed present in the source data loader (`research-spice-shadow-price-pred-qianli/src/shadow_price_prediction/data_loader.py`). They flow through `load_training_data()` into the DataFrame. No computation needed in features.py — just wire them through.

| Feature | Monotone | Physical meaning |
|---------|----------|-----------------|
| `density_mean` | 1 | Expected flow as fraction of limit. Higher = closer to binding. Captures distribution location — two constraints with same prob_exceed_100 but different density_mean have very different binding profiles. |
| `density_variance` | 0 | Flow uncertainty. High variance = wide distribution = harder to predict. May help calibration and worst-month robustness. |
| `density_entropy` | 0 | Information content of flow distribution. High entropy = uniform/uninformative density. Low entropy = peaked/certain. |
| `tail_concentration` | 1 | prob_exceed_100 / (prob_exceed_80 + 1e-9). How peaked the tail is near the binding threshold. Discriminates "gradually increasing" from "sharply peaked at limit" profiles. |
| `prob_band_95_100` | 1 | P(95% < flow < 100%). Mass in the near-binding band. Constraints with mass concentrated here are "on the edge" — strong NDCG discriminators. |
| `prob_band_100_105` | 1 | P(100% < flow < 105%). Mass in mild overload. Captures severity gradient among binders. |
| `hist_da_max_season` | 1 | Peak seasonal DA shadow price. Captures extreme historical events that the mean (hist_da) misses. Enriches historical signal — the #1 importance category. |

### File changes required

#### 1. `ml/config.py` — FeatureConfig.step1_features

Add after the constraint metadata section:

```python
# --- Distribution shape features ---
("density_mean", 1),
("density_variance", 0),
("density_entropy", 0),
# --- Near-boundary band features ---
("tail_concentration", 1),
("prob_band_95_100", 1),
("prob_band_100_105", 1),
# --- Historical enrichment ---
("hist_da_max_season", 1),
```

#### 2. `ml/features.py` — source_features set

Expand the `source_features` set to include the new columns:

```python
source_features = {
    "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac",
    "is_interface", "constraint_limit",
    "density_mean", "density_variance", "density_entropy",
    "tail_concentration", "prob_band_95_100", "prob_band_100_105",
    "hist_da_max_season",
}
```

#### 3. `ml/data_loader.py` — diagnostic check in `_load_real()`

Expand the `new_cols` list to include all source-loader features:

```python
new_cols = [
    "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac",
    "is_interface", "constraint_limit",
    "density_mean", "density_variance", "density_entropy",
    "tail_concentration", "prob_band_95_100", "prob_band_100_105",
    "hist_da_max_season",
]
```

#### 4. `ml/data_loader.py` — `_load_smoke()` synthetic data

Add synthetic generators for the 7 new features:

```python
# Distribution shape features
data["density_mean"] = (rng.uniform(0.5, 1.2, n) * np.where(binding, 1.05, 0.85)).tolist()
data["density_variance"] = np.abs(rng.randn(n) * 0.1).tolist()
data["density_entropy"] = rng.uniform(1.0, 5.0, n).tolist()

# Near-boundary band features
data["tail_concentration"] = np.where(binding, rng.uniform(0.3, 0.9, n), rng.uniform(0.01, 0.3, n)).tolist()
data["prob_band_95_100"] = np.abs(rng.randn(n) * 0.05).tolist()
data["prob_band_100_105"] = np.abs(rng.randn(n) * 0.03).tolist()

# Historical enrichment
data["hist_da_max_season"] = np.where(binding, rng.lognormal(2, 1, n), rng.exponential(0.5, n)).tolist()
```

Also add these to `sf_meta_features` (rename to `source_loader_features`):
```python
source_loader_features = {
    "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac",
    "is_interface", "constraint_limit",
    "density_mean", "density_variance", "density_entropy",
    "tail_concentration", "prob_band_95_100", "prob_band_100_105",
    "hist_da_max_season",
}
```

#### 5. Tests — update expected feature counts and lists

Update test assertions for feature count (19 → 26) and any hardcoded feature lists.

### What NOT to change

- Hyperparameters: keep v0 defaults (n_estimators=200, max_depth=4, etc.)
- train_months: keep 14 (HARD MAX)
- threshold_beta: keep 0.7
- Existing 19 features: do NOT remove any
- gates.json and evaluate.py: NEVER modify

### Worker verification checklist

1. **Before coding**: Print all available columns from the source loader DataFrame to confirm the 7 new columns exist
2. **After config change**: Run smoke tests (`SMOKE_TEST=true python -m pytest ml/tests/ -v`)
3. **After real data run**: Verify 26 features in model, check feature importance report
4. **Critical check**: NDCG bot2 ≥ 0.6362 (L3 floor). If below, the version CANNOT be promoted.

## Expected Impact

| Metric | v0007 (champion) | Expected direction | Reasoning |
|--------|------------------|-------------------|-----------|
| S1-AUC | 0.8485 | Maintain or improve (+0.005?) | Adding information; distribution features are orthogonal to shift factors |
| S1-AP | 0.4391 | Maintain or improve | density_mean and hist_da_max_season directly help positive class ranking |
| S1-NDCG | 0.7333 | **Improve (+0.01 to +0.02)** | Band features discriminate binding intensity — directly targets ranking quality |
| S1-VCAP@100 | 0.0247 | Improve | Better ranking → better value capture at top |
| S1-BRIER | 0.1395 | Maintain | More features may help calibration slightly |

**Target for promotion**: Maintain AUC ≥ 0.845, AP ≥ 0.430, and improve NDCG to ≥ 0.740 with bot2 ≥ 0.660.

## Risk Assessment

1. **Feature existence**: LOW — all 7 confirmed in source loader code. Worker should still verify at runtime.
2. **Overfitting**: LOW-MEDIUM — 26 features with 200 trees and colsample_bytree=0.8 (only ~21 features sampled per tree). The regularization should be adequate.
3. **NDCG regression**: MEDIUM — if distribution shape features add noise to relative ordering among binders, NDCG could worsen. Bot2 floor is 0.6362 — margin is thin.
4. **Memory**: LOW — 7 additional float columns on ~270K rows = ~15 MB. Well within budget.

## Layer 3 Non-Regression Floors (v0007 as champion)

These are the absolute floors the new version must meet:

| Metric | Champion bot2 | L3 floor (bot2 - 0.02) |
|--------|--------------|------------------------|
| S1-AUC | 0.8188 | **0.7988** |
| S1-AP | 0.3685 | **0.3485** |
| S1-VCAP@100 | 0.0094 | **-0.0106** |
| S1-NDCG | 0.6562 | **0.6362** (tightest) |

## Why these features and not others

- **NOT density_skewness/kurtosis/cv**: These were in v0's original 14 features and were identified as noise candidates (1.3% combined importance). Removed during v0006 feature pruning. density_mean/variance/entropy are strictly more informative — they capture location and spread, not just higher-order shape moments.
- **NOT forecast_horizon**: Always 0 for f0 (prompt month). Would be a constant column. Useful only if/when we model f1/f2.
- **NOT derived interactions** (sf_x_exceed, hist_da_x_sf): Save for iter 2 — let the raw features establish signal first before adding interactions.

## Batch Strategy (3 iterations)

- **Iter 1** (this): Add 7 distribution/band/seasonal features. Establish signal.
- **Iter 2**: Depends on results. If NDCG improves → add derived interactions (sf_x_exceed, overload_severity). If NDCG flat → try monotone constraint tuning on new features or targeted feature selection.
- **Iter 3**: Final optimization. If cumulative gains are strong → prepare for HUMAN_SYNC with gate calibration recommendations.
