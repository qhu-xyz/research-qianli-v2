# Direction — Iter 2 (feat-eng-3-20260303-104101)

## Hypothesis H11: Derived Interaction Features + colsample_bytree Tuning

**Core idea**: v0008 confirmed that additive features improve NDCG (bot2 +0.0101), but VCAP@100 regressed (4W/8L, bot2 -0.0033). This suggests the 26-feature model is spreading importance too thinly for top-100 ranking — the most extreme binding cases require sharp feature interactions that raw features don't capture. Adding 3 derived interaction features specifically targeting binding severity, combined with increasing colsample_bytree from 0.8 to 0.9 (ensuring more features are available per tree), should improve VCAP@100 without sacrificing the NDCG gains.

**Secondary target**: Continue improving 2021-04 (NDCG 0.651, the absolute worst month) through features that better capture spring transition binding patterns.

## Specific Changes

### Add 3 new derived interaction features (26 → 29 features)

These features are computed from existing columns in `features.py` — they don't require new source loader columns. They should be added to the `interaction_cols` set and the `with_columns` computation block.

| Feature | Formula | Monotone | Physical meaning |
|---------|---------|----------|-----------------|
| `band_severity` | `prob_band_95_100 * expected_overload` | 1 | Combines near-binding mass with overload severity. Constraints that are both on the edge AND severe when they bind should rank highest. Targets VCAP@100 directly. |
| `sf_exceed_interaction` | `sf_max_abs * prob_exceed_100` | 1 | Combines network topology signal (shift factors from v0007) with physical exceedance probability. Constraints that are both topologically important AND physically overloaded are the highest-value targets. |
| `hist_seasonal_band` | `hist_da_max_season * prob_band_100_105` | 1 | Combines seasonal historical extremes with mild overload band. Constraints with high historical peaks AND current mild overload are likely to bind heavily. Enriches the #7 feature (hist_da_max_season) with a physical signal. |

### Increase colsample_bytree from 0.8 to 0.9

With 29 features and colsample_bytree=0.8, each tree sees only ~23 features. This means some trees may miss the critical features for top-100 ranking (especially the 3 new interactions). Increasing to 0.9 ensures ~26 features per tree, covering virtually all important features while still maintaining feature subsampling regularization.

**Why 0.9 not 1.0**: Full colsample_bytree (1.0) eliminates feature subsampling entirely, which could reduce regularization and hurt generalization. 0.9 is a conservative increase that maintains regularization while reducing the probability of critical feature dropout.

### File changes required

#### 1. `ml/config.py` — FeatureConfig.step1_features

Add after the historical enrichment section:
```python
# --- Derived interaction features (targeting VCAP@100) ---
("band_severity", 1),
("sf_exceed_interaction", 1),
("hist_seasonal_band", 1),
```

#### 2. `ml/config.py` — HyperparamConfig.colsample_bytree

Change from 0.8 to 0.9:
```python
colsample_bytree: float = 0.9
```

#### 3. `ml/features.py` — interaction_cols set + computation

Expand the interaction_cols set:
```python
interaction_cols = {
    "exceed_severity_ratio", "hist_physical_interaction", "overload_exceedance_product",
    "band_severity", "sf_exceed_interaction", "hist_seasonal_band",
}
```

Add new computations to the `with_columns` block:
```python
(pl.col("prob_band_95_100") * pl.col("expected_overload"))
    .alias("band_severity"),
(pl.col("sf_max_abs") * pl.col("prob_exceed_100"))
    .alias("sf_exceed_interaction"),
(pl.col("hist_da_max_season") * pl.col("prob_band_100_105"))
    .alias("hist_seasonal_band"),
```

**Note**: The source features `prob_band_95_100`, `expected_overload`, `sf_max_abs`, `prob_exceed_100`, `hist_da_max_season`, and `prob_band_100_105` are all already loaded from the source data loader before `prepare_features()` is called. These interactions just need to be computed from existing columns.

#### 4. `ml/data_loader.py` — `_load_smoke()` synthetic data

Add synthetic generators for the 3 new interaction features:
```python
# Derived interaction features (for smoke tests only — computed in features.py for real data)
data["band_severity"] = (np.array(data["prob_band_95_100"]) * np.array(data["expected_overload"])).tolist()
data["sf_exceed_interaction"] = (np.array(data["sf_max_abs"]) * np.array(data["prob_exceed_100"])).tolist()
data["hist_seasonal_band"] = (np.array(data["hist_da_max_season"]) * np.array(data["prob_band_100_105"])).tolist()
```

**Actually, these should NOT be added to the smoke data** — they're computed by `features.py` from the component columns, which already exist in smoke data. The `interaction_cols` check in `features.py` will compute them automatically. Only add to `_load_smoke()` if the test expects them as pre-existing columns (check whether the smoke tests go through `features.py`).

**Clarification**: The existing interactions (hist_physical_interaction, overload_exceedance_product) are generated in smoke data directly AND computed in features.py. Follow the same pattern for consistency — add them to smoke data directly even though features.py will also compute them (the features.py computation will overwrite).

#### 5. `ml/tests/test_config.py` — update expected feature counts and lists

Update test assertions for feature count (26 → 29), monotone constraint string, and feature name list. Also update colsample_bytree from 0.8 to 0.9.

### What NOT to change

- train_months: keep 14 (HARD MAX)
- threshold_beta: keep 0.7
- Existing 26 features: do NOT remove any
- Other hyperparameters: keep v0 defaults (n_estimators=200, max_depth=4, etc.)
- gates.json and evaluate.py: NEVER modify

### Worker verification checklist

1. **Before coding**: Verify all component columns exist in the loaded DataFrame (prob_band_95_100, expected_overload, sf_max_abs, prob_exceed_100, hist_da_max_season, prob_band_100_105)
2. **After config change**: Run smoke tests (`SMOKE_TEST=true python -m pytest ml/tests/ -v`)
3. **After real data run**: Verify 29 features in model, check feature importance report — specifically check importance of the 3 new interactions
4. **Critical checks**:
   - VCAP@100 bot2 ≥ -0.0139 (L3 floor with v0008 as champion). Target: improve VCAP@100 vs v0008.
   - NDCG bot2 ≥ 0.6463 (L3 floor). Should maintain or improve.
   - All other Group A L3 floors: AUC ≥ 0.7999, AP ≥ 0.3526.

## Expected Impact

| Metric | v0008 (champion) | Expected direction | Reasoning |
|--------|------------------|-------------------|-----------|
| S1-AUC | 0.8498 | Maintain | Interactions don't add new signal, just sharpen existing |
| S1-AP | 0.4418 | Maintain | Same reasoning |
| S1-NDCG | 0.7346 | Maintain or slight improve | band_severity targets binding intensity ranking |
| S1-VCAP@100 | 0.0240 | **Improve** | Direct target — band_severity and sf_exceed_interaction capture top-of-distribution binding severity |
| S1-BRIER | 0.1383 | Maintain | Interactions typically don't help calibration |

**Key expectation**: VCAP@100 W/L should flip from 4W/8L to >6W/6L. Bot2 should maintain or improve. If VCAP@100 regresses further, the interaction features are not capturing the right signal for extreme binding events.

## Risk Assessment

1. **VCAP@100 further regression**: MEDIUM — if interactions dilute rather than concentrate top-100 signal, VCAP@100 could worsen. L3 margin is +0.0167 — can tolerate bot2 regression of up to 0.0167 before failing.
2. **NDCG regression**: LOW — interactions between confirmed high-signal features should not hurt ranking. colsample_bytree increase helps by ensuring critical features are always available.
3. **Overfitting**: LOW — 29 features with 200 trees and colsample_bytree=0.9 (~26 features/tree). Still well-regularized. The 3 interactions are multiplicative, which XGBoost can approximate with depth but may benefit from explicit computation.
4. **colsample_bytree change**: LOW — 0.8→0.9 is a minor conservative increase. If it causes any degradation, it will be visible across all metrics uniformly.

## Layer 3 Non-Regression Floors (v0008 as champion)

| Metric | Champion bot2 | L3 floor (bot2 - 0.02) |
|--------|--------------|------------------------|
| S1-AUC | 0.8199 | **0.7999** |
| S1-AP | 0.3726 | **0.3526** |
| S1-VCAP@100 | 0.0061 | **-0.0139** |
| S1-NDCG | 0.6663 | **0.6463** (tightest) |

## Why these interactions and not others

- **NOT density_mean * prob_exceed_100**: density_mean has monotone +1 and prob_exceed_100 has monotone +1 — their product is monotonically correlated with prob_exceed_100 alone, adding limited new signal. band_severity (prob_band_95_100 * expected_overload) captures the near-boundary specificity that density_mean misses.
- **NOT tail_concentration * expected_overload**: tail_concentration has low importance (0.37%, #16) — building interactions on it risks amplifying noise. prob_band_95_100 (#5, 3.82%) is a much stronger base.
- **NOT hist_da * sf_max_abs**: Already implicitly captured by hist_physical_interaction (hist_da * prob_exceed_100) + sf_max_abs as separate features. sf_exceed_interaction uses sf_max_abs * prob_exceed_100 instead, which is more interpretable (network importance × physical overload).

## Batch Strategy Remaining

- **Iter 2** (this): Derived interactions + colsample_bytree=0.9. Target: VCAP@100 recovery.
- **Iter 3**: Depends on results. If VCAP@100 recovers → final optimization (consider n_estimators increase to 300 or learning_rate decrease). If still degraded → investigate feature selection or colsample_bytree=1.0.
