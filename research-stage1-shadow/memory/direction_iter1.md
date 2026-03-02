# Direction — Iteration 1 (hp-tune-20260302-144146)

## Hypothesis

**H4: Interaction features provide new discriminative signal for ranking quality.**

The model is feature-limited at AUC ~0.835 with 14 independent features. Previous iteration (H3, v0003) proved that HP tuning cannot improve ranking — deeper trees + slower learning degraded AUC in 11/12 months while only improving calibration (BRIER). The 14 features have reached their informational ceiling for independent-feature splits.

Adding physically-motivated interaction features that capture cross-feature combinations should provide new discriminative signal that XGBoost depth-4 trees cannot easily discover through single-feature splits (a 2-feature interaction requires 2 splits, consuming half the tree depth; a pre-computed interaction needs only 1 split).

## Specific Changes

### 1. Revert HyperparamConfig to v0 defaults (CRITICAL — do this first)

**File**: `ml/config.py`, class `HyperparamConfig`

The config currently has v0003 values from the previous batch. These MUST be reverted:

| Param | Current (v0003) | Revert to (v0) |
|-------|----------------|----------------|
| `n_estimators` | 400 | **200** |
| `max_depth` | 6 | **4** |
| `learning_rate` | 0.05 | **0.1** |
| `min_child_weight` | 5 | **10** |

All other HP defaults (`subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=1.0`, `random_state=42`) remain unchanged.

### 2. Add 3 interaction features to FeatureConfig

**File**: `ml/config.py`, class `FeatureConfig`

Append these 3 features to `step1_features` (after the existing 14):

```python
# --- Interaction features (computed in prepare_features) ---
("exceed_severity_ratio", 1),       # prob_exceed_110 / (prob_exceed_90 + 1e-6)
("hist_physical_interaction", 1),   # hist_da * prob_exceed_100
("overload_exceedance_product", 1), # expected_overload * prob_exceed_105
```

Total features: 14 → 17. All 3 new features get monotone constraint +1 (higher = more likely to bind).

### 3. Compute interaction features in prepare_features

**File**: `ml/features.py`, function `prepare_features`

Before the `df.select(cols)` line, compute the 3 new columns from existing features using polars expressions:

```python
def prepare_features(
    df: pl.DataFrame, config: FeatureConfig
) -> tuple[np.ndarray, list[str]]:
    cols = config.features
    print(f"[features] mem before prepare: {mem_mb():.0f} MB")

    # Compute interaction features from base columns
    df = df.with_columns([
        # Tail concentration: how much exceedance is in the extreme tail
        (pl.col("prob_exceed_110") / (pl.col("prob_exceed_90") + 1e-6))
            .alias("exceed_severity_ratio"),
        # Historical × physical confirmation
        (pl.col("hist_da") * pl.col("prob_exceed_100"))
            .alias("hist_physical_interaction"),
        # Severity-weighted likelihood
        (pl.col("expected_overload") * pl.col("prob_exceed_105"))
            .alias("overload_exceedance_product"),
    ])

    X = df.select(cols).fill_null(0).to_numpy()
    return X, cols
```

**Key design choice**: Features are computed in `prepare_features()`, not in the data loader. This means:
- No changes to `data_loader.py` (real or smoke mode)
- Smoke test data works automatically (interactions computed from random base features)
- All 14 base columns remain available as inputs to the interaction computation

### 4. Update tests

Existing tests should still pass since the default `FeatureConfig()` now includes 17 features and `prepare_features()` computes the interactions. Verify:
- `tests/test_features.py`: Feature count assertion changes from 14 → 17
- `tests/test_pipeline.py`: Smoke test should pass end-to-end with 17 features
- `tests/test_config.py`: Feature list length assertion may need updating

If any test hardcodes `14` for feature count, update to `17`.

### 5. Run benchmark

After all code changes + tests pass:
```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-stage1-shadow
python ml/benchmark.py --version-id ${VERSION_ID} --ptype f0 --class-type onpeak
```

Then run compare:
```bash
python ml/compare.py --gates-path registry/gates.json --registry-dir registry
```

## Feature Rationale

### exceed_severity_ratio = prob_exceed_110 / (prob_exceed_90 + 1e-6)

**Physical meaning**: Measures tail concentration of exceedance probability. A constraint with prob_exceed_110=0.30 and prob_exceed_90=0.40 (ratio=0.75) is far more severe than one with prob_exceed_110=0.01 and prob_exceed_90=0.40 (ratio=0.025). Both have similar prob_exceed_90, but the first has most of its exceedance in the extreme tail, indicating near-certain heavy binding.

**Why XGBoost can't easily discover this**: With monotone constraints, prob_exceed_110 and prob_exceed_90 are both constrained (+1 and +1). The ratio captures a RELATIVE signal (concentration) that requires the model to split on both features in a specific order, using 2 of 4 available depth levels.

### hist_physical_interaction = hist_da × prob_exceed_100

**Physical meaning**: Combines two independent information sources — historical binding (has this constraint bound before?) and physical likelihood (is the current flow distribution likely to exceed the limit?). When both are high, the constraint is a very strong binding candidate. When either is low, the signal is weak. This AND-like combination is exactly what matters for precision: we want constraints where BOTH signals agree.

**Business relevance**: This is the most important feature for precision. A constraint with high hist_da but low physical exceedance may have changed (line upgrade, topology change). A constraint with high physical exceedance but no history is unproven. Both signals together = high confidence.

### overload_exceedance_product = expected_overload × prob_exceed_105

**Physical meaning**: Expected MW overload × probability of significant (>105%) exceedance. This captures "how badly AND how likely" in a single number. A constraint with 50 MW expected overload and 40% chance of exceeding 105% is a stronger binding candidate than one with 100 MW overload but only 5% chance of exceedance.

**Why it helps VCAP@100**: This feature directly captures the VALUE component — constraints with high overload × high exceedance probability tend to produce larger shadow prices, improving value capture at top-K.

## Expected Impact

| Metric | v0 Mean | Expected Direction | Rationale |
|--------|---------|-------------------|-----------|
| S1-AUC | 0.8348 | +0.005 to +0.015 | New discriminative signal for separation |
| S1-AP | 0.3936 | +0.010 to +0.025 | Better ranking of positives in imbalanced setting |
| S1-NDCG | 0.7333 | +0.005 to +0.015 | Better ranking quality at top positions |
| S1-VCAP@100 | 0.0149 | +0.005 to +0.020 | overload_exceedance_product targets value capture directly |
| S1-BRIER | 0.1503 | ±0.002 | Neutral — features help ranking, not calibration |

**Win/loss target**: AUC improvement in ≥8/12 months (vs 0/11 for v0003). If < 6/12, the features aren't providing consistent signal.

## Risk Assessment

### Low Risk
- **No HP changes vs v0** — reverting to proven defaults isolates the feature effect
- **Additive features** — 14 base features unchanged; 3 new features can only add signal
- **Monotone constraints preserved** — all 3 new features have physically justified +1 constraints
- **No data changes** — same 12 eval months, same train/val splits, same data loader

### Medium Risk
- **Feature noise**: If interaction features are noisy (high variance across months), they could slightly degrade stability. Mitigated by: monotone constraints prevent overfitting to noise direction; subsample=0.8 + colsample=0.8 provide regularization.
- **Collinearity**: `exceed_severity_ratio` is correlated with `prob_exceed_110` (numerator). XGBoost handles collinearity well (greedy split selection), but the tree may split redundantly. If AUC doesn't improve, this is a possible explanation.

### Not a Risk
- **BRIER regression**: These features improve ranking, not calibration. BRIER should be neutral (±0.002). The 0.02 headroom to BRIER floor (0.170) is safe.
- **Gate failures**: v0 passes all gates with ~0.05 headroom. Even a small regression would not breach floors.

## Success Criteria

1. **Primary**: AUC mean > 0.8348 (any improvement over v0)
2. **Primary**: AP mean > 0.3936 (any improvement over v0)
3. **Consistency**: AUC improved in ≥8/12 months (statistical significance)
4. **No regression**: All Group A gates pass all 3 layers
5. **Tail safety**: Bottom-2 AUC ≥ 0.810 (v0 = 0.811)
