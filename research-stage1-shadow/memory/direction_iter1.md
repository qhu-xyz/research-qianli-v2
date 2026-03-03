# Direction — Iteration 1 (feat-eng-2-20260303-092848, v0006)

## Hypothesis H9: Shift Factor + Constraint Metadata Features

**Hypothesis**: Adding 6 new features from entirely new signal categories (network topology via shift factors + constraint structural metadata) will break the AUC ceiling at ~0.836 because the current model is feature-starved, not complexity-starved. These features capture WHERE in the network a constraint sits (shift factors) and WHAT TYPE of constraint it is (interface vs line, MW limit) — information completely absent from the current density-curve-based feature set.

**Evidence supporting this hypothesis**:
1. 6 real-data experiments confirmed the model is feature-limited: HP tuning (-0.0025 AUC), window expansion (+0.0013 ceiling), interaction features (+0.0000), feature pruning (tradeoff only). The AUC operating range across all experiments is [0.832, 0.836] — a 0.004 span.
2. Feature importance shows 79% of model signal comes from historical trend/level features. Physical flow features provide only 18%. Adding a completely independent signal class (network topology) could break the ceiling.
3. The source data loader already computes all 6 features — they just need to be wired through.

## Specific Changes

### 1. Verify new columns exist in loaded DataFrame

In `ml/data_loader.py`, add a diagnostic print after data loading (after `train_data = pl.from_pandas(train_data_pd)`, around line 122):

```python
# Diagnostic: verify new feature columns are available
new_cols = ["sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit"]
available = [c for c in new_cols if c in train_data.columns]
missing = [c for c in new_cols if c not in train_data.columns]
print(f"[data_loader] new feature columns available: {available}")
if missing:
    print(f"[data_loader] WARNING: missing columns: {missing}")
```

**If any columns are missing**: The source loader may require specific config flags or the columns may be named differently. Check the source loader's `load_training_data()` output columns and adapt. Do NOT proceed with missing columns — report in handoff.

### 2. Update `ml/config.py` — FeatureConfig

Replace the `step1_features` list to add 6 new features (20 total: 13 retained from v0006 + 1 restored + 6 new):

```python
step1_features: list[tuple[str, int]] = field(
    default_factory=lambda: [
        # --- Density exceedance probabilities (core 5) ---
        ("prob_exceed_110", 1),
        ("prob_exceed_105", 1),
        ("prob_exceed_100", 1),
        ("prob_exceed_95", 1),
        ("prob_exceed_90", 1),
        # --- Density below-threshold probabilities ---
        ("prob_below_100", -1),
        ("prob_below_95", -1),
        ("prob_below_90", -1),
        # --- Severity signal ---
        ("expected_overload", 1),
        # --- Historical DA shadow price ---
        ("hist_da", 1),
        ("hist_da_trend", 1),
        # --- Interaction features (retained: top 2 of 3) ---
        ("hist_physical_interaction", 1),
        ("overload_exceedance_product", 1),
        # --- NEW: Shift factor features (network topology) ---
        ("sf_max_abs", 1),      # peak node sensitivity
        ("sf_mean_abs", 1),     # average sensitivity
        ("sf_std", 0),          # sensitivity spread (unconstrained)
        ("sf_nonzero_frac", 0), # constraint reach (unconstrained)
        # --- NEW: Constraint metadata ---
        ("is_interface", 0),    # flowgate vs line (unconstrained)
        ("constraint_limit", 0), # MW limit log-transformed (unconstrained)
    ]
)
```

**Key design decisions**:
- Start from the v0004 feature set (13 features from v0006 + the 2 interaction features already present = current 13). This is the CURRENT config (v0006 config has 13 features). We keep all 13 and ADD 6.
- `sf_max_abs` and `sf_mean_abs` get monotone=1 (higher sensitivity → more likely to bind)
- `sf_std`, `sf_nonzero_frac`, `is_interface`, `constraint_limit` get monotone=0 (direction uncertain)
- Total: 19 features (13 current + 6 new)

### 3. Update `ml/features.py` — prepare_features()

The new features (sf_max_abs, sf_mean_abs, sf_std, sf_nonzero_frac, is_interface, constraint_limit) should already exist in the DataFrame from the source loader. They do NOT require computation like interaction features. Update the `missing` check to also exclude them from the "must compute" set:

In `prepare_features()`, update the missing-column check. Currently:
```python
interaction_cols = {"exceed_severity_ratio", "hist_physical_interaction", "overload_exceedance_product"}
...
missing = set(cols) - set(df.columns) - interaction_cols
```

The new features are NOT interaction features — they come from the source loader. The existing logic should work if they're in the DataFrame. **However**, verify this by:
1. First checking if the columns exist in df.columns
2. If any new column is missing, raise a clear error (not silently fill with 0)

Add after the interaction computation block:
```python
# Verify source-loader features exist (not computed here — must come from data_loader)
source_features = {"sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit"}
source_needed = source_features & set(cols)
source_missing = source_needed - set(df.columns)
if source_missing:
    raise ValueError(
        f"Source-loader features missing from DataFrame: {source_missing}. "
        f"These must come from MisoDataLoader. Available columns: {sorted(df.columns)}"
    )
```

### 4. Update `ml/data_loader.py` — _load_smoke()

The smoke test generator needs to produce the new columns. Add synthetic values:
```python
# Add new feature columns for smoke test
for feat in ["sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit"]:
    if feat == "is_interface":
        data[feat] = (rng.random(n) < 0.3).astype(float).tolist()  # 30% interface
    elif feat == "constraint_limit":
        data[feat] = np.log1p(rng.uniform(100, 2000, n)).tolist()  # log-transformed MW
    else:
        data[feat] = np.abs(rng.randn(n)).tolist()  # positive SF features
```

### 5. Update tests

In `ml/tests/`, update any test that checks feature count, feature names, or monotone constraint strings to account for the new 19-feature configuration.

### 6. Do NOT change

- `train_months`: keep at 14 (HARD MAX)
- `threshold_beta`: keep at 0.7
- HPs: keep v0 defaults
- `evaluate.py`: do NOT modify
- `gates.json`: do NOT modify

## Expected Impact

| Metric | Expected Direction | Reasoning |
|--------|-------------------|-----------|
| S1-AUC | **+0.003 to +0.010** | Network topology is a fundamentally new signal class. If shift factors discriminate at all, this could push AUC past 0.840. |
| S1-AP | **+0.005 to +0.015** | More features for positive-class ranking. AP has been stagnant at ~0.394. |
| S1-VCAP@100 | **+0.005 to +0.020** | Topology features may help rank the most valuable constraints. |
| S1-NDCG | **+0.005 to +0.015** | Better ranking quality overall. |
| S1-BRIER | **neutral to +0.003** | More features could slightly degrade calibration (ongoing trend). |

**Success criteria** (from human input):
- **Promotion-worthy**: AUC > 0.840 AND 8+/12 wins AND AP > 0.400
- **Encouraging**: AUC > 0.837 with 7+/12 wins → continue refining
- **Marginal**: AUC 0.835-0.837 → features helped but not enough
- **Dead end**: AUC ≤ 0.835 or <6/12 wins → features don't discriminate, pivot

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| SF columns missing from source loader output | LOW | Verified: source loader computes all 6. Diagnostic print will catch issues. |
| SF features are noise (no discriminative power) | MEDIUM | Even if SF features are weak, they shouldn't HURT (monotone constraints prevent overfitting). 6 experiments show feature additions don't degrade AUC. |
| AP bot2 continues declining (now at -0.0094 vs v0) | MEDIUM | More features could either help (more signal) or hurt (more complexity for weak months). Monitor closely. Margin to Layer 3 fail is 0.0106. |
| BRIER headroom narrows further (now 0.0163) | MEDIUM | Group B (non-blocking). More features historically worsen BRIER slightly (+0.002-0.004). |
| Memory/compute increase from 6 more features | LOW | 19 features is still modest for XGBoost. No OOM risk. |
| Smoke test fails on new columns | LOW | Explicitly adding synthetic data for new columns in _load_smoke(). |

## Feature Importance Watch

After training, check feature importance for the 6 new features:
- If sf_max_abs or sf_mean_abs are in the top 5 → strong signal, this is a breakthrough
- If all 6 are below 2% → they're noise, but the base model is unaffected
- If is_interface or constraint_limit show >5% → structural metadata matters, explore further in iter 2

## Connection to Future Iterations

- **If H9 succeeds (AUC > 0.837)**: Iter 2 adds distribution shape + probability band features (density_mean, density_variance, density_entropy, tail_concentration, prob_band_95_100, prob_band_100_105). Cumulative with iter 1.
- **If H9 is marginal (0.835-0.837)**: Iter 2 tries a "kitchen sink" approach — add ALL remaining features at once (~25 features), then prune in iter 3.
- **If H9 fails (≤ 0.835)**: Iter 2 pivots to ranking-focused objectives (LambdaRank) or selective monotone constraint relaxation, since new data sources didn't help.
