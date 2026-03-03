# Direction — Iteration 1 (feat-eng-20260302-194243)

## Hypothesis

**H5: Training window expansion (10→14 months) breaks AUC ceiling by addressing late-2022 distribution shift**

The AUC ceiling at ~0.835 is confirmed across two independent levers:
- HP tuning (v0003): AUC 0W/11L — complexity not the bottleneck
- Interaction features (v0002): AUC 5W/6L/1T — information ceiling within feature set

The weakest months (2022-09, 2022-12) consistently underperform, and early months benefit more from changes than late months. This pattern suggests the 10-month rolling window may not capture sufficient seasonal diversity. Expanding to 14 months provides 40% more training examples, including data from seasons absent in the shorter window.

For eval month 2022-09 with `train_months=14`:
- v0 sees: ~2021-09 to 2022-07 (10 months) — misses spring 2021
- New sees: ~2021-05 to 2022-07 (14 months) — captures spring/summer 2021 patterns

## Specific Changes

### 1. Revert FeatureConfig to v0 baseline (14 features)

**File**: `ml/config.py` → `FeatureConfig.step1_features`

Remove the 3 interaction features added for v0002. This isolates the training window effect cleanly.

```python
# REMOVE these 3 entries from step1_features:
("exceed_severity_ratio", 1),
("hist_physical_interaction", 1),
("overload_exceedance_product", 1),
```

After removal, `step1_features` should have exactly 14 entries (the v0 baseline set).

### 2. Guard interaction feature computation in features.py

**File**: `ml/features.py` → `prepare_features()`

The interaction features are unconditionally computed on line 38-45. Guard them so they only compute if the config requests them:

```python
# In prepare_features(), BEFORE the df.select(cols):
interaction_cols = {"exceed_severity_ratio", "hist_physical_interaction", "overload_exceedance_product"}
if interaction_cols & set(cols):
    df = df.with_columns([
        (pl.col("prob_exceed_110") / (pl.col("prob_exceed_90") + 1e-6))
            .alias("exceed_severity_ratio"),
        (pl.col("hist_da") * pl.col("prob_exceed_100"))
            .alias("hist_physical_interaction"),
        (pl.col("expected_overload") * pl.col("prob_exceed_105"))
            .alias("overload_exceedance_product"),
    ])
```

Also add a schema guard: verify all requested feature columns exist in the DataFrame before `df.select(cols)`. This addresses the Codex MEDIUM from iter1:

```python
missing = set(cols) - set(df.columns) - interaction_cols  # interactions may be computed above
if missing:
    raise ValueError(f"Missing feature columns in data: {missing}")
```

### 3. Fix benchmark.py train_months plumbing (BUG FIX)

**File**: `ml/benchmark.py`

Three changes required:

**(a)** Add `train_months` parameter to `_eval_single_month()`:
```python
def _eval_single_month(
    auction_month: str,
    class_type: str,
    ptype: str,
    hyperparam_config: HyperparamConfig,
    feature_config: FeatureConfig,
    threshold_beta: float = 0.7,
    train_months: int = 10,       # <-- ADD THIS
    val_months: int = 2,          # <-- ADD THIS
) -> dict | None:
```

And pass them to PipelineConfig:
```python
config = PipelineConfig(
    auction_month=auction_month,
    class_type=class_type,
    period_type=ptype,
    threshold_beta=threshold_beta,
    train_months=train_months,    # <-- ADD THIS
    val_months=val_months,        # <-- ADD THIS
)
```

**(b)** Add `train_months` parameter to `run_benchmark()`:
```python
def run_benchmark(
    version_id: str,
    eval_months: list[str],
    class_type: str = "onpeak",
    ptype: str = "f0",
    registry_dir: str = "registry",
    hyperparam_config: HyperparamConfig | None = None,
    feature_config: FeatureConfig | None = None,
    threshold_beta: float = 0.7,
    train_months: int = 10,       # <-- ADD THIS
    val_months: int = 2,          # <-- ADD THIS
    overrides: dict | None = None,
) -> dict:
```

Extract `train_months` and `val_months` from overrides:
```python
if overrides:
    from ml.pipeline import _apply_overrides
    pc_dummy = PipelineConfig(threshold_beta=threshold_beta)
    hyperparam_config, pc_dummy = _apply_overrides(hyperparam_config, pc_dummy, overrides)
    threshold_beta = pc_dummy.threshold_beta
    train_months = pc_dummy.train_months    # <-- ADD THIS
    val_months = pc_dummy.val_months        # <-- ADD THIS
```

Pass them to each `_eval_single_month()` call:
```python
metrics = _eval_single_month(
    month, class_type, ptype, hyperparam_config, feature_config,
    threshold_beta, train_months, val_months  # <-- ADD THESE
)
```

**(c)** Fix hardcoded eval_config (line 178):
```python
"eval_config": {
    "eval_months": eval_months,
    "class_type": class_type,
    "ptype": ptype,
    "train_months": train_months,   # <-- was hardcoded 10
    "val_months": val_months,        # <-- was hardcoded 2
    "threshold_beta": threshold_beta,
},
```

### 4. Set train_months=14 for this version

**File**: `ml/config.py` → `PipelineConfig`

Change the default:
```python
train_months: int = 14  # was 10
```

**Rationale for changing the default rather than using overrides**: The overrides mechanism works, but changing the default makes the experiment config explicit in `config.py` and ensures it flows through all code paths (benchmark, pipeline, etc.) without needing every caller to pass it.

### 5. Update stale docstrings

**File**: `ml/features.py`
- Line 20: Change "14 feature columns" → "feature columns" (count depends on config)
- Line 30: Change "Feature matrix of shape (n_samples, 14)" → "Feature matrix of shape (n_samples, n_features)"

**File**: `ml/data_loader.py`
- Line 24: Change "14 feature columns" → "feature columns"

### 6. Keep all v0 hyperparameters

**IMPORTANT**: Do NOT change any hyperparameters. Keep v0 defaults:
- `n_estimators=200`, `max_depth=4`, `learning_rate=0.1`
- `subsample=0.8`, `colsample_bytree=0.8`
- `reg_alpha=0.1`, `reg_lambda=1.0`, `min_child_weight=10`

This isolates the training window effect.

## Run Instructions

After making the above changes:

1. Run tests: `python -m pytest ml/tests/ -v`
2. Run benchmark: `python ml/benchmark.py --version-id ${VERSION_ID}`
3. Run validate: `python ml/validate.py --version-id ${VERSION_ID}`
4. Run compare: `python ml/compare.py --version-id ${VERSION_ID} --baseline v0`
5. Commit, then write handoff

## Expected Impact

| Metric | Expected Direction | Rationale |
|--------|-------------------|-----------|
| S1-AUC | +0.002 to +0.008 | More diverse training examples improve discrimination |
| S1-AP | +0.005 to +0.015 | Better positive ranking from seasonal diversity |
| S1-NDCG | +0.002 to +0.010 | Improved ranking quality from richer training |
| S1-VCAP@100 | +0.001 to +0.005 | Top-100 benefit from better tail discrimination |
| S1-BRIER | ±0.002 | Neutral — calibration mostly unaffected by window size |
| Late-2022 months | Strongest improvement | 2022-09, 2022-12 gain most from additional historical context |

**Success criteria**: AUC improvement in ≥7/12 months AND mean AUC > 0.835

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Stale data dilution (old patterns hurt) | LOW | 14 months is moderate; not reaching back 2+ years |
| Data availability for earliest months | LOW | 2020-09 needs data from 2019-05; MISO data available |
| Compute time increase (~40%) | MEDIUM | 14-month lookback loads more data; monitor mem_mb() closely |
| Overfitting to seasonal patterns | LOW | Larger training set generally regularizes, not overfits |
| Neutral result (window isn't the answer) | MEDIUM | If AUC stays at ~0.835, need fundamentally new features or methods |

## What NOT to Change

- Hyperparameters (keep v0 defaults)
- `threshold_beta` (keep 0.7)
- `threshold_scaling_factor` (keep 1.0)
- `gates.json` or `evaluate.py` (HUMAN-WRITE-ONLY)
- `registry/v0/` (immutable)
- `val_months` (keep 2)
