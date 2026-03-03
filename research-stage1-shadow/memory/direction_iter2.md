# Direction — Iteration 2 (feat-eng-20260302-194243)

## Hypothesis

**H6: Combining 14-month training window with interaction features produces additive improvement**

Three real-data iterations have isolated individual levers:
- v0003-HP (HP tuning): AUC -0.0025, 0W/11L — model not complexity-limited
- v0002 (interaction features): AUC +0.0000, 5W/6L/1T — but NDCG 8W/4L and AP 7W/5L (positive ranking signal)
- v0003 (14-month window): AUC +0.0013, 7W/4L/1T — best AUC signal, VCAP@100 9W/3L

The two positive-signal levers (interactions and window expansion) were tested independently. If their effects are additive, the combined version should show AUC ~+0.0013 and NDCG ~+0.0035 — potentially the first version with robust enough improvement to promote. Even partial additivity would strengthen the win/loss ratios.

This also includes bug fixes for f2p parsing (Codex HIGH) and dual-default fragility (Claude MEDIUM).

## Specific Changes

### 1. Re-add interaction features to FeatureConfig

**File**: `ml/config.py` → `FeatureConfig.step1_features`

Add back the 3 interaction features after the existing 14 base features:

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
        # --- Distribution shape (unconstrained) ---
        ("density_skewness", 0),
        ("density_kurtosis", 0),
        ("density_cv", 0),
        # --- Historical DA shadow price ---
        ("hist_da", 1),
        ("hist_da_trend", 1),
        # --- Interaction features ---
        ("exceed_severity_ratio", 1),
        ("hist_physical_interaction", 1),
        ("overload_exceedance_product", 1),
    ]
)
```

Total: 17 features (14 base + 3 interactions). All monotone +1 for interactions.

### 2. Keep train_months=14 (from v0003)

**File**: `ml/config.py` → `PipelineConfig`

No change needed — `train_months` is already 14 from iteration 1.

### 3. Fix f2p period type parsing (BUG FIX — Codex HIGH)

**File**: `ml/data_loader.py`, line 103

Current code:
```python
horizon = int(ptype[1:]) if ptype.startswith("f") else 3
```

`int("2p")` crashes for ptype="f2p". Fix with explicit mapping:

```python
# Parse forecast horizon from ptype (e.g., "f0"→0, "f1"→1, "f2"→2, "f2p"→2)
_PTYPE_HORIZON = {"f0": 0, "f1": 1, "f2": 2, "f2p": 2}
horizon = _PTYPE_HORIZON.get(ptype, 3)
```

### 4. Fix dual-default fragility in benchmark.py (Claude MEDIUM)

**File**: `ml/benchmark.py`

Change `_eval_single_month()` and `run_benchmark()` to use `None` sentinel with fallback to PipelineConfig defaults, instead of hardcoding `train_months=14`:

**(a)** In `_eval_single_month()` signature (line 36-37):
```python
def _eval_single_month(
    auction_month: str,
    class_type: str,
    ptype: str,
    hyperparam_config: HyperparamConfig,
    feature_config: FeatureConfig,
    threshold_beta: float = 0.7,
    train_months: int | None = None,
    val_months: int | None = None,
) -> dict | None:
```

Add fallback at start of function body:
```python
    _defaults = PipelineConfig()
    if train_months is None:
        train_months = _defaults.train_months
    if val_months is None:
        val_months = _defaults.val_months
```

**(b)** In `run_benchmark()` signature (line 98-99):
```python
def run_benchmark(
    ...
    train_months: int | None = None,
    val_months: int | None = None,
    ...
) -> dict:
```

Add fallback at start of function body (before the overrides block):
```python
    _defaults = PipelineConfig()
    if train_months is None:
        train_months = _defaults.train_months
    if val_months is None:
        val_months = _defaults.val_months
```

And update the `pc_dummy` creation in the overrides block to use the resolved values:
```python
    if overrides:
        from ml.pipeline import _apply_overrides
        pc_dummy = PipelineConfig(threshold_beta=threshold_beta, train_months=train_months, val_months=val_months)
        ...
```

### 5. Update tests

**File**: `ml/tests/test_config.py`
- Update feature count assertions: 14 → 17
- Update monotone constraint string to include the 3 new entries: `(1,1,1,1,1,-1,-1,-1,1,0,0,0,1,1,1,1,1)`
- Update feature name list to include interaction features
- `train_months` default stays at 14 (already updated in iter1)

**File**: `ml/tests/test_features.py`
- Update shape assertion: 14 → 17
- Ensure test data includes base columns needed for interaction computation (prob_exceed_110, prob_exceed_90, hist_da, prob_exceed_100, expected_overload, prob_exceed_105)

**File**: `ml/tests/conftest.py` (if applicable)
- Ensure synthetic fixture generates columns for all 17 features or at minimum the base columns needed for interaction computation

### 6. Keep all other settings unchanged

**IMPORTANT**: Do NOT change:
- Hyperparameters (keep v0 defaults: n_estimators=200, max_depth=4, etc.)
- `threshold_beta` (keep 0.7)
- `threshold_scaling_factor` (keep 1.0)
- `val_months` (keep 2)
- `gates.json` or `evaluate.py` (HUMAN-WRITE-ONLY)
- `registry/v0/` (immutable)

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
| S1-AUC | +0.001 to +0.003 | Window expansion provides base signal; interactions may add slightly |
| S1-AP | +0.001 to +0.004 | Both levers showed marginal AP improvement independently |
| S1-NDCG | +0.003 to +0.006 | Interactions had strongest NDCG effect (8W/4L); window adds (7W/4L) |
| S1-VCAP@100 | +0.002 to +0.006 | Both levers improved top-100 value capture |
| S1-BRIER | ±0.002 | Neutral — neither lever strongly affected calibration |
| Late-2022 months | Mixed — 2022-12 improves, 2022-09 likely still weak | Different failure modes |

**Success criteria**: NDCG and AUC both show ≥8/12 month wins AND mean improvement > iter1 on at least 2/4 Group A metrics.

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Non-additive effects (combined ≈ each individually) | MEDIUM | If so, confirms 14-feature AUC ceiling. Need fundamentally new features for iter3. |
| Interaction features hurt with 14-month window | LOW | Interactions were neutral on AUC with 10-month window. Unlikely to flip negative with more data. |
| Compute time increase (~40% from window + marginal from interactions) | MEDIUM | Monitor mem_mb() closely. 14-month window already validated in iter1. |
| Combined bottom-2 regression | LOW-MEDIUM | Both levers showed slight bottom-2 AP/NDCG regression. Combined may amplify. Monitor carefully. |

## What NOT to Change

- Hyperparameters (keep v0 defaults)
- `threshold_beta` (keep 0.7)
- `threshold_scaling_factor` (keep 1.0)
- `gates.json` or `evaluate.py` (HUMAN-WRITE-ONLY)
- `registry/v0/` (immutable)
- `val_months` (keep 2)
- `train_months` (keep 14 — already set in iter1)
