# Direction — Iteration 2 (hp-tune-20260302-144146)

## Hypothesis

**H5: Longer training window addresses distribution shift and improves tail stability.**

Two iterations have confirmed the AUC ceiling at ~0.835 with the current feature set:
- v0003 (HP tuning): AUC 0W/11L — model complexity is not the bottleneck
- v0002 (interaction features): AUC 5W/6L/1T — pre-computed interactions add no new information

The persistent late-2022 weakness (2022-09 AP=0.314, 2022-12 AUC=0.809) and the pattern that v0002 interaction features helped early months (2020–2021H1) but not late months (2022) points to **distribution shift** as the dominant remaining problem. The 10-month rolling training window means the model for late-2022 evaluation months trains on only ~mid-2021 to late-2022 data — a period of rapid change. Expanding to 14 months provides:

1. **More diverse training examples** — capturing a wider range of grid conditions
2. **Better representation of transitions** — the model sees how the distribution evolved
3. **Potential tail stabilization** — worst months may improve if the training set covers analogous historical patterns

## Specific Changes

### 1. Expand training window: train_months 10 → 14

**File**: `ml/config.py`, class `PipelineConfig`

```python
train_months: int = 14  # was 10
```

This is the ONLY config change. Keep val_months=2 (validation window unchanged).

**Impact on data loading**: In `data_loader.py`, `lookback = config.train_months + config.val_months + horizon`. For f0: lookback goes from 10+2+0=12 to 14+2+0=16 months. Each evaluation month will load 16 months of history instead of 12. This means ~33% more training rows per month (~360K instead of ~270K).

### 2. Update benchmark.py hardcoded train_months

**File**: `ml/benchmark.py`, line ~178

The output dict hardcodes `"train_months": 10`. Update to read from config:

```python
"train_months": 14,  # was hardcoded 10
```

Or better, wire it from the PipelineConfig. But since _eval_single_month creates its own PipelineConfig, the simplest fix is to just change the hardcoded value to match. Alternatively, update _eval_single_month to accept train_months parameter and propagate it.

**Recommended approach**: Change the default in `PipelineConfig` (config.py) and update the hardcoded `10` in benchmark.py line 178 to `14`. This ensures both pipeline.py and benchmark.py use the new window.

### 3. Keep all 3 interaction features (NO CHANGE)

The 3 interaction features from v0002 remain:
- `exceed_severity_ratio`: prob_exceed_110 / (prob_exceed_90 + 1e-6)
- `hist_physical_interaction`: hist_da × prob_exceed_100
- `overload_exceedance_product`: expected_overload × prob_exceed_105

Rationale: Marginally positive for ranking (NDCG 8W/4L), computationally cheap, and may interact positively with longer training windows. Keep unless results show they actively harm.

### 4. Keep HP defaults (NO CHANGE)

All hyperparameters remain at v0 defaults:
- n_estimators=200, max_depth=4, lr=0.1, min_child_weight=10
- subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0

### 5. Fix stale docstrings (LOW priority, do if time permits)

Both reviewers noted docstrings still reference "14 features":
- `features.py:24` — "containing at least the 14 feature columns" → 14 base
- `features.py:31` — "Feature matrix of shape (n_samples, 14)" → 17
- `config.py:46` — "Return list of feature names (14 items)" → 17

### 6. Add schema guard for interaction features (MEDIUM priority, do if time permits)

Codex recommended validating that required base columns exist before computing interactions:

```python
# In prepare_features, before with_columns:
required_base = ["prob_exceed_110", "prob_exceed_90", "hist_da",
                  "prob_exceed_100", "expected_overload", "prob_exceed_105"]
missing = [c for c in required_base if c not in df.columns]
if missing:
    raise ValueError(f"Missing base columns for interaction features: {missing}")
```

### 7. Update tests

- `test_config.py`: Update `train_months` assertion from 10 → 14
- Any other tests that hardcode train_months=10

### 8. Run benchmark

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

## Expected Impact

| Metric | v0 Mean | Expected Direction | Rationale |
|--------|---------|-------------------|-----------|
| S1-AUC | 0.8348 | +0.002 to +0.010 | More diverse training → better generalization |
| S1-AP | 0.3936 | +0.005 to +0.015 | Better coverage of temporal patterns |
| S1-NDCG | 0.7333 | +0.003 to +0.010 | Improved ranking stability |
| S1-BRIER | 0.1503 | ±0.003 | More data could improve or slightly worsen calibration |
| Bottom-2 AUC | 0.8105 | +0.005 to +0.015 | Primary target: stabilize worst months |

**Win/loss target**: AUC improvement in ≥7/12 months. Bottom-2 AUC improvement (>0.8105).

**Primary success criterion**: Bottom-2 metrics improve (tail stabilization), not just mean.

## Risk Assessment

### Low Risk
- **Simple change** — only 1 config value (train_months) and 1 hardcoded constant
- **No feature or HP changes** — isolates the training window effect
- **More data generally helps** — standard ML principle, unlikely to harm
- **Additive** — all features and monotone constraints preserved

### Medium Risk
- **Memory increase**: ~33% more training rows per month. At ~270K rows/month × 14 months ≈ 3.8M rows of training data. With 17 features, this is ~260 MB of float64 data. Should be within the 40 GiB script budget, but monitor `mem_mb()` closely.
- **Training time**: ~33% more data means ~33% longer XGBoost training per month (tree construction is roughly linear in data size). 12 months × ~33% overhead. May approach the 1200s timeout.
- **Diminishing returns for early months**: Evaluation months like 2020-09 would train on data starting from ~2019-05. If very early data is stale or reflects different grid conditions, it could add noise. However, XGBoost's subsample=0.8 provides regularization against this.

### Not a Risk
- **Gate failures**: v0 passes with ~0.05 headroom. Even a small regression from stale training data would not breach floors.
- **Interaction feature conflicts**: The 3 interaction features are additive and computed at feature-prep time — unaffected by training window changes.

## Success Criteria

1. **Primary**: Bottom-2 AUC > 0.8105 (tail improvement over v0)
2. **Primary**: AUC mean > 0.8348 (any mean improvement)
3. **Consistency**: AUC improved in ≥7/12 months
4. **No regression**: All Group A gates pass all 3 layers
5. **Tail focus**: Late-2022 months (2022-09, 2022-12) should be the primary beneficiaries — check these specifically
