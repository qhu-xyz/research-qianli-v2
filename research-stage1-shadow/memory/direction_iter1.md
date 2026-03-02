# Direction — Iteration 1 (feat-eng-20260302-154125)

## Hypothesis

**H5: Training window expansion (10→14 months) improves ranking quality by exposing the model to more diverse market regimes, directly addressing the late-2022 distribution shift.**

Revert to v0's 14 base features (remove 3 interaction features from v0002) to isolate the training window effect.

## Rationale

Two independent levers have failed to break the AUC ceiling at ~0.835:
- **v0003** (HP tuning): AUC -0.0025, 0W/11L — model is not complexity-limited
- **v0002** (interaction features): AUC +0.0000, 5W/6L/1T — interactions don't add new information

Both reviewers and the previous orchestrator converge on: **the dominant remaining problem is temporal non-stationarity (late-2022 distribution shift)**. The 10-month rolling training window may not capture enough regime diversity.

Evidence for distribution shift:
- Weakest months: 2022-09 (AP=0.315), 2022-12 (AUC=0.809), 2022-06 (REC=0.313)
- Interaction features helped early months (2020-09 to 2021-04) but not late months (2022+)
- Late-2022 weakness unchanged across HP tuning AND interaction features

Why remove interaction features:
- v0002 showed bottom-2 regressed on 3/4 Group A metrics (AP -0.0017, VCAP@100 -0.0008, NDCG -0.0013)
- Gains concentrated in distribution middle, not tails
- VCAP@500 -0.0043 and VCAP@1000 -0.0031 (hurt broader ranking)
- Clean single-variable comparison (v0 features + new window) gives the clearest signal

## Specific Changes

### 1. Fix benchmark.py train_months plumbing (BUG — blocking for this experiment)

**Problem**: `_eval_single_month` creates `PipelineConfig` with default `train_months=10`, ignoring any override. The `run_benchmark` overrides set `train_months` on `pc_dummy` but never pass it to `_eval_single_month`.

**File**: `ml/benchmark.py`

**Changes**:
- Add `train_months: int = 10` and `val_months: int = 2` parameters to `_eval_single_month` signature
- Pass them when constructing `PipelineConfig` inside `_eval_single_month`:
  ```python
  config = PipelineConfig(
      auction_month=auction_month,
      class_type=class_type,
      period_type=ptype,
      threshold_beta=threshold_beta,
      train_months=train_months,
      val_months=val_months,
  )
  ```
- Extract `train_months` and `val_months` from `pc_dummy` in `run_benchmark` after `_apply_overrides`, and pass them to `_eval_single_month`:
  ```python
  train_months = pc_dummy.train_months
  val_months = pc_dummy.val_months
  ```
- Fix the hardcoded `"train_months": 10` on line 178 — use the actual value:
  ```python
  "train_months": train_months,
  "val_months": val_months,
  ```

### 2. Remove interaction features — revert to v0's 14 base features

**File**: `ml/config.py`
- Remove the 3 interaction feature tuples from `FeatureConfig.step1_features`:
  - `("exceed_severity_ratio", 1)`
  - `("hist_physical_interaction", 1)`
  - `("overload_exceedance_product", 1)`
- Update docstring: features property should say "14 items" (already does, but verify)

**File**: `ml/features.py`
- Remove the `df.with_columns([...])` block that computes the 3 interaction features (lines 38-45)
- Fix docstrings: "14 feature columns" is correct after removal; "shape (n_samples, 14)" is correct

**File**: `ml/data_loader.py`
- Fix docstring on line 24: "14 feature columns" is correct after removal

### 3. Set train_months=14 in PipelineConfig default

**File**: `ml/config.py`
- Change `train_months: int = 10` to `train_months: int = 14` in `PipelineConfig`

### 4. Update tests

**File**: `ml/tests/test_config.py`
- Update assertion `assert pc.train_months == 10` → `assert pc.train_months == 14`
- Update any feature count assertions from 17 back to 14

**File**: `ml/tests/test_features.py` (if exists)
- Remove test data for interaction features
- Update expected feature count assertions from 17 to 14

### 5. Add schema guard for base feature columns (from Codex review, MEDIUM)

**File**: `ml/features.py`
- Before `df.select(cols)`, add validation that all base columns exist in the DataFrame:
  ```python
  missing = [c for c in cols if c not in df.columns]
  if missing:
      raise ValueError(f"Missing feature columns in DataFrame: {missing}")
  ```

## Run Configuration

The worker should run benchmark with:
```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-stage1-shadow

python -m ml.benchmark --version-id ${VERSION_ID} --ptype f0 --class-type onpeak
```

No `--overrides` needed since `train_months=14` is set as the new default.

## Expected Impact

| Metric | Direction | Rationale |
|--------|-----------|-----------|
| S1-AUC | +0.002–0.008 | More diverse training data → better generalization, especially late-2022 |
| S1-AP | +0.005–0.015 | Ranking quality should improve with broader regime coverage |
| S1-NDCG | +0.002–0.010 | Better ordering of borderline cases |
| S1-VCAP@100 | ~neutral | Top-100 captures extreme cases that may not benefit from more training data |
| S1-BRIER | ~neutral | Calibration effects are unclear; monitor closely |

**Primary success metric**: AUC improvement in late-2022 months (2022-09, 2022-12).
**Secondary**: AUC win count ≥ 8/12 months.

## Risk Assessment

1. **Stale pattern inclusion** (MEDIUM): Older months in the training window may contain patterns that no longer hold. The 14-month window includes data from ~1 year before the auction month. If market dynamics changed, this could hurt rather than help. Mitigation: compare early-months vs late-months performance to detect this.

2. **Data loading time** (LOW): Loading 4 additional months of training data per eval month increases runtime. At ~270K rows/month, this adds ~1.1M rows per eval month (~40% more data). Training time scales roughly linearly with data size. Total benchmark time may increase from ~45min to ~60min. Within worker timeout.

3. **Overfitting risk** (LOW): More training data generally reduces overfitting, not increases it. This is the safest direction available.

4. **train_months plumbing fix could introduce bugs** (LOW): The benchmark.py fix is mechanical — adding parameter passthrough. Test by verifying that v0 baseline results are reproduced when train_months=10 is explicitly passed.

## Validation Checklist

- [ ] All existing tests pass after changes
- [ ] Benchmark with train_months=10 (explicit override) reproduces v0 metrics (±0.001) — confirms the plumbing fix doesn't break anything
- [ ] Benchmark with train_months=14 (new default) completes successfully for all 12 months
- [ ] No month produces empty training or validation set with the expanded window
- [ ] eval_config in metrics.json correctly shows train_months=14
