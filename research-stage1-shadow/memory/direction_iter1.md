# Direction — Iteration 1

## Batch
`smoke-v5-20260227-184427`

## Hypothesis
**Determinism check**: Running the pipeline with identical v0 hyperparameters and configuration will reproduce v0 metrics exactly (within noise tolerance of 0.02). This validates that the full pipeline infrastructure — version allocation, training, evaluation, gate comparison, and artifact registration — works correctly end-to-end.

## Specific Changes
**None.** This iteration uses the exact same configuration as v0:

### Pipeline Config (do NOT modify)
- `auction_month`: `2021-07`
- `class_type`: `onpeak`
- `period_type`: `f0`
- `train_months`: 10
- `val_months`: 2
- `threshold_beta`: 0.7
- `threshold_scaling_factor`: 1.0
- `scale_pos_weight_auto`: true

### Hyperparameters (do NOT modify — use defaults from `ml/config.py`)
- `n_estimators`: 200
- `max_depth`: 4
- `learning_rate`: 0.1
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `reg_alpha`: 0.1
- `reg_lambda`: 1.0
- `min_child_weight`: 10
- `random_state`: 42

### Features (do NOT modify — use defaults from `ml/config.py`)
All 14 features with monotone constraints as defined in `FeatureConfig.step1_features`.

## Worker Instructions

1. **Read `VERSION_ID` from state.json** (the PROJECT_DIR copy, not worktree):
   ```bash
   VERSION_ID=$(jq -r '.version_id' "${PROJECT_DIR}/state.json")
   ```

2. **Run the pipeline** with no overrides:
   ```bash
   cd /home/xyz/workspace/pmodel && source .venv/bin/activate
   cd "${WORKTREE_DIR}"
   SMOKE_TEST=true python -m ml.pipeline \
     --version-id "${VERSION_ID}" \
     --auction-month 2021-07 \
     --class-type onpeak \
     --period-type f0 \
     --registry-dir registry
   ```

3. **Run tests**:
   ```bash
   python -m pytest ml/tests/ -v
   ```

4. **Run comparison against v0**:
   ```bash
   python -m ml.compare --candidate "${VERSION_ID}" --baseline v0 --registry-dir registry
   ```

5. **Verify determinism**: Compare `registry/${VERSION_ID}/metrics.json` against `registry/v0/metrics.json`. Every metric should match within noise tolerance (0.02). If metrics diverge beyond noise tolerance, this is a bug — investigate.

6. **Commit and write handoff**: Follow the standard commit + handoff protocol from the runbook.

## Expected Impact

| Gate | v0 Value | Expected v0001 Value | Gate Floor | Status |
|------|----------|---------------------|------------|--------|
| S1-AUC | 0.75 | 0.75 (±0.02) | 0.65 | PASS |
| S1-AP | 0.5909 | 0.5909 (±0.02) | 0.12 | PASS |
| S1-VCAP@100 | 1.0 | 1.0 | 0.95 | PASS |
| S1-VCAP@500 | 1.0 | 1.0 | 0.95 | PASS |
| S1-VCAP@1000 | 1.0 | 1.0 | 0.95 | PASS |
| S1-NDCG | 0.5044 | 0.5044 (±0.02) | 0.4544 | PASS |
| S1-BRIER | 0.2021 | 0.2021 (±0.02) | 0.2221 | PASS |
| S1-REC | 0.0 | 0.0 | 0.4 | **FAIL** (expected) |
| S1-CAP@100 | 0.0 | 0.0 | -0.05 | PASS |
| S1-CAP@500 | 0.0 | 0.0 | -0.05 | PASS |

**S1-REC failing is expected** — the v0 model predicts no positives at threshold=0.82 on 20-sample SMOKE_TEST data (2 positives, binding_rate=0.1). This is a known baseline limitation, not a regression.

## Risk Assessment

1. **Non-determinism risk** (LOW): XGBoost with `random_state=42` and identical data should be deterministic. If metrics diverge, the cause is likely data loading order or floating-point accumulation differences — worth investigating but not a showstopper.

2. **Infrastructure risk** (MEDIUM): This is the first run of the full pipeline loop. Potential issues:
   - Version directory already exists (handled by pipeline's `FileExistsError` catch)
   - Handoff JSON paths not matching expectations
   - Worker worktree not having latest code

3. **Gate calibration note**: S1-REC floor of 0.4 is impossible to meet on SMOKE_TEST data with 2 positives and a high threshold. This is a known issue — gate calibration adjustments (if any) should happen at HUMAN_SYNC after iteration 3, not during this infrastructure validation batch.

## Success Criteria
- All metrics match v0 within ±0.02
- Pipeline completes all 6 phases without error
- Tests pass
- Comparison report generated successfully
- Handoff JSON written correctly
