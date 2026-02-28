# Direction — Iteration 1

**Batch**: smoke-v6-20260227-190225
**Iteration**: 1 of 3
**Objective**: Infrastructure validation — determinism check

## Hypothesis

Running the pipeline with identical v0 configuration (no ML changes) will produce metrics that match v0 within noise tolerance (0.02). This confirms the pipeline is deterministic and all infrastructure components (version allocation, training, evaluation, gate comparison) work correctly.

## Specific Changes

**No ML code changes.** The worker should:

1. **Allocate a new version** via the pipeline's version allocation mechanism (will be assigned `v0001` or similar)
2. **Run the training pipeline** with default hyperparameters — these must match v0 exactly:
   - Use SMOKE_TEST mode (`SMOKE_TEST=true` or equivalent env var)
   - Same auction month (2021-07), class (onpeak), period (f0)
   - Same model hyperparameters as v0 baseline
   - Same train/test split
3. **Run evaluation** using `ml/evaluate.py` against the test set
4. **Run gate comparison** against v0 baseline metrics
5. **Run tests**: `python -m pytest ml/tests/ -v`

## Expected Impact

All metrics should match v0 within noise tolerance (±0.02):

| Gate | v0 Value | Floor | Expected v0001 | Gate Status |
|------|----------|-------|-----------------|-------------|
| S1-AUC | 0.75 | 0.65 | ~0.75 | PASS |
| S1-AP | 0.5909 | 0.12 | ~0.5909 | PASS |
| S1-VCAP@100 | 1.0 | 0.95 | ~1.0 | PASS |
| S1-VCAP@500 | 1.0 | 0.95 | ~1.0 | PASS |
| S1-VCAP@1000 | 1.0 | 0.95 | ~1.0 | PASS |
| S1-NDCG | 0.5044 | 0.4544 | ~0.5044 | PASS |
| S1-BRIER | 0.2021 | 0.2221 | ~0.2021 | PASS |
| S1-REC | 0.0 | 0.4 | ~0.0 | FAIL (Group B, expected) |
| S1-CAP@100 | 0.0 | -0.05 | ~0.0 | PASS |
| S1-CAP@500 | 0.0 | -0.05 | ~0.0 | PASS |

**Key**: Group A gates should all pass. S1-REC (Group B) is expected to fail because the v0 model predicts no positives at threshold 0.82 (pred_binding_rate=0.0), so recall is 0.0.

## Success Criteria

1. All metrics match v0 within ±0.02 (noise_tolerance from gates.json)
2. All tests pass (`pytest ml/tests/ -v`)
3. Gate comparison runs without errors
4. Version directory `registry/v0001/` is created with: `metrics.json`, `comparison.json`, model artifacts

## Risk Assessment

- **Low risk**: This is a determinism check with no code changes
- **Possible issue**: Non-determinism in LightGBM training could cause small metric drift — if metrics differ by >0.02, investigate random seed handling
- **Possible issue**: SMOKE_TEST data generation might not be fully deterministic if seed is not fixed — check that synthetic data is identical across runs
- **Infrastructure risk**: If version allocation, evaluation, or gate comparison fails, this indicates a pipeline bug that must be fixed before proceeding to iterations 2-3

## Notes for Reviewers

This iteration is pure infrastructure validation. Reviewers should focus on:
- Is the pipeline infrastructure correct and complete?
- Are metrics computed correctly?
- Is the gate comparison logic sound?
- Are there any bugs in the training/evaluation code?
- Is the version registry structure clean and traceable?

Do NOT focus on ML improvements this iteration — that comes in iterations 2-3.
