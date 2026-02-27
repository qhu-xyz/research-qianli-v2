# Direction — Iteration 1

## Batch
- **Batch ID**: infra-v3-20260227-182424
- **Iteration**: 1 of 3
- **Mode**: SMOKE_TEST (infrastructure validation)

## Hypothesis

**Default hyperparameters reproduce v0 metrics within noise tolerance (0.02).**

This is a determinism check — not an ML improvement attempt. The goal is to verify that the full pipeline (version allocation, training, evaluation, gate comparison, registry write) works correctly end-to-end.

## Specific Changes

**None.** The worker should run the pipeline with identical configuration to v0:

1. **Do NOT modify any ML code** — use default `HyperparamConfig()` and `FeatureConfig()` (which match v0's config.json)
2. Run the pipeline via: `python -m ml.pipeline --version-id ${VERSION_ID} --auction-month 2021-07 --class-type onpeak --period-type f0`
3. Run tests: `python -m pytest ml/tests/ -v`
4. Run comparison: `python -m ml.compare --candidate ${VERSION_ID} --baseline v0`
5. Confirm all Group A gate statuses match expectations

## Expected Impact

| Gate | v0 Value | Expected v1 | Floor | Expected Status |
|------|----------|-------------|-------|-----------------|
| S1-AUC | 0.75 | 0.75 | 0.65 | PASS |
| S1-AP | 0.5909 | 0.5909 | 0.12 | PASS |
| S1-VCAP@100 | 1.0 | 1.0 | 0.95 | PASS |
| S1-VCAP@500 | 1.0 | 1.0 | 0.95 | PASS |
| S1-VCAP@1000 | 1.0 | 1.0 | 0.95 | PASS |
| S1-NDCG | 0.5044 | 0.5044 | 0.4544 | PASS |
| S1-BRIER | 0.2021 | 0.2021 | 0.2221 | PASS |
| S1-REC | 0.0 | 0.0 | 0.4 | FAIL (expected) |
| S1-CAP@100 | 0.0 | 0.0 | -0.05 | PASS |
| S1-CAP@500 | 0.0 | 0.0 | -0.05 | PASS |

- All metrics should be **identical** to v0 (same config, same data, same random_state=42)
- S1-REC failure is expected — the model predicts no positives at threshold 0.82 on 20-sample synthetic data
- If any metric deviates by more than noise_tolerance (0.02), that signals a pipeline bug

## Risk Assessment

- **Low risk**: No code changes, so failure would indicate a pipeline bug (useful signal)
- **Possible failure mode**: Non-determinism in data loading order or feature computation could cause small metric drift — if observed, document the source
- **Infrastructure concerns to validate**:
  - Version directory created correctly at `registry/${VERSION_ID}/`
  - `config.json`, `metrics.json`, `meta.json` all written with correct schema
  - Model file saved at `registry/${VERSION_ID}/model/classifier.ubj`
  - Comparison report generated correctly
  - Tests pass without modification

## Success Criteria

1. All tests pass (`pytest ml/tests/ -v`)
2. Metrics match v0 within noise tolerance
3. Comparison report shows correct gate pass/fail status
4. Version registered in `registry/${VERSION_ID}/` with all artifacts
5. Handoff JSON written with status "done"

## Notes for Reviewers

This iteration is about **infrastructure correctness**, not ML quality. Review focus should be:
- Is the pipeline code correct?
- Are metrics computed consistently?
- Is the registry/versioning system working?
- Are there any code quality issues in `ml/` that should be addressed in iteration 2?
- Is the gate calibration reasonable for the SMOKE_TEST data characteristics?
