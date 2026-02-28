# Changes Summary — v0001 (Iteration 1)

**Batch**: smoke-v6-20260227-190225
**Iteration**: 1 of 3
**Hypothesis**: Infrastructure validation — determinism check

## Changes Made

**No ML code changes.** This iteration validates that the pipeline infrastructure is deterministic: running with identical v0 configuration produces identical metrics.

## Pipeline Execution

- Ran `SMOKE_TEST=true python ml/pipeline.py --version-id v0001 --auction-month 2021-07 --class-type onpeak --period-type f0`
- All 66 tests passed (`python -m pytest ml/tests/ -v`)
- Gate comparison executed against v0 baseline

## Results

All v0001 metrics are **identical** to v0 (zero drift):

| Metric | v0 | v0001 | Delta | Gate |
|--------|-----|-------|-------|------|
| S1-AUC | 0.75 | 0.75 | 0.0 | PASS |
| S1-AP | 0.5909 | 0.5909 | 0.0 | PASS |
| S1-VCAP@100 | 1.0 | 1.0 | 0.0 | PASS |
| S1-VCAP@500 | 1.0 | 1.0 | 0.0 | PASS |
| S1-VCAP@1000 | 1.0 | 1.0 | 0.0 | PASS |
| S1-NDCG | 0.5044 | 0.5044 | 0.0 | PASS |
| S1-BRIER | 0.2021 | 0.2021 | 0.0 | PASS |
| S1-REC | 0.0 | 0.0 | 0.0 | FAIL (Group B, expected) |
| S1-CAP@100 | 0.0 | 0.0 | 0.0 | PASS |
| S1-CAP@500 | 0.0 | 0.0 | 0.0 | PASS |

## Conclusion

Infrastructure validation **PASSED**. The pipeline is fully deterministic:
- Synthetic data generation (seed=42) is reproducible
- XGBoost training (random_state=42) produces identical models
- Threshold optimization, evaluation, and gate comparison all function correctly
- Version registry structure is clean and complete

S1-REC fails its Group B floor (0.4) because the v0 model predicts no positives at threshold 0.82 (pred_binding_rate=0.0, recall=0.0). This is expected and documented in the direction file.
