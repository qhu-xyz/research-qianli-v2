# v0001 Changes Summary — Iteration 1

## Hypothesis
Determinism check: Running the pipeline with identical v0 hyperparameters reproduces v0 metrics exactly.

## Changes Made
None. This iteration uses the exact same configuration as v0 — no code modifications, no hyperparameter changes, no feature changes.

## Pipeline Execution
- Command: `SMOKE_TEST=true python -m ml.pipeline --version-id v0001 --auction-month 2021-07 --class-type onpeak --period-type f0 --registry-dir registry`
- All 6 phases completed successfully
- Peak memory: 212 MB

## Results

All metrics match v0 exactly (delta = 0.0 for every metric):

| Metric | v0 | v0001 | Delta |
|--------|-----|-------|-------|
| S1-AUC | 0.75 | 0.75 | 0.0 |
| S1-AP | 0.5909 | 0.5909 | 0.0 |
| S1-VCAP@100 | 1.0 | 1.0 | 0.0 |
| S1-VCAP@500 | 1.0 | 1.0 | 0.0 |
| S1-VCAP@1000 | 1.0 | 1.0 | 0.0 |
| S1-NDCG | 0.5044 | 0.5044 | 0.0 |
| S1-BRIER | 0.2021 | 0.2021 | 0.0 |
| S1-REC | 0.0 | 0.0 | 0.0 |
| S1-CAP@100 | 0.0 | 0.0 | 0.0 |
| S1-CAP@500 | 0.0 | 0.0 | 0.0 |
| threshold | 0.820287 | 0.820287 | 0.0 |

## Gate Status
- 9/10 gates PASS (same as v0)
- S1-REC FAIL (expected — v0 model predicts no positives at threshold=0.82 on 20-sample SMOKE_TEST data)

## Conclusion
Pipeline infrastructure is validated end-to-end. The full loop — version allocation, data loading, feature preparation, training, threshold optimization, evaluation, registration, and comparison — works correctly and produces deterministic results.
