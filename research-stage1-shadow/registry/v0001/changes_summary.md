# v0001 — Infrastructure Validation (Iteration 1)

**Batch:** smoke-v4-20260227-183323
**Date:** 2026-02-27

## Hypothesis

Running the pipeline with identical default hyperparameters to v0 confirms pipeline determinism and end-to-end infrastructure correctness.

## Changes Made

**No ML code changes.** This iteration is purely infrastructure validation:

1. Ran the full pipeline (`ml/pipeline.py`) with default config for v0001
2. Ran comparison (`ml/compare.py`) against v0 baseline
3. Verified all 66 tests pass

## Results

| Gate | v0 | v0001 | Delta | Status |
|------|-----|-------|-------|--------|
| S1-AUC | 0.7500 | 0.7500 | 0.0 | **PASS** (floor 0.65) |
| S1-AP | 0.5909 | 0.5909 | 0.0 | **PASS** (floor 0.12) |
| S1-VCAP@100 | 1.0000 | 1.0000 | 0.0 | **PASS** (floor 0.95) |
| S1-VCAP@500 | 1.0000 | 1.0000 | 0.0 | **PASS** (floor 0.95) |
| S1-VCAP@1000 | 1.0000 | 1.0000 | 0.0 | **PASS** (floor 0.95) |
| S1-NDCG | 0.5044 | 0.5044 | 0.0 | **PASS** (floor 0.4544) |
| S1-BRIER | 0.2021 | 0.2021 | 0.0 | **PASS** (floor 0.2221) |
| S1-REC | 0.0000 | 0.0000 | 0.0 | **FAIL** (floor 0.4, Group B) |
| S1-CAP@100 | 0.0000 | 0.0000 | 0.0 | **PASS** (floor -0.05) |
| S1-CAP@500 | 0.0000 | 0.0000 | 0.0 | **PASS** (floor -0.05) |

**All Group A gates pass.** S1-REC fails (Group B) — expected. The model's optimal threshold (0.82) produces zero positive predictions on SMOKE_TEST data (20 samples, 2 positive, binding_rate=0.1). This is a data limitation, not a model bug.

## Conclusion

Pipeline is deterministic: v0001 metrics are **identical** to v0 (zero delta on all metrics). The full pipeline (load → features → train → threshold → evaluate → register) completes successfully. Infrastructure is validated.
