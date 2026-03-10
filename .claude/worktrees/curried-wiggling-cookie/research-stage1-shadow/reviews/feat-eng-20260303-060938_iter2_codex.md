## Review: feat-eng-20260303-060938 Iteration 2

### Summary
v0005 (train window 18 months + feature-importance export) is technically stable and passes all Group A/Group B gates on mean and tail-safety. Relative to v0, ranking quality is mixed: AUC +0.0013, VCAP@100 +0.0044, NDCG +0.0032, but AP regresses -0.0007. Relative to iter1/v0004, all Group A means are flat-to-down (AUC -0.0002, AP -0.0022, VCAP@100 -0.0012, NDCG -0.0006), supporting diminishing returns for 14->18 month expansion.

Across 12 months, gains are not fully uniform. AUC is 7W/5L and AP is 6W/6L; 2022-09 remains the weakest AP month (0.2986), and 2022 regime weakness persists (2022 means: AUC 0.8302, AP 0.3552, NDCG 0.7131 vs 2021 means: 0.8390, 0.4116, 0.7488). Three-layer tails are safe by current thresholds, but note that Layer-3 non-regression is not actively enforced in code when champion is null.

### Gate Analysis
(Champion in this table uses v0 reference means because `registry/champion.json` is null.)

| Gate | Value | Floor | Champion | Pass | Delta |
|------|-------|-------|----------|------|-------|
| S1-AUC | 0.8361 | 0.7848 | 0.8348 | YES | +0.0013 |
| S1-AP | 0.3929 | 0.3436 | 0.3936 | YES | -0.0007 |
| S1-VCAP@100 | 0.0193 | -0.0351 | 0.0149 | YES | +0.0044 |
| S1-NDCG | 0.7365 | 0.6833 | 0.7333 | YES | +0.0032 |
| S1-BRIER | 0.1525 | 0.1703 | 0.1503 | YES | +0.0022 (worse; lower is better) |
| S1-VCAP@500 | 0.0845 | 0.0408 | 0.0908 | YES | -0.0063 |
| S1-VCAP@1000 | 0.1481 | 0.1091 | 0.1591 | YES | -0.0110 |
| S1-REC | 0.4182 | 0.1000 | 0.4192 | YES | -0.0010 |
| S1-CAP@100 | 0.7825 | 0.7325 | 0.7825 | YES | +0.0000 |
| S1-CAP@500 | 0.7723 | 0.7240 | 0.7740 | YES | -0.0017 |

Three-layer detail (v0005):
- Layer 1 (mean): all gates pass. Closest mean margin is S1-BRIER (+0.0178 to ceiling) and, in Group A, S1-AP (+0.0493 over floor).
- Layer 2 (tail safety): 0 months beyond tail_floor for every gate. Weakest Group A months: AUC=2021-12 (0.8133), AP=2022-09 (0.2986), VCAP@100=2022-12 (0.0014), NDCG=2021-04 (0.6664).
- Layer 3 (tail non-regression vs v0 reference with tolerance 0.02): Group A bottom_2 deltas are AUC +0.0051, AP -0.0075, VCAP@100 +0.0010, NDCG -0.0017; all pass. Closest to fail is AP (margin +0.0125).

### Code Findings
1. [HIGH] Threshold leakage remains: threshold optimization and evaluation are performed on the same validation fold, which inflates threshold-dependent metrics (BRIER/REC/CAP/precision). Evidence: `find_optimal_threshold(...)` then immediate `evaluate_classifier(...)` on same `val_df` in `ml/benchmark.py:75-82` and `ml/pipeline.py:104-125`.
2. [MEDIUM] Threshold boundary mismatch remains: PR-curve thresholds are inclusive at boundary, but inference uses strict `>` in `apply_threshold`, which can drop boundary positives. Evidence: threshold generation in `ml/threshold.py:33-37` vs application in `ml/threshold.py:49-64`.
3. [MEDIUM] Layer-3 gate is effectively disabled when champion is null. In this run, `champion.json` is null, so tail non-regression defaults to pass. Evidence: `registry/champion.json:1`, `ml/compare.py:219-232`, and champion loading logic `ml/compare.py:509-517`.
4. [LOW] New feature-importance output has no direct test coverage for file existence/schema and for ensuring `_feature_importance` never leaks into `metrics.json`. Current benchmark tests only check metrics/config/meta. Evidence: `ml/benchmark.py:191-249` vs tests in `ml/tests/test_benchmark.py:16-128`.

### Recommendations
1. Do not promote v0005; keep v0004 as working baseline. The 18-month expansion did not improve Group A means vs iter1 and materially hurt AP.
2. Keep focus on precision-aligned ranking metrics; prioritize AP stability in weak months (especially 2022-09) rather than further window expansion.
3. Add a small test set split (or rolling holdout) for threshold tuning to remove leakage from threshold-dependent metrics while keeping Group A ranking evaluation unchanged.
4. Add tests for `feature_importance.json` generation and schema, plus assertion that `metrics.json` per-month values remain numeric-only.
5. Treat `ptype` parsing fallback behavior as explicit (raise on malformed values) to avoid silent horizon assumptions.

### Gate Calibration
- No immediate floor changes recommended for Group A mean/tail floors; margins are still wide.
- `noise_tolerance=0.02` appears loose relative to observed bottom-2 shifts (Group A worst observed here: AP -0.0075 vs v0). After a real champion is set, consider metric-specific Layer-3 tolerances, e.g. AUC/AP/NDCG around 0.01, while keeping VCAP@100 at or above current tolerance until more tail history accumulates.
- Continue monitoring S1-BRIER as the closest mean gate to boundary (0.0178 headroom) and VCAP@500 tail behavior (improved vs v0004 but still below v0).
