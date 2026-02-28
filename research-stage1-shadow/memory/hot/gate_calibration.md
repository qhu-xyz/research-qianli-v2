# Gate Calibration — After Iteration 1

## Current State (SMOKE_TEST, n=20, n_positive=2)

| Gate | v0 Value | Floor | Headroom | Risk | Notes |
|------|----------|-------|----------|------|-------|
| S1-AUC | 0.7500 | 0.65 | +0.10 | Low | Reasonable |
| S1-AP | 0.5909 | 0.12 | +0.47 | Low | Floor very generous — consider raising after real data |
| S1-VCAP@100 | 1.0000 | 0.95 | +0.05 | N/A | Saturated at n=20, untested |
| S1-VCAP@500 | 1.0000 | 0.95 | +0.05 | N/A | Saturated (K >> n) |
| S1-VCAP@1000 | 1.0000 | 0.95 | +0.05 | N/A | Saturated (K >> n) |
| S1-NDCG | 0.5044 | 0.4544 | +0.05 | Medium | Exactly v0_offset, no margin |
| S1-BRIER | 0.2021 | 0.2221 | +0.02 | **HIGH** | Tightest gate — any threshold change risks flipping |
| S1-REC (B) | 0.0000 | 0.40 | -0.40 | **FAIL** | Primary iter2 target |
| S1-CAP@100 (B) | 0.0000 | -0.05 | +0.05 | Low | Trivially passes (negative floor) |
| S1-CAP@500 (B) | 0.0000 | -0.05 | +0.05 | Low | Trivially passes (negative floor) |

## Key Observations

1. **S1-BRIER is the binding constraint** for any threshold-lowering strategy. Lowering threshold will cause the model to predict more positives, which shifts Brier score. With only 0.02 headroom, this gate could easily flip.

2. **S1-REC requires the model to predict positives**. Current threshold 0.82 pushes pred_binding_rate to 0.0. Lowering threshold_beta from 0.7 to ~0.3 should lower the F-beta optimal threshold substantially, allowing some positive predictions.

3. **VCAP@K gates are meaningless at n=20**. These will only matter on real data where K << n_samples.

4. **noise_tolerance=0.02 is coarse at n=20** (Codex). One AUC pairwise swap = ~0.028. Don't read too much into small deltas.

## Recommendations
- **No floor changes yet** — insufficient data for recalibration
- **Monitor S1-BRIER closely in iter2** — any threshold change is a risk
- **Recalibrate all floors after first real-data run** with uncertainty-aware thresholds

## After Iteration 1 (smoke-v7) — No Changes

Both reviewers again agree: no gate floor changes justified. Iteration produced zero delta (hypothesis failed — beta direction was inverted). When beta > 1 is applied in iter2, S1-BRIER is almost certain to shift because producing positive predictions changes calibration. If S1-BRIER flips (value exceeds 0.2221 floor), assess whether the floor is too tight at n=20 and recommend recalibration at HUMAN_SYNC.

Additional Codex observation: `S1-VCAP@K` and `S1-CAP@K` should become informational when K >= n_samples, as they're mathematically saturated.
