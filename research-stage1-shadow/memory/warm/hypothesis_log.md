# Hypothesis Log

> **NOTE**: H1–H2 below were tested during SMOKE_TEST runs (n=20).
> H2 status is FAILED (beta direction was inverted). See experiment_log.md for details.

## H1 (Iteration 1): Infrastructure determinism — CONFIRMED
**Hypothesis**: Running pipeline with identical v0 config produces bit-for-bit identical metrics.
**Result**: Confirmed — all 10 metrics exactly match v0 (zero delta).
**Implication**: Pipeline is deterministic. Any future metric changes can be attributed to code/config changes.

## H2 (Iteration 2): Threshold-beta reduction fixes S1-REC
**Hypothesis**: Lowering threshold_beta from 0.7 to 0.3 will lower the optimal threshold enough to produce positive predictions, moving S1-REC from 0.0 toward the 0.4 floor.
**Risk**: S1-BRIER has only 0.02 headroom. More positive predictions could degrade calibration and flip BRIER.
**Status**: Pending — iteration 2
