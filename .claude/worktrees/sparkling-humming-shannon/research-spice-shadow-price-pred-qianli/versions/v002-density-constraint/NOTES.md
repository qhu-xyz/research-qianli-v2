# v002-xgb-20260220-002 — F2.0 + density_skewness constraint=0

## Hypothesis

The `density_skewness` feature has a negative empirical relationship with binding probability (higher skewness → lower binding), but we constrain it to monotonicity=+1 (positive). This mismatch forces the classifier to partially ignore the feature. Removing the constraint (setting monotonicity=0) should let XGBoost use skewness information freely, improving AUC.

## Changes

| Parameter | Baseline (v000) | v001 | This version |
|-----------|-----------------|------|--------------|
| `ThresholdConfig.threshold_beta` | 0.5 | 2.0 | **2.0** (from v001) |
| `density_skewness` monotonicity (step1) | +1 | +1 | **0** (no constraint) |
| `density_skewness` monotonicity (step2) | +1 | +1 | **0** (no constraint) |

Cumulative: carries v001's F2.0 threshold + removes density_skewness monotonicity constraint.

## Results

| Gate | Onpeak | Offpeak | Mean | Floor | vs v000 | vs v001 |
|------|-------:|--------:|-----:|------:|--------:|--------:|
| S1-AUC | 0.6588 | 0.6666 | 0.6627 | 0.80 | -3.3 pp | -0.1 pp |
| S1-REC | 0.3275 | 0.3309 | 0.3292 | 0.30 | +5.6 pp | -1.0 pp |
| S2-SPR | 0.3761 | 0.4464 | 0.4113 | 0.30 | -0.1 pp | +0.9 pp |
| C-VC@1000 | 0.7848 | 0.8112 | 0.7980 | 0.50 | -5.3 pp | +0.3 pp |
| C-RMSE | $1,216 | $1,274 | $1,245 | $2,000 | **-$311** | -$51 |

Promotable: **No** — S1-AUC fails floor; C-VC@1000 regresses vs champion.

## Conclusion

The density_skewness constraint change was **essentially neutral** — all gates within 1 pp of v001. The feature either has low importance or XGBoost already learned to work around the constraint. This confirms that the AUC gap (0.66 vs 0.80) is not caused by monotonicity mismatch on this feature.

**Lesson**: The AUC problem is structural, not caused by a single feature constraint. Need to look at broader changes: more features, ensemble diversity, or hyperparameter tuning.

## Run Notes

- Original run (2026-02-20) stopped at 50% (PY20 only). Completed PY21 on 2026-02-22 using concurrent runner (`--concurrency 4`, ~20 min wall time for 16 remaining periods).
