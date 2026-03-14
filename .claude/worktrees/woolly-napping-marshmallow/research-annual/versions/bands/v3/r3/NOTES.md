# R3 Band Calibration v2 — Per-Class Stratified Bands

## Method

Same as R3 v1 (6 quantile bins by |mtm_1st_mean|), but with separate widths per class (onpeak, offpeak) within each bin.

## Class Parity Improvement

| Quarter | Pooled gap | Per-class gap |
|---------|----------:|-------------:|
| aq1 | 1.02pp | 0.14pp |
| aq2 | 0.26pp | 0.08pp |
| aq3 | 0.08pp | 0.06pp |
| aq4 | 0.13pp | 0.01pp |

Biggest improvement in aq1 (7x better). R3 aq1 had the worst pooled gap (1.02pp) because R3 summer onpeak has 20% higher MAE than offpeak.

## Coverage Accuracy (LOO, P95)

| Quarter | Target | Actual | Error |
|---------|-------:|-------:|------:|
| aq1 | 95.0% | 94.57% | -0.43pp |
| aq2 | 95.0% | 94.79% | -0.21pp |
| aq3 | 95.0% | 94.54% | -0.46pp |
| aq4 | 95.0% | 94.66% | -0.34pp |

## Width Comparison vs v1

Width-neutral (±0.4%). Class stratification doesn't inflate widths.

## Gate Results

All HARD gates (BG1-BG3) PASS. BG5 SOFT fail (PY 2022, same as v1). BG7 class parity now passes.

## Decision

Promote as v2. Class parity gap reduced from 0.08-1.02pp to 0.01-0.14pp. Width-neutral. BG7 passes.
