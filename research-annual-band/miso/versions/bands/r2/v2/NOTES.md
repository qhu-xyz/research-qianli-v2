# R2 Band Calibration v2 — Per-Class Stratified Bands

## Method

Same as R2 v1 (6 quantile bins by |mtm_1st_mean|), but with separate widths per class (onpeak, offpeak) within each bin.

## Class Parity Improvement

| Quarter | Pooled gap | Per-class gap |
|---------|----------:|-------------:|
| aq1 | 0.43pp | 0.07pp |
| aq2 | 0.04pp | 0.09pp |
| aq3 | 0.58pp | 0.00pp |
| aq4 | 0.15pp | 0.07pp |

Biggest improvement in aq1 (6x) and aq3 (perfect parity).

## Coverage Accuracy (LOO, P95)

| Quarter | Target | Actual | Error |
|---------|-------:|-------:|------:|
| aq1 | 95.0% | 94.63% | -0.37pp |
| aq2 | 95.0% | 94.71% | -0.29pp |
| aq3 | 95.0% | 94.60% | -0.40pp |
| aq4 | 95.0% | 94.64% | -0.36pp |

## Width Comparison vs v1

Width-neutral (±0.2%). Per-class stratification doesn't inflate widths because the pooled quantile was already between the two class quantiles.

## Gate Results

All HARD gates (BG1-BG3) PASS. BG5 SOFT fail (PY 2022, same as v1). BG7 class parity now passes.

## Decision

Promote as v2. Class parity gap reduced from 0.04-0.58pp to 0.00-0.09pp. Width-neutral. BG7 passes.
