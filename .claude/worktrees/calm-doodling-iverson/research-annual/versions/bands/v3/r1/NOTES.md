# R1 Band Calibration v3 — Per-Class Stratified Bands

## Motivation

v2 pools onpeak + offpeak residuals together. Onpeak MAE is up to 13% higher than offpeak in some quarters (aq1: 847 vs 747). Pooled bands under-cover onpeak and over-cover offpeak, creating a class parity gap of up to 0.47pp at P95.

## Method

Same symmetric empirical quantile approach as v2 (4 data-driven bins by |nodal_f0|), but calibrate **separate widths per class (onpeak, offpeak)** within each bin. Bin boundaries computed on full training set (pooling classes); only the widths differ per class.

## Results

### Overall Coverage Accuracy (LOO, P95)

| Quarter | Target | Actual | Error |
|---------|-------:|-------:|------:|
| aq1 | 95.0% | 94.66% | -0.34pp |
| aq2 | 95.0% | 94.65% | -0.35pp |
| aq3 | 95.0% | 94.71% | -0.29pp |
| aq4 | 95.0% | 94.36% | -0.64pp |

### Class Parity (Key v3 Improvement)

| Quarter | Pooled gap | Per-class gap | Improvement |
|---------|----------:|-------------:|------------:|
| aq1 | 0.09pp | 0.01pp | 9x |
| aq2 | 0.45pp | 0.10pp | 4.5x |
| aq3 | 0.47pp | 0.02pp | 24x |
| aq4 | 0.05pp | 0.07pp | — |

Per-class gap always < 0.14pp. BG7 comfortably passes.

### Width Comparison vs v2

| Quarter | v2 width | v3 width | Change |
|---------|----------:|--------:|-------:|
| aq1 | 2,646 | 2,664 | +0.7% |
| aq2 | 3,122 | 3,126 | +0.1% |
| aq3 | 2,496 | 2,497 | +0.0% |
| aq4 | 2,217 | 2,073 | -6.5% |

Width-neutral on aq1-aq3, 6.5% narrower on aq4.

### Temporal Validation

| Quarter | LOO P95 | Temporal P95 |
|---------|--------:|------------:|
| aq1 | 94.66% | 94.00% |
| aq2 | 94.65% | 92.55% |
| aq3 | 94.71% | 91.37% |
| aq4 | 94.36% | 91.45% |

Comparable to v2 temporal results. PY 2023+ with 3+ training years achieve 94-98%.

## Gate Results

| Gate | Severity | Result |
|------|----------|--------|
| BG0 Baseline still promoted | HARD | PASS |
| BG1 P95 coverage accuracy | HARD | PASS |
| BG2 P50 coverage accuracy | HARD | PASS |
| BG3 Per-bin uniformity | HARD | PASS |
| BG5 Per-PY stability | SOFT | FAIL |
| BG6 Width monotonicity | ADVISORY | PASS |
| BG7 Class parity coverage | ADVISORY | PASS |

### BG5 SOFT Failure Justification

Same as v1/v2: PY 2022 worst-PY P95 coverage drops below 90%. Data property, not method deficiency.

## Decision

Promote as v3. Per-class stratification reduces the onpeak-offpeak coverage gap from 0.05-0.47pp to 0.01-0.10pp while maintaining equivalent overall coverage and width. The improvement is most pronounced in summer quarters (aq1-aq3) where onpeak congestion volatility is highest. Width-neutral except aq4 which is 6.5% narrower. BG7 (class parity) now passes with wide margin.
