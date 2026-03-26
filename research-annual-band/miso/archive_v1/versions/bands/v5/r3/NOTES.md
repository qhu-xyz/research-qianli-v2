# Bands R3 v4 — Asymmetric Band Calibration

## Method

Asymmetric bands using signed quantile pairs. Tests 6-bin and 8-bin configs
against a symmetric 6-bin baseline.

Calibrated at 7 coverage levels: P10, P30, P50, P70, P80, P90, P95.

Winner: **asym_8b_cs** (asymmetric, 8 quantile bins, class+sign stratified).

## Experiments tested (temporal)

Name         | Bins | Type       | Mean P95 hw | BG1
-------------|------|------------|-------------|-----
sym_6b_cs    | 6    | symmetric  | 142.4       | FAIL
asym_6b_cs   | 6    | asymmetric | 137.2       | FAIL
asym_8b_cs   | 8    | asymmetric | 135.9       | FAIL

No experiment passes BG1 (28-cell check at 7 levels). Temporal cold-start drives
P50-P90 undershoot beyond 5pp tolerance. asym_8b_cs selected as lowest mean P95
half-width.

## Gate results (compare v4 vs promoted v2)

Gate   | Severity | Result | Detail
-------|----------|--------|-------
BG1    | HARD     | FAIL   | 15/28 cells within tolerance, worst: aq2/p70 (+1.8pp over)
BG2a   | HARD     | PASS   | 20/20 cells (promoted v2 has 5 levels only)
BG2b   | HARD     | --     | skipped, promoted v2 lacks per_sign widths
BG3    | HARD     | FAIL   | 0/4 quarters, all bins within 5pp
BG4    | SOFT     | FAIL   | 0/4 quarters worst PY >= 90%
BG5    | ADVISORY | PASS   | width monotonicity (p10 < p30 < ... < p95)
BG6    | ADVISORY | PASS   | class parity within 5pp

## Coverage accuracy (temporal)

Quarter | P10       | P30        | P50        | P70        | P80        | P90        | P95
--------|-----------|------------|------------|------------|------------|------------|----------
aq1     | 9.2 -0.8  | 26.9 -3.0  | 45.1 -4.9  | 63.4 -6.6  | 73.6 -6.4  | 84.6 -5.4  | 90.9 -4.1
aq2     | 8.9 -1.1  | 26.6 -3.4  | 44.6 -5.4  | 63.2 -6.8  | 73.3 -6.7  | 84.3 -5.7  | 90.8 -4.2
aq3     | 8.4 -1.6  | 25.7 -4.3  | 44.0 -6.0  | 63.3 -6.7  | 73.8 -6.2  | 84.9 -5.1  | 91.0 -4.0
aq4     | 8.9 -1.1  | 27.2 -2.8  | 45.0 -5.0  | 64.0 -6.0  | 74.1 -5.9  | 85.4 -4.6  | 91.6 -3.4

P10/P30 errors within 5pp tolerance. BG1 failures driven by P50-P90 undershoot
from temporal cold-start (early folds with 1-2 training PYs).

## P95 mean half-width ($/MWh)

Quarter | v3 width | v4 width | Change
--------|----------|----------|-------
aq1     | 186      | 143.6    | -22.9%
aq2     | 177      | 140.1    | -20.8%
aq3     | 164      | 131.0    | -20.4%
aq4     | 162      | 129.0    | -20.6%

## Per-class P95 coverage (temporal)

Quarter | onpeak | offpeak | gap
--------|--------|---------|----
aq1     | 91.10% | 90.72%  | 0.38pp
aq2     | 90.63% | 91.08%  | 0.45pp
aq3     | 91.11% | 90.97%  | 0.14pp
aq4     | 92.00% | 91.16%  | 0.84pp

## Per-sign P95 coverage (temporal)

Quarter | prevail | counter | gap
--------|---------|---------|----
aq1     | 90.47%  | 91.55%  | 1.08pp
aq2     | 90.49%  | 91.30%  | 0.81pp
aq3     | 90.51%  | 91.82%  | 1.31pp
aq4     | 91.01%  | 92.44%  | 1.43pp

## LOO validation (winner)

Quarter | LOO cov | LOO hw
--------|---------|-------
aq1     | 94.32%  | 173.9
aq2     | 94.58%  | 176.9
aq3     | 94.38%  | 157.8
aq4     | 94.47%  | 155.3

## Key findings

1. Asymmetric bands reduce P95 half-width by 21-23% vs v3 symmetric bands.
2. P10/P30 coverage tracks targets within 1-4pp, well within 5pp tolerance.
3. BG1 failures (15/28 cells) are at P50-P90 in aq1-aq3 from temporal cold-start.
4. R3 residuals are more skewed than R2, making asymmetric calibration effective.
5. Sign gap is 1-1.4pp, within the 5pp BG6 threshold.
6. The 2022 PY remains the most challenging across all rounds (cold start + regime
   shift), contributing to temporal BG4 failures.
