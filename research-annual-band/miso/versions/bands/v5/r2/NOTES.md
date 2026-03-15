# Bands R2 v4 — Asymmetric Band Experiment (Symmetric Winner)

## Method

Tested asymmetric (signed quantile pair) and symmetric bands with 6 and 8 bins.
Asymmetric experiments failed BG1 on temporal validation, so the symmetric
baseline (sym_6b_cs) was selected as winner.

Calibrated at 7 coverage levels: P10, P30, P50, P70, P80, P90, P95.

Winner: **sym_6b_cs** (symmetric, 6 quantile bins, class+sign stratified).

## Experiments tested (temporal)

Name         | Bins | Type       | Mean P95 hw | BG1
-------------|------|------------|-------------|-----
sym_6b_cs    | 6    | symmetric  | 174.9       | PASS
asym_6b_cs   | 6    | asymmetric | 169.4       | FAIL
asym_8b_cs   | 8    | asymmetric | 167.6       | FAIL

Asymmetric experiments are 3-4% narrower but fail BG1 (28-cell check at 7 levels).
sym_6b_cs passes all 28 cells.

## Gate results (compare v4 vs promoted v2)

Gate   | Severity | Result | Detail
-------|----------|--------|-------
BG1    | HARD     | PASS   | 28/28 cells within tolerance
BG2a   | HARD     | PASS   | 20/20 cells (promoted v2 has 5 levels only)
BG2b   | HARD     | --     | skipped, promoted v2 lacks per_sign widths
BG3    | HARD     | FAIL   | 0/4 quarters, all bins within 5pp
BG4    | SOFT     | FAIL   | 0/4 quarters worst PY >= 90%
BG5    | ADVISORY | PASS   | width monotonicity (p10 < p30 < ... < p95)
BG6    | ADVISORY | PASS   | class parity within 5pp

## Coverage accuracy (temporal)

Quarter | P10       | P30        | P50        | P70        | P80        | P90        | P95
--------|-----------|------------|------------|------------|------------|------------|----------
aq1     | 9.5 -0.5  | 28.2 -1.8  | 47.4 -2.6  | 66.6 -3.4  | 76.3 -3.7  | 86.3 -3.7  | 91.9 -3.1
aq2     | 10.2 +0.2 | 29.5 -0.5  | 48.1 -1.9  | 66.8 -3.2  | 76.3 -3.7  | 86.1 -3.9  | 91.6 -3.4
aq3     | 9.9 -0.1  | 29.2 -0.8  | 48.0 -2.0  | 67.2 -2.8  | 76.7 -3.4  | 86.9 -3.1  | 92.5 -2.5
aq4     | 10.2 +0.1 | 29.4 -0.6  | 48.1 -1.9  | 67.1 -2.9  | 76.9 -3.1  | 86.6 -3.4  | 92.2 -2.8

All 28 cells within 5pp tolerance. P10/P30 errors are within 2pp.

## P95 mean half-width ($/MWh)

Quarter | v3 width | v4 width | Change
--------|----------|----------|-------
aq1     | 210      | 176.4    | -16.1%
aq2     | 223      | 182.0    | -18.5%
aq3     | 203      | 169.9    | -16.2%
aq4     | 206      | 171.4    | -16.9%

## Per-class P95 coverage (temporal)

Quarter | onpeak | offpeak | gap
--------|--------|---------|----
aq1     | 92.09% | 91.64%  | 0.45pp
aq2     | 91.85% | 91.37%  | 0.48pp
aq3     | 92.88% | 92.05%  | 0.83pp
aq4     | 92.62% | 91.74%  | 0.88pp

## Per-sign P95 coverage (temporal)

Quarter | prevail | counter | gap
--------|---------|---------|----
aq1     | 91.42%  | 92.49%  | 1.07pp
aq2     | 90.91%  | 92.59%  | 1.68pp
aq3     | 92.02%  | 93.12%  | 1.10pp
aq4     | 91.44%  | 93.22%  | 1.78pp

## LOO validation (winner)

Quarter | LOO cov | LOO hw
--------|---------|-------
aq1     | 94.61%  | 215.7
aq2     | 94.75%  | 227.6
aq3     | 94.59%  | 204.3
aq4     | 94.61%  | 212.1

## Key findings

1. R2 asymmetric bands fail BG1 at 7 levels (P50-P90 undershoot from cold-start),
   consistent with earlier 5-level results.
2. The symmetric winner (sym_6b_cs) now passes all 28 BG1 cells including P10/P30.
3. P10/P30 coverage tracks targets closely: P10 within 0.5pp, P30 within 1.8pp.
4. Width reduction vs v3 is 16-19% from the 6-bin quantile boundary refinement.
5. R2 data has more symmetric residual distributions than R1, explaining why
   asymmetric bands provide less benefit here.
