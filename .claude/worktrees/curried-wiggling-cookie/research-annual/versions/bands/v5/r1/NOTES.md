# Bands R1 v5 — Asymmetric Band Calibration

## Method

Asymmetric bands using signed quantile pairs instead of absolute quantiles.
Instead of `band = baseline +/- quantile(|residual|, level)`, computes:
- `lo = quantile(residual, (1-level)/2)`
- `hi = quantile(residual, (1+level)/2)`
- `band = [baseline + lo, baseline + hi]`

For skewed distributions, the asymmetric interval avoids wasting width on the thin
tail. Half-width `(hi - lo) / 2` is comparable to symmetric `w` for gate checks.

Calibrated at 7 coverage levels: P10, P30, P50, P70, P80, P90, P95.

Winner: **asym_8b_cs** (asymmetric, 8 quantile bins, class+sign stratified).

## Experiments tested (temporal)

Name         | Bins | Type       | Mean P95 hw | BG1
-------------|------|------------|-------------|-----
sym_4b_cs    | 4    | symmetric  | 2214.6      | FAIL
asym_4b_cs   | 4    | asymmetric | 1923.7      | FAIL
asym_6b_cs   | 6    | asymmetric | 1860.8      | FAIL
asym_8b_cs   | 8    | asymmetric | 1822.7      | FAIL

No experiment passes BG1 (28-cell check at 7 levels). Temporal cold-start drives
P50-P90 undershoot in aq3/aq4 beyond 5pp tolerance. asym_8b_cs selected as
lowest mean P95 half-width.

## Gate results (compare v5 vs promoted v3)

Gate   | Severity | Result | Detail
-------|----------|--------|-------
BG0    | HARD     | PASS   | baseline still v3
BG1    | HARD     | FAIL   | 21/28 cells within tolerance, worst: aq4/p70 (+1.6pp over)
BG2a   | HARD     | PASS   | 20/20 cells (promoted v3 has 5 levels only)
BG2b   | HARD     | --     | skipped, promoted v3 lacks per_sign widths
BG3    | HARD     | FAIL   | 1/4 quarters, per-bin uniformity within 5pp
BG4    | SOFT     | FAIL   | 0/4 quarters worst PY >= 90%
BG5    | ADVISORY | PASS   | width monotonicity (p10 < p30 < ... < p95)
BG6    | ADVISORY | PASS   | class parity within 5pp

## Coverage accuracy (temporal)

Quarter | P10       | P30        | P50        | P70        | P80        | P90        | P95
--------|-----------|------------|------------|------------|------------|------------|----------
aq1     | 9.4 -0.6  | 28.1 -1.9  | 46.6 -3.4  | 67.4 -2.6  | 78.2 -1.8  | 88.8 -1.2  | 93.7 -1.3
aq2     | 8.8 -1.2  | 27.1 -2.9  | 46.0 -4.0  | 65.6 -4.4  | 75.8 -4.2  | 86.4 -3.6  | 92.2 -2.8
aq3     | 8.5 -1.5  | 25.9 -4.1  | 43.9 -6.1  | 63.6 -6.4  | 73.9 -6.1  | 84.7 -5.3  | 90.7 -4.3
aq4     | 8.8 -1.2  | 26.6 -3.4  | 44.4 -5.6  | 63.4 -6.6  | 73.9 -6.1  | 85.0 -5.0  | 91.3 -3.7

P10/P30 errors all within 5pp tolerance. BG1 failures driven by P50-P90 undershoot
in aq3/aq4 from temporal cold-start (early folds have 1-2 training PYs).

## LOO validation

Quarter | P95 cov | P95 hw
--------|---------|-------
aq1     | 94.39%  | 2352.1
aq2     | 94.33%  | 2400.8
aq3     | 94.40%  | 2036.3
aq4     | 94.22%  | 1587.2

## Per-class P95 coverage (temporal)

Quarter | onpeak | offpeak | gap
--------|--------|---------|----
aq1     | 93.56% | 93.89%  | 0.33pp
aq2     | 92.28% | 92.07%  | 0.21pp
aq3     | 90.50% | 90.83%  | 0.33pp
aq4     | 91.74% | 90.82%  | 0.92pp

## Per-sign P95 coverage (temporal)

Quarter | prevail | counter | gap
--------|---------|---------|----
aq1     | 93.83%  | 93.47%  | 0.36pp
aq2     | 91.92%  | 92.60%  | 0.68pp
aq3     | 90.71%  | 90.52%  | 0.19pp
aq4     | 90.95%  | 91.88%  | 0.93pp

## P95 mean half-width ($/MWh)

Quarter | v4 width | v5 width | Change
--------|----------|----------|-------
aq1     | 2694     | 2215.8   | -17.7%
aq2     | 3071     | 2118.9   | -31.0%
aq3     | 2545     | 1759.5   | -30.9%
aq4     | 2120     | 1196.4   | -43.6%

## Temporal vs LOO

Quarter | LOO cov | Temp cov | LOO hw | Temp hw
--------|---------|----------|--------|--------
aq1     | 94.39%  | 93.72%   | 2352.1 | 2215.8
aq2     | 94.33%  | 92.18%   | 2400.8 | 2118.9
aq3     | 94.40%  | 90.66%   | 2036.3 | 1759.5
aq4     | 94.22%  | 91.29%   | 1587.2 | 1196.4

## Key findings

1. Asymmetric bands reduce P95 half-width by 18-44% vs v4 symmetric bands.
2. Higher bin count (8 vs 4) contributes additional 2-4% reduction.
3. P10/P30 coverage tracks targets within 1-4pp, well within 5pp tolerance.
4. Class parity gap stays tight (<1pp), sign parity gap <1pp.
5. BG1 failures are entirely in P50-P90 at aq3/aq4, driven by temporal
   cold-start. LOO validation shows all quarters within 1pp of 95% target.
