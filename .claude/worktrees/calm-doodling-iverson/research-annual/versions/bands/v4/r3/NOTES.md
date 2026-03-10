# Bands R3 v3 — Per-Class-Sign Stratified Calibration

## Method

Extends v2 by adding sign(baseline) as a third stratification axis.
Calibration cells: (|baseline| bin x class_type x sign_seg) = 6 bins x 2 classes x 2 signs = 24 cells.
Fallback chain: (bin, class, sign) -> (bin, class) -> (bin, pooled) when cell has < 500 rows.
Gate-facing metrics use temporal expanding-window (min 3 training PYs).

## Gate results (temporal, vs v2)

Gate   | Severity | Result | Detail
-------|----------|--------|-------
BG1    | HARD     | PASS   | P95 coverage within 3pp all 4 quarters
BG2    | HARD     | PASS   | P50 coverage within 5pp all 4 quarters
BG3    | HARD     | PASS   | per-bin uniformity within 5pp
BG4    | SOFT     | FAIL   | 3/4 quarters slightly wider
BG5    | SOFT     | FAIL   | worst PY below 90% (cold-start folds)
BG6    | ADVISORY | PASS   | width monotonicity
BG7    | ADVISORY | PASS   | class parity within 5pp

## P95 coverage accuracy (temporal, min_train_pys=3)

Quarter | Target | Actual | Error
--------|--------|--------|------
aq1     | 95.0%  | 94.51% | -0.49pp
aq2     | 95.0%  | 93.94% | -1.06pp
aq3     | 95.0%  | 94.53% | -0.47pp
aq4     | 95.0%  | 94.59% | -0.41pp

## Per-sign P95 coverage (temporal, per_class_sign)

Quarter | prevail | counter | gap
--------|---------|---------|----
aq1     | 93.91%  | 95.41%  | 1.50pp
aq2     | 93.54%  | 94.49%  | 0.95pp
aq3     | 94.25%  | 94.94%  | 0.69pp
aq4     | 93.89%  | 95.61%  | 1.72pp

## Per-class P95 coverage (temporal, per_class_sign)

Quarter | onpeak | offpeak | gap
--------|--------|---------|----
aq1     | 94.24% | 94.80%  | 0.56pp
aq2     | 93.92% | 93.96%  | 0.04pp
aq3     | 94.44% | 94.63%  | 0.19pp
aq4     | 94.57% | 94.60%  | 0.03pp

## P95 mean width ($/MWh)

Quarter | v2 width | v3 width | Change
--------|----------|----------|-------
aq1     | 186      | 186.2    | +0.2%
aq2     | 179      | 176.8    | -1.0%
aq3     | 166      | 164.5    | -0.6%
aq4     | 163      | 162.4    | -0.6%

## LOO vs temporal P95 coverage

Quarter | LOO     | Temporal
--------|---------|---------
aq1     | 94.52%  | 94.51%
aq2     | 94.78%  | 93.94%
aq3     | 94.52%  | 94.53%
aq4     | 94.63%  | 94.59%

## Soft gate justifications

**BG4 (width)**: Width changes are minimal (-1% to +0.2%). AQ1 is 0.2% wider;
AQ2-4 are narrower. Net effect is slight width reduction.

**BG5 (per-PY stability)**: Temporal cold-start folds (1-2 training PYs) have
poor coverage. Folds with 3+ training PYs (production-relevant) all exceed 94%.
Same pattern as v2.

## Experiment comparison (LOO P95 coverage)

Config          | aq1    | aq2    | aq3    | aq4
----------------|--------|--------|--------|-------
pooled          | 94.58% | 94.81% | 94.54% | 94.67%
per_class       | 94.57% | 94.79% | 94.54% | 94.66%
per_class_sign  | 94.52% | 94.78% | 94.52% | 94.63%

## Key finding

Sign split maintains aggregate coverage within 0.1pp of v2 while redistributing
band widths by sign segment. Counter-flow paths (baseline < 0) get wider bands
matching their wider residual tails. R3 shows the largest sign coverage gaps
(up to 1.7pp on temporal), which is expected since R3 residuals have the most
pronounced prevail/counter asymmetry. Width is essentially unchanged or slightly
narrower overall.
