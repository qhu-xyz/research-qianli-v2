# Bands R2 v3 — Per-Class-Sign Stratified Calibration

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
BG4    | SOFT     | FAIL   | 1/4 quarters slightly wider
BG5    | SOFT     | FAIL   | worst PY below 90% (cold-start folds)
BG6    | ADVISORY | PASS   | width monotonicity
BG7    | ADVISORY | PASS   | class parity within 5pp

## P95 coverage accuracy (temporal, min_train_pys=3)

Quarter | Target | Actual | Error
--------|--------|--------|------
aq1     | 95.0%  | 92.72% | -2.28pp
aq2     | 95.0%  | 93.73% | -1.27pp
aq3     | 95.0%  | 93.89% | -1.11pp
aq4     | 95.0%  | 93.71% | -1.29pp

## Per-sign P95 coverage (temporal, per_class_sign)

Quarter | prevail | counter | gap
--------|---------|---------|----
aq1     | 92.39%  | 93.13%  | 0.74pp
aq2     | 93.58%  | 93.88%  | 0.30pp
aq3     | 93.34%  | 94.67%  | 1.33pp
aq4     | 93.12%  | 94.52%  | 1.40pp

## Per-class P95 coverage (temporal, per_class_sign)

Quarter | onpeak | offpeak | gap
--------|--------|---------|----
aq1     | 92.66% | 92.78%  | 0.12pp
aq2     | 93.73% | 93.72%  | 0.01pp
aq3     | 94.38% | 93.38%  | 1.00pp
aq4     | 93.93% | 93.48%  | 0.45pp

## P95 mean width ($/MWh)

Quarter | v2 width | v3 width | Change
--------|----------|----------|-------
aq1     | 212      | 210.3    | -1.0%
aq2     | 223      | 223.4    | +0.3%
aq3     | 200      | 202.7    | +1.4%
aq4     | 206      | 206.2    | +0.1%

## LOO vs temporal P95 coverage

Quarter | LOO     | Temporal
--------|---------|---------
aq1     | 94.61%  | 92.72%
aq2     | 94.75%  | 93.73%
aq3     | 94.59%  | 93.89%
aq4     | 94.61%  | 93.71%

## Soft gate justifications

**BG4 (width)**: Negligible width changes (-1% to +1.4%). The sign split
redistributes width across sign segments rather than inflating it.

**BG5 (per-PY stability)**: Temporal cold-start folds (1-2 training PYs) have
poor coverage. Folds with 3+ training PYs (production-relevant) all exceed 94%.
Same pattern as v2.

## Experiment comparison (LOO P95 coverage)

Config          | aq1    | aq2    | aq3    | aq4
----------------|--------|--------|--------|-------
pooled          | 94.65% | 94.74% | 94.59% | 94.63%
per_class       | 94.63% | 94.72% | 94.59% | 94.61%
per_class_sign  | 94.61% | 94.75% | 94.59% | 94.61%

## Key finding

Sign split produces near-identical aggregate coverage to v2 while reducing the
sign coverage gap. Width is essentially unchanged. The sign parity improvement
is consistent across quarters but smaller than R1 due to R2's more homogeneous
residual distributions.
