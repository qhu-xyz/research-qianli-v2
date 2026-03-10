# Band Calibration v1 — Empirical Quantile Bins

## Method

Symmetric bands around the promoted v3 baseline (nodal f0 stitch):
- `lower = nodal_f0 - width`, `upper = nodal_f0 + width`
- Width = empirical quantile of `|mcp - nodal_f0|` within each `|nodal_f0|` bin
- 4 bins: tiny [0,50), small [50,250), medium [250,1000), large [1000,inf)
- 5 coverage levels: P50, P70, P80, P90, P95
- Validation: leave-one-PY-out (train on 5 PYs, test on 1)

Zero learned parameters beyond the bin quantiles themselves.

## Results

### Aggregate LOO Coverage Accuracy

| Quarter | P50 actual | P50 error | P95 actual | P95 error |
|---------|----------:|----------:|----------:|----------:|
| aq1 | 50.57% | +0.57pp | 94.70% | -0.30pp |
| aq2 | 50.10% | +0.10pp | 94.70% | -0.30pp |
| aq3 | 49.97% | -0.03pp | 94.67% | -0.33pp |
| aq4 | 49.82% | -0.18pp | 94.90% | -0.10pp |

All levels within 0.6pp of target. Slight undercoverage at P95 (~0.3pp) is expected from LOO: train set is 5/6 of data, so extreme quantiles are slightly underestimated.

### Per-Bin P95 Coverage

All bins across all quarters are within 1.3pp of 95% target. The large bin shows the most undercoverage (up to -1.0pp in aq2), consistent with heavier tails at high |f0| values.

### Band Widths (LOO-averaged, $/MWh)

| Bin | aq1 P95 | aq2 P95 | aq3 P95 | aq4 P95 |
|-----|--------:|--------:|--------:|--------:|
| tiny | 933 | 806 | 700 | 546 |
| small | 1,585 | 1,502 | 1,216 | 1,012 |
| medium | 3,158 | 3,287 | 2,626 | 2,182 |
| large | 7,597 | 9,994 | 7,872 | 7,366 |

Width monotonicity holds perfectly: p50 < p70 < p80 < p90 < p95 for all bins and quarters.

Large-bin P95 widths are consistent with v3's per-bin MAE values (MAE_large 2054-2396), as P95 width ~ 3-4x MAE for heavy-tailed distributions.

### Stability (Per-PY)

PY 2022 is the worst planning year across all quarters, consistent with baseline findings:

| Quarter | Worst PY | Worst P95 cov | Range | Width CV |
|---------|----------|:-------------:|------:|--------:|
| aq1 | 2022 | 87.73% | 10.30pp | 0.086 |
| aq2 | 2022 | 85.52% | 13.08pp | 0.082 |
| aq3 | 2022 | 87.34% | 12.11pp | 0.085 |
| aq4 | 2022 | 93.15% | 5.88pp | 0.030 |

**BG5 SOFT failure justification:** PY 2022 worst-PY coverage drops below 90% for aq1-aq3 (87.7%, 85.5%, 87.3%). This is expected: PY 2022 has the highest MAE across all baselines (MAE 915-1546 vs median 757-874) due to extreme congestion volatility. The calibration from the other 5 PYs underestimates the tails for this outlier year. This is a fundamental property of the data, not a flaw in the method. Options for v2: asymmetric bands or wider safety margin for extreme years.

### Clearing Probabilities

Buy-positive paths (baseline > 0, ~65% of paths):
- P95 lower band captures ~99% of actual MCPs
- Even P50 captures ~89-90%

Buy-negative paths (baseline < 0, ~35% of paths):
- P95 captures ~95-96%
- P50 captures only 55-64%

The asymmetry reflects positive bias in nodal_f0: it systematically underestimates positive MCPs, so lower bands are generous for buy-positive but tight for buy-negative.

## Decision

Promote as first band calibration version. Coverage accuracy is within tolerance at all levels. PY 2022 instability is a known data property, not a method deficiency. Width CV is low (3-9%), indicating stable calibration across folds.

## Future Improvements

- **v2:** Asymmetric bands (separate upper/lower widths, accounts for positive bias)
- **v3:** Data-driven bin boundaries
- **v4:** Per-class calibration (separate onpeak/offpeak)
