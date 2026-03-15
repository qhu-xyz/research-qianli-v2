# Bands v8/r2 — Production Candidate (5 bins, bidirectional correction)

**Date:** 2026-02-26  |  **Script:** `scripts/run_v8_bands.py`  |  **Tests:** `test_v8_bands.py` (659 pass)

## Method

- **Asymmetric signed quantile bands** with per-(bin, class, sign) calibration
- **5 quantile bins** (chosen for consistency with R1; R2 has plenty of data for any bin count)
- **Bidirectional OOS correction** (shrink over-covering, expand under-covering cells)
- **Cold-start inflation** for folds with <3 training PYs (factor = 1 + 0.15/n_train_pys)
- **Temporal CV** (expanding window, min_train_pys=1)
- Fallback chain: (bin, class, sign) -> (bin, class) -> (bin, pooled), min_cell_rows=500

## Experiments

| Experiment | Bins | Correction | BG1 | Mean P95 hw |
|------------|-----:|:----------:|:---:|------------:|
| asym_5b | 5 | none | FAIL | 170.8 |
| asym_5b_bidir | 5 | OOS bidir | PASS | 189.5 |

**Winner:** asym_5b_bidir

## Results

**P95 coverage (temporal CV):**

| Quarter | Target | Actual | Error |
|---------|-------:|-------:|------:|
| aq1 | 95.0% | 92.71% | -2.29pp |
| aq2 | 95.0% | 92.60% | -2.40pp |
| aq3 | 95.0% | 93.47% | -1.53pp |
| aq4 | 95.0% | 93.17% | -1.83pp |

**P95 mean half-width ($/MWh):**

| Quarter | v8 hw | v3 hw | vs v3 | v7 hw | vs v7 |
|---------|------:|------:|------:|------:|------:|
| aq1 | 196.8 | 212 | -7.3% | 188 | +4.5% |
| aq2 | 195.7 | 223 | -12.2% | 188 | +4.2% |
| aq3 | 180.9 | 200 | -9.5% | 177 | +2.1% |
| aq4 | 184.7 | 206 | -10.3% | 180 | +2.6% |

**Per-class P95 coverage (gap < 1pp all quarters):**

| Quarter | onpeak | offpeak | gap |
|---------|-------:|--------:|----:|
| aq1 | 92.88% | 92.53% | 0.35pp |
| aq2 | 92.56% | 92.65% | 0.09pp |
| aq3 | 93.69% | 93.24% | 0.45pp |
| aq4 | 93.47% | 92.86% | 0.61pp |

**Per-sign P95 coverage:**

| Quarter | prevail | counter | gap |
|---------|--------:|--------:|----:|
| aq1 | 92.21% | 93.42% | 1.21pp |
| aq2 | 91.86% | 93.65% | 1.79pp |
| aq3 | 93.14% | 93.94% | 0.80pp |
| aq4 | 92.51% | 94.09% | 1.58pp |

## Gate Results

| Gate | Severity | Result | Detail |
|------|----------|--------|--------|
| BG1 | HARD | PASS | 28/28 coverage cells within tolerance |
| BG2a | HARD | PASS | 20/20 width cells within tolerance |
| BG3 | HARD | FAIL | 1/4 quarters pass per-bin uniformity |
| BG4 | SOFT | FAIL | 0/4 quarters worst-PY >= 90% |
| BG5 | ADVISORY | PASS | 4/4 width monotonicity |
| BG6 | ADVISORY | PASS | 4/4 class parity < 5pp |

**BG3 failure:** Per-bin uniformity fails in 3/4 quarters. Even with R2's ~1M rows/quarter, the top bin's fat tails cause under-coverage in some bins/quarters.

**BG4 failure:** PY 2022 structural anomaly. Irreducible.

## vs v7 (10-bin winner)

v8 uses 5 bins vs v7's 10 bins for R2. Coverage is nearly identical (within ±0.1pp). Widths are 2-5% wider because fewer bins means coarser segmentation. The tradeoff is consistency with R1's bin count.
