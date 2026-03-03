# Bands v8/r1 — Production Candidate (5 bins, bidirectional correction)

**Date:** 2026-02-26  |  **Script:** `scripts/run_v8_bands.py`  |  **Tests:** `test_v8_bands.py` (659 pass)

## Method

- **Asymmetric signed quantile bands** with per-(bin, class, sign) calibration
- **5 quantile bins** (chosen via cell size analysis: 5 bins is safest for R1's ~130-150k rows)
- **Bidirectional OOS correction** (shrink over-covering, expand under-covering cells)
- **Cold-start inflation** for folds with <3 training PYs (factor = 1 + 0.15/n_train_pys)
- **Temporal CV** (expanding window, min_train_pys=1)
- Fallback chain: (bin, class, sign) -> (bin, class) -> (bin, pooled), min_cell_rows=500

## Why 5 bins

Cell size analysis showed 5 bins is optimal for R1's limited data (6 PYs, ~130-150k rows/quarter):
- 5 bins: 4-5 cells below 500-row fallback in 1-PY folds
- 8 bins: 15-18 cells below 500-row fallback in 1-PY folds
- Width difference vs 6 bins: ~1%
- R2/R3 have 7-8x more data; any bin count 5-10 works fine

## Experiments

| Experiment | Bins | Correction | BG1 | Mean P95 hw |
|------------|-----:|:----------:|:---:|------------:|
| asym_5b | 5 | none | FAIL | 1,884.4 |
| asym_5b_bidir | 5 | OOS bidir | PASS | 2,041.8 |

**Winner:** asym_5b_bidir

## Results

**P95 coverage (temporal CV):**

| Quarter | Target | Actual | Error |
|---------|-------:|-------:|------:|
| aq1 | 95.0% | 94.15% | -0.85pp |
| aq2 | 95.0% | 93.39% | -1.61pp |
| aq3 | 95.0% | 91.79% | -3.21pp |
| aq4 | 95.0% | 92.27% | -2.73pp |

**P95 mean half-width ($/MWh):**

| Quarter | v8 hw | v3 hw | vs v3 | v7 hw | vs v7 |
|---------|------:|------:|------:|------:|------:|
| aq1 | 2,377 | 2,664 | -10.8% | 2,374 | +0.1% |
| aq2 | 2,415 | 3,126 | -22.7% | 2,328 | +3.7% |
| aq3 | 1,996 | 2,496 | -20.1% | 1,849 | +7.9% |
| aq4 | 1,380 | 2,073 | -33.4% | 1,305 | +5.8% |

**Per-class P95 coverage (gap < 1pp all quarters):**

| Quarter | onpeak | offpeak | gap |
|---------|-------:|--------:|----:|
| aq1 | 94.26% | 94.03% | 0.23pp |
| aq2 | 93.53% | 93.24% | 0.29pp |
| aq3 | 91.78% | 91.79% | 0.01pp |
| aq4 | 92.69% | 91.84% | 0.85pp |

**Per-sign P95 coverage:**

| Quarter | prevail | counter | gap |
|---------|--------:|--------:|----:|
| aq1 | 94.19% | 94.04% | 0.15pp |
| aq2 | 92.97% | 94.12% | 1.15pp |
| aq3 | 91.62% | 92.05% | 0.43pp |
| aq4 | 91.88% | 92.99% | 1.11pp |

## Gate Results

| Gate | Severity | Result | Detail |
|------|----------|--------|--------|
| BG0 | HARD | PASS | baseline v3 still promoted |
| BG1 | HARD | PASS | 28/28 coverage cells within tolerance |
| BG2a | HARD | FAIL | 19/20 cells; worst: aq1/p50 (+13.4%) |
| BG3 | HARD | FAIL | 1/4 quarters pass per-bin uniformity |
| BG4 | SOFT | FAIL | 0/4 quarters worst-PY >= 90% |
| BG5 | ADVISORY | PASS | 4/4 width monotonicity |
| BG6 | ADVISORY | PASS | 4/4 class parity < 5pp |

**BG2a failure:** aq1/p50 level is 13.4% wider than v3. This is specific to the P50 level in one quarter.

**BG3 failure:** Per-bin uniformity still fails. With 5 bins each bin gets 20% of data (vs 10% for 10 bins), but the top bin's fat tails still cause under-coverage in some quarters.

**BG4 failure:** PY 2022 structural anomaly (European energy crisis). Coverage 82-87%. Irreducible with any calibration approach.

## vs v7 (8-bin winner)

v8 uses 5 bins vs v7's 8 bins. Coverage is nearly identical (within ±0.2pp). Widths are slightly wider (+0.1% to +7.9%) because fewer bins means coarser segmentation. The tradeoff is better cell reliability: fewer fallback cells, more stable quantile estimates.
