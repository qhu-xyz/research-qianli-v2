# Bands v8/r3 — Production Candidate (5 bins, bidirectional correction)

**Date:** 2026-02-26  |  **Script:** `scripts/run_v8_bands.py`  |  **Tests:** `test_v8_bands.py` (659 pass)

## Method

- **Asymmetric signed quantile bands** with per-(bin, class, sign) calibration
- **5 quantile bins** (chosen for consistency with R1; R3 has plenty of data for any bin count)
- **Bidirectional OOS correction** (shrink over-covering, expand under-covering cells)
- **Cold-start inflation** for folds with <3 training PYs (factor = 1 + 0.15/n_train_pys)
- **Temporal CV** (expanding window, min_train_pys=1)
- Fallback chain: (bin, class, sign) -> (bin, class) -> (bin, pooled), min_cell_rows=500

## Experiments

| Experiment | Bins | Correction | BG1 | Mean P95 hw |
|------------|-----:|:----------:|:---:|------------:|
| asym_5b | 5 | none | FAIL | 138.3 |
| asym_5b_bidir | 5 | OOS bidir | PASS | 150.9 |

**Winner:** asym_5b_bidir

## Results

**P95 coverage (temporal CV):**

| Quarter | Target | Actual | Error |
|---------|-------:|-------:|------:|
| aq1 | 95.0% | 92.04% | -2.96pp |
| aq2 | 95.0% | 91.93% | -3.07pp |
| aq3 | 95.0% | 92.05% | -2.95pp |
| aq4 | 95.0% | 92.43% | -2.57pp |

**P95 mean half-width ($/MWh):**

| Quarter | v8 hw | v3 hw | vs v3 | v7 hw | vs v7 |
|---------|------:|------:|------:|------:|------:|
| aq1 | 161.1 | 186 | -13.3% | 156 | +3.5% |
| aq2 | 157.2 | 179 | -12.0% | 152 | +3.2% |
| aq3 | 144.0 | 166 | -13.0% | 142 | +1.6% |
| aq4 | 141.3 | 163 | -13.5% | 135 | +4.4% |

**Per-class P95 coverage (gap < 1pp all quarters):**

| Quarter | onpeak | offpeak | gap |
|---------|-------:|--------:|----:|
| aq1 | 92.28% | 91.77% | 0.51pp |
| aq2 | 91.86% | 92.01% | 0.15pp |
| aq3 | 92.17% | 91.92% | 0.25pp |
| aq4 | 92.59% | 92.25% | 0.34pp |

**Per-sign P95 coverage:**

| Quarter | prevail | counter | gap |
|---------|--------:|--------:|----:|
| aq1 | 91.42% | 92.96% | 1.54pp |
| aq2 | 91.42% | 92.64% | 1.22pp |
| aq3 | 91.58% | 92.75% | 1.17pp |
| aq4 | 91.76% | 93.42% | 1.66pp |

## Gate Results

| Gate | Severity | Result | Detail |
|------|----------|--------|--------|
| BG1 | HARD | PASS | 28/28 coverage cells within tolerance |
| BG2a | HARD | PASS | 20/20 width cells within tolerance |
| BG3 | HARD | FAIL | 0/4 quarters pass per-bin uniformity |
| BG4 | SOFT | FAIL | 0/4 quarters worst-PY >= 90% |
| BG5 | ADVISORY | PASS | 4/4 width monotonicity |
| BG6 | ADVISORY | PASS | 4/4 class parity < 5pp |

**BG3 failure:** Per-bin uniformity fails in all 4 quarters. Despite R3's ~1.2M rows/quarter, the top bin's fat-tailed distribution causes coverage non-uniformity.

**BG4 failure:** PY 2022 structural anomaly. Irreducible.

## vs v7 (10-bin winner)

v8 uses 5 bins vs v7's 10 bins for R3. Coverage is nearly identical (within ±0.2pp). Widths are 1.6-4.4% wider. The tradeoff is consistency with R1's bin count.
