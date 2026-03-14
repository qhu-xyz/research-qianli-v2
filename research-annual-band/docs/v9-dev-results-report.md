# V9 Simplified Asymmetric Bands — Dev Results Report

**Date:** 2026-03-14
**Method:** Asymmetric signed quantile pairs, 5 bins x 2 classes = 10 cells, no sign split, no correction
**CV:** Temporal expanding, min_train_pys=2
**Dev PYs:** R1 2020-2024, R2/R3 2019-2024 (PY 2025 reserved as holdout)

---

## Table A: Overall Coverage Accuracy (All 8 Levels)

Coverage = % of test paths where `mcp` falls within `[baseline + lo, baseline + hi]`.
Error = actual - target. Negative = under-coverage.

### R1 (baseline = nodal_f0)

| Quarter | P10 | P30 | P50 | P70 | P80 | P90 | P95 | P99 |
|---------|----:|----:|----:|----:|----:|----:|----:|----:|
| aq1 | 10.8 | 31.5 | 53.2 | 72.4 | 82.0 | 91.4 | **95.5** | 98.6 |
| aq2 | 9.9 | 28.7 | 48.2 | 68.0 | 77.8 | 88.2 | **93.7** | 98.0 |
| aq3 | 8.9 | 27.9 | 47.1 | 67.1 | 77.7 | 88.4 | **93.8** | 97.9 |
| aq4 | 8.4 | 25.9 | 44.1 | 62.4 | 72.5 | 83.6 | **90.0** | 96.7 |

**Error vs target (pp):**

| Quarter | P10 | P30 | P50 | P70 | P80 | P90 | P95 | P99 |
|---------|----:|----:|----:|----:|----:|----:|----:|----:|
| aq1 | +0.8 | +1.5 | +3.2 | +2.4 | +2.0 | +1.4 | **+0.5** | -0.4 |
| aq2 | -0.1 | -1.3 | -1.8 | -2.0 | -2.2 | -1.8 | **-1.3** | -1.0 |
| aq3 | -1.1 | -2.1 | -2.9 | -2.9 | -2.3 | -1.6 | **-1.2** | -1.1 |
| aq4 | -1.6 | -4.1 | -5.9 | -7.6 | -7.5 | -6.4 | **-5.0** | -2.3 |

### R2 (baseline = mtm_1st_mean, prior round MCP)

| Quarter | P10 | P30 | P50 | P70 | P80 | P90 | P95 | P99 |
|---------|----:|----:|----:|----:|----:|----:|----:|----:|
| aq1 | 8.8 | 26.1 | 44.0 | 63.0 | 72.6 | 83.3 | **89.6** | 96.1 |
| aq2 | 8.6 | 26.4 | 44.3 | 63.0 | 72.6 | 83.5 | **89.8** | 96.4 |
| aq3 | 9.4 | 28.0 | 46.6 | 64.8 | 74.0 | 84.7 | **91.0** | 97.1 |
| aq4 | 9.2 | 26.8 | 44.9 | 63.6 | 73.4 | 84.0 | **90.3** | 97.0 |

**Error vs target (pp):**

| Quarter | P10 | P30 | P50 | P70 | P80 | P90 | P95 | P99 |
|---------|----:|----:|----:|----:|----:|----:|----:|----:|
| aq1 | -1.2 | -3.9 | -6.0 | -7.0 | -7.4 | -6.7 | **-5.4** | -2.9 |
| aq2 | -1.4 | -3.6 | -5.7 | -7.0 | -7.4 | -6.5 | **-5.2** | -2.6 |
| aq3 | -0.6 | -2.0 | -3.4 | -5.2 | -6.0 | -5.3 | **-4.0** | -1.9 |
| aq4 | -0.8 | -3.2 | -5.1 | -6.4 | -6.6 | -6.0 | **-4.7** | -2.0 |

### R3 (baseline = mtm_1st_mean, prior round MCP)

| Quarter | P10 | P30 | P50 | P70 | P80 | P90 | P95 | P99 |
|---------|----:|----:|----:|----:|----:|----:|----:|----:|
| aq1 | 9.4 | 27.5 | 46.2 | 65.0 | 74.6 | 85.0 | **91.5** | 97.5 |
| aq2 | 9.1 | 26.5 | 44.5 | 63.7 | 74.0 | 85.1 | **91.6** | 97.6 |
| aq3 | 8.6 | 26.1 | 45.4 | 64.9 | 75.3 | 86.3 | **92.0** | 97.8 |
| aq4 | 8.9 | 27.4 | 45.9 | 65.1 | 75.5 | 86.5 | **92.6** | 97.9 |

**Error vs target (pp):**

| Quarter | P10 | P30 | P50 | P70 | P80 | P90 | P95 | P99 |
|---------|----:|----:|----:|----:|----:|----:|----:|----:|
| aq1 | -0.6 | -2.5 | -3.8 | -5.0 | -5.4 | -5.0 | **-3.5** | -1.5 |
| aq2 | -0.9 | -3.5 | -5.5 | -6.3 | -6.0 | -4.9 | **-3.4** | -1.4 |
| aq3 | -1.4 | -3.9 | -4.6 | -5.1 | -4.7 | -3.7 | **-3.0** | -1.2 |
| aq4 | -1.1 | -2.6 | -4.1 | -4.9 | -4.5 | -3.5 | **-2.4** | -1.1 |

---

## Table B: Per-PY P95 Coverage (Stability Check)

### R1

| Quarter | PY2022 | PY2023 | PY2024 | Range | Worst |
|---------|-------:|-------:|-------:|------:|------:|
| aq1 | 89.9% | 98.4% | 97.2% | 8.6pp | 89.9% |
| aq2 | 86.2% | 98.3% | 95.0% | 12.1pp | 86.2% |
| aq3 | 86.3% | 97.3% | 95.8% | 11.0pp | 86.3% |
| aq4 | 87.3% | 92.3% | 92.5% | 5.6pp | 87.3% |

PY 2021 excluded (train=1PY, below min_train_pys=2).
PY 2022 is consistently the worst — trained on only 2020-2021 (2 PYs).

### R2

| Quarter | PY2021 | PY2022 | PY2023 | PY2024 | Range | Worst |
|---------|-------:|-------:|-------:|-------:|------:|------:|
| aq1 | 84.9% | 79.7% | 94.6% | 96.1% | 16.4pp | 79.7% |
| aq2 | 81.8% | 86.2% | 94.5% | 94.7% | 12.9pp | 81.8% |
| aq3 | 85.6% | 84.8% | 94.3% | 96.6% | 11.8pp | 84.8% |
| aq4 | 83.9% | 83.6% | 94.5% | 96.1% | 12.5pp | 83.6% |

### R3

| Quarter | PY2021 | PY2022 | PY2023 | PY2024 | Range | Worst |
|---------|-------:|-------:|-------:|-------:|------:|------:|
| aq1 | 86.3% | 82.0% | 96.7% | 97.4% | 15.4pp | 82.0% |
| aq2 | 88.7% | 83.5% | 95.4% | 95.7% | 12.2pp | 83.5% |
| aq3 | 88.0% | 83.0% | 96.1% | 97.5% | 14.5pp | 83.0% |
| aq4 | 90.0% | 84.6% | 96.2% | 96.5% | 11.8pp | 84.6% |

**Pattern:** PY 2021-2022 consistently under-cover (80-89%). PY 2023-2024 consistently over-cover (92-98%).
This is the temporal cold-start effect: early folds have 1-2 training PYs with noisy quantile estimates.
With 4+ training PYs (PY 2023+), P95 coverage is 92-98% — close to or above the 95% target.

---

## Table C: Per-Bin P95 Coverage

### R1

| Quarter | q1 (smallest \|f0\|) | q2 | q3 | q4 | q5 (largest \|f0\|) |
|---------|----:|----:|----:|----:|----:|
| aq1 | 96.5% | 96.5% | 96.2% | 96.4% | 93.1% |
| aq2 | 94.7% | 93.6% | 94.7% | 95.5% | 90.7% |
| aq3 | 94.4% | 94.9% | 95.2% | 95.9% | 89.9% |
| aq4 | 92.3% | 91.3% | 91.5% | 92.1% | 84.2% |

### R2

| Quarter | q1 | q2 | q3 | q4 | q5 |
|---------|----:|----:|----:|----:|----:|
| aq1 | 92.5% | 90.7% | 89.9% | 91.3% | 84.5% |
| aq2 | 91.6% | 90.7% | 90.4% | 92.4% | 84.9% |
| aq3 | 92.3% | 91.5% | 91.9% | 93.1% | 86.7% |
| aq4 | 92.9% | 91.5% | 91.4% | 91.5% | 84.8% |

### R3

| Quarter | q1 | q2 | q3 | q4 | q5 |
|---------|----:|----:|----:|----:|----:|
| aq1 | 94.1% | 92.8% | 92.1% | 93.1% | 86.5% |
| aq2 | 93.5% | 92.8% | 92.3% | 93.3% | 87.0% |
| aq3 | 93.8% | 93.4% | 92.7% | 93.8% | 87.6% |
| aq4 | 93.9% | 93.5% | 93.8% | 94.6% | 88.0% |

**Pattern:** q5 (largest |baseline|) consistently under-covers by 5-11pp vs q1-q4.
These are high-value paths with heavy-tailed residuals — the 97.5th percentile quantile estimate
is noisy with limited training data. R1 aq4 q5 is the worst cell at 84.2%.

---

## Table D: P95 Half-Width Comparison vs Promoted v3

### R1

| Quarter | v3 (promoted) | v9 (new) | Reduction |
|---------|-------------:|----------:|---------:|
| aq1 | 2,942 | 2,701 | **-8.2%** |
| aq2 | 3,239 | 2,772 | **-14.4%** |
| aq3 | 2,746 | 2,339 | **-14.8%** |
| aq4 | 1,733 | 1,369 | **-21.0%** |
| **Avg** | **2,665** | **2,295** | **-13.9%** |

### R2

| Quarter | v3 (promoted) | v9 (new) | Reduction |
|---------|-------------:|----------:|---------:|
| aq1 | 212 | 180 | **-15.1%** |
| aq2 | 223 | 186 | **-16.6%** |
| aq3 | 200 | 170 | **-14.8%** |
| aq4 | 206 | 173 | **-16.0%** |
| **Avg** | **210** | **177** | **-15.6%** |

### R3

| Quarter | v3 (promoted) | v9 (new) | Reduction |
|---------|-------------:|----------:|---------:|
| aq1 | 186 | 164 | **-11.8%** |
| aq2 | 179 | 155 | **-13.3%** |
| aq3 | 166 | 146 | **-12.0%** |
| aq4 | 163 | 144 | **-11.9%** |
| **Avg** | **173** | **152** | **-12.3%** |

### Summary Across All Rounds

| Round | v3 Avg Width | v9 Avg Width | Reduction | v9 Avg P95 Cov |
|-------|------------:|-----------:|---------:|-------:|
| R1 | 2,665 | 2,295 | **-13.9%** | 93.3% |
| R2 | 210 | 177 | **-15.6%** | 90.0% |
| R3 | 173 | 152 | **-12.3%** | 91.9% |

---

## Table E: Class Parity at P95

### R1

| Quarter | onpeak | offpeak | gap |
|---------|-------:|--------:|----:|
| aq1 | 95.02% | 96.09% | 1.07pp |
| aq2 | 93.51% | 93.81% | 0.30pp |
| aq3 | 93.94% | 93.64% | 0.30pp |
| aq4 | 90.29% | 89.70% | 0.59pp |

### R2

| Quarter | onpeak | offpeak | gap |
|---------|-------:|--------:|----:|
| aq1 | 89.80% | 89.42% | 0.38pp |
| aq2 | 89.74% | 89.96% | 0.22pp |
| aq3 | 91.46% | 90.43% | 1.03pp |
| aq4 | 90.65% | 89.93% | 0.72pp |

### R3

| Quarter | onpeak | offpeak | gap |
|---------|-------:|--------:|----:|
| aq1 | 91.07% | 91.89% | 0.82pp |
| aq2 | 91.67% | 91.46% | 0.21pp |
| aq3 | 91.88% | 92.22% | 0.34pp |
| aq4 | 93.02% | 92.13% | 0.89pp |

**All class parity gaps < 1.1pp.** Per-class stratification is working well.

---

## Table F: Per-PY P95 Width ($/MWh) — Stability of Band Widths

### R1

| Quarter | PY2022 | PY2023 | PY2024 | Width CV |
|---------|-------:|-------:|-------:|--------:|
| aq1 | 2,207 | 3,009 | 2,887 | 0.16 |
| aq2 | 2,160 | 3,163 | 2,994 | 0.19 |
| aq3 | 1,706 | 2,755 | 2,556 | 0.22 |
| aq4 | 805 | 1,683 | 1,795 | 0.37 |

### R2

| Quarter | PY2021 | PY2022 | PY2023 | PY2024 | Width CV |
|---------|-------:|-------:|-------:|-------:|--------:|
| aq1 | 104 | 153 | 234 | 231 | 0.35 |
| aq2 | 104 | 171 | 235 | 233 | 0.31 |
| aq3 | 105 | 152 | 212 | 212 | 0.28 |
| aq4 | 100 | 153 | 221 | 219 | 0.31 |

### R3

| Quarter | PY2021 | PY2022 | PY2023 | PY2024 | Width CV |
|---------|-------:|-------:|-------:|-------:|--------:|
| aq1 | 103 | 149 | 205 | 199 | 0.30 |
| aq2 | 102 | 133 | 194 | 191 | 0.28 |
| aq3 | 97 | 127 | 182 | 177 | 0.28 |
| aq4 | 102 | 124 | 177 | 174 | 0.25 |

**Pattern:** Widths grow as training set expands (expected — more extreme residuals are captured with more data).
Width CV ranges 0.16-0.37. The highest CV is R1 aq4 (0.37) because aq4 spans the most volatile spring months.

---

## Table G: Per-PY P99 Coverage

### R1

| Quarter | PY2022 | PY2023 | PY2024 | Range | Worst |
|---------|-------:|-------:|-------:|------:|------:|
| aq1 | 95.8% | 99.8% | 99.7% | 4.0pp | 95.8% |
| aq2 | 94.1% | 99.8% | 99.2% | 5.7pp | 94.1% |
| aq3 | 93.3% | 99.6% | 99.5% | 6.3pp | 93.3% |
| aq4 | 94.2% | 98.6% | 98.6% | 4.4pp | 94.2% |

### R2

| Quarter | PY2021 | PY2022 | PY2023 | PY2024 | Range | Worst |
|---------|-------:|-------:|-------:|-------:|------:|------:|
| aq1 | 93.7% | 90.3% | 99.1% | 99.3% | 9.0pp | 90.3% |
| aq2 | 91.6% | 94.8% | 99.0% | 99.0% | 7.4pp | 91.6% |
| aq3 | 93.8% | 94.8% | 99.1% | 99.4% | 5.6pp | 93.8% |
| aq4 | 93.5% | 94.2% | 99.3% | 99.5% | 6.0pp | 93.5% |

### R3

| Quarter | PY2021 | PY2022 | PY2023 | PY2024 | Range | Worst |
|---------|-------:|-------:|-------:|-------:|------:|------:|
| aq1 | 95.6% | 94.1% | 99.5% | 99.5% | 5.4pp | 94.1% |
| aq2 | 96.9% | 93.9% | 99.2% | 99.3% | 5.4pp | 93.9% |
| aq3 | 96.6% | 94.1% | 99.5% | 99.6% | 5.5pp | 94.1% |
| aq4 | 97.2% | 94.0% | 99.5% | 99.4% | 5.5pp | 94.0% |

P99 coverage is generally healthy (93-100%). Even the worst PY at P99 stays above 90%.

---

## Red Flag Analysis

| Flag | Threshold | Status | Detail |
|------|-----------|--------|--------|
| q5 P95 under-cover | > 5pp below target | **TRIGGERED** | R1 aq4 q5=84.2% (-10.8pp), R2 aq1 q5=84.5% (-10.5pp) |
| Any PY P95 < 88% | < 88% | **TRIGGERED** | R2 aq1 PY2022=79.7%, R3 aq1 PY2022=82.0% |
| Coverage non-monotonicity | any violation | **CLEAR** | All levels monotonic across all cells |
| Class parity gap > 3pp | > 3pp | **CLEAR** | Max gap = 1.07pp (R1 aq1) |
| R2/R3 P95 > 3pp from target | > 3pp | **TRIGGERED** | R2 all quarters 4.0-5.4pp below target |

### Interpretation

1. **q5 under-coverage** is structural — the top-quintile bin has the heaviest tails and the quantile estimates at 2.5th/97.5th percentile are noisy. This affects ~20% of paths (the highest-value ones).

2. **PY 2021-2022 under-coverage** is the cold-start effect — these folds have only 1-2 training PYs. In production (2026+), we'll have 5+ training PYs, so this won't occur. PY 2023-2024 (3-4 training PYs) show 92-98% P95 coverage.

3. **R2/R3 systematic under-coverage at P50-P95** (4-7pp) is more concerning than R1. Despite having ~7x more data per cell, R2/R3 coverage is consistently below targets across all quarters. This suggests the R2/R3 residual distribution is less stable across PYs than R1's — the prior-round MCP baseline introduces regime-change sensitivity that the expanding window doesn't fully capture.

### Width Reduction vs Coverage Tradeoff

| Round | Width Reduction | P95 Undershoot | Acceptable? |
|-------|:-:|:-:|:---:|
| R1 | -13.9% | -1.7pp avg (aq1 +0.5pp, aq4 -5.0pp) | aq1-aq3 yes, aq4 marginal |
| R2 | -15.6% | -4.8pp avg | Marginal — early PYs drag this down |
| R3 | -12.3% | -3.1pp avg | Acceptable — PY2023+ hits 95%+ |

---

## What This Means for Production (2026+)

In production, we train on 6+ PYs (2020-2025 or later). The cold-start folds (PY 2021-2022 with 1-2 training PYs) that drag down coverage will not exist. Looking at only PY 2023-2024 folds (3-4 training PYs):

| Round | PY2023-2024 Avg P95 Cov | Avg P95 Width |
|-------|:-:|:-:|
| R1 | 95.6% (aq1-aq3), 92.4% (aq4) | 2,700 |
| R2 | 95.1% | 225 |
| R3 | 96.2% | 190 |

With 5+ training PYs, we'd expect P95 coverage to be 93-96% — well within tolerance.
