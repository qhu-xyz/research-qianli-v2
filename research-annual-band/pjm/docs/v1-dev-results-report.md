# PJM V1 Dev Results Report

**Date:** 2026-03-16
**Method:** Asymmetric empirical quantile bands, 5 bins × 3 classes = 15 cells
**Baseline:** `mtm_1st_mean * 12` (annual) for all 4 rounds
**CV:** Temporal expanding, min_train_pys=2
**Dev PYs:** 2017-2024 (holdout: 2025)
**Scale:** Annual $ throughout

---

## COVERAGE ANALYSIS

### Table 1: Overall Coverage vs Target — All 8 Levels × 4 Rounds

| Round | P10 | P30 | P50 | P70 | P80 | P90 | P95 | P99 |
|-------|----:|----:|----:|----:|----:|----:|----:|----:|
| R1 | 12.2 | 36.8 | 59.3 | 78.4 | 86.4 | 93.8 | **97.1** | 99.5 |
| R2 | 11.1 | 33.4 | 54.3 | 73.8 | 82.6 | 91.3 | **95.7** | 99.1 |
| R3 | 10.8 | 32.5 | 53.2 | 73.2 | 82.4 | 91.6 | **95.9** | 99.2 |
| R4 | 10.8 | 31.4 | 52.0 | 72.1 | 81.2 | 90.2 | **94.9** | 98.9 |

**Error vs target (pp):**

| Round | P10 | P30 | P50 | P70 | P80 | P90 | P95 | P99 |
|-------|----:|----:|----:|----:|----:|----:|----:|----:|
| R1 | +2.2 | +6.8 | **+9.3** | +8.4 | +6.4 | +3.8 | +2.1 | +0.5 |
| R2 | +1.1 | +3.4 | +4.3 | +3.8 | +2.6 | +1.3 | +0.7 | +0.1 |
| R3 | +0.8 | +2.5 | +3.2 | +3.2 | +2.4 | +1.6 | +0.9 | +0.2 |
| R4 | +0.8 | +1.4 | +2.0 | +2.1 | +1.2 | +0.2 | -0.1 | -0.1 |

R1 over-covers significantly (P50 error = +9.3pp). R2-R4 are closer to target. R4 P95 is exactly on target.

### Table 2: Per-Bin P95 Coverage — 5 Bins × 4 Rounds

| Round | q1 | q2 | q3 | q4 | q5 | **Weakest** |
|-------|---:|---:|---:|---:|---:|:---:|
| R1 | 96.6 | 96.7 | 97.3 | 97.7 | 97.3 | q1 (96.6) |
| R2 | 96.6 | 96.4 | 96.0 | 95.6 | 93.8 | q5 (93.8) |
| R3 | 96.8 | 96.4 | 96.2 | 95.2 | 94.9 | q5 (94.9) |
| R4 | 96.5 | 95.8 | 95.1 | 94.5 | **92.7** | **q5 (92.7)** |

R4 q5 = 92.7% (flag: < 93% but above 90% threshold). R1 q5 is actually BETTER than q1 — unusual, likely because R1's wide bands over-cover everywhere.

### Table 3: Per-PY P95 Coverage — 4 Rounds

| PY | R1 | R2 | R3 | R4 |
|----|---:|---:|---:|---:|
| 2019 | 96.4 | 96.2 | 94.4 | 93.0 |
| 2020 | 96.2 | 95.9 | 96.9 | 95.2 |
| 2021 | 98.8 | 97.2 | 97.2 | 93.8 |
| **2022** | **94.0** | **88.5** | **86.2** | **83.0** |
| 2023 | 97.6 | 95.5 | 97.0 | 96.5 |
| 2024 | 97.4 | 96.7 | 97.1 | 96.9 |
| **Worst** | **94.0** | **88.5** | **86.2** | **83.0** |
| **Range** | **4.8pp** | **8.7pp** | **11.0pp** | **13.9pp** |

**PY 2022 is catastrophic** — R4 drops to 83.0% (flag: < 88%). This is the same regime shift year seen in MISO.

### Table 4: Per-Class P95 Coverage — 3 Classes × 4 Rounds

| Round | onpeak | dailyoffpeak | wkndonpeak | **Weakest** | **Gap** |
|-------|-------:|------------:|-----------:|:---:|:---:|
| R1 | 96.6 | 97.7 | 98.1 | onpeak | 1.5pp |
| R2 | 94.6 | 97.8 | 97.2 | onpeak | 3.2pp |
| R3 | 94.7 | 97.9 | 98.0 | onpeak | 3.3pp |
| R4 | **93.2** | 97.9 | 97.5 | **onpeak** | **4.7pp** |

Onpeak is consistently weakest. R4 onpeak = 93.2% (above 90% threshold but gap is 4.7pp).

Note: dailyoffpeak and wkndonpeak only testable for PY 2023-2024 (2 folds) due to R1 onpeak-only history.

### Table 5: Per-PY × Per-Bin P95 (Full Grid, Flag <85%)

#### R1

| PY | q1 | q2 | q3 | q4 | q5 |
|----|---:|---:|---:|---:|---:|
| 2019 | 96 | 96 | 96 | 96 | 97 |
| 2020 | 96 | 97 | 96 | 97 | 95 |
| 2021 | 98 | 99 | 99 | 98 | 100 |
| 2022 | 91 | 90 | 92 | 96 | 98 |
| 2023 | 98 | 97 | 98 | 98 | 96 |
| 2024 | 96 | 96 | 99 | 100 | 98 |

No cells below 85%. PY 2022 q2=90% is the weakest.

#### R2

| PY | q1 | q2 | q3 | q4 | q5 |
|----|---:|---:|---:|---:|---:|
| 2019 | 93 | 97 | 96 | 97 | 97 |
| 2020 | 98 | 97 | 96 | 96 | 92 |
| 2021 | 98 | 97 | 98 | 99 | 94 |
| **2022** | 92 | 90 | 88 | **87** | **87** |
| 2023 | 97 | 96 | 96 | 92 | 94 |
| 2024 | 96 | 97 | 97 | 97 | 97 |

PY 2022 q4=87%, q5=87% (flag: < 88%).

#### R3

| PY | q1 | q2 | q3 | q4 | q5 |
|----|---:|---:|---:|---:|---:|
| 2019 | 94 | 92 | 96 | 93 | 96 |
| 2020 | 98 | 97 | 97 | 96 | 96 |
| 2021 | 97 | 98 | 97 | 98 | 97 |
| **2022** | 88 | **85** | **86** | **86** | **86** |
| 2023 | 99 | 98 | 97 | 94 | 95 |
| 2024 | 96 | 97 | 97 | 98 | 98 |

PY 2022 q2=85% (flag: exactly at threshold). q3/q4/q5 = 86%.

#### R4

| PY | q1 | q2 | q3 | q4 | q5 |
|----|---:|---:|---:|---:|---:|
| 2019 | 95 | 92 | 92 | 91 | 95 |
| 2020 | 97 | 95 | 96 | 95 | 93 |
| 2021 | 96 | 95 | 95 | 92 | 91 |
| **2022** | 88 | 88 | **86** | **85** | **74** |
| 2023 | 98 | 98 | 96 | 95 | 92 |
| 2024 | 96 | 96 | 96 | 97 | 99 |

**PY 2022 R4 q5 = 74%** — the worst cell in the entire grid. Flag: far below 85%.

---

## WIDTH ANALYSIS

### Table 6: Width at All Levels — 4 Rounds (Annual $)

| Round | P10 | P30 | P50 | P70 | P80 | P90 | P95 | P99 |
|-------|----:|----:|----:|----:|----:|----:|----:|----:|
| R1 | 137 | 436 | 803 | 1,308 | 1,683 | 2,390 | **3,226** | 5,573 |
| R2 | 36 | 114 | 206 | 336 | 439 | 635 | **870** | 1,586 |
| R3 | 27 | 88 | 164 | 278 | 368 | 541 | **745** | 1,500 |
| R4 | 22 | 71 | 132 | 226 | 302 | 439 | **595** | 1,215 |

### Table 7: Per-PY P95 Width — Trend

| PY | R1 | R2 | R3 | R4 |
|----|---:|---:|---:|---:|
| 2019 | 3,613 | 927 | 767 | 540 |
| 2020 | 3,475 | 881 | 785 | 574 |
| 2021 | 3,446 | 866 | 744 | 572 |
| 2022 | 3,220 | 853 | 721 | 587 |
| 2023 | 3,245 | 947 | 826 | 714 |
| 2024 | 2,357 | 744 | 628 | 583 |

R1 width is shrinking over time: 3,613 (2019) → 2,357 (2024) = -35%. This matches the improving R1 MAE. R2-R4 widths are more stable.

### Table 8: Per-Class P95 Width

Not available from current metrics structure (width is aggregated across classes). Would need per-class width extraction from the per-PY data.

---

## COMBINED ANALYSIS

### Table 10: Coverage-Width Tradeoff Summary

| Round | Avg P95 Cov | Avg P95 Width | Worst Coverage Cell | Widest PY |
|-------|:-----------:|-------------:|---------------------|-----------|
| R1 | **97.1%** | **3,226** | PY2022 q2=90% | PY2019 (3,613) |
| R2 | **95.7%** | **870** | PY2022 q4=87% | PY2023 (947) |
| R3 | **95.9%** | **745** | PY2022 q2=85% | PY2023 (826) |
| R4 | **94.9%** | **595** | **PY2022 q5=74%** | PY2023 (714) |

R1 over-covers (+2.1pp at P95) with wide bands. R4 under-covers (-0.1pp) with the tightest bands.
PY 2022 R4 q5=74% is the single worst cell — both low coverage AND high-value paths.

### Table 11: PJM vs MISO Comparison (Directional Only)

Different RTOs, path universes, and settlement windows. MISO values are quarterly.

| Metric | PJM R1 | MISO R1 | PJM R2 | MISO R2 | PJM R4 | MISO R3 |
|--------|-------:|--------:|-------:|--------:|-------:|--------:|
| P95 Cov | 97.1% | 93.2% | 95.7% | 89.6% | 94.9% | 92.0% |
| P95 HW | 3,226 | 2,577 | 870 | 541 | 595 | 437 |
| Baseline MAE | 788 | 792 | 293 | 532 | 209 | 457 |

PJM has higher coverage but wider bands (partly because annual settlement = 12 months vs quarterly = 3 months).

---

## RED FLAG SUMMARY

| # | Flag | Round | PY | Bin/Class | Value | Severity |
|---|------|-------|:--:|-----------|------:|:---:|
| 1 | **P95 < 85%** | R4 | 2022 | q5 | **74%** | **CRITICAL** |
| 2 | P95 < 85% | R4 | 2022 | q4 | 85% | HIGH |
| 3 | P95 < 85% | R3 | 2022 | q2 | 85% | HIGH |
| 4 | P95 < 88% | R4 | 2022 | q3 | 86% | MEDIUM |
| 5 | P95 < 88% | R3 | 2022 | q3/q4/q5 | 86% | MEDIUM |
| 6 | P95 < 88% | R2 | 2022 | q4/q5 | 87% | MEDIUM |
| 7 | P95 < 90% | R2 | 2022 | overall | 88.5% | MEDIUM |
| 8 | R1 P50 over-cover | R1 | all | overall | +9.3pp | NOTE |

**Pattern:** ALL red flags are PY 2022. This was an extreme volatility year.
Excluding PY 2022, every cell in every round is above 90%.

**R1 over-coverage:** P50 error = +9.3pp means R1 bands are too wide at the median.
The R1 LT yr1 R5 baseline has higher MAE, so bands must be wider to reach 95% target.
The over-coverage at P50 is a consequence — not a bug.
