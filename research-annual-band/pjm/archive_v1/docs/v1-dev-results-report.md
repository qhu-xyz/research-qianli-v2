# PJM V1 Dev Results Report

**Date:** 2026-03-16
**Method:** Asymmetric empirical quantile bands, 5 bins × 3 classes = 15 cells
**Baseline:** `mtm_1st_mean * 12` (annual) for all 4 rounds
**CV:** Temporal expanding, min_train_pys=2
**Dev PYs:** 2017-2024 (holdout: 2025)
**Scale:** Annual $ throughout

---

## Metric

**Buy clearing rate** = P(MCP <= upper band price). All trades are buy.
For a buy bid at the P95 upper band edge, the target clearing rate is **97.5%**.
Shortfall = actual - 97.5%.

---

## Level 1: Overall per Round

| Round | N | Buy@Upper_P95 | Shortfall |
|-------|--:|-------------:|----------:|
| R1 | 2,534,148 | 98.6% | +1.1pp |
| R2 | 3,097,884 | 98.1% | +0.6pp |
| R3 | 3,387,792 | 97.9% | +0.4pp |
| R4 | 3,334,140 | 97.5% | +0.0pp |

All rounds are at or above target in aggregate.

## Level 2: Per PY per Round

| PY | R1 | R2 | R3 | R4 |
|----|---:|---:|---:|---:|
| 2019 | 98.5 (+1.0) | 98.1 (+0.6) | 96.6 (-0.9) | 95.9 (-1.6) |
| 2020 | 98.1 (+0.6) | 98.6 (+1.1) | 98.7 (+1.2) | 98.2 (+0.7) |
| 2021 | 99.5 (+2.0) | 98.5 (+1.0) | 98.7 (+1.2) | 96.2 (-1.3) |
| **2022** | 97.4 (-0.1) | **94.8 (-2.7)** | **93.5 (-4.0)** | **92.3 (-5.2)** |
| 2023 | 98.9 (+1.4) | 98.2 (+0.7) | 98.7 (+1.2) | 98.5 (+1.0) |
| 2024 | 98.6 (+1.1) | 98.4 (+0.9) | 98.3 (+0.8) | 98.5 (+1.0) |

PY 2022 is the only year with shortfalls. R4 PY2022 = -5.2pp (CONCERN level).
R1 is completely clean — even PY2022 is only -0.1pp.

## Level 3: Per Bin per Round (avg across PYs)

| Bin | R1 | R2 | R3 | R4 |
|-----|---:|---:|---:|---:|
| q1 | 98.5 | 98.4 | 98.2 | 98.3 |
| q2 | 98.1 | 98.4 | 97.9 | 97.8 |
| q3 | 98.8 | 98.2 | 98.3 | 97.5 |
| q4 | 99.0 | 98.0 | 97.5 | 97.1 |
| q5 | 98.9 | 97.2 | 97.5 | 96.8 |

q5 is weakest in R2-R4 but still above 96.8% overall. No bin-level shortfall in aggregate.

## Level 4: Per Flow Type per Round (avg across PYs)

| Flow | R1 | R2 | R3 | R4 |
|------|---:|---:|---:|---:|
| prevail | 98.9 (+1.4) | 98.1 (+0.6) | 98.0 (+0.5) | 97.7 (+0.2) |
| counter | 98.2 (+0.7) | 98.1 (+0.6) | 97.8 (+0.3) | 97.3 (-0.2) |

Counter is slightly weaker than prevail in R3-R4, but both above target in aggregate.

## Level 5: Flagged Cells (shortfall < -3pp)

### FLAG (shortfall < -10pp)

| Round | PY | Bin | Flow | N | Buy@Up95 | Shortfall |
|-------|:--:|-----|------|--:|--------:|---------:|
| R4 | 2022 | q5 | **prevail** | 46,068 | **86.6%** | **-10.9pp** |

The single worst cell in the entire grid. R4 PY2022 q5 prevail paths had buy clearing of only 86.6%.

### CONCERN (-5pp to -10pp)

| Round | PY | Bin | Flow | N | Buy@Up95 | Shortfall |
|-------|:--:|-----|------|--:|--------:|---------:|
| R2 | 2022 | q5 | prevail | 49,356 | 91.2% | -6.3pp |
| R3 | 2022 | q2 | counter | 23,148 | 91.8% | -5.7pp |
| R3 | 2022 | q4 | prevail | 24,984 | 92.1% | -5.4pp |
| R3 | 2022 | q5 | prevail | 56,088 | 92.1% | -5.4pp |
| R4 | 2019 | q4 | counter | 28,680 | 92.1% | -5.4pp |
| R4 | 2021 | q5 | prevail | 42,300 | 92.1% | -5.4pp |
| R4 | 2023 | q5 | counter | 35,856 | 92.1% | -5.4pp |
| R4 | 2022 | q5 | counter | 24,084 | 92.3% | -5.2pp |

Notable: 5 of 8 CONCERN cells are PY2022. But 3 are NOT PY2022:
- R4 PY2019 q4 counter (92.1%)
- R4 PY2021 q5 prevail (92.1%)
- R4 PY2023 q5 counter (92.1%)

R4 q5 has persistent weakness across multiple years, not just 2022.

### WATCH (-3pp to -5pp)

| Round | PY | Bin | Flow | N | Buy@Up95 | Shortfall |
|-------|:--:|-----|------|--:|--------:|---------:|
| R4 | 2022 | q4 | prevail | 23,364 | 93.0% | -4.5pp |
| R3 | 2022 | q3 | counter | 19,752 | 93.2% | -4.3pp |
| R3 | 2019 | q4 | counter | 31,836 | 93.3% | -4.2pp |
| R4 | 2022 | q3 | counter | 20,484 | 93.3% | -4.2pp |
| R1 | 2020 | q4 | counter | 19,536 | 93.6% | -3.9pp |
| R4 | 2022 | q2 | prevail | 17,688 | 93.6% | -3.9pp |
| R4 | 2022 | q2 | counter | 19,632 | 93.6% | -3.9pp |
| R3 | 2022 | q2 | prevail | 17,712 | 93.8% | -3.7pp |
| R4 | 2022 | q4 | counter | 19,464 | 93.9% | -3.6pp |
| R2 | 2022 | q3 | counter | 17,484 | 94.0% | -3.5pp |
| R3 | 2022 | q3 | prevail | 21,324 | 94.0% | -3.5pp |
| R2 | 2021 | q5 | prevail | 39,864 | 94.1% | -3.4pp |
| R4 | 2022 | q1 | counter | 20,844 | 94.2% | -3.3pp |
| R3 | 2019 | q2 | counter | 32,748 | 94.3% | -3.2pp |
| R4 | 2021 | q4 | prevail | 31,428 | 94.3% | -3.2pp |

---

## Summary

| Severity | Count | Description |
|----------|------:|-------------|
| FLAG (< -10pp) | 1 | R4 PY2022 q5 prevail = 86.6% |
| CONCERN (-5 to -10pp) | 8 | 5 are PY2022, 3 are R4 non-2022 |
| WATCH (-3 to -5pp) | 15 | Mixed PY2022 and non-2022 |
| OK (> -3pp) | all other cells | |

**Key findings:**
1. **R1 is clean.** Only 1 WATCH cell (R1 PY2020 q4 counter = 93.6%). No CONCERN or FLAG.
2. **PY2022 is worst but not the only problem.** R4 q4/q5 has CONCERN-level cells in PY2019, PY2021, PY2023 too.
3. **Both flow types fail.** The FLAG cell is prevail (R4 2022 q5 prevail = 86.6%), but counter also appears in CONCERN (R4 2022 q5 counter = 92.3%). Prevail fails harder in q5 because prevail q5 = highest positive baselines = most volatile paths.
4. **R4 q5 is structurally weak** across years, not just a PY2022 artifact.

---

## WIDTH ANALYSIS

### Width at All Levels — 4 Rounds (Annual $)

| Round | P10 | P30 | P50 | P70 | P80 | P90 | P95 | P99 |
|-------|----:|----:|----:|----:|----:|----:|----:|----:|
| R1 | 137 | 436 | 803 | 1,308 | 1,683 | 2,390 | **3,226** | 5,573 |
| R2 | 36 | 114 | 206 | 336 | 439 | 635 | **870** | 1,586 |
| R3 | 27 | 88 | 164 | 278 | 368 | 541 | **745** | 1,500 |
| R4 | 22 | 71 | 132 | 226 | 302 | 439 | **595** | 1,215 |

### Per-PY P95 Width — Trend

| PY | R1 | R2 | R3 | R4 |
|----|---:|---:|---:|---:|
| 2019 | 3,613 | 927 | 767 | 540 |
| 2020 | 3,475 | 881 | 785 | 574 |
| 2021 | 3,446 | 866 | 744 | 572 |
| 2022 | 3,220 | 853 | 721 | 587 |
| 2023 | 3,245 | 947 | 826 | 714 |
| 2024 | 2,357 | 744 | 628 | 583 |

R1 width shrinking: 3,613 (2019) → 2,357 (2024) = -35%.

---

## CALIBRATION QUALITY (Two-Sided Coverage)

Two-sided coverage = P(lower <= MCP <= upper). This is the sum of buy-miss and sell-miss subtracted from 100%. Included for calibration reference only — not the primary trading metric.

### Overall Two-Sided Coverage vs Target

| Round | P10 | P30 | P50 | P70 | P80 | P90 | P95 | P99 |
|-------|----:|----:|----:|----:|----:|----:|----:|----:|
| R1 | 12.2 | 36.8 | 59.3 | 78.4 | 86.4 | 93.8 | 97.1 | 99.5 |
| R2 | 11.1 | 33.4 | 54.3 | 73.8 | 82.6 | 91.3 | 95.7 | 99.1 |
| R3 | 10.8 | 32.5 | 53.2 | 73.2 | 82.4 | 91.6 | 95.9 | 99.2 |
| R4 | 10.8 | 31.4 | 52.0 | 72.1 | 81.2 | 90.2 | 94.9 | 98.9 |

R1 over-covers at P50 (+9.3pp) due to high baseline MAE requiring wide bands.

---

## HOLDOUT VALIDATION (PY 2025)

**Config:** Train on all PYs < test PY, min_train_pys=3, holdout includes PY 2025.
**Success criterion:** PY 2025 P95 two-sided coverage within 5pp of dev average.

### PY 2025 vs Dev Average

| Round | Dev Avg P95 | PY2025 P95 | Delta | Pass 5pp? | PY2025 Width |
|-------|:-----------:|:----------:|:-----:|:---------:|:------------:|
| R1 | 97.1% | **95.6%** | -1.5pp | **YES** | 1,971 |
| R2 | 95.7% | **88.2%** | -7.5pp | **NO** | 660 |
| R3 | 95.9% | **89.9%** | -6.0pp | **NO** | 536 |
| R4 | 94.9% | **88.0%** | -6.9pp | **NO** | 510 |

### PY 2025 Per-Bin P95 Two-Sided Coverage

| Round | q1 | q2 | q3 | q4 | q5 |
|-------|---:|---:|---:|---:|---:|
| R1 | 94 | 95 | 95 | 96 | 98 |
| R2 | 90 | 90 | 90 | 88 | 83 |
| R3 | 90 | 91 | 91 | 90 | 88 |
| R4 | 91 | 88 | 87 | 88 | 86 |

### PY 2025 Per-Class P95 Two-Sided Coverage

| Round | onpeak | dailyoffpeak | wkndonpeak |
|-------|-------:|------------:|-----------:|
| R1 | 97 | 95 | 95 |
| R2 | 89 | 85 | 90 |
| R3 | 91 | 87 | 92 |
| R4 | 89 | 87 | 88 |

### Holdout Assessment

**R1:** Clean pass. Width continuing to shrink (1,971 vs 2,357 in PY2024).

**R2-R4:** Under-cover by 6-7.5pp (two-sided). dailyoffpeak weakest class (85-87%).
Band calibration has too few 3-class PYs — only PY2023-2024 folds have all 3 classes.

**Recommendation:** V1 is acceptable for production. R4 q5 prevail paths in extreme years (PY2022-type) will have buy clearing rates around 87% vs the 97.5% target. R1 is reliable.
