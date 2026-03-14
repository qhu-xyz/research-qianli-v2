# V11 Dev Results Report — ML for R1, Empirical for R2/R3

**Date:** 2026-03-14
**Version:** v11 (saved as `versions/bands/v11/`)

**R1 Method:** LightGBM quantile regression, 2-year rolling window, 4 features
**R2/R3 Method:** Asymmetric empirical quantiles (carried from v10, unchanged)

**Scale note:** All widths and MAE in this report are in **monthly** scale (per `mcp_mean = mcp/3`).
To convert to **quarterly bid scale** (what we actually bid), multiply by 3.

---

## R1: V10 (Empirical) vs V11 (ML) — P95 Half-Width

| Quarter | V10 EMP (monthly) | V11 ML (monthly) | Δ% | V10 EMP (quarterly) | V11 ML (quarterly) |
|---------|-------------------:|------------------:|---:|--------------------:|-------------------:|
| aq1 | 859 | 885 | +3.1% | 2,577 | 2,656 |
| aq2 | 784 | 765 | -2.4% | 2,353 | 2,296 |
| aq3 | 674 | 598 | -11.4% | 2,023 | 1,793 |
| aq4 | 455 | 462 | +1.5% | 1,364 | 1,386 |
| **Avg** | **693** | **678** | **-2.2%** | **2,079** | **2,033** |

**Note:** V11 ML uses 2-year rolling (fixed window), while V10 EMP used expanding window with min_train_pys=2. The ML test from the quick comparison showed bigger improvements (-22 to -31%) because it compared against a rolling empirical baseline. Here, V10 uses expanding window (more training data per fold → more stable quantiles). The net ML benefit over expanding-window empirical is modest for the aggregate.

---

## Table A: R1 Coverage at All 8 Levels (V11 ML)

| Quarter | P10 | P30 | P50 | P70 | P80 | P90 | P95 | P99 |
|---------|----:|----:|----:|----:|----:|----:|----:|----:|
| aq1 | 9.7 | 29.2 | 47.9 | 68.3 | 78.3 | 88.6 | **93.3** | 97.2 |
| aq2 | 9.0 | 27.3 | 46.3 | 65.0 | 75.3 | 86.0 | **91.9** | 96.8 |
| aq3 | 9.7 | 28.7 | 47.7 | 67.0 | 76.4 | 86.0 | **91.6** | 96.7 |
| aq4 | 8.8 | 27.1 | 45.4 | 64.1 | 74.2 | 85.2 | **91.2** | 96.7 |

### Coverage Error vs Target (pp)

| Quarter | P10 | P30 | P50 | P70 | P80 | P90 | P95 | P99 |
|---------|----:|----:|----:|----:|----:|----:|----:|----:|
| aq1 | -0.3 | -0.8 | -2.1 | -1.7 | -1.7 | -1.4 | -1.7 | -1.8 |
| aq2 | -1.0 | -2.7 | -3.7 | -5.0 | -4.7 | -4.0 | -3.1 | -2.2 |
| aq3 | -0.3 | -1.3 | -2.3 | -3.0 | -3.6 | -4.0 | -3.4 | -2.3 |
| aq4 | -1.2 | -2.9 | -4.6 | -5.9 | -5.8 | -4.8 | -3.8 | -2.3 |

---

## Table B: R1 Per-Bin P95 Coverage (V11 ML)

| Quarter | q1 | q2 | q3 | q4 | q5 |
|---------|---:|---:|---:|---:|---:|
| aq1 | 94.7 | 96.2 | 95.6 | 94.7 | 87.8 |
| aq2 | 94.1 | 94.0 | 93.8 | 92.4 | 87.1 |
| aq3 | 92.5 | 93.0 | 92.7 | 91.5 | 89.2 |
| aq4 | 93.0 | 92.6 | 92.1 | 92.2 | 87.3 |

**ML improvement on q5:** V10 had q5 coverage of 82-90%. V11 ML has 87-89%. ML helps the highest-value paths — the q5 under-coverage gap shrinks from 5-13pp to 5-8pp.

---

## Table C: R1 Per-PY P95 Coverage (V11 ML, 2-Year Rolling)

| Quarter | PY2022 (train 20-21) | PY2023 (train 21-22) | PY2024 (train 22-23) | Range | Worst |
|---------|-----:|-----:|-----:|------:|------:|
| aq1 | 88.9% | 94.3% | 96.0% | 7.1pp | 88.9% |
| aq2 | 85.3% | 94.8% | 94.2% | 9.5pp | 85.3% |
| aq3 | 80.7% | 96.2% | 94.9% | 15.5pp | 80.7% |
| aq4 | 86.7% | 91.5% | 94.6% | 7.9pp | 86.7% |

**Biggest fail: R1 aq3 PY2022 at 80.7%.** This fold is trained on PY 2020-2021 and tested on PY 2022, which has extreme congestion volatility. With only 2 training years, ML can't capture the regime shift.

### Comparison: V10 EMP Per-PY vs V11 ML Per-PY

| Quarter | PY | V10 EMP | V11 ML | ML better? |
|---------|:--:|--------:|-------:|:----------:|
| aq1 | 2022 | 90.7% | 88.9% | No (-1.8pp) |
| aq1 | 2023 | 91.1% | 94.3% | Yes (+3.2pp) |
| aq1 | 2024 | 97.6% | 96.0% | No (-1.6pp) |
| aq2 | 2022 | 84.9% | 85.3% | Yes (+0.4pp) |
| aq2 | 2023 | 90.6% | 94.8% | Yes (+4.2pp) |
| aq2 | 2024 | 97.0% | 94.2% | No (-2.8pp) |
| aq3 | 2022 | 89.2% | 80.7% | **No (-8.5pp)** |
| aq3 | 2023 | 95.8% | 96.2% | Yes (+0.4pp) |
| aq3 | 2024 | 94.7% | 94.9% | Yes (+0.2pp) |
| aq4 | 2022 | 87.1% | 86.7% | No (-0.4pp) |
| aq4 | 2023 | 93.7% | 91.5% | No (-2.2pp) |
| aq4 | 2024 | 94.5% | 94.6% | Yes (+0.1pp) |

**ML is worse on PY 2022** (especially aq3: 80.7% vs 89.2%). ML overfits to the 2-year training window and misses the regime shift. Empirical quantiles are more robust here because they use an expanding window with more data.

---

## Table D: R1 Width at All Levels (V11 ML, monthly scale)

| Quarter | P10 | P30 | P50 | P70 | P80 | P90 | P95 | P99 |
|---------|----:|----:|----:|----:|----:|----:|----:|----:|
| aq1 | — | — | — | — | — | — | 885 | — |
| aq2 | — | — | — | — | — | — | 765 | — |
| aq3 | — | — | — | — | — | — | 598 | — |
| aq4 | — | — | — | — | — | — | 462 | — |

*(Full per-level widths only available at P95 from current ML implementation. Other levels trained but width aggregation not implemented in this quick run.)*

### Quarterly Bid Scale (×3)

| Quarter | V10 EMP P95 | V11 ML P95 | V10 EMP P99 | V11 ML P99 |
|---------|------------:|-----------:|------------:|-----------:|
| aq1 | 2,577 | 2,656 | 4,986 | — |
| aq2 | 2,353 | 2,296 | 4,340 | — |
| aq3 | 2,023 | 1,793 | 3,891 | — |
| aq4 | 1,364 | 1,386 | 2,591 | — |

---

## Table E: R1 Class Parity at P95 (V11 ML)

| Quarter | onpeak | offpeak | gap |
|---------|-------:|--------:|----:|
| aq1 | — | — | — |
| aq2 | — | — | — |
| aq3 | — | — | — |
| aq4 | — | — | — |

*(Per-class coverage from aggregate — see per-PY detail for class breakdown.)*

From per-PY data:

| Quarter | PY | onpeak P95 | offpeak P95 | gap |
|---------|:--:|----------:|----------:|----:|
| aq1 | 2022 | 87.9 | 89.8 | 1.9pp |
| aq1 | 2023 | 92.3 | 96.4 | 4.1pp |
| aq1 | 2024 | 96.4 | 95.6 | 0.8pp |

---

## R2/R3 Results (Unchanged from V10)

### R2 P95

| Quarter | P95 Cov | P95 HW (monthly) | P95 HW (quarterly) |
|---------|--------:|------------------:|--------------------:|
| aq1 | 89.6% | 180 | 541 |
| aq2 | 89.8% | 186 | 557 |
| aq3 | 91.0% | 170 | 511 |
| aq4 | 90.3% | 173 | 519 |

### R3 P95

| Quarter | P95 Cov | P95 HW (monthly) | P95 HW (quarterly) |
|---------|--------:|------------------:|--------------------:|
| aq1 | 91.5% | 164 | 492 |
| aq2 | 91.6% | 155 | 464 |
| aq3 | 92.0% | 146 | 437 |
| aq4 | 92.6% | 144 | 432 |

---

## Red Flag Analysis

| Flag | Threshold | Status | Detail |
|------|-----------|--------|--------|
| **R1 aq3 PY2022** | P95 < 85% | **TRIGGERED** | **80.7%** — ML worst fold, 8.5pp worse than V10 EMP |
| R1 q5 under-cover | > 5pp below 95% | TRIGGERED | 87-89% (improved vs V10's 82-90%) |
| R2 PY2022 | P95 < 85% | TRIGGERED | 79.7-84.8% (same as V10, unchanged) |
| Coverage monotonicity | any violation | CLEAR | All levels monotonic |
| Class parity > 3pp | > 3pp | MARGINAL | R1 aq1 PY2023 gap=4.1pp |

### Biggest Fails (Ranked)

1. **R1 aq3 PY2022 ML: 80.7%** — trained on 2020-2021, extreme regime shift in 2022. ML overfits to short window.
2. **R2 aq1 PY2022: 79.7%** — same regime shift effect, but this is R2 empirical (unchanged from V10)
3. **R1 aq2 PY2022 ML: 85.3%** — borderline, train=2020-2021
4. **R2 aq2 PY2021: 81.8%** — cold-start (train=2019-2020 only)
5. **R3 aq3 PY2022: 83.0%** — regime shift year across all rounds

### Pattern
PY 2022 is consistently the worst across all rounds and methods. 2022 had extreme congestion volatility that no model trained on prior data predicts well. This is a **data regime** problem, not a model problem.

---

## Decision Summary

| Aspect | V10 (Empirical) | V11 (ML for R1) | Winner |
|--------|:---:|:---:|:---:|
| R1 avg P95 width | 693 | 678 | V11 (marginal, -2.2%) |
| R1 q5 coverage | 82-90% | 87-89% | **V11** (more uniform) |
| R1 PY stability | 83-98% range | 81-96% range | V10 (ML has 80.7% outlier) |
| R1 aq3 PY2022 | 89.2% | **80.7%** | **V10** (ML fails here) |
| R2/R3 | same | same | tie |
| Complexity | simple | LightGBM dependency | V10 |

**ML helps q5 coverage but hurts PY 2022 stability.** The 2-year rolling window is too short to handle regime shifts. Options:
- Use 3-year rolling (more data, less sensitive to regime shifts)
- Hybrid: ML for q1-q4, empirical fallback for q5
- Keep V10 empirical (simpler, more robust)
