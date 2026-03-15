# Findings: Annual FTR Data Profile

**Source:** `notebooks/01_data_profile.ipynb` executed 2026-02-11
**Data:** MISO annual + f0p cleared trades, PY 2019-2025 (Jun Y to May Y+1)

---

## 1.1 Annual Cleared Trades Profile

**Total:** 11,498,880 rows across 7 PYs, 3 rounds, 4 quarters, 2 class types.

### 3x4 Grid: Total Rows (all PYs, both class_types)

| | aq1 | aq2 | aq3 | aq4 |
|---|---:|---:|---:|---:|
| **R1** | 620,520 | 631,326 | 572,433 | 569,295 |
| **R2** | 1,063,809 | 1,091,559 | 1,003,830 | 1,020,261 |
| **R3** | 1,255,008 | 1,279,842 | 1,191,363 | 1,199,634 |

### 3x4 Grid: Total MW Cleared

| | aq1 | aq2 | aq3 | aq4 |
|---|---:|---:|---:|---:|
| **R1** | 3,197,020 | 3,135,797 | 3,052,008 | 3,011,856 |
| **R2** | 4,374,713 | 4,418,327 | 4,245,969 | 4,180,664 |
| **R3** | 4,852,782 | 4,950,603 | 4,756,853 | 4,589,237 |

### 3x4 Grid: Avg MW per Row

| | aq1 | aq2 | aq3 | aq4 |
|---|---:|---:|---:|---:|
| **R1** | 5.15 | 4.97 | 5.33 | 5.29 |
| **R2** | 4.11 | 4.05 | 4.23 | 4.10 |
| **R3** | 3.87 | 3.87 | 3.99 | 3.83 |

**Observation:** R1 has the smallest number of trades but the largest average MW per trade. R3 has the most trades but smallest MW per trade. All 4 quarters are well-populated across all rounds and all PYs.

### Per-PY Summary

| PY | Rows | Unique Paths | Total MW | Rounds | Quarters |
|---|---:|---:|---:|---:|---:|
| 2019 | 1,160,565 | 58,836 | 5,014,170 | 3 | 4 |
| 2020 | 1,346,646 | 76,327 | 4,839,987 | 3 | 4 |
| 2021 | 1,690,560 | 88,404 | 5,849,924 | 3 | 4 |
| 2022 | 1,372,179 | 75,208 | 5,708,580 | 3 | 4 |
| 2023 | 1,990,440 | 116,008 | 7,780,376 | 3 | 4 |
| 2024 | 1,869,240 | 91,248 | 8,518,136 | 3 | 4 |
| 2025 | 2,069,250 | 115,559 | 11,054,656 | 3 | 4 |

**Trend:** Market participation growing — PY 2025 has ~2x the rows and MW of PY 2019.

---

## 1.2 f0p Cleared Trades Profile

**Total:** 12,711,961 rows across 7 PYs.

| PY | f0 rows | f1 rows | q4 rows | f0 total MW | f0 avg MW |
|---|---:|---:|---:|---:|---:|
| 2019 | 690,540 | 354,463 | 258,645 | 1,790,294 | 2.59 |
| 2020 | 777,305 | 472,583 | 390,633 | 1,813,794 | 2.33 |
| 2021 | 796,498 | 480,904 | 388,524 | 1,996,884 | 2.51 |
| 2022 | 785,101 | 491,446 | 409,869 | 2,327,910 | 2.97 |
| 2023 | 1,061,799 | 630,084 | 494,901 | 2,942,574 | 2.77 |
| 2024 | 1,105,316 | 650,856 | 510,279 | 3,478,605 | 3.15 |
| 2025 | 853,082 | 561,114 | 548,019 | 3,146,291 | 3.69 |

---

## 1.3 Annual vs f0p Volume Comparison

### Per-PY Comparison

| PY | Annual Total MW | f0p Total MW | Annual Avg MW/row | f0p Avg MW/row | MW Ratio | Total MW Ratio |
|---|---:|---:|---:|---:|---:|---:|
| 2019 | 5,014,170 | 3,211,136 | 4.32 | 2.46 | 1.75 | 1.56 |
| 2020 | 4,839,987 | 3,708,316 | 3.59 | 2.26 | 1.59 | 1.31 |
| 2021 | 5,849,924 | 4,142,542 | 3.46 | 2.49 | 1.39 | 1.41 |
| 2022 | 5,708,580 | 5,015,675 | 4.16 | 2.97 | 1.40 | 1.14 |
| 2023 | 7,780,376 | 6,205,509 | 3.91 | 2.84 | 1.38 | 1.25 |
| 2024 | 8,518,136 | 7,556,710 | 4.56 | 3.33 | 1.37 | 1.13 |
| 2025 | 11,054,656 | 7,702,849 | 5.34 | 3.93 | 1.36 | 1.44 |

**Overall averages:**
- Annual avg MW/row: **4.24**
- f0p avg MW/row: **2.95**
- Ratio: **1.44x** (annual trades are ~44% larger per row)

### Per Period x Class

| Period | Class | Mean MW | Median MW | Count |
|---|---|---:|---:|---:|
| aq1 | offpeak | 4.14 | 1.80 | 1,416,471 |
| aq1 | onpeak | 4.31 | 1.80 | 1,522,866 |
| aq2 | offpeak | 4.12 | 1.80 | 1,451,874 |
| aq2 | onpeak | 4.20 | 1.70 | 1,550,853 |
| aq3 | offpeak | 4.30 | 1.80 | 1,342,935 |
| aq3 | onpeak | 4.40 | 1.80 | 1,424,691 |
| aq4 | offpeak | 4.14 | 1.80 | 1,363,599 |
| aq4 | onpeak | 4.31 | 1.80 | 1,425,591 |
| f0 | offpeak | 2.85 | 1.10 | 2,967,219 |
| f0 | onpeak | 2.91 | 1.20 | 3,102,422 |
| f1 | offpeak | 2.89 | 1.20 | 1,781,977 |
| f1 | onpeak | 2.99 | 1.30 | 1,859,473 |
| q4 | offpeak | 3.08 | 1.40 | 1,464,774 |
| q4 | onpeak | 3.14 | 1.50 | 1,536,096 |

**Key finding:** Annual median MW is 1.80 across all quarters vs f0's 1.10-1.20. The mean is pulled up by large trades. The difference is moderate, not dramatic.

### MCP Distributions

| Round | Period | Mean MCP | Median MCP | Std |
|---|---|---:|---:|---:|
| R1 | aq1 | 418 | 124 | 2,057 |
| R1 | aq2 | 558 | 119 | 2,448 |
| R1 | aq3 | 452 | 109 | 2,139 |
| R1 | aq4 | 427 | 102 | 1,972 |
| R2 | aq1 | 392 | 108 | 2,138 |
| R2 | aq2 | 452 | 98 | 2,464 |
| R3 | aq1 | 386 | 105 | 2,150 |
| f0 | — | 101 | 20 | 923 |
| f1 | — | 141 | 35 | 834 |
| q4 | — | 442 | 104 | 2,101 |

**Key finding:** Annual MCPs are ~3-4x larger than f0 MCPs (median ~100+ vs ~20-35). Annual MCPs are comparable to q4 MCPs. This is consistent with quarterly scaling in the band generator (q4 gets 3x multiplier).

---

## 1.4 Submitted Bid Stacks

Not extracted in this pass. Submission notebooks exist at `/home/xyz/workspace/pmodel/notebook/hz/2025-planning-year/2025-26-annual/miso/submission/r1-r3/trades/` but require separate analysis to extract bid price spreads from notebook outputs.

---

## Assumption Verdicts from Data Profile

| # | Assumption | Verdict | Evidence |
|---|-----------|---------|----------|
| **A2** | Annual per-period volume >> f0p | **NUANCED** | Annual avg MW/row = 4.24 vs f0p = 2.95 (1.44x ratio). Moderate difference, not dramatic. Median MW is 1.80 vs 1.10 (1.6x). But total annual MW per PY is comparable to total f0p MW (ratio 1.1-1.6x depending on year). |
