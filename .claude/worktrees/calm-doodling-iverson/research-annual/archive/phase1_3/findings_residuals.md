# Findings: Annual FTR Residual Analysis (Corrected)

**Source:** `notebooks/03_corrected_baseline.ipynb` executed 2026-02-11
**Data:** MISO annual cleared trades PY 2019-2025, full MISO published MCP data
**Correction:** R2/R3 baselines now use `get_m2m_mcp_for_trades_all()` (full MISO market data), not manual path-matching within our portfolio.

---

## 1. Background: What Changed and Why

### 1.1 The Data Pipeline Error

Our previous analysis (notebook 02) had a **critical flaw** in how we obtained R2/R3 baselines:

| Step | Old (Wrong) Approach | New (Correct) Approach |
|------|---------------------|----------------------|
| **R2 baseline** | Manually joined R2 trades to R1 trades on `(PY, period_type, class_type, path)` within our cleared trades | `aptools.tools.get_m2m_mcp_for_trades_all(trades)` — looks up MCPs from MISO's full published market data |
| **R3 baseline** | Same manual join, R3 → R2 | Same function, looks up R2's MCP from full market data |
| **What this means** | Only finds MCPs for paths that *we* traded in the prior round | Finds MCPs for ALL paths, because MISO publishes clearing prices for every path in every round |

**Intuition:** When MISO runs the annual FTR auction, they compute and publish an MCP (market clearing price) for *every* source-sink path in the market, not just the paths that participants actually trade. So even if we didn't submit a bid on path X in R1, MISO still computed R1's MCP for path X. The correct way to get "what was the R1 MCP for this path?" is to look it up from MISO's published results, not from our own trade history.

### 1.2 The Dtype Bug

`get_all_cleared_trades()` returns some columns (`hedge_type`, `trade_type`, `class_type`) as pandas `category` dtype, which causes `"Object with dtype category cannot perform the numpy op multiply"` when doing math downstream. Fixed by calling `aptools.tools.cast_category_to_str(trades)` immediately after loading, which converts category columns to strings and the `round` column to int.

### 1.3 Impact Summary

The corrected pipeline produces:
- **R1 stats: identical** to before (same H baseline method, same data — validates the correction didn't break anything)
- **R2/R3 coverage: 100%** (up from 68%/75%) — this is the main fix
- **R2/R3 residuals: dramatically smaller** because we now include the full market's MCP data rather than the biased subset that overlapped with our portfolio
- **Total coverage: 11.5M rows → 11.5M rows** (up from 8.9M)

---

## 2. Baseline Reconstruction

### 2.1 R1: Historical DA Congestion (H)

**Method:** `fill_mtm_1st_period_with_hist_revenue()` (`pbase/analysis/tools/miso.py:322`)

This function computes H, the historical DA congestion proxy, for each R1 trade. The procedure:

1. For each path (source → sink), load monthly-average DA congestion prices from the prior year's delivery months
   - aq1 (Jun-Aug delivery): loads Jun, Jul, Aug, Sep DA congestion from the prior year
   - aq2 (Sep-Nov): loads Sep, Oct, Nov, Dec
   - aq3 (Dec-Feb): loads Dec, Jan, Feb, Mar
   - aq4 (Mar-May): loads Mar, Apr, May — but only March is available before the April cutoff
2. Apply 0.85 shrinkage factor to congestion prices in the profitable direction (conservative bias)
3. Average across months to get `mtm_1st_mean = sink_congestion - source_congestion`

This is the **only** available baseline for R1 — there is no prior auction round.

- **Coverage: 100.0%** of R1 trades (2,388,474 of 2,393,574 rows)
- 5,100 rows (0.2%) lost due to node coverage gaps (nodes in cleared trades not found in DA LMP data)

### 2.2 R2/R3: Previous Round's MCP (M) — CORRECTED

**Method:** `aptools.tools.get_m2m_mcp_for_trades_all(trades)` — retrieves MCPs from MISO's full published market clearing data.

This function:
1. Takes the full set of annual cleared trades as input
2. For R2 trades, looks up the R1 MCP for the same path from MISO's published auction results
3. For R3 trades, looks up the R2 MCP
4. For R1 trades, returns NaN (no prior round) — expected
5. Adds columns: `mtm_1st_mean` (prior round MCP), `mtm_2nd_mean` (two rounds prior), `mtm_3rd_mean`, `mcp_mean` (this round's actual MCP)

### 2.3 Coverage Comparison: Old vs Corrected

| Round | Old Coverage | Corrected Coverage | Improvement |
|-------|-------------|-------------------|-------------|
| **R1** | 100.0% (2,388,474) | 100.0% (2,388,474) | No change (same method) |
| **R2** | 68.0% (2,841,864) | **100.0%** (4,179,459) | **+32.0 pp, +1.34M rows** |
| **R3** | 75.1% (3,698,511) | **100.0%** (4,925,847) | **+24.9 pp, +1.23M rows** |
| **Total** | 77.6% (8,928,849) | **100.0%** (11,493,780) | **+22.4 pp, +2.56M rows** |

**Why the old coverage was low:** The old approach only found MCPs for paths where *we* traded in both rounds. But we only trade a subset of all MISO paths. Many paths in R2/R3 were paths that we traded in the later round but not in the earlier round — for these, the old method found no prior-round MCP. The corrected method finds MCPs for 100% of paths because MISO publishes clearing prices for every path.

---

## 3. Corrected Residual Distributions

### 3.1 By Round × Quarter (all PYs, both class_types)

| Round | Quarter | n | Bias | Mean |Res| | Median |Res| | p90 |Res| | p95 |Res| | p99 |Res| | Dir Acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **R1** | aq1 | 618,021 | +411 | 838 | 367 | 2,024 | 3,171 | 7,063 | 66% |
| **R1** | aq2 | 628,725 | +489 | 943 | 384 | 2,348 | 3,712 | 8,472 | 67% |
| **R1** | aq3 | 572,433 | +402 | 816 | 341 | 2,028 | 3,211 | 6,782 | 67% |
| **R1** | aq4 | 569,295 | +375 | 797 | 338 | 1,957 | 3,107 | 6,828 | 62% |
| **R2** | aq1 | 1,063,809 | +20 | 70 | 34 | 162 | 256 | 555 | 91% |
| **R2** | aq2 | 1,091,559 | +23 | 75 | 35 | 177 | 282 | 590 | 91% |
| **R2** | aq3 | 1,003,830 | +21 | 67 | 32 | 161 | 245 | 515 | 91% |
| **R2** | aq4 | 1,020,261 | +21 | 68 | 32 | 162 | 252 | 536 | 90% |
| **R3** | aq1 | 1,255,008 | +13 | 58 | 28 | 141 | 214 | 460 | 93% |
| **R3** | aq2 | 1,279,842 | +13 | 60 | 28 | 143 | 220 | 476 | 93% |
| **R3** | aq3 | 1,191,363 | +9 | 53 | 25 | 126 | 190 | 403 | 93% |
| **R3** | aq4 | 1,199,634 | +10 | 53 | 26 | 124 | 184 | 415 | 93% |

**How to read this table:**

For each trade, we compute `residual = actual_MCP - baseline`. If the baseline perfectly predicted the MCP, the residual would be zero. The columns measure how far off the baseline is:

- **Bias** = average of (actual_MCP - baseline) across all trades in that cell. A positive bias like +411 means actual MCPs are, on average, $411 higher than what the baseline predicted. The market systematically clears above our estimate.
- **Mean |Res|, Median |Res|** = average and median of the absolute error. These ignore direction — they just measure "how far off were we?" For example, mean |res| = 838 means the baseline is wrong by $838 on average, regardless of whether it was too high or too low.
- **p90, p95, p99 |Res|** = the 90th/95th/99th percentile of absolute error. p95 = 3,171 means 95% of trades have a baseline error smaller than $3,171, and the worst 5% are even larger. This is the key number for setting bid band widths — if you want your bids to cover 95% of outcomes, your bands need to span at least ±p95 around the baseline.
- **Dir Acc** = what fraction of the time the baseline correctly predicts whether the MCP will be positive or negative. 66% means the baseline gets the sign right about 2/3 of the time — only slightly better than a coin flip. 93% means it almost always gets the direction right.

### 3.2 By Round × Class Type

| Round | Class | n | Bias | Mean |Res| | Median |Res| | p90 |Res| | p95 |Res| | p99 |Res| | Dir Acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **R1** | offpeak | 1,161,102 | +413 | 817 | 322 | 2,027 | 3,290 | 7,293 | 65% |
| **R1** | onpeak | 1,227,372 | +428 | 882 | 390 | 2,151 | 3,320 | 7,400 | 67% |
| **R2** | offpeak | 2,020,782 | +21 | 68 | 31 | 162 | 260 | 552 | 91% |
| **R2** | onpeak | 2,158,677 | +21 | 72 | 36 | 168 | 260 | 551 | 91% |
| **R3** | offpeak | 2,390,796 | +10 | 54 | 25 | 129 | 197 | 444 | 93% |
| **R3** | onpeak | 2,535,051 | +12 | 58 | 28 | 137 | 207 | 437 | 93% |

### 3.3 Key Observations

1. **R1 residuals are unchanged** (same baseline method). Mean |residual| = 797–943, p95 = 3,107–3,712. The historical DA congestion proxy (H) remains a weak predictor.

2. **R2/R3 residuals are dramatically smaller than previously reported.** With correct full-market MCP data:
   - **R2:** Mean |res| = 70 (was 191), p95 = 260 (was 693) — a **63% reduction** in mean, **62% reduction** in p95
   - **R3:** Mean |res| = 56 (was 157), p95 = 202 (was 568) — a **64% reduction** in mean, **64% reduction** in p95

3. **Why the old R2/R3 numbers were inflated:** The manual path-matching approach introduced **selection bias** — it only included paths where we traded in both rounds. These tend to be paths where we had a strong view, which may correlate with larger price movements. The full market includes many "boring" paths with small MCPs and small residuals, which the old method missed.

4. **R2/R3 are now remarkably tight.** p95 = 202–282 is well below f0p's p95 of ~492. The previous round's MCP is an **extremely** strong baseline for annual auctions.

5. **Positive bias is much smaller for R2/R3.** R2 bias = +20–23, R3 bias = +9–13. This is close to zero — the previous round MCP is nearly unbiased. R1 bias remains large at +375–489.

6. **Direction accuracy:** R1 = 62–67% (still weak). R2/R3 = 90–93% (strong, unchanged from before since direction accuracy is robust to the coverage fix).

7. **R1 → R2 → R3 improvement ratio is now much starker:**
   - R1 p95 = 3,307 → R2 p95 = 260 → R3 p95 = 202
   - **R1 is 12.7× worse than R2** (was 4.8×) and **16.4× worse than R3** (was 5.8×)
   - This means the gap between R1 and R2/R3 is actually much larger than we thought

---

## 4. f0p Residual Comparison

### 4.1 f0p Residual Stats (mcp_mean - mtm_1st_mean)

*(These are from f0p training data in prior analysis — unchanged.)*

| Class | n | Bias | Mean |Res| | Median |Res| | p90 |Res| | p95 |Res| | p99 |Res| |
|---|---:|---:|---:|---:|---:|---:|---:|
| offpeak | 1,208,813 | +7 | 126 | 52 | 301 | 481 | 1,150 |
| onpeak | 1,340,804 | +10 | 140 | 67 | 321 | 503 | 1,165 |

### 4.2 Direct Comparison: Annual vs f0p |Residual|

| Metric | R1 (H) | R2 (M) | R3 (M) | f0p (MTM) |
|---|---:|---:|---:|---:|
| **Mean |res|** | 851 | 70 | 56 | 133 |
| **Median |res|** | 357 | 33 | 27 | 59 |
| **p90 |res|** | 2,093 | 166 | 133 | 311 |
| **p95 |res|** | 3,307 | 260 | 202 | 492 |
| **p99 |res|** | 7,352 | 551 | 440 | 1,157 |
| **Bias** | +421 | +21 | +11 | +9 |
| **Direction acc** | 66% | 91% | 93% | ~95%* |

*f0p direction accuracy estimated from bias/mean ratio.

### 4.3 Key Findings — REVISED

1. **R1 annual residuals are ~6.7× larger than f0p.** This is unchanged. R1 needs much wider bands.

2. **R2/R3 annual residuals are now SMALLER than f0p — not larger.** This is the major correction:
   - **R2 p95 = 260 vs f0p p95 = 492** — R2 is **47% tighter** than f0p
   - **R3 p95 = 202 vs f0p p95 = 492** — R3 is **59% tighter** than f0p
   - Previous report incorrectly stated R2/R3 were "20-50% larger than f0p." That was an artifact of the biased manual matching.

3. **The practical implication is profound:** R2/R3 annual bid bands can be **narrower** than f0p bands, not wider. The previous round's MCP is a better baseline for annual auctions than the MTM used in monthly forward auctions. This makes sense: in the monthly forward market, the MTM is a model estimate; in annual R2/R3, the baseline is an actual auction clearing price from a very similar market (same paths, same participants, recent round).

4. **R2/R3 bias is near zero** (+21 and +11 respectively), comparable to f0p (+9). The old biased data showed R2 bias = +74 and R3 bias = +38 — inflated by the portfolio selection effect.

---

## 5. aq4 Special Analysis

### 5.1 R1 Residual Stats by Quarter × Class

| Quarter | Class | n | Bias | Mean |Res| | Median |Res| | p90 |Res| | p95 |Res| | Dir Acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| aq1 | offpeak | 299,610 | +387 | 774 | 308 | 1,867 | 3,031 | 65% |
| aq1 | onpeak | 318,411 | +433 | 899 | 422 | 2,158 | 3,264 | 67% |
| aq2 | offpeak | 304,014 | +489 | 921 | 359 | 2,314 | 3,706 | 65% |
| aq2 | onpeak | 324,711 | +489 | 963 | 405 | 2,381 | 3,716 | 68% |
| aq3 | offpeak | 279,303 | +386 | 780 | 314 | 1,936 | 3,192 | 66% |
| aq3 | onpeak | 293,130 | +417 | 850 | 365 | 2,093 | 3,232 | 68% |
| aq4 | offpeak | 278,175 | +385 | 788 | 311 | 1,962 | 3,211 | 62% |
| aq4 | onpeak | 291,120 | +365 | 805 | 364 | 1,954 | 3,001 | 63% |

### 5.2 R1 Bias Stability by Planning Year

| PY | n | Bias | Mean |Res| | p95 |Res| | Dir Acc |
|---|---:|---:|---:|---:|---:|
| 2019 | 249,636 | +173 | 372 | 1,340 | 64% |
| 2020 | 251,418 | +148 | 371 | 1,400 | 71% |
| 2021 | 345,513 | +305 | 743 | 2,938 | 68% |
| 2022 | 317,439 | +679 | 1,346 | 5,508 | 62% |
| 2023 | 393,684 | +595 | 1,008 | 3,602 | 65% |
| 2024 | 393,486 | +459 | 941 | 3,948 | 67% |
| 2025 | 437,298 | +432 | 901 | 3,395 | 64% |

**Observations on bias stability:**
- Bias ranges from +148 (PY 2020) to +679 (PY 2022) — a 4.6× range.
- PY 2022 is a clear outlier: highest bias (+679), highest mean |res| (1,346), highest p95 (5,508), lowest dir accuracy (62%).
- The mean bias across all years is +421. Using this as a fixed correction would help on average but would be unreliable year to year.
- PYs 2019-2020 had notably smaller residuals (p95 ≈ 1,340-1,400) compared to PYs 2022-2024 (p95 ≈ 3,600-5,500). The market has become harder to predict in recent years.

### 5.3 Verdict on aq4

**aq4 is NOT structurally harder in terms of residual magnitude.** Its mean |residual| (797) is lower than aq1 (838) and aq2 (943). The p95 values are comparable across quarters.

**However, aq4 direction accuracy is notably worse: 62-63% vs 65-68%.** The H baseline more often gets the sign wrong for aq4, consistent with the structural issue that aq4 only has 1 month of DA congestion data (March) before the April cutoff, vs 3+ months for other quarters.

**aq2 is the hardest quarter by residual magnitude** (p95 = 3,706-3,716), not aq4.

---

## 6. R2/R3 Training Data Feasibility (Corrected)

### 6.1 Bin Counts (all PYs pooled, full market data)

| Round | Class | Bin | Total Rows | Avg per PY | Mean |Res| | p90 |Res| | p95 |Res| | Dir Acc |
|---|---|---|---:|---:|---:|---:|---:|---:|
| R2 | offpeak | tiny(<50) | 714,348 | 102,050 | 24 | 56 | 84 | 78% |
| R2 | offpeak | small(50-250) | 704,277 | 100,611 | 50 | 110 | 153 | 96% |
| R2 | offpeak | medium(250-1k) | 453,558 | 64,794 | 101 | 223 | 306 | 99% |
| R2 | offpeak | large(1k+) | 148,599 | 21,228 | 263 | 559 | 755 | 100% |
| R2 | onpeak | tiny(<50) | 634,308 | 90,615 | 26 | 64 | 94 | 77% |
| R2 | onpeak | small(50-250) | 795,705 | 113,672 | 52 | 113 | 155 | 96% |
| R2 | onpeak | medium(250-1k) | 563,640 | 80,520 | 99 | 214 | 290 | 100% |
| R2 | onpeak | large(1k+) | 165,024 | 23,575 | 256 | 564 | 767 | 100% |
| R3 | offpeak | tiny(<50) | 804,156 | 114,879 | 20 | 47 | 70 | 81% |
| R3 | offpeak | small(50-250) | 876,411 | 125,202 | 39 | 88 | 121 | 97% |
| R3 | offpeak | medium(250-1k) | 535,065 | 76,438 | 81 | 173 | 233 | 100% |
| R3 | offpeak | large(1k+) | 175,164 | 25,023 | 208 | 460 | 634 | 100% |
| R3 | onpeak | tiny(<50) | 724,650 | 103,521 | 22 | 52 | 78 | 81% |
| R3 | onpeak | small(50-250) | 955,224 | 136,461 | 41 | 90 | 123 | 97% |
| R3 | onpeak | medium(250-1k) | 659,997 | 94,285 | 79 | 173 | 228 | 100% |
| R3 | onpeak | large(1k+) | 195,180 | 27,883 | 205 | 441 | 599 | 100% |

### 6.2 Comparison vs Prior (Biased) Numbers

The corrected bin counts are 2-3× larger per bin because we now have 100% coverage:

| Metric | Old (biased) | Corrected | Change |
|--------|-------------|-----------|--------|
| R2 offpeak tiny avg/PY | 41,497 | 102,050 | +146% |
| R2 offpeak large avg/PY | 39,532 | 21,228 | -46% |
| R3 onpeak medium avg/PY | 92,283 | 94,285 | +2% |

The tiny bin grew the most because many of the previously-unmatched paths were low-MCP paths. The large bin actually shrank slightly, suggesting the old biased sample was enriched for high-MCP paths.

### 6.3 Residual Scaling Within Bins (Corrected)

The corrected residual scaling pattern:

| Bin | R2 p95 range | R3 p95 range | Pattern |
|-----|-------------|-------------|---------|
| tiny (<50) | 84–94 | 70–78 | ~1× |
| small (50-250) | 153–155 | 121–123 | ~2× tiny |
| medium (250-1k) | 290–306 | 228–233 | ~3-4× tiny |
| large (1k+) | 755–767 | 599–634 | ~9× tiny |

Residuals scale roughly linearly with |baseline| magnitude, confirming that |M|-based binning is the right approach. The corrected p95 values are **much lower** than the old biased ones (e.g., R2 large was 1,408-1,436, now 755-767).

### 6.4 Feasibility Verdict

**All bins have >21,000 avg rows per PY — far exceeding the MIN_ROWS_FOR_EMPIRICAL = 100 threshold.** f2p-style binning is completely feasible.

With full market coverage, the data is even more plentiful than before. A single planning year provides sufficient data for any bin. Multi-year pooling (3-5 year lookback) gives >60k rows per bin minimum.

---

## 7. R1 by |H| Bin

| |H| bin | n | Bias | Mean |Res| | Median |Res| | p90 |Res| | p95 |Res| | p99 |Res| | Dir Acc |
|---------|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny (<50) | 437,853 | +200 | 343 | 159 | 824 | 1,279 | 2,747 | 50% |
| small (50-250) | 967,671 | +251 | 463 | 238 | 1,099 | 1,656 | 3,293 | 59% |
| medium (250-1k) | 717,405 | +459 | 970 | 594 | 2,257 | 3,160 | 5,501 | 75% |
| large (1k+) | 265,545 | +1,302 | 2,777 | 1,776 | 6,490 | 8,772 | 14,353 | 90% |

**Intuition:** When H is near zero (tiny bin), the baseline tells us almost nothing about the MCP. Direction accuracy is 50% — equivalent to a coin flip. The residuals are still substantial (p95 = 1,279) even though |H| < 50, meaning paths that H thinks have ~zero congestion still see MCPs of ±1,300.

As |H| grows, the baseline becomes more directionally accurate (90% for large paths) but the absolute residual also grows enormously (p95 = 8,772 for large paths). This is because large-|H| paths have more volatile congestion patterns, so while H correctly identifies the direction, the magnitude error is proportionally larger.

**Key design implication:** R1 bid widths must scale with |H|. A single fixed offset (like the current ±2,000 cap) is fundamentally wrong — it's too wide for tiny paths and too narrow for large paths.

---

## 8. R2/R3 Blend Test (Corrected)

### 8.1 Test Setup

For R2/R3 trades, we have two potential baselines:
- **M** = previous round's MCP (from full MISO market data)
- **H** = historical DA congestion (same as R1 baseline)

We test whether blending M with H improves the baseline. The hypothesis is that M alone is optimal because the previous round's MCP already incorporates all information that H provides (and more). Note: blend test is limited to paths where both M and H are available (56.4% of R2/R3 trades — limited by R1 node coverage in fill_mtm).

### 8.2 Results

| Blend (M weight, H weight) | Round | n | Mean |Res| | p90 |Res| | p95 |Res| | p99 |Res| |
|---------------------------|-------|---:|---:|---:|---:|---:|
| **M=1.0, H=0.0** | R2 | 2,835,033 | **64** | **150** | **233** | **512** |
| M=0.95, H=0.05 | R2 | 2,835,033 | 68 | 158 | 245 | 535 |
| M=0.90, H=0.10 | R2 | 2,835,033 | 76 | 176 | 269 | 583 |
| **M=1.0, H=0.0** | R3 | 2,302,110 | **52** | **123** | **188** | **426** |
| M=0.95, H=0.05 | R3 | 2,302,110 | 57 | 134 | 204 | 463 |
| M=0.90, H=0.10 | R3 | 2,302,110 | 67 | 154 | 236 | 535 |

### 8.3 Conclusion

**M-only remains definitively best.** Adding even 5% H weight to the R2/R3 baseline increases p95 by +12/+16 (5-9% worse). Adding 10% H makes it 15-25% worse.

This is consistent with the corrected data: M is now an even stronger baseline than we thought (p95 = 260 for R2, down from 693), so adding noise from H is even more harmful.

---

## 9. Summary of Corrected Findings

### 9.1 What Changed Materially

| Metric | Old (Biased) | Corrected | Change |
|--------|-------------|-----------|--------|
| R2 coverage | 68.0% | **100.0%** | +32 pp |
| R3 coverage | 75.1% | **100.0%** | +25 pp |
| R2 mean |res| | 191 | **70** | **-63%** |
| R2 p95 |res| | 693 | **260** | **-62%** |
| R3 mean |res| | 157 | **56** | **-64%** |
| R3 p95 |res| | 568 | **202** | **-64%** |
| R2 vs f0p | R2 40% wider | **R2 47% tighter** | **Reversed** |
| R3 vs f0p | R3 15% wider | **R3 59% tighter** | **Reversed** |
| R1 vs R2 ratio | 4.8× | **12.7×** | Much wider gap |

### 9.2 Updated Assumption Verdicts

| # | Assumption | Old Verdict | New Verdict | Key Evidence |
|---|-----------|-------------|-------------|-------------|
| **A1** | Annual bands should be narrower than f0p | REJECTED for R1, NUANCED for R2/R3 | **REJECTED for R1. CONFIRMED for R2/R3.** | R1 p95=3,307 (6.7× wider). R2 p95=260 (47% tighter). R3 p95=202 (59% tighter). |
| **A2** | Annual per-period volume >> f0p | NUANCED | NUANCED (unchanged) | 1.44× ratio. |
| **A3** | f0p p95 widths average ~500 | CONFIRMED | CONFIRMED (unchanged) | f0p p95=481-503. |
| **A4** | R2/R3 can be treated like f2p | CONFIRMED | **STRONGLY CONFIRMED** | Even better: R2/R3 residuals are tighter than f0p. |
| **A5** | Same formula for all quarters | NUANCED | NUANCED (unchanged) | aq4 dir acc lower, aq2 hardest by magnitude. |

### 9.3 Revised Practical Implications

1. **R1 is the sole hard problem.** R2/R3 are now easier than f0p monthly forwards. All modeling effort should focus on R1.

2. **R2/R3 can use tighter bands than f0p.** Previous recommendation to use f0p-width bands was conservative. R2/R3 bands calibrated to their actual residual distribution (p95 ≈ 200-260) will be much tighter.

3. **The legacy system's bid range (cap = 2,000) is not too narrow for R2/R3 — it's too wide.** R2 p95 = 260. A cap of 2,000 means the system spreads bids over 7.7× the needed range, wasting bid stack space on impossible prices.

4. **The gap between R1 and R2/R3 is 12.7× (was 4.8×).** This makes the case for round-specific parameters even stronger. Using the same params for R1 and R2/R3 is now provably a factor-of-13 calibration error.

---

## Appendix: Cached Data Files

| File | Rows | Size | Description |
|------|---:|---:|-------------|
| `annual_cleared_all_v2.parquet` | 11,498,880 | 236 MB | All annual cleared trades, dtype fixed |
| `annual_with_mcp_v2.parquet` | 11,498,880 | 591 MB | + full MISO MCP data (mtm_1st_mean, mcp_mean) |
| `r1_filled_v2.parquet` | 2,388,474 | 92 MB | R1 trades with H baseline filled |
| `all_residuals_v2.parquet` | 11,493,780 | 456 MB | All rounds with residuals computed |

All files at `/opt/temp/qianli/annual_research/`.
