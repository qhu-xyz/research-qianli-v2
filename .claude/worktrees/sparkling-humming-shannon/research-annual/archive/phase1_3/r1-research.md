# R1 Baseline Research

**Date:** 2026-02-12
**Goal:** Find a better MCP prediction (baseline) for R1 annual FTR auctions, where the only available signal is H = historical DA congestion.
**Current production baseline:** `H = mean(sink_DA_congestion) - mean(source_DA_congestion)` with 0.85 shrinkage, from prior-year delivery months.
**Current production performance:** p95 |residual| = 3,307, direction accuracy = 66%, bias = +421.

---

## Table of Contents

1. [Experiment Setup](#1-experiment-setup)
2. [Experiment A: Prior-Year R1 MCP as Baseline](#2-experiment-a-prior-year-r1-mcp-as-baseline)
3. [Experiment B: Shrinkage Factor Sweep](#3-experiment-b-shrinkage-factor-sweep)
4. [Experiment C: Bias Correction (Leave-One-Year-Out)](#4-experiment-c-bias-correction-leave-one-year-out)
5. [Experiment D: DA Feature Engineering](#5-experiment-d-da-feature-engineering)
   - D1: Volatility Premium
   - D2: Middle-Month Weighting
   - D3: Percentile-Based Baseline
   - D4: Positive-Only Aggregation
   - D5: Trend / Recency Weighting
6. [Experiment E: Combined R1 Improvement](#6-experiment-e-combined-r1-improvement)
7. [Summary and Recommendations](#7-summary-and-recommendations)

---

## 1. Experiment Setup

### Data

| Dataset | Rows | Description |
|---|---:|---|
| `all_residuals_v2.parquet` | 11,493,780 | All rounds with residuals (R1 subset: 2,388,474) |
| `annual_with_mcp_v2.parquet` | 11,498,880 | Full MISO MCP data including mtm_now_* |
| `r1_filled_v2.parquet` | 2,388,474 | R1 trades with H baseline filled |
| Monthly DA congestion | ~469k rows | `MisoDaLmpMonthlyAgg`, all nodes, 2017-2024 |

### How H is Computed Today

For each R1 path (source → sink), for a given quarter (e.g., aq1 = Jun-Aug delivery):

1. Load monthly-average DA congestion for source and sink nodes from the prior year's delivery months
2. Apply 0.85 shrinkage to congestion in the profitable direction (positive sink × 0.85, negative source × 0.85)
3. Average across months: `H = mean(sink_congestion) - mean(source_congestion)`

Code: `fill_mtm_1st_period_with_hist_revenue()` in `pbase/analysis/tools/miso.py:322`.

### Evaluation Metrics

- **Bias** = mean(MCP - baseline). Positive = baseline systematically underestimates.
- **Mean |Res|** = mean of |MCP - baseline|. Average absolute error.
- **p95 |Res|** = 95th percentile of |MCP - baseline|. Determines bid band width needed for 95% coverage.
- **Dir Acc** = fraction of trades where sign(MCP) == sign(baseline). How often we get the direction right.

### Train/Test Split (for Experiment D)

- **Train:** PY 2019-2022 (discovery, parameter tuning)
- **Test:** PY 2023-2025 (out-of-sample validation)

Experiments A-C use leave-one-year-out cross-validation across all PYs 2019-2025.

---

## 2. Experiment A: Prior-Year R1 MCP as Baseline

### Hypothesis

Using PY-1's R1 auction MCP as the R1 baseline should outperform H, because an actual auction clearing price embeds participant behavior and forward-looking expectations that backward-looking DA congestion cannot.

### Method

For each PY, look up PY-1's R1 MCP for the same path `(period_type, class_type, source_id, sink_id)`. Compute residual = current MCP - prior-year MCP. Compare against H on the same matched subset.

### Coverage

Only 32% of R1 paths appear in the prior year's R1 auction. Path participation changes significantly year-to-year.

| PY | Total R1 Trades | Matched to PY-1 | Coverage |
|---:|---:|---:|---:|
| 2020 | 252,222 | 81,876 | 32.5% |
| 2021 | 346,584 | 107,793 | 31.1% |
| 2022 | 318,069 | 104,808 | 33.0% |
| 2023 | 394,770 | 120,870 | 30.6% |
| 2024 | 393,828 | 131,430 | 33.4% |
| 2025 | 437,871 | 142,746 | 32.6% |

### Results (matched paths, fair comparison)

| PY | Method | n | Bias | Mean |Res| | p95 | Dir Acc |
|---:|---|---:|---:|---:|---:|---:|
| 2020 | prior-MCP | 81,654 | -87 | 352 | 1,342 | 81.8% |
| 2020 | H-baseline | 81,654 | +202 | 315 | 1,182 | 68.2% |
| 2021 | prior-MCP | 107,616 | +49 | 553 | 2,508 | 79.8% |
| 2021 | H-baseline | 107,616 | +237 | 626 | 2,520 | 64.4% |
| 2022 | prior-MCP | 104,514 | +409 | 984 | 4,299 | 81.5% |
| 2022 | H-baseline | 104,514 | +738 | 1,187 | 4,741 | 56.7% |
| 2023 | prior-MCP | 120,567 | -257 | 916 | 3,677 | 79.8% |
| 2023 | H-baseline | 120,567 | +700 | 949 | 3,482 | 60.2% |
| 2024 | prior-MCP | 131,199 | -94 | 620 | 2,365 | 81.4% |
| 2024 | H-baseline | 131,199 | +494 | 730 | 2,846 | 63.2% |
| 2025 | prior-MCP | 142,641 | -23 | 506 | 1,803 | 81.4% |
| 2025 | H-baseline | 142,641 | +524 | 778 | 2,948 | 59.6% |

### Aggregate (all PYs 2020-2025, matched paths)

| Method | n | Bias | Mean |Res| | Median |Res| | p95 | p99 | Dir Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Prior-year MCP** | 688,191 | **-8** | **661** | 247 | **2,624** | 6,378 | **80.9%** |
| H-baseline | 688,191 | +499 | 782 | 331 | 3,045 | 6,961 | 61.7% |

### Verdict

**Prior-year MCP is substantially better than H on matched paths: p95 improves 14% (2,624 vs 3,045), direction accuracy improves +19pp (81% vs 62%), bias is near zero.**

Limitation: only 32% of paths have prior-year data. The remaining 68% must fall back to H.

---

## 3. Experiment B: Shrinkage Factor Sweep

### Hypothesis

The 0.85 shrinkage applied to profitable-direction congestion is hand-tuned. A different value might reduce residuals.

### Method

Approximate: `H_new = H / 0.85 × new_shrinkage`. This assumes all congestion was in the profitable direction. Directionally correct for comparison. All 2,388,474 R1 rows.

### Results

| Shrinkage | Bias | Mean |Res| | Median |Res| | p95 | Dir Acc |
|---:|---:|---:|---:|---:|---:|
| 0.50 | +439 | 899 | 349 | 3,603 | 66.1% |
| 0.70 | +429 | 867 | 352 | 3,415 | 66.1% |
| **0.85 (current)** | **+421** | **851** | **357** | **3,307** | **66.1%** |
| 0.90 | +418 | 847 | 360 | 3,280 | 66.1% |
| 0.95 | +416 | 843 | 362 | 3,252 | 66.1% |
| **1.00 (no shrinkage)** | **+413** | **841** | **365** | **3,230** | **66.1%** |

### Verdict

**Shrinkage = 1.0 is marginally better: p95 = 3,230 vs 3,307 (-2.3%). Direction accuracy is unchanged.** The improvement is negligible. The shrinkage parameter is not a meaningful lever — the problem is the weakness of DA data itself, not how it's scaled.

---

## 4. Experiment C: Bias Correction (Leave-One-Year-Out)

### Hypothesis

H systematically underestimates MCP by +421 on average. Adding a per-quarter bias correction should center predictions. The bias varies by quarter: aq2 is highest (+489), aq4 lowest (+375).

### Method

For each PY, estimate the per-quarter bias from all OTHER PYs' residuals (leave-one-year-out), then apply: `H_corrected = H + bias_estimate(quarter)`. This is proper out-of-sample evaluation.

### Per-PY Out-of-Sample Results

| PY | Method | n | Bias | Mean |Res| | p95 | Dir Acc |
|---:|---|---:|---:|---:|---:|---:|
| 2019 | Uncorrected | 249,636 | +173 | 372 | 1,340 | 64.7% |
| 2019 | Corrected | 249,636 | -277 | 477 | 1,285 | 69.2% |
| 2020 | Uncorrected | 251,418 | +148 | 371 | 1,400 | 71.1% |
| 2020 | Corrected | 251,418 | -304 | 510 | 1,392 | 67.3% |
| 2021 | Uncorrected | 345,513 | +305 | 743 | 2,938 | 67.8% |
| 2021 | Corrected | 345,513 | -137 | 784 | 2,839 | 71.8% |
| 2022 | Uncorrected | 317,439 | +679 | 1,346 | 5,508 | 62.7% |
| 2022 | Corrected | 317,439 | +298 | 1,283 | 5,378 | 71.8% |
| 2023 | Uncorrected | 393,684 | +595 | 1,008 | 3,602 | 65.3% |
| 2023 | Corrected | 393,684 | +211 | 942 | 3,390 | 70.5% |
| 2024 | Uncorrected | 393,486 | +459 | 941 | 3,948 | 67.9% |
| 2024 | Corrected | 393,486 | +45 | 952 | 3,816 | 69.6% |
| 2025 | Uncorrected | 437,298 | +432 | 901 | 3,395 | 64.5% |
| 2025 | Corrected | 437,298 | +12 | 877 | 3,263 | 67.0% |

### Overall Aggregate

| Method | n | Bias | Mean |Res| | Median |Res| | p95 | Dir Acc |
|---|---:|---:|---:|---:|---:|---:|
| Uncorrected | 2,388,474 | +421 | 851 | 357 | 3,307 | 66.1% |
| **Corrected** | 2,388,474 | **+3** | 860 | 422 | **3,157** | **69.6%** |

### Per-Quarter Aggregate

| Quarter | Method | Bias | Mean |Res| | p95 | Dir Acc |
|---|---|---:|---:|---:|---:|
| aq1 | Uncorrected | +411 | 838 | 3,171 | 66.7% |
| aq1 | Corrected | +5 | 840 | 3,035 | 69.4% |
| aq2 | Uncorrected | +489 | 943 | 3,712 | 67.4% |
| aq2 | Corrected | +2 | 972 | 3,548 | 69.1% |
| aq3 | Uncorrected | +402 | 816 | 3,211 | 67.4% |
| aq3 | Corrected | +2 | 820 | 3,067 | 71.1% |
| aq4 | Uncorrected | +375 | 797 | 3,107 | 62.9% |
| aq4 | Corrected | +5 | 798 | 2,984 | **68.8%** |

### Verdict

**Bias correction provides consistent improvement: p95 drops 4.5% (3,307 → 3,157), direction accuracy improves +3.5pp (66.1% → 69.6%).** Bias goes from +421 to near-zero (+3).

aq4 benefits most in direction accuracy (+5.9pp), which makes sense — aq4 had the worst dir acc (62.9%) due to having only 1 month of DA data.

Trade-off: mean |res| increases slightly (851 → 860) and median increases (357 → 422). The correction helps the tails and direction at the cost of a slight increase in typical errors. Acceptable trade-off since p95 (which determines band width) is the primary metric.

Caution: overcorrects for low-bias years (PY 2019-2020 had bias ~+150 but correction applied ~+430). If market reverts to low-bias regime, correction would hurt.

---

## 5. Experiment D: DA Feature Engineering

### Motivation

The current H uses the simple mean of monthly DA congestion. This discards distributional information. Can we extract better features from the same underlying DA data?

### Setup

- Loaded monthly DA congestion (`MisoDaLmpMonthlyAgg`) for all 2,465+ MISO nodes, 2017-2024
- For each R1 path × PY × quarter, computed node-level features from the same months that `fill_mtm` uses
- Built path-level features: `feature(sink) - feature(source)` for each feature
- **Train:** PY 2019-2022 (299,590 path-quarter combos). **Test:** PY 2023-2025 (321,897 combos).

### D1: Volatility Premium

**Hypothesis:** Nodes with spiky DA congestion (high standard deviation) should have higher FTR MCPs. Spiky = more revenue potential.

**Feature:** `path_std_diff = std(sink_congestion_across_months) - std(source_congestion_across_months)`

**Results:**

| Dataset | corr(path_std_diff, residual) | corr(sink_std, residual) | corr(source_std, residual) |
|---|---:|---:|---:|
| Train 2019-2022 | **+0.007** | +0.060 | +0.059 |
| Test 2023-2025 | **-0.037** | +0.070 | +0.102 |

Path-level volatility has near-zero correlation with residuals, and the sign flips between train and test (overfitting risk). The individual node volatilities (sink_std, source_std) correlate weakly with the *level* of residuals but cancel out in the path spread.

| Baseline | Train p95 | Test p95 |
|---|---:|---:|
| H (α=0) | 3,073 | 3,806 |
| H + 0.1×std | 3,066 | 3,810 |
| H + 0.5×std | 3,101 | 3,902 |
| H + 1.0×std | 3,233 | 4,168 |

**Verdict: REJECTED.** Volatility adds noise, not signal. Higher α makes predictions worse.

### D2: Middle-Month Weighting

**Hypothesis:** Congestion in the middle of the delivery period (e.g., mid-July for aq1 Jun-Aug) is more structural and more valuable than edge-month congestion. Weight the middle month more heavily.

**Features tested:**
- Equal-weight mean (current): `(M1 + M2 + M3) / 3`
- Middle-weighted: `0.25×M1 + 0.50×M2 + 0.25×M3`
- Middle-month only: just M2

**Results:**

| Baseline | Train p95 | Train Dir Acc | Test p95 | Test Dir Acc |
|---|---:|---:|---:|---:|
| **Equal-weight mean** | **3,623** | **69.6%** | **3,952** | **70.8%** |
| Middle-weighted (50/25/25) | 3,632 | 69.3% | 4,007 | 70.5% |
| Middle-month only | 3,805 | 65.9% | 4,382 | 67.2% |

Per-quarter (test set):

| Quarter | Equal-weight p95 | Middle-weighted p95 | Difference |
|---|---:|---:|---:|
| aq1 (Jun-Aug) | 4,032 | 4,045 | +0.3% worse |
| aq2 (Sep-Nov) | 4,807 | 4,842 | +0.7% worse |
| aq3 (Dec-Feb) | 3,465 | 3,546 | +2.3% worse |
| aq4 (Mar-May) | 3,511 | 3,591 | +2.3% worse |

**Verdict: REJECTED.** Equal-weight mean wins across the board. Using fewer months or weighting unevenly adds noise. The market treats all months in a quarter roughly equally when pricing FTR paths.

### D3: Percentile-Based Baseline

**Hypothesis:** Mean is dragged down by zero-congestion months. Using p75 or max captures "when congestion actually happens" and should be more predictive.

**Results:**

| Baseline | Train Bias | Train p95 | Test Bias | Test p95 |
|---|---:|---:|---:|---:|
| **Mean** | **+274** | **3,623** | **+191** | **3,952** |
| Median | +291 | 3,601 | +212 | 4,045 |
| p75 | +277 | 3,720 | +204 | 4,091 |
| Max month | +262 | 3,989 | +195 | 4,510 |

**Verdict: REJECTED.** Mean is the best. Median is close on train but worse on test. Higher percentiles amplify month-to-month noise. p75 is 3.5% worse, max is 14% worse on test.

### D4: Positive-Only Aggregation

**Hypothesis:** Only counting positive (profitable-direction) congestion removes noise from negative-congestion months.

**Feature:** `h_positive = mean(max(0, sink_cong)) - mean(max(0, source_cong))`

**Results:**

| Baseline | Train Bias | Train p95 | Test Bias | Test p95 |
|---|---:|---:|---:|---:|
| **Mean (all months)** | **+274** | **3,623** | **+191** | **3,952** |
| Positive-only | +352 | 3,898 | +404 | 4,657 |

**Verdict: REJECTED.** Positive-only is substantially worse: +18% on test p95. Zeroing out negative months removes real information (negative congestion IS signal that the path is less valuable) and inflates bias.

### D5: Trend / Recency Weighting

**Hypothesis:** Recent months are more predictive. If congestion is trending up, the MCP will be higher than the flat average.

**Feature:** Linearly weighted average: later months get higher weight. Also tested raw trend (last month - first month).

**Results:**

| Dataset | corr(path_trend, residual) |
|---|---:|
| Train 2019-2022 | +0.024 |
| Test 2023-2025 | +0.011 |

| Baseline | Train p95 | Test p95 |
|---|---:|---:|
| **Equal-weight** | **3,623** | **3,952** |
| Recency-weighted | 3,681 | 4,006 |

**Verdict: REJECTED.** Near-zero correlation between trend and residual. Recency weighting adds +1.4% to test p95. Month-to-month DA congestion does not have useful momentum for predicting annual FTR auction outcomes.

### D Summary

| Theory | Feature | Train p95 | Test p95 | vs Mean |
|---|---|---:|---:|---:|
| Baseline | **H mean (current production)** | **3,623** | **3,952** | **—** |
| D1 Volatility | H + α×std | 3,066-3,233 | 3,810-4,168 | Worse |
| D2 Middle-month | 50/25/25 weighting | 3,632 | 4,007 | +1.4% worse |
| D3 Percentile | p75 | 3,720 | 4,091 | +3.5% worse |
| D3 Percentile | max month | 3,989 | 4,510 | +14% worse |
| D4 Positive-only | max(0, cong) | 3,898 | 4,657 | +18% worse |
| D5 Recency | Later months weighted more | 3,681 | 4,006 | +1.4% worse |

**All DA feature alternatives are worse than the simple mean.** The equal-weight mean already extracts the maximum predictive value from monthly DA congestion data.

---

## 6. Experiment E: Combined R1 Improvement

### Method

Combine the improvements that work (Experiments A + B + C):
1. Remove shrinkage (1.0 instead of 0.85)
2. Apply leave-one-year-out per-quarter bias correction
3. Use prior-year MCP where available (32% of paths), corrected H otherwise

### Results (PY 2020-2025)

| Method | n | Bias | Mean |Res| | p95 | Dir Acc |
|---|---:|---:|---:|---:|---:|
| H (current, shrinkage=0.85) | 2,138,838 | +450 | 906 | 3,521 | 66.3% |
| H (shrinkage=1.0) | 2,138,838 | +441 | 896 | 3,434 | 66.3% |
| H (shrinkage=1.0) + bias correction | 2,138,838 | +36 | 886 | 3,287 | 70.2% |
| **Prior-year MCP** (32% matched only) | 299,694 | -43 | 602 | **2,344** | **79.0%** |
| Combined (prior-MCP or corrected-H) | 2,138,838 | +44 | 878 | 3,300 | 71.4% |

### Blending Prior-Year MCP with Corrected H (where both available)

On the 32% of paths with both signals, what's the optimal blend?

| α (prior MCP weight) | Bias | Mean |Res| | p95 | Dir Acc |
|---:|---:|---:|---:|---:|
| 0.00 (corrected H only) | -103 | 656 | 2,236 | 70.7% |
| 0.25 | -88 | 550 | 1,898 | 73.8% |
| **0.50** | **-73** | **504** | **1,796** | **76.5%** |
| 0.75 | -58 | 525 | 1,978 | 78.3% |
| 1.00 (prior MCP only) | -43 | 602 | 2,344 | 79.0% |

On matched paths, **50/50 blend achieves p95 = 1,796** — a 41% improvement over H.

### Per-PY Impact

| PY | Prior MCP Coverage | H p95 | Combined p95 | Change |
|---:|---:|---:|---:|---:|
| 2020 | 32% | 1,400 | 1,439 | -2.8% (hurt) |
| 2021 | 16% | 2,938 | 2,791 | +5.0% |
| 2022 | 12% | 5,508 | 5,226 | +5.1% |
| 2023 | 11% | 3,602 | 3,373 | +6.4% |
| 2024 | 10% | 3,948 | 3,725 | +5.6% |
| 2025 | 10% | 3,395 | 3,129 | +7.8% |

### Verdict

Combined approach gives 5-8% p95 improvement for PYs 2021-2025. Limited by 32% coverage of prior-year MCP — the 68% fallback to corrected H dominates the overall numbers.

---

## 7. Summary and Recommendations

### What Works

| Improvement | Coverage | p95 Impact | Effort |
|---|---|---|---|
| **Bias correction (LOO per-quarter)** | 100% | -4.5% (3,307 → 3,157) | Low: add ~+400 offset per quarter |
| **Prior-year MCP** | 32% | -14% on matched (3,045 → 2,624) | Medium: lookup PY-1 MCPs |
| **50/50 blend (on matched paths)** | 32% | -41% on matched (3,045 → 1,796) | Medium: blend two signals |
| **Remove 0.85 shrinkage** | 100% | -2.3% (3,307 → 3,230) | Trivial: change one parameter |

### What Doesn't Work

| Approach | Why It Fails |
|---|---|
| Volatility/spikiness of DA congestion | Near-zero correlation with residual; sign flips between train/test |
| Middle-month weighting | Market treats all months equally; weighting adds noise |
| p75/max instead of mean | Amplifies month-to-month noise without adding signal |
| Positive-only congestion | Removes real information; inflates bias |
| Trend/recency weighting | DA congestion has no useful momentum for auction prediction |
| Multi-year averaging (n=2,3) | Tested in Phase 2; adds stale data, does not help |

### Why the DA Approach Has a Ceiling

The fundamental issue is **not** how we aggregate DA congestion — it's that last year's DA congestion is structurally disconnected from this year's auction clearing price. The FTR auction MCP is driven by:

- Participant bidding behavior and portfolio strategies
- Forward-looking constraint forecasts (SPICE/DA signals)
- Market structure changes (new generation, transmission upgrades)
- Gas price / load forecast changes

None of these are captured by backward-looking DA congestion. The simple mean already extracts the maximum signal available. All attempts to "be clever" with the same underlying data add noise.

### Recommended R1 Baseline Strategy

**Two-tier approach:**

**Tier 1 — Paths with prior-year R1 MCP (32% of paths):**
```
baseline = 0.50 × prior_year_r1_mcp + 0.50 × H_corrected
```
Expected p95 ≈ 1,796 on these paths.

**Tier 2 — All other paths (68%):**
```
baseline = H_corrected = (H / 0.85) + bias_estimate(quarter)
```
Where `bias_estimate` ≈ +410 (aq1), +490 (aq2), +400 (aq3), +375 (aq4), estimated from prior PYs.
Expected p95 ≈ 3,157 on these paths.

### Untested Avenues

| Idea | Why Not Tested | Potential |
|---|---|---|
| **SPICE/DA signal data** | Not accessible in research environment (import/data access issue) | High — these are forward-looking constraint forecasts, fundamentally different from backward-looking DA |
| **Daily DA volatility** | pnode_id format mismatch between daily loader and trade data; needs debugging | Low — monthly features already showed no signal |
| **Cross-path similarity** | Would need clustering/embedding of path characteristics | Unknown |
| **External data** (gas prices, load forecasts) | Not in current data pipeline | Medium — could explain year-to-year bias variation |

---

## Experiment F: Direction Accuracy by |MCP| Magnitude

### Motivation

When |MCP| is near zero, "getting the direction right" is meaningless — a $2 path predicted at -$1 is "wrong direction" but the error is negligible. Filtering by |MCP| threshold shows true predictive power on paths that matter.

### Results: H Baseline

| |MCP| Threshold | n | Dir Acc | Bias | Mean |Res| | p95 |
|---:|---:|---:|---:|---:|---:|
| >= 0 (all) | 636,165 | 67.4% | +417 | 898 | 3,503 |
| >= 50 | 548,582 | 69.8% | +466 | 1,016 | 3,819 |
| >= 100 | 496,194 | 71.6% | +504 | 1,104 | 4,041 |
| >= 250 | 390,561 | 75.7% | +607 | 1,336 | 4,588 |
| >= 500 | 286,859 | 79.8% | +770 | 1,682 | 5,335 |
| >= 1000 | 180,069 | 84.8% | +1,084 | 2,309 | 6,571 |

### Results: By |H| Threshold

| |H| Threshold | n | Dir Acc | Bias | Mean |Res| | p95 |
|---:|---:|---:|---:|---:|---:|
| >= 0 (all) | 636,165 | 67.4% | +417 | 898 | 3,503 |
| >= 50 | 525,677 | 70.7% | +463 | 1,011 | 3,875 |
| >= 100 | 433,626 | 73.7% | +519 | 1,143 | 4,289 |
| >= 250 | 272,218 | 80.0% | +672 | 1,509 | 5,347 |
| >= 500 | 160,189 | 85.0% | +896 | 1,999 | 6,717 |
| >= 1000 | 76,499 | 90.6% | +1,274 | 2,816 | 8,847 |

### Interpretation

H's direction accuracy is quite good (85-91%) for paths with |H| >= 500. The 67% headline number is dragged down by the ~180k paths with near-zero H where the baseline is effectively random. For paths where H has a meaningful signal, it correctly predicts direction — the problem is magnitude, not direction.

---

## Experiment G: Nodal MCP Stitching

### Motivation

Experiment A found prior-year R1 MCP beats H, but only 32% of paths have a prior-year path match. MISO publishes **per-node MCPs** (not per-path). Since `path_MCP = sink_node_MCP - source_node_MCP`, we can reconstruct prior-year MCPs for ANY path where both nodes have data — regardless of whether that exact path was traded last year.

### Method

1. Load nodal MCPs via `MisoCalculator.get_mcp_df()` for each delivery month
2. Extract the annual R1 column (identified from `mcp_auction_df` metadata)
3. Average across delivery months within each quarter
4. For each current-year R1 path, compute `prior_year_path_MCP = sink_node_MCP(PY-1) - source_node_MCP(PY-1)`
5. Handle node renames via `MisoNodalReplacement` (920 replacements in database, 2013-2025)

### Coverage

| PY | Total Paths | Nodal Coverage | Old Path-Match |
|---:|---:|---:|---:|
| 2020 | 67,183 | 60,197 (89.6%) | ~32% |
| 2021 | 90,902 | 83,073 (91.4%) | ~32% |
| 2022 | 83,416 | 77,207 (92.6%) | ~32% |
| 2023 | 106,208 | 97,750 (92.0%) | ~32% |
| 2024 | 105,608 | 96,437 (91.3%) | ~32% |
| 2025 | 117,311 | 108,595 (92.6%) | ~32% |

**Coverage: 32% → 92%.** The remaining 8% are paths with at least one node that didn't exist in the prior year's nodal MCP data.

### Results: Nodal Prior-MCP vs H (Per-PY)

| PY | n | Method | Bias | Mean |Res| | p95 | Dir Acc |
|---:|---:|---|---:|---:|---:|---:|
| 2020 | 60,197 | Nodal | +91 | 409 | 1,535 | 72.9% |
| 2020 | 60,197 | H | +143 | 376 | 1,393 | 71.9% |
| 2021 | 83,073 | Nodal | +262 | 873 | 3,743 | 72.0% |
| 2021 | 83,073 | H | +281 | 773 | 3,073 | 68.9% |
| 2022 | 77,207 | Nodal | +743 | 1,565 | 6,835 | 73.2% |
| 2022 | 77,207 | H | +631 | 1,371 | 5,667 | 63.8% |
| 2023 | 97,750 | Nodal | +453 | 1,085 | 4,144 | 71.8% |
| 2023 | 97,750 | H | +584 | 1,022 | 3,663 | 66.5% |
| 2024 | 96,437 | Nodal | +173 | 938 | 3,813 | 76.1% |
| 2024 | 96,437 | H | +479 | 1,018 | 4,375 | 69.6% |
| 2025 | 108,595 | Nodal | +263 | 854 | 3,334 | 78.2% |
| 2025 | 108,595 | H | +428 | 964 | 3,619 | 66.0% |

### Aggregate

| Method | n | Bias | Mean |Res| | Median |Res| | p95 | Dir Acc |
|---|---:|---:|---:|---:|---:|---:|
| Nodal prior-MCP | 523,259 | +333 | 969 | 370 | 3,923 | **74.3%** |
| H baseline | 523,259 | +440 | 947 | 404 | **3,708** | 67.6% |

### Direction Accuracy by |MCP| Threshold

| |MCP| >= | n | Dir Acc (Nodal) | Dir Acc (H) | Nodal Advantage |
|---:|---:|---:|---:|---:|
| 0 | 523,259 | 74.3% | 67.6% | +6.7pp |
| 50 | 453,214 | 76.5% | 69.8% | +6.7pp |
| 100 | 411,884 | 77.8% | 71.5% | +6.3pp |
| 250 | 328,046 | 80.2% | 75.4% | +4.8pp |
| 500 | 244,446 | 82.8% | 79.5% | +3.3pp |
| 1000 | 156,388 | 85.6% | 84.5% | +1.1pp |

Nodal advantage is strongest for medium-|MCP| paths (+6-7pp) and narrows at high |MCP| where both converge.

### Blend: α × Nodal + (1-α) × H

| α (nodal weight) | Bias | Mean |Res| | p95 | Dir Acc |
|---:|---:|---:|---:|---:|
| 0.00 (H only) | +440 | 947 | 3,708 | 67.6% |
| **0.25** | **+413** | **914** | **3,639** | **71.6%** |
| 0.50 | +386 | 906 | 3,652 | 74.1% |
| 0.75 | +360 | 925 | 3,749 | 75.8% |
| 1.00 (Nodal only) | +333 | 969 | 3,923 | 74.3% |

Optimal blend by p95: **α=0.25** (p95 = 3,639, -1.9% vs H-only, dir acc +4pp).

### Why Nodal p95 Is Worse Than Path-Match p95

| Method | Coverage | p95 | Dir Acc |
|---|---:|---:|---:|
| Path-match (Exp A) | 32% | 2,624 | 80.9% |
| Nodal stitch (Exp G) | 92% | 3,923 | 74.3% |

The path-match's lower p95 is **selection bias**: the 32% of paths that persist year-to-year are the stable, predictable ones. Nodal stitching includes all paths — including volatile/new ones where year-over-year MCP changes are large. The path-match result overstates the method's quality; the nodal result is more representative.

### Verdict

**Nodal stitching solves the coverage problem (32% → 92%) and provides a consistent +6pp direction accuracy advantage over H.** However, it does NOT improve magnitude accuracy — p95 is 6% worse than H. The prior year's auction outcome tells you WHICH DIRECTION the market will clear, but year-over-year magnitude shifts are too large for it to improve tail coverage.

**Recommended blend: 25% nodal prior-MCP + 75% H** — captures the directional advantage while keeping H's tighter magnitude.

---

## 8. Review Session (2026-02-12)

### Reviewer Context

Post-analysis QA review requested after observing that several theory conclusions looked suspicious despite plausible narrative framing.

### Findings (Consistency and Method Risk)

| Severity | Finding | Evidence |
|---|---|---|
| High | **Coverage contradiction for prior-year MCP**: Experiment A reports ~32% coverage, while Experiment E implies ~14% overall matched coverage and 10-16% for PY 2021-2025. | Sec 2 coverage table vs Sec 6 rows `Prior-year MCP (32% matched only)` and Per-PY Impact table |
| High | **Combined model framing conflict**: table shows combined p95 = 3,300 vs corrected-H p95 = 3,287 (worse), but verdict text frames combined as improvement. | Sec 6 Results and Verdict |
| High | **Sign convention error in Per-PY change column**: 2020 p95 rises 1,400 -> 1,439 but listed as -2.8% (hurt). | Sec 6 Per-PY Impact |
| High | **Apples-to-oranges improvement claim**: 50/50 blend gain is presented vs 3,045 baseline, but blend table baseline on the matched subset is 2,236 at alpha=0. | Sec 6 blending table and summary table |
| Medium | **Shrinkage sweep is approximate**, not exact recomputation using the production profitable-direction logic. | Sec 3 Method note |
| Medium | **Formula mismatch risk** between validated correction form and recommended deployment expression. | Sec 4 method (`H + bias`) vs Sec 7 recommendation (`H/0.85 + bias`) |

### Reviewer Verdict

The document is useful as exploratory research, but not yet decision-grade for production baseline changes. Directional conclusions may still be valid, but headline performance claims must be recomputed under a single, consistent sample definition.

### Required Fixes Before Production Adoption

1. Recompute all Sec 6 metrics from one canonical dataset and one canonical filter, then regenerate every coverage and p95 figure.
2. Standardize reporting units and sign conventions in the Per-PY change table.
3. Rebuild summary claims so each improvement is compared against the correct in-sample/out-of-sample baseline population.
4. Re-run shrinkage test with exact production logic (not the linear approximation).
5. Lock one deployable formula for `H_corrected` and ensure experiment code + summary text use the same expression.

### Suggested Next Review Session

After recomputation, run a second QA session focused on:
- reproducibility (single script reproduces all tables),
- leakage checks for all out-of-sample claims,
- stability by PY regime (low-bias vs high-bias years),
- deployment guardrails (fallback behavior when prior-year MCP unavailable).

---

## 9. Comprehensiveness Gaps and Expansion Plan

### Why Current Research Is Not Comprehensive

Most experiments vary only how prior-year monthly DA congestion is aggregated. That is a single signal family with limited structural information. As a result, the work is strong on "what does not help within DA-only transforms," but weak on broader model-space coverage.

### Missing Dimensions

| Dimension | Current Status | Gap |
|---|---|---|
| Feature breadth | Mostly DA monthly transforms (mean/percentile/weighting/trend/std) | No forward-looking fundamental drivers included |
| Market microstructure | Limited prior-year MCP carryover | No bid/award/competition structure features |
| Constraint context | Indirectly via DA congestion only | No explicit constraint-level mapping or shift-factor state |
| Regime awareness | Acknowledged low/high-bias years | No formal regime model or adaptive calibration |
| Robustness testing | Basic train/test and some LOO | No stress tests by topology/event regimes |
| Economic utility | Error metrics (p95, mean abs, direction) | No PnL utility/backtest under bidding rules |
| Statistical confidence | Point estimates only | No confidence intervals / significance testing |
| Reproducibility | Narrative tables | No single reproducible pipeline artifact |

### Required Expansion Tracks

1. Add forward-looking features (SPICE/constraint forecasts, load/weather, fuel spreads) and evaluate incremental lift over DA-only baseline.
2. Add market-structure features (path liquidity, participation persistence, prior award density, clearing depth proxies).
3. Add path/constraint topology features (voltage class, hub/interface relationship, outage/upgrade proximity where available).
4. Add regime modeling (year-level and quarter-level intercept adaptation; change-point or clustered regimes).
5. Add economic evaluation (simulate bid bands, hit rate, realized spread capture, tail-loss control), not only prediction error.
6. Add uncertainty quantification (bootstrap CIs for p95/dir acc improvements; reject changes without statistically credible lift).
7. Build one end-to-end reproducible script/notebook that regenerates every table in this report from raw inputs.

### Minimum Standard for "Comprehensive" R1 Baseline Research

The research should not be called comprehensive unless all conditions below are met:
- At least one non-DA forward-looking feature family is tested out-of-sample.
- At least one market-structure feature family is tested out-of-sample.
- Performance is reported in both error metrics and economic utility metrics.
- Results include uncertainty bounds and regime-segmented performance.
- All tables are reproducible from a single versioned pipeline.

### Priority Execution Order

1. Recompute and fix internal inconsistencies from Session 8.
2. Integrate forward-looking signal data (highest expected lift).
3. Add economic backtest framework for bid decision relevance.
4. Add regime-aware calibration and uncertainty reporting.
5. Publish v2 report only after reproducibility checks pass.
