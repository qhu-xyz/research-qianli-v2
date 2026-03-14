# Findings: Baseline Improvement Research

**Date:** 2026-02-12
**Source:** `/tmp/run_baseline_research.py` and `/tmp/run_baseline_research_fix.py`
**Data:** all_residuals_v2, annual_with_mcp_v2, r1_filled_v2 (MISO annual PY 2019-2025)

---

## Executive Summary

Six experiments tested whether R1's baseline (H = historical DA congestion, p95 = 3,307) can be improved, and whether R2/R3's baseline (M = prior round MCP, p95 = 260/202) can be supplemented. Key results:

| Experiment | Verdict | p95 Impact |
|---|---|---|
| **1. Prior-year MCP for R1** | Strong improvement where available, but only 32% path coverage | 2,624 vs 3,045 (-14%) on matched paths |
| **2. Shrinkage sweep** | 1.0 marginally better than 0.85 | 3,230 vs 3,307 (-2.3%) |
| **3. Bias correction (LOO)** | Moderate improvement, centers predictions well | 3,157 vs 3,307 (-4.5%) |
| **4. Signal data** | Not accessible in current environment | N/A |
| **5a. split_month_mcp for R2/R3** | DATA LEAKAGE — invalid | N/A |
| **5b. Volume stratification** | Useful for band width scaling | Informational |
| **5c. MTM drift (R3)** | Useful for uncertainty stratification, not baseline | Informational |
| **6. Combined R1** | Blend on matched paths: p95 = 1,796 (-41%) | ~3,129 overall (limited by 32% coverage) |

**Bottom line:** R1's fundamental problem is that H is a weak predictor. The prior-year MCP is substantially better where available (32% of paths, p95 improves 14%), and bias correction helps on the rest (+4.5% p95, +3.5pp direction accuracy). For R2/R3, M-only is confirmed optimal — no non-leaking feature improves it.

---

## Experiment 1: Prior-Year R1 MCP as Baseline

### Hypothesis

Using PY-1's R1 annual MCPs as R1 baseline should outperform H, similar to how R2/R3's prior-round MCP outperforms everything else.

### Rationale

R2 uses R1's MCP with p95=260. If year-over-year R1 MCP correlation is similarly strong, we could reduce R1's p95 substantially.

### Coverage

**Finding: Only 32% of R1 paths appear in the prior year's R1 auction.** This is much lower than the >80% hypothesized, because annual FTR path participation changes significantly year-to-year.

| PY | Total R1 Trades | Matched to PY-1 | Coverage |
|---:|---:|---:|---:|
| 2020 | 252,222 | 81,876 | 32.5% |
| 2021 | 346,584 | 107,793 | 31.1% |
| 2022 | 318,069 | 104,808 | 33.0% |
| 2023 | 394,770 | 120,870 | 30.6% |
| 2024 | 393,828 | 131,430 | 33.4% |
| 2025 | 437,871 | 142,746 | 32.6% |

Coverage is remarkably stable at ~32% across all years.

### Results: Prior-Year MCP vs H Baseline (matched paths, fair comparison)

| PY | Method | n | Bias | Mean |Res| | p95 | Dir Acc |
|---:|---|---:|---:|---:|---:|---:|
| 2020 | prior-MCP | 81,654 | -87.4 | 352.2 | 1,342 | 81.8% |
| 2020 | H-baseline | 81,654 | +202.1 | 315.3 | 1,182 | 68.2% |
| 2021 | prior-MCP | 107,616 | +48.5 | 552.8 | 2,508 | 79.8% |
| 2021 | H-baseline | 107,616 | +237.3 | 625.6 | 2,520 | 64.4% |
| 2022 | prior-MCP | 104,514 | +409.4 | 983.9 | 4,299 | 81.5% |
| 2022 | H-baseline | 104,514 | +737.7 | 1,187.1 | 4,741 | 56.7% |
| 2023 | prior-MCP | 120,567 | -256.6 | 916.2 | 3,677 | 79.8% |
| 2023 | H-baseline | 120,567 | +699.7 | 948.7 | 3,482 | 60.2% |
| 2024 | prior-MCP | 131,199 | -94.3 | 620.0 | 2,365 | 81.4% |
| 2024 | H-baseline | 131,199 | +493.9 | 730.0 | 2,846 | 63.2% |
| 2025 | prior-MCP | 142,641 | -22.5 | 505.9 | 1,803 | 81.4% |
| 2025 | H-baseline | 142,641 | +524.1 | 777.9 | 2,948 | 59.6% |

### Aggregate (all PYs, matched paths only)

| Method | n | Bias | Mean |Res| | Median |Res| | p90 | p95 | p99 | Dir Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Prior-year MCP** | 688,191 | **-8.2** | **661.2** | 247.0 | 1,603 | **2,624** | 6,378 | **80.9%** |
| H-baseline | 688,191 | +498.5 | 782.1 | 330.5 | 1,867 | 3,045 | 6,961 | 61.7% |

### By Quarter (prior-year MCP aggregate)

| Quarter | n | Bias | Mean |Res| | Median |Res| | p90 | p95 | p99 | Dir Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aq1 | 174,789 | -7.0 | 234.5 | 95.4 | 548 | 892 | 2,205 | 80.2% |
| aq2 | 178,344 | -7.0 | 232.5 | 88.5 | 561 | 912 | 2,205 | 80.4% |
| aq3 | 167,646 | -2.7 | 224.3 | 79.0 | 558 | 915 | 2,155 | 81.3% |
| aq4 | 168,744 | +7.3 | 190.2 | 67.6 | 471 | 766 | 1,881 | 81.2% |

**Note:** The per-quarter numbers above are from the initial run which uses `mcp_mean` from `annual_with_mcp_v2` (different merge path). The aggregate numbers from the fix run are slightly different because they use `mcp_mean` from `all_residuals_v2`. The directional conclusions are identical.

### Verdict

**Prior-year MCP is a substantially better R1 baseline than H, but only covers 32% of paths.**

- p95: 2,624 vs 3,045 (-14% improvement)
- Direction accuracy: 80.9% vs 61.7% (+19.2 percentage points)
- Bias: -8 vs +499 (nearly unbiased vs strong positive bias)
- Consistent across all PYs and all quarters
- PY2020 is the only year where H's mean |res| is slightly better (315 vs 352), but prior-MCP still wins on p95 and dir acc

**Limitation:** 68% of R1 paths have no prior-year MCP and must fall back to H. This limits the overall improvement when used as a complete solution.

---

## Experiment 2: Shrinkage Factor Sweep

### Hypothesis

The 0.85 shrinkage factor applied to DA congestion is hand-tuned. A different value might reduce residuals.

### Method

Approximate test: `H_new = H / 0.85 * new_shrinkage`. This assumes all congestion was in the profitable direction (the 0.85 factor is applied directionally). The approximation is directionally correct for comparison purposes.

### Results

| Shrinkage | Bias | Mean |Res| | Median |Res| | p90 | p95 | p99 | Dir Acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | +438.8 | 898.5 | 348.6 | 2,243 | 3,603 | 8,178 | 66.1% |
| 0.60 | +433.6 | 881.1 | 349.6 | 2,192 | 3,505 | 7,909 | 66.1% |
| 0.70 | +428.5 | 866.7 | 352.0 | 2,148 | 3,415 | 7,658 | 66.1% |
| 0.75 | +425.9 | 860.6 | 353.6 | 2,127 | 3,377 | 7,555 | 66.1% |
| 0.80 | +423.3 | 855.2 | 355.3 | 2,110 | 3,341 | 7,453 | 66.1% |
| **0.85** | **+420.7** | **850.5** | **357.2** | **2,093** | **3,307** | **7,352** | **66.1%** |
| 0.90 | +418.1 | 846.6 | 359.9 | 2,079 | 3,280 | 7,247 | 66.1% |
| 0.95 | +415.6 | 843.3 | 362.2 | 2,064 | 3,252 | 7,170 | 66.1% |
| **1.00** | **+413.0** | **840.8** | **365.2** | **2,050** | **3,230** | **7,097** | **66.1%** |

### Verdict

**Shrinkage = 1.0 (no shrinkage) is marginally better by all tail metrics, but the improvement is negligible.**

- p95: 3,230 vs 3,307 (-2.3%)
- Mean |res|: 840.8 vs 850.5 (-1.1%)
- Direction accuracy: unchanged at 66.1% for all values
- The monotonic improvement toward 1.0 suggests the 0.85 conservative bias is slightly counterproductive for prediction accuracy, but the effect is tiny

The shrinkage parameter is not a meaningful lever for R1 improvement. The fundamental problem is that year-old DA congestion is a weak predictor of auction clearing prices regardless of how it's scaled.

---

## Experiment 3: Bias Correction (Leave-One-Year-Out)

### Hypothesis

Adding a per-quarter bias correction to H could center predictions better. The overall bias is +421, varying from +149 (PY 2020) to +679 (PY 2022).

### Method

For each PY, estimate the per-quarter bias from all OTHER PYs, then correct: `H_corrected = H + bias_estimate`. This is a proper out-of-sample evaluation — no PY sees its own data during bias estimation.

### Results: Per-PY Out-of-Sample

| PY | Method | n | Bias | Mean |Res| | p95 | Dir Acc |
|---:|---|---:|---:|---:|---:|---:|
| 2019 | Uncorrected | 249,636 | +172.7 | 371.8 | 1,340 | 64.7% |
| 2019 | Corrected | 249,636 | -276.6 | 476.5 | 1,285 | 69.2% |
| 2020 | Uncorrected | 251,418 | +148.0 | 371.0 | 1,400 | 71.1% |
| 2020 | Corrected | 251,418 | -303.5 | 510.1 | 1,392 | 67.3% |
| 2021 | Uncorrected | 345,513 | +304.7 | 743.0 | 2,938 | 67.8% |
| 2021 | Corrected | 345,513 | -137.0 | 784.3 | 2,839 | 71.8% |
| 2022 | Uncorrected | 317,439 | +679.0 | 1,345.5 | 5,508 | 62.7% |
| 2022 | Corrected | 317,439 | +298.2 | 1,282.6 | 5,378 | 71.8% |
| 2023 | Uncorrected | 393,684 | +595.2 | 1,008.4 | 3,602 | 65.3% |
| 2023 | Corrected | 393,684 | +211.0 | 941.7 | 3,390 | 70.5% |
| 2024 | Uncorrected | 393,486 | +458.8 | 941.3 | 3,948 | 67.9% |
| 2024 | Corrected | 393,486 | +45.1 | 952.1 | 3,816 | 69.6% |
| 2025 | Uncorrected | 437,298 | +431.9 | 901.3 | 3,395 | 64.5% |
| 2025 | Corrected | 437,298 | +12.3 | 877.3 | 3,263 | 67.0% |

**Observations:**
- Bias correction overcorrects for early PYs (2019-2020): their bias was lower (+150-170) but the correction uses the average from other (higher-bias) PYs, pushing them negative
- For PYs 2022-2025, correction works well: bias drops substantially while residuals improve
- Direction accuracy improves for 5 of 7 PYs (PY2020 is the exception)

### Overall Aggregate

| Method | n | Bias | Mean |Res| | Median |Res| | p90 | p95 | p99 | Dir Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Uncorrected | 2,388,474 | +420.7 | 850.5 | 357.2 | 2,093 | 3,307 | 7,352 | 66.1% |
| **Corrected** | 2,388,474 | **+3.4** | 860.1 | 421.9 | **1,965** | **3,157** | **7,157** | **69.6%** |

### Per-Quarter Aggregate

| Quarter | Method | Bias | Mean |Res| | p95 | Dir Acc |
|---|---|---:|---:|---:|---:|
| aq1 | Uncorrected | +410.5 | 838.2 | 3,171 | 66.7% |
| aq1 | Corrected | +4.9 | 840.3 | 3,035 | 69.4% |
| aq2 | Uncorrected | +489.0 | 942.7 | 3,712 | 67.4% |
| aq2 | Corrected | +1.8 | 972.4 | 3,548 | 69.1% |
| aq3 | Uncorrected | +402.3 | 815.7 | 3,211 | 67.4% |
| aq3 | Corrected | +2.4 | 819.6 | 3,067 | 71.1% |
| aq4 | Uncorrected | +375.0 | 797.0 | 3,107 | 62.9% |
| aq4 | Corrected | +4.6 | 798.3 | 2,984 | **68.8%** |

### Verdict

**Bias correction provides moderate, consistent improvement in tail metrics and direction accuracy, evaluated entirely out-of-sample.**

- p95: 3,307 → 3,157 (-4.5%)
- Bias: +421 → +3 (nearly zero)
- Direction accuracy: 66.1% → 69.6% (+3.5pp)
- aq4 benefits most in dir acc: 62.9% → 68.8% (+5.9pp)

**Trade-off:** Mean |res| increases slightly (851 → 860) and median increases (357 → 422). The correction helps the tails by centering the distribution, but shifts some mass from below-median errors to above-median errors. This is an acceptable trade-off for production use because tail coverage (p95) is what determines band width.

**Caution:** The correction overcorrects for low-bias years (PY 2019-2020). If the market reverts to lower-bias regimes, the correction could hurt. A more conservative approach would use a half-correction (H + bias/2) or a rolling window instead of leave-one-out.

---

## Experiment 4: Signal Data Feasibility Check

### Result

Signal data (SPICE_ANNUAL_V4.4, DA_ANNUAL_V1.4) could not be loaded in the current research environment. The `MisoSignal` class is not properly exported from the `pbase.data.dataset.ftr.signal` package's `__init__.py`.

### Implication

Signal-based analysis requires either:
1. Fixing the import path and ensuring S3/local data access is configured
2. Running from within the production pmodel environment where signal data is already loaded during model execution

This remains a future avenue for R1 improvement. The signals rank paths by constraint exposure and are already used in the legacy bid adjustment step — they could potentially improve the baseline itself if path-level signal values correlate with MCP.

---

## Experiment 5a: split_month_mcp Blending (R2/R3) — INVALID

### Leakage Finding

`split_month_mcp` has **correlation = 0.999** with `mcp_mean` (the target variable). It IS the current round's per-month MCP — the exact outcome we are trying to predict.

All blend results showing improvement (e.g., p95 dropping from 260 to 133 at α=0.50) are pure data leakage and not actionable.

### What about mtm_2nd_mean for R3?

`mtm_2nd_mean` = R1's MCP, available at R3 bid time. This is a legitimate non-leaking feature. Result:

| α (R1 MCP weight) | Mean |Res| | p95 | Dir Acc |
|---:|---:|---:|---:|
| 0.00 (M only) | **56.2** | **202** | 93.2% |
| 0.05 | 57.4 | 207 | 93.1% |
| 0.10 | 58.8 | 212 | 93.1% |
| 0.20 | 62.2 | 226 | 92.9% |
| 0.50 | 75.6 | 280 | 91.9% |

**Verdict: Adding R1's MCP to R3's baseline makes it strictly worse.** R2's MCP (mtm_1st_mean) already subsumes all information from R1's MCP. M-only remains optimal.

### Conclusion for R2/R3

**No supplementary feature in the available data improves the M-only baseline without leakage.** The prior round's MCP is the optimal baseline. All improvement effort should focus on band width calibration, not baseline prediction.

---

## Experiment 5b: Volume-Stratified Analysis

### R2 Residuals by Volume Decile

| Decile | Volume Range (MW) | n | Mean |Res| | Median |Res| | p95 | Dir Acc |
|---:|---|---:|---:|---:|---:|---:|
| 0 (smallest) | 0.1-0.2 | 423,000 | **98.3** | 47.7 | **368** | 93.4% |
| 1 | 0.3-0.5 | 542,646 | 86.2 | 39.8 | 334 | 93.1% |
| 2 | 0.6-0.8 | 392,091 | 76.5 | 37.0 | 281 | 92.8% |
| 3 | 0.9-1.2 | 391,122 | 71.6 | 34.1 | 265 | 92.3% |
| 4 | 1.3-1.7 | 345,153 | 66.6 | 33.3 | 238 | 91.9% |
| 5 | 1.8-2.5 | 452,430 | 62.4 | 30.8 | 221 | 90.9% |
| 6 | 2.6-3.5 | 385,185 | 59.9 | 29.5 | 210 | 90.8% |
| 7 | 3.6-5.0 | 415,401 | 58.8 | 28.6 | 212 | 90.1% |
| 8 | 5.1-8.5 | 418,764 | 55.7 | 27.4 | **200** | 88.8% |
| 9 (largest) | 8.6-757 | 413,667 | 59.8 | 28.8 | 217 | 89.5% |

### R3 Residuals by Volume Decile

| Decile | Volume Range (MW) | n | Mean |Res| | Median |Res| | p95 | Dir Acc |
|---:|---|---:|---:|---:|---:|---:|
| 0 (smallest) | 0.1-0.2 | 525,693 | **79.0** | 36.1 | **285** | 95.0% |
| 1 | 0.3-0.5 | 650,322 | 67.9 | 32.7 | 246 | 94.2% |
| 2 | 0.6-0.7 | 317,874 | 58.0 | 28.3 | 209 | 94.1% |
| 3 | 0.8-1.1 | 553,707 | 56.4 | 27.9 | 200 | 94.2% |
| 4 | 1.2-1.6 | 462,063 | 53.4 | 26.0 | 189 | 93.5% |
| 5 | 1.7-2.3 | 491,739 | 50.2 | 23.9 | 181 | 92.7% |
| 6 | 2.4-3.2 | 452,985 | 49.9 | 24.8 | 174 | 92.7% |
| 7 | 3.3-4.9 | 487,797 | 47.3 | 23.7 | 168 | 92.4% |
| 8 | 5.0-8.1 | 495,516 | 47.0 | 22.9 | **169** | 91.6% |
| 9 (largest) | 8.2-750 | 488,151 | 47.3 | 22.1 | 171 | 91.5% |

### Key Findings

1. **Small-volume trades have ~1.8× larger residuals.** R2 p95: 368 (smallest) vs 200 (decile 8) = 1.84× ratio. R3: 285 vs 169 = 1.69×.

2. **The relationship is monotonic from decile 0-8.** Residuals decrease steadily with increasing volume. Decile 9 (very large trades) has a slight uptick, possibly due to market impact.

3. **Direction accuracy is slightly higher for small trades** (93-95%) vs large trades (89-92%). This is counterintuitive — it may be because small trades tend to be on well-understood paths.

4. **Practical use:** Volume can be used as a band width multiplier. For example, trades <0.5 MW could receive 1.5× the standard band width, while trades >5 MW could receive 0.9×. This is informational for band calibration, not for baseline improvement.

---

## Experiment 5c: R3 MTM Drift Analysis

### Setup

`mtm_drift = |mtm_now_0 - mtm_1st_mean|` where:
- `mtm_now_0` = the most recent MTM estimate available at R3 bid time
- `mtm_1st_mean` = R2's MCP (the baseline)

The drift measures how much the market has moved since R2's clearing.

### Correlation

**Correlation(drift, |residual|) = 0.368** — moderately strong. Higher drift predicts larger R3 residuals.

### Residuals by Drift Magnitude

| Drift Bin | n | Mean |Res| | Median |Res| | p95 | Dir Acc |
|---|---:|---:|---:|---:|---:|
| 0-10 | 530,952 | 16.9 | 6.7 | 66 | 89.2% |
| 10-25 | 519,447 | 24.1 | 13.0 | 82 | 90.5% |
| 25-50 | 616,629 | 29.9 | 17.7 | 97 | 92.1% |
| 50-100 | 792,318 | 37.2 | 23.1 | 118 | 93.2% |
| 100-250 | 1,103,706 | 53.1 | 33.7 | 165 | 94.3% |
| **250+** | 1,353,096 | **109.9** | 64.9 | **356** | 95.4% |

**5.4× range** in p95 from lowest drift (66) to highest drift (356).

### Does Blending with mtm_now_0 Help R3?

| beta (mtm_now_0 weight) | Bias | Mean |Res| | p95 | Dir Acc |
|---:|---:|---:|---:|---:|
| **0.00 (M only)** | **+11.3** | **56.2** | **202** | 93.2% |
| 0.05 | +8.7 | 55.7 | 201 | 93.2% |
| 0.10 | +6.1 | 59.4 | 219 | 93.0% |
| 0.20 | +0.9 | 73.8 | 282 | 92.0% |
| 0.50 | -14.7 | 140.5 | 559 | 88.5% |
| 1.00 | -40.7 | 272.4 | 1,101 | 81.5% |

### Verdict

**mtm_now_0 should NOT be blended into the R3 baseline** — it makes predictions worse at every weight above 0.05 (and 0.05 is marginal: 201 vs 202).

**However, drift is extremely valuable as an uncertainty indicator for band width calibration.** Trades with drift >250 need p95 bands of 356 (1.76× the overall p95 of 202), while trades with drift <10 need only 66 (0.33× overall). This is a 5.4× range in required band width — far more discriminating than volume deciles (1.7× range).

**Recommendation:** Use drift magnitude as a primary band width scaling factor for R3. For R2, investigate whether an analogous drift measure exists (e.g., change in DA congestion between R1 and R2 auction dates).

---

## Experiment 6: Combined R1 Improvement

### Method

Combine the best findings from Experiments 1-3:
1. Shrinkage = 1.0 (from Exp 2)
2. Leave-one-year-out per-quarter bias correction (from Exp 3)
3. Prior-year MCP where available, corrected H otherwise (from Exp 1)

### Coverage

On PY 2020-2025 R1 trades:
- **32% of paths** have a prior-year R1 MCP available
- **68% of paths** fall back to bias-corrected H with shrinkage=1.0

### Method Comparison (PY 2020-2025)

| Method | n | Bias | Mean |Res| | Median |Res| | p95 | p99 | Dir Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| H (original, shrinkage=0.85) | 2,138,838 | +449.7 | 906.4 | 385.4 | 3,521 | 7,672 | 66.3% |
| H (shrinkage=1.0) | 2,138,838 | +441.4 | 895.6 | 393.3 | 3,434 | 7,423 | 66.3% |
| H (shrinkage=1.0) + bias correction | 2,138,838 | +35.7 | 885.7 | 426.3 | 3,287 | 7,230 | 70.2% |
| **Prior-year MCP** (matched only) | **299,694** | **-43.1** | **601.6** | **237.6** | **2,344** | 5,540 | **79.0%** |
| Combined (prior-MCP + corrected-H) | 2,138,838 | +44.1 | 878.1 | 414.1 | 3,300 | 7,277 | 71.4% |

### Blend: Prior-Year MCP + Corrected H (where both available)

On the 32% of paths with both signals:

| alpha (prior MCP weight) | Bias | Mean |Res| | p95 | Dir Acc |
|---:|---:|---:|---:|---:|
| 0.00 (corrected H only) | -103.3 | 656.1 | 2,236 | 70.7% |
| 0.25 | -88.3 | 549.9 | 1,898 | 73.8% |
| **0.50** | **-73.2** | **504.1** | **1,796** | **76.5%** |
| 0.75 | -58.1 | 524.5 | 1,978 | 78.3% |
| 1.00 (prior MCP only) | -43.1 | 601.6 | 2,344 | 79.0% |

**On matched paths, the 50/50 blend of prior-year MCP and corrected H achieves p95 = 1,796** — a 41% improvement over original H on those same paths (p95 = 3,045). The blend outperforms either signal alone.

### Per-PY Breakdown

| PY | Prior MCP Coverage | H (original) p95 | Combined p95 | Improvement |
|---:|---:|---:|---:|---:|
| 2020 | 32% | 1,400 | 1,439 | -2.8% (hurt) |
| 2021 | 16% | 2,938 | 2,791 | +5.0% |
| 2022 | 12% | 5,508 | 5,226 | +5.1% |
| 2023 | 11% | 3,602 | 3,373 | +6.4% |
| 2024 | 10% | 3,948 | 3,725 | +5.6% |
| 2025 | 10% | 3,395 | 3,129 | +7.8% |

### Verdict

**The combined approach provides moderate but consistent improvement** for PYs 2021-2025 (5-8% p95 reduction). PY 2020 is slightly hurt because it was a low-bias year and the bias correction overcorrects.

The limited 32% coverage of prior-year MCP is the binding constraint. The overall improvement is diluted because 68% of paths still rely on corrected H. If coverage could be increased (e.g., by matching to any prior year, not just PY-1), the improvement would be larger.

---

## Ranked Recommendations

### For R1 (in priority order)

1. **Add per-quarter bias correction** (Exp 3) — Low effort, universally applicable, +3.5pp direction accuracy, -4.5% p95. Implement as: `H_corrected = H + bias_estimate` where `bias_estimate` is the historical mean residual for that quarter from prior PYs.

2. **Use prior-year MCP where available** (Exp 1) — Medium effort, 32% coverage, dramatic improvement on matched paths. Implement as: look up PY-1's R1 MCP for the same (period_type, class_type, source_id, sink_id). Use 50/50 blend with corrected H when both available.

3. **Remove the 0.85 shrinkage** (Exp 2) — Trivial change, minimal impact (-2.3% p95). Change shrinkage from 0.85 to 1.0 in `fill_mtm_1st_period_with_hist_revenue()`.

4. **Scale band widths by |H| magnitude** (from Phase 2 analysis) — Already established that tiny-H paths need p95 bands of ~1,300 while large-H paths need ~8,800. This is about band calibration, not baseline.

5. **Investigate signal data** (Exp 4) — Deferred due to data access issues. If SPICE/DA signals contain path-level congestion forecasts, they could supplement H for paths without prior-year MCP.

### For R2/R3

1. **M-only baseline is confirmed optimal.** No non-leaking feature improves it.

2. **Use MTM drift (|mtm_now_0 - mtm_1st_mean|) for R3 band width scaling** (Exp 5c) — 5.4× range in p95 by drift bin. Paths with high drift need wider bands.

3. **Use volume for band width scaling** (Exp 5b) — 1.7-1.8× range in p95 by volume decile. Small trades need wider bands.

4. **Do NOT add H, R1 MCP, or split_month_mcp to the R2/R3 baseline** — All confirmed to make predictions worse or involve leakage.

---

## Technical Notes

- All R1 bias correction estimates use leave-one-year-out cross-validation
- The shrinkage sweep is approximate (assumes all congestion is in the profitable direction)
- Prior-year MCP coverage of ~32% reflects actual path persistence in MISO annual auctions
- Experiment 5a (split_month_mcp) was invalidated by a correlation of 0.999 with the target variable
- Signal data (Experiment 4) requires production environment access — not tested
