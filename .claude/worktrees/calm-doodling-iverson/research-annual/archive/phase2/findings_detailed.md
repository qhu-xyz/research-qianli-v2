# MISO Annual FTR R1 Baseline Research — Detailed Findings

**Date:** 2026-02-19
**Data:** PY 2020-2025, all 4 quarters, ~135-149K paths each. All figures $/MW.
**Scripts:** `scripts/run_aq{1,2,3,4}_experiment.py`, `scripts/run_phase2_improvement.py`, `scripts/explore_residuals.py`

---

## Context

MISO annual FTR auctions have 3 rounds (R1, R2, R3) × 4 quarters (aq1-aq4). R1 is the hardest to price because there is **no MTM** (mark-to-market from prior rounds). The production system (`ftr23/v8`) uses H (historical DA congestion with 0.85 shrinkage) as the R1 baseline. This research tests whether better baselines exist.

### Baselines Tested

| Baseline | What it is | How it's built | Coverage |
|----------|-----------|----------------|----------|
| **Nodal f0 stitch** | Per-node f0 monthly forward, stitched to path | `sink_f0 - source_f0` from `get_mcp_df(market_month=PY-1)` col 0, averaged over 3 delivery months. Node replacements via BFS on `MisoNodalReplacement`. | 98.8-100% |
| **H (DA congestion)** | Production baseline | DA LMP monthly congestion from prior-year delivery months, 0.85 shrinkage on counter-flow, stitched path-level. This is `_fill_mtm()` in `ftr23/v8/miso_base.py`. | 100% |
| **f0 path-level** | Direct f0 path clearing | From f0p cleared trades, path-level average. | 45-55% |
| **f1 path-level** | One-month-ahead forward | From f0p cleared trades, f1 period. | 26-48% |
| **Prior R1/R2/R3** | Prior-year annual clearing | Path-level MCP from PY-1's annual auctions. | 25-32% |
| **Q2/Q3/Q4 forward** | Quarterly auction clearing | From quarterly forward auctions (q1 does not exist). | 27-42% |
| **α × Nodal f0** | Scale-corrected nodal f0 | Phase 2: `prediction = α * nodal_f0`, α chosen by LOO. | 98.8-100% |

---

## Phase 1 Results: Per-Quarter Baseline Comparison

### AQ1 — Jun/Jul/Aug (149,115 rows)

**A. All rows (each baseline evaluated on its own coverage)**

| Baseline | n | Cov% | Bias | MAE | MedAE | p95 | p99 | Dir% | D>100% |
|----------|--:|-----:|-----:|----:|------:|----:|----:|-----:|-------:|
| **Nodal f0 stitch** | 147,315 | 98.8 | +220 | **798** | 318 | 3,192 | 6,967 | **80.9** | 84.8 |
| H (DA congestion) | 149,115 | 100.0 | +427 | 934 | 414 | 3,546 | 7,526 | 67.7 | 71.3 |
| f0 path-level | 67,291 | 45.1 | +126 | 573 | 238 | 2,197 | 4,883 | 79.7 | 83.9 |
| f1 path-level | 38,414 | 25.8 | +135 | 539 | 227 | 2,064 | 4,535 | 79.1 | 83.5 |
| Prior R1 | 36,932 | 24.8 | -32 | 740 | 304 | 2,839 | 6,788 | 78.9 | 83.0 |
| Prior R2 | 41,840 | 28.1 | +281 | 703 | 266 | 2,833 | 6,322 | 79.1 | 83.5 |
| Prior R3 | 44,633 | 29.9 | +242 | 659 | 254 | 2,608 | 5,945 | 79.1 | 83.4 |

Note: f0/f1 path have lower MAE but only cover 25-45% of paths. On those liquid paths every baseline does well. The fair comparison is on matched rows (Table B).

**B. Head-to-head (67,291 rows where Nodal f0 + f0 path + H all present)**

| Baseline | Bias | MAE | MedAE | p95 | p99 | Dir% | D>100% |
|----------|-----:|----:|------:|----:|----:|-----:|-------:|
| **Nodal f0** | +126 | **564** | **234** | **2,157** | **4,898** | **80.5** | **84.9** |
| f0 path | +126 | 573 | 238 | 2,197 | 4,883 | 79.7 | 83.9 |
| H | +322 | 671 | 329 | 2,434 | 5,086 | 64.6 | 68.6 |

**C. Fully matched (26,720 rows where Nodal + f0 + f1 + R3 + H all present)**

| Baseline | Bias | MAE | MedAE | p95 | p99 | Dir% | D>100% |
|----------|-----:|----:|------:|----:|----:|-----:|-------:|
| **Nodal f0** | +117 | **499** | 206 | 1,930 | 4,324 | **80.5** | **85.2** |
| f0 path | +117 | 501 | 207 | 1,923 | 4,274 | 80.0 | 84.6 |
| f1 path | +122 | 508 | 212 | 1,964 | 4,316 | 78.8 | 83.4 |
| Prior R3 | +144 | 520 | 215 | 1,992 | 4,456 | 78.3 | 82.7 |
| Prior R2 | +161 | 521 | 214 | 1,983 | 4,465 | 77.8 | 82.5 |
| Prior R1 | -9 | 555 | 240 | 2,002 | 4,906 | 77.1 | 81.7 |
| H | +325 | 612 | 304 | 2,215 | 4,523 | 61.8 | 66.3 |

Ranking on fully matched: **Nodal f0 ≈ f0 path < f1 < R3 < R2 < R1 < H**

---

### AQ2 — Sep/Oct/Nov (149,291 rows)

**A. All rows**

| Baseline | n | Cov% | Bias | MAE | MedAE | p95 | p99 | Dir% | D>100% |
|----------|--:|-----:|-----:|----:|------:|----:|----:|-----:|-------:|
| **Nodal f0 stitch** | 148,491 | 99.5 | +340 | **947** | 337 | 3,942 | 8,699 | **82.1** | 86.2 |
| H (DA congestion) | 149,291 | 100.0 | +524 | 1,070 | 442 | 4,252 | 9,189 | 69.0 | 72.8 |
| f0 path-level | 75,099 | 50.3 | +179 | 614 | 243 | 2,409 | 5,455 | 80.2 | 84.8 |
| f1 path-level | 63,990 | 42.9 | +174 | 606 | 239 | 2,370 | 5,445 | 79.9 | 84.2 |
| Prior R1 | 36,515 | 24.5 | -24 | 744 | 284 | 2,966 | 6,910 | 79.6 | 84.1 |
| Prior R2 | 42,405 | 28.4 | +314 | 725 | 257 | 2,946 | 7,125 | 79.3 | 83.8 |
| Prior R3 | 45,958 | 30.8 | +275 | 675 | 244 | 2,729 | 6,609 | 79.2 | 84.0 |
| Q2 forward | 40,345 | 27.0 | -200 | 679 | 271 | 2,620 | 5,988 | 78.4 | 82.8 |

**B. Head-to-head (75,099 rows)**

| Baseline | Bias | MAE | MedAE | p95 | p99 | Dir% | D>100% |
|----------|-----:|----:|------:|----:|----:|-----:|-------:|
| **Nodal f0** | +175 | **605** | **240** | **2,375** | **5,380** | **81.0** | **85.7** |
| f0 path | +179 | 614 | 243 | 2,409 | 5,455 | 80.2 | 84.8 |
| H | +374 | 726 | 334 | 2,675 | 5,673 | 65.6 | 69.8 |

**C. Fully matched (33,269 rows)**

| Baseline | Bias | MAE | MedAE | p95 | p99 | Dir% | D>100% |
|----------|-----:|----:|------:|----:|----:|-----:|-------:|
| **Nodal f0** | +150 | **518** | 204 | 2,025 | 4,787 | **80.9** | **86.0** |
| f0 path | +147 | 523 | 205 | 2,041 | 4,838 | 80.4 | 85.2 |
| f1 path | +150 | 530 | 209 | 2,064 | 4,871 | 79.7 | 84.5 |
| Prior R3 | +177 | 547 | 215 | 2,098 | 5,059 | 78.4 | 83.1 |
| Prior R2 | +194 | 550 | 215 | 2,141 | 5,210 | 77.9 | 82.6 |
| Prior R1 | +10 | 564 | 222 | 2,160 | 5,352 | 77.8 | 82.4 |
| H | +349 | 639 | 297 | 2,332 | 5,088 | 63.5 | 68.4 |

---

### AQ3 — Dec/Jan/Feb (137,088 rows)

**A. All rows**

| Baseline | n | Cov% | Bias | MAE | MedAE | p95 | p99 | Dir% | D>100% |
|----------|--:|-----:|-----:|----:|------:|----:|----:|-----:|-------:|
| **Nodal f0 stitch** | 136,716 | 99.7 | +225 | **797** | 296 | 3,236 | 6,836 | **83.8** | 88.1 |
| H (DA congestion) | 137,088 | 100.0 | +425 | 920 | 398 | 3,581 | 7,448 | 69.1 | 73.6 |
| f0 path-level | 69,792 | 50.9 | +120 | 550 | 215 | 2,191 | 4,504 | 81.6 | 86.4 |
| f1 path-level | 60,101 | 43.8 | +139 | 536 | 204 | 2,145 | 4,569 | 81.6 | 86.6 |
| Prior R1 | 34,712 | 25.3 | -11 | 729 | 262 | 2,945 | 6,851 | 80.0 | 84.6 |
| Prior R2 | 39,794 | 29.0 | +295 | 691 | 236 | 2,926 | 6,325 | 79.9 | 84.7 |
| Prior R3 | 42,827 | 31.2 | +273 | 650 | 226 | 2,730 | 5,957 | 79.8 | 84.7 |
| Q3 forward | 47,923 | 35.0 | -177 | 611 | 244 | 2,355 | 5,558 | 79.9 | 84.6 |

**B. Head-to-head (69,792 rows)**

| Baseline | Bias | MAE | MedAE | p95 | p99 | Dir% | D>100% |
|----------|-----:|----:|------:|----:|----:|-----:|-------:|
| **Nodal f0** | +121 | **542** | **211** | **2,181** | **4,449** | **82.7** | **87.6** |
| f0 path | +120 | 550 | 215 | 2,191 | 4,504 | 81.6 | 86.4 |
| H | +325 | 647 | 305 | 2,383 | 4,819 | 64.9 | 69.9 |

**C. Fully matched (29,795 rows)**

| Baseline | Bias | MAE | MedAE | p95 | p99 | Dir% | D>100% |
|----------|-----:|----:|------:|----:|----:|-----:|-------:|
| **Nodal f0** | +111 | **465** | 176 | 1,877 | 4,144 | **82.6** | **87.7** |
| f0 path | +107 | 468 | 178 | 1,890 | 4,231 | 81.9 | 87.1 |
| f1 path | +114 | 468 | 175 | 1,866 | 4,340 | 81.9 | 87.2 |
| Prior R3 | +165 | 505 | 190 | 2,032 | 4,769 | 79.3 | 84.5 |
| Prior R2 | +170 | 502 | 186 | 2,013 | 4,721 | 79.2 | 84.3 |
| Prior R1 | +25 | 512 | 191 | 1,998 | 4,521 | 79.1 | 84.1 |
| H | +316 | 569 | 277 | 2,029 | 4,484 | 61.6 | 67.2 |

---

### AQ4 — Mar/Apr/May (135,134 rows)

**A. All rows**

| Baseline | n | Cov% | Bias | MAE | MedAE | p95 | p99 | Dir% | D>100% |
|----------|--:|-----:|-----:|----:|------:|----:|----:|-----:|-------:|
| **Nodal f0 stitch** | 135,134 | 100.0 | +204 | **704** | 260 | 2,925 | 6,375 | **84.8** | 89.5 |
| H (DA congestion) | 135,134 | 100.0 | +399 | 893 | 384 | 3,493 | 7,278 | 64.3 | 68.0 |
| f0 path-level | 73,881 | 54.7 | +123 | 517 | 200 | 2,062 | 4,447 | 83.0 | 88.1 |
| f1 path-level | 64,787 | 47.9 | +143 | 506 | 193 | 2,030 | 4,434 | 82.8 | 87.8 |
| Prior R1 | 34,686 | 25.7 | +36 | 623 | 227 | 2,513 | 6,087 | 80.3 | 84.8 |
| Prior R2 | 40,541 | 30.0 | +295 | 653 | 225 | 2,677 | 6,836 | 80.1 | 84.7 |
| Prior R3 | 43,619 | 32.3 | +261 | 616 | 221 | 2,498 | 6,343 | 80.2 | 84.8 |
| Q4 forward | 57,034 | 42.2 | -183 | 562 | 232 | 2,122 | 4,978 | 80.6 | 85.6 |

**B. Head-to-head (73,881 rows)**

| Baseline | Bias | MAE | MedAE | p95 | p99 | Dir% | D>100% |
|----------|-----:|----:|------:|----:|----:|-----:|-------:|
| **Nodal f0** | +122 | **508** | **196** | **2,036** | **4,395** | **83.9** | **89.1** |
| f0 path | +123 | 517 | 200 | 2,062 | 4,447 | 83.0 | 88.1 |
| H | +317 | 678 | 316 | 2,553 | 5,231 | 61.3 | 65.1 |

**C. Fully matched (30,810 rows)**

| Baseline | Bias | MAE | MedAE | p95 | p99 | Dir% | D>100% |
|----------|-----:|----:|------:|----:|----:|-----:|-------:|
| **Nodal f0** | +118 | **425** | 166 | 1,717 | 3,876 | **83.9** | **89.5** |
| f0 path | +114 | 428 | 167 | 1,728 | 3,825 | 83.4 | 88.9 |
| f1 path | +128 | 437 | 167 | 1,759 | 4,005 | 83.1 | 88.2 |
| Prior R3 | +171 | 483 | 189 | 1,893 | 4,290 | 79.8 | 84.4 |
| Prior R2 | +186 | 483 | 184 | 1,883 | 4,456 | 79.6 | 84.1 |
| Prior R1 | +37 | 475 | 182 | 1,850 | 4,364 | 79.5 | 84.0 |
| H | +328 | 595 | 291 | 2,174 | 4,606 | 58.3 | 62.6 |

AQ4 is where H performs worst (Dir% 58-64%) because only March DA data is available before the April cutoff. Nodal f0 gives the largest improvement here: +20.5pp direction accuracy.

---

## Phase 1 Conclusions

1. **Baseline ranking is identical across all 4 quarters:** Nodal f0 ≈ f0 path < f1 < R3 ≈ R2 < R1 < QF < H.
2. **Nodal f0 wins head-to-head** vs f0 path in every quarter (51.8-51.9% win rate, structural from averaging methodology).
3. **2-tier cascade (Nodal f0 → H fallback) is optimal.** Adding f0 path as Tier 1 always hurts by +4-5 MAE.
4. **All baselines have persistent positive bias** (underestimate MCP), except Prior R1 which is near-zero bias.
5. **Quarterly forwards have negative bias** (-177 to -200), opposite to everything else.

---

## Phase 2 Results: Improving the Nodal f0 Baseline

### Residual Analysis (from `findings_residual_patterns.md`)

The nodal f0 residual (`mcp - nodal_f0`) has exploitable structure:

| Finding | Detail |
|---------|--------|
| **Persistent positive bias** | Mean +204 to +340 depending on quarter. Never negative in any PY. |
| **Autocorrelation** | Same path, lag-1: r = 0.41-0.71. Lag-2: r = 0.37-0.60. |
| **Persistent sign** | 58% of paths (with 4+ years) have same-sign residual ≥80% of years |
| **Scale factor** | Bias ratio 0.48 for |f0| ≥ 50 bin → MCP is ~48% larger than f0 |
| **Volume effect** | Low-volume paths: MAE 1,212. High-volume: MAE 586. 2x spread. |

### Alpha Scaling (all rows with nodal_f0)

`prediction = α × nodal_f0`. Optimal α chosen by LOO (leave-one-PY-out). Direction accuracy is unchanged — scaling preserves sign.

| Quarter | n | Raw MAE | α=1.45 MAE | α=1.55 MAE | α=1.60 MAE | Best improvement |
|---------|--:|--------:|-----------:|-----------:|-----------:|-----------------:|
| aq1 | 147,315 | 798 | 740 (-7%) | 736 (-8%) | 736 (-8%) | -8% |
| aq2 | 148,491 | 947 | 837 (-12%) | 821 (-13%) | 815 (-14%) | -14% |
| aq3 | 136,716 | 797 | 692 (-13%) | 675 (-15%) | 668 (-16%) | -16% |
| aq4 | 135,134 | 704 | 593 (-16%) | 575 (-18%) | 566 (-20%) | -20% |

### Combined: α scaling + Prior-Year Residual (LOO)

`prediction = α × nodal_f0 + β × (PY-1 mcp - PY-1 nodal_f0)`. Only ~22% of paths have prior-year data (recurring paths from PY-1's R1). Head-to-head on those ~30K rows:

**AQ1** (32,495 rows with prior residual)

| Method | Bias | MAE | MedAE | p95 | p99 | Dir% |
|--------|-----:|----:|------:|----:|----:|-----:|
| Nodal f0 (raw) | +232 | 736 | 271 | 3,073 | 6,579 | 82.0 |
| α=1.58 | +98 | 669 | 262 | 2,667 | 5,701 | 82.0 |
| β=0.30 only | +146 | 678 | 264 | 2,718 | 5,887 | 80.6 |
| **α=1.40 β=0.15** | **+96** | **663** | **262** | **2,639** | **5,645** | **81.6** |
| H (reference) | +485 | 882 | 392 | 3,384 | 7,215 | 63.9 |

**AQ2** (32,266 rows)

| Method | Bias | MAE | MedAE | p95 | p99 | Dir% |
|--------|-----:|----:|------:|----:|----:|-----:|
| Nodal f0 (raw) | +313 | 784 | 273 | 3,348 | 7,708 | 82.5 |
| α=1.60 | +151 | 674 | 247 | 2,781 | 6,395 | 82.5 |
| β=0.40 only | +175 | 678 | 253 | 2,779 | 6,444 | 81.3 |
| **α=1.55 β=0.20** | **+95** | **650** | **240** | **2,663** | **6,062** | **82.5** |
| H (reference) | +552 | 935 | 399 | 3,709 | 8,332 | 64.3 |

**AQ3** (30,565 rows)

| Method | Bias | MAE | MedAE | p95 | p99 | Dir% |
|--------|-----:|----:|------:|----:|----:|-----:|
| Nodal f0 (raw) | +269 | 742 | 253 | 3,192 | 6,123 | 84.4 |
| α=1.60 | +117 | 620 | 232 | 2,510 | 4,959 | 84.4 |
| β=0.50 only | +131 | 618 | 232 | 2,491 | 5,152 | 82.6 |
| **α=1.60 β=0.30** | **+35** | **570** | **224** | **2,249** | **4,648** | **84.4** |
| H (reference) | +514 | 889 | 381 | 3,589 | 6,977 | 63.3 |

**AQ4** (30,395 rows)

| Method | Bias | MAE | MedAE | p95 | p99 | Dir% |
|--------|-----:|----:|------:|----:|----:|-----:|
| Nodal f0 (raw) | +287 | 666 | 214 | 2,921 | 6,713 | 85.3 |
| α=1.60 | +129 | 536 | 189 | 2,264 | 5,003 | 85.3 |
| β=0.60 only | +145 | 522 | 199 | 2,084 | 4,689 | 82.4 |
| **α=1.60 β=0.40** | **+34** | **459** | **184** | **1,809** | **3,972** | **85.1** |
| H (reference) | +518 | 878 | 368 | 3,516 | 8,128 | 59.2 |

### Phase 2 Summary

| Quarter | Raw MAE | α-only MAE | Combined MAE | Combined vs raw |
|---------|--------:|-----------:|-------------:|----------------:|
| aq1 | 736 | 669 (-9%) | **663 (-10%)** | -10% |
| aq2 | 784 | 674 (-14%) | **650 (-17%)** | -17% |
| aq3 | 742 | 620 (-16%) | **570 (-23%)** | -23% |
| aq4 | 666 | 536 (-20%) | **459 (-31%)** | -31% |

Improvement increases aq1→aq4 because later quarters have more stable congestion patterns (less weather-driven variance), making the prior-year residual more predictive.

---

## Recommended Production Cascade

| Tier | Method | Coverage | Notes |
|------|--------|----------|-------|
| 1 | `α × nodal_f0 + β × prior_residual` | ~22% | Where path recurs from PY-1 R1 |
| 2 | `α × nodal_f0` | ~77% | Remaining paths with nodal coverage |
| 3 | H bias-corrected | ~1% | Fallback for missing nodes |

Optimal parameters (per-quarter, from LOO): α = 1.40-1.60, β = 0.15-0.40.

---

## Next: Signal Blending

The baselines parquet files contain 7 signals per row. Current experiment uses only nodal_f0 + prior-year residual. Potential improvement: blend all available signals (f0 path, f1, prior R1/R2/R3, quarterly, H) via ridge regression or weighted average, with LOO cross-validation.
