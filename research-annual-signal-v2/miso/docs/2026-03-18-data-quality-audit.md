# Annual Signal V2 — Data Quality Audit

**Date**: 2026-03-18
**Scope**: All 15 eval groups (12 dev + 3 holdout) x 2 class types = 30 slices

---

## 1. Bugs Fixed This Session

### 1.1 Abs_SP@K == VC@K (now fixed)

**Root cause**: `features.py:194` set `total_da_sp_quarter = table["realized_shadow_price"].sum()` — the branch-level SP sum, same denominator as VC@K. The intended cross-universe denominator (total DA SP including unmapped cids) was lost.

**Fix**: `ground_truth.py` now exports `onpeak_total_da_sp` and `offpeak_total_da_sp` in GT diagnostics. `features.py` uses these as the `total_da_sp_quarter` column. Abs_SP@K now correctly answers: "what fraction of ALL realized DA value did top-K capture?"

**Impact**: Abs_SP@K is now 81-91% of VC@K on dev, 62-84% on holdout. The gap represents DA SP from constraints that never map to any branch in the V6.1 universe.

### 1.2 Registry results regenerated

All 4 model ladder jobs re-run with corrected Abs_SP@K:
- `registry/onpeak/m2_dev/results.json`
- `registry/onpeak/m2_holdout/results.json`
- `registry/offpeak/m2_dev/results.json`
- `registry/offpeak/m2_holdout/results.json`

VC@K, Recall@K, NB metrics, Dang metrics unchanged. Only Abs_SP@K values corrected.

---

## 2. Three Concerns Investigated

### 2.1 "1,000 published constraints capture ALL realized binding value" — FALSE

The claim was overstated. Two sources of loss:

1. **Branch universe coverage gap**: 8-38% of class-specific DA SP comes from constraints that don't map to any V6.1 branch (unmapped cids, ambiguous drops).
2. **Publisher slot waste**: 1,000 constraint rows correspond to only ~733-746 unique branches (branch_cap=3 expands each branch to multiple constraints). Sibling constraints eat slots.

Measured branch-level VC@1000 on 2024-06 published V7 artifact: **0.518-0.641** across 8 slices, not 1.00.

### 2.2 VC@400 double counting — registry is clean, published artifact is not

- **Registry VC@400**: Computed at branch level in `run_model_ladder.py:39-65`. Model table asserts unique branch_names (`features.py:206`). No double counting.
- **Published V7 artifact**: If evaluated at constraint-row level, inflated because multiple constraints map to the same branch. The "naive" vs "branch-collapsed" gap is substantial (~0.1-0.2 VC).

### 2.3 PLESNLEEDS11_1 case study — weak signal, not zero

Branch key is `PLESNLEEDS11_1 1` (trailing space+1). Present in universe, classified as `history_zero`, `is_nb_12 = True`.

v0c gives it score ~0.003 (rank ~1955/2300):
- `da_rank_value` = sentinel (no history) -> da_norm ~ 0.000
- `bf_12` = 0.000 -> bf_norm = 0.000
- `rt_max` = 0.137 -> rt_norm = 0.009 (small density signal)

2 of 3 v0c components are dead. The density signal alone is too weak to push it into any K threshold.

---

## 3. Coverage Gap Trend — The Most Critical Finding

The fraction of DA SP that successfully maps to branches is **declining over time**:

| PY | Unmapped CIDs (range) | Abs/VC ratio (range) | DA SP leaked (range) |
|----|:---:|:---:|:---:|
| 2022-06 | 41-48 | 0.85-0.92 | 8-15% |
| 2023-06 | 54-92 | 0.82-0.94 | 6-18% |
| 2024-06 | 43-315 | 0.74-0.95 | 5-26% |
| **2025-06** | **245-444** | **0.62-0.76** | **24-38%** |

### 3.1 Worst slices

| Group | Class | Abs/VC | Unmapped CIDs | DA SP leaked |
|-------|-------|:---:|:---:|:---:|
| 2025-06/aq2 | offpeak | 0.619 | 444 | 38.1% |
| 2025-06/aq3 | onpeak | 0.644 | 345 | 35.6% |
| 2025-06/aq2 | onpeak | 0.671 | 444 | 32.9% |
| 2025-06/aq3 | offpeak | 0.677 | 345 | 32.3% |
| 2024-06/aq4 | onpeak | 0.740 | 315 | 26.0% |
| 2025-06/aq1 | onpeak | 0.735 | 245 | 26.5% |

### 3.2 Gap decomposition — three distinct problems

The Abs/VC gap has three sources. The first was initially overstated because CID-level matching misses branch-level recovery.

**Source A — CID-unmapped DA constraints**: DA constraint_ids not in the SPICE bridge. Initially reported as 23-30% of DA SP for 2025-06. However, many of these are new CIDs on branches that already exist in the SPICE universe.

**Source A corrected — Supplement key matching**: Using `MisoDaShadowPriceSupplement` structured keys (`key1+key3` for XF transformers, `key2+key3` for LN lines), 86/129 CID-unmapped constraints in 2025-06 are recoverable — they are new constraint formulations on known branches.

**Source B — Density universe coverage (out-of-universe branches)**: Binding branches that map through the bridge but fail the density threshold filter. Stable at 2-14% across all PYs.

| PY | CID-unmapped SP% | Recovered SP% | Residual SP% | Out-of-Universe SP% |
|----|:---:|:---:|:---:|:---:|
| 2022-06 | 3-4% | ~1.6% | **~1.0%** | 4-10% |
| 2023-06 | 2-5% | ~0.2% | **~1.4%** | 4-14% |
| 2024-06 | 2-20% | ~0.5% | **~1.2%** | 2-7% |
| 2025-06 | 29% | **~22%** | **~6.7%** | 2-12% |

Note: "Recovered" = DA CIDs matched to SPICE branches via supplement keys. "Residual" = CID-unmapped minus Recovered (includes CIDs with no supplement entry). These figures are from a sampled month (aq2/offpeak), not full-quarter aggregates.

**Key insight**: The true 2025-06 gap from unmapped branches is ~6.7%, not 29%. The other ~22% is new CIDs on branches we already model — recoverable by implementing supplement key matching in the GT pipeline.

**Source C — Ambiguous CIDs**: Dropped by `bridge.py:86`. SP = 0 in all groups checked. Not a material contributor.

### 3.3 Consequence for reported metrics

- **Dev VC@400 is meaningful** — the denominator (branch-mapped SP) covers 85-95% of total DA SP for 2022-2024 aq1-2 groups
- **Holdout VC@400 looks strongest** (0.75-0.82) but the universe it measures against covers only 62-76% of total DA SP
- **Holdout Abs_SP@400 is the honest number**: v0c captures 61-69% of actual DA value on holdout
- **Model-vs-model comparisons remain valid**: all models face the same denominator, so relative rankings are unaffected
- **With supplement key matching implemented**, the holdout Abs_SP@400 would improve because more DA SP would be credited to modeled branches

### 3.4 Root cause

Three factors contribute to the Abs/VC gap:

1. **New CIDs on known branches (recoverable)**: MISO creates new constraint formulations (new contingencies, new operating conditions) on existing physical branches. These get new CIDs not in the SPICE bridge, but the branch is already modeled. Fix: implement supplement key matching using `MisoDaShadowPriceSupplement` keys.

2. **Genuinely new branches**: Some DA constraints monitor physical transmission elements never included in any SPICE planning model. This is the irreducible gap (~6.7% for 2025-06, 0.1-2.0% for earlier PYs).

3. **Density threshold filtering**: The `is_active` threshold in `load_collapsed()` excludes branches where the density model predicts near-zero binding probability. Stable at 2-14% across all PYs. A deliberate precision/recall tradeoff.

---

## 4. Per-Group Diagnostics

### 4.1 Universe sizes and binding counts

| Group | Class | N_branches | N_binding | BranchSP | TotalDA_SP | Abs/VC | N_estab | N_dorm | N_zero |
|-------|-------|:---:|:---:|---:|---:|:---:|:---:|:---:|:---:|
| 2022-06/aq1 | onpeak | 2427 | 299 | 1,912,106 | 2,089,770 | 0.915 | 726 | 925 | 776 |
| 2022-06/aq1 | offpeak | 2427 | 261 | 1,335,440 | 1,469,828 | 0.909 | 633 | 1018 | 776 |
| 2022-06/aq2 | onpeak | 2225 | 366 | 1,418,417 | 1,649,973 | 0.860 | 709 | 819 | 697 |
| 2022-06/aq2 | offpeak | 2225 | 314 | 1,422,034 | 1,586,181 | 0.897 | 631 | 897 | 697 |
| 2022-06/aq3 | onpeak | 1953 | 264 | 908,695 | 1,066,875 | 0.852 | 643 | 719 | 591 |
| 2022-06/aq3 | offpeak | 1953 | 257 | 1,084,630 | 1,228,854 | 0.883 | 575 | 787 | 591 |
| 2022-06/aq4 | onpeak | 2395 | 347 | 1,148,901 | 1,264,009 | 0.909 | 741 | 880 | 774 |
| 2022-06/aq4 | offpeak | 2395 | 313 | 1,327,026 | 1,440,088 | 0.921 | 654 | 967 | 774 |
| 2023-06/aq1 | onpeak | 2512 | 290 | 1,125,850 | 1,284,360 | 0.877 | 775 | 1068 | 669 |
| 2023-06/aq1 | offpeak | 2512 | 260 | 598,445 | 682,347 | 0.877 | 672 | 1171 | 669 |
| 2023-06/aq2 | onpeak | 2278 | 382 | 1,621,641 | 1,753,223 | 0.925 | 741 | 964 | 573 |
| 2023-06/aq2 | offpeak | 2278 | 330 | 1,797,798 | 1,903,863 | 0.944 | 656 | 1049 | 573 |
| 2023-06/aq3 | onpeak | 2274 | 300 | 1,200,271 | 1,466,719 | 0.818 | 756 | 944 | 574 |
| 2023-06/aq3 | offpeak | 2274 | 277 | 1,216,018 | 1,461,073 | 0.832 | 676 | 1024 | 574 |
| 2023-06/aq4 | onpeak | 2382 | 381 | 1,270,737 | 1,413,911 | 0.899 | 785 | 975 | 622 |
| 2023-06/aq4 | offpeak | 2382 | 327 | 1,194,718 | 1,313,957 | 0.909 | 695 | 1065 | 622 |
| 2024-06/aq1 | onpeak | 2339 | 345 | 795,653 | 859,251 | 0.926 | 724 | 1110 | 505 |
| 2024-06/aq1 | offpeak | 2339 | 295 | 516,381 | 555,520 | 0.930 | 656 | 1178 | 505 |
| 2024-06/aq2 | onpeak | 2281 | 394 | 1,131,289 | 1,240,442 | 0.912 | 724 | 1074 | 483 |
| 2024-06/aq2 | offpeak | 2281 | 366 | 1,037,973 | 1,098,009 | 0.945 | 648 | 1150 | 483 |
| 2024-06/aq3 | onpeak | 1918 | 244 | 797,432 | 970,915 | 0.821 | 664 | 874 | 380 |
| 2024-06/aq3 | offpeak | 1918 | 223 | 826,930 | 902,618 | 0.916 | 600 | 938 | 380 |
| 2024-06/aq4 | onpeak | 2004 | 278 | 701,878 | 948,591 | 0.740 | 697 | 912 | 395 |
| 2024-06/aq4 | offpeak | 2004 | 241 | 756,512 | 938,413 | 0.806 | 628 | 981 | 395 |
| 2025-06/aq1 | onpeak | 2218 | 241 | 747,092 | 1,016,356 | 0.735 | 777 | 1099 | 342 |
| 2025-06/aq1 | offpeak | 2218 | 224 | 476,780 | 627,197 | 0.760 | 680 | 1196 | 342 |
| 2025-06/aq2 | onpeak | 2210 | 281 | 938,095 | 1,398,348 | 0.671 | 768 | 1092 | 350 |
| 2025-06/aq2 | offpeak | 2210 | 281 | 867,977 | 1,402,323 | 0.619 | 674 | 1186 | 350 |
| 2025-06/aq3 | onpeak | 1706 | 230 | 988,548 | 1,534,835 | 0.644 | 669 | 803 | 234 |
| 2025-06/aq3 | offpeak | 1706 | 222 | 1,070,988 | 1,581,468 | 0.677 | 589 | 883 | 234 |

### 4.2 v0c component distributions (onpeak)

| Group | da_rank sentinel% | bf_12=0% | v0c median | v0c P95 | score<0.001% |
|-------|:---:|:---:|:---:|:---:|:---:|
| 2022-06/aq1 | 36.0% | 70.1% | 0.108 | 0.466 | 17.9% |
| 2022-06/aq2 | 35.3% | 68.1% | 0.110 | 0.472 | 18.6% |
| 2022-06/aq3 | 34.5% | 67.1% | 0.119 | 0.491 | 14.8% |
| 2022-06/aq4 | 36.5% | 69.1% | 0.106 | 0.469 | 17.0% |
| 2023-06/aq1 | 30.9% | 69.1% | 0.130 | 0.466 | 17.6% |
| 2023-06/aq2 | 29.6% | 67.5% | 0.134 | 0.474 | 14.7% |
| 2023-06/aq3 | 30.0% | 66.8% | 0.137 | 0.480 | 13.7% |
| 2023-06/aq4 | 30.6% | 67.0% | 0.133 | 0.473 | 16.8% |
| 2024-06/aq1 | 25.5% | 69.0% | 0.153 | 0.473 | 13.3% |
| 2024-06/aq2 | 25.4% | 68.3% | 0.157 | 0.486 | 12.5% |
| 2024-06/aq3 | 23.8% | 65.4% | 0.171 | 0.505 | 10.1% |
| 2024-06/aq4 | 23.4% | 65.2% | 0.167 | 0.501 | 11.0% |
| 2025-06/aq1 | 19.0% | 65.0% | 0.185 | 0.504 | 7.6% |
| 2025-06/aq2 | 19.4% | 65.2% | 0.174 | 0.501 | 10.0% |
| 2025-06/aq3 | 17.1% | 60.8% | 0.191 | 0.505 | 8.3% |

**Trends**:
- da_rank sentinel% (no history): decreasing 36% -> 17%. More branches accumulate history over time. Good.
- bf_12=0%: decreasing 70% -> 61%. Slowly improving. ~2/3 of branches still have zero BF.
- v0c median: increasing 0.108 -> 0.191. Scores are getting less sparse.
- score<0.001%: decreasing 18% -> 8%. Fewer truly dead branches.
- rt_max=0%: always 0.0%. Every branch has some density signal.

### 4.3 NB detection

| PY | NB12 bind% (onpeak range) | NB12 bind% (offpeak range) |
|----|:---:|:---:|
| 2022-06 | 35-44% | 32-42% |
| 2023-06 | 32-39% | 34-38% |
| 2024-06 | 26-36% | 24-34% |
| 2025-06 | 19-22% | 19-26% |

Novel binders are declining. The universe is maturing — fewer branches bind for the first time. This explains why NB model improvement attempts (phases 6, 10) showed diminishing returns on holdout.

### 4.4 Label tier distribution

Tier 3 SP fraction (fraction of total SP held by the top tertile of binders) is volatile:
- Range: 0.159 (2025-06/aq2 offpeak) to 0.603 (2023-06/aq2 onpeak)
- No consistent trend
- Indicates heavy-tail behavior varies substantially across quarters

---

## 5. Corrected v0c Champion Metrics

### 5.1 Onpeak

| Metric | Dev (12 groups) | Holdout (3 groups) |
|--------|:---:|:---:|
| VC@50 | 0.314 | 0.365 |
| VC@200 | 0.604 | 0.595 |
| VC@250 | 0.639 | 0.631 |
| VC@400 | 0.712 | 0.754 |
| **Abs_SP@50** | **0.278** | **0.296** |
| **Abs_SP@200** | **0.536** | **0.482** |
| **Abs_SP@400** | **0.631** | **0.611** |
| Recall@400 | 0.448 | 0.539 |
| NB12_SP@400 | 0.152 | 0.174 |
| Dg20k_R@400 | 0.796 | 0.869 |
| Dg40k_R@400 | 0.693 | 0.556 |

### 5.2 Offpeak

| Metric | Dev (12 groups) | Holdout (3 groups) |
|--------|:---:|:---:|
| VC@50 | 0.401 | 0.373 |
| VC@200 | 0.689 | 0.659 |
| VC@250 | 0.718 | 0.685 |
| VC@400 | 0.778 | 0.824 |
| **Abs_SP@50** | **0.366** | **0.312** |
| **Abs_SP@200** | **0.628** | **0.551** |
| **Abs_SP@400** | **0.709** | **0.689** |
| Recall@400 | 0.491 | 0.589 |
| NB12_SP@400 | 0.136 | 0.229 |
| Dg20k_R@400 | 0.864 | 0.911 |
| Dg40k_R@400 | 0.714 | 0.917 |

### 5.3 Abs_SP / VC ratio

| | Dev | Holdout |
|---|:---:|:---:|
| Onpeak @400 | 0.886 | 0.811 |
| Offpeak @400 | 0.911 | 0.836 |

Holdout coverage is 8-10 percentage points worse than dev, consistent with the growing unmapped CID trend.

---

## 6. Edge Cases

| Issue | Group | Detail |
|-------|-------|--------|
| Worst coverage | 2025-06/aq2 offpeak | Abs/VC = 0.619, 444 unmapped cids, 38% DA SP leaked |
| Anomalous unmapped jump | 2024-06/aq4 | 315 unmapped cids vs 43-128 for aq1-3 |
| Smallest universe | 2025-06/aq3 | 1706 branches (vs 2200-2500 typical) |
| Lowest Tier 3 concentration | 2025-06/aq2 offpeak | T3_SP_frac = 0.159 (SP spread very evenly) |
| Highest Tier 3 concentration | 2023-06/aq2 onpeak | T3_SP_frac = 0.603 (extreme heavy tail) |
| Lowest NB bind rate | 2025-06/aq1 onpeak | 18.7% (down from 35-44% in 2022) |

---

## 7. Open Items

1. **Implement supplement key matching** — Use `MisoDaShadowPriceSupplement` keys (`key1+key3` for XF, `key2+key3` for LN) in the GT pipeline to recover DA CIDs on known SPICE branches. Recovers 86/129 CID-unmapped constraints (~22% of DA SP) for 2025-06/aq2/offpeak sampled month. Full-quarter and all-class validation still needed. See `docs/coverage-analysis-runbook.md` for algorithm and examples.

2. **Published artifact validation** — `signal_publisher.py` needs branch-collapsed VC. Without it, anyone evaluating the V7 parquet gets inflated numbers from sibling constraints.

3. **Doc corrections** — Retract "ALL binding captured" claim. Update Abs_SP columns in model comparison report. Correct PLESNLEEDS case study.

4. **Zero-SF filter in publisher** — One-liner fix, 0.34% of slots wasted on average, 2023-06 outlier at 1.3%.
