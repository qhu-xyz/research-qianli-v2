# PJM June First-Monthly MCP Prediction — Research Summary

## Problem Statement

PJM's first monthly auction (June) covers 12 forward months: f0 (Jun) through f11 (May next year). For MTM purposes, we predict each month's MCP from the annual auction result. The current approach uses `annual_mcp / 12` (flat, hour-weighted distribution). We investigated whether a better seasonal distribution improves prediction accuracy.

**Units**: MCP values are in total dollars over the contract period (hourly_mcp × hours_in_month), not $/MWh.

**Data**: 62K deduplicated onpeak obligation trades across PY2020-2025, plus pool data from `/opt/temp/qianli/pjm_mcp_pred_training2`.

---

## V1: Path-Level Historical Distributions

**Hypothesis**: Use last year's per-path monthly distribution (how much of the annual each month received) to predict this year's per-month MCPs.

### Methods tested
1. **Naive /12**: `annual_mcp / 12` (current, hour-weighted)
2. **Distribution**: `annual_mcp × last_year_pct[fx]`
3. **Distribution + ratio**: accounts for `sum(monthly) ≠ annual`

### Key findings
- **Aggregate seasonal pattern exists but is small**: Sep-Oct peak (~9-10%), spring trough (~7.5%), max deviation from flat only ±1.3%
- **Path-level distributions are extremely unstable**: 30-56 percentage points average year-over-year change per path
- **Pathological values**: ~2% of paths have percentages |pct| > 1 (up to ±567×) due to near-zero denominators
- **Sign flips**: ~9-13% of paths have monthly_sum/annual with opposite signs (different auctions, opinions shift)

### Results
| Method | MAE | Win Rate |
|--------|-----|----------|
| Naive /12 | 176.94 | baseline |
| Clipped per-path dist | 183.25 | 47.0% (worse) |
| **Blend 50/50 naive+clipped** | **166.53** | **52.9%** |

**On trades**: Best method = 1.8% MAE reduction, 51.1% win rate. Not worth the complexity.

**V1 verdict**: Path-level historical distributions too noisy. Abandoned.

---

## V2: Market-Wide (MW) Seasonal Factor

**Hypothesis**: Instead of per-path factors, compute a single 12-element vector of seasonal factors from all paths in all training years. Apply uniformly.

### Formula
```
pred_fx = annual_path_mcp × factor[fx]
```
Where `factor[fx] = median(fx_path_mcp / annual_path_mcp)` across all cleared paths, expanding window, clipped to [-0.5, 0.5].

For `mtm_1st_mean` adjustment: `adjusted = mtm_1st_mean × (factor[fx] × 12)` since the original is `annual/12`.

### Why this works
1. Captures seasonal shape — summer/fall get more, spring/late-winter less
2. Robust — median over 100K+ ratios, extremely stable
3. No noise amplification — single scalar multiply per path

### Results on trades (62K, PY2020-2025, onpeak)
| Method | MAE | Win Rate | Improvement |
|--------|-----|----------|-------------|
| Current mtm_1st | 162.97 | baseline | — |
| **MW factor** | **150.06** | **62.0%** | **+7.9%** |

### Every year improves
| Year | Current MAE | MW MAE | Win% | Improvement |
|------|------------|--------|------|-------------|
| PY2020 | 87.86 | 78.45 | 61.7% | +10.7% |
| PY2021 | 109.63 | 98.64 | 60.1% | +10.0% |
| PY2022 | 236.18 | 222.25 | 58.9% | +5.9% |
| PY2023 | 167.52 | 151.98 | 62.7% | +9.3% |
| PY2024 | 165.15 | 151.82 | 62.7% | +8.1% |
| PY2025 | 219.57 | 204.90 | 66.0% | +6.7% |

### Scales with path value
| Value tier | MW improvement | Win rate |
|------------|---------------|----------|
| All trades | +7.9% | 62.0% |
| $1K+ paths | +10.4% | 68.2% |
| $5K+ paths | +14.2% | 78.0% |

### Statistical significance
- Paired t-test: t=47.1, p≈0
- Wilcoxon signed-rank: p≈0
- Bootstrap 95% CI: [12.37, 13.46] — entirely positive
- Significant in every year, month, and value tier individually

### Baseline impact (end-to-end V2 formula)
MW only modifies `mtm_1st_mean`, which is ~22% of the full V2 baseline (`0.65 × avg(mtm1,mtm2,mtm3) + 0.35 × avg(rev1,rev2,rev3)`):
- Pool baseline improvement: **+0.6%**
- Trades baseline improvement: **+1.6%**
- Dilution is structural and expected

---

## Failed Alternatives (Exhaustive)

### 1. Node-level seasonal factors (V2, script 01)
- Compute per-node monthly/annual ratios, then path = sink_factor - source_factor
- **Result: 9× worse** than naive (MAE from 163 → ~1400)
- **Root cause**: node-level ratios are extremely noisy (std=4.57 on mean=1.07). Subtraction of two noisy estimates compounds error

### 2. Node-level traceback (V2, script 08)
- Use previous year's per-node monthly/annual ratios
- **Result: -418%** (catastrophic)
- Same noise compounding issue

### 3. Path-level traceback (V2, script 09)
- Use previous year's per-path monthly/annual ratio
- Coverage: only 28.5% of trades have prior-year data on the same path
- With expanding window (all prior years): 62.4% coverage
- **Results on covered trades**:
  - 1 year history: -8.2% (worse)
  - 3 years: +9.7%
  - **5+ years: +17.9%** — but only covers **6% of trades**
- At 5+ years, path-level is statistically indistinguishable from MW (p=0.14)
- **Best hybrid (path + MW blend)**: saves $0.16/trade — economically negligible

### 4. Normalized MW factors (sum = 1.0)
- Force factors to sum to 1.0 instead of natural ~0.88
- **Result: +3.0% improvement** vs unnormalized +7.9%
- Significantly worse — the ~0.88 sum is feature, not bug

### 5. Per-month regression (annual only)
- Regress each month's MCP on annual MCP
- Result: ~1% improvement (beta ≈ 1/12, no lift)

### 6. Blends with naive /12
- All blend ratios tested; pure MW outperforms every blend

---

## The 88% Factor Sum Explained

MW factors sum to ~0.88, suggesting "the market always shrinks by 12%." Investigation showed this is **not real shrinkage**:

- **Jensen's inequality**: `sum(median per-month ratio) ≠ median(sum of per-path ratios)`
  - Sum of medians = 0.88
  - Median of sums = 0.95
- **Actual per-year path sums**: PY2020=0.91, PY2024=0.99 — no consistent direction
- **MW's value comes from shape correction** (which months get more/less), not level scaling
- Even in PY2024 where actual sum ≈ 1.0, MW still wins (+8.1%)
- Normalizing to 1.0 destroys useful signal → significantly worse results

---

## Factor Table (PY2025 Training, Expanding Median)

| fx | Month | Factor | Adj. ratio (×12) |
|----|-------|--------|------------------|
| f0 | Jun | 0.065385 | 0.7846 |
| f1 | Jul | 0.074600 | 0.8952 |
| f2 | Aug | 0.074281 | 0.8914 |
| f3 | Sep | 0.078534 | 0.9424 |
| f4 | Oct | 0.081646 | 0.9798 |
| f5 | Nov | 0.074560 | 0.8947 |
| f6 | Dec | 0.077243 | 0.9269 |
| f7 | Jan | 0.080911 | 0.9709 |
| f8 | Feb | 0.074113 | 0.8894 |
| f9 | Mar | 0.067282 | 0.8074 |
| f10 | Apr | 0.065445 | 0.7853 |
| f11 | May | 0.065657 | 0.7879 |

Sum = 0.8797

---

## Recommendation

**Ship the MW seasonal factor as-is** (unnormalized, expanding median):
- Minimal implementation: 12-element lookup table per planning year
- Applied as scalar multiply on `mtm_1st_mean` before the baseline blend
- No path-level or node-level enhancements — they add complexity without meaningful improvement
- Statistically robust across all years, months, value tiers, and flow types


and run v3 pipeline for each ptype to verify it works. compare the results against v2(so you need also rerun v2) --> NO smoke test, test with real data until optimization. purpose is to verify    
  banding and clear probs can be generated and v3 is an improvement when comparing with v2

- test on: pool data  (the training data in pjm2)
- /opt/temp/shiyi/trash/pjm_onpeak.parquet

Do u understnad 100pct what i mean?

1. the numbers look way off, let's focus on f0 first.
2. for pjm v3 against vs, f0, the mtm_1st_mean adjustment, does it help? 
3. why does the band 50% have only around 20ish coverage? this is for lower or upper?
4. also when i say coverage stability: i mean to test whether a bid price's designated clearing prob matches the true cp.


so are the following true?
1. 