# V2: Market-Wide Seasonal Factors for PJM June First-Monthly MCP

## Problem
Same as V1: predict per-month MCPs for PJM's June first-monthly auction (f0-f11).
Current approach: `annual_mcp / 12` (flat distribution). Goal: beat this.

## V1 Recap
Path-level and node-level seasonal decomposition approaches failed:
- Path-level distributions too noisy (30-56 pct pts YoY change)
- Node-level factors amplify errors through subtraction (MAE 1485 vs 163)
- Best v1 result: 1.8% MAE improvement (not worth complexity)

## V2 Approach: Market-Wide Seasonal Factor

Instead of per-path or per-node decomposition, compute a **single 12-vector of seasonal factors** applied uniformly to all paths:

```
pred_fx = annual_path_mcp * market_wide_factor[fx]
```

Where `factor[fx] = median(fx_path_mcp / annual_path_mcp)` across all cleared paths and training years, with extreme ratios clipped to [-0.5, 0.5].

### Why this works
1. **Captures seasonal shape** — summer/fall months get more, spring/late-winter less
2. **Robust to noise** — median over ~100K+ ratios is extremely stable
3. **No amplification** — single scalar multiplication, no source/sink subtraction

### Key insight: systematic bias in naive /12
The actual monthly MCPs are systematically different from annual/12 at the path level:
- `sum(monthly_mcps)` averages 88-95% of `annual_mcp` (from different auction at different time)
- The monthly distribution is NOT flat — summer/fall gets more, spring gets less
- Naive /12 has a mean signed error of +108 (systematically overestimates)

## Results

### Trades (62K dedup, PY 2020-2025, onpeak)

| Method | MAE | Win vs Current | Improvement |
|--------|-----|---------------|-------------|
| Current mtm_1st | 162.97 | baseline | — |
| Naive /12 | 162.89 | 48.5% | +0.0% |
| **MW factor (expanding median)** | **150.06** | **62.0%** | **+7.9%** |

### Per year (all years improve)

| Year | Current MAE | MW MAE | Win% | Improvement |
|------|------------|--------|------|-------------|
| PY2020 | 87.86 | 78.45 | 61.7% | +10.7% |
| PY2021 | 109.63 | 98.64 | 60.1% | +10.0% |
| PY2022 | 236.18 | 222.25 | 58.9% | +5.9% |
| PY2023 | 167.52 | 151.98 | 62.7% | +9.3% |
| PY2024 | 165.15 | 151.82 | 62.7% | +8.1% |
| PY2025 | 219.57 | 204.90 | 66.0% | +6.7% |

### Per month (11/12 months improve)

| fx | Month | Current | MW | Improvement |
|----|-------|---------|-----|-------------|
| f0 | Jun | 473.90 | 420.02 | +11.4% |
| f1 | Jul | 84.42 | 80.42 | +4.7% |
| f2 | Aug | 95.79 | 89.94 | +6.1% |
| f3 | Sep | 73.87 | 72.22 | +2.2% |
| f4 | Oct | 100.98 | 98.19 | +2.8% |
| f5 | Nov | 89.60 | 86.56 | +3.4% |
| f6 | Dec | 121.97 | 119.56 | +2.0% |
| f7 | Jan | 178.50 | 178.64 | -0.1% |
| f8 | Feb | 158.05 | 156.66 | +0.9% |
| f9 | Mar | 101.66 | 88.69 | +12.8% |
| f10 | Apr | 125.66 | 106.95 | +14.9% |
| f11 | May | 121.95 | 106.20 | +12.9% |

### Excluding f0 (still significant)
MW factor MAE = 106.76 vs current 113.10 — **5.6% improvement, 61.1% win rate**

### Baseline impact (end-to-end V2 formula)
MW only changes mtm_1st_mean (1/3 of mtm avg, which is 65% of blend = ~22% of baseline weight).
- **Pool baseline improvement: +0.6%**
- **Trades baseline improvement: +1.6%**
- Dilution is expected and acceptable — mtm_1st is a small piece of the V2 blend.

### High-value paths (scales with value)
| Value tier | MW improvement | Win rate |
|------------|---------------|----------|
| All trades | +7.9% | 62.0% |
| $1K+ | +10.4% | 68.2% |
| $5K+ | +14.2% | 78.0% |

## Statistical Significance

MW improvement is statistically unambiguous:
- **Paired t-test**: t=47.1, p≈0
- **Wilcoxon signed-rank**: p≈0
- **Bootstrap 95% CI**: [12.37, 13.46] — entirely positive
- Every individual year is significant (all p < 1e-40)
- Every path value tier is significant (all p < 1e-6)

## The 88% Factor Sum: Jensen's Inequality, Not "Market Shrinkage"

The MW factors sum to ~0.88, not 1.0. This initially looks like "the market always shrinks by 12%." Investigation shows it's a mathematical artifact:

- **sum of medians ≠ median of sums**: `sum(median per-month ratio) = 0.88` vs `median(sum of per-path ratios) = 0.95` — Jensen's inequality
- **Actual path-level sum ratios vary by year**: PY2020=0.91, PY2024=0.99 — no consistent shrinkage
- **Normalizing to sum=1.0 hurts**: +3.0% improvement vs +7.9% unnormalized
- **MW value comes from shape correction** (which months get more/less), not level scaling

## Factor Table (Expanding Median, PY2025 Training)

| fx | Month | Factor | Adjustment ratio (×12) |
|----|-------|--------|----------------------|
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

Sum = 0.8797. Implementation: `adjusted_mtm_1st = mtm_1st_mean × adjustment_ratio[fx]`

## Failed Approaches (for the record)

1. **Node-level seasonal factors** (script 01): 9x worse than naive — errors amplify through sink-source subtraction. Node factor sums: mean=1.07, std=4.57 (extremely noisy).
2. **Node-level traceback** (script 08): -418% — catastrophic failure. Previous year's per-node monthly/annual ratios compound noise.
3. **Path-level traceback** (script 09): Works with 5+ years of history (+17.9%) but coverage only 6% of trades. Statistically indistinguishable from MW at 5+ years. Best hybrid saves $0.16/trade — economically negligible.
4. **Normalized MW factors** (sum to 1.0): +3.0% vs unnormalized +7.9%. Significantly worse.
5. **Per-month regression (annual only)**: ~1% improvement (beta ≈ 1/12, minimal lift)
6. **Blends with naive**: Pure MW outperforms all blends on trades

## Recommendation

**Ship the MW seasonal factor as-is** (unnormalized, expanding median):
- Implementation: 12-element lookup table per planning year, applied as scalar multiply on mtm_1st_mean
- No path-level or node-level enhancements needed — they add complexity without meaningful improvement
- Formula: `adjusted_mtm_1st = mtm_1st_mean × (factor[fx] × 12)`

## Key Scripts
| Script | Purpose |
|--------|---------|
| `01_node_seasonal_factors.py` | Node-level approach (failed, 9x worse) |
| `02_diagnose_node_ratios.py` | Root cause analysis of node ratio instability |
| `03_alternative_approaches.py` | Tests regression, prev-year, MW factor approaches |
| `04_refine_market_wide.py` | MW factor refinement (windows, blends, normalization) |
| `05_mw_trades_deep.py` | Deep-dive on trades: per-year, per-month, diagnostics |
| `06_tail_risk.py` | Tail risk analysis |
| `07_baseline_impact.py` | End-to-end baseline impact (+0.6% pool, +1.6% trades) |
| `08_followup_analysis.py` | Statistical significance, normalized vs unnormalized, node traceback |
| `09_path_level_traceback.py` | Path-level traceback: coverage, depth analysis, hybrid blending |
