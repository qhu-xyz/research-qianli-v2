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

## Factor Table (Expanding Median, All Years 2017-2025)

| fx | Month | Factor | vs flat 1/12 |
|----|-------|--------|-------------|
| f0 | Jun | 0.067766 | -1.557% |
| f1 | Jul | 0.078116 | -0.522% |
| f2 | Aug | 0.076500 | -0.683% |
| f3 | Sep | 0.078461 | -0.487% |
| f4 | Oct | 0.080659 | -0.268% |
| f5 | Nov | 0.072311 | -1.102% |
| f6 | Dec | 0.077031 | -0.630% |
| f7 | Jan | 0.080770 | -0.256% |
| f8 | Feb | 0.072676 | -1.066% |
| f9 | Mar | 0.067650 | -1.568% |
| f10 | Apr | 0.065235 | -1.810% |
| f11 | May | 0.065671 | -1.766% |

Sum = 0.883 (not 1.0 — by design, since monthly MCPs from June R1 systematically differ from annual R4)

## Failed approaches (for the record)
1. **Node-level seasonal factors**: 9x worse than naive (errors amplify through subtraction)
2. **Per-month regression (annual only)**: ~1% improvement (beta ≈ 1/12, minimal lift)
3. **Normalized MW factors** (sum to 1.0): Worse than unnormalized
4. **Blends with naive**: Pure MW outperforms all blends on trades

## Key Scripts
| Script | Purpose |
|--------|---------|
| `01_node_seasonal_factors.py` | Node-level approach (failed, 9x worse) |
| `02_diagnose_node_ratios.py` | Root cause analysis of node ratio instability |
| `03_alternative_approaches.py` | Tests regression, prev-year, MW factor approaches |
| `04_refine_market_wide.py` | MW factor refinement (windows, blends, normalization) |
| `05_mw_trades_deep.py` | Deep-dive on trades: per-year, per-month, diagnostics |
