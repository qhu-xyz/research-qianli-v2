# V1: PJM June First-Monthly MCP Prediction

## Problem
PJM's first monthly auction (June) covers f0 (Jun) through f11 (May next year). For MTM purposes, we predict each month's MCP using `annual_mcp / 12`. This is potentially inaccurate because monthly MCP distributions are not flat.

## Hypothesis
Use last year's actual monthly distribution (from the June auction) to predict this year's per-month MCPs, instead of flat 1/12.

Three methods tested:
1. **Naive /12**: `annual_mcp / 12` (current approach, hour-weighted)
2. **Distribution**: `annual_mcp * last_year_pct[fx]` where `pct[fx] = last_year_f[fx] / last_year_sum`
3. **Distribution + ratio**: `annual_mcp * last_year_pct[fx] * (last_year_sum / last_year_annual)`

## Findings

### Phase 1: Aggregate distribution analysis (24h, 9 years, PY 2017-2025)
- Confirmed moderate seasonal pattern: Sep-Oct peak (~9-10%), spring trough (~7.5%)
- Maximum deviation from flat: +1.3% (Oct), -0.9% (Jun)
- Positive and negative MTM paths show similar shapes
- Year-to-year std = 1-3% per month

### Phase 2: Path-level backtesting (onpeak, PY 2020-2025)
- **589K prediction samples** across 49K paths over 6 years
- Path-level distributions are extremely unstable year-to-year (**30-56 pct points** average change in allocation)
- ~9-13% of paths have `monthly_sum / annual` with opposite signs
- ~2% of paths have pathological percentage values (|pct| > 1, up to ±567x) due to near-zero monthly sums

### Backtesting results (bug-fixed, with percentage clipping)

| Method | MAE | WinVsNaive |
|--------|-----|------------|
| Naive /12 | 176.94 | baseline |
| Clipped per-path dist | 183.25 | 47.0% |
| **Blend 50 naive + 50 clipped** | **166.53** | **52.9%** |

### Trades verification (13,375 June onpeak trades)

| Method | MAE | WinVsCurrent |
|--------|-----|--------------|
| Current mtm_1st_mean | 94.90 | baseline |
| **Blend 50 naive + 50 clipped** | **93.17** | **51.1%** |

**Improvement: 1.8% MAE reduction, 51.1% win rate** — not sufficient to justify the complexity.

## Conclusion
The naive /12 approach is already near-optimal for June's first-monthly MCP prediction. Path-level monthly distributions are too noisy year-over-year to add meaningful signal, and the aggregate seasonal pattern deviates from flat by only ~1%.

**V1 verdict: no actionable improvement found. Move to v2 with a fundamentally different approach.**

## Key Scripts
| Script | Purpose |
|--------|---------|
| `scripts/18_all_years.py` | 9-year aggregate distribution analysis (24h class) |
| `scripts/23_clipped_backtest.py` | Path-level backtesting with clipping (onpeak, definitive) |
| `scripts/15_audit.py` | Correctness audit of MCP data |

## Data
- `data/mcp_distribution_{2020,2021,2022}.parquet` — early exploratory data dumps
