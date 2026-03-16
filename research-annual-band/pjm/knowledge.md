# PJM Annual Band Research — Knowledge Base

**Last updated:** 2026-03-15
**Status:** MTM data loaded, scale + baseline confirmed

---

## Auction Structure

| Aspect | PJM Annual | MISO Annual |
|--------|-----------|-------------|
| Period type | `a` (12-month, Jun-May) | `aq1`-`aq4` (quarterly) |
| Rounds | **4** (R1-R4) | 3 (R1-R3) |
| Settlement | 12 months | 3 months per quarter |
| Auction month convention | June (month 6) | June (month 6) |
| Class types (production) | `onpeak`, `dailyoffpeak`, `wkndonpeak` | `onpeak`, `offpeak` |
| Hedge types | `obligation` (97%) + `option` (3%) | `obligation` only |

## Key Difference from MISO: ALL Rounds Have Prior MCP

PJM R1 annual is NOT the first auction for the planning year. Long-term auctions (yr1-yr3)
clear before R1 and establish prior clearing prices. Therefore:

- **All 4 rounds have `mtm_1st_mean`** — R1 coverage = 99.9%, R2-R4 = 100%
- **No need for H baseline** (historical DA congestion fallback)
- **No need for nodal_f0 stitch** (the complex baseline MISO R1 requires)
- **Only filter to `hedge_type == 'obligation'`** — options (3%) not banded

This makes PJM annual banding **much simpler** than MISO.

## mtm_1st_mean Source (CONFIRMED)

The M2M system returns all auction clearing prices for a path, ordered by recency:

```
For R1 PY2024 annual:
  ar1Jun24 = Annual R1 (own MCP, LEAKY)
  lr5Jun24 = Long-term yr1 R5 (~March 2024, NON-LEAKY) ← this is mtm_1st
  lr4Jun24 = Long-term yr1 R4 (~December 2023)          ← this is mtm_2nd
  lr3Jun24 = Long-term yr1 R3 (~October 2023)           ← this is mtm_3rd
  ...
```

| Round | `mtm_1st_mean` source | When it clears |
|-------|----------------------|----------------|
| R1 | **Long-term yr1 Round 5** (same PY) | ~March |
| R2 | Annual R1 (same PY) | ~April (same auction event) |
| R3 | Annual R2 (same PY) | ~April (same auction event) |
| R4 | Annual R3 (same PY) | ~April (same auction event) |

**All 4 annual rounds clear in April** within the same auction event (days apart, not months).
R1's baseline is the long-term yr1 R5 MCP (~March) — NOT the prior year's R4 annual MCP.

The MAE gap between R1 (65.7) and R2-R4 (17-24) is due to sequential price discovery
within the April auction, not timing differences. R2 sees R1's price from hours/days earlier,
R3 sees R2's, etc. Each round refines within the same event.

## Scale Convention (CONFIRMED)

| Column | Scale | Formula |
|--------|-------|---------|
| `mcp` | **Annual total** (12-month) | The actual clearing price |
| `mcp_mean` | **Monthly average** | `mcp / 12` (confirmed: ratio = 12.0000) |
| `mtm_1st_mean` | **Monthly average** | Prior round's `mcp_mean` |

**Convention:** Use `mcp` (annual total) as target. Scale baseline to annual: `mtm_1st_mean * 12`.
Or equivalently, work in monthly scale (`mcp_mean` vs `mtm_1st_mean`) and multiply at the end.

**For consistency with MISO:** Use `mcp` directly and `baseline = mtm_1st_mean * 12`.

## Baseline Performance (mtm_1st_mean × 12 vs mcp)

Residual = `mcp_mean - mtm_1st_mean` (monthly scale, for comparison):

| Round | n | Bias | MAE | P95 |
|-------|--:|-----:|----:|----:|
| **R1** | 3,896,628 | -8.8 | **65.7** | 251 |
| **R2** | 4,749,876 | +4.1 | **24.4** | 88 |
| **R3** | 5,365,044 | +1.6 | **19.2** | 70 |
| **R4** | 5,362,812 | +0.1 | **17.4** | 65 |

In annual scale (×12):

| Round | MAE (annual) | P95 (annual) |
|-------|-------------:|-------------:|
| R1 | **788** | 3,012 |
| R2 | **293** | 1,056 |
| R3 | **230** | 840 |
| R4 | **209** | 780 |

**Key insight:** R1 MAE (788 annual) is 3-4x worse than R2-R4 because the long-term
auction clearing price is a weaker baseline than the prior annual round's MCP.
But it's still much better than having NO baseline (MISO R1 without nodal_f0).

**Comparison with MISO:**

| Metric | PJM R1 (annual) | MISO R1 (quarterly) |
|--------|----------------:|--------------------:|
| Baseline MAE | 788 | 792 (nodal_f0 × 3) |
| Baseline source | mtm_1st_mean (long-term MCP) | nodal_f0 (stitched f0 forwards) |
| Baseline coverage | 99.9% | 89-100% |

PJM R1 has similar MAE to MISO R1 but with 99.9% coverage and no complex stitch.

## Data Profile

| Dimension | Value |
|-----------|-------|
| Raw data | `/opt/temp/qianli/annual_research/pjm_annual_cleared_all.parquet` (238 MB) |
| With MTM | `/opt/temp/qianli/annual_research/pjm_annual_with_mcp.parquet` (403 MB) |
| Total rows (obligations) | 19.4M |
| PYs | 2017-2025 (9 years) |
| Unique paths | 194,778 |
| Rows per round | R1: 3.9M, R2: 4.7M, R3: 5.4M, R4: 5.4M |

### Class Types in Data

| Class | Rows | Production? |
|-------|-----:|:---:|
| `onpeak` | 12.1M | Yes |
| `dailyoffpeak` | 4.1M | Yes |
| `wkndonpeak` | 4.2M | Yes |
| `offpeak` | 6.6M | Legacy — filter out |
| `24h` | 0.9M | Legacy — filter out |
| `option` | 0.9M | Filter out (obligation only) |

### Split Structure

| Class | Split Rows/Trade | `mcp / sum(split)` |
|-------|---:|---:|
| onpeak | 24 | 0.5 |
| dailyoffpeak | 24 | 0.5 |
| wkndonpeak | 12 | 1.0 |

## Band Calibration Results (Preliminary)

### R1 Baseline Improvement Attempts

| Approach | MAE (monthly) | vs mtm_1st |
|----------|---:|---:|
| **mtm_1st_mean (lr5)** | **65.7** | — |
| Nodal f0 stitch | 110.3 | +68% (worse) |
| Best blend (90% mtm + 10% f0) | 64.7 | -1.4% |
| ML regression (various features) | 82.8 | +26% (worse) |
| Prior year R1 MCP blend | 54.8 | -2.7% (low coverage) |

**Conclusion:** mtm_1st_mean is the best available R1 baseline. Cannot improve >3%.

### R1 MAE By Era

| Period | MAE (monthly) | Notes |
|--------|---:|---|
| PY 2017-2020 | ~105 | Only onpeak traded in R1 |
| PY 2021 | 51.8 | Low volatility year |
| PY 2022 | 95.8 | High volatility |
| PY 2023-2025 | ~52 | All 3 classes traded, lower MAE |

### Band Width Results (All Rounds, Annual Scale)

| Round | P95 Cov | P95 HW (annual) | Baseline |
|-------|:---:|---:|---|
| R1 (all years) | 96.8% | 3,290 | LT yr1 R5 × 12 |
| R1 (recent 2yr) | 97.1% | 2,132 | LT yr1 R5 × 12 |
| R2 | 93.9% | 836 | R1 MCP × 12 |
| R3 | 93.9% | 698 | R2 MCP × 12 |
| R4 | 91.9% | 560 | R3 MCP × 12 |

### Key Findings

1. **R1 bands are 4-6x wider than R2-R4** — structural, due to LT→annual price discovery gap
2. **Recent years (2023+) have much tighter R1 bands** — using recent data only gives 35% narrower widths
3. **PY 2017-2022 R1 had only onpeak class** — dailyoffpeak/wkndonpeak appear from PY 2023
4. **R1 vs R2 residuals are uncorrelated (r=0.087)** — the price genuinely changes between March and April auctions
5. **R1/R2 MAE ratio is shrinking**: PY2017=3.4x → PY2025=1.6x, suggesting improving LT liquidity

## Next Steps

1. ~~Check mcp_mean = mcp/12~~ ✅ Confirmed
2. ~~Check R1 mtm_1st_mean coverage~~ ✅ 99.9%
3. ~~Check residual magnitude across rounds~~ ✅ Done
4. ~~Run asymmetric quantile band calibration~~ ✅ Preliminary results
5. ~~Test baseline improvement approaches~~ ✅ mtm_1st is best, can't improve >3%
6. Full V10-equivalent run with holdout validation
7. Generate comprehensive report (like MISO v9 report)
8. Production port planning
