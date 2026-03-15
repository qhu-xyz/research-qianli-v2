# PJM Annual Band Research — Knowledge Base

**Last updated:** 2026-03-15
**Status:** Data exploration in progress

---

## Auction Structure

| Aspect | PJM Annual | MISO Annual |
|--------|-----------|-------------|
| Period type | `a` (12-month, Jun-May) | `aq1`-`aq4` (quarterly) |
| Rounds | **4** (R1-R4) | 3 (R1-R3) |
| Settlement | 12 months | 3 months per quarter |
| Auction month convention | June (month 6) | June (month 6) |
| Class types (production) | `onpeak`, `dailyoffpeak`, `wkndonpeak` | `onpeak`, `offpeak` |

## Key Difference from MISO: R1 Has Prior Round MCP

PJM R1 annual is NOT the first auction for the planning year. Long-term auctions (yr1-yr3)
clear before R1 and establish prior clearing prices. Therefore:

- **All 4 rounds have `mtm_1st_mean`** (prior round/long-term MCP)
- **No need for H baseline** (historical DA congestion fallback)
- **No need for nodal_f0 stitch** (the complex baseline computation MISO R1 requires)

This makes PJM annual banding SIMPLER than MISO.

## Data Profile

| Dimension | Value |
|-----------|-------|
| Data source | `/opt/temp/qianli/annual_research/pjm_annual_cleared_all.parquet` |
| Total rows | 27.9M |
| PYs | 2017-2025 (9 years) |
| Unique paths | 194,778 |
| Rows per round | R1: 5.6M, R2: 6.8M, R3: 7.7M, R4: 7.7M |

### Class Types in Data

| Class | Rows | Production? |
|-------|-----:|:---:|
| `onpeak` | 12.1M | Yes |
| `dailyoffpeak` | 4.1M | Yes |
| `wkndonpeak` | 4.2M | Yes |
| `offpeak` | 6.6M | Legacy — filter out |
| `24h` | 0.9M | Legacy — filter out |

### Split Structure

| Class | Split Rows/Trade | `mcp / sum(split)` |
|-------|---:|---:|
| onpeak | 24 | 0.5 |
| dailyoffpeak | 24 | 0.5 |
| wkndonpeak | 12 | 1.0 |

`mcp` = annual total clearing price (12-month). This is the prediction target.

## Scale Convention

**TBD — need to verify:**
- Is `mcp_mean = mcp / 12` (monthly average)?
- Or is `mcp_mean` computed differently?
- All baselines and band widths should be in **annual scale** (matching `mcp`)

## Baseline Strategy

**All rounds use `mtm_1st_mean` (prior round MCP).** Unlike MISO where R1 requires
a separate nodal_f0 baseline.

**TBD:**
- What scale is `mtm_1st_mean` in? Monthly or annual?
- How does `get_m2m_mcp_for_trades_all()` handle PJM annual?
- R1 vs R2-R4 residual comparison needed

## Open Questions

1. What is the relationship between `mcp` and `mcp_mean` for PJM annual?
2. Do all 4 rounds have good `mtm_1st_mean` coverage?
3. How does residual magnitude compare across rounds?
4. Is the same asymmetric quantile approach applicable?
5. What bin count and MIN_CELL_ROWS work with 194K paths?
