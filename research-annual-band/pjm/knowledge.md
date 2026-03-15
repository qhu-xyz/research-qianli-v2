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

## Next Steps

1. ~~Check mcp_mean = mcp/12~~ ✅ Confirmed
2. ~~Check R1 mtm_1st_mean coverage~~ ✅ 99.9%
3. ~~Check residual magnitude across rounds~~ ✅ Done
4. Compute residuals in annual scale, run asymmetric quantile band calibration
5. Temporal CV with same framework as MISO
6. Compare PJM vs MISO results
