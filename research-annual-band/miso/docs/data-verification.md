# MISO Canonical Data Verification Report

**Date:** 2026-03-19
**Canonical source:** `MisoApTools.get_trades_of_given_duration(participant=None, start_month='2018-06', end_month_in='2026-06')`
**Pipeline:** `get_trades_of_given_duration` → `merge_cleared_volume(merge_mcp=True)` → `get_m2m_mcp_for_trades_all` → filter buy-only → derive planning_year → drop mcp_mean

## Data Summary

| Item | Value |
|------|-------|
| Total unique paths | 2,390,713 |
| Total monthly rows | 3,972,807 |
| PYs | 2018-2025 (8 years) |
| Rounds | 1, 2, 3 |
| Quarters | aq1, aq2, aq3, aq4 |
| Classes | onpeak, offpeak |
| MCP scale | Quarterly (mcp / mcp_mean ratio = 3.000) |
| `mcp_mean` | Dropped after verification (deprecated) |
| Trade type | Buy only (219,836 sells dropped = 5.2%) |
| `break_offpeak` | Confirmed no-op for MISO |

## Dev / Holdout Split

| Round | Dev (PY 2018-2022) | Holdout (PY 2023-2025) | Holdout % |
|-------|-------------------:|----------------------:|----------:|
| R1 | 361,646 | 329,763 | 48% |
| R2 | 424,987 | 406,916 | 49% |
| R3 | 451,756 | 415,645 | 48% |

## Key Column Availability

| Column | R1 | R2 | R3 |
|--------|:--:|:--:|:--:|
| mcp | 0% null | 0% null | 0% null |
| mtm_1st_mean | **100% null** (no prior annual round) | 0% null | 0% null |
| mtm_2nd_mean | — | varies | varies |
| bid_price | 0% null | 0% null | 0% null |
| cleared_volume | 0% null | 0% null | 0% null |

R1 mtm_1st_mean is 100% null because R1 is the first annual round — there is no prior round MCP.
R1 baseline uses nodal_f0 stitch (computed separately via MisoCalculator).

## Consistency with Old Data

Compared on shared PYs (2019-2025), buy-only trades.

| Check | Result | Gate | Status |
|-------|--------|------|--------|
| R1 row counts (aq3/aq4) | Exact match | ±1% | **PASS** |
| R1 row counts (aq1/aq2) | ≤0.9% diff | ±1% | **PASS** |
| R2 row counts (old buy-only) | Exact match | ±1% | **PASS** |
| R3 row counts (old buy-only) | Exact match | ±1% | **PASS** |
| MCP values | 100% match (max diff 0.005) | corr > 0.999 | **PASS** |
| mtm_1st_mean values | 100% match (max diff 0.0000) | exact | **PASS** |
| Path overlap (full key) | 100% of old buy paths in new | >98% | **PASS** |
| Class types per PY | Identical (shared PYs) | identical | **PASS** |
| MCP scale | ratio = 3.000 | 2.9-3.1 | **PASS** |

## Incremental Coverage vs Old

| Item | Old | New |
|------|-----|-----|
| PY range | 2019-2025 | **2018-2025** (+1 year) |
| Sell trades | Included | **Excluded** (buy-only) |
| Data source | `get_all_cleared_trades()` | `get_trades_of_given_duration()` |
| `mcp_mean` | Present (monthly) | Dropped after verification |

## Row Granularity

The canonical data has multiple rows per path (ratio ~1.25-2.27x depending on round/quarter).
This is from multiple cleared bid points per path (different trades_id, bid_price, bid_volume),
NOT from split_market_month expansion. MCP and mtm_1st_mean are constant within each path group
(verified: max within-group std = 0.0 for both columns).

## Cached Artifacts

| File | Rows | Size | Description |
|------|-----:|-----:|-------------|
| `miso/data/canonical_annual_paths.parquet` | 2,390,713 | 275 MB | One row per path (deduplicated) |
| `miso/data/canonical_annual_monthly.parquet` | 3,972,807 | 337 MB | All rows (multiple bid points per path) |

## Known Issues

- `mar_2026` partition has ArrowTypeError (schema mismatch in `market_round` field). Does not affect
  current data (PY2018-2025). Flagged as Phase 2 risk for March 2026 revenue features.
