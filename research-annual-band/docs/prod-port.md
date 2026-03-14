# V10 Annual Band Generator — Production Port Plan

**Date:** 2026-03-14 (rewritten to resolve contradictions)
**Version:** V10 (empirical asymmetric quantile bands)
**Status:** Research complete. Pre-port validation pending.
**Production Target:** PY 2026 annual auction (R1 ~April 2026, R2 ~May, R3 ~June)

---

## Resolved Design Decisions

These decisions are final. Everything below is consistent with them.

1. **Scale:** MONTHLY internally. All baselines, band edges, and artifact values are monthly.
   `scale_bid_prices_by_duration()` (untouched) converts bid_price columns to quarterly (×3) at the end.

2. **Integration seam:** `elif` branch inside `generate_bands_for_group()` in `band_generator.py`.
   The annual branch returns `result` (a DataFrame with `bid_price_1..10, clearing_prob_1..10` already assigned).
   It skips `apply_corrected_bid_prices()` by returning early before that call.

3. **Nodal f0 lookup:** Node-level, keyed by `(node_id, period_type, planning_year, class_type)`.
   Path baseline computed at inference: `baseline = sink_mcp - source_mcp` (monthly).

4. **CP assignment:** Handled inside the annual generator using empirical rates from the artifact.
   `apply_corrected_bid_prices()` is NOT called for annual — the annual branch returns early.

---

## PY 2026 Data Availability

| Data | Available Now? | When Available |
|------|:-:|:---|
| Training data (PY 2019-2025 historical MCPs) | **Yes** | Now |
| R1 baseline (nodal_f0 for PY 2026 delivery) | **No** | ~April 2026 |
| R2/R3 baseline (prior round MCP) | **No** | After prior round clears |

**Steps 1-2 can run NOW.** Steps 3-6 need ~April data for R1 (R2/R3 can use historical for testing).

---

## Pre-Port Validation Checklist

All steps run in the research repo. No production code touched.
All values are in **monthly scale** (matching the resolved production contract).

### 1. Holdout Validation (PY 2025)

**What:** Run V10 on full PYs (2020-2025) with min_train_pys=3.

**How:** In `run_v9_bands.py`, set `VERSION_ID = "v10_holdout"`, include PY 2025, set `MIN_TRAIN_PYS = 3`.

**Success criteria:** PY 2025 fold P95 coverage within 5pp of dev average per round.

### 2. Empirical Clearing Probabilities in Artifact

**What:** Add actual buy/sell clearing rates to `calibration_artifact.json`.

**Why:** Theoretical CPs are off by 2-5pp at P50-P80. Empirical CPs from temporal CV are more accurate.

**Schema:**
```json
{
  "calibration": {
    "aq1": {
      "boundaries": [0, 42.3, 109.1, 248.5, 620.0, "inf"],
      "bin_labels": ["q1", "q2", "q3", "q4", "q5"],
      "bin_pairs": { "q1": { "onpeak": { "p95": [-180.2, 120.5], ... }, ... }, ... },
      "empirical_clearing_rates": {
        "buy_prevail":     { "lower_p95": 5.4, "upper_p95": 98.0, ... },
        "buy_counter":     { "lower_p95": 2.4, "upper_p95": 96.6, ... },
        "sell_prevail":    { "lower_p95": 94.6, "upper_p95": 2.0, ... },
        "sell_counter":    { "lower_p95": 97.6, "upper_p95": 3.4, ... },
        "buy_prevail_q5":  { "lower_p95": 13.9, "upper_p95": 94.6, ... },
        "buy_counter_q5":  { "lower_p95": 2.4, "upper_p95": 94.6, ... },
        "sell_prevail_q5": { "lower_p95": 86.1, "upper_p95": 5.4, ... },
        "sell_counter_q5": { "lower_p95": 97.6, "upper_p95": 5.4, ... }
      }
    }
  }
}
```

All values in monthly scale. `boundaries` are monthly |baseline| thresholds.

### 3. Nodal f0 Lookup Table (R1 Only)

**What:** Pre-compute per-node f0 MCPs for the target PY's delivery months.

**Schema:** Node-level, NOT path-level.

| Column | Type | Notes |
|--------|------|-------|
| node_id | str | MISO pnode ID |
| period_type | str | aq1-aq4 (determines delivery months) |
| planning_year | int | PY of the auction |
| class_type | str | onpeak or offpeak |
| node_mcp | float | Monthly avg of f0 MCPs for delivery months |

Key: `(node_id, period_type, planning_year, class_type)`.

**Deriving `planning_year`:** For MISO annual auctions, `auction_month` is always June of the
planning year (convention in pbase). So: `planning_year = int(auction_month[:4])`.
Example: `auction_month = "2026-06"` → `planning_year = 2026`.
This is computed inside `generate_annual_bands_for_group()`, not passed by the caller.

**Why node-level, not path-level:** The research code (`run_aq1_experiment.py` lines 396-439) stores
node-level MCPs and computes `nodal_f0 = sink_mcp - source_mcp` at join time. Storing path-level
would require pre-computing the cross product of all (source, sink) pairs (~27K² = 729M rows).
Node-level is ~50K rows and the path stitch is a simple join.

**At inference:** Join on `(node_id=source_id, period_type, planning_year, class_type)` for source,
then `(node_id=sink_id, ...)` for sink. `baseline = sink_mcp - source_mcp` (monthly).
`planning_year` derived from `auction_month` as above.

**Blocked until ~April 2026** for PY 2026 data. Can test with PY 2025 historical now.

### 4. End-to-End Inference Test

**What:** Simulate the production inference path in the research repo.

**Input:** Sample R1 trades with columns: `source_id, sink_id, class_type, trade_type, period_type, round`.

**Steps:**
1. Join nodal lookup → compute `baseline = sink_mcp - source_mcp` (monthly)
2. Load calibration artifact for this (round, period_type)
3. Assign bins: `|baseline|` vs `boundaries` from artifact → bin label
4. Look up (bin, class) quantile pairs from `bin_pairs`
5. Compute band edges: `lower/upper = baseline + lo/hi` (monthly)
6. Determine `flow_type` = prevail if baseline > 0, counter otherwise
7. Determine `bin_group` = "q5" if `|baseline| >= boundaries[4]`, else "" (base key)
8. Look up empirical CP: key = `f"{trade_type}_{flow_type}_q5"` if q5 else `f"{trade_type}_{flow_type}"`.
   If q5 key missing in artifact, fall back to base key.
9. Select 10 band edges, sort by trade_type, output `bid_price_1..10, clearing_prob_1..10`

**Output columns (monthly scale):** `baseline, lower_p10..upper_p99, bid_price_1..10, clearing_prob_1..10`

`scale_bid_prices_by_duration()` will multiply `bid_price_*` by 3 downstream.

**Success criteria:**
- No null band edges (except H-fallback paths)
- CPs in 0-100, monotonic per row
- `bid_price_*` sorted descending for buy, ascending for sell
- <10s for ~30K paths

### 5. Fallback for Missing Nodes (R1)

1. `nodal_f0` (monthly) — 89-100% coverage
2. `mtm_1st_mean` (H baseline, monthly, already in trades) — 100% coverage
3. Every path MUST get bands. Explicit warning on each fallback.

### 6. Bug Checks

- [ ] `mcp_mean` not used anywhere in inference code
- [ ] All baselines in MONTHLY scale (no ×3 in the generator)
- [ ] class_type validated (`raise ValueError` on unexpected values)
- [ ] No silent fallbacks
- [ ] No `.get(key, {})` on critical calibration data
- [ ] CPs validated: 0-100 range, monotonic
- [ ] Band edges in monthly scale; `scale_bid_prices_by_duration()` handles ×3

---

## Production Port (after all 6 pre-port steps pass)

### Existing Production Flow

```
BaseFtrModel.run() (base.py:566)
  └── generate_bands(df=total_trades, ...) (band_generator.py:1563)
        └── for each (auction_month, period_type) group:
              generate_bands_for_group(trades, auction_month, period_type, ...)
                ├── if period_type in {"f0", "f1"}:
                │     result, val_df = generate_f0_f1_bands(...)  # LightGBM
                │
                ├── else:  # f2+, q*, aq*, a, yr*
                │     result, val_df = generate_f2p_bands(...)    # rule-based
                │
                ├── empirical_rates = calculate_empirical_clearing_rates(val_df)
                ├── result = apply_corrected_bid_prices(result, empirical_rates)
                ├── result = scale_bid_prices_by_duration(result)   # ×3 for aq*
                └── return result
```

### Changed Flow (annual only)

```python
# In generate_bands_for_group() (band_generator.py:1422):

lgbm_ptypes = {"f0", "f1"}
annual_ptypes = {"aq1", "aq2", "aq3", "aq4"}

if period_type in lgbm_ptypes:
    # ... unchanged ...

elif period_type in annual_ptypes:
    # V10: asymmetric bands + CP from pre-calibrated artifact
    from .annual_band_generator import generate_annual_bands_for_group
    round_num = int(trades["round"].iloc[0])
    result = generate_annual_bands_for_group(
        trades=trades.copy(),
        period_type=period_type,
        class_type=class_type,
        auction_month=auction_month,
        round_num=round_num,
    )
    # Annual handles CP assignment and bid_price sorting internally.
    # Skip apply_corrected_bid_prices() — go directly to scaling.
    result = scale_bid_prices_by_duration(result)
    return result

else:
    # ... unchanged (f2+, q*) ...

# Below here: existing apply_corrected_bid_prices + scale (runs for f0/f1 and f2p, NOT annual)
```

**Key:** The annual branch returns EARLY after `scale_bid_prices_by_duration()`.
It does NOT fall through to `apply_corrected_bid_prices()`.

### `generate_annual_bands_for_group()` — Return Contract

Returns a single `pd.DataFrame` (same as `generate_bands_for_group()`), NOT a tuple.

The returned DataFrame has:
- All original trade columns preserved
- `baseline` column (monthly)
- `bid_price_1..10` (monthly, BEFORE scaling — `scale_bid_prices_by_duration` handles ×3)
- `clearing_prob_1..10` (0-100)

### New File: `annual_band_generator.py`

```python
def generate_annual_bands_for_group(
    trades: pd.DataFrame,
    period_type: str,         # aq1-aq4
    class_type: str,          # onpeak/offpeak
    auction_month: str,       # e.g. "2026-06"
    round_num: int,           # 1, 2, or 3
    artifact_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Generate V10 asymmetric bands for one annual group.

    Returns a single DataFrame with bid_price_1..10 and clearing_prob_1..10.
    Band edges and bid_prices are in MONTHLY scale.
    Caller (generate_bands_for_group) applies scale_bid_prices_by_duration() after.
    """
```

### Nodal f0 Lookup

Stored as node-level lookup: `(node_id, period_type, planning_year, class_type) → node_mcp`.

At inference:
```python
# Join source
trades = trades.merge(lookup.rename(columns={"node_id": "source_id", "node_mcp": "_src_mcp"}),
                      on=["source_id", join_keys], how="left")
# Join sink
trades = trades.merge(lookup.rename(columns={"node_id": "sink_id", "node_mcp": "_snk_mcp"}),
                      on=["sink_id", join_keys], how="left")
# Stitch
trades["baseline"] = trades["_snk_mcp"] - trades["_src_mcp"]
```

For R2/R3: `baseline = mtm_1st_mean` (already in trades, already monthly).

### What Changes

- `generate_bands_for_group()` in `band_generator.py` — add `elif annual_ptypes` branch with early return
- New file: `annual_band_generator.py`
- New artifacts: `calibration_artifact.json` per round + `nodal_f0_lookup_PY{YYYY}.parquet`

### What Does NOT Change

- `autotuning.py` — untouched
- `generate_bands()` — untouched
- `scale_bid_prices_by_duration()` — untouched
- `apply_corrected_bid_prices()` — untouched (simply not reached for annual due to early return)
- `BaseFtrModel.run()` — untouched
- All monthly f0p logic — untouched

---

## Timeline

| Step | Est. Effort | Can Do Now? |
|------|:-:|:-:|
| 1. Holdout validation | 5 min | **Yes** |
| 2. Empirical CPs in artifact | 30 min | **Yes** |
| 3. Nodal f0 lookup (PY 2025 for test) | 1 hour | **Yes** (historical) |
| 3b. Nodal f0 lookup (PY 2026 for prod) | 30 min | **No** (~April) |
| 4. E2E inference test | 1 hour | **Yes** (with PY 2025 data) |
| 5. Fallback chain | 30 min | **Yes** |
| 6. Bug checks | 30 min | **Yes** |
| **Total pre-port** | **~4 hours** | |
| 7. Build `annual_band_generator.py` | 2 hours | After steps 1-6 |
| 8. Integration test | 1 hour | After step 7 |
