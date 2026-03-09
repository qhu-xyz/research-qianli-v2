# Extending the LTR Pipeline to Non-f0 Period Types

**Date:** 2026-03-08
**Depends on:** Stage 5 experiment-setup.md (f0/onpeak pipeline)
**Status:** Planning — no code changes yet

---

## 1. What Period Types Exist?

V6.2B (`SPICE_F0P_V6.2B.R1`) contains data for multiple period types, not just f0.

### MISO Auction Schedule (from `iso_configs.py`)

| Auction Month | Available Period Types |
|---|---|
| 5, 6 | f0 |
| 4, 7 | f0, f1 |
| 3, 9 | f0, f1, f2 |
| 2, 8, 11 | f0, f1, f2, f3 |
| 12 | f0, f1, f2 |
| 10 | f0, f1, q3, q4 |
| 1 | f0, f1, q4 |
| 7 | f0, f1, q2, q3, q4 |

### Period Type Definitions

| Period | Meaning | Delivery Month | Example (auction=2022-09) |
|---|---|---|---|
| f0 | Current month | auction_month + 0 | Sep 2022 |
| f1 | Next month | auction_month + 1 | Oct 2022 |
| f2 | Month + 2 | auction_month + 2 | Nov 2022 |
| f3 | Month + 3 | auction_month + 3 | Dec 2022 |
| q2 | Quarter (Sep-Nov) | 3 months | Sep-Nov 2022 |
| q3 | Quarter (Dec-Feb) | 3 months | Dec 2022 - Feb 2023 |
| q4 | Quarter (Mar-May) | 3 months | Mar-May 2023 |

### Data Coverage in V6.2B

| Period Type | Months Available | Eval Window (2020-2023) |
|---|---|---|
| f0 | 106 | Full coverage |
| f1 | 89 | Good coverage |
| f2 | 54 | Moderate — only months 2,3,8,9,11,12 |
| f3 | 27 | Sparse — only months 2,8,11 |
| q2-q4 | 9-27 | Very sparse — specific months only |

---

## 2. What's the Same Across Period Types?

For a given auction_month, all period types share:

| Component | Same? | Details |
|---|---|---|
| Schema | YES | Identical 21 columns across f0/f1/f2/f3 |
| `shadow_price_da` | YES | 60-month historical lookback, computed at auction time — doesn't depend on delivery month |
| `da_rank_value` | YES | = rank(shadow_price_da) — same because shadow_price_da is same |
| V6.2B formula | YES | Same formula, same weights |
| Evaluation framework | YES | VC@k, Recall@k, Spearman, gates — all reusable |
| LTR methodology | YES | LightGBM lambdarank, query groups, rank-transform |
| Ground truth source | YES | `get_da_shadow_by_peaktype()` — just for a different delivery month |
| Constraint ID format | YES | Same string IDs, same direct join with DA |

---

## 3. What Differs Across Period Types

### 3.1 Constraint Universe (IMPORTANT)

Each period type has a **different constraint universe** for the same auction month.

Example for auction_month=2022-09:

| Period | Rows | Unique to this ptype | Shared with f0 |
|---|---|---|---|
| f0 | 600 | 282 | — |
| f1 | 623 | 207 | 274 (46%) |
| f2 | 559 | 201 | 216 (39%) |
| All three | — | — | 172 (29%) |

**Implication:** You CANNOT mix training data across period types. An f0 model trained on f0 constraints cannot predict f1 constraints — different universes. Each period type needs its own model.

### 3.2 Flow Features Differ

For shared constraints, flow-based features differ because they forecast different delivery months:

| Feature | Same across ptypes? | Why |
|---|---|---|
| `shadow_price_da` | YES | Historical lookback, independent of delivery month |
| `da_rank_value` | YES | = rank(shadow_price_da) |
| `ori_mean` | NO | Flow forecast for delivery month (max_diff ~0.48 between f0/f1) |
| `mix_mean` | NO | Same — different delivery month |
| `mean_branch_max` | NO | Same — different delivery month |
| `density_mix_rank_value` | NO | Rank of mix_mean — differs because mix_mean differs (max_diff ~0.63) |
| `density_ori_rank_value` | NO | Same logic |

### 3.3 Ground Truth: Delivery Month, NOT Auction Month

**This is the most critical difference.**

For period type fN with auction_month A:
- **Delivery month = A + N months**
- **Ground truth = realized DA for the delivery month**

```
auction_month = 2022-09

f0: ground truth = realized DA for Sep 2022 (delivery = auction)
f1: ground truth = realized DA for Oct 2022 (delivery = auction + 1)
f2: ground truth = realized DA for Nov 2022 (delivery = auction + 2)
f3: ground truth = realized DA for Dec 2022 (delivery = auction + 3)
```

```python
import pandas as pd

auction_month = pd.Timestamp("2022-09-01")
period_offset = 1  # f1

delivery_month = auction_month + pd.DateOffset(months=period_offset)
# delivery_month = 2022-10-01

# Ground truth: realized DA for October 2022
st = delivery_month
et = delivery_month + pd.DateOffset(months=1)
da_shadow = aptools.tools.get_da_shadow_by_peaktype(st=st, et_ex=et, peak_type="onpeak")
```

### 3.4 Spice6 Density: Use market_month = delivery_month

Spice6 density is partitioned by `market_month`. For non-f0 periods:

```
f0: density/auction_month=2022-09/market_month=2022-09/...
f1: density/auction_month=2022-09/market_month=2022-10/...
f2: density/auction_month=2022-09/market_month=2022-11/...
```

The `spice6_loader.py` already takes `auction_month` and `period_type` but internally needs to resolve the correct `market_month`. Currently it uses `auction_month` as `market_month` (hardcoded f0 assumption) — this must be changed.

### 3.5 Spice6 ML Predictions: Same market_month Pattern

```
ml_pred/auction_month=2022-09/market_month=2022-09/  → f0
ml_pred/auction_month=2022-09/market_month=2022-10/  → f1
ml_pred/auction_month=2022-09/market_month=2022-11/  → f2
```

These exist and have the same schema. Available for all market_months that the density covers.

### 3.6 Historical DA for Training Labels

For training an fN model for eval_month M:
- Training months: M-8 through M-1
- Each training month T needs:
  - Features from V6.2B/{T}/{fN}/onpeak (if fN exists for month T — check auction schedule)
  - Ground truth from realized DA for **T + N** (the delivery month for that training auction)

Example — training an **f1 model** for eval auction_month 2022-09:

| Training auction_month | V6.2B features from | Ground truth from | Notes |
|---|---|---|---|
| 2022-01 | V6.2B/2022-01/f1/onpeak | realized DA **Feb 2022** | f1 delivery = Jan+1 = Feb |
| 2022-02 | V6.2B/2022-02/f1/onpeak | realized DA **Mar 2022** | |
| 2022-03 | V6.2B/2022-03/f1/onpeak | realized DA **Apr 2022** | |
| 2022-04 | V6.2B/2022-04/f1/onpeak | realized DA **May 2022** | |
| 2022-05 | **SKIP** | — | f1 not available in May (schedule: f0 only) |
| 2022-06 | **SKIP** | — | f1 not available in June (schedule: f0 only) |
| 2022-07 | V6.2B/2022-07/f1/onpeak | realized DA **Aug 2022** | |
| 2022-08 | V6.2B/2022-08/f1/onpeak | realized DA **Sep 2022** | |

**Test:** V6.2B/2022-09/f1/onpeak → ground truth = realized DA **Oct 2022**

**Key complication:** Not every month has f1 data. The auction schedule determines which months have which period types. Training must skip months where the period type doesn't exist.

---

## 4. Historical DA Feature (shadow_price_da) — No Change Needed

`shadow_price_da` in V6.2B is a 60-month historical lookback computed at auction time. It does NOT depend on the delivery month (verified: identical values across f0/f1/f2 for shared constraints).

However, the `dz_flow.py` code in pbase shows that the production signal adjusts the **recent history window** based on forecast horizon:

```python
# From pbase/analysis/tools/dz_flow.py lines 64-67
if shift_period <= 2:
    recent_n_months = 2 - shift_period  # f0→2, f1→1, f2→0
else:
    recent_n_months = -1  # f3+: no recent months
```

This means for longer horizons, fewer recent months of historical DA are used (because those months haven't happened yet at prediction time). But this adjustment is baked into how `shadow_price_da` was computed upstream — we don't need to replicate it. The V6.2B parquet already has the correct `shadow_price_da` per period type (and it happens to be identical because the 60-month lookback dominates).

For the **custom historical DA features** (Group E in experiment-setup.md), we DO need to adjust the lookback window:

```python
# For f0 (shift=0): cutoff = auction_month - 1
# For f1 (shift=1): cutoff = auction_month - 1 (same — DA for auction_month hasn't happened yet)
# For f2 (shift=2): cutoff = auction_month - 1 (same reasoning)
# The cutoff is always auction_month - 1 because we run before the auction
cutoff_month = auction_month - MonthBegin(1)
```

The cutoff doesn't change with period type because it's based on **when we run** (before the auction), not the delivery month. The V6.7B pipeline confirms this (`data_loader.py` line 198: `cutoff_month = auction_month - pd.offsets.MonthBegin(1)`).

---

## 5. Implementation Plan

### Phase 1: f0 First (Current Stage 5 Plan)
Complete the f0 pipeline end-to-end. This is the baseline.

### Phase 2: Parameterize for Period Type

**Registry structure is already in place.** Results are stored at `registry/{period_type}/{class_type}/{version_id}/`,
with per-slice `gates.json` and `champion.json`. Use `ml.registry_paths` helpers (e.g., `registry_root()`,
`holdout_root()`) to construct paths — never hardcode. This means adding a new period type (e.g., f1)
just requires running experiments that write to `registry/f1/onpeak/`.

Remaining code changes needed to generalize from f0 to fN:

| Module | Change |
|---|---|
| `config.py` | Add `period_offset: int` to `PipelineConfig` (0=f0, 1=f1, etc.) |
| `data_loader.py` | Compute `delivery_month = auction_month + period_offset` for ground truth fetch |
| `data_loader.py` | Pass correct `market_month` to spice6_loader (= delivery_month) |
| `data_loader.py` | Skip training months where period type doesn't exist (check auction schedule) |
| `spice6_loader.py` | Accept `market_month` parameter instead of assuming `= auction_month` |
| Ground truth cache | Cache realized DA by delivery_month (already needed for f0 — same cache works) |
| `benchmark.py` | Filter eval months to those where period type exists |

### Phase 3: Run f1

- Same versioning: v0 (formula baseline), v1 (LTR Groups A+B), v1b (+ da_rank_value), etc.
- Expect different performance characteristics: f1 is harder (forecasting further ahead)
- Fewer training months available (f1 missing in May/June)

### Phase 4: f2/f3 if Warranted

- f2: 54 months — limited but usable
- f3: 27 months — may be too sparse for reliable training/eval
- Quarterly (q2-q4): each covers 3 delivery months — need to decide how to handle ground truth aggregation (sum across 3 months? per-month?)

---

## 6. Quarter Period Types (q2, q3, q4) — Open Questions

Quarter types are more complex:

- **q2 covers Sep-Nov**: a single set of constraints ranked for 3 months
- **Ground truth**: Should it be realized DA summed across all 3 months? Or the worst month?
- **Constraint universe**: Different from fN — need to verify schema/overlap
- **Training**: Very sparse — q2 only exists in July auctions (9 months total)

Recommendation: Skip quarters until f0/f1/f2 are done. They require separate design decisions.

---

## 7. Auction Schedule Reference

From `iso_configs.py` — determines which months have which period types:

```python
MISO_AUCTION_SCHEDULE = {
    5: ["f0"],                              # May: f0 only
    6: ["f0"],                              # Jun: f0 only
    7: ["f0", "f1", "q2", "q3", "q4"],      # Jul: most types
    8: ["f0", "f1", "f2", "f3"],            # Aug: all monthly
    9: ["f0", "f1", "f2"],                  # Sep
    10: ["f0", "f1", "q3", "q4"],           # Oct
    11: ["f0", "f1", "f2", "f3"],           # Nov: all monthly
    12: ["f0", "f1", "f2"],                 # Dec
    1: ["f0", "f1", "q4"],                  # Jan
    2: ["f0", "f1", "f2", "f3"],            # Feb: all monthly
    3: ["f0", "f1", "f2"],                  # Mar
    4: ["f0", "f1"],                        # Apr
}
```

Use this to determine which training months to skip when building rolling windows for non-f0 models. A month missing from the schedule for a given period type means V6.2B won't have data for that period type.

---

## 8. Key Formulas

```python
# Period type to offset
def period_offset(period_type: str) -> int:
    """f0→0, f1→1, f2→2, f3→3."""
    return int(period_type[1:])

# Delivery month from auction month and period type
def delivery_month(auction_month: str, period_type: str) -> str:
    offset = period_offset(period_type)
    dt = pd.Timestamp(auction_month) + pd.DateOffset(months=offset)
    return dt.strftime("%Y-%m")

# Check if period type exists for a given auction month
def has_period_type(auction_month: str, period_type: str) -> bool:
    month_num = pd.Timestamp(auction_month).month
    schedule = MISO_AUCTION_SCHEDULE.get(month_num, ["f0"])
    return period_type in schedule

# Spice6 density/ml_pred market_month
def market_month(auction_month: str, period_type: str) -> str:
    return delivery_month(auction_month, period_type)  # same thing
```
