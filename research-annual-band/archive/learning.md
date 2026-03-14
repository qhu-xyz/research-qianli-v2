# Annual FTR Trading Framework — Stage 1 Research (Adversarial Review)

> **Methodology:** This document was produced through adversarial self-review — every claim
> is traced to a specific file:line, and contradictions between the codebase and prior
> assumptions are flagged with **[CORRECTION]** markers. Unresolved questions are marked
> **[OPEN]**.

## Table of Contents
1. [Task Intent](#1-task-intent)
2. [Market Structure — Annual Auctions](#2-market-structure)
3. [Which Framework Actually Runs Annual in Production?](#3-which-framework-runs-annual)
4. [FTR24/v1 Pipeline Architecture](#4-ftr24v1-pipeline)
5. [MCP Prediction — The Core Question](#5-mcp-prediction)
6. [`_fill_mtm` — Complete Verified Code](#6-fill-mtm)
7. [`fill_mtm_1st_period_with_hist_revenue` — The Parallel Implementation](#7-fill-mtm-pbase)
8. [The 0.85 Shrinkage Factor — Forensic Analysis](#8-shrinkage-factor)
9. [Band Generator — DOES NOT Support Annual](#9-band-generator)
10. [`_set_bid_curve` — Legacy, Not Active](#10-set-bid-curve)
11. [Bid Price Post-Processing: What "simple_v3" Actually Means](#11-simple-v3)
12. [Optimizer — Path Selection, Bid Points, Volume Assignment](#12-optimizer)
13. [Clearing Probabilities — V6 Empirical](#13-clearing-probabilities)
14. [Annual-Specific Configuration (Production Params)](#14-annual-config)
15. [Signal Architecture for Annual](#15-signals)
16. [PJM Annual — Detailed Analysis](#16-pjm-annual)
17. [SPP Annual](#17-spp-annual)
18. [Submission Notebooks — Actual Production Workflow](#18-notebooks)
19. [FTR23 Legacy — What Actually Ran for PY 2025](#19-ftr23-legacy)
20. [Historical Performance](#20-performance)
21. [Data Access — APTools API](#21-data-access)
22. [Prior Research Applicable to Annual](#22-prior-research)
23. [Answering the Memo's Open Questions](#23-answers)
24. [Key Files Reference](#24-key-files)
25. [Critical Gaps and Open Items for Stage 2](#25-gaps)

---

## 1. Task Intent

**Goal:** Understand the existing annual FTR bid pricing framework, then design an improved
MCP prediction formula and bid price strategy for annual auctions.

**Key challenge:** For MISO Annual Round 1, there is no `mtm_1st_mean` (previous auction
clearing price). How do we predict MCP?

**Desired pattern:** Replicate the f0p model's approach — predict MCP baseline → set bid
price bands → assign clearing probabilities → optimize volumes — adapted for annual's
unique constraints.

---

## 2. Market Structure — Annual Auctions

### MISO
- **4 quarterly periods:** `aq1` (Jun–Aug), `aq2` (Sep–Nov), `aq3` (Dec–Feb), `aq4` (Mar–May)
- **3 auction rounds:** R1, R2, R3
- **Planning year:** June Y to May Y+1
- **R1 submission:** ~April (before PY starts)
- **R1 key issue:** No previous clearing price → `mtm_1st_mean` = NaN
- **R2–R3:** `mtm_1st_mean` = previous round's MCP

### PJM
- **Period types:** `a` (annual), `yr1`, `yr2`, `yr3` (long-term)
- **4 auction rounds:** R1, R2, R3, R4
- **3 class types:** onpeak, dailyoffpeak, wkndonpeak
- **MTM availability:** PJM has MTMs for ALL rounds including R1 (via `get_m2m_mcp_for_trades_all`)
  - **[CORRECTION]** The first version of this doc said "to verify extent" — now verified:
    PJM does NOT have the MISO R1 problem. PJM fetches MTM normally for all rounds.
  - Source: `trade_finalizer.py:66-72` — the R1 NaN bypass is conditional on `model._rto == "miso" or model._rto == "spp"` only.

### SPP
- **Period types:** `af0`, `af1`, `af2`, `af3`, `aq2`, `aq3`, `aq4`
- Same R1 issue as MISO — no prior clearing for first round
- **[CORRECTION]** SPP has NO dedicated FtrModel subclass in FTR24/v1. Handled generically.
- SPP `fill_mtm` in pbase has source shrinkage COMMENTED OUT (`pbase/analysis/tools/spp.py:358`)

---

## 3. Which Framework Actually Runs Annual in Production?

**[CORRECTION]** This is the single most important finding of the adversarial review.

The FTR24/v1 framework has annual params files (`miso_a_offpeak.py`) and pipeline
support, **BUT** the PY 2025 submission notebooks show:

| Round | MISO Framework | PJM Framework |
|-------|---------------|---------------|
| R1 | `power_trading.model.ftr23.v1.miso_models_a_prod_r1` | `power_trading.model.ftr23.v1.pjm_models_a_prod_r1` |
| R2 | `pmodel.base.ftr23.v2` (inferred from R3 pattern) | `pjm_models_a_prod_r2` with `use_prod_bids=True` |
| R3 | `pmodel.base.ftr23.v2.autotuning` with `miso_models_a_prod_r3` | `pjm_models_a_prod_r3` (inferred) |

**Evidence:**
- MISO R3 offpeak notebook cell: `from pmodel.base.ftr23.v2.autotuning import FtrModelParamAutoTuning`
  - File: `/home/xyz/workspace/pmodel/notebook/hz/2025-planning-year/2025-26-annual/miso/submission/r3/trades/generate_trades_on.ipynb`
- PJM R1 notebook: `module_name=f"power_trading.model.ftr23.v1.{rto}_models_a_prod_r1"`
  - File: `/home/xyz/workspace/pmodel/notebook/hz/2025-planning-year/2025-26-annual/pjm/submission/r1/trades/generate_trades.ipynb`

**Implication:** FTR24/v1 annual is configured but may not have been used in PY 2025
production. The FTR24/v1 band generator **cannot** run annual (see Section 9). It is
likely that FTR24/v1 annual params exist for PY 2026, while PY 2025 ran on FTR23.

**[OPEN]** Verify whether FTR24/v1 will be used for PY 2026 annual, and if so, how the
band generator gap will be addressed.

---

## 4. FTR24/v1 Pipeline Architecture

**Entry point:** `FtrModelParamAutoTuning.generate_trades()` in `autotuning.py`

**Pipeline stages:**
```
1. _init_auction_params()      → validate period type for auction month
2. load_all_constraint_sets()  → load signals (constraint_loader.py)
3. generate_candidate_trades() → path pool generation (trade_generator.py)
4. reduce_trades_by_exposure() → filter by exposure (trade_filter.py)
5. sell_trades_by_exposure()   → identify positions to sell
6. process_existing_positions()→ MTM and pricing (trade_finalizer.py)
7. [generate_bands()]          → MCP bands + clearing probs (band_generator.py)
8. Path selection optimization → (optimizer/path_selection.py)
9. Bid point selection         → (optimizer/bid_points.py)
10. Volume assignment          → (optimizer/volume_assignment.py)
11. finalize_trades()          → format for submission
```

**Key abstraction:** `BaseFtrModel` (base) → `MisoFtrModel` (miso_base.py), `PjmFtrModel` (pjm_base.py)
**Directory:** `/home/xyz/workspace/pmodel/src/pmodel/base/ftr24/v1/`

**[CORRECTION]** Step 7 (`generate_bands()`) is called at `base.py:467`:
```python
total_trades = generate_bands(df=total_trades, treat_all_as_buy=True, base_path=_base_path)
```
This call does NOT check period type before invoking. For annual (aq1-aq4), it WILL FAIL
(see Section 9).

---

## 5. MCP Prediction — The Core Question

### 5.1 For f0p (Monthly) Trades — The Reference Pattern

```
baseline = w_mtm * mtm_1st_mean + w_rev * 1(rev)
```
Where:
- `mtm_1st_mean` = most recent clearing price (from last auction)
- `1(rev)` = current month's revenue, scaled
- Weights vary by period type (f0: 0.77/0.23, f1: 0.85/0.15, etc.)
- Then: `mcp_pred = y_pred_mcp` (from ML model) or `mcp_pred = mtm_1st_mean` (if no ML)

### 5.2 For Annual Trades — Current Approach

**In `trade_finalizer.py:66-72` (`add_price_related_columns`):**
```python
if (model._rto == "miso" or model._rto == "spp")
    and set(trades["period_type"]).issubset(model._aptools.tools.annual_period_types)
    and list(map(int, trades["round"].unique())) == [1]:
    trades["mtm_1st_mean"] = np.nan    # Force NaN for MISO/SPP R1
else:
    trades = model._aptools.tools.get_m2m_mcp_for_trades_all(trades_all=trades)
```

**Then at `trade_finalizer.py:74-76`:**
```python
if trades["mtm_1st_mean"].isna().all():
    trades = model._fill_mtm(trades)
```

### 5.3 Final MCP Prediction

**At `trade_finalizer.py:85-94`:**
```python
if generate_features_params_dict["prediction_class_instance"] is None:
    trades["mcp_pred"] = trades["mtm_1st_mean"]           # Current annual behavior
else:
    trades = make_prediction(...)
    scaler = 1
    trades["mcp_pred"] = trades["y_pred_mcp"] * scaler + trades["mtm_1st_mean"] * (1 - scaler)
```

**For annual currently:** `prediction_class_instance = None` (set in `miso_a_offpeak.py:351`), so:
```
mcp_pred = mtm_1st_mean = historical DA congestion (from _fill_mtm, with 0.85 shrinkage)
```

**This means:** For annual R1, the MCP prediction is last year's DA congestion prices
with 0.85 shrinkage. No ML model, no 1(rev), no sophisticated prediction.

---

## 6. `_fill_mtm` — Complete Verified Code

**Location:** `miso_base.py:221-261`

```python
def _fill_mtm(self, trades):
    self.logger.info("filling mtm for MISO trades")
    non_mtm_trades = trades[trades["mtm_1st_mean"].isna()].copy()
    mtm_trades = trades[trades["mtm_1st_mean"].notna()].copy()
    if non_mtm_trades.empty:
        return trades
    planning_year = trades["auction_date"].unique()[0].year
    period_type = trades["period_type"].unique()[0]
    class_type = trades["class_type"].unique()[0]
    assert period_type in ["aq1", "aq2", "aq3", "aq4"]
    month_st = pd.Timestamp(f"{planning_year}-06") + pd.offsets.MonthBegin((int(period_type[2]) - 1) * 3)
    cutoff_month = pd.Timestamp(f"{planning_year}-04")
    month_list = [
        month_st + pd.offsets.MonthBegin(i) - pd.DateOffset(years=j)
        for i in range(3) for j in range(1, 2)
    ]
    month_list = [month for month in month_list if month < cutoff_month]
    list_da = [self._aptools.tools._da_lmp_monthlyagg_loader.load_data(loading_month=month)
               for month in month_list]
    da = pd.concat(list_da)
    da = da[da["class_type"] == class_type].copy()
    da["planning_year"] = da["year"].astype(int) - (da["month"].astype(int) < 6).astype(int)
    common_nodes = set(non_mtm_trades["source_id"]).union(
        non_mtm_trades["sink_id"]).intersection(da["pnode_id"])
    non_mtm_trades = non_mtm_trades[
        non_mtm_trades["source_id"].isin(common_nodes) &
        non_mtm_trades["sink_id"].isin(common_nodes)
    ].copy()
    source_da_df = da[da["pnode_id"].isin(non_mtm_trades["source_id"])]
    sink_da_df = da[da["pnode_id"].isin(non_mtm_trades["sink_id"])]
    source_da_df.loc[source_da_df["congestion_price_da_monthly"] < 0,
                     "congestion_price_da_monthly"] *= 0.85
    sink_da_df.loc[sink_da_df["congestion_price_da_monthly"] > 0,
                   "congestion_price_da_monthly"] *= 0.85
    source_da_df = source_da_df.groupby(["planning_year", "pnode_id"])[
        "congestion_price_da_monthly"].mean()
    sink_da_df = sink_da_df.groupby(["planning_year", "pnode_id"])[
        "congestion_price_da_monthly"].mean()
    source_da_df = source_da_df.groupby("pnode_id").mean()
    sink_da_df = sink_da_df.groupby("pnode_id").mean()
    non_mtm_trades["mtm_1st_mean"] = (
        non_mtm_trades["sink_id"].map(sink_da_df) -
        non_mtm_trades["source_id"].map(source_da_df)
    )
    non_mtm_trades["mtm_2nd_mean"] = non_mtm_trades["mtm_1st_mean"]
    non_mtm_trades["mtm_3rd_mean"] = non_mtm_trades["mtm_1st_mean"]
    non_mtm_trades["mtm_1st"] = non_mtm_trades["mtm_1st_mean"] * 3     # quarterly scaling
    non_mtm_trades["mtm_1st_period"] = non_mtm_trades["mtm_1st"]
    return pd.concat([mtm_trades, non_mtm_trades])
```

### Algorithm Walkthrough (Example: aq1, PY 2026)

1. `month_st` = 2026-06 + 0*3 = 2026-06 (start of aq1: Jun)
2. Quarter months: Jun-2026, Jul-2026, Aug-2026
3. `range(1, 2)` → look back exactly 1 year: Jun-2025, Jul-2025, Aug-2025
4. `cutoff_month` = 2026-04
5. Filter: keep months before April 2026 → all 3 months pass (Jun/Jul/Aug 2025 < Apr 2026)
6. Load DA LMP monthly aggregated congestion for those 3 months
7. Apply 0.85 shrinkage (see Section 8)
8. Group by (planning_year, pnode_id) → mean → group by pnode_id → mean across years
9. `mtm_1st_mean = sink_congestion - source_congestion`
10. `mtm_1st = mtm_1st_mean * 3` (quarterly scaling: 3 months in a quarter)

### Edge Cases

- **aq3 (Dec–Feb):** `month_st` = 2026-12. Look back 1 year: Dec-2025, Jan-2026, Feb-2026.
  Cutoff = 2026-04. All pass. But `da["planning_year"]` calculation assigns Dec-2025 to PY2025,
  Jan/Feb-2026 also to PY2025 (month < 6). So grouping works.
- **aq4 (Mar–May):** `month_st` = 2027-03. Look back: Mar-2026, Apr-2026, May-2026.
  Cutoff = 2026-04. Only Mar-2026 passes (< Apr-2026). **aq4 uses only 1 month of data.**
  - **[OPEN]** Is this intentional? Seems like a data coverage weakness for aq4.

---

## 7. `fill_mtm_1st_period_with_hist_revenue` — The Parallel Implementation

**Location:** `pbase/src/pbase/analysis/tools/miso.py:322-366`

Nearly identical to `_fill_mtm` but with key differences:

| Aspect | `_fill_mtm` (pmodel) | `fill_mtm_1st_period_with_hist_revenue` (pbase) |
|--------|------|------|
| **Years of history** | 1 year (hardcoded: `range(1, 2)`) | Configurable `n_years` param (default 1) |
| **Logging** | `self.logger.info()` | `print()` |
| **Called from** | `trade_finalizer.py:75` | `autotuning.py:349-362` (autotuning path) |
| **Base class** | `MisoFtrModel._fill_mtm()` | `MisoApTools.fill_mtm_1st_period_with_hist_revenue()` |
| **0.85 shrinkage** | Yes, hardcoded | Yes, hardcoded |

**Base class definition** (`base.py:880-881`):
```python
def _fill_mtm(self, trades: pd.DataFrame) -> pd.DataFrame:
    return trades  # No-op default
```

**PJM:** Inherits no-op. PJM has NO `_fill_mtm` override. Relies entirely on `get_m2m_mcp_for_trades_all()`.

---

## 8. The 0.85 Shrinkage Factor — Forensic Analysis

### Where It Appears (4 files)

| File | Lines | Source Shrinkage | Sink Shrinkage |
|------|-------|-----------------|----------------|
| `pmodel/base/ftr24/v1/miso_base.py` | 248-249 | `< 0` → `*= 0.85` | `> 0` → `*= 0.85` |
| `pbase/analysis/tools/miso.py` | 353-354 | `< 0` → `*= 0.85` | `> 0` → `*= 0.85` |
| `pbase/analysis/tools/spp.py` | 358 | **COMMENTED OUT** | `> 0` → `*= 0.85` |
| `pbase/analysis/tools/ercot.py` | 536-537 | **COMMENTED OUT** | **COMMENTED OUT** |

### What It Does

Applies a **15% haircut** to historical congestion prices in one direction:
- **Source nodes:** If congestion < 0 (paying), shrink by 15% → less negative → conservative
- **Sink nodes:** If congestion > 0 (earning), shrink by 15% → less positive → conservative

Net effect: Reduces the predicted path value (sink - source) conservatively.

### Configurability

**None.** The 0.85 factor is:
- NOT a function parameter
- NOT a class variable
- NOT in any config file
- NOT documented in any comment
- NOT justified in any commit message visible from the code

The 0.85 also appears in optimizer files:
- `optimizer/path_selection.py:141` — "Split MTM into negative and positive components with shrinkage"
- `optimizer/volume_assignment.py:170` — same comment
- `optimizer/bid_points.py:423` — same comment

**[OPEN]** Origin unknown. Likely empirically determined. Should be parameterized for Stage 2.

---

## 9. Band Generator — DOES NOT Support Annual

**[CORRECTION]** This is a critical finding. The first version of this document implied
the band generator handles annual. It does not.

### Evidence 1: BASELINE_CONFIG missing annual period types

**`band_generator.py:121-129`:**
```python
BASELINE_CONFIG = {
    "f0": {"mtm_weight": 0.77, "rev_weight": 0.23},
    "f1": {"mtm_weight": 0.85, "rev_weight": 0.15},
    "f2": {"mtm_weight": 0.94, "rev_weight": 0.06},
    "f3": {"mtm_weight": 0.93, "rev_weight": 0.07},
    "q2": {"mtm_weight": 0.92, "rev_weight": 0.08},
    "q3": {"mtm_weight": 0.93, "rev_weight": 0.07},
    "q4": {"mtm_weight": 0.91, "rev_weight": 0.09},
    # aq1, aq2, aq3, aq4 ARE NOT DEFINED
}
```

Calling `compute_baseline()` with `ptype="aq1"` hits `BASELINE_CONFIG[ptype]` at line 400
→ **KeyError: 'aq1'**

### Evidence 2: Method selection excludes annual

**`band_generator.py:1189-1213`:**
```python
f0_ptypes = ["f0"]
f1_ptypes = ["f1"]
f2p_ptypes = ["f2", "f3", "q2", "q3", "q4"]

if period_type in f0_ptypes + f1_ptypes:
    # LightGBM + conformal
elif period_type in f2p_ptypes:
    # Rule-based binning
else:
    raise ValueError(f"Unknown period_type: {period_type}")
    # ← aq1-aq4 hit this branch
```

### Evidence 3: Quarterly scaling excludes annual

**`band_generator.py:274`:**
```python
QUARTERLY_PTYPES = ["q2", "q3", "q4"]  # aq1-aq4 NOT included
```

### Evidence 4: No training data exists

Training data at `/opt/temp/qianli/mcp_pred_training/` only contains `f0` and `f2` period types.

### Implication

The FTR24/v1 pipeline calls `generate_bands()` unconditionally at `base.py:467`. If the
pipeline runs with annual period types, it will crash. This confirms that PY 2025 annual
ran on FTR23 (which does NOT use `generate_bands`), and that FTR24/v1 annual support
requires adding annual period type handling to the band generator.

---

## 10. `_set_bid_curve` — Legacy, Not Active

**[CORRECTION]** The first version stated `_set_bid_curve` handles bid curve construction.
In reality, it is **legacy code** not used in the current production pipeline.

### MISO Implementation (`miso_base.py:112-160`)

- Takes `mcp_pred_*` columns and distributes them into multi-point bid curves
- Scale: 3× for quarterly periods (aq1-aq4 contain "q")
- Volume distributed via power function across price points

### PJM Implementation (`pjm_base.py:67-112`)

- Scale: 12× for annual periods (`a`, `yr1`, `yr2`, `yr3`)
- Uses diffed volume distribution (cumulative increments)

### Why It's Legacy

- `base.py:467` calls `generate_bands()` which produces `bid_price_1..10` and `clearing_prob_1..10`
- `_set_bid_curve` is only called if there are pre-existing `mcp_pred_*` columns (which
  `generate_bands()` does not produce — it produces band columns instead)
- The f0p config (miso_f0p_offpeak.py) does NOT call `_set_bid_curve`
- **No evidence of `_set_bid_curve` being invoked in any production notebook or pipeline**

---

## 11. Bid Price Post-Processing: What "simple_v3" Actually Means

**`miso_a_offpeak.py:291`:**
```python
BID_PRICE_POST_PROCESSING = "simple_v3"
```

**[CORRECTION]** "simple_v3" is NOT a separate algorithm. It is a config label that
currently has no dispatch mechanism consuming it. The actual bid price generation
happens through `generate_bands()` which is called unconditionally.

### Historical alternatives (from `params/dev/sc/miso_models_f0p_old.py:688-693`):

```python
# "post_processing": "clear_all",     # Not implemented
# "post_processing": "simple",         # Not implemented
# "post_processing": "simple_v2",      # Not implemented
"post_processing": "simple_v3",        # Active (routes to generate_bands)
# "post_processing": "band_20_60",     # Not implemented
```

None of the alternatives are implemented in the current codebase. The `post_processing`
key is stored in params but not consumed by any dispatch function.

---

## 12. Optimizer — Path Selection, Bid Points, Volume Assignment

### Three-Stage Optimization

**Stage 1: Path Selection** (`optimizer/path_selection.py`)
- Selects which paths to bid on
- Gurobi LP/QP solver
- Objective: maximize expected revenue - cost
- Subject to: exposure limits, volume bounds, LSR constraints

**Stage 2: Bid Point Selection** (`optimizer/bid_points.py`)
- For each selected path, choose which points on the bid curve to submit
- Decomposes bid curves into individual points via `decompose_bid_offer_curve()`
- Creates `mcp_pred_1..10` columns from the `mcp_pred` column
- Clearing probability handling (`bid_points.py:423`):
  ```python
  clear_prob_arr = trades[clear_prob_col].to_numpy() / 100
  clear_prob_arr[sell_arr] = 1 - clear_prob_arr[sell_arr]
  exposure *= clear_prob_arr[:, np.newaxis]
  estimated_mcp = trades[estimated_mcp_col].copy() * clear_prob_arr
  ```

**Stage 3: Volume Assignment** (`optimizer/volume_assignment.py`)
- Distributes MW across selected bid points
- Minimum 0.1 MW per point
- L2 regularization on bid price magnitude

### Key Parameters (from `miso_a_offpeak.py`)

- **Total volume:** 8000 MW (`800 * MULTIPLIER=10`)
- **Per path:** 10 MW (up to 20 for bid points)
- **Min bid points per path:** 5
- **Solver:** Gurobi, method=2, time limit=180s
- **LSR lower bound:** tier0=1.0, tier1=1.0
- **Negative exposure bounds:** tier10=-8 to tier23=-55 MW

---

## 13. Clearing Probabilities — V6 Empirical

**Location:** `band_generator.py:288-386`

### How Empirical Clearing Rates Are Calculated

```python
For BUY trades:   clearing_prob = % of historical rows where mcp_mean < band_value
For SELL trades:  clearing_prob = % of historical rows where mcp_mean > band_value
```

Segmented by flow_type:
- `"prevail"` if `mtm_1st_mean > 0`
- `"counter"` if `mtm_1st_mean <= 0`

Result structure:
```python
{
    'buy_prevail':  {'upper_99': 98.2, 'upper_50': 72.1, ...},
    'buy_counter':  {'upper_99': 99.1, 'upper_50': 81.3, ...},
    'sell_prevail': {'upper_99': 1.8,  'upper_50': 27.9, ...},
    'sell_counter': {'upper_99': 0.9,  'upper_50': 18.7, ...},
}
```

### Hybrid Logic

Uses empirical for most bands, rule-based fallback for bands where rule outperforms:
```python
RULE_WINS_BANDS = frozenset({"lower_95", "upper_50", "upper_70"})
```

### Rule-Based Fallback

```python
CLEARING_PROBS_BUY_FALLBACK = {
    "upper_95": 97.5,
    "upper_50": 75.0,
    "lower_50": 25.0,
    ...
}
```

Minimum rows for empirical: `MIN_ROWS_FOR_EMPIRICAL = 100`

**[OPEN]** Annual has far less data than monthly. Empirical rates may be unreliable
even if the code supported annual period types.

---

## 14. Annual-Specific Configuration (Production Params)

**File:** `params/prod/auc2603/miso_a_offpeak.py`

```python
MODEL_TYPE = "a"
period_types = ["aq1", "aq2", "aq3", "aq4"]
auction_rounds = [1, 2, 3]
MULTIPLIER = 10  # volume scaling

# Signals — SAME f0p signals for ALL periods (no short/long-term split)
ANNUAL_SIGNALS = [
    "TEST.Signal.MISO.DZ_F0P_V5.1.R1",
    "TEST.TEST.Signal.MISO.SPICE_F0P_V6.2C.R1",
]
# ALL 4 periods map to ANNUAL_SIGNALS:
# dict_set_signal_name_path_reduce = {p: ANNUAL_SIGNALS for p in ["aq1","aq2","aq3","aq4"]}

# No ML prediction
prediction_class_instance = None
BID_PRICE_POST_PROCESSING = "simple_v3"

# Exposure/volume
total_volume_lb = 800 * MULTIPLIER  # 8000 MW
volume_per_path_ub = MULTIPLIER     # 10 MW
```

### Autotuning Parameters Per (period, round)

```python
cpr = 0.8, cpr_offset = 0.2
lr = (2/3, 1)
prevail_cost_path_ub = 600
counter_cost_path_ub = 400
```

---

## 15. Signal Architecture for Annual

### FTR24/v1 Annual Signal Flow

**All 4 periods use the SAME signal list** (verified at `miso_a_offpeak.py:329-332`):
```python
all_periods = short_term_periods + long_term_periods  # [] + ["aq1","aq2","aq3","aq4"]
dict_set_signal_name_path_reduce = {p: ANNUAL_SIGNALS for p in all_periods}
dict_set_signal_name_cluster_matrix = {p: ANNUAL_SIGNALS for p in all_periods}
dict_set_signal_name_objective = {p: ANNUAL_SIGNALS for p in all_periods}
dict_set_signal_name_optimization_constraints = {p: ANNUAL_SIGNALS for p in all_periods}
```

**Key difference from f0p:** f0p uses short-term signals for f0/f1 and long-term signals
for f2+. Annual uses a single signal set for everything.

### FTR23 Annual Signals (Legacy — Actually Used in PY 2025)

FTR23 had **annual-specific** signals:
```python
short_term_signal = [
    "TEST.Signal.MISO.SPICE_ANNUAL_V4.5.R{auction_round}",
    "TEST.Signal.MISO.DA_ANNUAL_V1.4.R{auction_round}",
    "TEST.Signal.MISO.DZ_ANNUAL_V11.R{auction_round}",
    "TEST.Signal.MISO.TMP_VIOLATED_{tag}.R{auction_round}",
]
```

### Evolution

| Era | Signals |
|-----|---------|
| FTR23 PY2025 | SPICE_ANNUAL, DA_ANNUAL, DZ_ANNUAL (annual-specific) |
| FTR24 PY2026 | DZ_F0P, SPICE_F0P (f0p signals reused for annual) |

**[OPEN]** Is using f0p signals for annual a regression or a deliberate simplification?

### CIA Calibration

Annual config includes CIA-specific tiers (10-14) and DZ tiers (20-24):
```python
"cia": {
    "participants": ["DCTN", "SEM1", "7532"],
    "signal_config": {
        "TEST.Signal.MISO.AUC_SPICE_BEFORE_CLEAR_V1.1CIA1.R1": {
            "unit_expo_th": -999,
            "white_list_tiers": [10, 11, 12, 13, 14],
        },
    },
}
```

---

## 16. PJM Annual — Detailed Analysis

### MTM Handling

**PJM has NO special R1 MTM handling.** All rounds, including R1, use:
```python
trades = model._aptools.tools.get_m2m_mcp_for_trades_all(trades_all=trades)
```

PJM inherits the base `_fill_mtm()` which is a no-op (`base.py:880-881`).
This means PJM annual R1 has real MTM data, unlike MISO.

### Bid Price Scaling

**`pjm_base.py:70-73`:**
```python
if period_type in ["a", "yr1", "yr2", "yr3"]:
    scale = 12  # Annual = 12 months
```

### Class Types

PJM annual has 3 class types (vs MISO's 2):
- `onpeak`
- `dailyoffpeak`
- `wkndonpeak`

### PJM Annual Signals

**DA signal:** `TEST.Signal.PJM.DA_ANNUAL_V1.6.R1`
- 2-year lookback
- 5 tiers of ~50 constraints each
- Generated in: `/home/xyz/workspace/pmodel/notebook/hz/2025-planning-year/2025-26-annual/pjm/submission/r1/da_signal/annual_da_signal_cont.ipynb`

### PJM R1 vs R2

| Aspect | R1 | R2 |
|--------|----|----|
| Module | `pjm_models_a_prod_r1` | `pjm_models_a_prod_r2` |
| `use_prod_bids` | No | Yes (incorporates R1 cleared trades) |
| Output (OnPeak) | 417 trades, 6100 MW | 666 trades, 6601 MW buy + 1669 MW sell |

### `ftr5v15`

**[CORRECTION]** First version mentioned `ftr5v15`. NOT FOUND in the codebase.
Likely refers to an external or historical system, not relevant to current FTR24/v1.

---

## 17. SPP Annual

- Period types: `af0`, `af1`, `af2`, `af3`, `aq2`, `aq3`, `aq4`
- NO dedicated `SppFtrModel` in FTR24/v1
- SPP `fill_mtm` in pbase (`spp.py:326-379`) has source shrinkage **commented out**
- ERCOT has both source and sink shrinkage commented out (`ercot.py:536-537`)

---

## 18. Submission Notebooks — Actual Production Workflow

### Directory Structure

```
/home/xyz/workspace/pmodel/notebook/hz/2025-planning-year/2025-26-annual/
├── miso/
│   ├── submission/r{1,2,3}/trades/generate_trades_{on,off}.ipynb
│   ├── submission/r{1,2,3}/da_signal/annual_da_signal.ipynb
│   ├── submission/r{1,2,3}/spice/spice-signal-annual.ipynb
│   ├── generate_trades_{on,off}.ipynb
│   └── mtm_value_trend_annual.ipynb
├── pjm/
│   ├── submission/r{1,2,3,4}/trades/model_{onpeak,dailyoffpeak,wkndonpeak}_a_auto.ipynb
│   └── submission/r{1,2,3,4}/da_signal/annual_da_signal_cont.ipynb
```

### Three-Round Iterative Workflow (MISO)

**R1 (Early April):**
- Generate base portfolio using full model
- Apply constraint signals (DA + SPICE)
- Validate with `aptools.tools.just_simulate(trades)`
- ~2,070 trades

**R2 (Mid-April):**
- Load R1 existing bids
- Generate complementary portfolio
- `use_prod_bids=True` for PJM
- ~2,530 trades

**R3 (May 1st cutoff):**
- Load R1+R2 existing bids from parquet:
  ```python
  ep = pd.concat([
      pd.read_parquet('/opt/data/tmp/tmp/2025_annual/miso/offpeak_wrong_r12_r1.parquet'),
      pd.read_parquet('/opt/data/tmp/tmp/2025_annual/miso/offpeak_wrong_r12_r2.parquet'),
  ])
  ```
- Generate final adjustments
- **Manual path filtering:**
  ```python
  paths = {
      'aq4': ['NIPS.MNTIC2.NU,NIPS.ROSEWRWF', ...],  # MAGNETON-RENOLD4
      'aq3': ['CIN.CAYUGA.3,CIN.CAYUGA.2', ...],       # CAYUGA 9P 1
  }
  # Removed 6 buy trades (4 from aq4, 2 from aq3)
  ```
- **Manual bid ladder adjustments:**
  ```python
  buy_sell_mask = np.where(trades['trade_type'] == 'buy', -1, 1)
  trades['bid_price_4'] += 0.03 * buy_sell_mask
  trades['bid_price_3'] += 0.02 * buy_sell_mask
  trades['bid_price_2'] += 0.01 * buy_sell_mask
  ```
- ~3,534 trades after filtering

### Strategy Naming

```python
trades['strategy_name'] = f'{rto.upper()}.FTR23_V1.2.A.{class_type}.R{auction_round}'
# Example: 'MISO.FTR23_V1.2.A.OFFPEAK.R3'
```

### Trade Counts (PY 2025 R3)

- MISO on-peak: 3,540 → 3,534 after filtering
- MISO off-peak: 3,310 → 3,299 after filtering
- Per period: ~800-900 per aq1-aq4

---

## 19. FTR23 Legacy — What Actually Ran for PY 2025

### FTR23 Bid Pricing: Power Function

**Location:** `ftr23/v2/base.py:99-198`

```python
rank = MinMaxScaler(0, 1).fit_transform(exposure_tier_rank)
offset = MinMaxScaler(offset_lb, offset_ub).fit_transform(power_func(rank))
change_pct = MinMaxScaler(change_lb, change_ub).fit_transform(power_func(rank))
price_change = |mcp_pred_modify| * change_pct + offset
predicted_bid_1 = mcp_pred_modify + price_change
```

### Actual Parameter Values (PY 2025 R3, from `miso_models_a_prod_r3.py`)

**Buy Counter-Flow:**

| Param | Value | Meaning |
|-------|-------|---------|
| offset_lb | 30 | Min $ offset above mcp_pred |
| offset_ub | 70 | Max $ offset above mcp_pred |
| change_lb | 0.3 | Min 30% of \|mcp_pred\| added |
| change_ub | 0.7 | Max 70% of \|mcp_pred\| added |
| power | 2 | Quadratic scaling curve |

**Buy Prevailing-Flow:**

| Param | Value | Meaning |
|-------|-------|---------|
| offset_lb | -50 | Can bid $50 below mcp_pred |
| offset_ub | 30 | Up to $30 above mcp_pred |
| change_lb | -0.5 | Can reduce by 50% |
| change_ub | 0.3 | Up to 30% increase |
| power | 2 | Quadratic |

### Volume Per Round (MISO Annual)

| Round | Volume (MW) | CPR (aq1-aq2) | CPR (aq3-aq4) |
|-------|-------------|---------------|---------------|
| R1 | 8100 | 0.9 | 0.9 |
| R2 | 8600 | 0.9 | 0.9 |
| R3 | 9100 | 1.1 | 2.0 |

---

## 20. Historical Performance

### MISO On-Peak Annual (from submission notebooks)

| PY | Profit ($M) | Credit Req ($M) | Return | Score |
|----|------------|----------------|--------|-------|
| 2019 | 2.93 | 16.52 | 17.7% | 0.81 |
| 2020 | 7.50 | 16.69 | 44.9% | 1.69 |
| 2021 | 12.46 | 24.73 | 50.4% | 2.38 |
| 2022 | 13.21 | 29.08 | 45.4% | 2.22 |
| 2023 | 4.15 | 26.09 | 15.9% | 0.49 |
| 2024 | 4.13 | 22.53 | 18.3% | 0.33 |

### Clearing Rate (MISO R2 On-Peak)

- Bid volume: 35,703 MW → Cleared: 18,567 MW → **52% clear rate**
- Buying: 55-66% clear rate
- Selling: 32-37% clear rate

---

## 21. Data Access — APTools API

```python
from pbase.analysis.tools.all_positions import MisoApTools, PjmApTools

miso_aptools = MisoApTools()
pjm_aptools = PjmApTools()

# Cleared trades
trades = miso_aptools.get_all_cleared_trades(start_date="2026-05-01", end_date="2026-05-01")
trades = pjm_aptools.get_all_cleared_trades(start_date="2028-06-01", end_date="2028-06-01")

# Original trades (MISO only)
trades = miso_aptools.get_ori_trades_of_market_month(participant="AMGA", market_month="2025-11")
trades = miso_aptools.get_ori_trades_of_planning_year(participant="DCMW", planning_year=2025)
# Filter: trades[trades["period_type"].isin(["aq1", "aq2", "aq3", "aq4"])]

# MTM data
trades = aptools.tools.get_m2m_mcp_for_trades_all(trades_all=trades)

# Fill MTM for annual R1
trades = aptools.tools.fill_mtm_1st_period_with_hist_revenue(trades)

# Simulation
res = aptools.tools.just_simulate(trades)
credit = aptools.tools.calculate_credit_requirement(trades)
```

### Participant Tickers
- `AMGA`, `AMAZ` — trading tickers
- `DCMW` — also appears for original trades

---

## 22. Prior Research Applicable to Annual

### research-mcp-coverage (HIGH relevance)

**Path:** `/home/xyz/workspace/research-qianli/research-mcp-coverage/`

Two-step MCP prediction with uncertainty quantification:
1. **Baseline formula:** `MCP = 0.8 * mtm_1st_mean + 0.2 * 1(rev)` (found better weights)
2. **GBDT on residuals** → conformal calibration for coverage bands

Key finding: **Simple baseline outperforms ML models** (MAE 146.22 vs Ridge/LightGBM).

Revenue trend features (rev-ndays): lookback windows [1, 2, 3, 5, 7, 10, 14, 30] days.
Trend features: `diff_1_3 = rev-1days - rev-3days`, etc.

### research-bid-price-v5 (HIGH relevance)

**Path:** `/home/xyz/workspace/research-qianli/research-bid-price-v5/`

**MISO findings:**
- **2-Regime Ridge Model:** 24.7% MAE improvement over mtm1 baseline
- Regime split at |mtm| = 1500
- Low regime: `14.72 + 0.765*mtm1 + 0.030*mtm2 + 0.161*rev1 + 0.036*rev2`
- High regime: `32.53 + 0.832*mtm1 - 0.071*mtm2 + 0.203*rev1 + 0.029*rev2`

**PJM findings:**
- 2-tier baseline (4-source): 83.82 MAE vs 89.18 production (-6%)
- Low tier: 60% mtm + 15% rev + 15% mtm2 + 10% rev2
- High tier: 20% mtm + 40% rev + 20% mtm2 + 20% rev2

**Conformal banding:** One-piece conformal with P95 cap recommended.

### research-ML-bid-price (MEDIUM-HIGH relevance)

**Path:** `/home/xyz/workspace/research-qianli/research-ML-bid-price/`

**Spread features** (`mtm - revenue`) are highly informative:
- 49 spread features, 17 in top 50 importance
- Spread momentum, acceleration, z-scores
- F1: 0.7183, AUC: 0.7730

### research-params-search (MEDIUM relevance)

**Path:** `/home/xyz/workspace/research-qianli/research-params-search/`

Parameter tuning for bid price power functions — useful for FTR23-style pricing.

### research-backtest-augmentation (MEDIUM relevance)

Backtest engine v1-v5 with PnL decomposition. Useful for validating predictions.

### research-framework/competitor-analysis (LOW-MEDIUM relevance)

T01-T13 findings: constraint exposure, pricing strategies, signal disagreement trading.

---

## 23. Answering the Memo's Open Questions

### Q: "since in MISO we do not have mtms for R1, we rely more on 同比/环比 revenue?"

**Answer: Partially correct.** `_fill_mtm` uses **year-over-year (同比)** DA congestion
prices from the same quarter months of the previous PY. NOT month-over-month (环比).
The shrinkage factor is 0.85 (hardcoded, undocumented).

### Q: "we do some kind of percentile quantification to decide our bid price?"

**Answer: In FTR23, yes.** FTR23 used exposure tier ranking with percentile-based
power function curves. In FTR24, the band generator uses empirical percentiles for
clearing probabilities — but it **cannot run for annual period types**.

### Q: "what about R2? what about PJM?"

**R2-R3:** `mtm_1st_mean` from previous round's MCP via `get_m2m_mcp_for_trades_all()`.
**PJM:** Has MTMs for ALL rounds including R1. No special fill needed. Uses 12x bid
price scaling. 3 class types. 4 auction rounds.

### Q: "what pricing strategy did we actually use?"

**For PY 2025 (actual production):** FTR23 v2 with power function pricing.
**For PY 2026 (FTR24 config):** `prediction_class_instance = None`, `mcp_pred = mtm_1st_mean`,
band generator (would need annual support added).

### Q: "for annual, how to PREDICT MCP? what would be the formula?"

**Current formula (R1):**
```
mcp_pred = (sink_DA_congestion_prev_year - source_DA_congestion_prev_year) × 0.85_shrinkage
```

**This is the main target for improvement in Stage 2.** Promising directions from prior research:
1. Multi-source baseline: mtm + rev + mtm2 + rev2 (from bid-price-v5)
2. 2-regime approach at |mtm| threshold (24.7% improvement proven on monthly)
3. Spread features from historical mtm-revenue gaps
4. Conformal banding for uncertainty quantification

---

## 24. Key Files Reference

### Core Pipeline (FTR24/v1)

| File | Lines | Purpose |
|------|-------|---------|
| `base.py` | 928 | Abstract base model, pipeline orchestration |
| `miso_base.py` | 261 | MISO: `_fill_mtm` (221-261), `_set_bid_curve` (112-160, LEGACY) |
| `pjm_base.py` | 179 | PJM: 12x scaling, class types, NO `_fill_mtm` |
| `autotuning.py` | 1445 | Production entry point, param management |
| `trade_finalizer.py` | 484 | MTM fill (66-76), MCP pred (85-94), bid adjust (99-150) |
| `band_generator.py` | 1709 | MCP bands + clearing probs (**NO annual support**) |
| `trade_generator.py` | 446 | Candidate path generation |
| `trade_filter.py` | 221 | Exposure-based filtering |
| `constraint_loader.py` | 520 | Signal loading, CIA calibration |

### Optimizer

| File | Purpose |
|------|---------|
| `optimizer/path_selection.py` | Path selection + 0.85 shrinkage comment (line 141) |
| `optimizer/bid_points.py` | Bid point selection, clearing prob handling (line 423) |
| `optimizer/volume_assignment.py` | Volume distribution + 0.85 shrinkage comment (line 170) |
| `optimizer/solver.py` | Gurobi solver wrapper |

### Configuration

| File | Purpose |
|------|---------|
| `params/schema.py` | Parameter dataclass definitions |
| `params/prod/auc2603/miso_a_offpeak.py` | MISO annual offpeak production config |
| `params/prod/auc2603/miso_f0p_offpeak.py` | f0p reference config |

### APTools (pbase)

| File | Lines | Purpose |
|------|-------|---------|
| `pbase/analysis/tools/miso.py` | 322-366 | `fill_mtm_1st_period_with_hist_revenue` |
| `pbase/analysis/tools/spp.py` | 326-379 | SPP version (source shrinkage commented out) |
| `pbase/analysis/tools/base.py` | 17296-17360 | `get_m2m_mcp_for_trades_all` |

### Research

| Directory | Relevance | Key Finding |
|-----------|-----------|-------------|
| `research-mcp-coverage/` | HIGH | Baseline 0.8*mtm + 0.2*rev, 2-step methodology |
| `research-bid-price-v5/` | HIGH | 2-regime Ridge (-24.7% MAE), conformal bands |
| `research-ML-bid-price/` | MEDIUM-HIGH | Spread features (49), F1=0.7183 |
| `research-f0p-band/` | HIGH | Band generation V5-V7, clearing prob calibration |
| `research-params-search/` | MEDIUM | Power function param tuning |

---

## 25. Critical Gaps and Open Items for Stage 2

### Design Gaps (Must Address)

1. **Band generator has no annual support.** `BASELINE_CONFIG` is missing aq1-aq4.
   `generate_bands_for_group()` raises ValueError for annual. No training data exists.
   → Must either extend band_generator or use FTR23-style pricing for annual.

2. **0.85 shrinkage is an undocumented magic number.** No parameterization, no justification.
   → Should be calibrated empirically or replaced with a principled approach.

3. **`_fill_mtm` uses only 1 year of history.** `range(1, 2)` is hardcoded. The pbase
   version supports `n_years` parameter but pmodel does not.
   → Should evaluate multi-year lookback (the pbase version already supports it).

4. **aq4 gets only 1 month of data** due to cutoff_month logic (Mar-2026 < Apr-2026 passes,
   but Apr-2026 and May-2026 do not).
   → Evaluate if this data sparsity hurts aq4 prediction quality.

### Research Questions

5. **MCP prediction formula for R1:** Current formula is crude. Proven improvements from
   prior research:
   - Multi-source weighted baseline (mtm + rev + mtm2 + rev2)
   - 2-regime approach by |mtm| threshold
   - Spread features (mtm - revenue gap)
   - Revenue trend features (rev-ndays)

6. **R1 vs R2-R3 strategy:** Should we use fundamentally different approaches?
   R2-R3 have real MTM from prior round clearing.

7. **Signal selection:** FTR24 uses f0p signals (DZ_F0P, SPICE_F0P) for annual.
   FTR23 had annual-specific signals (SPICE_ANNUAL, DA_ANNUAL, DZ_ANNUAL).
   Which performs better?

8. **Clearing probability calibration for annual:** Much less data than monthly.
   Empirical rates may be unreliable.

9. **PJM annual strategy:** PJM has real R1 MTM. Should PJM annual use a different
   approach than MISO annual?

### Data to Gather for Stage 2

- Historical cleared annual trades across PY 2019-2025 (via ApTools)
- Predicted vs actual MCP comparison for each round
- Revenue analysis per planning year and quarter
- The seasonal revenue patterns relevant to aq1-aq4 prediction
- Competitor clearing patterns for annual auctions
