# F1 Period Type Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run the full f1 experiment ladder — v0 (V6.2B formula), v1 (optimized blend), v2 (full ML with binding_freq) — with correct temporal leakage guards, producing results in `registry/f1/onpeak/`.

**Architecture:** Fix 5 issues in the shared ML modules (GT join, spice6 market_month, binding_freq cutoff separation, training window backfill for missing months, auction schedule filtering), add `--ptype` CLI flag to both run scripts, then run a 3-version f1 experiment ladder. 16 tasks total.

**Tech Stack:** Python, Polars, LightGBM, existing ml/ modules

---

## Background: f1 Timing Model

For f1, auction_month A has delivery_month = A + 1. The auction for month A happens ~mid(A-1).

**Concrete example:** f1 target row (auction=2025-02, delivery=2025-03)
- Decision happens ~mid 2025-01
- Last fully known realized DA month = **2024-12**
- Last safe training row: (auction=2024-11, f1) → delivery=2024-12 (complete)
- NOT safe: (auction=2024-12, f1) → delivery=2025-01 (incomplete at decision time)

**Key rules:**
- **Row inclusion**: include training row (A, fN) only if A+N ≤ last_full_known (= eval_auction_month - 2)
- **Training window**: collect **8 usable rows** walking backward, not 8 calendar months. A row is "usable" if (a) fN exists for that auction month per auction schedule, AND (b) its delivery_month ≤ last_full_known. For f0 this is invisible (f0 exists every month), but for f1 May/Jun are missing — a fixed 8-calendar-month window would yield only 6 rows.
- **Minimum history**: skip eval months where fewer than 6 usable training rows exist. For f1, the earliest months (2020-07, 2020-08) only have 4-5 usable prior rows since f1 data starts 2020-01. These are excluded from dev eval. Expected: **~28 dev months** (not 30).
- **Feature cutoff** (binding_freq on ANY row): use realized months ≤ last_full_known. For compute_bf's strict-`<` semantics, pass cutoff = last_full_known + 1 month (= `auction_month - 1` for the target row)
- **Ground truth**: realized DA for **delivery_month** (A+N), not auction_month
- **Spice6**: load with **market_month = delivery_month** (A+N), not auction_month

**Data availability:**
- f1 V6.2B data: 34 months on disk (2020-01 to 2023-04), **30 within eval window** (`_FULL_EVAL_MONTHS` starts at 2020-06; removes 6 May/Jun months). For ML (v2), **~28 months** after min-history skip (early months lack 6+ usable training rows). v0/v1 formula evaluation uses all 30.
- f1 holdout: 20 months (2024-01 to 2025-12, missing May/Jun)
- Spice6 density: exists for f1 market_months (verified auction_month+1 paths)
- Realized DA cache: 2019-06 to 2025-12 (covers all delivery months)

---

## Task 1: Add `delivery_month` helper to config.py

**Files:**
- Modify: `ml/config.py`
- Modify: `ml/tests/test_config.py` (or create if doesn't exist)

**Why:** Multiple modules need to compute delivery_month = auction_month + N. Centralize this.

**Step 1: Add the auction schedule and helpers to config.py**

Add after the `REALIZED_DA_CACHE` line (~line 118):

```python
# ── MISO Auction Schedule ──
# Determines which period types exist for each calendar month.
# Source: iso_configs.py. May/Jun = f0 only; Jul/Aug/Nov = most types.
MISO_AUCTION_SCHEDULE: dict[int, list[str]] = {
    1: ["f0", "f1", "q4"],
    2: ["f0", "f1", "f2", "f3"],
    3: ["f0", "f1", "f2"],
    4: ["f0", "f1"],
    5: ["f0"],
    6: ["f0"],
    7: ["f0", "f1", "q2", "q3", "q4"],
    8: ["f0", "f1", "f2", "f3"],
    9: ["f0", "f1", "f2"],
    10: ["f0", "f1", "q3", "q4"],
    11: ["f0", "f1", "f2", "f3"],
    12: ["f0", "f1", "f2"],
}


def period_offset(period_type: str) -> int:
    """f0→0, f1→1, f2→2, f3→3. Only valid for monthly types (f0-f3)."""
    if not period_type.startswith("f"):
        raise ValueError(f"period_offset only supports monthly types (f0-f3), got '{period_type}'")
    return int(period_type[1:])


def delivery_month(auction_month: str, period_type: str) -> str:
    """Compute delivery month from auction month and period type."""
    import pandas as pd
    offset = period_offset(period_type)
    if offset == 0:
        return auction_month
    dt = pd.Timestamp(auction_month) + pd.DateOffset(months=offset)
    return dt.strftime("%Y-%m")


def has_period_type(auction_month: str, period_type: str) -> bool:
    """Check if period type exists for a given auction month."""
    import pandas as pd
    month_num = pd.Timestamp(auction_month).month
    return period_type in MISO_AUCTION_SCHEDULE.get(month_num, ["f0"])


def f1_eval_months(full: bool = False) -> list[str]:
    """Return eval months that have f1 data, from _FULL_EVAL_MONTHS or _DEFAULT_EVAL_MONTHS."""
    source = _FULL_EVAL_MONTHS if full else _DEFAULT_EVAL_MONTHS
    return [m for m in source if has_period_type(m, "f1")]
```

**Step 2: Write tests**

```python
# ml/tests/test_config_helpers.py
from ml.config import period_offset, delivery_month, has_period_type, f1_eval_months

def test_period_offset():
    assert period_offset("f0") == 0
    assert period_offset("f1") == 1
    assert period_offset("f3") == 3

def test_period_offset_rejects_quarterly():
    import pytest
    with pytest.raises(ValueError, match="monthly types"):
        period_offset("q4")

def test_delivery_month_f0():
    assert delivery_month("2022-09", "f0") == "2022-09"

def test_delivery_month_f1():
    assert delivery_month("2022-09", "f1") == "2022-10"

def test_delivery_month_f2_year_wrap():
    assert delivery_month("2022-11", "f2") == "2023-01"

def test_has_period_type_f1_in_july():
    assert has_period_type("2022-07", "f1") is True

def test_has_period_type_f1_not_in_may():
    assert has_period_type("2022-05", "f1") is False

def test_f1_eval_months_excludes_may_june():
    months = f1_eval_months(full=True)
    for m in months:
        month_num = int(m.split("-")[1])
        assert month_num not in (5, 6), f"f1 should not include {m}"
    assert len(months) > 0
```

**Step 3: Run tests**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage5-tier
source /home/xyz/workspace/pmodel/.venv/bin/activate
python -m pytest ml/tests/test_config_helpers.py -v
```

**Step 4: Commit**

```bash
git add ml/config.py ml/tests/test_config_helpers.py
git commit -m "feat: add delivery_month, auction schedule, period helpers to config"
```

---

## Task 2: Fix ground truth join in data_loader.py (BUG #1)

**Files:**
- Modify: `ml/data_loader.py:86-89`

**Why:** `load_v62b_month` joins realized DA on `auction_month`, but for f1 the ground truth is realized DA for `delivery_month = auction_month + 1`. This is the #1 correctness bug.

**Step 1: Update load_v62b_month**

Replace lines 86-92:

```python
    # Join realized DA ground truth
    # For fN, ground truth = realized DA for delivery_month (= auction_month + N)
    from ml.config import delivery_month as _delivery_month
    gt_month = _delivery_month(auction_month, period_type)
    realized = load_realized_da(gt_month, peak_type=class_type)
    df = df.join(realized, on="constraint_id", how="left")
    df = df.with_columns(pl.col("realized_sp").fill_null(0.0))

    n_binding = len(df.filter(pl.col("realized_sp") > 0))
    print(f"[data_loader] realized DA: {n_binding}/{len(df)} binding for {auction_month} "
          f"(gt_month={gt_month})")
```

**Step 2: Verify f0 still works identically**

```bash
python -c "
from ml.data_loader import load_v62b_month, clear_month_cache
import numpy as np

# f0: delivery_month == auction_month, so no change in behavior
clear_month_cache()
df = load_v62b_month('2022-09', 'f0', 'onpeak')
n_bind = len(df.filter(df['realized_sp'] > 0))
print(f'f0 2022-09: {len(df)} rows, {n_bind} binding')
assert n_bind > 0, 'f0 should have binding constraints'
"
```

**Step 3: Test f1 loads correct delivery month**

```bash
python -c "
from ml.data_loader import load_v62b_month, clear_month_cache

clear_month_cache()
df = load_v62b_month('2022-09', 'f1', 'onpeak')
n_bind = len(df.filter(df['realized_sp'] > 0))
print(f'f1 2022-09 (delivery=2022-10): {len(df)} rows, {n_bind} binding')
# Should show gt_month=2022-10 in the log output
assert n_bind > 0, 'f1 should have binding constraints for delivery month 2022-10'
"
```

**Step 4: Commit**

```bash
git add ml/data_loader.py
git commit -m "fix: GT join uses delivery_month for non-f0 period types"
```

---

## Task 3: Fix spice6 market_month in spice6_loader.py (BUG #2)

**Files:**
- Modify: `ml/spice6_loader.py:15-40`

**Why:** Line 38 hardcodes `market_month={auction_month}`. For f1, spice6 density is partitioned by `market_month = delivery_month = auction_month + 1`. Loading the wrong market_month gives incorrect density features (not leakage, but wrong data).

**Step 1: Update load_spice6_density signature and path**

```python
def load_spice6_density(
    auction_month: str,
    period_type: str = "f0",
) -> pl.DataFrame:
    """Load and aggregate spice6 density features for one month.

    Parameters
    ----------
    auction_month : str
        Month in YYYY-MM format.
    period_type : str
        Period type (f0, f1, etc.). Determines market_month = delivery_month.
    """
    from ml.config import delivery_month as _delivery_month
    market_month = _delivery_month(auction_month, period_type)
    market_round = "1"  # all monthly period types use round 1
    base = (
        Path(SPICE6_DENSITY_BASE)
        / f"auction_month={auction_month}"
        / f"market_month={market_month}"
        / f"market_round={market_round}"
    )
```

The rest of the function (lines 42-82) stays the same.

**Step 2: Verify f0 unchanged**

```bash
python -c "
from ml.spice6_loader import load_spice6_density
df = load_spice6_density('2022-09', 'f0')
print(f'f0 spice6 2022-09: {len(df)} rows')
assert len(df) > 0
"
```

**Step 3: Test f1 loads correct market_month**

```bash
python -c "
from ml.spice6_loader import load_spice6_density
df = load_spice6_density('2022-09', 'f1')
print(f'f1 spice6 2022-09 (market=2022-10): {len(df)} rows')
assert len(df) > 0, 'f1 spice6 should exist for auction=2022-09, market=2022-10'
"
```

**Step 4: Commit**

```bash
git add ml/spice6_loader.py
git commit -m "fix: spice6 uses delivery_month as market_month for non-f0"
```

---

## Task 4: Separate row-inclusion lag from feature-computation cutoff in run_v10e_lagged.py

**Files:**
- Modify: `scripts/run_v10e_lagged.py`

**Why:** The current code applies a single `lag` parameter to both (a) how far back the training window shifts and (b) the binding_freq cutoff for every row. For f1 with lag=2:
- Training window must shift by 2 (correct) — because last safe training row's GT needs delivery_month ≤ last_full_known
- But binding_freq cutoff for ALL rows (train + test) should use lag=1 — because bf depends on decision timing, not delivery offset

The invariant: **at decision time (~mid M-1), last complete realized DA = M-2**. The bf cutoff is always `months < M-1` in compute_bf's strict-< semantics, i.e., through M-2. This is lag=1 regardless of period type.

**Step 1: Introduce `as_of_lag` concept**

In `run_variant()`, change the enrichment calls to always use `lag=1` for binding_freq computation, while keeping the training window shift at `lag` (which is N+1 for fN):

Replace the current `enrich_df` call pattern (lines 164-175):

```python
        # Enrich training months — bf always uses lag=1 (decision-time cutoff)
        # Row inclusion is controlled by the training window (shifted by `lag`)
        BF_LAG = 1  # binding_freq cutoff = decision timing, always 1
        parts = []
        for tm in train_month_strs:
            part = train_df.filter(pl.col("query_month") == tm)
            if len(part) > 0:
                part = enrich_df(part, tm, bs, lag=BF_LAG)
                parts.append(part)
        if not parts:
            print(f"  {m}: SKIP (no training data)")
            continue
        train_df = pl.concat(parts)
        test_df = enrich_df(test_df, m, bs, lag=BF_LAG)
```

The training window shift (lines 148-158) stays at `lag` — that controls which rows are included, not how features are computed.

**CRITICAL: per-row feature cutoff, NOT per-target cutoff.**

Each row's binding_freq is computed relative to **that row's own auction_month**, not the target row's.
The `enrich_df(part, tm, ...)` call passes `tm` (the training row's month), so:
- Training row (2024-11, f1): `enrich_df(part, "2024-11", bs, lag=1)` → cutoff=2024-10 → bf sees through 2024-09
- Target row (2025-02, f1): `enrich_df(test_df, "2025-02", bs, lag=1)` → cutoff=2025-01 → bf sees through 2024-12

If someone passes the **target month** to all `enrich_df` calls (including training rows), that is **temporal leakage** — it gives training rows access to future binding data they would not have had at their own decision time.

**Step 2: Update the docstring for enrich_df**

```python
def enrich_df(df: pl.DataFrame, month: str, bs: dict[str, set[str]], lag: int = 1) -> pl.DataFrame:
    """Add features. lag=1 means bf cutoff = prev_month(month), i.e., through M-2.

    This lag should ALWAYS be 1 (decision-time cutoff). The training window shift
    (which rows to include) is handled separately by the caller.
    """
```

**Step 3: Verify f0 results unchanged**

Run a quick single-month f0 test before and after:

```bash
python -c "
import sys, json
sys.path.insert(0, '.')
from scripts.run_v10e_lagged import *
bs = load_all_binding_sets('onpeak')
# Single month, lag=1 (f0)
pm = run_variant('test-f0', ['2022-09'], bs, lag=1, class_type='onpeak')
print(json.dumps(pm['2022-09'], indent=2))
"
# Save output, compare after change — VC@20 etc. should be identical for f0
```

**Step 4: Commit**

```bash
git add scripts/run_v10e_lagged.py
git commit -m "fix: separate row-inclusion lag from bf cutoff (bf always lag=1)"
```

---

## Task 5: Add `collect_usable_months` helper + use in run_v10e_lagged.py only

**Files:**
- Modify: `ml/config.py` (add helper)
- Modify: `scripts/run_v10e_lagged.py` (use helper for training month generation)
- Create: `ml/tests/test_config_helpers.py` (add tests)
- **DO NOT modify**: `ml/data_loader.py` — keep the generic loader contract unchanged

**Why:** The current `run_v10e_lagged.py` generates a fixed contiguous window of calendar months. For f1, May and June have no data — so a target like (2022-09, f1) would skip May+June and end up with only **6 training months**. The correct behavior is to walk backward collecting 8 *usable* rows.

**Design decision (per code review):** Do NOT push this logic into the shared generic loader (`load_train_val_test`). That loader is used by `pipeline.py`, `benchmark.py`, and older experiments — changing its contract would silently break them. Instead, `collect_usable_months` is a standalone helper in `config.py`, and only the lagged/as-of scripts (`run_v10e_lagged.py`) call it. The script owns training-month selection; the loader stays dumb.

**Step 1: Add `collect_usable_months` to config.py**

```python
def collect_usable_months(
    target_auction_month: str,
    period_type: str,
    n_months: int = 8,
    min_months: int = 6,
    max_lookback: int = 24,
) -> list[str]:
    """Walk backward from target, collect n_months usable training auction months.

    A month is "usable" if:
      1. The period_type exists for that month (per MISO auction schedule)
      2. delivery_month(month, period_type) <= last_full_known

    last_full_known = target_auction_month - 2 (last complete realized DA at decision time).

    For f0, this returns the same months as a contiguous 8-month window.
    For f1+, it skips months where fN doesn't exist (May/Jun for f1).

    Returns most-recent-first order. Caller should reverse for chronological.
    Returns empty list if fewer than min_months are available (caller should skip).
    """
    import pandas as pd

    target_ts = pd.Timestamp(target_auction_month)
    # At decision time (~mid target-1), last complete realized DA = target - 2
    last_full_known = (target_ts - pd.DateOffset(months=2)).strftime("%Y-%m")

    usable = []
    for i in range(1, max_lookback + 1):
        candidate = (target_ts - pd.DateOffset(months=i)).strftime("%Y-%m")
        if not has_period_type(candidate, period_type):
            continue
        dm = delivery_month(candidate, period_type)
        if dm > last_full_known:
            continue
        usable.append(candidate)
        if len(usable) >= n_months:
            break

    if len(usable) < min_months:
        return []  # not enough history — caller should skip this eval month

    return usable
```

**Step 2: Update `run_variant` in run_v10e_lagged.py**

Replace the current `shifted_eval` / `load_train_val_test` / manual `train_month_strs` pattern (lines 148-158) with:

```python
        from ml.config import collect_usable_months
        train_month_strs = collect_usable_months(m, period_type, n_months=8)
        if not train_month_strs:
            print(f"  {m}: SKIP (insufficient training history, need ≥6 usable months)")
            continue
        actual_depth = len(train_month_strs)
        train_month_strs = list(reversed(train_month_strs))  # chronological order

        # Load each training month individually (no load_train_val_test — script owns month selection)
        parts = []
        for tm in train_month_strs:
            try:
                part = load_v62b_month(tm, period_type, class_type)
                part = part.with_columns(pl.lit(tm).alias("query_month"))
                part = enrich_df(part, tm, bs, lag=BF_LAG)
                parts.append(part)
            except FileNotFoundError:
                print(f"  {tm}: SKIP (not found)")
```

This removes the dependency on `load_train_val_test` entirely for the lagged path. The script is the single owner of training-month selection.

**Step 3: Write tests**

```python
# ml/tests/test_config_helpers.py (add to existing)
from ml.config import collect_usable_months

def test_collect_usable_f0_returns_8_contiguous():
    months = collect_usable_months("2022-09", "f0", n_months=8)
    assert len(months) == 8
    # f0 exists every month; latest safe: delivery(2022-07, f0)=2022-07 <= last_full_known=2022-07
    assert months[0] == "2022-07"
    assert months[-1] == "2021-12"

def test_collect_usable_f1_skips_may_june():
    months = collect_usable_months("2022-09", "f1", n_months=8)
    assert len(months) == 8
    for m in months:
        month_num = int(m.split("-")[1])
        assert month_num not in (5, 6), f"f1 should skip {m}"
    # last_full_known = 2022-07. delivery(2022-07, f1)=2022-08 > 2022-07 → NOT usable.
    # Latest safe: 2022-04, delivery=2022-05 <= 2022-07 → usable.
    assert months[0] == "2022-04"

def test_collect_usable_f0_matches_old_contiguous_window():
    """f0 collect_usable should produce identical months to the old lag=1 contiguous window."""
    months = collect_usable_months("2022-09", "f0", n_months=8)
    assert months == ["2022-07", "2022-06", "2022-05", "2022-04",
                      "2022-03", "2022-02", "2022-01", "2021-12"]

def test_collect_usable_f1_early_month_insufficient():
    """Early f1 months don't have 6 usable rows — should return empty."""
    # 2020-07 is the first f1 eval month. f1 data starts 2020-01.
    # Usable: 2020-04, 2020-03, 2020-02, 2020-01 = only 4 < min_months=6
    months = collect_usable_months("2020-07", "f1", n_months=8, min_months=6)
    assert months == [], f"Expected empty, got {months}"

def test_collect_usable_f1_enough_history():
    """Later f1 months should have sufficient history."""
    months = collect_usable_months("2021-01", "f1", n_months=8, min_months=6)
    assert len(months) >= 6
```

**Step 4: Run tests**

```bash
python -m pytest ml/tests/test_config_helpers.py -v
```

**Step 5: Commit**

```bash
git add ml/config.py scripts/run_v10e_lagged.py ml/tests/test_config_helpers.py
git commit -m "fix: training window collects N usable rows, skips missing period types"
```

---

## Task 6: Verify V6.2B / Spice6 snapshot provenance for f1

**Files:**
- No code changes — verification only
- Output: documented assumption in plan and saved config

**Why (per code review):** The plan fixes GT join, spice6 market_month, binding_freq cutoff, and row inclusion. But it does not prove that the V6.2B snapshot features (`da_rank_value`, `density_mix_rank_value`, `density_ori_rank_value`) and Spice6 density outputs (`prob_exceed_110`, `constraint_limit`) are point-in-time safe for f1. These features drive ~33% of model importance. If the snapshots were regenerated after the auction, they would be leaky.

**What we believe:** The V6.2B signal pipeline generates all period types for an auction month at the same time, before the auction. The Spice6 density model runs forward-looking simulations before the auction. Both are pre-decision artifacts.

**Step 1: Verify V6.2B f1 snapshot timestamps**

```bash
# Check file modification times for f1 vs f0 for the same auction month
stat /opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1/2022-09/f0/onpeak
stat /opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1/2022-09/f1/onpeak
# If both have the same timestamp → generated together, pre-auction
```

**Step 2: Verify Spice6 density timestamps**

```bash
stat /opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/auction_month=2022-09/market_month=2022-09/market_round=1/
stat /opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/auction_month=2022-09/market_month=2022-10/market_round=1/
```

**Step 3: Spot-check that V6.2B f1 features differ from f0 only in flow columns**

```bash
python -c "
import polars as pl
f0 = pl.read_parquet('/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1/2022-09/f0/onpeak')
f1 = pl.read_parquet('/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1/2022-09/f1/onpeak')
shared = set(f0['constraint_id'].to_list()) & set(f1['constraint_id'].to_list())
print(f'Shared constraints: {len(shared)}')
# For shared constraints, da_rank_value should be identical (historical, not delivery-dependent)
# ori_mean, mix_mean should differ (flow forecasts for different delivery months)
merged = f0.filter(pl.col('constraint_id').is_in(list(shared))).sort('constraint_id').select(['constraint_id', 'da_rank_value', 'ori_mean']).rename({'da_rank_value': 'da_f0', 'ori_mean': 'ori_f0'})
merged2 = f1.filter(pl.col('constraint_id').is_in(list(shared))).sort('constraint_id').select(['constraint_id', 'da_rank_value', 'ori_mean']).rename({'da_rank_value': 'da_f1', 'ori_mean': 'ori_f1'})
j = merged.join(merged2, on='constraint_id')
print(f'da_rank_value identical: {(j[\"da_f0\"] == j[\"da_f1\"]).all()}')
print(f'ori_mean identical: {(j[\"ori_f0\"] == j[\"ori_f1\"]).all()}')
print(f'ori_mean max diff: {(j[\"ori_f0\"] - j[\"ori_f1\"]).abs().max()}')
"
```

If da_rank_value is identical across f0/f1 and ori_mean differs, this is **consistent with** the snapshot being generated pre-auction with delivery-month-specific flow forecasts. However, matching timestamps and identical historical features do not **prove** the files were generated before the actual auction cutoff, nor that upstream feature construction avoided post-cutoff data. This remains an upstream assumption.

**Step 4: Document the assumption**

Record in the final config/metrics:
```json
{
  "assumptions": [
    "V6.2B snapshots assumed to be pre-auction artifacts (consistent: same timestamps for f0/f1, identical da_rank_value across ptypes)",
    "Spice6 density outputs assumed to be forward-looking model runs, not realized data",
    "Neither assumption is formally proven — would require upstream pipeline audit"
  ]
}
```

If timestamp checks are inconsistent (e.g., f1 generated later than f0), flag as a known risk.

**No commit needed — this is a verification step.**

---

## Task 7: Add --ptype flag to run_v10e_lagged.py (uses collect_usable_months from Task 5)

**Files:**
- Modify: `scripts/run_v10e_lagged.py` (argparser + all hardcoded "f0" references)

**Why:** The script hardcodes "f0" in 4 places (lines 152, 161, 237-238, 288-289). Must parameterize for f1.

**Step 1: Add --ptype to argparser** (line 272-276)

For f0, keep `--lag` for backward compatibility (existing `v10e-lag1` naming). For f1+, row selection is fully handled by `collect_usable_months` — `--lag` is not used. Record the effective lag as metadata only.

```python
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="V10e with production lag")
    parser.add_argument("--class-type", default="onpeak", choices=["onpeak", "offpeak"])
    parser.add_argument("--ptype", default="f0", help="Period type (f0, f1, f2, f3)")
    parser.add_argument("--lag", type=int, default=None,
                        help="Production lag for f0 only (default: 1). Ignored for f1+ "
                             "(collect_usable_months handles row selection).")
    parser.add_argument("--dev-only", action="store_true", help="Skip holdout")
    args = parser.parse_args()

    class_type = args.class_type
    period_type = args.ptype

    from ml.config import period_offset
    # For f0, lag controls the old shifted_eval logic (backward compat)
    # For f1+, lag is metadata only — collect_usable_months owns row selection
    effective_lag = period_offset(period_type) + 1
    if period_type == "f0":
        lag = args.lag if args.lag is not None else 1
    else:
        lag = effective_lag  # metadata only, not used in row selection
        if args.lag is not None:
            print(f"[main] WARNING: --lag ignored for {period_type} "
                  f"(collect_usable_months handles row selection)")
```

**Step 2: Replace all hardcoded "f0" references**

Note: Task 5 already replaced `shifted_eval`/`load_train_val_test` with `collect_usable_months` + per-month `load_v62b_month` for ALL period types (including f0, which produces identical results — verified by test). So the `load_train_val_test` call no longer exists. The remaining hardcoded "f0" references to fix:

Line 161 (`load_v62b_month` for test month):
```python
test_df = load_v62b_month(m, period_type, class_type)
```

Lines 237-238 (saved eval_config):
```python
"period_type": period_type, "lag": lag,
```

Lines 288-289 (registry/holdout paths):
```python
reg_slice = registry_root(period_type, class_type, base_dir=ROOT / "registry")
ho_slice = holdout_root(period_type, class_type, base_dir=ROOT / "holdout")
```

**Step 3: Filter eval months by auction schedule**

For f1, months 5 and 6 don't have f1 data. Add filtering after the eval months are determined:

```python
    from ml.config import has_period_type

    # Filter eval months to those where period_type exists
    dev_eval = [m for m in _FULL_EVAL_MONTHS if has_period_type(m, period_type)]
    holdout_eval = [m for m in HOLDOUT_MONTHS if has_period_type(m, period_type)]
    print(f"[main] {period_type}: {len(dev_eval)} dev months, {len(holdout_eval)} holdout months")
```

Use `dev_eval` and `holdout_eval` instead of `_FULL_EVAL_MONTHS` and `HOLDOUT_MONTHS` in the `run_variant` calls.

**Step 4: Fix the comparison block**

The comparison block (lines 300-334) tries to load v0 and v10e baselines. For f1, these won't exist initially. Wrap in existence checks:

```python
    # ── Comparison ──
    v0_path = reg_slice / "v0" / "metrics.json"
    if v0_path.exists():
        v0_dev = json.load(open(v0_path))["aggregate"]["mean"]
        lag_dev_clean = {m: {k: v for k, v in met.items() if not k.startswith("_")} for m, met in dev_pm.items()}
        lag_dev = aggregate_months(lag_dev_clean)["mean"]
        labels = {"v0 (formula)": v0_dev, version_id: lag_dev}

        # Optionally add no-lag baseline if it exists
        nolag_path = reg_slice / "v10e" / "metrics.json"
        if nolag_path.exists():
            labels["v10e (no lag)"] = json.load(open(nolag_path))["aggregate"]["mean"]

        print_comparison(labels, f"DEV COMPARISON ({period_type}/{class_type}, {len(dev_eval)} months)")
    else:
        print(f"\n[main] No v0 baseline in {reg_slice} — run v0 for {period_type} first")
```

Same pattern for holdout comparison.

**Step 5: Thread `period_type` into `run_variant` and `save_results`**

Pass `period_type` down so it's recorded in the saved metrics.

**Step 6: Commit**

```bash
git add scripts/run_v10e_lagged.py
git commit -m "feat: add --ptype flag to run_v10e_lagged for f1/f2/f3 support"
```

---

## Task 8: Add --ptype flag to run_v0_formula_baseline.py

**Files:**
- Modify: `scripts/run_v0_formula_baseline.py`

**Why:** Same issue — hardcodes "f0" in signal path, registry path, holdout path, and eval_config. Need to run `python scripts/run_v0_formula_baseline.py --ptype f1 --full --holdout` for f1 baseline.

**Step 1: Add --ptype to argparser** (line 111)

```python
    parser.add_argument("--ptype", default="f0", help="Period type (f0, f1, f2, f3)")
```

**Step 2: Replace hardcoded "f0"**

Line 38 (signal path in `evaluate_month`):
```python
def evaluate_month(month: str, class_type: str = "onpeak", period_type: str = "f0") -> dict:
    path = Path(V62B_SIGNAL_BASE) / month / period_type / class_type
```

The realized DA join in `evaluate_month` must also use delivery_month:
```python
    from ml.config import delivery_month as _delivery_month
    gt_month = _delivery_month(month, period_type)
    realized = load_realized_da(gt_month, peak_type=class_type)
```

Line 160 (registry path):
```python
    v0_dir = registry_root(period_type, class_type, base_dir=base_registry) / version_id
```

Line 202 (gates/champion):
```python
    slice_dir = registry_root(period_type, class_type, base_dir=base_registry)
```

Line 234 (holdout path):
```python
    holdout_dir = holdout_root(period_type, class_type, base_dir=base_holdout) / version_id
```

All `evaluate_month` calls must pass `period_type`:
```python
    per_month[month] = evaluate_month(month, class_type=class_type, period_type=period_type)
```

**Step 3: Filter eval months by auction schedule**

```python
    from ml.config import has_period_type
    period_type = args.ptype
    eval_months = _FULL_EVAL_MONTHS if args.full else _DEFAULT_EVAL_MONTHS
    eval_months = [m for m in eval_months if has_period_type(m, period_type)]
```

Same for holdout months.

**Step 4: Adjust expected-value validation**

The expected numbers check (lines 141-155) is only valid for f0/onpeak. Guard it:

```python
    if class_type == "onpeak" and period_type == "f0":
        expected = {"VC@20": 0.2817, ...}
        ...
```

**Step 5: Commit**

```bash
git add scripts/run_v0_formula_baseline.py
git commit -m "feat: add --ptype flag to run_v0_formula_baseline for f1/f2/f3 support"
```

---

## Task 9: Add guard for missing gates.json/champion.json in compare.py

**Files:**
- Modify: `ml/compare.py` (~lines 382-396)

**Why:** New slices (f1/onpeak, f0/offpeak) won't have gates.json or champion.json until v0 baseline runs. `run_comparison` crashes on these.

**Step 1: Add existence guards**

```python
    # Load gates
    if gp.exists():
        with open(gp) as f:
            gates_data = json.load(f)
        gates = gates_data.get("gates", {})
    else:
        print(f"[compare] No gates.json at {gp} — skipping gate checks")
        gates = {}

    # Load champion
    if cp.exists():
        with open(cp) as f:
            champion_data = json.load(f)
        current_champion = champion_data.get("version", None)
    else:
        print(f"[compare] No champion.json at {cp} — no champion set")
        current_champion = None
```

**Step 2: Commit**

```bash
git add ml/compare.py
git commit -m "fix: guard missing gates.json/champion.json in compare.py"
```

---

## Task 10: Deprecate run_36mo_comparison.py

**Files:**
- Modify: `scripts/run_36mo_comparison.py` (add deprecation notice at top)

**Step 1: Add deprecation**

```python
"""DEPRECATED: Results archived to archive/registry/. Superseded by v10e-lag1.

This script used flat registry paths (registry/{version_id}/) which no longer exist.
Use `python -m ml.compare --ptype f0 --class-type onpeak` instead.
"""
```

**Step 2: Commit**

```bash
git add scripts/run_36mo_comparison.py
git commit -m "deprecate: mark run_36mo_comparison.py as deprecated (stale flat paths)"
```

---

## Task 11: Run f1 v0 baseline

**Files:**
- Output: `registry/f1/onpeak/v0/metrics.json`, `config.json`, `meta.json`
- Output: `registry/f1/onpeak/gates.json`, `champion.json`
- Output: `holdout/f1/onpeak/v0/metrics.json`

**Step 1: Run f1 v0 with full eval + holdout**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage5-tier
source /home/xyz/workspace/pmodel/.venv/bin/activate
python scripts/run_v0_formula_baseline.py --ptype f1 --full --holdout --class-type onpeak
```

Expected:
- 30 dev months for v0 formula (May/Jun excluded; v0 has no training window so no min-history skip)
- 20 holdout months
- Results in `registry/f1/onpeak/v0/` and `holdout/f1/onpeak/v0/`
- Gates and champion initialized from v0

Note: v0 (formula) evaluates all 30 f1 dev months since it has no training window. v2 (ML) will evaluate ~28 months due to the min-history skip for early months. This is expected — the comparison table should note the different month counts.

**Step 2: Sanity check**

```bash
python -c "
import json
dev = json.load(open('registry/f1/onpeak/v0/metrics.json'))
ho = json.load(open('holdout/f1/onpeak/v0/metrics.json'))
print(f'Dev: {dev[\"n_months\"]} months, VC@20={dev[\"aggregate\"][\"mean\"][\"VC@20\"]:.4f}')
print(f'Holdout: {ho[\"n_months\"]} months, VC@20={ho[\"aggregate\"][\"mean\"][\"VC@20\"]:.4f}')
"
```

**Step 3: Commit results**

```bash
git add registry/f1/ holdout/f1/
git commit -m "data: f1 v0 formula baseline (dev + holdout)"
```

---

## Task 12: Run f1 v1 — blend grid search

**Why:** The V6.2B production formula is `(0.60, 0.30, 0.10)`. For f0, grid search found `(0.85, 0, 0.15)` was better (density_mix has ~0 correlation with realized DA). For f1, the density features forecast a **different delivery month**, so the optimal blend may differ. We must search, not assume.

**Verified:** The V6.2B formula `(0.60, 0.30, 0.10)` reproduces `rank_ori` exactly for both f0 and f1 (`max_abs_diff = 0.0`). So v0 is correct for all period types.

**Files:**
- Create: `scripts/run_f1_blend_search.py`
- Output: `registry/f1/onpeak/v1/metrics.json`, `config.json`
- Output: `holdout/f1/onpeak/v1/metrics.json`

**Step 1: Create the blend search script**

The script should:
1. Load f1 V6.2B data for each eval month (using `load_v62b_month` which now joins GT on delivery_month)
2. Grid search over (da, dmix, dori) triplets that sum to 1.0, step 0.05 — same 231-triplet grid as v7
3. For each triplet, compute `score = -(w_da * da_rank_value + w_dmix * density_mix_rank_value + w_dori * density_ori_rank_value)`
4. Evaluate VC@20 on the 30 f1 dev months
5. Select best triplet by mean VC@20
6. Run that blend on holdout (20 months)
7. Save as `v1` in `registry/f1/onpeak/`

Key differences from the f0 v7 search:
- Uses `load_v62b_month(month, "f1", "onpeak")` — this joins GT on delivery_month (Task 2 fix)
- Eval months filtered by `has_period_type(m, "f1")` — excludes May/Jun
- No temporal leakage concern here since the blend is purely formula-based (no realized DA features)

```python
#!/usr/bin/env python
"""F1 blend search: grid-search da/dmix/dori weights on f1 dev months.

Mirrors the f0 v7 experiment but on f1 data with correct GT (delivery_month).
"""
import itertools, json, sys, time
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import has_period_type, _FULL_EVAL_MONTHS
from ml.data_loader import load_v62b_month, clear_month_cache
from ml.evaluate import aggregate_months, evaluate_ltr
from ml.registry_paths import registry_root, holdout_root

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_MONTHS = [f"{y:04d}-{m:02d}" for y in (2024, 2025) for m in range(1, 13)]

def search_blend(eval_months, period_type="f1", class_type="onpeak"):
    # Preload all months
    data = {}
    for m in eval_months:
        df = load_v62b_month(m, period_type, class_type)
        data[m] = {
            "da": df["da_rank_value"].to_numpy(),
            "dmix": df["density_mix_rank_value"].to_numpy(),
            "dori": df["density_ori_rank_value"].to_numpy(),
            "actual": df["realized_sp"].to_numpy().astype(np.float64),
        }
        clear_month_cache()

    # Grid: step 0.05, sum to 1.0
    steps = [round(x * 0.05, 2) for x in range(21)]  # 0.00 to 1.00
    triplets = [(a, b, c) for a in steps for b in steps for c in steps
                if abs(a + b + c - 1.0) < 1e-9]
    print(f"[blend] Searching {len(triplets)} triplets on {len(eval_months)} months")

    best_vc20, best_weights = -1, None
    results = []
    for w_da, w_dmix, w_dori in triplets:
        per_month = {}
        for m, d in data.items():
            scores = -(w_da * d["da"] + w_dmix * d["dmix"] + w_dori * d["dori"])
            metrics = evaluate_ltr(d["actual"], scores)
            per_month[m] = metrics
        agg = aggregate_months(per_month)
        mean_vc20 = agg["mean"]["VC@20"]
        results.append({"da": w_da, "dmix": w_dmix, "dori": w_dori, "VC@20": mean_vc20})
        if mean_vc20 > best_vc20:
            best_vc20 = mean_vc20
            best_weights = (w_da, w_dmix, w_dori)

    print(f"[blend] Best: da={best_weights[0]}, dmix={best_weights[1]}, "
          f"dori={best_weights[2]}, VC@20={best_vc20:.4f}")
    return best_weights, results
```

**Step 2: Run search, evaluate holdout, save**

The main function should:
- Run grid search on dev months
- Evaluate best blend on holdout
- Save full results (grid + best) to `registry/f1/onpeak/v1/`
- Compare with v0

**Step 3: Commit**

```bash
git add scripts/run_f1_blend_search.py registry/f1/ holdout/f1/
git commit -m "data: f1 v1 blend search (dev + holdout)"
```

**Key question to answer:** Does the f0-optimal `(0.85, 0, 0.15)` transfer to f1, or does f1 need different weights? This determines whether the v2 ML feature `v7_formula_score` should use f1-specific weights.

**Important: v1 dev metrics are in-sample.** The blend weights are selected on the same 30 dev months used for reporting. This makes v1 dev numbers optimistic relative to v0 (which has no tuning). The holdout numbers are the fair comparison. Overfitting risk is low (3-parameter grid, 30 months), but all saved config and comparison output should label v1 dev as `"note": "weights selected on dev — holdout is the fair comparison"`. The primary purpose of v1 is finding the right blend for v2's formula_score feature, not being a standalone production model.

---

## Task 13: Run f1 v2 — full ML model (binding_freq + features)

**Why:** This is the f1 equivalent of v10e-lag1 for f0. Uses the v1 blend weights as `formula_score` feature plus binding_freq and other ML features.

**Files:**
- Modify: `scripts/run_v10e_lagged.py` — parameterize blend weights
- Output: `registry/f1/onpeak/v2/metrics.json`
- Output: `holdout/f1/onpeak/v2/metrics.json`

**Step 1: Parameterize blend weights in `enrich_df`**

Currently `enrich_df` hardcodes `V7_DA=0.85, V7_DMIX=0.00, V7_DORI=0.15`. These are f0-specific. Add an optional parameter:

```python
V7_DA, V7_DMIX, V7_DORI = 0.85, 0.00, 0.15  # f0 default

# Per-period-type blend weights (updated after blend search)
BLEND_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "f0": (0.85, 0.00, 0.15),
    # "f1": to be filled from Task 12 results
}

def enrich_df(df, month, bs, lag=1, blend_weights=None):
    """..."""
    cutoff = month
    for _ in range(lag):
        cutoff = prev_month(cutoff)

    cids = df["constraint_id"].to_list()
    w_da, w_dmix, w_dori = blend_weights or (V7_DA, V7_DMIX, V7_DORI)
    df = df.with_columns(
        (w_da * pl.col("da_rank_value")
         + w_dmix * pl.col("density_mix_rank_value")
         + w_dori * pl.col("density_ori_rank_value")
        ).alias("v7_formula_score")
    )
    # ... rest unchanged
```

In `run_variant` / `main`, look up blend weights by period_type:

```python
    blend = BLEND_WEIGHTS.get(period_type, (V7_DA, V7_DMIX, V7_DORI))
    # pass to enrich_df calls
```

**Step 2: Run f1 v2**

```bash
python scripts/run_v10e_lagged.py --ptype f1 --class-type onpeak
```

This auto-computes `lag=2` for f1 and uses f1-specific blend weights (from Task 12).

Expected:
- `version_id = v2` (or `v10e-lag2` — see naming note below)
- 30 dev months, 20 holdout months
- Comparison table vs v0, v1

**Step 3: Version naming**

For the f1 registry, use a clean version ladder:
- `v0` = V6.2B formula (0.60, 0.30, 0.10)
- `v1` = optimized blend (from Task 12)
- `v2` = full ML model (v1 blend as feature + binding_freq + spice6)

Update the `version_id` construction in `main()`:
```python
    if period_type == "f0":
        version_id = f"v10e-lag{lag}"  # preserve f0 naming
    else:
        version_id = "v2"  # clean f1 naming
```

Record in saved config:
```json
{
    "row_inclusion_lag": 2,
    "bf_cutoff_lag": 1,
    "blend_weights": {"da": 0.85, "dmix": 0.00, "dori": 0.15},
    "note": "row_inclusion_lag=N+1 for fN; bf_cutoff_lag=1 always (decision-time)"
}
```

**Step 4: Commit**

```bash
git add scripts/run_v10e_lagged.py registry/f1/ holdout/f1/
git commit -m "data: f1 v2 full ML model (dev + holdout)"
```

---

## Task 14: Run f1 v0/v1/v2 for offpeak

**Files:**
- Output: `registry/f1/offpeak/v0/`, `holdout/f1/offpeak/v0/`
- Output: `registry/f1/offpeak/v1/`, `holdout/f1/offpeak/v1/`
- Output: `registry/f1/offpeak/v2/`, `holdout/f1/offpeak/v2/`

**Step 1: Run v0 offpeak**

```bash
python scripts/run_v0_formula_baseline.py --ptype f1 --full --holdout --class-type offpeak
```

**Step 2: Run v1 blend search offpeak**

```bash
python scripts/run_f1_blend_search.py --class-type offpeak
```

Note: offpeak may have different optimal blend weights than onpeak.

**Step 3: Run v2 ML offpeak**

```bash
python scripts/run_v10e_lagged.py --ptype f1 --class-type offpeak
```

**Step 4: Commit**

```bash
git add registry/f1/ holdout/f1/
git commit -m "data: f1 offpeak v0/v1/v2 (dev + holdout)"
```

---

## Task 15: Comparison table + analysis

**Files:**
- Output: `registry/f1/onpeak/comparison.json`

**Step 1: Print f1 version ladder results**

```bash
python -c "
import json

for ct in ['onpeak', 'offpeak']:
    print(f'\n=== f1 {ct} ===')
    for v in ['v0', 'v1', 'v2']:
        path = f'registry/f1/{ct}/{v}/metrics.json'
        try:
            d = json.load(open(path))
            m = d['aggregate']['mean']
            print(f'{v}: VC@20={m[\"VC@20\"]:.4f} VC@100={m[\"VC@100\"]:.4f} Spearman={m[\"Spearman\"]:.4f}')
        except FileNotFoundError:
            print(f'{v}: not found')
"
```

**Step 2: Answer key questions**

- Does the f0 blend `(0.85, 0, 0.15)` transfer to f1?
- How much does blend optimization (v1 vs v0) contribute?
- How much do ML features (v2 vs v1) contribute?
- Does f1 performance hold up on holdout?

**Step 3: Commit**

```bash
git add registry/f1/
git commit -m "data: f1 comparison tables"
```

---

## Task 16: Update docs and memory

**Files:**
- Modify: `multi-period-extension.md` — update Phase 1 status with actual results
- Modify: `mem.md` — add f1 results
- Modify: `CLAUDE.md` — note f1 support

**Step 1: Record results in mem.md**

Add f1 section with actual numbers from Tasks 11-14, including:
- v0/v1/v2 dev and holdout numbers
- Optimal blend weights for f1 (from v1 search)
- Whether f0 blend transfers

**Step 2: Update multi-period-extension.md Phase 1**

Mark Phase 1 as complete with actual performance numbers.

**Step 3: Commit**

```bash
git add multi-period-extension.md mem.md CLAUDE.md
git commit -m "docs: f1 implementation complete with results"
```

---

## Gap Review

Reviewed after v3 review integration (16-task version ladder):

**Gaps found and fixed (v3 review):**
1. **Minimum-history policy** (High): Early f1 eval months (2020-07, 2020-08) only have 4-5 usable training rows. Fixed: `collect_usable_months` returns empty if `< min_months` (default 6). Caller skips. Expected ~28 ML dev months, not 30. v0 formula still evaluates all 30 (no training window).
2. **`--lag` inconsistency** (Medium): After Task 5, `collect_usable_months` owns row selection for f1+, making `--lag` dead logic. Fixed: `--lag` kept for f0 backward compat only, ignored with warning for f1+, recorded as metadata.
3. **v1 in-sample** (Medium): Blend weights tuned on same dev months used for reporting. Fixed: v1 dev labeled "selected on dev — holdout is the fair comparison" in saved config.
4. **Task 6 timestamp overstatement** (Medium): File mtimes don't prove pre-auction generation. Fixed: softened to "consistent with" pre-auction, documented as upstream assumption.
5. **Quarterly guard** (Low): `period_offset("q4")` would give wrong semantics. Fixed: `period_offset` raises ValueError on non-`f` types. Quarterly entries kept in `MISO_AUCTION_SCHEDULE` for `has_period_type`.

**Previously verified (no gaps):**
6. **Two GT fix sites**: `data_loader.py` (Task 2) and `run_v0_formula_baseline.py` (Task 8) both fixed independently.
7. **`pipeline.py` / `benchmark.py` not broken**: Default `period_type="f0"` → `delivery_month == auction_month`. No change.
8. **Existing tests not broken**: `test_data_loader.py` uses default f0. No change.
9. **`run_holdout_test.py` out of scope**: Uses `run_pipeline` for old champions. f1 holdout handled by dedicated scripts.
10. **Blend weights propagated**: `enrich_df` parameterized in Task 13 so v2 uses f1-specific weights.
11. **V6.2B formula identical across ptypes**: Verified `max_abs_diff = 0.0` for f0/f1.
12. **`spice6_loader.py` stub**: `market_round = "1"` always — harmless. Task 3 fixes the real bug.

---

## Verification Checklist

After all tasks, verify:

1. **f0 unchanged**: Run `python scripts/run_v10e_lagged.py --ptype f0 --dev-only` and compare VC@20 with existing `registry/f0/onpeak/v10e-lag1/metrics.json` — must be identical
2. **f1 GT correctness**: For f1 auction 2022-09, verify the log shows `gt_month=2022-10`
3. **f1 spice6 correctness**: Verify log shows loading from `market_month=2022-10` for f1 auction 2022-09
4. **f1 bf cutoff (per-row, not per-target)**: Verify each row's binding_freq uses that row's own `prev_month(auction_month)` as cutoff, NOT the target row's cutoff. E.g., training row (2024-11, f1) should see bf through 2024-09, while target (2025-02, f1) sees through 2024-12
5. **f1 training window backfill**: For target (2022-09, f1), verify `collect_usable_months` returns 8 months, none in May/Jun, and the latest is 2022-04 (not 2022-07, because delivery(2022-07,f1)=2022-08 > last_full_known=2022-07)
6. **Registry structure**: `ls registry/f1/onpeak/` shows `v0/`, `v1/`, `v2/`, `gates.json`, `champion.json`
7. **No May/Jun f1 months**: Verify no eval results for months 5 or 6
8. **Version ladder monotonic**: v0 ≤ v1 ≤ v2 on VC@20 (blend search should beat formula, ML should beat blend)
9. **v2 blend weights match v1**: Verify `v7_formula_score` in v2 uses the weights found by v1 search, not f0's `(0.85, 0, 0.15)` (unless they happen to be the same)
10. **Holdout consistency**: v2 holdout VC@20 improvement over v0 should be directionally consistent with dev (may be smaller, but not negative)
11. **Min-history skip**: Verify early f1 months (2020-07, 2020-08) are skipped for ML (v2) due to `< 6` usable training rows. v0/v1 formula should still include them.
12. **`--lag` ignored for f1**: Run with `--ptype f1 --lag 3` and verify warning is printed and results are unchanged
13. **v1 labeled as in-sample**: Check `registry/f1/onpeak/v1/config.json` contains the "selected on dev" note
14. **`period_offset("q4")` raises**: Quick `python -c "from ml.config import period_offset; period_offset('q4')"` should raise ValueError
