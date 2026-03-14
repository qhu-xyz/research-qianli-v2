# Integration Guide: MW Seasonal Factor → Banding Pipeline v3

## Overview

Create a **v3** in `/home/xyz/workspace/research-qianli/research-f0p-band/pjm/v3/` (note: v3 already exists as a blend search experiment — this would be a new v3 or the next available version number). The change is small: multiply `mtm_1st_mean` by a period-type-specific scalar before it enters `compute_baseline_v2()`.

---

## What `mtm_1st_mean` Is and Where It Comes From

### Computation chain (pbase → pmodel → banding)

1. **pbase** (`pbase/analysis/tools/base.py:17313`): `get_m2m_mcp_for_paths()` fetches nodal MCPs from the MCP calculator. For each path: `mtm_1st = sink_node_mcp - source_node_mcp`. This is the raw MCP in total-dollar units for the **first available clearing horizon**.

2. **pbase** (`base.py:17338`): `get_m2m_mcp_for_trades()` converts to per-month-mean:
   ```python
   mtm_1st_mean = mtm_1st * mcp_scale_back_factor / months_in_duration
   ```
   - For monthly period types (f0-f11): `months_in_duration = 1`, `mcp_scale_back_factor = 1` → `mtm_1st_mean = mtm_1st`
   - For annual: `months_in_duration = 12`, `mcp_scale_back_factor ≈ total_hours / first_month_hours`
   - **Key**: for f0 (June, current auction month), `mtm_1st_mean` = the raw June auction MCP for that path. For f1-f11, it uses the most recent clearing horizon's MCP, amortized per month.

3. **pmodel** (`trade_finalizer.py:49`): `add_price_related_columns()` calls `get_m2m_mcp_for_trades_all()` to populate `mtm_1st_mean` on the trades DataFrame. Then sets `mcp_pred = mtm_1st_mean`.

4. **Training data**: `generate_training_data_pjm.py` runs Steps 1-5 of the pipeline, which populates `mtm_1st_mean` via the above chain, then saves to parquet at `/opt/temp/qianli/pjm_mcp_pred_training2`.

5. **Banding**: `band_generator.py` reads training parquet → calls `prepare_features()` → calls `compute_baseline_v2()` which uses `mtm_1st_mean` as input.

### What the MW factor does

For PJM's **June first-monthly auction** (the auction that covers f0=Jun through f11=May), the current prediction uses `annual_mcp / 12` as `mtm_1st_mean` for each month. The MW factor corrects this flat distribution to match actual seasonal patterns:

```python
adjusted_mtm_1st = mtm_1st_mean * MW_ADJ_RATIOS[period_type]
```

---

## The Change

### Factor table (hardcoded constants, trained on PY2017-2025)

```python
# Market-wide seasonal adjustment ratios for PJM June first-monthly auction.
# Computed as: median(fx_path_mcp / annual_path_mcp) * 12, expanding window.
# Sum of underlying factors = 0.88, NOT normalized to 1.0 (by design).
MW_SEASONAL_ADJ = {
    "f0":  0.7846,  # Jun — most aggressive correction
    "f1":  0.8952,  # Jul
    "f2":  0.8914,  # Aug
    "f3":  0.9424,  # Sep
    "f4":  0.9798,  # Oct — closest to 1.0
    "f5":  0.8947,  # Nov
    "f6":  0.9269,  # Dec
    "f7":  0.9709,  # Jan
    "f8":  0.8894,  # Feb
    "f9":  0.8074,  # Mar
    "f10": 0.7853,  # Apr
    "f11": 0.7879,  # May
}
```

### Where to apply it

**In `prepare_features()`**, before `compute_baseline_v2()` is called. The banding pipeline flow is:

```
band_generator.py:595-601:
    if use_v2:
        df = compute_baseline_v2(df, ptype)     # ← mtm_1st_mean consumed here
    else:
        df = compute_baseline(df, ptype)
    df = compute_residual(df)
    df = compute_segment(df)                     # ← |mtm_1st_mean| < threshold
    df = compute_derived_features(df)            # ← feat_mtm_horizon_vol, feat_mtm_trend
```

The adjustment should happen **before** line 598 so that `compute_baseline_v2()`, `compute_segment()`, and `compute_derived_features()` all see the corrected value.

### Scope guard: June-only

This adjustment is **only valid for the June first-monthly auction**. You must guard on `auction_month`:

```python
def is_june_auction(df):
    """Check if this batch is from a June auction month."""
    if "auction_month" not in df.columns:
        return False
    months = df["auction_month"].unique()
    return len(months) == 1 and str(months[0]).endswith("-06")
```

Non-June months should NOT be adjusted.

---

## Implementation: Monkey-Patch Approach

Following the pattern from `v2/baseline_patch.py` and `v4/feature_patch.py`:

### `v_new/seasonal_patch.py`

```python
"""MW seasonal factor patch for PJM June first-monthly auction.

Adjusts mtm_1st_mean by period-type-specific seasonal factors before
baseline computation. Only applies when the data is from a June auction.

Research: research-first-monthly-pjm/v2/REPORT.md
Impact: +7.9% MAE improvement on mtm_1st_mean, +1.6% on V2 baseline (trades)

Usage:
    from seasonal_patch import apply_seasonal_patch
    apply_seasonal_patch()  # Call BEFORE generate_bands()
"""
import logging
import pandas as pd

logger = logging.getLogger(__name__)

MW_SEASONAL_ADJ = {
    "f0": 0.7846, "f1": 0.8952, "f2": 0.8914, "f3": 0.9424,
    "f4": 0.9798, "f5": 0.8947, "f6": 0.9269, "f7": 0.9709,
    "f8": 0.8894, "f9": 0.8074, "f10": 0.7853, "f11": 0.7879,
}


def adjust_mtm_seasonal(df: pd.DataFrame) -> pd.DataFrame:
    """Adjust mtm_1st_mean by MW seasonal factor, keyed on period_type.

    Only applies to June auction months. Returns df unchanged otherwise.
    """
    if "period_type" not in df.columns or "mtm_1st_mean" not in df.columns:
        return df

    # Guard: only adjust June auctions
    if "auction_month" in df.columns:
        months = df["auction_month"].unique()
        if not (len(months) == 1 and str(months[0]).endswith("-06")):
            return df

    df = df.copy()
    for ptype, ratio in MW_SEASONAL_ADJ.items():
        mask = df["period_type"] == ptype
        if mask.any():
            df.loc[mask, "mtm_1st_mean"] *= ratio
    return df


def apply_seasonal_patch() -> None:
    """Monkey-patch prepare_features to apply seasonal adjustment."""
    import pmodel.base.ftr24.v1.band_generator as bg

    original_prepare = bg.prepare_features

    def prepare_features_seasonal(df, ptype, band_version="v1"):
        df = adjust_mtm_seasonal(df)
        return original_prepare(df, ptype, band_version)

    bg.prepare_features = prepare_features_seasonal
    logger.info("Seasonal patch applied: mtm_1st_mean adjusted by MW factors for June auctions")
```

### `v_new/run_benchmark.py` (skeleton)

```python
"""Run benchmark with MW seasonal patch applied."""
import os
os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"

from pbase.config.ray import init_ray
import pmodel, lightgbm as lgb
init_ray(extra_modules=[pmodel, lgb])

# Apply patches: V2 baseline + V4 features + seasonal adjustment
from pjm.v2.baseline_patch import apply_baseline_patch
from pjm.v4.feature_patch import apply_feature_patch
from seasonal_patch import apply_seasonal_patch

apply_baseline_patch()
apply_feature_patch()
apply_seasonal_patch()

# Then run benchmark as usual (same as v4/run_benchmark.py)
```

---

## Cascading Effects (What to Watch For)

### 1. Segment threshold — LOW RISK
`compute_segment()` splits on `|mtm_1st_mean| < SEGMENT_THRESHOLD`. Scaling mtm_1st down by 2-22% could shift some borderline paths from "high" to "low". Since most paths are well above/below threshold, the practical impact is small. Check segment distribution in benchmark output.

### 2. LightGBM features — MEDIUM RISK (but likely helpful)
- `feat_mtm_horizon_vol = |mtm_1st_mean - mtm_2nd_mean|`: mtm_1st is now scaled but mtm_2nd is NOT. This changes the feature semantics — it now captures both horizon spread AND seasonal adjustment. Could help (seasonal-corrected mtm_1st is closer to truth, so the vol captures genuine horizon uncertainty) or hurt (introduces artificial spread).
- `feat_mtm_trend = mtm_1st_mean - mtm_3rd_mean`: same issue.
- The LightGBM model will retrain on the adjusted data, so it should adapt.

### 3. Flow type assignment — NO RISK
`prevail if mtm_1st_mean > 0 else counter`. Scaling preserves sign (all ratios > 0). No flow type changes.

### 4. Conformal calibration — NO RISK
V4 strata use `srev_bucket × flow_type`. Neither changes.

### 5. `mcp_pred` and `bought_price` — IMPORTANT
In `trade_finalizer.py:61`: `trades["mcp_pred"] = trades["mtm_1st_mean"]`. In the banding pipeline, the patch adjusts mtm_1st_mean INSIDE `prepare_features()`, but `mcp_pred` is set earlier in `add_price_related_columns()` (Step 3 of the pipeline). So the production pipeline's `mcp_pred` is NOT affected by the banding patch. The adjustment only flows through the banding baseline and features. This is correct — you want the banding model's internal baseline to use the adjusted value, but the final trade pricing uses the original.

For pool data (loaded from parquet), `mtm_1st_mean` IS the stored value, and the patch adjusts it in the DataFrame before banding. There's no separate `mcp_pred` computation for pool data.

---

## Size of the Change

This is a **small change**:
- 1 new file: `seasonal_patch.py` (~50 lines)
- 1 config dict: 12 hardcoded floats
- 1 wrapper function: wraps `prepare_features()` with a single `adjust_mtm_seasonal()` call before delegation
- No changes to pmodel, pbase, training data generation, or any existing code
- Follows exact same monkey-patch pattern as V2 baseline and V4 features

The benchmark rerun is the main work — the code change is trivial.

---

## Validation Checklist

1. [ ] Run pool benchmark (f0 onpeak, all available months)
2. [ ] Run trades benchmark (f0 onpeak)
3. [ ] Compare baseline MAE vs V4/V14 (expect ~1-2% improvement)
4. [ ] Compare band calibration (clearing probability accuracy should be similar or better)
5. [ ] Check segment distribution shift (count of low vs high paths)
6. [ ] Verify June-only guard works (non-June months should show identical results to V4)
7. [ ] Check LightGBM feature importance — do mtm_horizon_vol and mtm_trend shift?

---

## Research Reference

Full research at: `/home/xyz/workspace/research-qianli-v2/research-first-monthly-pjm/`
- `SUMMARY.md` — complete findings across V1 and V2
- `v2/REPORT.md` — detailed V2 results with all statistical tests
- `v2/scripts/07_baseline_impact.py` — end-to-end baseline impact measurement
- `v2/scripts/08_followup_analysis.py` — statistical significance proof
