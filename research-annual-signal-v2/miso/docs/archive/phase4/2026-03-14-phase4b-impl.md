# Phase 4b: Value-Aware Track B Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build value-aware Track B with new features (constraint propagation, recency, shape) and dynamic slot allocation to reach NB12_SP@50 ≥ 10%.

**Architecture:** Three phases: (1) extract branch↔CID mapping from data_loader, (2) build new feature groups in `ml/features_trackb.py`, (3) run Experiment 1 (log1p(SP) regression + dynamic R). Experiments 2-3 only if Experiment 1 doesn't hit target.

**Tech Stack:** LightGBM regression, polars, numpy

**Spec:** `docs/superpowers/specs/2026-03-14-phase4b-value-aware-track-b-design.md`

---

## Chunk 1: Data Pipeline + Feature Engineering

### Task 1: Extract and cache branch↔CID mapping from data_loader

**Files:**
- Modify: `ml/data_loader.py` (add CID mapping extraction to `load_collapsed`)

The branch↔CID mapping exists at Step 7 of `load_collapsed()` (`active_cids` DataFrame
with `constraint_id`, `branch_name`) but is discarded after Level 2 collapse. We need to
cache it as a separate artifact.

- [ ] **Step 1: Add CID mapping cache to `load_collapsed`**

Add a helper that extracts and caches the branch↔CID mapping alongside the existing
branch-level cache. The mapping is the `active_cids` DataFrame (constraint_id, branch_name)
plus the `is_active` flag and Level 1 density means.

In `ml/data_loader.py`, add a new function and modify `load_collapsed`:

```python
def _cid_mapping_cache_path(planning_year: str, aq_quarter: str) -> Path:
    """Cache path for branch↔CID mapping."""
    threshold_tag = f"{UNIVERSE_THRESHOLD:.6e}".replace(".", "p").replace("+", "")
    return COLLAPSED_CACHE_DIR / f"{planning_year}_{aq_quarter}_cid_map_t{threshold_tag}.parquet"


def load_cid_mapping(planning_year: str, aq_quarter: str) -> pl.DataFrame:
    """Load branch↔CID mapping for one (PY, quarter).

    Returns DataFrame with columns:
      constraint_id, branch_name, is_active

    This is the pre-aggregation CID-level data that load_collapsed() uses
    internally but doesn't expose. Cached alongside the branch-level cache.
    """
    cache_path = _cid_mapping_cache_path(planning_year, aq_quarter)
    if cache_path.exists():
        return pl.read_parquet(cache_path)

    # Force load_collapsed to run (which will also cache the cid mapping)
    load_collapsed(planning_year, aq_quarter)

    assert cache_path.exists(), f"CID mapping not cached after load_collapsed: {cache_path}"
    return pl.read_parquet(cache_path)
```

Then inside `load_collapsed()`, after Step 7 (where `cid_with_branch` is computed), add:

```python
    # Cache CID mapping for downstream use (Phase 4b constraint propagation)
    cid_map_path = _cid_mapping_cache_path(planning_year, aq_quarter)
    if not cid_map_path.exists():
        cid_with_branch.select(
            ["constraint_id", "branch_name", "is_active"]
        ).write_parquet(str(cid_map_path))
```

Insert this right after Step 7 (the bridge diagnostic log, around line 178), before Step 4.

- [ ] **Step 2: Test that CID mapping is produced and loadable**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. python3 -c "
from ml.data_loader import load_cid_mapping
m = load_cid_mapping('2022-06', 'aq1')
print(f'Shape: {m.shape}')
print(f'Columns: {m.columns}')
print(f'Unique branches: {m[\"branch_name\"].n_unique()}')
print(f'Unique CIDs: {m[\"constraint_id\"].n_unique()}')
print(f'Sample:')
print(m.head(5))
"`

Expected: DataFrame with constraint_id, branch_name, is_active columns. Multiple CIDs per branch.

- [ ] **Step 3: Commit**

```bash
git add ml/data_loader.py
git commit -m "phase4b: extract and cache branch-CID mapping from load_collapsed

Exposes the pre-aggregation CID-level data as load_cid_mapping().
Cached alongside branch-level cache in COLLAPSED_CACHE_DIR.
Required for Phase 4b constraint propagation features."
```

---

### Task 2: Build `ml/features_trackb.py` — constraint propagation + recency + shape features

**Files:**
- Create: `ml/features_trackb.py`
- Create: `tests/test_features_trackb.py`

- [ ] **Step 1: Create the feature module**

```python
"""Phase 4b Track B features: constraint propagation, recency, density shape.

These features are computed for dormant branches only (bf_combined_12 == 0).
They supplement the existing 12 density bin features from Phase 3.

Constraint propagation: for each dormant branch's CIDs, compute how actively
those CIDs bind on OTHER branches. "Cold branch, hot constraint" signal.

Recency: how recently the branch last bound, historical SP from older windows.

Shape: distribution statistics from density bins (entropy, skewness, etc.)
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from ml.data_loader import load_cid_mapping
from ml.config import SELECTED_BINS, get_bf_cutoff_month, BF_FLOOR_MONTH

logger = logging.getLogger(__name__)


def compute_constraint_propagation(
    planning_year: str,
    aq_quarter: str,
    monthly_binding: pl.DataFrame,
    dormant_branches: list[str],
    cutoff_month: str,
) -> pl.DataFrame:
    """Compute constraint propagation features for dormant branches.

    For each dormant branch, look up its CIDs, find all OTHER branches
    containing those CIDs, and aggregate their binding activity.

    Args:
        planning_year: eval PY
        aq_quarter: eval quarter
        monthly_binding: monthly binding table from compute_history_features
            (columns: month, branch_name, combined_bound, combined_sp, ...)
        dormant_branches: list of dormant branch names
        cutoff_month: BF cutoff month (only use months <= this)

    Returns:
        DataFrame with columns: branch_name, max_cid_bf_other, mean_cid_bf_other,
        sum_cid_sp_other, max_cid_sp_other, n_active_cids_other, active_cid_ratio_other
    """
    # Load branch↔CID mapping
    cid_map = load_cid_mapping(planning_year, aq_quarter)

    # Get CIDs for dormant branches
    dormant_cids = cid_map.filter(pl.col("branch_name").is_in(dormant_branches))

    if len(dormant_cids) == 0:
        return _empty_constraint_propagation(dormant_branches)

    # Get CIDs for ALL branches (for cross-reference)
    # For each CID, compute binding stats on NON-dormant branches
    all_cid_branch = cid_map.select(["constraint_id", "branch_name"])

    # Monthly binding within window
    binding_in_window = monthly_binding.filter(
        pl.col("month") <= cutoff_month
    )

    if len(binding_in_window) == 0:
        return _empty_constraint_propagation(dormant_branches)

    # Per-branch binding stats (BF and total SP)
    branch_stats = binding_in_window.group_by("branch_name").agg(
        (pl.col("combined_bound").sum() / pl.col("combined_bound").len()).alias("branch_bf"),
        pl.col("combined_sp").sum().alias("branch_total_sp"),
    )

    # Join CID mapping to branch stats: for each CID, get stats of ALL branches it appears on
    cid_branch_stats = all_cid_branch.join(branch_stats, on="branch_name", how="left")
    cid_branch_stats = cid_branch_stats.with_columns(
        pl.col("branch_bf").fill_null(0.0),
        pl.col("branch_total_sp").fill_null(0.0),
    )

    # For each dormant branch's CIDs, get OTHER branches' stats
    # Join dormant CIDs to all branch stats for those CIDs
    dormant_cid_list = dormant_cids.select("constraint_id").unique()
    other_stats = (
        cid_branch_stats
        .join(dormant_cid_list, on="constraint_id", how="inner")
        # Exclude dormant branches themselves
        .filter(~pl.col("branch_name").is_in(dormant_branches))
    )

    if len(other_stats) == 0:
        return _empty_constraint_propagation(dormant_branches)

    # Per-CID: aggregate other-branch stats
    per_cid = other_stats.group_by("constraint_id").agg(
        pl.col("branch_bf").max().alias("cid_max_bf_other"),
        pl.col("branch_bf").mean().alias("cid_mean_bf_other"),
        pl.col("branch_total_sp").sum().alias("cid_sum_sp_other"),
        pl.col("branch_total_sp").max().alias("cid_max_sp_other"),
        (pl.col("branch_bf") > 0).any().cast(pl.Int64).alias("cid_has_active_other"),
    )

    # Join back to dormant CIDs and aggregate to branch level
    dormant_with_cid_stats = dormant_cids.join(per_cid, on="constraint_id", how="left")
    dormant_with_cid_stats = dormant_with_cid_stats.with_columns(
        pl.col("cid_max_bf_other").fill_null(0.0),
        pl.col("cid_mean_bf_other").fill_null(0.0),
        pl.col("cid_sum_sp_other").fill_null(0.0),
        pl.col("cid_max_sp_other").fill_null(0.0),
        pl.col("cid_has_active_other").fill_null(0),
    )

    # Branch-level aggregation
    result = dormant_with_cid_stats.group_by("branch_name").agg(
        pl.col("cid_max_bf_other").max().alias("max_cid_bf_other"),
        pl.col("cid_mean_bf_other").mean().alias("mean_cid_bf_other"),
        pl.col("cid_sum_sp_other").sum().alias("sum_cid_sp_other"),
        pl.col("cid_max_sp_other").max().alias("max_cid_sp_other"),
        pl.col("cid_has_active_other").sum().cast(pl.Int64).alias("n_active_cids_other"),
    )

    # Add count_cids for ratio
    cid_counts = dormant_cids.group_by("branch_name").agg(pl.len().alias("_n_cids"))
    result = result.join(cid_counts, on="branch_name", how="left")
    result = result.with_columns(
        (pl.col("n_active_cids_other") / pl.col("_n_cids").cast(pl.Float64)).alias("active_cid_ratio_other")
    ).drop("_n_cids")

    # Ensure all dormant branches are present (left join with zeros)
    all_dormant = pl.DataFrame({"branch_name": dormant_branches})
    result = all_dormant.join(result, on="branch_name", how="left")
    for col in result.columns:
        if col != "branch_name":
            result = result.with_columns(pl.col(col).fill_null(0.0))

    return result


def _empty_constraint_propagation(dormant_branches: list[str]) -> pl.DataFrame:
    """Return zero-filled constraint propagation features."""
    return pl.DataFrame({
        "branch_name": dormant_branches,
        "max_cid_bf_other": [0.0] * len(dormant_branches),
        "mean_cid_bf_other": [0.0] * len(dormant_branches),
        "sum_cid_sp_other": [0.0] * len(dormant_branches),
        "max_cid_sp_other": [0.0] * len(dormant_branches),
        "n_active_cids_other": [0.0] * len(dormant_branches),
        "active_cid_ratio_other": [0.0] * len(dormant_branches),
    })


def compute_recency_features(
    monthly_binding: pl.DataFrame,
    dormant_branches: list[str],
    cutoff_month: str,
) -> pl.DataFrame:
    """Compute recency features for dormant branches.

    months_since_last_bind: months from cutoff to most recent binding
    historical_max_sp: peak single-month SP from any period
    historical_sp_12_24m: total SP in months 13-24 before cutoff
    historical_sp_24_36m: total SP in months 25-36 before cutoff
    n_historical_binding_months: count of months where branch had SP > 0
    """
    # Filter to dormant branches
    dormant_binding = monthly_binding.filter(
        (pl.col("branch_name").is_in(dormant_branches))
        & (pl.col("month") <= cutoff_month)
    )

    if len(dormant_binding) == 0:
        return _empty_recency(dormant_branches)

    # Parse cutoff month for arithmetic
    cutoff_y = int(cutoff_month[:4])
    cutoff_m = int(cutoff_month[5:7])

    def _months_diff(month_str: str) -> int:
        y, m = int(month_str[:4]), int(month_str[5:7])
        return (cutoff_y - y) * 12 + (cutoff_m - m)

    dormant_binding = dormant_binding.with_columns(
        pl.col("month").map_elements(
            _months_diff, return_dtype=pl.Int64
        ).alias("months_ago")
    )

    # Per-branch aggregation
    result = dormant_binding.group_by("branch_name").agg(
        # months_since_last_bind: min months_ago where combined_bound == True
        pl.col("months_ago").filter(pl.col("combined_bound")).min().alias("months_since_last_bind"),
        # historical_max_sp: peak single-month combined_sp
        pl.col("combined_sp").max().alias("historical_max_sp"),
        # Count of binding months
        pl.col("combined_bound").sum().alias("n_historical_binding_months"),
    )

    # historical_sp_12_24m and 24_36m
    sp_12_24 = (
        dormant_binding
        .filter((pl.col("months_ago") >= 12) & (pl.col("months_ago") < 24))
        .group_by("branch_name")
        .agg(pl.col("combined_sp").sum().alias("historical_sp_12_24m"))
    )
    sp_24_36 = (
        dormant_binding
        .filter((pl.col("months_ago") >= 24) & (pl.col("months_ago") < 36))
        .group_by("branch_name")
        .agg(pl.col("combined_sp").sum().alias("historical_sp_24_36m"))
    )

    result = result.join(sp_12_24, on="branch_name", how="left")
    result = result.join(sp_24_36, on="branch_name", how="left")

    # Fill nulls and ensure all dormant branches present
    all_dormant = pl.DataFrame({"branch_name": dormant_branches})
    result = all_dormant.join(result, on="branch_name", how="left")

    result = result.with_columns(
        pl.col("months_since_last_bind").fill_null(999),
        pl.col("historical_max_sp").fill_null(0.0),
        pl.col("n_historical_binding_months").fill_null(0),
        pl.col("historical_sp_12_24m").fill_null(0.0),
        pl.col("historical_sp_24_36m").fill_null(0.0),
    )

    return result


def _empty_recency(dormant_branches: list[str]) -> pl.DataFrame:
    return pl.DataFrame({
        "branch_name": dormant_branches,
        "months_since_last_bind": [999] * len(dormant_branches),
        "historical_max_sp": [0.0] * len(dormant_branches),
        "n_historical_binding_months": [0] * len(dormant_branches),
        "historical_sp_12_24m": [0.0] * len(dormant_branches),
        "historical_sp_24_36m": [0.0] * len(dormant_branches),
    })


def compute_density_shape(
    collapsed: pl.DataFrame,
    dormant_branches: list[str],
) -> pl.DataFrame:
    """Compute density shape features for dormant branches.

    Uses the existing bin_*_cid_max columns from the collapsed model table.
    Note: bins are NOT calibrated probabilities — they are raw simulation
    counts/scores. These features capture distribution shape, not tail probability.
    """
    dormant = collapsed.filter(pl.col("branch_name").is_in(dormant_branches))

    # Get the cid_max bin columns (these are the primary signal columns)
    bin_max_cols = [c for c in dormant.columns if c.endswith("_cid_max") and c.startswith("bin_")]

    if len(dormant) == 0 or len(bin_max_cols) == 0:
        return _empty_shape(dormant_branches)

    # Collect bin values as arrays for shape computation
    results = []
    for row in dormant.iter_rows(named=True):
        vals = np.array([row[c] for c in bin_max_cols], dtype=np.float64)
        vals_pos = vals[vals > 0] if (vals > 0).any() else vals

        # Tail sums (bins with numeric part >= 100, 110)
        tail_100_cols = [c for c in bin_max_cols if _bin_num(c) >= 100]
        tail_110_cols = [c for c in bin_max_cols if _bin_num(c) >= 110]
        tail_sum_100 = sum(row[c] for c in tail_100_cols)
        tail_sum_110 = sum(row[c] for c in tail_110_cols)

        # Shape stats
        total = vals.sum()
        if total > 0:
            probs = vals / total
            entropy = -float(np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
        else:
            entropy = 0.0

        mean_v = vals.mean()
        std_v = vals.std()
        cv = float(std_v / mean_v) if mean_v > 0 else 0.0

        if len(vals) >= 3 and std_v > 0:
            skew = float(np.mean(((vals - mean_v) / std_v) ** 3))
            kurt = float(np.mean(((vals - mean_v) / std_v) ** 4)) - 3.0
        else:
            skew = 0.0
            kurt = 0.0

        results.append({
            "branch_name": row["branch_name"],
            "tail_sum_ge_100": tail_sum_100,
            "tail_sum_ge_110": tail_sum_110,
            "density_entropy": entropy,
            "density_skewness": skew,
            "density_kurtosis": kurt,
            "density_cv": cv,
        })

    return pl.DataFrame(results)


def _bin_num(col_name: str) -> float:
    """Extract numeric bin value from column name like 'bin_100_cid_max'."""
    # Format: bin_{num}_cid_max where num can be negative (e.g., bin_-50_cid_max)
    parts = col_name.replace("bin_", "").replace("_cid_max", "")
    try:
        return float(parts)
    except ValueError:
        return 0.0


def _empty_shape(dormant_branches: list[str]) -> pl.DataFrame:
    return pl.DataFrame({
        "branch_name": dormant_branches,
        "tail_sum_ge_100": [0.0] * len(dormant_branches),
        "tail_sum_ge_110": [0.0] * len(dormant_branches),
        "density_entropy": [0.0] * len(dormant_branches),
        "density_skewness": [0.0] * len(dormant_branches),
        "density_kurtosis": [0.0] * len(dormant_branches),
        "density_cv": [0.0] * len(dormant_branches),
    })


# ── Feature list constants ──────────────────────────────────────────────

CONSTRAINT_PROP_FEATURES = [
    "max_cid_bf_other", "mean_cid_bf_other",
    "sum_cid_sp_other", "max_cid_sp_other",
    "n_active_cids_other", "active_cid_ratio_other",
]

RECENCY_FEATURES = [
    "months_since_last_bind", "historical_max_sp",
    "n_historical_binding_months",
    "historical_sp_12_24m", "historical_sp_24_36m",
]

SHAPE_FEATURES = [
    "tail_sum_ge_100", "tail_sum_ge_110",
    "density_entropy", "density_skewness",
    "density_kurtosis", "density_cv",
]

PHASE4B_TRACK_B_FEATURES = (
    # Phase 3 features (from selected_features.json)
    ["count_cids", "bin_80_cid_max", "bin_70_cid_max", "bin_90_cid_max",
     "count_active_cids", "bin_60_cid_max", "bin_100_cid_max", "bin_110_cid_max",
     "bin_-50_cid_max", "bin_120_cid_max", "bin_-100_cid_max", "bin_150_cid_max"]
    + CONSTRAINT_PROP_FEATURES
    + RECENCY_FEATURES
    + SHAPE_FEATURES
)
```

- [ ] **Step 2: Clear collapsed cache so CID mapping gets generated**

Run: `rm -f /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/data/collapsed_cache/*.parquet`

This forces `load_collapsed()` to re-run and produce the CID mapping cache files.

- [ ] **Step 3: Smoke test the feature module**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. python3 -c "
from ml.features import build_model_table
from ml.features_trackb import (
    compute_constraint_propagation, compute_recency_features,
    compute_density_shape, PHASE4B_TRACK_B_FEATURES,
)
from ml.config import get_bf_cutoff_month
from ml.history_features import compute_history_features

mt = build_model_table('2022-06', 'aq1')
dormant = mt.filter(mt['cohort'] == 'history_dormant')['branch_name'].to_list()
print(f'Dormant branches: {len(dormant)}')

# Constraint propagation
cutoff = get_bf_cutoff_month('2022-06')
_, monthly_binding = compute_history_features('2022-06', 'aq1', mt['branch_name'].to_list())
cp = compute_constraint_propagation('2022-06', 'aq1', monthly_binding, dormant, cutoff)
print(f'Constraint prop shape: {cp.shape}')
print(cp.describe())

# Recency
rec = compute_recency_features(monthly_binding, dormant, cutoff)
print(f'Recency shape: {rec.shape}')
print(rec.describe())

# Shape
sh = compute_density_shape(mt, dormant)
print(f'Shape shape: {sh.shape}')
print(sh.describe())

print(f'Total Phase4b features: {len(PHASE4B_TRACK_B_FEATURES)}')
"`

- [ ] **Step 4: Run existing tests to verify no regressions**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/ -q`
Expected: All 103 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ml/features_trackb.py ml/data_loader.py
git commit -m "phase4b: add Track B features — constraint propagation, recency, shape

Constraint propagation: cross-branch CID binding signal (cold branch, hot constraint).
Recency: months_since_last_bind, historical SP windows.
Shape: density entropy, skewness, kurtosis, CV, tail sums.
33 total features (12 Phase 3 + 6 constraint + 5 recency + 6 shape + 4 aggregation)."
```

---

## Chunk 2: Experiment 1 — Regression + Dynamic R

### Task 3: Add tau support to merge_tracks

**Files:**
- Modify: `ml/merge.py`
- Modify: `tests/test_merge.py`

- [ ] **Step 1: Add tau parameter to merge_tracks**

```python
# In ml/merge.py, change the function signature and add tau logic:

def merge_tracks(
    track_a: pl.DataFrame,
    track_b: pl.DataFrame,
    k: int,
    r: int,
    tau: float | None = None,
) -> tuple[pl.DataFrame, np.ndarray]:
    # ... existing docstring, add tau description ...

    a = track_a.with_columns(pl.lit("A").alias("track"))
    b = track_b.with_columns(pl.lit("B").alias("track"))

    merged = pl.concat([a, b], how="diagonal")

    n_a = len(track_a)
    n_b = len(track_b)

    # Score-thresholded R: only include Track B branches with score >= tau
    if tau is not None:
        b_scores_all = track_b["score"].to_numpy().astype(np.float64)
        qualified_mask = b_scores_all >= tau
        r_actual = min(r, int(qualified_mask.sum()))
    else:
        r_actual = min(r, n_b)

    n_a_slots = min(k - r_actual, n_a)

    # Top Track A indices
    a_scores = track_a["score"].to_numpy().astype(np.float64)
    a_order = np.argsort(a_scores)[::-1][:n_a_slots]

    # Top Track B indices (only qualified if tau set)
    b_scores = track_b["score"].to_numpy().astype(np.float64)
    if tau is not None:
        # Sort only qualified branches, take top r_actual
        qualified_indices = np.where(qualified_mask)[0]
        qualified_scores = b_scores[qualified_indices]
        top_qualified = qualified_indices[np.argsort(qualified_scores)[::-1][:r_actual]]
        b_order_merged = top_qualified + n_a
    else:
        b_order = np.argsort(b_scores)[::-1][:r_actual]
        b_order_merged = b_order + n_a

    top_k_indices = np.concatenate([a_order, b_order_merged])

    return merged, top_k_indices
```

- [ ] **Step 2: Add test for tau**

Add to `tests/test_merge.py`:

```python
def test_merge_tracks_tau_filters():
    """tau parameter filters Track B to only include branches with score >= tau."""
    from ml.merge import merge_tracks

    a = pl.DataFrame({"score": [10.0, 9.0, 8.0, 7.0, 6.0],
                       "branch_name": ["a1","a2","a3","a4","a5"],
                       "realized_shadow_price": [100.0]*5})
    b = pl.DataFrame({"score": [0.8, 0.3, 0.1, 0.05, 0.02],
                       "branch_name": ["b1","b2","b3","b4","b5"],
                       "realized_shadow_price": [50.0]*5})

    # Without tau: r=3 -> top 3 from Track B
    _, idx_no_tau = merge_tracks(a, b, k=5, r=3, tau=None)
    assert len(idx_no_tau) == 5  # 2 from A + 3 from B

    # With tau=0.2: only b1(0.8) and b2(0.3) qualify -> r_actual=2
    _, idx_tau = merge_tracks(a, b, k=5, r=3, tau=0.2)
    assert len(idx_tau) == 5  # 3 from A + 2 from B (only 2 qualified)

    # With tau=0.9: only b1(0.8) doesn't qualify either -> r_actual=0
    _, idx_high_tau = merge_tracks(a, b, k=5, r=3, tau=0.9)
    assert len(idx_high_tau) == 5  # all 5 from A
```

- [ ] **Step 3: Run merge tests**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/test_merge.py -v`
Expected: All PASS (5 existing + 1 new = 6).

- [ ] **Step 4: Commit**

```bash
git add ml/merge.py tests/test_merge.py
git commit -m "phase4b: add tau parameter to merge_tracks for dynamic R

Score-thresholded allocation: only Track B branches with score >= tau
get reserved slots. Unused slots return to Track A."
```

---

### Task 4: Create `scripts/run_phase4b_regression.py` — Experiment 1

**Files:**
- Create: `scripts/run_phase4b_regression.py`

This is the main experiment script. It:
1. Builds model tables + Phase 4b features for dormant branches
2. Trains LightGBM regression on `log1p(realized_shadow_price)` for dormant branches
3. Sweeps tau on dev at K=50 and K=100
4. Reports all metrics with Phase 3/4a comparison
5. Holdout validation with registry save

- [ ] **Step 1: Create the experiment script**

The script follows the same structure as `run_phase4a_experiment.py` but with:
- LightGBM regression target instead of binary
- Phase 4b features (33 total)
- Dynamic R via tau parameter
- Dormant-only population (history_zero in universe with score=0)
- Tau sweep on dev: candidate thresholds from predicted log1p(SP) percentiles

(Full script code provided — see implementation below)

- [ ] **Step 2: Run dev sweep**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run python scripts/run_phase4b_regression.py 2>&1 | tee /tmp/phase4b_dev_sweep.txt`

Expected: Tables for each tau value at K=50 and K=100 with NB12_SP, VC, dormant/zero breakdown.

- [ ] **Step 3: Analyze and select winner**

From dev sweep: pick tau that maximizes NB12_SP@50 subject to VC@50 > v0c - 0.01.

- [ ] **Step 4: Run holdout validation**

Run with winning tau:
`PYTHONPATH=. uv run python scripts/run_phase4b_regression.py --holdout --tau50 <TAU50> --tau100 <TAU100> --r50 5 --r100 15 --version p4b_reg_<details>`

- [ ] **Step 5: Commit results**

```bash
git add scripts/run_phase4b_regression.py registry/p4b_*/
git commit -m "phase4b: Experiment 1 — log1p(SP) regression on dormant with dynamic R

<Results summary>"
```
