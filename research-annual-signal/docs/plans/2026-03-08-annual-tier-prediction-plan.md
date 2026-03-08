# Annual Constraint Tier Prediction — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an ML pipeline that predicts which MISO constraints will bind in each annual auction quarter (aq1-aq4), producing 5-tier rankings that beat the V6.1 formula baseline.

**Architecture:** V6.1 annual parquets define the constraint universe (rows). Ground truth is realized DA shadow prices fetched via Ray for the target quarter. LightGBM lambdarank learns to rank constraints using V6.1 features + spice6 density features. Evaluation uses VC@k, Recall@k, NDCG, Spearman with 3-layer promotion gates.

**Tech Stack:** Python, polars, LightGBM, numpy, scipy, Ray (for ground truth only)

**Spec:** `/home/xyz/workspace/research-qianli-v2/research-annual-signal/experiment-setup.md`

**Reuse from stage4:** `/home/xyz/workspace/research-qianli-v2/research-stage4-tier/ml/` (evaluate.py, compare.py, train.py copied verbatim; others adapted)

**Working directory:** `/home/xyz/workspace/research-qianli-v2/research-annual-signal/`

**Venv:** `cd /home/xyz/workspace/pmodel && source .venv/bin/activate`

---

### Task 1: Scaffold repo structure

**Files:**
- Create: `ml/__init__.py`
- Create: `scripts/.gitkeep`
- Create: `registry/.gitkeep`

**Step 1: Create directory structure**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-annual-signal
mkdir -p ml scripts registry
```

**Step 2: Create __init__.py**

```python
# ml/__init__.py
```

**Step 3: Commit**

```bash
git add ml/__init__.py scripts/ registry/ docs/
git commit -m "task 1: scaffold annual-signal repo structure"
```

---

### Task 2: Copy reusable modules from stage4

**Files:**
- Copy: `ml/evaluate.py` (verbatim from stage4)
- Copy: `ml/compare.py` (verbatim from stage4)
- Copy: `ml/train.py` (verbatim from stage4)
- Copy: `ml/features.py` (verbatim from stage4)

These modules have NO monthly-specific logic — they operate on numpy arrays and polars DataFrames with generic `query_month`/`query_group` columns.

**Step 1: Copy files**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-annual-signal
cp ../research-stage4-tier/ml/evaluate.py ml/evaluate.py
cp ../research-stage4-tier/ml/compare.py ml/compare.py
cp ../research-stage4-tier/ml/train.py ml/train.py
cp ../research-stage4-tier/ml/features.py ml/features.py
```

**Step 2: Verify imports work**

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-annual-signal
python -c "from ml.evaluate import evaluate_ltr; print('evaluate OK')"
python -c "from ml.train import train_ltr_model; print('train OK')"
```

**Step 3: Rename `query_month` to `query_group` in features.py**

In `ml/features.py`, rename the column reference from `query_month` to `query_group` for clarity (annual uses year_aq, not month). The `compute_query_groups` function computes group sizes from a sorted column — works identically regardless of column name.

```python
# ml/features.py — change line 37: months = df["query_month"].to_list()
# to: months = df["query_group"].to_list()
```

**Step 4: Commit**

```bash
git add ml/evaluate.py ml/compare.py ml/train.py ml/features.py
git commit -m "task 2: copy reusable modules from stage4 (evaluate, compare, train, features)"
```

---

### Task 3: Annual config module

**Files:**
- Create: `ml/config.py`

This replaces stage4's config.py with annual-specific paths, feature sets, eval groups, and the same LTRConfig/PipelineConfig/GateConfig dataclasses.

**Step 1: Write config.py**

```python
"""Annual LTR pipeline configuration.

Mirrors stage4 monthly config but adapted for annual auction structure:
- V6.1 annual signal (year/aq partitions) instead of V6.2B monthly
- Expanding-window train (all prior years) instead of rolling 8-month
- Eval groups = (planning_year, aq_round) instead of month
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# -- Data paths --
V61_SIGNAL_BASE = "/opt/data/xyz-dataset/signal_data/miso/constraints/Signal.MISO.SPICE_ANNUAL_V6.1"
SPICE_DATA_BASE = "/opt/data/xyz-dataset/spice_data/miso"

# -- Leakage guard --
_LEAKY_FEATURES: set[str] = {
    "rank", "rank_ori", "tier",
    "shadow_sign", "shadow_price",
    "da_rank_value",       # redundant with shadow_price_da (Spearman = -1.0)
    "density_mix_rank",    # integer version of density_mix_rank_value
    "mean_branch_max_fillna",  # redundant with mean_branch_max
}

# -- Feature Set A: V6.1 base (6 features) --
_V61_FEATURES: list[str] = [
    "shadow_price_da",          # historical DA shadow price (NOT realized)
    "mean_branch_max",          # max branch loading forecast
    "ori_mean",                 # mean flow, baseline scenario
    "mix_mean",                 # mean flow, mixed scenario
    "density_mix_rank_value",   # percentile rank of mix flow (lower = more binding)
    "density_ori_rank_value",   # percentile rank of ori flow (lower = more binding)
]
_V61_MONOTONE: list[int] = [1, 1, 1, 1, -1, -1]

# -- Feature Set B: V6.1 + spice6 density (11 features) --
_SPICE6_FEATURES: list[str] = [
    "prob_exceed_110",
    "prob_exceed_100",
    "prob_exceed_90",
    "prob_exceed_85",
    "prob_exceed_80",
]
_SPICE6_MONOTONE: list[int] = [1, 1, 1, 1, 1]

# -- Feature Set C: Full (13 features) --
_STRUCTURAL_FEATURES: list[str] = [
    "constraint_limit",
    "rate_a",
]
_STRUCTURAL_MONOTONE: list[int] = [0, 0]

# -- Composite feature lists --
SET_A_FEATURES = list(_V61_FEATURES)
SET_A_MONOTONE = list(_V61_MONOTONE)

SET_B_FEATURES = _V61_FEATURES + _SPICE6_FEATURES
SET_B_MONOTONE = _V61_MONOTONE + _SPICE6_MONOTONE

SET_C_FEATURES = _V61_FEATURES + _SPICE6_FEATURES + _STRUCTURAL_FEATURES
SET_C_MONOTONE = _V61_MONOTONE + _SPICE6_MONOTONE + _STRUCTURAL_MONOTONE

# -- Eval groups --
# Each group = (planning_year, aq_round) as "YYYY-06/aqN"
PLANNING_YEARS = [
    "2019-06", "2020-06", "2021-06", "2022-06",
    "2023-06", "2024-06", "2025-06",
]
AQ_ROUNDS = ["aq1", "aq2", "aq3", "aq4"]

# Quarter -> market months mapping
AQ_MARKET_MONTHS: dict[str, list[int]] = {
    "aq1": [6, 7, 8],    # Jun-Aug
    "aq2": [9, 10, 11],   # Sep-Nov
    "aq3": [12, 1, 2],    # Dec-Feb (crosses year boundary)
    "aq4": [3, 4, 5],     # Mar-May
}

# Eval splits (expanding window)
EVAL_SPLITS: dict[str, dict] = {
    "split1": {"train_years": ["2019-06", "2020-06", "2021-06"], "eval_year": "2022-06"},
    "split2": {"train_years": ["2019-06", "2020-06", "2021-06", "2022-06"], "eval_year": "2023-06"},
    "split3": {"train_years": ["2019-06", "2020-06", "2021-06", "2022-06", "2023-06"], "eval_year": "2024-06"},
}
# 2025-06 held out for final validation

SCREEN_EVAL_GROUPS: list[str] = [
    "2022-06/aq1", "2023-06/aq2", "2024-06/aq3", "2024-06/aq4",
]

DEFAULT_EVAL_GROUPS: list[str] = [
    f"{year}/{aq}"
    for year in ["2022-06", "2023-06", "2024-06"]
    for aq in AQ_ROUNDS
]

HOLDOUT_EVAL_GROUPS: list[str] = [
    f"2025-06/{aq}" for aq in AQ_ROUNDS
]


def get_market_months(planning_year: str, aq_round: str) -> list[str]:
    """Return YYYY-MM strings for the 3 market months in a quarter.

    Example: get_market_months("2022-06", "aq1") -> ["2022-06", "2022-07", "2022-08"]
    """
    base_year = int(planning_year.split("-")[0])
    months = AQ_MARKET_MONTHS[aq_round]
    result = []
    for m in months:
        if aq_round == "aq3" and m in (1, 2):
            year = base_year + 1
        elif aq_round == "aq4":
            year = base_year + 1
        else:
            year = base_year
        result.append(f"{year:04d}-{m:02d}")
    return result


@dataclass
class LTRConfig:
    """Learning-to-rank configuration."""
    features: list[str] = field(default_factory=lambda: list(SET_B_FEATURES))
    monotone_constraints: list[int] = field(default_factory=lambda: list(SET_B_MONOTONE))
    backend: str = "lightgbm"
    n_estimators: int = 100
    learning_rate: float = 0.05
    min_child_weight: int = 25
    num_leaves: int = 31
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 1.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 20

    def __post_init__(self) -> None:
        if len(self.monotone_constraints) != len(self.features):
            raise ValueError(
                f"len(monotone) != len(features): "
                f"{len(self.monotone_constraints)} != {len(self.features)}"
            )
        filtered_features = []
        filtered_mono = []
        removed = []
        for feat, mono in zip(self.features, self.monotone_constraints):
            if feat in _LEAKY_FEATURES:
                removed.append(feat)
                continue
            filtered_features.append(feat)
            filtered_mono.append(mono)
        if removed:
            print(f"[config] WARNING: dropped leaky features: {sorted(set(removed))}")
            self.features = filtered_features
            self.monotone_constraints = filtered_mono

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "monotone_constraints": list(self.monotone_constraints),
            "backend": self.backend,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "num_leaves": self.num_leaves,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LTRConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    ltr: LTRConfig = field(default_factory=LTRConfig)

    def to_dict(self) -> dict[str, Any]:
        return {"ltr": self.ltr.to_dict()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineConfig:
        return cls(ltr=LTRConfig.from_dict(d["ltr"]))


@dataclass
class GateConfig:
    """Quality gates loaded from JSON."""
    gates: dict[str, Any] = field(default_factory=dict)
    noise_tolerance: float = 0.0
    tail_max_failures: int = 0

    @classmethod
    def from_json(cls, path: str | Path) -> GateConfig:
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text())
        return cls(
            gates=data.get("gates", {}),
            noise_tolerance=data.get("noise_tolerance", 0.0),
            tail_max_failures=data.get("tail_max_failures", 0),
        )
```

**Step 2: Test config**

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-annual-signal
python -c "
from ml.config import get_market_months, LTRConfig, SET_A_FEATURES, SET_B_FEATURES
print('Set A:', SET_A_FEATURES)
print('Set B:', SET_B_FEATURES)
print('aq1 2022:', get_market_months('2022-06', 'aq1'))
print('aq3 2022:', get_market_months('2022-06', 'aq3'))
print('aq4 2022:', get_market_months('2022-06', 'aq4'))
cfg = LTRConfig()
print('Default features:', cfg.features)
"
```

Expected output should show:
- aq1 2022 -> ["2022-06", "2022-07", "2022-08"]
- aq3 2022 -> ["2022-12", "2023-01", "2023-02"] (crosses year boundary)
- aq4 2022 -> ["2023-03", "2023-04", "2023-05"]

**Step 3: Commit**

```bash
git add ml/config.py
git commit -m "task 3: annual config module (paths, feature sets, eval groups)"
```

---

### Task 4: V6.1 data loader

**Files:**
- Create: `ml/data_loader.py`

Loads V6.1 annual parquets and enriches with spice6 density features (aggregated across 3 market months per quarter).

**Step 1: Write data_loader.py**

```python
"""Data loading for annual LTR ranking pipeline.

Loads V6.1 annual signal and enriches with spice6 density features
aggregated across the 3 market months in each quarter.
"""
from __future__ import annotations

import gc
import resource
from pathlib import Path

import polars as pl

from ml.config import (
    V61_SIGNAL_BASE,
    SPICE_DATA_BASE,
    get_market_months,
)


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_v61_group(planning_year: str, aq_round: str) -> pl.DataFrame:
    """Load V6.1 annual signal for one (planning_year, aq_round).

    Parameters
    ----------
    planning_year : str
        Planning year in YYYY-06 format.
    aq_round : str
        Auction quarter (aq1, aq2, aq3, aq4).

    Returns
    -------
    pl.DataFrame
        V6.1 data with query_group column added.
    """
    path = Path(V61_SIGNAL_BASE) / planning_year / aq_round / "onpeak"
    if not path.exists():
        raise FileNotFoundError(f"V6.1 data not found: {path}")
    df = pl.read_parquet(str(path))
    if "__index_level_0__" in df.columns:
        df = df.drop("__index_level_0__")

    query_group = f"{planning_year}/{aq_round}"
    df = df.with_columns(pl.lit(query_group).alias("query_group"))
    return df


def load_spice6_density_annual(
    planning_year: str,
    aq_round: str,
) -> pl.DataFrame:
    """Load and aggregate spice6 density features for a quarter (3 months).

    Aggregates across all outage_dates AND all 3 market_months in the quarter.
    Returns one row per (constraint_id, flow_direction).
    """
    market_months = get_market_months(planning_year, aq_round)
    density_score_path = Path(SPICE_DATA_BASE) / "MISO_SPICE_DENSITY_SIGNAL_SCORE.parquet"
    constraint_limit_path = Path(SPICE_DATA_BASE) / "MISO_SPICE_CONSTRAINT_LIMIT.parquet"

    # Load density scores filtered to annual auction_type and these market months
    scores = (
        pl.scan_parquet(str(density_score_path))
        .filter(
            (pl.col("auction_type") == "annual")
            & (pl.col("auction_month") == planning_year)
            & (pl.col("market_month").is_in(market_months))
        )
        .collect()
    )

    if len(scores) == 0:
        return pl.DataFrame()

    # Detect exceedance columns (numeric columns like "80", "85", "90", "100", "110")
    exceed_cols = [c for c in scores.columns if c.isdigit() and int(c) >= 80]

    density = scores.group_by(["constraint_id", "flow_direction"]).agg([
        pl.col(c).mean().alias(f"prob_exceed_{c}") for c in exceed_cols
    ])

    # Load constraint limits
    if constraint_limit_path.exists():
        limits = (
            pl.scan_parquet(str(constraint_limit_path))
            .filter(
                (pl.col("auction_type") == "annual")
                & (pl.col("auction_month") == planning_year)
                & (pl.col("market_month").is_in(market_months))
            )
            .collect()
        )
        if len(limits) > 0:
            limit_agg = limits.group_by("constraint_id").agg(
                pl.col("limit").mean().alias("constraint_limit")
            )
            density = density.join(limit_agg, on="constraint_id", how="left")

    if "constraint_limit" not in density.columns:
        density = density.with_columns(pl.lit(0.0).alias("constraint_limit"))

    return density


def load_v61_enriched(planning_year: str, aq_round: str) -> pl.DataFrame:
    """Load V6.1 data enriched with spice6 density features.

    Parameters
    ----------
    planning_year : str
        Planning year in YYYY-06 format.
    aq_round : str
        Auction quarter (aq1-aq4).

    Returns
    -------
    pl.DataFrame
        V6.1 data with spice6 density columns added.
    """
    df = load_v61_group(planning_year, aq_round)

    spice6 = load_spice6_density_annual(planning_year, aq_round)
    if len(spice6) > 0:
        df = df.join(
            spice6,
            on=["constraint_id", "flow_direction"],
            how="left",
        )
        spice6_cols = [c for c in spice6.columns if c not in ("constraint_id", "flow_direction")]
        df = df.with_columns([pl.col(c).fill_null(0.0) for c in spice6_cols])
        n_matched = len(df.filter(pl.col("prob_exceed_110") > 0))
        print(f"[data_loader] spice6 enrichment: {n_matched}/{len(df)} matched")
    else:
        print(f"[data_loader] WARNING: no spice6 data for {planning_year}/{aq_round}")
        for col in ["prob_exceed_110", "prob_exceed_100", "prob_exceed_90",
                     "prob_exceed_85", "prob_exceed_80", "constraint_limit"]:
            df = df.with_columns(pl.lit(0.0).alias(col))

    return df


def load_multiple_groups(groups: list[str]) -> pl.DataFrame:
    """Load and concatenate multiple (planning_year/aq_round) groups.

    Parameters
    ----------
    groups : list[str]
        Group IDs as "YYYY-06/aqN" strings.

    Returns
    -------
    pl.DataFrame
        Concatenated data with query_group column.
    """
    dfs = []
    for group_id in groups:
        planning_year, aq_round = group_id.split("/")
        try:
            df = load_v61_enriched(planning_year, aq_round)
            dfs.append(df)
            print(f"[data_loader] loaded {group_id}: {len(df)} rows, mem={mem_mb():.0f} MB")
        except FileNotFoundError:
            print(f"[data_loader] WARNING: skipping {group_id} (not found)")

    if not dfs:
        raise ValueError(f"No data found for groups: {groups}")
    return pl.concat(dfs, how="diagonal")
```

**Step 2: Test data loading**

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-annual-signal
python -c "
from ml.data_loader import load_v61_enriched
df = load_v61_enriched('2024-06', 'aq1')
print(f'Rows: {len(df)}')
print(f'Columns: {sorted(df.columns)}')
print(f'Has prob_exceed_110: {\"prob_exceed_110\" in df.columns}')
print(f'Non-zero prob_exceed_110: {len(df.filter(df[\"prob_exceed_110\"] > 0))}')
"
```

Expected: ~425 rows, all V6.1 columns + spice6 density columns, high enrichment rate.

**Step 3: Commit**

```bash
git add ml/data_loader.py
git commit -m "task 4: V6.1 data loader with spice6 density enrichment"
```

---

### Task 5: Ground truth module

**Files:**
- Create: `ml/ground_truth.py`

Fetches realized DA shadow prices via Ray and caches to parquet. This is the MOST CRITICAL module — it produces the target labels.

**Step 1: Write ground_truth.py**

```python
"""Ground truth: realized DA constraint shadow prices.

Fetches realized DA shadow prices for a target quarter via
MisoApTools.tools.get_da_shadow_by_peaktype() and maps them
to V6.1 constraint universe via branch_name.

REQUIRES RAY. Call init_ray() before using any function here.
Results are cached to parquet to avoid repeated Ray calls.
"""
from __future__ import annotations

import gc
import os
import resource
from pathlib import Path

import pandas as pd
import polars as pl

from ml.config import get_market_months, SPICE_DATA_BASE


CACHE_DIR = Path("cache/ground_truth")


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _get_constraint_branch_map() -> pl.DataFrame:
    """Load constraint_id -> branch_name mapping from SPICE constraint info."""
    info_path = Path(SPICE_DATA_BASE) / "MISO_SPICE_CONSTRAINT_INFO.parquet"
    info = pl.read_parquet(str(info_path))
    # Get unique constraint_id -> branch_name mapping
    mapping = info.select(["constraint_id", "branch_name"]).unique()
    return mapping


def fetch_realized_da_quarter(
    planning_year: str,
    aq_round: str,
    peak_type: str = "onpeak",
) -> pl.DataFrame:
    """Fetch realized DA shadow prices for a quarter.

    Returns DataFrame with columns: branch_name, realized_shadow_price.
    The shadow price is sum(abs(shadow_price)) across all hours in the quarter,
    aggregated per branch_name.
    """
    from pbase.analysis.tools.all_positions import MisoApTools

    market_months = get_market_months(planning_year, aq_round)
    aptools = MisoApTools()

    all_da = []
    for mm in market_months:
        year, month = mm.split("-")
        st = pd.Timestamp(f"{year}-{month}-01", tz="US/Central")
        # End of month (exclusive)
        et = st + pd.offsets.MonthBegin(1)
        print(f"[ground_truth] Fetching DA shadow: {st.date()} to {et.date()}, mem={mem_mb():.0f} MB")

        da_shadow = aptools.tools.get_da_shadow_by_peaktype(
            st=st, et_ex=et, peak_type=peak_type,
        )
        if da_shadow is not None and len(da_shadow) > 0:
            all_da.append(da_shadow)

    if not all_da:
        return pl.DataFrame({"branch_name": [], "realized_shadow_price": []})

    da_pd = pd.concat(all_da)
    da_pl = pl.from_pandas(da_pd.reset_index())

    # Map monitored_facility -> branch_name via constraint info
    # The DA shadow returns constraint names as monitored_facility
    # We need to map these to branch_name for joining with V6.1
    constraint_map = _get_constraint_branch_map()

    # DA shadow might have monitored_facility or constraint_id column
    # Try to join on available columns
    if "monitored_facility" in da_pl.columns:
        # Aggregate per monitored_facility first
        per_facility = da_pl.group_by("monitored_facility").agg(
            pl.col("shadow_price").abs().sum().alias("realized_shadow_price")
        )
        # Map to branch_name
        result = per_facility.join(
            constraint_map.rename({"constraint_id": "monitored_facility"}),
            on="monitored_facility",
            how="left",
        )
    elif "branch_name" in da_pl.columns:
        result = da_pl.group_by("branch_name").agg(
            pl.col("shadow_price").abs().sum().alias("realized_shadow_price")
        )
    else:
        print(f"[ground_truth] WARNING: unexpected columns: {da_pl.columns}")
        return pl.DataFrame({"branch_name": [], "realized_shadow_price": []})

    # Aggregate by branch_name (multiple facilities may map to same branch)
    if "branch_name" in result.columns:
        result = result.group_by("branch_name").agg(
            pl.col("realized_shadow_price").sum()
        )
    else:
        print(f"[ground_truth] WARNING: no branch_name after mapping")
        return pl.DataFrame({"branch_name": [], "realized_shadow_price": []})

    return result.filter(pl.col("realized_shadow_price") > 0)


def get_ground_truth(
    planning_year: str,
    aq_round: str,
    v61_df: pl.DataFrame,
    cache: bool = True,
) -> pl.DataFrame:
    """Get realized DA shadow prices aligned with V6.1 constraint universe.

    Parameters
    ----------
    planning_year, aq_round : str
        Target quarter.
    v61_df : pl.DataFrame
        V6.1 data for this quarter (defines constraint universe).
    cache : bool
        If True, cache results to parquet.

    Returns
    -------
    pl.DataFrame
        V6.1 data with 'realized_shadow_price' column added.
        Constraints that didn't bind get realized_shadow_price = 0.0.
    """
    cache_path = CACHE_DIR / f"{planning_year}_{aq_round}.parquet"

    if cache and cache_path.exists():
        realized = pl.read_parquet(str(cache_path))
        print(f"[ground_truth] loaded from cache: {cache_path} ({len(realized)} binding)")
    else:
        realized = fetch_realized_da_quarter(planning_year, aq_round)
        if cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            realized.write_parquet(str(cache_path))
            print(f"[ground_truth] cached to {cache_path}")

    # Join: V6.1 constraint universe LEFT JOIN realized DA on branch_name/equipment
    join_col = "branch_name" if "branch_name" in v61_df.columns else "equipment"
    result = v61_df.join(
        realized,
        left_on=join_col,
        right_on="branch_name",
        how="left",
    )
    result = result.with_columns(
        pl.col("realized_shadow_price").fill_null(0.0)
    )

    n_binding = len(result.filter(pl.col("realized_shadow_price") > 0))
    print(f"[ground_truth] {planning_year}/{aq_round}: {n_binding}/{len(result)} constraints binding")

    return result
```

**Step 2: Test ground truth (requires Ray — run manually)**

This requires Ray so it cannot be auto-tested. Create a small test script:

```python
# scripts/test_ground_truth.py
"""Test ground truth fetching for one quarter. Requires Ray."""
import os
os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"

from pbase.config.ray import init_ray
import pmodel
init_ray(extra_modules=[pmodel])

from ml.data_loader import load_v61_group
from ml.ground_truth import get_ground_truth

v61 = load_v61_group("2024-06", "aq1")
result = get_ground_truth("2024-06", "aq1", v61, cache=True)
print(f"Result: {len(result)} rows")
n_binding = len(result.filter(result["realized_shadow_price"] > 0))
print(f"Binding: {n_binding}/{len(result)} ({100*n_binding/len(result):.1f}%)")
print(result.filter(result["realized_shadow_price"] > 0).sort("realized_shadow_price", descending=True).head(10))
```

Run: `python scripts/test_ground_truth.py`

Expected: ~425 rows, some fraction binding (10-30%), cached to `cache/ground_truth/2024-06_aq1.parquet`.

**Step 3: Commit**

```bash
git add ml/ground_truth.py scripts/test_ground_truth.py
git commit -m "task 5: ground truth module (realized DA via Ray, cached to parquet)"
```

---

### Task 6: Cache ALL ground truth

**Files:**
- Create: `scripts/cache_all_ground_truth.py`

Run this ONCE to pre-fetch all 28 quarters. After this, no more Ray needed for ground truth.

**Step 1: Write cache script**

```python
# scripts/cache_all_ground_truth.py
"""Pre-fetch and cache realized DA shadow prices for all 28 quarters."""
import os
os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"

from pbase.config.ray import init_ray
import pmodel
init_ray(extra_modules=[pmodel])

import ray
from ml.config import PLANNING_YEARS, AQ_ROUNDS
from ml.data_loader import load_v61_group
from ml.ground_truth import get_ground_truth

for year in PLANNING_YEARS:
    for aq in AQ_ROUNDS:
        try:
            v61 = load_v61_group(year, aq)
            get_ground_truth(year, aq, v61, cache=True)
        except Exception as e:
            print(f"[cache] ERROR {year}/{aq}: {e}")

ray.shutdown()
print("[cache] Done. All ground truth cached.")
```

**Step 2: Run it**

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-annual-signal
python scripts/cache_all_ground_truth.py
```

**Step 3: Verify cache**

```bash
ls -la cache/ground_truth/
# Should have 28 parquet files (7 years x 4 rounds)
```

**Step 4: Commit**

```bash
git add scripts/cache_all_ground_truth.py
# Do NOT commit cache/ — add to .gitignore
echo "cache/" >> .gitignore
git add .gitignore
git commit -m "task 6: cache all ground truth (28 quarters via Ray)"
```

---

### Task 7: Pipeline module (annual-specific)

**Files:**
- Create: `ml/pipeline.py`

Adapts stage4's pipeline for annual structure: expanding-window train, realized DA ground truth.

**Step 1: Write pipeline.py**

```python
"""Annual LTR pipeline: load -> features -> train -> predict -> evaluate.

For each eval group (planning_year/aq_round):
1. Train on all prior years (expanding window)
2. Predict on eval group
3. Evaluate against realized DA shadow prices
"""
from __future__ import annotations

import gc
import resource
from typing import Any

import numpy as np
import polars as pl

from ml.config import PipelineConfig, EVAL_SPLITS, AQ_ROUNDS
from ml.data_loader import load_v61_enriched, load_multiple_groups
from ml.evaluate import evaluate_ltr
from ml.features import compute_query_groups, prepare_features
from ml.ground_truth import get_ground_truth
from ml.train import predict_scores, train_ltr_model


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _get_train_groups(eval_group: str) -> list[str]:
    """Determine training groups for a given eval group using expanding window."""
    eval_year = eval_group.split("/")[0]
    for split_name, split_def in EVAL_SPLITS.items():
        if split_def["eval_year"] == eval_year:
            train_years = split_def["train_years"]
            return [f"{y}/{aq}" for y in train_years for aq in AQ_ROUNDS]
    raise ValueError(f"No training split defined for eval group: {eval_group}")


def run_pipeline(
    config: PipelineConfig,
    version_id: str,
    eval_group: str,
) -> dict[str, Any]:
    """Run the annual LTR pipeline for a single eval group.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.
    version_id : str
        Version identifier.
    eval_group : str
        Eval group as "YYYY-06/aqN".

    Returns
    -------
    dict with "metrics" key.
    """
    planning_year, aq_round = eval_group.split("/")
    print(f"[pipeline] version={version_id} eval={eval_group}")

    # Phase 1: Load train data
    print(f"[phase 1] Loading train data ... (mem={mem_mb():.0f} MB)")
    train_groups = _get_train_groups(eval_group)
    train_df = load_multiple_groups(train_groups)

    # Add ground truth labels to train data
    print(f"[phase 1b] Adding ground truth to train ... (mem={mem_mb():.0f} MB)")
    train_parts = []
    for tg in train_groups:
        ty, tq = tg.split("/")
        part = train_df.filter(pl.col("query_group") == tg)
        part = get_ground_truth(ty, tq, part, cache=True)
        train_parts.append(part)
    train_df = pl.concat(train_parts, how="diagonal")

    # Phase 2: Load test data
    print(f"[phase 2] Loading test data ... (mem={mem_mb():.0f} MB)")
    test_df = load_v61_enriched(planning_year, aq_round)
    test_df = test_df.with_columns(pl.lit(eval_group).alias("query_group"))
    test_df = get_ground_truth(planning_year, aq_round, test_df, cache=True)

    # Phase 3: Prepare features
    print(f"[phase 3] Preparing features ... (mem={mem_mb():.0f} MB)")
    train_df = train_df.sort("query_group")
    X_train, _ = prepare_features(train_df, config.ltr)
    y_train = train_df["realized_shadow_price"].to_numpy().astype(np.float64)
    groups_train = compute_query_groups(train_df)

    X_test, _ = prepare_features(test_df, config.ltr)
    actual_sp = test_df["realized_shadow_price"].to_numpy().astype(np.float64)

    print(f"[phase 3] train={X_train.shape} groups={groups_train} test={X_test.shape} "
          f"(mem={mem_mb():.0f} MB)")

    del train_df
    gc.collect()

    # Phase 4: Train
    print(f"[phase 4] Training LTR model ({config.ltr.backend}) ... (mem={mem_mb():.0f} MB)")
    model = train_ltr_model(X_train, y_train, groups_train, config.ltr)

    del X_train, y_train, groups_train
    gc.collect()

    # Phase 5: Predict + Evaluate
    print(f"[phase 5] Predicting and evaluating ... (mem={mem_mb():.0f} MB)")
    scores = predict_scores(model, X_test)
    metrics = evaluate_ltr(actual_sp, scores)

    # Feature importance
    feat_names = config.ltr.features
    if hasattr(model, "feature_importance"):
        importance = model.feature_importance(importance_type="gain")
    else:
        importance = model.feature_importances_
    metrics["_feature_importance"] = {
        name: float(imp)
        for name, imp in sorted(zip(feat_names, importance), key=lambda x: x[1], reverse=True)
    }

    del X_test, scores, actual_sp, test_df
    gc.collect()

    print(f"[pipeline] complete (mem={mem_mb():.0f} MB)")
    for key, value in metrics.items():
        if key.startswith("_"):
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    return {"metrics": metrics}
```

**Step 2: Commit**

```bash
git add ml/pipeline.py
git commit -m "task 7: annual pipeline (expanding-window train, realized DA ground truth)"
```

---

### Task 8: Benchmark runner

**Files:**
- Create: `ml/benchmark.py`

Adapts stage4's benchmark for annual groups.

**Step 1: Write benchmark.py**

```python
"""Multi-group benchmark for annual LTR ranking pipeline.

Runs pipeline for each eval group independently.

CLI:
  python ml/benchmark.py --version-id v1 --screen     # 4 groups (fast)
  python ml/benchmark.py --version-id v1               # 12 groups (default)
"""
import argparse
import gc
import json
import resource
from pathlib import Path

from ml.config import PipelineConfig, SCREEN_EVAL_GROUPS, DEFAULT_EVAL_GROUPS
from ml.evaluate import aggregate_months  # works for any groups, name is legacy
from ml.pipeline import run_pipeline


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_benchmark(
    version_id: str,
    eval_groups: list[str],
    registry_dir: str = "registry",
    config: PipelineConfig | None = None,
    mode: str = "eval",
) -> dict:
    """Run benchmark across multiple eval groups."""
    if config is None:
        config = PipelineConfig()

    print(f"[benchmark] {mode.upper()} MODE: {len(eval_groups)} eval groups")

    per_group = {}
    skipped = []
    for group_id in eval_groups:
        print(f"\n[benchmark] === {group_id} === mem: {mem_mb():.0f} MB")
        try:
            result = run_pipeline(config=config, version_id=version_id, eval_group=group_id)
            metrics = result.get("metrics", {})
            if metrics:
                per_group[group_id] = metrics
            else:
                skipped.append(group_id)
        except Exception as e:
            print(f"[benchmark] ERROR {group_id}: {e}")
            skipped.append(group_id)
        gc.collect()

    if skipped:
        print(f"\n[benchmark] Skipped {len(skipped)} groups: {skipped}")

    # Extract feature importance
    importance_per_group = {}
    for group_id in list(per_group.keys()):
        imp = per_group[group_id].pop("_feature_importance", None)
        if imp:
            importance_per_group[group_id] = imp

    # Aggregate (reuse monthly aggregate function — works on any dict of dicts)
    agg = aggregate_months(per_group)

    result = {
        "eval_config": {
            "eval_groups": eval_groups,
            "mode": mode,
        },
        "per_month": per_group,  # key name kept for compare.py compatibility
        "aggregate": agg,
        "n_months": len(per_group),
        "n_months_requested": len(eval_groups),
        "skipped_months": skipped,
    }

    # Save to registry
    registry_path = Path(registry_dir)
    version_dir = registry_path / version_id
    version_dir.mkdir(parents=True, exist_ok=True)

    with open(version_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(version_dir / "config.json", "w") as f:
        json.dump({"ltr": config.ltr.to_dict(), "eval_config": result["eval_config"]}, f, indent=2)
    with open(version_dir / "meta.json", "w") as f:
        json.dump({"n_groups": len(per_group), "version_id": version_id}, f, indent=2)

    print(f"\n[benchmark] Wrote metrics to {version_dir / 'metrics.json'}")
    print(f"[benchmark] Complete: {len(per_group)} groups evaluated")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run annual LTR benchmark")
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--screen", action="store_true", help="4 groups (fast)")
    parser.add_argument("--eval-groups", nargs="+", default=None)
    parser.add_argument("--registry-dir", default="registry")
    args = parser.parse_args()

    if args.eval_groups:
        eval_groups = args.eval_groups
        mode = "custom"
    elif args.screen:
        eval_groups = SCREEN_EVAL_GROUPS
        mode = "screen"
    else:
        eval_groups = DEFAULT_EVAL_GROUPS
        mode = "eval"

    run_benchmark(
        version_id=args.version_id,
        eval_groups=eval_groups,
        registry_dir=args.registry_dir,
        mode=mode,
    )


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add ml/benchmark.py
git commit -m "task 8: benchmark runner (multi-group annual evaluation)"
```

---

### Task 9: v0 baseline — V6.1 formula vs realized DA

**Files:**
- Create: `scripts/run_v0_baseline.py`

Evaluates the stored V6.1 `rank` column against realized DA ground truth for all 12 eval groups. This is the benchmark that ML must beat.

**Step 1: Write v0 baseline script**

```python
# scripts/run_v0_baseline.py
"""v0 Baseline: V6.1 formula rank evaluated against realized DA.

No training. Just loads V6.1 rank and evaluates against ground truth.
Produces registry/v0/metrics.json and calibrates gates.

Formula: rank_ori = 0.60*da_rank + 0.30*density_mix_rank + 0.10*density_ori_rank
Score for evaluation: 1 - rank  (higher = more binding)
"""
import json
import gc
from pathlib import Path

from ml.config import DEFAULT_EVAL_GROUPS, SCREEN_EVAL_GROUPS, AQ_ROUNDS
from ml.data_loader import load_v61_group
from ml.evaluate import evaluate_ltr, aggregate_months
from ml.ground_truth import get_ground_truth

import numpy as np


def run_v0_baseline(eval_groups: list[str], registry_dir: str = "registry") -> dict:
    per_group = {}

    for group_id in eval_groups:
        planning_year, aq_round = group_id.split("/")
        print(f"\n[v0] === {group_id} ===")

        v61 = load_v61_group(planning_year, aq_round)
        v61 = get_ground_truth(planning_year, aq_round, v61, cache=True)

        # V6.1 formula score: rank column (lower = more binding)
        # Invert: 1 - rank so higher = more binding for evaluate_ltr
        scores = 1.0 - v61["rank"].to_numpy().astype(np.float64)
        actual = v61["realized_shadow_price"].to_numpy().astype(np.float64)

        metrics = evaluate_ltr(actual, scores)
        per_group[group_id] = metrics

        n_binding = (actual > 0).sum()
        print(f"  Binding: {n_binding}/{len(actual)} ({100*n_binding/len(actual):.1f}%)")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

        gc.collect()

    agg = aggregate_months(per_group)

    result = {
        "eval_config": {"eval_groups": eval_groups, "mode": "eval"},
        "per_month": per_group,
        "aggregate": agg,
        "n_months": len(per_group),
        "n_months_requested": len(eval_groups),
        "skipped_months": [],
    }

    # Save to registry
    version_dir = Path(registry_dir) / "v0"
    version_dir.mkdir(parents=True, exist_ok=True)

    with open(version_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    config = {
        "formula": "rank_ori = 0.60*da_rank_value + 0.30*density_mix_rank_value + 0.10*density_ori_rank_value",
        "score": "1 - rank (inverted so higher = more binding)",
        "note": "No training. Evaluates stored V6.1 formula rank against realized DA ground truth.",
    }
    with open(version_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[v0] === AGGREGATE (mean across {len(per_group)} groups) ===")
    for k, v in agg.get("mean", {}).items():
        print(f"  {k}: {v:.4f}")

    # Calibrate gates from v0 results
    _calibrate_gates(agg, registry_dir)

    return result


def _calibrate_gates(agg: dict, registry_dir: str) -> None:
    """Calibrate 3-layer gates from v0 aggregate metrics."""
    blocking_metrics = ["VC@20", "VC@100", "Recall@20", "Recall@50", "Recall@100", "NDCG"]
    monitor_metrics = ["Spearman", "Tier0-AP", "Tier01-AP"]

    gates = {}
    for metric in blocking_metrics + monitor_metrics:
        mean_val = agg.get("mean", {}).get(metric)
        min_val = agg.get("min", {}).get(metric)
        if mean_val is None:
            continue
        group = "A" if metric in blocking_metrics else "B"
        gates[metric] = {
            "floor": round(0.9 * mean_val, 4),         # L1: 90% of v0 mean
            "tail_floor": round(min_val, 4) if min_val is not None else None,  # L2: v0 min
            "direction": "higher",
            "group": group,
        }

    gates_data = {
        "gates": gates,
        "noise_tolerance": 0.02,
        "tail_max_failures": 1,
        "calibrated_from": "v0",
    }

    gates_path = Path(registry_dir) / "gates.json"
    with open(gates_path, "w") as f:
        json.dump(gates_data, f, indent=2)
    print(f"[v0] Calibrated gates from v0 -> {gates_path}")

    champion_path = Path(registry_dir) / "champion.json"
    with open(champion_path, "w") as f:
        json.dump({"version": "v0"}, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", action="store_true")
    args = parser.parse_args()

    groups = SCREEN_EVAL_GROUPS if args.screen else DEFAULT_EVAL_GROUPS
    run_v0_baseline(groups)
```

**Step 2: Run v0 baseline**

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-annual-signal
python scripts/run_v0_baseline.py --screen   # fast: 4 groups
python scripts/run_v0_baseline.py            # full: 12 groups
```

**Step 3: Verify registry outputs**

```bash
cat registry/v0/metrics.json | python -m json.tool | head -30
cat registry/gates.json | python -m json.tool
cat registry/champion.json
```

**Step 4: Commit**

```bash
git add scripts/run_v0_baseline.py registry/v0/ registry/gates.json registry/champion.json
git commit -m "task 9: v0 baseline — V6.1 formula vs realized DA ground truth"
```

---

### Task 10: v1 ML experiment — Set A (6 features)

**Files:**
- Create: `scripts/run_v1_experiment.py`

First ML version: same 6 features as V6.1, but learned weights via LightGBM lambdarank.

**Step 1: Write v1 experiment script**

```python
# scripts/run_v1_experiment.py
"""v1: LightGBM LambdaRank with V6.1 base features (Set A, 6 features).

Same information available to V6.1 formula, but ML-learned weights.
"""
from ml.benchmark import run_benchmark
from ml.config import (
    PipelineConfig, LTRConfig,
    SET_A_FEATURES, SET_A_MONOTONE,
    SCREEN_EVAL_GROUPS, DEFAULT_EVAL_GROUPS,
)
from ml.compare import run_comparison

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--screen", action="store_true")
args = parser.parse_args()

config = PipelineConfig(
    ltr=LTRConfig(
        features=SET_A_FEATURES,
        monotone_constraints=SET_A_MONOTONE,
        backend="lightgbm",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
    ),
)

groups = SCREEN_EVAL_GROUPS if args.screen else DEFAULT_EVAL_GROUPS
mode = "screen" if args.screen else "eval"

run_benchmark(
    version_id="v1",
    eval_groups=groups,
    config=config,
    mode=mode,
)

# Compare against v0
run_comparison(
    batch_id="annual",
    iteration=1,
    output_path="reports/v1_comparison.md",
)
```

**Step 2: Run v1**

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-annual-signal
python scripts/run_v1_experiment.py --screen   # fast: 4 groups
python scripts/run_v1_experiment.py            # full: 12 groups
```

**Step 3: Review results**

```bash
cat reports/v1_comparison.md
```

**Step 4: Commit**

```bash
git add scripts/run_v1_experiment.py registry/v1/ reports/
git commit -m "task 10: v1 ML experiment — Set A (6 V6.1 features)"
```

---

### Task 11: v2 ML experiment — Set B (11 features)

**Files:**
- Create: `scripts/run_v2_experiment.py`

Add spice6 density exceedance probabilities.

**Step 1: Write v2 experiment script**

```python
# scripts/run_v2_experiment.py
"""v2: LightGBM LambdaRank with V6.1 + spice6 density (Set B, 11 features)."""
from ml.benchmark import run_benchmark
from ml.config import (
    PipelineConfig, LTRConfig,
    SET_B_FEATURES, SET_B_MONOTONE,
    SCREEN_EVAL_GROUPS, DEFAULT_EVAL_GROUPS,
)
from ml.compare import run_comparison

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--screen", action="store_true")
args = parser.parse_args()

config = PipelineConfig(
    ltr=LTRConfig(
        features=SET_B_FEATURES,
        monotone_constraints=SET_B_MONOTONE,
        backend="lightgbm",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
    ),
)

groups = SCREEN_EVAL_GROUPS if args.screen else DEFAULT_EVAL_GROUPS
mode = "screen" if args.screen else "eval"

run_benchmark(
    version_id="v2",
    eval_groups=groups,
    config=config,
    mode=mode,
)

run_comparison(
    batch_id="annual",
    iteration=2,
    output_path="reports/v2_comparison.md",
)
```

**Step 2: Run and review**

```bash
python scripts/run_v2_experiment.py --screen
python scripts/run_v2_experiment.py
cat reports/v2_comparison.md
```

**Step 3: Commit**

```bash
git add scripts/run_v2_experiment.py registry/v2/ reports/
git commit -m "task 11: v2 ML experiment — Set B (11 features, +spice6 density)"
```

---

### Task 12: Update mem.md with results

**Files:**
- Modify: `mem.md`

**Step 1: Update mem.md with actual v0/v1/v2 numbers**

After running all experiments, update mem.md with actual metrics tables comparing v0, v1, v2.

**Step 2: Commit**

```bash
git add mem.md
git commit -m "task 12: update mem.md with v0/v1/v2 results"
```

---

## Summary of Execution Order

| Task | Description | Depends on | Ray needed? |
|------|------------|-----------|------------|
| 1 | Scaffold repo | — | No |
| 2 | Copy reusable modules | 1 | No |
| 3 | Config module | 1 | No |
| 4 | Data loader | 3 | No |
| 5 | Ground truth module | 3 | Yes (write) |
| 6 | Cache all ground truth | 4, 5 | Yes (one-time) |
| 7 | Pipeline module | 2, 3, 4, 5 | No (uses cache) |
| 8 | Benchmark runner | 7 | No |
| 9 | v0 baseline | 4, 5, 8 | No (uses cache) |
| 10 | v1 ML experiment | 7, 8, 9 | No |
| 11 | v2 ML experiment | 7, 8, 9 | No |
| 12 | Update mem.md | 9, 10, 11 | No |

**Tasks 1-4 can be done without Ray.** Task 5-6 require Ray (one-time ground truth caching). Tasks 7-12 use cached ground truth — no Ray needed.

**Parallelizable:** Tasks 1-3 are independent scaffolding. Tasks 10-11 are independent experiments (after 9).
