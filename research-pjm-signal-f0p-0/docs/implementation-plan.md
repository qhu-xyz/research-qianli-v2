# PJM V7.0b Constraint-Tier ML Signal — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PJM V7.0b constraint-tier signal that replaces V6.2B formula scoring with ML-scored tiers for f0/f1 across 3 class types, with V6.2B passthrough for f2-f11.

**Architecture:** Fork-and-adapt the proven MISO stage5 ML pipeline (`research-stage5-tier/ml/`, 14 modules) for PJM. The critical PJM-specific adaptation is the branch-level constraint→DA target join (naive join captures only 46% of binding value; branch-level captures 96-99%). The pipeline follows the same walk-forward evaluation with LightGBM LambdaRank, 9 features, tiered labels, and row-percentile tiering. After research validation, deploy as a signal writer analogous to `research-miso-signal7/`.

**Tech Stack:** Python, polars, LightGBM (LambdaRank), NumPy, pbase (PjmApTools for DA data), Ray (for DA fetch only).

---

## File Structure

```
research-pjm-signal-f0p-0/
├── ml/
│   ├── __init__.py
│   ├── config.py              # PJM paths, auction schedule, LTRConfig, PipelineConfig
│   ├── branch_mapping.py      # constraint_id → branch_name mapping via constraint_info
│   ├── realized_da.py         # Fetch + cache realized DA by branch_name
│   ├── spice6_loader.py       # Load spice6 density features (PJM paths)
│   ├── data_loader.py         # Load V6.2B + spice6 + realized DA (branch-level join)
│   ├── features.py            # Feature prep + query groups (reuse from MISO)
│   ├── train.py               # LightGBM LambdaRank training (copy from MISO)
│   ├── evaluate.py            # VC@20, Recall, NDCG metrics (copy from MISO)
│   ├── v62b_formula.py        # V6.2B formula reproduction (copy from MISO)
│   ├── pipeline.py            # load → train → predict → evaluate (adapted)
│   ├── benchmark.py           # Walk-forward multi-month evaluation (adapted)
│   ├── compare.py             # Gate comparison system (copy from MISO)
│   ├── registry_paths.py      # Registry path helpers (copy from MISO)
│   └── tests/
│       ├── __init__.py
│       ├── test_branch_mapping.py
│       ├── test_realized_da.py
│       └── test_config.py
├── scripts/
│   ├── cache_realized_da.py      # Preflight: cache all needed realized DA months
│   ├── run_v0_formula_baseline.py # V0 baseline for all 6 slices
│   ├── run_v2_ml.py              # V2 ML model for all 6 slices
│   ├── run_blend_search.py       # Blend search for all 6 slices
│   └── run_holdout.py            # Holdout evaluation
├── v70/
│   ├── __init__.py
│   ├── cache.py               # Realized DA cache management for deployment
│   ├── inference.py           # ML inference (train on history, score target)
│   └── signal_writer.py       # Rank/tier computation, signal assembly
├── registry/                  # Dev evaluation results
│   ├── f0/
│   │   ├── onpeak/
│   │   ├── dailyoffpeak/
│   │   └── wkndonpeak/
│   └── f1/
│       ├── onpeak/
│       ├── dailyoffpeak/
│       └── wkndonpeak/
├── holdout/                   # Holdout evaluation results (same structure)
├── data/
│   └── realized_da/           # Local realized DA cache (branch_name keyed)
└── docs/
    └── implementation-plan.md # This file
```

### Key difference from MISO

| Module | MISO | PJM change |
|--------|------|------------|
| `config.py` | MISO paths, `MISO_AUCTION_SCHEDULE` | PJM paths, `PJM_AUCTION_SCHEDULE`, 3 class types |
| `branch_mapping.py` | N/A (MISO joins on constraint_id) | **NEW**: constraint_info → branch_name mapping |
| `realized_da.py` | Join on `constraint_id` | Join on `branch_name` via branch mapping |
| `data_loader.py` | Join realized DA on `constraint_id` | Join realized DA on `branch_name` |
| `spice6_loader.py` | MISO density path | PJM density path |
| `train.py` | — | Copy verbatim |
| `evaluate.py` | — | Copy verbatim |
| `v62b_formula.py` | — | Copy verbatim |
| `features.py` | — | Copy verbatim (add `v7_formula_score` derived feature) |
| `benchmark.py` | 2 class types | 3 class types |

---

## Chunk 1: Core ML Modules (Config, Branch Mapping, Realized DA)

### Task 1: Create `ml/__init__.py` and copy unchanged modules

Copy modules that need zero PJM-specific changes from MISO stage5.

**Files:**
- Create: `ml/__init__.py`
- Create: `ml/train.py` (copy from `research-stage5-tier/ml/train.py`)
- Create: `ml/evaluate.py` (copy from `research-stage5-tier/ml/evaluate.py`)
- Create: `ml/v62b_formula.py` (copy from `research-stage5-tier/ml/v62b_formula.py`)
- Create: `ml/registry_paths.py` (copy from `research-stage5-tier/ml/registry_paths.py`)
- Create: `ml/compare.py` (copy from `research-stage5-tier/ml/compare.py`)
- Create: `ml/tests/__init__.py`

- [ ] **Step 1: Copy unchanged modules**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
mkdir -p ml/tests
touch ml/__init__.py ml/tests/__init__.py

# These modules have NO MISO-specific logic — copy verbatim
cp ../research-stage5-tier/ml/train.py ml/train.py
cp ../research-stage5-tier/ml/evaluate.py ml/evaluate.py
cp ../research-stage5-tier/ml/v62b_formula.py ml/v62b_formula.py
cp ../research-stage5-tier/ml/registry_paths.py ml/registry_paths.py
cp ../research-stage5-tier/ml/compare.py ml/compare.py
```

- [ ] **Step 2: Verify imports work**

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
python -c "from ml.train import train_ltr_model, predict_scores; print('train OK')"
python -c "from ml.evaluate import evaluate_ltr, aggregate_months; print('evaluate OK')"
python -c "from ml.v62b_formula import v62b_score, dense_rank_normalized; print('v62b OK')"
python -c "from ml.registry_paths import registry_root, holdout_root; print('registry OK')"
```

Expected: all print OK, no errors.

- [ ] **Step 3: Commit**

```bash
git add ml/
git commit -m "copy unchanged MISO modules: train, evaluate, v62b_formula, registry_paths, compare"
```

---

### Task 2: Create `ml/config.py` with PJM paths and auction schedule

This is the central configuration module. Key PJM differences:
- All data paths point to PJM locations
- PJM auction schedule: f0-f11, varies by month (May: f0 only, June: all 12)
- 3 class types: onpeak, dailyoffpeak, wkndonpeak
- Leaky features include PJM-specific columns

**Files:**
- Create: `ml/config.py`
- Test: `ml/tests/test_config.py`

- [ ] **Step 1: Write config test**

```python
# ml/tests/test_config.py
"""Tests for PJM config module."""
import pytest
from ml.config import (
    delivery_month,
    has_period_type,
    period_offset,
    collect_usable_months,
    PJM_CLASS_TYPES,
    V62B_SIGNAL_BASE,
    SPICE6_DENSITY_BASE,
)


def test_period_offset():
    assert period_offset("f0") == 0
    assert period_offset("f1") == 1
    assert period_offset("f11") == 11


def test_delivery_month():
    assert delivery_month("2025-01", "f0") == "2025-01"
    assert delivery_month("2025-01", "f1") == "2025-02"
    assert delivery_month("2025-06", "f11") == "2026-05"


def test_has_period_type_may_f0_only():
    """May auctions have only f0."""
    assert has_period_type("2025-05", "f0") is True
    assert has_period_type("2025-05", "f1") is False


def test_has_period_type_june_all():
    """June auctions have f0-f11."""
    for i in range(12):
        assert has_period_type("2025-06", f"f{i}") is True


def test_class_types():
    assert PJM_CLASS_TYPES == ["onpeak", "dailyoffpeak", "wkndonpeak"]


def test_pjm_paths_exist():
    from pathlib import Path
    assert Path(V62B_SIGNAL_BASE).exists(), f"V6.2B path missing: {V62B_SIGNAL_BASE}"
    assert Path(SPICE6_DENSITY_BASE).exists(), f"Spice6 density path missing: {SPICE6_DENSITY_BASE}"


def test_collect_usable_months_f0():
    """f0 should return 8 contiguous months for a well-covered eval month."""
    months = collect_usable_months("2023-06", "f0", n_months=8)
    assert len(months) == 8
    # Most recent should be 2 months before target (lag built into collect_usable_months)
    assert months[0] <= "2023-04"


def test_collect_usable_months_f1_skips_gaps():
    """f1 should skip months where f1 doesn't exist (May)."""
    months = collect_usable_months("2023-06", "f1", n_months=8)
    assert len(months) >= 6  # may be < 8 due to gaps
    for m in months:
        assert has_period_type(m, "f1"), f"{m} should have f1"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
python -m pytest ml/tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ml.config'`

- [ ] **Step 3: Write `ml/config.py`**

```python
# ml/config.py
"""PJM LTR pipeline configuration.

Adapted from research-stage5-tier/ml/config.py for PJM market structure.
Key PJM differences:
  - 3 class types: onpeak, dailyoffpeak, wkndonpeak
  - f0 through f11 period types (vs MISO f0-f3)
  - PJM-specific data paths
  - Branch-level DA join (not constraint_id)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── PJM class types ──
PJM_CLASS_TYPES = ["onpeak", "dailyoffpeak", "wkndonpeak"]

# ── Leakage guard ──
_LEAKY_FEATURES: set[str] = {
    "rank", "rank_ori", "tier",
    "shadow_sign", "shadow_price",
}

# ── V10E features (9, same as MISO champion) ──
V10E_FEATURES = [
    "binding_freq_1", "binding_freq_3", "binding_freq_6", "binding_freq_12",
    "binding_freq_15", "v7_formula_score", "prob_exceed_110", "constraint_limit",
    "da_rank_value",
]
V10E_MONOTONE = [1, 1, 1, 1, 1, -1, 1, 0, -1]

# ── Eval months (PJM-specific, based on V6.2B availability from 2017-06) ──
# Dev: 36 months (2020-06 to 2023-05), same range as MISO
_FULL_EVAL_MONTHS: list[str] = [
    f"{y:04d}-{m:02d}"
    for y in range(2020, 2024)
    for m in range(1, 13)
    if (y, m) >= (2020, 6) and (y, m) <= (2023, 5)
]

# Screen: 4 fast months
_SCREEN_EVAL_MONTHS: list[str] = [
    "2020-12", "2021-09", "2022-06", "2023-03",
]

# Default: 12 representative months
_DEFAULT_EVAL_MONTHS: list[str] = [
    "2020-09", "2020-12", "2021-03", "2021-06",
    "2021-09", "2021-12", "2022-03", "2022-06",
    "2022-09", "2022-12", "2023-03", "2023-05",
]

# Holdout: 2024-2025
HOLDOUT_MONTHS: list[str] = [
    f"{y:04d}-{m:02d}" for y in (2024, 2025) for m in range(1, 13)
]

# ── Data paths ──
V62B_SIGNAL_BASE = "/opt/data/xyz-dataset/signal_data/pjm/constraints/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1"
SPICE6_DENSITY_BASE = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/density"
SPICE6_MLPRED_BASE = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/ml_pred"
SPICE6_CI_BASE = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/constraint_info"
REALIZED_DA_CACHE = str(Path(__file__).resolve().parent.parent / "data" / "realized_da")

# ── PJM Auction Schedule ──
# Determined from actual V6.2B data on disk. PJM has f0-f11.
# June exposes all 12 monthly periods; schedule shrinks as planning year progresses.
# Source: pbase/src/pbase/data/const/period/pjm.py + V6.2B directory enumeration.
PJM_AUCTION_SCHEDULE: dict[int, list[str]] = {
    1: ["f0", "f1", "f2"],
    2: ["f0", "f1"],
    3: ["f0", "f1", "f2"],
    4: ["f0", "f1"],
    5: ["f0"],
    6: ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"],
    7: ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"],
    8: ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"],
    9: ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"],
    10: ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"],
    11: ["f0", "f1", "f2", "f3", "f4", "f5", "f6"],
    12: ["f0", "f1", "f2", "f3", "f4", "f5"],
}


def period_offset(period_type: str) -> int:
    """f0→0, f1→1, ..., f11→11."""
    if not period_type.startswith("f"):
        raise ValueError(f"period_offset only supports fN types, got '{period_type}'")
    return int(period_type[1:])


def delivery_month(auction_month: str, period_type: str) -> str:
    """Compute delivery month = auction_month + period_offset."""
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
    return period_type in PJM_AUCTION_SCHEDULE.get(month_num, ["f0"])


def collect_usable_months(
    target_auction_month: str,
    period_type: str,
    n_months: int = 8,
    min_months: int = 6,
    max_lookback: int = 24,
) -> list[str]:
    """Walk backward from target, collect n_months usable training auction months.

    A month is "usable" if:
      1. The period_type exists for that month (per PJM auction schedule)
      2. delivery_month(month, period_type) <= last_full_known

    last_full_known = target_auction_month - 2 (last complete realized DA at decision time).
    Returns most-recent-first order. Caller should reverse for chronological.
    """
    import pandas as pd

    target_ts = pd.Timestamp(target_auction_month)
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
        return []
    return usable


@dataclass
class LTRConfig:
    """Learning-to-rank configuration."""

    features: list[str] = field(default_factory=lambda: list(V10E_FEATURES))
    monotone_constraints: list[int] = field(default_factory=lambda: list(V10E_MONOTONE))
    backend: str = "lightgbm"
    label_mode: str = "tiered"
    n_estimators: int = 100
    learning_rate: float = 0.05
    min_child_weight: int = 25
    num_leaves: int = 31
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 1.0
    reg_lambda: float = 1.0
    max_depth: int = 5
    early_stopping_rounds: int = 20

    def __post_init__(self) -> None:
        if len(self.monotone_constraints) != len(self.features):
            raise ValueError(
                f"len(monotone_constraints) must match len(features) "
                f"({len(self.monotone_constraints)} != {len(self.features)})"
            )
        filtered_features: list[str] = []
        filtered_mono: list[int] = []
        for feat, mono in zip(self.features, self.monotone_constraints):
            if feat in _LEAKY_FEATURES:
                print(f"[config] WARNING: dropped leaky feature: {feat}")
                continue
            filtered_features.append(feat)
            filtered_mono.append(mono)
        self.features = filtered_features
        self.monotone_constraints = filtered_mono

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LTRConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    ltr: LTRConfig = field(default_factory=LTRConfig)
    train_months: int = 8
    val_months: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"ltr": self.ltr.to_dict(), "train_months": self.train_months, "val_months": self.val_months}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineConfig:
        return cls(ltr=LTRConfig.from_dict(d["ltr"]), train_months=d["train_months"], val_months=d["val_months"])


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

**IMPORTANT**: The `PJM_AUCTION_SCHEDULE` above is a best-guess from the handoff docs. The implementing agent MUST verify it against the actual V6.2B directory structure on disk by running:

```python
from pathlib import Path
base = Path("/opt/data/xyz-dataset/signal_data/pjm/constraints/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1")
schedule = {}
for month_dir in sorted(base.iterdir()):
    ptypes = sorted([p.name for p in month_dir.iterdir() if p.is_dir()])
    month_num = int(month_dir.name.split("-")[1])
    if month_num not in schedule:
        schedule[month_num] = ptypes
    else:
        # Take the union across years
        schedule[month_num] = sorted(set(schedule[month_num]) | set(ptypes))
print(schedule)
```

Update `PJM_AUCTION_SCHEDULE` with the verified values before proceeding.

- [ ] **Step 4: Run tests**

```bash
python -m pytest ml/tests/test_config.py -v
```

Expected: All tests PASS. If `PJM_AUCTION_SCHEDULE` is wrong, fix it first.

- [ ] **Step 5: Commit**

```bash
git add ml/config.py ml/tests/test_config.py
git commit -m "feat: add PJM config with auction schedule, paths, and LTRConfig"
```

---

### Task 3: Create `ml/branch_mapping.py` — constraint_id → branch_name mapping

This is the **most critical PJM-specific module**. It maps V6.2B `constraint_id` to `branch_name` via `constraint_info`, then maps DA `monitored_facility` to `branch_name`. Without this, the target join captures only 46% of binding value.

Reference: `research-spice-shadow-price-pred/src/shadow_price_prediction/data_loader.py:805`

**Files:**
- Create: `ml/branch_mapping.py`
- Test: `ml/tests/test_branch_mapping.py`

- [ ] **Step 1: Write branch mapping test**

```python
# ml/tests/test_branch_mapping.py
"""Tests for PJM branch mapping module."""
import polars as pl
import pytest
from ml.branch_mapping import load_constraint_info, build_branch_map, map_da_to_branches


def test_load_constraint_info_returns_dataframe():
    """constraint_info should load for a known month."""
    ci = load_constraint_info("2025-01", period_type="f0")
    assert isinstance(ci, pl.DataFrame)
    assert len(ci) > 0
    assert "constraint_id" in ci.columns
    assert "branch_name" in ci.columns


def test_build_branch_map_has_match_str():
    """Branch map should have match_str for DA joining."""
    ci = load_constraint_info("2025-01", period_type="f0")
    bmap = build_branch_map(ci)
    assert "match_str" in bmap.columns
    assert "branch_name" in bmap.columns
    # match_str should be uppercase
    for s in bmap["match_str"].head(5).to_list():
        assert s == s.upper(), f"match_str not uppercase: {s}"


def test_map_da_to_branches_captures_most_value():
    """Branch-level join should capture >90% of DA value."""
    ci = load_constraint_info("2025-01", period_type="f0")
    bmap = build_branch_map(ci)

    # Load a month of DA data
    from ml.realized_da import _fetch_raw_da
    da_df = _fetch_raw_da("2025-01", "onpeak")
    if len(da_df) == 0:
        pytest.skip("No DA data available for 2025-01")

    result = map_da_to_branches(da_df, bmap)
    assert "branch_name" in result.columns
    assert "realized_sp" in result.columns

    # Check coverage: matched value / total value
    total_value = da_df["shadow_price"].abs().sum()
    matched_value = result["realized_sp"].sum()
    coverage = matched_value / total_value if total_value > 0 else 0
    print(f"DA value coverage: {coverage:.1%}")
    assert coverage > 0.90, f"Branch mapping coverage too low: {coverage:.1%}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest ml/tests/test_branch_mapping.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `ml/branch_mapping.py`**

```python
# ml/branch_mapping.py
"""PJM constraint_id → branch_name mapping via constraint_info.

The naive join (constraint_id.split(":")[0] → DA monitored_facility) captures
only ~46% of DA binding value. The branch-level join via constraint_info
captures 96-99%.

Reference: research-spice-shadow-price-pred/src/shadow_price_prediction/data_loader.py:805

How it works:
  1. constraint_info maps each constraint_id to a branch_name
  2. constraint_id.split(":")[0] → monitored_facility → .upper() → match_str
  3. DA monitored_facility (uppercased) matches match_str → branch_name
  4. Interface fallback: prefix-match for interface contingencies
  5. Aggregate realized_sp by branch_name
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from ml.config import SPICE6_CI_BASE


def load_constraint_info(
    auction_month: str,
    period_type: str = "f0",
) -> pl.DataFrame:
    """Load constraint_info for an auction month.

    constraint_info is stored only under class_type=onpeak (by design —
    it's physical topology, class-invariant).
    """
    path = (
        Path(SPICE6_CI_BASE)
        / f"auction_month={auction_month}"
        / "market_round=1"
        / f"period_type={period_type}"
        / "class_type=onpeak"
    )
    if not path.exists():
        return pl.DataFrame()

    # Read all parquet files in the directory
    parquet_files = list(path.glob("*.parquet"))
    if not parquet_files:
        return pl.DataFrame()

    dfs = [pl.read_parquet(str(f)) for f in parquet_files]
    df = pl.concat(dfs) if len(dfs) > 1 else dfs[0]

    # Ensure constraint_id is string
    if "constraint_id" in df.columns:
        df = df.with_columns(pl.col("constraint_id").cast(pl.String))

    return df


def build_branch_map(ci: pl.DataFrame) -> pl.DataFrame:
    """Build branch mapping from constraint_info.

    Returns DataFrame with columns: constraint_id, branch_name, match_str, type.
    match_str = monitored_facility.upper() (extracted from constraint_id).
    """
    if len(ci) == 0:
        return pl.DataFrame(schema={
            "constraint_id": pl.String,
            "branch_name": pl.String,
            "match_str": pl.String,
            "type": pl.String,
        })

    # Extract monitored_facility from constraint_id (split on ":")
    result = ci.select([
        pl.col("constraint_id"),
        pl.col("branch_name"),
        pl.col("type") if "type" in ci.columns else pl.lit("branch_constraint").alias("type"),
    ]).unique()

    # Build match_str: monitored_facility = constraint_id.split(":")[0], uppercased
    result = result.with_columns(
        pl.col("constraint_id")
        .str.split(":")
        .list.first()
        .str.to_uppercase()
        .alias("match_str")
    )

    return result


def map_da_to_branches(
    da_df: pl.DataFrame,
    branch_map: pl.DataFrame,
) -> pl.DataFrame:
    """Map DA shadow prices to branch_names and aggregate.

    Parameters
    ----------
    da_df : pl.DataFrame
        Raw DA data with columns: monitored_facility, shadow_price.
    branch_map : pl.DataFrame
        From build_branch_map(), with columns: branch_name, match_str, type.

    Returns
    -------
    pl.DataFrame
        Columns: branch_name, realized_sp (sum of |shadow_price| per branch).
    """
    if len(da_df) == 0 or len(branch_map) == 0:
        return pl.DataFrame(schema={
            "branch_name": pl.String,
            "realized_sp": pl.Float64,
        })

    # Uppercase DA monitored_facility for matching
    da = da_df.with_columns(
        pl.col("monitored_facility").str.to_uppercase().alias("match_str")
    )

    # Deduplicate branch_map on match_str (take first branch_name per match_str)
    # Separate interface and non-interface entries
    non_interface = branch_map.filter(pl.col("type") != "interface")
    interface_map = branch_map.filter(pl.col("type") == "interface")

    # Step 1: Direct match on match_str (non-interface)
    direct_map = non_interface.select(["match_str", "branch_name"]).unique(subset=["match_str"])
    da = da.join(direct_map, on="match_str", how="left")

    # Step 2: Interface fallback — prefix match for unmatched DA rows
    if len(interface_map) > 0:
        unmatched = da.filter(pl.col("branch_name").is_null())
        if len(unmatched) > 0:
            # For each unmatched DA row, try to prefix-match against interface match_strs
            interface_strs = interface_map.select(["match_str", "branch_name"]).unique(subset=["match_str"])
            interface_prefixes = interface_strs["match_str"].to_list()
            interface_branch_names = interface_strs["branch_name"].to_list()

            # Build a mapping: for each unmatched match_str, find interface prefix
            unmatched_strs = unmatched["match_str"].unique().to_list()
            prefix_map = {}
            for ums in unmatched_strs:
                # Try splitting on space and matching the first word
                first_word = ums.split(" ")[0] if " " in ums else ums
                for ip, ibn in zip(interface_prefixes, interface_branch_names):
                    if first_word == ip or ums.startswith(ip + " "):
                        prefix_map[ums] = ibn
                        break

            if prefix_map:
                prefix_df = pl.DataFrame({
                    "match_str": list(prefix_map.keys()),
                    "interface_branch": list(prefix_map.values()),
                })
                da = da.join(prefix_df, on="match_str", how="left")
                da = da.with_columns(
                    pl.col("branch_name").fill_null(pl.col("interface_branch"))
                )
                if "interface_branch" in da.columns:
                    da = da.drop("interface_branch")

    # Aggregate: sum |shadow_price| by branch_name
    result = (
        da.filter(pl.col("branch_name").is_not_null())
        .group_by("branch_name")
        .agg(pl.col("shadow_price").abs().sum().alias("realized_sp"))
    )

    return result
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest ml/tests/test_branch_mapping.py -v
```

Expected: All PASS. The coverage test should show >90%.

- [ ] **Step 5: Commit**

```bash
git add ml/branch_mapping.py ml/tests/test_branch_mapping.py
git commit -m "feat: add PJM branch mapping (constraint_id → branch_name via constraint_info)"
```

---

### Task 4: Create `ml/realized_da.py` — Fetch and cache realized DA by branch_name

PJM realized DA is fetched via `PjmApTools`, aggregated by `branch_name` (not `constraint_id`), and cached as parquet.

**Files:**
- Create: `ml/realized_da.py`
- Test: `ml/tests/test_realized_da.py`

- [ ] **Step 1: Write realized DA test**

```python
# ml/tests/test_realized_da.py
"""Tests for PJM realized DA loader."""
import polars as pl
import pytest
from pathlib import Path
from ml.realized_da import load_realized_da, fetch_and_cache_month, _fetch_raw_da


def test_fetch_raw_da_returns_data():
    """Raw DA fetch should return monitored_facility + shadow_price."""
    # This requires Ray — skip if not available
    try:
        df = _fetch_raw_da("2024-06", "onpeak")
    except Exception:
        pytest.skip("Ray not available or DA fetch failed")
    assert isinstance(df, pl.DataFrame)
    assert "monitored_facility" in df.columns
    assert "shadow_price" in df.columns
    assert len(df) > 0


def test_load_realized_da_has_branch_name():
    """Cached realized DA should have branch_name and realized_sp columns."""
    cache_dir = str(Path(__file__).resolve().parent.parent.parent / "data" / "realized_da")
    try:
        df = load_realized_da("2024-06", peak_type="onpeak", cache_dir=cache_dir)
    except FileNotFoundError:
        pytest.skip("No cached data for 2024-06")
    assert "branch_name" in df.columns
    assert "realized_sp" in df.columns
    assert df["realized_sp"].dtype == pl.Float64
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest ml/tests/test_realized_da.py -v
```

Expected: FAIL

- [ ] **Step 3: Write `ml/realized_da.py`**

```python
# ml/realized_da.py
"""PJM realized DA shadow price loader and fetcher.

Key difference from MISO: PJM realized DA is aggregated by branch_name
(via branch mapping from constraint_info), not by constraint_id.

load_realized_da    -- read a cached month from parquet
fetch_and_cache_month -- fetch from PJM API via Ray, map to branches, cache
_fetch_raw_da       -- raw DA fetch (no branch mapping)

REQUIRES RAY for fetch_and_cache_month.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

from ml.config import REALIZED_DA_CACHE


def _cache_path(month: str, peak_type: str, cache_dir: str) -> Path:
    """Return cache file path."""
    if peak_type == "onpeak":
        return Path(cache_dir) / f"{month}.parquet"
    return Path(cache_dir) / f"{month}_{peak_type}.parquet"


def load_realized_da(
    month: str,
    peak_type: str = "onpeak",
    cache_dir: str = REALIZED_DA_CACHE,
) -> pl.DataFrame:
    """Read cached realized DA shadow prices for a month.

    Returns
    -------
    pl.DataFrame
        Columns: [branch_name (String), realized_sp (Float64)]
    """
    p = _cache_path(month, peak_type, cache_dir)
    if not p.exists():
        raise FileNotFoundError(f"No cached realized DA for {month}/{peak_type}: {p}")
    df = pl.read_parquet(str(p))
    return df.select(
        pl.col("branch_name").cast(pl.String),
        pl.col("realized_sp").cast(pl.Float64),
    )


def _fetch_raw_da(month: str, peak_type: str) -> pl.DataFrame:
    """Fetch raw PJM DA shadow prices for one month.

    Returns polars DataFrame with columns: monitored_facility, shadow_price.
    """
    from pbase.analysis.tools.all_positions import PjmApTools

    st = pd.Timestamp(f"{month}-01")
    et = st + pd.offsets.MonthBegin(1)

    aptools = PjmApTools()
    da_shadow = aptools.tools.get_da_shadow_by_peaktype(
        st=st, et_ex=et, peak_type=peak_type,
    )

    if da_shadow is None or len(da_shadow) == 0:
        return pl.DataFrame(schema={
            "monitored_facility": pl.String,
            "shadow_price": pl.Float64,
        })

    return pl.from_pandas(da_shadow.reset_index()).select([
        pl.col("monitored_facility").cast(pl.String),
        pl.col("shadow_price").cast(pl.Float64),
    ])


def fetch_and_cache_month(
    month: str,
    peak_type: str = "onpeak",
    cache_dir: str = REALIZED_DA_CACHE,
    period_type: str = "f0",
) -> Path:
    """Fetch realized DA, map to branches via constraint_info, and cache.

    The branch mapping uses constraint_info for the given month. If
    constraint_info is unavailable, falls back to a nearby month.

    Returns path to cached parquet with columns: branch_name, realized_sp.
    """
    from ml.branch_mapping import load_constraint_info, build_branch_map, map_da_to_branches

    out_path = _cache_path(month, peak_type, cache_dir)
    if out_path.exists():
        return out_path

    # Fetch raw DA
    raw_da = _fetch_raw_da(month, peak_type)

    if len(raw_da) == 0:
        df = pl.DataFrame(schema={
            "branch_name": pl.String,
            "realized_sp": pl.Float64,
        })
    else:
        # Load constraint_info for branch mapping
        # Try the target month first, then fall back to nearby months
        ci = load_constraint_info(month, period_type=period_type)
        if len(ci) == 0:
            # Try adjacent months
            for offset in [1, -1, 2, -2]:
                alt = (pd.Timestamp(month) + pd.DateOffset(months=offset)).strftime("%Y-%m")
                ci = load_constraint_info(alt, period_type=period_type)
                if len(ci) > 0:
                    print(f"[realized_da] Using constraint_info from {alt} for {month}")
                    break

        if len(ci) == 0:
            print(f"[realized_da] WARNING: no constraint_info for {month}, using raw aggregation")
            # Fallback: aggregate by monitored_facility directly
            df = (
                raw_da
                .group_by("monitored_facility")
                .agg(pl.col("shadow_price").abs().sum().alias("realized_sp"))
                .rename({"monitored_facility": "branch_name"})
            )
        else:
            bmap = build_branch_map(ci)
            df = map_da_to_branches(raw_da, bmap)

    # Write atomically
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    df.write_parquet(str(tmp))
    tmp.rename(out_path)

    print(f"[realized_da] Cached {month}/{peak_type}: {len(df)} branches")
    return out_path
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest ml/tests/test_realized_da.py -v
```

Expected: PASS (may skip tests if Ray is not available or data not cached yet).

- [ ] **Step 5: Commit**

```bash
git add ml/realized_da.py ml/tests/test_realized_da.py
git commit -m "feat: add PJM realized DA with branch-level aggregation"
```

---

### Task 5: Create `ml/spice6_loader.py` — PJM spice6 density features

Adapted from MISO's `spice6_loader.py`. Only the base path changes.

**Files:**
- Create: `ml/spice6_loader.py`

- [ ] **Step 1: Write `ml/spice6_loader.py`**

Copy from `research-stage5-tier/ml/spice6_loader.py` and change the import path:

```python
# ml/spice6_loader.py
"""Load spice6 density features for PJM.

Identical logic to MISO spice6_loader.py, only the base path changes.
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from ml.config import SPICE6_DENSITY_BASE, delivery_month as _delivery_month

logger = logging.getLogger(__name__)


def load_spice6_density(
    auction_month: str,
    period_type: str = "f0",
) -> pl.DataFrame:
    """Load and aggregate spice6 density features for one month.

    Returns
    -------
    pl.DataFrame
        Columns: constraint_id, flow_direction, prob_exceed_110, prob_exceed_100,
        prob_exceed_90, prob_exceed_85, prob_exceed_80, constraint_limit.
    """
    market_month = _delivery_month(auction_month, period_type)
    logger.info("spice6 density: auction=%s ptype=%s market_month=%s",
                auction_month, period_type, market_month)
    market_round = "1"
    base = (
        Path(SPICE6_DENSITY_BASE)
        / f"auction_month={auction_month}"
        / f"market_month={market_month}"
        / f"market_round={market_round}"
    )

    if not base.exists():
        return pl.DataFrame()

    score_dfs = []
    limit_dfs = []
    use_legacy_schema = None

    for od_dir in sorted(base.iterdir()):
        if not od_dir.name.startswith("outage_date="):
            continue
        score_path = od_dir / "score_df.parquet"
        if score_path.exists():
            if use_legacy_schema is None:
                use_legacy_schema = True
        else:
            score_path = od_dir / "score.parquet"
            if use_legacy_schema is None:
                use_legacy_schema = False
        limit_path = od_dir / "limit.parquet"
        if score_path.exists():
            score_dfs.append(pl.read_parquet(score_path))
        if limit_path.exists():
            limit_dfs.append(pl.read_parquet(limit_path))

    if not score_dfs:
        return pl.DataFrame()

    all_scores = pl.concat(score_dfs)

    if use_legacy_schema:
        density = all_scores.group_by(["constraint_id", "flow_direction"]).agg([
            pl.col("110").mean().alias("prob_exceed_110"),
            pl.col("100").mean().alias("prob_exceed_100"),
            pl.col("90").mean().alias("prob_exceed_90"),
            pl.col("85").mean().alias("prob_exceed_85"),
            pl.col("80").mean().alias("prob_exceed_80"),
        ])
    else:
        density = all_scores.group_by(["constraint_id", "flow_direction"]).agg([
            pl.col("score").mean().alias("prob_exceed_110"),
        ])
        for col in ["prob_exceed_100", "prob_exceed_90", "prob_exceed_85", "prob_exceed_80"]:
            density = density.with_columns(pl.lit(0.0).alias(col))

    if limit_dfs:
        all_limits = pl.concat(limit_dfs)
        limits = all_limits.group_by("constraint_id").agg(
            pl.col("limit").mean().alias("constraint_limit")
        )
        density = density.join(limits, on="constraint_id", how="left")
    else:
        density = density.with_columns(pl.lit(0.0).alias("constraint_limit"))

    return density
```

- [ ] **Step 2: Verify spice6 loads for a known month**

```bash
python -c "
from ml.spice6_loader import load_spice6_density
df = load_spice6_density('2025-01', 'f0')
print(f'Loaded {len(df)} rows, columns: {df.columns}')
print(df.head(3))
"
```

Expected: prints rows with prob_exceed_110, constraint_limit, etc.

- [ ] **Step 3: Commit**

```bash
git add ml/spice6_loader.py
git commit -m "feat: add PJM spice6 density loader"
```

---

### Task 6: Create `ml/features.py` — Feature preparation

Copy from MISO with the `v7_formula_score` derived feature. The binding frequency enrichment will be done in the script layer (same as MISO's `run_v10e_lagged.py`), not in this module.

**Files:**
- Create: `ml/features.py`

- [ ] **Step 1: Write `ml/features.py`**

```python
# ml/features.py
"""Feature preparation for PJM LTR pipeline.

Identical to MISO features.py. Binding frequency enrichment happens in
the script layer (run_v2_ml.py), not here.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ml.config import LTRConfig


def _add_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add v62b_formula_score if not already present."""
    if "v62b_formula_score" not in df.columns:
        has_cols = all(c in df.columns for c in
                       ["da_rank_value", "density_mix_rank_value", "density_ori_rank_value"])
        if has_cols:
            df = df.with_columns(
                (0.60 * pl.col("da_rank_value")
                 + 0.30 * pl.col("density_mix_rank_value")
                 + 0.10 * pl.col("density_ori_rank_value")
                ).alias("v62b_formula_score")
            )
    return df


def prepare_features(
    df: pl.DataFrame,
    cfg: LTRConfig,
) -> tuple[np.ndarray, list[int]]:
    """Extract feature matrix from df, fill nulls with 0."""
    df = _add_derived_features(df)
    cols = list(cfg.features)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[features] WARNING: {len(missing)} features missing, filling with 0: {missing}")

    X_parts = []
    for c in cols:
        if c in df.columns:
            X_parts.append(df[c].fill_null(0.0).to_numpy().astype(np.float64))
        else:
            X_parts.append(np.zeros(len(df), dtype=np.float64))

    X = np.column_stack(X_parts) if X_parts else np.zeros((len(df), 0))
    return X, list(cfg.monotone_constraints)


def compute_query_groups(df: pl.DataFrame) -> np.ndarray:
    """Compute query group sizes from query_month column."""
    months = df["query_month"].to_list()
    groups = []
    if not months:
        return np.array([], dtype=np.int32)

    current = months[0]
    count = 0
    for m in months:
        if m == current:
            count += 1
        else:
            groups.append(count)
            current = m
            count = 1
    groups.append(count)
    return np.array(groups, dtype=np.int32)
```

- [ ] **Step 2: Commit**

```bash
git add ml/features.py
git commit -m "feat: add PJM feature preparation module"
```

---

### Task 7: Create `ml/data_loader.py` — V6.2B + spice6 + realized DA

The data loader joins V6.2B signal data with spice6 density features and realized DA ground truth. **Key PJM change**: realized DA is joined on `branch_name` (V6.2B already has this column), not `constraint_id`.

**Files:**
- Create: `ml/data_loader.py`

- [ ] **Step 1: Write `ml/data_loader.py`**

```python
# ml/data_loader.py
"""Data loading for PJM LTR ranking pipeline.

Loads V6.2B signal data, enriches with spice6 density features,
and joins realized DA shadow prices as ground truth.

KEY PJM DIFFERENCE: ground truth joins on branch_name (not constraint_id).
V6.2B parquet already has a branch_name column.
"""
from __future__ import annotations

import resource
from pathlib import Path

import polars as pl

from ml.config import V62B_SIGNAL_BASE
from ml.realized_da import load_realized_da
from ml.spice6_loader import load_spice6_density


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


_MONTH_CACHE: dict[tuple[str, str, str, str | None], pl.DataFrame] = {}


def clear_month_cache() -> None:
    _MONTH_CACHE.clear()


def load_v62b_month(
    auction_month: str,
    period_type: str = "f0",
    class_type: str = "onpeak",
    cache_dir: str | None = None,
) -> pl.DataFrame:
    """Load V6.2B signal data enriched with spice6 density + realized DA.

    PJM-specific: realized DA is joined on branch_name, not constraint_id.
    V6.2B parquet already contains a branch_name column.
    """
    cache_key = (auction_month, period_type, class_type, cache_dir)
    if cache_key in _MONTH_CACHE:
        return _MONTH_CACHE[cache_key]

    path = Path(V62B_SIGNAL_BASE) / auction_month / period_type / class_type
    if not path.exists():
        raise FileNotFoundError(f"V6.2B data not found: {path}")
    df = pl.read_parquet(str(path))
    if "__index_level_0__" in df.columns:
        df = df.drop("__index_level_0__")

    # Ensure constraint_id and branch_name are strings
    df = df.with_columns(
        pl.col("constraint_id").cast(pl.String),
        pl.col("branch_name").cast(pl.String),
    )

    # Enrich with spice6 density features
    spice6 = load_spice6_density(auction_month, period_type)
    if len(spice6) > 0:
        df = df.join(spice6, on=["constraint_id", "flow_direction"], how="left")
        spice6_cols = [c for c in spice6.columns if c not in ("constraint_id", "flow_direction")]
        df = df.with_columns([pl.col(c).fill_null(0.0) for c in spice6_cols])
        n_matched = len(df.filter(pl.col("prob_exceed_110") > 0))
        print(f"[data_loader] spice6: {n_matched}/{len(df)} matched")
    else:
        print(f"[data_loader] WARNING: no spice6 data for {auction_month}")
        for col in ["prob_exceed_110", "prob_exceed_100", "prob_exceed_90",
                     "prob_exceed_85", "prob_exceed_80", "constraint_limit"]:
            df = df.with_columns(pl.lit(0.0).alias(col))

    # Join realized DA ground truth on branch_name
    from ml.config import delivery_month as _delivery_month
    gt_month = _delivery_month(auction_month, period_type)

    # Map class_type to peak_type for DA fetch
    peak_type = class_type  # PJM uses same names: onpeak, dailyoffpeak, wkndonpeak
    da_kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    try:
        realized = load_realized_da(gt_month, peak_type=peak_type, **da_kwargs)
        # Join on branch_name (PJM-specific)
        df = df.join(realized, on="branch_name", how="left")
        df = df.with_columns(pl.col("realized_sp").fill_null(0.0))
        n_binding = len(df.filter(pl.col("realized_sp") > 0))
        print(f"[data_loader] realized DA: {n_binding}/{len(df)} binding for {auction_month} "
              f"(gt_month={gt_month})")
    except FileNotFoundError:
        print(f"[data_loader] WARNING: no realized DA for {gt_month}/{peak_type}")
        df = df.with_columns(pl.lit(0.0).alias("realized_sp"))

    _MONTH_CACHE[cache_key] = df
    return df
```

- [ ] **Step 2: Smoke-test the data loader**

```bash
python -c "
from ml.data_loader import load_v62b_month
df = load_v62b_month('2023-06', 'f0', 'onpeak')
print(f'Loaded: {len(df)} rows, {df.columns}')
n_binding = len(df.filter(df['realized_sp'] > 0))
print(f'Binding: {n_binding}/{len(df)}')
"
```

Expected: prints row count, columns including `realized_sp`, and binding count.
Note: This will only work after realized DA is cached (Task 9). If not cached, it falls back to realized_sp=0.

- [ ] **Step 3: Commit**

```bash
git add ml/data_loader.py
git commit -m "feat: add PJM data loader with branch-level DA join"
```

---

## Chunk 2: Scripts (Cache, V0 Baseline, V2 ML, Blend, Holdout)

### Task 8: Create `ml/pipeline.py` — single-month pipeline

Adapted from MISO's pipeline.py. Used by the benchmark harness.

**Files:**
- Create: `ml/pipeline.py`

- [ ] **Step 1: Write `ml/pipeline.py`**

Copy from `research-stage5-tier/ml/pipeline.py` — the only change is the import paths (now `from ml.config` instead of cross-repo). The code is identical since `data_loader.py` already handles the PJM-specific branch join.

```python
# ml/pipeline.py
"""LTR pipeline: load -> features -> train -> predict -> evaluate."""
from __future__ import annotations

import gc
import resource
from typing import Any

import numpy as np
import polars as pl

from ml.config import PipelineConfig
from ml.data_loader import load_v62b_month
from ml.evaluate import evaluate_ltr
from ml.features import compute_query_groups, prepare_features
from ml.train import predict_scores, train_ltr_model


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_pipeline(
    config: PipelineConfig,
    version_id: str,
    eval_month: str,
    period_type: str = "f0",
    class_type: str = "onpeak",
) -> dict[str, Any]:
    """Run the LTR pipeline for a single evaluation month."""
    print(f"[pipeline] version={version_id} eval_month={eval_month}")

    import pandas as pd
    eval_ts = pd.Timestamp(eval_month)
    total_lookback = config.train_months + config.val_months

    train_month_strs = []
    for i in range(total_lookback, config.val_months, -1):
        m = (eval_ts - pd.DateOffset(months=i)).strftime("%Y-%m")
        train_month_strs.append(m)

    dfs = []
    for m in train_month_strs:
        try:
            df = load_v62b_month(m, period_type, class_type)
            df = df.with_columns(pl.lit(m).alias("query_month"))
            dfs.append(df)
        except FileNotFoundError:
            print(f"[pipeline] WARNING: skipping {m}")
    if not dfs:
        return {"metrics": {}}
    train_df = pl.concat(dfs)

    test_df = load_v62b_month(eval_month, period_type, class_type)
    test_df = test_df.with_columns(pl.lit(eval_month).alias("query_month"))

    train_df = train_df.sort("query_month")
    X_train, _ = prepare_features(train_df, config.ltr)
    y_train = train_df["realized_sp"].to_numpy().astype(np.float64)
    groups_train = compute_query_groups(train_df)

    del dfs
    gc.collect()

    model = train_ltr_model(X_train, y_train, groups_train, config.ltr)
    del X_train, y_train, groups_train
    gc.collect()

    X_test, _ = prepare_features(test_df, config.ltr)
    scores = predict_scores(model, X_test)
    actual_sp = test_df["realized_sp"].to_numpy().astype(np.float64)

    metrics = evaluate_ltr(actual_sp, scores)

    feat_names = config.ltr.features
    if hasattr(model, "feature_importance"):
        importance = model.feature_importance(importance_type="gain")
    else:
        importance = model.feature_importances_
    metrics["_feature_importance"] = {
        name: float(imp)
        for name, imp in sorted(zip(feat_names, importance), key=lambda x: x[1], reverse=True)
    }

    del X_test, scores, actual_sp, test_df, model
    gc.collect()

    for key, value in metrics.items():
        if key.startswith("_"):
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")

    return {"metrics": metrics}
```

- [ ] **Step 2: Commit**

```bash
git add ml/pipeline.py
git commit -m "feat: add PJM pipeline module"
```

---

### Task 9: Create `scripts/cache_realized_da.py` — Preflight DA cache

This script must be run FIRST (requires Ray) to cache all realized DA months needed for evaluation. It fetches PJM DA via `PjmApTools`, maps to branches, and caches.

**Files:**
- Create: `scripts/cache_realized_da.py`

- [ ] **Step 1: Write the cache script**

```python
#!/usr/bin/env python
"""Cache PJM realized DA shadow prices for all months needed by the ML pipeline.

Must be run before any experiment scripts. Requires Ray connection.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python scripts/cache_realized_da.py --peak-types onpeak dailyoffpeak wkndonpeak
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import REALIZED_DA_CACHE, _FULL_EVAL_MONTHS, HOLDOUT_MONTHS
from ml.realized_da import fetch_and_cache_month, _cache_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peak-types", nargs="+",
                        default=["onpeak", "dailyoffpeak", "wkndonpeak"])
    parser.add_argument("--start", default="2019-01",
                        help="First month to cache (YYYY-MM)")
    parser.add_argument("--end", default="2026-02",
                        help="Last month to cache (YYYY-MM)")
    parser.add_argument("--cache-dir", default=REALIZED_DA_CACHE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Init Ray
    os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"
    from pbase.config.ray import init_ray
    import pmodel
    init_ray(extra_modules=[pmodel])

    import pandas as pd

    # Generate all months in range
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    months = []
    current = start
    while current <= end:
        months.append(current.strftime("%Y-%m"))
        current += pd.DateOffset(months=1)

    print(f"[cache] Caching {len(months)} months × {len(args.peak_types)} peak types")
    print(f"[cache] Cache dir: {args.cache_dir}")

    if args.dry_run:
        # Count existing vs missing
        existing = sum(
            1 for m in months for pt in args.peak_types
            if _cache_path(m, pt, args.cache_dir).exists()
        )
        total = len(months) * len(args.peak_types)
        print(f"[cache] DRY RUN: {existing}/{total} already cached, {total - existing} to fetch")
        return

    t0 = time.time()
    fetched = 0
    skipped = 0
    already_cached = 0

    for pt in args.peak_types:
        for m in months:
            p = _cache_path(m, pt, args.cache_dir)
            if p.exists():
                already_cached += 1
                continue
            try:
                fetch_and_cache_month(m, pt, cache_dir=args.cache_dir)
                fetched += 1
            except Exception as e:
                print(f"[cache] SKIP {m}/{pt}: {e}")
                skipped += 1

    elapsed = time.time() - t0
    print(f"\n[cache] Done in {elapsed:.0f}s: {fetched} fetched, "
          f"{already_cached} cached, {skipped} skipped")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the cache script**

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
python scripts/cache_realized_da.py --dry-run
```

Expected: prints count of existing vs missing files. Then run without `--dry-run` to actually fetch.

- [ ] **Step 3: Commit**

```bash
git add scripts/cache_realized_da.py
git commit -m "feat: add realized DA cache script for PJM (branch-level)"
```

---

### Task 10: Create `scripts/run_v0_formula_baseline.py`

Run V6.2B formula baseline for all 6 ML slices. Establishes gates and champion.

**Files:**
- Create: `scripts/run_v0_formula_baseline.py`

- [ ] **Step 1: Write the v0 baseline script**

Adapt from `research-stage5-tier/scripts/run_v0_formula_baseline.py`. Key changes:
- Loop over 3 class types instead of 2
- Use `branch_name` for realized DA join (already handled by `data_loader.py`)
- PJM auction schedule for f1 filtering

```python
#!/usr/bin/env python
"""V0 formula baseline for PJM: evaluate V6.2B formula against realized DA.

Runs all 6 ML slices: f0×{onpeak,dailyoffpeak,wkndonpeak} + f1×{same}.
Saves to registry/{ptype}/{ctype}/v0/ and calibrates gates.json + champion.json.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
    python scripts/run_v0_formula_baseline.py
    python scripts/run_v0_formula_baseline.py --ptype f0 --class-type onpeak  # single slice
    python scripts/run_v0_formula_baseline.py --holdout  # include holdout
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    V62B_SIGNAL_BASE, PJM_CLASS_TYPES, _FULL_EVAL_MONTHS,
    HOLDOUT_MONTHS, has_period_type,
)
from ml.evaluate import aggregate_months, evaluate_ltr
from ml.realized_da import load_realized_da
from ml.registry_paths import registry_root, holdout_root
from ml.v62b_formula import v62b_score

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "registry"
HOLDOUT_DIR = ROOT / "holdout"


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def evaluate_month(month: str, class_type: str, period_type: str) -> dict:
    """Evaluate V6.2B formula on one month against realized DA."""
    from ml.config import delivery_month as _delivery_month

    path = Path(V62B_SIGNAL_BASE) / month / period_type / class_type
    df = pl.read_parquet(str(path))
    df = df.with_columns(pl.col("branch_name").cast(pl.String))

    gt_month = _delivery_month(month, period_type)
    realized = load_realized_da(gt_month, peak_type=class_type)
    df = df.join(realized, on="branch_name", how="left")
    df = df.with_columns(pl.col("realized_sp").fill_null(0.0))

    actual = df["realized_sp"].to_numpy().astype(np.float64)

    # Negate because lower rank_value = more binding, but eval expects higher = better
    scores = -v62b_score(
        da_rank_value=df["da_rank_value"].to_numpy(),
        density_mix_rank_value=df["density_mix_rank_value"].to_numpy(),
        density_ori_rank_value=df["density_ori_rank_value"].to_numpy(),
    )

    metrics = evaluate_ltr(actual, scores)
    n_binding = int((actual > 0).sum())
    print(f"  {month}: n={len(df)}, binding={n_binding}, "
          f"VC@20={metrics['VC@20']:.4f}, VC@100={metrics['VC@100']:.4f}")

    del df, realized, actual, scores
    gc.collect()
    return metrics


def build_gates(agg: dict) -> dict:
    group_a = ["VC@20", "VC@100", "Recall@20", "Recall@50", "Recall@100", "NDCG"]
    gates = {}
    means = agg["mean"]
    mins = agg["min"]
    for metric in group_a:
        if metric in means:
            gates[metric] = {
                "floor": round(0.9 * means[metric], 4),
                "tail_floor": round(mins[metric], 4),
                "direction": "higher",
                "group": "A",
            }
    return {
        "gates": gates,
        "noise_tolerance": 0.02,
        "tail_max_failures": 1,
        "calibrated_from": "v0",
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
    }


def run_slice(
    period_type: str,
    class_type: str,
    eval_months: list[str],
    holdout: bool = False,
):
    """Run v0 for one (ptype, ctype) slice."""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"[v0] {period_type}/{class_type} ({len(eval_months)} eval months)")
    print(f"{'='*60}")

    per_month: dict[str, dict] = {}
    for month in eval_months:
        try:
            per_month[month] = evaluate_month(month, class_type, period_type)
        except FileNotFoundError as e:
            print(f"  {month}: SKIP ({e})")

    if not per_month:
        print(f"[v0] No valid months for {period_type}/{class_type}")
        return

    agg = aggregate_months(per_month)
    means = agg["mean"]
    print(f"\n[v0] Aggregate ({len(per_month)} months):")
    for metric in ["VC@20", "VC@50", "VC@100", "Recall@20", "NDCG", "Spearman"]:
        print(f"  {metric:<12} {means.get(metric, 0):.4f}")

    # Save to registry
    v0_dir = registry_root(period_type, class_type, base_dir=REGISTRY) / "v0"
    v0_dir.mkdir(parents=True, exist_ok=True)

    metrics_out = {
        "eval_config": {"eval_months": sorted(per_month.keys()), "class_type": class_type,
                        "period_type": period_type, "mode": "eval"},
        "per_month": per_month, "aggregate": agg,
        "n_months": len(per_month), "n_months_requested": len(eval_months),
        "skipped_months": sorted(set(eval_months) - set(per_month.keys())),
    }
    with open(v0_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    config_out = {
        "method": "v62b_formula",
        "formula": "-(0.60*da_rank_value + 0.30*density_mix_rank_value + 0.10*density_ori_rank_value)",
        "ground_truth": f"realized_da (branch-level sum, {class_type})",
    }
    with open(v0_dir / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    # Gates + champion
    slice_dir = registry_root(period_type, class_type, base_dir=REGISTRY)
    gates_data = build_gates(agg)
    with open(slice_dir / "gates.json", "w") as f:
        json.dump(gates_data, f, indent=2)

    champion_data = {
        "version": "v0",
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "reason": f"initial baseline ({period_type}/{class_type})",
    }
    with open(slice_dir / "champion.json", "w") as f:
        json.dump(champion_data, f, indent=2)

    print(f"[v0] Saved to {v0_dir} ({time.time()-t0:.1f}s)")

    # Holdout
    if holdout:
        ho_months = [m for m in HOLDOUT_MONTHS if has_period_type(m, period_type)]
        print(f"\n[v0] Holdout ({len(ho_months)} months)...")
        ho_pm: dict[str, dict] = {}
        for month in ho_months:
            try:
                ho_pm[month] = evaluate_month(month, class_type, period_type)
            except FileNotFoundError:
                print(f"  {month}: SKIP")

        if ho_pm:
            ho_agg = aggregate_months(ho_pm)
            ho_dir = holdout_root(period_type, class_type, base_dir=HOLDOUT_DIR) / "v0"
            ho_dir.mkdir(parents=True, exist_ok=True)
            ho_out = {
                "eval_config": {"eval_months": sorted(ho_pm.keys()), "class_type": class_type,
                                "period_type": period_type, "mode": "holdout"},
                "per_month": ho_pm, "aggregate": ho_agg,
                "n_months": len(ho_pm), "n_months_requested": len(ho_months),
                "skipped_months": sorted(set(ho_months) - set(ho_pm.keys())),
            }
            with open(ho_dir / "metrics.json", "w") as f:
                json.dump(ho_out, f, indent=2)
            ho_means = ho_agg["mean"]
            print(f"[v0] Holdout aggregate ({len(ho_pm)} months):")
            for metric in ["VC@20", "VC@50", "VC@100", "Recall@20", "NDCG"]:
                print(f"  {metric:<12} {ho_means.get(metric, 0):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptype", default=None, help="Period type (f0 or f1). Default: both.")
    parser.add_argument("--class-type", default=None, help="Class type. Default: all 3.")
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--full", action="store_true", help="Use 36 eval months")
    args = parser.parse_args()

    ptypes = [args.ptype] if args.ptype else ["f0", "f1"]
    ctypes = [args.class_type] if args.class_type else PJM_CLASS_TYPES

    for ptype in ptypes:
        eval_months = _FULL_EVAL_MONTHS if args.full else _FULL_EVAL_MONTHS
        eval_months = [m for m in eval_months if has_period_type(m, ptype)]
        for ctype in ctypes:
            run_slice(ptype, ctype, eval_months, holdout=args.holdout)

    print("\n[v0] All slices complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run v0 for a single slice first**

```bash
python scripts/run_v0_formula_baseline.py --ptype f0 --class-type onpeak
```

Expected: prints per-month metrics, aggregate, saves to `registry/f0/onpeak/v0/metrics.json`.

- [ ] **Step 3: Run v0 for all 6 slices**

```bash
python scripts/run_v0_formula_baseline.py --holdout
```

Expected: saves v0 results for all 6 slices to registry/ and holdout/.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_v0_formula_baseline.py registry/ holdout/
git commit -m "feat: v0 formula baseline for all 6 PJM slices"
```

---

### Task 11: Create `scripts/run_v2_ml.py` — ML model with binding frequency

The v2 ML script trains LightGBM LambdaRank with 9 features (5 binding freq + v7_formula_score + prob_exceed_110 + constraint_limit + da_rank_value).

**Key**: Binding frequency uses `branch_name` keys (not constraint_id). BF_LAG=1 always.

**Files:**
- Create: `scripts/run_v2_ml.py`

- [ ] **Step 1: Write the v2 ML script**

Adapt from `research-stage5-tier/scripts/run_v10e_lagged.py`. Key PJM changes:
- `load_all_binding_sets()` reads branch_name from cached parquet (not constraint_id)
- `compute_bf()` operates on branch_name
- Loop over 3 class types
- Per-slice blend weights (start with MISO defaults, will tune in blend search)

```python
#!/usr/bin/env python
"""V2 ML model for PJM: LightGBM LambdaRank with 9 features.

Runs dev (36mo) and holdout (24mo) for all 6 ML slices.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
    python scripts/run_v2_ml.py --ptype f0 --class-type onpeak
    python scripts/run_v2_ml.py  # all 6 slices
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    REALIZED_DA_CACHE, LTRConfig, PipelineConfig, V10E_FEATURES, V10E_MONOTONE,
    _FULL_EVAL_MONTHS, HOLDOUT_MONTHS, PJM_CLASS_TYPES,
    has_period_type, collect_usable_months,
)
from ml.data_loader import load_v62b_month
from ml.evaluate import aggregate_months, evaluate_ltr
from ml.features import compute_query_groups, prepare_features
from ml.registry_paths import registry_root, holdout_root
from ml.train import predict_scores, train_ltr_model

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "registry"
HOLDOUT_DIR = ROOT / "holdout"

# Per-(period_type, class_type) blend weights for v7_formula_score.
# Start with MISO defaults. Will be tuned by blend search script.
BLEND_WEIGHTS: dict[tuple[str, str], tuple[float, float, float]] = {
    ("f0", "onpeak"): (0.85, 0.00, 0.15),
    ("f0", "dailyoffpeak"): (0.85, 0.00, 0.15),
    ("f0", "wkndonpeak"): (0.85, 0.00, 0.15),
    ("f1", "onpeak"): (0.70, 0.00, 0.30),
    ("f1", "dailyoffpeak"): (0.80, 0.00, 0.20),
    ("f1", "wkndonpeak"): (0.80, 0.00, 0.20),
}
_DEFAULT_BLEND = (0.85, 0.00, 0.15)


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_all_binding_sets(
    peak_type: str = "onpeak",
    cache_dir: str = REALIZED_DA_CACHE,
) -> dict[str, set[str]]:
    """Load all cached DA into {month: set(branch_names)}.

    PJM-specific: keys are branch_name, not constraint_id.
    """
    binding_sets: dict[str, set[str]] = {}
    if peak_type == "onpeak":
        pattern = "[0-9][0-9][0-9][0-9]-[0-9][0-9].parquet"
    else:
        pattern = f"*_{peak_type}.parquet"
    for f in sorted(Path(cache_dir).glob(pattern)):
        df = pl.read_parquet(str(f))
        month = f.stem.replace(f"_{peak_type}", "")
        binding_sets[month] = set(
            df.filter(pl.col("realized_sp") > 0)["branch_name"].to_list()
        )
    print(f"[bf] Loaded {len(binding_sets)} months of {peak_type} binding sets")
    return binding_sets


def compute_bf(
    branch_names: list[str], month: str,
    bs: dict[str, set[str]], lookback: int,
) -> np.ndarray:
    """Compute binding frequency for branch_names."""
    prior = [m for m in sorted(bs.keys()) if m < month][-lookback:]
    n = len(prior)
    if n == 0:
        return np.zeros(len(branch_names), dtype=np.float64)
    freq = np.zeros(len(branch_names), dtype=np.float64)
    for m in prior:
        s = bs[m]
        for i, bn in enumerate(branch_names):
            if bn in s:
                freq[i] += 1
    return freq / n


def prev_month(m: str) -> str:
    ts = pd.Timestamp(m)
    return (ts - pd.DateOffset(months=1)).strftime("%Y-%m")


def enrich_df(
    df: pl.DataFrame, month: str, bs: dict[str, set[str]],
    blend_weights: tuple[float, float, float],
) -> pl.DataFrame:
    """Add binding_freq and formula score features. BF_LAG=1 always."""
    cutoff = prev_month(month)  # BF sees months strictly before M-1
    w_da, w_dmix, w_dori = blend_weights

    # PJM-specific: use branch_name for BF, not constraint_id
    branch_names = df["branch_name"].to_list()

    df = df.with_columns(
        (w_da * pl.col("da_rank_value")
         + w_dmix * pl.col("density_mix_rank_value")
         + w_dori * pl.col("density_ori_rank_value")
        ).alias("v7_formula_score")
    )
    for lb in [1, 3, 6, 12, 15]:
        col_name = f"binding_freq_{lb}"
        if col_name not in df.columns:
            freq = compute_bf(branch_names, cutoff, bs, lb)
            df = df.with_columns(pl.Series(col_name, freq))
    return df


def run_variant(
    label: str,
    eval_months: list[str],
    bs: dict[str, set[str]],
    class_type: str,
    period_type: str,
) -> dict[str, dict]:
    blend = BLEND_WEIGHTS.get((period_type, class_type), _DEFAULT_BLEND)
    print(f"\n[{label}] 9f, {len(eval_months)} months, ptype={period_type}, "
          f"class_type={class_type}, blend={blend}")

    cfg = PipelineConfig(
        ltr=LTRConfig(
            features=list(V10E_FEATURES),
            monotone_constraints=list(V10E_MONOTONE),
            backend="lightgbm",
            label_mode="tiered",
        ),
        train_months=8,
        val_months=0,
    )

    per_month: dict[str, dict] = {}

    for m in eval_months:
        t0 = time.time()
        train_month_strs = collect_usable_months(m, period_type, n_months=8)
        if not train_month_strs:
            print(f"  {m}: SKIP (insufficient history)")
            continue
        train_month_strs = list(reversed(train_month_strs))

        parts = []
        for tm in train_month_strs:
            try:
                part = load_v62b_month(tm, period_type, class_type)
                part = part.with_columns(pl.lit(tm).alias("query_month"))
                part = enrich_df(part, tm, bs, blend)
                parts.append(part)
            except FileNotFoundError:
                pass
        if not parts:
            continue
        train_df = pl.concat(parts)

        try:
            test_df = load_v62b_month(m, period_type, class_type)
        except FileNotFoundError:
            continue
        test_df = test_df.with_columns(pl.lit(m).alias("query_month"))
        test_df = enrich_df(test_df, m, bs, blend)

        train_df = train_df.sort("query_month")
        X_train, _ = prepare_features(train_df, cfg.ltr)
        y_train = train_df["realized_sp"].to_numpy().astype(np.float64)
        groups = compute_query_groups(train_df)

        model = train_ltr_model(X_train, y_train, groups, cfg.ltr)
        X_test, _ = prepare_features(test_df, cfg.ltr)
        scores = predict_scores(model, X_test)
        actual = test_df["realized_sp"].to_numpy().astype(np.float64)

        metrics = evaluate_ltr(actual, scores)
        per_month[m] = metrics

        if hasattr(model, "feature_importance"):
            imp = model.feature_importance(importance_type="gain")
            metrics["_fi"] = dict(zip(cfg.ltr.features, [float(x) for x in imp]))

        elapsed = time.time() - t0
        n_bind = int((actual > 0).sum())
        print(f"  {m}: VC@20={metrics['VC@20']:.4f} binding={n_bind} ({elapsed:.1f}s)")

        del train_df, test_df, parts, X_train, y_train, groups, model, X_test, actual, scores
        gc.collect()

    if per_month:
        clean = {m: {k: v for k, v in met.items() if not k.startswith("_")}
                 for m, met in per_month.items()}
        agg = aggregate_months(clean)
        means = agg["mean"]
        print(f"  => VC@20={means['VC@20']:.4f} VC@100={means['VC@100']:.4f} "
              f"NDCG={means['NDCG']:.4f} Spearman={means['Spearman']:.4f}")

    return per_month


def save_results(label, per_month, eval_months, dest_dir, class_type, period_type):
    clean = {m: {k: v for k, v in met.items() if not k.startswith("_")}
             for m, met in per_month.items()}
    agg = aggregate_months(clean)
    d = dest_dir / label
    d.mkdir(parents=True, exist_ok=True)

    result = {
        "eval_config": {"eval_months": sorted(clean.keys()), "class_type": class_type,
                        "period_type": period_type, "lag": 1},
        "per_month": clean, "aggregate": agg, "n_months": len(clean),
        "n_months_requested": len(eval_months),
        "skipped_months": sorted(set(eval_months) - set(clean.keys())),
    }
    with open(d / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    config = {
        "method": "lightgbm_lambdarank_tiered",
        "features": list(V10E_FEATURES),
        "lag": 1, "period_type": period_type, "class_type": class_type,
        "blend_weights": dict(zip(["da", "dmix", "dori"],
                                  BLEND_WEIGHTS.get((period_type, class_type), _DEFAULT_BLEND))),
    }
    with open(d / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved to {d}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptype", default=None)
    parser.add_argument("--class-type", default=None)
    parser.add_argument("--dev-only", action="store_true")
    args = parser.parse_args()

    ptypes = [args.ptype] if args.ptype else ["f0", "f1"]
    ctypes = [args.class_type] if args.class_type else PJM_CLASS_TYPES
    t_start = time.time()

    for ptype in ptypes:
        for ctype in ctypes:
            print(f"\n{'='*70}")
            print(f"V2 ML: {ptype}/{ctype}")
            print(f"{'='*70}")

            bs = load_all_binding_sets(peak_type=ctype)

            dev_eval = [m for m in _FULL_EVAL_MONTHS if has_period_type(m, ptype)]
            reg_slice = registry_root(ptype, ctype, base_dir=REGISTRY)

            dev_pm = run_variant("v2", dev_eval, bs, class_type=ctype, period_type=ptype)
            if dev_pm:
                save_results("v2", dev_pm, dev_eval, reg_slice,
                             class_type=ctype, period_type=ptype)

            if not args.dev_only:
                ho_eval = [m for m in HOLDOUT_MONTHS if has_period_type(m, ptype)]
                ho_slice = holdout_root(ptype, ctype, base_dir=HOLDOUT_DIR)
                ho_pm = run_variant("v2-holdout", ho_eval, bs,
                                    class_type=ctype, period_type=ptype)
                if ho_pm:
                    save_results("v2", ho_pm, ho_eval, ho_slice,
                                 class_type=ctype, period_type=ptype)

            # Print comparison vs v0
            v0_path = reg_slice / "v0" / "metrics.json"
            if v0_path.exists() and dev_pm:
                v0_agg = json.load(open(v0_path))["aggregate"]["mean"]
                clean = {m: {k: v for k, v in met.items() if not k.startswith("_")}
                         for m, met in dev_pm.items()}
                v2_agg = aggregate_months(clean)["mean"]
                print(f"\n  {'Metric':<12} {'v0':>8} {'v2':>8} {'delta':>8}")
                for met in ["VC@20", "VC@50", "VC@100", "Recall@20", "NDCG"]:
                    v0_v = v0_agg.get(met, 0)
                    v2_v = v2_agg.get(met, 0)
                    delta = (v2_v / v0_v - 1) * 100 if v0_v > 0 else 0
                    print(f"  {met:<12} {v0_v:>8.4f} {v2_v:>8.4f} {delta:>+7.1f}%")

    print(f"\n[main] All done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run v2 for a single slice first**

```bash
python scripts/run_v2_ml.py --ptype f0 --class-type onpeak --dev-only
```

Expected: prints per-month VC@20, aggregate, saves to `registry/f0/onpeak/v2/`.

- [ ] **Step 3: Run v2 for all slices**

```bash
python scripts/run_v2_ml.py
```

- [ ] **Step 4: Commit**

```bash
git add scripts/run_v2_ml.py registry/ holdout/
git commit -m "feat: v2 ML model for all 6 PJM slices"
```

---

### Task 12: Create `scripts/run_blend_search.py`

Search for optimal `(w_ml, 0, w_formula)` blend per slice.

**Files:**
- Create: `scripts/run_blend_search.py`

- [ ] **Step 1: Write blend search script**

This runs the v2 ML model with different blend weights for `v7_formula_score` and evaluates which produces the best dev VC@20. Saves the best blend as v1 in the registry.

The blend weights control how `v7_formula_score` (a feature) is computed:
`v7_formula_score = w_da * da_rank_value + w_dmix * density_mix_rank_value + w_dori * density_ori_rank_value`

Search grid: `w_da` from 0.50 to 1.00 step 0.05, `w_dori = 1 - w_da`, `w_dmix = 0`.

This script can be adapted from `research-stage5-tier/scripts/run_f1_blend_search.py`.

- [ ] **Step 2: Run blend search for f0/onpeak**

```bash
python scripts/run_blend_search.py --ptype f0 --class-type onpeak
```

- [ ] **Step 3: Run for all 6 slices and commit**

```bash
python scripts/run_blend_search.py
git add scripts/run_blend_search.py registry/
git commit -m "feat: blend search for all 6 PJM slices"
```

---

### Task 13: Create `scripts/run_holdout.py` — Final holdout evaluation

Re-run best version (v2 with optimized blend) on held-out months (2024-2025).

**Files:**
- Create: `scripts/run_holdout.py`

- [ ] **Step 1: Write holdout script**

This is a thin wrapper that runs `run_variant()` from `run_v2_ml.py` on holdout months with the best blend weights from the blend search.

- [ ] **Step 2: Run holdout and commit**

```bash
python scripts/run_holdout.py
git add scripts/run_holdout.py holdout/
git commit -m "feat: holdout evaluation for all 6 PJM slices"
```

---

## Chunk 3: Deployment Signal Writer

### Task 14: Create `v70/` deployment modules

Adapt from `research-miso-signal7/v70/` for PJM.

**Files:**
- Create: `v70/__init__.py`
- Create: `v70/cache.py`
- Create: `v70/inference.py`
- Create: `v70/signal_writer.py`
- Create: `scripts/generate_v70_signal.py`

- [ ] **Step 1: Create `v70/cache.py`**

Same as MISO `cache.py` but with PJM paths and 3 class types.

```python
# v70/cache.py
"""Realized DA cache management for PJM V7.0 signal generation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import collect_usable_months, delivery_month, has_period_type

REALIZED_DA_CACHE = os.environ.get(
    "PJM_REALIZED_DA_CACHE",
    str(Path(__file__).resolve().parent.parent / "data" / "realized_da"),
)

_MAX_BF_LOOKBACK = 15


def _cache_path(month: str, peak_type: str) -> Path:
    if peak_type == "onpeak":
        return Path(REALIZED_DA_CACHE) / f"{month}.parquet"
    return Path(REALIZED_DA_CACHE) / f"{month}_{peak_type}.parquet"


def _prev_month(m: str) -> str:
    import pandas as pd
    return (pd.Timestamp(m) - pd.DateOffset(months=1)).strftime("%Y-%m")


def _months_before(month: str, n: int) -> list[str]:
    import pandas as pd
    ts = pd.Timestamp(month)
    return [(ts - pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(1, n + 1)]


def required_realized_da_months(
    auction_month: str, ptypes: list[str], ctypes: list[str],
) -> set[tuple[str, str]]:
    needed: set[tuple[str, str]] = set()
    for ptype in ptypes:
        if not has_period_type(auction_month, ptype):
            continue
        train_months = collect_usable_months(auction_month, ptype, n_months=8)
        for tm in train_months:
            dm = delivery_month(tm, ptype)
            for ct in ctypes:
                needed.add((dm, ct))
            bf_cutoff = _prev_month(tm)
            for bf_month in _months_before(bf_cutoff, _MAX_BF_LOOKBACK):
                for ct in ctypes:
                    needed.add((bf_month, ct))
        bf_cutoff = _prev_month(auction_month)
        for bf_month in _months_before(bf_cutoff, _MAX_BF_LOOKBACK):
            for ct in ctypes:
                needed.add((bf_month, ct))
    return needed


def ensure_realized_da_cache(
    auction_month: str, ptypes: list[str], ctypes: list[str],
) -> None:
    needed = required_realized_da_months(auction_month, ptypes, ctypes)
    missing = [(m, pt) for m, pt in sorted(needed) if not _cache_path(m, pt).exists()]

    if not missing:
        print(f"[cache] All {len(needed)} required DA files present")
        return

    print(f"[cache] {len(missing)}/{len(needed)} missing, fetching...")
    from ml.realized_da import fetch_and_cache_month

    fetched, skipped = 0, 0
    for month, peak_type in missing:
        try:
            out = fetch_and_cache_month(month, peak_type, cache_dir=REALIZED_DA_CACHE)
            fetched += 1 if out.exists() else 0
        except Exception as e:
            skipped += 1
            print(f"[cache]   SKIP {month}/{peak_type}: {e}")

    print(f"[cache] Fetched {fetched}, skipped {skipped}")
```

- [ ] **Step 2: Create `v70/signal_writer.py`**

```python
# v70/signal_writer.py
"""Rank/tier computation and signal assembly for PJM V7.0."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.config import PJM_AUCTION_SCHEDULE


def compute_rank_tier(
    scores: np.ndarray,
    v62b_rank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Row-percentile rank with V6.2B tie-breaking.

    1. ML score descending (higher = more binding = lower rank)
    2. V6.2B rank_ori ascending (lower = more binding, tie-break)
    3. Original index ascending (final tie-break)
    """
    n = len(scores)
    if n == 0:
        return np.array([]), np.array([], dtype=int)
    order = np.lexsort((np.arange(n), v62b_rank, -scores))
    rank = np.empty(n, dtype=np.float64)
    rank[order] = (np.arange(n) + 1) / n
    tier = np.clip(np.ceil(rank * 5).astype(int) - 1, 0, 4)
    return rank, tier


def available_ptypes(auction_month: str) -> list[str]:
    import pandas as pd
    month_num = pd.Timestamp(auction_month).month
    return PJM_AUCTION_SCHEDULE.get(month_num, ["f0"])
```

- [ ] **Step 3: Create `v70/inference.py`**

Adapt from MISO's `v70/inference.py`. Key PJM changes:
- binding sets keyed on `branch_name`
- 3 class types
- PJM paths

- [ ] **Step 4: Create `scripts/generate_v70_signal.py`**

Adapt from MISO's `scripts/generate_v70_signal.py`. Key changes:
- `"pjm"` instead of `"miso"` in `ConstraintsSignal` calls
- PJM signal names
- 3 class types
- No `SO_MW_Transfer` exception

- [ ] **Step 5: Commit**

```bash
git add v70/ scripts/generate_v70_signal.py
git commit -m "feat: PJM V7.0 signal deployment pipeline"
```

---

## Execution Order Summary

The tasks must be executed in this order:

1. **Task 1**: Copy unchanged modules (train, evaluate, v62b_formula, registry_paths, compare)
2. **Task 2**: Create `ml/config.py` — **verify PJM_AUCTION_SCHEDULE against real data**
3. **Task 3**: Create `ml/branch_mapping.py` — **most critical PJM adaptation**
4. **Task 4**: Create `ml/realized_da.py` — fetch + cache DA by branch_name
5. **Task 5**: Create `ml/spice6_loader.py` — PJM density features
6. **Task 6**: Create `ml/features.py` — feature preparation
7. **Task 7**: Create `ml/data_loader.py` — V6.2B + spice6 + realized DA
8. **Task 8**: Create `ml/pipeline.py` — single-month pipeline
9. **Task 9**: Create `scripts/cache_realized_da.py` + **run it** (requires Ray)
10. **Task 10**: Create `scripts/run_v0_formula_baseline.py` + **run it**
11. **Task 11**: Create `scripts/run_v2_ml.py` + **run it**
12. **Task 12**: Create `scripts/run_blend_search.py` + **run it**
13. **Task 13**: Create `scripts/run_holdout.py` + **run it**
14. **Task 14**: Create `v70/` deployment modules

Tasks 1-8 can be done without Ray. Task 9 requires Ray and must complete before Tasks 10-13. Task 14 can be done after all research is validated.

---

## Critical Invariants to Verify

After each major phase, verify these:

1. **Branch mapping coverage**: `map_da_to_branches()` captures >90% of DA value for test months
2. **No temporal leakage**: BF features use months strictly before `prev_month(auction_month)`, training window uses `collect_usable_months()` which respects delivery month lag
3. **V0 baseline sanity**: VC@20 should be in 0.15-0.35 range (if 0.0 or >0.5, something is wrong)
4. **V2 vs V0 improvement**: ML should beat formula by >20% on VC@20 (if not, check target join)
5. **Registry schema**: every slice has `metrics.json`, `config.json`, `gates.json`, `champion.json`
6. **Memory**: never exceed 40 GiB for research scripts (`mem_mb()` at each stage)
7. **LightGBM threads**: always `num_threads=4` (already hardcoded in `train.py`)
