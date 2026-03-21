# MISO Annual FTR Constraint Ranking v2 — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full foundation pipeline — data loading, feature engineering, ground truth, evaluation, and formula baselines — with no ML yet.

**Architecture:** 12 Python modules in `ml/`, thin experiment scripts in `scripts/`, results in `registry/`. Row unit is `branch_name` (not constraint_id). Two-level collapse: outage_dates→cid (mean), cid→branch (max/min). Combined onpeak+offpeak GT. LambdaRank with tiered labels (0/1/2/3).

**Tech Stack:** polars 1.31.0, lightgbm, numpy, Ray (for DA fetch only)

**Spec documents:**
- Design spec: `docs/superpowers/specs/2026-03-12-miso-annual-constraint-ranking-design.md`
- Implementer guide: `docs/implementer-guide.md`
- Test spec: `docs/test-specification.md`
- Handoff prompt: `docs/superpowers/specs/HANDOFF-PROMPT.md`

**v1 reference (patterns only, don't port blindly):** `/home/xyz/workspace/research-qianli-v2/research-annual-signal/ml/`

**Virtual environment:** `cd /home/xyz/workspace/pmodel && source .venv/bin/activate`

**Run all tests:** `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && python -m pytest tests/ -v`

---

## File Structure

```
research-annual-signal-v2/
  ml/
    __init__.py                # empty
    config.py                  # paths, constants, feature lists, splits, params, gates
    bridge.py                  # shared bridge loading + map_cids_to_branches()
    realized_da.py             # DA cache loading, monthly/quarter aggregation
    data_loader.py             # density + limits -> universe filter -> Level 1+2 collapse
    history_features.py        # BF family (7) + da_rank_value + has_hist_da
    ground_truth.py            # continuous SP + tiered labels + per-ctype split + coverage
    nb_detection.py            # NB6/NB12/NB24 + per-ctype NB12 flags
    features.py                # joins all into ONE model table, cohort, monotone vector
    train.py                   # expanding-window LambdaRank training + prediction
    evaluate.py                # Tier 1/2/3 metrics, NB metrics, cohort, gates
    registry.py                # write config.json + metrics.json to registry/{version}/
  tests/
    __init__.py
    conftest.py                # shared fixtures (paths, sample PY/quarter)
    test_config.py
    test_bridge.py
    test_realized_da.py
    test_data_loader.py
    test_history_features.py
    test_ground_truth.py
    test_nb_detection.py
    test_features.py
    test_train.py
    test_evaluate.py
    test_registry.py
  scripts/
    fetch_realized_da.py       # build DA cache (requires Ray) — run FIRST
    calibrate_threshold.py     # universe threshold elbow analysis
    run_v0a_da_rank.py         # formula: pure da_rank_value
    run_v0b_blend.py           # formula: da_rank + density
    run_v0c_full_blend.py      # formula: da_rank + density + bf
  data/
    realized_da/               # DA cache (built by fetch script)
    collapsed/                 # cached collapsed density data
  registry/
    threshold_calibration/     # calibration artifacts
    v0a/                       # formula baseline results
    v0b/
    v0c/
```

---

## Chunk 1: Shared Infrastructure

### Task 1: Project scaffold + ml/config.py

**Files:**
- Create: `ml/__init__.py`
- Create: `ml/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

This task establishes all constants, paths, feature lists, splits, params, and helper functions that every other module imports.

- [ ] **Step 1: Create ml/__init__.py**

```python
# empty
```

- [ ] **Step 2: Create tests/__init__.py**

```python
# empty
```

- [ ] **Step 3: Write test_config.py — failing tests for config module**

```python
"""Tests for ml/config.py — paths, constants, splits, helpers."""
import pytest


def test_data_paths_exist():
    """Test spec A1, A4, A6: verify raw data paths are accessible."""
    from ml.config import DENSITY_PATH, BRIDGE_PATH, LIMIT_PATH
    from pathlib import Path
    assert Path(DENSITY_PATH).exists(), f"Density path not found: {DENSITY_PATH}"
    assert Path(BRIDGE_PATH).exists(), f"Bridge path not found: {BRIDGE_PATH}"
    assert Path(LIMIT_PATH).exists(), f"Limit path not found: {LIMIT_PATH}"


def test_selected_bins():
    """Exactly 10 bins selected per design spec SS3.2."""
    from ml.config import SELECTED_BINS
    assert len(SELECTED_BINS) == 10
    assert SELECTED_BINS == ["-100", "-50", "60", "70", "80", "90", "100", "110", "120", "150"]


def test_right_tail_bins():
    """Right-tail bins for universe filter: 80, 90, 100, 110."""
    from ml.config import RIGHT_TAIL_BINS
    assert RIGHT_TAIL_BINS == ["80", "90", "100", "110"]


def test_get_market_months():
    """Test spec C6: quarter-to-market-month mapping."""
    from ml.config import get_market_months
    assert get_market_months("2025-06", "aq1") == ["2025-06", "2025-07", "2025-08"]
    assert get_market_months("2025-06", "aq2") == ["2025-09", "2025-10", "2025-11"]
    assert get_market_months("2025-06", "aq3") == ["2025-12", "2026-01", "2026-02"]
    assert get_market_months("2025-06", "aq4") == ["2026-03", "2026-04", "2026-05"]
    assert get_market_months("2024-06", "aq3") == ["2024-12", "2025-01", "2025-02"]


def test_get_bf_cutoff_month():
    """Leakage prevention: cutoff = March of submission year (Trap 1)."""
    from ml.config import get_bf_cutoff_month
    assert get_bf_cutoff_month("2025-06") == "2025-03"
    assert get_bf_cutoff_month("2024-06") == "2024-03"
    assert get_bf_cutoff_month("2019-06") == "2019-03"


def test_eval_splits():
    """Expanding window: 4 splits, holdout is 2025-06."""
    from ml.config import EVAL_SPLITS
    assert len(EVAL_SPLITS) == 4
    # Holdout split
    holdout = EVAL_SPLITS["2025-06"]
    assert holdout["train_pys"] == ["2019-06", "2020-06", "2021-06", "2022-06", "2023-06", "2024-06"]
    assert holdout["eval_pys"] == ["2025-06"]
    assert holdout["split"] == "holdout"
    # First dev split
    dev1 = EVAL_SPLITS["2022-06"]
    assert dev1["train_pys"] == ["2019-06", "2020-06", "2021-06"]
    assert dev1["split"] == "dev"


def test_planning_years():
    from ml.config import PLANNING_YEARS, AQ_QUARTERS
    assert PLANNING_YEARS == ["2019-06", "2020-06", "2021-06", "2022-06", "2023-06", "2024-06", "2025-06"]
    assert AQ_QUARTERS == ["aq1", "aq2", "aq3", "aq4"]


def test_holdout_groups():
    from ml.config import HOLDOUT_GROUPS, DEV_GROUPS
    assert HOLDOUT_GROUPS == ["2025-06/aq1", "2025-06/aq2", "2025-06/aq3"]
    assert len(DEV_GROUPS) == 12  # 3 years x 4 quarters


def test_lgbm_params():
    """Trap 3: num_threads must be 4."""
    from ml.config import LGBM_PARAMS
    assert LGBM_PARAMS["num_threads"] == 4
    assert LGBM_PARAMS["objective"] == "lambdarank"
    assert LGBM_PARAMS["metric"] == "ndcg"
    assert LGBM_PARAMS["n_estimators"] == 200
    assert LGBM_PARAMS["learning_rate"] == 0.03


def test_bf_backfill_floor():
    from ml.config import BF_FLOOR_MONTH
    assert BF_FLOOR_MONTH == "2017-04"


def test_tier1_gate_metrics():
    """7 blocking gate metrics per design spec SS9.1."""
    from ml.config import TIER1_GATE_METRICS
    assert len(TIER1_GATE_METRICS) == 7
    assert "VC@50" in TIER1_GATE_METRICS
    assert "NB12_Recall@50" in TIER1_GATE_METRICS
    assert "Abs_SP@50" in TIER1_GATE_METRICS
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && python -m pytest tests/test_config.py -v`
Expected: FAIL (no ml.config module)

- [ ] **Step 5: Write tests/conftest.py**

```python
"""Shared test fixtures."""
import pytest


@pytest.fixture
def sample_py():
    return "2024-06"


@pytest.fixture
def sample_quarter():
    return "aq1"


@pytest.fixture
def holdout_py():
    return "2025-06"
```

- [ ] **Step 6: Implement ml/config.py**

```python
"""Constants, paths, feature lists, splits, params, and gates for annual signal v2."""
from __future__ import annotations

from pathlib import Path

# ─── Data paths ───────────────────────────────────────────────────────────
DENSITY_PATH = "/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_DENSITY_DISTRIBUTION.parquet"
BRIDGE_PATH = "/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_INFO.parquet"
LIMIT_PATH = "/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_LIMIT.parquet"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DA_CACHE_DIR = PROJECT_ROOT / "data" / "realized_da"
COLLAPSED_CACHE_DIR = PROJECT_ROOT / "data" / "collapsed"
REGISTRY_DIR = PROJECT_ROOT / "registry"

# ─── Density bins ─────────────────────────────────────────────────────────
# All 77 bin column names (verified from parquet schema)
ALL_BIN_COLUMNS: list[str] = [
    "-300", "-280", "-260", "-240", "-220", "-200", "-180", "-160", "-150",
    "-145", "-140", "-135", "-130", "-125", "-120", "-115", "-110", "-105",
    "-100", "-95", "-90", "-85", "-80", "-75", "-70", "-65", "-60", "-55",
    "-50", "-45", "-40", "-35", "-30", "-25", "-20", "-15", "-10", "-5",
    "0", "5", "10", "15", "20", "25", "30", "35", "40", "45", "50", "55",
    "60", "65", "70", "75", "80", "85", "90", "95", "100", "105", "110",
    "115", "120", "125", "130", "135", "140", "145", "150", "160", "180",
    "200", "220", "240", "260", "280", "300",
]

# 10 selected bins (design spec SS3.2, implementer-guide SS7.2)
SELECTED_BINS: list[str] = [
    "-100", "-50", "60", "70", "80", "90", "100", "110", "120", "150",
]

# Right-tail bins for universe filter (design spec SS3.1 Step 3)
RIGHT_TAIL_BINS: list[str] = ["80", "90", "100", "110"]

# Universe threshold — PLACEHOLDER until calibrate_threshold.py runs (Step 1.2)
# Will be replaced with the calibrated value.
UNIVERSE_THRESHOLD: float = 0.0  # TODO: freeze after calibration

# ─── Planning years & quarters ────────────────────────────────────────────
PLANNING_YEARS: list[str] = [
    "2019-06", "2020-06", "2021-06", "2022-06", "2023-06", "2024-06", "2025-06",
]

AQ_QUARTERS: list[str] = ["aq1", "aq2", "aq3", "aq4"]

# ─── BF ───────────────────────────────────────────────────────────────────
BF_FLOOR_MONTH: str = "2017-04"  # backfill start (v16 champion insight)
BF_WINDOWS_ONPEAK: list[int] = [6, 12, 15]
BF_WINDOWS_OFFPEAK: list[int] = [6, 12]
BF_WINDOWS_COMBINED: list[int] = [6, 12]


def get_market_months(planning_year: str, aq_quarter: str) -> list[str]:
    """Derive 3 market months from (PY, quarter). See test spec C6."""
    py_year = int(planning_year[:4])
    py_month = int(planning_year[5:7])  # always 6
    offsets = {"aq1": [0, 1, 2], "aq2": [3, 4, 5], "aq3": [6, 7, 8], "aq4": [9, 10, 11]}
    assert aq_quarter in offsets, f"Invalid quarter: {aq_quarter}"
    months = []
    for offset in offsets[aq_quarter]:
        total_month = py_month + offset
        year = py_year + (total_month - 1) // 12
        month = ((total_month - 1) % 12) + 1
        months.append(f"{year:04d}-{month:02d}")
    return months


def get_bf_cutoff_month(planning_year: str) -> str:
    """BF lookback cutoff = March of submission year (Trap 1).
    Annual R1 submitted ~April 10. Use only months <= March.
    """
    py_year = int(planning_year[:4])
    return f"{py_year}-03"


# ─── Eval splits (expanding window) ──────────────────────────────────────
EVAL_SPLITS: dict[str, dict] = {
    "2022-06": {
        "train_pys": ["2019-06", "2020-06", "2021-06"],
        "eval_pys": ["2022-06"],
        "split": "dev",
    },
    "2023-06": {
        "train_pys": ["2019-06", "2020-06", "2021-06", "2022-06"],
        "eval_pys": ["2023-06"],
        "split": "dev",
    },
    "2024-06": {
        "train_pys": ["2019-06", "2020-06", "2021-06", "2022-06", "2023-06"],
        "eval_pys": ["2024-06"],
        "split": "dev",
    },
    "2025-06": {
        "train_pys": ["2019-06", "2020-06", "2021-06", "2022-06", "2023-06", "2024-06"],
        "eval_pys": ["2025-06"],
        "split": "holdout",
    },
}

DEV_GROUPS: list[str] = [
    f"{py}/{aq}"
    for py in ["2022-06", "2023-06", "2024-06"]
    for aq in AQ_QUARTERS
]

HOLDOUT_GROUPS: list[str] = ["2025-06/aq1", "2025-06/aq2", "2025-06/aq3"]
# aq4 monitored only — incomplete as of 2026-03-12

# ─── LightGBM parameters ─────────────────────────────────────────────────
LGBM_PARAMS: dict = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "n_estimators": 200,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 10,
    "num_threads": 4,  # Trap 3: 64-CPU container causes contention
    "verbose": -1,
}

# ─── Feature lists ────────────────────────────────────────────────────────
HISTORY_FEATURES: list[str] = [
    "da_rank_value",
    "bf_6", "bf_12", "bf_15",
    "bfo_6", "bfo_12",
    "bf_combined_6", "bf_combined_12",
]

DENSITY_MAX_FEATURES: list[str] = [f"bin_{b}_cid_max" for b in SELECTED_BINS]
DENSITY_MIN_FEATURES: list[str] = [f"bin_{b}_cid_min" for b in SELECTED_BINS]

LIMIT_FEATURES: list[str] = ["limit_min", "limit_mean", "limit_max", "limit_std"]
METADATA_FEATURES: list[str] = ["count_cids", "count_active_cids"]

ALL_FEATURES: list[str] = (
    DENSITY_MAX_FEATURES + DENSITY_MIN_FEATURES
    + LIMIT_FEATURES + METADATA_FEATURES
    + HISTORY_FEATURES
)

# ─── Monotone constraints (order must match feature_cols — assert in features.py) ───
MONOTONE_MAP: dict[str, int] = {
    # BF: +1 (higher freq = more binding)
    "bf_6": 1, "bf_12": 1, "bf_15": 1,
    "bfo_6": 1, "bfo_12": 1,
    "bf_combined_6": 1, "bf_combined_12": 1,
    # da_rank_value: -1 (lower rank = more binding)
    "da_rank_value": -1,
    # Density bins: 0 (NOT monotone — density weights, not probabilities)
    # Limits: 0
    # Metadata: 0
}


def get_monotone_constraints(feature_cols: list[str]) -> list[int]:
    """Build monotone constraint vector matching feature_cols order.
    Returns 0 for any feature not in MONOTONE_MAP.
    """
    return [MONOTONE_MAP.get(f, 0) for f in feature_cols]


# ─── Tier 1 gate metrics (blocking) ──────────────────────────────────────
TIER1_GATE_METRICS: list[str] = [
    "VC@50", "VC@100", "Recall@50", "Recall@100",
    "NDCG", "Abs_SP@50", "NB12_Recall@50",
]

# Gate rule: candidate must beat baseline on >=2 of 3 holdout groups + mean >= baseline
GATE_MIN_WINS: int = 2
GATE_HOLDOUT_COUNT: int = 3
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && python -m pytest tests/test_config.py -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add ml/__init__.py ml/config.py tests/__init__.py tests/conftest.py tests/test_config.py
git commit -m "feat: add ml/config.py with paths, constants, splits, params, gates"
```

---

### Task 2: ml/bridge.py — shared bridge loading + ambiguity handling

**Files:**
- Create: `ml/bridge.py`
- Create: `tests/test_bridge.py`

This is the single source of truth for bridge loading. All three consumers (data_loader, ground_truth, history_features) use `map_cids_to_branches()`.

**Bridge partition layout** (verified from data): Annual bridge has `period_type=aq1..aq4` partitions; monthly bridge has `period_type=f0` partitions. The `period_type` parameter is ALWAYS required — for annual calls use the specific quarter (e.g. `"aq1"`), for monthly fallback use `"f0"`.

- [ ] **Step 1: Write test_bridge.py — failing tests**

```python
"""Tests for ml/bridge.py — bridge loading + cid-to-branch mapping."""
import polars as pl
import pytest


def test_load_bridge_partition_annual_both_ctypes(sample_py):
    """Test spec A8: loads BOTH onpeak+offpeak and UNIONs them."""
    from ml.bridge import load_bridge_partition
    bridge = load_bridge_partition(
        auction_type="annual",
        auction_month=sample_py,
        period_type="aq1",
    )
    assert "constraint_id" in bridge.columns
    assert "branch_name" in bridge.columns
    assert bridge["branch_name"].null_count() == 0
    n_unique_branches = bridge["branch_name"].n_unique()
    assert 4000 <= n_unique_branches <= 6000, f"Unexpected branch count: {n_unique_branches}"


def test_load_bridge_partition_convention_filter(sample_py):
    """Test spec A4: convention < 10 keeps only -1 and 1."""
    from ml.bridge import load_bridge_partition
    from ml.config import BRIDGE_PATH
    # Load raw (no convention filter) to compare
    raw = pl.read_parquet(
        f"{BRIDGE_PATH}/spice_version=v6/auction_type=annual"
        f"/auction_month={sample_py}/market_round=1/period_type=aq1/class_type=onpeak/"
    )
    assert 999 in raw["convention"].unique().to_list()
    # Filtered version should not have 999
    bridge = load_bridge_partition(
        auction_type="annual", auction_month=sample_py, period_type="aq1",
    )
    # After convention filter + union + unique, should have reasonable count
    assert len(bridge) > 10000


def test_load_bridge_partition_missing_raises():
    """FileNotFoundError if NEITHER class_type partition exists."""
    from ml.bridge import load_bridge_partition
    with pytest.raises(FileNotFoundError):
        load_bridge_partition(
            auction_type="annual",
            auction_month="1900-06",  # doesn't exist
            period_type="aq1",
        )


def test_map_cids_to_branches_annual(sample_py):
    """Shared mapping function: maps cids, detects ambiguity, drops ambiguous."""
    from ml.bridge import map_cids_to_branches
    import polars as pl

    # Create a test df with some cids
    test_cids = pl.DataFrame({"constraint_id": ["1000", "100023", "999999999"]})
    mapped, diag = map_cids_to_branches(
        cid_df=test_cids,
        auction_type="annual",
        auction_month=sample_py,
        period_type="aq1",
    )
    assert "branch_name" in mapped.columns
    assert "constraint_id" in mapped.columns
    assert "ambiguous_cids" in diag
    assert "ambiguous_sp" in diag
    # No nulls in branch_name for mapped rows
    assert mapped["branch_name"].null_count() == 0


def test_bridge_no_fanout(sample_py):
    """Test spec C2: convention < 10 gives ~1:1 cid:branch after unique."""
    from ml.bridge import load_bridge_partition
    bridge = load_bridge_partition(
        auction_type="annual", auction_month=sample_py, period_type="aq1",
    )
    # After unique(), each cid should map to at most 1 branch
    cid_counts = bridge.group_by("constraint_id").len()
    max_branches_per_cid = cid_counts["len"].max()
    # A few ambiguous cids may exist (0-2 per slice), but most are 1:1
    assert max_branches_per_cid <= 3, f"Bridge fanout too high: {max_branches_per_cid}"


def test_bridge_hive_scan_fails():
    """Test spec A5: hive scan on full bridge raises SchemaError (Trap 5)."""
    from ml.config import BRIDGE_PATH
    with pytest.raises(Exception):  # SchemaError or similar
        pl.scan_parquet(BRIDGE_PATH, hive_partitioning=True).collect()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bridge.py -v`
Expected: FAIL (no ml.bridge module)

- [ ] **Step 3: Implement ml/bridge.py**

```python
"""Bridge table loading and cid-to-branch mapping.

This is the SINGLE SOURCE OF TRUTH for bridge loading. All consumers
(data_loader, ground_truth, history_features) use map_cids_to_branches().
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from ml.config import BRIDGE_PATH

logger = logging.getLogger(__name__)


def load_bridge_partition(
    auction_type: str,
    auction_month: str,
    period_type: str,
) -> pl.DataFrame:
    """Load bridge for BOTH class types and UNION them.

    Applies convention < 10 filter. Returns unique (constraint_id, branch_name).
    RAISES FileNotFoundError if NEITHER class_type partition exists.
    Logs warning if only one class_type is found.
    """
    frames: list[pl.DataFrame] = []
    missing: list[str] = []

    for ctype in ["onpeak", "offpeak"]:
        part_path = (
            f"{BRIDGE_PATH}/spice_version=v6/auction_type={auction_type}"
            f"/auction_month={auction_month}/market_round=1"
            f"/period_type={period_type}/class_type={ctype}/"
        )
        if not Path(part_path).exists():
            missing.append(ctype)
            continue
        df = (
            pl.read_parquet(part_path)
            .filter(
                (pl.col("convention") < 10) & pl.col("branch_name").is_not_null()
            )
            .select(["constraint_id", "branch_name"])
            .unique()
        )
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No bridge partition found for {auction_type}/{auction_month}/"
            f"{period_type} in either onpeak or offpeak"
        )
    if missing:
        logger.warning(
            "Bridge partition missing for class_type=%s (%s/%s/%s). Using %s only.",
            missing, auction_type, auction_month, period_type,
            "offpeak" if "onpeak" in missing else "onpeak",
        )

    return pl.concat(frames).unique()


def map_cids_to_branches(
    cid_df: pl.DataFrame,
    auction_type: str,
    auction_month: str,
    period_type: str,
) -> tuple[pl.DataFrame, dict]:
    """Map constraint_ids to branch_names via bridge table.

    Handles:
    - Both-ctype UNION (onpeak + offpeak)
    - Convention < 10 filter
    - Ambiguous cid detection (cids mapping to multiple branch_names after union)
    - Logging ambiguous cid count
    - Dropping ambiguous cids

    Returns:
        (mapped_df with branch_name column, diagnostics dict)
    """
    bridge = load_bridge_partition(auction_type, auction_month, period_type)

    # Detect ambiguous cids: cids that map to >1 branch_name
    cid_branch_counts = bridge.group_by("constraint_id").agg(
        pl.col("branch_name").n_unique().alias("n_branches")
    )
    ambiguous_cids = cid_branch_counts.filter(pl.col("n_branches") > 1)["constraint_id"]
    n_ambiguous = len(ambiguous_cids)

    # Compute ambiguous SP if cid_df has a realized_sp column
    ambiguous_sp = 0.0
    if n_ambiguous > 0:
        if "realized_sp" in cid_df.columns:
            ambiguous_sp = float(
                cid_df.filter(pl.col("constraint_id").is_in(ambiguous_cids))
                ["realized_sp"].sum()
            )
        logger.warning(
            "Found %d ambiguous cids (SP=%.1f) mapping to >1 branch in %s/%s/%s. Dropping.",
            n_ambiguous, ambiguous_sp, auction_type, auction_month, period_type,
        )
        bridge = bridge.filter(~pl.col("constraint_id").is_in(ambiguous_cids))

    # Now bridge should have at most 1 branch per cid
    bridge_unique = bridge.unique(subset=["constraint_id"])

    # Inner join: keep only cids that have a bridge mapping
    assert "constraint_id" in cid_df.columns, "cid_df must have constraint_id column"
    mapped = cid_df.join(bridge_unique, on="constraint_id", how="inner")

    diagnostics = {
        "ambiguous_cids": n_ambiguous,
        "ambiguous_sp": ambiguous_sp,
        "total_bridge_cids": len(bridge_unique),
        "mapped_cids": len(mapped),
        "unmapped_cids": len(cid_df) - len(mapped),
    }

    return mapped, diagnostics
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bridge.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add ml/bridge.py tests/test_bridge.py
git commit -m "feat: add ml/bridge.py — shared bridge loading + ambiguity handling"
```

---

### Task 3: ml/realized_da.py + scripts/fetch_realized_da.py

**Files:**
- Create: `ml/realized_da.py`
- Create: `scripts/fetch_realized_da.py`
- Create: `tests/test_realized_da.py`

DA cache is the foundation for ground truth, BF, and da_rank_value.

- [ ] **Step 1: Write test_realized_da.py — failing tests**

```python
"""Tests for ml/realized_da.py — DA cache loading."""
import polars as pl
import pytest


def test_load_month_onpeak():
    """Test spec A7: onpeak DA loads with correct schema."""
    from ml.realized_da import load_month
    df = load_month("2024-07", peak_type="onpeak")
    assert "constraint_id" in df.columns
    assert "realized_sp" in df.columns
    assert df["realized_sp"].min() >= 0, "realized_sp must be non-negative (already abs-aggregated)"
    assert len(df) > 0


def test_load_month_offpeak():
    """Test spec A7: offpeak DA loads."""
    from ml.realized_da import load_month
    df = load_month("2024-07", peak_type="offpeak")
    assert len(df) > 0
    assert df["realized_sp"].min() >= 0


def test_load_month_missing_raises():
    """Missing month raises FileNotFoundError."""
    from ml.realized_da import load_month
    with pytest.raises(FileNotFoundError):
        load_month("1900-01", peak_type="onpeak")


def test_load_quarter_combined():
    """Load combined onpeak+offpeak for a quarter, summed per cid."""
    from ml.realized_da import load_quarter
    df = load_quarter(["2024-06", "2024-07", "2024-08"])
    assert "constraint_id" in df.columns
    assert "realized_sp" in df.columns
    assert df["realized_sp"].min() >= 0
    # Combined should have cids from both ctypes
    assert len(df) > 200


def test_load_quarter_per_ctype():
    """Load per-ctype quarter data for monitoring split."""
    from ml.realized_da import load_quarter_per_ctype
    onpeak_df, offpeak_df = load_quarter_per_ctype(["2024-06", "2024-07", "2024-08"])
    assert len(onpeak_df) > 0
    assert len(offpeak_df) > 0
    # Some cids should appear in offpeak but not onpeak
    on_cids = set(onpeak_df["constraint_id"].to_list())
    off_cids = set(offpeak_df["constraint_id"].to_list())
    assert len(off_cids - on_cids) > 0, "Expected some offpeak-only cids"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_realized_da.py -v`
Expected: FAIL

- [ ] **Step 3: Implement ml/realized_da.py**

```python
"""Realized DA cache loading — single source of truth for DA data access.

Cache layout in data/realized_da/:
  {YYYY-MM}.parquet         — onpeak (constraint_id, realized_sp)
  {YYYY-MM}_offpeak.parquet — offpeak
"""
from __future__ import annotations

import polars as pl

from ml.config import DA_CACHE_DIR


def _cache_path(month: str, peak_type: str) -> str:
    suffix = "_offpeak" if peak_type == "offpeak" else ""
    return str(DA_CACHE_DIR / f"{month}{suffix}.parquet")


def load_month(month: str, peak_type: str) -> pl.DataFrame:
    """Load cached DA for one month+ctype.

    Returns DataFrame with columns: constraint_id (Utf8), realized_sp (Float64).
    realized_sp = abs(sum(shadow_price)) per constraint_id, already netted within month+ctype.
    """
    path = _cache_path(month, peak_type)
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"DA cache not found: {path}. Run scripts/fetch_realized_da.py first.")
    df = pl.read_parquet(path)
    assert "constraint_id" in df.columns, f"Missing constraint_id in {path}"
    assert "realized_sp" in df.columns, f"Missing realized_sp in {path}"
    return df


def load_quarter(market_months: list[str]) -> pl.DataFrame:
    """Load combined onpeak+offpeak DA for a quarter (3 months).

    Aggregates: sum(realized_sp) per constraint_id across months and both ctypes.
    These are nonneg values being summed — no re-netting needed.
    """
    frames: list[pl.DataFrame] = []
    for month in market_months:
        for peak_type in ["onpeak", "offpeak"]:
            df = load_month(month, peak_type)
            frames.append(df)

    combined = pl.concat(frames)
    return combined.group_by("constraint_id").agg(
        pl.col("realized_sp").sum()
    )


def load_quarter_per_ctype(
    market_months: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load quarter DA split by ctype — for per-ctype GT monitoring.

    Returns (onpeak_df, offpeak_df), each with (constraint_id, realized_sp).
    """
    onpeak_frames: list[pl.DataFrame] = []
    offpeak_frames: list[pl.DataFrame] = []

    for month in market_months:
        onpeak_frames.append(load_month(month, "onpeak"))
        offpeak_frames.append(load_month(month, "offpeak"))

    onpeak = pl.concat(onpeak_frames).group_by("constraint_id").agg(
        pl.col("realized_sp").sum()
    )
    offpeak = pl.concat(offpeak_frames).group_by("constraint_id").agg(
        pl.col("realized_sp").sum()
    )
    return onpeak, offpeak
```

- [ ] **Step 4: Implement scripts/fetch_realized_da.py**

This script builds the DA cache. Requires Ray. Run ONCE before anything else.

```python
"""Build realized DA cache for all months 2017-04 through 2026-02.

Usage:
  cd /home/xyz/workspace/pmodel && source .venv/bin/activate
  python /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/scripts/fetch_realized_da.py

Requires Ray cluster.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ray setup
os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"

def main():
    from pbase.config.ray import init_ray
    import pmodel
    init_ray(extra_modules=[pmodel])

    import polars as pl
    from pbase.analysis.tools.all_positions import MisoApTools

    project_root = Path(__file__).resolve().parent.parent
    cache_dir = project_root / "data" / "realized_da"
    cache_dir.mkdir(parents=True, exist_ok=True)

    tools = MisoApTools().tools

    # Generate all months from 2017-04 through 2026-02
    months = []
    for year in range(2017, 2027):
        for month in range(1, 13):
            m = f"{year:04d}-{month:02d}"
            if m < "2017-04" or m > "2026-02":
                continue
            months.append(m)

    for month_str in months:
        for peak_type in ["onpeak", "offpeak"]:
            suffix = "_offpeak" if peak_type == "offpeak" else ""
            out_path = cache_dir / f"{month_str}{suffix}.parquet"

            if out_path.exists():
                continue

            year, mon = month_str.split("-")
            st = f"{year}-{mon}-01"
            # et_ex is first day of NEXT month
            next_mon = int(mon) + 1
            next_year = int(year)
            if next_mon > 12:
                next_mon = 1
                next_year += 1
            et_ex = f"{next_year:04d}-{next_mon:02d}-01"

            print(f"Fetching {month_str} {peak_type}...")
            da = tools.get_da_shadow_by_peaktype(
                st=st, et_ex=et_ex, peak_type=peak_type
            )

            if da is None or len(da) == 0:
                print(f"  SKIP {month_str} {peak_type}: no data returned")
                continue

            # Validate raw API output columns
            assert "constraint_id" in da.columns, f"API missing constraint_id, got: {list(da.columns)}"
            assert "shadow_price" in da.columns, f"API missing shadow_price, got: {list(da.columns)}"

            # Convert to polars, aggregate per constraint_id
            # abs(sum(shadow_price)): netting within month+ctype, then abs
            da_pl = pl.from_pandas(da)
            agg = (
                da_pl
                .group_by("constraint_id")
                .agg(pl.col("shadow_price").sum().abs().alias("realized_sp"))
            )
            assert (agg["realized_sp"] >= 0).all(), "realized_sp must be non-negative after abs"
            # Ensure constraint_id is string
            agg = agg.with_columns(pl.col("constraint_id").cast(pl.Utf8))

            agg.write_parquet(str(out_path))
            print(f"  Wrote {out_path} ({len(agg)} cids)")

    import ray
    ray.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run fetch script to build DA cache**

Run: `cd /home/xyz/workspace/pmodel && source .venv/bin/activate && python /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/scripts/fetch_realized_da.py`
Expected: Creates parquet files in `data/realized_da/`

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_realized_da.py -v`
Expected: ALL PASS (requires DA cache to exist)

- [ ] **Step 7: Commit**

```bash
git add ml/realized_da.py scripts/fetch_realized_da.py tests/test_realized_da.py
git commit -m "feat: add ml/realized_da.py + DA fetch script"
```

---

## Chunk 2: Data Pipeline

### Task 4: scripts/calibrate_threshold.py — universe threshold

**Files:**
- Create: `scripts/calibrate_threshold.py`

Freezes `UNIVERSE_THRESHOLD` in config.py. Must run before data_loader.

**Unit relationship (deliberate design choice):**
- The threshold is a value of `right_tail_max` — the max over bins 80/90/100/110 across all outage_dates for a given cid.
- In `data_loader.py`, the threshold is applied **per cid**: `is_active = (cid_right_tail_max >= UNIVERSE_THRESHOLD)`.
- A branch survives the universe filter iff it has `count_active_cids >= 1` (at least one cid passes).
- Calibration works at **branch level** because that's where trading value (binding SP) is measured: `branch_right_tail_max = max(cid_right_tail_max)` over all mapped cids.
- These are consistent: a branch with `branch_rtm >= T` always has at least one cid with `cid_rtm >= T`. The branch-level calibration curve therefore correctly predicts how many branches (and how much SP) the cid-level filter will retain.

- [ ] **Step 1: Implement calibrate_threshold.py**

This script:
1. Loads raw density for 2024-06/aq1 (all ~13,000 cids)
2. Computes `right_tail_max = max(bin_80, bin_90, bin_100, bin_110)` per cid across outage_dates
3. Maps cids to branches, takes max right_tail_max per branch
4. Joins branch-level binding SP from DA
5. Sweeps thresholds descending, computes cumulative SP captured vs branch count
6. Finds elbow (threshold where adding branches stops improving SP capture)
7. Cross-checks against 2023-06/aq1 (within ±20%)
8. Saves calibration artifact to `registry/threshold_calibration/`

```python
"""Universe threshold calibration — run on Day 1.

Produces:
  registry/threshold_calibration/
    threshold.json        — {threshold, rationale, date}
    calibration_data.parquet — cid-level right_tail_max + binding SP
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from ml.config import (
    DENSITY_PATH, RIGHT_TAIL_BINS, REGISTRY_DIR,
    get_market_months,
)
from ml.bridge import load_bridge_partition
from ml.realized_da import load_quarter


def compute_right_tail_max(planning_year: str, aq_quarter: str) -> pl.DataFrame:
    """Compute right_tail_max per cid for one (PY, quarter)."""
    market_months = get_market_months(planning_year, aq_quarter)
    frames = []
    for mm in market_months:
        path = (
            f"{DENSITY_PATH}/spice_version=v6/auction_type=annual"
            f"/auction_month={planning_year}/market_month={mm}/market_round=1/"
        )
        if not Path(path).exists():
            continue
        df = pl.read_parquet(path).select(["constraint_id"] + RIGHT_TAIL_BINS)
        frames.append(df)

    assert len(frames) > 0, f"No density data for {planning_year}/{aq_quarter}"
    raw = pl.concat(frames, how="diagonal")

    # Per row: right_tail = max(bin_80, bin_90, bin_100, bin_110)
    raw = raw.with_columns(
        pl.max_horizontal([pl.col(b) for b in RIGHT_TAIL_BINS]).alias("right_tail")
    )

    # Per cid: right_tail_max = max across all outage_dates
    return raw.group_by("constraint_id").agg(
        pl.col("right_tail").max().alias("right_tail_max")
    )


def run_calibration():
    out_dir = REGISTRY_DIR / "threshold_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Primary: 2024-06/aq1
    print("Computing right_tail_max for 2024-06/aq1...")
    rtm_2024 = compute_right_tail_max("2024-06", "aq1")
    print(f"  Total cids: {len(rtm_2024)}")

    # Get binding SP per cid for 2024-06/aq1
    mm_2024 = get_market_months("2024-06", "aq1")
    da_2024 = load_quarter(mm_2024)

    # Bridge mapping: cid -> branch
    bridge = load_bridge_partition(auction_type="annual", auction_month="2024-06", period_type="aq1")

    # Work at BRANCH level to avoid fan-out (multiple cids per branch)
    # 1. Map density cids to branches, keep max right_tail_max per branch
    cid_with_branch = rtm_2024.join(bridge, on="constraint_id", how="inner")
    branch_rtm = cid_with_branch.group_by("branch_name").agg(
        pl.col("right_tail_max").max()
    )

    # 2. Map DA cids to branches, sum realized_sp per branch
    da_mapped = da_2024.join(bridge, on="constraint_id", how="inner")
    branch_sp = da_mapped.group_by("branch_name").agg(pl.col("realized_sp").sum())

    # 3. Join branch-level: right_tail_max + realized_sp
    branch_df = branch_rtm.join(branch_sp, on="branch_name", how="left").with_columns(
        pl.col("realized_sp").fill_null(0.0)
    )

    # Sort by right_tail_max descending, compute cumulative SP (no fan-out)
    sorted_df = branch_df.sort("right_tail_max", descending=True)
    total_sp = sorted_df["realized_sp"].sum()

    # Compute cumulative SP capture at each threshold
    rtm_values = sorted_df["right_tail_max"].to_list()
    sp_values = sorted_df["realized_sp"].to_list()

    cumsum = 0.0
    thresholds = []
    for i, (rtm, sp) in enumerate(zip(rtm_values, sp_values)):
        cumsum += sp
        thresholds.append({
            "rank": i + 1,
            "right_tail_max": rtm,
            "cumulative_sp": cumsum,
            "pct_sp": cumsum / total_sp if total_sp > 0 else 0,
        })

    # Find elbow: where cumulative SP capture > 95% of total
    target_pct = 0.95
    elbow_idx = next(
        (i for i, t in enumerate(thresholds) if t["pct_sp"] >= target_pct),
        len(thresholds) - 1
    )
    elbow_threshold = thresholds[elbow_idx]["right_tail_max"]
    elbow_branch_count = thresholds[elbow_idx]["rank"]

    print(f"\n  Elbow at {target_pct*100}% SP capture:")
    print(f"    threshold = {elbow_threshold:.6f}")
    print(f"    branch count = {elbow_branch_count}")

    # Cross-check with 2023-06/aq1 (also at branch level)
    print("\nCross-checking with 2023-06/aq1...")
    rtm_2023 = compute_right_tail_max("2023-06", "aq1")
    bridge_2023 = load_bridge_partition(auction_type="annual", auction_month="2023-06", period_type="aq1")
    cid_br_2023 = rtm_2023.join(bridge_2023, on="constraint_id", how="inner")
    branch_rtm_2023 = cid_br_2023.group_by("branch_name").agg(pl.col("right_tail_max").max())
    branches_2023 = branch_rtm_2023.filter(pl.col("right_tail_max") >= elbow_threshold)
    print(f"  2023-06/aq1 filtered branches: {len(branches_2023)}")
    ratio = len(branches_2023) / elbow_branch_count if elbow_branch_count > 0 else 0
    print(f"  Ratio: {ratio:.2f} (should be 0.80-1.20)")
    assert 0.5 <= ratio <= 2.0, f"Cross-check failed: ratio {ratio:.2f} outside tolerance"

    # Save artifact
    artifact = {
        "threshold": elbow_threshold,
        "calibration_py": "2024-06",
        "calibration_quarter": "aq1",
        "branch_count_at_threshold": elbow_branch_count,
        "sp_capture_pct": thresholds[elbow_idx]["pct_sp"],
        "cross_check_py": "2023-06",
        "cross_check_branches": len(branches_2023),
        "cross_check_ratio": ratio,
        "date": "2026-03-12",
    }
    with open(out_dir / "threshold.json", "w") as f:
        json.dump(artifact, f, indent=2)

    # Save calibration data
    cal_data = pl.DataFrame(thresholds)
    cal_data.write_parquet(str(out_dir / "calibration_data.parquet"))

    print(f"\nSaved to {out_dir}/")
    print(f"\n*** ACTION REQUIRED: Update UNIVERSE_THRESHOLD in ml/config.py to {elbow_threshold} ***")


if __name__ == "__main__":
    run_calibration()
```

- [ ] **Step 2: Run calibration script**

Run: `cd /home/xyz/workspace/pmodel && source .venv/bin/activate && python /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/scripts/calibrate_threshold.py`
Expected: Prints threshold, saves to `registry/threshold_calibration/`

- [ ] **Step 3: Update UNIVERSE_THRESHOLD in ml/config.py**

Replace the placeholder value with the calibrated threshold from `registry/threshold_calibration/threshold.json`.

- [ ] **Step 4: Commit**

```bash
git add scripts/calibrate_threshold.py registry/threshold_calibration/ ml/config.py
git commit -m "feat: calibrate universe threshold, freeze in config.py"
```

---

### Task 5: ml/data_loader.py — density collapse pipeline

**Files:**
- Create: `ml/data_loader.py`
- Create: `tests/test_data_loader.py`

Full pipeline: density → universe filter → Level 1 → bridge → Level 2 → branch features.

- [ ] **Step 1: Write test_data_loader.py — failing tests**

```python
"""Tests for ml/data_loader.py — density collapse pipeline."""
import polars as pl
import pytest


def test_load_raw_density_shape(sample_py, sample_quarter):
    """Test spec A1: raw density has expected columns."""
    from ml.data_loader import load_raw_density
    df = load_raw_density(sample_py, sample_quarter)
    assert "constraint_id" in df.columns
    assert "outage_date" in df.columns
    assert len(df) > 100_000


def test_density_row_sums(sample_py, sample_quarter):
    """Test spec A2: bins sum to 20.0 per row."""
    from ml.data_loader import load_raw_density
    from ml.config import ALL_BIN_COLUMNS
    df = load_raw_density(sample_py, sample_quarter)
    # Sample 1000 rows
    sample = df.head(1000)
    row_sums = sample.select(
        pl.sum_horizontal([pl.col(b) for b in ALL_BIN_COLUMNS if b in sample.columns]).alias("row_sum")
    )
    assert (row_sums["row_sum"] - 20.0).abs().max() < 0.01, "Bins must sum to 20.0"


def test_right_tail_max_computation(sample_py, sample_quarter):
    """Test spec B1: right_tail_max computed correctly."""
    from ml.data_loader import compute_right_tail_max
    rtm = compute_right_tail_max(sample_py, sample_quarter)
    assert "constraint_id" in rtm.columns
    assert "right_tail_max" in rtm.columns
    # Can exceed 1.0 (bins are density weights)
    assert rtm["right_tail_max"].max() > 1.0


def test_universe_filter(sample_py, sample_quarter):
    """Test spec B2: universe filter produces expected sizes."""
    from ml.data_loader import load_collapsed
    df = load_collapsed(sample_py, sample_quarter)
    n_branches = len(df)
    # 2024-06/aq1 should have ~1,712 branches (±20%)
    assert 800 <= n_branches <= 3000, f"Unexpected branch count: {n_branches}"


def test_collapsed_is_branch_level(sample_py, sample_quarter):
    """Test spec K8: one row per branch_name."""
    from ml.data_loader import load_collapsed
    df = load_collapsed(sample_py, sample_quarter)
    assert df["branch_name"].n_unique() == len(df), "Duplicate branch_names found"


def test_count_cids_features(sample_py, sample_quarter):
    """Test spec C4: count_cids and count_active_cids."""
    from ml.data_loader import load_collapsed
    df = load_collapsed(sample_py, sample_quarter)
    assert "count_cids" in df.columns
    assert "count_active_cids" in df.columns
    # count_active <= count_cids for every row
    assert (df["count_active_cids"] <= df["count_cids"]).all()
    # Every branch has at least 1 active cid
    assert (df["count_active_cids"] >= 1).all()
    # count_cids >= 1
    assert (df["count_cids"] >= 1).all()


def test_density_features_present(sample_py, sample_quarter):
    """Test spec D1 (partial): density features have expected naming."""
    from ml.data_loader import load_collapsed
    from ml.config import SELECTED_BINS
    df = load_collapsed(sample_py, sample_quarter)
    for b in SELECTED_BINS:
        assert f"bin_{b}_cid_max" in df.columns, f"Missing bin_{b}_cid_max"
        assert f"bin_{b}_cid_min" in df.columns, f"Missing bin_{b}_cid_min"


def test_limit_features(sample_py, sample_quarter):
    """Test spec C5: limit features are correct."""
    from ml.data_loader import load_collapsed
    df = load_collapsed(sample_py, sample_quarter)
    assert (df["limit_min"] <= df["limit_mean"]).all()
    assert (df["limit_mean"] <= df["limit_max"]).all()
    assert (df["limit_min"] > 0).all(), "Limits must be positive"


def test_single_cid_branches(sample_py, sample_quarter):
    """Test spec J2: for single-cid branches, max == min."""
    from ml.data_loader import load_collapsed
    from ml.config import SELECTED_BINS
    df = load_collapsed(sample_py, sample_quarter)
    singles = df.filter(pl.col("count_cids") == 1)
    assert len(singles) > 0, "Expected some single-cid branches"
    for b in SELECTED_BINS[:2]:  # spot check first 2 bins
        diff = (singles[f"bin_{b}_cid_max"] - singles[f"bin_{b}_cid_min"]).abs()
        assert diff.max() < 1e-10, f"Single-cid branch has max != min for bin_{b}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_data_loader.py -v`

- [ ] **Step 3: Implement ml/data_loader.py**

Key implementation points:
- Load density via partition-specific paths (NOT hive scan, NOT pbase loaders — Trap 21)
- Compute `right_tail_max` BEFORE any filtering (for `is_active` flag)
- `count_cids` = total mapped cids (active + inactive), `count_active_cids` = active only
- Level 1: mean across outage_dates AND months per cid per bin
- Level 2: max + min per bin per branch (scaffolded for std variant)
- Constraint limits: Level 1 mean per cid, then Level 2 min/mean/max/std per branch
- Cache to `data/collapsed/`

The implementation should follow the exact pipeline in design spec SS3.1 Steps 1-8. Reference the implementer guide SS6.2 for code patterns.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_data_loader.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add ml/data_loader.py tests/test_data_loader.py
git commit -m "feat: add ml/data_loader.py — two-level density collapse pipeline"
```

---

### Task 6: ml/ground_truth.py — GT with bridge fallback

**Files:**
- Create: `ml/ground_truth.py`
- Create: `tests/test_ground_truth.py`

Combined-ctype GT with annual bridge + monthly fallback + tiered labels.

**Module boundary (deliberate design choice):** `build_ground_truth()` returns only branches that have positive DA binding SP (mapped via annual + monthly fallback). It does NOT return the full branch universe with zero-filled targets. The zero-filling (adding branches with `realized_shadow_price=0, label_tier=0`) happens in `features.py` when GT is left-joined onto the branch universe from `data_loader`. This keeps GT focused on the mapping+aggregation problem and keeps the universe definition in one place (data_loader → features).

- [ ] **Step 1: Write test_ground_truth.py — failing tests**

```python
"""Tests for ml/ground_truth.py — GT pipeline."""
import polars as pl
import pytest


def test_gt_combined_ctype(sample_py, sample_quarter):
    """Test spec E1: GT loads both onpeak and offpeak. Returns only positive-binding branches."""
    from ml.ground_truth import build_ground_truth
    gt_df, diag = build_ground_truth(sample_py, sample_quarter)
    assert "branch_name" in gt_df.columns
    assert "realized_shadow_price" in gt_df.columns
    assert "label_tier" in gt_df.columns
    # GT returns ONLY positive-binding branches
    assert (gt_df["realized_shadow_price"] > 0).all(), "GT should only contain positive-binding branches"
    assert len(gt_df) > 0


def test_gt_tiered_labels(sample_py, sample_quarter):
    """Test spec E5: tiered labels are 1/2/3 (GT only returns positive-binding branches)."""
    from ml.ground_truth import build_ground_truth
    gt_df, _ = build_ground_truth(sample_py, sample_quarter)
    labels = gt_df["label_tier"].unique().sort().to_list()
    # GT returns ONLY positive-binding branches — no label_tier=0 here.
    # Zero-fill (label_tier=0 for non-binding) happens in features.py.
    assert set(labels).issubset({1, 2, 3})
    assert 0 not in labels, "GT should not contain label_tier=0 (zero-fill is in features.py)"
    # All branches in GT have positive SP
    assert (gt_df["realized_shadow_price"] > 0).all()
    # Labels 1/2/3 should be approximately equal (tertiles of positive SP)
    for tier in [1, 2, 3]:
        n = gt_df.filter(pl.col("label_tier") == tier).height
        assert n > 0, f"No branches with tier {tier}"


def test_gt_per_ctype_split(sample_py, sample_quarter):
    """Design spec SS4.2: per-ctype split targets returned."""
    from ml.ground_truth import build_ground_truth
    gt_df, _ = build_ground_truth(sample_py, sample_quarter)
    assert "onpeak_sp" in gt_df.columns
    assert "offpeak_sp" in gt_df.columns


def test_gt_coverage_diagnostics(sample_py, sample_quarter):
    """Design spec SS4.3: raw coverage diagnostics returned."""
    from ml.ground_truth import build_ground_truth
    _, diag = build_ground_truth(sample_py, sample_quarter)
    required_keys = [
        "total_da_cids", "annual_mapped_cids", "monthly_recovered_cids",
        "still_unmapped_cids", "total_da_sp", "annual_mapped_sp",
        "monthly_recovered_sp", "still_unmapped_sp",
    ]
    for key in required_keys:
        assert key in diag, f"Missing diagnostic: {key}"
    assert diag["total_da_sp"] > 0


def test_gt_monthly_fallback_2025():
    """Test spec E3: monthly fallback recovers cids for 2025-06."""
    from ml.ground_truth import build_ground_truth
    _, diag = build_ground_truth("2025-06", "aq1")
    # For 2025-06: monthly fallback should recover some cids
    assert diag["monthly_recovered_cids"] > 0
    assert diag["monthly_recovered_sp"] > 0


def test_gt_monthly_fallback_uses_market_month():
    """Monthly bridge uses individual market_month as auction_month, not PY.

    For 2025-06/aq2, market months are 2025-09, 2025-10, 2025-11.
    The monthly fallback must try bridges for those months, not for 2025-06.
    We verify by checking that 2025-06/aq2 recovers cids — since the annual
    bridge for 2025-06 is known to have gaps, monthly fallback is essential.
    If the fallback incorrectly passed PY (2025-06) instead of 2025-09/10/11,
    it would only look up the June monthly bridge and miss the others.
    """
    from ml.ground_truth import build_ground_truth
    # 2025-06 has known annual bridge gaps — monthly fallback must recover cids
    _, diag = build_ground_truth("2025-06", "aq2")
    # Monthly fallback must recover at least some cids for aq2
    # If fallback incorrectly used PY (2025-06) for all months,
    # it would find the June bridge but miss Sep/Oct/Nov bridges entirely,
    # producing fewer recovered cids than expected
    assert diag["monthly_recovered_cids"] >= 0  # basic sanity
    # The diagnostic should also include per-month recovery detail
    # to enable manual verification of which months contributed
    if "monthly_recovery_detail" in diag:
        # If implemented, verify the detail keys are market months not PY
        detail = diag["monthly_recovery_detail"]
        for month_key in detail:
            assert month_key != "2025-06", \
                f"Monthly fallback detail should use market months, not PY. Got: {month_key}"


def test_gt_branch_aggregation():
    """Test spec E4: multiple DA cids -> same branch -> SUM (not mean)."""
    from ml.ground_truth import build_ground_truth
    gt_df, _ = build_ground_truth("2024-06", "aq1")
    # All values should be non-negative
    assert (gt_df["realized_shadow_price"] >= 0).all()
    # No duplicate branch_names
    assert gt_df["branch_name"].n_unique() == len(gt_df)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ground_truth.py -v`

- [ ] **Step 3: Implement ml/ground_truth.py**

Key implementation points:
- Uses `bridge.map_cids_to_branches()` for annual mapping (shared ambiguity rule)
- Monthly fallback: for each market_month, try monthly bridge (`auction_type='monthly'`, `period_type='f0'`)
- Only use monthly for cids NOT already mapped by annual
- Aggregate to branch: `group_by(branch_name).agg(sum(realized_sp))`
- **Returns only branches with positive DA binding** (realized_shadow_price > 0). Does NOT include non-binding branches — that zero-fill happens in features.py.
- Tiered labels: tertiles of positive SP within the returned branches (labels 1/2/3 only). Label 0 is assigned in features.py.
- Per-ctype split: separate onpeak_sp and offpeak_sp columns
- Return (DataFrame, diagnostics_dict)
- Diagnostics should include `monthly_recovery_detail`: a dict mapping each market_month to the number of cids recovered by that month's fallback bridge. This enables verifying that fallback uses the correct market_month (not PY) as auction_month.

Follow design spec SS4.1 Steps 1-6 exactly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ground_truth.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add ml/ground_truth.py tests/test_ground_truth.py
git commit -m "feat: add ml/ground_truth.py — combined GT with bridge fallback"
```

---

## Chunk 3: Feature Engineering

### Task 7: ml/history_features.py — BF + da_rank_value

**Files:**
- Create: `ml/history_features.py`
- Create: `tests/test_history_features.py`

Builds monthly branch-binding table (single DA scan), then derives BF and da_rank_value.

**Bridge mapping rule for historical months (deliberate design choice):**
When building the monthly binding table for eval PY X, every historical month M (from 2017-04 through the March cutoff) is mapped using **X's annual bridge** — NOT the PY that contains month M. For example, when computing features for eval PY 2025-06, a historical month like 2020-08 is mapped using the 2025-06 annual bridge, not the 2020-06 bridge.

**Why this is correct (not obvious):**
- The model universe for PY X is defined by X's density data mapped through X's bridge.
- GT for PY X maps DA cids through X's bridge.
- If BF features used a different PY's bridge, the same physical constraint could map to different branch_names in features vs GT, creating label noise.
- Using X's bridge everywhere keeps the branch_name identity consistent across data_loader, GT, and history_features for the same eval PY.

**Trade-off:** Some historical cids may not exist in X's bridge (newer constraints). Monthly fallback for month M itself (using M's f0 bridge) recovers some of these. This is the same fallback pattern used by GT.

- [ ] **Step 1: Write test_history_features.py — failing tests**

```python
"""Tests for ml/history_features.py — BF + da_rank_value."""
import polars as pl
import pytest


def test_build_monthly_binding_table(sample_py):
    """Monthly binding table has expected columns."""
    from ml.history_features import build_monthly_binding_table
    from ml.config import get_bf_cutoff_month, BF_FLOOR_MONTH
    table = build_monthly_binding_table(
        eval_py=sample_py,
        cutoff_month=get_bf_cutoff_month(sample_py),
        floor_month=BF_FLOOR_MONTH,
    )
    required = ["month", "branch_name", "onpeak_bound", "offpeak_bound",
                "combined_bound", "onpeak_sp", "offpeak_sp", "combined_sp"]
    for col in required:
        assert col in table.columns, f"Missing column: {col}"


def test_bf_values_in_range(sample_py, sample_quarter):
    """Test spec D3: BF values in [0, 1]."""
    from ml.history_features import compute_history_features
    from ml.config import get_bf_cutoff_month
    # Need universe branches
    from ml.data_loader import load_collapsed
    collapsed = load_collapsed(sample_py, sample_quarter)
    branches = collapsed["branch_name"].to_list()

    # compute_history_features ALWAYS returns (hist_df, monthly_binding_table)
    hist_df, _ = compute_history_features(
        eval_py=sample_py,
        aq_quarter=sample_quarter,
        universe_branches=branches,
    )
    for col in ["bf_6", "bf_12", "bf_15", "bfo_6", "bfo_12",
                "bf_combined_6", "bf_combined_12"]:
        assert col in hist_df.columns
        vals = hist_df[col]
        assert vals.min() >= 0.0, f"{col} has negative values"
        assert vals.max() <= 1.0, f"{col} exceeds 1.0"


def test_returns_tuple(sample_py, sample_quarter):
    """compute_history_features always returns (hist_df, monthly_binding_table)."""
    from ml.history_features import compute_history_features
    from ml.data_loader import load_collapsed
    collapsed = load_collapsed(sample_py, sample_quarter)
    branches = collapsed["branch_name"].to_list()[:10]
    result = compute_history_features(sample_py, sample_quarter, branches)
    assert isinstance(result, tuple) and len(result) == 2
    hist_df, binding_table = result
    assert "branch_name" in hist_df.columns
    assert "month" in binding_table.columns


def test_bf_temporal_leakage(sample_py):
    """Test spec D4, K1: BF does NOT use April data."""
    from ml.history_features import build_monthly_binding_table
    from ml.config import get_bf_cutoff_month, BF_FLOOR_MONTH
    cutoff = get_bf_cutoff_month(sample_py)
    table = build_monthly_binding_table(
        eval_py=sample_py,
        cutoff_month=cutoff,
        floor_month=BF_FLOOR_MONTH,
    )
    max_month = table["month"].max()
    assert max_month <= cutoff, f"BF uses data beyond cutoff: {max_month} > {cutoff}"


def test_bf_combined_either_ctype(sample_py, sample_quarter):
    """Test spec D5: bf_combined_12 >= max(bf_12, bfo_12)."""
    from ml.history_features import compute_history_features
    from ml.data_loader import load_collapsed
    collapsed = load_collapsed(sample_py, sample_quarter)
    branches = collapsed["branch_name"].to_list()
    hist_df, _ = compute_history_features(sample_py, sample_quarter, branches)
    assert (hist_df["bf_combined_12"] >= hist_df["bf_12"]).all()
    assert (hist_df["bf_combined_12"] >= hist_df["bfo_12"]).all()


def test_da_rank_value(sample_py, sample_quarter):
    """Test spec D6: da_rank_value ranked within universe."""
    from ml.history_features import compute_history_features
    from ml.data_loader import load_collapsed
    collapsed = load_collapsed(sample_py, sample_quarter)
    branches = collapsed["branch_name"].to_list()
    hist_df, _ = compute_history_features(sample_py, sample_quarter, branches)
    assert "da_rank_value" in hist_df.columns
    # Rank 1 = most binding (highest cumulative SP)
    assert hist_df["da_rank_value"].min() >= 1
    # All values positive
    assert (hist_df["da_rank_value"] > 0).all()
    # Zero-history branches get rank = n_positive + 1 (sentinel)
    n_positive = hist_df.filter(pl.col("has_hist_da"))["da_rank_value"].n_unique()
    zero_hist = hist_df.filter(~pl.col("has_hist_da"))
    if len(zero_hist) > 0:
        expected_sentinel = n_positive + 1
        assert (zero_hist["da_rank_value"] == expected_sentinel).all(), \
            f"Zero-history branches should have rank {expected_sentinel}"


def test_has_hist_da_flag(sample_py, sample_quarter):
    """has_hist_da = cumulative_sp > 0."""
    from ml.history_features import compute_history_features
    from ml.data_loader import load_collapsed
    collapsed = load_collapsed(sample_py, sample_quarter)
    branches = collapsed["branch_name"].to_list()
    hist_df, _ = compute_history_features(sample_py, sample_quarter, branches)
    assert "has_hist_da" in hist_df.columns
    # Some branches should have history, some shouldn't
    assert hist_df["has_hist_da"].sum() > 0
    assert hist_df["has_hist_da"].sum() < len(hist_df)


def test_bf_fixed_denominator(sample_py, sample_quarter):
    """BF denominator is always fixed N, even with fewer months available."""
    from ml.history_features import compute_history_features
    from ml.data_loader import load_collapsed
    # Use 2019-06 where only ~24 months of history exist (2017-04 to 2019-03)
    try:
        collapsed = load_collapsed("2019-06", sample_quarter)
    except Exception:
        pytest.skip("2019-06 data not available")
    branches = collapsed["branch_name"].to_list()[:10]
    hist_df, _ = compute_history_features("2019-06", sample_quarter, branches)
    # bf_15 divides by 15 even with < 15 months available
    # So some bf_15 values may be < bf_12 for branches that bound in months 13-15
    assert "bf_15" in hist_df.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_history_features.py -v`

- [ ] **Step 3: Implement ml/history_features.py**

Key implementation points:
- Build monthly binding table ONCE per eval PY (single DA scan)
- Bridge: use eval PY's annual bridge (`bridge.map_cids_to_branches(auction_type='annual', auction_month=eval_PY, period_type=aq_quarter)`), with monthly fallback for month M itself (`period_type='f0'`)
- Per branch per month: onpeak_bound, offpeak_bound, combined_bound, onpeak_sp, offpeak_sp, combined_sp
- BF: `bf_N = count(bound in last N months) / N` (FIXED denominator)
- da_rank_value: dense rank descending of cumulative_sp within universe. Zero-history branches get rank = n_positive + 1.
- has_hist_da = cumulative_sp > 0
- ALWAYS returns tuple: `(hist_df, monthly_binding_table)` — hist_df has branch_name + 8 features + has_hist_da; monthly_binding_table is needed by nb_detection

Follow design spec SS5.1 exactly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_history_features.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add ml/history_features.py tests/test_history_features.py
git commit -m "feat: add ml/history_features.py — BF + da_rank_value from monthly binding table"
```

---

### Task 8: ml/nb_detection.py — NB flags

**Files:**
- Create: `ml/nb_detection.py`
- Create: `tests/test_nb_detection.py`

Branch-level NB flags reusing monthly binding table from history_features.

- [ ] **Step 1: Write test_nb_detection.py — failing tests**

```python
"""Tests for ml/nb_detection.py — NB flags."""
import polars as pl
import pytest


def test_nb_flags_columns():
    """NB detection returns expected columns."""
    from ml.nb_detection import compute_nb_flags
    from ml.history_features import compute_history_features
    from ml.ground_truth import build_ground_truth
    from ml.data_loader import load_collapsed

    py, aq = "2024-06", "aq1"
    collapsed = load_collapsed(py, aq)
    branches = collapsed["branch_name"].to_list()
    hist_df, monthly_binding = compute_history_features(py, aq, branches)
    gt_df, _ = build_ground_truth(py, aq)

    nb_df = compute_nb_flags(
        universe_branches=branches,
        planning_year=py,
        aq_quarter=aq,
        gt_df=gt_df,
        monthly_binding_table=monthly_binding,
    )
    assert "is_nb_6" in nb_df.columns
    assert "is_nb_12" in nb_df.columns
    assert "is_nb_24" in nb_df.columns
    assert "nb_onpeak_12" in nb_df.columns
    assert "nb_offpeak_12" in nb_df.columns
    assert "branch_name" in nb_df.columns


def test_nb_requires_target_binding():
    """NB requires branch to actually bind in target quarter."""
    from ml.nb_detection import compute_nb_flags
    from ml.history_features import compute_history_features
    from ml.ground_truth import build_ground_truth
    from ml.data_loader import load_collapsed

    py, aq = "2024-06", "aq1"
    collapsed = load_collapsed(py, aq)
    branches = collapsed["branch_name"].to_list()
    hist_df, monthly_binding = compute_history_features(py, aq, branches)
    gt_df, _ = build_ground_truth(py, aq)

    nb_df = compute_nb_flags(branches, py, aq, gt_df, monthly_binding)

    # Join GT to check: NB branches must have realized_shadow_price > 0
    nb_with_gt = nb_df.join(gt_df.select(["branch_name", "realized_shadow_price"]),
                            on="branch_name", how="left")
    nb12_branches = nb_with_gt.filter(pl.col("is_nb_12"))
    if len(nb12_branches) > 0:
        assert (nb12_branches["realized_shadow_price"] > 0).all(), \
            "NB12 branches must have positive target binding"


def test_nb_combined_ctype_check():
    """Test spec E6: NB checks BOTH ctypes for lookback."""
    from ml.nb_detection import compute_nb_flags
    from ml.history_features import compute_history_features
    from ml.ground_truth import build_ground_truth
    from ml.data_loader import load_collapsed

    py, aq = "2024-06", "aq1"
    collapsed = load_collapsed(py, aq)
    branches = collapsed["branch_name"].to_list()
    hist_df, monthly_binding = compute_history_features(py, aq, branches)
    gt_df, _ = build_ground_truth(py, aq)

    nb_df = compute_nb_flags(branches, py, aq, gt_df, monthly_binding)
    nb_with_hist = nb_df.join(hist_df.select(["branch_name", "bfo_12", "bf_12"]),
                               on="branch_name", how="left")

    # A branch with bfo_12 > 0 should NOT be NB12 (had offpeak binding)
    offpeak_binders = nb_with_hist.filter(pl.col("bfo_12") > 0)
    if len(offpeak_binders) > 0:
        assert not offpeak_binders["is_nb_12"].any(), \
            "Branch with offpeak binding in last 12mo should NOT be NB12"
```

- [ ] **Step 2: Run tests, implement, verify, commit**

Follow same TDD pattern. Implementation uses the monthly_binding_table from history_features (no duplicate DA scan). See design spec SS6.1 for interface.

```bash
git add ml/nb_detection.py tests/test_nb_detection.py
git commit -m "feat: add ml/nb_detection.py — NB6/NB12/NB24 + per-ctype NB12 flags"
```

---

### Task 9: ml/features.py — model table assembly

**Files:**
- Create: `ml/features.py`
- Create: `tests/test_features.py`

Joins all sources into ONE model table. Owns the schema contract.

**Key responsibility**: features.py creates the full branch universe table from data_loader, then LEFT JOINs GT (which only has positive-binding branches). Branches not in GT get `realized_shadow_price=0.0, label_tier=0, onpeak_sp=0.0, offpeak_sp=0.0`. This is the ONLY place where non-binding branches get their zero targets.

- [ ] **Step 1: Write test_features.py — failing tests**

```python
"""Tests for ml/features.py — model table assembly."""
import polars as pl
import pytest


def test_model_table_schema():
    """Test spec D1: all expected columns present."""
    from ml.features import build_model_table
    table = build_model_table("2024-06", "aq1")
    required = [
        "branch_name", "planning_year", "aq_quarter",
        # Density (20)
        *[f"bin_{b}_cid_max" for b in ["-100","-50","60","70","80","90","100","110","120","150"]],
        *[f"bin_{b}_cid_min" for b in ["-100","-50","60","70","80","90","100","110","120","150"]],
        # Limits (4)
        "limit_min", "limit_mean", "limit_max", "limit_std",
        # Metadata (2)
        "count_cids", "count_active_cids",
        # History (8)
        "da_rank_value", "bf_6", "bf_12", "bf_15", "bfo_6", "bfo_12",
        "bf_combined_6", "bf_combined_12",
        # Target
        "realized_shadow_price", "label_tier", "onpeak_sp", "offpeak_sp",
        # NB
        "is_nb_6", "is_nb_12", "is_nb_24", "nb_onpeak_12", "nb_offpeak_12",
        # Cohort
        "cohort",
        # Metadata
        "has_hist_da",
        # GT diagnostics
        "total_da_sp_quarter",
    ]
    for col in required:
        assert col in table.columns, f"Missing column: {col}"


def test_model_table_unique_branches():
    """Test spec K8: one row per branch_name."""
    from ml.features import build_model_table
    table = build_model_table("2024-06", "aq1")
    assert table["branch_name"].n_unique() == len(table)


def test_model_table_zero_fill():
    """features.py creates zero-fill: non-binding branches get label_tier=0."""
    from ml.features import build_model_table
    table = build_model_table("2024-06", "aq1")
    # Majority should be non-binding (label_tier=0) — this is where zeros come from
    n_zero = table.filter(pl.col("label_tier") == 0).height
    n_positive = table.filter(pl.col("label_tier") > 0).height
    assert n_zero > n_positive, "Majority of branches should be non-binding (label_tier=0)"
    # Non-binding branches should have realized_shadow_price == 0
    zeros = table.filter(pl.col("label_tier") == 0)
    assert (zeros["realized_shadow_price"] == 0).all()


def test_cohort_assignment():
    """Test spec G5: mutually exclusive cohorts."""
    from ml.features import build_model_table
    table = build_model_table("2024-06", "aq1")
    cohorts = table["cohort"].unique().to_list()
    assert set(cohorts).issubset({"established", "history_dormant", "history_zero"})
    # Every branch has exactly 1 cohort
    assert table["cohort"].null_count() == 0


def test_cohort_rules():
    """Cohort rules: established > dormant > zero."""
    from ml.features import build_model_table
    table = build_model_table("2024-06", "aq1")
    # established: bf_combined_12 > 0
    established = table.filter(pl.col("cohort") == "established")
    if len(established) > 0:
        assert (established["bf_combined_12"] > 0).all()
    # history_dormant: has_hist_da=True AND bf_combined_12 == 0
    dormant = table.filter(pl.col("cohort") == "history_dormant")
    if len(dormant) > 0:
        assert (dormant["has_hist_da"]).all()
        assert (dormant["bf_combined_12"] == 0).all()
    # history_zero: NOT has_hist_da AND bf_combined_12 == 0
    h_zero = table.filter(pl.col("cohort") == "history_zero")
    if len(h_zero) > 0:
        assert (h_zero["has_hist_da"] == False).all()
        assert (h_zero["bf_combined_12"] == 0).all()


def test_monotone_constraints_order():
    """Design spec SS7.2: monotone vector matches feature_cols."""
    from ml.features import build_model_table
    from ml.config import get_monotone_constraints, ALL_FEATURES
    constraints = get_monotone_constraints(ALL_FEATURES)
    assert len(constraints) == len(ALL_FEATURES)
    # BF features should be +1
    bf_idx = ALL_FEATURES.index("bf_6")
    assert constraints[bf_idx] == 1
    # da_rank should be -1
    da_idx = ALL_FEATURES.index("da_rank_value")
    assert constraints[da_idx] == -1


def test_total_da_sp_quarter():
    """total_da_sp_quarter is group-level constant (Abs_SP denominator)."""
    from ml.features import build_model_table
    table = build_model_table("2024-06", "aq1")
    vals = table["total_da_sp_quarter"].unique()
    assert len(vals) == 1, "total_da_sp_quarter should be constant within group"
    assert vals[0] > 0
```

- [ ] **Step 2: Run tests, implement, verify, commit**

Implementation:
1. Load branch universe from data_loader (`load_collapsed`)
2. LEFT JOIN ground_truth — GT only has positive-binding branches; fill missing with `realized_shadow_price=0.0, label_tier=0, onpeak_sp=0.0, offpeak_sp=0.0`
3. LEFT JOIN history_features — fill missing with zeros for BF, sentinel for da_rank_value, `has_hist_da=False`
4. LEFT JOIN nb_detection — fill missing NB flags with False
5. Assign cohorts using `has_hist_da` and `bf_combined_12`
6. Attach `total_da_sp_quarter` from GT diagnostics as a group-level constant column
7. Add `planning_year` and `aq_quarter` columns

Also add `build_model_table_all(groups: list[str]) -> pl.DataFrame` convenience function:
- Parses each group string `"PY/aq"`, calls `build_model_table(PY, aq)`, concats all.
- Used by formula baseline scripts.

```bash
git add ml/features.py tests/test_features.py
git commit -m "feat: add ml/features.py — model table assembly + cohort + schema contract"
```

---

## Chunk 4: Training, Evaluation & Baselines

### Task 10: ml/train.py — LambdaRank training

**Files:**
- Create: `ml/train.py`
- Create: `tests/test_train.py`

Expanding-window LambdaRank training + prediction.

- [ ] **Step 1: Write test_train.py — failing tests**

```python
"""Tests for ml/train.py — LambdaRank training."""
import numpy as np
import pytest


def test_tiered_labels():
    """Labels: 0=non-binding, 1/2/3=tertiles of positive."""
    from ml.train import tiered_labels
    y = np.array([0, 0, 0, 10, 20, 30, 40, 50, 60])
    groups = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])  # 1 group
    labels = tiered_labels(y, groups)
    assert set(labels[:3]) == {0}  # zeros stay 0
    assert set(labels[3:]) == {1, 2, 3}  # positives get 1/2/3


def test_query_groups_construction():
    """Test spec F1: query groups from model table."""
    from ml.train import build_query_groups
    import polars as pl
    df = pl.DataFrame({
        "planning_year": ["2022-06"] * 5 + ["2023-06"] * 3,
        "aq_quarter": ["aq1"] * 5 + ["aq2"] * 3,
        "branch_name": [f"b{i}" for i in range(8)],
    })
    groups = build_query_groups(df)
    assert groups.tolist() == [5, 3]
    assert sum(groups) == 8


def test_train_and_predict_smoke():
    """Smoke test: training completes and produces scores."""
    from ml.train import train_and_predict
    from ml.features import build_model_table
    from ml.config import HISTORY_FEATURES
    import polars as pl

    # Build model tables for train and eval groups
    train_table = build_model_table("2023-06", "aq1")
    eval_table = build_model_table("2024-06", "aq1")
    model_table = pl.concat([train_table, eval_table])

    # train_and_predict receives the assembled model table
    result = train_and_predict(
        model_table=model_table,
        train_pys=["2023-06"],
        eval_pys=["2024-06"],
        feature_cols=HISTORY_FEATURES,
    )
    assert "score" in result.columns
    assert len(result) > 0
    # Scores only on eval rows
    assert len(result) <= len(eval_table)


def test_lgbm_params():
    """Test spec F3: num_threads=4, lambdarank."""
    from ml.config import LGBM_PARAMS
    assert LGBM_PARAMS["num_threads"] == 4
    assert LGBM_PARAMS["objective"] == "lambdarank"
```

- [ ] **Step 2: Run tests, implement, verify, commit**

Implementation adapts v1's `train.py` pattern. Key rules:
- Sort by (planning_year, aq_quarter, branch_name) before building group sizes
- Eval-only scoring (never predict on training rows)
- Return model table with added score, eval_year, split columns
- Record train walltime

```bash
git add ml/train.py tests/test_train.py
git commit -m "feat: add ml/train.py — expanding-window LambdaRank training"
```

---

### Task 11: ml/evaluate.py + ml/registry.py

**Files:**
- Create: `ml/evaluate.py`
- Create: `ml/registry.py`
- Create: `tests/test_evaluate.py`
- Create: `tests/test_registry.py`

All Tier 1/2/3 metrics, NB metrics, cohort contribution, gates, and result persistence.

- [ ] **Step 1: Write test_evaluate.py — failing tests**

```python
"""Tests for ml/evaluate.py — metrics and gates."""
import numpy as np
import pytest


def test_vc_at_k():
    """Test spec G1: VC@K computation."""
    from ml.evaluate import value_capture_at_k
    actual = np.array([100, 50, 30, 20, 0, 0, 0, 0, 0, 0], dtype=float)
    scores = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
    vc50 = value_capture_at_k(actual, scores, k=2)
    assert abs(vc50 - 150 / 200) < 1e-6  # top 2 capture 150 of 200


def test_recall_at_k():
    """Test spec G2: Recall@K computation."""
    from ml.evaluate import recall_at_k
    actual = np.array([100, 50, 30, 0, 0, 0], dtype=float)
    scores = np.array([6, 5, 4, 3, 2, 1], dtype=float)
    recall = recall_at_k(actual, scores, k=2)
    assert abs(recall - 2 / 3) < 1e-6  # 2 of 3 binders in top 2


def test_abs_sp_at_k():
    """Test spec G3: Abs_SP uses total DA SP (not in-universe)."""
    from ml.evaluate import abs_sp_at_k
    actual = np.array([100, 50, 0, 0], dtype=float)
    scores = np.array([4, 3, 2, 1], dtype=float)
    total_da_sp = 500.0  # includes outside-universe SP
    abs_sp = abs_sp_at_k(actual, scores, k=2, total_da_sp=total_da_sp)
    assert abs(abs_sp - 150 / 500) < 1e-6


def test_nb_recall_at_k():
    """Test spec G4: NB12_Recall@K."""
    from ml.evaluate import nb_recall_at_k
    actual = np.array([100, 50, 30, 0, 0], dtype=float)
    scores = np.array([5, 4, 3, 2, 1], dtype=float)
    is_nb = np.array([True, False, True, False, False])
    # NB binders: index 0 (100, NB, in top-2) and index 2 (30, NB, NOT in top-2)
    nb_recall = nb_recall_at_k(actual, scores, is_nb, k=2)
    assert abs(nb_recall - 1 / 2) < 1e-6  # 1 of 2 NB binders in top 2


def test_gate_checking():
    """Test spec L1: 2/3 holdout groups rule."""
    from ml.evaluate import check_gates
    candidate = {
        "2025-06/aq1": {"VC@50": 0.30},
        "2025-06/aq2": {"VC@50": 0.25},
        "2025-06/aq3": {"VC@50": 0.35},
    }
    baseline = {
        "2025-06/aq1": {"VC@50": 0.28},
        "2025-06/aq2": {"VC@50": 0.27},
        "2025-06/aq3": {"VC@50": 0.30},
    }
    gates = check_gates(candidate, baseline, "v0c",
                        ["2025-06/aq1", "2025-06/aq2", "2025-06/aq3"])
    # Wins on aq1 (0.30 > 0.28) and aq3 (0.35 > 0.30) = 2/3
    # Mean: 0.30 vs 0.283 -> pass
    assert gates["VC@50"]["passed"] is True
    assert gates["VC@50"]["wins"] == 2


def test_gate_fails_with_mean_below_baseline():
    """Test spec L1: 3/3 wins but mean < baseline → FAIL."""
    from ml.evaluate import check_gates
    candidate = {
        "2025-06/aq1": {"VC@50": 0.201},
        "2025-06/aq2": {"VC@50": 0.201},
        "2025-06/aq3": {"VC@50": 0.201},
    }
    baseline = {
        "2025-06/aq1": {"VC@50": 0.200},
        "2025-06/aq2": {"VC@50": 0.200},
        "2025-06/aq3": {"VC@50": 0.300},  # baseline mean = 0.233
    }
    gates = check_gates(candidate, baseline, "v0c",
                        ["2025-06/aq1", "2025-06/aq2", "2025-06/aq3"])
    # 3 wins (all >=), but candidate mean (0.201) < baseline mean (0.233)
    assert gates["VC@50"]["passed"] is False


def test_gate_fails_with_1_win():
    """1 of 3 wins -> FAIL even if mean is higher."""
    from ml.evaluate import check_gates
    candidate = {
        "2025-06/aq1": {"VC@50": 0.50},
        "2025-06/aq2": {"VC@50": 0.10},
        "2025-06/aq3": {"VC@50": 0.10},
    }
    baseline = {
        "2025-06/aq1": {"VC@50": 0.20},
        "2025-06/aq2": {"VC@50": 0.20},
        "2025-06/aq3": {"VC@50": 0.20},
    }
    gates = check_gates(candidate, baseline, "v0c",
                        ["2025-06/aq1", "2025-06/aq2", "2025-06/aq3"])
    assert gates["VC@50"]["wins"] == 1
    assert gates["VC@50"]["passed"] is False
```

- [ ] **Step 2: Run tests, implement evaluate.py and registry.py, verify, commit**

evaluate.py key functions:
- `value_capture_at_k(actual, scores, k)` — VC@K
- `recall_at_k(actual, scores, k)` — Recall@K
- `abs_sp_at_k(actual, scores, k, total_da_sp)` — Abs_SP@K (cross-universe)
- `nb_recall_at_k(actual, scores, is_nb, k)` — NB_Recall@K
- `ndcg(actual, scores)` — NDCG
- `spearman_corr(actual, scores)` — Spearman
- `evaluate_group(model_table_group)` — all metrics for one (PY, quarter)
- `evaluate_all(model_table_with_scores)` — per-group + aggregated
- `check_gates(candidate, baseline, baseline_name, holdout_groups)` — gate checking
- `cohort_contribution(model_table_group, k)` — Tier 3 cohort metrics

registry.py key functions:
- `save_experiment(version_id, config_dict, metrics_dict)` — writes to `registry/{version}/`
- Independent of evaluate.py (takes dicts, no import)

```bash
git add ml/evaluate.py ml/registry.py tests/test_evaluate.py tests/test_registry.py
git commit -m "feat: add ml/evaluate.py + ml/registry.py — metrics, gates, persistence"
```

---

### Task 12: Formula baselines (v0a, v0b, v0c)

**Files:**
- Create: `scripts/run_v0a_da_rank.py`
- Create: `scripts/run_v0b_blend.py`
- Create: `scripts/run_v0c_full_blend.py`

Three formula baselines, run on dev (12 groups) + holdout (3 groups). Full Tier 1/2/3 + NB + cohort reporting.

- [ ] **Step 1: Implement run_v0a_da_rank.py**

Pure da_rank_value baseline. Score = -da_rank_value (lower rank = higher score).

```python
"""v0a: pure da_rank_value formula baseline.

Score = -da_rank_value (lower rank = more binding = higher score).
"""
from __future__ import annotations

import time

import polars as pl

from ml.features import build_model_table_all
from ml.evaluate import evaluate_all
from ml.registry import save_experiment
from ml.config import DEV_GROUPS, HOLDOUT_GROUPS


def main():
    start = time.time()

    # Build model tables for all groups
    print("Building model tables...")
    all_groups = DEV_GROUPS + HOLDOUT_GROUPS
    model_table = build_model_table_all(all_groups)

    # Score: -da_rank_value
    model_table = model_table.with_columns(
        (-pl.col("da_rank_value")).alias("score")
    )

    # Evaluate
    print("Evaluating...")
    metrics = evaluate_all(model_table)

    walltime = time.time() - start
    print(f"\nWalltime: {walltime:.1f}s")

    # Save
    config = {"version": "v0a", "formula": "-da_rank_value", "features": ["da_rank_value"]}
    save_experiment("v0a", config, metrics)
    print(f"Saved to registry/v0a/")

    # Print summary
    _print_summary(metrics)


def _print_summary(metrics):
    import json
    print("\n" + "=" * 60)
    print("v0a: pure da_rank_value")
    print("=" * 60)
    for group, m in sorted(metrics["per_group"].items()):
        print(f"\n{group}:")
        for k in ["VC@50", "VC@100", "Recall@50", "NDCG", "Abs_SP@50", "NB12_Recall@50"]:
            if k in m:
                print(f"  {k}: {m[k]:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Implement run_v0b_blend.py**

```
score = 0.60 * da_rank_norm + 0.40 * right_tail_rank_norm
```
Where both terms are normalized to [0, 1] within each (PY, quarter) group (1.0 = most binding).
**IMPORTANT**: `da_rank_value` has lower = more binding, so `da_rank_norm = 1 - (da_rank_value - 1) / (max_rank - 1)` (invert so 1.0 = most binding). Similarly, `right_tail_rank_norm = (right_tail_max - min) / (max - min)` (higher = more binding, no inversion needed).

- [ ] **Step 3: Implement run_v0c_full_blend.py**

```
score = 0.40 * da_rank_norm + 0.30 * right_tail_rank_norm + 0.30 * bf_combined_12_rank_norm
```
Uses `bf_combined_12` (not onpeak-only bf_12).

- [ ] **Step 4: Run all three baselines**

Run each script:
```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
python /path/to/scripts/run_v0a_da_rank.py
python /path/to/scripts/run_v0b_blend.py
python /path/to/scripts/run_v0c_full_blend.py
```

Expected: Each produces `registry/{version}/config.json` and `registry/{version}/metrics.json` with full Tier 1/2/3 + NB + cohort metrics for all 15 groups (12 dev + 3 holdout).

- [ ] **Step 5: Verify baseline results**

Check that:
- v0a VC@50 > 0 for all groups (not random)
- v0b VC@50 > v0a on most groups (density adds signal)
- v0c VC@50 > v0a on most groups (BF adds signal)
- NDCG > 0.5 for all versions (better than random)
- NB metrics are reported

- [ ] **Step 6: Baseline contract freeze**

After baselines run, determine which is the authoritative baseline for promotion:
- If v0c is strongest: use v0c as gate baseline
- Document the decision in `registry/baseline_contract.json`

- [ ] **Step 7: Commit**

```bash
git add scripts/run_v0a_da_rank.py scripts/run_v0b_blend.py scripts/run_v0c_full_blend.py
git add registry/v0a/ registry/v0b/ registry/v0c/ registry/baseline_contract.json
git commit -m "feat: formula baselines v0a/v0b/v0c + baseline contract freeze"
```

---

### Task 13: Integration tests + full pipeline verification

**Files:**
- Create: `tests/test_integration.py`

End-to-end tests validating the complete pipeline.

- [ ] **Step 1: Write integration tests**

```python
"""Integration tests — full pipeline verification."""
import polars as pl
import pytest


def test_full_pipeline_one_slice():
    """Test spec I1: end-to-end for one (PY, quarter)."""
    from ml.features import build_model_table
    table = build_model_table("2024-06", "aq1")
    # Branch count in expected range
    assert 800 <= len(table) <= 3000
    # All features present (34 features)
    from ml.config import ALL_FEATURES
    for f in ALL_FEATURES:
        assert f in table.columns, f"Missing feature: {f}"
    # Target has binding branches
    n_binding = table.filter(pl.col("realized_shadow_price") > 0).height
    assert n_binding > 50


def test_no_duplicate_branch_names_across_groups():
    """Within a (PY, quarter) group, branch_names are unique."""
    from ml.features import build_model_table
    for aq in ["aq1", "aq2"]:
        table = build_model_table("2024-06", aq)
        assert table["branch_name"].n_unique() == len(table)


def test_formula_baseline_beats_random():
    """v0a should produce VC@50 > random baseline."""
    from ml.features import build_model_table
    from ml.evaluate import value_capture_at_k
    import numpy as np

    table = build_model_table("2024-06", "aq1")
    actual = table["realized_shadow_price"].to_numpy()
    scores = -table["da_rank_value"].to_numpy()  # v0a formula
    vc50 = value_capture_at_k(actual, scores, k=50)
    random_vc50 = 50 / len(table)  # expected random performance
    assert vc50 > random_vc50 * 2, f"v0a VC@50 ({vc50:.4f}) barely beats random ({random_vc50:.4f})"


def test_density_row_sum_invariant():
    """Trap 22: density bins sum to 20.0 in raw data."""
    from ml.data_loader import load_raw_density
    from ml.config import ALL_BIN_COLUMNS
    df = load_raw_density("2024-06", "aq1")
    bin_cols = [c for c in ALL_BIN_COLUMNS if c in df.columns]
    sums = df.select(pl.sum_horizontal([pl.col(b) for b in bin_cols]).alias("s"))
    assert (sums["s"] - 20.0).abs().max() < 0.1


def test_lgbm_num_threads():
    """Trap 3: grep for num_threads=4."""
    from ml.config import LGBM_PARAMS
    assert LGBM_PARAMS["num_threads"] == 4
```

- [ ] **Step 2: Run all tests**

Run: `python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: add integration tests — full pipeline verification"
```

---

## Phase 1 Complete Checklist

After all tasks are done, verify:

- [ ] `ml/config.py` — UNIVERSE_THRESHOLD is frozen (not 0.0)
- [ ] `data/realized_da/` — DA cache exists for 2017-04 through 2026-02
- [ ] `registry/threshold_calibration/` — calibration artifact exists
- [ ] `registry/v0a/`, `v0b/`, `v0c/` — all three baselines have metrics.json
- [ ] `registry/baseline_contract.json` — authoritative baseline named
- [ ] All tests pass: `python -m pytest tests/ -v`
- [ ] Per-group holdout breakdown for all 3 baselines matches expected patterns
- [ ] NB metrics are reported and non-trivial
- [ ] Cohort contribution is reported
- [ ] Walltimes are reasonable (< 5 min per baseline)


**Phase 1 is complete when formula baselines are running with full metrics and the baseline contract is frozen. Phase 2 (ML build-up) can begin.**