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

# Universe threshold — calibrated 2026-03-12 on 2024-06/aq1 at BRANCH level.
# 95% of annual-bridge-mapped branch SP captured at 2,339 branches.
# Applied at CID level (cid is "active" if right_tail_max >= threshold).
# Cross-check 2023-06/aq1: ratio=1.07.
# See registry/threshold_calibration/threshold.json for full artifact.
UNIVERSE_THRESHOLD: float = 0.0003467728739657263

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
    "NDCG", "Abs_SP@50",
]

TWO_TRACK_GATE_METRICS: list[str] = ["VC@50", "Recall@50", "Abs_SP@50"]

# Gate rule: candidate must beat baseline on >=2 of 3 holdout groups + mean >= baseline
GATE_MIN_WINS: int = 2
GATE_HOLDOUT_COUNT: int = 3

# ─── Phase 5: new metric universe ──────────────────────────────────────
PHASE5_K_LEVELS: list[int] = [150, 200, 300, 400]
DANGEROUS_THRESHOLD: float = 50000.0
PHASE5_GATE_METRICS_150_300: list[str] = [
    "VC@150", "VC@300", "Recall@150", "Recall@300",
    "Abs_SP@150", "Abs_SP@300", "Dang_Recall@300",
]
PHASE5_GATE_METRICS_200_400: list[str] = [
    "VC@200", "VC@400", "Recall@200", "Recall@400",
    "Abs_SP@200", "Abs_SP@400", "Dang_Recall@400",
]

# ─── Phase 6: class-specific pipeline ──────────────────────────────────
CLASS_TYPES: list[str] = ["onpeak", "offpeak"]

# Per-class column mappings
CLASS_BF_COL: dict[str, str] = {"onpeak": "bf_12", "offpeak": "bfo_12"}
CLASS_TARGET_COL: dict[str, str] = {"onpeak": "onpeak_sp", "offpeak": "offpeak_sp"}
CLASS_NB_FLAG_COL: dict[str, str] = {"onpeak": "nb_onpeak_12", "offpeak": "nb_offpeak_12"}
# Cross-class BF: the OTHER class's BF column
CROSS_CLASS_BF_COL: dict[str, str] = {"onpeak": "bfo_12", "offpeak": "bf_12"}
