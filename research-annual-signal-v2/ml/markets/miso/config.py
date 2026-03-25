"""MISO-specific constants, paths, and eval splits for annual signal pipeline."""
from __future__ import annotations

from pathlib import Path

# ─── Data paths (MISO NFS) ───────────────────────────────────────────
DENSITY_PATH = "/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_DENSITY_DISTRIBUTION.parquet"
BRIDGE_PATH = "/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_INFO.parquet"
LIMIT_PATH = "/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_LIMIT.parquet"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DA_CACHE_DIR = PROJECT_ROOT / "data" / "realized_da"
COLLAPSED_CACHE_DIR = PROJECT_ROOT / "data" / "collapsed"
REGISTRY_DIR = PROJECT_ROOT / "registry"

# ─── Density bins ─────────────────────────────────────────────────────
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

SELECTED_BINS: list[str] = [
    "-100", "-50", "60", "70", "80", "90", "100", "110", "120", "150",
]

RIGHT_TAIL_BINS: list[str] = ["80", "90", "100", "110"]

UNIVERSE_THRESHOLD: float = 0.0003467728739657263

# ─── Planning years & quarters ────────────────────────────────────────
PLANNING_YEARS: list[str] = [
    "2019-06", "2020-06", "2021-06", "2022-06", "2023-06", "2024-06", "2025-06",
]
AQ_QUARTERS: list[str] = ["aq1", "aq2", "aq3", "aq4"]

# ─── BF ───────────────────────────────────────────────────────────────
BF_FLOOR_MONTH: str = "2016-04"
BF_WINDOWS_ONPEAK: list[int] = [6, 12, 15]
BF_WINDOWS_OFFPEAK: list[int] = [6, 12]
BF_WINDOWS_COMBINED: list[int] = [6, 12]

# ─── Class types ──────────────────────────────────────────────────────
CLASS_TYPES: list[str] = ["onpeak", "offpeak"]
CLASS_BF_COL: dict[str, str] = {"onpeak": "bf_12", "offpeak": "bfo_12"}
CLASS_TARGET_COL: dict[str, str] = {"onpeak": "onpeak_sp", "offpeak": "offpeak_sp"}
CLASS_NB_FLAG_COL: dict[str, str] = {"onpeak": "nb_onpeak_12", "offpeak": "nb_offpeak_12"}
CROSS_CLASS_BF_COL: dict[str, str] = {"onpeak": "bfo_12", "offpeak": "bf_12"}


def get_market_months(planning_year: str, aq_quarter: str) -> list[str]:
    """Derive 3 market months from (PY, quarter)."""
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
    """Legacy BF cutoff = March of submission year. Use get_history_cutoff_month instead."""
    py_year = int(planning_year[:4])
    return f"{py_year}-03"
