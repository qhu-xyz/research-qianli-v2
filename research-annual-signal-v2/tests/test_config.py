"""Tests for ml/config.py — paths, constants, splits, params."""
import pytest


def test_data_paths_exist():
    """Data paths point to existing locations."""
    from pathlib import Path
    from ml.config import DENSITY_PATH, BRIDGE_PATH, LIMIT_PATH
    assert Path(DENSITY_PATH).exists(), f"Density not found: {DENSITY_PATH}"
    assert Path(BRIDGE_PATH).exists(), f"Bridge not found: {BRIDGE_PATH}"
    assert Path(LIMIT_PATH).exists(), f"Limits not found: {LIMIT_PATH}"


def test_bin_columns():
    """77 bin columns, all strings."""
    from ml.config import ALL_BIN_COLUMNS
    assert len(ALL_BIN_COLUMNS) == 77
    assert ALL_BIN_COLUMNS[0] == "-300"
    assert ALL_BIN_COLUMNS[-1] == "300"


def test_selected_bins():
    """10 selected bins from implementer-guide SS7.2."""
    from ml.config import SELECTED_BINS, ALL_BIN_COLUMNS
    assert len(SELECTED_BINS) == 10
    for b in SELECTED_BINS:
        assert b in ALL_BIN_COLUMNS, f"Selected bin {b} not in ALL_BIN_COLUMNS"


def test_right_tail_bins():
    """Right-tail bins for universe filter: 80, 90, 100, 110."""
    from ml.config import RIGHT_TAIL_BINS
    assert RIGHT_TAIL_BINS == ["80", "90", "100", "110"]


def test_get_market_months():
    """Market months derived from (PY, quarter)."""
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
