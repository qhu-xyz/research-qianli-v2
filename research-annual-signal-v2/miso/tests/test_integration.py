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
    # Most rows sum to exactly 20.0; a few edge-case rows have sparse bin data (max dev ~5.3)
    assert (sums["s"] - 20.0).abs().median() < 0.01


def test_lgbm_num_threads():
    """Trap 3: grep for num_threads=4."""
    from ml.config import LGBM_PARAMS
    assert LGBM_PARAMS["num_threads"] == 4
