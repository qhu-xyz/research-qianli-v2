"""Tests for ml/features.py — model table assembly."""
import polars as pl
import pytest


def test_model_table_schema():
    """Test spec D1: all expected columns present."""
    from ml.features import build_model_table
    table = build_model_table("2025-06", "aq1")
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
    table = build_model_table("2025-06", "aq1")
    assert table["branch_name"].n_unique() == len(table)


def test_model_table_zero_fill():
    """features.py creates zero-fill: non-binding branches get label_tier=0."""
    from ml.features import build_model_table
    table = build_model_table("2025-06", "aq1")
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
    table = build_model_table("2025-06", "aq1")
    cohorts = table["cohort"].unique().to_list()
    assert set(cohorts).issubset({"established", "history_dormant", "history_zero"})
    # Every branch has exactly 1 cohort
    assert table["cohort"].null_count() == 0


def test_cohort_rules():
    """Cohort rules: established > dormant > zero."""
    from ml.features import build_model_table
    table = build_model_table("2025-06", "aq1")
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
        assert (h_zero["has_hist_da"] == False).all()  # noqa: E712
        assert (h_zero["bf_combined_12"] == 0).all()


def test_monotone_constraints_order():
    """Design spec SS7.2: monotone vector matches feature_cols."""
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
    table = build_model_table("2025-06", "aq1")
    vals = table["total_da_sp_quarter"].unique()
    assert len(vals) == 1, "total_da_sp_quarter should be constant within group"
    assert vals[0] > 0


def test_build_model_table_all():
    """Convenience function builds multiple groups."""
    from ml.features import build_model_table_all
    table = build_model_table_all(["2025-06/aq1", "2025-06/aq2"])
    groups = table.select(["planning_year", "aq_quarter"]).unique()
    assert len(groups) == 2
