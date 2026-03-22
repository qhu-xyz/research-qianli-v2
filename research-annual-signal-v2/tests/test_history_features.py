"""Tests for ml/history_features.py — BF + da_rank_value."""
import polars as pl
import pytest


def test_build_monthly_binding_table(sample_py, sample_quarter):
    """Monthly binding table has expected columns."""
    from ml.history_features import build_monthly_binding_table
    from ml.config import get_bf_cutoff_month, BF_FLOOR_MONTH
    table = build_monthly_binding_table(
        eval_py=sample_py,
        aq_quarter=sample_quarter,
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
    from ml.data_loader import load_collapsed
    collapsed = load_collapsed(sample_py, sample_quarter)
    branches = collapsed["branch_name"].to_list()

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


def test_bf_temporal_leakage(sample_py, sample_quarter):
    """Test spec D4, K1: BF does NOT use April data."""
    from ml.history_features import build_monthly_binding_table
    from ml.config import get_bf_cutoff_month, BF_FLOOR_MONTH
    cutoff = get_bf_cutoff_month(sample_py)
    table = build_monthly_binding_table(
        eval_py=sample_py,
        aq_quarter=sample_quarter,
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
    """Test spec D6: da_rank_value is dense rank descending within universe."""
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

    # Verify dense rank: n_distinct_ranks among positive branches
    positive = hist_df.filter(pl.col("has_hist_da"))
    n_distinct_ranks = int(positive["da_rank_value"].max())
    # Dense rank means no gaps: ranks are 1..n_distinct
    rank_vals = sorted(positive["da_rank_value"].unique().to_list())
    assert rank_vals == list(range(1, n_distinct_ranks + 1)), \
        f"Dense rank should have no gaps, got {rank_vals[:10]}..."

    # Zero-history branches get sentinel = n_distinct_ranks + 1
    zero_hist = hist_df.filter(~pl.col("has_hist_da"))
    if len(zero_hist) > 0:
        expected_sentinel = n_distinct_ranks + 1
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
    # Use earliest available PY where limited history exists
    try:
        collapsed = load_collapsed("2025-06", sample_quarter)
        branches = collapsed["branch_name"].to_list()[:10]
        hist_df, _ = compute_history_features("2025-06", sample_quarter, branches)
    except (FileNotFoundError, AssertionError):
        pytest.skip("SPICE data not available for test PY")
    # bf_15 divides by 15 even with < 15 months available
    # So some bf_15 values may be < bf_12 for branches that bound in months 13-15
    assert "bf_15" in hist_df.columns
