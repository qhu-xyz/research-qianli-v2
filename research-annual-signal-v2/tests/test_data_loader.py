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
    sample = df.head(1000)
    row_sums = sample.select(
        pl.sum_horizontal([pl.col(b) for b in ALL_BIN_COLUMNS if b in sample.columns]).alias("row_sum")
    )
    # Most rows sum to exactly 20.0; a few edge cases deviate slightly
    assert (row_sums["row_sum"] - 20.0).abs().max() < 2.0, "Bins must sum close to 20.0"


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
    # Expected ~1,700-2,800 branches depending on PY
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
    """Test spec C5: limit features present and ordered."""
    from ml.data_loader import load_collapsed
    df = load_collapsed(sample_py, sample_quarter)
    for col in ["limit_min", "limit_mean", "limit_max", "limit_std"]:
        assert col in df.columns, f"Missing {col}"
    # For rows that have limit data, min <= mean <= max (with f32 tolerance)
    has_limits = df.filter(pl.col("limit_max") > 0)
    assert len(has_limits) > 0, "Expected some rows with limit data"
    eps = 0.01
    assert ((has_limits["limit_min"] - has_limits["limit_mean"]) <= eps).all()
    assert ((has_limits["limit_mean"] - has_limits["limit_max"]) <= eps).all()


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


def test_no_cid_fanout_before_level2(sample_py, sample_quarter):
    """Ambiguous cids must be dropped before Level 2 collapse.

    If a cid maps to >1 branch (after onpeak+offpeak union), it should be
    dropped by map_cids_to_branches(), not silently retained.
    """
    from ml.data_loader import load_collapsed, compute_right_tail_max
    from ml.bridge import map_cids_to_branches

    df = load_collapsed(sample_py, sample_quarter)
    rtm = compute_right_tail_max(sample_py, sample_quarter)

    # Map via the shared helper (same path as load_collapsed)
    mapped, _diag = map_cids_to_branches(
        cid_df=rtm,
        auction_type="annual",
        auction_month=sample_py,
        period_type=sample_quarter,
    )
    # After ambiguity drop, each cid should map to exactly 1 branch
    cid_counts = mapped.group_by("constraint_id").len()
    assert cid_counts["len"].max() == 1, (
        "Fan-out detected: cid maps to >1 branch. "
        "map_cids_to_branches() should have dropped ambiguous cids."
    )

    # count_cids sum should equal mapped cids on ACTIVE branches only
    # (inactive branches are filtered out by load_collapsed)
    active_branch_names = df["branch_name"].to_list()
    mapped_on_active = mapped.filter(
        pl.col("branch_name").is_in(active_branch_names)
    )
    total_counted = int(df["count_cids"].sum())
    assert total_counted == len(mapped_on_active), (
        f"count_cids sum ({total_counted}) != mapped cids on active branches "
        f"({len(mapped_on_active)}). Possible cid duplication before Level 2."
    )
