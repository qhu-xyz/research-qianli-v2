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


def test_load_month_invalid_peak_type_raises():
    """Invalid peak_type must raise, not silently fall back to onpeak."""
    from ml.realized_da import load_month
    with pytest.raises(AssertionError, match="Invalid peak_type"):
        load_month("2024-07", peak_type="badpeak")


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
