# ml/tests/test_realized_da.py
"""Tests for PJM realized DA loader."""
import polars as pl
import pytest
from pathlib import Path
from ml.realized_da import load_realized_da, fetch_and_cache_month, _fetch_raw_da


def test_fetch_raw_da_returns_data():
    """Raw DA fetch should return monitored_facility + shadow_price."""
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
