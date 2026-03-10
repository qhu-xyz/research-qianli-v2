"""Tests for realized DA shadow price loader."""
import polars as pl
import pytest
from pathlib import Path

from ml.realized_da import load_realized_da


def test_load_realized_da_from_cache(tmp_path: Path):
    """load_realized_da reads a cached parquet with correct schema."""
    df = pl.DataFrame({
        "constraint_id": ["72691", "1026FG", "99999"],
        "realized_sp": [150.0, 0.0, 42.5],
    })
    month = "2022-06"
    df.write_parquet(str(tmp_path / f"{month}.parquet"))

    result = load_realized_da(month, cache_dir=str(tmp_path))
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["constraint_id", "realized_sp"]
    assert result.schema["constraint_id"] == pl.String
    assert result.schema["realized_sp"] == pl.Float64
    assert len(result) == 3


def test_load_realized_da_missing_month(tmp_path: Path):
    """load_realized_da raises FileNotFoundError for uncached month."""
    with pytest.raises(FileNotFoundError):
        load_realized_da("2099-01", cache_dir=str(tmp_path))
