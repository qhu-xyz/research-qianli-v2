"""Tests for ml.data_loader — smoke test data generation."""
from __future__ import annotations

import os

import polars as pl
import pytest

from ml.config import PipelineConfig
from ml.data_loader import load_data, mem_mb


@pytest.fixture()
def cfg() -> PipelineConfig:
    return PipelineConfig()


@pytest.fixture()
def smoke_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set SMOKE_TEST=true so load_data dispatches to _load_smoke."""
    monkeypatch.setenv("SMOKE_TEST", "true")


# ---------------------------------------------------------------------------
# Shape & column tests
# ---------------------------------------------------------------------------

class TestLoadSmokeData:
    def test_shapes(self, cfg: PipelineConfig, smoke_env: None) -> None:
        """Smoke mode returns 80-row train and 20-row val DataFrames."""
        train_df, val_df = load_data(cfg, auction_month="2025-06", class_type="peak", period_type="monthly")
        assert isinstance(train_df, pl.DataFrame)
        assert isinstance(val_df, pl.DataFrame)
        assert train_df.shape[0] == 80
        assert val_df.shape[0] == 20

    def test_feature_columns_present(self, cfg: PipelineConfig, smoke_env: None) -> None:
        """All 34 tier features must be present in both DataFrames."""
        train_df, val_df = load_data(cfg, auction_month="2025-06", class_type="peak", period_type="monthly")
        expected_features = cfg.tier.features
        assert len(expected_features) == 34

        for col in expected_features:
            assert col in train_df.columns, f"Missing column {col} in train_df"
            assert col in val_df.columns, f"Missing column {col} in val_df"

    def test_target_column_present(self, cfg: PipelineConfig, smoke_env: None) -> None:
        """Target column actual_shadow_price must exist in both DataFrames."""
        train_df, val_df = load_data(cfg, auction_month="2025-06", class_type="peak", period_type="monthly")
        assert "actual_shadow_price" in train_df.columns
        assert "actual_shadow_price" in val_df.columns

    def test_feature_dtypes_numeric(self, cfg: PipelineConfig, smoke_env: None) -> None:
        """All feature columns must be numeric (Float64)."""
        train_df, _ = load_data(cfg, auction_month="2025-06", class_type="peak", period_type="monthly")
        for col in cfg.tier.features:
            assert train_df[col].dtype == pl.Float64, f"Column {col} has dtype {train_df[col].dtype}, expected Float64"


# ---------------------------------------------------------------------------
# Tier distribution
# ---------------------------------------------------------------------------

class TestSmokeDataHasTiers:
    def test_some_positive_shadow_price(self, cfg: PipelineConfig, smoke_env: None) -> None:
        """At least some rows should have actual_shadow_price > 0 (binding constraints)."""
        train_df, val_df = load_data(cfg, auction_month="2025-06", class_type="peak", period_type="monthly")
        all_prices = pl.concat([train_df, val_df])["actual_shadow_price"]
        n_positive = (all_prices > 0).sum()
        assert n_positive >= 3, f"Expected at least 3 positive prices, got {n_positive}"

    def test_some_negative_shadow_price(self, cfg: PipelineConfig, smoke_env: None) -> None:
        """At least some rows should have actual_shadow_price < 0 (tier 4)."""
        train_df, val_df = load_data(cfg, auction_month="2025-06", class_type="peak", period_type="monthly")
        all_prices = pl.concat([train_df, val_df])["actual_shadow_price"]
        n_negative = (all_prices < 0).sum()
        assert n_negative >= 3, f"Expected at least 3 negative prices, got {n_negative}"

    def test_has_high_value_constraints(self, cfg: PipelineConfig, smoke_env: None) -> None:
        """At least some rows should have high shadow prices (tier 0/1)."""
        train_df, val_df = load_data(cfg, auction_month="2025-06", class_type="peak", period_type="monthly")
        all_prices = pl.concat([train_df, val_df])["actual_shadow_price"]
        n_high = (all_prices >= 1000).sum()
        assert n_high >= 3, f"Expected at least 3 high-value prices, got {n_high}"


# ---------------------------------------------------------------------------
# Metadata columns
# ---------------------------------------------------------------------------

class TestSmokeDataHasMetadata:
    def test_constraint_id_present(self, cfg: PipelineConfig, smoke_env: None) -> None:
        """constraint_id column must exist in both DataFrames."""
        train_df, val_df = load_data(cfg, auction_month="2025-06", class_type="peak", period_type="monthly")
        assert "constraint_id" in train_df.columns
        assert "constraint_id" in val_df.columns

    def test_branch_name_present(self, cfg: PipelineConfig, smoke_env: None) -> None:
        """branch_name column must exist in both DataFrames."""
        train_df, val_df = load_data(cfg, auction_month="2025-06", class_type="peak", period_type="monthly")
        assert "branch_name" in train_df.columns
        assert "branch_name" in val_df.columns

    def test_metadata_are_strings(self, cfg: PipelineConfig, smoke_env: None) -> None:
        """constraint_id and branch_name should be string columns."""
        train_df, _ = load_data(cfg, auction_month="2025-06", class_type="peak", period_type="monthly")
        assert train_df["constraint_id"].dtype == pl.Utf8
        assert train_df["branch_name"].dtype == pl.Utf8


# ---------------------------------------------------------------------------
# Real data requires Ray / pbase
# ---------------------------------------------------------------------------

class TestLoadRealRequiresInfra:
    def test_real_mode_dispatches_to_real_loader(self, cfg: PipelineConfig) -> None:
        """Without SMOKE_TEST, load_data dispatches to _load_real which needs pbase."""
        os.environ.pop("SMOKE_TEST", None)
        with pytest.raises(Exception):
            load_data(cfg, auction_month="2025-06", class_type="peak", period_type="monthly")


# ---------------------------------------------------------------------------
# mem_mb helper
# ---------------------------------------------------------------------------

class TestMemMb:
    def test_mem_mb_returns_positive_float(self) -> None:
        """mem_mb() should return a positive number (current RSS in MB)."""
        m = mem_mb()
        assert isinstance(m, float)
        assert m > 0
