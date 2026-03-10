import os

import numpy as np
import pytest

from ml.config import FeatureConfig, PipelineConfig


@pytest.fixture(autouse=True)
def smoke_mode(monkeypatch):
    monkeypatch.setenv("SMOKE_TEST", "true")


def test_smoke_returns_correct_shapes():
    from ml.data_loader import load_data

    config = PipelineConfig()
    train_df, val_df = load_data(config)
    # 100 total: 80 train, 20 val
    assert train_df.shape[0] == 80
    assert val_df.shape[0] == 20


def test_smoke_has_all_feature_columns():
    from ml.data_loader import load_data

    config = PipelineConfig()
    fc = FeatureConfig()
    train_df, val_df = load_data(config)
    for feat in fc.features:
        assert feat in train_df.columns, f"Missing feature: {feat}"
        assert feat in val_df.columns, f"Missing feature in val: {feat}"


def test_smoke_has_required_columns():
    from ml.data_loader import load_data

    config = PipelineConfig()
    train_df, _ = load_data(config)
    assert "actual_shadow_price" in train_df.columns
    assert "constraint_id" in train_df.columns
    assert "auction_month" in train_df.columns


def test_smoke_no_nans_in_features():
    from ml.data_loader import load_data

    config = PipelineConfig()
    fc = FeatureConfig()
    train_df, val_df = load_data(config)
    for feat in fc.features:
        assert train_df[feat].null_count() == 0
        assert val_df[feat].null_count() == 0


def test_smoke_shadow_prices_non_negative():
    from ml.data_loader import load_data

    config = PipelineConfig()
    train_df, val_df = load_data(config)
    assert (train_df["actual_shadow_price"].to_numpy() >= 0).all()
    assert (val_df["actual_shadow_price"].to_numpy() >= 0).all()


def test_smoke_deterministic():
    from ml.data_loader import load_data

    config = PipelineConfig()
    t1, v1 = load_data(config)
    t2, v2 = load_data(config)
    np.testing.assert_array_equal(
        t1["actual_shadow_price"].to_numpy(),
        t2["actual_shadow_price"].to_numpy(),
    )
