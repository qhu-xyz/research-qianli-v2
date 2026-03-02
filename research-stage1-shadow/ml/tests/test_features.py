import numpy as np
import polars as pl
import pytest

from ml.config import FeatureConfig
from ml.features import compute_binary_labels, compute_scale_pos_weight, prepare_features


def test_prepare_features_shape():
    """prepare_features returns (n_samples, 17) array with interaction features."""
    rng = np.random.RandomState(42)
    fc = FeatureConfig()
    # Only need the 14 base features — interactions are computed in prepare_features
    base_features = [f[0] for f in fc.step1_features[:14]]
    data = {feat: rng.randn(50).tolist() for feat in base_features}
    df = pl.DataFrame(data)

    X, cols = prepare_features(df, fc)
    assert X.shape == (50, 17)
    assert len(cols) == 17


def test_prepare_features_fills_nulls():
    """Null values are replaced with 0."""
    fc = FeatureConfig()
    # Only need the 14 base features — interactions are computed in prepare_features
    base_features = [f[0] for f in fc.step1_features[:14]]
    data = {feat: [None, 1.0, 2.0] for feat in base_features}
    df = pl.DataFrame(data)

    X, _ = prepare_features(df, fc)
    assert not np.any(np.isnan(X))
    assert X[0, 0] == 0.0


def test_compute_binary_labels_range():
    """Labels are 0 or 1."""
    df = pl.DataFrame({"actual_shadow_price": [0.0, 0.0, 5.0, 10.0, 0.0]})
    labels = compute_binary_labels(df)
    assert set(np.unique(labels)).issubset({0, 1})
    np.testing.assert_array_equal(labels, [0, 0, 1, 1, 0])


def test_compute_binary_labels_custom_threshold():
    """Custom threshold works correctly."""
    df = pl.DataFrame({"actual_shadow_price": [0.0, 3.0, 5.0, 10.0]})
    labels = compute_binary_labels(df, threshold=4.0)
    np.testing.assert_array_equal(labels, [0, 0, 1, 1])


def test_compute_scale_pos_weight_known():
    """90 negatives, 10 positives -> weight = 9.0."""
    labels = np.array([0] * 90 + [1] * 10)
    spw = compute_scale_pos_weight(labels)
    assert spw == 9.0


def test_compute_scale_pos_weight_no_positives():
    """All negatives returns 1.0."""
    labels = np.array([0] * 100)
    spw = compute_scale_pos_weight(labels)
    assert spw == 1.0


def test_compute_scale_pos_weight_balanced():
    """50/50 split returns 1.0."""
    labels = np.array([0] * 50 + [1] * 50)
    spw = compute_scale_pos_weight(labels)
    assert spw == 1.0
