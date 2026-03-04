"""Tests for ml.features — feature preparation for tier classifier."""
from __future__ import annotations

import numpy as np
import polars as pl

from ml.config import TierConfig
from ml.features import (
    compute_sample_weights,
    compute_tier_labels,
    prepare_features,
)


def _make_sample_df(n: int = 5) -> pl.DataFrame:
    """Build a minimal DataFrame with all tier feature columns + actual_shadow_price."""
    cfg = TierConfig()
    data: dict[str, list[float]] = {}
    for i, feat in enumerate(cfg.features):
        data[feat] = [float(i + row) for row in range(n)]
    # actual_shadow_price spanning multiple tiers
    data["actual_shadow_price"] = [5000.0, 1500.0, 200.0, 10.0, -5.0][:n]
    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# prepare_features
# ---------------------------------------------------------------------------

class TestPrepareFeatures:
    def test_shape_and_monotone_length(self):
        """Output X has shape (n, 34) and monotone list has length 34."""
        df = _make_sample_df(n=5)
        cfg = TierConfig()
        X, monotone = prepare_features(df, cfg)

        assert isinstance(X, np.ndarray)
        assert X.shape == (5, len(cfg.features))
        assert isinstance(monotone, list)
        assert len(monotone) == len(cfg.features)

    def test_monotone_values_match_config(self):
        """Returned monotone constraints must match the config exactly."""
        df = _make_sample_df(n=5)
        cfg = TierConfig()
        _, monotone = prepare_features(df, cfg)
        assert monotone == list(cfg.monotone_constraints)

    def test_values_extracted_correctly(self):
        """First column of X should match the first feature column in df."""
        df = _make_sample_df(n=5)
        cfg = TierConfig()
        X, _ = prepare_features(df, cfg)
        first_feat = cfg.features[0]
        expected = df[first_feat].to_numpy().astype(np.float64)
        np.testing.assert_array_equal(X[:, 0], expected)

    def test_null_filled_with_zero(self):
        """Null values in features should be filled with 0.0."""
        df = _make_sample_df(n=5)
        cfg = TierConfig()
        first_feat = cfg.features[0]
        df = df.with_columns(
            pl.when(pl.col(first_feat) == df[first_feat][0])
            .then(None)
            .otherwise(pl.col(first_feat))
            .alias(first_feat)
        )
        X, _ = prepare_features(df, cfg)
        assert X[0, 0] == 0.0


# ---------------------------------------------------------------------------
# compute_tier_labels
# ---------------------------------------------------------------------------

class TestComputeTierLabels:
    def test_tier_assignment(self):
        """actual_shadow_price maps to correct tiers."""
        cfg = TierConfig()
        # [5000, 1500, 200, 10, -5]
        # Tier 0: [3000, inf) => 5000
        # Tier 1: [1000, 3000) => 1500
        # Tier 2: [100, 1000) => 200
        # Tier 3: [0, 100) => 10
        # Tier 4: (-inf, 0) => -5
        actual = np.array([5000.0, 1500.0, 200.0, 10.0, -5.0])
        labels = compute_tier_labels(actual, cfg)
        expected = np.array([0, 1, 2, 3, 4])
        np.testing.assert_array_equal(labels, expected)

    def test_boundary_values(self):
        """Values at exact boundaries."""
        cfg = TierConfig()
        # 0 -> tier 3 (in [0, 100))
        # 100 -> tier 2 (in [100, 1000))
        # 1000 -> tier 1 (in [1000, 3000))
        # 3000 -> tier 0 (in [3000, inf))
        actual = np.array([0.0, 100.0, 1000.0, 3000.0])
        labels = compute_tier_labels(actual, cfg)
        expected = np.array([3, 2, 1, 0])
        np.testing.assert_array_equal(labels, expected)

    def test_output_dtype(self):
        """Labels should be integer type."""
        cfg = TierConfig()
        actual = np.array([5000.0, 10.0, -5.0])
        labels = compute_tier_labels(actual, cfg)
        assert labels.dtype in (np.int32, np.int64)


# ---------------------------------------------------------------------------
# compute_sample_weights
# ---------------------------------------------------------------------------

class TestComputeSampleWeights:
    def test_weights_match_class_weights(self):
        """Each sample gets weight from its tier's class weight."""
        cfg = TierConfig()
        labels = np.array([0, 1, 2, 3, 4])
        weights = compute_sample_weights(labels, cfg)
        expected = np.array([10.0, 5.0, 2.0, 1.0, 0.5])
        np.testing.assert_array_equal(weights, expected)

    def test_weights_shape(self):
        """Output has same length as input."""
        cfg = TierConfig()
        labels = np.array([0, 0, 1, 2, 3, 4, 4, 4])
        weights = compute_sample_weights(labels, cfg)
        assert weights.shape == labels.shape

    def test_custom_class_weights(self):
        """Custom class weights should be used."""
        cfg = TierConfig()
        cfg.class_weights = {0: 20, 1: 10, 2: 5, 3: 2, 4: 1}
        labels = np.array([0, 4])
        weights = compute_sample_weights(labels, cfg)
        np.testing.assert_array_equal(weights, np.array([20.0, 1.0]))
