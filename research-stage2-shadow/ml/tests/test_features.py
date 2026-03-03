"""Tests for ml.features — feature preparation for classifier and regressor."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml.config import ClassifierConfig, RegressorConfig
from ml.features import (
    compute_binary_labels,
    compute_regression_target,
    compute_scale_pos_weight,
    prepare_clf_features,
    prepare_reg_features,
)


def _make_sample_df(n: int = 3) -> pl.DataFrame:
    """Build a minimal DataFrame with all 24 regressor columns + actual_shadow_price."""
    cfg = RegressorConfig()
    data: dict[str, list[float]] = {}
    for i, feat in enumerate(cfg.features):
        data[feat] = [float(i + row) for row in range(n)]
    # Add actual_shadow_price for label functions
    data["actual_shadow_price"] = [10.0, -5.0, 0.0]
    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# prepare_clf_features
# ---------------------------------------------------------------------------

class TestPrepareClfFeatures:
    def test_shape_and_monotone_length(self):
        """Output X has shape (3, 13) and monotone list has length 13."""
        df = _make_sample_df(n=3)
        cfg = ClassifierConfig()
        X, monotone = prepare_clf_features(df, cfg)

        assert isinstance(X, np.ndarray)
        assert X.shape == (3, 13)
        assert isinstance(monotone, list)
        assert len(monotone) == 13

    def test_monotone_values_match_config(self):
        """Returned monotone constraints must match the config exactly."""
        df = _make_sample_df(n=3)
        cfg = ClassifierConfig()
        _, monotone = prepare_clf_features(df, cfg)
        assert monotone == list(cfg.monotone_constraints)

    def test_values_extracted_correctly(self):
        """First column of X should match the first feature column in df."""
        df = _make_sample_df(n=3)
        cfg = ClassifierConfig()
        X, _ = prepare_clf_features(df, cfg)
        first_feat = cfg.features[0]
        expected = df[first_feat].to_numpy().astype(np.float64)
        np.testing.assert_array_equal(X[:, 0], expected)

    def test_null_filled_with_zero(self):
        """Null values in features should be filled with 0.0."""
        df = _make_sample_df(n=3)
        # Inject a null into the first feature column
        cfg = ClassifierConfig()
        first_feat = cfg.features[0]
        df = df.with_columns(
            pl.when(pl.col(first_feat) == df[first_feat][0])
            .then(None)
            .otherwise(pl.col(first_feat))
            .alias(first_feat)
        )
        X, _ = prepare_clf_features(df, cfg)
        assert X[0, 0] == 0.0


# ---------------------------------------------------------------------------
# prepare_reg_features
# ---------------------------------------------------------------------------

class TestPrepareRegFeatures:
    def test_shape_and_monotone_length(self):
        """Output X has shape (3, 24) and monotone list has length 24."""
        df = _make_sample_df(n=3)
        cfg = RegressorConfig()
        X, monotone = prepare_reg_features(df, cfg)

        assert isinstance(X, np.ndarray)
        assert X.shape == (3, 24)
        assert isinstance(monotone, list)
        assert len(monotone) == 24

    def test_monotone_values_match_config(self):
        """Returned monotone constraints must match the config exactly."""
        df = _make_sample_df(n=3)
        cfg = RegressorConfig()
        _, monotone = prepare_reg_features(df, cfg)
        assert monotone == list(cfg.monotone_constraints)

    def test_values_extracted_correctly(self):
        """Last column of X should match the last feature column in df."""
        df = _make_sample_df(n=3)
        cfg = RegressorConfig()
        X, _ = prepare_reg_features(df, cfg)
        last_feat = cfg.features[-1]
        expected = df[last_feat].to_numpy().astype(np.float64)
        np.testing.assert_array_equal(X[:, -1], expected)

    def test_null_filled_with_zero(self):
        """Null values in features should be filled with 0.0."""
        df = _make_sample_df(n=3)
        cfg = RegressorConfig()
        last_feat = cfg.features[-1]
        df = df.with_columns(
            pl.when(pl.col(last_feat) == df[last_feat][1])
            .then(None)
            .otherwise(pl.col(last_feat))
            .alias(last_feat)
        )
        X, _ = prepare_reg_features(df, cfg)
        assert X[1, -1] == 0.0


# ---------------------------------------------------------------------------
# compute_binary_labels
# ---------------------------------------------------------------------------

class TestComputeBinaryLabels:
    def test_default_threshold(self):
        """actual_shadow_price > 0.0 should be 1, else 0."""
        df = _make_sample_df(n=3)
        # actual_shadow_price = [10.0, -5.0, 0.0]
        labels = compute_binary_labels(df)
        expected = np.array([1, 0, 0], dtype=int)
        np.testing.assert_array_equal(labels, expected)

    def test_custom_threshold(self):
        """With threshold=5.0, only 10.0 > 5.0 => label 1."""
        df = _make_sample_df(n=3)
        labels = compute_binary_labels(df, threshold=5.0)
        expected = np.array([1, 0, 0], dtype=int)
        np.testing.assert_array_equal(labels, expected)

    def test_all_below_threshold(self):
        """When all values are below threshold, all labels are 0."""
        df = _make_sample_df(n=3)
        labels = compute_binary_labels(df, threshold=100.0)
        expected = np.array([0, 0, 0], dtype=int)
        np.testing.assert_array_equal(labels, expected)


# ---------------------------------------------------------------------------
# compute_regression_target
# ---------------------------------------------------------------------------

class TestComputeRegressionTarget:
    def test_log1p_transform(self):
        """Target = log1p(max(0, actual_shadow_price))."""
        df = _make_sample_df(n=3)
        # actual_shadow_price = [10.0, -5.0, 0.0]
        target = compute_regression_target(df)
        expected = np.array([
            np.log1p(10.0),   # max(0, 10.0)  = 10.0
            np.log1p(0.0),    # max(0, -5.0)  = 0.0
            np.log1p(0.0),    # max(0, 0.0)   = 0.0
        ])
        np.testing.assert_allclose(target, expected)

    def test_output_dtype(self):
        """Output should be a float64 numpy array."""
        df = _make_sample_df(n=3)
        target = compute_regression_target(df)
        assert target.dtype == np.float64


# ---------------------------------------------------------------------------
# compute_scale_pos_weight
# ---------------------------------------------------------------------------

class TestComputeScalePosWeight:
    def test_ratio_calculation(self):
        """scale_pos_weight = n_neg / n_pos."""
        labels = np.array([1, 0, 0, 0, 0])
        weight = compute_scale_pos_weight(labels)
        assert weight == pytest.approx(4.0)

    def test_no_positives_returns_one(self):
        """If n_positive == 0, return 1.0 to avoid division by zero."""
        labels = np.array([0, 0, 0])
        weight = compute_scale_pos_weight(labels)
        assert weight == 1.0

    def test_balanced_classes(self):
        """Equal positives and negatives => weight 1.0."""
        labels = np.array([1, 1, 0, 0])
        weight = compute_scale_pos_weight(labels)
        assert weight == pytest.approx(1.0)
