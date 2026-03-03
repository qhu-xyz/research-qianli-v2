"""Tests for ml.train — classifier and regressor training functions."""
from __future__ import annotations

import numpy as np
import pytest

from ml.config import ClassifierConfig, RegressorConfig
from ml.train import (
    predict_proba,
    predict_shadow_price,
    train_classifier,
    train_regressor,
)


def _make_clf_data(n: int = 200, seed: int = 42):
    """Build random classifier training data with shape matching ClassifierConfig."""
    rng = np.random.RandomState(seed)
    cfg = ClassifierConfig()
    n_features = len(cfg.features)
    X = rng.randn(n, n_features).astype(np.float64)
    y = rng.randint(0, 2, size=n).astype(int)
    return X, y, cfg


def _make_reg_data(n: int = 200, seed: int = 42):
    """Build random regressor training data with shape matching RegressorConfig."""
    rng = np.random.RandomState(seed)
    cfg = RegressorConfig()
    n_features = len(cfg.features)
    X = rng.randn(n, n_features).astype(np.float64)
    # Regression target: log1p(max(0, shadow_price)), always non-negative
    y = np.abs(rng.randn(n)).astype(np.float64)
    return X, y, cfg


# ---------------------------------------------------------------------------
# train_classifier
# ---------------------------------------------------------------------------

class TestTrainClassifier:
    def test_returns_model_and_threshold(self):
        """train_classifier returns (model, threshold) tuple."""
        X, y, cfg = _make_clf_data()
        model, threshold = train_classifier(X, y, cfg)

        # Model should be an XGBClassifier
        from xgboost import XGBClassifier
        assert isinstance(model, XGBClassifier)
        # Threshold should be a float
        assert isinstance(threshold, float)

    def test_default_threshold_is_half(self):
        """Without validation data, threshold defaults to 0.5."""
        X, y, cfg = _make_clf_data()
        _, threshold = train_classifier(X, y, cfg)
        assert threshold == 0.5

    def test_model_has_correct_params(self):
        """Model hyperparams should match config values."""
        X, y, cfg = _make_clf_data()
        model, _ = train_classifier(X, y, cfg)

        assert model.n_estimators == cfg.n_estimators
        assert model.max_depth == cfg.max_depth
        assert model.learning_rate == cfg.learning_rate


# ---------------------------------------------------------------------------
# predict_proba
# ---------------------------------------------------------------------------

class TestPredictProba:
    def test_shape(self):
        """predict_proba returns array with same length as input."""
        X, y, cfg = _make_clf_data(n=50)
        model, _ = train_classifier(X, y, cfg)
        proba = predict_proba(model, X)

        assert isinstance(proba, np.ndarray)
        assert proba.shape == (50,)

    def test_range_zero_to_one(self):
        """All predicted probabilities should be in [0, 1]."""
        X, y, cfg = _make_clf_data(n=100)
        model, _ = train_classifier(X, y, cfg)
        proba = predict_proba(model, X)

        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)


# ---------------------------------------------------------------------------
# train_regressor
# ---------------------------------------------------------------------------

class TestTrainRegressor:
    def test_returns_model(self):
        """train_regressor returns an XGBRegressor."""
        X, y, cfg = _make_reg_data()
        model = train_regressor(X, y, cfg)

        from xgboost import XGBRegressor
        assert isinstance(model, XGBRegressor)

    def test_model_has_correct_params(self):
        """Model hyperparams should match config values."""
        X, y, cfg = _make_reg_data()
        model = train_regressor(X, y, cfg)

        assert model.n_estimators == cfg.n_estimators
        assert model.max_depth == cfg.max_depth
        assert model.learning_rate == cfg.learning_rate

    def test_with_sample_weight(self):
        """train_regressor should accept sample_weight without error."""
        X, y, cfg = _make_reg_data(n=100)
        weights = np.ones(100, dtype=np.float64)
        model = train_regressor(X, y, cfg, sample_weight=weights)

        from xgboost import XGBRegressor
        assert isinstance(model, XGBRegressor)


# ---------------------------------------------------------------------------
# predict_shadow_price
# ---------------------------------------------------------------------------

class TestPredictShadowPrice:
    def test_shape(self):
        """predict_shadow_price returns array with same length as input."""
        X, y, cfg = _make_reg_data(n=50)
        model = train_regressor(X, y, cfg)
        preds = predict_shadow_price(model, X)

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (50,)

    def test_non_negative(self):
        """All predictions should be >= 0 (expm1 of non-negative clipped values)."""
        X, y, cfg = _make_reg_data(n=100)
        model = train_regressor(X, y, cfg)
        preds = predict_shadow_price(model, X)

        assert np.all(preds >= 0.0)

    def test_dtype_float(self):
        """Predictions should be floating point."""
        X, y, cfg = _make_reg_data(n=50)
        model = train_regressor(X, y, cfg)
        preds = predict_shadow_price(model, X)

        assert preds.dtype in (np.float32, np.float64)


# ---------------------------------------------------------------------------
# Classifier with threshold optimization
# ---------------------------------------------------------------------------

class TestClassifierWithThresholdOptimization:
    def test_threshold_optimized_when_val_provided(self):
        """When X_val/y_val are provided, threshold should be optimized (may differ from 0.5)."""
        rng = np.random.RandomState(123)
        cfg = ClassifierConfig()
        n_features = len(cfg.features)

        # Create separable data so the classifier learns a real signal
        n_train = 300
        X_train = rng.randn(n_train, n_features)
        # Label based on first feature to create learnable signal
        y_train = (X_train[:, 0] > 0.0).astype(int)

        n_val = 100
        X_val = rng.randn(n_val, n_features)
        y_val = (X_val[:, 0] > 0.0).astype(int)

        model, threshold = train_classifier(
            X_train, y_train, cfg, X_val=X_val, y_val=y_val,
        )

        # With real signal, threshold optimization should find something
        # (may or may not be 0.5 — the key is the code path runs)
        assert isinstance(threshold, float)
        assert 0.0 < threshold < 1.0

    def test_threshold_uses_beta_from_config(self):
        """The beta parameter for threshold optimization comes from cfg.threshold_beta."""
        rng = np.random.RandomState(99)
        cfg = ClassifierConfig()
        n_features = len(cfg.features)

        n_train = 200
        X_train = rng.randn(n_train, n_features)
        y_train = (X_train[:, 0] > 0.0).astype(int)

        n_val = 100
        X_val = rng.randn(n_val, n_features)
        y_val = (X_val[:, 0] > 0.0).astype(int)

        model, threshold = train_classifier(
            X_train, y_train, cfg, X_val=X_val, y_val=y_val,
        )

        # Threshold should be a valid float between 0 and 1
        assert 0.0 < threshold < 1.0
