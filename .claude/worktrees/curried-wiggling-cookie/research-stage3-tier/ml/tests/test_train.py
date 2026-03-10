"""Tests for ml.train — tier classifier training functions."""
from __future__ import annotations

import numpy as np
import pytest

from ml.config import TierConfig
from ml.train import (
    compute_tier_ev_score,
    predict_tier,
    predict_tier_probabilities,
    train_tier_classifier,
)


def _make_tier_data(n: int = 500, seed: int = 42):
    """Build random tier training data with shape matching TierConfig."""
    rng = np.random.RandomState(seed)
    cfg = TierConfig()
    n_features = len(cfg.features)
    X = rng.randn(n, n_features).astype(np.float64)
    # 5-class labels (0-4)
    y = rng.randint(0, 5, size=n).astype(int)
    return X, y, cfg


# ---------------------------------------------------------------------------
# train_tier_classifier
# ---------------------------------------------------------------------------

class TestTrainTierClassifier:
    def test_returns_model(self):
        """train_tier_classifier returns an XGBClassifier."""
        X, y, cfg = _make_tier_data()
        model = train_tier_classifier(X, y, cfg)

        from xgboost import XGBClassifier
        assert isinstance(model, XGBClassifier)

    def test_model_has_correct_params(self):
        """Model hyperparams should match config values."""
        X, y, cfg = _make_tier_data()
        model = train_tier_classifier(X, y, cfg)

        assert model.n_estimators == cfg.n_estimators
        assert model.max_depth == cfg.max_depth
        assert model.learning_rate == cfg.learning_rate

    def test_with_sample_weight(self):
        """train_tier_classifier should accept sample_weight without error."""
        X, y, cfg = _make_tier_data(n=200)
        weights = np.ones(200, dtype=np.float64)
        model = train_tier_classifier(X, y, cfg, sample_weight=weights)

        from xgboost import XGBClassifier
        assert isinstance(model, XGBClassifier)

    def test_early_stopping(self):
        """train_tier_classifier with val set uses early stopping."""
        X, y, cfg = _make_tier_data(n=400)
        cfg.early_stopping_rounds = 10
        cfg.n_estimators = 500
        X_train, X_val = X[:300], X[300:]
        y_train, y_val = y[:300], y[300:]
        model = train_tier_classifier(
            X_train, y_train, cfg,
            X_val=X_val, y_val=y_val,
        )
        # Should have stopped early (best_iteration < n_estimators)
        assert hasattr(model, "best_iteration")
        assert model.best_iteration < cfg.n_estimators


# ---------------------------------------------------------------------------
# predict_tier_probabilities
# ---------------------------------------------------------------------------

class TestPredictTierProbabilities:
    def test_shape(self):
        """predict_tier_probabilities returns (n_samples, num_class) array."""
        X, y, cfg = _make_tier_data(n=100)
        model = train_tier_classifier(X, y, cfg)
        proba = predict_tier_probabilities(model, X)

        assert isinstance(proba, np.ndarray)
        assert proba.shape == (100, cfg.num_class)

    def test_rows_sum_to_one(self):
        """Each row of probabilities should sum to ~1.0."""
        X, y, cfg = _make_tier_data(n=100)
        model = train_tier_classifier(X, y, cfg)
        proba = predict_tier_probabilities(model, X)

        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_non_negative(self):
        """All probabilities should be >= 0."""
        X, y, cfg = _make_tier_data(n=100)
        model = train_tier_classifier(X, y, cfg)
        proba = predict_tier_probabilities(model, X)

        assert np.all(proba >= 0.0)


# ---------------------------------------------------------------------------
# predict_tier
# ---------------------------------------------------------------------------

class TestPredictTier:
    def test_shape(self):
        """predict_tier returns array with same length as input."""
        X, y, cfg = _make_tier_data(n=50)
        model = train_tier_classifier(X, y, cfg)
        tiers = predict_tier(model, X)

        assert isinstance(tiers, np.ndarray)
        assert tiers.shape == (50,)

    def test_valid_tier_range(self):
        """All predicted tiers should be in [0, 4]."""
        X, y, cfg = _make_tier_data(n=100)
        model = train_tier_classifier(X, y, cfg)
        tiers = predict_tier(model, X)

        assert np.all(tiers >= 0)
        assert np.all(tiers <= 4)


# ---------------------------------------------------------------------------
# compute_tier_ev_score
# ---------------------------------------------------------------------------

class TestComputeTierEvScore:
    def test_shape(self):
        """compute_tier_ev_score returns array with same length as rows in proba."""
        proba = np.array([
            [0.1, 0.2, 0.3, 0.3, 0.1],
            [0.5, 0.3, 0.1, 0.05, 0.05],
        ])
        midpoints = [4000, 2000, 550, 50, 0]
        ev = compute_tier_ev_score(proba, midpoints)

        assert isinstance(ev, np.ndarray)
        assert ev.shape == (2,)

    def test_values(self):
        """Verify EV score = proba @ midpoints."""
        proba = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],  # 100% tier 0 => EV = 4000
            [0.0, 0.0, 0.0, 0.0, 1.0],  # 100% tier 4 => EV = 0
        ])
        midpoints = [4000, 2000, 550, 50, 0]
        ev = compute_tier_ev_score(proba, midpoints)

        np.testing.assert_allclose(ev[0], 4000.0)
        np.testing.assert_allclose(ev[1], 0.0)

    def test_mixed_probabilities(self):
        """50/50 split between tier 0 and tier 4."""
        proba = np.array([[0.5, 0.0, 0.0, 0.0, 0.5]])
        midpoints = [4000, 2000, 550, 50, 0]
        ev = compute_tier_ev_score(proba, midpoints)

        np.testing.assert_allclose(ev[0], 2000.0)
