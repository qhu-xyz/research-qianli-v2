"""Tests for LTR training."""
import numpy as np
import pytest
from ml.config import LTRConfig
from ml.train import train_ltr_model, predict_scores


def test_train_lightgbm():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 5)
    y = rng.rand(200) * 1000
    groups = np.array([100, 100])
    cfg = LTRConfig(
        features=[f"f{i}" for i in range(5)],
        monotone_constraints=[0] * 5,
        backend="lightgbm",
        n_estimators=10,
    )
    model = train_ltr_model(X, y, groups, cfg)
    assert model is not None


def test_predict_scores_shape():
    rng = np.random.RandomState(42)
    X_train = rng.randn(200, 5)
    y_train = rng.rand(200) * 1000
    groups = np.array([100, 100])
    cfg = LTRConfig(
        features=[f"f{i}" for i in range(5)],
        monotone_constraints=[0] * 5,
        backend="lightgbm",
        n_estimators=10,
    )
    model = train_ltr_model(X_train, y_train, groups, cfg)
    X_test = rng.randn(50, 5)
    scores = predict_scores(model, X_test)
    assert scores.shape == (50,)


def test_train_lightgbm_with_early_stopping():
    rng = np.random.RandomState(42)
    X_train = rng.randn(200, 5)
    y_train = rng.rand(200) * 1000
    groups_train = np.array([100, 100])
    X_val = rng.randn(100, 5)
    y_val = rng.rand(100) * 1000
    groups_val = np.array([100])
    cfg = LTRConfig(
        features=[f"f{i}" for i in range(5)],
        monotone_constraints=[0] * 5,
        backend="lightgbm",
        n_estimators=50,
    )
    model = train_ltr_model(
        X_train, y_train, groups_train, cfg,
        X_val=X_val, y_val=y_val, groups_val=groups_val,
    )
    assert model is not None
