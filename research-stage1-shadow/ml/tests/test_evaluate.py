import numpy as np
import pytest

from ml.evaluate import evaluate_classifier


def test_evaluate_returns_all_gate_metrics(synthetic_features, synthetic_labels):
    """All 10 gate metrics must be present in the output."""
    from ml.config import FeatureConfig, HyperparamConfig
    from ml.threshold import apply_threshold, find_optimal_threshold
    from ml.train import predict_proba, train_classifier

    model = train_classifier(
        synthetic_features, synthetic_labels, HyperparamConfig(), FeatureConfig()
    )
    proba = predict_proba(model, synthetic_features)
    threshold, _ = find_optimal_threshold(synthetic_labels, proba)
    y_pred = apply_threshold(proba, threshold)

    rng = np.random.RandomState(42)
    fake_sp = np.where(
        synthetic_labels == 1,
        rng.lognormal(mean=3, sigma=1.5, size=synthetic_labels.shape),
        0.0,
    )

    metrics = evaluate_classifier(synthetic_labels, proba, y_pred, fake_sp, threshold)

    required_keys = [
        "S1-AUC",
        "S1-AP",
        "S1-VCAP@100",
        "S1-VCAP@500",
        "S1-VCAP@1000",
        "S1-NDCG",
        "S1-BRIER",
        "S1-REC",
        "S1-CAP@100",
        "S1-CAP@500",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing gate metric: {key}"


def test_evaluate_auc_range(synthetic_features, synthetic_labels):
    """AUC should be between 0 and 1."""
    from ml.config import FeatureConfig, HyperparamConfig
    from ml.train import predict_proba, train_classifier

    model = train_classifier(
        synthetic_features, synthetic_labels, HyperparamConfig(), FeatureConfig()
    )
    proba = predict_proba(model, synthetic_features)
    y_pred = (proba > 0.5).astype(int)

    rng = np.random.RandomState(42)
    fake_sp = np.where(
        synthetic_labels == 1,
        rng.lognormal(mean=3, sigma=1.5, size=synthetic_labels.shape),
        0.0,
    )

    metrics = evaluate_classifier(synthetic_labels, proba, y_pred, fake_sp, 0.5)
    auc = metrics["S1-AUC"]
    if not np.isnan(auc):
        assert 0 <= auc <= 1


def test_evaluate_brier_range(synthetic_features, synthetic_labels):
    """Brier score should be between 0 and 1."""
    from ml.config import FeatureConfig, HyperparamConfig
    from ml.train import predict_proba, train_classifier

    model = train_classifier(
        synthetic_features, synthetic_labels, HyperparamConfig(), FeatureConfig()
    )
    proba = predict_proba(model, synthetic_features)
    y_pred = (proba > 0.5).astype(int)

    rng = np.random.RandomState(42)
    fake_sp = np.where(
        synthetic_labels == 1,
        rng.lognormal(mean=3, sigma=1.5, size=synthetic_labels.shape),
        0.0,
    )

    metrics = evaluate_classifier(synthetic_labels, proba, y_pred, fake_sp, 0.5)
    brier = metrics["S1-BRIER"]
    if not np.isnan(brier):
        assert 0 <= brier <= 1


def test_evaluate_vcap_range():
    """Value capture should be between 0 and 1."""
    rng = np.random.RandomState(42)
    n = 200
    y_true = (rng.random(n) < 0.1).astype(int)
    y_proba = rng.random(n)
    y_pred = (y_proba > 0.5).astype(int)
    shadow_prices = np.where(y_true == 1, rng.lognormal(3, 1.5, size=n), 0.0)

    metrics = evaluate_classifier(y_true, y_proba, y_pred, shadow_prices, 0.5)
    assert 0 <= metrics["S1-VCAP@100"] <= 1


def test_evaluate_perfect_classifier():
    """Perfect classifier should have AUC=1 and BRIER=0."""
    n = 200
    y_true = np.array([0] * 180 + [1] * 20)
    y_proba = np.where(y_true == 1, 0.99, 0.01)
    y_pred = y_true.copy()
    shadow_prices = np.where(y_true == 1, 50.0, 0.0)

    metrics = evaluate_classifier(y_true, y_proba, y_pred, shadow_prices, 0.5)
    assert metrics["S1-AUC"] > 0.99
    assert metrics["S1-BRIER"] < 0.01
    assert metrics["S1-REC"] == 1.0


def test_evaluate_capture_at_k():
    """CAP@K should work correctly for known inputs."""
    rng = np.random.RandomState(42)
    n = 200
    y_true = (rng.random(n) < 0.1).astype(int)
    y_proba = rng.random(n)
    # Predict all as binding -- should capture everything
    y_pred = np.ones(n, dtype=int)
    shadow_prices = np.where(y_true == 1, rng.lognormal(3, 1.5, size=n), 0.0)

    metrics = evaluate_classifier(y_true, y_proba, y_pred, shadow_prices, 0.0)
    # All predicted as binding => CAP@K should be 1.0
    assert metrics["S1-CAP@100"] == 1.0
    assert metrics["S1-CAP@500"] == 1.0


def test_evaluate_monitoring_metrics():
    """Monitoring metrics (precision, f1, etc.) should be present."""
    rng = np.random.RandomState(42)
    n = 200
    y_true = (rng.random(n) < 0.1).astype(int)
    y_proba = rng.random(n)
    y_pred = (y_proba > 0.5).astype(int)
    shadow_prices = np.where(y_true == 1, rng.lognormal(3, 1.5, size=n), 0.0)

    metrics = evaluate_classifier(y_true, y_proba, y_pred, shadow_prices, 0.5)
    assert "precision" in metrics
    assert "f1" in metrics
    assert "threshold" in metrics
    assert "n_samples" in metrics
    assert "binding_rate" in metrics


def test_aggregate_months():
    """aggregate_months computes mean, std, min, max, bottom_2_mean."""
    from ml.evaluate import aggregate_months
    per_month = {
        "2020-09": {"S1-AUC": 0.72, "S1-BRIER": 0.09},
        "2020-11": {"S1-AUC": 0.65, "S1-BRIER": 0.12},
        "2021-01": {"S1-AUC": 0.71, "S1-BRIER": 0.10},
    }
    agg = aggregate_months(per_month)
    assert abs(agg["mean"]["S1-AUC"] - 0.6933) < 0.001
    assert agg["min"]["S1-AUC"] == 0.65
    assert agg["max"]["S1-AUC"] == 0.72
    # bottom_2_mean for AUC (higher=better): worst 2 = (0.65, 0.71) => 0.68
    assert abs(agg["bottom_2_mean"]["S1-AUC"] - 0.68) < 0.001
    # For BRIER (lower is better), bottom_2 should be worst 2 (highest values)
    # bottom_2_mean = (0.12 + 0.10) / 2 = 0.11
    assert abs(agg["bottom_2_mean"]["S1-BRIER"] - 0.11) < 0.001


def test_aggregate_months_empty():
    """aggregate_months handles empty input."""
    from ml.evaluate import aggregate_months
    agg = aggregate_months({})
    assert agg["mean"] == {}


def test_aggregate_months_single():
    """aggregate_months with single month uses that month's values."""
    from ml.evaluate import aggregate_months
    agg = aggregate_months({"2021-01": {"S1-AUC": 0.75}})
    assert agg["mean"]["S1-AUC"] == 0.75
    assert agg["bottom_2_mean"]["S1-AUC"] == 0.75
    assert agg["std"]["S1-AUC"] == 0.0
