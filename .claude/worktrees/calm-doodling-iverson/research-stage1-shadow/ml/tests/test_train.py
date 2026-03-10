import numpy as np

from ml.config import FeatureConfig, HyperparamConfig
from ml.train import predict_proba, train_classifier
from ml.threshold import apply_threshold, find_optimal_threshold


def test_train_returns_xgb_classifier(synthetic_features, synthetic_labels):
    from xgboost import XGBClassifier

    model = train_classifier(
        synthetic_features, synthetic_labels, HyperparamConfig(), FeatureConfig()
    )
    assert isinstance(model, XGBClassifier)


def test_predict_proba_shape(synthetic_features, synthetic_labels):
    model = train_classifier(
        synthetic_features, synthetic_labels, HyperparamConfig(), FeatureConfig()
    )
    proba = predict_proba(model, synthetic_features)
    assert proba.shape == (synthetic_features.shape[0],)
    assert np.all((proba >= 0) & (proba <= 1))


def test_predict_proba_deterministic(synthetic_features, synthetic_labels):
    """Same seed produces same predictions."""
    model1 = train_classifier(
        synthetic_features, synthetic_labels, HyperparamConfig(), FeatureConfig()
    )
    model2 = train_classifier(
        synthetic_features, synthetic_labels, HyperparamConfig(), FeatureConfig()
    )
    p1 = predict_proba(model1, synthetic_features)
    p2 = predict_proba(model2, synthetic_features)
    np.testing.assert_array_almost_equal(p1, p2)


# --- Threshold tests ---


def test_threshold_in_valid_range(synthetic_features, synthetic_labels):
    model = train_classifier(
        synthetic_features, synthetic_labels, HyperparamConfig(), FeatureConfig()
    )
    proba = predict_proba(model, synthetic_features)
    threshold, fbeta = find_optimal_threshold(synthetic_labels, proba)
    assert 0 < threshold < 1
    assert 0 <= fbeta <= 1


def test_apply_threshold():
    proba = np.array([0.1, 0.5, 0.9])
    result = apply_threshold(proba, 0.5)
    np.testing.assert_array_equal(result, np.array([0, 0, 1]))  # 0.5 exactly is NOT above 0.5


def test_apply_threshold_all_above():
    proba = np.array([0.6, 0.7, 0.8])
    result = apply_threshold(proba, 0.5)
    np.testing.assert_array_equal(result, np.array([1, 1, 1]))


def test_apply_threshold_all_below():
    proba = np.array([0.1, 0.2, 0.3])
    result = apply_threshold(proba, 0.5)
    np.testing.assert_array_equal(result, np.array([0, 0, 0]))


def test_threshold_scaling_factor(synthetic_features, synthetic_labels):
    model = train_classifier(
        synthetic_features, synthetic_labels, HyperparamConfig(), FeatureConfig()
    )
    proba = predict_proba(model, synthetic_features)
    t1, _ = find_optimal_threshold(synthetic_labels, proba, scaling_factor=1.0)
    t2, _ = find_optimal_threshold(synthetic_labels, proba, scaling_factor=2.0)
    assert abs(t2 - 2 * t1) < 1e-10
