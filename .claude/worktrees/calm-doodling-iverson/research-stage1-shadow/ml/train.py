"""Model training for shadow price classification pipeline."""

import resource

import numpy as np
from xgboost import XGBClassifier

from ml.config import FeatureConfig, HyperparamConfig
from ml.features import compute_scale_pos_weight


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: HyperparamConfig,
    feature_config: FeatureConfig,
) -> XGBClassifier:
    """Train an XGBoost classifier with monotone constraints.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Binary training labels.
    config : HyperparamConfig
        Hyperparameter configuration.
    feature_config : FeatureConfig
        Feature configuration (for monotone constraints).

    Returns
    -------
    model : XGBClassifier
        Fitted XGBoost classifier.
    """
    spw = compute_scale_pos_weight(y_train)
    print(f"[train] scale_pos_weight={spw:.2f}, mem before fit: {mem_mb():.0f} MB")

    model = XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        min_child_weight=config.min_child_weight,
        random_state=config.random_state,
        scale_pos_weight=spw,
        monotone_constraints=feature_config.get_monotone_constraints_str(),
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    print(f"[train] mem after fit: {mem_mb():.0f} MB")
    return model


def predict_proba(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    """Return P(binding) for each sample.

    Parameters
    ----------
    model : XGBClassifier
        Fitted classifier.
    X : np.ndarray
        Feature matrix.

    Returns
    -------
    proba : np.ndarray
        Probability of binding (class 1) for each sample.
    """
    return model.predict_proba(X)[:, 1]
