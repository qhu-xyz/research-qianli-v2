"""Model training for classifier and regressor pipelines.

train_classifier  -- XGBClassifier with config + optional threshold tuning.
predict_proba     -- positive-class probability extraction.
train_regressor   -- XGBRegressor with config + optional sample weights.
predict_shadow_price -- inverse log1p transform with non-negativity clamp.
"""
from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from ml.config import ClassifierConfig, RegressorConfig
from ml.features import compute_scale_pos_weight
from ml.threshold import find_optimal_threshold


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: ClassifierConfig,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> tuple[XGBClassifier, float]:
    """Train an XGBClassifier.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape ``(n_samples, n_features)``.
    y_train : np.ndarray
        Binary labels of shape ``(n_samples,)`` with values in {0, 1}.
    cfg : ClassifierConfig
        Classifier configuration (hyperparams + monotone constraints).
    X_val : np.ndarray, optional
        Validation feature matrix for threshold optimization.
    y_val : np.ndarray, optional
        Validation labels for threshold optimization.

    Returns
    -------
    model : XGBClassifier
        Trained classifier.
    threshold : float
        Decision threshold. Optimized via F-beta if validation data is
        provided, otherwise 0.5.
    """
    scale_pos_weight = compute_scale_pos_weight(y_train)
    monotone = tuple(cfg.monotone_constraints)

    model = XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        min_child_weight=cfg.min_child_weight,
        scale_pos_weight=scale_pos_weight,
        monotone_constraints=monotone,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Threshold optimization
    if X_val is not None and y_val is not None:
        val_proba = predict_proba(model, X_val)
        threshold = find_optimal_threshold(
            y_val, val_proba, beta=cfg.threshold_beta,
        )
    else:
        threshold = 0.5

    return model, threshold


def predict_proba(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities.

    Parameters
    ----------
    model : XGBClassifier
        Trained classifier.
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.

    Returns
    -------
    np.ndarray
        Probability of positive class, shape ``(n_samples,)``.
    """
    return model.predict_proba(X)[:, 1]


def train_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: RegressorConfig,
    sample_weight: np.ndarray | None = None,
) -> XGBRegressor:
    """Train an XGBRegressor.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape ``(n_samples, n_features)``.
    y_train : np.ndarray
        Regression targets of shape ``(n_samples,)``, typically
        ``log1p(max(0, shadow_price))``.
    cfg : RegressorConfig
        Regressor configuration (hyperparams + monotone constraints).
    sample_weight : np.ndarray, optional
        Per-sample weights of shape ``(n_samples,)``.

    Returns
    -------
    XGBRegressor
        Trained regressor model.
    """
    monotone = tuple(cfg.monotone_constraints)

    model = XGBRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        min_child_weight=cfg.min_child_weight,
        monotone_constraints=monotone,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    return model


def predict_shadow_price(model: XGBRegressor, X: np.ndarray) -> np.ndarray:
    """Predict shadow prices with inverse log1p transform.

    Computes ``expm1(max(0, raw_prediction))`` to invert the
    ``log1p(max(0, price))`` target transform and guarantee non-negative
    outputs.

    Parameters
    ----------
    model : XGBRegressor
        Trained regressor.
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.

    Returns
    -------
    np.ndarray
        Predicted shadow prices, shape ``(n_samples,)``, all >= 0.
    """
    raw = model.predict(X)
    return np.expm1(np.maximum(0.0, raw))
