"""Model training for tier classification pipeline.

train_tier_classifier       -- XGBClassifier multi:softprob for 5 tiers.
predict_tier_probabilities  -- (n_samples, 5) probability matrix.
predict_tier                -- argmax tier labels.
compute_tier_ev_score       -- probability-weighted expected shadow price.
"""
from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

from ml.config import TierConfig


def train_tier_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: TierConfig,
    sample_weight: np.ndarray | None = None,
) -> XGBClassifier:
    """Train a multi-class XGBClassifier for tier prediction.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape ``(n_samples, n_features)``.
    y_train : np.ndarray
        Tier labels of shape ``(n_samples,)`` with values in {0, 1, 2, 3, 4}.
    cfg : TierConfig
        Tier configuration (hyperparams + monotone constraints).
    sample_weight : np.ndarray, optional
        Per-sample weights. If None, computed from cfg.class_weights.

    Returns
    -------
    XGBClassifier
        Trained multi-class classifier.
    """
    if sample_weight is None:
        sample_weight = np.array(
            [cfg.class_weights.get(int(label), 1.0) for label in y_train],
            dtype=np.float64,
        )

    monotone = tuple(cfg.monotone_constraints)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=cfg.num_class,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        min_child_weight=cfg.min_child_weight,
        monotone_constraints=monotone,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    return model


def predict_tier_probabilities(
    model: XGBClassifier,
    X: np.ndarray,
    num_class: int = 5,
) -> np.ndarray:
    """Return tier probability matrix.

    Always returns ``(n_samples, num_class)`` even if some classes had zero
    training samples (XGBoost drops unseen classes from ``predict_proba``).

    Parameters
    ----------
    model : XGBClassifier
        Trained multi-class classifier.
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    num_class : int
        Expected number of classes (default 5).

    Returns
    -------
    np.ndarray
        Probability matrix of shape ``(n_samples, num_class)``, columns = tiers 0..num_class-1.
    """
    raw_proba = model.predict_proba(X)
    if raw_proba.shape[1] == num_class:
        return raw_proba
    # Pad missing classes with zero probability
    n_samples = raw_proba.shape[0]
    full_proba = np.zeros((n_samples, num_class), dtype=np.float64)
    learned_classes = model.classes_.astype(int)
    for i, cls in enumerate(learned_classes):
        full_proba[:, cls] = raw_proba[:, i]
    return full_proba


def predict_tier(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    """Return predicted tier labels (argmax of probabilities).

    Returns
    -------
    np.ndarray
        Integer tier labels of shape ``(n_samples,)``.
    """
    return model.predict(X).astype(int)


def compute_tier_ev_score(
    proba: np.ndarray,
    midpoints: list[float],
) -> np.ndarray:
    """Compute probability-weighted expected shadow price from tier probabilities.

    tier_ev = sum(P(tier=t) * midpoint[t]) for t in [0, 1, 2, 3, 4]

    This serves as the continuous ranking signal for capital allocation.

    Parameters
    ----------
    proba : np.ndarray
        Probability matrix of shape ``(n_samples, 5)``.
    midpoints : list[float]
        Tier midpoints, e.g. [4000, 2000, 550, 50, 0] for tiers 0-4.

    Returns
    -------
    np.ndarray
        Expected shadow price of shape ``(n_samples,)``.
    """
    return proba @ np.array(midpoints, dtype=np.float64)
