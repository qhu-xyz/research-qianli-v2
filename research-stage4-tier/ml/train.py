"""XGBoost learning-to-rank training for tier ranking pipeline."""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from ml.config import LTRConfig


def train_ltr_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    cfg: LTRConfig,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    groups_val: np.ndarray | None = None,
) -> xgb.XGBRanker:
    """Train XGBoost ranker with pairwise objective.

    Parameters
    ----------
    X_train, y_train : arrays
        Training features and relevance labels (shadow prices).
    groups_train : array
        Query group sizes for training data.
    cfg : LTRConfig
        Model configuration.
    X_val, y_val, groups_val : arrays, optional
        Validation data for early stopping.

    Returns
    -------
    xgb.XGBRanker
        Trained ranker model.
    """
    monotone = tuple(cfg.monotone_constraints)
    use_early_stopping = X_val is not None and y_val is not None

    model = xgb.XGBRanker(
        objective=cfg.objective,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        min_child_weight=cfg.min_child_weight,
        monotone_constraints=monotone,
        tree_method="hist",
        random_state=42,
        early_stopping_rounds=cfg.early_stopping_rounds if use_early_stopping else None,
    )

    fit_kwargs: dict = {"group": groups_train}
    if use_early_stopping and groups_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["eval_group"] = [groups_val]
        fit_kwargs["verbose"] = False

    model.fit(X_train, y_train, **fit_kwargs)

    if use_early_stopping and hasattr(model, "best_iteration"):
        print(f"[train] early stopping: best_iteration={model.best_iteration} "
              f"of {cfg.n_estimators}")

    return model


def predict_scores(model: xgb.XGBRanker, X: np.ndarray) -> np.ndarray:
    """Predict ranking scores. Higher = more binding."""
    return model.predict(X)
