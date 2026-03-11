"""Learning-to-rank training: LightGBM lambdarank (default) or XGBoost rank:pairwise.

LightGBM is 22x faster than XGBoost on our data (~3s vs ~70s per month).
LightGBM lambdarank requires integer labels — we rank-transform shadow prices
within each query group before training.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ml.config import LTRConfig

# LightGBM deadlocks with auto-detected thread count (64) in this container.
# Cap at 4 threads — fast enough for our data size (~5k rows).
_LGB_NUM_THREADS = 4


def _rank_transform_labels(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Rank-transform continuous labels to integers within each query group.

    LightGBM lambdarank requires integer relevance labels.
    Raw rank-transform preserves full ordering (~600 levels per group).
    Cost: ~1.5s/iteration on this data size. With n_estimators=100 and
    early_stopping=20, training takes ~30-150s per month.
    """
    y_rank = np.zeros(len(y), dtype=np.int32)
    offset = 0
    for g in groups:
        chunk = y[offset : offset + g]
        y_rank[offset : offset + g] = np.argsort(np.argsort(chunk))
        offset += g
    return y_rank


def _tiered_labels(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Convert continuous labels to tiered relevance within each query group.

    Tiers (per group):
      0 = non-binding (realized_sp == 0)  — ~88% of constraints
      1 = binding (realized_sp > 0, bottom 50% of binding)
      2 = strongly binding (top 50% of binding)
      3 = highly binding (top 20% of binding)

    This avoids the raw rank problem where 528 non-binding constraints
    get 528 distinct labels, causing LightGBM to learn noise.
    """
    y_tier = np.zeros(len(y), dtype=np.int32)
    offset = 0
    for g in groups:
        chunk = y[offset : offset + g]
        binding_mask = chunk > 0
        n_binding = binding_mask.sum()
        if n_binding > 0:
            binding_vals = chunk[binding_mask]
            p50 = np.percentile(binding_vals, 50)
            p80 = np.percentile(binding_vals, 80)
            tier = np.zeros(g, dtype=np.int32)
            tier[binding_mask] = 1
            tier[binding_mask & (chunk >= p50)] = 2
            tier[binding_mask & (chunk >= p80)] = 3
            y_tier[offset : offset + g] = tier
        offset += g
    return y_tier


def _log_value_labels(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Convert continuous shadow prices to log-scaled integer relevance labels.

    Non-binding (y=0) -> 0. Binding -> log1p(y) scaled and discretized.
    Winsorizes at p99 within each group to prevent outlier domination.
    Produces ~10-100 distinct label levels (vs 4 in tiered mode).
    """
    y_label = np.zeros(len(y), dtype=np.int32)
    offset = 0
    for g in groups:
        chunk = y[offset : offset + g]
        binding_mask = chunk > 0
        n_binding = binding_mask.sum()
        if n_binding > 0:
            binding_vals = chunk[binding_mask]
            p99 = np.percentile(binding_vals, 99)
            clipped = np.minimum(binding_vals, p99)
            log_vals = np.log1p(clipped)
            max_log = log_vals.max()
            if max_log > 0:
                scaled = (log_vals / max_log * 99).astype(np.int32) + 1
            else:
                scaled = np.ones(n_binding, dtype=np.int32)
            labels = np.zeros(g, dtype=np.int32)
            labels[binding_mask] = scaled
            y_label[offset : offset + g] = labels
        offset += g
    return y_label


def _train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    cfg: LTRConfig,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    groups_val: np.ndarray | None = None,
) -> Any:
    """Train LightGBM lambdarank model."""
    import lightgbm as lgb

    if cfg.label_mode == "tiered":
        y_rank = _tiered_labels(y_train, groups_train)
    elif cfg.label_mode == "log_value":
        y_rank = _log_value_labels(y_train, groups_train)
    else:
        y_rank = _rank_transform_labels(y_train, groups_train)
    max_label = int(y_rank.max())
    label_gain = list(range(max_label + 1))

    train_data = lgb.Dataset(
        X_train,
        label=y_rank,
        group=groups_train.tolist(),
        feature_name=cfg.features,
    )

    mono_str = ",".join(str(m) for m in cfg.monotone_constraints)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [20, 100],
        "num_leaves": cfg.num_leaves,
        "learning_rate": cfg.learning_rate,
        "min_data_in_leaf": cfg.min_child_weight,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "reg_alpha": cfg.reg_alpha,
        "reg_lambda": cfg.reg_lambda,
        "label_gain": label_gain,
        "monotone_constraints": mono_str,
        "verbose": -1,
        "seed": 42,
        "num_threads": _LGB_NUM_THREADS,
    }

    callbacks = []
    valid_sets = []
    valid_names = []

    if X_val is not None and y_val is not None and groups_val is not None:
        if cfg.label_mode == "tiered":
            y_val_rank = _tiered_labels(y_val, groups_val)
        elif cfg.label_mode == "log_value":
            y_val_rank = _log_value_labels(y_val, groups_val)
        else:
            y_val_rank = _rank_transform_labels(y_val, groups_val)
        max_val_label = int(y_val_rank.max())
        if max_val_label > max_label:
            label_gain = list(range(max_val_label + 1))
            params["label_gain"] = label_gain
        val_data = lgb.Dataset(
            X_val, label=y_val_rank, group=groups_val.tolist(), reference=train_data,
        )
        valid_sets = [val_data]
        valid_names = ["val"]
        if cfg.early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(cfg.early_stopping_rounds, verbose=False))

    model = lgb.train(
        params,
        train_data,
        num_boost_round=cfg.n_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks if callbacks else None,
    )

    print(f"[train] LightGBM: {model.current_iteration()} iterations")
    return model


def _train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    cfg: LTRConfig,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    groups_val: np.ndarray | None = None,
) -> Any:
    """Train XGBoost rank:pairwise model (fallback, 22x slower)."""
    import xgboost as xgb

    monotone = tuple(cfg.monotone_constraints)

    model = xgb.XGBRanker(
        objective="rank:pairwise",
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
    )

    model.fit(X_train, y_train, group=groups_train)
    print(f"[train] XGBoost: {cfg.n_estimators} iterations")
    return model


def _train_lightgbm_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    cfg: LTRConfig,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    groups_val: np.ndarray | None = None,
) -> Any:
    """Train LightGBM regressor — predict realized_sp, rank by predicted value."""
    import lightgbm as lgb

    mono_str = ",".join(str(m) for m in cfg.monotone_constraints)

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=cfg.features,
    )

    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": cfg.num_leaves,
        "learning_rate": cfg.learning_rate,
        "min_data_in_leaf": cfg.min_child_weight,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "reg_alpha": cfg.reg_alpha,
        "reg_lambda": cfg.reg_lambda,
        "monotone_constraints": mono_str,
        "verbose": -1,
        "seed": 42,
        "num_threads": _LGB_NUM_THREADS,
    }

    callbacks = []
    valid_sets = []
    valid_names = []

    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        valid_sets = [val_data]
        valid_names = ["val"]
        if cfg.early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(cfg.early_stopping_rounds, verbose=False))

    model = lgb.train(
        params,
        train_data,
        num_boost_round=cfg.n_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks if callbacks else None,
    )

    print(f"[train] LightGBM regression: {model.current_iteration()} iterations")
    return model


def train_ltr_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    cfg: LTRConfig,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    groups_val: np.ndarray | None = None,
) -> Any:
    """Train LTR model using configured backend.

    Parameters
    ----------
    X_train, y_train : arrays
        Training features and relevance labels (shadow prices).
    groups_train : array
        Query group sizes for training data.
    cfg : LTRConfig
        Model configuration (cfg.backend selects LightGBM or XGBoost).
    X_val, y_val, groups_val : arrays, optional
        Validation data for early stopping.

    Returns
    -------
    model
        Trained ranker model (lightgbm.Booster or xgboost.XGBRanker).
    """
    if cfg.backend == "lightgbm":
        return _train_lightgbm(X_train, y_train, groups_train, cfg, X_val, y_val, groups_val)
    elif cfg.backend == "lightgbm_regression":
        return _train_lightgbm_regression(X_train, y_train, groups_train, cfg, X_val, y_val, groups_val)
    elif cfg.backend == "xgboost":
        return _train_xgboost(X_train, y_train, groups_train, cfg, X_val, y_val, groups_val)
    else:
        raise ValueError(f"Unknown backend: {cfg.backend}")


def predict_scores(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict ranking scores. Higher = more binding."""
    return model.predict(X)
