"""LambdaRank training with expanding window + eval-only prediction.

Key rules:
  - Sort by (planning_year, aq_quarter, branch_name) before building groups
  - Eval-only scoring (never predict on training rows)
  - Tiered labels: 0=non-binding, 1/2/3=tertiles of positive SP
"""
from __future__ import annotations

import logging
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from ml.config import LGBM_PARAMS, get_monotone_constraints

logger = logging.getLogger(__name__)


def tiered_labels(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Convert continuous SP to tiered labels per query group.

    Within each group:
      - y == 0 -> label 0
      - y > 0  -> tertile labels 1/2/3 (1=lowest third, 3=highest third)

    Args:
        y: realized_shadow_price array (aligned with groups)
        groups: query group sizes array (sum == len(y))
    """
    labels = np.zeros(len(y), dtype=np.int32)
    offset = 0
    for g_size in groups:
        g_y = y[offset : offset + g_size]
        pos_mask = g_y > 0
        n_pos = pos_mask.sum()
        if n_pos > 0:
            # Rank positives within group, assign tertiles
            pos_vals = g_y[pos_mask]
            # argsort of argsort gives rank (0-based)
            ranks = np.empty(n_pos, dtype=np.int32)
            ranks[pos_vals.argsort().argsort()] = np.arange(n_pos)
            # Tertile boundaries
            t1 = n_pos // 3
            t2 = 2 * n_pos // 3
            tier = np.where(ranks < t1, 1, np.where(ranks < t2, 2, 3))
            labels[offset : offset + g_size][pos_mask] = tier
        offset += g_size
    assert offset == len(y), f"Group sizes don't sum to len(y): {offset} != {len(y)}"
    return labels


def build_query_groups(df: pl.DataFrame) -> np.ndarray:
    """Build LightGBM query group sizes from sorted model table.

    Expects df sorted by (planning_year, aq_quarter, branch_name).
    Returns array of group sizes where each group = one (PY, quarter).
    """
    group_counts = (
        df.group_by(["planning_year", "aq_quarter"], maintain_order=True)
        .agg(pl.len().alias("count"))
    )
    return group_counts["count"].to_numpy()


def train_and_predict(
    model_table: pl.DataFrame,
    train_pys: list[str],
    eval_pys: list[str],
    feature_cols: list[str],
    use_monotone: bool = False,
) -> tuple[pl.DataFrame, dict]:
    """Train LambdaRank on train_pys, predict on eval_pys.

    Args:
        model_table: concatenated model tables with all PYs
        train_pys: planning years for training
        eval_pys: planning years for evaluation
        feature_cols: feature column names

    Returns:
        (scored_eval_df, train_info) where train_info contains:
          - feature_importance: dict[str, float] normalized to sum=1.0
          - walltime: float seconds
          - n_train_rows: int
    """
    t0 = time.time()

    # Sort consistently
    model_table = model_table.sort(["planning_year", "aq_quarter", "branch_name"])

    # Split train / eval
    train_df = model_table.filter(pl.col("planning_year").is_in(train_pys))
    eval_df = model_table.filter(pl.col("planning_year").is_in(eval_pys))

    assert len(train_df) > 0, f"No training data for {train_pys}"
    assert len(eval_df) > 0, f"No eval data for {eval_pys}"

    # Features and target
    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df["realized_shadow_price"].to_numpy().astype(np.float64)
    train_groups = build_query_groups(train_df)

    X_eval = eval_df.select(feature_cols).to_numpy()
    y_eval = eval_df["realized_shadow_price"].to_numpy().astype(np.float64)
    eval_groups = build_query_groups(eval_df)

    # Tiered labels
    train_labels = tiered_labels(y_train, train_groups)
    eval_labels = tiered_labels(y_eval, eval_groups)

    # LightGBM datasets
    train_dataset = lgb.Dataset(
        X_train, label=train_labels, group=train_groups,
        feature_name=feature_cols, free_raw_data=False,
    )
    eval_dataset = lgb.Dataset(
        X_eval, label=eval_labels, group=eval_groups,
        reference=train_dataset, free_raw_data=False,
    )

    # Train
    params = {**LGBM_PARAMS}
    if use_monotone:
        params["monotone_constraints"] = get_monotone_constraints(feature_cols)
    n_estimators = params.pop("n_estimators", 200)

    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=n_estimators,
        valid_sets=[eval_dataset],
        valid_names=["eval"],
    )

    # Score eval only
    scores = model.predict(X_eval)

    # Feature importance (gain-based, normalized)
    raw_imp = model.feature_importance(importance_type="gain")
    total_imp = raw_imp.sum()
    if total_imp > 0:
        norm_imp = raw_imp / total_imp
    else:
        norm_imp = raw_imp
    feature_importance = dict(zip(feature_cols, norm_imp.tolist()))

    walltime = time.time() - t0
    logger.info(
        "Train %s -> eval %s: %d train rows, %d eval rows, %.1fs",
        train_pys, eval_pys, len(train_df), len(eval_df), walltime,
    )

    # Attach scores to eval df
    result = eval_df.with_columns(pl.Series("score", scores))
    train_info = {
        "feature_importance": feature_importance,
        "walltime": walltime,
        "n_train_rows": len(train_df),
    }
    return result, train_info
