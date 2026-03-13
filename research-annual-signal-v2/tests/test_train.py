"""Tests for ml/train.py — LambdaRank training."""
import numpy as np
import pytest


def test_tiered_labels():
    """Labels: 0=non-binding, 1/2/3=tertiles of positive."""
    from ml.train import tiered_labels
    y = np.array([0, 0, 0, 10, 20, 30, 40, 50, 60])
    groups = np.array([9])  # 1 group of size 9
    labels = tiered_labels(y, groups)
    assert set(labels[:3]) == {0}  # zeros stay 0
    assert set(labels[3:]) == {1, 2, 3}  # positives get 1/2/3


def test_query_groups_construction():
    """Test spec F1: query groups from model table."""
    from ml.train import build_query_groups
    import polars as pl
    df = pl.DataFrame({
        "planning_year": ["2022-06"] * 5 + ["2023-06"] * 3,
        "aq_quarter": ["aq1"] * 5 + ["aq2"] * 3,
        "branch_name": [f"b{i}" for i in range(8)],
    })
    groups = build_query_groups(df)
    assert groups.tolist() == [5, 3]
    assert sum(groups) == 8


def test_train_and_predict_smoke():
    """Smoke test: training completes and produces scores."""
    from ml.train import train_and_predict
    from ml.features import build_model_table
    from ml.config import HISTORY_FEATURES
    import polars as pl

    # Build model tables for train and eval groups
    train_table = build_model_table("2023-06", "aq1")
    eval_table = build_model_table("2024-06", "aq1")
    model_table = pl.concat([train_table, eval_table], how="diagonal")

    # train_and_predict receives the assembled model table
    result, train_info = train_and_predict(
        model_table=model_table,
        train_pys=["2023-06"],
        eval_pys=["2024-06"],
        feature_cols=HISTORY_FEATURES,
    )
    assert "score" in result.columns
    assert len(result) > 0
    # Scores only on eval rows
    assert len(result) <= len(eval_table)
    # Feature importance
    assert "feature_importance" in train_info
    assert len(train_info["feature_importance"]) == len(HISTORY_FEATURES)
    assert abs(sum(train_info["feature_importance"].values()) - 1.0) < 1e-6
    assert "walltime" in train_info
    assert train_info["walltime"] > 0
    assert "n_train_rows" in train_info
    assert train_info["n_train_rows"] > 0


def test_lgbm_params():
    """Test spec F3: num_threads=4, lambdarank."""
    from ml.config import LGBM_PARAMS
    assert LGBM_PARAMS["num_threads"] == 4
    assert LGBM_PARAMS["objective"] == "lambdarank"
