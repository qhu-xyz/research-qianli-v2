# ml/pipeline.py
"""LTR pipeline: load -> features -> train -> predict -> evaluate."""
from __future__ import annotations

import gc
import resource
from typing import Any

import numpy as np
import polars as pl

from ml.config import PipelineConfig
from ml.data_loader import load_v62b_month
from ml.evaluate import evaluate_ltr
from ml.features import compute_query_groups, prepare_features
from ml.train import predict_scores, train_ltr_model


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_pipeline(
    config: PipelineConfig,
    version_id: str,
    eval_month: str,
    period_type: str = "f0",
    class_type: str = "onpeak",
) -> dict[str, Any]:
    """Run the LTR pipeline for a single evaluation month."""
    print(f"[pipeline] version={version_id} eval_month={eval_month}")

    import pandas as pd
    eval_ts = pd.Timestamp(eval_month)
    total_lookback = config.train_months + config.val_months

    train_month_strs = []
    for i in range(total_lookback, config.val_months, -1):
        m = (eval_ts - pd.DateOffset(months=i)).strftime("%Y-%m")
        train_month_strs.append(m)

    dfs = []
    for m in train_month_strs:
        try:
            df = load_v62b_month(m, period_type, class_type)
            df = df.with_columns(pl.lit(m).alias("query_month"))
            dfs.append(df)
        except FileNotFoundError:
            print(f"[pipeline] WARNING: skipping {m}")
    if not dfs:
        return {"metrics": {}}
    train_df = pl.concat(dfs)

    test_df = load_v62b_month(eval_month, period_type, class_type)
    test_df = test_df.with_columns(pl.lit(eval_month).alias("query_month"))

    train_df = train_df.sort("query_month")
    X_train, _ = prepare_features(train_df, config.ltr)
    y_train = train_df["realized_sp"].to_numpy().astype(np.float64)
    groups_train = compute_query_groups(train_df)

    del dfs
    gc.collect()

    model = train_ltr_model(X_train, y_train, groups_train, config.ltr)
    del X_train, y_train, groups_train
    gc.collect()

    X_test, _ = prepare_features(test_df, config.ltr)
    scores = predict_scores(model, X_test)
    actual_sp = test_df["realized_sp"].to_numpy().astype(np.float64)

    metrics = evaluate_ltr(actual_sp, scores)

    feat_names = config.ltr.features
    if hasattr(model, "feature_importance"):
        importance = model.feature_importance(importance_type="gain")
    else:
        importance = model.feature_importances_
    metrics["_feature_importance"] = {
        name: float(imp)
        for name, imp in sorted(zip(feat_names, importance), key=lambda x: x[1], reverse=True)
    }

    del X_test, scores, actual_sp, test_df, model
    gc.collect()

    for key, value in metrics.items():
        if key.startswith("_"):
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")

    return {"metrics": metrics}
