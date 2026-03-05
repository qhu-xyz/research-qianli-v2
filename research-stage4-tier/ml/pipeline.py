"""LTR pipeline: load -> features -> train -> predict -> evaluate.

5-phase pipeline with memory tracking.
"""
from __future__ import annotations

import gc
import resource
from typing import Any

import numpy as np
import polars as pl

from ml.config import PipelineConfig
from ml.data_loader import load_train_val_test
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
    """Run the LTR pipeline for a single evaluation month.

    Returns dict with "metrics" key.
    """
    print(f"[pipeline] version={version_id} eval_month={eval_month}")

    # Phase 1: Load data
    print(f"[phase 1] Loading data ... (mem={mem_mb():.0f} MB)")
    train_df, val_df, test_df = load_train_val_test(
        eval_month, config.train_months, config.val_months,
        period_type, class_type,
    )

    # Phase 2: Prepare features
    print(f"[phase 2] Preparing features ... (mem={mem_mb():.0f} MB)")

    # Sort by query_month for proper group computation
    train_df = train_df.sort("query_month")
    val_df = val_df.sort("query_month")

    X_train, _ = prepare_features(train_df, config.ltr)
    y_train = train_df["shadow_price_da"].to_numpy().astype(np.float64)
    groups_train = compute_query_groups(train_df)

    X_val, _ = prepare_features(val_df, config.ltr)
    y_val = val_df["shadow_price_da"].to_numpy().astype(np.float64)
    groups_val = compute_query_groups(val_df)

    print(f"[phase 2] train={X_train.shape} val={X_val.shape} "
          f"groups_train={groups_train} groups_val={groups_val} "
          f"(mem={mem_mb():.0f} MB)")

    del train_df, val_df
    gc.collect()

    # Phase 3: Train
    # NOTE: We do NOT pass eval_set for early stopping because XGBoost's
    # default NDCG eval metric requires labels in [0, 31] but shadow prices
    # are raw floats (0 to 100k+). Training uses all n_estimators instead.
    print(f"[phase 3] Training LTR model ... (mem={mem_mb():.0f} MB)")
    model = train_ltr_model(
        X_train, y_train, groups_train, config.ltr,
    )
    del X_train, y_train, groups_train, X_val, y_val, groups_val
    gc.collect()

    # Phase 4: Predict on test
    print(f"[phase 4] Predicting on test ... (mem={mem_mb():.0f} MB)")
    X_test, _ = prepare_features(test_df, config.ltr)
    scores = predict_scores(model, X_test)
    actual_sp = test_df["shadow_price_da"].to_numpy().astype(np.float64)

    # Phase 5: Evaluate
    print(f"[phase 5] Evaluating ... (mem={mem_mb():.0f} MB)")
    metrics = evaluate_ltr(actual_sp, scores)

    # Feature importance
    importance = model.feature_importances_
    feat_names = config.ltr.features
    metrics["_feature_importance"] = {
        name: float(imp)
        for name, imp in sorted(
            zip(feat_names, importance),
            key=lambda x: x[1],
            reverse=True,
        )
    }

    del X_test, scores, actual_sp, test_df
    gc.collect()

    print(f"[pipeline] complete (mem={mem_mb():.0f} MB)")
    for key, value in metrics.items():
        if key.startswith("_"):
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    return {"metrics": metrics}
