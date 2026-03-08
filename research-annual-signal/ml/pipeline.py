"""Annual LTR pipeline: load -> features -> train -> predict -> evaluate.

For each eval group (planning_year/aq_round):
1. Train on all prior years (expanding window)
2. Predict on eval group
3. Evaluate against realized DA shadow prices
"""
from __future__ import annotations

import gc
import resource
from typing import Any

import numpy as np
import polars as pl

from ml.config import PipelineConfig, EVAL_SPLITS, AQ_ROUNDS
from ml.data_loader import load_v61_enriched, load_multiple_groups
from ml.evaluate import evaluate_ltr
from ml.features import compute_query_groups, prepare_features
from ml.ground_truth import get_ground_truth
from ml.train import predict_scores, train_ltr_model


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _get_train_groups(eval_group: str) -> list[str]:
    """Determine training groups for a given eval group using expanding window."""
    eval_year = eval_group.split("/")[0]
    for split_name, split_def in EVAL_SPLITS.items():
        if split_def["eval_year"] == eval_year:
            train_years = split_def["train_years"]
            return [f"{y}/{aq}" for y in train_years for aq in AQ_ROUNDS]
    raise ValueError(f"No training split defined for eval group: {eval_group}")


def run_pipeline(
    config: PipelineConfig,
    version_id: str,
    eval_group: str,
) -> dict[str, Any]:
    """Run the annual LTR pipeline for a single eval group.

    Returns dict with "metrics" key.
    """
    planning_year, aq_round = eval_group.split("/")
    print(f"[pipeline] version={version_id} eval={eval_group}")

    # Phase 1: Load train data
    print(f"[phase 1] Loading train data ... (mem={mem_mb():.0f} MB)")
    train_groups = _get_train_groups(eval_group)
    train_df = load_multiple_groups(train_groups)

    # Add ground truth labels to train data
    print(f"[phase 1b] Adding ground truth to train ... (mem={mem_mb():.0f} MB)")
    train_parts = []
    for tg in train_groups:
        ty, tq = tg.split("/")
        part = train_df.filter(pl.col("query_group") == tg)
        part = get_ground_truth(ty, tq, part, cache=True)
        train_parts.append(part)
    train_df = pl.concat(train_parts, how="diagonal")

    # Phase 2: Load test data
    print(f"[phase 2] Loading test data ... (mem={mem_mb():.0f} MB)")
    test_df = load_v61_enriched(planning_year, aq_round)
    test_df = test_df.with_columns(pl.lit(eval_group).alias("query_group"))
    test_df = get_ground_truth(planning_year, aq_round, test_df, cache=True)

    # Phase 3: Prepare features
    print(f"[phase 3] Preparing features ... (mem={mem_mb():.0f} MB)")
    train_df = train_df.sort("query_group")
    X_train, _ = prepare_features(train_df, config.ltr)
    y_train = train_df["realized_shadow_price"].to_numpy().astype(np.float64)
    groups_train = compute_query_groups(train_df)

    X_test, _ = prepare_features(test_df, config.ltr)
    actual_sp = test_df["realized_shadow_price"].to_numpy().astype(np.float64)

    print(f"[phase 3] train={X_train.shape} groups={groups_train} test={X_test.shape} "
          f"(mem={mem_mb():.0f} MB)")

    del train_df
    gc.collect()

    # Phase 4: Train
    print(f"[phase 4] Training LTR model ({config.ltr.backend}) ... (mem={mem_mb():.0f} MB)")
    model = train_ltr_model(X_train, y_train, groups_train, config.ltr)

    del X_train, y_train, groups_train
    gc.collect()

    # Phase 5: Predict + Evaluate
    print(f"[phase 5] Predicting and evaluating ... (mem={mem_mb():.0f} MB)")
    scores = predict_scores(model, X_test)
    metrics = evaluate_ltr(actual_sp, scores)

    # Feature importance
    feat_names = config.ltr.features
    if hasattr(model, "feature_importance"):
        importance = model.feature_importance(importance_type="gain")
    else:
        importance = model.feature_importances_
    metrics["_feature_importance"] = {
        name: float(imp)
        for name, imp in sorted(zip(feat_names, importance), key=lambda x: x[1], reverse=True)
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
