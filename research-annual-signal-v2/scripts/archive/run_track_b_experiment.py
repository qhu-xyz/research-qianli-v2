"""Phase 3.2: Track B model development — binary classifier for NB candidates.

Trains on Track B population (cohort in {history_dormant, history_zero}) only.
Target: realized_shadow_price > 0 (any binding = positive).

Models:
  3.2.1 - LightGBM binary classifier with class weights
  3.2.2 - Logistic regression baseline

Usage:
    PYTHONPATH=. uv run python scripts/run_track_b_experiment.py
"""
from __future__ import annotations

import json
import logging
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from ml.config import (
    EVAL_SPLITS, DEV_GROUPS, AQ_QUARTERS, REGISTRY_DIR,
)
from ml.features import build_model_table_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_track_b_features() -> list[str]:
    """Load selected Track B features from Phase 3.1 analysis."""
    path = REGISTRY_DIR / "nb_analysis" / "selected_features.json"
    assert path.exists(), f"Run Phase 3.1 first: {path}"
    with open(path) as f:
        data = json.load(f)
    return data["track_b_features"]


def train_lgbm_binary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, dict]:
    """Train LightGBM binary classifier with scale_pos_weight."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale = n_neg / n_pos if n_pos > 0 else 1.0

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 5,
        "scale_pos_weight": scale,
        "num_threads": 4,
        "verbose": -1,
    }

    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, free_raw_data=False)

    model = lgb.train(params, train_ds, num_boost_round=200)
    scores = model.predict(X_eval)

    raw_imp = model.feature_importance(importance_type="gain")
    total = raw_imp.sum()
    fi = dict(zip(feature_names, (raw_imp / total).tolist())) if total > 0 else {}

    return scores, {"model": "lgbm_binary", "feature_importance": fi}


def train_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, dict]:
    """Train L2-regularized logistic regression."""
    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
    )
    lr.fit(X_train, y_train)
    scores = lr.predict_proba(X_eval)[:, 1]

    coefs = dict(zip(feature_names, lr.coef_[0].tolist()))
    return scores, {"model": "logistic", "coefficients": coefs}


def evaluate_track_b(y_true: np.ndarray, scores: np.ndarray, k: int = 10) -> dict:
    """Evaluate Track B model: AUC, precision@K, recall@K."""
    auc = roc_auc_score(y_true, scores) if y_true.sum() > 0 and y_true.sum() < len(y_true) else 0.5
    top_k = np.argsort(scores)[::-1][:k]
    pred_at_k = np.zeros(len(y_true), dtype=int)
    pred_at_k[top_k] = 1

    prec_k = precision_score(y_true, pred_at_k, zero_division=0)
    rec_k = recall_score(y_true, pred_at_k, zero_division=0)

    return {"AUC": float(auc), f"Precision@{k}": float(prec_k), f"Recall@{k}": float(rec_k)}


def main():
    t0 = time.time()

    features = load_track_b_features()
    logger.info("Track B features (%d): %s", len(features), features)

    # Build model tables
    all_needed: set[str] = set()
    for split_info in EVAL_SPLITS.values():
        for py in split_info["train_pys"] + split_info["eval_pys"]:
            for aq in AQ_QUARTERS:
                all_needed.add(f"{py}/{aq}")
    all_needed.discard("2025-06/aq4")

    model_table = build_model_table_all(sorted(all_needed))

    # Filter to Track B only
    track_b_all = model_table.filter(
        pl.col("cohort").is_in(["history_dormant", "history_zero"])
    )

    # Binary target
    track_b_all = track_b_all.with_columns(
        (pl.col("realized_shadow_price") > 0).cast(pl.Int32).alias("target_bind")
    )

    # Expanding-window train/eval on dev splits only
    results: dict = {"lgbm": {}, "logistic": {}}

    for eval_key, split_info in EVAL_SPLITS.items():
        if split_info["split"] != "dev":
            continue

        train_pys = split_info["train_pys"]
        eval_pys = split_info["eval_pys"]

        train_df = track_b_all.filter(pl.col("planning_year").is_in(train_pys))
        eval_df = track_b_all.filter(pl.col("planning_year").is_in(eval_pys))

        if len(train_df) == 0 or len(eval_df) == 0:
            continue

        X_train = train_df.select(features).to_numpy().astype(np.float64)
        y_train = train_df["target_bind"].to_numpy()
        X_eval = eval_df.select(features).to_numpy().astype(np.float64)
        y_eval = eval_df["target_bind"].to_numpy()

        logger.info(
            "Split %s: train=%d (pos=%.1f%%), eval=%d (pos=%.1f%%)",
            eval_key, len(train_df), y_train.mean() * 100,
            len(eval_df), y_eval.mean() * 100,
        )

        # LightGBM binary
        lgbm_scores, lgbm_info = train_lgbm_binary(X_train, y_train, X_eval, features)
        lgbm_metrics = evaluate_track_b(y_eval, lgbm_scores)
        results["lgbm"][eval_key] = {**lgbm_metrics, **lgbm_info}

        # Logistic regression
        lr_scores, lr_info = train_logistic(X_train, y_train, X_eval, features)
        lr_metrics = evaluate_track_b(y_eval, lr_scores)
        results["logistic"][eval_key] = {**lr_metrics, **lr_info}

    # Print comparison
    print(f"\n{'='*80}")
    print("  Phase 3.2: Track B Model Comparison (Dev Only)")
    print(f"{'='*80}\n")

    header = f"{'Split':<12} {'Model':<12} {'AUC':>8} {'P@10':>8} {'R@10':>8}"
    print(header)
    print("-" * len(header))
    for eval_key in sorted(results["lgbm"].keys()):
        for model_name in ["lgbm", "logistic"]:
            m = results[model_name][eval_key]
            print(
                f"{eval_key:<12} {model_name:<12} "
                f"{m['AUC']:>8.4f} {m.get('Precision@10', 0):>8.4f} "
                f"{m.get('Recall@10', 0):>8.4f}"
            )

    # Means
    for model_name in ["lgbm", "logistic"]:
        aucs = [results[model_name][k]["AUC"] for k in results[model_name]]
        print(f"\n  {model_name} mean AUC: {sum(aucs)/len(aucs):.4f}")

    # Save results
    out_dir = REGISTRY_DIR / "track_b_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Saved to %s (%.1fs)", out_dir, time.time() - t0)


if __name__ == "__main__":
    main()
