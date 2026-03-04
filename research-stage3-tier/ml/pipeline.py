"""Main pipeline orchestration: load -> train tier model -> evaluate.

6-phase pipeline with crash-recovery support via ``from_phase``.
Prints mem_mb() at key stages for memory tracking.

Phases:
  1. Load train/val data (6+2 month lookback)
  2. Prepare features and tier labels
  3. Train multi-class XGBoost tier classifier
  4. Load target-month test data
  5. Evaluate on test data
  6. Return results

CLI usage::

    python ml/pipeline.py --version-id v0001 --auction-month 2021-07 \\
        --class-type onpeak --period-type f0
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
from typing import Any

import numpy as np

from ml.config import PipelineConfig
from ml.data_loader import load_data, load_test_data
from ml.evaluate import evaluate_tier_pipeline
from ml.features import (
    compute_interaction_features,
    compute_sample_weights,
    compute_tier_labels,
    prepare_features,
)
from ml.train import (
    compute_tier_ev_score,
    predict_tier,
    predict_tier_probabilities,
    train_tier_classifier,
)


def mem_mb() -> float:
    """Current process RSS in megabytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_pipeline(
    config: PipelineConfig,
    version_id: str,
    auction_month: str,
    class_type: str,
    period_type: str,
    from_phase: int = 1,
) -> dict[str, Any]:
    """Run the 6-phase tier classification pipeline.

    Parameters
    ----------
    config : PipelineConfig
        Full pipeline configuration.
    version_id : str
        Version identifier for logging.
    auction_month : str
        Auction month in YYYY-MM format.
    class_type : str
        Class type, e.g. "onpeak" or "offpeak".
    period_type : str
        Period type, e.g. "f0" or "monthly".
    from_phase : int
        Phase to start from (1-6) for crash recovery.

    Returns
    -------
    dict
        Dictionary with "metrics" (evaluation metrics dict).
    """
    print(f"[pipeline] version={version_id} month={auction_month} "
          f"class={class_type} period={period_type}")
    print(f"[pipeline] from_phase={from_phase}")

    # State variables populated across phases
    train_df = None
    val_df = None
    test_df = None
    X_train = None
    y_train_tier = None
    model = None
    metrics: dict[str, Any] = {}

    # ── Phase 1: Load train/val data ─────────────────────────────────────
    if from_phase <= 1:
        print(f"[phase 1] Loading train/val data ... (mem={mem_mb():.0f} MB)")
        train_df, val_df = load_data(config, auction_month, class_type, period_type)
        print(f"[phase 1] train={len(train_df)} val={len(val_df)} "
              f"(mem={mem_mb():.0f} MB)")

    # ── Phase 2: Prepare features and tier labels ────────────────────────
    if from_phase <= 2:
        assert train_df is not None and val_df is not None, (
            "Phase 2 requires Phase 1 data. Use from_phase <= 1."
        )
        print(f"[phase 2] Preparing features ... (mem={mem_mb():.0f} MB)")

        # Compute interaction features on train/val
        train_df = compute_interaction_features(train_df)
        val_df = compute_interaction_features(val_df)

        # Feature matrix
        X_train, _ = prepare_features(train_df, config.tier)

        # Tier labels
        actual_sp_train = train_df["actual_shadow_price"].to_numpy().astype(np.float64)
        y_train_tier = compute_tier_labels(actual_sp_train, config.tier)

        tier_counts = {int(t): int((y_train_tier == t).sum()) for t in range(5)}
        print(f"[phase 2] features={X_train.shape[1]} "
              f"tier_dist={tier_counts} (mem={mem_mb():.0f} MB)")

        # Free val early (not needed for training — no threshold optimization)
        del val_df
        gc.collect()

    # ── Phase 3: Train multi-class XGBoost ───────────────────────────────
    if from_phase <= 3:
        assert X_train is not None and y_train_tier is not None, (
            "Phase 3 requires Phase 2 features. Use from_phase <= 2."
        )
        print(f"[phase 3] Training tier classifier ... (mem={mem_mb():.0f} MB)")

        sample_weights = compute_sample_weights(y_train_tier, config.tier)
        model = train_tier_classifier(
            X_train, y_train_tier, config.tier,
            sample_weight=sample_weights,
        )
        print(f"[phase 3] model trained (mem={mem_mb():.0f} MB)")

        # Free training data
        del X_train, y_train_tier, sample_weights, train_df
        gc.collect()

    # ── Phase 4: Load target-month test data ─────────────────────────────
    if from_phase <= 4:
        print(f"[phase 4] Loading test data ... (mem={mem_mb():.0f} MB)")
        test_df = load_test_data(config, auction_month, class_type, period_type)
        test_df = compute_interaction_features(test_df)
        print(f"[phase 4] test={len(test_df)} (mem={mem_mb():.0f} MB)")

    # ── Phase 5: Evaluate on test data ───────────────────────────────────
    if from_phase <= 5:
        assert model is not None, (
            "Phase 5 requires trained model. Use from_phase <= 3."
        )
        assert test_df is not None, (
            "Phase 5 requires test data. Use from_phase <= 4."
        )
        print(f"[phase 5] Evaluating on test data ... (mem={mem_mb():.0f} MB)")

        # Features on test
        X_test, _ = prepare_features(test_df, config.tier)

        # Predictions
        tier_proba = predict_tier_probabilities(model, X_test)
        pred_tier_labels = predict_tier(model, X_test)
        tier_ev = compute_tier_ev_score(tier_proba, config.tier.tier_midpoints)

        # Actuals
        actual_sp = test_df["actual_shadow_price"].to_numpy().astype(np.float64)
        actual_tier = compute_tier_labels(actual_sp, config.tier)

        # Evaluate
        metrics = evaluate_tier_pipeline(
            actual_shadow_price=actual_sp,
            actual_tier=actual_tier,
            pred_tier=pred_tier_labels,
            tier_proba=tier_proba,
            tier_ev_score=tier_ev,
        )

        # Feature importance
        importance = model.feature_importances_
        feat_names = config.tier.features
        metrics["_feature_importance"] = {
            name: float(imp)
            for name, imp in sorted(
                zip(feat_names, importance),
                key=lambda x: x[1],
                reverse=True,
            )
        }

        del X_test, tier_proba, pred_tier_labels, tier_ev, actual_sp, actual_tier
        del test_df
        gc.collect()

        print(f"[phase 5] evaluation complete (mem={mem_mb():.0f} MB)")

    # ── Phase 6: Return results ──────────────────────────────────────────
    print(f"[phase 6] Pipeline complete (mem={mem_mb():.0f} MB)")
    print("[metrics]")
    for key, value in metrics.items():
        if key.startswith("_"):
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    return {"metrics": metrics}


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run tier classification pipeline.",
    )
    parser.add_argument(
        "--version-id", required=True,
        help="Version identifier, e.g. v0001",
    )
    parser.add_argument(
        "--auction-month", required=True,
        help="Auction month in YYYY-MM format",
    )
    parser.add_argument(
        "--class-type", required=True,
        help="Class type, e.g. onpeak or offpeak",
    )
    parser.add_argument(
        "--period-type", required=True,
        help="Period type, e.g. f0 or monthly",
    )
    parser.add_argument(
        "--from-phase", type=int, default=1,
        help="Phase to resume from (1-6) for crash recovery",
    )
    parser.add_argument(
        "--config-override", type=str, default=None,
        help="JSON string to override PipelineConfig fields",
    )
    return parser.parse_args()


def _apply_config_overrides(
    config: PipelineConfig,
    overrides_json: str,
) -> PipelineConfig:
    """Apply JSON overrides to a PipelineConfig.

    Supports top-level fields (train_months, val_months)
    and nested tier fields via "tier" key.
    """
    overrides = json.loads(overrides_json)

    # Top-level overrides
    if "train_months" in overrides:
        config.train_months = overrides["train_months"]
    if "val_months" in overrides:
        config.val_months = overrides["val_months"]

    # Tier overrides
    tier_overrides = overrides.get("tier", {})
    for key, value in tier_overrides.items():
        if hasattr(config.tier, key):
            setattr(config.tier, key, value)

    return config


def main() -> None:
    """CLI entry point."""
    args = _parse_args()

    config = PipelineConfig()
    if args.config_override:
        config = _apply_config_overrides(config, args.config_override)

    result = run_pipeline(
        config=config,
        version_id=args.version_id,
        auction_month=args.auction_month,
        class_type=args.class_type,
        period_type=args.period_type,
        from_phase=args.from_phase,
    )

    print("Done.")


if __name__ == "__main__":
    main()
