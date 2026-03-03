"""Main pipeline orchestration: load -> clf -> reg -> EV-score -> evaluate.

6-phase pipeline with crash-recovery support via ``from_phase``.
Prints mem_mb() at key stages for memory tracking.

CLI usage::

    python ml/pipeline.py --version-id v0001 --auction-month 2021-07 \\
        --class-type onpeak --period-type f0

Optional ``--config-override`` accepts a JSON string to override
PipelineConfig fields.
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import warnings
from typing import Any

import numpy as np

from ml.config import PipelineConfig, RegressorConfig
from ml.data_loader import load_data
from ml.evaluate import evaluate_pipeline
from ml.features import (
    compute_binary_labels,
    compute_regression_target,
    prepare_clf_features,
    prepare_reg_features,
)
from ml.train import (
    predict_proba,
    predict_shadow_price,
    train_classifier,
    train_regressor,
)


def mem_mb() -> float:
    """Current process RSS in megabytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# Minimum binding samples before fallback to all samples
_MIN_BINDING_SAMPLES = 5


def run_pipeline(
    config: PipelineConfig,
    version_id: str,
    auction_month: str,
    class_type: str,
    period_type: str,
    from_phase: int = 1,
) -> dict[str, Any]:
    """Run the 6-phase shadow-price pipeline.

    Phases
    ------
    1. Load data (train_df, val_df)
    2. Prepare features and labels
    3. Train classifier (frozen config)
    4. Train regressor (gated or unified)
    5. Evaluate on validation set
    6. Return results

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
        Dictionary with "metrics" (evaluation metrics dict) and
        "threshold" (float, classifier decision threshold).
    """
    print(f"[pipeline] version={version_id} month={auction_month} "
          f"class={class_type} period={period_type}")
    print(f"[pipeline] from_phase={from_phase}")

    # State variables populated across phases
    train_df = None
    val_df = None
    X_train_clf = None
    y_train_binary = None
    X_val_clf = None
    y_val_binary = None
    X_train_reg = None
    y_train_reg = None
    clf_model = None
    threshold = 0.5
    reg_model = None
    metrics: dict[str, Any] = {}

    # ── Phase 1: Load data ──────────────────────────────────────────────
    if from_phase <= 1:
        print(f"[phase 1] Loading data ... (mem={mem_mb():.0f} MB)")
        train_df, val_df = load_data(config, auction_month, class_type, period_type)
        print(f"[phase 1] train={len(train_df)} val={len(val_df)} "
              f"(mem={mem_mb():.0f} MB)")

    # ── Phase 2: Prepare features ───────────────────────────────────────
    if from_phase <= 2:
        assert train_df is not None and val_df is not None, (
            "Phase 2 requires Phase 1 data. Use from_phase <= 1."
        )
        print(f"[phase 2] Preparing features ... (mem={mem_mb():.0f} MB)")

        # Classifier features
        X_train_clf, _ = prepare_clf_features(train_df, config.classifier)
        X_val_clf, _ = prepare_clf_features(val_df, config.classifier)

        # Binary labels
        y_train_binary = compute_binary_labels(train_df)
        y_val_binary = compute_binary_labels(val_df)

        # Regressor features (from train only -- val prepared later)
        X_train_reg, _ = prepare_reg_features(train_df, config.regressor)

        # Regression target (from train only)
        y_train_reg = compute_regression_target(train_df)

        n_binding_train = int(np.sum(y_train_binary == 1))
        print(f"[phase 2] clf_features={X_train_clf.shape[1]} "
              f"reg_features={X_train_reg.shape[1]} "
              f"binding_train={n_binding_train} "
              f"(mem={mem_mb():.0f} MB)")

    # ── Phase 3: Train classifier (frozen config) ───────────────────────
    if from_phase <= 3:
        assert X_train_clf is not None and y_train_binary is not None, (
            "Phase 3 requires Phase 2 features. Use from_phase <= 2."
        )
        print(f"[phase 3] Training classifier ... (mem={mem_mb():.0f} MB)")
        clf_model, threshold = train_classifier(
            X_train_clf,
            y_train_binary,
            config.classifier,
            X_val=X_val_clf,
            y_val=y_val_binary,
        )
        print(f"[phase 3] threshold={threshold:.4f} (mem={mem_mb():.0f} MB)")

    # ── Phase 4: Train regressor ────────────────────────────────────────
    if from_phase <= 4:
        assert X_train_reg is not None and y_train_reg is not None, (
            "Phase 4 requires Phase 2 features. Use from_phase <= 2."
        )
        assert y_train_binary is not None, (
            "Phase 4 requires Phase 2 labels. Use from_phase <= 2."
        )
        print(f"[phase 4] Training regressor ... (mem={mem_mb():.0f} MB)")

        unified = config.regressor.unified_regressor

        if unified:
            # Unified mode: train on ALL samples
            print("[phase 4] mode=unified (all samples)")
            X_reg_fit = X_train_reg
            y_reg_fit = y_train_reg
        else:
            # Gated mode: train only on binding samples
            binding_mask = y_train_binary == 1
            n_binding = int(np.sum(binding_mask))

            if n_binding < _MIN_BINDING_SAMPLES:
                warnings.warn(
                    f"Only {n_binding} binding samples (< {_MIN_BINDING_SAMPLES}). "
                    f"Falling back to all samples for regressor training.",
                    stacklevel=2,
                )
                X_reg_fit = X_train_reg
                y_reg_fit = y_train_reg
            else:
                print(f"[phase 4] mode=gated (binding only, n={n_binding})")
                X_reg_fit = X_train_reg[binding_mask]
                y_reg_fit = y_train_reg[binding_mask]

        reg_model = train_regressor(X_reg_fit, y_reg_fit, config.regressor)
        print(f"[phase 4] regressor trained (mem={mem_mb():.0f} MB)")

        # Free training data
        del X_reg_fit, y_reg_fit
        gc.collect()

    # ── Phase 5: Evaluate on validation set ─────────────────────────────
    if from_phase <= 5:
        assert clf_model is not None and reg_model is not None, (
            "Phase 5 requires trained models. Use from_phase <= 3."
        )
        assert val_df is not None, (
            "Phase 5 requires validation data. Use from_phase <= 1."
        )
        print(f"[phase 5] Evaluating ... (mem={mem_mb():.0f} MB)")

        # Classifier probabilities on val
        val_proba = predict_proba(clf_model, X_val_clf)

        # Regressor features on val
        X_val_reg, _ = prepare_reg_features(val_df, config.regressor)

        # Regressor predictions on val
        unified = config.regressor.unified_regressor

        if unified:
            # Unified: predict on all val samples
            val_shadow = predict_shadow_price(reg_model, X_val_reg)
        else:
            # Gated: only predict on samples above classifier threshold
            val_shadow = np.zeros(len(val_df), dtype=np.float64)
            above_mask = val_proba >= threshold
            if np.any(above_mask):
                val_shadow[above_mask] = predict_shadow_price(
                    reg_model, X_val_reg[above_mask]
                )

        # EV scores
        if config.ev_scoring:
            ev_scores = val_proba * val_shadow
        else:
            ev_scores = val_shadow.copy()

        # Actual shadow prices
        actual = val_df["actual_shadow_price"].to_numpy().astype(np.float64)

        # Evaluate
        metrics = evaluate_pipeline(
            actual_shadow_price=actual,
            pred_proba=val_proba,
            pred_shadow_price=val_shadow,
            ev_scores=ev_scores,
        )

        # Free intermediates
        del X_val_reg, val_proba, val_shadow, ev_scores, actual
        gc.collect()

        print(f"[phase 5] evaluation complete (mem={mem_mb():.0f} MB)")

    # ── Phase 6: Return results ─────────────────────────────────────────
    print(f"[phase 6] Pipeline complete (mem={mem_mb():.0f} MB)")
    print("[metrics]")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    return {
        "metrics": metrics,
        "threshold": threshold,
    }


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run stage-2 shadow-price pipeline.",
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

    Supports top-level fields (train_months, val_months, ev_scoring)
    and nested regressor fields via "regressor" key.
    Classifier overrides are ignored (frozen config).
    """
    overrides = json.loads(overrides_json)

    # Top-level overrides
    if "train_months" in overrides:
        config.train_months = overrides["train_months"]
    if "val_months" in overrides:
        config.val_months = overrides["val_months"]
    if "ev_scoring" in overrides:
        config.ev_scoring = overrides["ev_scoring"]

    # Regressor overrides
    reg_overrides = overrides.get("regressor", {})
    for key, value in reg_overrides.items():
        if hasattr(config.regressor, key):
            setattr(config.regressor, key, value)

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

    print(f"\nFinal threshold: {result['threshold']:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
