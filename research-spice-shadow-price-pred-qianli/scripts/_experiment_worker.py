"""Experiment pipeline worker — runs ONE (auction_month, class_type, period_type).

Uses the CURRENT codebase with optional config overrides passed as JSON.

Usage (from pmodel venv):
  PYTHONPATH=/path/to/src:$PYTHONPATH python _experiment_worker.py \
      --auction-month 2020-07 --class-type onpeak --period-type f0 \
      --output-dir /tmp/experiment_v001 \
      --config-overrides '{"threshold_beta": 2.0}'
"""

import argparse
import gc
import json
import resource
import sys
import time

import pandas as pd


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def apply_overrides(config, overrides: dict):
    """Apply config overrides to a PredictionConfig instance.

    Supported overrides:
      threshold_beta: float — ThresholdConfig.threshold_beta
      threshold_override: float — bypass optimization, use fixed threshold for all groups
      train_months: int — TrainingConfig.train_months
      val_months: int — TrainingConfig.val_months
      step1_features: list[list] — FeatureConfig.step1_features (list of [name, mono] pairs)
      step2_features: list[list] — FeatureConfig.step2_features
      default_clf_n_estimators: int
      default_clf_max_depth: int
      default_clf_learning_rate: float
      default_clf_min_child_weight: int
      branch_clf_n_estimators: int
      branch_clf_max_depth: int
      branch_clf_learning_rate: float
      branch_clf_min_child_weight: int
      default_reg_n_estimators: int
      default_reg_max_depth: int
      default_reg_learning_rate: float
      default_reg_min_child_weight: int
      branch_reg_n_estimators: int
      branch_reg_max_depth: int
      branch_reg_learning_rate: float
      branch_reg_min_child_weight: int
      {default,branch}_{clf,reg}_{subsample,colsample_bytree,reg_alpha,reg_lambda}: float
    """
    from dataclasses import replace

    if "threshold_beta" in overrides:
        config.threshold = replace(config.threshold, threshold_beta=overrides["threshold_beta"])
    if "threshold_override" in overrides:
        config.threshold = replace(config.threshold, threshold_override=overrides["threshold_override"])

    if "train_months" in overrides:
        config.training = replace(config.training, train_months=overrides["train_months"])
    if "val_months" in overrides:
        config.training = replace(config.training, val_months=overrides["val_months"])

    if "step1_features" in overrides:
        from shadow_price_prediction.config import FeatureConfig
        new_s1 = [(f[0], f[1]) for f in overrides["step1_features"]]
        s2 = config.features.step2_features
        config.features = FeatureConfig(step1_features=new_s1, step2_features=s2)

    if "step2_features" in overrides:
        from shadow_price_prediction.config import FeatureConfig
        s1 = config.features.step1_features
        new_s2 = [(f[0], f[1]) for f in overrides["step2_features"]]
        config.features = FeatureConfig(step1_features=s1, step2_features=new_s2)

    # XGBoost parameter overrides for default classifiers
    for spec in config.models.default_classifiers:
        for key in ("n_estimators", "max_depth", "learning_rate", "min_child_weight", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"):
            override_key = f"default_clf_{key}"
            if override_key in overrides:
                spec.config.params[key] = overrides[override_key]

    # XGBoost parameter overrides for branch classifiers
    for spec in config.models.branch_classifiers:
        for key in ("n_estimators", "max_depth", "learning_rate", "min_child_weight", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"):
            override_key = f"branch_clf_{key}"
            if override_key in overrides:
                spec.config.params[key] = overrides[override_key]

    # XGBoost parameter overrides for default regressors
    for spec in config.models.default_regressors:
        for key in ("n_estimators", "max_depth", "learning_rate", "min_child_weight", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"):
            override_key = f"default_reg_{key}"
            if override_key in overrides:
                spec.config.params[key] = overrides[override_key]

    # XGBoost parameter overrides for branch regressors
    for spec in config.models.branch_regressors:
        for key in ("n_estimators", "max_depth", "learning_rate", "min_child_weight", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"):
            override_key = f"branch_reg_{key}"
            if override_key in overrides:
                spec.config.params[key] = overrides[override_key]

    return config


def main():
    parser = argparse.ArgumentParser(description="Experiment pipeline single-run worker")
    parser.add_argument("--auction-month", required=True, help="e.g. 2020-07")
    parser.add_argument("--class-type", required=True, choices=["onpeak", "offpeak"])
    parser.add_argument("--period-type", required=True, choices=["f0", "f1"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-overrides", default="{}", help="JSON string of config overrides")
    args = parser.parse_args()

    am = pd.Timestamp(args.auction_month)
    class_type = args.class_type
    period_type = args.period_type
    output_dir = args.output_dir
    overrides = json.loads(args.config_overrides)

    # Market month depends on period type
    if period_type == "f0":
        mm = am
    elif period_type == "f1":
        mm = am + pd.DateOffset(months=1)
    else:
        raise ValueError(f"Unsupported period_type: {period_type}")

    tag = f"{am.strftime('%Y-%m')}/{class_type}/{period_type}"
    print(f"[{time.strftime('%H:%M:%S')}] Worker starting: {tag}  mem={mem_mb():.0f} MB")
    if overrides:
        print(f"  Config overrides: {json.dumps(overrides, indent=2)}")

    # --- Ray init ---
    from pbase.config.ray import init_ray
    import pmodel
    import shadow_price_prediction

    init_ray(address="ray://10.8.0.36:10001", extra_modules=[pmodel, shadow_price_prediction])
    print(f"[{time.strftime('%H:%M:%S')}] Ray initialized. mem={mem_mb():.0f} MB")

    # --- Import current pipeline ---
    from shadow_price_prediction import ShadowPricePipeline, PredictionConfig

    config = PredictionConfig(
        market_round=1,
        period_type=period_type,
        class_type=class_type,
    )

    # Apply overrides
    config = apply_overrides(config, overrides)

    pipeline = ShadowPricePipeline(config)
    t0 = time.time()

    try:
        results_per_outage, final_results, metrics, _, _ = pipeline.run(
            test_periods=[(am, mm)],
            class_type=class_type,
            verbose=True,
            use_parallel=False,
        )
        elapsed = time.time() - t0

        if results_per_outage is not None and not results_per_outage.empty:
            import os
            os.makedirs(output_dir, exist_ok=True)
            fname = f"results_{am.strftime('%Y%m')}_{class_type}_{period_type}.parquet"
            out_path = os.path.join(output_dir, fname)
            results_per_outage.to_parquet(out_path)
            n_rows = len(results_per_outage)
            n_binding = int((results_per_outage["actual_binding"] == 1).sum())
            print(f"\n[SAVED] {out_path} ({n_rows:,} rows, {n_binding:,} binding, {elapsed:.0f}s)")

            # ── Save new artifacts (threshold, feature importance, train manifest) ──
            _save_worker_artifacts(pipeline, am, class_type, output_dir, config)
        else:
            print(f"\n[WARN] No results for {tag} (elapsed={elapsed:.0f}s)")
            sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] {tag}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
    finally:
        del pipeline
        gc.collect()

    # --- Cleanup ---
    import ray
    ray.shutdown()
    gc.collect()
    print(f"[{time.strftime('%H:%M:%S')}] Worker done: {tag}  mem={mem_mb():.0f} MB")


def _save_worker_artifacts(pipeline, auction_month, class_type, output_dir, config):
    """Save threshold decisions, feature importances, and train manifest alongside parquets."""
    from shadow_price_prediction.naming import (
        worker_threshold_path,
        worker_feature_importance_path,
        worker_train_manifest_path,
    )
    from shadow_price_prediction.pipeline import extract_train_provenance

    am_str = auction_month.strftime("%Y-%m")

    # Get the trained models for this auction month
    period_models = pipeline.trained_models.get(auction_month)
    if period_models is None:
        print(f"  [WARN] No trained models found for {am_str}, skipping artifact save")
        return

    # Threshold decisions
    try:
        thresholds = period_models.extract_threshold_decisions()
        thr_path = worker_threshold_path(output_dir, am_str, class_type)
        with open(thr_path, "w") as f:
            json.dump(thresholds, f, indent=2, default=str)
        print(f"  [ARTIFACT] Thresholds saved: {thr_path}")
    except Exception as e:
        print(f"  [WARN] Failed to save thresholds: {e}")

    # Feature importances
    try:
        importances = period_models.extract_feature_importances()
        fi_path = worker_feature_importance_path(output_dir, am_str, class_type)
        with open(fi_path, "w") as f:
            json.dump(importances, f, indent=2, default=str)
        print(f"  [ARTIFACT] Feature importances saved: {fi_path}")
    except Exception as e:
        print(f"  [WARN] Failed to save feature importances: {e}")

    # Train manifest (provenance)
    try:
        train_data = pipeline.train_data.get(auction_month)
        if train_data is not None:
            feature_cols = config.features.all_features
            provenance = extract_train_provenance(train_data, auction_month, class_type, feature_cols)
            tm_path = worker_train_manifest_path(output_dir, am_str, class_type)
            with open(tm_path, "w") as f:
                json.dump(provenance, f, indent=2, default=str)
            print(f"  [ARTIFACT] Train manifest saved: {tm_path}")
        else:
            print(f"  [WARN] No train_data found for {am_str}, skipping train manifest")
    except Exception as e:
        print(f"  [WARN] Failed to save train manifest: {e}")


if __name__ == "__main__":
    main()
