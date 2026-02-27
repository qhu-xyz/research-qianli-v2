"""End-to-end ML pipeline for shadow price classification.

Phases: load -> features -> train -> threshold -> evaluate -> register -> save model.

CLI: python ml/pipeline.py --version-id v0 --auction-month 2021-07 --class-type onpeak --period-type f0
"""

import argparse
import json
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ml.config import FeatureConfig, HyperparamConfig, PipelineConfig
from ml.data_loader import load_data
from ml.evaluate import evaluate_classifier
from ml.features import compute_binary_labels, prepare_features
from ml.registry import register_version
from ml.threshold import apply_threshold, find_optimal_threshold
from ml.train import predict_proba, train_classifier


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_pipeline(
    config: PipelineConfig,
    hyperparam_config: HyperparamConfig | None = None,
    feature_config: FeatureConfig | None = None,
    from_phase: int = 1,
) -> dict:
    """Run the full ML pipeline.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.
    hyperparam_config : HyperparamConfig or None
        Hyperparameter config (default created if None).
    feature_config : FeatureConfig or None
        Feature config (default created if None).
    from_phase : int
        Phase to start from (for crash recovery).

    Returns
    -------
    metrics : dict
        Evaluation metrics from the pipeline run.
    """
    if hyperparam_config is None:
        hyperparam_config = HyperparamConfig()
    if feature_config is None:
        feature_config = FeatureConfig()

    version_id = config.version_id or "v0"
    registry_dir = Path(config.registry_dir)

    print(f"[pipeline] Starting pipeline for {version_id}")
    print(f"[pipeline] mem at start: {mem_mb():.0f} MB")

    # Phase 1: Load data
    if from_phase <= 1:
        print(f"\n[pipeline] Phase 1: Loading data...")
        print(f"[pipeline] mem before load: {mem_mb():.0f} MB")
        train_df, val_df = load_data(config)
        print(f"[pipeline] train shape: {train_df.shape}, val shape: {val_df.shape}")
        print(f"[pipeline] mem after load: {mem_mb():.0f} MB")

    # Phase 2: Prepare features
    if from_phase <= 2:
        print(f"\n[pipeline] Phase 2: Preparing features...")
        print(f"[pipeline] mem before features: {mem_mb():.0f} MB")
        X_train, feature_names = prepare_features(train_df, feature_config)
        y_train = compute_binary_labels(train_df)
        X_val, _ = prepare_features(val_df, feature_config)
        y_val = compute_binary_labels(val_df)
        print(f"[pipeline] X_train: {X_train.shape}, y_train positive rate: {y_train.mean():.3f}")
        print(f"[pipeline] X_val: {X_val.shape}, y_val positive rate: {y_val.mean():.3f}")
        print(f"[pipeline] mem after features: {mem_mb():.0f} MB")

    # Phase 3: Train model
    if from_phase <= 3:
        print(f"\n[pipeline] Phase 3: Training model...")
        print(f"[pipeline] mem before train: {mem_mb():.0f} MB")
        model = train_classifier(X_train, y_train, hyperparam_config, feature_config)
        print(f"[pipeline] mem after train: {mem_mb():.0f} MB")

    # Phase 4: Threshold optimization
    if from_phase <= 4:
        print(f"\n[pipeline] Phase 4: Optimizing threshold...")
        print(f"[pipeline] mem before threshold: {mem_mb():.0f} MB")
        val_proba = predict_proba(model, X_val)
        threshold, max_fbeta = find_optimal_threshold(
            y_val,
            val_proba,
            beta=config.threshold_beta,
            scaling_factor=config.threshold_scaling_factor,
        )
        print(f"[pipeline] optimal threshold: {threshold:.4f}, max F-beta: {max_fbeta:.4f}")
        print(f"[pipeline] mem after threshold: {mem_mb():.0f} MB")

    # Phase 5: Evaluate
    if from_phase <= 5:
        print(f"\n[pipeline] Phase 5: Evaluating model...")
        print(f"[pipeline] mem before evaluate: {mem_mb():.0f} MB")

        # Evaluate on validation set
        val_pred = apply_threshold(val_proba, threshold)
        val_shadow_prices = val_df["actual_shadow_price"].to_numpy()

        metrics = evaluate_classifier(
            y_val, val_proba, val_pred, val_shadow_prices, threshold
        )
        print(f"[pipeline] Evaluation metrics:")
        for key, value in sorted(metrics.items()):
            if key.startswith("S1-"):
                print(f"  {key}: {value}")
        print(f"[pipeline] mem after evaluate: {mem_mb():.0f} MB")

    # Phase 6: Register
    if from_phase <= 6:
        print(f"\n[pipeline] Phase 6: Registering version...")
        print(f"[pipeline] mem before register: {mem_mb():.0f} MB")

        resolved_config = {
            "pipeline": {
                "auction_month": config.auction_month,
                "class_type": config.class_type,
                "period_type": config.period_type,
                "version_id": version_id,
                "train_months": config.train_months,
                "val_months": config.val_months,
                "threshold_beta": config.threshold_beta,
                "threshold_scaling_factor": config.threshold_scaling_factor,
                "scale_pos_weight_auto": config.scale_pos_weight_auto,
            },
            "hyperparams": hyperparam_config.to_dict(),
            "features": feature_config.features,
        }

        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "feature_count": len(feature_config.features),
            "train_samples": X_train.shape[0],
            "val_samples": X_val.shape[0],
            "threshold": threshold,
            "max_fbeta": max_fbeta,
        }

        # Save model to temp location
        model_dir = registry_dir / version_id / "model"
        # register_version creates the directory, so we save model after registration
        try:
            register_version(
                registry_dir=registry_dir,
                version_id=version_id,
                config=resolved_config,
                metrics=metrics,
                meta=meta,
            )
        except FileExistsError:
            print(f"[pipeline] WARNING: {version_id} already exists, overwriting metrics")
            import json as json_mod
            version_dir = registry_dir / version_id
            with open(version_dir / "config.json", "w") as f:
                json_mod.dump(resolved_config, f, indent=2)
            with open(version_dir / "metrics.json", "w") as f:
                json_mod.dump(metrics, f, indent=2)
            with open(version_dir / "meta.json", "w") as f:
                json_mod.dump(meta, f, indent=2)

        # Save model file
        model_dir = registry_dir / version_id / "model"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "classifier.ubj"
        model.save_model(str(model_path))
        print(f"[pipeline] model saved to {model_path}")

        print(f"[pipeline] mem after register: {mem_mb():.0f} MB")

    print(f"\n[pipeline] Pipeline complete for {version_id}")
    print(f"[pipeline] mem at end: {mem_mb():.0f} MB")
    return metrics


def _apply_overrides(
    hyperparam_config: HyperparamConfig,
    pipeline_config: PipelineConfig,
    overrides: dict,
) -> tuple[HyperparamConfig, PipelineConfig]:
    """Apply JSON overrides to config objects.

    Keys are matched against HyperparamConfig fields first,
    then PipelineConfig fields. Unknown keys raise ValueError.
    """
    hp_fields = {f for f in HyperparamConfig.__dataclass_fields__}
    pc_fields = {f for f in PipelineConfig.__dataclass_fields__}

    for key, value in overrides.items():
        if key in hp_fields:
            setattr(hyperparam_config, key, value)
        elif key in pc_fields:
            setattr(pipeline_config, key, value)
        else:
            raise ValueError(
                f"Unknown override key: '{key}'. "
                f"Valid keys: {sorted(hp_fields | pc_fields)}"
            )

    return hyperparam_config, pipeline_config


def main():
    parser = argparse.ArgumentParser(
        description="Run shadow price classification pipeline"
    )
    parser.add_argument("--version-id", required=True, help="Version identifier")
    parser.add_argument("--auction-month", default=None, help="Auction month (YYYY-MM)")
    parser.add_argument("--class-type", default="onpeak", help="Class type")
    parser.add_argument("--period-type", default="f0", help="Period type")
    parser.add_argument("--from-phase", type=int, default=1, help="Phase to start from")
    parser.add_argument("--registry-dir", default="registry", help="Registry directory")
    parser.add_argument(
        "--overrides",
        default=None,
        help='JSON string of config overrides (e.g. \'{"n_estimators": 300}\')',
    )
    args = parser.parse_args()

    pipeline_config = PipelineConfig(
        version_id=args.version_id,
        auction_month=args.auction_month,
        class_type=args.class_type,
        period_type=args.period_type,
        registry_dir=args.registry_dir,
    )
    hyperparam_config = HyperparamConfig()
    feature_config = FeatureConfig()

    if args.overrides:
        overrides = json.loads(args.overrides)
        hyperparam_config, pipeline_config = _apply_overrides(
            hyperparam_config, pipeline_config, overrides
        )

    metrics = run_pipeline(
        config=pipeline_config,
        hyperparam_config=hyperparam_config,
        feature_config=feature_config,
        from_phase=args.from_phase,
    )

    print("\n[pipeline] Final metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
