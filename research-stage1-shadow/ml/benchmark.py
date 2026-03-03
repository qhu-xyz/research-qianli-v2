"""Multi-month benchmark evaluation for shadow price classification.

Runs the full pipeline (load -> train -> threshold -> evaluate) for each
evaluation month independently. In real mode, uses Ray to parallelize
across months.

CLI: python ml/benchmark.py --version-id v0 --ptype f0 --class-type onpeak
"""

import argparse
import gc
import json
import os
import resource
from pathlib import Path

from ml.config import FeatureConfig, HyperparamConfig, PipelineConfig
from ml.data_loader import load_data
from ml.evaluate import aggregate_months, evaluate_classifier
from ml.features import compute_binary_labels, prepare_features
from ml.threshold import apply_threshold, find_optimal_threshold
from ml.train import predict_proba, train_classifier


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _eval_single_month(
    auction_month: str,
    class_type: str,
    ptype: str,
    hyperparam_config: HyperparamConfig,
    feature_config: FeatureConfig,
    threshold_beta: float = 0.7,
    train_months: int | None = None,
    val_months: int | None = None,
) -> dict | None:
    """Train + evaluate on a single auction month. Returns metrics dict or None if skipped."""
    if train_months is None:
        train_months = PipelineConfig().train_months
    if val_months is None:
        val_months = PipelineConfig().val_months
    print(f"[benchmark] Evaluating {auction_month} (ptype={ptype}), mem: {mem_mb():.0f} MB")

    config = PipelineConfig(
        auction_month=auction_month,
        class_type=class_type,
        period_type=ptype,
        threshold_beta=threshold_beta,
        train_months=train_months,
        val_months=val_months,
    )

    # Load data (train + val)
    train_df, val_df = load_data(config)
    print(f"[benchmark]   train={train_df.shape}, val={val_df.shape}")

    if val_df.shape[0] == 0:
        print(f"[benchmark]   SKIP {auction_month}: empty validation set (ptype={ptype})")
        del train_df, val_df
        gc.collect()
        return None

    # Prepare features
    X_train, _ = prepare_features(train_df, feature_config)
    y_train = compute_binary_labels(train_df)
    X_val, _ = prepare_features(val_df, feature_config)
    y_val = compute_binary_labels(val_df)

    # Train
    model = train_classifier(X_train, y_train, hyperparam_config, feature_config)

    # Threshold
    val_proba = predict_proba(model, X_val)
    threshold, max_fbeta = find_optimal_threshold(y_val, val_proba, beta=threshold_beta)

    # Evaluate on val set
    val_pred = apply_threshold(val_proba, threshold)
    val_sp = val_df["actual_shadow_price"].to_numpy()
    metrics = evaluate_classifier(y_val, val_proba, val_pred, val_sp, threshold)

    # Cleanup
    del train_df, val_df, X_train, X_val, y_train, y_val, model, val_proba, val_pred, val_sp
    gc.collect()

    print(f"[benchmark]   AUC={metrics['S1-AUC']}, AP={metrics['S1-AP']}, "
          f"BRIER={metrics['S1-BRIER']}, mem: {mem_mb():.0f} MB")
    return metrics


def run_benchmark(
    version_id: str,
    eval_months: list[str],
    class_type: str = "onpeak",
    ptype: str = "f0",
    registry_dir: str = "registry",
    hyperparam_config: HyperparamConfig | None = None,
    feature_config: FeatureConfig | None = None,
    threshold_beta: float = 0.7,
    train_months: int | None = None,
    val_months: int | None = None,
    overrides: dict | None = None,
) -> dict:
    """Run benchmark across multiple evaluation months.

    Parameters
    ----------
    version_id : str
        Version ID to register results under (e.g. "v0").
    eval_months : list[str]
        List of auction months to evaluate (e.g. ["2020-09", "2020-11"]).
    class_type : str
        "onpeak" or "offpeak".
    ptype : str
        Period type ("f0", "f1", etc.).
    registry_dir : str
        Path to registry directory.
    hyperparam_config : HyperparamConfig or None
        Override hyperparameters.
    feature_config : FeatureConfig or None
        Override feature config.
    threshold_beta : float
        F-beta parameter for threshold optimization.
    overrides : dict or None
        Config overrides (applied to hyperparam + pipeline configs).

    Returns
    -------
    result : dict
        {"per_month": {...}, "aggregate": {...}, "eval_config": {...}, ...}
    """
    if train_months is None:
        train_months = PipelineConfig().train_months
    if val_months is None:
        val_months = PipelineConfig().val_months
    if hyperparam_config is None:
        hyperparam_config = HyperparamConfig()
    if feature_config is None:
        feature_config = FeatureConfig()

    if overrides:
        from ml.pipeline import _apply_overrides
        pc_dummy = PipelineConfig(threshold_beta=threshold_beta, train_months=train_months, val_months=val_months)
        hyperparam_config, pc_dummy = _apply_overrides(hyperparam_config, pc_dummy, overrides)
        threshold_beta = pc_dummy.threshold_beta
        train_months = pc_dummy.train_months
        val_months = pc_dummy.val_months

    smoke = os.environ.get("SMOKE_TEST", "false").lower() == "true"

    # Init Ray once for all months (data_loader uses Ray internally for loading;
    # we keep it alive across months to avoid 12 init/shutdown cycles).
    # NOTE: Ray-parallel dispatch of entire eval months doesn't work because
    # Ray cluster workers lack polars/sklearn. Instead we run months sequentially,
    # each using Ray only for data loading via MisoDataLoader.
    we_inited_ray = False
    if not smoke:
        import ray
        if not ray.is_initialized():
            os.environ.setdefault("RAY_ADDRESS", "ray://10.8.0.36:10001")
            from pbase.config.ray import init_ray
            import pmodel
            import ml as shadow_ml
            init_ray(extra_modules=[pmodel, shadow_ml])
            we_inited_ray = True

    per_month = {}
    skipped = []
    for month in eval_months:
        metrics = _eval_single_month(
            month, class_type, ptype, hyperparam_config, feature_config,
            threshold_beta, train_months, val_months
        )
        if metrics is None:
            skipped.append(month)
            continue
        per_month[month] = metrics

    if skipped:
        print(f"[benchmark] Skipped {len(skipped)} months with empty val: {skipped}")

    if we_inited_ray:
        import ray
        ray.shutdown()

    # Aggregate
    agg = aggregate_months(per_month)

    result = {
        "eval_config": {
            "eval_months": eval_months,
            "class_type": class_type,
            "ptype": ptype,
            "train_months": train_months,
            "val_months": val_months,
            "threshold_beta": threshold_beta,
        },
        "per_month": per_month,
        "aggregate": agg,
        "n_months": len(per_month),
        "n_months_requested": len(eval_months),
        "skipped_months": skipped,
        "threshold_per_month": {m: per_month[m].get("threshold") for m in per_month},
    }

    # Register in registry
    registry_path = Path(registry_dir)
    version_dir = registry_path / version_id
    version_dir.mkdir(parents=True, exist_ok=True)

    with open(version_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[benchmark] Wrote metrics to {version_dir / 'metrics.json'}")

    config_out = {
        "hyperparams": hyperparam_config.to_dict(),
        "features": feature_config.features,
        "eval_config": result["eval_config"],
    }
    with open(version_dir / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    meta = {"n_months": len(per_month), "version_id": version_id}
    with open(version_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    skip_msg = f", {len(skipped)} skipped" if skipped else ""
    print(f"[benchmark] Benchmark complete: {len(per_month)} months evaluated{skip_msg}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run multi-month benchmark evaluation")
    parser.add_argument("--version-id", required=True, help="Version ID (e.g. v0)")
    parser.add_argument("--ptype", default="f0", help="Period type")
    parser.add_argument("--class-type", default="onpeak", help="Class type")
    parser.add_argument("--eval-months", nargs="+", default=None,
                        help="Eval months (default: read from gates.json)")
    parser.add_argument("--registry-dir", default="registry", help="Registry directory")
    parser.add_argument("--gates-path", default="registry/gates.json", help="Gates JSON")
    parser.add_argument("--overrides", default=None, help="JSON config overrides")
    args = parser.parse_args()

    eval_months = args.eval_months
    if eval_months is None:
        with open(args.gates_path) as f:
            gates = json.load(f)
        eval_months = gates.get("eval_months", {}).get("primary", [])
        if not eval_months:
            raise ValueError("No eval_months in gates.json and none provided via --eval-months")

    overrides = json.loads(args.overrides) if args.overrides else None

    run_benchmark(
        version_id=args.version_id,
        eval_months=eval_months,
        class_type=args.class_type,
        ptype=args.ptype,
        registry_dir=args.registry_dir,
        overrides=overrides,
    )


if __name__ == "__main__":
    main()
