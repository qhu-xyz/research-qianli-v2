"""Multi-month benchmark evaluation for stage-2 shadow price regression.

Runs the full pipeline (load -> clf -> reg -> EV-score -> evaluate) for each
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

from ml.config import PipelineConfig, RegressorConfig
from ml.evaluate import aggregate_months
from ml.pipeline import run_pipeline


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _eval_single_month(
    auction_month: str,
    class_type: str,
    period_type: str,
    config: PipelineConfig,
) -> dict | None:
    """Run stage-2 pipeline on a single auction month. Returns metrics dict or None if skipped."""
    print(f"[benchmark] Evaluating {auction_month} (period_type={period_type}), mem: {mem_mb():.0f} MB")

    result = run_pipeline(
        config=config,
        version_id="_bench",
        auction_month=auction_month,
        class_type=class_type,
        period_type=period_type,
    )

    metrics = result.get("metrics", {})
    if not metrics:
        print(f"[benchmark]   SKIP {auction_month}: empty metrics")
        return None

    # Store threshold alongside metrics for per-month tracking
    metrics["threshold"] = result.get("threshold", 0.5)

    gc.collect()

    ev_vc100 = metrics.get("EV-VC@100", "N/A")
    ev_ndcg = metrics.get("EV-NDCG", "N/A")
    spearman = metrics.get("Spearman", "N/A")
    print(f"[benchmark]   EV-VC@100={ev_vc100}, EV-NDCG={ev_ndcg}, "
          f"Spearman={spearman}, mem: {mem_mb():.0f} MB")
    return metrics


def run_benchmark(
    version_id: str,
    eval_months: list[str],
    class_type: str = "onpeak",
    period_type: str = "f0",
    registry_dir: str = "registry",
    config: PipelineConfig | None = None,
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
    period_type : str
        Period type ("f0", "f1", etc.).
    registry_dir : str
        Path to registry directory.
    config : PipelineConfig or None
        Pipeline config. Uses default if None.
    overrides : dict or None
        Config overrides applied to regressor + pipeline.

    Returns
    -------
    result : dict
        {"per_month": {...}, "aggregate": {...}, "eval_config": {...}, ...}
    """
    if config is None:
        config = PipelineConfig()

    if overrides:
        from ml.pipeline import _apply_config_overrides
        config = _apply_config_overrides(config, json.dumps(overrides))

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
        metrics = _eval_single_month(month, class_type, period_type, config)
        if metrics is None:
            skipped.append(month)
            continue
        per_month[month] = metrics

    if skipped:
        print(f"[benchmark] Skipped {len(skipped)} months with empty val: {skipped}")

    if we_inited_ray:
        import ray
        ray.shutdown()

    # Extract feature importance before aggregation (not a numeric metric).
    # In stage 2, feature importance is from the REGRESSOR model.
    importance_per_month = {}
    for month in list(per_month.keys()):
        imp = per_month[month].pop("_feature_importance", None)
        if imp:
            importance_per_month[month] = imp

    # Extract threshold before aggregation (not a gate metric)
    threshold_per_month = {}
    for month in list(per_month.keys()):
        threshold_per_month[month] = per_month[month].pop("threshold", None)

    # Aggregate
    agg = aggregate_months(per_month)

    result = {
        "eval_config": {
            "eval_months": eval_months,
            "class_type": class_type,
            "period_type": period_type,
            "train_months": config.train_months,
            "val_months": config.val_months,
            "ev_scoring": config.ev_scoring,
        },
        "per_month": per_month,
        "aggregate": agg,
        "n_months": len(per_month),
        "n_months_requested": len(eval_months),
        "skipped_months": skipped,
        "threshold_per_month": threshold_per_month,
    }

    # Register in registry
    registry_path = Path(registry_dir)
    version_dir = registry_path / version_id
    version_dir.mkdir(parents=True, exist_ok=True)

    with open(version_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[benchmark] Wrote metrics to {version_dir / 'metrics.json'}")

    if importance_per_month:
        import statistics
        all_features = list(next(iter(importance_per_month.values())).keys())
        mean_imp = {}
        std_imp = {}
        for feat in all_features:
            vals = [importance_per_month[m].get(feat, 0.0) for m in importance_per_month]
            mean_imp[feat] = statistics.mean(vals)
            std_imp[feat] = statistics.stdev(vals) if len(vals) > 1 else 0.0

        fi_data = {
            "importance_type": "gain",
            "model": "regressor",
            "n_months": len(importance_per_month),
            "per_month": importance_per_month,
            "aggregate": {
                "mean": mean_imp,
                "std": std_imp,
            },
            "ranked": sorted(mean_imp.items(), key=lambda x: x[1], reverse=True),
        }
        with open(version_dir / "feature_importance.json", "w") as f:
            json.dump(fi_data, f, indent=2)
        print(f"[benchmark] Wrote feature importance to {version_dir / 'feature_importance.json'}")

    config_out = {
        "classifier": config.classifier.to_dict(),
        "regressor": config.regressor.to_dict(),
        "pipeline": {
            "train_months": config.train_months,
            "val_months": config.val_months,
            "ev_scoring": config.ev_scoring,
        },
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
    parser = argparse.ArgumentParser(description="Run multi-month benchmark evaluation (stage 2)")
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
        period_type=args.ptype,
        registry_dir=args.registry_dir,
        overrides=overrides,
    )


if __name__ == "__main__":
    main()
