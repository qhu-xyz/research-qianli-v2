"""Multi-month benchmark for LTR ranking pipeline.

Runs the full pipeline (load -> train -> predict -> evaluate) for each
evaluation month independently.

Two-stage evaluation:
  --screen  12 representative months (~36s with LightGBM) — default for hypothesis testing
  --full    36 rolling months (~108s with LightGBM) — only run if screening passes

CLI:
  python ml/benchmark.py --version-id v2 --screen              # hypothesis screening (default)
  python ml/benchmark.py --version-id v2 --full                 # comprehensive validation
  python ml/benchmark.py --version-id v2 --eval-months 2021-06  # specific months
"""

import argparse
import gc
import json
import resource
import statistics
from pathlib import Path

from ml.config import PipelineConfig, _SCREEN_EVAL_MONTHS, _FULL_EVAL_MONTHS
from ml.evaluate import aggregate_months
from ml.pipeline import run_pipeline


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _eval_single_month(
    eval_month: str,
    class_type: str,
    period_type: str,
    config: PipelineConfig,
    version_id: str,
) -> dict | None:
    """Run LTR pipeline on a single eval month. Returns metrics dict or None."""
    print(f"[benchmark] Evaluating {eval_month} (period_type={period_type}), mem: {mem_mb():.0f} MB")

    result = run_pipeline(
        config=config,
        version_id=version_id,
        eval_month=eval_month,
        class_type=class_type,
        period_type=period_type,
    )

    metrics = result.get("metrics", {})
    if not metrics:
        print(f"[benchmark]   SKIP {eval_month}: empty metrics")
        return None

    gc.collect()

    vc100 = metrics.get("VC@100", "N/A")
    ndcg = metrics.get("NDCG", "N/A")
    recall100 = metrics.get("Recall@100", "N/A")
    print(f"[benchmark]   VC@100={vc100}, NDCG={ndcg}, Recall@100={recall100}, mem: {mem_mb():.0f} MB")
    return metrics


def run_benchmark(
    version_id: str,
    eval_months: list[str],
    class_type: str = "onpeak",
    period_type: str = "f0",
    registry_dir: str = "registry",
    config: PipelineConfig | None = None,
    mode: str = "screen",
) -> dict:
    """Run benchmark across multiple evaluation months.

    Parameters
    ----------
    version_id : str
        Version ID to register results under.
    eval_months : list[str]
        List of evaluation months.
    class_type : str
        "onpeak" or "offpeak".
    period_type : str
        Period type ("f0", "f1", etc.).
    registry_dir : str
        Path to registry directory.
    config : PipelineConfig or None
        Pipeline config. Uses default if None.
    mode : str
        "screen" (12 months, hypothesis testing) or "full" (36 months, validation).
    """
    if config is None:
        config = PipelineConfig()

    print(f"[benchmark] {mode.upper()} MODE: {len(eval_months)} eval months")

    per_month = {}
    skipped = []
    for month in eval_months:
        metrics = _eval_single_month(month, class_type, period_type, config, version_id)
        if metrics is None:
            skipped.append(month)
            continue
        per_month[month] = metrics

    if skipped:
        print(f"[benchmark] Skipped {len(skipped)} months: {skipped}")

    # Extract feature importance before aggregation
    importance_per_month = {}
    for month in list(per_month.keys()):
        imp = per_month[month].pop("_feature_importance", None)
        if imp:
            importance_per_month[month] = imp

    # Aggregate
    agg = aggregate_months(per_month)

    result = {
        "eval_config": {
            "eval_months": eval_months,
            "class_type": class_type,
            "period_type": period_type,
            "train_months": config.train_months,
            "val_months": config.val_months,
            "mode": mode,
        },
        "per_month": per_month,
        "aggregate": agg,
        "n_months": len(per_month),
        "n_months_requested": len(eval_months),
        "skipped_months": skipped,
    }

    # Register in registry
    registry_path = Path(registry_dir)
    version_dir = registry_path / version_id
    version_dir.mkdir(parents=True, exist_ok=True)

    with open(version_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[benchmark] Wrote metrics to {version_dir / 'metrics.json'}")

    if importance_per_month:
        all_features = list(next(iter(importance_per_month.values())).keys())
        mean_imp = {}
        std_imp = {}
        for feat in all_features:
            vals = [importance_per_month[m].get(feat, 0.0) for m in importance_per_month]
            mean_imp[feat] = statistics.mean(vals)
            std_imp[feat] = statistics.stdev(vals) if len(vals) > 1 else 0.0

        fi_data = {
            "importance_type": "gain",
            "model": "ltr_ranker",
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
        "ltr": config.ltr.to_dict(),
        "pipeline": {
            "train_months": config.train_months,
            "val_months": config.val_months,
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
    parser = argparse.ArgumentParser(description="Run multi-month LTR benchmark")
    parser.add_argument("--version-id", required=True, help="Version ID (e.g. v2)")
    parser.add_argument("--ptype", default="f0", help="Period type")
    parser.add_argument("--class-type", default="onpeak", help="Class type")
    parser.add_argument("--eval-months", nargs="+", default=None,
                        help="Eval months (overrides --screen/--full)")
    parser.add_argument("--registry-dir", default="registry", help="Registry directory")
    parser.add_argument("--screen", action="store_true", default=True,
                        help="Screen: 12 representative months (default)")
    parser.add_argument("--full", action="store_true",
                        help="Full: 36 rolling months (run after screening passes)")
    args = parser.parse_args()

    if args.eval_months:
        eval_months = args.eval_months
        mode = "custom"
    elif args.full:
        eval_months = _FULL_EVAL_MONTHS
        mode = "full"
    else:
        eval_months = _SCREEN_EVAL_MONTHS
        mode = "screen"

    run_benchmark(
        version_id=args.version_id,
        eval_months=eval_months,
        class_type=args.class_type,
        period_type=args.ptype,
        registry_dir=args.registry_dir,
        mode=mode,
    )


if __name__ == "__main__":
    main()
