"""Multi-group benchmark for annual LTR ranking pipeline.

Runs pipeline for each eval group independently.

CLI:
  python ml/benchmark.py --version-id v1 --screen     # 4 groups (fast)
  python ml/benchmark.py --version-id v1               # 12 groups (default)
"""
import argparse
import gc
import json
import resource
from pathlib import Path
from typing import Any

from ml.config import PipelineConfig, SCREEN_EVAL_GROUPS, DEFAULT_EVAL_GROUPS
from ml.evaluate import aggregate_months  # works for any groups, name is legacy
from ml.pipeline import train_for_year, evaluate_group


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_benchmark(
    version_id: str,
    eval_groups: list[str],
    registry_dir: str = str(Path(__file__).resolve().parent.parent / "registry"),
    config: PipelineConfig | None = None,
    mode: str = "eval",
) -> dict:
    """Run benchmark across multiple eval groups."""
    if config is None:
        config = PipelineConfig()

    print(f"[benchmark] {mode.upper()} MODE: {len(eval_groups)} eval groups")

    # Group eval_groups by year so we train once per year
    from collections import OrderedDict
    year_groups: dict[str, list[str]] = OrderedDict()
    for g in eval_groups:
        year = g.split("/")[0]
        year_groups.setdefault(year, []).append(g)

    per_group = {}
    skipped = []
    model_cache: dict[str, Any] = {}  # year -> trained model

    for year, groups_in_year in year_groups.items():
        # Train once per year
        if year not in model_cache:
            print(f"\n[benchmark] Training model for eval_year={year} ... mem: {mem_mb():.0f} MB")
            try:
                model_cache[year] = train_for_year(config, year)
            except Exception as e:
                print(f"[benchmark] ERROR training for {year}: {e}")
                import traceback
                traceback.print_exc()
                skipped.extend(groups_in_year)
                continue

        model = model_cache[year]

        # Evaluate each quarter with the cached model
        for group_id in groups_in_year:
            print(f"\n[benchmark] === {group_id} === mem: {mem_mb():.0f} MB")
            try:
                metrics = evaluate_group(config, model, group_id)
                if metrics:
                    per_group[group_id] = metrics
                else:
                    skipped.append(group_id)
            except Exception as e:
                print(f"[benchmark] ERROR {group_id}: {e}")
                import traceback
                traceback.print_exc()
                skipped.append(group_id)
            gc.collect()

    del model_cache
    gc.collect()

    if skipped:
        print(f"\n[benchmark] Skipped {len(skipped)} groups: {skipped}")

    # Extract feature importance
    for group_id in list(per_group.keys()):
        per_group[group_id].pop("_feature_importance", None)

    # Aggregate (reuse monthly aggregate function — works on any dict of dicts)
    agg = aggregate_months(per_group)

    result = {
        "eval_config": {
            "eval_groups": eval_groups,
            "mode": mode,
        },
        "per_month": per_group,  # key name kept for compare.py compatibility
        "aggregate": agg,
        "n_months": len(per_group),
        "n_months_requested": len(eval_groups),
        "skipped_months": skipped,
    }

    # Save to registry
    registry_path = Path(registry_dir)
    version_dir = registry_path / version_id
    version_dir.mkdir(parents=True, exist_ok=True)

    with open(version_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(version_dir / "config.json", "w") as f:
        json.dump({"ltr": config.ltr.to_dict(), "eval_config": result["eval_config"]}, f, indent=2)
    with open(version_dir / "meta.json", "w") as f:
        json.dump({"n_groups": len(per_group), "version_id": version_id}, f, indent=2)

    print(f"\n[benchmark] Wrote metrics to {version_dir / 'metrics.json'}")
    print(f"[benchmark] Complete: {len(per_group)} groups evaluated")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run annual LTR benchmark")
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--screen", action="store_true", help="4 groups (fast)")
    parser.add_argument("--eval-groups", nargs="+", default=None)
    parser.add_argument("--registry-dir", default="registry")
    args = parser.parse_args()

    if args.eval_groups:
        eval_groups = args.eval_groups
        mode = "custom"
    elif args.screen:
        eval_groups = SCREEN_EVAL_GROUPS
        mode = "screen"
    else:
        eval_groups = DEFAULT_EVAL_GROUPS
        mode = "eval"

    run_benchmark(
        version_id=args.version_id,
        eval_groups=eval_groups,
        registry_dir=args.registry_dir,
        mode=mode,
    )


if __name__ == "__main__":
    main()
