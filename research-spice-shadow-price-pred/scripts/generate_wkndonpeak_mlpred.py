"""
Generate PJM wkndonpeak ml_pred data to fill the gap for V7.0.

All computation runs on the Ray cluster. The head node only coordinates.
Each auction month is processed as a separate Ray task that writes
final_results.parquet directly to disk and returns only a status string.

Output: /opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/ml_pred/
        auction_month={A}/market_month={M}/class_type=wkndonpeak/final_results.parquet

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    PYTHONPATH=.../research-spice-shadow-price-pred/src python .../scripts/generate_wkndonpeak_mlpred.py
"""

import argparse
import os
import resource
import time
from collections import defaultdict

os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"

OUTPUT_DIR = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/ml_pred/"
CLASS_TYPE = "wkndonpeak"


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def build_auction_groups(config, start_month, end_month):
    """Build {auction_month: [market_months]} matching the PJM schedule."""
    import pandas as pd

    groups = defaultdict(list)
    for auction_month in pd.date_range(start_month, end_month, freq="MS"):
        market_st = auction_month
        market_et = market_st + pd.offsets.MonthBegin(12)
        if market_et.month > 5:
            market_et = market_et.replace(month=5)

        all_periods = config.iso.auction_schedule.get(auction_month.month, [])
        for market_month in pd.date_range(start=market_st, end=market_et, freq="MS"):
            f_period = f"f{(market_month.year - auction_month.year) * 12 + market_month.month - auction_month.month}"
            if f_period in all_periods:
                groups[auction_month].append(market_month)

    return dict(groups)


def count_existing(output_dir, class_type):
    """Count how many final_results.parquet files already exist for this class_type."""
    from pathlib import Path

    count = 0
    base = Path(output_dir)
    if base.exists():
        for f in base.rglob("final_results.parquet"):
            if f"class_type={class_type}" in str(f):
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Generate PJM wkndonpeak ml_pred")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    parser.add_argument("--max-concurrent", type=int, default=12, help="Max concurrent Ray tasks (default: 12)")
    parser.add_argument("--start", default="2018-06", help="First auction month (default: 2018-06)")
    parser.add_argument("--end", default="2026-01", help="Last auction month (default: 2026-01)")
    parser.add_argument("--refresh", action="store_true", help="Overwrite existing results")
    args = parser.parse_args()

    # Import after env var is set
    import ray
    from pbase.config.ray import init_ray
    import shadow_price_prediction

    init_ray(extra_modules=[shadow_price_prediction])

    import pandas as pd
    from pbase.utils.ray import ray_map_bounded
    from shadow_price_prediction.config import PredictionConfig
    from shadow_price_prediction.iso_configs import PJM_ISO_CONFIG, PJM_HORIZON_GROUPS

    # Configure — same config as onpeak/dailyoffpeak runs
    config = PredictionConfig()
    config.iso = PJM_ISO_CONFIG
    config.horizon_groups = PJM_HORIZON_GROUPS
    config.class_type = CLASS_TYPE

    # Build auction groups
    auction_groups = build_auction_groups(config, args.start, args.end)
    total_periods = sum(len(mms) for mms in auction_groups.values())
    existing_count = count_existing(OUTPUT_DIR, CLASS_TYPE)

    print(f"PJM wkndonpeak ml_pred generation (Ray remote)")
    print(f"  Auction months: {len(auction_groups)} ({args.start} to {args.end})")
    print(f"  Total test periods: {total_periods}")
    print(f"  Existing wkndonpeak files: {existing_count}")
    print(f"  Max concurrent tasks: {args.max_concurrent}")
    print(f"  Refresh: {args.refresh}")
    print(f"  Head memory: {mem_mb():.0f} MB")

    if args.dry_run:
        print("\n[DRY RUN] Would submit the following auction months:")
        for am in sorted(auction_groups)[:10]:
            print(f"  {am.strftime('%Y-%m')} ({len(auction_groups[am])} market months)")
        if len(auction_groups) > 10:
            print(f"  ... and {len(auction_groups) - 10} more")
        return

    # Put config in object store once — all workers share the same ref
    config_ref = ray.put(config)

    # Define the remote worker — all heavy work happens here, on the cluster
    @ray.remote(num_cpus=2, scheduling_strategy="SPREAD")
    def process_auction_month_remote(config_ref, auction_month, market_months, output_dir, refresh):
        """Run _process_auction_month on a Ray worker, return only a status string."""
        from shadow_price_prediction.pipeline import _process_auction_month

        config = config_ref  # Ray auto-resolves ObjectRef
        try:
            _process_auction_month(
                config=config,
                auction_month=auction_month,
                market_months=market_months,
                train_only=False,
                verbose=True,
                output_dir=output_dir,
                refresh=refresh,
                branch_name=None,
            )
            return f"OK: {auction_month.strftime('%Y-%m')} ({len(market_months)} periods)"
        except Exception as e:
            return f"FAIL: {auction_month.strftime('%Y-%m')}: {e}"

    # Build args list for ray_map_bounded
    args_list = [
        (config_ref, am, mms, OUTPUT_DIR, args.refresh)
        for am, mms in sorted(auction_groups.items())
    ]

    print(f"\nSubmitting {len(args_list)} auction months to Ray cluster...")
    t0 = time.time()

    try:
        results = ray_map_bounded(
            process_auction_month_remote,
            args_list,
            max_concurrent=args.max_concurrent,
        )
    except BaseException:
        print("\nInterrupted — Ray tasks cancelled.")
        raise

    elapsed = time.time() - t0

    # Report
    ok = [r for r in results if r and r.startswith("OK")]
    fail = [r for r in results if r and r.startswith("FAIL")]
    new_count = count_existing(OUTPUT_DIR, CLASS_TYPE)

    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Succeeded: {len(ok)}/{len(results)}")
    print(f"  Failed:    {len(fail)}/{len(results)}")
    print(f"  wkndonpeak files before: {existing_count}")
    print(f"  wkndonpeak files after:  {new_count}")
    print(f"  New files generated:     {new_count - existing_count}")
    print(f"  Head memory: {mem_mb():.0f} MB")

    if fail:
        print("\nFailed auction months:")
        for f in fail:
            print(f"  {f}")


if __name__ == "__main__":
    main()
