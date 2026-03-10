"""Legacy pipeline worker — runs ONE (auction_month, class_type, period_type).

Invoked as a subprocess by run_legacy_baseline.py with _legacy/src on PYTHONPATH
so that ``import shadow_price_prediction`` resolves to the original unmodified code.

Usage (from pmodel venv):
  PYTHONPATH=/path/to/_legacy/src:$PYTHONPATH python _legacy_worker.py \
      --auction-month 2020-07 --class-type onpeak --period-type f0 \
      --output-dir /tmp/legacy_baseline
"""

import argparse
import gc
import resource
import sys
import time

import pandas as pd


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    parser = argparse.ArgumentParser(description="Legacy pipeline single-run worker")
    parser.add_argument("--auction-month", required=True, help="e.g. 2020-07")
    parser.add_argument("--class-type", required=True, choices=["onpeak", "offpeak"])
    parser.add_argument("--period-type", required=True, choices=["f0", "f1"])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    am = pd.Timestamp(args.auction_month)
    class_type = args.class_type
    period_type = args.period_type
    output_dir = args.output_dir

    # Market month depends on period type
    if period_type == "f0":
        mm = am
    elif period_type == "f1":
        mm = am + pd.DateOffset(months=1)
    else:
        raise ValueError(f"Unsupported period_type: {period_type}")

    tag = f"{am.strftime('%Y-%m')}/{class_type}/{period_type}"
    print(f"[{time.strftime('%H:%M:%S')}] Worker starting: {tag}  mem={mem_mb():.0f} MB")

    # --- Ray init ---
    from pbase.config.ray import init_ray
    import pmodel
    import shadow_price_prediction

    init_ray(address="ray://10.8.0.36:10001", extra_modules=[pmodel, shadow_price_prediction])
    print(f"[{time.strftime('%H:%M:%S')}] Ray initialized. mem={mem_mb():.0f} MB")

    # --- Import legacy pipeline ---
    from shadow_price_prediction import ShadowPricePipeline, PredictionConfig

    # Use ALL defaults (the whole point of "legacy baseline")
    config = PredictionConfig(
        market_round=1,
        period_type=period_type,
        class_type=class_type,
    )

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
            # Save results_per_outage to parquet
            import os
            os.makedirs(output_dir, exist_ok=True)
            fname = f"results_{am.strftime('%Y%m')}_{class_type}_{period_type}.parquet"
            out_path = os.path.join(output_dir, fname)
            results_per_outage.to_parquet(out_path)
            n_rows = len(results_per_outage)
            n_binding = int((results_per_outage["actual_binding"] == 1).sum())
            print(f"\n[SAVED] {out_path} ({n_rows:,} rows, {n_binding:,} binding, {elapsed:.0f}s)")
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


if __name__ == "__main__":
    main()
