"""Compare v0 vs v0007 configs across all 36 months of 2020-2022.

Runs both configs in parallel (two processes) for time efficiency.
Each process runs all months sequentially using Ray for data loading.

Usage:
    source /home/xyz/workspace/pmodel/.venv/bin/activate
    cd /home/xyz/workspace/research-qianli-v2/research-stage2-shadow
    PYTHONPATH=. python scripts/compare_v0_v7.py
"""

import json
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


ALL_MONTHS = [f"{y}-{m:02d}" for y in [2020, 2021, 2022] for m in range(1, 13)]

# v0 config: reg_lambda=1.0, mcw=10, reg_alpha=0.1 (original baseline)
V0_OVERRIDES = {"regressor": {"min_child_weight": 10, "reg_alpha": 0.1}}

# v0007 config: reg_lambda=1.0, mcw=25, reg_alpha=1.0 (current committed defaults)
V7_OVERRIDES = None  # no overrides needed, defaults ARE v0007


def run_config(label: str, version_id: str, overrides: dict | None) -> dict:
    """Run benchmark for a single config across all months. Runs in subprocess."""
    os.environ.setdefault("RAY_ADDRESS", "ray://10.8.0.36:10001")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from ml.benchmark import run_benchmark

    print(f"[{label}] Starting {len(ALL_MONTHS)} months, mem: {mem_mb():.0f} MB")
    t0 = time.time()

    result = run_benchmark(
        version_id=version_id,
        eval_months=ALL_MONTHS,
        class_type="onpeak",
        period_type="f0",
        overrides=overrides,
    )

    elapsed = time.time() - t0
    n = result.get("n_months", 0)
    skipped = result.get("skipped_months", [])
    agg = result.get("aggregate", {}).get("mean", {})
    print(f"\n[{label}] Done: {n}/{len(ALL_MONTHS)} months in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    if skipped:
        print(f"[{label}] Skipped: {skipped}")
    print(f"[{label}] EV-VC@100={agg.get('EV-VC@100', '?'):.4f}  "
          f"EV-VC@500={agg.get('EV-VC@500', '?'):.4f}  "
          f"EV-NDCG={agg.get('EV-NDCG', '?'):.4f}  "
          f"Spearman={agg.get('Spearman', '?'):.4f}  "
          f"C-RMSE={agg.get('C-RMSE', '?'):.0f}")

    return {"label": label, "result": result, "elapsed": elapsed}


def print_comparison(v0_result: dict, v7_result: dict):
    """Print per-month and aggregate comparison table."""
    v0_pm = v0_result["result"]["per_month"]
    v7_pm = v7_result["result"]["per_month"]
    v0_agg = v0_result["result"]["aggregate"]["mean"]
    v7_agg = v7_result["result"]["aggregate"]["mean"]

    all_months_done = sorted(set(v0_pm.keys()) & set(v7_pm.keys()))

    metrics = ["EV-VC@100", "EV-VC@500", "EV-NDCG", "Spearman", "C-RMSE", "C-MAE"]
    higher_better = {"EV-VC@100": True, "EV-VC@500": True, "EV-NDCG": True,
                     "Spearman": True, "C-RMSE": False, "C-MAE": False}

    print("\n" + "=" * 120)
    print("  MONTH-BY-MONTH COMPARISON: v0 vs v0007 (2020-2022)")
    print("=" * 120)

    # Header
    hdr = f"{'Month':>8}"
    for m in metrics:
        hdr += f" | {m + ' v0':>12} {m + ' v7':>12} {'Delta':>8}"
    print(hdr)
    print("-" * 120)

    # Per-month detail for key metrics only
    wins = {m: 0 for m in metrics}
    for month in all_months_done:
        row = f"{month:>8}"
        for m in metrics:
            val0 = v0_pm[month].get(m, float("nan"))
            val7 = v7_pm[month].get(m, float("nan"))
            if m in ("C-RMSE", "C-MAE"):
                delta = val0 - val7  # lower is better, positive delta = v7 wins
                pct = f"+{delta:.0f}" if delta > 0 else f"{delta:.0f}"
                if delta > 0:
                    wins[m] += 1
                row += f" | {val0:>12.1f} {val7:>12.1f} {pct:>8}"
            else:
                delta = val7 - val0  # higher is better
                pct = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
                if delta > 0:
                    wins[m] += 1
                row += f" | {val0:>12.4f} {val7:>12.4f} {pct:>8}"
        print(row)

    n = len(all_months_done)
    print("-" * 120)

    # Win counts
    row = f"{'v7 wins':>8}"
    for m in metrics:
        w = wins[m]
        row += f" | {'':>12} {'':>12} {f'{w}/{n}':>8}"
    print(row)

    # Aggregate
    print("\n" + "=" * 80)
    print("  AGGREGATE COMPARISON")
    print("=" * 80)
    print(f"  {'Metric':<15} {'v0 mean':>12} {'v0007 mean':>12} {'Delta':>10} {'Delta%':>8} {'Better':>8}")
    print("-" * 80)
    for m in metrics:
        val0 = v0_agg.get(m, float("nan"))
        val7 = v7_agg.get(m, float("nan"))
        if higher_better.get(m, True):
            delta = val7 - val0
            better = "v0007" if delta > 0 else "v0"
        else:
            delta = val0 - val7
            better = "v0007" if delta > 0 else "v0"
        pct = (delta / abs(val0) * 100) if val0 != 0 else 0
        if m in ("C-RMSE", "C-MAE"):
            print(f"  {m:<15} {val0:>12.1f} {val7:>12.1f} {delta:>+10.1f} {pct:>+7.1f}% {better:>8}")
        else:
            print(f"  {m:<15} {val0:>12.4f} {val7:>12.4f} {delta:>+10.4f} {pct:>+7.1f}% {better:>8}")
    print("=" * 80)
    print(f"\n  Months evaluated: {n} (of {len(ALL_MONTHS)} requested)")
    print(f"  v0 skipped: {v0_result['result'].get('skipped_months', [])}")
    print(f"  v7 skipped: {v7_result['result'].get('skipped_months', [])}")
    print(f"  v0 time: {v0_result['elapsed']:.0f}s  v7 time: {v7_result['elapsed']:.0f}s")

    # Save comparison JSON
    comparison = {
        "months_evaluated": all_months_done,
        "n_months": n,
        "v0_aggregate": v0_agg,
        "v7_aggregate": v7_agg,
        "v0_per_month": v0_pm,
        "v7_per_month": v7_pm,
        "wins_v7": wins,
    }
    out_path = Path("registry/comparisons/v0_vs_v0007_full_2020_2022.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Saved: {out_path}")


def main():
    t0 = time.time()
    print(f"Starting parallel comparison: v0 vs v0007 across {len(ALL_MONTHS)} months")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Run both configs in parallel using ProcessPoolExecutor
    # max_workers=2: one process per config
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(run_config, "v0", "v0-reeval", V0_OVERRIDES): "v0",
            pool.submit(run_config, "v0007", "v0007-reeval", V7_OVERRIDES): "v0007",
        }

        results = {}
        for fut in as_completed(futures):
            label = futures[fut]
            try:
                results[label] = fut.result()
            except Exception as e:
                print(f"ERROR: {label} failed: {e}")
                import traceback
                traceback.print_exc()

    if "v0" in results and "v0007" in results:
        print_comparison(results["v0"], results["v0007"])
    else:
        print("ERROR: one or both configs failed")
        for label, r in results.items():
            print(f"  {label}: {r.get('elapsed', '?')}s, {r['result'].get('n_months', '?')} months")

    print(f"\nTotal wall time: {time.time() - t0:.0f}s ({(time.time() - t0)/60:.1f}m)")


if __name__ == "__main__":
    main()
