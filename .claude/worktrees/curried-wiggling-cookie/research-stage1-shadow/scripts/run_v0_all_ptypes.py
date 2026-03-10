"""Run v0 baseline benchmarks for all cascade ptypes and record timing."""
import json
import time
import sys
import os

# Must run from project root with venv activated
sys.path.insert(0, os.getcwd())
os.environ.setdefault("PYTHONPATH", os.getcwd())

from ml.benchmark import run_benchmark

EVAL_MONTHS_PRIMARY = [
    "2020-09", "2020-11", "2021-01", "2021-04", "2021-06", "2021-08",
    "2021-10", "2021-12", "2022-03", "2022-06", "2022-09", "2022-12",
]

# MISO auction schedule: which months support which ptypes
# June (6) and May (5) only have f0
# Jan (1), Apr (4), Jun (6), Jul (7), Oct (10) don't have f2
PTYPE_EXCLUDES = {
    "f0": set(),  # all months have f0
    "f1": {"06"},  # June doesn't have f1
    "f2": {"01", "04", "06", "07", "10"},  # these months lack f2
}

PTYPES = ["f1", "f2"]  # f0 already done
timing = {}

for ptype in PTYPES:
    excludes = PTYPE_EXCLUDES.get(ptype, set())
    eval_months = [m for m in EVAL_MONTHS_PRIMARY if m.split("-")[1] not in excludes]
    version_id = f"v0-{ptype}"

    print(f"\n{'='*60}")
    print(f"Running v0 benchmark for ptype={ptype}")
    print(f"Eval months ({len(eval_months)}): {eval_months}")
    print(f"Excluded month-numbers: {excludes}")
    print(f"Version ID: {version_id}")
    print(f"{'='*60}\n", flush=True)

    t0 = time.time()
    try:
        result = run_benchmark(
            version_id=version_id,
            eval_months=eval_months,
            class_type="onpeak",
            ptype=ptype,
            registry_dir="registry",
        )
        elapsed = time.time() - t0
        n_evaluated = result.get("n_months", len(eval_months))
        n_skipped = len(result.get("skipped_months", []))
        timing[ptype] = {
            "elapsed_seconds": round(elapsed, 1),
            "elapsed_minutes": round(elapsed / 60, 1),
            "n_months_requested": len(eval_months),
            "n_months_evaluated": n_evaluated,
            "n_months_skipped": n_skipped,
            "skipped_months": result.get("skipped_months", []),
            "status": "success",
        }
        print(f"\n[timing] ptype={ptype}: {elapsed:.1f}s ({elapsed/60:.1f} min) for {n_evaluated}/{len(eval_months)} months ({n_skipped} skipped)", flush=True)

        # Print key aggregates
        agg = result.get("aggregate", {}).get("mean", result.get("aggregate", {}))
        for k in ["S1-AUC", "S1-AP", "S1-NDCG", "S1-BRIER"]:
            if k in agg:
                print(f"  {k}: {agg[k]:.4f}", flush=True)

    except Exception as e:
        elapsed = time.time() - t0
        timing[ptype] = {
            "elapsed_seconds": round(elapsed, 1),
            "status": f"failed: {e}",
        }
        print(f"\n[timing] ptype={ptype}: FAILED after {elapsed:.1f}s: {e}", flush=True)
        import traceback
        traceback.print_exc()

# Save timing summary
timing_path = "registry/v0_benchmark_timing.json"
# Include f0 timing from previous run (approximate)
timing["f0"] = {"elapsed_minutes": "~35 (previous run)", "n_months": 12, "status": "success"}
with open(timing_path, "w") as f:
    json.dump(timing, f, indent=2)
print(f"\n{'='*60}")
print(f"Timing summary saved to {timing_path}")
print(json.dumps(timing, indent=2))
