"""Run v0 and v1 baselines, populate gates, compare, and promote v0.

Steps:
  1. Run v0 baseline (14-feat clf, 34-feat reg) across 12 eval months
  2. Run v1 baseline (29-feat clf, 34-feat reg) across 12 eval months
  3. Populate gates from v0 metrics
  4. Run comparison (v0 vs v1)
  5. Promote v0 as initial champion

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    PYTHONPATH=.../research-stage2-shadow python scripts/run_baseline.py
"""
from __future__ import annotations

import gc
import os
import resource
import sys
import time

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.benchmark import run_benchmark
from ml.compare import run_comparison
from ml.config import ClassifierConfig, PipelineConfig
from ml.populate_v0_gates import populate_v0_gates
from ml.registry import promote_version


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


EVAL_MONTHS = [
    "2020-09", "2020-11", "2021-01", "2021-03", "2021-05", "2021-07",
    "2021-09", "2021-11", "2022-03", "2022-06", "2022-09", "2022-12",
]

REGISTRY_DIR = os.path.join(project_root, "registry")
GATES_PATH = os.path.join(project_root, "registry", "gates.json")
CHAMPION_PATH = os.path.join(project_root, "registry", "champion.json")


def main():
    t0 = time.time()
    print(f"[baseline] Starting baseline runs, mem: {mem_mb():.0f} MB")
    print(f"[baseline] eval_months: {EVAL_MONTHS}")

    # ── Step 1: Run v0 (14-feat classifier) ─────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1: v0 baseline (14-feature classifier)")
    print("=" * 60)

    v0_config = PipelineConfig(classifier=ClassifierConfig.v0())
    print(f"[baseline] v0 clf features: {len(v0_config.classifier.features)}")
    print(f"[baseline] v0 reg features: {len(v0_config.regressor.features)}")
    print(f"[baseline] train_months={v0_config.train_months}, val_months={v0_config.val_months}")

    v0_result = run_benchmark(
        version_id="v0",
        eval_months=EVAL_MONTHS,
        class_type="onpeak",
        period_type="f0",
        registry_dir=REGISTRY_DIR,
        config=v0_config,
    )

    t1 = time.time()
    print(f"\n[baseline] v0 complete: {v0_result['n_months']} months, "
          f"{(t1 - t0) / 60:.1f} min, mem: {mem_mb():.0f} MB")

    # Print v0 summary
    v0_mean = v0_result["aggregate"]["mean"]
    print("[baseline] v0 mean metrics:")
    for k in ["AUC", "AP", "EV-VC@100", "EV-VC@500", "EV-NDCG", "Spearman", "C-RMSE"]:
        if k in v0_mean:
            print(f"  {k}: {v0_mean[k]:.4f}")

    gc.collect()

    # ── Step 2: Run v1 (29-feat classifier) ─────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2: v1 baseline (29-feature classifier)")
    print("=" * 60)

    v1_config = PipelineConfig(classifier=ClassifierConfig.v1())
    print(f"[baseline] v1 clf features: {len(v1_config.classifier.features)}")
    print(f"[baseline] v1 reg features: {len(v1_config.regressor.features)}")

    v1_result = run_benchmark(
        version_id="v1",
        eval_months=EVAL_MONTHS,
        class_type="onpeak",
        period_type="f0",
        registry_dir=REGISTRY_DIR,
        config=v1_config,
    )

    t2 = time.time()
    print(f"\n[baseline] v1 complete: {v1_result['n_months']} months, "
          f"{(t2 - t1) / 60:.1f} min, mem: {mem_mb():.0f} MB")

    # Print v1 summary
    v1_mean = v1_result["aggregate"]["mean"]
    print("[baseline] v1 mean metrics:")
    for k in ["AUC", "AP", "EV-VC@100", "EV-VC@500", "EV-NDCG", "Spearman", "C-RMSE"]:
        if k in v1_mean:
            print(f"  {k}: {v1_mean[k]:.4f}")

    gc.collect()

    # ── Step 3: Populate gates from v0 ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3: Populate gates from v0 baseline")
    print("=" * 60)

    populate_v0_gates(registry_dir=REGISTRY_DIR, gates_path=GATES_PATH)

    # ── Step 4: Promote v0 as champion ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4: Promote v0 as initial champion")
    print("=" * 60)

    promote_version(REGISTRY_DIR, "v0", CHAMPION_PATH)
    print("[baseline] v0 promoted as champion")

    # ── Step 5: Run comparison ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 5: Compare v0 vs v1")
    print("=" * 60)

    report_path = os.path.join(project_root, "registry", "reports", "baseline_comparison.md")
    run_comparison(
        batch_id="baseline",
        iteration=0,
        registry_dir=REGISTRY_DIR,
        gates_path=GATES_PATH,
        champion_path=CHAMPION_PATH,
        output_path=report_path,
    )

    # ── Summary ─────────────────────────────────────────────────────────
    t3 = time.time()
    print("\n" + "=" * 60)
    print("  BASELINE COMPLETE")
    print("=" * 60)
    print(f"Total time: {(t3 - t0) / 60:.1f} min")
    print(f"Peak memory: {mem_mb():.0f} MB")
    print(f"Champion: v0")
    print(f"\nv0 vs v1 comparison:")

    for k in ["AUC", "AP", "EV-VC@100", "EV-VC@500", "EV-NDCG", "Spearman", "C-RMSE", "C-MAE"]:
        v0v = v0_mean.get(k, 0)
        v1v = v1_mean.get(k, 0)
        delta = v1v - v0v
        better = "v1" if (delta > 0 and k != "C-RMSE" and k != "C-MAE") or \
                         (delta < 0 and k in ("C-RMSE", "C-MAE")) else "v0"
        print(f"  {k:15s}  v0={v0v:.4f}  v1={v1v:.4f}  delta={delta:+.4f}  ({better})")

    print(f"\nReport: {report_path}")
    print(f"Gates: {GATES_PATH}")


if __name__ == "__main__":
    main()
