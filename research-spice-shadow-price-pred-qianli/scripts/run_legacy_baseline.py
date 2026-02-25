"""Legacy Baseline Benchmark — run unmodified pipeline, score comprehensively.

Three modes:
  --mode smoke    Run one period (2020-07/onpeak/f0) to validate wiring
  --mode full     Run all 32 test periods (8 months × 2 class × 2 period_types)
  --mode score    Score existing parquets (skip pipeline runs)

Usage (from pmodel venv):
  cd /home/xyz/workspace/pmodel && source .venv/bin/activate

  # Smoke test
  PYTHONPATH=/.../research-spice-shadow-price-pred-qianli/src:$PYTHONPATH \
    python /.../scripts/run_legacy_baseline.py --mode smoke

  # Full benchmark
  PYTHONPATH=/.../research-spice-shadow-price-pred-qianli/src:$PYTHONPATH \
    python /.../scripts/run_legacy_baseline.py --mode full

  # Score only (after runs are done)
  PYTHONPATH=/.../research-spice-shadow-price-pred-qianli/src:$PYTHONPATH \
    python /.../scripts/run_legacy_baseline.py --mode score
"""

import argparse
import json
import gc
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_SRC = PROJECT_ROOT / "_legacy" / "src"
CURRENT_SRC = PROJECT_ROOT / "src"
DEFAULT_OUTPUT_DIR = "/opt/temp/tmp/pw_data/spice6/legacy_baseline"
WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "_legacy_worker.py"
REGISTRY_DIR = PROJECT_ROOT / "versions"
PMODEL_VENV_PYTHON = "/home/xyz/workspace/pmodel/.venv/bin/python"

# ── Benchmark Scope ────────────────────────────────────────────────────
# PY20: Jun 2020 - May 2021, PY21: Jun 2021 - May 2022
# Quarterly sampling: Summer, Fall, Winter, Spring

BENCHMARK_MONTHS = {
    "PY20": [
        ("2020-07", "Summer"),
        ("2020-10", "Fall"),
        ("2021-01", "Winter"),
        ("2021-04", "Spring"),
    ],
    "PY21": [
        ("2021-07", "Summer"),
        ("2021-10", "Fall"),
        ("2022-01", "Winter"),
        ("2022-04", "Spring"),
    ],
}

CLASS_TYPES = ["onpeak", "offpeak"]
PERIOD_TYPES = ["f0", "f1"]


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ── Run helpers ────────────────────────────────────────────────────────

def run_worker(auction_month: str, class_type: str, period_type: str, output_dir: str) -> bool:
    """Run legacy worker as subprocess. Returns True on success."""
    env = os.environ.copy()
    # CRITICAL: legacy code on PYTHONPATH first, so `import shadow_price_prediction`
    # resolves to the unmodified version
    env["PYTHONPATH"] = f"{LEGACY_SRC}:{env.get('PYTHONPATH', '')}"

    cmd = [
        PMODEL_VENV_PYTHON,
        str(WORKER_SCRIPT),
        "--auction-month", auction_month,
        "--class-type", class_type,
        "--period-type", period_type,
        "--output-dir", output_dir,
    ]

    tag = f"{auction_month}/{class_type}/{period_type}"
    print(f"\n{'─'*60}")
    print(f"[{time.strftime('%H:%M:%S')}] Launching: {tag}")
    print(f"{'─'*60}")

    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=False, timeout=1800)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"[{time.strftime('%H:%M:%S')}] OK: {tag} ({elapsed:.0f}s)")
        return True
    else:
        print(f"[{time.strftime('%H:%M:%S')}] FAIL: {tag} (rc={result.returncode}, {elapsed:.0f}s)")
        return False


def list_all_runs():
    """Generate all (py, auction_month, season, class_type, period_type) tuples."""
    runs = []
    for py, months in BENCHMARK_MONTHS.items():
        for am_str, season in months:
            for ct in CLASS_TYPES:
                for pt in PERIOD_TYPES:
                    runs.append((py, am_str, season, ct, pt))
    return runs


def expected_parquet(auction_month: str, class_type: str, period_type: str, output_dir: str) -> Path:
    am_compact = auction_month.replace("-", "")
    return Path(output_dir) / f"results_{am_compact}_{class_type}_{period_type}.parquet"


# ── Scoring ────────────────────────────────────────────────────────────

def score_all_results(output_dir: str):
    """Load all parquets, score each, produce aggregation tables."""
    from shadow_price_prediction.evaluation import score_results_df, print_score_report

    runs = list_all_runs()
    all_scores = []
    missing = []

    for py, am_str, season, ct, pt in runs:
        pq_path = expected_parquet(am_str, ct, pt, output_dir)
        if not pq_path.exists():
            missing.append(f"{am_str}/{ct}/{pt}")
            continue

        df = pd.read_parquet(pq_path)
        scores = score_results_df(df)
        scores["_meta"] = {
            "planning_year": py,
            "auction_month": am_str,
            "season": season,
            "class_type": ct,
            "period_type": pt,
            "parquet_path": str(pq_path),
        }
        all_scores.append(scores)

        print_score_report(scores, label=f"{py} | {am_str} | {ct} | {pt} | {season}")

    if missing:
        print(f"\n[WARN] Missing {len(missing)} parquets: {missing}")

    if not all_scores:
        print("[ERROR] No scored results to aggregate.")
        return None

    # ── Aggregation ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("AGGREGATION TABLES")
    print(f"{'='*70}")

    # Build flat rows for aggregation
    flat_rows = []
    for s in all_scores:
        m = s["_meta"]
        row = {
            "planning_year": m["planning_year"],
            "auction_month": m["auction_month"],
            "season": m["season"],
            "class_type": m["class_type"],
            "period_type": m["period_type"],
            # Stage 1
            "auc_roc": s["stage1"]["auc_roc"],
            "avg_precision": s["stage1"]["avg_precision"],
            "brier_score": s["stage1"]["brier_score"],
            "precision": s["stage1"]["precision"],
            "recall": s["stage1"]["recall"],
            "f1": s["stage1"]["f1"],
            "fbeta_0.5": s["stage1"]["fbeta_0.5"],
            "fbeta_2.0": s["stage1"]["fbeta_2.0"],
            "binding_rate": s["stage1"]["binding_rate"],
            "pred_binding_rate": s["stage1"]["pred_binding_rate"],
            # Stage 2
            "spearman_tp": s["stage2"]["spearman_tp"],
            "mae_tp": s["stage2"]["mae_tp"],
            "rmse_tp": s["stage2"]["rmse_tp"],
            "bias_tp": s["stage2"]["bias_tp"],
            "n_tp": s["stage2"]["n_tp"],
            # Combined
            "rmse_all": s["combined"]["rmse_all"],
            "mae_all": s["combined"]["mae_all"],
            # Ranking — outage
            "ndcg_outage": s["ranking_outage"].get("ndcg", float("nan")),
            # Ranking — constraint
            "ndcg_constraint": s.get("ranking_constraint", {}).get("ndcg", float("nan")),
        }

        # ValCap@K from constraint ranking
        rc_topk = s.get("ranking_constraint", {}).get("topk", {})
        for k in (50, 100, 260, 520, 1000):
            vc = rc_topk.get(k, {}).get("value_capture", float("nan"))
            row[f"valcap_cst_{k}"] = vc

        # ValCap@K from outage ranking
        ro_topk = s.get("ranking_outage", {}).get("topk", {})
        for k in (100, 250, 500, 1000, 2000):
            vc = ro_topk.get(k, {}).get("value_capture", float("nan"))
            row[f"valcap_out_{k}"] = vc
            lift = ro_topk.get(k, {}).get("lift", float("nan"))
            row[f"lift_out_{k}"] = lift

        flat_rows.append(row)

    agg_df = pd.DataFrame(flat_rows)

    # Key metrics for summaries
    key_metrics = [
        "auc_roc", "avg_precision", "precision", "recall",
        "fbeta_0.5", "fbeta_2.0", "brier_score",
        "spearman_tp", "mae_tp", "rmse_tp", "bias_tp",
        "rmse_all", "ndcg_outage", "ndcg_constraint",
        "valcap_cst_1000",
    ]

    def _agg_table(df, group_col, label):
        """Print mean metrics grouped by a column."""
        print(f"\n── {label} ──")
        grouped = df.groupby(group_col)[key_metrics].mean()
        with pd.option_context("display.float_format", "{:.4f}".format, "display.max_columns", 20, "display.width", 200):
            print(grouped.to_string())
        return grouped

    # Per-class_type
    for ct in CLASS_TYPES:
        ct_df = agg_df[agg_df["class_type"] == ct]
        if ct_df.empty:
            continue

        print(f"\n{'='*70}")
        print(f"  CLASS TYPE: {ct.upper()}")
        print(f"{'='*70}")

        _agg_table(ct_df, "planning_year", f"Per Planning Year ({ct})")
        _agg_table(ct_df, "period_type", f"Per Horizon ({ct})")
        _agg_table(ct_df, "season", f"Per Season ({ct})")

        # Overall mean for this class type
        print(f"\n── Overall Mean ({ct}) ──")
        overall = ct_df[key_metrics].mean()
        for metric, val in overall.items():
            print(f"  {metric:25s} = {val:.4f}")

    # ── Gate table ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("HARD GATE BASELINE VALUES")
    print(f"{'='*70}")

    gate_table = {}
    for ct in CLASS_TYPES:
        ct_df = agg_df[agg_df["class_type"] == ct]
        if ct_df.empty:
            continue
        gate_table[ct] = {
            "S1-AUC": round(float(ct_df["auc_roc"].mean()), 4),
            "S1-REC": round(float(ct_df["recall"].mean()), 4),
            "S2-SPR": round(float(ct_df["spearman_tp"].mean()), 4),
            "C-VC@1000": round(float(ct_df["valcap_cst_1000"].mean()), 4),
            "C-RMSE": round(float(ct_df["rmse_all"].mean()), 2),
        }
        print(f"\n  {ct}:")
        for gate, val in gate_table[ct].items():
            print(f"    {gate:15s} = {val}")

    # ── Save to JSON ───────────────────────────────────────────────────
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    baseline_path = REGISTRY_DIR / "legacy_baseline.json"

    baseline_json = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "description": "Legacy unmodified pipeline baseline (defaults from commit b32bf6b)",
        "benchmark_scope": {
            "planning_years": list(BENCHMARK_MONTHS.keys()),
            "months": {py: [m for m, _ in months] for py, months in BENCHMARK_MONTHS.items()},
            "class_types": CLASS_TYPES,
            "period_types": PERIOD_TYPES,
            "n_scored": len(all_scores),
            "n_missing": len(missing),
        },
        "gate_values": gate_table,
        "per_run_scores": [
            {
                **s["_meta"],
                "stage1": s["stage1"],
                "stage2": s["stage2"],
                "combined": s["combined"],
                "ranking_outage": _sanitize_topk(s.get("ranking_outage", {})),
                "ranking_constraint": _sanitize_topk(s.get("ranking_constraint", {})),
            }
            for s in all_scores
        ],
    }

    with open(baseline_path, "w") as f:
        json.dump(baseline_json, f, indent=2, default=_json_default)
    print(f"\n[SAVED] {baseline_path}")

    # Save aggregated CSV for quick reference
    csv_path = REGISTRY_DIR / "legacy_baseline_agg.csv"
    agg_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"[SAVED] {csv_path}")

    return baseline_json


def _sanitize_topk(d: dict) -> dict:
    """Convert integer keys in topk to strings for JSON."""
    out = {}
    for k, v in d.items():
        if k == "topk" and isinstance(v, dict):
            out["topk"] = {str(kk): vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def _json_default(obj):
    """JSON fallback for numpy/pandas types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return repr(obj)


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Legacy Baseline Benchmark")
    parser.add_argument(
        "--mode",
        choices=["smoke", "full", "score"],
        default="smoke",
        help="smoke = 1 run, full = all 32, score = score existing parquets",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for parquets (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    output_dir = args.output_dir

    print(f"[{time.strftime('%H:%M:%S')}] Legacy Baseline Benchmark — mode={args.mode}")
    print(f"  Output: {output_dir}")
    print(f"  Legacy src: {LEGACY_SRC}")
    print(f"  mem={mem_mb():.0f} MB")

    if args.mode == "smoke":
        # Single period: PY20 Summer, onpeak, f0
        ok = run_worker("2020-07", "onpeak", "f0", output_dir)
        if ok:
            print(f"\n[{time.strftime('%H:%M:%S')}] Smoke test passed. Scoring...")
            from shadow_price_prediction.evaluation import score_results_df, print_score_report
            pq = expected_parquet("2020-07", "onpeak", "f0", output_dir)
            if pq.exists():
                df = pd.read_parquet(pq)
                scores = score_results_df(df)
                print_score_report(scores, label="SMOKE TEST: 2020-07 / onpeak / f0")
            else:
                print(f"[ERROR] Parquet not found: {pq}")
        else:
            print("[FAIL] Smoke test failed.")
            sys.exit(1)

    elif args.mode == "full":
        runs = list_all_runs()
        total = len(runs)
        results = {"ok": [], "fail": []}

        for i, (py, am_str, season, ct, pt) in enumerate(runs, 1):
            # Check if parquet already exists (resume support)
            pq = expected_parquet(am_str, ct, pt, output_dir)
            if pq.exists():
                print(f"\n[{i}/{total}] SKIP (exists): {am_str}/{ct}/{pt}")
                results["ok"].append(f"{am_str}/{ct}/{pt}")
                continue

            print(f"\n[{i}/{total}] Running: {py} {am_str} {season} {ct} {pt}")
            ok = run_worker(am_str, ct, pt, output_dir)
            if ok:
                results["ok"].append(f"{am_str}/{ct}/{pt}")
            else:
                results["fail"].append(f"{am_str}/{ct}/{pt}")

            # Brief pause to let Ray cluster recover
            gc.collect()
            time.sleep(2)

        print(f"\n{'='*70}")
        print(f"BENCHMARK COMPLETE: {len(results['ok'])}/{total} OK, {len(results['fail'])} FAIL")
        print(f"{'='*70}")

        if results["fail"]:
            print(f"  Failed: {results['fail']}")

        # Score all available results
        print(f"\n[{time.strftime('%H:%M:%S')}] Scoring all results...")
        score_all_results(output_dir)

    elif args.mode == "score":
        score_all_results(output_dir)

    print(f"\n[{time.strftime('%H:%M:%S')}] Done. mem={mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
