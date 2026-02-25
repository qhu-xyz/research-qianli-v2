"""Experiment Benchmark Runner — run current pipeline with config overrides.

Modes:
  --mode smoke    Run one period (2020-07/onpeak/f0) to validate wiring
  --mode full     Run all 32 test periods concurrently (8 months x 2 class x 2 period_types)
  --mode score    Score existing parquets (skip pipeline runs)
  --mode timing   Run 2 workers sequential vs parallel to measure speedup

Usage (from pmodel venv):
  cd /home/xyz/workspace/pmodel && source .venv/bin/activate

  # Smoke test with F2.0 threshold
  PYTHONPATH=/.../src:$PYTHONPATH python /.../scripts/run_experiment.py \
    --mode smoke --version-id v001-threshold-f2 \
    --overrides '{"threshold_beta": 2.0}'

  # Full benchmark (4 concurrent workers)
  PYTHONPATH=/.../src:$PYTHONPATH python /.../scripts/run_experiment.py \
    --mode full --version-id v001-threshold-f2 \
    --concurrency 4 --overrides '{"threshold_beta": 2.0}'

  # Score only
  PYTHONPATH=/.../src:$PYTHONPATH python /.../scripts/run_experiment.py \
    --mode score --version-id v001-threshold-f2

  # Timing test (sequential vs parallel comparison)
  PYTHONPATH=/.../src:$PYTHONPATH python /.../scripts/run_experiment.py \
    --mode timing --version-id timing-test
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURRENT_SRC = PROJECT_ROOT / "src"
BASE_OUTPUT_DIR = "/opt/temp/tmp/pw_data/spice6/experiments"
WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "_experiment_worker.py"
REGISTRY_DIR = PROJECT_ROOT / "versions"
PMODEL_VENV_PYTHON = "/home/xyz/workspace/pmodel/.venv/bin/python"

# ── Benchmark Scope ────────────────────────────────────────────────────
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
WORKER_TIMEOUT = 1800  # seconds (30 min) per worker subprocess


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ── Run helpers ────────────────────────────────────────────────────────

def run_worker(auction_month: str, class_type: str, period_type: str,
               output_dir: str, overrides_json: str) -> bool:
    """Run experiment worker as subprocess. Returns True on success."""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{CURRENT_SRC}:{env.get('PYTHONPATH', '')}"

    cmd = [
        PMODEL_VENV_PYTHON,
        str(WORKER_SCRIPT),
        "--auction-month", auction_month,
        "--class-type", class_type,
        "--period-type", period_type,
        "--output-dir", output_dir,
        "--config-overrides", overrides_json,
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


def _build_worker_cmd(auction_month: str, class_type: str, period_type: str,
                      output_dir: str, overrides_json: str) -> list[str]:
    """Build the subprocess command list for a single worker."""
    return [
        PMODEL_VENV_PYTHON,
        str(WORKER_SCRIPT),
        "--auction-month", auction_month,
        "--class-type", class_type,
        "--period-type", period_type,
        "--output-dir", output_dir,
        "--config-overrides", overrides_json,
    ]


def _interleave_runs(runs: list[tuple]) -> list[tuple]:
    """Reorder runs via round-robin across auction months.

    This spreads concurrent workers across different auction months so they
    don't all try to load the same month's data simultaneously.  Runs are
    grouped by (auction_month, class_type) and dealt out in rotation.

    Parameters
    ----------
    runs : list[tuple]
        Each element is (planning_year, auction_month, season, class_type, period_type).

    Returns
    -------
    list[tuple]
        Same elements, reordered.
    """
    from collections import OrderedDict

    buckets: OrderedDict[str, list] = OrderedDict()
    for run in runs:
        key = f"{run[1]}_{run[3]}"  # auction_month + class_type
        buckets.setdefault(key, []).append(run)

    interleaved = []
    bucket_iters = [iter(v) for v in buckets.values()]
    while bucket_iters:
        next_round_iters = []
        for it in bucket_iters:
            item = next(it, None)
            if item is not None:
                interleaved.append(item)
                next_round_iters.append(it)
        bucket_iters = next_round_iters

    return interleaved


def run_workers_concurrent(
    runs: list[tuple],
    output_dir: str,
    overrides_json: str,
    max_concurrent: int = 4,
) -> dict[str, list[str]]:
    """Launch experiment workers as concurrent subprocesses.

    Parameters
    ----------
    runs : list[tuple]
        Each element is (planning_year, auction_month, season, class_type, period_type).
    output_dir : str
        Directory for worker output (parquets + artifacts).
    overrides_json : str
        JSON string of config overrides passed to each worker.
    max_concurrent : int
        Maximum simultaneous worker subprocesses.

    Returns
    -------
    dict with keys "ok", "fail", "skip" — each a list of "{am}/{ct}/{pt}" tags.
    """
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{CURRENT_SRC}:{env.get('PYTHONPATH', '')}"

    results: dict[str, list[str]] = {"ok": [], "fail": [], "skip": []}

    # Filter out already-completed runs
    pending = []
    for run in runs:
        _py, am_str, _season, ct, pt = run
        tag = f"{am_str}/{ct}/{pt}"
        pq = expected_parquet(am_str, ct, pt, output_dir)
        if pq.exists():
            print(f"  [SKIP] {tag} (parquet exists)")
            results["skip"].append(tag)
        else:
            pending.append(run)

    if not pending:
        print("[INFO] All runs already completed.")
        return results

    total = len(pending)
    print(f"\n[{time.strftime('%H:%M:%S')}] Launching {total} workers (max_concurrent={max_concurrent})")

    # Track active workers: list of (Popen, log_fh, tag, start_time)
    active: list[tuple[subprocess.Popen, object, str, float]] = []
    pending_iter = iter(pending)
    launched = 0
    completed = 0

    try:
        while completed < total:
            # Launch workers up to max_concurrent
            while len(active) < max_concurrent:
                run = next(pending_iter, None)
                if run is None:
                    break
                _py, am_str, _season, ct, pt = run
                tag = f"{am_str}/{ct}/{pt}"

                log_path = Path(output_dir) / f"log_{am_str.replace('-', '')}_{ct}_{pt}.txt"
                log_fh = open(log_path, "w")

                cmd = _build_worker_cmd(am_str, ct, pt, output_dir, overrides_json)
                proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)

                active.append((proc, log_fh, tag, time.time()))
                launched += 1
                print(f"  [{time.strftime('%H:%M:%S')}] [{launched}/{total}] Started: {tag} (pid={proc.pid})")

            # Poll active workers
            still_active = []
            for proc, log_fh, tag, t0 in active:
                rc = proc.poll()
                if rc is not None:
                    # Process finished
                    elapsed = time.time() - t0
                    log_fh.close()
                    completed += 1
                    if rc == 0:
                        results["ok"].append(tag)
                        print(f"  [{time.strftime('%H:%M:%S')}] [{completed}/{total}] OK: {tag} ({elapsed:.0f}s)")
                    else:
                        results["fail"].append(tag)
                        print(f"  [{time.strftime('%H:%M:%S')}] [{completed}/{total}] FAIL: {tag} (rc={rc}, {elapsed:.0f}s)")
                elif time.time() - t0 > WORKER_TIMEOUT:
                    # Timeout — kill worker
                    proc.kill()
                    proc.wait()
                    log_fh.close()
                    completed += 1
                    results["fail"].append(tag)
                    print(f"  [{time.strftime('%H:%M:%S')}] [{completed}/{total}] TIMEOUT: {tag} (>{WORKER_TIMEOUT}s)")
                else:
                    still_active.append((proc, log_fh, tag, t0))

            active = still_active
            if active:
                time.sleep(2)

    except KeyboardInterrupt:
        print(f"\n[{time.strftime('%H:%M:%S')}] KeyboardInterrupt — terminating {len(active)} active workers...")
        for proc, log_fh, tag, _ in active:
            proc.terminate()
        # Give processes a moment to clean up
        for proc, log_fh, tag, _ in active:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log_fh.close()
            results["fail"].append(tag)
        print(f"[{time.strftime('%H:%M:%S')}] All workers terminated.")

    n_ok = len(results["ok"])
    n_fail = len(results["fail"])
    n_skip = len(results["skip"])
    print(f"\n{'='*70}")
    print(f"CONCURRENT RUN COMPLETE: {n_ok} ok, {n_fail} fail, {n_skip} skip (total={n_ok + n_fail + n_skip})")
    print(f"{'='*70}")
    if results["fail"]:
        print(f"  Failed: {results['fail']}")

    return results


# ── Scoring ────────────────────────────────────────────────────────────

def score_all_results(output_dir: str, version_id: str):
    """Load all parquets, score each, register in registry."""
    from shadow_price_prediction.evaluation import score_results_df, print_score_report
    from shadow_price_prediction.registry import ModelRegistry

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

    flat_rows = []
    for s in all_scores:
        m = s["_meta"]
        row = {
            "planning_year": m["planning_year"],
            "auction_month": m["auction_month"],
            "season": m["season"],
            "class_type": m["class_type"],
            "period_type": m["period_type"],
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
            "spearman_tp": s["stage2"]["spearman_tp"],
            "mae_tp": s["stage2"]["mae_tp"],
            "rmse_tp": s["stage2"]["rmse_tp"],
            "bias_tp": s["stage2"]["bias_tp"],
            "n_tp": s["stage2"]["n_tp"],
            "rmse_all": s["combined"]["rmse_all"],
            "mae_all": s["combined"]["mae_all"],
            "ndcg_outage": s["ranking_outage"].get("ndcg", float("nan")),
            "ndcg_constraint": s.get("ranking_constraint", {}).get("ndcg", float("nan")),
        }

        # Constraint-level @K metrics (all sub-metrics at each K)
        rc = s.get("ranking_constraint", {})
        rc_topk = rc.get("topk", {})
        row["n_constraints"] = rc.get("n_constraints", 0)
        row["n_binding_cst"] = rc.get("n_binding", 0)
        for k in (20, 50, 100, 200, 300, 500, 1000):
            kd = rc_topk.get(k, {})
            row[f"recall_cst_{k}"] = kd.get("recall", float("nan"))
            row[f"prec_cst_{k}"] = kd.get("precision", float("nan"))
            row[f"valcap_cst_{k}"] = kd.get("value_capture", float("nan"))
            row[f"lift_cst_{k}"] = kd.get("lift", float("nan"))
            row[f"meanval_cst_{k}"] = kd.get("mean_value", float("nan"))
            row[f"capture_cst_{k}"] = kd.get("capture", float("nan"))

        ro_topk = s.get("ranking_outage", {}).get("topk", {})
        for k in (100, 250, 500, 1000, 2000):
            row[f"valcap_out_{k}"] = ro_topk.get(k, {}).get("value_capture", float("nan"))
            row[f"lift_out_{k}"] = ro_topk.get(k, {}).get("lift", float("nan"))

        flat_rows.append(row)

    agg_df = pd.DataFrame(flat_rows)

    key_metrics = [
        "auc_roc", "avg_precision", "precision", "recall",
        "fbeta_0.5", "fbeta_2.0", "brier_score",
        "spearman_tp", "mae_tp", "rmse_tp", "bias_tp",
        "rmse_all", "ndcg_outage", "ndcg_constraint",
        "valcap_cst_1000",
    ]

    # @K metrics for constraint-level tracking
    CST_K = (20, 50, 100, 200, 300, 500, 1000)
    atk_metrics = {
        "recall":  [f"recall_cst_{k}"  for k in CST_K],
        "prec":    [f"prec_cst_{k}"    for k in CST_K],
        "valcap":  [f"valcap_cst_{k}"  for k in CST_K],
        "lift":    [f"lift_cst_{k}"    for k in CST_K],
        "meanval": [f"meanval_cst_{k}" for k in CST_K],
        "capture": [f"capture_cst_{k}" for k in CST_K],
    }

    def _agg_table(df, group_col, label):
        print(f"\n── {label} ──")
        grouped = df.groupby(group_col)[key_metrics].mean()
        with pd.option_context("display.float_format", "{:.4f}".format, "display.max_columns", 20, "display.width", 200):
            print(grouped.to_string())

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
        print(f"\n── Overall Mean ({ct}) ──")
        overall = ct_df[key_metrics].mean()
        for metric, val in overall.items():
            print(f"  {metric:25s} = {val:.4f}")

    # ── Constraint-level @K summary ──────────────────────────────────
    print(f"\n{'='*70}")
    print("CONSTRAINT-LEVEL @K METRICS  (median binding ≈ 300)")
    print(f"{'='*70}")
    for ct in CLASS_TYPES:
        ct_df = agg_df[agg_df["class_type"] == ct]
        if ct_df.empty:
            continue
        print(f"\n  {ct}:")
        avg_bind = ct_df["n_binding_cst"].mean()
        avg_cst = ct_df["n_constraints"].mean()
        print(f"    avg binding: {avg_bind:.0f} / {avg_cst:.0f} ({avg_bind/avg_cst:.1%})")
        header = f"    {'K':>5}  {'Recall@K':>9}  {'Prec@K':>8}  {'VC@K':>8}  {'Capture@K':>10}  {'Lift@K':>7}  {'MeanVal@K':>10}"
        print(header)
        print(f"    {'-----':>5}  {'---------':>9}  {'--------':>8}  {'--------':>8}  {'----------':>10}  {'-------':>7}  {'----------':>10}")
        for k in CST_K:
            rec = ct_df[f"recall_cst_{k}"].mean()
            prec = ct_df[f"prec_cst_{k}"].mean()
            vc = ct_df[f"valcap_cst_{k}"].mean()
            cap = ct_df[f"capture_cst_{k}"].mean()
            lift = ct_df[f"lift_cst_{k}"].mean()
            mv = ct_df[f"meanval_cst_{k}"].mean()
            print(f"    {k:>5}  {rec:>9.4f}  {prec:>8.4f}  {vc:>8.4f}  {cap:>10.4f}  {lift:>7.2f}  ${mv:>9,.0f}")

    # ── Gate table ─────────────────────────────────────────────────────
    def _compute_gates(df):
        """Compute gate values from a DataFrame slice."""
        return {
            "S1-AUC": round(float(df["auc_roc"].mean()), 4),
            "S1-REC": round(float(df["recall"].mean()), 4),
            "S1-PREC": round(float(df["precision"].mean()), 4),
            "S1-F1": round(float(df["f1"].mean()), 4),
            "S2-SPR": round(float(df["spearman_tp"].mean()), 4),
            "C-VC@100": round(float(df["valcap_cst_100"].mean()), 4),
            "C-VC@500": round(float(df["valcap_cst_500"].mean()), 4),
            "C-VC@1000": round(float(df["valcap_cst_1000"].mean()), 4),
            "C-CAP@20": round(float(df["capture_cst_20"].mean()), 4),
            "C-CAP@200": round(float(df["capture_cst_200"].mean()), 4),
            "C-CAP@1000": round(float(df["capture_cst_1000"].mean()), 4),
            "C-RMSE": round(float(df["rmse_all"].mean()), 2),
        }

    print(f"\n{'='*70}")
    print("HARD GATE VALUES  (overall = across period types)")
    print(f"{'='*70}")

    gate_table = {}
    for ct in CLASS_TYPES:
        ct_df = agg_df[agg_df["class_type"] == ct]
        if ct_df.empty:
            continue
        gate_table[ct] = _compute_gates(ct_df)
        print(f"\n  {ct}:")
        for gate, val in gate_table[ct].items():
            print(f"    {gate:15s} = {val}")

    # ── Per-period_type gate table ────────────────────────────────────
    print(f"\n{'='*70}")
    print("GATE VALUES BY PERIOD TYPE  (f0 = current month, f1 = next month)")
    print(f"{'='*70}")

    gate_table_by_period = {}
    for ct in CLASS_TYPES:
        for pt in PERIOD_TYPES:
            slice_df = agg_df[(agg_df["class_type"] == ct) & (agg_df["period_type"] == pt)]
            if slice_df.empty:
                continue
            key = f"{ct}/{pt}"
            gate_table_by_period[key] = _compute_gates(slice_df)
            print(f"\n  {key}:")
            for gate, val in gate_table_by_period[key].items():
                print(f"    {gate:15s} = {val}")

    # ── Register in registry ──────────────────────────────────────────
    reg = ModelRegistry(str(REGISTRY_DIR))

    try:
        ver = reg.get_version(version_id)
    except FileNotFoundError:
        print(f"\n[WARN] Version {version_id} not found in registry. Skipping registration.")
        return gate_table

    per_run_scores = [
        {
            **s["_meta"],
            "stage1": s["stage1"],
            "stage2": s["stage2"],
            "combined": s["combined"],
            "ranking_outage": _sanitize_topk(s.get("ranking_outage", {})),
            "ranking_constraint": _sanitize_topk(s.get("ranking_constraint", {})),
        }
        for s in all_scores
    ]

    ver.record_metrics(
        gate_values=gate_table,
        gate_values_by_period=gate_table_by_period,
        benchmark_scope={
            "planning_years": list(BENCHMARK_MONTHS.keys()),
            "months": {py: [m for m, _ in months] for py, months in BENCHMARK_MONTHS.items()},
            "class_types": CLASS_TYPES,
            "period_types": PERIOD_TYPES,
            "n_scored": len(all_scores),
            "n_missing": len(missing),
        },
        per_run_scores=per_run_scores,
    )
    print(f"\n[REGISTRY] Recorded metrics for {version_id}")

    # ── Aggregate and record new artifacts ─────────────────────────────
    _aggregate_and_record_artifacts(ver, output_dir)

    # Compute and store version hash over all artifacts
    version_hash = ver.compute_version_hash()
    print(f"[REGISTRY] Version hash: {version_hash}")

    # Check gates
    gate_result = reg.check_gates(version_id)
    print(f"\n{gate_result.summary_table()}")

    # Save aggregated CSV for this version
    csv_path = Path(output_dir) / f"{version_id}_agg.csv"
    agg_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"[SAVED] {csv_path}")

    return gate_table


def _aggregate_and_record_artifacts(ver, output_dir: str) -> None:
    """Aggregate per-run worker artifacts and register in the version."""
    from shadow_price_prediction.naming import (
        worker_threshold_path,
        worker_feature_importance_path,
        worker_train_manifest_path,
    )

    threshold_agg = {}
    fi_agg = {"stage1": {}, "stage2": {}}
    train_manifest_agg = {}

    for py, months in BENCHMARK_MONTHS.items():
        for am_str, season in months:
            for ct in CLASS_TYPES:
                key = f"{am_str.replace('-', '')}_{ct}"

                # Thresholds
                thr_path = worker_threshold_path(output_dir, am_str, ct)
                if thr_path.exists():
                    with open(thr_path) as f:
                        data = json.load(f)
                    threshold_agg[key] = data

                # Feature importances
                fi_path = worker_feature_importance_path(output_dir, am_str, ct)
                if fi_path.exists():
                    with open(fi_path) as f:
                        data = json.load(f)
                    # Merge into aggregated structure
                    for stage in ("stage1", "stage2"):
                        for group_name, group_data in data.get(stage, {}).items():
                            if group_name not in fi_agg[stage]:
                                fi_agg[stage][group_name] = {}
                            fi_agg[stage][group_name][key] = group_data

                # Train manifest
                tm_path = worker_train_manifest_path(output_dir, am_str, ct)
                if tm_path.exists():
                    with open(tm_path) as f:
                        data = json.load(f)
                    train_manifest_agg[key] = data

    n_thr = len(threshold_agg)
    n_fi = sum(len(v) for v in fi_agg.values())
    n_tm = len(train_manifest_agg)

    if threshold_agg:
        ver.record_threshold_manifest(threshold_agg)
        print(f"[REGISTRY] Recorded threshold manifest ({n_thr} entries)")
    else:
        # Save empty to satisfy artifact contract
        ver.record_threshold_manifest({})
        print("[REGISTRY] No threshold artifacts found — saved empty manifest")

    if any(fi_agg[s] for s in ("stage1", "stage2")):
        ver.record_feature_importance(fi_agg)
        print(f"[REGISTRY] Recorded feature importances ({n_fi} entries)")
    else:
        ver.record_feature_importance({"stage1": {}, "stage2": {}})
        print("[REGISTRY] No feature importance artifacts found — saved empty manifest")

    if train_manifest_agg:
        ver.record_train_manifest(train_manifest_agg)
        print(f"[REGISTRY] Recorded train manifest ({n_tm} entries)")
    else:
        ver.record_train_manifest({})
        print("[REGISTRY] No train manifest artifacts found — saved empty manifest")


def _sanitize_topk(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if k == "topk" and isinstance(v, dict):
            out["topk"] = {str(kk): vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experiment Benchmark Runner")
    parser.add_argument("--mode", choices=["smoke", "full", "score", "timing"], default="smoke")
    parser.add_argument("--version-id", required=True, help="Model version ID, e.g. v001-xgb-20260220-001")
    parser.add_argument("--overrides", default="{}", help="JSON string of config overrides")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: BASE_OUTPUT_DIR/version_id)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max simultaneous worker subprocesses for --mode full (default: 4)")
    args = parser.parse_args()

    version_id = args.version_id
    overrides_json = args.overrides
    output_dir = args.output_dir or str(Path(BASE_OUTPUT_DIR) / version_id)

    print(f"[{time.strftime('%H:%M:%S')}] Experiment Benchmark — mode={args.mode}")
    print(f"  Version: {version_id}")
    print(f"  Output: {output_dir}")
    print(f"  Overrides: {overrides_json}")
    if args.mode == "full":
        print(f"  Concurrency: {args.concurrency}")
    print(f"  mem={mem_mb():.0f} MB")

    if args.mode == "smoke":
        ok = run_worker("2020-07", "onpeak", "f0", output_dir, overrides_json)
        if ok:
            print(f"\n[{time.strftime('%H:%M:%S')}] Smoke test passed. Scoring...")
            from shadow_price_prediction.evaluation import score_results_df, print_score_report
            pq = expected_parquet("2020-07", "onpeak", "f0", output_dir)
            if pq.exists():
                df = pd.read_parquet(pq)
                scores = score_results_df(df)
                print_score_report(scores, label=f"SMOKE TEST: 2020-07/onpeak/f0 [{version_id}]")
            else:
                print(f"[ERROR] Parquet not found: {pq}")
        else:
            print("[FAIL] Smoke test failed.")
            sys.exit(1)

    elif args.mode == "full":
        runs = _interleave_runs(list_all_runs())
        t_start = time.time()

        results = run_workers_concurrent(
            runs, output_dir, overrides_json, max_concurrent=args.concurrency,
        )

        elapsed = time.time() - t_start
        n_ok = len(results["ok"]) + len(results["skip"])
        n_fail = len(results["fail"])
        print(f"\nBENCHMARK COMPLETE: {n_ok}/{n_ok + n_fail} OK, {n_fail} FAIL ({elapsed:.0f}s wall)")

        if n_fail > 0:
            print(f"  Failed: {results['fail']}")

        print(f"\n[{time.strftime('%H:%M:%S')}] Scoring all results...")
        score_all_results(output_dir, version_id)

    elif args.mode == "score":
        score_all_results(output_dir, version_id)

    elif args.mode == "timing":
        import shutil

        timing_runs = [
            ("PY20", "2020-07", "Summer", "onpeak", "f0"),
            ("PY20", "2020-07", "Summer", "offpeak", "f0"),
        ]

        seq_dir = str(Path(output_dir) / "_timing_seq")
        par_dir = str(Path(output_dir) / "_timing_par")

        # Clean up any prior timing dirs
        for d in (seq_dir, par_dir):
            if os.path.exists(d):
                shutil.rmtree(d)

        print(f"\n{'='*60}")
        print("TIMING TEST: sequential vs parallel")
        print(f"{'='*60}")

        # Sequential: run 2 workers one at a time
        print(f"\n[SEQ] Running {len(timing_runs)} workers sequentially...")
        t0 = time.time()
        for _py, am_str, _season, ct, pt in timing_runs:
            run_worker(am_str, ct, pt, seq_dir, overrides_json)
        t_seq = time.time() - t0

        # Parallel: run same 2 workers concurrently
        print(f"\n[PAR] Running {len(timing_runs)} workers in parallel (concurrency=2)...")
        t0 = time.time()
        run_workers_concurrent(timing_runs, par_dir, overrides_json, max_concurrent=2)
        t_par = time.time() - t0

        speedup = t_seq / t_par if t_par > 0 else float("inf")

        print(f"\n{'='*60}")
        print("TIMING TEST RESULTS")
        print(f"{'='*60}")
        print(f"  Sequential: {len(timing_runs)} workers in {t_seq:.0f}s")
        print(f"  Parallel:   {len(timing_runs)} workers in {t_par:.0f}s (concurrency=2)")
        print(f"  Speedup:    {speedup:.1f}x")
        print(f"{'='*60}")

        # Clean up timing dirs
        for d in (seq_dir, par_dir):
            if os.path.exists(d):
                shutil.rmtree(d)
        print("[CLEANUP] Timing directories removed.")

    print(f"\n[{time.strftime('%H:%M:%S')}] Done. mem={mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
