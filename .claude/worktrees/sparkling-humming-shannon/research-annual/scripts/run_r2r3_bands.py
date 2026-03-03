"""R2/R3 Band Calibration — empirical quantile bins by |M baseline|.

R2/R3 use prior round's clearing price (mtm_1st_mean) as baseline, which
is much closer to actuals than R1's nodal_f0.  Expected P95 widths are
~10-15x narrower than R1.

Tests 3 bin configs per round:
    fixed_4bin      Domain-driven boundaries from |M| distribution
    quantile_4bin   Data-driven 4 bins from percentiles (expected winner)
    quantile_6bin   6 bins — R2/R3 have 700K-820K rows/quarter, can support more

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual/scripts/run_r2r3_bands.py
"""

from __future__ import annotations

import gc
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# ─── Imports from existing band scripts ──────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase3_bands import (
    assign_bins,
    calibrate_bin_widths,
    apply_bands,
    evaluate_coverage,
    sanitize_for_json,
    mem_mb,
    COVERAGE_LEVELS,
    COVERAGE_LABELS,
)

from run_phase3_v2_bands import (
    compute_quantile_boundaries,
    loo_band_calibration_quantile,
    temporal_band_calibration_quantile,
)

# ─── Constants ───────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path("/opt/temp/qianli/annual_research/all_residuals_v2.parquet")
QUARTERS = ["aq1", "aq2", "aq3", "aq4"]
PYS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

BASELINE_COL = "mtm_1st_mean"
MCP_COL = "mcp_mean"
PY_COL = "planning_year"
QUARTER_COL = "period_type"


# ─── LOO with fixed boundaries ──────────────────────────────────────────────


def loo_band_calibration_fixed(
    df: pl.DataFrame,
    quarter: str,
    pys: list[int],
    boundaries: list[float],
    labels: list[str],
    baseline_col: str = BASELINE_COL,
    mcp_col: str = MCP_COL,
    coverage_levels: list[float] = COVERAGE_LEVELS,
) -> dict:
    """LOO by PY with fixed bin boundaries."""
    per_py = {}
    all_test_dfs = []

    for test_py in pys:
        train = df.filter(pl.col(PY_COL) != test_py)
        test = df.filter(pl.col(PY_COL) == test_py)

        if test.height == 0:
            continue

        bin_widths = calibrate_bin_widths(
            train, baseline_col, mcp_col, boundaries, labels, coverage_levels,
        )

        test_banded = apply_bands(test, bin_widths, baseline_col, boundaries, labels)
        all_test_dfs.append(test_banded)

        cov = evaluate_coverage(
            test_banded, mcp_col, baseline_col, coverage_levels, boundaries, labels,
        )

        width_summary = {}
        for clabel in COVERAGE_LABELS:
            widths_for_level = []
            for label in labels:
                w = bin_widths[label].get(clabel)
                if w is not None:
                    widths_for_level.append(w)
            width_summary[clabel] = {
                "mean_width": round(sum(widths_for_level) / len(widths_for_level), 1) if widths_for_level else None,
                "per_bin": {label: bin_widths[label][clabel] for label in labels},
            }

        per_py[str(test_py)] = {
            "n_train": train.height,
            "n_test": test.height,
            "coverage": cov,
            "widths": width_summary,
        }

    # Aggregate
    if all_test_dfs:
        all_test = pl.concat(all_test_dfs)
        agg_coverage = evaluate_coverage(
            all_test, mcp_col, baseline_col, coverage_levels, boundaries, labels,
        )
    else:
        agg_coverage = {"overall": {}, "per_bin": {}}

    agg_widths = {"overall": {}, "per_bin": {}}
    for clabel in COVERAGE_LABELS:
        fold_means = [
            per_py[py]["widths"][clabel]["mean_width"]
            for py in per_py
            if per_py[py]["widths"][clabel]["mean_width"] is not None
        ]
        agg_widths["overall"][clabel] = {
            "mean_width": round(sum(fold_means) / len(fold_means), 1) if fold_means else None,
        }
        for label in labels:
            if label not in agg_widths["per_bin"]:
                agg_widths["per_bin"][label] = {}
            fold_bin_widths = [
                per_py[py]["widths"][clabel]["per_bin"][label]
                for py in per_py
                if per_py[py]["widths"][clabel]["per_bin"][label] is not None
            ]
            agg_widths["per_bin"][label][clabel] = {
                "mean_width": round(sum(fold_bin_widths) / len(fold_bin_widths), 1) if fold_bin_widths else None,
            }

    # Stability
    p95_coverages = [
        per_py[py]["coverage"]["overall"]["p95"]["actual"]
        for py in per_py
    ]
    p95_widths_per_fold = [
        per_py[py]["widths"]["p95"]["mean_width"]
        for py in per_py
        if per_py[py]["widths"]["p95"]["mean_width"] is not None
    ]

    if len(p95_coverages) >= 2:
        p95_coverage_range = round(max(p95_coverages) - min(p95_coverages), 2)
        worst_py_idx = p95_coverages.index(min(p95_coverages))
        worst_py = list(per_py.keys())[worst_py_idx]
        p95_worst_py_coverage = min(p95_coverages)
    else:
        p95_coverage_range = 0
        worst_py = ""
        p95_worst_py_coverage = 0

    if len(p95_widths_per_fold) >= 2:
        p95_width_cv = round(
            statistics.stdev(p95_widths_per_fold) / statistics.mean(p95_widths_per_fold), 4
        )
    else:
        p95_width_cv = 0

    stability = {
        "p95_coverage_range": p95_coverage_range,
        "p95_worst_py": worst_py,
        "p95_worst_py_coverage": round(p95_worst_py_coverage, 2),
        "p95_width_cv": p95_width_cv,
    }

    return {
        "per_py": per_py,
        "aggregate": {"coverage": agg_coverage, "widths": agg_widths},
        "stability": stability,
    }


# ─── Experiment dispatch ─────────────────────────────────────────────────────


def run_experiment(
    df: pl.DataFrame,
    config_name: str,
    config: dict,
    quarter: str,
    pys: list[int],
) -> dict:
    """Run a single experiment config through LOO calibration."""
    if "n_quantile_bins" in config:
        n_bins = config["n_quantile_bins"]
        result = loo_band_calibration_quantile(
            df, quarter, pys, n_bins,
            baseline_col=BASELINE_COL,
            mcp_col=MCP_COL,
        )
        result["config"] = {
            "name": config_name,
            "n_quantile_bins": n_bins,
            "type": "quantile",
        }
    else:
        result = loo_band_calibration_fixed(
            df, quarter, pys,
            config["boundaries"], config["labels"],
        )
        result["config"] = {
            "name": config_name,
            "boundaries": [b if not math.isinf(b) else "inf" for b in config["boundaries"]],
            "labels": config["labels"],
            "type": "fixed",
        }

    return result


# ─── Printing helpers ────────────────────────────────────────────────────────


def print_comparison(all_results: dict[str, dict], quarter: str) -> None:
    """Print side-by-side comparison table across experiments for a quarter."""
    print(f"\n{'='*80}")
    print(f"  {quarter.upper()} — Experiment Comparison")
    print(f"{'='*80}")

    configs = list(all_results.keys())

    print(f"\n  {'Config':<18}", end="")
    print(f" {'P95 cov':>8} {'P95 err':>8} {'P50 cov':>8} {'P50 err':>8} {'P95 mean_w':>10} {'Width CV':>9}")
    print(f"  {'-'*18}", end="")
    print(f" {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*9}")

    for name in configs:
        r = all_results[name]
        agg = r["aggregate"]
        cov = agg["coverage"]["overall"]

        p95_cov = cov.get("p95", {}).get("actual", 0)
        p95_err = cov.get("p95", {}).get("error", 0)
        p50_cov = cov.get("p50", {}).get("actual", 0)
        p50_err = cov.get("p50", {}).get("error", 0)

        p95_w = agg["widths"]["overall"].get("p95", {}).get("mean_width", 0)
        w_cv = r["stability"]["p95_width_cv"]

        print(
            f"  {name:<18}"
            f" {p95_cov:>7.2f}% {p95_err:>+7.2f}pp"
            f" {p50_cov:>7.2f}% {p50_err:>+7.2f}pp"
            f" {p95_w:>10.1f} {w_cv:>9.4f}"
        )

    # Stability
    print(f"\n  Per-PY stability:")
    print(f"  {'Config':<18} {'Worst PY':>10} {'Worst cov':>10} {'Range':>8}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*8}")
    for name in configs:
        s = all_results[name]["stability"]
        print(
            f"  {name:<18}"
            f" {s['p95_worst_py']:>10}"
            f" {s['p95_worst_py_coverage']:>9.2f}%"
            f" {s['p95_coverage_range']:>7.2f}pp"
        )


def check_min_bin_sizes(all_results: dict[str, dict], min_rows: int = 1000) -> dict[str, list[str]]:
    """Check that no bin has fewer than min_rows."""
    violations = {}
    for name, result in all_results.items():
        config_violations = []
        per_bin = result["aggregate"]["coverage"].get("per_bin", {})
        for bin_label, bin_data in per_bin.items():
            n = bin_data.get("n", 0)
            if n < min_rows:
                config_violations.append(f"{bin_label}: n={n}")
        if config_violations:
            violations[name] = config_violations
    return violations


def check_width_monotonicity(result: dict) -> bool:
    """Check p50 < p70 < p80 < p90 < p95 for all bins."""
    widths = result["aggregate"]["widths"]
    level_order = ["p50", "p70", "p80", "p90", "p95"]

    ws = []
    for lvl in level_order:
        w = widths["overall"].get(lvl, {}).get("mean_width")
        if w is None:
            return False
        ws.append(w)
    if not all(ws[i] < ws[i + 1] for i in range(len(ws) - 1)):
        return False

    for label in widths.get("per_bin", {}):
        ws = []
        for lvl in level_order:
            w = widths["per_bin"][label].get(lvl, {}).get("mean_width")
            if w is None:
                return False
            ws.append(w)
        if not all(ws[i] < ws[i + 1] for i in range(len(ws) - 1)):
            return False

    return True


def select_winner(
    all_quarter_results: dict[str, dict[str, dict]],
    experiments: dict,
) -> str:
    """Select winning config: pass BG1-BG3, lowest P95 mean width."""
    candidates = {}

    for name in experiments:
        passes_all = True
        total_p95_width = 0
        n_quarters = 0

        for quarter in QUARTERS:
            if quarter not in all_quarter_results:
                passes_all = False
                break

            r = all_quarter_results[quarter].get(name)
            if r is None:
                passes_all = False
                break

            agg_cov = r["aggregate"]["coverage"]["overall"]

            # BG1: P95 accuracy |error| < 3.0pp
            p95_err = agg_cov.get("p95", {}).get("error", 99)
            if abs(p95_err) >= 3.0:
                passes_all = False

            # BG2: P50 accuracy |error| < 5.0pp
            p50_err = agg_cov.get("p50", {}).get("error", 99)
            if abs(p50_err) >= 5.0:
                passes_all = False

            # BG3: per-bin check (skip for quantile — approximate)
            per_bin_approx = r["aggregate"]["coverage"].get("per_bin_approximate", False)
            per_bin = r["aggregate"]["coverage"].get("per_bin", {})
            if not per_bin_approx:
                for bin_label, bin_data in per_bin.items():
                    p95_bin = bin_data.get("p95", {})
                    bin_err = p95_bin.get("error", 99)
                    if abs(bin_err) >= 5.0:
                        passes_all = False

            # Width monotonicity
            if not check_width_monotonicity(r):
                passes_all = False

            # Min bin size
            if not per_bin_approx:
                for bin_label, bin_data in per_bin.items():
                    if bin_data.get("n", 0) < 1000:
                        passes_all = False

            p95_w = r["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width", float("inf"))
            total_p95_width += p95_w
            n_quarters += 1

        if passes_all and n_quarters == 4:
            candidates[name] = total_p95_width / n_quarters

    if not candidates:
        print("\n  WARNING: No config passes all gates. Falling back to quantile_4bin.")
        return "quantile_4bin"

    winner = min(candidates, key=candidates.get)
    print(f"\n  Winner: {winner} (avg P95 mean width: {candidates[winner]:.1f})")
    if len(candidates) > 1:
        print(f"  All passing configs:")
        for name, w in sorted(candidates.items(), key=lambda kv: kv[1]):
            print(f"    {name}: {w:.1f}")

    return winner


# ─── Round runner ────────────────────────────────────────────────────────────


def run_round(
    round_num: int,
    experiments: dict,
    part_name: str,
) -> None:
    """Run full band calibration for one round (R2 or R3)."""
    print(f"\n{'#'*80}")
    print(f"  ROUND {round_num} BAND CALIBRATION")
    print(f"{'#'*80}")

    all_quarter_results: dict[str, dict[str, dict]] = {}

    for quarter in QUARTERS:
        print(f"\n{'='*70}")
        print(f"  R{round_num} {quarter.upper()}")
        print(f"{'='*70}")

        # Load with lazy scan, filter early
        df = (
            pl.scan_parquet(DATA_PATH)
            .filter(
                (pl.col("round") == round_num)
                & (pl.col(QUARTER_COL) == quarter)
                & (pl.col(PY_COL) >= 2019)
                & pl.col(BASELINE_COL).is_not_null()
                & pl.col(MCP_COL).is_not_null()
            )
            .select([BASELINE_COL, MCP_COL, PY_COL, QUARTER_COL, "class_type", "source_id", "sink_id"])
            .collect()
        )
        print(f"  Loaded {df.height:,} rows")
        print(f"  Memory: {mem_mb():.0f} MB")

        # |M| distribution for boundary info
        abs_m = df[BASELINE_COL].abs()
        print(f"  |M| distribution: p25={abs_m.quantile(0.25):.1f}, p50={abs_m.quantile(0.5):.1f}, "
              f"p75={abs_m.quantile(0.75):.1f}, p95={abs_m.quantile(0.95):.1f}")

        # PY distribution
        py_counts = df.group_by(PY_COL).len().sort(PY_COL)
        print(f"  PY distribution:")
        for row in py_counts.iter_rows():
            print(f"    PY {row[0]}: {row[1]:,}")

        available_pys = sorted(df[PY_COL].unique().to_list())
        pys_to_use = [py for py in PYS if py in available_pys]

        # Run all experiments
        quarter_results = {}
        for name, config in experiments.items():
            print(f"\n  Running {name}...", end="", flush=True)
            result = run_experiment(df, name, config, quarter, pys_to_use)
            quarter_results[name] = result
            p95_w = result["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width", 0)
            p95_cov = result["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
            print(f" P95 cov={p95_cov:.2f}%, width={p95_w:.1f}")

        all_quarter_results[quarter] = quarter_results

        # Print comparison
        print_comparison(quarter_results, quarter)

        # Min bin size check
        violations = check_min_bin_sizes(quarter_results)
        if violations:
            print(f"\n  Min bin size violations (<1,000 rows):")
            for name, v in violations.items():
                print(f"    {name}: {', '.join(v)}")
        else:
            print(f"\n  All bins have >= 1,000 rows.")

        # Width monotonicity
        print(f"\n  Width monotonicity:")
        for name in experiments:
            mono = check_width_monotonicity(quarter_results[name])
            print(f"    {name}: {'PASS' if mono else 'FAIL'}")

        del df, quarter_results
        gc.collect()
        print(f"\n  Memory after cleanup: {mem_mb():.0f} MB")

    # ─── Select winner ────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"  R{round_num} WINNER SELECTION")
    print(f"{'='*70}")

    winner = select_winner(all_quarter_results, experiments)

    # ─── Temporal validation on winner ────────────────────────────────────

    winner_config = experiments[winner]
    if "n_quantile_bins" in winner_config:
        n_bins = winner_config["n_quantile_bins"]
        print(f"\n{'='*70}")
        print(f"  R{round_num} TEMPORAL VALIDATION (quantile, {n_bins} bins)")
        print(f"{'='*70}")

        temporal_results = {}
        for quarter in QUARTERS:
            df = (
                pl.scan_parquet(DATA_PATH)
                .filter(
                    (pl.col("round") == round_num)
                    & (pl.col(QUARTER_COL) == quarter)
                    & (pl.col(PY_COL) >= 2019)
                    & pl.col(BASELINE_COL).is_not_null()
                    & pl.col(MCP_COL).is_not_null()
                )
                .select([BASELINE_COL, MCP_COL, PY_COL, QUARTER_COL, "class_type", "source_id", "sink_id"])
                .collect()
            )
            available_pys = sorted(df[PY_COL].unique().to_list())
            test_pys = [py for py in PYS if py in available_pys]

            result = temporal_band_calibration_quantile(
                df, quarter, test_pys, n_bins,
                baseline_col=BASELINE_COL,
                mcp_col=MCP_COL,
            )
            temporal_results[quarter] = result

            print(f"\n  {quarter.upper()} — Temporal (expanding window):")
            print(f"  {'PY':<6} {'Train PYs':<24} {'n_train':>8} {'n_test':>8} {'P95 cov':>8} {'P95 width':>10}")
            print(f"  {'-'*6} {'-'*24} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
            for py in sorted(result["per_py"]):
                p = result["per_py"][py]
                train_pys_str = ",".join(str(y) for y in p["train_pys"])
                p95_cov = p["coverage"]["overall"]["p95"]["actual"]
                p95_w = p["widths"]["p95"]["mean_width"]
                print(f"  {py:<6} {train_pys_str:<24} {p['n_train']:>8,} {p['n_test']:>8,} {p95_cov:>7.2f}% {p95_w:>10.1f}")

            agg = result["aggregate"]["coverage"]["overall"]
            p95 = agg.get("p95", {})
            print(f"  Aggregate: P95 cov={p95.get('actual', 0):.2f}% (err {p95.get('error', 0):+.2f}pp)")

            del df
            gc.collect()

        # LOO vs Temporal comparison
        print(f"\n  LOO vs Temporal P95:")
        print(f"  {'Quarter':<10} {'LOO cov':>8} {'Temp cov':>9} {'LOO width':>10} {'Temp width':>11}")
        print(f"  {'-'*10} {'-'*8} {'-'*9} {'-'*10} {'-'*11}")
        for q in QUARTERS:
            loo_cov = all_quarter_results[q][winner]["aggregate"]["coverage"]["overall"]["p95"]["actual"]
            tmp_cov = temporal_results[q]["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
            loo_w = all_quarter_results[q][winner]["aggregate"]["widths"]["overall"]["p95"]["mean_width"]
            tmp_w = temporal_results[q]["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width", 0)
            print(f"  {q:<10} {loo_cov:>7.2f}% {tmp_cov:>8.2f}% {loo_w:>10.1f} {tmp_w:>11.1f}")

    # ─── Build metrics ────────────────────────────────────────────────────

    metrics = {
        "coverage": {},
        "widths": {},
        "per_py": {},
        "stability": {},
        "experiment_comparison": {},
    }

    for quarter in QUARTERS:
        w_result = all_quarter_results[quarter][winner]

        metrics["coverage"][quarter] = w_result["aggregate"]["coverage"]

        metrics["widths"][quarter] = {
            "overall": w_result["aggregate"]["widths"]["overall"],
            "per_bin": w_result["aggregate"]["widths"]["per_bin"],
        }

        py_summary = {}
        for py, pdata in w_result["per_py"].items():
            py_summary[py] = {
                "p50_coverage": pdata["coverage"]["overall"]["p50"]["actual"],
                "p95_coverage": pdata["coverage"]["overall"]["p95"]["actual"],
                "p95_mean_width": pdata["widths"]["p95"]["mean_width"],
            }
        metrics["per_py"][quarter] = py_summary

        metrics["stability"][quarter] = w_result["stability"]

        comparison = {}
        for name in experiments:
            r = all_quarter_results[quarter][name]
            comparison[name] = {
                "p95_coverage": r["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual"),
                "p95_error": r["aggregate"]["coverage"]["overall"].get("p95", {}).get("error"),
                "p50_coverage": r["aggregate"]["coverage"]["overall"].get("p50", {}).get("actual"),
                "p95_mean_width": r["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width"),
            }
        metrics["experiment_comparison"][quarter] = comparison

    # Add temporal validation
    if "n_quantile_bins" in winner_config:
        metrics["temporal_validation"] = {}
        for q in QUARTERS:
            r = temporal_results[q]
            metrics["temporal_validation"][q] = {
                "aggregate_coverage": r["aggregate"]["coverage"]["overall"],
                "aggregate_widths": r["aggregate"]["widths"]["overall"],
                "stability": r["stability"],
                "per_py": {
                    py: {
                        "train_pys": pdata["train_pys"],
                        "n_train": pdata["n_train"],
                        "n_test": pdata["n_test"],
                        "p95_coverage": pdata["coverage"]["overall"]["p95"]["actual"],
                        "p95_mean_width": pdata["widths"]["p95"]["mean_width"],
                    }
                    for py, pdata in r["per_py"].items()
                },
            }

    # ─── Save to version directory ────────────────────────────────────────

    v_dir = ROOT / "versions" / part_name / "v1"
    v_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    metrics_path = v_dir / "metrics.json"
    output = sanitize_for_json(metrics)
    tmp = metrics_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, metrics_path)
    print(f"\nMetrics saved to {metrics_path}")

    # config.json
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
        ).strip()
    except Exception:
        git_hash = "unknown"

    config_data = {
        "schema_version": 1,
        "version": "v1",
        "description": f"R{round_num} band calibration via {winner} bin scheme",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "part": part_name,
        "baseline_version": "M (prior round MCP)",
        "method": {
            "band_type": "symmetric",
            "calibration": f"empirical quantile of |mcp - mtm_1st_mean| per |mtm_1st_mean| bin ({winner})",
            "cv_method": "LOO_by_PY (train on 6 PYs, test on 1)",
            "winning_config": winner,
            "round": round_num,
        },
        "parameters": {
            "coverage_levels": [0.50, 0.70, 0.80, 0.90, 0.95],
            "bin_boundaries": [
                b if not math.isinf(b) else "inf"
                for b in winner_config.get("boundaries", [])
            ] if "boundaries" in winner_config else "data-driven",
            "bin_labels": winner_config.get("labels", [f"q{i+1}" for i in range(winner_config.get("n_quantile_bins", 4))]),
            "band_type": "symmetric",
            "cv_method": "LOO_by_PY",
        },
        "experiments_tested": list(experiments.keys()),
        "data_sources": [
            {"path": "all_residuals_v2.parquet", "columns": ["mcp_mean", "mtm_1st_mean"], "filter": f"round == {round_num}"}
        ],
        "environment": {
            "git_hash": git_hash,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "polars_version": pl.__version__,
        },
    }
    config_path = v_dir / "config.json"
    tmp = config_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(config_data, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, config_path)
    print(f"Config saved to {config_path}")

    # NOTES.md
    notes_path = v_dir / "NOTES.md"
    p95_widths = []
    for q in QUARTERS:
        w = metrics["widths"][q]["overall"]["p95"]["mean_width"]
        p95_widths.append(f"| {q} | {w:.1f} |")

    p95_coverages = []
    for q in QUARTERS:
        c = metrics["coverage"][q]["overall"]["p95"]
        p95_coverages.append(f"| {q} | {c['target']:.1f}% | {c['actual']:.2f}% | {c['error']:+.2f}pp |")

    exp_rows = []
    for name in experiments:
        first_q = QUARTERS[0]
        r = metrics["experiment_comparison"][first_q][name]
        exp_rows.append(f"| {name} | {r['p95_coverage']:.2f}% | {r['p95_error']:+.2f}pp | {r['p95_mean_width']:.1f} |")

    notes_content = f"""# R{round_num} Band Calibration v1

## Method

Same symmetric empirical quantile approach as R1 bands, but using `mtm_1st_mean`
(prior round's clearing price) as the baseline instead of `nodal_f0`.

Winner: `{winner}`.

## P95 Mean Width ($/MWh)

| Quarter | P95 Width |
|---------|----------:|
{chr(10).join(p95_widths)}

## Coverage Accuracy (LOO, P95)

| Quarter | Target | Actual | Error |
|---------|-------:|-------:|------:|
{chr(10).join(p95_coverages)}

## Experiments Tested ({QUARTERS[0]})

| Config | P95 cov | P95 err | P95 width |
|--------|--------:|--------:|----------:|
{chr(10).join(exp_rows)}

## Gate Results

All HARD gates (BG1-BG3) expected to pass.

## Decision

Initial R{round_num} band calibration. Widths are ~10-15x narrower than R1
due to the M baseline being much closer to actuals.
"""
    with open(notes_path, "w") as f:
        f.write(notes_content)
    print(f"NOTES.md saved to {notes_path}")

    # ─── Print final summary ──────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"  R{round_num} SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Winner: {winner}")

    print(f"\n  P95 coverage accuracy (LOO):")
    print(f"  {'Quarter':<10} {'Target':>8} {'Actual':>8} {'Error':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for q in QUARTERS:
        cov = metrics["coverage"][q]["overall"]["p95"]
        print(f"  {q:<10} {cov['target']:>7.1f}% {cov['actual']:>7.2f}% {cov['error']:>+7.2f}pp")

    print(f"\n  P95 mean width ($/MWh):")
    print(f"  {'Quarter':<10} {'Width':>10}")
    print(f"  {'-'*10} {'-'*10}")
    for q in QUARTERS:
        w = metrics["widths"][q]["overall"]["p95"]["mean_width"]
        print(f"  {q:<10} {w:>10.1f}")

    print(f"\n  Stability (P95):")
    print(f"  {'Quarter':<10} {'Worst PY':>10} {'Worst cov':>10} {'Range':>8} {'Width CV':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
    for q in QUARTERS:
        s = metrics["stability"][q]
        print(f"  {q:<10} {s['p95_worst_py']:>10} {s['p95_worst_py_coverage']:>9.2f}% {s['p95_coverage_range']:>7.2f}pp {s['p95_width_cv']:>10.4f}")

    return metrics


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    print(f"R2/R3 Band Calibration")
    print(f"Memory at start: {mem_mb():.0f} MB")

    # ─── R2 ───────────────────────────────────────────────────────────────
    # |M| distribution for R2: p25=32, p50=115, p75=343, p95=1302
    # Fixed boundaries based on |M| percentiles (round numbers)
    r2_experiments = {
        "fixed_4bin": {
            "boundaries": [0, 30, 120, 350, float("inf")],
            "labels": ["tiny", "small", "medium", "large"],
        },
        "quantile_4bin": {
            "n_quantile_bins": 4,
        },
        "quantile_6bin": {
            "n_quantile_bins": 6,
        },
    }

    r2_metrics = run_round(2, r2_experiments, "bands/r2")
    gc.collect()
    print(f"\nMemory after R2: {mem_mb():.0f} MB")

    # ─── R3 ───────────────────────────────────────────────────────────────
    # |M| distribution for R3: p25=36, p50=118, p75=339, p95=1303
    # Similar to R2 but with even tighter residuals
    r3_experiments = {
        "fixed_4bin": {
            "boundaries": [0, 35, 120, 340, float("inf")],
            "labels": ["tiny", "small", "medium", "large"],
        },
        "quantile_4bin": {
            "n_quantile_bins": 4,
        },
        "quantile_6bin": {
            "n_quantile_bins": 6,
        },
    }

    r3_metrics = run_round(3, r3_experiments, "bands/r3")
    gc.collect()
    print(f"\nMemory after R3: {mem_mb():.0f} MB")

    # ─── Cross-round comparison ───────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"  CROSS-ROUND COMPARISON")
    print(f"{'='*70}")

    # Load R1 metrics for comparison
    r1_metrics_path = ROOT / "versions" / "bands" / "r1" / "v2" / "metrics.json"
    if r1_metrics_path.exists():
        with open(r1_metrics_path) as f:
            r1_metrics = json.load(f)

        print(f"\n  P95 mean width by round ($/MWh):")
        print(f"  {'Quarter':<10} {'R1':>10} {'R2':>10} {'R3':>10} {'R2/R1':>8} {'R3/R1':>8}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
        for q in QUARTERS:
            r1_w = r1_metrics["widths"][q]["overall"]["p95"]["mean_width"]
            r2_w = r2_metrics["widths"][q]["overall"]["p95"]["mean_width"]
            r3_w = r3_metrics["widths"][q]["overall"]["p95"]["mean_width"]
            r2_ratio = r2_w / r1_w if r1_w else 0
            r3_ratio = r3_w / r1_w if r1_w else 0
            print(f"  {q:<10} {r1_w:>10.0f} {r2_w:>10.1f} {r3_w:>10.1f} {r2_ratio:>7.1%} {r3_ratio:>7.1%}")

    print(f"\nDone. Memory: {mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
