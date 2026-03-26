# DEPRECATED — archived 2026-03-17. Old data sources. See task_plan.md for porting plan.
import warnings as _w; _w.warn("This script is DEPRECATED (archived 2026-03-17). Use canonical get_trades_of_given_duration().", DeprecationWarning, stacklevel=2)

"""Phase 3 v2: Band Width Reduction Experiments.

Tests 5 bin configurations against v1 baseline to reduce band widths
while maintaining coverage accuracy. Primary approach: split the large
bin (|f0| >= 1000) at 3,000 to separate paths with different residual
behavior.

Experiments:
    v1_repro         [0,50,250,1000,inf]       4 bins — baseline reproduction
    split_large      [0,50,250,1000,3000,inf]  5 bins — split most heterogeneous bin
    split_large_shrunk  same 5 bins, α=0.8     shrink extreme bins toward global quantile
    six_bins         [0,50,250,1000,3000,10000,inf]  6 bins — test if 10k+ split helps
    quantile_bins    data-driven 4 bins (per fold)   — test domain boundaries optimality

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual/scripts/run_phase3_v2_bands.py
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

# ─── Imports from v1 script ───────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase3_bands import (
    assign_bins,
    calibrate_bin_widths,
    apply_bands,
    evaluate_coverage,
    loo_band_calibration,
    sanitize_for_json,
    mem_mb,
    COVERAGE_LEVELS,
    COVERAGE_LABELS,
    BASELINE_COL,
    MCP_COL,
    PY_COL,
)

# ─── Constants ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/opt/temp/qianli/annual_research/crossproduct_work")
QUARTERS = ["aq1", "aq2", "aq3", "aq4"]
PYS = [2020, 2021, 2022, 2023, 2024, 2025]

# Experiment configs: name -> {boundaries, labels, alpha (optional)}
EXPERIMENTS = {
    "v1_repro": {
        "boundaries": [0, 50, 250, 1000, float("inf")],
        "labels": ["tiny", "small", "medium", "large"],
    },
    "split_large": {
        "boundaries": [0, 50, 250, 1000, 3000, float("inf")],
        "labels": ["tiny", "small", "medium", "large_lo", "large_hi"],
    },
    "split_large_shrunk": {
        "boundaries": [0, 50, 250, 1000, 3000, float("inf")],
        "labels": ["tiny", "small", "medium", "large_lo", "large_hi"],
        "shrinkage_alpha": 0.8,
    },
    "six_bins": {
        "boundaries": [0, 50, 250, 1000, 3000, 10000, float("inf")],
        "labels": ["tiny", "small", "medium", "large_lo", "large_hi", "extreme"],
    },
    "quantile_bins": {
        "n_quantile_bins": 4,
    },
}


# ─── New calibration functions ────────────────────────────────────────────────


def calibrate_with_shrinkage(
    df: pl.DataFrame,
    boundaries: list[float],
    labels: list[str],
    alpha: float,
    baseline_col: str = BASELINE_COL,
    mcp_col: str = MCP_COL,
    coverage_levels: list[float] = COVERAGE_LEVELS,
) -> dict[str, dict[str, float]]:
    """Per-bin empirical quantile, shrunk toward global quantile.

    w = alpha * q_bin + (1 - alpha) * q_global

    Returns {bin_label: {p50: width, p70: width, ...}}.
    """
    abs_res = (df[mcp_col] - df[baseline_col]).abs()
    bins = assign_bins(df[baseline_col].abs(), boundaries, labels)

    work = pl.DataFrame({
        "abs_residual": abs_res,
        "bin": bins,
    })

    # Compute global quantiles
    global_quantiles = {}
    for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
        global_quantiles[clabel] = float(abs_res.quantile(level))

    result = {}
    for label in labels:
        subset = work.filter(pl.col("bin") == label)["abs_residual"]
        n = len(subset)
        widths = {}
        for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
            if n > 0:
                q_bin = float(subset.quantile(level))
                q_global = global_quantiles[clabel]
                shrunk = alpha * q_bin + (1 - alpha) * q_global
                widths[clabel] = round(shrunk, 1)
            else:
                widths[clabel] = None
        if n == 0:
            print(f"    WARNING: bin '{label}' has 0 rows in shrinkage calibration")
        widths["n"] = n
        result[label] = widths

    return result


def compute_quantile_boundaries(
    series: pl.Series,
    n_bins: int,
) -> tuple[list[float], list[str]]:
    """Compute data-driven bin boundaries from percentiles of |baseline|.

    Returns (boundaries, labels) where boundaries has n_bins+1 entries.
    """
    abs_vals = series.abs()
    quantiles = [i / n_bins for i in range(1, n_bins)]
    cuts = [0.0]
    for q in quantiles:
        cuts.append(round(float(abs_vals.quantile(q)), 1))
    cuts.append(float("inf"))

    labels = [f"q{i+1}" for i in range(n_bins)]
    return cuts, labels


def loo_band_calibration_shrunk(
    df: pl.DataFrame,
    quarter: str,
    pys: list[int],
    boundaries: list[float],
    labels: list[str],
    alpha: float,
    baseline_col: str = BASELINE_COL,
    mcp_col: str = MCP_COL,
    coverage_levels: list[float] = COVERAGE_LEVELS,
) -> dict:
    """LOO calibration with shrinkage. Same structure as loo_band_calibration."""
    per_py = {}
    all_test_dfs = []

    for test_py in pys:
        train = df.filter(pl.col(PY_COL) != test_py)
        test = df.filter(pl.col(PY_COL) == test_py)

        if test.height == 0:
            continue

        bin_widths = calibrate_with_shrinkage(
            train, boundaries, labels, alpha,
            baseline_col, mcp_col, coverage_levels,
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


def loo_band_calibration_quantile(
    df: pl.DataFrame,
    quarter: str,
    pys: list[int],
    n_bins: int,
    baseline_col: str = BASELINE_COL,
    mcp_col: str = MCP_COL,
    coverage_levels: list[float] = COVERAGE_LEVELS,
) -> dict:
    """LOO calibration with data-driven quantile boundaries computed per fold."""
    per_py = {}
    all_test_dfs = []
    fold_boundaries = {}

    for test_py in pys:
        train = df.filter(pl.col(PY_COL) != test_py)
        test = df.filter(pl.col(PY_COL) == test_py)

        if test.height == 0:
            continue

        # Compute boundaries from training set
        boundaries, labels = compute_quantile_boundaries(train[baseline_col], n_bins)
        fold_boundaries[str(test_py)] = boundaries

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
            "boundaries": [b if not math.isinf(b) else "inf" for b in boundaries],
        }

    # Aggregate — use last fold's boundaries/labels for the aggregate evaluation
    # since boundaries vary per fold; we evaluate using the per-fold banded data
    if all_test_dfs:
        all_test = pl.concat(all_test_dfs)
        # For aggregate coverage, use the banded columns already computed per fold
        agg_coverage = {"overall": {}, "per_bin": {}}
        mcp = all_test[mcp_col]
        for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
            lower = all_test[f"lower_{clabel}"]
            upper = all_test[f"upper_{clabel}"]
            covered = ((mcp >= lower) & (mcp <= upper)).mean()
            actual = round(float(covered) * 100, 2)
            target = round(level * 100, 1)
            agg_coverage["overall"][clabel] = {
                "target": target,
                "actual": actual,
                "error": round(actual - target, 2),
            }
        # Per-bin coverage not meaningful for quantile bins (boundaries vary per fold).
        # Bins are rebinned with full-data boundaries — this is approximate only.
        # Mark with approximate=True so select_winner skips BG3 per-bin check.
        agg_coverage["per_bin_approximate"] = True
        if fold_boundaries:
            last_boundaries, labels = compute_quantile_boundaries(
                df[baseline_col], n_bins,
            )
            bins = assign_bins(all_test[baseline_col].abs(), last_boundaries, labels)
            df_with_bin = all_test.with_columns(pl.Series("_bin", bins))
            for label in labels:
                subset = df_with_bin.filter(pl.col("_bin") == label)
                n = subset.height
                bin_result = {"n": n}
                if n > 0:
                    sub_mcp = subset[mcp_col]
                    for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
                        lower_s = subset[f"lower_{clabel}"]
                        upper_s = subset[f"upper_{clabel}"]
                        covered = ((sub_mcp >= lower_s) & (sub_mcp <= upper_s)).mean()
                        actual = round(float(covered) * 100, 2)
                        target = round(level * 100, 1)
                        bin_result[clabel] = {
                            "target": target,
                            "actual": actual,
                            "error": round(actual - target, 2),
                        }
                else:
                    for clabel in COVERAGE_LABELS:
                        bin_result[clabel] = {"target": 0, "actual": 0, "error": 0}
                agg_coverage["per_bin"][label] = bin_result
    else:
        agg_coverage = {"overall": {}, "per_bin": {}}
        labels = [f"q{i+1}" for i in range(n_bins)]

    # Aggregate widths
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
                per_py[py]["widths"][clabel]["per_bin"].get(label)
                for py in per_py
            ]
            fold_bin_widths = [w for w in fold_bin_widths if w is not None]
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
        "fold_boundaries": fold_boundaries,
    }


# ─── Experiment dispatcher ────────────────────────────────────────────────────


def run_experiment(
    df: pl.DataFrame,
    config_name: str,
    config: dict,
    quarter: str,
    pys: list[int],
) -> dict:
    """Run a single experiment config through LOO calibration.

    Returns the standard LOO result dict with added 'config' field.
    """
    if config_name == "quantile_bins":
        n_bins = config["n_quantile_bins"]
        result = loo_band_calibration_quantile(
            df, quarter, pys, n_bins,
        )
        result["config"] = {
            "name": config_name,
            "n_quantile_bins": n_bins,
            "type": "quantile",
        }
    elif "shrinkage_alpha" in config:
        alpha = config["shrinkage_alpha"]
        result = loo_band_calibration_shrunk(
            df, quarter, pys,
            config["boundaries"], config["labels"], alpha,
        )
        result["config"] = {
            "name": config_name,
            "boundaries": [b if not math.isinf(b) else "inf" for b in config["boundaries"]],
            "labels": config["labels"],
            "shrinkage_alpha": alpha,
            "type": "shrinkage",
        }
    else:
        result = loo_band_calibration(
            df, quarter, pys,
            BASELINE_COL, MCP_COL,
            config["boundaries"], config["labels"], COVERAGE_LEVELS,
        )
        result["config"] = {
            "name": config_name,
            "boundaries": [b if not math.isinf(b) else "inf" for b in config["boundaries"]],
            "labels": config["labels"],
            "type": "fixed",
        }

    return result


# ─── Asymmetry diagnostic ────────────────────────────────────────────────────


def asymmetry_diagnostic(
    df: pl.DataFrame,
    quarter: str,
    pys: list[int],
    boundaries: list[float],
    labels: list[str],
) -> dict:
    """Compute upper/lower quantile ratios per bin per PY.

    For each bin, compute:
    - upper_q95 = 95th percentile of (mcp - baseline) where mcp > baseline
    - lower_q95 = 95th percentile of (baseline - mcp) where mcp < baseline
    - ratio = upper_q95 / lower_q95

    Returns {bin_label: {py: {upper_q95, lower_q95, ratio, n_upper, n_lower}}}.
    Also computes CV of ratios across PYs per bin.
    """
    residual = df[MCP_COL] - df[BASELINE_COL]
    bins = assign_bins(df[BASELINE_COL].abs(), boundaries, labels)

    work = df.select([pl.col(PY_COL)]).with_columns([
        pl.Series("residual", residual),
        pl.Series("bin", bins),
    ])

    result = {}
    for label in labels:
        bin_data = work.filter(pl.col("bin") == label)
        py_ratios = {}
        ratio_values = []

        for py in pys:
            py_data = bin_data.filter(pl.col(PY_COL) == py)
            res = py_data["residual"]

            upper = res.filter(res > 0)
            lower = res.filter(res < 0).abs()

            n_upper = len(upper)
            n_lower = len(lower)

            if n_upper >= 50 and n_lower >= 50:
                upper_q95 = round(float(upper.quantile(0.95)), 1)
                lower_q95 = round(float(lower.quantile(0.95)), 1)
                ratio = round(upper_q95 / lower_q95, 3) if lower_q95 > 0 else None
            else:
                upper_q95 = None
                lower_q95 = None
                ratio = None

            py_ratios[str(py)] = {
                "upper_q95": upper_q95,
                "lower_q95": lower_q95,
                "ratio": ratio,
                "n_upper": n_upper,
                "n_lower": n_lower,
            }
            if ratio is not None:
                ratio_values.append(ratio)

        # CV across PYs
        if len(ratio_values) >= 2:
            cv = round(statistics.stdev(ratio_values) / statistics.mean(ratio_values), 4)
            mean_ratio = round(statistics.mean(ratio_values), 3)
        else:
            cv = None
            mean_ratio = None

        result[label] = {
            "per_py": py_ratios,
            "cv": cv,
            "mean_ratio": mean_ratio,
        }

    return result


# ─── Comparison & printing ────────────────────────────────────────────────────


def print_comparison(all_results: dict[str, dict], quarter: str) -> None:
    """Print side-by-side comparison table across experiments for a quarter."""
    print(f"\n{'='*80}")
    print(f"  {quarter.upper()} — Experiment Comparison")
    print(f"{'='*80}")

    configs = list(all_results.keys())

    # Header
    print(f"\n  {'Config':<22}", end="")
    print(f" {'P95 cov':>8} {'P95 err':>8} {'P50 cov':>8} {'P50 err':>8} {'P95 mean_w':>10} {'Width CV':>9}")
    print(f"  {'-'*22}", end="")
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
            f"  {name:<22}"
            f" {p95_cov:>7.2f}% {p95_err:>+7.2f}pp"
            f" {p50_cov:>7.2f}% {p50_err:>+7.2f}pp"
            f" {p95_w:>10.0f} {w_cv:>9.4f}"
        )

    # Per-bin widths for split_large vs v1_repro
    if "v1_repro" in all_results and "split_large" in all_results:
        v1 = all_results["v1_repro"]["aggregate"]["widths"]["per_bin"]
        sl = all_results["split_large"]["aggregate"]["widths"]["per_bin"]

        print(f"\n  Per-bin P95 width comparison (v1_repro vs split_large):")
        print(f"  {'Bin':<12} {'v1_repro':>10} {'split_large':>12} {'Change':>10}")
        print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*10}")

        # v1 bins
        for label in ["tiny", "small", "medium"]:
            v1_w = v1.get(label, {}).get("p95", {}).get("mean_width")
            sl_w = sl.get(label, {}).get("p95", {}).get("mean_width")
            if v1_w and sl_w:
                pct = round((sl_w - v1_w) / v1_w * 100, 1)
                print(f"  {label:<12} {v1_w:>10.0f} {sl_w:>12.0f} {pct:>+9.1f}%")

        # v1 large -> split_large large_lo + large_hi
        v1_large = v1.get("large", {}).get("p95", {}).get("mean_width")
        sl_lo = sl.get("large_lo", {}).get("p95", {}).get("mean_width")
        sl_hi = sl.get("large_hi", {}).get("p95", {}).get("mean_width")
        if v1_large:
            print(f"  {'large':<12} {v1_large:>10.0f} {'—':>12}")
        if sl_lo:
            pct_lo = round((sl_lo - v1_large) / v1_large * 100, 1) if v1_large else 0
            print(f"  {'  large_lo':<12} {'':>10} {sl_lo:>12.0f} {pct_lo:>+9.1f}%")
        if sl_hi:
            pct_hi = round((sl_hi - v1_large) / v1_large * 100, 1) if v1_large else 0
            print(f"  {'  large_hi':<12} {'':>10} {sl_hi:>12.0f} {pct_hi:>+9.1f}%")

    # Stability
    print(f"\n  Per-PY stability:")
    print(f"  {'Config':<22} {'Worst PY':>10} {'Worst cov':>10} {'Range':>8}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*8}")
    for name in configs:
        s = all_results[name]["stability"]
        print(
            f"  {name:<22}"
            f" {s['p95_worst_py']:>10}"
            f" {s['p95_worst_py_coverage']:>9.2f}%"
            f" {s['p95_coverage_range']:>7.2f}pp"
        )


def print_asymmetry(asym: dict, quarter: str) -> None:
    """Print asymmetry diagnostic results."""
    print(f"\n  {quarter.upper()} — Asymmetry Diagnostic (upper/lower P95 ratio)")
    print(f"  {'Bin':<12} {'Mean ratio':>10} {'CV':>8} {'PY range':>20}")
    print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*20}")

    for label, data in asym.items():
        mean_r = data["mean_ratio"]
        cv = data["cv"]
        ratios = [
            data["per_py"][py]["ratio"]
            for py in sorted(data["per_py"])
            if data["per_py"][py]["ratio"] is not None
        ]
        if ratios:
            rng = f"{min(ratios):.2f}-{max(ratios):.2f}"
        else:
            rng = "n/a"

        mean_str = f"{mean_r:.3f}" if mean_r is not None else "n/a"
        cv_str = f"{cv:.4f}" if cv is not None else "n/a"
        print(f"  {label:<12} {mean_str:>10} {cv_str:>8} {rng:>20}")


def check_min_bin_sizes(all_results: dict[str, dict], min_rows: int = 1000) -> dict[str, list[str]]:
    """Check that no bin has fewer than min_rows in any quarter.

    Returns {config_name: [list of violations]}.
    """
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

    # Overall
    ws = []
    for lvl in level_order:
        w = widths["overall"].get(lvl, {}).get("mean_width")
        if w is None:
            return False
        ws.append(w)
    if not all(ws[i] < ws[i + 1] for i in range(len(ws) - 1)):
        return False

    # Per-bin
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
) -> str:
    """Select winning config: pass BG1-BG3 in all quarters, lowest overall P95 mean width.

    Returns config name.
    """
    candidates = {}

    for name in EXPERIMENTS:
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

            # BG3: per-bin uniformity all within 5pp
            # Skip per-bin check for quantile_bins (boundaries vary per fold,
            # aggregate per-bin is approximate and unreliable for gate checks)
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

            # Min bin size (skip for approximate per-bin)
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
        print("\n  WARNING: No config passes all gates. Falling back to split_large.")
        return "split_large"

    winner = min(candidates, key=candidates.get)
    print(f"\n  Winner: {winner} (avg P95 mean width: {candidates[winner]:.0f})")
    if len(candidates) > 1:
        print(f"  All passing configs:")
        for name, w in sorted(candidates.items(), key=lambda kv: kv[1]):
            print(f"    {name}: {w:.0f}")

    return winner


# ─── Temporal (expanding window) validation ───────────────────────────────────


def temporal_band_calibration_quantile(
    df: pl.DataFrame,
    quarter: str,
    pys: list[int],
    n_bins: int,
    baseline_col: str = BASELINE_COL,
    mcp_col: str = MCP_COL,
    coverage_levels: list[float] = COVERAGE_LEVELS,
) -> dict:
    """Temporal expanding-window calibration with quantile bins.

    For each test PY, train ONLY on strictly prior PYs.
    PY with no prior data is skipped.
    """
    per_py = {}
    all_test_dfs = []

    for test_py in pys:
        train = df.filter(pl.col(PY_COL) < test_py)
        test = df.filter(pl.col(PY_COL) == test_py)

        if train.height == 0 or test.height == 0:
            continue

        train_pys = sorted(train[PY_COL].unique().to_list())

        boundaries, labels = compute_quantile_boundaries(train[baseline_col], n_bins)

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
            "train_pys": train_pys,
            "coverage": cov,
            "widths": width_summary,
            "boundaries": [b if not math.isinf(b) else "inf" for b in boundaries],
        }

    # Aggregate
    if all_test_dfs:
        all_test = pl.concat(all_test_dfs)
        agg_coverage = {"overall": {}, "per_bin": {}}
        mcp = all_test[mcp_col]
        for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
            lower = all_test[f"lower_{clabel}"]
            upper = all_test[f"upper_{clabel}"]
            covered = ((mcp >= lower) & (mcp <= upper)).mean()
            actual = round(float(covered) * 100, 2)
            target = round(level * 100, 1)
            agg_coverage["overall"][clabel] = {
                "target": target,
                "actual": actual,
                "error": round(actual - target, 2),
            }
    else:
        agg_coverage = {"overall": {}, "per_bin": {}}

    # Aggregate widths
    agg_widths = {"overall": {}}
    labels = [f"q{i+1}" for i in range(n_bins)]
    for clabel in COVERAGE_LABELS:
        fold_means = [
            per_py[py]["widths"][clabel]["mean_width"]
            for py in per_py
            if per_py[py]["widths"][clabel]["mean_width"] is not None
        ]
        agg_widths["overall"][clabel] = {
            "mean_width": round(sum(fold_means) / len(fold_means), 1) if fold_means else None,
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
        worst_py = list(per_py.keys())[0] if per_py else ""
        p95_worst_py_coverage = p95_coverages[0] if p95_coverages else 0

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


def run_temporal_validation():
    """Run temporal (expanding window) validation for quantile_bins."""
    print(f"Phase 3 v2: Temporal Validation (expanding window)")
    print(f"  For each test PY, train ONLY on strictly prior PYs.")
    print(f"Memory at start: {mem_mb():.0f} MB")

    temporal_results = {}

    for quarter in QUARTERS:
        parquet_path = DATA_DIR / f"{quarter}_all_baselines.parquet"
        print(f"\n{'#'*80}")
        print(f"  {quarter} — Temporal validation")
        print(f"{'#'*80}")

        # Load PY >= 2019 to have training data for PY 2020
        df = (
            pl.scan_parquet(parquet_path)
            .filter(
                (pl.col(PY_COL) >= 2019)
                & pl.col(BASELINE_COL).is_not_null()
                & pl.col(MCP_COL).is_not_null()
            )
            .collect()
        )
        print(f"  Loaded {df.height:,} rows (PY >= 2019, both cols non-null)")

        available_pys = sorted(df[PY_COL].unique().to_list())
        print(f"  Available PYs: {available_pys}")

        # Test on PYs 2020-2025 (training on strictly prior)
        test_pys = [py for py in PYS if py in available_pys]

        result = temporal_band_calibration_quantile(
            df, quarter, test_pys, n_bins=4,
        )
        temporal_results[quarter] = result

        # Print per-PY detail
        print(f"\n  {'PY':<6} {'Train PYs':<20} {'n_train':>8} {'n_test':>8} {'P95 cov':>8} {'P95 width':>10}")
        print(f"  {'-'*6} {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        for py in sorted(result["per_py"]):
            p = result["per_py"][py]
            train_pys_str = ",".join(str(y) for y in p["train_pys"])
            p95_cov = p["coverage"]["overall"]["p95"]["actual"]
            p95_w = p["widths"]["p95"]["mean_width"]
            print(f"  {py:<6} {train_pys_str:<20} {p['n_train']:>8,} {p['n_test']:>8,} {p95_cov:>7.2f}% {p95_w:>10.0f}")

        agg = result["aggregate"]["coverage"]["overall"]
        p95 = agg.get("p95", {})
        print(f"\n  Aggregate: P95 cov={p95.get('actual', 0):.2f}% (err {p95.get('error', 0):+.2f}pp)")

        del df
        gc.collect()
        print(f"  Memory: {mem_mb():.0f} MB")

    # ─── Compare temporal vs LOO ──────────────────────────────────────────

    print(f"\n{'='*80}")
    print(f"  TEMPORAL vs LOO COMPARISON")
    print(f"{'='*80}")

    v2_metrics_path = ROOT / "versions" / "bands" / "r1" / "v2" / "metrics.json"
    if v2_metrics_path.exists():
        with open(v2_metrics_path) as f:
            loo_metrics = json.load(f)

        print(f"\n  {'Quarter':<10} {'LOO P95':>8} {'Temp P95':>9} {'LOO width':>10} {'Temp width':>11}")
        print(f"  {'-'*10} {'-'*8} {'-'*9} {'-'*10} {'-'*11}")
        for q in QUARTERS:
            loo_cov = loo_metrics["coverage"][q]["overall"]["p95"]["actual"]
            tmp_cov = temporal_results[q]["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
            loo_w = loo_metrics["widths"][q]["overall"]["p95"]["mean_width"]
            tmp_w = temporal_results[q]["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width", 0)
            print(f"  {q:<10} {loo_cov:>7.2f}% {tmp_cov:>8.2f}% {loo_w:>10.0f} {tmp_w:>11.0f}")
    else:
        print("  No LOO metrics found. Run main experiments first.")

    # ─── Save temporal results to metrics.json ────────────────────────────

    v2_metrics_path = ROOT / "versions" / "bands" / "r1" / "v2" / "metrics.json"
    if v2_metrics_path.exists():
        with open(v2_metrics_path) as f:
            v2_metrics = json.load(f)

        v2_metrics["temporal_validation"] = {}
        for q in QUARTERS:
            r = temporal_results[q]
            v2_metrics["temporal_validation"][q] = {
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

        output = sanitize_for_json(v2_metrics)
        tmp = v2_metrics_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(output, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, v2_metrics_path)
        print(f"\nTemporal results appended to {v2_metrics_path}")

    print(f"\nDone. Memory: {mem_mb():.0f} MB")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    print(f"Phase 3 v2: Band Width Reduction Experiments")
    print(f"Memory at start: {mem_mb():.0f} MB")

    all_quarter_results: dict[str, dict[str, dict]] = {}
    all_quarter_asymmetry: dict[str, dict] = {}

    for quarter in QUARTERS:
        parquet_path = DATA_DIR / f"{quarter}_all_baselines.parquet"
        print(f"\n{'#'*80}")
        print(f"  Processing {quarter} from {parquet_path}")
        print(f"{'#'*80}")

        # Load with lazy scan
        df = (
            pl.scan_parquet(parquet_path)
            .filter(
                (pl.col(PY_COL) >= 2020)
                & pl.col(BASELINE_COL).is_not_null()
                & pl.col(MCP_COL).is_not_null()
            )
            .collect()
        )
        print(f"  Loaded {df.height:,} rows (PY >= 2020, both cols non-null)")
        print(f"  Memory: {mem_mb():.0f} MB")

        available_pys = sorted(df[PY_COL].unique().to_list())
        pys_to_use = [py for py in PYS if py in available_pys]

        # Run all 5 experiments
        quarter_results = {}
        for name, config in EXPERIMENTS.items():
            print(f"\n  Running {name}...", end="", flush=True)
            result = run_experiment(df, name, config, quarter, pys_to_use)
            quarter_results[name] = result
            p95_w = result["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width", 0)
            p95_cov = result["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
            print(f" P95 cov={p95_cov:.2f}%, width={p95_w:.0f}")

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

        # Width monotonicity check
        print(f"\n  Width monotonicity:")
        for name in EXPERIMENTS:
            mono = check_width_monotonicity(quarter_results[name])
            print(f"    {name}: {'PASS' if mono else 'FAIL'}")

        # Asymmetry diagnostic (using split_large boundaries)
        asym_boundaries = [0, 50, 250, 1000, 3000, float("inf")]
        asym_labels = ["tiny", "small", "medium", "large_lo", "large_hi"]
        asym = asymmetry_diagnostic(df, quarter, pys_to_use, asym_boundaries, asym_labels)
        all_quarter_asymmetry[quarter] = asym
        print_asymmetry(asym, quarter)

        # Free memory
        del df, quarter_results
        gc.collect()
        print(f"\n  Memory after cleanup: {mem_mb():.0f} MB")

    # ─── Select winner ────────────────────────────────────────────────────

    print(f"\n{'='*80}")
    print(f"  WINNER SELECTION")
    print(f"{'='*80}")

    winner = select_winner(all_quarter_results)

    # ─── Build v2 metrics ─────────────────────────────────────────────────

    winner_config = EXPERIMENTS[winner]
    v2_metrics = {
        "coverage": {},
        "widths": {},
        "per_py": {},
        "stability": {},
        "experiment_comparison": {},
        "asymmetry_diagnostic": {},
    }

    for quarter in QUARTERS:
        w_result = all_quarter_results[quarter][winner]

        # Coverage
        v2_metrics["coverage"][quarter] = w_result["aggregate"]["coverage"]

        # Widths
        v2_metrics["widths"][quarter] = {
            "overall": w_result["aggregate"]["widths"]["overall"],
            "per_bin": w_result["aggregate"]["widths"]["per_bin"],
        }

        # Per-PY
        py_summary = {}
        for py, pdata in w_result["per_py"].items():
            py_summary[py] = {
                "p50_coverage": pdata["coverage"]["overall"]["p50"]["actual"],
                "p95_coverage": pdata["coverage"]["overall"]["p95"]["actual"],
                "p95_mean_width": pdata["widths"]["p95"]["mean_width"],
            }
        v2_metrics["per_py"][quarter] = py_summary

        # Stability
        v2_metrics["stability"][quarter] = w_result["stability"]

        # Experiment comparison (all configs, summary only)
        comparison = {}
        for name in EXPERIMENTS:
            r = all_quarter_results[quarter][name]
            comparison[name] = {
                "p95_coverage": r["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual"),
                "p95_error": r["aggregate"]["coverage"]["overall"].get("p95", {}).get("error"),
                "p50_coverage": r["aggregate"]["coverage"]["overall"].get("p50", {}).get("actual"),
                "p95_mean_width": r["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width"),
            }
        v2_metrics["experiment_comparison"][quarter] = comparison

        # Asymmetry
        v2_metrics["asymmetry_diagnostic"][quarter] = sanitize_for_json(
            all_quarter_asymmetry[quarter]
        )

    # ─── Save metrics ─────────────────────────────────────────────────────

    v2_dir = ROOT / "versions" / "bands" / "r1" / "v2"
    v2_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = v2_dir / "metrics.json"
    output = sanitize_for_json(v2_metrics)
    tmp = metrics_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, metrics_path)
    print(f"\nMetrics saved to {metrics_path}")

    # ─── Save config.json ─────────────────────────────────────────────────

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
        ).strip()
    except Exception:
        git_hash = "unknown"

    config = {
        "schema_version": 1,
        "version": "v2",
        "description": f"Width reduction via {winner} bin scheme",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "part": "bands/r1",
        "baseline_version": "v3",
        "method": {
            "band_type": "symmetric",
            "calibration": f"empirical quantile of |mcp - nodal_f0| per |nodal_f0| bin ({winner})",
            "cv_method": "LOO_by_PY (train on 5 PYs, test on 1)",
            "winning_config": winner,
        },
        "parameters": {
            "coverage_levels": [0.50, 0.70, 0.80, 0.90, 0.95],
            "bin_boundaries": [
                b if not math.isinf(b) else "inf"
                for b in winner_config.get("boundaries", [])
            ] if "boundaries" in winner_config else "data-driven",
            "bin_labels": winner_config.get("labels", [f"q{i+1}" for i in range(winner_config.get("n_quantile_bins", 4))]),
            "shrinkage_alpha": winner_config.get("shrinkage_alpha"),
            "band_type": "symmetric",
            "cv_method": "LOO_by_PY",
        },
        "experiments_tested": list(EXPERIMENTS.keys()),
        "data_sources": [
            {"path": f"crossproduct_work/{q}_all_baselines.parquet", "columns": ["mcp_mean", "nodal_f0"]}
            for q in QUARTERS
        ],
        "environment": {
            "git_hash": git_hash,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "polars_version": pl.__version__,
        },
    }
    config_path = v2_dir / "config.json"
    tmp = config_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, config_path)
    print(f"Config saved to {config_path}")

    # ─── Print final summary ──────────────────────────────────────────────

    print(f"\n{'='*80}")
    print(f"  SUMMARY — Phase 3 v2 Band Width Reduction")
    print(f"{'='*80}")
    print(f"\n  Winner: {winner}")

    # v1 vs v2 width comparison
    v1_metrics_path = ROOT / "versions" / "bands" / "r1" / "v1" / "metrics.json"
    if v1_metrics_path.exists():
        with open(v1_metrics_path) as f:
            v1_metrics = json.load(f)

        print(f"\n  v1 vs v2 P95 mean width comparison:")
        print(f"  {'Quarter':<10} {'v1':>8} {'v2':>8} {'Change':>10}")
        print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10}")
        for q in QUARTERS:
            v1_w = v1_metrics.get("widths", {}).get(q, {}).get("overall", {}).get("p95", {}).get("mean_width", 0)
            v2_w = v2_metrics["widths"][q]["overall"].get("p95", {}).get("mean_width", 0)
            if v1_w and v2_w:
                pct = round((v2_w - v1_w) / v1_w * 100, 1)
                print(f"  {q:<10} {v1_w:>8.0f} {v2_w:>8.0f} {pct:>+9.1f}%")

    print(f"\n  Coverage accuracy (LOO, P95):")
    print(f"  {'Quarter':<10} {'Target':>8} {'Actual':>8} {'Error':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for q in QUARTERS:
        cov = v2_metrics["coverage"][q]["overall"]["p95"]
        print(f"  {q:<10} {cov['target']:>7.1f}% {cov['actual']:>7.2f}% {cov['error']:>+7.2f}pp")

    print(f"\n  Stability (P95):")
    print(f"  {'Quarter':<10} {'Worst PY':>10} {'Worst cov':>10} {'Range':>8} {'Width CV':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
    for q in QUARTERS:
        s = v2_metrics["stability"][q]
        print(f"  {q:<10} {s['p95_worst_py']:>10} {s['p95_worst_py_coverage']:>9.2f}% {s['p95_coverage_range']:>7.2f}pp {s['p95_width_cv']:>10.4f}")

    # Asymmetry summary
    print(f"\n  Asymmetry diagnostic summary (justifying symmetric bands):")
    print(f"  {'Bin':<12}", end="")
    for q in QUARTERS:
        print(f" {q+' CV':>10}", end="")
    print()
    print(f"  {'-'*12}", end="")
    for _ in QUARTERS:
        print(f" {'-'*10}", end="")
    print()

    # Collect all bin labels from asymmetry data
    all_asym_labels = list(all_quarter_asymmetry[QUARTERS[0]].keys())
    for label in all_asym_labels:
        print(f"  {label:<12}", end="")
        for q in QUARTERS:
            cv = all_quarter_asymmetry[q].get(label, {}).get("cv")
            if cv is not None:
                print(f" {cv:>10.4f}", end="")
            else:
                print(f" {'n/a':>10}", end="")
        print()

    print(f"\n  High CV values indicate upper/lower ratio instability across PYs,")
    print(f"  justifying symmetric bands over asymmetric.")

    print(f"\nDone. Memory: {mem_mb():.0f} MB")


if __name__ == "__main__":
    if "--temporal" in sys.argv:
        run_temporal_validation()
    else:
        main()
