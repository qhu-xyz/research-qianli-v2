# DEPRECATED — archived 2026-03-17. Old data sources. See task_plan.md for porting plan.
import warnings as _w; _w.warn("This script is DEPRECATED (archived 2026-03-17). Use canonical get_trades_of_given_duration().", DeprecationWarning, stacklevel=2)

"""Bands v3: Per-Class Stratified Band Calibration.

Current bands (v2 for R1, v1 for R2/R3) pool onpeak + offpeak residuals.
Onpeak MAE is up to 20% higher than offpeak in some quarters, meaning
pooled bands under-cover onpeak and over-cover offpeak.

v3 calibrates separate widths per (|baseline| bin, class_type) pair.
Bin boundaries are computed on the full training set (pooling classes),
only the widths differ per class.

Tests per round:
    pooled      v2/v1 reproduction (pooled onpeak + offpeak)
    per_class   separate widths per class within each bin

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual/scripts/run_v3_bands.py
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
R1_DATA_DIR = Path("/opt/temp/qianli/annual_research/crossproduct_work")
R2R3_DATA_PATH = Path("/opt/temp/qianli/annual_research/all_residuals_v2.parquet")
QUARTERS = ["aq1", "aq2", "aq3", "aq4"]
R1_PYS = [2020, 2021, 2022, 2023, 2024, 2025]
R2R3_PYS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

MCP_COL = "mcp_mean"
PY_COL = "planning_year"
CLASS_COL = "class_type"
CLASSES = ["onpeak", "offpeak"]

MIN_CLASS_BIN_ROWS = 500  # fall back to pooled if fewer rows


# ─── Per-class calibration functions ─────────────────────────────────────────


def calibrate_bin_widths_per_class(
    df: pl.DataFrame,
    baseline_col: str,
    mcp_col: str = MCP_COL,
    class_col: str = CLASS_COL,
    boundaries: list[float] | None = None,
    labels: list[str] | None = None,
    coverage_levels: list[float] = COVERAGE_LEVELS,
) -> dict[str, dict[str, dict[str, float]]]:
    """Per-(bin, class) empirical quantile of |residual|.

    Returns {bin_label: {class: {p50: width, ...}, "_pooled": {...}}}.
    Falls back to pooled estimate if class has < MIN_CLASS_BIN_ROWS in a bin.
    """
    abs_res = (df[mcp_col] - df[baseline_col]).abs()
    bins = assign_bins(df[baseline_col].abs(), boundaries, labels)

    work = pl.DataFrame({
        "abs_residual": abs_res,
        "bin": bins,
        "class_type": df[class_col],
    })

    result = {}
    for label in labels:
        bin_data = work.filter(pl.col("bin") == label)

        # Pooled estimate (fallback)
        pooled_subset = bin_data["abs_residual"]
        n_pooled = len(pooled_subset)
        pooled_widths = {}
        for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
            if n_pooled > 0:
                pooled_widths[clabel] = round(float(pooled_subset.quantile(level)), 1)
            else:
                pooled_widths[clabel] = None
        pooled_widths["n"] = n_pooled

        class_widths = {"_pooled": pooled_widths}

        for cls in CLASSES:
            cls_subset = bin_data.filter(pl.col("class_type") == cls)["abs_residual"]
            n_cls = len(cls_subset)

            if n_cls >= MIN_CLASS_BIN_ROWS:
                widths = {}
                for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
                    widths[clabel] = round(float(cls_subset.quantile(level)), 1)
                widths["n"] = n_cls
            else:
                # Fall back to pooled
                widths = dict(pooled_widths)
                widths["n"] = n_cls
                widths["_fallback"] = True

            class_widths[cls] = widths

        result[label] = class_widths

    return result


def apply_bands_per_class(
    df: pl.DataFrame,
    bin_widths_per_class: dict[str, dict[str, dict[str, float]]],
    baseline_col: str,
    class_col: str = CLASS_COL,
    boundaries: list[float] | None = None,
    labels: list[str] | None = None,
) -> pl.DataFrame:
    """Add lower/upper columns using class-specific widths."""
    abs_bl = df[baseline_col].abs()
    bins = assign_bins(abs_bl, boundaries, labels)
    df = df.with_columns([
        pl.Series("_bin", bins),
    ])

    for clabel in COVERAGE_LABELS:
        # Build width series: for each row, look up (bin, class) -> width
        width_values = []
        for i in range(df.height):
            bin_label = df["_bin"][i]
            cls = df[class_col][i]
            if bin_label is not None and cls in bin_widths_per_class.get(bin_label, {}):
                w = bin_widths_per_class[bin_label][cls].get(clabel)
            elif bin_label is not None:
                w = bin_widths_per_class[bin_label]["_pooled"].get(clabel)
            else:
                w = None
            width_values.append(w)

        width_series = pl.Series("_w", width_values, dtype=pl.Float64)

        df = df.with_columns([
            (pl.col(baseline_col) - width_series).alias(f"lower_{clabel}"),
            (pl.col(baseline_col) + width_series).alias(f"upper_{clabel}"),
        ])

    return df.drop("_bin")


def apply_bands_per_class_fast(
    df: pl.DataFrame,
    bin_widths_per_class: dict[str, dict[str, dict[str, float]]],
    baseline_col: str,
    class_col: str = CLASS_COL,
    boundaries: list[float] | None = None,
    labels: list[str] | None = None,
) -> pl.DataFrame:
    """Vectorized version of apply_bands_per_class using join."""
    abs_bl = df[baseline_col].abs()
    bins = assign_bins(abs_bl, boundaries, labels)
    df = df.with_columns(pl.Series("_bin", bins))

    # Build lookup table: (bin, class) -> {p50: w, p70: w, ...}
    rows = []
    for bin_label in labels:
        for cls in CLASSES:
            entry = {"_bin": bin_label, CLASS_COL: cls}
            class_data = bin_widths_per_class.get(bin_label, {}).get(cls)
            if class_data is None:
                class_data = bin_widths_per_class.get(bin_label, {}).get("_pooled", {})
            for clabel in COVERAGE_LABELS:
                entry[f"_w_{clabel}"] = class_data.get(clabel)
            rows.append(entry)

    lookup = pl.DataFrame(rows)

    df = df.join(lookup, on=["_bin", CLASS_COL], how="left")

    for clabel in COVERAGE_LABELS:
        df = df.with_columns([
            (pl.col(baseline_col) - pl.col(f"_w_{clabel}")).alias(f"lower_{clabel}"),
            (pl.col(baseline_col) + pl.col(f"_w_{clabel}")).alias(f"upper_{clabel}"),
        ])

    drop_cols = ["_bin"] + [f"_w_{clabel}" for clabel in COVERAGE_LABELS]
    return df.drop(drop_cols)


def evaluate_per_class_coverage(
    df: pl.DataFrame,
    mcp_col: str = MCP_COL,
    class_col: str = CLASS_COL,
    coverage_levels: list[float] = COVERAGE_LEVELS,
) -> dict:
    """Coverage per class. Returns {onpeak: {p50: {target, actual, error}, ...}, offpeak: {...}}."""
    result = {}
    mcp = df[mcp_col]

    for cls in CLASSES:
        mask = df[class_col] == cls
        sub = df.filter(mask)
        n = sub.height
        cls_result = {"n": n}
        if n > 0:
            sub_mcp = sub[mcp_col]
            for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
                lower = sub[f"lower_{clabel}"]
                upper = sub[f"upper_{clabel}"]
                covered = ((sub_mcp >= lower) & (sub_mcp <= upper)).mean()
                actual = round(float(covered) * 100, 2)
                target = round(level * 100, 1)
                cls_result[clabel] = {
                    "target": target,
                    "actual": actual,
                    "error": round(actual - target, 2),
                }
        result[cls] = cls_result

    return result


# ─── LOO calibration ────────────────────────────────────────────────────────


def loo_per_class_quantile(
    df: pl.DataFrame,
    quarter: str,
    pys: list[int],
    n_bins: int,
    baseline_col: str,
    mcp_col: str = MCP_COL,
    class_col: str = CLASS_COL,
    coverage_levels: list[float] = COVERAGE_LEVELS,
) -> dict:
    """LOO by PY with per-class quantile calibration."""
    per_py = {}
    all_test_dfs = []
    fold_boundaries = {}

    for test_py in pys:
        train = df.filter(pl.col(PY_COL) != test_py)
        test = df.filter(pl.col(PY_COL) == test_py)

        if test.height == 0:
            continue

        # Compute boundaries from training set (pooling classes)
        boundaries, labels = compute_quantile_boundaries(train[baseline_col], n_bins)
        fold_boundaries[str(test_py)] = boundaries

        # Calibrate per (bin, class)
        bin_widths = calibrate_bin_widths_per_class(
            train, baseline_col, mcp_col, class_col,
            boundaries, labels, coverage_levels,
        )

        # Apply per-class bands
        test_banded = apply_bands_per_class_fast(
            test, bin_widths, baseline_col, class_col, boundaries, labels,
        )
        all_test_dfs.append(test_banded)

        # Overall coverage
        mcp_s = test_banded[mcp_col]
        overall_cov = {"overall": {}}
        for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
            lower = test_banded[f"lower_{clabel}"]
            upper = test_banded[f"upper_{clabel}"]
            covered = ((mcp_s >= lower) & (mcp_s <= upper)).mean()
            actual = round(float(covered) * 100, 2)
            target = round(level * 100, 1)
            overall_cov["overall"][clabel] = {
                "target": target, "actual": actual,
                "error": round(actual - target, 2),
            }

        # Per-class coverage
        per_class_cov = evaluate_per_class_coverage(
            test_banded, mcp_col, class_col, coverage_levels,
        )

        # Per-bin coverage (approximate — use full-data boundaries for rebinning)
        bins = assign_bins(test_banded[baseline_col].abs(), boundaries, labels)
        df_with_bin = test_banded.with_columns(pl.Series("_bin", bins))
        per_bin_cov = {}
        for label in labels:
            subset = df_with_bin.filter(pl.col("_bin") == label)
            n = subset.height
            bin_result = {"n": n}
            if n > 0:
                sub_mcp = subset[mcp_col]
                for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
                    lower_s = subset[f"lower_{clabel}"]
                    upper_s = subset[f"upper_{clabel}"]
                    cov_val = ((sub_mcp >= lower_s) & (sub_mcp <= upper_s)).mean()
                    actual = round(float(cov_val) * 100, 2)
                    target = round(level * 100, 1)
                    bin_result[clabel] = {
                        "target": target, "actual": actual,
                        "error": round(actual - target, 2),
                    }
            else:
                for clabel in COVERAGE_LABELS:
                    bin_result[clabel] = {"target": 0, "actual": 0, "error": 0}
            per_bin_cov[label] = bin_result

        # Width summary
        width_summary = {}
        for clabel in COVERAGE_LABELS:
            widths_for_level = []
            for label in labels:
                # Use average of class widths
                for cls in CLASSES:
                    w = bin_widths[label][cls].get(clabel)
                    if w is not None:
                        widths_for_level.append(w)
            width_summary[clabel] = {
                "mean_width": round(sum(widths_for_level) / len(widths_for_level), 1) if widths_for_level else None,
                "per_bin": {},
            }
            for label in labels:
                cls_widths = {}
                for cls in CLASSES:
                    cls_widths[cls] = bin_widths[label][cls].get(clabel)
                # Average across classes for summary
                vals = [v for v in cls_widths.values() if v is not None]
                cls_widths["avg"] = round(sum(vals) / len(vals), 1) if vals else None
                width_summary[clabel]["per_bin"][label] = cls_widths

        cov_result = overall_cov
        cov_result["per_bin"] = per_bin_cov
        cov_result["per_bin_approximate"] = True  # boundaries vary per fold
        cov_result["per_class"] = per_class_cov

        per_py[str(test_py)] = {
            "n_train": train.height,
            "n_test": test.height,
            "coverage": cov_result,
            "widths": width_summary,
            "boundaries": [b if not math.isinf(b) else "inf" for b in boundaries],
        }

    # ─── Aggregate ────────────────────────────────────────────────────

    if all_test_dfs:
        all_test = pl.concat(all_test_dfs)

        # Overall aggregate coverage
        agg_coverage = {"overall": {}, "per_bin": {}, "per_bin_approximate": True}
        mcp = all_test[mcp_col]
        for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
            lower = all_test[f"lower_{clabel}"]
            upper = all_test[f"upper_{clabel}"]
            covered = ((mcp >= lower) & (mcp <= upper)).mean()
            actual = round(float(covered) * 100, 2)
            target = round(level * 100, 1)
            agg_coverage["overall"][clabel] = {
                "target": target, "actual": actual,
                "error": round(actual - target, 2),
            }

        # Aggregate per-bin (using full-data boundaries for rebinning)
        full_boundaries, full_labels = compute_quantile_boundaries(df[baseline_col], n_bins)
        bins = assign_bins(all_test[baseline_col].abs(), full_boundaries, full_labels)
        df_with_bin = all_test.with_columns(pl.Series("_bin", bins))
        for label in full_labels:
            subset = df_with_bin.filter(pl.col("_bin") == label)
            n = subset.height
            bin_result = {"n": n}
            if n > 0:
                sub_mcp = subset[mcp_col]
                for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
                    lower_s = subset[f"lower_{clabel}"]
                    upper_s = subset[f"upper_{clabel}"]
                    cov_val = ((sub_mcp >= lower_s) & (sub_mcp <= upper_s)).mean()
                    actual = round(float(cov_val) * 100, 2)
                    target = round(level * 100, 1)
                    bin_result[clabel] = {
                        "target": target, "actual": actual,
                        "error": round(actual - target, 2),
                    }
            else:
                for clabel in COVERAGE_LABELS:
                    bin_result[clabel] = {"target": 0, "actual": 0, "error": 0}
            agg_coverage["per_bin"][label] = bin_result

        # Aggregate per-class coverage
        agg_per_class = evaluate_per_class_coverage(
            all_test, mcp_col, CLASS_COL, coverage_levels,
        )
        agg_coverage["per_class"] = agg_per_class
    else:
        agg_coverage = {"overall": {}, "per_bin": {}}
        full_labels = [f"q{i+1}" for i in range(n_bins)]

    # Aggregate widths
    labels = full_labels if all_test_dfs else [f"q{i+1}" for i in range(n_bins)]
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
                per_py[py]["widths"][clabel]["per_bin"].get(label, {}).get("avg")
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


# ─── Temporal validation with per-class ──────────────────────────────────────


def temporal_per_class_quantile(
    df: pl.DataFrame,
    quarter: str,
    pys: list[int],
    n_bins: int,
    baseline_col: str,
    mcp_col: str = MCP_COL,
    class_col: str = CLASS_COL,
    coverage_levels: list[float] = COVERAGE_LEVELS,
    min_train_pys: int = 3,
) -> dict:
    """Temporal expanding-window calibration with per-class quantile bins.

    Args:
        min_train_pys: Minimum number of training PYs for a fold to be included
            in aggregate metrics. Cold-start folds (1-2 training PYs) drag the
            aggregate down but aren't relevant to production where we always have
            3+ years of history.
    """
    per_py = {}
    all_test_dfs = []
    filtered_test_dfs = []  # Only folds meeting min_train_pys

    for test_py in pys:
        train = df.filter(pl.col(PY_COL) < test_py)
        test = df.filter(pl.col(PY_COL) == test_py)

        if train.height == 0 or test.height == 0:
            continue

        train_pys = sorted(train[PY_COL].unique().to_list())
        meets_min = len(train_pys) >= min_train_pys

        boundaries, labels = compute_quantile_boundaries(train[baseline_col], n_bins)

        bin_widths = calibrate_bin_widths_per_class(
            train, baseline_col, mcp_col, class_col,
            boundaries, labels, coverage_levels,
        )

        test_banded = apply_bands_per_class_fast(
            test, bin_widths, baseline_col, class_col, boundaries, labels,
        )
        all_test_dfs.append(test_banded)
        if meets_min:
            filtered_test_dfs.append(test_banded)

        # Overall coverage
        mcp_s = test_banded[mcp_col]
        overall_cov = {}
        for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
            lower = test_banded[f"lower_{clabel}"]
            upper = test_banded[f"upper_{clabel}"]
            covered = ((mcp_s >= lower) & (mcp_s <= upper)).mean()
            actual = round(float(covered) * 100, 2)
            target = round(level * 100, 1)
            overall_cov[clabel] = {
                "target": target, "actual": actual,
                "error": round(actual - target, 2),
            }

        # Width summary
        width_summary = {}
        for clabel in COVERAGE_LABELS:
            widths_for_level = []
            for label in labels:
                for cls in CLASSES:
                    w = bin_widths[label][cls].get(clabel)
                    if w is not None:
                        widths_for_level.append(w)
            width_summary[clabel] = {
                "mean_width": round(sum(widths_for_level) / len(widths_for_level), 1) if widths_for_level else None,
            }

        per_py[str(test_py)] = {
            "n_train": train.height,
            "n_test": test.height,
            "train_pys": train_pys,
            "meets_min_train_pys": meets_min,
            "coverage": {"overall": overall_cov},
            "widths": width_summary,
        }

    # ─── Aggregate (using only folds meeting min_train_pys) ──────────

    use_dfs = filtered_test_dfs if filtered_test_dfs else all_test_dfs
    if use_dfs:
        all_test = pl.concat(use_dfs)

        # Overall aggregate coverage
        agg_coverage = {"overall": {}, "per_bin": {}, "per_class": {}}
        mcp = all_test[mcp_col]
        for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
            lower = all_test[f"lower_{clabel}"]
            upper = all_test[f"upper_{clabel}"]
            covered = ((mcp >= lower) & (mcp <= upper)).mean()
            actual = round(float(covered) * 100, 2)
            target = round(level * 100, 1)
            agg_coverage["overall"][clabel] = {
                "target": target, "actual": actual,
                "error": round(actual - target, 2),
            }

        # Aggregate per-bin coverage (using full-data boundaries for rebinning)
        full_boundaries, full_labels = compute_quantile_boundaries(df[baseline_col], n_bins)
        bins = assign_bins(all_test[baseline_col].abs(), full_boundaries, full_labels)
        df_with_bin = all_test.with_columns(pl.Series("_bin", bins))
        for label in full_labels:
            subset = df_with_bin.filter(pl.col("_bin") == label)
            n = subset.height
            bin_result = {"n": n}
            if n > 0:
                sub_mcp = subset[mcp_col]
                for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
                    lower_s = subset[f"lower_{clabel}"]
                    upper_s = subset[f"upper_{clabel}"]
                    cov_val = ((sub_mcp >= lower_s) & (sub_mcp <= upper_s)).mean()
                    actual = round(float(cov_val) * 100, 2)
                    target = round(level * 100, 1)
                    bin_result[clabel] = {
                        "target": target, "actual": actual,
                        "error": round(actual - target, 2),
                    }
            else:
                for clabel in COVERAGE_LABELS:
                    bin_result[clabel] = {"target": 0, "actual": 0, "error": 0}
            agg_coverage["per_bin"][label] = bin_result

        # Aggregate per-class coverage
        agg_per_class = evaluate_per_class_coverage(
            all_test, mcp_col, CLASS_COL, coverage_levels,
        )
        agg_coverage["per_class"] = agg_per_class
    else:
        agg_coverage = {"overall": {}, "per_bin": {}, "per_class": {}}

    # Aggregate widths (filtered folds only)
    filtered_pys = [
        py for py in per_py
        if per_py[py]["meets_min_train_pys"]
    ] if any(per_py[py]["meets_min_train_pys"] for py in per_py) else list(per_py.keys())

    agg_widths = {"overall": {}}
    for clabel in COVERAGE_LABELS:
        fold_means = [
            per_py[py]["widths"][clabel]["mean_width"]
            for py in filtered_pys
            if per_py[py]["widths"][clabel]["mean_width"] is not None
        ]
        agg_widths["overall"][clabel] = {
            "mean_width": round(sum(fold_means) / len(fold_means), 1) if fold_means else None,
        }

    # Stability (using only folds meeting min_train_pys)
    p95_coverages = [
        per_py[py]["coverage"]["overall"]["p95"]["actual"]
        for py in filtered_pys
    ]
    if len(p95_coverages) >= 2:
        p95_coverage_range = round(max(p95_coverages) - min(p95_coverages), 2)
        worst_py_idx = p95_coverages.index(min(p95_coverages))
        worst_py = filtered_pys[worst_py_idx]
        p95_worst_py_coverage = min(p95_coverages)
    else:
        p95_coverage_range = 0
        worst_py = filtered_pys[0] if filtered_pys else ""
        p95_worst_py_coverage = p95_coverages[0] if p95_coverages else 0

    stability = {
        "p95_coverage_range": p95_coverage_range,
        "p95_worst_py": worst_py,
        "p95_worst_py_coverage": round(p95_worst_py_coverage, 2),
        "min_train_pys": min_train_pys,
        "n_folds_total": len(per_py),
        "n_folds_filtered": len(filtered_pys),
    }

    return {
        "per_py": per_py,
        "aggregate": {"coverage": agg_coverage, "widths": agg_widths},
        "stability": stability,
    }


# ─── Printing ────────────────────────────────────────────────────────────────


def print_comparison(
    pooled_result: dict,
    per_class_result: dict,
    quarter: str,
    round_num: int,
) -> None:
    """Print pooled vs per_class comparison."""
    print(f"\n{'='*80}")
    print(f"  R{round_num} {quarter.upper()} — Pooled vs Per-Class")
    print(f"{'='*80}")

    print(f"\n  {'Config':<14}", end="")
    print(f" {'P95 cov':>8} {'P95 err':>8} {'P50 cov':>8} {'P50 err':>8} {'P95 mean_w':>10}")
    print(f"  {'-'*14}", end="")
    print(f" {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    for name, r in [("pooled", pooled_result), ("per_class", per_class_result)]:
        agg = r["aggregate"]
        cov = agg["coverage"]["overall"]
        p95_cov = cov.get("p95", {}).get("actual", 0)
        p95_err = cov.get("p95", {}).get("error", 0)
        p50_cov = cov.get("p50", {}).get("actual", 0)
        p50_err = cov.get("p50", {}).get("error", 0)
        p95_w = agg["widths"]["overall"].get("p95", {}).get("mean_width", 0)
        print(
            f"  {name:<14}"
            f" {p95_cov:>7.2f}% {p95_err:>+7.2f}pp"
            f" {p50_cov:>7.2f}% {p50_err:>+7.2f}pp"
            f" {p95_w:>10.1f}"
        )

    # Per-class coverage for per_class method
    pc = per_class_result["aggregate"]["coverage"].get("per_class", {})
    if pc:
        print(f"\n  Per-class P95 coverage (per_class method):")
        print(f"  {'Class':<10} {'n':>10} {'Target':>8} {'Actual':>8} {'Error':>8}")
        print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
        for cls in CLASSES:
            c = pc.get(cls, {})
            p95 = c.get("p95", {})
            print(f"  {cls:<10} {c.get('n', 0):>10,} {p95.get('target', 0):>7.1f}% {p95.get('actual', 0):>7.2f}% {p95.get('error', 0):>+7.2f}pp")

    # Pooled per-class coverage for comparison
    pc_pooled = pooled_result["aggregate"]["coverage"].get("per_class", {})
    if pc_pooled:
        print(f"\n  Per-class P95 coverage (pooled method):")
        print(f"  {'Class':<10} {'n':>10} {'Target':>8} {'Actual':>8} {'Error':>8}")
        print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
        for cls in CLASSES:
            c = pc_pooled.get(cls, {})
            p95 = c.get("p95", {})
            print(f"  {cls:<10} {c.get('n', 0):>10,} {p95.get('target', 0):>7.1f}% {p95.get('actual', 0):>7.2f}% {p95.get('error', 0):>+7.2f}pp")

    # Class parity gap
    if pc:
        on_cov = pc.get("onpeak", {}).get("p95", {}).get("actual", 0)
        off_cov = pc.get("offpeak", {}).get("p95", {}).get("actual", 0)
        gap = abs(on_cov - off_cov)
        print(f"\n  BG7 class parity gap (per_class): {gap:.2f}pp (threshold <5pp)")
    if pc_pooled:
        on_cov = pc_pooled.get("onpeak", {}).get("p95", {}).get("actual", 0)
        off_cov = pc_pooled.get("offpeak", {}).get("p95", {}).get("actual", 0)
        gap = abs(on_cov - off_cov)
        print(f"  BG7 class parity gap (pooled):    {gap:.2f}pp")


# ─── Round runner ────────────────────────────────────────────────────────────


def run_round(
    round_num: int,
    n_bins: int,
    baseline_col: str,
    data_loader,
    pys: list[int],
    part_name: str,
    version_id: str,
    prior_version_part: str | None = None,
) -> dict:
    """Run v3 per-class experiment for one round."""
    print(f"\n{'#'*80}")
    print(f"  ROUND {round_num} — PER-CLASS BAND CALIBRATION (v3)")
    print(f"{'#'*80}")

    all_quarter_pooled = {}
    all_quarter_perclass = {}

    for quarter in QUARTERS:
        df = data_loader(quarter)
        print(f"\n  R{round_num} {quarter.upper()}: {df.height:,} rows")
        print(f"  Memory: {mem_mb():.0f} MB")

        # Class distribution
        for cls in CLASSES:
            n = df.filter(pl.col(CLASS_COL) == cls).height
            print(f"    {cls}: {n:,}")

        available_pys = sorted(df[PY_COL].unique().to_list())
        pys_to_use = [py for py in pys if py in available_pys]

        # Pooled (v2 reproduction)
        print(f"\n  Running pooled...", end="", flush=True)
        pooled_result = loo_per_class_quantile(
            df, quarter, pys_to_use, n_bins, baseline_col,
        )
        # For pooled, re-run with standard (non-per-class) calibration
        pooled_result_std = loo_band_calibration_quantile(
            df, quarter, pys_to_use, n_bins,
            baseline_col=baseline_col,
            mcp_col=MCP_COL,
        )
        # Add per-class coverage from pooled bands
        all_test_dfs = []
        for test_py in pys_to_use:
            train = df.filter(pl.col(PY_COL) != test_py)
            test = df.filter(pl.col(PY_COL) == test_py)
            if test.height == 0:
                continue
            boundaries, labels = compute_quantile_boundaries(train[baseline_col], n_bins)
            bin_widths = calibrate_bin_widths(
                train, baseline_col, MCP_COL, boundaries, labels, COVERAGE_LEVELS,
            )
            test_banded = apply_bands(test, bin_widths, baseline_col, boundaries, labels)
            all_test_dfs.append(test_banded)
        if all_test_dfs:
            all_test_pooled = pl.concat(all_test_dfs)
            pooled_per_class = evaluate_per_class_coverage(
                all_test_pooled, MCP_COL, CLASS_COL, COVERAGE_LEVELS,
            )
            pooled_result_std["aggregate"]["coverage"]["per_class"] = pooled_per_class
            del all_test_pooled
        p95_w = pooled_result_std["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width", 0)
        p95_cov = pooled_result_std["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
        print(f" P95 cov={p95_cov:.2f}%, width={p95_w:.1f}")
        all_quarter_pooled[quarter] = pooled_result_std

        # Per-class
        print(f"  Running per_class...", end="", flush=True)
        perclass_result = loo_per_class_quantile(
            df, quarter, pys_to_use, n_bins, baseline_col,
        )
        p95_w = perclass_result["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width", 0)
        p95_cov = perclass_result["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
        print(f" P95 cov={p95_cov:.2f}%, width={p95_w:.1f}")
        all_quarter_perclass[quarter] = perclass_result

        # Comparison
        print_comparison(pooled_result_std, perclass_result, quarter, round_num)

        del df
        gc.collect()
        print(f"\n  Memory: {mem_mb():.0f} MB")

    # ─── Temporal validation on per_class ─────────────────────────────

    print(f"\n{'='*70}")
    print(f"  R{round_num} TEMPORAL VALIDATION (per_class)")
    print(f"{'='*70}")

    temporal_results = {}
    for quarter in QUARTERS:
        df = data_loader(quarter)
        available_pys = sorted(df[PY_COL].unique().to_list())
        test_pys = [py for py in pys if py in available_pys]

        result = temporal_per_class_quantile(
            df, quarter, test_pys, n_bins, baseline_col,
        )
        temporal_results[quarter] = result

        print(f"\n  {quarter.upper()}:")
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

    # LOO vs Temporal
    print(f"\n  LOO vs Temporal P95 (per_class):")
    print(f"  {'Quarter':<10} {'LOO cov':>8} {'Temp cov':>9} {'LOO width':>10} {'Temp width':>11}")
    print(f"  {'-'*10} {'-'*8} {'-'*9} {'-'*10} {'-'*11}")
    for q in QUARTERS:
        loo_cov = all_quarter_perclass[q]["aggregate"]["coverage"]["overall"]["p95"]["actual"]
        tmp_cov = temporal_results[q]["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
        loo_w = all_quarter_perclass[q]["aggregate"]["widths"]["overall"]["p95"]["mean_width"]
        tmp_w = temporal_results[q]["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width", 0)
        print(f"  {q:<10} {loo_cov:>7.2f}% {tmp_cov:>8.2f}% {loo_w:>10.1f} {tmp_w:>11.1f}")

    # ─── Build metrics ──────────────────────────────────────────────
    # R1: LOO for gate-facing metrics (only 6 PYs → too few temporal folds)
    # R2/R3: temporal for gate-facing metrics (7 PYs → 3 mature folds)

    winner = "per_class"
    winner_results = all_quarter_perclass
    use_temporal_primary = round_num >= 2

    metrics = {
        "coverage": {},           # gates read this
        "widths": {},
        "stability": {},
        "per_class_coverage": {},  # BG7
        "per_py": {},
        "temporal_validation": {},  # temporal detail (diagnostic for R1, source for R2/R3)
        "loo_validation": {},      # LOO detail (source for R1, diagnostic for R2/R3)
        "experiment_comparison": {},
    }

    for quarter in QUARTERS:
        t = temporal_results[quarter]
        w = winner_results[quarter]

        if use_temporal_primary:
            # R2/R3: temporal as primary gate metric
            metrics["coverage"][quarter] = t["aggregate"]["coverage"]
            metrics["widths"][quarter] = {
                "overall": t["aggregate"]["widths"]["overall"],
            }
            metrics["stability"][quarter] = t["stability"]
            metrics["per_class_coverage"][quarter] = t["aggregate"]["coverage"].get("per_class", {})
        else:
            # R1: LOO as primary gate metric
            metrics["coverage"][quarter] = w["aggregate"]["coverage"]
            metrics["widths"][quarter] = {
                "overall": w["aggregate"]["widths"]["overall"],
                "per_bin": w["aggregate"]["widths"]["per_bin"],
            }
            metrics["stability"][quarter] = w["stability"]
            metrics["per_class_coverage"][quarter] = w["aggregate"]["coverage"].get("per_class", {})

        # LOO per-PY detail
        metrics["loo_validation"][quarter] = {
            "coverage": w["aggregate"]["coverage"],
            "widths": {
                "overall": w["aggregate"]["widths"]["overall"],
                "per_bin": w["aggregate"]["widths"]["per_bin"],
            },
            "stability": w["stability"],
            "per_class_coverage": w["aggregate"]["coverage"].get("per_class", {}),
            "per_py": {
                py: {
                    "p50_coverage": pdata["coverage"]["overall"]["p50"]["actual"],
                    "p95_coverage": pdata["coverage"]["overall"]["p95"]["actual"],
                    "p95_mean_width": pdata["widths"]["p95"]["mean_width"],
                }
                for py, pdata in w["per_py"].items()
            },
        }

        # Temporal per-PY detail
        metrics["temporal_validation"][quarter] = {
            "aggregate_coverage": t["aggregate"]["coverage"]["overall"],
            "aggregate_widths": t["aggregate"]["widths"]["overall"],
            "stability": t["stability"],
            "per_py": {
                py: {
                    "train_pys": pdata["train_pys"],
                    "meets_min_train_pys": pdata["meets_min_train_pys"],
                    "n_train": pdata["n_train"],
                    "n_test": pdata["n_test"],
                    "p95_coverage": pdata["coverage"]["overall"]["p95"]["actual"],
                    "p95_mean_width": pdata["widths"]["p95"]["mean_width"],
                }
                for py, pdata in t["per_py"].items()
            },
        }

        # Per-PY summary (from whichever is primary)
        if use_temporal_primary:
            metrics["per_py"][quarter] = {
                py: {
                    "train_pys": pdata["train_pys"],
                    "meets_min_train_pys": pdata["meets_min_train_pys"],
                    "n_train": pdata["n_train"],
                    "n_test": pdata["n_test"],
                    "p95_coverage": pdata["coverage"]["overall"]["p95"]["actual"],
                    "p95_mean_width": pdata["widths"]["p95"]["mean_width"],
                }
                for py, pdata in t["per_py"].items()
            }
        else:
            metrics["per_py"][quarter] = {
                py: {
                    "p50_coverage": pdata["coverage"]["overall"]["p50"]["actual"],
                    "p95_coverage": pdata["coverage"]["overall"]["p95"]["actual"],
                    "p95_mean_width": pdata["widths"]["p95"]["mean_width"],
                }
                for py, pdata in w["per_py"].items()
            }

        # Experiment comparison (pooled vs per_class, from LOO)
        comparison = {}
        for name, results in [("pooled", all_quarter_pooled), ("per_class", all_quarter_perclass)]:
            r = results[quarter]
            comparison[name] = {
                "p95_coverage": r["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual"),
                "p95_error": r["aggregate"]["coverage"]["overall"].get("p95", {}).get("error"),
                "p50_coverage": r["aggregate"]["coverage"]["overall"].get("p50", {}).get("actual"),
                "p95_mean_width": r["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width"),
            }
            pc = r["aggregate"]["coverage"].get("per_class", {})
            if pc:
                on_cov = pc.get("onpeak", {}).get("p95", {}).get("actual", 0)
                off_cov = pc.get("offpeak", {}).get("p95", {}).get("actual", 0)
                comparison[name]["class_gap_pp"] = round(abs(on_cov - off_cov), 2)
        metrics["experiment_comparison"][quarter] = comparison

    # ─── Save ─────────────────────────────────────────────────────────

    v_dir = ROOT / "versions" / part_name / version_id
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

    baseline_ver = "v3" if round_num == 1 else "M (prior round MCP)"
    config_data = {
        "schema_version": 1,
        "version": version_id,
        "description": f"R{round_num} per-class stratified bands ({n_bins} quantile bins)",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "part": part_name,
        "baseline_version": baseline_ver,
        "method": {
            "band_type": "symmetric",
            "calibration": f"empirical quantile of |mcp - baseline| per (|baseline| bin, class_type)",
            "cv_method": "temporal_expanding" if round_num >= 2 else "LOO_by_PY",
            "gate_metric": "temporal" if round_num >= 2 else "LOO",
            "bin_scheme": f"quantile_{n_bins}bin",
            "stratification": "per_class (onpeak, offpeak)",
            "round": round_num,
        },
        "parameters": {
            "coverage_levels": [0.50, 0.70, 0.80, 0.90, 0.95],
            "n_bins": n_bins,
            "bin_boundaries": "data-driven (quantile)",
            "stratify_by": "class_type",
            "min_class_bin_rows": MIN_CLASS_BIN_ROWS,
            "band_type": "symmetric",
            "cv_method": "temporal_expanding",
            "min_train_pys": 3,
        },
        "experiments_tested": ["pooled", "per_class"],
        "data_sources": [{
            "path": "all_residuals_v2.parquet" if round_num > 1 else "crossproduct_work/aq*_all_baselines.parquet",
            "columns": [MCP_COL, baseline_col, CLASS_COL],
        }],
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

    # ─── Print final summary ──────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"  R{round_num} v3 SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Method: per_class stratified, {n_bins} quantile bins")

    gate_label = "temporal, min_train_pys=3" if use_temporal_primary else "LOO"
    print(f"\n  P95 coverage accuracy ({gate_label}):")
    print(f"  {'Quarter':<10} {'Target':>8} {'Actual':>8} {'Error':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for q in QUARTERS:
        cov = metrics["coverage"][q]["overall"]["p95"]
        print(f"  {q:<10} {cov['target']:>7.1f}% {cov['actual']:>7.2f}% {cov['error']:>+7.2f}pp")

    print(f"\n  Per-class P95 coverage (class parity):")
    print(f"  {'Quarter':<10} {'onpeak':>10} {'offpeak':>10} {'gap':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for q in QUARTERS:
        pc = metrics["per_class_coverage"][q]
        on = pc.get("onpeak", {}).get("p95", {}).get("actual", 0)
        off = pc.get("offpeak", {}).get("p95", {}).get("actual", 0)
        gap = abs(on - off)
        print(f"  {q:<10} {on:>9.2f}% {off:>9.2f}% {gap:>7.2f}pp")

    print(f"\n  P95 mean width ($/MWh):")
    print(f"  {'Quarter':<10} {'Width':>10}")
    print(f"  {'-'*10} {'-'*10}")
    for q in QUARTERS:
        w = metrics["widths"][q]["overall"]["p95"]["mean_width"]
        print(f"  {q:<10} {w:>10.1f}")

    # Compare to prior version
    if prior_version_part:
        prior_metrics_path = ROOT / "versions" / prior_version_part / "metrics.json"
        if prior_metrics_path.exists():
            with open(prior_metrics_path) as f:
                prior_metrics = json.load(f)
            print(f"\n  vs prior version ({prior_version_part}):")
            print(f"  {'Quarter':<10} {'Prior w':>10} {'v3 w':>10} {'Change':>10} {'Prior gap':>10} {'v3 gap':>8}")
            print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
            for q in QUARTERS:
                prior_w = prior_metrics["widths"][q]["overall"]["p95"]["mean_width"]
                v3_w = metrics["widths"][q]["overall"]["p95"]["mean_width"]
                pct = round((v3_w - prior_w) / prior_w * 100, 1) if prior_w else 0
                # Prior class gap
                prior_comp = metrics["experiment_comparison"][q]["pooled"]
                prior_gap = prior_comp.get("class_gap_pp", "n/a")
                v3_comp = metrics["experiment_comparison"][q]["per_class"]
                v3_gap = v3_comp.get("class_gap_pp", "n/a")
                print(f"  {q:<10} {prior_w:>10.0f} {v3_w:>10.1f} {pct:>+9.1f}% {prior_gap:>9}pp {v3_gap:>7}pp")

    return metrics


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    print(f"Bands v3: Per-Class Stratified Calibration")
    print(f"Memory at start: {mem_mb():.0f} MB")

    # ─── R1 ───────────────────────────────────────────────────────────

    def r1_loader(quarter: str) -> pl.DataFrame:
        parquet_path = R1_DATA_DIR / f"{quarter}_all_baselines.parquet"
        return (
            pl.scan_parquet(parquet_path)
            .filter(
                (pl.col(PY_COL) >= 2019)
                & pl.col("nodal_f0").is_not_null()
                & pl.col(MCP_COL).is_not_null()
            )
            .collect()
        )

    r1_metrics = run_round(
        round_num=1, n_bins=4, baseline_col="nodal_f0",
        data_loader=r1_loader, pys=R1_PYS,
        part_name="bands/r1", version_id="v3",
        prior_version_part="bands/r1/v2",
    )
    gc.collect()
    print(f"\nMemory after R1: {mem_mb():.0f} MB")

    # ─── R2 ───────────────────────────────────────────────────────────

    def r2_loader(quarter: str) -> pl.DataFrame:
        return (
            pl.scan_parquet(R2R3_DATA_PATH)
            .filter(
                (pl.col("round") == 2)
                & (pl.col("period_type") == quarter)
                & (pl.col(PY_COL) >= 2019)
                & pl.col("mtm_1st_mean").is_not_null()
                & pl.col(MCP_COL).is_not_null()
            )
            .select(["mtm_1st_mean", MCP_COL, PY_COL, "period_type", CLASS_COL, "source_id", "sink_id"])
            .collect()
        )

    r2_metrics = run_round(
        round_num=2, n_bins=6, baseline_col="mtm_1st_mean",
        data_loader=r2_loader, pys=R2R3_PYS,
        part_name="bands/r2", version_id="v2",
        prior_version_part="bands/r2/v1",
    )
    gc.collect()
    print(f"\nMemory after R2: {mem_mb():.0f} MB")

    # ─── R3 ───────────────────────────────────────────────────────────

    def r3_loader(quarter: str) -> pl.DataFrame:
        return (
            pl.scan_parquet(R2R3_DATA_PATH)
            .filter(
                (pl.col("round") == 3)
                & (pl.col("period_type") == quarter)
                & (pl.col(PY_COL) >= 2019)
                & pl.col("mtm_1st_mean").is_not_null()
                & pl.col(MCP_COL).is_not_null()
            )
            .select(["mtm_1st_mean", MCP_COL, PY_COL, "period_type", CLASS_COL, "source_id", "sink_id"])
            .collect()
        )

    r3_metrics = run_round(
        round_num=3, n_bins=6, baseline_col="mtm_1st_mean",
        data_loader=r3_loader, pys=R2R3_PYS,
        part_name="bands/r3", version_id="v2",
        prior_version_part="bands/r3/v1",
    )
    gc.collect()
    print(f"\nMemory after R3: {mem_mb():.0f} MB")

    print(f"\nDone. Memory: {mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
