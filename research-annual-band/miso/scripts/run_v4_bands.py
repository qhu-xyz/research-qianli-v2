"""Bands v4: Per-Class + Sign Stratified Band Calibration.

v3 calibrates bands per (|baseline| bin, class_type).
Investigation (investigate_sign_split.py) confirmed prevail (baseline > 0) and
counter (baseline < 0) have different residual distributions within the same
bin — counter paths have wider tails, especially in middle bins. Splitting
calibration by sign improves counter P95 coverage by +0.63pp on average.

v4 adds sign_seg (prevail/counter) as a third stratification axis:
    R1: 4 bins x 2 classes x 2 signs = 16 cells
    R2/R3: 6 bins x 2 classes x 2 signs = 24 cells

Tests per round:
    pooled          v2/v1 reproduction (pooled onpeak + offpeak)
    per_class       v3 reproduction (separate widths per class)
    per_class_sign  v4 new (separate widths per class x sign)

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual/scripts/run_v4_bands.py
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

from run_v3_bands import (
    calibrate_bin_widths_per_class,
    apply_bands_per_class_fast,
    evaluate_per_class_coverage,
    loo_per_class_quantile,
    temporal_per_class_quantile,
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
SIGN_SEGS = ["prevail", "counter"]

MIN_CELL_ROWS = 500  # min rows for (bin, class, sign) cell before fallback


# ─── Per-class-sign calibration functions ────────────────────────────────────


def add_sign_seg(df: pl.DataFrame, baseline_col: str) -> pl.DataFrame:
    """Add sign_seg column: prevail/counter/zero."""
    return df.with_columns(
        pl.when(pl.col(baseline_col) > 0).then(pl.lit("prevail"))
        .when(pl.col(baseline_col) < 0).then(pl.lit("counter"))
        .otherwise(pl.lit("zero"))
        .alias("sign_seg")
    )


def calibrate_bin_widths_per_class_sign(
    df: pl.DataFrame,
    baseline_col: str,
    mcp_col: str = MCP_COL,
    class_col: str = CLASS_COL,
    boundaries: list[float] | None = None,
    labels: list[str] | None = None,
    coverage_levels: list[float] = COVERAGE_LEVELS,
) -> dict[str, dict]:
    """Per-(bin, class, sign_seg) empirical quantile of |residual|.

    Returns {bin_label: {(class, sign): {p50: w, ...}, class: {...}, _pooled: {...}}}.

    Fallback chain: (bin, class, sign) -> (bin, class) -> (bin, _pooled).
    Zero-baseline paths use the (bin, class) fallback.
    """
    abs_res = (df[mcp_col] - df[baseline_col]).abs()
    bins = assign_bins(df[baseline_col].abs(), boundaries, labels)

    # Add sign_seg if not present
    if "sign_seg" not in df.columns:
        df = add_sign_seg(df, baseline_col)

    work = pl.DataFrame({
        "abs_residual": abs_res,
        "bin": bins,
        "class_type": df[class_col],
        "sign_seg": df["sign_seg"],
    })

    result = {}
    for label in labels:
        bin_data = work.filter(pl.col("bin") == label)

        # Pooled estimate (ultimate fallback)
        pooled_subset = bin_data["abs_residual"]
        n_pooled = len(pooled_subset)
        pooled_widths = {}
        for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
            if n_pooled > 0:
                pooled_widths[clabel] = round(float(pooled_subset.quantile(level)), 1)
            else:
                pooled_widths[clabel] = None
        pooled_widths["n"] = n_pooled

        cell_widths = {"_pooled": pooled_widths}

        # Per-class estimate (mid-level fallback)
        for cls in CLASSES:
            cls_subset = bin_data.filter(pl.col("class_type") == cls)["abs_residual"]
            n_cls = len(cls_subset)

            if n_cls >= MIN_CELL_ROWS:
                widths = {}
                for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
                    widths[clabel] = round(float(cls_subset.quantile(level)), 1)
                widths["n"] = n_cls
            else:
                widths = dict(pooled_widths)
                widths["n"] = n_cls
                widths["_fallback"] = "pooled"
            cell_widths[cls] = widths

            # Per-class-sign estimate (finest level)
            for seg in SIGN_SEGS:
                seg_subset = bin_data.filter(
                    (pl.col("class_type") == cls) & (pl.col("sign_seg") == seg)
                )["abs_residual"]
                n_seg = len(seg_subset)

                if n_seg >= MIN_CELL_ROWS:
                    seg_widths = {}
                    for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
                        seg_widths[clabel] = round(float(seg_subset.quantile(level)), 1)
                    seg_widths["n"] = n_seg
                else:
                    # Fall back to class-level
                    seg_widths = dict(cell_widths[cls])
                    seg_widths["n"] = n_seg
                    seg_widths["_fallback"] = "class"
                cell_widths[(cls, seg)] = seg_widths

        result[label] = cell_widths

    return result


def apply_bands_per_class_sign_fast(
    df: pl.DataFrame,
    bin_widths: dict[str, dict],
    baseline_col: str,
    class_col: str = CLASS_COL,
    boundaries: list[float] | None = None,
    labels: list[str] | None = None,
) -> pl.DataFrame:
    """Vectorized band application using join on (bin, class, sign_seg)."""
    abs_bl = df[baseline_col].abs()
    bins = assign_bins(abs_bl, boundaries, labels)

    # Add sign_seg if not present
    if "sign_seg" not in df.columns:
        df = add_sign_seg(df, baseline_col)

    df = df.with_columns(pl.Series("_bin", bins))

    # Build lookup table: (bin, class, sign_seg) -> widths
    rows = []
    for bin_label in labels:
        cell = bin_widths.get(bin_label, {})
        for cls in CLASSES:
            for seg in SIGN_SEGS:
                entry = {"_bin": bin_label, CLASS_COL: cls, "sign_seg": seg}
                # Lookup with fallback chain
                key = (cls, seg)
                if key in cell:
                    data = cell[key]
                elif cls in cell:
                    data = cell[cls]
                else:
                    data = cell.get("_pooled", {})
                for clabel in COVERAGE_LABELS:
                    entry[f"_w_{clabel}"] = data.get(clabel)
                rows.append(entry)
            # Also handle zero sign_seg -> fall back to class level
            entry = {"_bin": bin_label, CLASS_COL: cls, "sign_seg": "zero"}
            data = cell.get(cls, cell.get("_pooled", {}))
            for clabel in COVERAGE_LABELS:
                entry[f"_w_{clabel}"] = data.get(clabel)
            rows.append(entry)

    lookup = pl.DataFrame(rows)

    df = df.join(lookup, on=["_bin", CLASS_COL, "sign_seg"], how="left")

    for clabel in COVERAGE_LABELS:
        df = df.with_columns([
            (pl.col(baseline_col) - pl.col(f"_w_{clabel}")).alias(f"lower_{clabel}"),
            (pl.col(baseline_col) + pl.col(f"_w_{clabel}")).alias(f"upper_{clabel}"),
        ])

    drop_cols = ["_bin"] + [f"_w_{clabel}" for clabel in COVERAGE_LABELS]
    return df.drop(drop_cols)


def evaluate_per_sign_coverage(
    df: pl.DataFrame,
    mcp_col: str = MCP_COL,
    coverage_levels: list[float] = COVERAGE_LEVELS,
) -> dict:
    """Coverage broken out by sign_seg. Returns {prevail: {n, p95: {...}}, counter: {...}}."""
    result = {}
    for seg in SIGN_SEGS:
        sub = df.filter(pl.col("sign_seg") == seg)
        n = sub.height
        seg_result = {"n": n}
        if n > 0:
            sub_mcp = sub[mcp_col]
            for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
                lower = sub[f"lower_{clabel}"]
                upper = sub[f"upper_{clabel}"]
                covered = ((sub_mcp >= lower) & (sub_mcp <= upper)).mean()
                actual = round(float(covered) * 100, 2)
                target = round(level * 100, 1)
                seg_result[clabel] = {
                    "target": target,
                    "actual": actual,
                    "error": round(actual - target, 2),
                }
        result[seg] = seg_result
    return result


# ─── LOO calibration with sign split ────────────────────────────────────────


def loo_per_class_sign_quantile(
    df: pl.DataFrame,
    quarter: str,
    pys: list[int],
    n_bins: int,
    baseline_col: str,
    mcp_col: str = MCP_COL,
    class_col: str = CLASS_COL,
    coverage_levels: list[float] = COVERAGE_LEVELS,
) -> dict:
    """LOO by PY with per-class-sign quantile calibration."""
    # Pre-add sign_seg
    if "sign_seg" not in df.columns:
        df = add_sign_seg(df, baseline_col)

    per_py = {}
    all_test_dfs = []
    fold_boundaries = {}

    for test_py in pys:
        train = df.filter(pl.col(PY_COL) != test_py)
        test = df.filter(pl.col(PY_COL) == test_py)

        if test.height == 0:
            continue

        boundaries, labels = compute_quantile_boundaries(train[baseline_col], n_bins)
        fold_boundaries[str(test_py)] = boundaries

        bin_widths = calibrate_bin_widths_per_class_sign(
            train, baseline_col, mcp_col, class_col,
            boundaries, labels, coverage_levels,
        )

        test_banded = apply_bands_per_class_sign_fast(
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

        # Per-sign coverage
        per_sign_cov = evaluate_per_sign_coverage(
            test_banded, mcp_col, coverage_levels,
        )

        # Per-bin coverage
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
                for cls in CLASSES:
                    for seg in SIGN_SEGS:
                        key = (cls, seg)
                        w = bin_widths[label].get(key, {}).get(clabel)
                        if w is not None:
                            widths_for_level.append(w)
            width_summary[clabel] = {
                "mean_width": round(sum(widths_for_level) / len(widths_for_level), 1) if widths_for_level else None,
                "per_bin": {},
            }
            for label in labels:
                cls_sign_widths = {}
                for cls in CLASSES:
                    for seg in SIGN_SEGS:
                        key = (cls, seg)
                        cls_sign_widths[f"{cls}_{seg}"] = bin_widths[label].get(key, {}).get(clabel)
                vals = [v for v in cls_sign_widths.values() if v is not None]
                cls_sign_widths["avg"] = round(sum(vals) / len(vals), 1) if vals else None
                width_summary[clabel]["per_bin"][label] = cls_sign_widths

        cov_result = overall_cov
        cov_result["per_bin"] = per_bin_cov
        cov_result["per_bin_approximate"] = True
        cov_result["per_class"] = per_class_cov
        cov_result["per_sign"] = per_sign_cov

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

        # Aggregate per-bin
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

        # Aggregate per-class
        agg_per_class = evaluate_per_class_coverage(
            all_test, mcp_col, CLASS_COL, coverage_levels,
        )
        agg_coverage["per_class"] = agg_per_class

        # Aggregate per-sign
        agg_per_sign = evaluate_per_sign_coverage(
            all_test, mcp_col, coverage_levels,
        )
        agg_coverage["per_sign"] = agg_per_sign
    else:
        agg_coverage = {"overall": {}, "per_bin": {}}
        full_labels = [f"q{i+1}" for i in range(n_bins)]

    # Aggregate widths
    labels_agg = full_labels if all_test_dfs else [f"q{i+1}" for i in range(n_bins)]
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
        for label in labels_agg:
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


# ─── Temporal validation with sign split ─────────────────────────────────────


def temporal_per_class_sign_quantile(
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
    """Temporal expanding-window with per-class-sign calibration."""
    if "sign_seg" not in df.columns:
        df = add_sign_seg(df, baseline_col)

    per_py = {}
    all_test_dfs = []
    filtered_test_dfs = []

    for test_py in pys:
        train = df.filter(pl.col(PY_COL) < test_py)
        test = df.filter(pl.col(PY_COL) == test_py)

        if train.height == 0 or test.height == 0:
            continue

        train_pys = sorted(train[PY_COL].unique().to_list())
        meets_min = len(train_pys) >= min_train_pys

        boundaries, labels = compute_quantile_boundaries(train[baseline_col], n_bins)

        bin_widths = calibrate_bin_widths_per_class_sign(
            train, baseline_col, mcp_col, class_col,
            boundaries, labels, coverage_levels,
        )

        test_banded = apply_bands_per_class_sign_fast(
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
                    for seg in SIGN_SEGS:
                        key = (cls, seg)
                        w = bin_widths[label].get(key, {}).get(clabel)
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

        agg_coverage = {"overall": {}, "per_bin": {}, "per_class": {}, "per_sign": {}}
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

        # Aggregate per-bin
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

        # Aggregate per-class
        agg_per_class = evaluate_per_class_coverage(
            all_test, mcp_col, CLASS_COL, coverage_levels,
        )
        agg_coverage["per_class"] = agg_per_class

        # Aggregate per-sign
        agg_per_sign = evaluate_per_sign_coverage(
            all_test, mcp_col, coverage_levels,
        )
        agg_coverage["per_sign"] = agg_per_sign
    else:
        agg_coverage = {"overall": {}, "per_bin": {}, "per_class": {}, "per_sign": {}}

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

    # Stability (filtered folds only)
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
    per_class_sign_result: dict,
    quarter: str,
    round_num: int,
) -> None:
    """Print pooled vs per_class vs per_class_sign comparison."""
    print(f"\n{'='*90}")
    print(f"  R{round_num} {quarter.upper()} — Pooled vs Per-Class vs Per-Class-Sign")
    print(f"{'='*90}")

    print(f"\n  {'Config':<16}", end="")
    print(f" {'P95 cov':>8} {'P95 err':>8} {'P50 cov':>8} {'P50 err':>8} {'P95 mean_w':>10}")
    print(f"  {'-'*16}", end="")
    print(f" {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    for name, r in [
        ("pooled", pooled_result),
        ("per_class", per_class_result),
        ("per_class_sign", per_class_sign_result),
    ]:
        agg = r["aggregate"]
        cov = agg["coverage"]["overall"]
        p95_cov = cov.get("p95", {}).get("actual", 0)
        p95_err = cov.get("p95", {}).get("error", 0)
        p50_cov = cov.get("p50", {}).get("actual", 0)
        p50_err = cov.get("p50", {}).get("error", 0)
        p95_w = agg["widths"]["overall"].get("p95", {}).get("mean_width", 0)
        print(
            f"  {name:<16}"
            f" {p95_cov:>7.2f}% {p95_err:>+7.2f}pp"
            f" {p50_cov:>7.2f}% {p50_err:>+7.2f}pp"
            f" {p95_w:>10.1f}"
        )

    # Per-class P95 coverage comparison
    for method_name, result in [("per_class", per_class_result), ("per_class_sign", per_class_sign_result)]:
        pc = result["aggregate"]["coverage"].get("per_class", {})
        if pc:
            print(f"\n  Per-class P95 coverage ({method_name}):")
            print(f"  {'Class':<10} {'n':>10} {'Target':>8} {'Actual':>8} {'Error':>8}")
            print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
            for cls in CLASSES:
                c = pc.get(cls, {})
                p95 = c.get("p95", {})
                print(f"  {cls:<10} {c.get('n', 0):>10,} {p95.get('target', 0):>7.1f}% {p95.get('actual', 0):>7.2f}% {p95.get('error', 0):>+7.2f}pp")

    # Per-sign P95 coverage (per_class_sign only)
    ps = per_class_sign_result["aggregate"]["coverage"].get("per_sign", {})
    if ps:
        print(f"\n  Per-sign P95 coverage (per_class_sign):")
        print(f"  {'Sign':>10} {'n':>10} {'Target':>8} {'Actual':>8} {'Error':>8}")
        print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
        for seg in SIGN_SEGS:
            c = ps.get(seg, {})
            p95 = c.get("p95", {})
            print(f"  {seg:>10} {c.get('n', 0):>10,} {p95.get('target', 0):>7.1f}% {p95.get('actual', 0):>7.2f}% {p95.get('error', 0):>+7.2f}pp")

    # Sign parity gap
    if ps:
        prev_cov = ps.get("prevail", {}).get("p95", {}).get("actual", 0)
        ctr_cov = ps.get("counter", {}).get("p95", {}).get("actual", 0)
        sign_gap = abs(prev_cov - ctr_cov)
        print(f"\n  Sign parity gap (per_class_sign): {sign_gap:.2f}pp")

    # Class parity gap
    pc_sign = per_class_sign_result["aggregate"]["coverage"].get("per_class", {})
    if pc_sign:
        on_cov = pc_sign.get("onpeak", {}).get("p95", {}).get("actual", 0)
        off_cov = pc_sign.get("offpeak", {}).get("p95", {}).get("actual", 0)
        gap = abs(on_cov - off_cov)
        print(f"  Class parity gap (per_class_sign): {gap:.2f}pp (threshold <5pp)")
    pc_cls = per_class_result["aggregate"]["coverage"].get("per_class", {})
    if pc_cls:
        on_cov = pc_cls.get("onpeak", {}).get("p95", {}).get("actual", 0)
        off_cov = pc_cls.get("offpeak", {}).get("p95", {}).get("actual", 0)
        gap = abs(on_cov - off_cov)
        print(f"  Class parity gap (per_class):      {gap:.2f}pp")


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
    """Run v4 per-class-sign experiment for one round."""
    print(f"\n{'#'*80}")
    print(f"  ROUND {round_num} — PER-CLASS-SIGN BAND CALIBRATION (v4)")
    print(f"{'#'*80}")

    all_quarter_pooled = {}
    all_quarter_perclass = {}
    all_quarter_perclasssign = {}

    for quarter in QUARTERS:
        df = data_loader(quarter)
        df = add_sign_seg(df, baseline_col)
        print(f"\n  R{round_num} {quarter.upper()}: {df.height:,} rows")
        print(f"  Memory: {mem_mb():.0f} MB")

        # Class distribution
        for cls in CLASSES:
            n = df.filter(pl.col(CLASS_COL) == cls).height
            print(f"    {cls}: {n:,}")

        # Sign distribution
        for seg in SIGN_SEGS:
            n = df.filter(pl.col("sign_seg") == seg).height
            print(f"    {seg}: {n:,}")
        n_zero = df.filter(pl.col("sign_seg") == "zero").height
        if n_zero > 0:
            print(f"    zero: {n_zero:,}")

        available_pys = sorted(df[PY_COL].unique().to_list())
        pys_to_use = [py for py in pys if py in available_pys]

        # ─── Experiment 1: Pooled (v2 reproduction) ──────────────────
        print(f"\n  Running pooled...", end="", flush=True)
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
            del all_test_pooled, all_test_dfs
        p95_w = pooled_result_std["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width", 0)
        p95_cov = pooled_result_std["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
        print(f" P95 cov={p95_cov:.2f}%, width={p95_w:.1f}")
        all_quarter_pooled[quarter] = pooled_result_std

        # ─── Experiment 2: Per-class (v3 reproduction) ───────────────
        print(f"  Running per_class...", end="", flush=True)
        perclass_result = loo_per_class_quantile(
            df, quarter, pys_to_use, n_bins, baseline_col,
        )
        p95_w = perclass_result["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width", 0)
        p95_cov = perclass_result["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
        print(f" P95 cov={p95_cov:.2f}%, width={p95_w:.1f}")
        all_quarter_perclass[quarter] = perclass_result

        # ─── Experiment 3: Per-class-sign (v4 new) ───────────────────
        print(f"  Running per_class_sign...", end="", flush=True)
        perclasssign_result = loo_per_class_sign_quantile(
            df, quarter, pys_to_use, n_bins, baseline_col,
        )
        p95_w = perclasssign_result["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width", 0)
        p95_cov = perclasssign_result["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
        print(f" P95 cov={p95_cov:.2f}%, width={p95_w:.1f}")
        all_quarter_perclasssign[quarter] = perclasssign_result

        # Comparison
        print_comparison(pooled_result_std, perclass_result, perclasssign_result, quarter, round_num)

        del df
        gc.collect()
        print(f"\n  Memory: {mem_mb():.0f} MB")

    # ─── Temporal validation on per_class_sign ────────────────────────

    print(f"\n{'='*70}")
    print(f"  R{round_num} TEMPORAL VALIDATION (per_class_sign)")
    print(f"{'='*70}")

    temporal_results = {}
    for quarter in QUARTERS:
        df = data_loader(quarter)
        df = add_sign_seg(df, baseline_col)
        available_pys = sorted(df[PY_COL].unique().to_list())
        test_pys = [py for py in pys if py in available_pys]

        result = temporal_per_class_sign_quantile(
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

        # Per-sign temporal aggregate
        ps = result["aggregate"]["coverage"].get("per_sign", {})
        if ps:
            for seg in SIGN_SEGS:
                c = ps.get(seg, {})
                p95_s = c.get("p95", {})
                print(f"    {seg}: P95 cov={p95_s.get('actual', 0):.2f}% (n={c.get('n', 0):,})")

        del df
        gc.collect()

    # LOO vs Temporal
    print(f"\n  LOO vs Temporal P95 (per_class_sign):")
    print(f"  {'Quarter':<10} {'LOO cov':>8} {'Temp cov':>9} {'LOO width':>10} {'Temp width':>11}")
    print(f"  {'-'*10} {'-'*8} {'-'*9} {'-'*10} {'-'*11}")
    for q in QUARTERS:
        loo_cov = all_quarter_perclasssign[q]["aggregate"]["coverage"]["overall"]["p95"]["actual"]
        tmp_cov = temporal_results[q]["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
        loo_w = all_quarter_perclasssign[q]["aggregate"]["widths"]["overall"]["p95"]["mean_width"]
        tmp_w = temporal_results[q]["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width", 0)
        print(f"  {q:<10} {loo_cov:>7.2f}% {tmp_cov:>8.2f}% {loo_w:>10.1f} {tmp_w:>11.1f}")

    # ─── Build metrics ──────────────────────────────────────────────

    winner = "per_class_sign"
    winner_results = all_quarter_perclasssign
    use_temporal_primary = round_num >= 2

    metrics = {
        "coverage": {},
        "widths": {},
        "stability": {},
        "per_class_coverage": {},
        "per_sign_coverage": {},
        "per_py": {},
        "temporal_validation": {},
        "loo_validation": {},
        "experiment_comparison": {},
    }

    for quarter in QUARTERS:
        t = temporal_results[quarter]
        w = winner_results[quarter]

        if use_temporal_primary:
            metrics["coverage"][quarter] = t["aggregate"]["coverage"]
            metrics["widths"][quarter] = {
                "overall": t["aggregate"]["widths"]["overall"],
            }
            metrics["stability"][quarter] = t["stability"]
            metrics["per_class_coverage"][quarter] = t["aggregate"]["coverage"].get("per_class", {})
            metrics["per_sign_coverage"][quarter] = t["aggregate"]["coverage"].get("per_sign", {})
        else:
            metrics["coverage"][quarter] = w["aggregate"]["coverage"]
            metrics["widths"][quarter] = {
                "overall": w["aggregate"]["widths"]["overall"],
                "per_bin": w["aggregate"]["widths"]["per_bin"],
            }
            metrics["stability"][quarter] = w["stability"]
            metrics["per_class_coverage"][quarter] = w["aggregate"]["coverage"].get("per_class", {})
            metrics["per_sign_coverage"][quarter] = w["aggregate"]["coverage"].get("per_sign", {})

        # LOO per-PY detail
        metrics["loo_validation"][quarter] = {
            "coverage": w["aggregate"]["coverage"],
            "widths": {
                "overall": w["aggregate"]["widths"]["overall"],
                "per_bin": w["aggregate"]["widths"]["per_bin"],
            },
            "stability": w["stability"],
            "per_class_coverage": w["aggregate"]["coverage"].get("per_class", {}),
            "per_sign_coverage": w["aggregate"]["coverage"].get("per_sign", {}),
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
            "per_sign_coverage": t["aggregate"]["coverage"].get("per_sign", {}),
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

        # Per-PY summary
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

        # Experiment comparison: pooled vs per_class vs per_class_sign
        comparison = {}
        for name, results in [
            ("pooled", all_quarter_pooled),
            ("per_class", all_quarter_perclass),
            ("per_class_sign", all_quarter_perclasssign),
        ]:
            r = results[quarter]
            entry = {
                "p95_coverage": r["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual"),
                "p95_error": r["aggregate"]["coverage"]["overall"].get("p95", {}).get("error"),
                "p50_coverage": r["aggregate"]["coverage"]["overall"].get("p50", {}).get("actual"),
                "p95_mean_width": r["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width"),
            }
            pc = r["aggregate"]["coverage"].get("per_class", {})
            if pc:
                on_cov = pc.get("onpeak", {}).get("p95", {}).get("actual", 0)
                off_cov = pc.get("offpeak", {}).get("p95", {}).get("actual", 0)
                entry["class_gap_pp"] = round(abs(on_cov - off_cov), 2)
            ps = r["aggregate"]["coverage"].get("per_sign", {})
            if ps:
                prev_cov = ps.get("prevail", {}).get("p95", {}).get("actual", 0)
                ctr_cov = ps.get("counter", {}).get("p95", {}).get("actual", 0)
                entry["sign_gap_pp"] = round(abs(prev_cov - ctr_cov), 2)
            comparison[name] = entry
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
        "description": f"R{round_num} per-class-sign stratified bands ({n_bins} quantile bins)",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "part": part_name,
        "baseline_version": baseline_ver,
        "method": {
            "band_type": "symmetric",
            "calibration": f"empirical quantile of |mcp - baseline| per (|baseline| bin, class_type, sign_seg)",
            "cv_method": "temporal_expanding" if round_num >= 2 else "LOO_by_PY",
            "gate_metric": "temporal" if round_num >= 2 else "LOO",
            "bin_scheme": f"quantile_{n_bins}bin",
            "stratification": "per_class_sign (onpeak/offpeak x prevail/counter)",
            "round": round_num,
        },
        "parameters": {
            "coverage_levels": [0.50, 0.70, 0.80, 0.90, 0.95],
            "n_bins": n_bins,
            "bin_boundaries": "data-driven (quantile)",
            "stratify_by": ["class_type", "sign_seg"],
            "min_cell_rows": MIN_CELL_ROWS,
            "fallback_chain": "(bin, class, sign) -> (bin, class) -> (bin, pooled)",
            "band_type": "symmetric",
            "cv_method": "temporal_expanding",
            "min_train_pys": 3,
        },
        "experiments_tested": ["pooled", "per_class", "per_class_sign"],
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
    print(f"  R{round_num} v4 SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Method: per_class_sign stratified, {n_bins} quantile bins")

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

    print(f"\n  Per-sign P95 coverage (sign parity):")
    print(f"  {'Quarter':<10} {'prevail':>10} {'counter':>10} {'gap':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for q in QUARTERS:
        ps = metrics["per_sign_coverage"][q]
        prev = ps.get("prevail", {}).get("p95", {}).get("actual", 0)
        ctr = ps.get("counter", {}).get("p95", {}).get("actual", 0)
        gap = abs(prev - ctr)
        print(f"  {q:<10} {prev:>9.2f}% {ctr:>9.2f}% {gap:>7.2f}pp")

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
            print(f"  {'Quarter':<10} {'Prior w':>10} {'v4 w':>10} {'Change':>10} {'Prior cls gap':>14} {'v4 cls gap':>11} {'v4 sign gap':>12}")
            print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*14} {'-'*11} {'-'*12}")
            for q in QUARTERS:
                prior_w = prior_metrics["widths"][q]["overall"]["p95"]["mean_width"]
                v4_w = metrics["widths"][q]["overall"]["p95"]["mean_width"]
                pct = round((v4_w - prior_w) / prior_w * 100, 1) if prior_w else 0
                # Prior class gap (from pooled experiment)
                prior_comp = metrics["experiment_comparison"][q].get("per_class", {})
                prior_gap = prior_comp.get("class_gap_pp", "n/a")
                v4_comp = metrics["experiment_comparison"][q].get("per_class_sign", {})
                v4_cls_gap = v4_comp.get("class_gap_pp", "n/a")
                v4_sign_gap = v4_comp.get("sign_gap_pp", "n/a")
                print(f"  {q:<10} {prior_w:>10.0f} {v4_w:>10.1f} {pct:>+9.1f}% {prior_gap:>13}pp {v4_cls_gap:>10}pp {v4_sign_gap:>11}pp")

    return metrics


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    print(f"Bands v4: Per-Class-Sign Stratified Calibration")
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
        part_name="bands/r1", version_id="v4",
        prior_version_part="bands/r1/v3",
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
        part_name="bands/r2", version_id="v3",
        prior_version_part="bands/r2/v2",
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
        part_name="bands/r3", version_id="v3",
        prior_version_part="bands/r3/v2",
    )
    gc.collect()
    print(f"\nMemory after R3: {mem_mb():.0f} MB")

    print(f"\nDone. Memory: {mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
