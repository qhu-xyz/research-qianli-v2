"""Bands v9: Simplified Asymmetric (Per-Class Only, No Sign Split).

Asymmetric signed quantile bands with per-(bin, class) calibration.
5 quantile bins x 2 classes = 10 cells per quarter per round.
No sign stratification, no correction.
Temporal expanding CV only (no LOO), min_train_pys=2 for dev.
8 coverage levels: P10, P30, P50, P70, P80, P90, P95, P99.

QUARTERLY SCALE: Target = mcp (quarterly clearing price).
Baselines scaled to quarterly: nodal_f0 * 3 for R1, mtm_1st_mean * 3 for R2/R3.
All band widths are natively in quarterly $/MWh — no monthly-to-quarterly conversion needed.

Dev PYs only (PY 2025 reserved as holdout).

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual-band/scripts/run_v9_bands.py
"""

from __future__ import annotations

import gc
import json
import math
import os
import resource
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# ─── Constants ───────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
R1_DATA_DIR = Path("/opt/temp/qianli/annual_research/crossproduct_work")
R2R3_DATA_PATH = Path("/opt/temp/qianli/annual_research/all_residuals_v2.parquet")
QUARTERS = ["aq1", "aq2", "aq3", "aq4"]

DEV_R1_PYS = [2020, 2021, 2022, 2023, 2024]
DEV_R2R3_PYS = [2019, 2020, 2021, 2022, 2023, 2024]

MCP_COL = "mcp"  # quarterly clearing price — the economically meaningful target
PY_COL = "planning_year"
CLASS_COL = "class_type"
CLASSES = ["onpeak", "offpeak"]

COVERAGE_LEVELS = [0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]
COVERAGE_LABELS = ["p10", "p30", "p50", "p70", "p80", "p90", "p95", "p99"]

MIN_CELL_ROWS = 500
MIN_TRAIN_PYS = 2
N_BINS = 5
VERSION_ID = "v9"


# ─── Utilities ───────────────────────────────────────────────────────────────


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def sanitize_for_json(obj):
    """Replace NaN/Inf with None for JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


def assign_bins(
    abs_baseline: pl.Series,
    boundaries: list[float],
    labels: list[str],
) -> pl.Series:
    """Assign each row to a bin based on |baseline| value."""
    exprs = []
    for i, label in enumerate(labels):
        lo, hi = boundaries[i], boundaries[i + 1]
        if math.isinf(hi):
            exprs.append(pl.when(pl.col("_abs_bl") >= lo).then(pl.lit(label)))
        else:
            exprs.append(
                pl.when((pl.col("_abs_bl") >= lo) & (pl.col("_abs_bl") < hi)).then(pl.lit(label))
            )
    tmp = pl.DataFrame({"_abs_bl": abs_baseline})
    return tmp.with_columns(pl.coalesce(exprs).alias("bin"))["bin"]


def compute_quantile_boundaries(
    series: pl.Series,
    n_bins: int,
) -> tuple[list[float], list[str]]:
    """Data-driven bin boundaries from percentiles of |baseline|."""
    abs_vals = series.abs()
    quantiles = [i / n_bins for i in range(1, n_bins)]
    cuts = [0.0]
    for q in quantiles:
        cuts.append(round(float(abs_vals.quantile(q)), 1))
    cuts.append(float("inf"))
    labels = [f"q{i + 1}" for i in range(n_bins)]
    return cuts, labels


# ─── Core: Calibration ──────────────────────────────────────────────────────


def calibrate_asymmetric_per_class(
    df: pl.DataFrame,
    baseline_col: str,
    mcp_col: str = MCP_COL,
    class_col: str = CLASS_COL,
    boundaries: list[float] | None = None,
    labels: list[str] | None = None,
    coverage_levels: list[float] = COVERAGE_LEVELS,
) -> dict[str, dict]:
    """Per-(bin, class) signed quantile pairs. No sign stratification.

    Returns {bin_label: {class: {p10: (lo, hi), ...}, _pooled: {...}}}.
    Fallback: (bin, class) -> (bin, _pooled).
    """
    residual = df[mcp_col] - df[baseline_col]
    bins = assign_bins(df[baseline_col].abs(), boundaries, labels)

    work = pl.DataFrame({
        "residual": residual,
        "bin": bins,
        "class_type": df[class_col],
    })

    result = {}
    fallback_stats = {"total": 0, "to_pooled": 0}

    for label in labels:
        bin_data = work.filter(pl.col("bin") == label)

        # Pooled estimate (fallback)
        pooled_subset = bin_data["residual"]
        n_pooled = len(pooled_subset)
        pooled_pairs = {}
        for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
            if n_pooled > 0:
                lo = round(float(pooled_subset.quantile((1 - level) / 2)), 1)
                hi = round(float(pooled_subset.quantile((1 + level) / 2)), 1)
                pooled_pairs[clabel] = (lo, hi)
            else:
                pooled_pairs[clabel] = None
        pooled_pairs["n"] = n_pooled

        cell_pairs = {"_pooled": pooled_pairs}

        # Per-class estimate
        for cls in CLASSES:
            fallback_stats["total"] += 1
            cls_subset = bin_data.filter(pl.col("class_type") == cls)["residual"]
            n_cls = len(cls_subset)

            if n_cls >= MIN_CELL_ROWS:
                pairs = {}
                for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
                    lo = round(float(cls_subset.quantile((1 - level) / 2)), 1)
                    hi = round(float(cls_subset.quantile((1 + level) / 2)), 1)
                    pairs[clabel] = (lo, hi)
                pairs["n"] = n_cls
            else:
                print(f"  WARNING: Cell ({label}, {cls}) has {n_cls} rows < {MIN_CELL_ROWS}, "
                      f"falling back to pooled ({n_pooled} rows)")
                if n_pooled == 0:
                    raise ValueError(
                        f"Cell ({label}, {cls}): both class ({n_cls}) and "
                        f"pooled ({n_pooled}) have insufficient rows"
                    )
                pairs = dict(pooled_pairs)
                pairs["n"] = n_cls
                pairs["_fallback"] = "pooled"
                fallback_stats["to_pooled"] += 1
            cell_pairs[cls] = pairs

        result[label] = cell_pairs

    result["_fallback_stats"] = fallback_stats
    return result


# ─── Core: Apply Bands ──────────────────────────────────────────────────────


def apply_asymmetric_bands_per_class_fast(
    df: pl.DataFrame,
    bin_pairs: dict[str, dict],
    baseline_col: str,
    class_col: str = CLASS_COL,
    boundaries: list[float] | None = None,
    labels: list[str] | None = None,
) -> pl.DataFrame:
    """Vectorized asymmetric band application using join on (bin, class)."""
    abs_bl = df[baseline_col].abs()
    bins = assign_bins(abs_bl, boundaries, labels)

    if "_bin" in df.columns:
        df = df.drop("_bin")
    df = df.with_columns(pl.Series("_bin", bins))

    rows = []
    for bin_label in labels:
        cell = bin_pairs.get(bin_label)
        if cell is None:
            raise ValueError(f"No calibration data for bin={bin_label}")
        for cls in CLASSES:
            entry = {"_bin": bin_label, CLASS_COL: cls}
            if cls in cell:
                data = cell[cls]
            elif "_pooled" in cell:
                data = cell["_pooled"]
            else:
                raise ValueError(f"No calibration data for bin={bin_label}, class={cls}")
            for clabel in COVERAGE_LABELS:
                lo_hi = data.get(clabel)
                if isinstance(lo_hi, (list, tuple)):
                    entry[f"_lo_{clabel}"] = lo_hi[0]
                    entry[f"_hi_{clabel}"] = lo_hi[1]
                else:
                    entry[f"_lo_{clabel}"] = None
                    entry[f"_hi_{clabel}"] = None
            rows.append(entry)

    lookup = pl.DataFrame(rows)
    df = df.join(lookup, on=["_bin", CLASS_COL], how="left")

    for clabel in COVERAGE_LABELS:
        df = df.with_columns([
            (pl.col(baseline_col) + pl.col(f"_lo_{clabel}")).alias(f"lower_{clabel}"),
            (pl.col(baseline_col) + pl.col(f"_hi_{clabel}")).alias(f"upper_{clabel}"),
        ])

    drop_cols = ["_bin"]
    drop_cols += [f"_lo_{clabel}" for clabel in COVERAGE_LABELS]
    drop_cols += [f"_hi_{clabel}" for clabel in COVERAGE_LABELS]
    return df.drop(drop_cols)


# ─── Evaluation ─────────────────────────────────────────────────────────────


def evaluate_coverage(
    df: pl.DataFrame,
    mcp_col: str = MCP_COL,
    baseline_col: str = "nodal_f0",
    coverage_levels: list[float] = COVERAGE_LEVELS,
    boundaries: list[float] | None = None,
    labels: list[str] | None = None,
) -> dict:
    """Actual coverage per level, overall and per-bin."""
    mcp = df[mcp_col]
    result = {"overall": {}, "per_bin": {}}

    for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
        lower = df[f"lower_{clabel}"]
        upper = df[f"upper_{clabel}"]
        covered = ((mcp >= lower) & (mcp <= upper)).mean()
        actual = round(float(covered) * 100, 2)
        target = round(level * 100, 1)
        result["overall"][clabel] = {
            "target": target,
            "actual": actual,
            "error": round(actual - target, 2),
        }

    bins = assign_bins(df[baseline_col].abs(), boundaries, labels)
    df_with_bin = df.with_columns(pl.Series("_eval_bin", bins))

    for label in labels:
        subset = df_with_bin.filter(pl.col("_eval_bin") == label)
        n = subset.height
        bin_result = {"n": n}
        if n > 0:
            sub_mcp = subset[mcp_col]
            for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
                lower = subset[f"lower_{clabel}"]
                upper = subset[f"upper_{clabel}"]
                covered = ((sub_mcp >= lower) & (sub_mcp <= upper)).mean()
                actual = round(float(covered) * 100, 2)
                target = round(level * 100, 1)
                bin_result[clabel] = {
                    "target": target, "actual": actual,
                    "error": round(actual - target, 2),
                }
        else:
            for clabel in COVERAGE_LABELS:
                bin_result[clabel] = {"target": 0, "actual": 0, "error": 0}
        result["per_bin"][label] = bin_result

    return result


def evaluate_per_class_coverage(
    df: pl.DataFrame,
    mcp_col: str = MCP_COL,
    class_col: str = CLASS_COL,
    coverage_levels: list[float] = COVERAGE_LEVELS,
) -> dict:
    """Coverage per class at all levels."""
    result = {}
    for cls in CLASSES:
        sub = df.filter(pl.col(class_col) == cls)
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
                    "target": target, "actual": actual,
                    "error": round(actual - target, 2),
                }
        result[cls] = cls_result
    return result


# ─── Experiment Runner ───────────────────────────────────────────────────────


def run_experiment(
    df: pl.DataFrame,
    quarter: str,
    pys: list[int],
    n_bins: int,
    baseline_col: str,
    min_train_pys: int = MIN_TRAIN_PYS,
    mcp_col: str = MCP_COL,
    class_col: str = CLASS_COL,
    coverage_levels: list[float] = COVERAGE_LEVELS,
) -> dict:
    """Temporal expanding CV for asymmetric per-class bands."""
    # Validate class_type
    actual_classes = set(df[class_col].unique().to_list())
    if not actual_classes <= {"onpeak", "offpeak"}:
        raise ValueError(f"Unexpected class_type values: {actual_classes - {'onpeak', 'offpeak'}}")

    per_py = {}
    all_test_dfs = []
    filtered_test_dfs = []
    fold_boundaries = {}

    for test_py in pys:
        train = df.filter(pl.col(PY_COL) < test_py)
        test = df.filter(pl.col(PY_COL) == test_py)

        if train.height == 0 or test.height == 0:
            continue

        train_pys = sorted(train[PY_COL].unique().to_list())
        meets_min = len(train_pys) >= min_train_pys

        boundaries, bin_labels = compute_quantile_boundaries(train[baseline_col], n_bins)
        fold_boundaries[str(test_py)] = boundaries

        n_tpys = len(train_pys)
        bin_pairs = calibrate_asymmetric_per_class(
            train, baseline_col, mcp_col, class_col,
            boundaries, bin_labels, coverage_levels,
        )

        test_banded = apply_asymmetric_bands_per_class_fast(
            test, bin_pairs, baseline_col, class_col, boundaries, bin_labels,
        )

        all_test_dfs.append(test_banded)
        if meets_min:
            filtered_test_dfs.append(test_banded)

        # Overall coverage
        overall_cov = {}
        mcp_s = test_banded[mcp_col]
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

        # Per-class coverage
        per_class_cov = evaluate_per_class_coverage(test_banded, mcp_col, class_col, coverage_levels)

        # Per-bin coverage
        test_bins = assign_bins(test_banded[baseline_col].abs(), boundaries, bin_labels)
        df_with_bin = test_banded.with_columns(pl.Series("_bin", test_bins))
        per_bin_cov = {}
        for label in bin_labels:
            subset = df_with_bin.filter(pl.col("_bin") == label)
            n = subset.height
            bin_result = {"n": n}
            if n > 0:
                sub_mcp = subset[mcp_col]
                for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
                    lower_s = subset[f"lower_{clabel}"]
                    upper_s = subset[f"upper_{clabel}"]
                    cov_val = ((sub_mcp >= lower_s) & (sub_mcp <= upper_s)).mean()
                    bin_result[clabel] = {
                        "target": round(level * 100, 1),
                        "actual": round(float(cov_val) * 100, 2),
                        "error": round(float(cov_val) * 100 - level * 100, 2),
                    }
            else:
                for clabel in COVERAGE_LABELS:
                    bin_result[clabel] = {"target": 0, "actual": 0, "error": 0}
            per_bin_cov[label] = bin_result

        # Width summary
        width_summary = {}
        for clabel in COVERAGE_LABELS:
            half_widths = []
            for label in bin_labels:
                for cls in CLASSES:
                    lo_hi = bin_pairs[label].get(cls, {}).get(clabel)
                    if isinstance(lo_hi, (list, tuple)):
                        half_widths.append((lo_hi[1] - lo_hi[0]) / 2)
            width_summary[clabel] = {
                "mean_width": round(sum(half_widths) / len(half_widths), 1) if half_widths else None,
                "per_bin": {},
            }
            for label in bin_labels:
                cls_hw = {}
                for cls in CLASSES:
                    lo_hi = bin_pairs[label].get(cls, {}).get(clabel)
                    if isinstance(lo_hi, (list, tuple)):
                        cls_hw[cls] = round((lo_hi[1] - lo_hi[0]) / 2, 1)
                    else:
                        cls_hw[cls] = None
                vals = [v for v in cls_hw.values() if v is not None]
                cls_hw["avg"] = round(sum(vals) / len(vals), 1) if vals else None
                width_summary[clabel]["per_bin"][label] = cls_hw

        # Per-fold logging
        p95_cov = overall_cov["p95"]["actual"]
        p95_w = width_summary["p95"]["mean_width"]
        p99_cov = overall_cov["p99"]["actual"]
        flag = "" if meets_min else " [excluded: <min_train_pys]"
        print(f"    PY={test_py}: train={n_tpys}PY, P95={p95_cov:.1f}% w={p95_w}, "
              f"P99={p99_cov:.1f}%{flag}")

        py_entry = {
            "n_train": train.height,
            "n_test": test.height,
            "train_pys": train_pys,
            "meets_min_train_pys": meets_min,
            "coverage": {"overall": overall_cov, "per_bin": per_bin_cov, "per_class": per_class_cov},
            "widths": width_summary,
            "boundaries": [b if not math.isinf(b) else "inf" for b in boundaries],
        }
        per_py[str(test_py)] = py_entry

    # ─── Aggregate ────────────────────────────────────────────────────

    use_dfs = filtered_test_dfs if filtered_test_dfs else all_test_dfs
    agg_pys = [
        py for py in per_py
        if per_py[py].get("meets_min_train_pys", True)
    ] if any(per_py[py].get("meets_min_train_pys", True) for py in per_py) else list(per_py.keys())

    if use_dfs:
        all_test = pl.concat(use_dfs)

        agg_coverage = evaluate_coverage(
            all_test, mcp_col, baseline_col, coverage_levels,
            *compute_quantile_boundaries(df[baseline_col], n_bins),
        )
        agg_per_class = evaluate_per_class_coverage(all_test, mcp_col, class_col, coverage_levels)
        agg_coverage["per_class"] = agg_per_class
    else:
        agg_coverage = {"overall": {}, "per_bin": {}, "per_class": {}}

    # Aggregate widths (mean across filtered folds)
    agg_widths = {}
    for clabel in COVERAGE_LABELS:
        fold_means = [
            per_py[py]["widths"][clabel]["mean_width"]
            for py in agg_pys
            if per_py[py]["widths"][clabel]["mean_width"] is not None
        ]
        agg_widths[clabel] = {
            "mean_width": round(sum(fold_means) / len(fold_means), 1) if fold_means else None,
        }

    # Stability
    p95_coverages = [
        per_py[py]["coverage"]["overall"]["p95"]["actual"]
        for py in agg_pys
    ]
    p95_widths_fold = [
        per_py[py]["widths"]["p95"]["mean_width"]
        for py in agg_pys
        if per_py[py]["widths"]["p95"]["mean_width"] is not None
    ]

    if len(p95_coverages) >= 2:
        p95_range = round(max(p95_coverages) - min(p95_coverages), 2)
        worst_py = agg_pys[p95_coverages.index(min(p95_coverages))]
        p95_worst = min(p95_coverages)
    else:
        p95_range = 0
        worst_py = agg_pys[0] if agg_pys else ""
        p95_worst = p95_coverages[0] if p95_coverages else 0

    width_cv = 0.0
    if len(p95_widths_fold) >= 2:
        width_cv = round(statistics.stdev(p95_widths_fold) / statistics.mean(p95_widths_fold), 4)

    stability = {
        "p95_coverage_range": p95_range,
        "p95_worst_py": worst_py,
        "p95_worst_py_coverage": round(p95_worst, 2),
        "p95_width_cv": width_cv,
        "min_train_pys": min_train_pys,
        "n_folds_total": len(per_py),
        "n_folds_filtered": len(agg_pys),
    }

    return {
        "per_py": per_py,
        "aggregate": {"coverage": agg_coverage, "widths": agg_widths},
        "stability": stability,
    }


# ─── Round Runner ────────────────────────────────────────────────────────────


def run_round(
    round_num: int,
    baseline_col: str,
    data_loader,
    pys: list[int],
    n_bins: int = N_BINS,
    version_id: str = VERSION_ID,
) -> dict:
    """Run v9 experiment for one round. Temporal CV only."""
    print(f"\n{'#' * 80}")
    print(f"  ROUND {round_num} — v9 SIMPLIFIED ASYMMETRIC (per-class only)")
    print(f"  CV: temporal, min_train_pys={MIN_TRAIN_PYS}")
    print(f"  Bins: {n_bins} quantile, Coverage: {len(COVERAGE_LEVELS)} levels")
    print(f"{'#' * 80}")

    all_results = {}

    for quarter in QUARTERS:
        df = data_loader(quarter)
        print(f"\n  R{round_num} {quarter.upper()}: {df.height:,} rows, mem={mem_mb():.0f}MB")
        for cls in CLASSES:
            n = df.filter(pl.col(CLASS_COL) == cls).height
            print(f"    {cls}: {n:,}")

        available_pys = sorted(df[PY_COL].unique().to_list())
        pys_to_use = [py for py in pys if py in available_pys]

        result = run_experiment(
            df, quarter, pys_to_use, n_bins, baseline_col,
            min_train_pys=MIN_TRAIN_PYS,
        )
        all_results[quarter] = result

        p95_cov = result["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
        p95_w = result["aggregate"]["widths"].get("p95", {}).get("mean_width", 0)
        print(f"  -> Aggregate: P95 cov={p95_cov:.2f}%, hw={p95_w:.1f}")

        del df
        gc.collect()

    # ─── Print comparison tables ──────────────────────────────────────

    print(f"\n{'=' * 120}")
    print(f"  R{round_num} RESULTS — ALL 8 COVERAGE LEVELS (TEMPORAL)")
    print(f"{'=' * 120}")

    header = f"  {'Quarter':<8}"
    for clabel in COVERAGE_LABELS:
        header += f" {clabel:>7}"
    print(header)
    print(f"  {'-' * 8}" + f" {'-' * 7}" * len(COVERAGE_LABELS))

    for q in QUARTERS:
        line = f"  {q:<8}"
        for clabel in COVERAGE_LABELS:
            cov = all_results[q]["aggregate"]["coverage"]["overall"].get(clabel, {}).get("actual", 0)
            line += f" {cov:>6.1f}%"
        print(line)

    # Per-PY stability
    print(f"\n  Per-PY P95 coverage:")
    all_py_keys = sorted({py for q in QUARTERS for py in all_results[q]["per_py"]})
    filtered_pys = [
        py for py in all_py_keys
        if any(all_results[q]["per_py"].get(py, {}).get("meets_min_train_pys", False) for q in QUARTERS)
    ]

    header = f"  {'Quarter':<8}"
    for py in filtered_pys:
        header += f" PY{py:>5}"
    header += f" {'Range':>7} {'Worst':>7}"
    print(header)
    print(f"  {'-' * 8}" + f" {'-' * 7}" * (len(filtered_pys) + 2))

    for q in QUARTERS:
        line = f"  {q:<8}"
        covs = []
        for py in filtered_pys:
            py_data = all_results[q]["per_py"].get(py)
            if py_data and py_data.get("meets_min_train_pys", False):
                c = py_data["coverage"]["overall"]["p95"]["actual"]
                covs.append(c)
                line += f" {c:>6.1f}%"
            else:
                line += f" {'—':>7}"
        if covs:
            rng = max(covs) - min(covs)
            worst = min(covs)
            line += f" {rng:>6.1f}p {worst:>6.1f}%"
        print(line)

    # Per-bin P95 coverage
    print(f"\n  Per-bin P95 coverage:")
    bin_labels_all = [f"q{i + 1}" for i in range(n_bins)]
    header = f"  {'Quarter':<8}"
    for bl in bin_labels_all:
        header += f" {bl:>7}"
    print(header)
    print(f"  {'-' * 8}" + f" {'-' * 7}" * n_bins)

    for q in QUARTERS:
        line = f"  {q:<8}"
        per_bin = all_results[q]["aggregate"]["coverage"].get("per_bin", {})
        for bl in bin_labels_all:
            c = per_bin.get(bl, {}).get("p95", {}).get("actual", 0)
            line += f" {c:>6.1f}%"
        print(line)

    # Per-class coverage
    print(f"\n  Per-class P95 coverage:")
    print(f"  {'Quarter':<8} {'onpeak':>10} {'offpeak':>10} {'gap':>8}")
    print(f"  {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 8}")
    for q in QUARTERS:
        pc = all_results[q]["aggregate"]["coverage"].get("per_class", {})
        on = pc.get("onpeak", {}).get("p95", {}).get("actual", 0)
        off = pc.get("offpeak", {}).get("p95", {}).get("actual", 0)
        gap = abs(on - off)
        print(f"  {q:<8} {on:>9.2f}% {off:>9.2f}% {gap:>7.2f}pp")

    # Width comparison vs promoted v3
    print(f"\n  P95 half-width (vs promoted v3):")
    ref_path = ROOT / "versions" / "bands" / "v3" / f"r{round_num}" / "metrics.json"
    ref_widths = {}
    if ref_path.exists():
        with open(ref_path) as f:
            ref_metrics = json.load(f)
        for q in QUARTERS:
            # Try temporal first
            tv = ref_metrics.get("temporal_validation", {}).get(q, {})
            w = tv.get("aggregate_widths", {}).get("p95", {}).get("mean_width")
            if w is None:
                w = ref_metrics.get("widths", {}).get(q, {}).get("overall", {}).get("p95", {}).get("mean_width", 0)
            ref_widths[q] = w

    print(f"  {'Quarter':<8} {'v3 width':>10} {'v9 width':>10} {'Δ%':>8}")
    print(f"  {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 8}")
    for q in QUARTERS:
        v3_w = ref_widths.get(q, 0)
        v9_w = all_results[q]["aggregate"]["widths"].get("p95", {}).get("mean_width", 0)
        pct = round((v9_w - v3_w) / v3_w * 100, 1) if v3_w else 0
        print(f"  {q:<8} {v3_w:>10.1f} {v9_w:>10.1f} {pct:>+7.1f}%")

    # ─── Save outputs ─────────────────────────────────────────────────

    v_dir = ROOT / "versions" / "bands" / version_id / f"r{round_num}"
    v_dir.mkdir(parents=True, exist_ok=True)

    # Build metrics
    metrics = {
        "coverage": {},
        "widths": {},
        "stability": {},
        "per_class_coverage": {},
        "per_py": {},
        "temporal_validation": {},
    }

    for quarter in QUARTERS:
        r = all_results[quarter]
        metrics["coverage"][quarter] = r["aggregate"]["coverage"]
        metrics["widths"][quarter] = {"overall": r["aggregate"]["widths"]}
        metrics["stability"][quarter] = r["stability"]
        metrics["per_class_coverage"][quarter] = r["aggregate"]["coverage"].get("per_class", {})

        metrics["temporal_validation"][quarter] = {
            "aggregate_coverage": r["aggregate"]["coverage"]["overall"],
            "aggregate_widths": r["aggregate"]["widths"],
            "stability": r["stability"],
            "per_py": {
                py: {
                    "train_pys": pdata.get("train_pys", []),
                    "meets_min_train_pys": pdata.get("meets_min_train_pys", True),
                    "n_train": pdata["n_train"],
                    "n_test": pdata["n_test"],
                    "p95_coverage": pdata["coverage"]["overall"]["p95"]["actual"],
                    "p95_mean_width": pdata["widths"]["p95"]["mean_width"],
                    "p99_coverage": pdata["coverage"]["overall"]["p99"]["actual"],
                    "p99_mean_width": pdata["widths"]["p99"]["mean_width"],
                    "per_bin_p95": {
                        bl: pdata["coverage"]["per_bin"].get(bl, {}).get("p95", {}).get("actual", 0)
                        for bl in [f"q{i+1}" for i in range(n_bins)]
                    },
                    "per_class_p95": {
                        cls: pdata["coverage"]["per_class"].get(cls, {}).get("p95", {}).get("actual", 0)
                        for cls in CLASSES
                    },
                }
                for py, pdata in r["per_py"].items()
            },
        }

        metrics["per_py"][quarter] = {
            py: {
                "train_pys": pdata.get("train_pys", []),
                "meets_min_train_pys": pdata.get("meets_min_train_pys", True),
                "n_train": pdata["n_train"],
                "n_test": pdata["n_test"],
                "p95_coverage": pdata["coverage"]["overall"]["p95"]["actual"],
                "p95_mean_width": pdata["widths"]["p95"]["mean_width"],
            }
            for py, pdata in r["per_py"].items()
        }

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
    print(f"\n  Saved: {metrics_path}")

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
        "description": f"R{round_num} simplified asymmetric bands ({n_bins} quantile bins, per-class only, no correction)",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "part": "bands",
        "round": f"r{round_num}",
        "baseline_version": baseline_ver,
        "method": {
            "band_type": "asymmetric",
            "calibration": "signed quantile pair of (mcp - baseline) per (bin, class)",
            "cv_method": "temporal_only",
            "bin_scheme": f"quantile_{n_bins}bin",
            "stratification": "per_class (onpeak/offpeak)",
            "correction_type": "none",
            "round": round_num,
        },
        "parameters": {
            "coverage_levels": COVERAGE_LEVELS,
            "n_bins": n_bins,
            "bin_boundaries": "data-driven (quantile)",
            "stratify_by": ["class_type"],
            "min_cell_rows": MIN_CELL_ROWS,
            "fallback_chain": "(bin, class) -> (bin, pooled)",
            "band_type": "asymmetric",
            "quantile_method": "signed residual quantiles at alpha_lo=(1-level)/2, alpha_hi=(1+level)/2",
            "cv_method": "temporal_only",
            "min_train_pys": MIN_TRAIN_PYS,
            "correction_type": "none",
        },
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
    print(f"  Saved: {config_path}")

    # calibration_artifact.json — full-data calibration for production inference
    artifact = {
        "version": version_id,
        "round": round_num,
        "method": "asymmetric_per_class",
        "n_bins": n_bins,
        "calibration": {},
    }
    for quarter in QUARTERS:
        qdf = data_loader(quarter)
        all_pys_q = sorted(qdf[PY_COL].unique().to_list())
        boundaries, bin_labels = compute_quantile_boundaries(qdf[baseline_col], n_bins)
        bin_pairs = calibrate_asymmetric_per_class(
            qdf, baseline_col, MCP_COL, CLASS_COL,
            boundaries, bin_labels, COVERAGE_LEVELS,
        )
        artifact["calibration"][quarter] = {
            "boundaries": [b if not math.isinf(b) else "inf" for b in boundaries],
            "bin_labels": bin_labels,
            "bin_pairs": sanitize_for_json(bin_pairs),
            "n_rows": qdf.height,
            "pys_used": all_pys_q,
        }
        del qdf
        gc.collect()

    artifact_path = v_dir / "calibration_artifact.json"
    tmp = artifact_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(sanitize_for_json(artifact), f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, artifact_path)
    print(f"  Saved: {artifact_path}")

    return metrics


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    print(f"Bands v9: Simplified Asymmetric (per-class only, dev PYs)")
    print(f"Coverage levels: {COVERAGE_LABELS}")
    print(f"min_train_pys={MIN_TRAIN_PYS}, n_bins={N_BINS}")
    print(f"Memory at start: {mem_mb():.0f} MB")

    # ─── R1 ───────────────────────────────────────────────────────────

    def r1_loader(quarter: str) -> pl.DataFrame:
        parquet_path = R1_DATA_DIR / f"{quarter}_all_baselines.parquet"
        df = (
            pl.scan_parquet(parquet_path)
            .filter(
                (pl.col(PY_COL) >= 2019)
                & pl.col("nodal_f0").is_not_null()
                & pl.col("mcp_mean").is_not_null()
            )
            .collect()
        )
        # mcp_mean in R1 baselines was patched to monthly (mcp/3).
        # Reconstruct quarterly mcp = mcp_mean * 3
        df = df.with_columns((pl.col("mcp_mean") * 3).alias("mcp"))
        # Scale baseline to quarterly (nodal_f0 is monthly avg of 3 delivery months)
        df = df.with_columns((pl.col("nodal_f0") * 3).alias("baseline_q"))
        return df

    r1_metrics = run_round(
        round_num=1, baseline_col="baseline_q",
        data_loader=r1_loader, pys=DEV_R1_PYS,
    )
    gc.collect()
    print(f"\nMemory after R1: {mem_mb():.0f} MB")

    # ─── R2 ───────────────────────────────────────────────────────────

    def r2_loader(quarter: str) -> pl.DataFrame:
        df = (
            pl.scan_parquet(R2R3_DATA_PATH)
            .filter(
                (pl.col("round") == 2)
                & (pl.col("period_type") == quarter)
                & (pl.col(PY_COL) >= 2019)
                & pl.col("mtm_1st_mean").is_not_null()
                & pl.col("mcp").is_not_null()
            )
            .select(["mtm_1st_mean", "mcp", PY_COL, "period_type", CLASS_COL, "source_id", "sink_id"])
            .collect()
        )
        # Scale baseline to quarterly (mtm_1st_mean is monthly)
        df = df.with_columns((pl.col("mtm_1st_mean") * 3).alias("baseline_q"))
        return df

    r2_metrics = run_round(
        round_num=2, baseline_col="baseline_q",
        data_loader=r2_loader, pys=DEV_R2R3_PYS,
    )
    gc.collect()
    print(f"\nMemory after R2: {mem_mb():.0f} MB")

    # ─── R3 ───────────────────────────────────────────────────────────

    def r3_loader(quarter: str) -> pl.DataFrame:
        df = (
            pl.scan_parquet(R2R3_DATA_PATH)
            .filter(
                (pl.col("round") == 3)
                & (pl.col("period_type") == quarter)
                & (pl.col(PY_COL) >= 2019)
                & pl.col("mtm_1st_mean").is_not_null()
                & pl.col("mcp").is_not_null()
            )
            .select(["mtm_1st_mean", "mcp", PY_COL, "period_type", CLASS_COL, "source_id", "sink_id"])
            .collect()
        )
        # Scale baseline to quarterly (mtm_1st_mean is monthly)
        df = df.with_columns((pl.col("mtm_1st_mean") * 3).alias("baseline_q"))
        return df

    r3_metrics = run_round(
        round_num=3, baseline_col="baseline_q",
        data_loader=r3_loader, pys=DEV_R2R3_PYS,
    )
    gc.collect()
    print(f"\nMemory after R3: {mem_mb():.0f} MB")

    print(f"\nDone. Memory: {mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
