"""Bands v8: Production Candidate (5 bins, bidirectional correction).

v7 explored 6/8/10 bins with bidirectional OOS correction. Cell size analysis
showed 5 bins is optimal for R1's limited data (~130-150k rows, 6 PYs):
  - 5 bins: 4-5 cells below 500-row fallback in 1-PY folds
  - 8 bins: 15-18 cells below 500-row fallback in 1-PY folds
  - Width difference vs 6 bins: ~1%

v8 is a minimal fork of v7 — same core functions, 5-bin experiment matrix only.
R2/R3 have 7-8x more data — any bin count from 5-10 is fine.

Method:
  - Asymmetric signed quantile bands with per-(bin, class, sign) calibration
  - Bidirectional OOS correction (shrink over-covering, expand under-covering)
  - Temporal CV for ALL rounds
  - Cold-start inflation for folds with <3 training PYs
  - 5 quantile bins

Experiments per round (identical across R1/R2/R3):
    asym_5b         — asymmetric, 5 bins, no correction
    asym_5b_bidir   — asymmetric, 5 bins, bidirectional correction

Winner = lowest mean P95 half-width across quarters, subject to BG1 pass.
All rounds use temporal CV.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual/scripts/run_v8_bands.py
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
R1_PYS = [2020, 2021, 2022, 2023, 2024, 2025]
R2R3_PYS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

MCP_COL = "mcp_mean"
PY_COL = "planning_year"
CLASS_COL = "class_type"
CLASSES = ["onpeak", "offpeak"]
SIGN_SEGS = ["prevail", "counter"]

COVERAGE_LEVELS = [0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]
COVERAGE_LABELS = ["p10", "p30", "p50", "p70", "p80", "p90", "p95"]

MIN_CELL_ROWS = 500

# BG1 tolerance per level
BG1_TOLERANCE = {
    "p10": 5.0, "p30": 5.0, "p50": 5.0,
    "p70": 5.0, "p80": 5.0, "p90": 5.0, "p95": 5.0,
}

# Bidirectional per-bin correction parameters (OOS-based in v7.1)
CORRECTION_TOLERANCE = 2.0   # pp — tolerance for OOS coverage measurement (wider for noisy validation)
CORRECTION_STEP = 0.05       # 5% per iteration (larger steps for bigger OOS deficits)
CORRECTION_MAX_ITER = 50     # max iterations per cell
COLD_START_BOOST = 0.15      # inflation factor for cold-start folds (1-2 training PYs)
MIN_CORRECTION_PYS = 3       # minimum training PYs for OOS holdout correction


# ─── Shared functions ────────────────────────────────────────────────────────


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def assign_bins(
    abs_baseline: pl.Series,
    boundaries: list[float],
    labels: list[str],
) -> pl.Series:
    """Assign each row to a bin based on |baseline| value. Returns String series."""
    exprs = []
    for i, label in enumerate(labels):
        lo, hi = boundaries[i], boundaries[i + 1]
        if math.isinf(hi):
            exprs.append(
                pl.when(pl.col("_abs_bl") >= lo).then(pl.lit(label))
            )
        else:
            exprs.append(
                pl.when((pl.col("_abs_bl") >= lo) & (pl.col("_abs_bl") < hi)).then(pl.lit(label))
            )

    tmp = pl.DataFrame({"_abs_bl": abs_baseline})
    result = tmp.with_columns(
        pl.coalesce(exprs).alias("bin")
    )["bin"]
    return result


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


def evaluate_coverage(
    df: pl.DataFrame,
    mcp_col: str = MCP_COL,
    baseline_col: str = "nodal_f0",
    coverage_levels: list[float] = COVERAGE_LEVELS,
    boundaries: list[float] | None = None,
    labels: list[str] | None = None,
) -> dict:
    """Actual coverage per level, overall and per-bin."""
    if boundaries is None:
        boundaries = [0, 50, 250, 1000, float("inf")]
    if labels is None:
        labels = ["tiny", "small", "medium", "large"]

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
    df_with_bin = df.with_columns(pl.Series("_bin", bins))

    for label in labels:
        subset = df_with_bin.filter(pl.col("_bin") == label)
        n = subset.height
        bin_result = {"n": n}
        if n == 0:
            for clabel in COVERAGE_LABELS:
                bin_result[clabel] = {"target": 0, "actual": 0, "error": 0}
        else:
            sub_mcp = subset[mcp_col]
            for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
                lower = subset[f"lower_{clabel}"]
                upper = subset[f"upper_{clabel}"]
                covered = ((sub_mcp >= lower) & (sub_mcp <= upper)).mean()
                actual = round(float(covered) * 100, 2)
                target = round(level * 100, 1)
                bin_result[clabel] = {
                    "target": target,
                    "actual": actual,
                    "error": round(actual - target, 2),
                }
        result["per_bin"][label] = bin_result

    return result


def evaluate_per_class_coverage(
    df: pl.DataFrame,
    mcp_col: str = MCP_COL,
    class_col: str = CLASS_COL,
    coverage_levels: list[float] = COVERAGE_LEVELS,
) -> dict:
    """Coverage per class. Returns {onpeak: {p50: {target, actual, error}, ...}, offpeak: {...}}."""
    result = {}
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


def add_sign_seg(df: pl.DataFrame, baseline_col: str) -> pl.DataFrame:
    """Add sign_seg column: prevail/counter/zero."""
    return df.with_columns(
        pl.when(pl.col(baseline_col) > 0).then(pl.lit("prevail"))
        .when(pl.col(baseline_col) < 0).then(pl.lit("counter"))
        .otherwise(pl.lit("zero"))
        .alias("sign_seg")
    )


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


# ─── Asymmetric calibration functions ────────────────────────────────────────


def calibrate_asymmetric_per_class_sign(
    df: pl.DataFrame,
    baseline_col: str,
    mcp_col: str = MCP_COL,
    class_col: str = CLASS_COL,
    boundaries: list[float] | None = None,
    labels: list[str] | None = None,
    coverage_levels: list[float] = COVERAGE_LEVELS,
) -> dict[str, dict]:
    """Per-(bin, class, sign_seg) signed quantile pairs.

    Returns {bin_label: {(class, sign): {p50: (lo, hi), ...}, class: {...}, _pooled: {...}}}.

    Fallback chain: (bin, class, sign) -> (bin, class) -> (bin, _pooled).
    Zero-baseline paths use the (bin, class) fallback.
    """
    residual = df[mcp_col] - df[baseline_col]
    bins = assign_bins(df[baseline_col].abs(), boundaries, labels)

    if "sign_seg" not in df.columns:
        df = add_sign_seg(df, baseline_col)

    work = pl.DataFrame({
        "residual": residual,
        "bin": bins,
        "class_type": df[class_col],
        "sign_seg": df["sign_seg"],
    })

    result = {}
    fallback_stats = {"total": 0, "to_class": 0, "to_pooled": 0}

    for label in labels:
        bin_data = work.filter(pl.col("bin") == label)

        # Pooled estimate (ultimate fallback)
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

        # Per-class estimate (mid-level fallback)
        for cls in CLASSES:
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
                pairs = dict(pooled_pairs)
                pairs["n"] = n_cls
                pairs["_fallback"] = "pooled"
                fallback_stats["to_pooled"] += 1
            cell_pairs[cls] = pairs

            # Per-class-sign estimate (finest level)
            for seg in SIGN_SEGS:
                fallback_stats["total"] += 1
                seg_subset = bin_data.filter(
                    (pl.col("class_type") == cls) & (pl.col("sign_seg") == seg)
                )["residual"]
                n_seg = len(seg_subset)

                if n_seg >= MIN_CELL_ROWS:
                    seg_pairs = {}
                    for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
                        lo = round(float(seg_subset.quantile((1 - level) / 2)), 1)
                        hi = round(float(seg_subset.quantile((1 + level) / 2)), 1)
                        seg_pairs[clabel] = (lo, hi)
                    seg_pairs["n"] = n_seg
                else:
                    seg_pairs = dict(cell_pairs[cls])
                    seg_pairs["n"] = n_seg
                    seg_pairs["_fallback"] = "class"
                    fallback_stats["to_class"] += 1
                cell_pairs[(cls, seg)] = seg_pairs

        result[label] = cell_pairs

    result["_fallback_stats"] = fallback_stats
    return result


def apply_asymmetric_bands_per_class_sign_fast(
    df: pl.DataFrame,
    bin_pairs: dict[str, dict],
    baseline_col: str,
    class_col: str = CLASS_COL,
    boundaries: list[float] | None = None,
    labels: list[str] | None = None,
) -> pl.DataFrame:
    """Vectorized asymmetric band application using join on (bin, class, sign_seg)."""
    abs_bl = df[baseline_col].abs()
    bins = assign_bins(abs_bl, boundaries, labels)

    if "sign_seg" not in df.columns:
        df = add_sign_seg(df, baseline_col)

    if "_bin" in df.columns:
        df = df.drop("_bin")
    df = df.with_columns(pl.Series("_bin", bins))

    # Build lookup table: (bin, class, sign_seg) -> (lo, hi) per level
    rows = []
    for bin_label in labels:
        cell = bin_pairs.get(bin_label, {})
        for cls in CLASSES:
            for seg in SIGN_SEGS:
                entry = {"_bin": bin_label, CLASS_COL: cls, "sign_seg": seg}
                key = (cls, seg)
                if key in cell:
                    data = cell[key]
                elif cls in cell:
                    data = cell[cls]
                else:
                    data = cell.get("_pooled", {})
                for clabel in COVERAGE_LABELS:
                    lo_hi = data.get(clabel)
                    if isinstance(lo_hi, (list, tuple)):
                        entry[f"_lo_{clabel}"] = lo_hi[0]
                        entry[f"_hi_{clabel}"] = lo_hi[1]
                    else:
                        entry[f"_lo_{clabel}"] = None
                        entry[f"_hi_{clabel}"] = None
                rows.append(entry)
            # Handle zero sign_seg -> fall back to class level
            entry = {"_bin": bin_label, CLASS_COL: cls, "sign_seg": "zero"}
            data = cell.get(cls, cell.get("_pooled", {}))
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
    df = df.join(lookup, on=["_bin", CLASS_COL, "sign_seg"], how="left")

    for clabel in COVERAGE_LABELS:
        df = df.with_columns([
            (pl.col(baseline_col) + pl.col(f"_lo_{clabel}")).alias(f"lower_{clabel}"),
            (pl.col(baseline_col) + pl.col(f"_hi_{clabel}")).alias(f"upper_{clabel}"),
        ])

    drop_cols = ["_bin"]
    drop_cols += [f"_lo_{clabel}" for clabel in COVERAGE_LABELS]
    drop_cols += [f"_hi_{clabel}" for clabel in COVERAGE_LABELS]
    return df.drop(drop_cols)


# ─── Bidirectional per-bin correction (v7 new) ───────────────────────────────


def bidirectional_bin_correction(
    train_df: pl.DataFrame,
    bin_pairs: dict[str, dict],
    baseline_col: str,
    boundaries: list[float],
    labels: list[str],
    mcp_col: str = MCP_COL,
    class_col: str = CLASS_COL,
    coverage_levels: list[float] = COVERAGE_LEVELS,
    coverage_labels: list[str] = COVERAGE_LABELS,
    tolerance: float = CORRECTION_TOLERANCE,
    step: float = CORRECTION_STEP,
    max_iter: int = CORRECTION_MAX_ITER,
) -> dict[str, dict]:
    """Bidirectional correction: shrink over-covering cells, expand under-covering.

    Args:
        train_df: Dataframe to measure coverage on. In practice this should be
            OOS validation data (e.g. inner temporal holdout) so that coverage
            measurement is genuine out-of-sample, not in-sample exact.
        bin_pairs: Pairs calibrated on a (possibly different) training set to correct.

    For each (bin, class, sign) cell and each coverage level:
      1. Measure coverage of bin_pairs on train_df
      2. If coverage > target + tolerance: shrink lo/hi inward (tighten)
         - Safety: stop if coverage drops below target
      3. If coverage < target - tolerance: expand lo/hi outward (widen)
      4. Repeat until within tolerance or max_iter reached

    Returns corrected bin_pairs dict (same structure as input).
    """
    import copy
    corrected = copy.deepcopy(bin_pairs)

    if "sign_seg" not in train_df.columns:
        train_df = add_sign_seg(train_df, baseline_col)

    bins = assign_bins(train_df[baseline_col].abs(), boundaries, labels)
    train_work = train_df.with_columns(pl.Series("_bin", bins))

    n_shrunk = 0
    n_expanded = 0
    n_unchanged = 0
    shrink_pcts = []
    expand_pcts = []

    for bin_label in labels:
        cell = corrected.get(bin_label)
        if cell is None:
            continue

        for cls in CLASSES:
            for seg in SIGN_SEGS:
                key = (cls, seg)

                # Get the data entry for this cell
                if key not in cell:
                    continue
                data = cell[key]

                # Skip if this cell is a fallback (it shares data with parent)
                if "_fallback" in data:
                    continue

                # Filter training data to this cell
                subset = train_work.filter(
                    (pl.col("_bin") == bin_label)
                    & (pl.col(class_col) == cls)
                    & (pl.col("sign_seg") == seg)
                )
                if subset.height < 10:
                    continue

                mcp_vals = subset[mcp_col]
                bl_vals = subset[baseline_col]

                for level, clabel in zip(coverage_levels, coverage_labels):
                    lo_hi = data.get(clabel)
                    if not isinstance(lo_hi, (list, tuple)):
                        continue

                    lo, hi = lo_hi
                    orig_lo, orig_hi = lo, hi
                    target_pct = level * 100

                    # Measure in-sample coverage
                    lower = bl_vals + lo
                    upper = bl_vals + hi
                    actual_pct = float(((mcp_vals >= lower) & (mcp_vals <= upper)).mean()) * 100

                    deficit = target_pct - actual_pct  # positive = under-covering

                    if deficit > tolerance:
                        # EXPAND: under-covering — widen bands outward
                        for _ in range(max_iter):
                            lo = lo * (1 + step) if lo < 0 else lo - abs(lo) * step if lo > 0 else lo - step
                            hi = hi * (1 + step) if hi > 0 else hi + abs(hi) * step if hi < 0 else hi + step

                            lower = bl_vals + lo
                            upper = bl_vals + hi
                            actual_pct = float(((mcp_vals >= lower) & (mcp_vals <= upper)).mean()) * 100

                            if actual_pct >= target_pct - tolerance:
                                break

                        data[clabel] = (round(lo, 1), round(hi, 1))
                        orig_width = orig_hi - orig_lo
                        new_width = hi - lo
                        if orig_width > 0:
                            expand_pcts.append((new_width - orig_width) / orig_width * 100)
                        n_expanded += 1

                    elif deficit < -tolerance:
                        # SHRINK: over-covering — tighten bands inward
                        for _ in range(max_iter):
                            prev_lo, prev_hi = lo, hi
                            # Move lo toward 0 (less negative)
                            lo = lo * (1 - step) if lo < 0 else lo + abs(lo) * step if lo > 0 else lo + step
                            # Move hi toward 0 (less positive)
                            hi = hi * (1 - step) if hi > 0 else hi - abs(hi) * step if hi < 0 else hi - step

                            lower = bl_vals + lo
                            upper = bl_vals + hi
                            new_pct = float(((mcp_vals >= lower) & (mcp_vals <= upper)).mean()) * 100

                            if new_pct < target_pct:
                                # Safety: revert — went too far
                                lo, hi = prev_lo, prev_hi
                                break

                            actual_pct = new_pct
                            if actual_pct <= target_pct + tolerance:
                                break

                        data[clabel] = (round(lo, 1), round(hi, 1))
                        orig_width = orig_hi - orig_lo
                        new_width = hi - lo
                        if orig_width > 0:
                            shrink_pcts.append((orig_width - new_width) / orig_width * 100)
                        n_shrunk += 1

                    else:
                        n_unchanged += 1

    avg_shrink = sum(shrink_pcts) / len(shrink_pcts) if shrink_pcts else 0
    avg_expand = sum(expand_pcts) / len(expand_pcts) if expand_pcts else 0
    print(f"    Correction summary: {n_shrunk} shrunk (avg {avg_shrink:.1f}%), "
          f"{n_expanded} expanded (avg {avg_expand:.1f}%), "
          f"{n_unchanged} unchanged")
    return corrected


def cold_start_inflate(
    bin_pairs: dict[str, dict],
    n_train_pys: int,
    labels: list[str],
    base_boost: float = COLD_START_BOOST,
    coverage_labels: list[str] = COVERAGE_LABELS,
) -> dict[str, dict]:
    """Inflate bands for cold-start folds with few training PYs.

    factor = 1.0 + base_boost / n_train_pys
    - 1 PY  -> 1.150x (15% wider)
    - 2 PYs -> 1.075x (7.5% wider)
    - 3 PYs -> 1.050x (5% wider) — only if called, but MIN_CORRECTION_PYS=3 uses OOS instead

    Inflation pushes lo further negative (or more negative) and hi further positive.
    """
    import copy
    factor = 1.0 + base_boost / max(n_train_pys, 1)
    inflated = copy.deepcopy(bin_pairs)

    n_inflated = 0
    for label in labels:
        cell = inflated.get(label)
        if cell is None:
            continue
        for cls in CLASSES:
            for seg in SIGN_SEGS:
                key = (cls, seg)
                if key not in cell:
                    continue
                data = cell[key]
                # Inflate even if this cell fell back to class/pooled level —
                # apply_bands looks up (cls, seg) first, so the inflated pair is used.
                for clabel in coverage_labels:
                    lo_hi = data.get(clabel)
                    if isinstance(lo_hi, (list, tuple)):
                        lo, hi = lo_hi
                        new_lo = lo * factor if lo < 0 else lo - abs(lo) * (factor - 1)
                        new_hi = hi * factor if hi > 0 else hi + abs(hi) * (factor - 1)
                        data[clabel] = (round(new_lo, 1), round(new_hi, 1))
                        n_inflated += 1

    print(f"    Cold-start inflation: factor={factor:.3f}, {n_inflated} values inflated")
    return inflated


# ─── Unified experiment runner ───────────────────────────────────────────────


def run_experiment(
    df: pl.DataFrame,
    quarter: str,
    pys: list[int],
    n_bins: int,
    baseline_col: str,
    experiment_name: str,
    cv_mode: str = "temporal",
    min_train_pys: int = 1,
    apply_correction: bool = False,
    mcp_col: str = MCP_COL,
    class_col: str = CLASS_COL,
    coverage_levels: list[float] = COVERAGE_LEVELS,
) -> dict:
    """Temporal/LOO experiment runner for asymmetric bands.

    Args:
        cv_mode: "temporal" for expanding-window (default), "loo" for leave-one-PY-out.
        min_train_pys: Minimum training PYs for temporal mode (folds with fewer are excluded from aggregation).
        apply_correction: If True, apply bidirectional per-bin correction after calibration.

    Returns same structure as v6's run_experiment (minus symmetric).
    """
    if "sign_seg" not in df.columns:
        df = add_sign_seg(df, baseline_col)

    per_py = {}
    all_test_dfs = []
    filtered_test_dfs = []
    fold_boundaries = {}

    for test_py in pys:
        if cv_mode == "loo":
            train = df.filter(pl.col(PY_COL) != test_py)
        else:
            train = df.filter(pl.col(PY_COL) < test_py)

        test = df.filter(pl.col(PY_COL) == test_py)

        if train.height == 0 or test.height == 0:
            continue

        train_pys = sorted(train[PY_COL].unique().to_list())
        meets_min = len(train_pys) >= min_train_pys if cv_mode == "temporal" else True

        boundaries, bin_labels = compute_quantile_boundaries(train[baseline_col], n_bins)
        fold_boundaries[str(test_py)] = boundaries

        # Calibrate + apply (asymmetric only in v8)
        # Correction strategy: OOS-based (inner temporal holdout) or cold-start inflation
        n_tpys = len(train_pys)
        correction_mode = "none"

        if apply_correction and n_tpys >= MIN_CORRECTION_PYS:
            # OOS correction: hold out latest training PY for coverage validation.
            # Calibrate on inner_train → measure real OOS coverage on inner_val → correct.
            latest_train_py = max(train_pys)
            inner_train = train.filter(pl.col(PY_COL) < latest_train_py)
            inner_val = train.filter(pl.col(PY_COL) == latest_train_py)

            # Calibrate on inner_train using the same boundaries (from full train)
            bin_pairs = calibrate_asymmetric_per_class_sign(
                inner_train, baseline_col, mcp_col, class_col,
                boundaries, bin_labels, coverage_levels,
            )
            # OOS coverage correction: inner_val is genuinely out-of-sample for inner_train
            bin_pairs = bidirectional_bin_correction(
                inner_val, bin_pairs, baseline_col, boundaries, bin_labels,
                mcp_col, class_col, coverage_levels, COVERAGE_LABELS,
            )
            correction_mode = f"oos_holdout(val_py={latest_train_py})"

        elif apply_correction and n_tpys < MIN_CORRECTION_PYS:
            # Cold start: calibrate on full (small) train, inflate bands
            bin_pairs = calibrate_asymmetric_per_class_sign(
                train, baseline_col, mcp_col, class_col,
                boundaries, bin_labels, coverage_levels,
            )
            bin_pairs = cold_start_inflate(bin_pairs, n_tpys, labels=bin_labels)
            correction_mode = f"cold_start(n={n_tpys})"

        else:
            # No correction: calibrate on full train
            bin_pairs = calibrate_asymmetric_per_class_sign(
                train, baseline_col, mcp_col, class_col,
                boundaries, bin_labels, coverage_levels,
            )

        test_banded = apply_asymmetric_bands_per_class_sign_fast(
            test, bin_pairs, baseline_col, class_col, boundaries, bin_labels,
        )

        all_test_dfs.append(test_banded)
        if meets_min:
            filtered_test_dfs.append(test_banded)

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
                    actual_val = round(float(cov_val) * 100, 2)
                    target_val = round(level * 100, 1)
                    bin_result[clabel] = {
                        "target": target_val, "actual": actual_val,
                        "error": round(actual_val - target_val, 2),
                    }
            else:
                for clabel in COVERAGE_LABELS:
                    bin_result[clabel] = {"target": 0, "actual": 0, "error": 0}
            per_bin_cov[label] = bin_result

        # Width summary (asymmetric only)
        width_summary = {}
        for clabel in COVERAGE_LABELS:
            half_widths = []
            for label in bin_labels:
                for cls in CLASSES:
                    for seg in SIGN_SEGS:
                        key = (cls, seg)
                        lo_hi = bin_pairs[label].get(key, {}).get(clabel)
                        if isinstance(lo_hi, (list, tuple)):
                            half_widths.append((lo_hi[1] - lo_hi[0]) / 2)
            width_summary[clabel] = {
                "mean_width": round(sum(half_widths) / len(half_widths), 1) if half_widths else None,
                "per_bin": {},
            }
            for label in bin_labels:
                cls_sign_hw = {}
                for cls in CLASSES:
                    for seg in SIGN_SEGS:
                        key = (cls, seg)
                        lo_hi = bin_pairs[label].get(key, {}).get(clabel)
                        if isinstance(lo_hi, (list, tuple)):
                            cls_sign_hw[f"{cls}_{seg}"] = round((lo_hi[1] - lo_hi[0]) / 2, 1)
                        else:
                            cls_sign_hw[f"{cls}_{seg}"] = None
                vals = [v for v in cls_sign_hw.values() if v is not None]
                cls_sign_hw["avg"] = round(sum(vals) / len(vals), 1) if vals else None
                width_summary[clabel]["per_bin"][label] = cls_sign_hw

        # Per-sign width stats: mean + max across bins for each (sign, level)
        sign_widths = {}
        for sign in SIGN_SEGS:
            sign_widths[sign] = {}
            for clabel in COVERAGE_LABELS:
                bin_vals = []
                for label in bin_labels:
                    cell_ws = []
                    for cls in CLASSES:
                        key = (cls, sign)
                        lo_hi = bin_pairs[label].get(key, {}).get(clabel)
                        if isinstance(lo_hi, (list, tuple)):
                            cell_ws.append((lo_hi[1] - lo_hi[0]) / 2)
                    if cell_ws:
                        bin_vals.append(sum(cell_ws) / len(cell_ws))
                sign_widths[sign][clabel] = {
                    "mean_width": round(sum(bin_vals) / len(bin_vals), 1) if bin_vals else None,
                    "max_width": round(max(bin_vals), 1) if bin_vals else None,
                }

        cov_result = overall_cov
        cov_result["per_bin"] = per_bin_cov
        cov_result["per_bin_approximate"] = True
        cov_result["per_class"] = per_class_cov
        cov_result["per_sign"] = per_sign_cov

        py_entry = {
            "n_train": train.height,
            "n_test": test.height,
            "coverage": cov_result,
            "widths": width_summary,
            "widths_by_sign": sign_widths,
            "boundaries": [b if not math.isinf(b) else "inf" for b in boundaries],
        }
        if cv_mode == "temporal":
            py_entry["train_pys"] = train_pys
            py_entry["meets_min_train_pys"] = meets_min

        # Per-fold logging
        p95_cov = overall_cov["overall"]["p95"]["actual"]
        p95_w = width_summary["p95"]["mean_width"]
        print(f"  Fold PY={test_py}: train_pys={n_tpys}, "
              f"P95_cov={p95_cov:.1f}%, P95_width={p95_w}, "
              f"correction={correction_mode}")

        per_py[str(test_py)] = py_entry

    # ─── Aggregate ────────────────────────────────────────────────────

    if cv_mode == "temporal":
        use_dfs = filtered_test_dfs if filtered_test_dfs else all_test_dfs
    else:
        use_dfs = all_test_dfs

    if use_dfs:
        all_test = pl.concat(use_dfs)

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
        agg_bins = assign_bins(all_test[baseline_col].abs(), full_boundaries, full_labels)
        df_with_bin = all_test.with_columns(pl.Series("_bin", agg_bins))
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
    if cv_mode == "temporal":
        agg_pys = [
            py for py in per_py
            if per_py[py].get("meets_min_train_pys", True)
        ] if any(per_py[py].get("meets_min_train_pys", True) for py in per_py) else list(per_py.keys())
    else:
        agg_pys = list(per_py.keys())

    labels_agg = full_labels if use_dfs else [f"q{i+1}" for i in range(n_bins)]
    agg_widths = {"overall": {}, "per_bin": {}}
    for clabel in COVERAGE_LABELS:
        fold_means = [
            per_py[py]["widths"][clabel]["mean_width"]
            for py in agg_pys
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
                for py in agg_pys
            ]
            fold_bin_widths = [w for w in fold_bin_widths if w is not None]
            agg_widths["per_bin"][label][clabel] = {
                "mean_width": round(sum(fold_bin_widths) / len(fold_bin_widths), 1) if fold_bin_widths else None,
            }

    # Aggregate per-sign widths
    agg_widths_by_sign = {}
    for sign in SIGN_SEGS:
        agg_widths_by_sign[sign] = {}
        for clabel in COVERAGE_LABELS:
            fold_means = [
                per_py[py]["widths_by_sign"][sign][clabel]["mean_width"]
                for py in agg_pys
                if per_py[py]["widths_by_sign"][sign][clabel]["mean_width"] is not None
            ]
            fold_maxes = [
                per_py[py]["widths_by_sign"][sign][clabel]["max_width"]
                for py in agg_pys
                if per_py[py]["widths_by_sign"][sign][clabel]["max_width"] is not None
            ]
            agg_widths_by_sign[sign][clabel] = {
                "mean_width": round(sum(fold_means) / len(fold_means), 1) if fold_means else None,
                "max_width": round(sum(fold_maxes) / len(fold_maxes), 1) if fold_maxes else None,
            }
    agg_widths["per_sign"] = agg_widths_by_sign

    # Stability
    p95_coverages = [
        per_py[py]["coverage"]["overall"]["p95"]["actual"]
        for py in agg_pys
    ]
    p95_widths_per_fold = [
        per_py[py]["widths"]["p95"]["mean_width"]
        for py in agg_pys
        if per_py[py]["widths"]["p95"]["mean_width"] is not None
    ]

    if len(p95_coverages) >= 2:
        p95_coverage_range = round(max(p95_coverages) - min(p95_coverages), 2)
        worst_py_idx = p95_coverages.index(min(p95_coverages))
        worst_py = agg_pys[worst_py_idx]
        p95_worst_py_coverage = min(p95_coverages)
    else:
        p95_coverage_range = 0
        worst_py = agg_pys[0] if agg_pys else ""
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
    if cv_mode == "temporal":
        stability["min_train_pys"] = min_train_pys
        stability["n_folds_total"] = len(per_py)
        stability["n_folds_filtered"] = len(agg_pys)

    return {
        "per_py": per_py,
        "aggregate": {"coverage": agg_coverage, "widths": agg_widths},
        "stability": stability,
    }


# ─── Winner selection ────────────────────────────────────────────────────────


def check_bg1(experiment_results: dict, quarters: list[str]) -> bool:
    """Check if experiment passes BG1: all 20 (level, quarter) cells within tolerance."""
    for q in quarters:
        overall = experiment_results[q]["aggregate"]["coverage"]["overall"]
        for clabel in COVERAGE_LABELS:
            error = abs(overall[clabel]["error"])
            if error > BG1_TOLERANCE[clabel]:
                return False
    return True


def select_winner(
    all_experiment_results: dict[str, dict[str, dict]],
    quarters: list[str],
) -> tuple[str, dict[str, dict]]:
    """Select winner: lowest mean P95 half-width across quarters, subject to BG1 pass.

    Args:
        all_experiment_results: {experiment_name: {quarter: run_experiment result}}
        quarters: list of quarter keys

    Returns:
        (winner_name, {quarter: result})
    """
    print(f"\n{'='*80}")
    print(f"  WINNER SELECTION")
    print(f"{'='*80}")

    print(f"\n  {'Experiment':<18} {'BG1':>5}", end="")
    for q in quarters:
        print(f" {q+' P95w':>10}", end="")
    print(f" {'Mean P95w':>10}")
    print(f"  {'-'*18} {'-'*5}", end="")
    for _ in quarters:
        print(f" {'-'*10}", end="")
    print(f" {'-'*10}")

    candidates = []
    for name, q_results in all_experiment_results.items():
        passes = check_bg1(q_results, quarters)
        p95_widths = []
        for q in quarters:
            w = q_results[q]["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width")
            p95_widths.append(w)
        mean_w = sum(w for w in p95_widths if w is not None) / len([w for w in p95_widths if w is not None]) if any(w is not None for w in p95_widths) else None

        status = "PASS" if passes else "FAIL"
        print(f"  {name:<18} {status:>5}", end="")
        for w in p95_widths:
            print(f" {w:>10.1f}" if w else f" {'n/a':>10}", end="")
        print(f" {mean_w:>10.1f}" if mean_w else f" {'n/a':>10}")

        if passes and mean_w is not None:
            candidates.append((name, mean_w, q_results))

    if not candidates:
        print("\n  WARNING: No experiment passes BG1! Selecting lowest width anyway.")
        for name, q_results in all_experiment_results.items():
            p95_widths = []
            for q in quarters:
                w = q_results[q]["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width")
                p95_widths.append(w)
            mean_w = sum(w for w in p95_widths if w is not None) / len([w for w in p95_widths if w is not None]) if any(w is not None for w in p95_widths) else None
            if mean_w is not None:
                candidates.append((name, mean_w, q_results))

    candidates.sort(key=lambda x: x[1])
    winner_name = candidates[0][0]
    winner_results = candidates[0][2]
    print(f"\n  Winner: {winner_name} (mean P95 half-width = {candidates[0][1]:.1f})")

    return winner_name, winner_results


# ─── Round runner ────────────────────────────────────────────────────────────


def run_round_v8(
    round_num: int,
    experiments: list[dict],
    baseline_col: str,
    data_loader,
    pys: list[int],
    version_id: str,
    prior_version_part: str | None = None,
) -> dict:
    """Run v8 experiments for one round.

    Each experiment dict: {"name": str, "n_bins": int, "correction": bool}
    All rounds use temporal CV, min_train_pys=1.
    Output path: versions/bands/{version_id}/r{round_num}/
    """
    primary_cv = "temporal"

    print(f"\n{'#'*80}")
    print(f"  ROUND {round_num} — v8 BAND EXPERIMENTS")
    print(f"  Primary CV: {primary_cv} (all rounds)")
    print(f"  Experiments: {', '.join(e['name'] for e in experiments)}")
    print(f"{'#'*80}")

    # ─── Phase 1: Per-quarter primary experiments ────────────────────

    # {experiment_name: {quarter: result}}
    all_results: dict[str, dict[str, dict]] = {e["name"]: {} for e in experiments}

    for quarter in QUARTERS:
        df = data_loader(quarter)
        df = add_sign_seg(df, baseline_col)
        print(f"\n  R{round_num} {quarter.upper()}: {df.height:,} rows")
        print(f"  Memory: {mem_mb():.0f} MB")

        # Distribution info
        for cls in CLASSES:
            n = df.filter(pl.col(CLASS_COL) == cls).height
            print(f"    {cls}: {n:,}")
        for seg in SIGN_SEGS:
            n = df.filter(pl.col("sign_seg") == seg).height
            print(f"    {seg}: {n:,}")
        n_zero = df.filter(pl.col("sign_seg") == "zero").height
        if n_zero > 0:
            print(f"    zero: {n_zero:,}")

        available_pys = sorted(df[PY_COL].unique().to_list())
        pys_to_use = [py for py in pys if py in available_pys]

        for exp in experiments:
            exp_name = exp["name"]
            print(f"\n  Running {exp_name}...")
            result = run_experiment(
                df, quarter, pys_to_use, exp["n_bins"], baseline_col,
                exp_name, cv_mode="temporal",
                min_train_pys=1, apply_correction=exp.get("correction", False),
            )
            p95_cov = result["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
            p95_w = result["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width", 0)
            print(f"  -> {exp_name}: P95 cov={p95_cov:.2f}%, half-width={p95_w:.1f}")
            all_results[exp_name][quarter] = result

        del df
        gc.collect()
        print(f"  Memory after cleanup: {mem_mb():.0f} MB")

    # ─── Phase 2: Print comparison table ─────────────────────────────

    print(f"\n{'='*120}")
    print(f"  R{round_num} EXPERIMENT COMPARISON (TEMPORAL)")
    print(f"{'='*120}")

    header = f"  {'Experiment':<18}"
    for q in QUARTERS:
        header += f" {q+' P95c':>10} {q+' P95w':>10}"
    header += f" {'Mean P95w':>10}"
    print(header)
    print(f"  {'-'*18}" + (f" {'-'*10} {'-'*10}" * len(QUARTERS)) + f" {'-'*10}")

    for exp in experiments:
        name = exp["name"]
        line = f"  {name:<18}"
        widths = []
        for q in QUARTERS:
            r = all_results[name][q]
            cov = r["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
            w = r["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width", 0)
            line += f" {cov:>9.2f}% {w:>10.1f}"
            if w:
                widths.append(w)
        mean_w = sum(widths) / len(widths) if widths else 0
        line += f" {mean_w:>10.1f}"
        print(line)

    # Compare to v3 (promoted) and v5 reference if available
    for ref_label, ref_path in [("v3 (promoted)", "bands/v3"), ("v5 (reference)", "bands/v5")]:
        ref_metrics_path = ROOT / "versions" / ref_path / f"r{round_num}" / "metrics.json"
        if ref_metrics_path.exists():
            with open(ref_metrics_path) as f:
                ref_metrics = json.load(f)
            line = f"  {ref_label:<18}"
            ref_ws = []
            for q in QUARTERS:
                ref_cov = ref_metrics.get("coverage", {}).get(q, {}).get("overall", {}).get("p95", {}).get("actual", 0)
                ref_w = ref_metrics.get("widths", {}).get(q, {}).get("overall", {}).get("p95", {}).get("mean_width", 0)
                line += f" {ref_cov:>9.2f}% {ref_w:>10.1f}"
                if ref_w:
                    ref_ws.append(ref_w)
            ref_mean = sum(ref_ws) / len(ref_ws) if ref_ws else 0
            line += f" {ref_mean:>10.1f}"
            print(line)

    # Per-class and per-sign coverage for each experiment
    for exp in experiments:
        name = exp["name"]
        print(f"\n  {name} per-class P95 coverage:")
        print(f"  {'Quarter':<10} {'onpeak':>10} {'offpeak':>10} {'gap':>8}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
        for q in QUARTERS:
            pc = all_results[name][q]["aggregate"]["coverage"].get("per_class", {})
            on = pc.get("onpeak", {}).get("p95", {}).get("actual", 0)
            off = pc.get("offpeak", {}).get("p95", {}).get("actual", 0)
            gap = abs(on - off)
            print(f"  {q:<10} {on:>9.2f}% {off:>9.2f}% {gap:>7.2f}pp")

    # ─── Phase 3: Select winner ──────────────────────────────────────

    winner_name, winner_q_results = select_winner(all_results, QUARTERS)
    winner_exp = next(e for e in experiments if e["name"] == winner_name)

    # ─── Phase 4: Secondary validation (LOO) on winner ───────────────

    secondary_cv = "loo"
    print(f"\n{'='*70}")
    print(f"  R{round_num} {secondary_cv.upper()} VALIDATION ON WINNER: {winner_name}")
    print(f"{'='*70}")

    secondary_results = {}
    for quarter in QUARTERS:
        df = data_loader(quarter)
        df = add_sign_seg(df, baseline_col)
        available_pys = sorted(df[PY_COL].unique().to_list())
        pys_to_use = [py for py in pys if py in available_pys]

        result = run_experiment(
            df, quarter, pys_to_use, winner_exp["n_bins"], baseline_col,
            winner_name, cv_mode=secondary_cv,
            min_train_pys=1, apply_correction=winner_exp.get("correction", False),
        )
        secondary_results[quarter] = result

        print(f"\n  {quarter.upper()} (LOO):")
        print(f"  {'PY':<6} {'n_train':>8} {'n_test':>8} {'P95 cov':>8} {'P95 hw':>10}")
        print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        for py in sorted(result["per_py"]):
            p = result["per_py"][py]
            p95_cov = p["coverage"]["overall"]["p95"]["actual"]
            p95_w = p["widths"]["p95"]["mean_width"]
            print(f"  {py:<6} {p['n_train']:>8,} {p['n_test']:>8,} {p95_cov:>7.2f}% {p95_w:>10.1f}")

        agg = result["aggregate"]["coverage"]["overall"]
        p95 = agg.get("p95", {})
        print(f"  Aggregate: P95 cov={p95.get('actual', 0):.2f}% (err {p95.get('error', 0):+.2f}pp)")

        del df
        gc.collect()

    # Primary vs secondary comparison
    print(f"\n  TEMPORAL vs LOO P95 ({winner_name}):")
    print(f"  {'Quarter':<10} {'temporal cov':>12} {'loo cov':>10} {'temporal hw':>12} {'loo hw':>10}")
    print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")
    for q in QUARTERS:
        pri_cov = winner_q_results[q]["aggregate"]["coverage"]["overall"]["p95"]["actual"]
        sec_cov = secondary_results[q]["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
        pri_w = winner_q_results[q]["aggregate"]["widths"]["overall"]["p95"]["mean_width"]
        sec_w = secondary_results[q]["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width", 0)
        print(f"  {q:<10} {pri_cov:>11.2f}% {sec_cov:>9.2f}% {pri_w:>12.1f} {sec_w:>10.1f}")

    # ─── Phase 5: Build metrics ──────────────────────────────────────

    # v8: temporal is always primary
    primary_results = winner_q_results
    temporal_results = winner_q_results
    loo_results = secondary_results

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
        pri = primary_results[quarter]
        t = temporal_results[quarter]
        w = loo_results[quarter]

        # Primary metrics for gates
        metrics["coverage"][quarter] = pri["aggregate"]["coverage"]
        metrics["widths"][quarter] = {
            "overall": pri["aggregate"]["widths"]["overall"],
            "per_sign": pri["aggregate"]["widths"].get("per_sign", {}),
        }
        metrics["stability"][quarter] = pri["stability"]
        metrics["per_class_coverage"][quarter] = pri["aggregate"]["coverage"].get("per_class", {})
        metrics["per_sign_coverage"][quarter] = pri["aggregate"]["coverage"].get("per_sign", {})

        # LOO validation
        metrics["loo_validation"][quarter] = {
            "coverage": w["aggregate"]["coverage"],
            "widths": {
                "overall": w["aggregate"]["widths"]["overall"],
                "per_bin": w["aggregate"]["widths"].get("per_bin", {}),
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

        # Temporal validation
        metrics["temporal_validation"][quarter] = {
            "aggregate_coverage": t["aggregate"]["coverage"]["overall"],
            "aggregate_widths": t["aggregate"]["widths"]["overall"],
            "stability": t["stability"],
            "per_sign_coverage": t["aggregate"]["coverage"].get("per_sign", {}),
            "per_py": {
                py: {
                    "train_pys": pdata.get("train_pys", []),
                    "meets_min_train_pys": pdata.get("meets_min_train_pys", True),
                    "n_train": pdata["n_train"],
                    "n_test": pdata["n_test"],
                    "p95_coverage": pdata["coverage"]["overall"]["p95"]["actual"],
                    "p95_mean_width": pdata["widths"]["p95"]["mean_width"],
                }
                for py, pdata in t["per_py"].items()
            },
        }

        # Per-PY summary from primary (temporal)
        metrics["per_py"][quarter] = {
            py: {
                "train_pys": pdata.get("train_pys", []),
                "meets_min_train_pys": pdata.get("meets_min_train_pys", True),
                "n_train": pdata["n_train"],
                "n_test": pdata["n_test"],
                "p95_coverage": pdata["coverage"]["overall"]["p95"]["actual"],
                "p95_mean_width": pdata["widths"]["p95"]["mean_width"],
            }
            for py, pdata in pri["per_py"].items()
        }

        # Experiment comparison
        comparison = {}
        for exp in experiments:
            name = exp["name"]
            r = all_results[name][quarter]
            entry = {
                "p95_coverage": r["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual"),
                "p95_error": r["aggregate"]["coverage"]["overall"].get("p95", {}).get("error"),
                "p50_coverage": r["aggregate"]["coverage"]["overall"].get("p50", {}).get("actual"),
                "p95_mean_width": r["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width"),
                "n_bins": exp["n_bins"],
                "correction": exp.get("correction", False),
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

    # ─── Phase 6: Save ───────────────────────────────────────────────

    v_dir = ROOT / "versions" / "bands" / version_id / f"r{round_num}"
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
    winner_correction = winner_exp.get("correction", False)
    config_data = {
        "schema_version": 1,
        "version": version_id,
        "description": f"R{round_num} asymmetric bands ({winner_exp['n_bins']} quantile bins, class+sign stratified, correction={'bidirectional' if winner_correction else 'off'})",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "part": "bands",
        "round": f"r{round_num}",
        "baseline_version": baseline_ver,
        "method": {
            "band_type": "asymmetric",
            "calibration": "signed quantile pair of (mcp - baseline) per (bin, class, sign)",
            "cv_method": "temporal_all",
            "gate_metric": "temporal",
            "bin_scheme": f"quantile_{winner_exp['n_bins']}bin",
            "stratification": "per_class_sign (onpeak/offpeak x prevail/counter)",
            "correction_type": "bidirectional" if winner_correction else "none",
            "round": round_num,
            "winner": winner_name,
            "experiments_tested": [e["name"] for e in experiments],
        },
        "parameters": {
            "coverage_levels": [0.50, 0.70, 0.80, 0.90, 0.95],
            "n_bins": winner_exp["n_bins"],
            "bin_boundaries": "data-driven (quantile)",
            "stratify_by": ["class_type", "sign_seg"],
            "min_cell_rows": MIN_CELL_ROWS,
            "fallback_chain": "(bin, class, sign) -> (bin, class) -> (bin, pooled)",
            "band_type": "asymmetric",
            "quantile_method": "signed residual quantiles at alpha_lo=(1-level)/2, alpha_hi=(1+level)/2",
            "winner_selection": "lowest mean P95 half-width, BG1 pass required",
            "cv_method": "temporal_all",
            "min_train_pys": 1,
            "correction_type": "bidirectional" if winner_correction else "none",
            "correction_tolerance_pp": CORRECTION_TOLERANCE,
            "correction_step": CORRECTION_STEP,
            "correction_max_iter": CORRECTION_MAX_ITER,
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
    print(f"Config saved to {config_path}")

    # ─── Phase 7: Print final summary ────────────────────────────────

    print(f"\n{'='*70}")
    print(f"  R{round_num} v8 SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Winner: {winner_name} (asymmetric, {winner_exp['n_bins']} bins, correction={'bidirectional' if winner_correction else 'off'})")

    print(f"\n  P95 coverage accuracy (temporal, min_train_pys=1):")
    print(f"  {'Quarter':<10} {'Target':>8} {'Actual':>8} {'Error':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for q in QUARTERS:
        cov = metrics["coverage"][q]["overall"]["p95"]
        print(f"  {q:<10} {cov['target']:>7.1f}% {cov['actual']:>7.2f}% {cov['error']:>+7.2f}pp")

    print(f"\n  Per-class P95 coverage:")
    print(f"  {'Quarter':<10} {'onpeak':>10} {'offpeak':>10} {'gap':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for q in QUARTERS:
        pc = metrics["per_class_coverage"][q]
        on = pc.get("onpeak", {}).get("p95", {}).get("actual", 0)
        off = pc.get("offpeak", {}).get("p95", {}).get("actual", 0)
        gap = abs(on - off)
        print(f"  {q:<10} {on:>9.2f}% {off:>9.2f}% {gap:>7.2f}pp")

    print(f"\n  Per-sign P95 coverage:")
    print(f"  {'Quarter':<10} {'prevail':>10} {'counter':>10} {'gap':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for q in QUARTERS:
        ps = metrics["per_sign_coverage"][q]
        prev = ps.get("prevail", {}).get("p95", {}).get("actual", 0)
        ctr = ps.get("counter", {}).get("p95", {}).get("actual", 0)
        gap = abs(prev - ctr)
        print(f"  {q:<10} {prev:>9.2f}% {ctr:>9.2f}% {gap:>7.2f}pp")

    print(f"\n  P95 mean half-width ($/MWh):")
    print(f"  {'Quarter':<10} {'Half-width':>10}")
    print(f"  {'-'*10} {'-'*10}")
    for q in QUARTERS:
        w = metrics["widths"][q]["overall"]["p95"]["mean_width"]
        print(f"  {q:<10} {w:>10.1f}")

    # Compare to v3 (promoted), v5, v6
    for ref_label, ref_path in [("v3 (promoted)", "bands/v3"), ("v5", "bands/v5"), ("v6", "bands/v6"), ("v7", "bands/v7")]:
        ref_metrics_path = ROOT / "versions" / ref_path / f"r{round_num}" / "metrics.json"
        if ref_metrics_path.exists():
            with open(ref_metrics_path) as f:
                ref_metrics = json.load(f)
            print(f"\n  vs {ref_label}:")
            print(f"  {'Quarter':<10} {'Ref P95cov':>10} {'v8 P95cov':>10} {'Cov delta':>10} {'Ref hw':>10} {'v8 hw':>10} {'Width chg':>10}")
            print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
            for q in QUARTERS:
                ref_cov = ref_metrics.get("coverage", {}).get(q, {}).get("overall", {}).get("p95", {}).get("actual", 0)
                ref_w = ref_metrics.get("widths", {}).get(q, {}).get("overall", {}).get("p95", {}).get("mean_width", 0)
                v8_cov = metrics["coverage"][q]["overall"]["p95"]["actual"]
                v8_w = metrics["widths"][q]["overall"]["p95"]["mean_width"]
                cov_delta = v8_cov - ref_cov
                pct = round((v8_w - ref_w) / ref_w * 100, 1) if ref_w else 0
                print(f"  {q:<10} {ref_cov:>9.2f}% {v8_cov:>9.2f}% {cov_delta:>+9.2f}pp {ref_w:>10.0f} {v8_w:>10.1f} {pct:>+9.1f}%")

    # Run pipeline validate
    print(f"\n  Running pipeline validate bands...")
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "pipeline" / "pipeline.py"), "validate", "bands"],
            capture_output=True, text=True, cwd=ROOT, timeout=60,
        )
        print(f"  {result.stdout.strip()}")
        if result.returncode != 0 and result.stderr:
            print(f"  STDERR: {result.stderr.strip()}")
    except Exception as e:
        print(f"  Validate failed: {e}")

    # Run pipeline compare
    round_flag = f"r{round_num}"
    print(f"\n  Running pipeline compare bands {version_id} --round {round_flag}...")
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "pipeline" / "pipeline.py"),
             "compare", "bands", version_id, "--round", round_flag],
            capture_output=True, text=True, cwd=ROOT, timeout=60,
        )
        print(f"  {result.stdout.strip()}")
        if result.returncode != 0 and result.stderr:
            print(f"  STDERR: {result.stderr.strip()}")
    except Exception as e:
        print(f"  Compare failed: {e}")

    return metrics


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    print(f"Bands v8: Production Candidate (5 bins, bidirectional correction)")
    print(f"Memory at start: {mem_mb():.0f} MB")

    # v8 experiment definitions: asymmetric only, 5 bins, with/without bidir correction
    experiments = [
        {"name": "asym_5b",       "n_bins": 5, "correction": False},
        {"name": "asym_5b_bidir", "n_bins": 5, "correction": True},
    ]

    # ─── R1: Temporal primary ────────────────────────────────────────

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

    r1_metrics = run_round_v8(
        round_num=1, experiments=experiments, baseline_col="nodal_f0",
        data_loader=r1_loader, pys=R1_PYS,
        version_id="v8",
        prior_version_part="bands/v7/r1",
    )
    gc.collect()
    print(f"\nMemory after R1: {mem_mb():.0f} MB")

    # ─── R2: Temporal primary ────────────────────────────────────────

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

    r2_metrics = run_round_v8(
        round_num=2, experiments=experiments, baseline_col="mtm_1st_mean",
        data_loader=r2_loader, pys=R2R3_PYS,
        version_id="v8",
        prior_version_part="bands/v7/r2",
    )
    gc.collect()
    print(f"\nMemory after R2: {mem_mb():.0f} MB")

    # ─── R3: Temporal primary ────────────────────────────────────────

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

    r3_metrics = run_round_v8(
        round_num=3, experiments=experiments, baseline_col="mtm_1st_mean",
        data_loader=r3_loader, pys=R2R3_PYS,
        version_id="v8",
        prior_version_part="bands/v7/r3",
    )
    gc.collect()
    print(f"\nMemory after R3: {mem_mb():.0f} MB")

    print(f"\nDone. Memory: {mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
