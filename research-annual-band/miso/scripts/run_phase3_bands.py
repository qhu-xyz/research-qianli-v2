"""Phase 3: R1 Band Calibration — empirical quantile bins by |baseline|.

For each R1 path, produces symmetric band widths at 5 coverage levels
(P50/P70/P80/P90/P95), calibrated per |nodal_f0| bin, validated via
leave-one-PY-out cross-validation.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual/scripts/run_phase3_bands.py
"""

from __future__ import annotations

import gc
import json
import math
import os
import resource
import sys
from pathlib import Path

import polars as pl

# ─── Constants ─────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/opt/temp/qianli/annual_research/crossproduct_work")
QUARTERS = ["aq1", "aq2", "aq3", "aq4"]
PYS = [2020, 2021, 2022, 2023, 2024, 2025]

COVERAGE_LEVELS = [0.50, 0.70, 0.80, 0.90, 0.95]
COVERAGE_LABELS = ["p50", "p70", "p80", "p90", "p95"]

BIN_BOUNDARIES = [0, 50, 250, 1000, float("inf")]
BIN_LABELS = ["tiny", "small", "medium", "large"]

BASELINE_COL = "nodal_f0"
MCP_COL = "mcp_mean"
PY_COL = "planning_year"
CLASS_COL = "class_type"


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ─── Core calibration functions ───────────────────────────────────────────────


def assign_bins(
    abs_baseline: pl.Series,
    boundaries: list[float] = BIN_BOUNDARIES,
    labels: list[str] = BIN_LABELS,
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

    # Build coalesce chain
    tmp = pl.DataFrame({"_abs_bl": abs_baseline})
    result = tmp.with_columns(
        pl.coalesce(exprs).alias("bin")
    )["bin"]
    return result


def calibrate_bin_widths(
    df: pl.DataFrame,
    baseline_col: str = BASELINE_COL,
    mcp_col: str = MCP_COL,
    boundaries: list[float] = BIN_BOUNDARIES,
    labels: list[str] = BIN_LABELS,
    coverage_levels: list[float] = COVERAGE_LEVELS,
) -> dict[str, dict[str, float]]:
    """Per-bin empirical quantile of |residual|.

    Returns {bin_label: {p50: width, p70: width, ...}}.
    """
    abs_res = (df[mcp_col] - df[baseline_col]).abs()
    bins = assign_bins(df[baseline_col].abs(), boundaries, labels)

    work = pl.DataFrame({
        "abs_residual": abs_res,
        "bin": bins,
    })

    result = {}
    for label in labels:
        subset = work.filter(pl.col("bin") == label)["abs_residual"]
        n = len(subset)
        widths = {}
        for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
            if n > 0:
                widths[clabel] = round(float(subset.quantile(level)), 1)
            else:
                widths[clabel] = None
        widths["n"] = n
        result[label] = widths

    return result


def apply_bands(
    df: pl.DataFrame,
    bin_widths: dict[str, dict[str, float]],
    baseline_col: str = BASELINE_COL,
    boundaries: list[float] = BIN_BOUNDARIES,
    labels: list[str] = BIN_LABELS,
) -> pl.DataFrame:
    """Add lower_{pct}, upper_{pct} columns. Symmetric: baseline +/- width."""
    abs_bl = df[baseline_col].abs()
    bins = assign_bins(abs_bl, boundaries, labels)
    df = df.with_columns(pl.Series("_bin", bins))

    for clabel in COVERAGE_LABELS:
        # Map bin -> width
        width_map = {label: bin_widths[label][clabel] for label in labels}
        width_series = df["_bin"].replace_strict(width_map, default=None).cast(pl.Float64)

        df = df.with_columns([
            (pl.col(baseline_col) - width_series).alias(f"lower_{clabel}"),
            (pl.col(baseline_col) + width_series).alias(f"upper_{clabel}"),
        ])

    return df.drop("_bin")


def evaluate_coverage(
    df: pl.DataFrame,
    mcp_col: str = MCP_COL,
    baseline_col: str = BASELINE_COL,
    coverage_levels: list[float] = COVERAGE_LEVELS,
    boundaries: list[float] = BIN_BOUNDARIES,
    labels: list[str] = BIN_LABELS,
) -> dict:
    """Actual coverage per level, overall and per-bin.

    Returns {overall: {p50: {target, actual, error}, ...}, per_bin: {tiny: {...}, ...}}.
    """
    mcp = df[mcp_col]
    result = {"overall": {}, "per_bin": {}}

    # Overall
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

    # Per-bin
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


def loo_band_calibration(
    df: pl.DataFrame,
    quarter: str,
    pys: list[int],
    baseline_col: str = BASELINE_COL,
    mcp_col: str = MCP_COL,
    boundaries: list[float] = BIN_BOUNDARIES,
    labels: list[str] = BIN_LABELS,
    coverage_levels: list[float] = COVERAGE_LEVELS,
) -> dict:
    """Full LOO: for each PY, calibrate on remaining PYs, test on held-out.

    Returns {
        per_py: {py: {coverage: {...}, widths: {...}}},
        aggregate: {coverage: {...}, widths: {...}},
        stability: {p95_coverage_range, p95_worst_py, p95_width_cv},
    }
    """
    per_py = {}
    all_test_dfs = []

    for test_py in pys:
        train = df.filter(pl.col(PY_COL) != test_py)
        test = df.filter(pl.col(PY_COL) == test_py)

        if test.height == 0:
            continue

        # Calibrate on train set
        bin_widths = calibrate_bin_widths(
            train, baseline_col, mcp_col, boundaries, labels, coverage_levels,
        )

        # Apply to test set
        test_banded = apply_bands(test, bin_widths, baseline_col, boundaries, labels)
        all_test_dfs.append(test_banded)

        # Evaluate on test set
        cov = evaluate_coverage(
            test_banded, mcp_col, baseline_col, coverage_levels, boundaries, labels,
        )

        # Width summary
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

    # Aggregate: combine all LOO test predictions
    if all_test_dfs:
        all_test = pl.concat(all_test_dfs)
        agg_coverage = evaluate_coverage(
            all_test, mcp_col, baseline_col, coverage_levels, boundaries, labels,
        )
    else:
        agg_coverage = {"overall": {}, "per_bin": {}}

    # Aggregate widths: average across LOO folds
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

    # Stability metrics for P95
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
        import statistics
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


def compute_clearing_probs(
    df: pl.DataFrame,
    baseline_col: str = BASELINE_COL,
    mcp_col: str = MCP_COL,
    coverage_levels: list[float] = COVERAGE_LEVELS,
) -> dict:
    """Empirical clearing rates per band level, segmented by sign(baseline).

    A buy trade clears profitably when actual MCP > bid price.
    With symmetric bands, the bid = baseline - width (lower band).
    Clearing rate = P(mcp > lower_band) for buy_positive (baseline > 0)
    and buy_negative (baseline < 0).
    """
    result = {}

    for sign_label, sign_filter in [
        ("buy_positive", pl.col(baseline_col) > 0),
        ("buy_negative", pl.col(baseline_col) < 0),
    ]:
        subset = df.filter(sign_filter)
        n = subset.height
        sign_result = {"n": n}
        if n > 0:
            mcp = subset[mcp_col]
            for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
                lower = subset[f"lower_{clabel}"]
                clearing_rate = round(float((mcp > lower).mean()) * 100, 2)
                sign_result[clabel] = clearing_rate
        result[sign_label] = sign_result

    return result


# ─── Formatting helpers ───────────────────────────────────────────────────────


def print_coverage_table(results: dict, quarter: str) -> None:
    """Print coverage accuracy table for a quarter."""
    print(f"\n{'='*60}")
    print(f"  {quarter.upper()} — Coverage Accuracy (LOO aggregated)")
    print(f"{'='*60}")

    overall = results["aggregate"]["coverage"]["overall"]
    print(f"\n  {'Level':<8} {'Target':>8} {'Actual':>8} {'Error':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for clabel in COVERAGE_LABELS:
        o = overall[clabel]
        print(f"  {clabel:<8} {o['target']:>7.1f}% {o['actual']:>7.2f}% {o['error']:>+7.2f}pp")

    # Per-bin at P95
    per_bin = results["aggregate"]["coverage"]["per_bin"]
    print(f"\n  Per-bin P95 coverage:")
    print(f"  {'Bin':<10} {'n':>8} {'Target':>8} {'Actual':>8} {'Error':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for label in BIN_LABELS:
        b = per_bin[label]
        p95 = b["p95"]
        print(f"  {label:<10} {b['n']:>8,} {p95['target']:>7.1f}% {p95['actual']:>7.2f}% {p95['error']:>+7.2f}pp")


def print_width_table(results: dict, quarter: str) -> None:
    """Print band width table for a quarter."""
    print(f"\n  {quarter.upper()} — Band Widths (avg across LOO folds)")
    widths = results["aggregate"]["widths"]

    print(f"\n  {'Bin':<10}", end="")
    for clabel in COVERAGE_LABELS:
        print(f" {clabel:>8}", end="")
    print()
    print(f"  {'-'*10}", end="")
    for _ in COVERAGE_LABELS:
        print(f" {'-'*8}", end="")
    print()

    for label in BIN_LABELS:
        print(f"  {label:<10}", end="")
        for clabel in COVERAGE_LABELS:
            w = widths["per_bin"][label][clabel]["mean_width"]
            if w is not None:
                print(f" {w:>8.0f}", end="")
            else:
                print(f" {'n/a':>8}", end="")
        print()


def print_stability_table(results: dict, quarter: str) -> None:
    """Print per-PY stability table for a quarter."""
    print(f"\n  {quarter.upper()} — Per-PY Stability")
    per_py = results["per_py"]

    print(f"\n  {'PY':<6} {'n_test':>8} {'P50 cov':>8} {'P95 cov':>8} {'P95 mean_w':>10}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for py in sorted(per_py.keys()):
        p = per_py[py]
        p50_cov = p["coverage"]["overall"]["p50"]["actual"]
        p95_cov = p["coverage"]["overall"]["p95"]["actual"]
        p95_w = p["widths"]["p95"]["mean_width"]
        print(f"  {py:<6} {p['n_test']:>8,} {p50_cov:>7.2f}% {p95_cov:>7.2f}% {p95_w:>10.0f}")

    stab = results["stability"]
    print(f"\n  P95 coverage range: {stab['p95_coverage_range']:.2f}pp")
    print(f"  P95 worst PY: {stab['p95_worst_py']} ({stab['p95_worst_py_coverage']:.2f}%)")
    print(f"  P95 width CV: {stab['p95_width_cv']:.4f}")


# ─── JSON sanitization ────────────────────────────────────────────────────────


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


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    print(f"Phase 3: R1 Band Calibration")
    print(f"Memory at start: {mem_mb():.0f} MB")

    all_metrics = {
        "coverage": {},
        "widths": {},
        "per_py": {},
        "stability": {},
        "clearing_probs": {},
    }

    for quarter in QUARTERS:
        parquet_path = DATA_DIR / f"{quarter}_all_baselines.parquet"
        print(f"\n{'#'*70}")
        print(f"  Processing {quarter} from {parquet_path}")
        print(f"{'#'*70}")

        # Load with lazy scan, filter early
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

        # Check PY distribution
        py_counts = df.group_by(PY_COL).len().sort(PY_COL)
        print(f"  PY distribution:")
        for row in py_counts.iter_rows():
            print(f"    PY {row[0]}: {row[1]:,} rows")

        available_pys = sorted(df[PY_COL].unique().to_list())
        pys_to_use = [py for py in PYS if py in available_pys]

        # Run LOO band calibration
        results = loo_band_calibration(
            df, quarter, pys_to_use,
            BASELINE_COL, MCP_COL,
            BIN_BOUNDARIES, BIN_LABELS, COVERAGE_LEVELS,
        )

        # Print results
        print_coverage_table(results, quarter)
        print_width_table(results, quarter)
        print_stability_table(results, quarter)

        # Also calibrate on all data for the "production" widths
        all_data_widths = calibrate_bin_widths(
            df, BASELINE_COL, MCP_COL,
            BIN_BOUNDARIES, BIN_LABELS, COVERAGE_LEVELS,
        )
        print(f"\n  {quarter.upper()} — Full-data calibrated widths (for production):")
        for label in BIN_LABELS:
            w = all_data_widths[label]
            print(f"    {label}: n={w['n']:,}  p50={w['p50']}  p70={w['p70']}  p80={w['p80']}  p90={w['p90']}  p95={w['p95']}")

        # Apply full-data bands for clearing prob computation
        df_banded = apply_bands(df, all_data_widths, BASELINE_COL, BIN_BOUNDARIES, BIN_LABELS)
        clearing = compute_clearing_probs(df_banded, BASELINE_COL, MCP_COL, COVERAGE_LEVELS)
        print(f"\n  {quarter.upper()} — Clearing probabilities:")
        for sign_label in ["buy_positive", "buy_negative"]:
            c = clearing[sign_label]
            print(f"    {sign_label} (n={c['n']:,}):", end="")
            for clabel in COVERAGE_LABELS:
                if clabel in c:
                    print(f"  {clabel}={c[clabel]:.1f}%", end="")
            print()

        # ─── Build metrics sections ───────────────────────────────────────

        # Coverage section
        agg_cov = results["aggregate"]["coverage"]
        all_metrics["coverage"][quarter] = agg_cov

        # Widths section
        all_metrics["widths"][quarter] = {
            "overall": results["aggregate"]["widths"]["overall"],
            "per_bin": results["aggregate"]["widths"]["per_bin"],
            "production_widths": all_data_widths,
        }

        # Per-PY section
        py_summary = {}
        for py, pdata in results["per_py"].items():
            py_summary[py] = {
                "p50_coverage": pdata["coverage"]["overall"]["p50"]["actual"],
                "p95_coverage": pdata["coverage"]["overall"]["p95"]["actual"],
                "p95_mean_width": pdata["widths"]["p95"]["mean_width"],
            }
        all_metrics["per_py"][quarter] = py_summary

        # Stability section
        all_metrics["stability"][quarter] = results["stability"]

        # Clearing probs section
        all_metrics["clearing_probs"][quarter] = clearing

        # Free memory
        del df, df_banded, results
        gc.collect()
        print(f"  Memory after cleanup: {mem_mb():.0f} MB")

    # ─── Save metrics ─────────────────────────────────────────────────────

    metrics_path = ROOT / "versions" / "bands" / "r1" / "v1" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    output = sanitize_for_json(all_metrics)
    tmp = metrics_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, metrics_path)
    print(f"\nMetrics saved to {metrics_path}")

    # ─── Print summary ────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"  SUMMARY — Phase 3 Band Calibration")
    print(f"{'='*70}")

    print(f"\n  Coverage accuracy (LOO, P95):")
    print(f"  {'Quarter':<10} {'Target':>8} {'Actual':>8} {'Error':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for q in QUARTERS:
        cov = all_metrics["coverage"][q]["overall"]["p95"]
        print(f"  {q:<10} {cov['target']:>7.1f}% {cov['actual']:>7.2f}% {cov['error']:>+7.2f}pp")

    print(f"\n  Coverage accuracy (LOO, P50):")
    print(f"  {'Quarter':<10} {'Target':>8} {'Actual':>8} {'Error':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for q in QUARTERS:
        cov = all_metrics["coverage"][q]["overall"]["p50"]
        print(f"  {q:<10} {cov['target']:>7.1f}% {cov['actual']:>7.2f}% {cov['error']:>+7.2f}pp")

    print(f"\n  Stability (P95):")
    print(f"  {'Quarter':<10} {'Worst PY':>10} {'Worst cov':>10} {'Range':>8} {'Width CV':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
    for q in QUARTERS:
        s = all_metrics["stability"][q]
        print(f"  {q:<10} {s['p95_worst_py']:>10} {s['p95_worst_py_coverage']:>9.2f}% {s['p95_coverage_range']:>7.2f}pp {s['p95_width_cv']:>10.4f}")

    print(f"\nDone. Memory: {mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
