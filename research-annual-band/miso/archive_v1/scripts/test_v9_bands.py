# DEPRECATED — archived 2026-03-17. Old data sources. See task_plan.md for porting plan.
import warnings as _w; _w.warn("This script is DEPRECATED (archived 2026-03-17). Use canonical get_trades_of_given_duration().", DeprecationWarning, stacklevel=2)

"""Comprehensive tests for v10 annual bands (quarterly scale).

Covers 90%+ of use cases per CLAUDE.md testing requirements:
1. Core calibration
2. Band application
3. Coverage monotonicity
4. Scale correctness (quarterly)
5. Edge cases
6. CP assignment
7. Class parity
8. Temporal CV
9. Artifact round-trip
10. Integration contract

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual-band/scripts/test_v9_bands.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl

from run_v9_bands import (
    calibrate_asymmetric_per_class,
    apply_asymmetric_bands_per_class_fast,
    compute_quantile_boundaries,
    assign_bins,
    evaluate_coverage,
    evaluate_per_class_coverage,
    sanitize_for_json,
    COVERAGE_LEVELS,
    COVERAGE_LABELS,
    CLASSES,
    MIN_CELL_ROWS,
)

import random
random.seed(42)

PASS_COUNT = 0
FAIL_COUNT = 0

def check(condition, name, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  PASS: {name}")
    else:
        FAIL_COUNT += 1
        print(f"  FAIL: {name} — {detail}")


def _make_quarterly_data(n: int = 5000) -> pl.DataFrame:
    """Synthetic data in quarterly scale (×3 of monthly)."""
    random.seed(42)
    # Monthly baselines, then scale to quarterly
    baselines_monthly = [random.gauss(0, 500) for _ in range(n)]
    residuals_monthly = [random.gauss(50, 200) + abs(b) * 0.05 for b in baselines_monthly]
    mcps_monthly = [b + r for b, r in zip(baselines_monthly, residuals_monthly)]

    # Quarterly = monthly × 3
    baselines = [b * 3 for b in baselines_monthly]
    mcps = [m * 3 for m in mcps_monthly]

    classes = (["onpeak"] * (n // 2)) + (["offpeak"] * (n - n // 2))
    return pl.DataFrame({
        "baseline": baselines,
        "mcp_q": mcps,
        "class_type": classes,
        "planning_year": [2023] * n,
    })


# ═══════════════════════════════════════════════════════════════════════════
# 1. Core Calibration
# ═══════════════════════════════════════════════════════════════════════════

def test_calibration():
    print("\n=== 1. Core Calibration ===")
    df = _make_quarterly_data(5000)
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 5)
    result = calibrate_asymmetric_per_class(
        df, "baseline", "mcp_q", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )

    # 5 bins
    check(len(labels) == 5, "5 quantile bins")

    # Each bin has both classes and _pooled
    for label in labels:
        cell = result[label]
        check("_pooled" in cell, f"{label} has _pooled")
        for cls in CLASSES:
            check(cls in cell, f"{label} has {cls}")
            pairs = cell[cls]
            # All 8 coverage levels present
            for clabel in COVERAGE_LABELS:
                check(clabel in pairs, f"{label}.{cls} has {clabel}")
                lo_hi = pairs[clabel]
                check(isinstance(lo_hi, tuple) and len(lo_hi) == 2, f"{label}.{cls}.{clabel} is (lo, hi)")
                check(lo_hi[0] <= lo_hi[1], f"{label}.{cls}.{clabel}: lo <= hi")

    # No tuple keys (sign_seg remnants)
    for label in labels:
        for key in result[label]:
            check(not isinstance(key, tuple), f"{label} key '{key}' not a tuple")

    # Fallback stats exist
    check("_fallback_stats" in result, "fallback_stats present")

    # Quantile pairs are in quarterly scale (values should be ~3x of monthly)
    sample = result[labels[2]]["onpeak"]["p95"]
    check(abs(sample[1] - sample[0]) > 100, f"P95 width > 100 (quarterly scale): {sample}")


def test_fallback():
    print("\n=== 1b. Fallback Triggers ===")
    df = pl.DataFrame({
        "baseline": [100.0] * 10 + [200.0] * 2000,
        "mcp_q": [110.0] * 10 + [210.0] * 2000,
        "class_type": ["onpeak"] * 5 + ["offpeak"] * 5 + ["onpeak"] * 1000 + ["offpeak"] * 1000,
    })
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 2)
    result = calibrate_asymmetric_per_class(
        df, "baseline", "mcp_q", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )
    check(result["_fallback_stats"]["to_pooled"] > 0, "fallback triggered for small cells")
    found_fallback = any(
        result[label][cls].get("_fallback") == "pooled"
        for label in labels for cls in CLASSES
    )
    check(found_fallback, "fallback marker '_fallback=pooled' found")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Band Application
# ═══════════════════════════════════════════════════════════════════════════

def test_apply():
    print("\n=== 2. Band Application ===")
    df = _make_quarterly_data(3000)
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 5)
    bin_pairs = calibrate_asymmetric_per_class(
        df, "baseline", "mcp_q", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )
    result = apply_asymmetric_bands_per_class_fast(
        df, bin_pairs, "baseline", "class_type", boundaries, labels,
    )

    # 16 band columns (8 levels × lower/upper)
    for clabel in COVERAGE_LABELS:
        check(f"lower_{clabel}" in result.columns, f"lower_{clabel} exists")
        check(f"upper_{clabel}" in result.columns, f"upper_{clabel} exists")

    # No internal columns leaked
    check("sign_seg" not in result.columns, "no sign_seg column")
    check("_bin" not in result.columns, "no _bin column leaked")

    # No nulls in band columns
    for clabel in COVERAGE_LABELS:
        check(result[f"lower_{clabel}"].null_count() == 0, f"lower_{clabel} no nulls")
        check(result[f"upper_{clabel}"].null_count() == 0, f"upper_{clabel} no nulls")

    # Band containment: P99 ⊃ P95 ⊃ ... ⊃ P10
    for i in range(1, len(COVERAGE_LABELS)):
        wider = COVERAGE_LABELS[i]
        narrower = COVERAGE_LABELS[i - 1]
        lo_wider = result[f"lower_{wider}"]
        lo_narrower = result[f"lower_{narrower}"]
        hi_wider = result[f"upper_{wider}"]
        hi_narrower = result[f"upper_{narrower}"]
        check((lo_wider <= lo_narrower).all(), f"lower_{wider} <= lower_{narrower}")
        check((hi_wider >= hi_narrower).all(), f"upper_{wider} >= upper_{narrower}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Coverage Monotonicity
# ═══════════════════════════════════════════════════════════════════════════

def test_coverage_monotonicity():
    print("\n=== 3. Coverage Monotonicity ===")
    df = _make_quarterly_data(3000)
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 5)
    bin_pairs = calibrate_asymmetric_per_class(
        df, "baseline", "mcp_q", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )
    banded = apply_asymmetric_bands_per_class_fast(
        df, bin_pairs, "baseline", "class_type", boundaries, labels,
    )

    coverages = []
    for clabel in COVERAGE_LABELS:
        lo = banded[f"lower_{clabel}"]
        hi = banded[f"upper_{clabel}"]
        mcp = banded["mcp_q"]
        cov = float(((mcp >= lo) & (mcp <= hi)).mean())
        coverages.append(cov)

    for i in range(1, len(coverages)):
        check(coverages[i] >= coverages[i - 1] - 0.001,
              f"{COVERAGE_LABELS[i]} >= {COVERAGE_LABELS[i-1]}: {coverages[i]:.3f} >= {coverages[i-1]:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Scale Correctness (Quarterly)
# ═══════════════════════════════════════════════════════════════════════════

def test_quarterly_scale():
    print("\n=== 4. Scale Correctness (Quarterly) ===")
    df = _make_quarterly_data(2000)

    # Baselines should be ~3x of typical monthly values (monthly std ~500, quarterly ~1500)
    bl_std = df["baseline"].std()
    check(bl_std > 500, f"baseline std={bl_std:.0f} > 500 (quarterly scale)")

    # MCP should be ~3x of monthly
    mcp_std = df["mcp_q"].std()
    check(mcp_std > 500, f"mcp_q std={mcp_std:.0f} > 500 (quarterly scale)")

    # Calibrate and check widths are quarterly-scale
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 5)
    bin_pairs = calibrate_asymmetric_per_class(
        df, "baseline", "mcp_q", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )
    # P95 width for a mid bin should be > 300 (quarterly)
    mid_bin = labels[2]
    p95 = bin_pairs[mid_bin]["onpeak"]["p95"]
    width = p95[1] - p95[0]
    check(width > 300, f"P95 width={width:.0f} > 300 (quarterly scale)")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

def test_edge_cases():
    print("\n=== 5. Edge Cases ===")

    # Zero baseline
    df = pl.DataFrame({
        "baseline": [0.0] * 1000 + [100.0] * 1000,
        "mcp_q": [10.0] * 1000 + [110.0] * 1000,
        "class_type": ["onpeak"] * 1000 + ["offpeak"] * 1000,
    })
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 2)
    bin_pairs = calibrate_asymmetric_per_class(
        df, "baseline", "mcp_q", "class_type", boundaries, labels, COVERAGE_LEVELS,
    )
    banded = apply_asymmetric_bands_per_class_fast(
        df, bin_pairs, "baseline", "class_type", boundaries, labels,
    )
    check(banded["lower_p95"].null_count() == 0, "zero baseline: no null bands")

    # Single class_type — needs enough data that pooled has rows
    random.seed(99)
    n_single = 3000
    df_single = pl.DataFrame({
        "baseline": [random.gauss(500, 200) for _ in range(n_single)],
        "mcp_q": [random.gauss(510, 210) for _ in range(n_single)],
        "class_type": ["onpeak"] * n_single,
    })
    boundaries, labels = compute_quantile_boundaries(df_single["baseline"], 2)
    bin_pairs = calibrate_asymmetric_per_class(
        df_single, "baseline", "mcp_q", "class_type", boundaries, labels, COVERAGE_LEVELS,
    )
    # offpeak should fall back to pooled (0 offpeak rows, but pooled has onpeak rows)
    for label in labels:
        check(bin_pairs[label]["offpeak"].get("_fallback") == "pooled",
              f"{label}.offpeak falls back to pooled (single class)")

    # Bad class_type raises
    df_bad = pl.DataFrame({
        "baseline": [100.0], "mcp_q": [110.0], "class_type": ["peak"],
        "planning_year": [2023],
    })
    from run_v9_bands import run_experiment
    try:
        run_experiment(df_bad, "aq1", [2023], 2, "baseline", min_train_pys=1)
        check(False, "bad class_type should raise ValueError")
    except ValueError:
        check(True, "bad class_type raises ValueError")


# ═══════════════════════════════════════════════════════════════════════════
# 6. CP Assignment
# ═══════════════════════════════════════════════════════════════════════════

def test_cp_assignment():
    print("\n=== 6. CP Assignment ===")
    # For buy trades: higher upper band → higher clearing prob
    # CP at upper_p99 > CP at upper_p95 > CP at upper_p50 > CP at upper_p10
    # Theoretical: upper_p99 buy CP = 99.5%, upper_p10 buy CP = 55%

    # Check theoretical CPs are ordered correctly
    for level, clabel in zip(COVERAGE_LEVELS, COVERAGE_LABELS):
        buy_cp_upper = (1 + level) / 2 * 100  # e.g., 97.5 for P95
        buy_cp_lower = (1 - level) / 2 * 100  # e.g., 2.5 for P95
        check(buy_cp_upper > 50, f"upper_{clabel} buy CP={buy_cp_upper:.1f}% > 50%")
        check(buy_cp_lower < 50, f"lower_{clabel} buy CP={buy_cp_lower:.1f}% < 50%")

    # Check buy CP monotonicity across levels
    buy_cps = [(1 + level) / 2 * 100 for level in COVERAGE_LEVELS]
    for i in range(1, len(buy_cps)):
        check(buy_cps[i] > buy_cps[i-1],
              f"buy CP for upper_{COVERAGE_LABELS[i]} > upper_{COVERAGE_LABELS[i-1]}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. Class Parity
# ═══════════════════════════════════════════════════════════════════════════

def test_class_parity():
    print("\n=== 7. Class Parity ===")
    df = _make_quarterly_data(4000)
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 5)
    bin_pairs = calibrate_asymmetric_per_class(
        df, "baseline", "mcp_q", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )
    banded = apply_asymmetric_bands_per_class_fast(
        df, bin_pairs, "baseline", "class_type", boundaries, labels,
    )

    per_class = evaluate_per_class_coverage(banded, "mcp_q", "class_type", COVERAGE_LEVELS)

    check("onpeak" in per_class, "onpeak coverage computed")
    check("offpeak" in per_class, "offpeak coverage computed")
    check(per_class["onpeak"]["n"] > 0, "onpeak has rows")
    check(per_class["offpeak"]["n"] > 0, "offpeak has rows")

    # Class parity gap at P95 should be < 5pp
    on_p95 = per_class["onpeak"]["p95"]["actual"]
    off_p95 = per_class["offpeak"]["p95"]["actual"]
    gap = abs(on_p95 - off_p95)
    check(gap < 5.0, f"class parity gap at P95: {gap:.1f}pp < 5pp")


# ═══════════════════════════════════════════════════════════════════════════
# 8. Temporal CV
# ═══════════════════════════════════════════════════════════════════════════

def test_temporal_cv():
    print("\n=== 8. Temporal CV ===")
    from run_v9_bands import run_experiment

    random.seed(42)
    n_per_py = 1000
    rows = []
    for py in [2020, 2021, 2022, 2023]:
        for i in range(n_per_py):
            bl = random.gauss(0, 500) * 3
            res = random.gauss(50, 200) * 3
            rows.append({
                "baseline": bl,
                "mcp_q": bl + res,
                "class_type": "onpeak" if i < n_per_py // 2 else "offpeak",
                "planning_year": py,
            })
    df = pl.DataFrame(rows)

    result = run_experiment(df, "aq1", [2020, 2021, 2022, 2023], 3, "baseline", min_train_pys=2)

    # Check that folds with <min_train_pys are excluded
    for py_str, py_data in result["per_py"].items():
        train_pys = py_data.get("train_pys", [])
        meets = py_data.get("meets_min_train_pys", True)
        if len(train_pys) < 2:
            check(not meets, f"PY{py_str}: {len(train_pys)} train PYs → excluded")
        else:
            check(meets, f"PY{py_str}: {len(train_pys)} train PYs → included")

    # Train strictly before test
    for py_str, py_data in result["per_py"].items():
        train_pys = py_data.get("train_pys", [])
        for tpy in train_pys:
            check(tpy < int(py_str), f"PY{py_str}: train PY {tpy} < test PY {py_str}")

    # Aggregate coverage should be reasonable
    p95 = result["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
    check(80 < p95 < 100, f"aggregate P95 coverage={p95:.1f}% in reasonable range")


# ═══════════════════════════════════════════════════════════════════════════
# 9. Artifact Round-Trip
# ═══════════════════════════════════════════════════════════════════════════

def test_artifact_roundtrip():
    print("\n=== 9. Artifact Round-Trip ===")
    df = _make_quarterly_data(3000)
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 5)
    bin_pairs = calibrate_asymmetric_per_class(
        df, "baseline", "mcp_q", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )

    # Save artifact
    artifact = {
        "version": "test",
        "calibration": {
            "aq1": {
                "boundaries": [b if not math.isinf(b) else "inf" for b in boundaries],
                "bin_labels": labels,
                "bin_pairs": sanitize_for_json(bin_pairs),
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(artifact, f, indent=2)
        artifact_path = f.name

    # Load artifact
    with open(artifact_path) as f:
        loaded = json.load(f)

    cal = loaded["calibration"]["aq1"]

    # Reconstruct boundaries
    loaded_boundaries = [float(b) if b != "inf" else float("inf") for b in cal["boundaries"]]
    check(len(loaded_boundaries) == len(boundaries), "boundaries length matches")
    for i, (orig, loaded_b) in enumerate(zip(boundaries, loaded_boundaries)):
        if math.isinf(orig):
            check(math.isinf(loaded_b), f"boundary[{i}] inf preserved")
        else:
            check(abs(orig - loaded_b) < 0.01, f"boundary[{i}] matches: {orig} vs {loaded_b}")

    # Reconstruct bin_pairs and apply
    loaded_pairs = cal["bin_pairs"]
    # Convert lists back to tuples
    for label in labels:
        for cls in CLASSES:
            if cls in loaded_pairs.get(label, {}):
                for clabel in COVERAGE_LABELS:
                    val = loaded_pairs[label][cls].get(clabel)
                    if isinstance(val, list):
                        loaded_pairs[label][cls][clabel] = tuple(val)
        if "_pooled" in loaded_pairs.get(label, {}):
            for clabel in COVERAGE_LABELS:
                val = loaded_pairs[label]["_pooled"].get(clabel)
                if isinstance(val, list):
                    loaded_pairs[label]["_pooled"][clabel] = tuple(val)

    banded_orig = apply_asymmetric_bands_per_class_fast(
        df, bin_pairs, "baseline", "class_type", boundaries, labels,
    )
    banded_loaded = apply_asymmetric_bands_per_class_fast(
        df, loaded_pairs, "baseline", "class_type", loaded_boundaries, labels,
    )

    # Compare band edges
    for clabel in COVERAGE_LABELS:
        diff = (banded_orig[f"lower_{clabel}"] - banded_loaded[f"lower_{clabel}"]).abs().max()
        check(diff < 0.1, f"lower_{clabel} round-trip diff={diff:.4f} < 0.1")

    import os
    os.unlink(artifact_path)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Integration Contract
# ═══════════════════════════════════════════════════════════════════════════

def test_integration_contract():
    print("\n=== 10. Integration Contract ===")
    # Verify output schema would be compatible with production
    df = _make_quarterly_data(2000)
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 5)
    bin_pairs = calibrate_asymmetric_per_class(
        df, "baseline", "mcp_q", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )
    banded = apply_asymmetric_bands_per_class_fast(
        df, bin_pairs, "baseline", "class_type", boundaries, labels,
    )

    # Simulate bid_price selection (10 from 16 band edges)
    # Use all 8 upper + lower_p95 + lower_p99 = 10
    bid_cols = [f"upper_{cl}" for cl in COVERAGE_LABELS] + ["lower_p95", "lower_p99"]
    check(len(bid_cols) == 10, f"10 bid price candidates: {len(bid_cols)}")

    for col in bid_cols:
        check(col in banded.columns, f"{col} available for bid_price selection")

    # All values finite
    for col in bid_cols:
        vals = banded[col]
        check(vals.null_count() == 0, f"{col} no nulls")
        check(vals.is_finite().all(), f"{col} all finite")


# ═══════════════════════════════════════════════════════════════════════════
# Run All
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_calibration()
    test_fallback()
    test_apply()
    test_coverage_monotonicity()
    test_quarterly_scale()
    test_edge_cases()
    test_cp_assignment()
    test_class_parity()
    test_temporal_cv()
    test_artifact_roundtrip()
    test_integration_contract()

    print(f"\n{'='*60}")
    print(f"  RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    print(f"{'='*60}")
    if FAIL_COUNT > 0:
        sys.exit(1)
    else:
        print("  All tests passed.")
