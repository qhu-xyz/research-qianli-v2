"""Tests for v9 simplified asymmetric bands (per-class only, no sign split).

All tests use synthetic data — no real data or Ray dependency.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual-band/scripts/test_v9_bands.py
"""

from __future__ import annotations

import sys
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
    COVERAGE_LEVELS,
    COVERAGE_LABELS,
    CLASSES,
    MIN_CELL_ROWS,
)

import random
random.seed(42)


def _make_synthetic_data(n: int = 2000) -> pl.DataFrame:
    """Create synthetic data with skewed residuals for testing."""
    random.seed(42)
    baselines = [random.gauss(0, 500) for _ in range(n)]
    # Skewed residuals: positive bias (baseline underestimates)
    residuals = [random.gauss(50, 200) + abs(b) * 0.05 for b in baselines]
    mcps = [b + r for b, r in zip(baselines, residuals)]
    classes = (["onpeak"] * (n // 2)) + (["offpeak"] * (n - n // 2))
    return pl.DataFrame({
        "baseline": baselines,
        "mcp": mcps,
        "class_type": classes,
    })


def test_calibrate_output_shape():
    """5 bins x 2 classes, all have quantile pairs at all 8 levels, no tuple keys."""
    df = _make_synthetic_data(5000)
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 5)
    result = calibrate_asymmetric_per_class(
        df, "baseline", "mcp", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )
    assert len(labels) == 5
    for label in labels:
        cell = result[label]
        # No tuple keys (sign_seg remnants)
        for key in cell:
            if isinstance(key, tuple):
                raise AssertionError(f"Found tuple key {key} — sign_seg not removed")
        # Both classes present
        for cls in CLASSES:
            assert cls in cell, f"Missing class {cls} in bin {label}"
            pairs = cell[cls]
            # All 8 coverage levels present
            for clabel in COVERAGE_LABELS:
                assert clabel in pairs, f"Missing {clabel} in ({label}, {cls})"
                lo_hi = pairs[clabel]
                assert isinstance(lo_hi, tuple) and len(lo_hi) == 2, f"Bad lo/hi for ({label}, {cls}, {clabel})"
                assert lo_hi[0] <= lo_hi[1], f"lo > hi for ({label}, {cls}, {clabel}): {lo_hi}"
        # Pooled also present
        assert "_pooled" in cell
    assert "_fallback_stats" in result
    print("PASS: test_calibrate_output_shape")


def test_apply_produces_16_band_columns():
    """lower/upper x 8 levels = 16 columns added, no sign_seg column."""
    df = _make_synthetic_data(2000)
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 5)
    bin_pairs = calibrate_asymmetric_per_class(
        df, "baseline", "mcp", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )
    result = apply_asymmetric_bands_per_class_fast(
        df, bin_pairs, "baseline", "class_type", boundaries, labels,
    )
    # 16 band columns
    for clabel in COVERAGE_LABELS:
        assert f"lower_{clabel}" in result.columns, f"Missing lower_{clabel}"
        assert f"upper_{clabel}" in result.columns, f"Missing upper_{clabel}"
    # No sign_seg
    assert "sign_seg" not in result.columns, "sign_seg column should not exist"
    # No internal columns leaked
    assert "_bin" not in result.columns
    for clabel in COVERAGE_LABELS:
        assert f"_lo_{clabel}" not in result.columns
        assert f"_hi_{clabel}" not in result.columns
    # Bands are finite
    for clabel in COVERAGE_LABELS:
        lo = result[f"lower_{clabel}"]
        hi = result[f"upper_{clabel}"]
        assert lo.null_count() == 0, f"lower_{clabel} has nulls"
        assert hi.null_count() == 0, f"upper_{clabel} has nulls"
    print("PASS: test_apply_produces_16_band_columns")


def test_coverage_monotonicity():
    """P10 coverage < P30 < P50 < ... < P99 on synthetic data."""
    df = _make_synthetic_data(3000)
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 5)
    bin_pairs = calibrate_asymmetric_per_class(
        df, "baseline", "mcp", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )
    banded = apply_asymmetric_bands_per_class_fast(
        df, bin_pairs, "baseline", "class_type", boundaries, labels,
    )

    # Compute actual coverage at each level
    coverages = []
    for clabel in COVERAGE_LABELS:
        lo = banded[f"lower_{clabel}"]
        hi = banded[f"upper_{clabel}"]
        mcp = banded["mcp"]
        cov = float(((mcp >= lo) & (mcp <= hi)).mean())
        coverages.append(cov)

    # Monotonicity: each level should cover >= previous
    for i in range(1, len(coverages)):
        assert coverages[i] >= coverages[i - 1] - 0.001, (
            f"Coverage not monotonic: {COVERAGE_LABELS[i-1]}={coverages[i-1]:.3f} > "
            f"{COVERAGE_LABELS[i]}={coverages[i]:.3f}"
        )
    print(f"PASS: test_coverage_monotonicity (coverages: "
          f"{', '.join(f'{cl}={c:.1%}' for cl, c in zip(COVERAGE_LABELS, coverages))})")


def test_fallback_triggers_and_records():
    """Small cell falls back to pooled, _fallback_stats records it."""
    # Create data where one class in one bin has very few rows
    df = pl.DataFrame({
        "baseline": [100.0] * 10 + [200.0] * 2000,
        "mcp": [110.0] * 10 + [210.0] * 2000,
        "class_type": ["onpeak"] * 5 + ["offpeak"] * 5 + ["onpeak"] * 1000 + ["offpeak"] * 1000,
    })
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 2)
    result = calibrate_asymmetric_per_class(
        df, "baseline", "mcp", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )
    assert result["_fallback_stats"]["to_pooled"] > 0, "Expected at least one fallback"
    # Check that fallback cells have _fallback marker
    found_fallback = False
    for label in labels:
        for cls in CLASSES:
            if result[label][cls].get("_fallback") == "pooled":
                found_fallback = True
    assert found_fallback, "Expected _fallback='pooled' marker in at least one cell"
    print("PASS: test_fallback_triggers_and_records")


def test_bad_class_type_raises():
    """class_type='peak' raises ValueError."""
    df = pl.DataFrame({
        "baseline": [100.0, 200.0],
        "mcp": [110.0, 210.0],
        "class_type": ["peak", "offpeak"],
        "planning_year": [2023, 2023],
    })
    from run_v9_bands import run_experiment
    try:
        run_experiment(df, "aq1", [2023], 2, "baseline", min_train_pys=1)
        raise AssertionError("Expected ValueError for bad class_type")
    except ValueError as e:
        assert "peak" in str(e).lower() or "Unexpected" in str(e), f"Wrong error: {e}"
    print("PASS: test_bad_class_type_raises")


def test_asymmetric_narrower_than_symmetric():
    """For skewed residuals, asymmetric total width < symmetric total width."""
    random.seed(123)
    n = 5000
    # Heavily right-skewed residuals
    baselines = [random.gauss(500, 200) for _ in range(n)]
    residuals = [abs(random.gauss(0, 100)) + 50 for _ in range(n)]  # all positive skew
    mcps = [b + r for b, r in zip(baselines, residuals)]

    df = pl.DataFrame({
        "baseline": baselines,
        "mcp": mcps,
        "class_type": ["onpeak"] * n,
    })

    boundaries, labels = compute_quantile_boundaries(df["baseline"], 3)

    # Asymmetric calibration
    asym_pairs = calibrate_asymmetric_per_class(
        df, "baseline", "mcp", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )

    # Symmetric: width = quantile(|residual|, level)
    residual_series = df["mcp"] - df["baseline"]
    abs_residual = residual_series.abs()
    sym_p95_width = float(abs_residual.quantile(0.95))

    # Asymmetric P95 total width (averaged across bins)
    asym_widths = []
    for label in labels:
        cell = asym_pairs[label]
        lo_hi = cell["onpeak"]["p95"]
        asym_widths.append(lo_hi[1] - lo_hi[0])
    asym_p95_avg_width = sum(asym_widths) / len(asym_widths)

    # Symmetric total width = 2 * sym_p95_width
    sym_total = 2 * sym_p95_width

    assert asym_p95_avg_width < sym_total, (
        f"Asymmetric ({asym_p95_avg_width:.1f}) should be narrower than "
        f"symmetric ({sym_total:.1f}) for right-skewed data"
    )
    pct_narrower = (1 - asym_p95_avg_width / sym_total) * 100
    print(f"PASS: test_asymmetric_narrower_than_symmetric "
          f"(asym={asym_p95_avg_width:.1f} vs sym={sym_total:.1f}, {pct_narrower:.1f}% narrower)")


def test_p99_wider_than_p95():
    """P99 band fully contains P95 band for every row."""
    df = _make_synthetic_data(2000)
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 5)
    bin_pairs = calibrate_asymmetric_per_class(
        df, "baseline", "mcp", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )
    banded = apply_asymmetric_bands_per_class_fast(
        df, bin_pairs, "baseline", "class_type", boundaries, labels,
    )

    lo_95 = banded["lower_p95"]
    hi_95 = banded["upper_p95"]
    lo_99 = banded["lower_p99"]
    hi_99 = banded["upper_p99"]

    # P99 should contain P95: lo_99 <= lo_95 and hi_99 >= hi_95
    assert (lo_99 <= lo_95).all(), "P99 lower bound should be <= P95 lower bound"
    assert (hi_99 >= hi_95).all(), "P99 upper bound should be >= P95 upper bound"
    print("PASS: test_p99_wider_than_p95")


if __name__ == "__main__":
    test_calibrate_output_shape()
    test_apply_produces_16_band_columns()
    test_coverage_monotonicity()
    test_fallback_triggers_and_records()
    test_bad_class_type_raises()
    test_asymmetric_narrower_than_symmetric()
    test_p99_wider_than_p95()
    print("\n=== All 7 v9 tests passed. ===")
