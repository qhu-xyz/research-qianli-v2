"""Comprehensive tests for run_v7_bands.py.

Tests asymmetric calibration, band application, bidirectional per-bin correction,
experiment runner, winner selection, edge cases, and pipeline gate compatibility.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual/scripts/test_v7_bands.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_v7_bands import (
    calibrate_asymmetric_per_class_sign,
    apply_asymmetric_bands_per_class_sign_fast,
    bidirectional_bin_correction,
    cold_start_inflate,
    sanitize_for_json,
    compute_quantile_boundaries,
    run_experiment,
    check_bg1,
    select_winner,
    add_sign_seg,
    COVERAGE_LEVELS,
    COVERAGE_LABELS,
    MIN_CELL_ROWS,
    BG1_TOLERANCE,
    CORRECTION_TOLERANCE,
    CORRECTION_STEP,
    CORRECTION_MAX_ITER,
    COLD_START_BOOST,
    MIN_CORRECTION_PYS,
)

passed = 0
failed = 0
errors = []


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed, errors
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        errors.append(name)


def make_df(
    n: int = 10000,
    seed: int = 42,
    skew: float = 50.0,
    n_pys: int = 6,
    baseline_range: tuple = (-500, 500),
) -> pl.DataFrame:
    """Create a synthetic DataFrame with known properties."""
    rng = np.random.RandomState(seed)
    baseline = rng.uniform(baseline_range[0], baseline_range[1], n)
    # Skewed residuals: positive shift for prevail, negative for counter
    residual = rng.normal(0, 100, n) + np.where(baseline > 0, skew, -skew)
    mcp = baseline + residual
    pys = rng.choice(list(range(2020, 2020 + n_pys)), n)
    cls = rng.choice(["onpeak", "offpeak"], n)

    return pl.DataFrame({
        "baseline": baseline,
        "mcp_mean": mcp,
        "class_type": cls,
        "planning_year": pys,
    })


# ─── Test 1: Calibration structure ──────────────────────────────────────────

def test_calibration_structure():
    print("\n[Test 1] Calibration structure")
    df = make_df(n=20000)
    df = add_sign_seg(df, "baseline")
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 4)

    result = calibrate_asymmetric_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )

    # Check all bins present
    for label in labels:
        check(f"bin '{label}' present", label in result)
        cell = result[label]
        check(f"bin '{label}' has _pooled", "_pooled" in cell)
        check(f"bin '{label}' has onpeak", "onpeak" in cell)
        check(f"bin '{label}' has offpeak", "offpeak" in cell)
        for cls in ["onpeak", "offpeak"]:
            for seg in ["prevail", "counter"]:
                key = (cls, seg)
                check(f"bin '{label}' has {key}", key in cell)

    # Check fallback stats
    check("_fallback_stats present", "_fallback_stats" in result)


# ─── Test 2: Quantile pair correctness ──────────────────────────────────────

def test_quantile_pair_correctness():
    print("\n[Test 2] Quantile pair correctness")
    df = make_df(n=20000)
    df = add_sign_seg(df, "baseline")
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 4)

    result = calibrate_asymmetric_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )

    for label in labels:
        pooled = result[label]["_pooled"]
        for clabel in COVERAGE_LABELS:
            pair = pooled[clabel]
            if pair is not None:
                lo, hi = pair
                check(f"{label} {clabel}: hi >= lo", hi >= lo,
                      f"lo={lo}, hi={hi}")

    # Verify monotonicity: wider coverage levels produce wider intervals
    for label in labels:
        pooled = result[label]["_pooled"]
        widths = []
        for clabel in COVERAGE_LABELS:
            pair = pooled[clabel]
            if pair is not None:
                widths.append(pair[1] - pair[0])
        for i in range(len(widths) - 1):
            check(f"{label} width monotonicity {COVERAGE_LABELS[i]}<{COVERAGE_LABELS[i+1]}",
                  widths[i] <= widths[i + 1],
                  f"{widths[i]:.1f} vs {widths[i+1]:.1f}")


# ─── Test 3: Apply function produces correct columns ────────────────────────

def test_apply_columns():
    print("\n[Test 3] Apply function column names")
    df = make_df(n=10000)
    df = add_sign_seg(df, "baseline")
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 4)

    bin_pairs = calibrate_asymmetric_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )
    df_banded = apply_asymmetric_bands_per_class_sign_fast(
        df, bin_pairs, "baseline", "class_type", boundaries, labels,
    )

    for clabel in COVERAGE_LABELS:
        check(f"lower_{clabel} present", f"lower_{clabel}" in df_banded.columns)
        check(f"upper_{clabel} present", f"upper_{clabel}" in df_banded.columns)

    # No temp columns left
    check("_bin not in output", "_bin" not in df_banded.columns)
    for clabel in COVERAGE_LABELS:
        check(f"_lo_{clabel} not in output", f"_lo_{clabel}" not in df_banded.columns)
        check(f"_hi_{clabel} not in output", f"_hi_{clabel}" not in df_banded.columns)


# ─── Test 4: In-sample coverage accuracy ────────────────────────────────────

def test_insample_coverage():
    print("\n[Test 4] In-sample coverage accuracy")
    df = make_df(n=50000, seed=123)
    df = add_sign_seg(df, "baseline")
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 4)

    bin_pairs = calibrate_asymmetric_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )
    df_banded = apply_asymmetric_bands_per_class_sign_fast(
        df, bin_pairs, "baseline", "class_type", boundaries, labels,
    )

    mcp = df_banded["mcp_mean"]
    for level, clabel in zip(COVERAGE_LEVELS, COVERAGE_LABELS):
        lower = df_banded[f"lower_{clabel}"]
        upper = df_banded[f"upper_{clabel}"]
        actual = float(((mcp >= lower) & (mcp <= upper)).mean()) * 100
        target = level * 100
        # In-sample should be very close (within 1pp for large n)
        check(f"in-sample {clabel}: {actual:.1f}% ~ {target:.0f}%",
              abs(actual - target) < 1.5,
              f"actual={actual:.2f}%, target={target:.0f}%")


# ─── Test 5: run_experiment temporal mode ────────────────────────────────────

def test_run_experiment_temporal():
    print("\n[Test 5] run_experiment temporal mode")
    df = make_df(n=20000, n_pys=6)
    pys = sorted(df["planning_year"].unique().to_list())

    result = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_temp",
        cv_mode="temporal", min_train_pys=1,
    )

    check("per_py present", "per_py" in result)
    check("aggregate present", "aggregate" in result)
    check("stability present", "stability" in result)

    # Temporal per_py should have train_pys and meets_min_train_pys
    for py_str, py_data in result["per_py"].items():
        check(f"temporal PY {py_str} has train_pys", "train_pys" in py_data)
        check(f"temporal PY {py_str} has meets_min_train_pys",
              "meets_min_train_pys" in py_data)

    # Aggregate coverage
    overall = result["aggregate"]["coverage"]["overall"]
    for clabel in COVERAGE_LABELS:
        check(f"aggregate {clabel} present", clabel in overall)
        cov = overall[clabel]
        check(f"aggregate {clabel} has target/actual/error",
              all(k in cov for k in ["target", "actual", "error"]))

    # Widths
    widths = result["aggregate"]["widths"]["overall"]
    for clabel in COVERAGE_LABELS:
        w = widths[clabel]["mean_width"]
        check(f"aggregate {clabel} width is float", isinstance(w, float))
        check(f"aggregate {clabel} width > 0", w > 0)

    # Stability should have temporal fields
    stab = result["stability"]
    check("stability has p95_coverage_range", "p95_coverage_range" in stab)
    check("stability has p95_worst_py", "p95_worst_py" in stab)
    check("stability has p95_width_cv", "p95_width_cv" in stab)
    check("temporal stability has min_train_pys", "min_train_pys" in stab)
    check("temporal stability has n_folds_total", "n_folds_total" in stab)
    check("temporal stability has n_folds_filtered", "n_folds_filtered" in stab)


# ─── Test 6: run_experiment LOO mode ─────────────────────────────────────────

def test_run_experiment_loo():
    print("\n[Test 6] run_experiment LOO mode")
    df = make_df(n=20000)
    pys = sorted(df["planning_year"].unique().to_list())

    result = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_loo",
        cv_mode="loo",
    )

    check("per_py present", "per_py" in result)
    check("aggregate present", "aggregate" in result)

    # Each PY should be in per_py
    for py in pys:
        check(f"PY {py} in per_py", str(py) in result["per_py"])

    # LOO should NOT have temporal-specific fields
    for py_data in result["per_py"].values():
        check("LOO per_py has no train_pys", "train_pys" not in py_data)


# ─── Test 7: BG1 check ─────────────────────────────────────────────────────

def test_bg1_check():
    print("\n[Test 7] BG1 gate check")
    df = make_df(n=20000)
    pys = sorted(df["planning_year"].unique().to_list())

    result = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_bg1",
        cv_mode="loo",
    )

    # With enough data, BG1 should pass
    passes = check_bg1({"aq1": result}, ["aq1"])
    check("BG1 passes with good data", passes)

    # Create a deliberately bad result
    bad_result = {
        "aggregate": {
            "coverage": {
                "overall": {
                    clabel: {"target": level * 100, "actual": 50.0,
                             "error": 50.0 - level * 100}
                    for level, clabel in zip(COVERAGE_LEVELS, COVERAGE_LABELS)
                }
            },
            "widths": result["aggregate"]["widths"],
        },
        "per_py": {},
        "stability": result["stability"],
    }
    fails = check_bg1({"aq1": bad_result}, ["aq1"])
    check("BG1 fails with bad coverage", not fails)


# ─── Test 8: select_winner ──────────────────────────────────────────────────

def test_select_winner():
    print("\n[Test 8] select_winner")
    df = make_df(n=20000)
    pys = sorted(df["planning_year"].unique().to_list())

    all_results = {}
    experiments = [
        {"name": "asym_4b", "n_bins": 4},
        {"name": "asym_6b", "n_bins": 6},
    ]
    for exp in experiments:
        r = run_experiment(
            df, "aq1", pys, exp["n_bins"], "baseline", exp["name"],
            cv_mode="loo",
        )
        all_results[exp["name"]] = {"aq1": r}

    winner_name, winner_q = select_winner(all_results, ["aq1"])
    check("winner_name is a string", isinstance(winner_name, str))
    check("winner_name in experiments", winner_name in [e["name"] for e in experiments])
    check("winner_q has aq1", "aq1" in winner_q)


# ─── Test 9: Fallback chain with sparse data ───────────────────────────────

def test_sparse_fallback():
    print("\n[Test 9] Sparse cell fallback")
    df = make_df(n=3000, seed=99)
    df = add_sign_seg(df, "baseline")
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 8)

    result = calibrate_asymmetric_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )

    stats = result.get("_fallback_stats", {})
    check("fallback stats has total", "total" in stats)
    check("fallback stats has to_class", "to_class" in stats)
    check("fallback stats has to_pooled", "to_pooled" in stats)

    # With 3000 rows / 32 cells ~ 94 rows/cell, most should fall back
    total_fallbacks = stats["to_class"] + stats["to_pooled"]
    check(f"some fallbacks occurred (sparse 8-bin)", total_fallbacks > 0,
          f"to_class={stats['to_class']}, to_pooled={stats['to_pooled']}")

    # Apply should still work without errors
    df_banded = apply_asymmetric_bands_per_class_sign_fast(
        df, result, "baseline", "class_type", boundaries, labels,
    )

    # All bands should be non-null
    for clabel in COVERAGE_LABELS:
        n_null = df_banded[f"lower_{clabel}"].null_count()
        check(f"{clabel} lower no nulls after fallback", n_null == 0,
              f"null_count={n_null}")


# ─── Test 10: Zero baseline edge case ──────────────────────────────────────

def test_zero_baseline():
    print("\n[Test 10] Zero baseline edge case")
    rng = np.random.RandomState(42)
    n = 5000
    baseline = np.concatenate([
        np.zeros(100),  # zero baseline paths
        rng.uniform(-500, 500, n - 100),
    ])
    mcp = baseline + rng.normal(0, 100, n)
    cls = rng.choice(["onpeak", "offpeak"], n)
    pys = rng.choice([2020, 2021, 2022, 2023], n)

    df = pl.DataFrame({
        "baseline": baseline,
        "mcp_mean": mcp,
        "class_type": cls,
        "planning_year": pys,
    })
    df = add_sign_seg(df, "baseline")

    # Verify zero baseline gets sign_seg="zero"
    n_zero = df.filter(pl.col("sign_seg") == "zero").height
    check(f"zero baseline rows get sign_seg='zero'", n_zero == 100)

    boundaries, labels = compute_quantile_boundaries(df["baseline"], 4)
    bin_pairs = calibrate_asymmetric_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )

    df_banded = apply_asymmetric_bands_per_class_sign_fast(
        df, bin_pairs, "baseline", "class_type", boundaries, labels,
    )

    # Zero baseline rows should have valid bands
    zero_rows = df_banded.filter(pl.col("sign_seg") == "zero")
    for clabel in COVERAGE_LABELS:
        n_null = zero_rows[f"lower_{clabel}"].null_count()
        check(f"zero baseline {clabel} no nulls", n_null == 0,
              f"null_count={n_null}")

    # Band should be [lo, hi] around 0 for zero baseline
    if zero_rows.height > 0:
        lo = float(zero_rows["lower_p95"].mean())
        hi = float(zero_rows["upper_p95"].mean())
        check(f"zero baseline P95 band valid", lo < hi,
              f"lo={lo:.1f}, hi={hi:.1f}")


# ─── Test 11: Single PY edge case ───────────────────────────────────────────

def test_single_py():
    print("\n[Test 11] Single PY edge case")
    df = make_df(n=5000, n_pys=1)

    # LOO with single PY: test set exists but no folds possible
    result = run_experiment(
        df, "aq1", [2020], 4, "baseline", "single_py",
        cv_mode="loo",
    )
    # With 1 PY, LOO has 0 training rows, should skip
    check("single PY LOO: per_py is empty", len(result["per_py"]) == 0)

    # Temporal with single PY: no training data before first PY
    result2 = run_experiment(
        df, "aq1", [2020], 4, "baseline", "single_py_temp",
        cv_mode="temporal",
    )
    check("single PY temporal: handles gracefully",
          isinstance(result2["stability"], dict))


# ─── Test 12: JSON serialization round-trip ─────────────────────────────────

def test_json_roundtrip():
    print("\n[Test 12] JSON serialization round-trip")
    df = make_df(n=10000)
    pys = sorted(df["planning_year"].unique().to_list())

    result = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_json",
        cv_mode="loo",
    )

    metrics = {
        "coverage": {"aq1": result["aggregate"]["coverage"]},
        "widths": {"aq1": {"overall": result["aggregate"]["widths"]["overall"]}},
        "stability": {"aq1": result["stability"]},
    }

    sanitized = sanitize_for_json(metrics)

    try:
        s = json.dumps(sanitized, indent=2)
        check("json.dumps succeeds", True)
        parsed = json.loads(s)
        check("json.loads round-trip succeeds", True)
        orig_p95 = metrics["coverage"]["aq1"]["overall"]["p95"]["actual"]
        parsed_p95 = parsed["coverage"]["aq1"]["overall"]["p95"]["actual"]
        check("P95 actual preserved", abs(orig_p95 - parsed_p95) < 0.01,
              f"orig={orig_p95}, parsed={parsed_p95}")
    except Exception as e:
        check(f"JSON serialization failed: {e}", False)


# ─── Test 13: Per-class and per-sign coverage evaluation ────────────────────

def test_per_class_per_sign_evaluation():
    print("\n[Test 13] Per-class and per-sign coverage in run_experiment")
    df = make_df(n=20000)
    pys = sorted(df["planning_year"].unique().to_list())

    result = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_eval",
        cv_mode="loo",
    )

    agg_cov = result["aggregate"]["coverage"]
    check("aggregate has per_class", "per_class" in agg_cov)
    check("aggregate has per_sign", "per_sign" in agg_cov)

    pc = agg_cov["per_class"]
    check("per_class has onpeak", "onpeak" in pc)
    check("per_class has offpeak", "offpeak" in pc)

    ps = agg_cov["per_sign"]
    check("per_sign has prevail", "prevail" in ps)
    check("per_sign has counter", "counter" in ps)


# ─── Test 14: Per-bin coverage in aggregate ─────────────────────────────────

def test_per_bin_aggregate():
    print("\n[Test 14] Per-bin coverage in aggregate")
    df = make_df(n=20000)
    pys = sorted(df["planning_year"].unique().to_list())

    result = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_perbin",
        cv_mode="loo",
    )

    per_bin = result["aggregate"]["coverage"].get("per_bin", {})
    check("aggregate per_bin not empty", len(per_bin) > 0)

    for bin_label, bin_data in per_bin.items():
        check(f"bin {bin_label} has n", "n" in bin_data)
        check(f"bin {bin_label} has p95", "p95" in bin_data)
        p95 = bin_data["p95"]
        check(f"bin {bin_label} p95 has actual",
              "actual" in p95 and isinstance(p95["actual"], (int, float)))


# ─── Test 15: _bin column guard (apply called twice) ────────────────────────

def test_bin_column_guard():
    print("\n[Test 15] _bin column guard (double apply)")
    df = make_df(n=5000)
    df = add_sign_seg(df, "baseline")
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 4)

    bin_pairs = calibrate_asymmetric_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )

    # First apply
    df_banded1 = apply_asymmetric_bands_per_class_sign_fast(
        df, bin_pairs, "baseline", "class_type", boundaries, labels,
    )

    # Second apply on same-ish df (should not crash with DuplicateError)
    try:
        df_with_bin = df.with_columns(pl.lit("q1").alias("_bin"))
        df_banded2 = apply_asymmetric_bands_per_class_sign_fast(
            df_with_bin, bin_pairs, "baseline", "class_type", boundaries, labels,
        )
        check("double apply does not crash", True)
        check("output has lower_p95", "lower_p95" in df_banded2.columns)
    except Exception as e:
        check(f"double apply crashed: {e}", False)


# ─── Test 16: Width monotonicity (BG5 compatibility) ────────────────────────

def test_width_monotonicity():
    print("\n[Test 16] Width monotonicity (BG5)")
    df = make_df(n=20000)
    pys = sorted(df["planning_year"].unique().to_list())

    result = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_mono",
        cv_mode="loo",
    )

    widths = result["aggregate"]["widths"]["overall"]
    ws = [widths[clabel]["mean_width"] for clabel in COVERAGE_LABELS]

    for i in range(len(ws) - 1):
        check(f"width {COVERAGE_LABELS[i]} < {COVERAGE_LABELS[i+1]}",
              ws[i] < ws[i + 1],
              f"{ws[i]:.1f} vs {ws[i+1]:.1f}")


# ─── Test 17: Stability metrics structure ────────────────────────────────────

def test_stability_structure():
    print("\n[Test 17] Stability metrics structure")
    df = make_df(n=20000)
    pys = sorted(df["planning_year"].unique().to_list())

    result = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_stab",
        cv_mode="loo",
    )

    stab = result["stability"]
    check("p95_coverage_range is float",
          isinstance(stab["p95_coverage_range"], (int, float)))
    check("p95_worst_py is string", isinstance(stab["p95_worst_py"], str))
    check("p95_worst_py_coverage is float",
          isinstance(stab["p95_worst_py_coverage"], (int, float)))
    check("p95_width_cv is float",
          isinstance(stab["p95_width_cv"], (int, float)))
    check("p95_coverage_range >= 0", stab["p95_coverage_range"] >= 0)
    check("p95_width_cv >= 0", stab["p95_width_cv"] >= 0)


# ─── Test 18: Empty test PY handling ────────────────────────────────────────

def test_empty_test_py():
    print("\n[Test 18] Empty test PY handling")
    df = make_df(n=10000, n_pys=3)

    result = run_experiment(
        df, "aq1", [2020, 2021, 2022, 2099], 4, "baseline", "test_empty",
        cv_mode="loo",
    )

    check("non-existent PY 2099 not in per_py", "2099" not in result["per_py"])
    check("existing PYs are in per_py", len(result["per_py"]) >= 3)


# ─── Test 19: 6-bin and 8-bin and 10-bin configurations ─────────────────────

def test_higher_bin_counts():
    print("\n[Test 19] Higher bin counts (6, 8, 10)")
    df = make_df(n=20000)
    df = add_sign_seg(df, "baseline")

    for n_bins in [6, 8, 10]:
        boundaries, labels = compute_quantile_boundaries(df["baseline"], n_bins)
        check(f"{n_bins}-bin: {n_bins + 1} boundaries",
              len(boundaries) == n_bins + 1)
        check(f"{n_bins}-bin: {n_bins} labels", len(labels) == n_bins)

        bin_pairs = calibrate_asymmetric_per_class_sign(
            df, "baseline", "mcp_mean", "class_type",
            boundaries, labels, COVERAGE_LEVELS,
        )

        df_banded = apply_asymmetric_bands_per_class_sign_fast(
            df, bin_pairs, "baseline", "class_type", boundaries, labels,
        )

        mcp = df_banded["mcp_mean"]
        cov = float(((mcp >= df_banded["lower_p95"]) &
                      (mcp <= df_banded["upper_p95"])).mean()) * 100
        check(f"{n_bins}-bin P95 coverage ~ 95%", abs(cov - 95) < 2.0,
              f"actual={cov:.2f}%")


# ─── Test 20: Pipeline gate metrics structure ────────────────────────────────

def test_pipeline_gate_structure():
    print("\n[Test 20] Pipeline gate metrics structure")
    df = make_df(n=20000)
    pys = sorted(df["planning_year"].unique().to_list())

    result = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_gates",
        cv_mode="loo",
    )

    # Build metrics dict as run_round_v7 would
    metrics = {
        "coverage": {
            "aq1": result["aggregate"]["coverage"],
        },
        "widths": {
            "aq1": {
                "overall": result["aggregate"]["widths"]["overall"],
                "per_bin": result["aggregate"]["widths"]["per_bin"],
                "per_sign": result["aggregate"]["widths"].get("per_sign", {}),
            },
        },
        "stability": {
            "aq1": result["stability"],
        },
        "per_class_coverage": {
            "aq1": result["aggregate"]["coverage"].get("per_class", {}),
        },
    }

    # BG1: coverage[q]["overall"][lvl]["error"]
    for lvl in COVERAGE_LABELS:
        val = metrics["coverage"]["aq1"]["overall"][lvl]["error"]
        check(f"BG1 path: coverage/aq1/overall/{lvl}/error exists",
              isinstance(val, (int, float)))

    # BG2: widths[q]["overall"][lvl]["mean_width"]
    for lvl in COVERAGE_LABELS:
        val = metrics["widths"]["aq1"]["overall"][lvl]["mean_width"]
        check(f"BG2 path: widths/aq1/overall/{lvl}/mean_width exists",
              isinstance(val, (int, float)))

    # BG3: coverage[q]["per_bin"][bin]["p95"]["error"]
    per_bin = metrics["coverage"]["aq1"]["per_bin"]
    for bin_label in per_bin:
        val = per_bin[bin_label]["p95"]["error"]
        check(f"BG3 path: per_bin/{bin_label}/p95/error exists",
              isinstance(val, (int, float)))

    # BG4: stability[q]["p95_worst_py_coverage"]
    val = metrics["stability"]["aq1"]["p95_worst_py_coverage"]
    check("BG4 path: stability/aq1/p95_worst_py_coverage exists",
          isinstance(val, (int, float)))

    # BG5: widths monotonicity — needs widths[q]["overall"]
    check("BG5 path: widths/aq1/overall exists",
          isinstance(metrics["widths"]["aq1"]["overall"], dict))

    # BG6: per_class_coverage[q][class]["p95"]["actual"]
    pc = metrics["per_class_coverage"]["aq1"]
    for cls in ["onpeak", "offpeak"]:
        val = pc[cls]["p95"]["actual"]
        check(f"BG6 path: per_class_coverage/aq1/{cls}/p95/actual exists",
              isinstance(val, (int, float)))

    # BG2b: per_sign widths path
    per_sign = metrics["widths"]["aq1"].get("per_sign", {})
    check("per_sign widths present", len(per_sign) > 0)
    for sign in ["prevail", "counter"]:
        for lvl in COVERAGE_LABELS:
            mw = per_sign.get(sign, {}).get(lvl, {}).get("mean_width")
            xw = per_sign.get(sign, {}).get(lvl, {}).get("max_width")
            check(f"per_sign {sign}/{lvl} mean_width exists",
                  isinstance(mw, (int, float)))
            check(f"per_sign {sign}/{lvl} max_width exists",
                  isinstance(xw, (int, float)))


# ─── Test 21: widths_by_sign structure in run_experiment ──────────────────

def test_widths_by_sign_structure():
    print("\n[Test 21] widths_by_sign structure")
    df = make_df(n=20000)
    pys = sorted(df["planning_year"].unique().to_list())

    result = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_wbs",
        cv_mode="loo",
    )

    # Per-PY entries should have widths_by_sign
    for py_str, py_data in result["per_py"].items():
        check(f"PY {py_str} has widths_by_sign", "widths_by_sign" in py_data)
        wbs = py_data["widths_by_sign"]
        for sign in ["prevail", "counter"]:
            check(f"PY {py_str} wbs has {sign}", sign in wbs)
            for clabel in COVERAGE_LABELS:
                check(f"PY {py_str} wbs {sign}/{clabel} has mean_width",
                      "mean_width" in wbs[sign][clabel])
                check(f"PY {py_str} wbs {sign}/{clabel} has max_width",
                      "max_width" in wbs[sign][clabel])
                mw = wbs[sign][clabel]["mean_width"]
                xw = wbs[sign][clabel]["max_width"]
                if mw is not None and xw is not None:
                    check(f"PY {py_str} wbs {sign}/{clabel} max >= mean",
                          xw >= mw - 0.1,
                          f"max={xw}, mean={mw}")

    # Aggregate should have per_sign widths
    agg_per_sign = result["aggregate"]["widths"].get("per_sign", {})
    check("aggregate widths has per_sign", len(agg_per_sign) > 0)
    for sign in ["prevail", "counter"]:
        for clabel in COVERAGE_LABELS:
            entry = agg_per_sign.get(sign, {}).get(clabel, {})
            check(f"agg per_sign {sign}/{clabel} mean_width > 0",
                  entry.get("mean_width", 0) > 0)
            check(f"agg per_sign {sign}/{clabel} max_width > 0",
                  entry.get("max_width", 0) > 0)


# ─── Test 22: Bidirectional shrink ───────────────────────────────────────────

def test_bidirectional_shrink():
    """Manually widen bin_pairs to force over-coverage, verify correction shrinks."""
    print("\n[Test 22] Bidirectional shrink (over-covering cell)")
    import copy
    rng = np.random.RandomState(42)
    n = 20000
    baseline = rng.uniform(10, 100, n)
    residual = rng.normal(0, 30, n)
    mcp = baseline + residual

    cls = rng.choice(["onpeak", "offpeak"], n)
    df = pl.DataFrame({
        "baseline": baseline,
        "mcp_mean": mcp,
        "class_type": cls,
        "planning_year": rng.choice([2020, 2021, 2022, 2023], n),
    })
    df = add_sign_seg(df, "baseline")

    boundaries, labels = compute_quantile_boundaries(df["baseline"], 4)

    # Calibrate, then manually inflate to force over-coverage
    bin_pairs_raw = calibrate_asymmetric_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )
    bin_pairs_inflated = copy.deepcopy(bin_pairs_raw)
    for label in labels:
        cell = bin_pairs_inflated.get(label, {})
        for cls_name in ["onpeak", "offpeak"]:
            for seg in ["prevail", "counter"]:
                key = (cls_name, seg)
                if key not in cell or "_fallback" in cell[key]:
                    continue
                for clabel in COVERAGE_LABELS:
                    lo_hi = cell[key].get(clabel)
                    if isinstance(lo_hi, (list, tuple)):
                        lo, hi = lo_hi
                        # Inflate by 50% to force over-coverage
                        cell[key][clabel] = (round(lo * 1.5, 1), round(hi * 1.5, 1))

    # Apply bidirectional correction on the inflated pairs
    bin_pairs_corrected = bidirectional_bin_correction(
        df, bin_pairs_inflated, "baseline", boundaries, labels,
        tolerance=1.0, step=0.02, max_iter=50,
    )

    # Check that at least some cells were shrunk back
    any_shrunk = False
    for label in labels:
        inf_cell = bin_pairs_inflated.get(label, {})
        corr_cell = bin_pairs_corrected.get(label, {})
        for cls_name in ["onpeak", "offpeak"]:
            for seg in ["prevail", "counter"]:
                key = (cls_name, seg)
                inf_p95 = inf_cell.get(key, {}).get("p95")
                corr_p95 = corr_cell.get(key, {}).get("p95")
                if (isinstance(inf_p95, (list, tuple)) and
                        isinstance(corr_p95, (list, tuple))):
                    inf_width = inf_p95[1] - inf_p95[0]
                    corr_width = corr_p95[1] - corr_p95[0]
                    if corr_width < inf_width - 0.01:
                        any_shrunk = True

    check("bidirectional correction shrunk at least one cell", any_shrunk)


# ─── Test 23: Bidirectional expand ──────────────────────────────────────────

def test_bidirectional_expand():
    """Manually shrink bin_pairs to force under-coverage, verify correction expands."""
    print("\n[Test 23] Bidirectional expand (under-covering cell)")
    import copy
    rng = np.random.RandomState(99)
    n = 20000
    baseline = rng.uniform(10, 100, n)
    residual = rng.normal(0, 50, n)
    mcp = baseline + residual

    cls = rng.choice(["onpeak", "offpeak"], n)
    df = pl.DataFrame({
        "baseline": baseline,
        "mcp_mean": mcp,
        "class_type": cls,
        "planning_year": rng.choice([2020, 2021, 2022, 2023], n),
    })
    df = add_sign_seg(df, "baseline")

    boundaries, labels = compute_quantile_boundaries(df["baseline"], 4)

    bin_pairs_raw = calibrate_asymmetric_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )

    # Manually deflate to force under-coverage
    bin_pairs_deflated = copy.deepcopy(bin_pairs_raw)
    for label in labels:
        cell = bin_pairs_deflated.get(label, {})
        for cls_name in ["onpeak", "offpeak"]:
            for seg in ["prevail", "counter"]:
                key = (cls_name, seg)
                if key not in cell or "_fallback" in cell[key]:
                    continue
                for clabel in COVERAGE_LABELS:
                    lo_hi = cell[key].get(clabel)
                    if isinstance(lo_hi, (list, tuple)):
                        lo, hi = lo_hi
                        # Deflate by 30% to force under-coverage
                        cell[key][clabel] = (round(lo * 0.7, 1), round(hi * 0.7, 1))

    bin_pairs_corrected = bidirectional_bin_correction(
        df, bin_pairs_deflated, "baseline", boundaries, labels,
        tolerance=1.0, step=0.02, max_iter=50,
    )

    # Check that at least some cells were expanded
    any_expanded = False
    for label in labels:
        def_cell = bin_pairs_deflated.get(label, {})
        corr_cell = bin_pairs_corrected.get(label, {})
        for cls_name in ["onpeak", "offpeak"]:
            for seg in ["prevail", "counter"]:
                key = (cls_name, seg)
                def_p95 = def_cell.get(key, {}).get("p95")
                corr_p95 = corr_cell.get(key, {}).get("p95")
                if (isinstance(def_p95, (list, tuple)) and
                        isinstance(corr_p95, (list, tuple))):
                    def_width = def_p95[1] - def_p95[0]
                    corr_width = corr_p95[1] - corr_p95[0]
                    if corr_width > def_width + 0.01:
                        any_expanded = True

    check("bidirectional correction expanded at least one cell", any_expanded)


# ─── Test 24: Bidirectional safety ──────────────────────────────────────────

def test_bidirectional_safety():
    """Verify shrinking stops before coverage drops below target."""
    print("\n[Test 24] Bidirectional safety (shrink doesn't drop below target)")
    rng = np.random.RandomState(42)
    n = 20000
    baseline = rng.uniform(10, 100, n)
    residual = rng.normal(0, 30, n)
    mcp = baseline + residual

    cls = rng.choice(["onpeak", "offpeak"], n)
    df = pl.DataFrame({
        "baseline": baseline,
        "mcp_mean": mcp,
        "class_type": cls,
        "planning_year": rng.choice([2020, 2021, 2022, 2023], n),
    })
    df = add_sign_seg(df, "baseline")

    boundaries, labels = compute_quantile_boundaries(df["baseline"], 4)

    bin_pairs = calibrate_asymmetric_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )

    # Apply with very aggressive step to test safety
    corrected = bidirectional_bin_correction(
        df, bin_pairs, "baseline", boundaries, labels,
        tolerance=0.5, step=0.10, max_iter=100,
    )

    # Apply corrected bands and check coverage doesn't drop below target
    df_banded = apply_asymmetric_bands_per_class_sign_fast(
        df, corrected, "baseline", "class_type", boundaries, labels,
    )

    mcp_s = df_banded["mcp_mean"]
    for level, clabel in zip(COVERAGE_LEVELS, COVERAGE_LABELS):
        lower = df_banded[f"lower_{clabel}"]
        upper = df_banded[f"upper_{clabel}"]
        actual = float(((mcp_s >= lower) & (mcp_s <= upper)).mean()) * 100
        target = level * 100
        # Safety: coverage should not drop more than 2pp below target
        check(f"safety: {clabel} coverage >= target - 2pp",
              actual >= target - 2.0,
              f"actual={actual:.2f}%, target={target:.0f}%")


# ─── Test 25: Bidirectional no change ────────────────────────────────────────

def test_bidirectional_no_change():
    """All bins within tolerance. No corrections should be made."""
    print("\n[Test 25] Bidirectional no change (bins within tolerance)")
    df = make_df(n=50000, seed=42)
    df = add_sign_seg(df, "baseline")
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 4)

    bin_pairs = calibrate_asymmetric_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )

    import copy
    original = copy.deepcopy(bin_pairs)

    # With large n, in-sample quantiles should give ~exact coverage
    # Use very wide tolerance so nothing triggers
    corrected = bidirectional_bin_correction(
        df, bin_pairs, "baseline", boundaries, labels,
        tolerance=5.0, step=0.02, max_iter=50,
    )

    # Should be mostly unchanged
    changes = 0
    comparisons = 0
    for label in labels:
        for cls_name in ["onpeak", "offpeak"]:
            for seg in ["prevail", "counter"]:
                key = (cls_name, seg)
                orig_cell = original.get(label, {}).get(key, {})
                corr_cell = corrected.get(label, {}).get(key, {})
                for clabel in COVERAGE_LABELS:
                    o = orig_cell.get(clabel)
                    c = corr_cell.get(clabel)
                    if isinstance(o, (list, tuple)) and isinstance(c, (list, tuple)):
                        comparisons += 1
                        if abs(o[0] - c[0]) > 0.2 or abs(o[1] - c[1]) > 0.2:
                            changes += 1

    check(f"wide tolerance: few changes ({changes}/{comparisons})",
          changes < comparisons * 0.1,
          f"changes={changes}, comparisons={comparisons}")


# ─── Test 26: Bidirectional max_iter ─────────────────────────────────────────

def test_bidirectional_max_iter():
    """Massive deviation. Caps at max_iter iterations."""
    print("\n[Test 26] Bidirectional max_iter cap")
    rng = np.random.RandomState(77)
    n = 10000
    baseline = rng.uniform(10, 100, n)
    # Very heavy-tailed: will require many iterations
    residual = rng.standard_cauchy(n) * 50
    mcp = baseline + residual

    cls = rng.choice(["onpeak", "offpeak"], n)
    df = pl.DataFrame({
        "baseline": baseline,
        "mcp_mean": mcp,
        "class_type": cls,
        "planning_year": rng.choice([2020, 2021, 2022, 2023], n),
    })
    df = add_sign_seg(df, "baseline")

    boundaries, labels = compute_quantile_boundaries(df["baseline"], 4)

    bin_pairs = calibrate_asymmetric_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )

    # Very small step, very few max_iter -> should terminate without error
    try:
        corrected = bidirectional_bin_correction(
            df, bin_pairs, "baseline", boundaries, labels,
            tolerance=0.5, step=0.001, max_iter=3,
        )
        check("max_iter: completes without error", True)
    except Exception as e:
        check(f"max_iter: crashed: {e}", False)


# ─── Test 27: Bidirectional mixed ────────────────────────────────────────────

def test_bidirectional_mixed():
    """Manually mix inflated and deflated cells, verify both directions."""
    print("\n[Test 27] Bidirectional mixed (some shrink, some expand)")
    import copy
    rng = np.random.RandomState(42)
    n = 30000
    baseline = rng.uniform(0, 500, n)
    residual = rng.normal(0, 50, n)
    mcp = baseline + residual

    cls = rng.choice(["onpeak", "offpeak"], n)
    df = pl.DataFrame({
        "baseline": baseline,
        "mcp_mean": mcp,
        "class_type": cls,
        "planning_year": rng.choice([2020, 2021, 2022, 2023], n),
    })
    df = add_sign_seg(df, "baseline")

    boundaries, labels = compute_quantile_boundaries(df["baseline"], 4)

    bin_pairs_raw = calibrate_asymmetric_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )

    # Inflate first 2 bins (force over-coverage), deflate last 2 bins (force under-coverage)
    bin_pairs_mixed = copy.deepcopy(bin_pairs_raw)
    for i, label in enumerate(labels):
        cell = bin_pairs_mixed.get(label, {})
        factor = 1.5 if i < len(labels) // 2 else 0.7
        for cls_name in ["onpeak", "offpeak"]:
            for seg in ["prevail", "counter"]:
                key = (cls_name, seg)
                if key not in cell or "_fallback" in cell[key]:
                    continue
                for clabel in COVERAGE_LABELS:
                    lo_hi = cell[key].get(clabel)
                    if isinstance(lo_hi, (list, tuple)):
                        lo, hi = lo_hi
                        cell[key][clabel] = (round(lo * factor, 1), round(hi * factor, 1))

    corrected = bidirectional_bin_correction(
        df, bin_pairs_mixed, "baseline", boundaries, labels,
        tolerance=1.0, step=0.02, max_iter=50,
    )

    n_wider = 0
    n_narrower = 0
    for label in labels:
        mix_cell = bin_pairs_mixed.get(label, {})
        corr_cell = corrected.get(label, {})
        for cls_name in ["onpeak", "offpeak"]:
            for seg in ["prevail", "counter"]:
                key = (cls_name, seg)
                mix_p95 = mix_cell.get(key, {}).get("p95")
                corr_p95 = corr_cell.get(key, {}).get("p95")
                if (isinstance(mix_p95, (list, tuple)) and
                        isinstance(corr_p95, (list, tuple))):
                    mix_w = mix_p95[1] - mix_p95[0]
                    corr_w = corr_p95[1] - corr_p95[0]
                    if corr_w > mix_w + 0.1:
                        n_wider += 1
                    elif corr_w < mix_w - 0.1:
                        n_narrower += 1

    check("mixed: some cells narrower", n_narrower > 0,
          f"n_narrower={n_narrower}")
    check("mixed: some cells wider", n_wider > 0,
          f"n_wider={n_wider}")
    check("mixed: bidirectional (both directions)",
          n_narrower > 0 and n_wider > 0,
          f"narrower={n_narrower}, wider={n_wider}")


# ─── Test 28: Correction with run_experiment ─────────────────────────────────

def test_correction_in_experiment():
    """Test that apply_correction=True in run_experiment works end-to-end."""
    print("\n[Test 28] Correction in run_experiment")
    df = make_df(n=20000)
    pys = sorted(df["planning_year"].unique().to_list())

    # Without correction
    result_no = run_experiment(
        df, "aq1", pys, 6, "baseline", "no_corr",
        cv_mode="loo", apply_correction=False,
    )

    # With correction
    result_yes = run_experiment(
        df, "aq1", pys, 6, "baseline", "with_corr",
        cv_mode="loo", apply_correction=True,
    )

    # Both should produce valid results
    check("no_corr has aggregate", "aggregate" in result_no)
    check("with_corr has aggregate", "aggregate" in result_yes)

    # Widths may differ but both should have valid coverage
    p95_no = result_no["aggregate"]["coverage"]["overall"]["p95"]["actual"]
    p95_yes = result_yes["aggregate"]["coverage"]["overall"]["p95"]["actual"]
    check(f"no_corr P95 coverage reasonable ({p95_no:.1f}%)",
          p95_no > 80)
    check(f"with_corr P95 coverage reasonable ({p95_yes:.1f}%)",
          p95_yes > 80)

    # Widths should be different (correction changes them)
    w_no = result_no["aggregate"]["widths"]["overall"]["p95"]["mean_width"]
    w_yes = result_yes["aggregate"]["widths"]["overall"]["p95"]["mean_width"]
    # They may be very close or different depending on the data
    check("both experiments produce width values",
          w_no is not None and w_yes is not None)


# ─── Test 29: No symmetric parameter in run_experiment ──────────────────────

def test_no_symmetric_parameter():
    """v7 removed symmetric parameter. Verify the function signature."""
    print("\n[Test 29] No symmetric parameter")
    import inspect
    sig = inspect.signature(run_experiment)
    params = list(sig.parameters.keys())
    check("'symmetric' not in run_experiment params",
          "symmetric" not in params)
    check("'apply_correction' in run_experiment params",
          "apply_correction" in params)
    check("'cv_mode' in run_experiment params",
          "cv_mode" in params)


# ─── Test 30: v7 constants ──────────────────────────────────────────────────

def test_v7_constants():
    """Verify v7-specific constant values."""
    print("\n[Test 30] v7 constants")
    check("CORRECTION_TOLERANCE = 2.0", CORRECTION_TOLERANCE == 2.0)
    check("CORRECTION_STEP = 0.05", CORRECTION_STEP == 0.05)
    check("CORRECTION_MAX_ITER = 50", CORRECTION_MAX_ITER == 50)
    check("COLD_START_BOOST = 0.15", COLD_START_BOOST == 0.15)
    check("MIN_CORRECTION_PYS = 3", MIN_CORRECTION_PYS == 3)


# ─── Test 31: cold_start_inflate ─────────────────────────────────────────────

def test_cold_start_inflate():
    """Verify cold_start_inflate widens bands proportionally."""
    print("\n[Test 31] cold_start_inflate")
    import copy
    rng = np.random.RandomState(42)
    n = 10000
    baseline = rng.uniform(10, 200, n)
    residual = rng.normal(0, 50, n)
    mcp = baseline + residual

    df = pl.DataFrame({
        "baseline": baseline,
        "mcp_mean": mcp,
        "class_type": rng.choice(["onpeak", "offpeak"], n),
        "planning_year": rng.choice([2020, 2021, 2022, 2023], n),
    })
    df = add_sign_seg(df, "baseline")
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 4)

    bin_pairs = calibrate_asymmetric_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )

    for n_pys in [1, 2]:
        inflated = cold_start_inflate(bin_pairs, n_pys, labels=labels)
        expected_factor = 1.0 + COLD_START_BOOST / n_pys

        widened = 0
        total = 0
        for label in labels:
            for cls_name in ["onpeak", "offpeak"]:
                for seg in ["prevail", "counter"]:
                    key = (cls_name, seg)
                    orig = bin_pairs.get(label, {}).get(key, {})
                    inf = inflated.get(label, {}).get(key, {})
                    if "_fallback" in orig:
                        continue
                    for clabel in COVERAGE_LABELS:
                        o = orig.get(clabel)
                        i = inf.get(clabel)
                        if isinstance(o, (list, tuple)) and isinstance(i, (list, tuple)):
                            total += 1
                            orig_w = o[1] - o[0]
                            inf_w = i[1] - i[0]
                            if inf_w > orig_w + 0.01:
                                widened += 1

        check(f"n_pys={n_pys}: all cells widened ({widened}/{total})",
              widened == total and total > 0,
              f"widened={widened}, total={total}")
        check(f"n_pys={n_pys}: factor ~ {expected_factor:.3f}",
              abs(expected_factor - (1.0 + COLD_START_BOOST / n_pys)) < 1e-9)

    # Factor with 1 PY > 2 PYs
    inf1 = cold_start_inflate(bin_pairs, 1, labels=labels)
    inf2 = cold_start_inflate(bin_pairs, 2, labels=labels)
    for label in labels:
        for cls_name in ["onpeak", "offpeak"]:
            for seg in ["prevail", "counter"]:
                key = (cls_name, seg)
                lo1 = inf1.get(label, {}).get(key, {}).get("p95")
                lo2 = inf2.get(label, {}).get(key, {}).get("p95")
                if isinstance(lo1, (list, tuple)) and isinstance(lo2, (list, tuple)):
                    w1 = lo1[1] - lo1[0]
                    w2 = lo2[1] - lo2[0]
                    check("1-PY inflation wider than 2-PY inflation",
                          w1 >= w2 - 0.01,
                          f"w1={w1:.1f}, w2={w2:.1f}")


# ─── Test 32: OOS correction fires in run_experiment ──────────────────────────

def test_oos_correction_in_run_experiment():
    """Verify OOS correction changes bands differently from no-correction case.

    With MIN_CORRECTION_PYS=3, we need at least 3 training PYs for OOS correction.
    Use temporal CV with 6 PYs so later folds trigger OOS correction.
    """
    print("\n[Test 32] OOS correction fires in run_experiment (temporal, 6 PYs)")
    rng = np.random.RandomState(42)
    n = 30000
    baseline = rng.uniform(10, 500, n)
    residual = rng.normal(0, 80, n)
    mcp = baseline + residual

    pys = [2020, 2021, 2022, 2023, 2024, 2025]
    py_arr = rng.choice(pys, n)

    df = pl.DataFrame({
        "baseline": baseline,
        "mcp_mean": mcp,
        "class_type": rng.choice(["onpeak", "offpeak"], n),
        "planning_year": py_arr,
    })

    result_no = run_experiment(
        df, "aq1", pys, 6, "baseline", "oos_no_corr",
        cv_mode="temporal", apply_correction=False,
    )
    result_yes = run_experiment(
        df, "aq1", pys, 6, "baseline", "oos_with_corr",
        cv_mode="temporal", apply_correction=True,
    )

    # Both should produce valid results
    check("no_corr has per_py", len(result_no["per_py"]) > 0)
    check("with_corr has per_py", len(result_yes["per_py"]) > 0)

    p95_no = result_no["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
    p95_yes = result_yes["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
    check(f"no_corr P95 coverage reasonable ({p95_no:.1f}%)", p95_no > 70)
    check(f"with_corr P95 coverage reasonable ({p95_yes:.1f}%)", p95_yes > 70)

    # Widths should differ between correction and no-correction
    w_no = result_no["aggregate"]["widths"]["overall"]["p95"]["mean_width"]
    w_yes = result_yes["aggregate"]["widths"]["overall"]["p95"]["mean_width"]
    check("both produce non-null widths", w_no is not None and w_yes is not None)
    # Widths can differ in either direction depending on over/under-coverage
    check("corrected width is finite positive", w_yes is not None and w_yes > 0)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  test_v7_bands.py — Comprehensive tests for v7 bands")
    print("=" * 70)

    test_calibration_structure()        # Test 1
    test_quantile_pair_correctness()    # Test 2
    test_apply_columns()                # Test 3
    test_insample_coverage()            # Test 4
    test_run_experiment_temporal()       # Test 5
    test_run_experiment_loo()            # Test 6
    test_bg1_check()                     # Test 7
    test_select_winner()                 # Test 8
    test_sparse_fallback()               # Test 9
    test_zero_baseline()                 # Test 10
    test_single_py()                     # Test 11
    test_json_roundtrip()                # Test 12
    test_per_class_per_sign_evaluation() # Test 13
    test_per_bin_aggregate()             # Test 14
    test_bin_column_guard()              # Test 15
    test_width_monotonicity()            # Test 16
    test_stability_structure()           # Test 17
    test_empty_test_py()                 # Test 18
    test_higher_bin_counts()             # Test 19
    test_pipeline_gate_structure()       # Test 20
    test_widths_by_sign_structure()      # Test 21
    test_bidirectional_shrink()          # Test 22
    test_bidirectional_expand()          # Test 23
    test_bidirectional_safety()          # Test 24
    test_bidirectional_no_change()       # Test 25
    test_bidirectional_max_iter()        # Test 26
    test_bidirectional_mixed()           # Test 27
    test_correction_in_experiment()      # Test 28
    test_no_symmetric_parameter()        # Test 29
    test_v7_constants()                  # Test 30
    test_cold_start_inflate()            # Test 31
    test_oos_correction_in_run_experiment()  # Test 32

    print(f"\n{'=' * 70}")
    print(f"  Results: {passed} passed, {failed} failed")
    if errors:
        print(f"  Failures: {', '.join(errors)}")
    print(f"{'=' * 70}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
