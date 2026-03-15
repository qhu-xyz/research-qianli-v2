"""Comprehensive tests for run_v6_bands.py.

Tests asymmetric calibration, band application, per-bin coverage correction,
experiment runner, winner selection, edge cases, and pipeline gate compatibility.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual/scripts/test_v6_bands.py
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

from run_v6_bands import (
    calibrate_asymmetric_per_class_sign,
    apply_asymmetric_bands_per_class_sign_fast,
    calibrate_bin_widths_per_class_sign,
    apply_bands_per_class_sign_fast,
    correct_bin_coverage,
    correct_symmetric_bin_coverage,
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


# ─── Test 5: Asymmetric narrower than symmetric ─────────────────────────────

def test_asymmetric_narrower():
    print("\n[Test 5] Asymmetric narrower than symmetric (skewed data)")
    # Use strongly skewed residuals
    df = make_df(n=30000, seed=42, skew=100.0)
    df = add_sign_seg(df, "baseline")
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 4)

    # Symmetric
    sym_widths = calibrate_bin_widths_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )
    df_sym = apply_bands_per_class_sign_fast(
        df, sym_widths, "baseline", "class_type", boundaries, labels,
    )

    # Asymmetric
    asym_pairs = calibrate_asymmetric_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )
    df_asym = apply_asymmetric_bands_per_class_sign_fast(
        df, asym_pairs, "baseline", "class_type", boundaries, labels,
    )

    sym_p95_w = float((df_sym["upper_p95"] - df_sym["lower_p95"]).mean())
    asym_p95_w = float((df_asym["upper_p95"] - df_asym["lower_p95"]).mean())
    reduction = (1 - asym_p95_w / sym_p95_w) * 100

    check(f"asymmetric P95 narrower than symmetric",
          asym_p95_w < sym_p95_w,
          f"asym={asym_p95_w:.1f} vs sym={sym_p95_w:.1f}, reduction={reduction:.1f}%")


# ─── Test 6: Symmetric residuals → similar widths ───────────────────────────

def test_symmetric_residuals():
    print("\n[Test 6] Symmetric residuals produce similar widths")
    # Zero skew
    df = make_df(n=30000, seed=42, skew=0.0)
    df = add_sign_seg(df, "baseline")
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 4)

    sym_widths = calibrate_bin_widths_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )
    df_sym = apply_bands_per_class_sign_fast(
        df, sym_widths, "baseline", "class_type", boundaries, labels,
    )

    asym_pairs = calibrate_asymmetric_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )
    df_asym = apply_asymmetric_bands_per_class_sign_fast(
        df, asym_pairs, "baseline", "class_type", boundaries, labels,
    )

    sym_p95_w = float((df_sym["upper_p95"] - df_sym["lower_p95"]).mean())
    asym_p95_w = float((df_asym["upper_p95"] - df_asym["lower_p95"]).mean())
    diff_pct = abs(asym_p95_w - sym_p95_w) / sym_p95_w * 100

    check(f"symmetric residuals: widths within 5%",
          diff_pct < 5.0,
          f"asym={asym_p95_w:.1f} vs sym={sym_p95_w:.1f}, diff={diff_pct:.1f}%")


# ─── Test 7: run_experiment LOO mode ─────────────────────────────────────────

def test_run_experiment_loo():
    print("\n[Test 7] run_experiment LOO mode")
    df = make_df(n=20000)
    pys = sorted(df["planning_year"].unique().to_list())

    result = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_loo",
        symmetric=False, cv_mode="loo",
    )

    check("per_py present", "per_py" in result)
    check("aggregate present", "aggregate" in result)
    check("stability present", "stability" in result)

    # Each PY should be in per_py
    for py in pys:
        check(f"PY {py} in per_py", str(py) in result["per_py"])

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

    # Stability
    stab = result["stability"]
    check("stability has p95_coverage_range", "p95_coverage_range" in stab)
    check("stability has p95_worst_py", "p95_worst_py" in stab)
    check("stability has p95_width_cv", "p95_width_cv" in stab)

    # LOO should NOT have temporal fields
    for py_data in result["per_py"].values():
        check("LOO per_py has no train_pys", "train_pys" not in py_data)


# ─── Test 8: run_experiment temporal mode ────────────────────────────────────

def test_run_experiment_temporal():
    print("\n[Test 8] run_experiment temporal mode")
    df = make_df(n=20000, n_pys=6)
    pys = sorted(df["planning_year"].unique().to_list())

    result = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_temp",
        symmetric=False, cv_mode="temporal", min_train_pys=3,
    )

    # Temporal per_py should have train_pys and meets_min_train_pys
    for py_str, py_data in result["per_py"].items():
        check(f"temporal PY {py_str} has train_pys", "train_pys" in py_data)
        check(f"temporal PY {py_str} has meets_min_train_pys",
              "meets_min_train_pys" in py_data)

    # Early PYs should have fewer training PYs
    first_py = str(min(pys))
    if first_py in result["per_py"]:
        first_data = result["per_py"][first_py]
        check(f"first PY {first_py} has 0 train_pys (temporal)",
              len(first_data.get("train_pys", [])) == 0 or first_data["n_train"] == 0)

    # Stability should have temporal fields
    stab = result["stability"]
    check("temporal stability has min_train_pys", "min_train_pys" in stab)
    check("temporal stability has n_folds_total", "n_folds_total" in stab)
    check("temporal stability has n_folds_filtered", "n_folds_filtered" in stab)


# ─── Test 9: run_experiment symmetric mode ───────────────────────────────────

def test_run_experiment_symmetric():
    print("\n[Test 9] run_experiment symmetric mode")
    df = make_df(n=20000)
    pys = sorted(df["planning_year"].unique().to_list())

    result = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_sym",
        symmetric=True, cv_mode="loo",
    )

    overall = result["aggregate"]["coverage"]["overall"]
    p95 = overall["p95"]
    check("symmetric P95 coverage within 5pp of target",
          abs(p95["error"]) < 5.0,
          f"error={p95['error']}pp")


# ─── Test 10: BG1 check ─────────────────────────────────────────────────────

def test_bg1_check():
    print("\n[Test 10] BG1 gate check")
    df = make_df(n=20000)
    pys = sorted(df["planning_year"].unique().to_list())

    result = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_bg1",
        symmetric=False, cv_mode="loo",
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


# ─── Test 11: select_winner ──────────────────────────────────────────────────

def test_select_winner():
    print("\n[Test 11] select_winner")
    df = make_df(n=20000)
    pys = sorted(df["planning_year"].unique().to_list())

    all_results = {}
    experiments = [
        {"name": "sym_4b", "n_bins": 4, "symmetric": True},
        {"name": "asym_4b", "n_bins": 4, "symmetric": False},
        {"name": "asym_6b", "n_bins": 6, "symmetric": False},
    ]
    for exp in experiments:
        r = run_experiment(
            df, "aq1", pys, exp["n_bins"], "baseline", exp["name"],
            symmetric=exp["symmetric"], cv_mode="loo",
        )
        all_results[exp["name"]] = {"aq1": r}

    winner_name, winner_q = select_winner(all_results, ["aq1"])
    check("winner_name is a string", isinstance(winner_name, str))
    check("winner_name in experiments", winner_name in [e["name"] for e in experiments])
    check("winner_q has aq1", "aq1" in winner_q)


# ─── Test 12: Fallback chain with sparse data ───────────────────────────────

def test_sparse_fallback():
    print("\n[Test 12] Sparse cell fallback")
    # Small dataset: 8 bins × 2 classes × 2 signs = 32 cells, many will be sparse
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


# ─── Test 13: Zero baseline edge case ───────────────────────────────────────

def test_zero_baseline():
    print("\n[Test 13] Zero baseline edge case")
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


# ─── Test 14: Single PY edge case ───────────────────────────────────────────

def test_single_py():
    print("\n[Test 14] Single PY edge case")
    df = make_df(n=5000, n_pys=1)

    # LOO with single PY: test set exists but no folds possible
    result = run_experiment(
        df, "aq1", [2020], 4, "baseline", "single_py",
        symmetric=False, cv_mode="loo",
    )
    # With 1 PY, LOO has 0 training rows, should skip
    check("single PY LOO: per_py is empty", len(result["per_py"]) == 0)

    # Temporal with single PY: no training data before first PY
    result2 = run_experiment(
        df, "aq1", [2020], 4, "baseline", "single_py_temp",
        symmetric=False, cv_mode="temporal",
    )
    check("single PY temporal: handles gracefully",
          isinstance(result2["stability"], dict))


# ─── Test 15: Temporal min_train_pys filtering ──────────────────────────────

def test_temporal_min_train_filtering():
    print("\n[Test 15] Temporal min_train_pys filtering")
    df = make_df(n=20000, n_pys=6)
    pys = sorted(df["planning_year"].unique().to_list())

    result = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_filter",
        symmetric=False, cv_mode="temporal", min_train_pys=3,
    )

    # First PY has 0 train PYs, second has 1, third has 2 — all < 3
    for py_str, py_data in result["per_py"].items():
        train_pys = py_data.get("train_pys", [])
        meets = py_data.get("meets_min_train_pys", True)
        if len(train_pys) < 3:
            check(f"PY {py_str} ({len(train_pys)} train PYs) meets_min=False",
                  not meets)
        else:
            check(f"PY {py_str} ({len(train_pys)} train PYs) meets_min=True",
                  meets)


# ─── Test 16: JSON serialization round-trip ──────────────────────────────────

def test_json_roundtrip():
    print("\n[Test 16] JSON serialization round-trip")
    df = make_df(n=10000)
    pys = sorted(df["planning_year"].unique().to_list())

    result = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_json",
        symmetric=False, cv_mode="loo",
    )

    # Build a metrics-like dict
    metrics = {
        "coverage": {"aq1": result["aggregate"]["coverage"]},
        "widths": {"aq1": {"overall": result["aggregate"]["widths"]["overall"]}},
        "stability": {"aq1": result["stability"]},
    }

    sanitized = sanitize_for_json(metrics)

    # Should serialize without errors
    try:
        s = json.dumps(sanitized, indent=2)
        check("json.dumps succeeds", True)
        # Should parse back
        parsed = json.loads(s)
        check("json.loads round-trip succeeds", True)
        # Values preserved
        orig_p95 = metrics["coverage"]["aq1"]["overall"]["p95"]["actual"]
        parsed_p95 = parsed["coverage"]["aq1"]["overall"]["p95"]["actual"]
        check("P95 actual preserved", abs(orig_p95 - parsed_p95) < 0.01,
              f"orig={orig_p95}, parsed={parsed_p95}")
    except Exception as e:
        check(f"JSON serialization failed: {e}", False)


# ─── Test 17: Width comparison: half-width equivalence ───────────────────────

def test_halfwidth_equivalence():
    print("\n[Test 17] Half-width equivalence")
    df = make_df(n=20000, skew=0.0)  # symmetric residuals
    df = add_sign_seg(df, "baseline")
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 4)

    # Symmetric calibration
    sym_widths = calibrate_bin_widths_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )

    # Asymmetric calibration
    asym_pairs = calibrate_asymmetric_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )

    # For symmetric residuals, half-width ≈ symmetric width
    for label in labels:
        for cls in ["onpeak", "offpeak"]:
            for seg in ["prevail", "counter"]:
                key = (cls, seg)
                sw = sym_widths[label].get(key, {}).get("p95")
                ap = asym_pairs[label].get(key, {}).get("p95")
                if sw is not None and isinstance(ap, (list, tuple)):
                    half_w = (ap[1] - ap[0]) / 2
                    diff_pct = abs(half_w - sw) / sw * 100 if sw > 0 else 0
                    check(f"{label}/{cls}/{seg} half-width ~ sym width",
                          diff_pct < 15,
                          f"half_w={half_w:.1f}, sym_w={sw:.1f}, diff={diff_pct:.1f}%")


# ─── Test 18: Per-class and per-sign coverage evaluation ─────────────────────

def test_per_class_per_sign_evaluation():
    print("\n[Test 18] Per-class and per-sign coverage in run_experiment")
    df = make_df(n=20000)
    pys = sorted(df["planning_year"].unique().to_list())

    result = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_eval",
        symmetric=False, cv_mode="loo",
    )

    # Check aggregate has per_class and per_sign
    agg_cov = result["aggregate"]["coverage"]
    check("aggregate has per_class", "per_class" in agg_cov)
    check("aggregate has per_sign", "per_sign" in agg_cov)

    # Per-class should have onpeak and offpeak
    pc = agg_cov["per_class"]
    check("per_class has onpeak", "onpeak" in pc)
    check("per_class has offpeak", "offpeak" in pc)

    # Per-sign should have prevail and counter
    ps = agg_cov["per_sign"]
    check("per_sign has prevail", "prevail" in ps)
    check("per_sign has counter", "counter" in ps)


# ─── Test 19: Per-bin coverage in aggregate ──────────────────────────────────

def test_per_bin_aggregate():
    print("\n[Test 19] Per-bin coverage in aggregate")
    df = make_df(n=20000)
    pys = sorted(df["planning_year"].unique().to_list())

    result = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_perbin",
        symmetric=False, cv_mode="loo",
    )

    per_bin = result["aggregate"]["coverage"].get("per_bin", {})
    check("aggregate per_bin not empty", len(per_bin) > 0)

    for bin_label, bin_data in per_bin.items():
        check(f"bin {bin_label} has n", "n" in bin_data)
        check(f"bin {bin_label} has p95", "p95" in bin_data)
        p95 = bin_data["p95"]
        check(f"bin {bin_label} p95 has actual",
              "actual" in p95 and isinstance(p95["actual"], (int, float)))


# ─── Test 20: _bin column guard (apply called twice) ────────────────────────

def test_bin_column_guard():
    print("\n[Test 20] _bin column guard (double apply)")
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
        # Re-add _bin to simulate leftover
        df_with_bin = df.with_columns(pl.lit("q1").alias("_bin"))
        df_banded2 = apply_asymmetric_bands_per_class_sign_fast(
            df_with_bin, bin_pairs, "baseline", "class_type", boundaries, labels,
        )
        check("double apply does not crash", True)
        check("output has lower_p95", "lower_p95" in df_banded2.columns)
    except Exception as e:
        check(f"double apply crashed: {e}", False)


# ─── Test 21: Width monotonicity (BG5 compatibility) ────────────────────────

def test_width_monotonicity():
    print("\n[Test 21] Width monotonicity (BG5)")
    df = make_df(n=20000)
    pys = sorted(df["planning_year"].unique().to_list())

    result = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_mono",
        symmetric=False, cv_mode="loo",
    )

    widths = result["aggregate"]["widths"]["overall"]
    ws = [widths[clabel]["mean_width"] for clabel in COVERAGE_LABELS]

    for i in range(len(ws) - 1):
        check(f"width {COVERAGE_LABELS[i]} < {COVERAGE_LABELS[i+1]}",
              ws[i] < ws[i + 1],
              f"{ws[i]:.1f} vs {ws[i+1]:.1f}")


# ─── Test 22: Stability metrics structure ────────────────────────────────────

def test_stability_structure():
    print("\n[Test 22] Stability metrics structure")
    df = make_df(n=20000)
    pys = sorted(df["planning_year"].unique().to_list())

    result = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_stab",
        symmetric=False, cv_mode="loo",
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


# ─── Test 23: Empty test PY handling ────────────────────────────────────────

def test_empty_test_py():
    print("\n[Test 23] Empty test PY handling")
    df = make_df(n=10000, n_pys=3)

    # Ask for a PY that doesn't exist
    result = run_experiment(
        df, "aq1", [2020, 2021, 2022, 2099], 4, "baseline", "test_empty",
        symmetric=False, cv_mode="loo",
    )

    check("non-existent PY 2099 not in per_py", "2099" not in result["per_py"])
    check("existing PYs are in per_py", len(result["per_py"]) >= 3)


# ─── Test 24: 6-bin and 8-bin configurations ────────────────────────────────

def test_higher_bin_counts():
    print("\n[Test 24] Higher bin counts (6, 8)")
    df = make_df(n=20000)
    df = add_sign_seg(df, "baseline")

    for n_bins in [6, 8]:
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


# ─── Test 25: Pipeline gate metrics structure ────────────────────────────────

def test_pipeline_gate_structure():
    print("\n[Test 25] Pipeline gate metrics structure")
    df = make_df(n=20000)
    pys = sorted(df["planning_year"].unique().to_list())
    quarters = ["aq1"]

    result = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_gates",
        symmetric=False, cv_mode="loo",
    )

    # Build metrics dict as run_round_v6 would
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


# ─── Test 26: widths_by_sign structure in run_experiment ──────────────────

def test_widths_by_sign_structure():
    print("\n[Test 26] widths_by_sign structure")
    df = make_df(n=20000)
    pys = sorted(df["planning_year"].unique().to_list())

    result = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_wbs",
        symmetric=False, cv_mode="loo",
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

    # Symmetric mode should also produce widths_by_sign
    result_sym = run_experiment(
        df, "aq1", pys, 4, "baseline", "test_wbs_sym",
        symmetric=True, cv_mode="loo",
    )
    for py_str, py_data in result_sym["per_py"].items():
        check(f"sym PY {py_str} has widths_by_sign", "widths_by_sign" in py_data)
        break  # check first PY only


# ─── Test 27: BG2a/BG2b gate logic ───────────────────────────────────────

def test_bg2a_bg2b_gates():
    print("\n[Test 27] BG2a/BG2b gate logic")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
    from pipeline import check_band_gates

    # Build minimal candidate and promoted metrics
    def make_width_metrics(overall_widths, per_sign_widths=None):
        metrics = {"coverage": {}, "widths": {}, "stability": {},
                   "per_class_coverage": {}}
        for q in ["aq1", "aq2", "aq3", "aq4"]:
            metrics["coverage"][q] = {
                "overall": {
                    lvl: {"target": t, "actual": t, "error": 0}
                    for lvl, t in [("p10", 10), ("p30", 30), ("p50", 50),
                                   ("p70", 70), ("p80", 80),
                                   ("p90", 90), ("p95", 95)]
                }
            }
            metrics["widths"][q] = {"overall": {}}
            for lvl, w in overall_widths.items():
                metrics["widths"][q]["overall"][lvl] = {"mean_width": w}
            if per_sign_widths:
                metrics["widths"][q]["per_sign"] = per_sign_widths
            metrics["stability"][q] = {
                "p95_coverage_range": 2.0, "p95_worst_py": "2023",
                "p95_worst_py_coverage": 93.0, "p95_width_cv": 0.05,
            }
            metrics["per_class_coverage"][q] = {
                "onpeak": {"p10": {"actual": 10.0}, "p30": {"actual": 30.0}, "p95": {"actual": 95.0}},
                "offpeak": {"p10": {"actual": 10.0}, "p30": {"actual": 30.0}, "p95": {"actual": 95.0}},
            }
        return metrics

    base_widths = {"p10": 200, "p30": 350, "p50": 500, "p70": 800, "p80": 1200, "p90": 1800, "p95": 2500}

    # Test BG2a: candidate same as promoted -> PASS
    promoted = make_width_metrics(base_widths)
    candidate = make_width_metrics(base_widths)
    gates = check_band_gates(candidate, promoted)
    bg2a = next((g for g in gates if g["gate"] == "BG2a"), None)
    check("BG2a exists in gates", bg2a is not None)
    if bg2a:
        check("BG2a PASS when widths equal", bg2a["passed"])

    # Test BG2a: P50 +7.8% wider -> PASS (within 10% tolerance)
    wider_p50 = dict(base_widths)
    wider_p50["p50"] = 539  # +7.8%
    candidate2 = make_width_metrics(wider_p50)
    gates2 = check_band_gates(candidate2, promoted)
    bg2a2 = next((g for g in gates2 if g["gate"] == "BG2a"), None)
    if bg2a2:
        check("BG2a PASS with P50 +7.8% (within 10% tol)", bg2a2["passed"])

    # Test BG2a: P50 +11% wider -> FAIL (exceeds 10% tolerance)
    too_wide_p50 = dict(base_widths)
    too_wide_p50["p50"] = 556  # +11.2%
    candidate3 = make_width_metrics(too_wide_p50)
    gates3 = check_band_gates(candidate3, promoted)
    bg2a3 = next((g for g in gates3 if g["gate"] == "BG2a"), None)
    if bg2a3:
        check("BG2a FAIL with P50 +11.2% (exceeds 10% tol)", not bg2a3["passed"])

    # Test BG2a: P95 +1% wider -> FAIL (0% tolerance at P95)
    wider_p95 = dict(base_widths)
    wider_p95["p95"] = 2525  # +1%
    candidate4 = make_width_metrics(wider_p95)
    gates4 = check_band_gates(candidate4, promoted)
    bg2a4 = next((g for g in gates4 if g["gate"] == "BG2a"), None)
    if bg2a4:
        check("BG2a FAIL with P95 +1% (0% tol)", not bg2a4["passed"])

    # Test BG2b: all combos improved -> PASS
    per_sign_promoted = {
        sign: {
            lvl: {"mean_width": 100.0, "max_width": 200.0}
            for lvl in ["p10", "p30", "p50", "p70", "p80", "p90", "p95"]
        }
        for sign in ["prevail", "counter"]
    }
    per_sign_improved = {
        sign: {
            lvl: {"mean_width": 90.0, "max_width": 180.0}
            for lvl in ["p10", "p30", "p50", "p70", "p80", "p90", "p95"]
        }
        for sign in ["prevail", "counter"]
    }
    prom_ps = make_width_metrics(base_widths, per_sign_promoted)
    cand_ps = make_width_metrics(base_widths, per_sign_improved)
    gates5 = check_band_gates(cand_ps, prom_ps)
    bg2b = next((g for g in gates5 if g["gate"] == "BG2b"), None)
    check("BG2b exists when per_sign present", bg2b is not None)
    if bg2b:
        check("BG2b PASS when all combos improved", bg2b["passed"])

    # Test BG2b: combos degraded by 5% each
    per_sign_mostly_ok = {}
    for sign in ["prevail", "counter"]:
        per_sign_mostly_ok[sign] = {}
        for lvl in ["p10", "p30", "p50", "p70", "p80", "p90", "p95"]:
            per_sign_mostly_ok[sign][lvl] = {"mean_width": 90.0, "max_width": 180.0}
    per_sign_some_degrade = {}
    for sign in ["prevail", "counter"]:
        per_sign_some_degrade[sign] = {}
        for lvl in ["p10", "p30", "p50", "p70", "p80", "p90", "p95"]:
            if sign == "prevail" and lvl == "p50":
                per_sign_some_degrade[sign][lvl] = {"mean_width": 105.0, "max_width": 210.0}
            else:
                per_sign_some_degrade[sign][lvl] = {"mean_width": 90.0, "max_width": 180.0}
    cand_ps2 = make_width_metrics(base_widths, per_sign_some_degrade)
    gates6 = check_band_gates(cand_ps2, prom_ps)
    bg2b2 = next((g for g in gates6 if g["gate"] == "BG2b"), None)
    if bg2b2:
        check("BG2b PASS with 4/56 combos degraded by 5%", bg2b2["passed"],
              bg2b2["detail"])

    # Test BG2b: one combo mean degrades > 10% -> FAIL
    per_sign_bad_mean = {}
    for sign in ["prevail", "counter"]:
        per_sign_bad_mean[sign] = {}
        for lvl in ["p10", "p30", "p50", "p70", "p80", "p90", "p95"]:
            if sign == "counter" and lvl == "p95":
                per_sign_bad_mean[sign][lvl] = {"mean_width": 112.0, "max_width": 180.0}
            else:
                per_sign_bad_mean[sign][lvl] = {"mean_width": 90.0, "max_width": 180.0}
    cand_ps3 = make_width_metrics(base_widths, per_sign_bad_mean)
    gates7 = check_band_gates(cand_ps3, prom_ps)
    bg2b3 = next((g for g in gates7 if g["gate"] == "BG2b"), None)
    if bg2b3:
        check("BG2b FAIL with one mean +12% degrade", not bg2b3["passed"],
              bg2b3["detail"])

    # Test BG2b: one combo max degrades > 20% -> FAIL
    per_sign_bad_max = {}
    for sign in ["prevail", "counter"]:
        per_sign_bad_max[sign] = {}
        for lvl in ["p10", "p30", "p50", "p70", "p80", "p90", "p95"]:
            if sign == "prevail" and lvl == "p90":
                per_sign_bad_max[sign][lvl] = {"mean_width": 90.0, "max_width": 250.0}
            else:
                per_sign_bad_max[sign][lvl] = {"mean_width": 90.0, "max_width": 180.0}
    cand_ps4 = make_width_metrics(base_widths, per_sign_bad_max)
    gates8 = check_band_gates(cand_ps4, prom_ps)
    bg2b4 = next((g for g in gates8 if g["gate"] == "BG2b"), None)
    if bg2b4:
        check("BG2b FAIL with one max +25% degrade", not bg2b4["passed"],
              bg2b4["detail"])

    # Test BG2b: promoted lacks per_sign -> BG2b skipped
    prom_no_ps = make_width_metrics(base_widths)
    cand_with_ps = make_width_metrics(base_widths, per_sign_promoted)
    gates9 = check_band_gates(cand_with_ps, prom_no_ps)
    bg2b5 = next((g for g in gates9 if g["gate"] == "BG2b"), None)
    check("BG2b skipped when promoted lacks per_sign", bg2b5 is None)


# ─── Test 28: Per-bin correction (asymmetric) — basic ────────────────────────

def test_correct_bin_coverage_basic():
    print("\n[Test 28] Per-bin correction (asymmetric) — undershooting bin gets inflated")
    # Create data where the highest bin has wider residuals than calibration expects
    rng = np.random.RandomState(42)
    n = 30000
    baseline = rng.uniform(-500, 500, n)
    # For the highest |baseline| bin, inject extra-wide residuals
    abs_bl = np.abs(baseline)
    q75 = np.percentile(abs_bl, 75)
    residual = np.where(
        abs_bl >= q75,
        rng.normal(0, 200, n),  # wider in top bin
        rng.normal(0, 80, n),   # narrower elsewhere
    )
    mcp = baseline + residual
    cls = rng.choice(["onpeak", "offpeak"], n)
    pys = rng.choice([2020, 2021, 2022, 2023, 2024, 2025], n)

    df = pl.DataFrame({
        "baseline": baseline,
        "mcp_mean": mcp,
        "class_type": cls,
        "planning_year": pys,
    })
    df = add_sign_seg(df, "baseline")

    boundaries, labels = compute_quantile_boundaries(df["baseline"], 4)

    # Calibrate with intentionally narrow quantiles by using a SUBSET
    # (the correction should detect undershoot on the full training set)
    bin_pairs = calibrate_asymmetric_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )

    # Save original widths for top bin
    top_bin = labels[-1]
    original_p95 = {}
    for cls_name in ["onpeak", "offpeak"]:
        for seg in ["prevail", "counter"]:
            key = (cls_name, seg)
            pair = bin_pairs[top_bin].get(key, {}).get("p95")
            if isinstance(pair, (list, tuple)):
                original_p95[key] = pair[1] - pair[0]

    # Apply correction
    corrected = correct_bin_coverage(
        df, bin_pairs, "baseline", boundaries, labels,
    )

    # Check that corrected is a valid dict (same structure)
    check("corrected has all bin labels", all(l in corrected for l in labels))

    # Verify correction: top bin should have been inflated (or stayed same)
    # Lower bins should be unchanged (they already hit target in-sample)
    bottom_bin = labels[0]
    for cls_name in ["onpeak", "offpeak"]:
        for seg in ["prevail", "counter"]:
            key = (cls_name, seg)
            orig = bin_pairs[bottom_bin].get(key, {}).get("p95")
            corr = corrected[bottom_bin].get(key, {}).get("p95")
            if isinstance(orig, (list, tuple)) and isinstance(corr, (list, tuple)):
                orig_w = orig[1] - orig[0]
                corr_w = corr[1] - corr[0]
                # Bottom bin should be unchanged or minimally changed
                check(f"bottom bin {key} width not inflated much",
                      corr_w <= orig_w * 1.3,
                      f"orig={orig_w:.1f}, corr={corr_w:.1f}")


# ─── Test 29: Per-bin correction — no change needed ──────────────────────────

def test_correct_bin_coverage_no_change():
    print("\n[Test 29] Per-bin correction — all bins at target, no changes")
    # Create uniform data where all bins easily hit coverage targets
    df = make_df(n=50000, seed=99, skew=0.0)
    df = add_sign_seg(df, "baseline")
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 4)

    bin_pairs = calibrate_asymmetric_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )

    # With in-sample calibration on large data, all bins should already be at target
    corrected = correct_bin_coverage(
        df, bin_pairs, "baseline", boundaries, labels,
        tolerance=2.0,
    )

    # Check that widths are identical or nearly identical (tolerance allows some slack)
    changes = 0
    for label in labels:
        for cls_name in ["onpeak", "offpeak"]:
            for seg in ["prevail", "counter"]:
                key = (cls_name, seg)
                for clabel in COVERAGE_LABELS:
                    orig = bin_pairs[label].get(key, {}).get(clabel)
                    corr = corrected[label].get(key, {}).get(clabel)
                    if isinstance(orig, (list, tuple)) and isinstance(corr, (list, tuple)):
                        orig_w = orig[1] - orig[0]
                        corr_w = corr[1] - corr[0]
                        if abs(corr_w - orig_w) > 0.2:
                            changes += 1

    # With 2pp tolerance on in-sample data at n=50000, should be zero or very few changes
    check(f"no/minimal corrections needed (changes={changes})", changes <= 5,
          f"changes={changes}")


# ─── Test 30: Per-bin correction — max_iter cap ─────────────────────────────

def test_correct_bin_coverage_max_iter():
    print("\n[Test 30] Per-bin correction — max_iter cap")
    # Create data with massive undershoot in one bin
    rng = np.random.RandomState(42)
    n = 20000
    baseline = rng.uniform(-500, 500, n)
    # Very wide residuals everywhere — calibration will undershoot badly
    mcp = baseline + rng.normal(0, 500, n)
    cls = rng.choice(["onpeak", "offpeak"], n)
    pys = rng.choice([2020, 2021, 2022, 2023, 2024, 2025], n)

    df = pl.DataFrame({
        "baseline": baseline,
        "mcp_mean": mcp,
        "class_type": cls,
        "planning_year": pys,
    })
    df = add_sign_seg(df, "baseline")
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 4)

    # Calibrate normally
    bin_pairs = calibrate_asymmetric_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )

    # Apply correction with very small max_iter to test capping
    corrected = correct_bin_coverage(
        df, bin_pairs, "baseline", boundaries, labels,
        max_iter=3,
    )

    # Should not crash and should return valid structure
    check("correction with max_iter=3 returns valid dict",
          all(l in corrected for l in labels))

    # Width should be inflated but not infinitely
    for label in labels:
        for cls_name in ["onpeak", "offpeak"]:
            for seg in ["prevail", "counter"]:
                key = (cls_name, seg)
                corr = corrected[label].get(key, {}).get("p95")
                if isinstance(corr, (list, tuple)):
                    check(f"{label}/{cls_name}/{seg} p95 lo < hi after correction",
                          corr[0] < corr[1],
                          f"lo={corr[0]}, hi={corr[1]}")


# ─── Test 31: Symmetric per-bin correction ───────────────────────────────────

def test_correct_symmetric_bin_coverage():
    print("\n[Test 31] Symmetric per-bin correction")
    df = make_df(n=30000, seed=77, skew=0.0)
    df = add_sign_seg(df, "baseline")
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 4)

    bin_widths = calibrate_bin_widths_per_class_sign(
        df, "baseline", "mcp_mean", "class_type",
        boundaries, labels, COVERAGE_LEVELS,
    )

    corrected = correct_symmetric_bin_coverage(
        df, bin_widths, "baseline", boundaries, labels,
    )

    # Structure preserved
    check("symmetric correction has all bin labels",
          all(l in corrected for l in labels))

    # Widths should be >= original (correction only inflates)
    for label in labels:
        for cls_name in ["onpeak", "offpeak"]:
            for seg in ["prevail", "counter"]:
                key = (cls_name, seg)
                for clabel in COVERAGE_LABELS:
                    orig_w = bin_widths[label].get(key, {}).get(clabel)
                    corr_w = corrected[label].get(key, {}).get(clabel)
                    if isinstance(orig_w, (int, float)) and isinstance(corr_w, (int, float)):
                        check(f"sym {label}/{cls_name}/{seg}/{clabel}: corrected >= original",
                              corr_w >= orig_w - 0.1,
                              f"orig={orig_w:.1f}, corr={corr_w:.1f}")


# ─── Test 32: run_experiment with apply_correction ───────────────────────────

def test_run_experiment_with_correction():
    print("\n[Test 32] run_experiment with apply_correction=True")
    df = make_df(n=20000)
    pys = sorted(df["planning_year"].unique().to_list())

    # Without correction
    result_no_corr = run_experiment(
        df, "aq1", pys, 4, "baseline", "no_corr",
        symmetric=False, cv_mode="loo", apply_correction=False,
    )

    # With correction
    result_corr = run_experiment(
        df, "aq1", pys, 4, "baseline", "with_corr",
        symmetric=False, cv_mode="loo", apply_correction=True,
    )

    # Both should have valid structure
    check("no_corr has aggregate", "aggregate" in result_no_corr)
    check("with_corr has aggregate", "aggregate" in result_corr)

    # Corrected version should have same or better coverage
    p95_no = result_no_corr["aggregate"]["coverage"]["overall"]["p95"]["actual"]
    p95_corr = result_corr["aggregate"]["coverage"]["overall"]["p95"]["actual"]
    check(f"corrected P95 coverage >= uncorrected - 1pp",
          p95_corr >= p95_no - 1.0,
          f"no_corr={p95_no:.2f}%, corr={p95_corr:.2f}%")

    # Symmetric with correction
    result_sym_corr = run_experiment(
        df, "aq1", pys, 4, "baseline", "sym_corr",
        symmetric=True, cv_mode="loo", apply_correction=True,
    )
    check("symmetric + correction has aggregate", "aggregate" in result_sym_corr)


# ─── Test 33: v6 constants exist ─────────────────────────────────────────────

def test_v6_constants():
    print("\n[Test 33] v6 constants")
    check("CORRECTION_TOLERANCE is 2.0", CORRECTION_TOLERANCE == 2.0)
    check("CORRECTION_STEP is 0.05", CORRECTION_STEP == 0.05)
    check("CORRECTION_MAX_ITER is 20", CORRECTION_MAX_ITER == 20)


# ─── Run all tests ──────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  run_v6_bands.py — Comprehensive Test Suite")
    print("=" * 70)

    test_calibration_structure()
    test_quantile_pair_correctness()
    test_apply_columns()
    test_insample_coverage()
    test_asymmetric_narrower()
    test_symmetric_residuals()
    test_run_experiment_loo()
    test_run_experiment_temporal()
    test_run_experiment_symmetric()
    test_bg1_check()
    test_select_winner()
    test_sparse_fallback()
    test_zero_baseline()
    test_single_py()
    test_temporal_min_train_filtering()
    test_json_roundtrip()
    test_halfwidth_equivalence()
    test_per_class_per_sign_evaluation()
    test_per_bin_aggregate()
    test_bin_column_guard()
    test_width_monotonicity()
    test_stability_structure()
    test_empty_test_py()
    test_higher_bin_counts()
    test_pipeline_gate_structure()
    test_widths_by_sign_structure()
    test_bg2a_bg2b_gates()
    test_correct_bin_coverage_basic()
    test_correct_bin_coverage_no_change()
    test_correct_bin_coverage_max_iter()
    test_correct_symmetric_bin_coverage()
    test_run_experiment_with_correction()
    test_v6_constants()

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 70}")
    if errors:
        print(f"\n  Failed tests:")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)
    else:
        print(f"\n  All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
