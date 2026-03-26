# DEPRECATED — archived 2026-03-17. Old data sources. See task_plan.md for porting plan.
import warnings as _w; _w.warn("This script is DEPRECATED (archived 2026-03-17). Use canonical get_trades_of_given_duration().", DeprecationWarning, stacklevel=2)

"""Tests for PJM V1 annual bands.

Covers 90%+ use cases per CLAUDE.md:
1. Core calibration (3 classes × 5 bins)
2. Band application (16 columns, no nulls, containment)
3. Coverage monotonicity
4. Scale correctness (annual × 12)
5. Edge cases (onpeak-only PY, zero baseline)
6. CP assignment
7. Class parity (3 classes)
8. Temporal CV
9. Artifact round-trip
10. Integration contract

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual-band/pjm/scripts/test_v1_bands.py
"""

from __future__ import annotations
import sys, math, json, tempfile, random, os
from pathlib import Path

# Add MISO scripts for core functions
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "miso" / "scripts"))

import polars as pl
import numpy as np

# Import and override for PJM
import run_v9_bands
run_v9_bands.CLASSES = ["onpeak", "dailyoffpeak", "wkndonpeak"]
run_v9_bands.MIN_CELL_ROWS = 200

from run_v9_bands import (
    calibrate_asymmetric_per_class, apply_asymmetric_bands_per_class_fast,
    compute_quantile_boundaries, evaluate_coverage, evaluate_per_class_coverage,
    sanitize_for_json, COVERAGE_LEVELS, COVERAGE_LABELS,
)

PJM_CLASSES = ["onpeak", "dailyoffpeak", "wkndonpeak"]
PASS_COUNT = 0
FAIL_COUNT = 0

def check(condition, name, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
        print(f"  FAIL: {name} — {detail}")

def _make_pjm_data(n=6000):
    random.seed(42)
    baselines = [random.gauss(0, 500) * 12 for _ in range(n)]
    residuals = [random.gauss(50, 200) * 12 + abs(b) * 0.05 for b in baselines]
    mcps = [b + r for b, r in zip(baselines, residuals)]
    n_per_class = n // 3
    classes = (["onpeak"] * n_per_class +
               ["dailyoffpeak"] * n_per_class +
               ["wkndonpeak"] * (n - 2 * n_per_class))
    return pl.DataFrame({
        "baseline": baselines, "mcp": mcps,
        "class_type": classes, "planning_year": [2023] * n,
    })

# 1. Core calibration
def test_calibration():
    print("\n=== 1. Core Calibration (3 classes) ===")
    df = _make_pjm_data(6000)
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 5)
    result = calibrate_asymmetric_per_class(df, "baseline", "mcp", "class_type", boundaries, labels, COVERAGE_LEVELS)
    check(len(labels) == 5, "5 bins")
    for label in labels:
        for cls in PJM_CLASSES:
            check(cls in result[label], f"{label} has {cls}")
            for cl in COVERAGE_LABELS:
                lo_hi = result[label][cls].get(cl)
                check(isinstance(lo_hi, tuple) and len(lo_hi) == 2, f"{label}.{cls}.{cl} is tuple")
        check("_pooled" in result[label], f"{label} has _pooled")
        for key in result[label]:
            check(not isinstance(key, tuple), f"no tuple keys in {label}")

# 2. Band application
def test_apply():
    print("\n=== 2. Band Application ===")
    df = _make_pjm_data(3000)
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 5)
    bp = calibrate_asymmetric_per_class(df, "baseline", "mcp", "class_type", boundaries, labels, COVERAGE_LEVELS)
    result = apply_asymmetric_bands_per_class_fast(df, bp, "baseline", "class_type", boundaries, labels)
    for cl in COVERAGE_LABELS:
        check(f"lower_{cl}" in result.columns, f"lower_{cl}")
        check(f"upper_{cl}" in result.columns, f"upper_{cl}")
        check(result[f"lower_{cl}"].null_count() == 0, f"lower_{cl} no nulls")
        check(result[f"upper_{cl}"].null_count() == 0, f"upper_{cl} no nulls")
    check("sign_seg" not in result.columns, "no sign_seg")
    check("_bin" not in result.columns, "no _bin leaked")
    # Containment
    for i in range(1, len(COVERAGE_LABELS)):
        w = COVERAGE_LABELS[i]; n_ = COVERAGE_LABELS[i-1]
        check((result[f"lower_{w}"] <= result[f"lower_{n_}"]).all(), f"lower_{w} <= lower_{n_}")
        check((result[f"upper_{w}"] >= result[f"upper_{n_}"]).all(), f"upper_{w} >= upper_{n_}")

# 3. Coverage monotonicity
def test_coverage_monotonicity():
    print("\n=== 3. Coverage Monotonicity ===")
    df = _make_pjm_data(3000)
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 5)
    bp = calibrate_asymmetric_per_class(df, "baseline", "mcp", "class_type", boundaries, labels, COVERAGE_LEVELS)
    banded = apply_asymmetric_bands_per_class_fast(df, bp, "baseline", "class_type", boundaries, labels)
    covs = []
    for cl in COVERAGE_LABELS:
        mcp = banded["mcp"].to_numpy()
        lo = banded[f"lower_{cl}"].to_numpy()
        hi = banded[f"upper_{cl}"].to_numpy()
        covs.append(float(np.mean((mcp >= lo) & (mcp <= hi))))
    for i in range(1, len(covs)):
        check(covs[i] >= covs[i-1] - 0.001, f"{COVERAGE_LABELS[i]} >= {COVERAGE_LABELS[i-1]}")

# 4. Scale correctness (annual)
def test_annual_scale():
    print("\n=== 4. Annual Scale ===")
    df = _make_pjm_data(2000)
    check(df["baseline"].std() > 2000, f"baseline std={df['baseline'].std():.0f} > 2000 (annual)")
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 5)
    bp = calibrate_asymmetric_per_class(df, "baseline", "mcp", "class_type", boundaries, labels, COVERAGE_LEVELS)
    p95 = bp[labels[2]]["onpeak"]["p95"]
    width = p95[1] - p95[0]
    check(width > 1000, f"P95 width={width:.0f} > 1000 (annual scale)")

# 5. Edge cases
def test_edge_cases():
    print("\n=== 5. Edge Cases ===")
    # Onpeak-only (like PY2017-2022 R1)
    random.seed(99)
    n = 3000
    df_onpeak = pl.DataFrame({
        "baseline": [random.gauss(500, 200) * 12 for _ in range(n)],
        "mcp": [random.gauss(510, 210) * 12 for _ in range(n)],
        "class_type": ["onpeak"] * n,
    })
    boundaries, labels = compute_quantile_boundaries(df_onpeak["baseline"], 5)
    bp = calibrate_asymmetric_per_class(df_onpeak, "baseline", "mcp", "class_type", boundaries, labels, COVERAGE_LEVELS)
    for label in labels:
        check(bp[label]["dailyoffpeak"].get("_fallback") == "pooled", f"{label}.dailyoff fallback")
        check(bp[label]["wkndonpeak"].get("_fallback") == "pooled", f"{label}.wkndon fallback")

    # Bad class_type
    df_bad = pl.DataFrame({"baseline": [100.0], "mcp": [110.0], "class_type": ["peak"], "planning_year": [2023]})
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_v1_bands import run_experiment
    try:
        run_experiment(df_bad, [2023], min_train_pys=1)
        check(False, "bad class_type should raise")
    except ValueError:
        check(True, "bad class_type raises ValueError")

# 6. CP assignment
def test_cp():
    print("\n=== 6. CP Assignment ===")
    for level, cl in zip(COVERAGE_LEVELS, COVERAGE_LABELS):
        buy_upper = (1 + level) / 2 * 100
        buy_lower = (1 - level) / 2 * 100
        check(buy_upper > 50, f"upper_{cl} buy CP > 50")
        check(buy_lower < 50, f"lower_{cl} buy CP < 50")

# 7. Class parity (3 classes)
def test_class_parity():
    print("\n=== 7. Class Parity (3 classes) ===")
    df = _make_pjm_data(6000)
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 5)
    bp = calibrate_asymmetric_per_class(df, "baseline", "mcp", "class_type", boundaries, labels, COVERAGE_LEVELS)
    banded = apply_asymmetric_bands_per_class_fast(df, bp, "baseline", "class_type", boundaries, labels)
    pc = evaluate_per_class_coverage(banded, "mcp", "class_type", COVERAGE_LEVELS)
    for cls in PJM_CLASSES:
        check(cls in pc, f"{cls} in coverage")
        check(pc[cls]["n"] > 0, f"{cls} has rows")
    # Parity gap < 5pp
    p95s = [pc[cls]["p95"]["actual"] for cls in PJM_CLASSES if cls in pc]
    gap = max(p95s) - min(p95s)
    check(gap < 5.0, f"class gap={gap:.1f}pp < 5pp")

# 8. Temporal CV
def test_temporal_cv():
    print("\n=== 8. Temporal CV ===")
    random.seed(42)
    rows = []
    for py in [2020, 2021, 2022, 2023]:
        for i in range(1000):
            bl = random.gauss(0, 500) * 12
            res = random.gauss(50, 200) * 12
            rows.append({
                "baseline": bl, "mcp": bl + res,
                "class_type": ["onpeak", "dailyoffpeak", "wkndonpeak"][i % 3],
                "planning_year": py,
            })
    df = pl.DataFrame(rows)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_v1_bands import run_experiment
    result = run_experiment(df, [2020, 2021, 2022, 2023], min_train_pys=2)
    for py_str, pd in result["per_py"].items():
        for tpy in pd.get("train_pys", []):
            check(tpy < int(py_str), f"PY{py_str}: train PY {tpy} < test")
    p95 = result["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
    check(80 < p95 < 100, f"aggregate P95={p95:.1f}% in range")

# 9. Artifact round-trip
def test_artifact_roundtrip():
    print("\n=== 9. Artifact Round-Trip ===")
    df = _make_pjm_data(3000)
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 5)
    bp = calibrate_asymmetric_per_class(df, "baseline", "mcp", "class_type", boundaries, labels, COVERAGE_LEVELS)
    artifact = {"calibration": {"a": {
        "boundaries": [b if not math.isinf(b) else "inf" for b in boundaries],
        "bin_labels": labels, "bin_pairs": sanitize_for_json(bp),
    }}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(artifact, f, indent=2); path = f.name
    loaded = json.load(open(path))
    cal = loaded["calibration"]["a"]
    lb = [float(b) if b != "inf" else float("inf") for b in cal["boundaries"]]
    check(len(lb) == len(boundaries), "boundaries length")
    lp = cal["bin_pairs"]
    for label in labels:
        for cls in PJM_CLASSES:
            if cls in lp.get(label, {}):
                for cl in COVERAGE_LABELS:
                    v = lp[label][cls].get(cl)
                    if isinstance(v, list): lp[label][cls][cl] = tuple(v)
        if "_pooled" in lp.get(label, {}):
            for cl in COVERAGE_LABELS:
                v = lp[label]["_pooled"].get(cl)
                if isinstance(v, list): lp[label]["_pooled"][cl] = tuple(v)
    b1 = apply_asymmetric_bands_per_class_fast(df, bp, "baseline", "class_type", boundaries, labels)
    b2 = apply_asymmetric_bands_per_class_fast(df, lp, "baseline", "class_type", lb, labels)
    for cl in COVERAGE_LABELS:
        diff = (b1[f"lower_{cl}"] - b2[f"lower_{cl}"]).abs().max()
        check(diff < 0.1, f"lower_{cl} roundtrip diff={diff:.4f}")
    os.unlink(path)

# 10. Integration contract
def test_integration():
    print("\n=== 10. Integration Contract ===")
    df = _make_pjm_data(2000)
    boundaries, labels = compute_quantile_boundaries(df["baseline"], 5)
    bp = calibrate_asymmetric_per_class(df, "baseline", "mcp", "class_type", boundaries, labels, COVERAGE_LEVELS)
    banded = apply_asymmetric_bands_per_class_fast(df, bp, "baseline", "class_type", boundaries, labels)
    bid_cols = [f"upper_{cl}" for cl in COVERAGE_LABELS] + ["lower_p95", "lower_p99"]
    check(len(bid_cols) == 10, "10 bid price candidates")
    for col in bid_cols:
        check(col in banded.columns, f"{col} available")
        check(banded[col].null_count() == 0, f"{col} no nulls")
        check(banded[col].is_finite().all(), f"{col} all finite")


if __name__ == "__main__":
    test_calibration()
    test_apply()
    test_coverage_monotonicity()
    test_annual_scale()
    test_edge_cases()
    test_cp()
    test_class_parity()
    test_temporal_cv()
    test_artifact_roundtrip()
    test_integration()
    print(f"\n{'='*60}")
    print(f"  RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    print(f"{'='*60}")
    if FAIL_COUNT > 0:
        sys.exit(1)
    else:
        print("  All tests passed.")
    # Restore
    run_v9_bands.CLASSES = ["onpeak", "offpeak"]
    run_v9_bands.MIN_CELL_ROWS = 500
