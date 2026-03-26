"""Multi-cell smoke test for PJM 7.0b publisher.

Tests the publish_signal pipeline across multiple (PY, round, class_type) cells.
For each cell:
  1. Build constraints + SF via publish_signal()
  2. Validate schema, nulls, tier counts, index uniqueness
  3. Compare overlap with V4.6 for constraint/equipment/convention/shadow_sign parity
  4. Record constraint row count, SF shape, tier counts
  5. Test save/load round-trip via dev path

Run:
  cd /home/xyz/workspace/pmodel && source .venv/bin/activate
  python /home/xyz/workspace/research-qianli-v2/research-annual-signal-pjm/scripts/smoke_test_multi_cell.py

Memory: each cell uses ~4-8 GB for SF loading. Runs sequentially.
"""
from __future__ import annotations

import gc
import json
import resource
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.markets.pjm.signal_publisher import (
    CONSTRAINT_COLUMNS,
    CONSTRAINT_INDEX_COLUMN,
    REQUIRED_NON_NULL_COLUMNS,
    SF_INDEX_COLUMN,
    V46_BASE,
    publish_signal,
    save_signal,
)

import polars as pl


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ── Smoke test cells ─────────────────────────────────────────────────
# Minimum set from the task spec:
#   2024-06/R1/onpeak, 2024-06/R2/dailyoffpeak, 2024-06/R3/wkndonpeak,
#   2021-06/R2/onpeak (older year), 2025-06/R4/onpeak (holdout year)
SMOKE_CELLS = [
    {"planning_year": "2024-06", "market_round": 1, "class_type": "onpeak"},
    {"planning_year": "2024-06", "market_round": 2, "class_type": "dailyoffpeak"},
    {"planning_year": "2024-06", "market_round": 3, "class_type": "wkndonpeak"},
    {"planning_year": "2021-06", "market_round": 2, "class_type": "onpeak"},
    {"planning_year": "2025-06", "market_round": 4, "class_type": "onpeak"},
]


def load_v46_for_overlap(py: str, rnd: int, ctype: str) -> pd.DataFrame | None:
    """Load V4.6 constraints for overlap comparison."""
    path = f"{V46_BASE}/TEST.Signal.PJM.SPICE_ANNUAL_V4.6.R{rnd}/{py}/a/{ctype}/"
    if not Path(path).is_dir():
        return None
    df = pl.read_parquet(path)
    if CONSTRAINT_INDEX_COLUMN not in df.columns:
        return None
    return df.select([
        pl.col(CONSTRAINT_INDEX_COLUMN).str.split("|").list.first().alias("constraint_id"),
        pl.col("constraint").alias("v46_constraint"),
        pl.col("equipment").alias("v46_equipment"),
        pl.col("convention").cast(pl.Int64).alias("v46_convention"),
        pl.col("shadow_sign").alias("v46_shadow_sign"),
    ]).to_pandas()


def validate_constraints(constraints_df: pd.DataFrame) -> dict:
    """Validate constraints parquet against the publication contract."""
    issues: list[str] = []

    # 1. Check all contracted columns present
    for col in CONSTRAINT_COLUMNS:
        if col not in constraints_df.columns:
            issues.append(f"MISSING_COLUMN: {col}")

    # 2. Check index uniqueness
    if constraints_df.index.duplicated().any():
        n_dup = int(constraints_df.index.duplicated().sum())
        issues.append(f"DUPLICATE_INDEX: {n_dup} duplicated index keys")

    # 3. Check no nulls in required columns
    for col in REQUIRED_NON_NULL_COLUMNS:
        if col in constraints_df.columns and constraints_df[col].isna().any():
            n_null = int(constraints_df[col].isna().sum())
            issues.append(f"NULL_IN_{col}: {n_null}")

    # 4. Check rank direction (low = best)
    if "rank" in constraints_df.columns and "tier" in constraints_df.columns:
        tier0 = constraints_df[constraints_df["tier"] == 0]
        other = constraints_df[constraints_df["tier"] > 0]
        if len(tier0) > 0 and len(other) > 0:
            if tier0["rank"].mean() > other["rank"].mean():
                issues.append("RANK_DIRECTION: tier 0 has higher mean rank than other tiers (should be lower)")

    # 5. Check tier assignment: tiers should be contiguous starting from 0
    if "tier" in constraints_df.columns:
        tiers_present = sorted(constraints_df["tier"].unique())
        expected = list(range(len(tiers_present)))
        if tiers_present != expected:
            issues.append(f"TIER_GAPS: tiers present {tiers_present}, expected {expected}")

    # 6. Check index format: "{constraint_id}|{shadow_sign}|spice"
    for idx_val in constraints_df.index[:5]:
        parts = str(idx_val).split("|")
        if len(parts) != 3 or parts[2] != "spice":
            issues.append(f"INDEX_FORMAT: {idx_val} does not match '{{cid}}|{{sign}}|spice'")
            break

    # 7. Check constraint_limit populated
    if "constraint_limit" in constraints_df.columns:
        n_null = int(constraints_df["constraint_limit"].isna().sum())
        if n_null > 0:
            issues.append(f"CONSTRAINT_LIMIT_NULL: {n_null}")

    # 8. Row count check
    n_rows = len(constraints_df)
    if n_rows > 1000:
        issues.append(f"TOO_MANY_ROWS: {n_rows} > 1000")
    if n_rows < 100:
        issues.append(f"SUSPICIOUSLY_FEW_ROWS: {n_rows} < 100")

    return {
        "n_rows": n_rows,
        "n_columns": len(constraints_df.columns),
        "issues": issues,
        "tier_counts": {str(k): int(v) for k, v in constraints_df["tier"].value_counts().sort_index().items()}
        if "tier" in constraints_df.columns else {},
        "pass": len(issues) == 0,
    }


def validate_sf(sf_df: pd.DataFrame, constraints_df: pd.DataFrame) -> dict:
    """Validate SF parquet against constraints and contract."""
    issues: list[str] = []

    # 1. Index name
    if sf_df.index.name != SF_INDEX_COLUMN:
        issues.append(f"SF_INDEX_NAME: got '{sf_df.index.name}', expected '{SF_INDEX_COLUMN}'")

    # 2. Columns match constraints index
    constraint_keys = set(constraints_df.index)
    sf_keys = set(sf_df.columns)
    missing_in_sf = constraint_keys - sf_keys
    extra_in_sf = sf_keys - constraint_keys
    if missing_in_sf:
        issues.append(f"SF_MISSING_COLUMNS: {len(missing_in_sf)} constraint keys not in SF")
    if extra_in_sf:
        issues.append(f"SF_EXTRA_COLUMNS: {len(extra_in_sf)} SF columns not in constraints")

    # 3. No NaN
    nan_count = int(sf_df.isna().sum().sum())
    if nan_count > 0:
        issues.append(f"SF_NAN: {nan_count} NaN values in SF")

    # 4. dtype float64
    non_float = [c for c in sf_df.columns if sf_df[c].dtype != "float64"]
    if non_float:
        issues.append(f"SF_DTYPE: {len(non_float)} non-float64 columns")

    return {
        "shape": list(sf_df.shape),
        "index_name": sf_df.index.name,
        "issues": issues,
        "pass": len(issues) == 0,
    }


def compute_overlap_parity(
    constraints_df: pd.DataFrame, v46_df: pd.DataFrame | None
) -> dict:
    """Compare overlapping constraints with V4.6 for parity."""
    if v46_df is None:
        return {"v46_available": False, "overlap_count": 0}

    # Extract constraint_id from our index
    our_cids = pd.Series(
        [str(idx).split("|")[0] for idx in constraints_df.index],
        index=constraints_df.index,
        name="constraint_id",
    )
    our_df = constraints_df.copy()
    our_df["constraint_id"] = our_cids.values

    merged = our_df.merge(v46_df, on="constraint_id", how="inner")
    n_overlap = len(merged)
    if n_overlap == 0:
        return {"v46_available": True, "overlap_count": 0}

    parity = {}
    # constraint parity
    if "constraint" in merged.columns and "v46_constraint" in merged.columns:
        match = (merged["constraint"] == merged["v46_constraint"]).mean()
        parity["constraint"] = float(match)

    # equipment parity
    if "equipment" in merged.columns and "v46_equipment" in merged.columns:
        match = (merged["equipment"] == merged["v46_equipment"]).mean()
        parity["equipment"] = float(match)

    # convention parity
    if "convention" in merged.columns and "v46_convention" in merged.columns:
        match = (merged["convention"].astype(int) == merged["v46_convention"].astype(int)).mean()
        parity["convention"] = float(match)

    # shadow_sign parity
    if "shadow_sign" in merged.columns and "v46_shadow_sign" in merged.columns:
        match = (merged["shadow_sign"].astype(int) == merged["v46_shadow_sign"].astype(int)).mean()
        parity["shadow_sign"] = float(match)

    return {
        "v46_available": True,
        "overlap_count": n_overlap,
        "parity": parity,
    }


def test_save_load_round_trip(
    py: str, rnd: int, ctype: str, constraints_df: pd.DataFrame, sf_df: pd.DataFrame
) -> dict:
    """Test save/load round-trip using dev path."""
    from pbase.data.dataset.signal.general import ConstraintsSignal, ShiftFactorSignal

    signal_name = f"TEST.Signal.PJM.SPICE_ANNUAL_V7.0B.R{rnd}"
    auction_month = pd.Timestamp(py)

    # Save to dev path
    cs = ConstraintsSignal(
        rto="pjm", signal_name=signal_name, period_type="a", class_type=ctype, is_dev=True
    )
    sf_loader = ShiftFactorSignal(
        rto="pjm", signal_name=signal_name, period_type="a", class_type=ctype, is_dev=True
    )

    try:
        cs_path = cs.save_data(constraints_df, auction_month=auction_month, dry_run=False)
        sf_path = sf_loader.save_data(sf_df, auction_month=auction_month, dry_run=False)
    except Exception as e:
        return {"save_ok": False, "error": str(e)}

    # Load back
    try:
        loaded_cs = cs.load_data(auction_month=auction_month)
        loaded_sf = sf_loader.load_data(auction_month=auction_month)
    except Exception as e:
        return {"save_ok": True, "load_ok": False, "error": str(e), "cs_path": cs_path, "sf_path": sf_path}

    # Verify shape
    cs_shape_match = loaded_cs.shape == constraints_df.shape
    sf_shape_match = loaded_sf.shape == sf_df.shape

    return {
        "save_ok": True,
        "load_ok": True,
        "cs_path": cs_path,
        "sf_path": sf_path,
        "cs_shape_match": cs_shape_match,
        "sf_shape_match": sf_shape_match,
        "loaded_cs_shape": list(loaded_cs.shape),
        "loaded_sf_shape": list(loaded_sf.shape),
        "pass": cs_shape_match and sf_shape_match,
    }


def run_single_cell(cell: dict) -> dict:
    """Run full smoke test for one publish cell."""
    py = cell["planning_year"]
    rnd = cell["market_round"]
    ctype = cell["class_type"]
    label = f"{py}/R{rnd}/{ctype}"

    print(f"\n{'='*80}")
    print(f"  SMOKE: {label}")
    print(f"{'='*80}")
    print(f"  mem_before: {mem_mb():.0f} MB")

    t0 = time.time()
    result: dict = {"cell": cell, "label": label}

    try:
        constraints_df, sf_df = publish_signal(
            planning_year=py,
            market_round=rnd,
            class_type=ctype,
        )
    except Exception as e:
        result["status"] = "PUBLISH_FAILED"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"  FAILED: {e}")
        return result

    elapsed = time.time() - t0
    result["publish_elapsed_s"] = round(elapsed, 1)
    print(f"  publish took {elapsed:.1f}s, mem={mem_mb():.0f} MB")

    # Validate constraints
    cs_val = validate_constraints(constraints_df)
    result["constraints"] = cs_val
    print(f"  constraints: {cs_val['n_rows']} rows, {cs_val['n_columns']} cols, pass={cs_val['pass']}")
    if cs_val["issues"]:
        for iss in cs_val["issues"]:
            print(f"    ISSUE: {iss}")
    print(f"  tier_counts: {cs_val.get('tier_counts', {})}")

    # Validate SF
    sf_val = validate_sf(sf_df, constraints_df)
    result["sf"] = sf_val
    print(f"  sf: {sf_val['shape']}, index={sf_val['index_name']}, pass={sf_val['pass']}")
    if sf_val["issues"]:
        for iss in sf_val["issues"]:
            print(f"    ISSUE: {iss}")

    # V4.6 overlap parity
    v46 = load_v46_for_overlap(py, rnd, ctype)
    overlap = compute_overlap_parity(constraints_df, v46)
    result["overlap"] = overlap
    if overlap.get("v46_available"):
        print(f"  v46 overlap: {overlap['overlap_count']} rows, parity={overlap.get('parity', {})}")
    else:
        print(f"  v46: not available for this cell")

    # Save/load round-trip
    try:
        rt = test_save_load_round_trip(py, rnd, ctype, constraints_df, sf_df)
        result["round_trip"] = rt
        print(f"  round_trip: save={rt['save_ok']}, load={rt.get('load_ok', 'N/A')}, pass={rt.get('pass', False)}")
        if not rt.get("pass", False) and "error" in rt:
            print(f"    RT ERROR: {rt['error']}")
    except Exception as e:
        result["round_trip"] = {"error": str(e)}
        print(f"  round_trip: EXCEPTION: {e}")

    # Overall status
    all_pass = cs_val["pass"] and sf_val["pass"]
    result["status"] = "PASS" if all_pass else "FAIL"
    print(f"  OVERALL: {result['status']}")

    # Free memory
    del constraints_df, sf_df
    gc.collect()
    print(f"  mem_after_gc: {mem_mb():.0f} MB")

    return result


def main():
    print("PJM 7.0b Multi-Cell Smoke Test")
    print(f"Cells to test: {len(SMOKE_CELLS)}")
    print(f"Starting mem: {mem_mb():.0f} MB")

    all_results: list[dict] = []
    for cell in SMOKE_CELLS:
        result = run_single_cell(cell)
        all_results.append(result)

    # Summary
    print(f"\n{'#'*80}")
    print("  SMOKE TEST SUMMARY")
    print(f"{'#'*80}")
    n_pass = sum(1 for r in all_results if r.get("status") == "PASS")
    n_fail = sum(1 for r in all_results if r.get("status") == "FAIL")
    n_error = sum(1 for r in all_results if r.get("status") == "PUBLISH_FAILED")

    for r in all_results:
        label = r["label"]
        status = r.get("status", "UNKNOWN")
        rows = r.get("constraints", {}).get("n_rows", "?")
        sf = r.get("sf", {}).get("shape", "?")
        overlap_n = r.get("overlap", {}).get("overlap_count", "?")
        elapsed = r.get("publish_elapsed_s", "?")
        rt_pass = r.get("round_trip", {}).get("pass", "?")
        print(f"  {label:40s} {status:15s} rows={str(rows):>5s} sf={str(sf):>20s} overlap={str(overlap_n):>4s} rt={str(rt_pass):>5s} {elapsed}s")

    print(f"\n  TOTAL: {n_pass} pass, {n_fail} fail, {n_error} error out of {len(all_results)} cells")

    # Save results
    out_dir = ROOT / "releases" / "pjm" / "annual" / "7.0b"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke_test_results.json"

    # Build serializable summary
    summary = {
        "n_cells": len(all_results),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_error": n_error,
        "cells": [],
    }
    for r in all_results:
        cell_summary = {
            "label": r["label"],
            "status": r.get("status", "UNKNOWN"),
            "publish_elapsed_s": r.get("publish_elapsed_s"),
            "constraint_rows": r.get("constraints", {}).get("n_rows"),
            "constraint_issues": r.get("constraints", {}).get("issues", []),
            "sf_shape": r.get("sf", {}).get("shape"),
            "sf_issues": r.get("sf", {}).get("issues", []),
            "tier_counts": r.get("constraints", {}).get("tier_counts", {}),
            "overlap_count": r.get("overlap", {}).get("overlap_count"),
            "overlap_parity": r.get("overlap", {}).get("parity"),
            "round_trip_pass": r.get("round_trip", {}).get("pass"),
        }
        if r.get("status") == "PUBLISH_FAILED":
            cell_summary["error"] = r.get("error")
        summary["cells"].append(cell_summary)

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Results written to {out_path}")

    # Decide readiness
    if n_pass == len(all_results):
        print("\n  VERDICT: ALL CELLS PASS — ready for publication review")
    elif n_error > 0:
        print(f"\n  VERDICT: {n_error} cells failed to publish — BLOCKER, see errors above")
    else:
        print(f"\n  VERDICT: {n_fail} cells have validation issues — review before publishing")


if __name__ == "__main__":
    main()
