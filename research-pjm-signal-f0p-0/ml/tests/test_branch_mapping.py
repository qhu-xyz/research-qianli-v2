# ml/tests/test_branch_mapping.py
"""Tests for PJM branch mapping module.

NOTE: This test must NOT import from ml.realized_da (Task 4).
Task 3 must be independently completable.
"""
import polars as pl
import pytest
from ml.branch_mapping import load_constraint_info, build_branch_map, map_da_to_branches


def test_load_constraint_info_returns_dataframe():
    """constraint_info should load for a known month."""
    ci = load_constraint_info("2025-01", period_type="f0")
    assert isinstance(ci, pl.DataFrame)
    assert len(ci) > 0
    assert "constraint_id" in ci.columns
    assert "branch_name" in ci.columns


def test_build_branch_map_has_match_str():
    """Branch map should have match_str for DA joining."""
    ci = load_constraint_info("2025-01", period_type="f0")
    bmap = build_branch_map(ci)
    assert "match_str" in bmap.columns
    assert "branch_name" in bmap.columns
    # match_str should be uppercase
    for s in bmap["match_str"].head(5).to_list():
        assert s == s.upper(), f"match_str not uppercase: {s}"


def test_map_da_to_branches_captures_most_value():
    """Branch-level join should capture >90% of DA value.

    NOTE: This test uses PjmApTools directly (not ml.realized_da which is Task 4).
    """
    ci = load_constraint_info("2025-01", period_type="f0")
    bmap = build_branch_map(ci)

    try:
        import pandas as pd
        from pbase.analysis.tools.all_positions import PjmApTools

        st = pd.Timestamp("2025-01-01")
        et = st + pd.offsets.MonthBegin(1)
        aptools = PjmApTools()
        da_shadow = aptools.tools.get_da_shadow_by_peaktype(st=st, et_ex=et, peak_type="onpeak")
        if da_shadow is None or len(da_shadow) == 0:
            pytest.skip("No DA data for 2025-01")
        da_df = pl.from_pandas(da_shadow.reset_index()).select([
            pl.col("monitored_facility").cast(pl.String),
            pl.col("shadow_price").cast(pl.Float64),
        ])
    except Exception as e:
        pytest.skip(f"Cannot fetch DA data: {e}")

    result = map_da_to_branches(da_df, bmap)
    assert "branch_name" in result.columns
    assert "realized_sp" in result.columns

    total_value = da_df["shadow_price"].abs().sum()
    matched_value = result["realized_sp"].sum()
    coverage = matched_value / total_value if total_value > 0 else 0
    print(f"DA value coverage: {coverage:.1%}")
    assert coverage > 0.90, f"Branch mapping coverage too low: {coverage:.1%}"
