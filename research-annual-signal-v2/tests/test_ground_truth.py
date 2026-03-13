"""Tests for ml/ground_truth.py — GT pipeline."""
import polars as pl
import pytest


def test_gt_combined_ctype(sample_py, sample_quarter):
    """Test spec E1: GT loads both onpeak and offpeak. Returns only positive-binding branches."""
    from ml.ground_truth import build_ground_truth
    gt_df, diag = build_ground_truth(sample_py, sample_quarter)
    assert "branch_name" in gt_df.columns
    assert "realized_shadow_price" in gt_df.columns
    assert "label_tier" in gt_df.columns
    # GT returns ONLY positive-binding branches
    assert (gt_df["realized_shadow_price"] > 0).all(), "GT should only contain positive-binding branches"
    assert len(gt_df) > 0


def test_gt_tiered_labels(sample_py, sample_quarter):
    """Test spec E5: tiered labels are 1/2/3 (GT only returns positive-binding branches)."""
    from ml.ground_truth import build_ground_truth
    gt_df, _ = build_ground_truth(sample_py, sample_quarter)
    labels = gt_df["label_tier"].unique().sort().to_list()
    # GT returns ONLY positive-binding branches — no label_tier=0 here.
    # Zero-fill (label_tier=0 for non-binding) happens in features.py.
    assert set(labels).issubset({1, 2, 3})
    assert 0 not in labels, "GT should not contain label_tier=0 (zero-fill is in features.py)"
    # All branches in GT have positive SP
    assert (gt_df["realized_shadow_price"] > 0).all()
    # Labels 1/2/3 should be approximately equal (tertiles of positive SP)
    for tier in [1, 2, 3]:
        n = gt_df.filter(pl.col("label_tier") == tier).height
        assert n > 0, f"No branches with tier {tier}"


def test_gt_per_ctype_split(sample_py, sample_quarter):
    """Design spec SS4.2: per-ctype split targets returned."""
    from ml.ground_truth import build_ground_truth
    gt_df, _ = build_ground_truth(sample_py, sample_quarter)
    assert "onpeak_sp" in gt_df.columns
    assert "offpeak_sp" in gt_df.columns


def test_gt_coverage_diagnostics(sample_py, sample_quarter):
    """Design spec SS4.3: raw coverage diagnostics returned.

    Note: total_da_cids includes zero-SP cids. annual_mapped_cids includes
    zero-SP cids that were mapped. These are RAW coverage counts, not
    positive-binding-only counts.
    """
    from ml.ground_truth import build_ground_truth
    _, diag = build_ground_truth(sample_py, sample_quarter)
    required_keys = [
        "total_da_cids", "annual_mapped_cids", "monthly_recovered_cids",
        "still_unmapped_cids", "total_da_sp", "annual_mapped_sp",
        "monthly_recovered_sp", "still_unmapped_sp",
    ]
    for key in required_keys:
        assert key in diag, f"Missing diagnostic: {key}"
    assert diag["total_da_sp"] > 0
    # Consistency: mapped + recovered + unmapped = total
    assert (
        diag["annual_mapped_cids"] + diag["monthly_recovered_cids"] + diag["still_unmapped_cids"]
    ) == diag["total_da_cids"], "Cid counts must partition total"


def test_gt_monthly_fallback_2025():
    """Test spec E3: monthly fallback recovers cids for 2025-06."""
    from ml.ground_truth import build_ground_truth
    _, diag = build_ground_truth("2025-06", "aq1")
    # For 2025-06: monthly fallback MUST recover some cids (annual bridge has known gaps)
    assert diag["monthly_recovered_cids"] > 0, (
        "2025-06 should have monthly fallback recoveries — annual bridge has known gaps"
    )
    assert diag["monthly_recovered_sp"] > 0


def test_gt_monthly_fallback_uses_market_month():
    """Monthly bridge uses individual market_month as auction_month, not PY."""
    from ml.ground_truth import build_ground_truth
    _, diag = build_ground_truth("2025-06", "aq2")
    # monthly_recovery_detail must exist and be keyed by market months
    assert "monthly_recovery_detail" in diag, "Missing monthly_recovery_detail in diagnostics"
    detail = diag["monthly_recovery_detail"]
    for month_key in detail:
        assert month_key != "2025-06", \
            f"Monthly fallback detail should use market months, not PY. Got: {month_key}"


def test_gt_branch_aggregation():
    """Test spec E4: multiple DA cids -> same branch -> SUM (not mean)."""
    from ml.ground_truth import build_ground_truth
    gt_df, _ = build_ground_truth("2024-06", "aq1")
    # All values should be positive (GT only returns positive-binding)
    assert (gt_df["realized_shadow_price"] > 0).all()
    # No duplicate branch_names
    assert gt_df["branch_name"].n_unique() == len(gt_df)


def test_gt_no_cid_fanout():
    """Each mapped cid should map to exactly 1 branch (ambiguous cids dropped)."""
    from ml.ground_truth import build_ground_truth
    from ml.realized_da import load_quarter_per_ctype
    from ml.config import get_market_months
    from ml.bridge import map_cids_to_branches

    # Build GT for a known PY
    gt_df, diag = build_ground_truth("2024-06", "aq1")

    # Verify the annual mapping itself has no fan-out
    market_months = get_market_months("2024-06", "aq1")
    onpeak_da, offpeak_da = load_quarter_per_ctype(market_months)
    combined = pl.concat([
        onpeak_da.rename({"realized_sp": "total_sp"}),
        offpeak_da.rename({"realized_sp": "total_sp"}),
    ]).group_by("constraint_id").agg(pl.col("total_sp").sum())

    mapped, _ = map_cids_to_branches(
        cid_df=combined,
        auction_type="annual",
        auction_month="2024-06",
        period_type="aq1",
    )
    # After map_cids_to_branches, each cid should appear at most once
    cid_counts = mapped.group_by("constraint_id").len()
    assert cid_counts["len"].max() == 1, "Fan-out detected: a cid maps to >1 branch after ambiguity drop"
