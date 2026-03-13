"""Tests for ml/nb_detection.py — NB flags."""
import polars as pl
import pytest


def _load_test_data():
    """Helper: load all inputs needed for NB tests."""
    from ml.nb_detection import compute_nb_flags
    from ml.history_features import compute_history_features
    from ml.ground_truth import build_ground_truth
    from ml.data_loader import load_collapsed

    py, aq = "2024-06", "aq1"
    collapsed = load_collapsed(py, aq)
    branches = collapsed["branch_name"].to_list()
    hist_df, monthly_binding = compute_history_features(py, aq, branches)
    gt_df, _ = build_ground_truth(py, aq)

    nb_df = compute_nb_flags(
        universe_branches=branches,
        planning_year=py,
        aq_quarter=aq,
        gt_df=gt_df,
        monthly_binding_table=monthly_binding,
    )
    return nb_df, hist_df, gt_df, branches


def test_nb_flags_columns():
    """NB detection returns expected columns."""
    nb_df, _, _, _ = _load_test_data()
    assert "is_nb_6" in nb_df.columns
    assert "is_nb_12" in nb_df.columns
    assert "is_nb_24" in nb_df.columns
    assert "nb_onpeak_12" in nb_df.columns
    assert "nb_offpeak_12" in nb_df.columns
    assert "branch_name" in nb_df.columns


def test_nb_requires_target_binding():
    """NB requires branch to actually bind in target quarter."""
    nb_df, _, gt_df, _ = _load_test_data()

    nb_with_gt = nb_df.join(gt_df.select(["branch_name", "realized_shadow_price"]),
                            on="branch_name", how="left")
    nb12_branches = nb_with_gt.filter(pl.col("is_nb_12"))
    if len(nb12_branches) > 0:
        assert (nb12_branches["realized_shadow_price"] > 0).all(), \
            "NB12 branches must have positive target binding"


def test_nb_combined_ctype_check():
    """Test spec E6: NB checks BOTH ctypes for lookback."""
    nb_df, hist_df, _, _ = _load_test_data()

    nb_with_hist = nb_df.join(hist_df.select(["branch_name", "bfo_12", "bf_12"]),
                               on="branch_name", how="left")

    # A branch with bfo_12 > 0 should NOT be NB12 (had offpeak binding)
    offpeak_binders = nb_with_hist.filter(pl.col("bfo_12") > 0)
    if len(offpeak_binders) > 0:
        assert not offpeak_binders["is_nb_12"].any(), \
            "Branch with offpeak binding in last 12mo should NOT be NB12"


def test_nb_covers_universe():
    """NB flags should cover entire universe."""
    nb_df, _, _, branches = _load_test_data()
    assert len(nb_df) == len(branches), "NB flags should cover entire universe"
    assert nb_df["branch_name"].n_unique() == len(nb_df), "Duplicate branches in NB output"
