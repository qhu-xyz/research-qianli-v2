"""Tests for data_loader — the critical ground-truth-swap module."""
import polars as pl

from ml.data_loader import load_v62b_month, load_train_val_test

# Engineered features that must NOT be present (they are useless per stage4 findings)
_BANNED_ENGINEERED = [
    "flow_utilization", "mix_utilization", "branch_utilization", "prob_exceed_max",
    "ori_mix_ratio", "branch_ori_ratio", "branch_mix_ratio",
    "ori_mix_diff", "branch_ori_diff", "branch_mix_diff",
    "rank_gap_mix_ori", "rank_gap_da_mix", "rank_gap_da_ori",
    "prob_spread_110_100", "prob_spread_100_90", "prob_spread_90_80", "prob_exceed_mean",
    "prob110_x_flow", "prob110_x_branch", "probmax_x_util", "probmean_x_util",
    "da_x_dmix", "da_x_dori", "da_x_prob110", "da_x_util",
    "da_rank_sq", "dmix_rank_sq", "flow_util_sq", "prob110_sq", "ori_mean_sq",
    "log_constraint_limit", "log_ori_mean", "log_branch_max",
    "da_prob110_util", "dmix_prob110_branch", "prob110_x_ori_mix_ratio", "da_x_branch_util",
]


def test_load_v62b_month_has_realized_sp():
    """realized_sp column must be present after ground-truth join."""
    df = load_v62b_month("2022-06")
    assert "realized_sp" in df.columns, "realized_sp missing — ground truth join failed"
    assert df.schema["realized_sp"] == pl.Float64


def test_load_v62b_month_has_spice6_features():
    """Spice6 density features must be present."""
    df = load_v62b_month("2022-06")
    for col in ["prob_exceed_110", "prob_exceed_100", "prob_exceed_90",
                "prob_exceed_85", "prob_exceed_80", "constraint_limit"]:
        assert col in df.columns, f"Missing spice6 feature: {col}"


def test_load_v62b_month_no_engineered_features():
    """Banned engineered features must be absent."""
    df = load_v62b_month("2022-06")
    present = [c for c in _BANNED_ENGINEERED if c in df.columns]
    assert len(present) == 0, f"Banned engineered features found: {present}"


def test_train_val_test_splits_have_realized_sp():
    """Train and test splits both have realized_sp with val_months=0."""
    train_df, val_df, test_df = load_train_val_test(
        "2022-06", train_months=2, val_months=0,
    )
    assert "realized_sp" in train_df.columns, "realized_sp missing from train"
    assert "realized_sp" in test_df.columns, "realized_sp missing from test"
    assert val_df is None, "val_df should be None when val_months=0"
    assert "query_month" in train_df.columns, "query_month missing from train"
