"""Tests for ml_pred feature loader."""
import polars as pl

from ml.mlpred_loader import load_mlpred


def test_load_mlpred_returns_correct_columns():
    """Load real data for 2022-06, verify expected columns present, leaky columns absent."""
    df = load_mlpred("2022-06")
    assert isinstance(df, pl.DataFrame)
    assert len(df) > 0, "Expected non-empty DataFrame for 2022-06"

    # Expected columns
    expected = {
        "constraint_id",
        "flow_direction",
        "predicted_shadow_price",
        "binding_probability",
        "binding_probability_scaled",
    }
    for col in expected:
        assert col in df.columns, f"Missing expected column: {col}"

    # Leaky columns must be absent
    leaky = {
        "actual_shadow_price", "actual_binding", "error", "abs_error",
        "model_used", "predicted_binding", "predicted_binding_count",
        "threshold", "hist_da", "branch_name", "auction_month", "market_month",
    }
    for col in leaky:
        assert col not in df.columns, f"Leaky column present: {col}"

    # prob_exceed_* columns must be absent (already from spice6 density)
    prob_cols = [c for c in df.columns if c.startswith("prob_exceed_")]
    assert len(prob_cols) == 0, f"prob_exceed_* columns should be dropped: {prob_cols}"


def test_load_mlpred_missing_month_returns_empty():
    """Load 1999-01, verify empty df."""
    df = load_mlpred("1999-01")
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 0
