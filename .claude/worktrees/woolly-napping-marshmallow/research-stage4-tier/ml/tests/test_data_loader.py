"""Tests for LTR data loader."""
import os
import numpy as np
import polars as pl
import pytest
from ml.data_loader import load_v62b_month, load_train_val_test

def test_load_v62b_month_returns_dataframe():
    """Integration test: requires V6.2B data on disk."""
    if not os.path.exists("/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1/2021-07"):
        pytest.skip("V6.2B data not available")
    df = load_v62b_month("2021-07", period_type="f0", class_type="onpeak")
    assert len(df) > 400
    assert "constraint_id" in df.columns
    assert "shadow_price_da" in df.columns
    assert "rank" in df.columns
    # V6.2B columns should be present
    assert "mean_branch_max" in df.columns
    assert "da_rank_value" in df.columns

def test_load_v62b_month_no_nulls():
    """shadow_price_da should have no nulls."""
    if not os.path.exists("/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1/2021-07"):
        pytest.skip("V6.2B data not available")
    df = load_v62b_month("2021-07", period_type="f0", class_type="onpeak")
    assert df["shadow_price_da"].null_count() == 0
