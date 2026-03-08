"""Tests for LTR feature preparation."""
import numpy as np
import polars as pl
import pytest
from ml.config import LTRConfig
from ml.features import prepare_features, compute_query_groups

def test_prepare_features_shape():
    cfg = LTRConfig(features=["a", "b", "c"], monotone_constraints=[1, 0, -1])
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, None], "c": [5.0, 6.0]})
    X, mono = prepare_features(df, cfg)
    assert X.shape == (2, 3)
    assert mono == [1, 0, -1]
    assert X[1, 1] == 0.0  # null filled

def test_compute_query_groups():
    df = pl.DataFrame({
        "query_month": ["2021-01", "2021-01", "2021-02", "2021-02", "2021-02"],
    })
    groups = compute_query_groups(df)
    assert groups.tolist() == [2, 3]  # 2 in first group, 3 in second
