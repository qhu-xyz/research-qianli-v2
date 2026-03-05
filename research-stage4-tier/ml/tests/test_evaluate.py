"""Tests for LTR evaluation metrics."""
import numpy as np
import pytest
from ml.evaluate import (
    value_capture_at_k, recall_at_k, ndcg, spearman_corr,
    tier_ap, evaluate_ltr, aggregate_months,
)

def test_vc_at_k_perfect():
    actual = np.array([100.0, 50.0, 10.0, 5.0, 1.0])
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # perfect ranking
    assert value_capture_at_k(actual, scores, 2) == pytest.approx(150 / 166, abs=1e-4)

def test_vc_at_k_worst():
    actual = np.array([100.0, 50.0, 10.0, 5.0, 1.0])
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # reverse ranking
    assert value_capture_at_k(actual, scores, 2) == pytest.approx(6 / 166, abs=1e-4)

def test_recall_at_k_perfect():
    actual = np.array([100.0, 50.0, 10.0, 5.0, 1.0])
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert recall_at_k(actual, scores, 2) == 1.0

def test_recall_at_k_partial():
    actual = np.array([100.0, 50.0, 10.0, 5.0, 1.0])
    scores = np.array([1.0, 5.0, 3.0, 4.0, 2.0])  # top-2 by score: idx 1,3
    # true top-2: idx 0,1. overlap = {1}
    assert recall_at_k(actual, scores, 2) == 0.5

def test_ndcg_perfect():
    actual = np.array([10.0, 5.0, 1.0])
    scores = np.array([3.0, 2.0, 1.0])
    assert ndcg(actual, scores) == pytest.approx(1.0, abs=1e-4)

def test_spearman_perfect():
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman_corr(actual, scores) == pytest.approx(1.0, abs=1e-4)

def test_tier_ap():
    actual = np.array([100.0, 50.0, 10.0, 5.0, 1.0])
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    # top-20% = 1 constraint. Perfect score should give AP=1
    ap = tier_ap(actual, scores, top_frac=0.2)
    assert ap == pytest.approx(1.0, abs=1e-4)

def test_evaluate_ltr_returns_all_metrics():
    actual = np.array([100.0, 50.0, 10.0, 5.0, 1.0] * 20)
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0] * 20)
    metrics = evaluate_ltr(actual, scores)
    expected_keys = {"VC@10", "VC@20", "VC@25", "VC@50", "VC@100", "VC@200",
                     "Recall@10", "Recall@20", "Recall@50", "Recall@100",
                     "NDCG", "Spearman", "Tier0-AP", "Tier01-AP",
                     "n_samples"}
    assert expected_keys.issubset(set(metrics.keys()))

def test_aggregate_months():
    pm = {
        "2021-01": {"VC@100": 0.8, "NDCG": 0.9},
        "2021-02": {"VC@100": 0.9, "NDCG": 0.95},
    }
    agg = aggregate_months(pm)
    assert agg["mean"]["VC@100"] == pytest.approx(0.85, abs=1e-4)
    assert "bottom_2_mean" in agg
