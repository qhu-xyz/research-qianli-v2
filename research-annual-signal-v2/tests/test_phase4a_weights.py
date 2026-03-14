"""Tests for Phase 4a sample weight computation."""
import numpy as np
import pytest


def test_tiered_weights_basic():
    """Tiered scheme: bottom 1/3 = 1.0, middle 1/3 = 3.0, top 1/3 = 10.0."""
    from scripts.run_phase4a_experiment import compute_sample_weights

    # 3 negatives + 6 positives with known SP ordering
    sp = np.array([0.0, 0.0, 0.0, 10.0, 20.0, 100.0, 200.0, 500.0, 1000.0])
    w = compute_sample_weights(sp, "tiered")

    # Negatives: weight 1.0
    assert w[0] == 1.0
    assert w[1] == 1.0
    assert w[2] == 1.0

    # 6 positives: bottom 2 = 1.0, middle 2 = 3.0, top 2 = 10.0
    pos_weights = w[3:]
    assert pos_weights[0] == 1.0   # SP=10, rank 0
    assert pos_weights[1] == 1.0   # SP=20, rank 1
    assert pos_weights[2] == 3.0   # SP=100, rank 2
    assert pos_weights[3] == 3.0   # SP=200, rank 3
    assert pos_weights[4] == 10.0  # SP=500, rank 4
    assert pos_weights[5] == 10.0  # SP=1000, rank 5


def test_tiered_weights_all_negative():
    """No positives -> all weights = 1.0."""
    from scripts.run_phase4a_experiment import compute_sample_weights

    sp = np.array([0.0, 0.0, 0.0])
    w = compute_sample_weights(sp, "tiered")
    np.testing.assert_array_equal(w, [1.0, 1.0, 1.0])


def test_continuous_weights_basic():
    """Continuous scheme: negatives=1.0, positives=1.0+min(log1p(SP), 12.0)."""
    from scripts.run_phase4a_experiment import compute_sample_weights

    sp = np.array([0.0, 100.0, 1e6])
    w = compute_sample_weights(sp, "continuous")

    assert w[0] == 1.0  # negative
    assert abs(w[1] - (1.0 + np.log1p(100.0))) < 1e-10  # log1p(100) ≈ 4.62
    assert abs(w[2] - 13.0) < 1e-10  # capped: 1.0 + 12.0 = 13.0


def test_continuous_weights_cap():
    """Cap kicks in at SP ≈ $162k (log1p(162754.79) ≈ 12.0)."""
    from scripts.run_phase4a_experiment import compute_sample_weights

    sp = np.array([200000.0, 500000.0])
    w = compute_sample_weights(sp, "continuous")
    # Both above cap -> weight = 13.0
    assert w[0] == 13.0
    assert w[1] == 13.0


def test_tiered_weights_single_positive():
    """Single positive -> always top tier (weight=10.0)."""
    from scripts.run_phase4a_experiment import compute_sample_weights

    sp = np.array([0.0, 0.0, 42.0])
    w = compute_sample_weights(sp, "tiered")
    assert w[0] == 1.0
    assert w[1] == 1.0
    assert w[2] == 10.0  # single positive: ranks < 0 always False -> top tier
