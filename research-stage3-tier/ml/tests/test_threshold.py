"""Tests for F-beta threshold optimization."""

import numpy as np
import pytest

from ml.threshold import apply_threshold, find_optimal_threshold


class TestFindOptimalThreshold:
    """Tests for find_optimal_threshold."""

    def test_find_optimal_threshold_returns_valid_range(self):
        """Threshold should be between 0 and 1 for typical binary data."""
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=200)
        y_proba = rng.rand(200)
        threshold = find_optimal_threshold(y_true, y_proba, beta=0.7)
        assert 0.0 < threshold < 1.0

    def test_find_optimal_threshold_default_beta(self):
        """Default beta=0.7 should work without explicit argument."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
        threshold = find_optimal_threshold(y_true, y_proba)
        assert 0.0 < threshold < 1.0

    def test_find_optimal_threshold_no_positive_predictions_fallback(self):
        """When all probabilities are very low and no threshold yields
        non-zero predictions meaningfully, return 0.5."""
        # All true labels are 0, all probabilities are near 0
        y_true = np.array([0, 0, 0, 0, 0])
        y_proba = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        threshold = find_optimal_threshold(y_true, y_proba)
        assert threshold == 0.5


class TestApplyThreshold:
    """Tests for apply_threshold."""

    def test_apply_threshold_basic(self):
        """Values at or above threshold become 1, below become 0."""
        y_proba = np.array([0.1, 0.5, 0.7, 0.3, 0.9])
        result = apply_threshold(y_proba, 0.5)
        expected = np.array([0, 1, 1, 0, 1])
        np.testing.assert_array_equal(result, expected)

    def test_apply_threshold_returns_int_array(self):
        """Output dtype should be integer."""
        y_proba = np.array([0.2, 0.8])
        result = apply_threshold(y_proba, 0.5)
        assert result.dtype == np.int64 or result.dtype == np.int32

    def test_apply_threshold_all_above(self):
        """All predictions above threshold should all be 1."""
        y_proba = np.array([0.6, 0.7, 0.8, 0.9])
        result = apply_threshold(y_proba, 0.5)
        np.testing.assert_array_equal(result, np.ones(4, dtype=int))

    def test_apply_threshold_all_below(self):
        """All predictions below threshold should all be 0."""
        y_proba = np.array([0.1, 0.2, 0.3, 0.4])
        result = apply_threshold(y_proba, 0.5)
        np.testing.assert_array_equal(result, np.zeros(4, dtype=int))


class TestThresholdWithPerfectSeparation:
    """Test with perfectly separated classes."""

    def test_perfect_separation_threshold_near_half(self):
        """With perfect separation at 0.5, optimal threshold should be near 0.5."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        threshold = find_optimal_threshold(y_true, y_proba, beta=0.7)
        # Threshold should fall in the gap between negative and positive classes
        assert 0.3 < threshold < 0.7, (
            f"Expected threshold between 0.3 and 0.7 for perfect separation, got {threshold}"
        )

    def test_perfect_separation_achieves_perfect_fbeta(self):
        """Applying the optimal threshold on perfectly separated data
        should yield perfect predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        threshold = find_optimal_threshold(y_true, y_proba, beta=0.7)
        preds = apply_threshold(y_proba, threshold)
        np.testing.assert_array_equal(preds, y_true)
