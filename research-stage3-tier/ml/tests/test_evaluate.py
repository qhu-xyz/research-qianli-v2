"""Tests for tier evaluation harness."""

import numpy as np
import pytest

from ml.config import TierConfig
from ml.evaluate import aggregate_months, evaluate_tier_pipeline
from ml.features import compute_tier_labels


class TestEvaluateTierPipelineBasic:
    """Basic smoke test: all keys present and types correct."""

    def test_evaluate_tier_pipeline_basic(self):
        """Create random data, verify all keys exist."""
        rng = np.random.RandomState(42)
        n = 2000
        cfg = TierConfig()
        actual_sp = rng.choice([5000.0, 1500.0, 200.0, 10.0, -5.0], size=n)
        actual_tier = compute_tier_labels(actual_sp, cfg)

        pred_tier = rng.randint(0, 5, size=n)
        tier_proba = rng.dirichlet([1, 1, 1, 1, 1], size=n)
        tier_ev_score = rng.rand(n) * 4000

        result = evaluate_tier_pipeline(actual_sp, actual_tier, pred_tier, tier_proba, tier_ev_score)

        # Group A keys
        assert "Tier-VC@100" in result
        assert "Tier-VC@500" in result
        assert "Tier-NDCG" in result
        assert "QWK" in result

        # Group B keys
        assert "Macro-F1" in result
        assert "Tier-Accuracy" in result
        assert "Adjacent-Accuracy" in result
        assert "Tier-Recall@0" in result
        assert "Tier-Recall@1" in result

        # Monitoring keys
        assert "n_samples" in result

        # Value types
        assert result["n_samples"] == n


class TestTierValueCapture:
    """Tests for tier-based value capture metrics."""

    def test_value_capture_perfect(self):
        """When tier_ev_score perfectly ranks by actual, VC should be high."""
        rng = np.random.RandomState(123)
        cfg = TierConfig()
        n = 2000
        actual_sp = np.zeros(n)
        # Top 200 have high actual values
        actual_sp[:200] = rng.uniform(50, 5000, size=200)
        actual_tier = compute_tier_labels(actual_sp, cfg)

        # tier_ev_score perfectly correlated with actual
        tier_ev_score = actual_sp.copy() + rng.uniform(0, 0.01, size=n)

        tier_proba = rng.dirichlet([1, 1, 1, 1, 1], size=n)
        pred_tier = rng.randint(0, 5, size=n)

        result = evaluate_tier_pipeline(actual_sp, actual_tier, pred_tier, tier_proba, tier_ev_score)

        assert result["Tier-VC@100"] > 0.4, (
            f"Expected VC@100 > 0.4 with perfect ranking, got {result['Tier-VC@100']}"
        )


class TestQWK:
    """Tests for Quadratic Weighted Kappa."""

    def test_qwk_perfect(self):
        """Perfect predictions give QWK close to 1.0."""
        rng = np.random.RandomState(789)
        cfg = TierConfig()
        n = 500
        actual_sp = rng.choice([5000.0, 1500.0, 200.0, 10.0, -5.0], size=n)
        actual_tier = compute_tier_labels(actual_sp, cfg)

        # Perfect tier prediction
        pred_tier = actual_tier.copy()

        tier_proba = rng.dirichlet([1, 1, 1, 1, 1], size=n)
        tier_ev_score = rng.rand(n) * 4000

        result = evaluate_tier_pipeline(actual_sp, actual_tier, pred_tier, tier_proba, tier_ev_score)

        assert result["QWK"] > 0.99, (
            f"Expected QWK ~1.0 for perfect prediction, got {result['QWK']}"
        )


class TestAggregateMonths:
    """Tests for aggregate_months."""

    def test_aggregate_months(self):
        """Verify mean, std, min, max, bottom_2_mean for 3 months."""
        per_month = {
            "2025-01": {"Tier-VC@100": 0.6, "QWK": 0.7},
            "2025-02": {"Tier-VC@100": 0.8, "QWK": 0.9},
            "2025-03": {"Tier-VC@100": 0.7, "QWK": 0.8},
        }

        agg = aggregate_months(per_month)

        # Check structure
        assert "mean" in agg
        assert "std" in agg
        assert "min" in agg
        assert "max" in agg
        assert "bottom_2_mean" in agg

        # Mean
        assert agg["mean"]["Tier-VC@100"] == pytest.approx(0.7, abs=1e-6)
        assert agg["mean"]["QWK"] == pytest.approx(0.8, abs=1e-6)

        # Min/Max
        assert agg["min"]["Tier-VC@100"] == pytest.approx(0.6, abs=1e-6)
        assert agg["max"]["Tier-VC@100"] == pytest.approx(0.8, abs=1e-6)

        # bottom_2_mean for Tier-VC@100 (higher is better) -> 2 lowest: 0.6, 0.7
        assert agg["bottom_2_mean"]["Tier-VC@100"] == pytest.approx(0.65, abs=1e-6)
