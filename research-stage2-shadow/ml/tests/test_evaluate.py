"""Tests for EV-based evaluation harness."""

import numpy as np
import pytest

from ml.evaluate import aggregate_months, evaluate_pipeline


class TestEvaluatePipelineBasic:
    """Basic smoke test: all keys present and types correct."""

    def test_evaluate_pipeline_basic(self):
        """Create random data with ~10% binding, verify all keys exist."""
        rng = np.random.RandomState(42)
        n = 2000
        actual = np.zeros(n)
        binding_idx = rng.choice(n, size=int(n * 0.1), replace=False)
        actual[binding_idx] = rng.uniform(1, 100, size=len(binding_idx))

        pred_proba = rng.rand(n)
        pred_shadow = rng.uniform(0, 50, size=n)
        ev_scores = rng.rand(n)

        result = evaluate_pipeline(actual, pred_proba, pred_shadow, ev_scores)

        # Group A keys
        assert "EV-VC@100" in result
        assert "EV-VC@500" in result
        assert "EV-NDCG" in result
        assert "Spearman" in result

        # Group B keys
        assert "C-RMSE" in result
        assert "C-MAE" in result
        assert "EV-VC@1000" in result
        assert "R-REC@500" in result

        # Monitoring keys
        assert "binding_rate" in result
        assert "n_samples" in result
        assert "n_binding" in result

        # Value types
        assert result["n_samples"] == n
        assert result["n_binding"] == len(binding_idx)
        assert 0.0 < result["binding_rate"] < 1.0


class TestValueCapture:
    """Tests for EV-based value capture metrics."""

    def test_value_capture_perfect(self):
        """When ev_scores perfectly rank by actual, VC should be high."""
        rng = np.random.RandomState(123)
        n = 2000
        actual = np.zeros(n)
        # Top 200 have high actual values
        actual[:200] = rng.uniform(50, 100, size=200)
        # ev_scores perfectly correlated with actual
        ev_scores = actual.copy() + rng.uniform(0, 0.01, size=n)

        pred_proba = rng.rand(n)
        pred_shadow = rng.uniform(0, 50, size=n)

        result = evaluate_pipeline(actual, pred_proba, pred_shadow, ev_scores)

        # Perfect ranking: top-100 by EV should capture a large fraction
        # of value since all value is in top 200
        assert result["EV-VC@100"] > 0.4, (
            f"Expected VC@100 > 0.4 with perfect ranking, got {result['EV-VC@100']}"
        )
        assert result["EV-VC@500"] > 0.95, (
            f"Expected VC@500 > 0.95 with perfect ranking, got {result['EV-VC@500']}"
        )

    def test_value_capture_random(self):
        """Random ranking should give moderate VC."""
        rng = np.random.RandomState(456)
        n = 2000
        actual = np.zeros(n)
        binding_idx = rng.choice(n, size=200, replace=False)
        actual[binding_idx] = rng.uniform(1, 100, size=200)

        # Random ev_scores — uncorrelated with actual
        ev_scores = rng.rand(n)
        pred_proba = rng.rand(n)
        pred_shadow = rng.uniform(0, 50, size=n)

        result = evaluate_pipeline(actual, pred_proba, pred_shadow, ev_scores)

        # Random: VC@100 should be modest (around 100/2000 = 5% of samples)
        # With 200 binding out of 2000, random top-100 captures ~10 binding
        assert result["EV-VC@100"] < 0.5, (
            f"Random ranking shouldn't capture >50% value at top-100, got {result['EV-VC@100']}"
        )
        # VC values should be non-negative
        assert result["EV-VC@100"] >= 0.0
        assert result["EV-VC@500"] >= 0.0


class TestSpearman:
    """Tests for Spearman rank correlation."""

    def test_spearman_perfect(self):
        """Perfect prediction gives Spearman ~1.0."""
        rng = np.random.RandomState(789)
        n = 500
        actual = np.zeros(n)
        binding_idx = rng.choice(n, size=100, replace=False)
        actual[binding_idx] = rng.uniform(1, 100, size=100)

        # pred_shadow = actual (perfect)
        pred_shadow = actual.copy()
        pred_proba = rng.rand(n)
        ev_scores = rng.rand(n)

        result = evaluate_pipeline(actual, pred_proba, pred_shadow, ev_scores)

        assert result["Spearman"] > 0.99, (
            f"Expected Spearman ~1.0 for perfect prediction, got {result['Spearman']}"
        )


class TestEdgeCases:
    """Edge case handling."""

    def test_edge_case_no_binding(self):
        """All zeros: Spearman=0, RMSE=0, MAE=0."""
        n = 500
        actual = np.zeros(n)
        pred_proba = np.random.rand(n)
        pred_shadow = np.random.rand(n) * 10
        ev_scores = np.random.rand(n)

        result = evaluate_pipeline(actual, pred_proba, pred_shadow, ev_scores)

        assert result["Spearman"] == 0.0
        assert result["C-RMSE"] == 0.0
        assert result["C-MAE"] == 0.0
        assert result["n_binding"] == 0
        assert result["binding_rate"] == 0.0


class TestAggregateMonths:
    """Tests for aggregate_months."""

    def test_aggregate_months(self):
        """Verify mean, std, min, max, bottom_2_mean for 3 months."""
        per_month = {
            "2025-01": {"EV-VC@100": 0.6, "C-RMSE": 10.0},
            "2025-02": {"EV-VC@100": 0.8, "C-RMSE": 20.0},
            "2025-03": {"EV-VC@100": 0.7, "C-RMSE": 15.0},
        }

        agg = aggregate_months(per_month)

        # Check structure
        assert "mean" in agg
        assert "std" in agg
        assert "min" in agg
        assert "max" in agg
        assert "bottom_2_mean" in agg

        # Mean
        assert agg["mean"]["EV-VC@100"] == pytest.approx(0.7, abs=1e-6)
        assert agg["mean"]["C-RMSE"] == pytest.approx(15.0, abs=1e-6)

        # Min/Max
        assert agg["min"]["EV-VC@100"] == pytest.approx(0.6, abs=1e-6)
        assert agg["max"]["EV-VC@100"] == pytest.approx(0.8, abs=1e-6)

        # bottom_2_mean for EV-VC@100 (higher is better) -> 2 lowest: 0.6, 0.7
        assert agg["bottom_2_mean"]["EV-VC@100"] == pytest.approx(0.65, abs=1e-6)

    def test_aggregate_lower_is_better(self):
        """Verify C-RMSE bottom_2_mean uses highest 2 values (worst months)."""
        per_month = {
            "2025-01": {"C-RMSE": 10.0, "C-MAE": 5.0},
            "2025-02": {"C-RMSE": 20.0, "C-MAE": 12.0},
            "2025-03": {"C-RMSE": 15.0, "C-MAE": 8.0},
        }

        agg = aggregate_months(per_month)

        # For C-RMSE (lower is better), bottom_2_mean = worst 2 = highest 2: 20, 15
        assert agg["bottom_2_mean"]["C-RMSE"] == pytest.approx(17.5, abs=1e-6)
        # For C-MAE (lower is better), bottom_2_mean = worst 2 = highest 2: 12, 8
        assert agg["bottom_2_mean"]["C-MAE"] == pytest.approx(10.0, abs=1e-6)
