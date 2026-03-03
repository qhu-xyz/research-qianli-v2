"""Tests for ml.compare — gate checking with stage-2 regression metrics."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.compare import (
    check_gates,
    check_gates_multi_month,
    evaluate_overall_pass,
    evaluate_overall_pass_multi_month,
)


# Stage 2 gates fixture (Group A: higher is better; Group B: mixed)
_GATES = {
    "EV-VC@100": {"floor": 0.50, "tail_floor": 0.40, "direction": "higher", "group": "A"},
    "EV-VC@500": {"floor": 0.70, "tail_floor": 0.60, "direction": "higher", "group": "A"},
    "EV-NDCG": {"floor": 0.80, "tail_floor": 0.70, "direction": "higher", "group": "A"},
    "Spearman": {"floor": 0.40, "tail_floor": 0.30, "direction": "higher", "group": "A"},
    "C-RMSE": {"floor": 150.0, "tail_floor": 200.0, "direction": "lower", "group": "B"},
    "C-MAE": {"floor": 100.0, "tail_floor": 130.0, "direction": "lower", "group": "B"},
    "EV-VC@1000": {"floor": 0.85, "tail_floor": 0.75, "direction": "higher", "group": "B"},
    "R-REC@500": {"floor": 0.60, "tail_floor": 0.50, "direction": "higher", "group": "B"},
}


class TestCheckGatesPassing:
    def test_all_gates_pass(self):
        """Version that exceeds all floors passes."""
        metrics = {
            "EV-VC@100": 0.65,
            "EV-VC@500": 0.80,
            "EV-NDCG": 0.90,
            "Spearman": 0.55,
            "C-RMSE": 120.0,
            "C-MAE": 80.0,
            "EV-VC@1000": 0.92,
            "R-REC@500": 0.70,
        }
        results = check_gates(metrics, _GATES)
        ga, gb = evaluate_overall_pass(results)
        assert ga is True
        assert gb is True

    def test_lower_is_better_pass(self):
        """C-RMSE and C-MAE pass when below floor (lower is better)."""
        metrics = {
            "EV-VC@100": 0.65,
            "EV-VC@500": 0.80,
            "EV-NDCG": 0.90,
            "Spearman": 0.55,
            "C-RMSE": 140.0,   # below 150 floor -> pass
            "C-MAE": 95.0,     # below 100 floor -> pass
            "EV-VC@1000": 0.90,
            "R-REC@500": 0.65,
        }
        results = check_gates(metrics, _GATES)
        assert results["C-RMSE"]["passed"] is True
        assert results["C-MAE"]["passed"] is True


class TestCheckGatesFailing:
    def test_below_floor_fails(self):
        """Version below Group A floor returns overall fail."""
        metrics = {
            "EV-VC@100": 0.30,   # below 0.50 floor
            "EV-VC@500": 0.80,
            "EV-NDCG": 0.90,
            "Spearman": 0.55,
            "C-RMSE": 120.0,
            "C-MAE": 80.0,
            "EV-VC@1000": 0.92,
            "R-REC@500": 0.70,
        }
        results = check_gates(metrics, _GATES)
        assert results["EV-VC@100"]["passed"] is False
        ga, gb = evaluate_overall_pass(results)
        assert ga is False
        assert gb is True  # Group B still passes

    def test_lower_is_better_above_floor_fails(self):
        """C-RMSE above floor (lower is better) fails."""
        metrics = {
            "EV-VC@100": 0.65,
            "EV-VC@500": 0.80,
            "EV-NDCG": 0.90,
            "Spearman": 0.55,
            "C-RMSE": 160.0,   # above 150 floor -> fail
            "C-MAE": 80.0,
            "EV-VC@1000": 0.92,
            "R-REC@500": 0.70,
        }
        results = check_gates(metrics, _GATES)
        assert results["C-RMSE"]["passed"] is False
        ga, gb = evaluate_overall_pass(results)
        assert ga is True   # Group A still passes (C-RMSE is Group B)
        assert gb is False

    def test_nan_metric_fails(self):
        """NaN metric value causes gate failure."""
        metrics = {
            "EV-VC@100": float("nan"),
            "EV-VC@500": 0.80,
            "EV-NDCG": 0.90,
            "Spearman": 0.55,
            "C-RMSE": 120.0,
            "C-MAE": 80.0,
            "EV-VC@1000": 0.92,
            "R-REC@500": 0.70,
        }
        results = check_gates(metrics, _GATES)
        assert results["EV-VC@100"]["passed"] is False


class TestMultiMonthGates:
    def test_multi_month_passing(self):
        """All gates pass with multi-month data."""
        per_month = {
            "2024-01": {
                "EV-VC@100": 0.60, "EV-VC@500": 0.75, "EV-NDCG": 0.85,
                "Spearman": 0.50, "C-RMSE": 130.0, "C-MAE": 90.0,
                "EV-VC@1000": 0.90, "R-REC@500": 0.65,
            },
            "2024-02": {
                "EV-VC@100": 0.70, "EV-VC@500": 0.82, "EV-NDCG": 0.92,
                "Spearman": 0.60, "C-RMSE": 110.0, "C-MAE": 70.0,
                "EV-VC@1000": 0.95, "R-REC@500": 0.75,
            },
            "2024-03": {
                "EV-VC@100": 0.65, "EV-VC@500": 0.78, "EV-NDCG": 0.88,
                "Spearman": 0.55, "C-RMSE": 125.0, "C-MAE": 85.0,
                "EV-VC@1000": 0.91, "R-REC@500": 0.68,
            },
        }
        results = check_gates_multi_month(per_month, _GATES)
        ga, gb = evaluate_overall_pass_multi_month(results)
        assert ga is True
        assert gb is True

    def test_multi_month_mean_fails(self):
        """Mean below floor causes gate failure."""
        per_month = {
            "2024-01": {
                "EV-VC@100": 0.30, "EV-VC@500": 0.75, "EV-NDCG": 0.85,
                "Spearman": 0.50, "C-RMSE": 130.0, "C-MAE": 90.0,
                "EV-VC@1000": 0.90, "R-REC@500": 0.65,
            },
            "2024-02": {
                "EV-VC@100": 0.40, "EV-VC@500": 0.82, "EV-NDCG": 0.92,
                "Spearman": 0.60, "C-RMSE": 110.0, "C-MAE": 70.0,
                "EV-VC@1000": 0.95, "R-REC@500": 0.75,
            },
        }
        results = check_gates_multi_month(per_month, _GATES)
        # Mean of EV-VC@100 = (0.30 + 0.40) / 2 = 0.35, below 0.50 floor
        assert results["EV-VC@100"]["mean_passed"] is False
        assert results["EV-VC@100"]["overall_passed"] is False
        ga, _ = evaluate_overall_pass_multi_month(results)
        assert ga is False
