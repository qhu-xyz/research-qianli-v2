"""Tests for ml.compare — gate checking with tier classification metrics."""
from __future__ import annotations

from ml.compare import (
    check_gates,
    check_gates_multi_month,
    evaluate_overall_pass,
    evaluate_overall_pass_multi_month,
)


# Tier gates fixture (all higher-is-better)
_GATES = {
    "Tier-VC@100": {"floor": 0.05, "tail_floor": 0.01, "direction": "higher", "group": "A"},
    "Tier-VC@500": {"floor": 0.20, "tail_floor": 0.05, "direction": "higher", "group": "A"},
    "Tier-NDCG": {"floor": 0.70, "tail_floor": 0.60, "direction": "higher", "group": "A"},
    "QWK": {"floor": 0.30, "tail_floor": 0.15, "direction": "higher", "group": "A"},
    "Macro-F1": {"floor": 0.30, "tail_floor": 0.25, "direction": "higher", "group": "B"},
    "Tier-Accuracy": {"floor": 0.93, "tail_floor": 0.92, "direction": "higher", "group": "B"},
    "Adjacent-Accuracy": {"floor": 0.96, "tail_floor": 0.95, "direction": "higher", "group": "B"},
    "Tier-Recall@0": {"floor": 0.30, "tail_floor": 0.05, "direction": "higher", "group": "B"},
    "Tier-Recall@1": {"floor": 0.08, "tail_floor": 0.02, "direction": "higher", "group": "B"},
}


class TestCheckGatesPassing:
    def test_all_gates_pass(self):
        """Version that exceeds all floors passes."""
        metrics = {
            "Tier-VC@100": 0.10,
            "Tier-VC@500": 0.25,
            "Tier-NDCG": 0.80,
            "QWK": 0.40,
            "Macro-F1": 0.40,
            "Tier-Accuracy": 0.95,
            "Adjacent-Accuracy": 0.98,
            "Tier-Recall@0": 0.40,
            "Tier-Recall@1": 0.12,
        }
        results = check_gates(metrics, _GATES)
        ga, gb = evaluate_overall_pass(results)
        assert ga is True
        assert gb is True


class TestCheckGatesFailing:
    def test_below_floor_fails(self):
        """Version below Group A floor returns overall fail."""
        metrics = {
            "Tier-VC@100": 0.02,   # below 0.05 floor
            "Tier-VC@500": 0.25,
            "Tier-NDCG": 0.80,
            "QWK": 0.40,
            "Macro-F1": 0.40,
            "Tier-Accuracy": 0.95,
            "Adjacent-Accuracy": 0.98,
            "Tier-Recall@0": 0.40,
            "Tier-Recall@1": 0.12,
        }
        results = check_gates(metrics, _GATES)
        assert results["Tier-VC@100"]["passed"] is False
        ga, gb = evaluate_overall_pass(results)
        assert ga is False
        assert gb is True  # Group B still passes

    def test_group_b_fail_doesnt_block_a(self):
        """Group B failure doesn't block Group A."""
        metrics = {
            "Tier-VC@100": 0.10,
            "Tier-VC@500": 0.25,
            "Tier-NDCG": 0.80,
            "QWK": 0.40,
            "Macro-F1": 0.20,   # below 0.30 floor -> fail
            "Tier-Accuracy": 0.95,
            "Adjacent-Accuracy": 0.98,
            "Tier-Recall@0": 0.40,
            "Tier-Recall@1": 0.12,
        }
        results = check_gates(metrics, _GATES)
        assert results["Macro-F1"]["passed"] is False
        ga, gb = evaluate_overall_pass(results)
        assert ga is True   # Group A still passes
        assert gb is False

    def test_nan_metric_fails(self):
        """NaN metric value causes gate failure."""
        metrics = {
            "Tier-VC@100": float("nan"),
            "Tier-VC@500": 0.25,
            "Tier-NDCG": 0.80,
            "QWK": 0.40,
            "Macro-F1": 0.40,
            "Tier-Accuracy": 0.95,
            "Adjacent-Accuracy": 0.98,
            "Tier-Recall@0": 0.40,
            "Tier-Recall@1": 0.12,
        }
        results = check_gates(metrics, _GATES)
        assert results["Tier-VC@100"]["passed"] is False


class TestMultiMonthGates:
    def test_multi_month_passing(self):
        """All gates pass with multi-month data."""
        per_month = {
            "2024-01": {
                "Tier-VC@100": 0.08, "Tier-VC@500": 0.22, "Tier-NDCG": 0.75,
                "QWK": 0.35, "Macro-F1": 0.35, "Tier-Accuracy": 0.94,
                "Adjacent-Accuracy": 0.97, "Tier-Recall@0": 0.35, "Tier-Recall@1": 0.10,
            },
            "2024-02": {
                "Tier-VC@100": 0.12, "Tier-VC@500": 0.28, "Tier-NDCG": 0.82,
                "QWK": 0.42, "Macro-F1": 0.42, "Tier-Accuracy": 0.95,
                "Adjacent-Accuracy": 0.98, "Tier-Recall@0": 0.45, "Tier-Recall@1": 0.15,
            },
            "2024-03": {
                "Tier-VC@100": 0.10, "Tier-VC@500": 0.25, "Tier-NDCG": 0.78,
                "QWK": 0.38, "Macro-F1": 0.38, "Tier-Accuracy": 0.94,
                "Adjacent-Accuracy": 0.97, "Tier-Recall@0": 0.40, "Tier-Recall@1": 0.12,
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
                "Tier-VC@100": 0.02, "Tier-VC@500": 0.22, "Tier-NDCG": 0.75,
                "QWK": 0.35, "Macro-F1": 0.35, "Tier-Accuracy": 0.94,
                "Adjacent-Accuracy": 0.97, "Tier-Recall@0": 0.35, "Tier-Recall@1": 0.10,
            },
            "2024-02": {
                "Tier-VC@100": 0.04, "Tier-VC@500": 0.28, "Tier-NDCG": 0.82,
                "QWK": 0.42, "Macro-F1": 0.42, "Tier-Accuracy": 0.95,
                "Adjacent-Accuracy": 0.98, "Tier-Recall@0": 0.45, "Tier-Recall@1": 0.15,
            },
        }
        results = check_gates_multi_month(per_month, _GATES)
        # Mean of Tier-VC@100 = (0.02 + 0.04) / 2 = 0.03, below 0.05 floor
        assert results["Tier-VC@100"]["mean_passed"] is False
        assert results["Tier-VC@100"]["overall_passed"] is False
        ga, _ = evaluate_overall_pass_multi_month(results)
        assert ga is False
