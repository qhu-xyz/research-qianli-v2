"""Tests for ml/evaluate.py — metrics and gates."""
import numpy as np
import pytest


def test_vc_at_k():
    """Test spec G1: VC@K computation."""
    from ml.evaluate import value_capture_at_k
    actual = np.array([100, 50, 30, 20, 0, 0, 0, 0, 0, 0], dtype=float)
    scores = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
    vc50 = value_capture_at_k(actual, scores, k=2)
    assert abs(vc50 - 150 / 200) < 1e-6  # top 2 capture 150 of 200


def test_recall_at_k():
    """Test spec G2: Recall@K computation."""
    from ml.evaluate import recall_at_k
    actual = np.array([100, 50, 30, 0, 0, 0], dtype=float)
    scores = np.array([6, 5, 4, 3, 2, 1], dtype=float)
    recall = recall_at_k(actual, scores, k=2)
    assert abs(recall - 2 / 3) < 1e-6  # 2 of 3 binders in top 2


def test_abs_sp_at_k():
    """Test spec G3: Abs_SP uses total DA SP (not in-universe)."""
    from ml.evaluate import abs_sp_at_k
    actual = np.array([100, 50, 0, 0], dtype=float)
    scores = np.array([4, 3, 2, 1], dtype=float)
    total_da_sp = 500.0  # includes outside-universe SP
    abs_sp = abs_sp_at_k(actual, scores, k=2, total_da_sp=total_da_sp)
    assert abs(abs_sp - 150 / 500) < 1e-6


def test_nb_recall_at_k():
    """Test spec G4: NB12_Recall@K."""
    from ml.evaluate import nb_recall_at_k
    actual = np.array([100, 50, 30, 0, 0], dtype=float)
    scores = np.array([5, 4, 3, 2, 1], dtype=float)
    is_nb = np.array([True, False, True, False, False])
    # NB binders: index 0 (100, NB, in top-2) and index 2 (30, NB, NOT in top-2)
    nb_recall = nb_recall_at_k(actual, scores, is_nb, k=2)
    assert abs(nb_recall - 1 / 2) < 1e-6  # 1 of 2 NB binders in top 2


def test_gate_checking():
    """Test spec L1: 2/3 holdout groups rule."""
    from ml.evaluate import check_gates
    candidate = {
        "2025-06/aq1": {"VC@50": 0.30},
        "2025-06/aq2": {"VC@50": 0.25},
        "2025-06/aq3": {"VC@50": 0.35},
    }
    baseline = {
        "2025-06/aq1": {"VC@50": 0.28},
        "2025-06/aq2": {"VC@50": 0.27},
        "2025-06/aq3": {"VC@50": 0.30},
    }
    gates = check_gates(candidate, baseline, "v0c",
                        ["2025-06/aq1", "2025-06/aq2", "2025-06/aq3"])
    # Wins on aq1 (0.30 > 0.28) and aq3 (0.35 > 0.30) = 2/3
    # Mean: 0.30 vs 0.283 -> pass
    assert gates["VC@50"]["passed"] is True
    assert gates["VC@50"]["wins"] == 2


def test_gate_fails_with_mean_below_baseline():
    """Test spec L1: 3/3 wins but mean < baseline -> FAIL."""
    from ml.evaluate import check_gates
    candidate = {
        "2025-06/aq1": {"VC@50": 0.201},
        "2025-06/aq2": {"VC@50": 0.201},
        "2025-06/aq3": {"VC@50": 0.201},
    }
    baseline = {
        "2025-06/aq1": {"VC@50": 0.200},
        "2025-06/aq2": {"VC@50": 0.200},
        "2025-06/aq3": {"VC@50": 0.300},  # baseline mean = 0.233
    }
    gates = check_gates(candidate, baseline, "v0c",
                        ["2025-06/aq1", "2025-06/aq2", "2025-06/aq3"])
    # 3 wins (all >=), but candidate mean (0.201) < baseline mean (0.233)
    assert gates["VC@50"]["passed"] is False


def test_gate_fails_with_1_win():
    """1 of 3 wins -> FAIL even if mean is higher."""
    from ml.evaluate import check_gates
    candidate = {
        "2025-06/aq1": {"VC@50": 0.50},
        "2025-06/aq2": {"VC@50": 0.10},
        "2025-06/aq3": {"VC@50": 0.10},
    }
    baseline = {
        "2025-06/aq1": {"VC@50": 0.20},
        "2025-06/aq2": {"VC@50": 0.20},
        "2025-06/aq3": {"VC@50": 0.20},
    }
    gates = check_gates(candidate, baseline, "v0c",
                        ["2025-06/aq1", "2025-06/aq2", "2025-06/aq3"])
    assert gates["VC@50"]["wins"] == 1
    assert gates["VC@50"]["passed"] is False


def test_ndcg_uses_tiered_labels():
    """Contract: NDCG uses label_tier (0/1/2/3), NOT continuous SP."""
    from ml.evaluate import ndcg
    # Tiered labels: if NDCG used continuous SP, result would differ
    tiered = np.array([3, 2, 1, 0, 0, 0], dtype=float)
    scores = np.array([6, 5, 4, 3, 2, 1], dtype=float)  # perfect ranking
    result = ndcg(tiered, scores)
    assert abs(result - 1.0) < 1e-6, "Perfect ranking should give NDCG=1.0"

    # Verify it does NOT accept continuous SP semantics: with SP [100, 1, 1, 0, 0, 0]
    # and scores ranking item 0 first, continuous NDCG ~ 1.0 but tiered NDCG < 1.0
    # if item with tier=3 is not ranked first
    tiered2 = np.array([1, 3, 2, 0, 0, 0], dtype=float)
    scores2 = np.array([6, 5, 4, 3, 2, 1], dtype=float)  # ranks tier=1 above tier=3
    result2 = ndcg(tiered2, scores2)
    assert result2 < 1.0, "Misordered tiers should give NDCG < 1.0"


def test_gates_restrict_to_tier1_metrics():
    """Contract: check_gates only checks TIER1_GATE_METRICS, not all metrics."""
    from ml.evaluate import check_gates
    from ml.config import TIER1_GATE_METRICS
    candidate = {
        "2025-06/aq1": {"VC@50": 0.30, "Spearman": 0.10, "n_branches": 100},
        "2025-06/aq2": {"VC@50": 0.30, "Spearman": 0.10, "n_branches": 100},
        "2025-06/aq3": {"VC@50": 0.30, "Spearman": 0.10, "n_branches": 100},
    }
    baseline = {
        "2025-06/aq1": {"VC@50": 0.25, "Spearman": 0.50, "n_branches": 100},
        "2025-06/aq2": {"VC@50": 0.25, "Spearman": 0.50, "n_branches": 100},
        "2025-06/aq3": {"VC@50": 0.25, "Spearman": 0.50, "n_branches": 100},
    }
    gates = check_gates(candidate, baseline, "v0c",
                        ["2025-06/aq1", "2025-06/aq2", "2025-06/aq3"])
    # Spearman and n_branches should NOT appear in gates (not Tier 1)
    assert "Spearman" not in gates, "Non-Tier1 metric 'Spearman' should not be gated"
    assert "n_branches" not in gates, "Non-Tier1 metric 'n_branches' should not be gated"
    # Only Tier 1 metrics that have data should appear
    for metric in gates:
        assert metric in TIER1_GATE_METRICS, f"Gate metric '{metric}' is not in TIER1_GATE_METRICS"


def test_nb12_recall_not_in_tier1_gates():
    """NB12_Recall@50 was removed from TIER1_GATE_METRICS (Phase 3.0.1)."""
    from ml.config import TIER1_GATE_METRICS
    assert "NB12_Recall@50" not in TIER1_GATE_METRICS


def test_two_track_gate_metrics():
    """Phase 3.0.1: TWO_TRACK_GATE_METRICS restricts gating to top-50 only."""
    from ml.config import TWO_TRACK_GATE_METRICS
    assert TWO_TRACK_GATE_METRICS == ["VC@50", "Recall@50", "Abs_SP@50"]


def test_check_gates_with_custom_gate_metrics():
    """Phase 3.0.1: check_gates accepts gate_metrics param to override TIER1_GATE_METRICS."""
    from ml.evaluate import check_gates
    candidate = {
        "2025-06/aq1": {"VC@50": 0.30, "VC@100": 0.40, "Recall@50": 0.25},
        "2025-06/aq2": {"VC@50": 0.30, "VC@100": 0.40, "Recall@50": 0.25},
        "2025-06/aq3": {"VC@50": 0.30, "VC@100": 0.40, "Recall@50": 0.25},
    }
    baseline = {
        "2025-06/aq1": {"VC@50": 0.25, "VC@100": 0.50, "Recall@50": 0.20},
        "2025-06/aq2": {"VC@50": 0.25, "VC@100": 0.50, "Recall@50": 0.20},
        "2025-06/aq3": {"VC@50": 0.25, "VC@100": 0.50, "Recall@50": 0.20},
    }
    gates = check_gates(
        candidate, baseline, "v0c",
        ["2025-06/aq1", "2025-06/aq2", "2025-06/aq3"],
        gate_metrics=["VC@50", "Recall@50"],
    )
    assert "VC@50" in gates
    assert "Recall@50" in gates
    assert "VC@100" not in gates, "VC@100 should be excluded by gate_metrics override"
