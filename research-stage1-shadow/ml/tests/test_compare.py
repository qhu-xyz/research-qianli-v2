import json

import numpy as np
import pytest

from ml.compare import build_comparison_table, check_gates, load_all_versions, run_comparison


@pytest.fixture
def sample_gates():
    return {
        "S1-AUC": {"floor": 0.65, "direction": "higher", "pending_v0": False},
        "S1-BRIER": {"floor": 0.10, "direction": "lower", "pending_v0": False},
        "S1-REC": {"floor": 0.40, "direction": "higher", "pending_v0": False, "group": "B"},
    }


def test_higher_direction_gate_pass(sample_gates):
    """Higher-direction gate: value >= floor => pass."""
    metrics = {"S1-AUC": 0.70, "S1-BRIER": 0.08, "S1-REC": 0.50}
    results = check_gates(metrics, sample_gates)
    assert results["S1-AUC"]["passed"] is True
    assert results["S1-AUC"]["delta"] > 0


def test_higher_direction_gate_fail(sample_gates):
    """Higher-direction gate: value < floor => fail."""
    metrics = {"S1-AUC": 0.60, "S1-BRIER": 0.08, "S1-REC": 0.50}
    results = check_gates(metrics, sample_gates)
    assert results["S1-AUC"]["passed"] is False


def test_lower_direction_gate_pass(sample_gates):
    """Lower-direction gate (Brier): value <= floor => pass."""
    metrics = {"S1-AUC": 0.70, "S1-BRIER": 0.08, "S1-REC": 0.50}
    results = check_gates(metrics, sample_gates)
    assert results["S1-BRIER"]["passed"] is True


def test_lower_direction_gate_fail(sample_gates):
    """Lower-direction gate (Brier): value > floor => fail."""
    metrics = {"S1-AUC": 0.70, "S1-BRIER": 0.15, "S1-REC": 0.50}
    results = check_gates(metrics, sample_gates)
    assert results["S1-BRIER"]["passed"] is False


def test_noise_tolerance_edge_case(sample_gates):
    """At exact boundary, should still pass."""
    metrics = {"S1-AUC": 0.65, "S1-BRIER": 0.10, "S1-REC": 0.40}
    results = check_gates(metrics, sample_gates)
    assert results["S1-AUC"]["passed"] is True
    assert results["S1-BRIER"]["passed"] is True
    assert results["S1-REC"]["passed"] is True


def test_regression_vs_champion(sample_gates):
    """If value regresses vs champion beyond noise_tolerance, gate fails."""
    champion_metrics = {"S1-AUC": 0.80, "S1-BRIER": 0.07, "S1-REC": 0.60}
    # S1-AUC: 0.70 < 0.80 - 0.02 = 0.78 => regression => fail
    metrics = {"S1-AUC": 0.70, "S1-BRIER": 0.08, "S1-REC": 0.50}
    results = check_gates(metrics, sample_gates, champion_metrics, noise_tolerance=0.02)
    assert results["S1-AUC"]["passed"] is False


def test_no_regression_within_tolerance(sample_gates):
    """Small regression within noise_tolerance should still pass."""
    champion_metrics = {"S1-AUC": 0.70, "S1-BRIER": 0.08, "S1-REC": 0.50}
    # S1-AUC: 0.69 >= 0.70 - 0.02 = 0.68 => no regression => pass (above floor too)
    metrics = {"S1-AUC": 0.69, "S1-BRIER": 0.08, "S1-REC": 0.50}
    results = check_gates(metrics, sample_gates, champion_metrics, noise_tolerance=0.02)
    assert results["S1-AUC"]["passed"] is True


def test_missing_v0_metrics(sample_gates):
    """Missing metric should result in passed=None."""
    metrics = {"S1-AUC": 0.70}  # Missing BRIER and REC
    results = check_gates(metrics, sample_gates)
    assert results["S1-BRIER"]["passed"] is None
    assert results["S1-REC"]["passed"] is None


def test_null_floor_gate():
    """Gate with null floor should result in passed=None."""
    gates = {"S1-VCAP@100": {"floor": None, "direction": "higher", "pending_v0": True}}
    metrics = {"S1-VCAP@100": 0.50}
    results = check_gates(metrics, gates)
    assert results["S1-VCAP@100"]["passed"] is None


def test_markdown_table_no_broken_pipes(sample_gates):
    """Markdown output must not have broken pipe characters in cells."""
    versions = {
        "v0": {"S1-AUC": 0.75, "S1-BRIER": 0.089, "S1-REC": 0.55},
        "v0001": {"S1-AUC": 0.70, "S1-BRIER": 0.095, "S1-REC": 0.50},
    }
    table = build_comparison_table(versions, sample_gates)
    # Each row should have the same number of pipes
    lines = table.strip().split("\n")
    pipe_counts = [line.count("|") for line in lines]
    assert len(set(pipe_counts)) == 1, f"Inconsistent pipe counts: {pipe_counts}"


def test_load_all_versions(tmp_path):
    """load_all_versions reads all v*/metrics.json."""
    reg = tmp_path / "registry"
    reg.mkdir()
    v0 = reg / "v0"
    v0.mkdir()
    (v0 / "metrics.json").write_text('{"S1-AUC": 0.75}')
    v1 = reg / "v0001"
    v1.mkdir()
    (v1 / "metrics.json").write_text('{"S1-AUC": 0.70}')

    versions = load_all_versions(str(reg))
    assert "v0" in versions
    assert "v0001" in versions
    assert versions["v0"]["S1-AUC"] == 0.75


def test_run_comparison_writes_outputs(tmp_path):
    """run_comparison produces JSON and optionally Markdown."""
    reg = tmp_path / "registry"
    reg.mkdir()
    (reg / "comparisons").mkdir()
    v0 = reg / "v0"
    v0.mkdir()
    (v0 / "metrics.json").write_text('{"S1-AUC": 0.75, "S1-BRIER": 0.089, "S1-REC": 0.55}')
    (reg / "champion.json").write_text('{"version": null, "promoted_at": null}')

    gates = {
        "version": 1,
        "noise_tolerance": 0.02,
        "gates": {
            "S1-AUC": {"floor": 0.65, "direction": "higher", "pending_v0": False},
            "S1-BRIER": {"floor": 0.10, "direction": "lower", "pending_v0": False},
            "S1-REC": {"floor": 0.40, "direction": "higher", "pending_v0": False},
        },
    }
    (reg / "gates.json").write_text(json.dumps(gates))

    output_md = str(tmp_path / "report.md")
    result = run_comparison(
        batch_id="test_batch",
        iteration=1,
        registry_dir=str(reg),
        gates_path=str(reg / "gates.json"),
        champion_path=str(reg / "champion.json"),
        output_path=output_md,
    )

    assert (tmp_path / "report.md").exists()
    assert (reg / "comparisons" / "test_batch_iter1.json").exists()
    assert "v0" in result["versions"]
