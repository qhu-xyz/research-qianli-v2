"""Tests for ml/registry.py — experiment persistence."""
import json


def test_save_and_load(tmp_path, monkeypatch):
    """Save experiment and load back."""
    import ml.registry as registry
    monkeypatch.setattr(registry, "REGISTRY_DIR", tmp_path)

    config = {"version": "test", "features": ["bf_6"]}
    metrics = {"per_group": {"2024-06/aq1": {"VC@50": 0.3}}}

    path = registry.save_experiment("test_v1", config, metrics)
    assert (path / "config.json").exists()
    assert (path / "metrics.json").exists()

    loaded_metrics = registry.load_metrics("test_v1")
    assert loaded_metrics["per_group"]["2024-06/aq1"]["VC@50"] == 0.3

    loaded_config = registry.load_config("test_v1")
    assert loaded_config["version"] == "test"


def test_save_nb_gate_results(tmp_path, monkeypatch):
    """Phase 3.0.3: nb_gate_results saved as separate JSON file."""
    import ml.registry as registry
    monkeypatch.setattr(registry, "REGISTRY_DIR", tmp_path)

    config = {"version": "test_nb", "features": ["bin_80_cid_max"]}
    metrics = {"per_group": {"2024-06/aq1": {"VC@50": 0.3}}}
    nb_results = {
        "passed": True,
        "total_count": 4,
        "min_total_count": 3,
        "per_group_counts": {"2025-06/aq1": 2, "2025-06/aq2": 1, "2025-06/aq3": 1},
    }

    path = registry.save_experiment("test_nb_v1", config, metrics, nb_gate_results=nb_results)
    nb_path = path / "nb_gate_results.json"
    assert nb_path.exists()

    with open(nb_path) as f:
        loaded = json.load(f)
    assert loaded["passed"] is True
    assert loaded["total_count"] == 4
