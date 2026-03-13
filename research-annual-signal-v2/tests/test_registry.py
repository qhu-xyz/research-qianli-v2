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
