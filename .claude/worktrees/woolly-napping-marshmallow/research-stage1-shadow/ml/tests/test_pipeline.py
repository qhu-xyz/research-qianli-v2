import json

import numpy as np
import pytest

from ml.config import PipelineConfig


@pytest.fixture(autouse=True)
def smoke_mode(monkeypatch):
    monkeypatch.setenv("SMOKE_TEST", "true")


def test_pipeline_smoke_creates_registry(tmp_path):
    """Pipeline creates registry directory with metrics.json."""
    from ml.pipeline import run_pipeline

    reg = tmp_path / "registry"
    reg.mkdir()
    (reg / "version_counter.json").write_text('{"next_id": 1}')

    config = PipelineConfig(version_id="v0001", registry_dir=str(reg))
    metrics = run_pipeline(config)

    assert (reg / "v0001" / "metrics.json").exists()
    assert "S1-AUC" in metrics


def test_pipeline_smoke_creates_config_json(tmp_path):
    """Pipeline writes resolved config to registry."""
    from ml.pipeline import run_pipeline

    reg = tmp_path / "registry"
    reg.mkdir()
    (reg / "version_counter.json").write_text('{"next_id": 1}')

    config = PipelineConfig(version_id="v0001", registry_dir=str(reg))
    run_pipeline(config)

    assert (reg / "v0001" / "config.json").exists()
    with open(reg / "v0001" / "config.json") as f:
        cfg = json.load(f)
    assert "hyperparams" in cfg
    assert "features" in cfg


def test_pipeline_smoke_creates_model(tmp_path):
    """Pipeline saves model file."""
    from ml.pipeline import run_pipeline

    reg = tmp_path / "registry"
    reg.mkdir()
    (reg / "version_counter.json").write_text('{"next_id": 1}')

    config = PipelineConfig(version_id="v0001", registry_dir=str(reg))
    run_pipeline(config)

    model_dir = reg / "v0001" / "model"
    assert model_dir.exists()
    assert (model_dir / "classifier.ubj.gz").exists()
    assert not (model_dir / "classifier.ubj").exists(), "Uncompressed model should be removed"


def test_pipeline_metrics_have_all_gates(tmp_path):
    """Pipeline returns all 10 gate metrics."""
    from ml.pipeline import run_pipeline

    reg = tmp_path / "registry"
    reg.mkdir()
    (reg / "version_counter.json").write_text('{"next_id": 1}')

    config = PipelineConfig(version_id="v0001", registry_dir=str(reg))
    metrics = run_pipeline(config)

    required_keys = [
        "S1-AUC", "S1-AP", "S1-VCAP@100", "S1-VCAP@500", "S1-VCAP@1000",
        "S1-NDCG", "S1-BRIER", "S1-REC", "S1-CAP@100", "S1-CAP@500",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing: {key}"


def test_pipeline_overrides(tmp_path):
    """Config overrides are applied correctly."""
    from ml.pipeline import _apply_overrides
    from ml.config import HyperparamConfig

    hc = HyperparamConfig()
    pc = PipelineConfig()
    hc, pc = _apply_overrides(hc, pc, {"n_estimators": 300, "threshold_beta": 0.5})
    assert hc.n_estimators == 300
    assert pc.threshold_beta == 0.5


def test_pipeline_overrides_unknown_key():
    """Unknown override keys raise ValueError."""
    from ml.pipeline import _apply_overrides
    from ml.config import HyperparamConfig

    hc = HyperparamConfig()
    pc = PipelineConfig()
    with pytest.raises(ValueError, match="Unknown override key"):
        _apply_overrides(hc, pc, {"nonexistent_param": 42})


def test_pipeline_deterministic(tmp_path):
    """Two runs with same config produce same metrics."""
    from ml.pipeline import run_pipeline

    reg1 = tmp_path / "reg1"
    reg1.mkdir()
    (reg1 / "version_counter.json").write_text('{"next_id": 1}')
    config1 = PipelineConfig(version_id="v0001", registry_dir=str(reg1))
    metrics1 = run_pipeline(config1)

    reg2 = tmp_path / "reg2"
    reg2.mkdir()
    (reg2 / "version_counter.json").write_text('{"next_id": 1}')
    config2 = PipelineConfig(version_id="v0001", registry_dir=str(reg2))
    metrics2 = run_pipeline(config2)

    for key in ["S1-AUC", "S1-AP", "S1-BRIER", "S1-REC"]:
        assert metrics1[key] == metrics2[key], f"Non-deterministic: {key}"


def test_from_phase_gt1_raises():
    """from_phase > 1 raises NotImplementedError."""
    from ml.pipeline import run_pipeline

    config = PipelineConfig(version_id="v_test")
    with pytest.raises(NotImplementedError, match="from_phase=2"):
        run_pipeline(config, from_phase=2)
