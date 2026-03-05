"""Tests for LTR pipeline configuration."""
import json
import pytest
from ml.config import LTRConfig, PipelineConfig, GateConfig

def test_ltr_config_defaults():
    cfg = LTRConfig()
    assert cfg.objective == "rank:pairwise"
    assert cfg.n_estimators == 400
    assert cfg.max_depth == 5
    assert cfg.early_stopping_rounds == 50
    assert len(cfg.features) == 40
    assert len(cfg.monotone_constraints) == len(cfg.features)

def test_ltr_config_roundtrip():
    cfg = LTRConfig()
    d = cfg.to_dict()
    cfg2 = LTRConfig.from_dict(d)
    assert cfg2.features == cfg.features
    assert cfg2.objective == cfg.objective
    assert cfg2.n_estimators == cfg.n_estimators

def test_pipeline_config_roundtrip():
    cfg = PipelineConfig()
    d = cfg.to_dict()
    cfg2 = PipelineConfig.from_dict(d)
    assert cfg2.train_months == 6
    assert cfg2.val_months == 2
    assert cfg2.ltr.objective == "rank:pairwise"

def test_gate_config_missing_file(tmp_path):
    cfg = GateConfig.from_json(tmp_path / "nonexistent.json")
    assert cfg.gates == {}

def test_gate_config_loads(tmp_path):
    data = {"gates": {"VC@100": {"floor": 0.8, "direction": "higher", "group": "A"}},
            "noise_tolerance": 0.02, "tail_max_failures": 1}
    path = tmp_path / "gates.json"
    path.write_text(json.dumps(data))
    cfg = GateConfig.from_json(path)
    assert "VC@100" in cfg.gates
    assert cfg.noise_tolerance == 0.02
