"""Tests for ml.config — TierConfig, PipelineConfig, GateConfig."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ml.config import GateConfig, PipelineConfig, TierConfig


# ---------------------------------------------------------------------------
# TierConfig
# ---------------------------------------------------------------------------

class TestTierConfig:
    def test_tier_config_defaults(self):
        """Verify 34 features, monotone constraints, hyperparams, class_weights."""
        cfg = TierConfig()

        assert len(cfg.features) == 34
        assert len(cfg.monotone_constraints) == len(cfg.features)

        # hyperparams
        assert cfg.n_estimators == 400
        assert cfg.max_depth == 5
        assert cfg.learning_rate == 0.05
        assert cfg.subsample == 0.8
        assert cfg.colsample_bytree == 0.8
        assert cfg.reg_alpha == 1.0
        assert cfg.reg_lambda == 1.0
        assert cfg.min_child_weight == 25

        # multi-class params
        assert cfg.num_class == 5

        # class weights
        assert cfg.class_weights == {0: 10, 1: 5, 2: 2, 3: 1, 4: 0.5}

    def test_tier_config_bins(self):
        """Verify default bin edges."""
        cfg = TierConfig()
        assert cfg.bins == [float("-inf"), 0, 100, 1000, 3000, float("inf")]

    def test_tier_config_midpoints(self):
        """Verify default tier midpoints for EV score."""
        cfg = TierConfig()
        assert cfg.tier_midpoints == [4000, 2000, 550, 50, 0]

    def test_tier_config_is_mutable(self):
        """TierConfig must be mutable (agentic loop can iterate)."""
        cfg = TierConfig()
        cfg.n_estimators = 800
        assert cfg.n_estimators == 800

    def test_tier_config_class_weights_mutable(self):
        """Class weights can be changed."""
        cfg = TierConfig()
        cfg.class_weights = {0: 20, 1: 10, 2: 3, 3: 1, 4: 0.3}
        assert cfg.class_weights[0] == 20


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    def test_pipeline_config_composition(self):
        """PipelineConfig wraps TierConfig with pipeline params."""
        cfg = PipelineConfig()

        assert isinstance(cfg.tier, TierConfig)
        assert cfg.train_months == 6
        assert cfg.val_months == 2

    def test_pipeline_config_roundtrip(self):
        """to_dict() then from_dict() produces equivalent config."""
        original = PipelineConfig()
        d = original.to_dict()

        # Dict must be JSON-serializable
        json_str = json.dumps(d)
        restored_dict = json.loads(json_str)

        restored = PipelineConfig.from_dict(restored_dict)

        # Compare all fields
        assert restored.tier.features == original.tier.features
        assert restored.tier.monotone_constraints == original.tier.monotone_constraints
        assert restored.tier.n_estimators == original.tier.n_estimators
        assert restored.tier.max_depth == original.tier.max_depth
        assert restored.tier.learning_rate == original.tier.learning_rate
        assert restored.tier.class_weights == original.tier.class_weights
        assert restored.tier.bins == original.tier.bins
        assert restored.tier.tier_midpoints == original.tier.tier_midpoints
        assert restored.tier.num_class == original.tier.num_class

        assert restored.train_months == original.train_months
        assert restored.val_months == original.val_months


# ---------------------------------------------------------------------------
# GateConfig
# ---------------------------------------------------------------------------

class TestGateConfig:
    def test_gate_config_loads_json(self):
        """Load GateConfig from a temp JSON file, verify fields."""
        data = {
            "gates": {
                "Tier-VC@100": {"floor": 0.1},
                "QWK": {"floor": 0.5},
            },
            "eval_months": {"start": 1, "end": 12},
            "noise_tolerance": 0.02,
            "tail_max_failures": 3,
            "cascade_stages": [],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            tmp_path = f.name

        try:
            cfg = GateConfig.from_json(tmp_path)
            assert "Tier-VC@100" in cfg.gates
            assert cfg.noise_tolerance == 0.02
            assert cfg.tail_max_failures == 3
        finally:
            Path(tmp_path).unlink()

    def test_gate_config_missing_file_graceful(self):
        """GateConfig handles nonexistent file gracefully with empty defaults."""
        cfg = GateConfig.from_json("/tmp/nonexistent_gates_file_abc123.json")
        assert cfg.gates == {}
        assert cfg.eval_months == {}
        assert cfg.noise_tolerance == 0.0
        assert cfg.tail_max_failures == 0
        assert cfg.cascade_stages == []
