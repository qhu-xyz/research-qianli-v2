"""Tests for ml.config — ClassifierConfig, RegressorConfig, PipelineConfig, GateConfig."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ml.config import ClassifierConfig, GateConfig, PipelineConfig, RegressorConfig


# ---------------------------------------------------------------------------
# ClassifierConfig
# ---------------------------------------------------------------------------

class TestClassifierConfig:
    def test_classifier_config_frozen_defaults(self):
        """Verify 13 features, monotone constraints, hyperparams, frozen."""
        cfg = ClassifierConfig()

        # 13 step-1 features
        assert len(cfg.features) == 13

        expected_features = [
            "prob_exceed_110",
            "prob_exceed_105",
            "prob_exceed_100",
            "prob_exceed_95",
            "prob_exceed_90",
            "prob_below_100",
            "prob_below_95",
            "prob_below_90",
            "expected_overload",
            "density_skewness",
            "density_kurtosis",
            "hist_da",
            "hist_da_trend",
        ]
        assert list(cfg.features) == expected_features

        # monotone constraints match features one-to-one
        expected_constraints = [1, 1, 1, 1, 1, -1, -1, -1, 1, 0, 0, 1, 1]
        assert list(cfg.monotone_constraints) == expected_constraints
        assert len(cfg.monotone_constraints) == len(cfg.features)

        # hyperparams
        assert cfg.n_estimators == 200
        assert cfg.max_depth == 4
        assert cfg.learning_rate == 0.1
        assert cfg.subsample == 0.8
        assert cfg.colsample_bytree == 0.8
        assert cfg.reg_alpha == 0.1
        assert cfg.reg_lambda == 1.0
        assert cfg.min_child_weight == 10

        # threshold_beta
        assert cfg.threshold_beta == 0.7

    def test_classifier_config_is_frozen(self):
        """ClassifierConfig must be immutable (frozen dataclass)."""
        cfg = ClassifierConfig()
        with pytest.raises(AttributeError):
            cfg.n_estimators = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RegressorConfig
# ---------------------------------------------------------------------------

class TestRegressorConfig:
    def test_regressor_config_defaults(self):
        """Verify 24 features, constraints, hyperparams, unified_regressor=False."""
        cfg = RegressorConfig()

        # 24 step-2 features: all 13 from classifier + 11 additional
        assert len(cfg.features) == 24

        # First 13 must match classifier features
        classifier_feats = list(ClassifierConfig().features)
        assert cfg.features[:13] == classifier_feats

        additional_features = [
            "prob_exceed_85",
            "prob_exceed_80",
            "tail_concentration",
            "prob_band_95_100",
            "prob_band_100_105",
            "density_mean",
            "density_variance",
            "density_entropy",
            "recent_hist_da",
            "season_hist_da_1",
            "season_hist_da_2",
        ]
        assert cfg.features[13:] == additional_features

        # monotone constraints for all 24
        expected_constraints = [
            # classifier 13
            1, 1, 1, 1, 1, -1, -1, -1, 1, 0, 0, 1, 1,
            # additional 11
            1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1,
        ]
        assert cfg.monotone_constraints == expected_constraints
        assert len(cfg.monotone_constraints) == len(cfg.features)

        # hyperparams
        assert cfg.n_estimators == 400
        assert cfg.max_depth == 5
        assert cfg.learning_rate == 0.05
        assert cfg.subsample == 0.8
        assert cfg.colsample_bytree == 0.8
        assert cfg.reg_alpha == 0.1
        assert cfg.reg_lambda == 1.0
        assert cfg.min_child_weight == 10

        # gated mode defaults
        assert cfg.unified_regressor is False
        assert cfg.value_weighted is False

    def test_regressor_config_is_mutable(self):
        """RegressorConfig must be mutable (agentic loop can iterate)."""
        cfg = RegressorConfig()
        cfg.n_estimators = 800
        assert cfg.n_estimators == 800


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    def test_pipeline_config_composition(self):
        """PipelineConfig wraps classifier + regressor with pipeline params."""
        cfg = PipelineConfig()

        assert isinstance(cfg.classifier, ClassifierConfig)
        assert isinstance(cfg.regressor, RegressorConfig)
        assert cfg.train_months == 10
        assert cfg.val_months == 2
        assert cfg.ev_scoring is True

    def test_pipeline_config_roundtrip(self):
        """to_dict() then from_dict() produces equivalent config."""
        original = PipelineConfig()
        d = original.to_dict()

        # Dict must be JSON-serializable
        json_str = json.dumps(d)
        restored_dict = json.loads(json_str)

        restored = PipelineConfig.from_dict(restored_dict)

        # Compare all fields
        assert restored.classifier.features == original.classifier.features
        assert restored.classifier.monotone_constraints == original.classifier.monotone_constraints
        assert restored.classifier.n_estimators == original.classifier.n_estimators
        assert restored.classifier.max_depth == original.classifier.max_depth
        assert restored.classifier.learning_rate == original.classifier.learning_rate
        assert restored.classifier.threshold_beta == original.classifier.threshold_beta

        assert restored.regressor.features == original.regressor.features
        assert restored.regressor.monotone_constraints == original.regressor.monotone_constraints
        assert restored.regressor.n_estimators == original.regressor.n_estimators
        assert restored.regressor.max_depth == original.regressor.max_depth
        assert restored.regressor.learning_rate == original.regressor.learning_rate
        assert restored.regressor.unified_regressor == original.regressor.unified_regressor
        assert restored.regressor.value_weighted == original.regressor.value_weighted

        assert restored.train_months == original.train_months
        assert restored.val_months == original.val_months
        assert restored.ev_scoring == original.ev_scoring


# ---------------------------------------------------------------------------
# GateConfig
# ---------------------------------------------------------------------------

class TestGateConfig:
    def test_gate_config_loads_json(self):
        """Load GateConfig from a temp JSON file, verify fields."""
        data = {
            "gates": {
                "rmse_cap": 5.0,
                "mae_cap": 3.0,
            },
            "eval_months": {"start": 1, "end": 12},
            "noise_tolerance": 0.02,
            "tail_max_failures": 3,
            "cascade_stages": ["classifier", "regressor"],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            tmp_path = f.name

        try:
            cfg = GateConfig.from_json(tmp_path)
            assert cfg.gates == {"rmse_cap": 5.0, "mae_cap": 3.0}
            assert cfg.eval_months == {"start": 1, "end": 12}
            assert cfg.noise_tolerance == 0.02
            assert cfg.tail_max_failures == 3
            assert cfg.cascade_stages == ["classifier", "regressor"]
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
