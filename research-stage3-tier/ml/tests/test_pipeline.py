"""Tests for ml.pipeline — main pipeline orchestration.

TDD RED phase: these tests should FAIL until ml/pipeline.py is implemented.
All tests run with SMOKE_TEST=true to use synthetic data.
"""
from __future__ import annotations

import os

import pytest

from ml.config import PipelineConfig, RegressorConfig
from ml.pipeline import run_pipeline


@pytest.fixture(autouse=True)
def _set_smoke_test(monkeypatch):
    """Ensure SMOKE_TEST=true for every test in this module."""
    monkeypatch.setenv("SMOKE_TEST", "true")


# ── Gate metric keys that must always be present ─────────────────────────

_REQUIRED_GATE_KEYS = {
    "EV-VC@100",
    "EV-VC@500",
    "EV-NDCG",
    "Spearman",
    "C-RMSE",
    "C-MAE",
}


class TestRunPipelineSmoke:
    """Smoke test: pipeline runs end-to-end and returns correct structure."""

    def test_run_pipeline_smoke(self):
        cfg = PipelineConfig()
        result = run_pipeline(
            config=cfg,
            version_id="v_test_smoke",
            auction_month="2021-07",
            class_type="onpeak",
            period_type="f0",
        )
        assert isinstance(result, dict)
        assert "metrics" in result
        assert "threshold" in result
        assert isinstance(result["metrics"], dict)
        assert isinstance(result["threshold"], float)


class TestPipelineMetricsKeys:
    """Verify all gate metric keys are present in the returned metrics."""

    def test_pipeline_metrics_keys(self):
        cfg = PipelineConfig()
        result = run_pipeline(
            config=cfg,
            version_id="v_test_keys",
            auction_month="2021-07",
            class_type="onpeak",
            period_type="f0",
        )
        metrics = result["metrics"]
        for key in _REQUIRED_GATE_KEYS:
            assert key in metrics, f"Missing metric key: {key}"
            assert isinstance(metrics[key], (int, float)), (
                f"Metric {key} should be numeric, got {type(metrics[key])}"
            )


class TestPipelineGatedMode:
    """Default (gated) mode: unified_regressor=False."""

    def test_pipeline_gated_mode(self):
        cfg = PipelineConfig()
        # Default: cfg.regressor.unified_regressor == False
        assert cfg.regressor.unified_regressor is False
        result = run_pipeline(
            config=cfg,
            version_id="v_test_gated",
            auction_month="2021-07",
            class_type="onpeak",
            period_type="f0",
        )
        assert "metrics" in result
        assert "threshold" in result
        # Threshold should be a valid float in [0, 1]
        assert 0.0 <= result["threshold"] <= 1.0


class TestPipelineUnifiedMode:
    """Unified regressor mode: unified_regressor=True."""

    def test_pipeline_unified_mode(self):
        reg_cfg = RegressorConfig(unified_regressor=True)
        cfg = PipelineConfig(regressor=reg_cfg)
        assert cfg.regressor.unified_regressor is True
        result = run_pipeline(
            config=cfg,
            version_id="v_test_unified",
            auction_month="2021-07",
            class_type="onpeak",
            period_type="f0",
        )
        assert "metrics" in result
        assert "threshold" in result
        # All gate keys present
        for key in _REQUIRED_GATE_KEYS:
            assert key in result["metrics"], f"Missing metric key: {key}"
