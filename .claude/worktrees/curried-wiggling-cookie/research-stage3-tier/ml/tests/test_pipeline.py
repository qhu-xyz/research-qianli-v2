"""Tests for ml.pipeline — main tier pipeline orchestration.

All tests run with SMOKE_TEST=true to use synthetic data.
"""
from __future__ import annotations

import os

import pytest

from ml.config import PipelineConfig, TierConfig
from ml.pipeline import run_pipeline


@pytest.fixture(autouse=True)
def _set_smoke_test(monkeypatch):
    """Ensure SMOKE_TEST=true for every test in this module."""
    monkeypatch.setenv("SMOKE_TEST", "true")


# ── Gate metric keys that must always be present ─────────────────────────

_REQUIRED_GATE_KEYS = {
    # Group A (blocking)
    "Tier-VC@100",
    "Tier-VC@500",
    "Tier0-AP",
    "Tier01-AP",
    # Group B (monitor)
    "Tier-NDCG",
    "QWK",
    "Macro-F1",
    "Value-QWK",
    "Tier-Recall@0",
    "Tier-Recall@1",
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
        assert isinstance(result["metrics"], dict)


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
