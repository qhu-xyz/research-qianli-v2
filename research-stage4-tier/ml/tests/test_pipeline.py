"""Tests for LTR pipeline orchestration."""
import os
import pytest
from ml.config import PipelineConfig
from ml.pipeline import run_pipeline

def test_pipeline_smoke():
    """Smoke test with real V6.2B data (requires data on disk)."""
    if not os.path.exists("/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1/2021-07"):
        pytest.skip("V6.2B data not available")
    cfg = PipelineConfig()
    # Use small n_estimators for speed
    cfg.ltr.n_estimators = 10
    cfg.ltr.early_stopping_rounds = 5
    result = run_pipeline(cfg, "test", "2021-07")
    metrics = result["metrics"]
    assert "VC@100" in metrics
    assert "Recall@100" in metrics
    assert "NDCG" in metrics
    assert metrics["n_samples"] > 400
