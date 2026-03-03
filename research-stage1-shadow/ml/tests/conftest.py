import numpy as np
import pytest


@pytest.fixture
def synthetic_features():
    rng = np.random.RandomState(42)
    return rng.randn(100, 13)


@pytest.fixture
def synthetic_labels():
    rng = np.random.RandomState(42)
    return (rng.random(100) < 0.07).astype(int)


@pytest.fixture
def mock_pipeline_config(tmp_path):
    from ml.config import PipelineConfig

    return PipelineConfig(version_id="v0001", registry_dir=str(tmp_path / "registry"))


@pytest.fixture
def tmp_registry(tmp_path):
    reg = tmp_path / "registry"
    reg.mkdir()
    v0 = reg / "v0"
    v0.mkdir()
    (v0 / "metrics.json").write_text(
        '{"S1-AUC": 0.75, "S1-AP": 0.15, "S1-BRIER": 0.089}'
    )
    (reg / "version_counter.json").write_text('{"next_id": 1}')
    (reg / "champion.json").write_text('{"version": null, "promoted_at": null}')
    return reg
