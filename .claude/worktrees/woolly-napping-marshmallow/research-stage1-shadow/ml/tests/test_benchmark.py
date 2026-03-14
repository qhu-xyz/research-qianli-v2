import json
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from ml.config import FeatureConfig


@pytest.fixture(autouse=True)
def smoke_mode(monkeypatch):
    monkeypatch.setenv("SMOKE_TEST", "true")


def test_benchmark_smoke_single_month(tmp_path):
    """Benchmark in smoke mode evaluates a single month and writes metrics."""
    from ml.benchmark import run_benchmark

    reg = tmp_path / "registry"
    reg.mkdir()
    (reg / "version_counter.json").write_text('{"next_id": 1}')

    result = run_benchmark(
        version_id="v0",
        eval_months=["2021-07"],
        class_type="onpeak",
        ptype="f0",
        registry_dir=str(reg),
    )

    assert "per_month" in result
    assert "2021-07" in result["per_month"]
    assert "aggregate" in result
    assert "S1-AUC" in result["aggregate"]["mean"]
    assert (reg / "v0" / "metrics.json").exists()


def test_benchmark_smoke_multi_month(tmp_path):
    """Benchmark with multiple months aggregates correctly."""
    from ml.benchmark import run_benchmark

    reg = tmp_path / "registry"
    reg.mkdir()
    (reg / "version_counter.json").write_text('{"next_id": 1}')

    result = run_benchmark(
        version_id="v0",
        eval_months=["2021-07", "2021-08"],
        class_type="onpeak",
        ptype="f0",
        registry_dir=str(reg),
    )

    assert len(result["per_month"]) == 2
    assert result["n_months"] == 2
    assert "bottom_2_mean" in result["aggregate"]


def test_benchmark_writes_config_and_meta(tmp_path):
    """Benchmark writes config.json and meta.json alongside metrics."""
    from ml.benchmark import run_benchmark

    reg = tmp_path / "registry"
    reg.mkdir()

    run_benchmark(
        version_id="v_test",
        eval_months=["2021-07"],
        class_type="onpeak",
        ptype="f0",
        registry_dir=str(reg),
    )

    assert (reg / "v_test" / "config.json").exists()
    assert (reg / "v_test" / "meta.json").exists()

    with open(reg / "v_test" / "meta.json") as f:
        meta = json.load(f)
    assert meta["n_months"] == 1
    assert meta["version_id"] == "v_test"


def _make_df(n: int, rng=None):
    """Helper to create a synthetic DataFrame with n rows."""
    if rng is None:
        rng = np.random.RandomState(42)
    fc = FeatureConfig()
    data = {feat: rng.randn(n).tolist() for feat in fc.features}
    binding = (rng.random(n) < 0.07).astype(bool)
    data["actual_shadow_price"] = np.where(binding, rng.lognormal(3, 1.5, size=n), 0.0).tolist()
    data["constraint_id"] = [f"C{i:04d}" for i in range(n)]
    data["auction_month"] = ["2021-07"] * n
    return pl.DataFrame(data)


def test_benchmark_skips_empty_val_month(tmp_path):
    """Months with empty validation sets are skipped, not crashed."""
    from ml.benchmark import run_benchmark

    good_train, good_val = _make_df(80), _make_df(20)
    empty_val = _make_df(0)  # 0-row DataFrame
    empty_train = _make_df(80)

    call_count = [0]
    def mock_load_data(config):
        call_count[0] += 1
        if config.auction_month == "2021-08":
            return empty_train, empty_val
        return good_train, good_val

    reg = tmp_path / "registry"
    reg.mkdir()

    with patch("ml.benchmark.load_data", side_effect=mock_load_data):
        result = run_benchmark(
            version_id="v_skip",
            eval_months=["2021-07", "2021-08"],
            class_type="onpeak",
            ptype="f0",
            registry_dir=str(reg),
        )

    assert result["n_months"] == 1
    assert "2021-07" in result["per_month"]
    assert "2021-08" not in result["per_month"]
    assert result["skipped_months"] == ["2021-08"]
    assert call_count[0] == 2
