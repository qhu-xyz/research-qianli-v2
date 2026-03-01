import json

import pytest


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
