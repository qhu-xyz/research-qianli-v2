"""Tests for registry path helpers."""
from pathlib import Path
import pytest
from ml.registry_paths import (
    registry_root, holdout_root, version_dir, holdout_version_dir,
    gates_path, champion_path,
)


def test_registry_root_defaults():
    assert registry_root() == Path("registry/f0/onpeak")


def test_registry_root_offpeak_f1():
    assert registry_root(period_type="f1", class_type="offpeak") == Path("registry/f1/offpeak")


def test_holdout_root_defaults():
    assert holdout_root() == Path("holdout/f0/onpeak")


def test_version_dir():
    assert version_dir("v10e-lag1") == Path("registry/f0/onpeak/v10e-lag1")


def test_holdout_version_dir():
    assert holdout_version_dir("v0", period_type="f0", class_type="offpeak") == Path("holdout/f0/offpeak/v0")


def test_gates_path():
    assert gates_path() == Path("registry/f0/onpeak/gates.json")


def test_champion_path():
    assert champion_path(period_type="f1", class_type="offpeak") == Path("registry/f1/offpeak/champion.json")


def test_version_dir_custom_base(tmp_path):
    p = version_dir("v0", period_type="f0", class_type="onpeak", base_dir=tmp_path)
    assert p == tmp_path / "f0" / "onpeak" / "v0"
