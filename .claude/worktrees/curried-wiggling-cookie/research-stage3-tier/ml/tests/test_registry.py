"""Tests for ml.registry — version allocation, registration, promotion."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ml.registry import allocate_version_id, get_champion, promote_version, register_version


class TestAllocateVersionId:
    def test_allocate_two_sequential(self, tmp_path):
        """Allocate two IDs, verify they are sequential."""
        counter_path = tmp_path / "version_counter.json"
        counter_path.write_text(json.dumps({"next_id": 1}))

        v1 = allocate_version_id(counter_path)
        v2 = allocate_version_id(counter_path)

        assert v1 == "v0001"
        assert v2 == "v0002"

        # Counter should now be at 3
        data = json.loads(counter_path.read_text())
        assert data["next_id"] == 3

    def test_allocate_preserves_padding(self, tmp_path):
        """Verify 4-digit zero-padded format."""
        counter_path = tmp_path / "version_counter.json"
        counter_path.write_text(json.dumps({"next_id": 99}))

        v = allocate_version_id(counter_path)
        assert v == "v0099"


class TestRegisterVersion:
    def test_register_creates_files(self, tmp_path):
        """Register a version, verify config/metrics/meta files exist."""
        registry_dir = tmp_path / "registry"
        registry_dir.mkdir()

        config = {"tier": {"n_estimators": 400}}
        metrics = {"Tier-VC@100": 0.85, "Tier-NDCG": 0.90}
        meta = {"version_id": "v0001"}

        version_dir = register_version(
            registry_dir=registry_dir,
            version_id="v0001",
            config=config,
            metrics=metrics,
            meta=meta,
        )

        assert version_dir.exists()
        assert (version_dir / "config.json").exists()
        assert (version_dir / "metrics.json").exists()
        assert (version_dir / "meta.json").exists()

        # Verify content
        loaded_config = json.loads((version_dir / "config.json").read_text())
        assert loaded_config["tier"]["n_estimators"] == 400

        loaded_metrics = json.loads((version_dir / "metrics.json").read_text())
        assert loaded_metrics["Tier-VC@100"] == 0.85

        loaded_meta = json.loads((version_dir / "meta.json").read_text())
        assert "registered_at" in loaded_meta

    def test_register_duplicate_fails(self, tmp_path):
        """Registering the same version ID twice raises FileExistsError."""
        registry_dir = tmp_path / "registry"
        registry_dir.mkdir()

        register_version(registry_dir, "v0001", {}, {}, {})
        with pytest.raises(FileExistsError):
            register_version(registry_dir, "v0001", {}, {}, {})


class TestPromoteAndGetChampion:
    def test_promote_and_get(self, tmp_path):
        """Promote a version and verify get_champion returns it."""
        champion_path = tmp_path / "champion.json"
        registry_dir = tmp_path / "registry"
        registry_dir.mkdir()

        # Initially no champion
        assert get_champion(champion_path) is None

        # Promote v0001
        promote_version(registry_dir, "v0001", champion_path)
        assert get_champion(champion_path) == "v0001"

        # Promote v0002 overwrites
        promote_version(registry_dir, "v0002", champion_path)
        assert get_champion(champion_path) == "v0002"
