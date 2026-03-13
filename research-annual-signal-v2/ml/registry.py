"""Experiment registry — save config + metrics to disk."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ml.config import REGISTRY_DIR

logger = logging.getLogger(__name__)


def save_experiment(version_id: str, config: dict, metrics: dict) -> Path:
    """Save experiment results to registry/{version_id}/.

    Creates:
      - registry/{version_id}/config.json
      - registry/{version_id}/metrics.json

    Returns path to version directory.
    """
    version_dir = REGISTRY_DIR / version_id
    version_dir.mkdir(parents=True, exist_ok=True)

    config_path = version_dir / "config.json"
    metrics_path = version_dir / "metrics.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info("Saved experiment %s to %s", version_id, version_dir)
    return version_dir


def load_metrics(version_id: str) -> dict:
    """Load metrics for a version."""
    metrics_path = REGISTRY_DIR / version_id / "metrics.json"
    assert metrics_path.exists(), f"No metrics for {version_id} at {metrics_path}"
    with open(metrics_path) as f:
        return json.load(f)


def load_config(version_id: str) -> dict:
    """Load config for a version."""
    config_path = REGISTRY_DIR / version_id / "config.json"
    assert config_path.exists(), f"No config for {version_id} at {config_path}"
    with open(config_path) as f:
        return json.load(f)
