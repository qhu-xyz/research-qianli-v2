"""Centralized path helpers for registry and holdout directories.

All experiment results live at:
    registry/{period_type}/{class_type}/{version_id}/
    holdout/{period_type}/{class_type}/{version_id}/

Each (period_type, class_type) slice has its own gates.json and champion.json.
"""
from __future__ import annotations

from pathlib import Path

_DEFAULT_PERIOD_TYPE = "f0"
_DEFAULT_CLASS_TYPE = "onpeak"


def registry_root(
    period_type: str = _DEFAULT_PERIOD_TYPE,
    class_type: str = _DEFAULT_CLASS_TYPE,
    base_dir: Path | str | None = None,
) -> Path:
    """Return registry/{period_type}/{class_type}."""
    base = Path(base_dir) if base_dir is not None else Path("registry")
    return base / period_type / class_type


def holdout_root(
    period_type: str = _DEFAULT_PERIOD_TYPE,
    class_type: str = _DEFAULT_CLASS_TYPE,
    base_dir: Path | str | None = None,
) -> Path:
    """Return holdout/{period_type}/{class_type}."""
    base = Path(base_dir) if base_dir is not None else Path("holdout")
    return base / period_type / class_type


def version_dir(
    version_id: str,
    period_type: str = _DEFAULT_PERIOD_TYPE,
    class_type: str = _DEFAULT_CLASS_TYPE,
    base_dir: Path | str | None = None,
) -> Path:
    """Return registry/{period_type}/{class_type}/{version_id}."""
    return registry_root(period_type, class_type, base_dir) / version_id


def holdout_version_dir(
    version_id: str,
    period_type: str = _DEFAULT_PERIOD_TYPE,
    class_type: str = _DEFAULT_CLASS_TYPE,
    base_dir: Path | str | None = None,
) -> Path:
    """Return holdout/{period_type}/{class_type}/{version_id}."""
    return holdout_root(period_type, class_type, base_dir) / version_id


def gates_path(
    period_type: str = _DEFAULT_PERIOD_TYPE,
    class_type: str = _DEFAULT_CLASS_TYPE,
    base_dir: Path | str | None = None,
) -> Path:
    """Return registry/{period_type}/{class_type}/gates.json."""
    return registry_root(period_type, class_type, base_dir) / "gates.json"


def champion_path(
    period_type: str = _DEFAULT_PERIOD_TYPE,
    class_type: str = _DEFAULT_CLASS_TYPE,
    base_dir: Path | str | None = None,
) -> Path:
    """Return registry/{period_type}/{class_type}/champion.json."""
    return registry_root(period_type, class_type, base_dir) / "champion.json"
