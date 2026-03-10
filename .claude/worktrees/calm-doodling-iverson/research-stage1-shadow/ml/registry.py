"""Version registry management for shadow price classification pipeline.

Handles version allocation, registration, and promotion.
"""

import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path


def allocate_version_id(counter_path: str | Path) -> str:
    """Atomically allocate the next version ID using flock.

    Format: v0001, v0002, v0003, ...

    Parameters
    ----------
    counter_path : str or Path
        Path to version_counter.json file.

    Returns
    -------
    version_id : str
        Allocated version identifier (e.g., "v0001").
    """
    counter_path = Path(counter_path)
    with open(counter_path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            data = json.load(f)
            next_id = data["next_id"]
            version_id = f"v{next_id:04d}"
            data["next_id"] = next_id + 1
            f.seek(0)
            f.truncate()
            json.dump(data, f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return version_id


def register_version(
    registry_dir: str | Path,
    version_id: str,
    config: dict,
    metrics: dict,
    meta: dict,
    model_path: str | Path | None = None,
) -> Path:
    """Register a new version in the registry.

    Creates registry/{version_id}/ with config.json, metrics.json, meta.json.
    If model_path is provided, copies model files to registry/{version_id}/model/.

    Parameters
    ----------
    registry_dir : str or Path
        Path to registry directory.
    version_id : str
        Version identifier (e.g., "v0001").
    config : dict
        Configuration used for this version.
    metrics : dict
        Evaluation metrics.
    meta : dict
        Additional metadata (creation time, etc.).
    model_path : str or Path or None
        Path to model file(s) to copy into the registry.

    Returns
    -------
    version_dir : Path
        Path to the created version directory.
    """
    registry_dir = Path(registry_dir)
    version_dir = registry_dir / version_id
    version_dir.mkdir(parents=True, exist_ok=False)

    with open(version_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(version_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    meta.setdefault("registered_at", datetime.now(timezone.utc).isoformat())
    with open(version_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if model_path is not None:
        import gzip
        import shutil

        model_path = Path(model_path)
        model_dir = version_dir / "model"
        model_dir.mkdir()

        if model_path.is_dir():
            for src_file in model_path.iterdir():
                dst = model_dir / src_file.name
                shutil.copy2(src_file, dst)
                if dst.suffix == ".ubj":
                    with open(dst, "rb") as f_in:
                        with gzip.open(str(dst) + ".gz", "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    dst.unlink()
        elif model_path.is_file():
            dst = model_dir / model_path.name
            shutil.copy2(model_path, dst)
            if dst.suffix == ".ubj":
                with open(dst, "rb") as f_in:
                    with gzip.open(str(dst) + ".gz", "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                dst.unlink()

    return version_dir


def promote_version(
    registry_dir: str | Path,
    version_id: str,
    champion_path: str | Path,
) -> None:
    """Promote a version to champion.

    Updates champion.json with the new version and timestamp.

    Parameters
    ----------
    registry_dir : str or Path
        Path to registry directory.
    version_id : str
        Version to promote.
    champion_path : str or Path
        Path to champion.json.
    """
    champion_path = Path(champion_path)
    data = {
        "version": version_id,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(champion_path, "w") as f:
        json.dump(data, f, indent=2)


def get_champion(champion_path: str | Path) -> str | None:
    """Get the current champion version ID.

    Parameters
    ----------
    champion_path : str or Path
        Path to champion.json.

    Returns
    -------
    version_id : str or None
        Current champion version ID, or None if no champion.
    """
    champion_path = Path(champion_path)
    if not champion_path.exists():
        return None
    with open(champion_path) as f:
        data = json.load(f)
    return data.get("version")
