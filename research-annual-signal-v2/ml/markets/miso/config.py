"""MISO-specific constants, paths, and eval splits for annual signal pipeline."""
from __future__ import annotations

from pathlib import Path

# ─── Data paths (MISO NFS) ───────────────────────────────────────────
DENSITY_PATH = "/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_DENSITY_DISTRIBUTION.parquet"
BRIDGE_PATH = "/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_INFO.parquet"
LIMIT_PATH = "/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_LIMIT.parquet"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DA_CACHE_DIR = PROJECT_ROOT / "data" / "realized_da"
COLLAPSED_CACHE_DIR = PROJECT_ROOT / "data" / "collapsed"
REGISTRY_DIR = PROJECT_ROOT / "registry"
