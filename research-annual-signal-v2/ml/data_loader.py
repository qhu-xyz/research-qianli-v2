"""Compatibility shim — real implementation moved to ml.markets.miso.data_loader."""
from ml.markets.miso.data_loader import (  # noqa: F401
    load_collapsed,
    load_cid_mapping,
    load_raw_density,
    load_constraint_limits,
    compute_right_tail_max,
    _load_limits,
    _cid_mapping_cache_path,
    COLLAPSED_CACHE_DIR,
)
