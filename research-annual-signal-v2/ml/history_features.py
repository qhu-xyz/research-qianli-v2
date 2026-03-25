"""Compatibility shim — real implementation moved to ml.markets.miso.history_features."""
from ml.markets.miso.history_features import (  # noqa: F401
    compute_history_features,
    build_monthly_binding_table,
    _generate_month_range,
)
