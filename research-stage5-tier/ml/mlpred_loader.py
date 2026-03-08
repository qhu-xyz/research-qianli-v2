"""Load ml_pred features for a single auction month.

Keeps: constraint_id, flow_direction, predicted_shadow_price,
       binding_probability, binding_probability_scaled.

Drops leaky columns (actual_shadow_price, actual_binding, error, etc.)
and prob_exceed_* (already loaded from spice6 density).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from ml.config import SPICE6_MLPRED_BASE

_KEEP_COLUMNS = [
    "constraint_id",
    "flow_direction",
    "predicted_shadow_price",
    "binding_probability",
    "binding_probability_scaled",
]


def load_mlpred(
    auction_month: str,
    class_type: str = "onpeak",
) -> pl.DataFrame:
    """Load ml_pred features for one auction month.

    Parameters
    ----------
    auction_month : str
        Month in YYYY-MM format.
    class_type : str
        "onpeak" or "offpeak".

    Returns
    -------
    pl.DataFrame
        Columns: constraint_id, flow_direction, predicted_shadow_price,
        binding_probability, binding_probability_scaled.
        Empty DataFrame if path doesn't exist.
    """
    path = (
        Path(SPICE6_MLPRED_BASE)
        / f"auction_month={auction_month}"
        / f"market_month={auction_month}"
        / f"class_type={class_type}"
        / "final_results.parquet"
    )

    if not path.exists():
        return pl.DataFrame(
            schema={
                "constraint_id": pl.String,
                "flow_direction": pl.Int64,
                "predicted_shadow_price": pl.Float64,
                "binding_probability": pl.Float64,
                "binding_probability_scaled": pl.Float64,
            }
        )

    df = pl.read_parquet(str(path))

    # Keep only non-leaky columns
    available = [c for c in _KEEP_COLUMNS if c in df.columns]
    df = df.select(available)

    return df
