"""Published output schema contract for annual signals.

This is the machine-readable counterpart to docs/contracts/output-schema.md.
It defines the expected constraint parquet schema for annual publication.
"""
from __future__ import annotations

CONSTRAINT_INDEX_COLUMN = "__index_level_0__"
SF_INDEX_COLUMN = "pnode_id"

CONSTRAINT_SIGNAL_COLUMNS = [
    "constraint_id",
    "flow_direction",
    "mean_branch_max",
    "mean_branch_max_fillna",
    "ori_mean",
    "branch_name",
    "bus_key",
    "bus_key_group",
    "mix_mean",
    "shadow_price_da",
    "density_mix_rank_value",
    "density_ori_rank_value",
    "da_rank_value",
    "rank_ori",
    "density_mix_rank",
    "rank",
    "tier",
    "shadow_sign",
    "shadow_price",
    "equipment",
    "constraint_limit",
]

REQUIRED_CONSTRAINT_NON_NULL_COLUMNS = [
    "constraint_id",
    "branch_name",
    "flow_direction",
    "shadow_sign",
    "tier",
    "bus_key",
    "constraint_limit",
]


def expected_constraint_output_columns() -> list[str]:
    """Return the full ordered constraint output schema."""
    return CONSTRAINT_SIGNAL_COLUMNS + [CONSTRAINT_INDEX_COLUMN]
