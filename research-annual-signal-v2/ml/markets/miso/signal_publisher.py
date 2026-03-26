"""M3: Annual signal publisher — produces V7.0 constraint + SF parquets.

Implements the 7-step pipeline from docs/pipeline-and-production-port.md §5.
All metadata derived from pbase/psignal sources. No silent fallbacks — raises
on unexpected nulls per CLAUDE.md code quality rules.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl

from ml.markets.miso.config import get_market_months
from ml.markets.miso.data_loader import load_cid_mapping, load_constraint_limits
from ml.markets.miso.features import build_class_publish_table
from ml.markets.miso.release_candidate import (
    BASE_PER_TIER,
    SPECIALIST_PER_TIER,
    score_publish_branches_72b,
)
from ml.markets.miso.scoring import score_v0c
from ml.products.annual.output_schema import (
    CONSTRAINT_INDEX_COLUMN,
    REQUIRED_CONSTRAINT_NON_NULL_COLUMNS,
    expected_constraint_output_columns,
)

logger = logging.getLogger(__name__)

DENSITY_SCORE_PATH = "/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_DENSITY_SIGNAL_SCORE.parquet"
SF_PATH = "/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_SF.parquet"

def publish_signal(
    planning_year: str,
    aq_quarter: str,
    class_type: str,
    market_round: int,
    tier_sizes: list[int],
    branch_cap: int = 3,
    chebyshev_threshold: float = 0.05,
    correlation_threshold: float = -0.21,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (constraints_df, sf_df) for one (PY, aq, class_type, round)."""
    bf_col = "bf_12" if class_type == "onpeak" else "bfo_12"
    total_target = sum(tier_sizes)

    # ── Step 1: Score branches ────────────────────────────────────────
    logger.info("Step 1: Scoring %s/%s/%s", planning_year, aq_quarter, class_type)
    model_table = build_class_publish_table(planning_year, aq_quarter, class_type, market_round=market_round)
    scores = score_v0c(model_table, bf_col)
    model_table = model_table.with_columns(pl.Series("v0c_score", scores))

    # ── Step 2: Expand branch → constraints ───────────────────────────
    logger.info("Step 2: Expanding to constraints")
    cid_map = load_cid_mapping(planning_year, aq_quarter, market_round=market_round)
    cid_limits = load_constraint_limits(planning_year, aq_quarter, market_round=market_round)
    branch_scores = model_table.select(
        ["branch_name", "v0c_score", "shadow_price_da", "da_rank_value"]
    ).unique(subset=["branch_name"])
    constraints = (
        cid_map
        .join(branch_scores, on="branch_name", how="left")
        .join(cid_limits, on="constraint_id", how="left")
    )
    n_before = len(constraints)
    constraints = constraints.filter(pl.col("v0c_score").is_not_null())
    n_null_limit = constraints.filter(pl.col("constraint_limit").is_null()).height
    if n_null_limit > 0:
        raise ValueError(
            f"{n_null_limit} constraints missing constraint_limit. "
            "Cannot publish without constraint_limit in the output schema."
        )
    logger.info("  %d constraints (%d dropped)", len(constraints), n_before - len(constraints))

    # ── Step 3: Join metadata ─────────────────────────────────────────
    logger.info("Step 3: Metadata")

    # 3a. flow_direction from MISO_SPICE_DENSITY_SIGNAL_SCORE.
    # Each CID has rows for both directions (+1, -1). We pick the direction with
    # the highest density signal score. This determines shadow_sign (= -flow_direction)
    # and therefore whether the published constraint is a buy or sell signal.
    # Confirmed: flow_direction SHOULD come from density_signal_score (teammate verified).
    flow_dir = _load_flow_direction(planning_year, market_round=market_round)
    constraints = constraints.join(flow_dir, on="constraint_id", how="left")
    n_null_fd = constraints.filter(pl.col("flow_direction").is_null()).height
    if n_null_fd > 0:
        raise ValueError(
            f"{n_null_fd} constraints missing flow_direction from density score. "
            "Cannot default — flow_direction determines shadow_sign and exposure direction."
        )

    # 3b. bus_key from pbase branches
    bus_key_df = _load_bus_key(planning_year, aq_quarter, class_type, market_round=market_round)
    constraints = constraints.join(bus_key_df, on="branch_name", how="left")
    n_null_bk = constraints.filter(pl.col("bus_key").is_null()).height
    if n_null_bk > 0:
        logger.warning("  %d constraints missing bus_key — using branch_name", n_null_bk)
        constraints = constraints.with_columns(
            pl.when(pl.col("bus_key").is_null())
            .then(pl.col("branch_name"))
            .otherwise(pl.col("bus_key"))
            .alias("bus_key")
        )

    # 3c. bus_key_group via union-find on bus pairs
    constraints = _compute_bus_key_group(constraints)

    # 3d. Density features (ori_mean, mix_mean, mean_branch_max)
    density_score = _load_density_score_branch(planning_year, flow_dir, constraints, market_round=market_round)
    constraints = constraints.join(density_score, on="branch_name", how="left")
    for col in ["ori_mean", "mix_mean", "mean_branch_max"]:
        n_null = constraints.filter(pl.col(col).is_null()).height
        if n_null > 0:
            logger.warning("  %d constraints missing %s — filling 0.0 (no density score)", n_null, col)
        constraints = constraints.with_columns(pl.col(col).fill_null(0.0))
    constraints = constraints.with_columns(pl.col("mean_branch_max").alias("mean_branch_max_fillna"))

    # 3e. Derived columns
    constraints = constraints.with_columns(
        (-pl.col("flow_direction")).alias("shadow_sign"),
    )
    constraints = constraints.with_columns(
        pl.col("shadow_sign").cast(pl.Float64).alias("shadow_price"),
        pl.col("branch_name").alias("equipment"),
    )

    # ── Step 4: Ranks ─────────────────────────────────────────────────
    logger.info("Step 4: Ranks")
    constraints = constraints.with_columns(pl.col("v0c_score").alias("rank"))

    for src, dst in [("mix_mean", "density_mix_rank_value"),
                     ("mix_mean", "density_mix_rank"),
                     ("ori_mean", "density_ori_rank_value")]:
        constraints = constraints.with_columns(
            pl.col(src).rank(method="dense", descending=True).cast(pl.Float64)
            .truediv(pl.col(src).count()).alias(dst)
        )
    constraints = constraints.with_columns(
        pl.col("da_rank_value").rank(method="dense").cast(pl.Float64)
        .truediv(pl.col("da_rank_value").count()).alias("da_rank_value")
    )
    constraints = constraints.with_columns(
        (0.60 * pl.col("da_rank_value")
         + 0.30 * pl.col("density_mix_rank_value")
         + 0.10 * pl.col("density_ori_rank_value")).alias("rank_ori")
    )

    # ── Step 5: Build SF ──────────────────────────────────────────────
    logger.info("Step 5: SF matrix")
    market_months = get_market_months(planning_year, aq_quarter)
    sf_raw = _load_sf(planning_year, market_months, market_round=market_round)
    logger.info("  SF raw: %d pnodes × %d constraints", sf_raw.shape[0], sf_raw.shape[1] - 1)

    # ── Step 6: Grouped dedup (walk-and-fill within bus_key_group) ────
    logger.info("Step 6: Grouped dedup (target=%d, branch_cap=%d)", total_target, branch_cap)
    constraints_sorted = constraints.sort("v0c_score", descending=True)

    sf_pd = sf_raw.to_pandas().set_index("pnode_id")

    selected_indices = []
    branch_counts: dict[str, int] = {}
    # Track selected SF vectors PER bus_key_group for within-group dedup
    group_sf_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    tier_idx = 0
    tier_count = 0

    cids = constraints_sorted["constraint_id"].to_list()
    branches = constraints_sorted["branch_name"].to_list()
    groups = constraints_sorted["bus_key_group"].to_list()

    for i in range(len(cids)):
        if tier_idx >= len(tier_sizes):
            break

        cid = cids[i]
        branch = branches[i]
        group = groups[i]

        # Branch cap (within group, matching psignal)
        if branch_counts.get(branch, 0) >= branch_cap:
            continue

        # Skip all-zero SF constraints (no price impact on any pnode)
        if cid in sf_pd.columns and sf_pd[cid].abs().sum() == 0:
            continue

        # SF similarity check WITHIN bus_key_group only
        if cid in sf_pd.columns:
            candidate_sf = sf_pd[cid].values
            skip = False
            for prev_sf in group_sf_vectors[group]:
                if _chebyshev_distance(candidate_sf, prev_sf) < chebyshev_threshold:
                    skip = True
                    break
                corr = _correlation(candidate_sf, prev_sf)
                if corr is not None and corr > correlation_threshold:
                    skip = True
                    break
            if skip:
                continue
            group_sf_vectors[group].append(candidate_sf)
        else:
            group_sf_vectors[group].append(np.zeros(len(sf_pd)))

        selected_indices.append(i)
        branch_counts[branch] = branch_counts.get(branch, 0) + 1
        tier_count += 1

        if tier_count >= tier_sizes[tier_idx]:
            tier_idx += 1
            tier_count = 0

    logger.info("  Selected %d across %d tiers", len(selected_indices), min(tier_idx + 1, len(tier_sizes)))

    # Assign tiers
    published = constraints_sorted[selected_indices]
    tiers = []
    t_idx, t_count = 0, 0
    for _ in range(len(selected_indices)):
        tiers.append(t_idx)
        t_count += 1
        if t_idx < len(tier_sizes) and t_count >= tier_sizes[t_idx]:
            t_idx += 1
            t_count = 0
    published = published.with_columns(pl.Series("tier", tiers).cast(pl.Int64))

    # ── Step 7: Validate + format ─────────────────────────────────────
    logger.info("Step 7: Validate")
    published = published.with_columns(
        (pl.col("constraint_id") + "|" + pl.col("shadow_sign").cast(pl.Utf8) + "|spice")
        .alias(CONSTRAINT_INDEX_COLUMN)
    )

    out_cols = expected_constraint_output_columns()
    for col in out_cols:
        if col not in published.columns:
            raise ValueError(f"Missing column: {col}")

    constraints_out = published.select(out_cols).to_pandas()

    # No-null check on critical columns
    for col in REQUIRED_CONSTRAINT_NON_NULL_COLUMNS:
        n_null = constraints_out[col].isna().sum()
        if n_null > 0:
            raise ValueError(f"Column {col} has {n_null} nulls")

    # SF output: subset + rename to pipe format
    pub_cids = published["constraint_id"].to_list()
    pub_signs = published["shadow_sign"].to_list()
    sf_cid_cols = [c for c in pub_cids if c in sf_pd.columns]
    missing_sf = set(pub_cids) - set(sf_pd.columns)
    if missing_sf:
        raise ValueError(
            f"{len(missing_sf)} published constraints missing SF coverage. "
            f"Cannot publish without SF. First 5: {sorted(missing_sf)[:5]}"
        )

    cid_to_sign = dict(zip(pub_cids, pub_signs))
    sf_out = sf_pd[sf_cid_cols].copy()
    sf_out = sf_out.rename(columns={c: f"{c}|{int(cid_to_sign[c])}|spice" for c in sf_cid_cols})
    sf_out = sf_out.reset_index()

    logger.info("  Constraints: %d × %d, SF: %d × %d",
                len(constraints_out), len(constraints_out.columns),
                len(sf_out), len(sf_out.columns) - 1)

    return constraints_out, sf_out


def publish_signal_72b(
    planning_year: str,
    aq_quarter: str,
    class_type: str,
    market_round: int,
    tier_sizes: list[int],
    branch_cap: int = 3,
    chebyshev_threshold: float = 0.05,
    correlation_threshold: float = -0.21,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build 7.2b publish outputs with constraint-level 150/50 tier assembly."""
    total_target = sum(tier_sizes)

    # ── Step 1: Score branches with 7.2b publish adapter ──────────────
    logger.info("Step 1: 7.2b branch scoring %s/%s/%s", planning_year, aq_quarter, class_type)
    model_table = score_publish_branches_72b(
        planning_year=planning_year,
        aq=aq_quarter,
        ct=class_type,
        market_round=market_round,
    )

    # ── Step 2: Expand branch → constraints ───────────────────────────
    logger.info("Step 2: Expanding 7.2b branches to constraints")
    cid_map = load_cid_mapping(planning_year, aq_quarter, market_round=market_round)
    cid_limits = load_constraint_limits(planning_year, aq_quarter, market_round=market_round)
    branch_scores = model_table.select(
        [
            "branch_name",
            "base_score",
            "specialist_score",
            "origin",
            "shadow_price_da",
            "da_rank_value",
        ]
    ).unique(subset=["branch_name"])
    constraints = (
        cid_map
        .join(branch_scores, on="branch_name", how="left")
        .join(cid_limits, on="constraint_id", how="left")
        .with_row_count("candidate_idx")
    )
    n_before = len(constraints)
    constraints = constraints.filter(pl.col("base_score").is_not_null())
    n_null_limit = constraints.filter(pl.col("constraint_limit").is_null()).height
    if n_null_limit > 0:
        raise ValueError(
            f"{n_null_limit} constraints missing constraint_limit. "
            "Cannot publish without constraint_limit in the output schema."
        )
    logger.info("  %d constraints (%d dropped)", len(constraints), n_before - len(constraints))

    # ── Step 3: Join metadata ─────────────────────────────────────────
    logger.info("Step 3: Metadata")
    flow_dir = _load_flow_direction(planning_year, market_round=market_round)
    constraints = constraints.join(flow_dir, on="constraint_id", how="left")
    n_null_fd = constraints.filter(pl.col("flow_direction").is_null()).height
    if n_null_fd > 0:
        raise ValueError(
            f"{n_null_fd} constraints missing flow_direction from density score. "
            "Cannot default — flow_direction determines shadow_sign and exposure direction."
        )

    bus_key_df = _load_bus_key(planning_year, aq_quarter, class_type, market_round=market_round)
    constraints = constraints.join(bus_key_df, on="branch_name", how="left")
    n_null_bk = constraints.filter(pl.col("bus_key").is_null()).height
    if n_null_bk > 0:
        logger.warning("  %d constraints missing bus_key — using branch_name", n_null_bk)
        constraints = constraints.with_columns(
            pl.when(pl.col("bus_key").is_null())
            .then(pl.col("branch_name"))
            .otherwise(pl.col("bus_key"))
            .alias("bus_key")
        )

    constraints = _compute_bus_key_group(constraints)

    density_score = _load_density_score_branch(planning_year, flow_dir, constraints, market_round=market_round)
    constraints = constraints.join(density_score, on="branch_name", how="left")
    for col in ["ori_mean", "mix_mean", "mean_branch_max"]:
        n_null = constraints.filter(pl.col(col).is_null()).height
        if n_null > 0:
            logger.warning("  %d constraints missing %s — filling 0.0 (no density score)", n_null, col)
        constraints = constraints.with_columns(pl.col(col).fill_null(0.0))
    constraints = constraints.with_columns(pl.col("mean_branch_max").alias("mean_branch_max_fillna"))
    constraints = constraints.with_columns(
        (-pl.col("flow_direction")).alias("shadow_sign"),
    )
    constraints = constraints.with_columns(
        pl.col("shadow_sign").cast(pl.Float64).alias("shadow_price"),
        pl.col("branch_name").alias("equipment"),
    )

    # ── Step 4: Rank features used in the published schema ────────────
    logger.info("Step 4: Rank features")
    constraints = _add_output_rank_features(constraints)

    # ── Step 5: Build SF ──────────────────────────────────────────────
    logger.info("Step 5: SF matrix")
    market_months = get_market_months(planning_year, aq_quarter)
    sf_raw = _load_sf(planning_year, market_months, market_round=market_round)
    logger.info("  SF raw: %d pnodes × %d constraints", sf_raw.shape[0], sf_raw.shape[1] - 1)
    sf_pd = sf_raw.to_pandas().set_index("pnode_id")

    # ── Step 6: 7.2b two-pool walk ────────────────────────────────────
    logger.info(
        "Step 6: 7.2b tier walk (target=%d, branch_cap=%d, specialist=%d/tier, base=%d/tier)",
        total_target, branch_cap, SPECIALIST_PER_TIER, BASE_PER_TIER,
    )
    specialist_pool = constraints.filter(pl.col("origin") == "specialist").sort(
        ["specialist_score", "base_score", "constraint_id"],
        descending=[True, True, False],
    )
    base_pool = constraints.filter(pl.col("origin") == "base").sort(
        ["base_score", "constraint_id"],
        descending=[True, False],
    )

    selection_df, audit_rows = _walk_tiers_72b(
        specialist_pool=specialist_pool,
        base_pool=base_pool,
        sf_pd=sf_pd,
        tier_sizes=tier_sizes,
        branch_cap=branch_cap,
        chebyshev_threshold=chebyshev_threshold,
        correlation_threshold=correlation_threshold,
    )

    if selection_df.height == 0:
        raise ValueError("7.2b publish walk selected zero constraints")

    published = (
        constraints.join(selection_df, on="candidate_idx", how="inner")
        .sort("publish_pos")
        .with_columns(
            (pl.lit(float(selection_df.height)) - pl.col("publish_pos").cast(pl.Float64))
            .alias("rank")
        )
    )

    for row in audit_rows:
        logger.info(
            "  Tier %d: selected=%d specialist=%d base=%d shortfall=%d rejected(branch_cap=%d sf=%d missing_sf=%d zero_sf=%d dup=%d)",
            row["tier"],
            row["selected_total"],
            row["selected_specialist"],
            row["selected_base"],
            row["specialist_shortfall"],
            row["rejected_branch_cap"],
            row["rejected_sf_dedup"],
            row["rejected_missing_sf"],
            row["rejected_zero_sf"],
            row["rejected_duplicate_cid"],
        )

    if published.height < total_target:
        logger.warning(
            "7.2b publish shortfall: selected %d of requested %d constraints",
            published.height, total_target,
        )

    # ── Step 7: Validate + format ─────────────────────────────────────
    logger.info("Step 7: Validate")
    constraints_out, sf_out = _finalize_outputs(published, sf_pd)
    logger.info("  Constraints: %d × %d, SF: %d × %d",
                len(constraints_out), len(constraints_out.columns),
                len(sf_out), len(sf_out.columns) - 1)
    return constraints_out, sf_out, audit_rows


# ── Helpers ───────────────────────────────────────────────────────────

def _add_output_rank_features(constraints: pl.DataFrame) -> pl.DataFrame:
    """Add rank-like metadata fields required by the 7.x published schema."""
    for src, dst in [
        ("mix_mean", "density_mix_rank_value"),
        ("mix_mean", "density_mix_rank"),
        ("ori_mean", "density_ori_rank_value"),
    ]:
        constraints = constraints.with_columns(
            pl.col(src).rank(method="dense", descending=True).cast(pl.Float64)
            .truediv(pl.col(src).count()).alias(dst)
        )
    constraints = constraints.with_columns(
        pl.col("da_rank_value").rank(method="dense").cast(pl.Float64)
        .truediv(pl.col("da_rank_value").count()).alias("da_rank_value")
    )
    return constraints.with_columns(
        (0.60 * pl.col("da_rank_value")
         + 0.30 * pl.col("density_mix_rank_value")
         + 0.10 * pl.col("density_ori_rank_value")).alias("rank_ori")
    )


def _walk_pool(
    rows: list[dict],
    start_idx: int,
    quota: int,
    sf_pd: pd.DataFrame,
    used_cids: set[str],
    branch_counts: dict[str, int],
    group_sf_vectors: dict[str, list[np.ndarray]],
    branch_cap: int,
    chebyshev_threshold: float,
    correlation_threshold: float,
) -> tuple[list[int], int, dict[str, int]]:
    """Sequentially consume one origin pool under shared publish-state constraints."""
    selected: list[int] = []
    stats = {
        "branch_cap": 0,
        "sf_dedup": 0,
        "missing_sf": 0,
        "zero_sf": 0,
        "duplicate_cid": 0,
    }

    idx = start_idx
    while idx < len(rows) and len(selected) < quota:
        row = rows[idx]
        idx += 1
        cid = row["constraint_id"]
        branch = row["branch_name"]
        group = row["bus_key_group"]

        if cid in used_cids:
            stats["duplicate_cid"] += 1
            continue
        if branch_counts.get(branch, 0) >= branch_cap:
            stats["branch_cap"] += 1
            continue
        if cid not in sf_pd.columns:
            stats["missing_sf"] += 1
            continue

        candidate_sf = sf_pd[cid].values
        if np.abs(candidate_sf).sum() == 0:
            stats["zero_sf"] += 1
            continue

        skip = False
        for prev_sf in group_sf_vectors[group]:
            if _chebyshev_distance(candidate_sf, prev_sf) < chebyshev_threshold:
                skip = True
                break
            corr = _correlation(candidate_sf, prev_sf)
            if corr is not None and corr > correlation_threshold:
                skip = True
                break
        if skip:
            stats["sf_dedup"] += 1
            continue

        selected.append(int(row["candidate_idx"]))
        used_cids.add(cid)
        branch_counts[branch] = branch_counts.get(branch, 0) + 1
        group_sf_vectors[group].append(candidate_sf)

    return selected, idx, stats


def _walk_tiers_72b(
    specialist_pool: pl.DataFrame,
    base_pool: pl.DataFrame,
    sf_pd: pd.DataFrame,
    tier_sizes: list[int],
    branch_cap: int,
    chebyshev_threshold: float,
    correlation_threshold: float,
    specialist_per_tier: int = SPECIALIST_PER_TIER,
) -> tuple[pl.DataFrame, list[dict[str, int]]]:
    """Constraint-level two-pool walk for 7.2b publication."""
    specialist_rows = specialist_pool.to_dicts()
    base_rows = base_pool.to_dicts()

    used_cids: set[str] = set()
    branch_counts: dict[str, int] = {}
    group_sf_vectors: dict[str, list[np.ndarray]] = defaultdict(list)

    spec_idx = 0
    base_idx = 0
    selected_entries: list[dict[str, int]] = []
    audit_rows: list[dict[str, int]] = []

    for tier, tier_size in enumerate(tier_sizes):
        specialist_target = min(specialist_per_tier, tier_size)
        base_target = max(tier_size - specialist_target, 0)

        spec_sel, spec_idx, spec_stats = _walk_pool(
            rows=specialist_rows,
            start_idx=spec_idx,
            quota=specialist_target,
            sf_pd=sf_pd,
            used_cids=used_cids,
            branch_counts=branch_counts,
            group_sf_vectors=group_sf_vectors,
            branch_cap=branch_cap,
            chebyshev_threshold=chebyshev_threshold,
            correlation_threshold=correlation_threshold,
        )

        specialist_shortfall = specialist_target - len(spec_sel)
        base_quota = base_target + specialist_shortfall

        base_sel, base_idx, base_stats = _walk_pool(
            rows=base_rows,
            start_idx=base_idx,
            quota=base_quota,
            sf_pd=sf_pd,
            used_cids=used_cids,
            branch_counts=branch_counts,
            group_sf_vectors=group_sf_vectors,
            branch_cap=branch_cap,
            chebyshev_threshold=chebyshev_threshold,
            correlation_threshold=correlation_threshold,
        )

        for candidate_idx in spec_sel + base_sel:
            selected_entries.append(
                {
                    "candidate_idx": candidate_idx,
                    "tier": tier,
                    "publish_pos": len(selected_entries),
                }
            )

        audit_rows.append(
            {
                "tier": tier,
                "selected_total": len(spec_sel) + len(base_sel),
                "selected_specialist": len(spec_sel),
                "selected_base": len(base_sel),
                "specialist_shortfall": specialist_shortfall,
                "rejected_branch_cap": spec_stats["branch_cap"] + base_stats["branch_cap"],
                "rejected_sf_dedup": spec_stats["sf_dedup"] + base_stats["sf_dedup"],
                "rejected_missing_sf": spec_stats["missing_sf"] + base_stats["missing_sf"],
                "rejected_zero_sf": spec_stats["zero_sf"] + base_stats["zero_sf"],
                "rejected_duplicate_cid": spec_stats["duplicate_cid"] + base_stats["duplicate_cid"],
            }
        )

        if len(spec_sel) + len(base_sel) == 0:
            break

    return pl.DataFrame(selected_entries), audit_rows


def _finalize_outputs(
    published: pl.DataFrame,
    sf_pd: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and format final constraint + SF outputs."""
    published = published.with_columns(
        (pl.col("constraint_id") + "|" + pl.col("shadow_sign").cast(pl.Utf8) + "|spice")
        .alias(CONSTRAINT_INDEX_COLUMN)
    )

    out_cols = expected_constraint_output_columns()
    for col in out_cols:
        if col not in published.columns:
            raise ValueError(f"Missing column: {col}")

    constraints_out = published.select(out_cols).to_pandas()
    for col in REQUIRED_CONSTRAINT_NON_NULL_COLUMNS:
        n_null = constraints_out[col].isna().sum()
        if n_null > 0:
            raise ValueError(f"Column {col} has {n_null} nulls")

    pub_cids = published["constraint_id"].to_list()
    pub_signs = published["shadow_sign"].to_list()
    sf_cid_cols = [c for c in pub_cids if c in sf_pd.columns]
    missing_sf = set(pub_cids) - set(sf_pd.columns)
    if missing_sf:
        raise ValueError(
            f"{len(missing_sf)} published constraints missing SF coverage. "
            f"Cannot publish without SF. First 5: {sorted(missing_sf)[:5]}"
        )

    cid_to_sign = dict(zip(pub_cids, pub_signs))
    sf_out = sf_pd[sf_cid_cols].copy()
    sf_out = sf_out.rename(columns={c: f"{c}|{int(cid_to_sign[c])}|spice" for c in sf_cid_cols})
    sf_out = sf_out.reset_index()
    return constraints_out, sf_out

def _load_flow_direction(planning_year: str, market_round: int) -> pl.DataFrame:
    """Load flow_direction from MISO_SPICE_DENSITY_SIGNAL_SCORE.

    Each CID has rows for both flow directions (+1 and -1). We pick the direction
    with the highest density signal score per CID. This is the canonical source
    for flow_direction in the annual signal (confirmed by teammate 2026-03-23).

    The opaque 'score' value is NOT used for ranking — only for direction selection.
    """
    ds = pl.scan_parquet(DENSITY_SCORE_PATH).filter(
        (pl.col("auction_type") == "annual")
        & (pl.col("auction_month") == planning_year)
        & (pl.col("market_round") == market_round)
    ).select(["constraint_id", "flow_direction", "score"]).collect()

    if len(ds) == 0:
        raise ValueError(f"No density signal score for {planning_year} round {market_round}")

    return (
        ds.sort("score", descending=True)
        .group_by("constraint_id").first()
        .select(["constraint_id", "flow_direction"])
    )


def _load_density_score_branch(
    planning_year: str,
    flow_dir: pl.DataFrame,
    constraints: pl.DataFrame,
    market_round: int,
) -> pl.DataFrame:
    """Load density signal score, pick correct direction, aggregate to branch level."""
    ds = pl.scan_parquet(DENSITY_SCORE_PATH).filter(
        (pl.col("auction_type") == "annual")
        & (pl.col("auction_month") == planning_year)
        & (pl.col("market_round") == market_round)
    ).select(["constraint_id", "flow_direction", "score"]).collect()

    # Match to chosen flow_direction
    ds_with_fd = ds.join(flow_dir.rename({"flow_direction": "fd_chosen"}), on="constraint_id", how="inner")
    ds_matched = ds_with_fd.filter(pl.col("flow_direction") == pl.col("fd_chosen"))

    # Map CID → branch
    cid_branch = constraints.select(["constraint_id", "branch_name"]).unique(subset=["constraint_id"])
    ds_branch = ds_matched.join(cid_branch, on="constraint_id", how="inner")

    # Branch-level mean score
    # NOTE: In V6.1, ori_mean and mix_mean can differ slightly. We use the same
    # density score for both as an approximation. Exact reproduction requires
    # running psignal's full density pipeline with ori/mix separation.
    branch_score = ds_branch.group_by("branch_name").agg(
        pl.col("score").mean().alias("ori_mean")
    )
    return branch_score.with_columns(
        pl.col("ori_mean").alias("mix_mean"),
        pl.col("ori_mean").alias("mean_branch_max"),
    )


def _load_bus_key(planning_year: str, aq_quarter: str, class_type: str, market_round: int) -> pl.DataFrame:
    """Load bus_key from pbase branches."""
    import sys
    sys.path.insert(0, "/home/xyz/workspace/psignal/src")
    sys.path.insert(0, "/home/xyz/workspace/pbase/src")
    from psignal.spice.data.source import SpiceDataSource

    ds = SpiceDataSource(rto="miso", auction_type="annual")
    branches = ds.load_branches(
        auction_month=planning_year, market_round=market_round,
        period_type=aq_quarter, class_type=class_type,
    )
    branches = branches.rename(columns={"memo": "branch_name"})
    branches["bus_key"] = (
        branches["from_number"].astype(int).astype(str)
        + "," + branches["to_number"].astype(int).astype(str)
    )
    return pl.from_pandas(branches[["branch_name", "bus_key"]].drop_duplicates("branch_name"))


def _compute_bus_key_group(constraints: pl.DataFrame) -> pl.DataFrame:
    """Compute bus_key_group via union-find on bus pairs (matching psignal base.py:166-172)."""
    bus_keys = constraints["bus_key"].to_list()

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Build union-find from all bus pairs
    for bk in bus_keys:
        if bk and "," in bk:
            parts = bk.split(",")
            if len(parts) == 2:
                for p in parts:
                    if p not in parent:
                        parent[p] = p
                union(parts[0], parts[1])

    # Map each bus_key to its group representative
    def get_group(bk: str) -> str:
        if not bk or "," not in bk:
            return bk
        first_bus = bk.split(",")[0]
        return find(first_bus) if first_bus in parent else bk

    groups = [get_group(bk) for bk in bus_keys]
    return constraints.with_columns(pl.Series("bus_key_group", groups))


def _load_sf(planning_year: str, market_months: list[str], market_round: int) -> pl.DataFrame:
    """Load and aggregate SF from MISO_SPICE_SF.parquet for the given quarter."""
    import glob

    frames = []
    for mm in market_months:
        pattern = (f"{SF_PATH}/spice_version=v6/auction_type=annual/"
                   f"auction_month={planning_year}/market_month={mm}/market_round={market_round}/*/")
        for pdir in glob.glob(pattern):
            for pf in glob.glob(f"{pdir}*.parquet"):
                frames.append(pl.read_parquet(pf))

    if not frames:
        raise ValueError(f"No SF data for {planning_year} {market_months}")

    combined = pl.concat(frames, how="diagonal")
    pnode_col = "pnode_id"
    sf_cols = [c for c in combined.columns if c != pnode_col]

    return combined.group_by(pnode_col).agg([pl.col(c).mean() for c in sf_cols])


def _chebyshev_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Max absolute difference (L-inf distance)."""
    valid = ~(np.isnan(a) | np.isnan(b))
    if not valid.any():
        return 0.0
    return float(np.max(np.abs(a[valid] - b[valid])))


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson correlation, or None if insufficient data."""
    valid = ~(np.isnan(a) | np.isnan(b))
    if valid.sum() < 3:
        return None
    a_v, b_v = a[valid], b[valid]
    a_std, b_std = a_v.std(), b_v.std()
    if a_std == 0 or b_std == 0:
        return None
    return float(np.corrcoef(a_v, b_v)[0, 1])
