"""Build full SPICE CID <-> branch <-> DA CID mapping tables for teammate review.

For each (PY, aq, class_type), produces one parquet with the complete round-trip:
  - Every SPICE CID in the density/bridge universe -> branch_name
  - Every DA CID from realized DA -> branch_name (via GT bridge logic)
  - Which branches are in the model universe, which are published

Usage:
    source /home/xyz/workspace/pmodel/.venv/bin/activate
    RAY_ADDRESS=ray://10.8.0.36:10001 PYTHONPATH=. uv run python scripts/v7_mapping_tables.py
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAY_ADDRESS", "ray://10.8.0.36:10001")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONSTRAINT_ROOT = "/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1"
OUTPUT_ROOT = Path("data/v7_verification")


def build_spice_to_branch(py: str, aq: str) -> pl.DataFrame:
    """All SPICE CIDs from the density universe -> branch_name via annual bridge.

    This is the mapping used by load_collapsed() and signal_publisher().
    """
    from ml.config import RIGHT_TAIL_BINS, UNIVERSE_THRESHOLD
    from ml.data_loader import load_raw_density
    from ml.bridge import load_bridge_partition

    # Load density, compute right_tail_max and is_active
    raw = load_raw_density(py, aq)
    raw = raw.with_columns(
        pl.max_horizontal([pl.col(b) for b in RIGHT_TAIL_BINS]).alias("right_tail")
    )
    cid_rtm = raw.group_by("constraint_id").agg(
        pl.col("right_tail").max().alias("right_tail_max")
    )
    cid_rtm = cid_rtm.with_columns(
        (pl.col("right_tail_max") >= UNIVERSE_THRESHOLD).alias("is_active")
    )

    # Map via annual bridge (same logic as load_collapsed)
    bridge = load_bridge_partition("annual", py, aq)

    # Detect ambiguous
    cid_counts = bridge.group_by("constraint_id").agg(
        pl.col("branch_name").n_unique().alias("n_branches")
    )
    ambig = set(cid_counts.filter(pl.col("n_branches") > 1)["constraint_id"].to_list())
    bridge_clean = bridge.filter(~pl.col("constraint_id").is_in(list(ambig))).unique(subset=["constraint_id"])

    # Join: SPICE CID -> branch
    spice_map = cid_rtm.join(bridge_clean, on="constraint_id", how="left")
    spice_map = spice_map.with_columns(
        pl.when(pl.col("constraint_id").is_in(list(ambig)))
        .then(pl.lit("ambiguous"))
        .when(pl.col("branch_name").is_null())
        .then(pl.lit("no_bridge"))
        .otherwise(pl.lit("mapped"))
        .alias("spice_mapping_status")
    )

    return spice_map.select([
        "constraint_id", "branch_name", "right_tail_max", "is_active", "spice_mapping_status",
    ]).rename({"constraint_id": "spice_cid"})


def build_da_to_branch(py: str, aq: str, ctype: str) -> pl.DataFrame:
    """All DA CIDs from realized DA -> branch_name via GT bridge logic.

    This is the mapping used by build_ground_truth().
    """
    from ml.config import get_market_months
    from ml.realized_da import load_month
    from ml.bridge import load_bridge_partition

    market_months = get_market_months(py, aq)

    # Load class-specific DA
    frames = []
    for mm in market_months:
        frames.append(load_month(mm, ctype))
    da = pl.concat(frames).group_by("constraint_id").agg(
        pl.col("realized_sp").sum()
    )

    # Annual bridge
    bridge = load_bridge_partition("annual", py, aq)
    cid_counts = bridge.group_by("constraint_id").agg(
        pl.col("branch_name").n_unique().alias("n_branches")
    )
    ambig = set(cid_counts.filter(pl.col("n_branches") > 1)["constraint_id"].to_list())
    bridge_clean = bridge.filter(~pl.col("constraint_id").is_in(list(ambig))).unique(subset=["constraint_id"])

    annual_cids = set(bridge_clean["constraint_id"].to_list())

    rows = []
    for row in da.iter_rows(named=True):
        cid = row["constraint_id"]
        sp = row["realized_sp"]

        if cid in ambig:
            rows.append({
                "da_cid": cid, "realized_sp": sp, "branch_name": None,
                "da_mapping_status": "ambiguous_dropped", "da_mapping_source": "annual",
            })
        elif cid in annual_cids:
            match = bridge_clean.filter(pl.col("constraint_id") == cid)
            branch = match["branch_name"][0] if len(match) > 0 else None
            rows.append({
                "da_cid": cid, "realized_sp": sp, "branch_name": branch,
                "da_mapping_status": "mapped", "da_mapping_source": "annual",
            })
        else:
            # Monthly fallback
            recovered = False
            for mm in market_months:
                try:
                    mb = load_bridge_partition("monthly", mm, "f0")
                    mb_clean = mb.unique(subset=["constraint_id"])
                    match = mb_clean.filter(pl.col("constraint_id") == cid)
                    if len(match) > 0:
                        rows.append({
                            "da_cid": cid, "realized_sp": sp,
                            "branch_name": match["branch_name"][0],
                            "da_mapping_status": "mapped",
                            "da_mapping_source": f"monthly_{mm}",
                        })
                        recovered = True
                        break
                except FileNotFoundError:
                    continue
            if not recovered:
                rows.append({
                    "da_cid": cid, "realized_sp": sp, "branch_name": None,
                    "da_mapping_status": "unmapped", "da_mapping_source": "none",
                })

    return pl.DataFrame(rows)


def build_full_mapping(py: str, aq: str, ctype: str) -> pl.DataFrame:
    """Build the full round-trip mapping table.

    Output: one row per (branch_name, spice_cid, da_cid) combination.
    Includes branches with only SPICE CIDs, only DA CIDs, or both.
    """
    from ml.data_loader import load_collapsed

    logger.info("Building SPICE -> branch map for %s/%s", py, aq)
    spice_map = build_spice_to_branch(py, aq)

    logger.info("Building DA -> branch map for %s/%s/%s", py, aq, ctype)
    da_map = build_da_to_branch(py, aq, ctype)

    # Model universe branches
    collapsed = load_collapsed(py, aq)
    universe_branches = set(collapsed["branch_name"].to_list())

    # Published branches + CIDs
    pub_path = f"{CONSTRAINT_ROOT}/{py}/{aq}/{ctype}/signal.parquet"
    published = pl.read_parquet(pub_path)
    pub_branches = set(published["branch_name"].to_list())
    pub_cids = set(published["constraint_id"].to_list())

    # --- Table 1: SPICE CID -> branch (with model universe + published flags) ---
    spice_out = spice_map.with_columns(
        pl.col("branch_name").is_in(list(universe_branches)).alias("in_model_universe"),
        pl.col("spice_cid").is_in(list(pub_cids)).alias("is_published"),
    )
    # Add published tier if published
    pub_tier = published.select(["constraint_id", "tier"]).rename({"constraint_id": "spice_cid"})
    spice_out = spice_out.join(pub_tier, on="spice_cid", how="left")

    # --- Table 2: DA CID -> branch (with model universe + realized SP) ---
    da_out = da_map.with_columns(
        pl.when(pl.col("branch_name").is_not_null())
        .then(pl.col("branch_name").is_in(list(universe_branches)))
        .otherwise(pl.lit(False))
        .alias("in_model_universe"),
        pl.when(pl.col("branch_name").is_not_null())
        .then(pl.col("branch_name").is_in(list(pub_branches)))
        .otherwise(pl.lit(False))
        .alias("branch_is_published"),
    )

    # --- Table 3: Branch-level summary ---
    # For each branch: SPICE CID count, DA CID count, total DA SP, universe/published status
    all_branches = set()
    spice_branches = spice_map.filter(pl.col("branch_name").is_not_null())["branch_name"].to_list()
    da_branches = da_map.filter(pl.col("branch_name").is_not_null())["branch_name"].to_list()
    all_branches.update(spice_branches)
    all_branches.update(da_branches)

    branch_rows = []
    for branch in sorted(all_branches):
        spice_cids_on_branch = spice_map.filter(pl.col("branch_name") == branch)
        da_cids_on_branch = da_map.filter(pl.col("branch_name") == branch)

        n_spice = len(spice_cids_on_branch)
        n_spice_active = int(spice_cids_on_branch.filter(pl.col("is_active"))["is_active"].sum()) if n_spice > 0 else 0
        n_da = len(da_cids_on_branch)
        da_sp = float(da_cids_on_branch["realized_sp"].sum()) if n_da > 0 else 0.0
        in_univ = branch in universe_branches
        is_pub = branch in pub_branches

        # Published CID count
        n_pub_cids = int(published.filter(pl.col("branch_name") == branch).height)

        branch_rows.append({
            "branch_name": branch,
            "n_spice_cids": n_spice,
            "n_spice_active": n_spice_active,
            "n_da_cids": n_da,
            "da_sp_total": da_sp,
            "in_model_universe": in_univ,
            "is_published": is_pub,
            "n_published_cids": n_pub_cids,
        })

    branch_summary = pl.DataFrame(branch_rows)

    return spice_out, da_out, branch_summary


def main():
    from pbase.config.ray import init_ray
    init_ray()

    slices = [
        ("2021-06", "aq1", "onpeak"),
        ("2021-06", "aq1", "offpeak"),
        ("2025-06", "aq2", "onpeak"),
        ("2025-06", "aq2", "offpeak"),
    ]

    for py, aq, ctype in slices:
        t0 = time.time()
        slug = f"{py}_{aq}_{ctype}"
        out_dir = OUTPUT_ROOT / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=== %s/%s/%s ===", py, aq, ctype)

        spice_out, da_out, branch_summary = build_full_mapping(py, aq, ctype)

        spice_out.write_parquet(str(out_dir / "spice_cid_to_branch.parquet"))
        da_out.write_parquet(str(out_dir / "da_cid_to_branch.parquet"))
        branch_summary.write_parquet(str(out_dir / "branch_summary.parquet"))

        logger.info("Saved 3 parquets to %s", out_dir)

        # Quick summary
        n_spice = len(spice_out)
        n_da = len(da_out)
        n_branches = len(branch_summary)
        n_both = int(branch_summary.filter(
            (pl.col("n_spice_cids") > 0) & (pl.col("n_da_cids") > 0)
        ).height)
        n_spice_only = int(branch_summary.filter(
            (pl.col("n_spice_cids") > 0) & (pl.col("n_da_cids") == 0)
        ).height)
        n_da_only = int(branch_summary.filter(
            (pl.col("n_spice_cids") == 0) & (pl.col("n_da_cids") > 0)
        ).height)
        da_mapped = int(da_out.filter(pl.col("da_mapping_status") == "mapped").height)
        da_unmapped = int(da_out.filter(pl.col("da_mapping_status") == "unmapped").height)

        print(f"\n  {py}/{aq}/{ctype}:")
        print(f"    SPICE CIDs: {n_spice}")
        print(f"    DA CIDs: {n_da} ({da_mapped} mapped, {da_unmapped} unmapped)")
        print(f"    Branches total: {n_branches}")
        print(f"      Both SPICE+DA: {n_both}")
        print(f"      SPICE only: {n_spice_only}")
        print(f"      DA only: {n_da_only}")
        print(f"    Time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
