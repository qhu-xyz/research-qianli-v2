"""Step 2-5: Build verification tables for one slice.

Produces:
  - real_da_merged.parquet
  - training_universe_map.parquet
  - published_signal_map.parquet
  - loss_waterfall.parquet
  - summary.json

Usage:
    source /home/xyz/workspace/pmodel/.venv/bin/activate
    RAY_ADDRESS=ray://10.8.0.36:10001 PYTHONPATH=. uv run python scripts/v7_step2_da_merge.py \
        --py 2025-06 --aq aq2 --class-type offpeak
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAY_ADDRESS", "ray://10.8.0.36:10001")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONSTRAINT_ROOT = "/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1"
SF_ROOT = "/opt/data/xyz-dataset/signal_data/miso/sf/TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1"
OUTPUT_ROOT = Path("data/v7_verification")


def build_real_da_merged(
    py: str, aq: str, ctype: str,
) -> tuple[pl.DataFrame, dict]:
    """Step 2: Trace every DA CID's mapping fate."""
    from ml.config import get_market_months, CLASS_TARGET_COL
    from ml.realized_da import load_month
    from ml.bridge import load_bridge_partition, map_cids_to_branches

    market_months = get_market_months(py, aq)
    peak_type = ctype  # "onpeak" or "offpeak"

    # Load class-specific DA
    frames = []
    for mm in market_months:
        frames.append(load_month(mm, peak_type))
    da = pl.concat(frames).group_by("constraint_id").agg(
        pl.col("realized_sp").sum()
    )
    logger.info("DA loaded: %d CIDs, SP=%.0f", len(da), da["realized_sp"].sum())

    # Try annual bridge
    try:
        annual_bridge = load_bridge_partition("annual", py, aq)
    except FileNotFoundError:
        annual_bridge = pl.DataFrame(schema={"constraint_id": pl.Utf8, "branch_name": pl.Utf8})

    annual_cids = set(annual_bridge["constraint_id"].to_list())

    # Detect ambiguous in annual
    cid_branch_counts = annual_bridge.group_by("constraint_id").agg(
        pl.col("branch_name").n_unique().alias("n_branches")
    )
    ambig_cids = set(
        cid_branch_counts.filter(pl.col("n_branches") > 1)["constraint_id"].to_list()
    )
    annual_unique = annual_bridge.filter(
        ~pl.col("constraint_id").is_in(list(ambig_cids))
    ).unique(subset=["constraint_id"])

    # Map DA CIDs via annual
    da_with_status = da.with_columns(
        pl.lit("").alias("branch_name"),
        pl.lit("").alias("mapping_status"),
        pl.lit("").alias("mapping_source"),
    )

    rows = []
    for row in da.iter_rows(named=True):
        cid = row["constraint_id"]
        sp = row["realized_sp"]

        if cid in ambig_cids:
            rows.append({
                "constraint_id": cid, "realized_sp": sp,
                "branch_name": None, "mapping_status": "ambiguous_dropped",
                "mapping_source": "annual",
            })
        elif cid in annual_cids:
            branch_rows = annual_unique.filter(pl.col("constraint_id") == cid)
            branch = branch_rows["branch_name"][0] if len(branch_rows) > 0 else None
            rows.append({
                "constraint_id": cid, "realized_sp": sp,
                "branch_name": branch, "mapping_status": "mapped",
                "mapping_source": "annual",
            })
        else:
            # Try monthly fallback
            recovered = False
            for mm in market_months:
                try:
                    monthly_bridge = load_bridge_partition("monthly", mm, "f0")
                    monthly_unique = monthly_bridge.unique(subset=["constraint_id"])
                    match = monthly_unique.filter(pl.col("constraint_id") == cid)
                    if len(match) > 0:
                        rows.append({
                            "constraint_id": cid, "realized_sp": sp,
                            "branch_name": match["branch_name"][0],
                            "mapping_status": "mapped",
                            "mapping_source": f"monthly_{mm}",
                        })
                        recovered = True
                        break
                except FileNotFoundError:
                    continue

            if not recovered:
                rows.append({
                    "constraint_id": cid, "realized_sp": sp,
                    "branch_name": None, "mapping_status": "unmapped",
                    "mapping_source": "none",
                })

    merged = pl.DataFrame(rows)

    # Summary stats
    status_counts = merged.group_by("mapping_status").agg(
        pl.len().alias("count"),
        pl.col("realized_sp").sum().alias("total_sp"),
    )
    logger.info("DA mapping status:\n%s", status_counts)

    diag = {
        "total_da_cids": len(merged),
        "total_da_sp": float(merged["realized_sp"].sum()),
        "mapped_cids": int(merged.filter(pl.col("mapping_status") == "mapped").height),
        "mapped_sp": float(merged.filter(pl.col("mapping_status") == "mapped")["realized_sp"].sum()),
        "unmapped_cids": int(merged.filter(pl.col("mapping_status") == "unmapped").height),
        "unmapped_sp": float(merged.filter(pl.col("mapping_status") == "unmapped")["realized_sp"].sum()),
        "ambiguous_cids": int(merged.filter(pl.col("mapping_status") == "ambiguous_dropped").height),
        "ambiguous_sp": float(merged.filter(pl.col("mapping_status") == "ambiguous_dropped")["realized_sp"].sum()),
    }

    return merged, diag


def build_training_universe_map(
    py: str, aq: str, ctype: str,
    da_merged: pl.DataFrame,
    published_cstr: pl.DataFrame,
) -> tuple[pl.DataFrame, dict]:
    """Step 3: One row per model-table branch with GT + published flag."""
    from ml.phase6.features import build_class_model_table
    from ml.phase6.scoring import score_v0c, _minmax
    from ml.config import CLASS_BF_COL

    table = build_class_model_table(py, aq, ctype)
    bf_col = CLASS_BF_COL[ctype]

    # Compute v0c score
    scores = score_v0c(table, bf_col)
    table = table.with_columns(pl.Series("v0c_score", scores))

    # Published branches
    pub_branches = set(published_cstr["branch_name"].to_list())
    pub_branch_cids = published_cstr.group_by("branch_name").agg(
        pl.len().alias("n_published_cids")
    )

    table = table.join(pub_branch_cids, on="branch_name", how="left")
    table = table.with_columns(
        pl.col("n_published_cids").fill_null(0),
        pl.col("branch_name").is_in(list(pub_branches)).alias("is_published"),
    )

    # Select output columns
    out_cols = [
        "branch_name", "realized_shadow_price", "cohort",
        "v0c_score", "da_rank_value", "shadow_price_da",
        "bf_12" if ctype == "onpeak" else "bfo_12",
        "has_hist_da", "is_nb_12",
        "is_published", "n_published_cids",
    ]
    out_cols = [c for c in out_cols if c in table.columns]
    result = table.select(out_cols)

    n_binding = int((result["realized_shadow_price"] > 0).sum())
    n_published = int(result["is_published"].sum())
    n_binding_published = int(
        (result.filter(pl.col("realized_shadow_price") > 0)["is_published"]).sum()
    )

    diag = {
        "n_branches": len(result),
        "n_binding": n_binding,
        "n_published": n_published,
        "n_binding_published": n_binding_published,
        "n_binding_not_published": n_binding - n_binding_published,
    }
    logger.info("Training universe: %d branches, %d binding, %d published, %d binding+published",
                len(result), n_binding, n_published, n_binding_published)

    return result, diag


def build_published_signal_map(
    py: str, aq: str, ctype: str,
    published_cstr: pl.DataFrame,
    training_map: pl.DataFrame,
) -> tuple[pl.DataFrame, dict]:
    """Step 4: One row per published constraint with branch-level DA."""
    # Get branch-level SP from training map
    branch_sp = training_map.select([
        "branch_name", "realized_shadow_price", "cohort", "v0c_score",
    ])

    # Join published constraints to branch SP
    result = published_cstr.select([
        "constraint_id", "branch_name", "tier", "rank",
        "da_rank_value", "shadow_sign", "flow_direction",
    ]).join(branch_sp, on="branch_name", how="left")

    # Count siblings
    sibling_counts = published_cstr.group_by("branch_name").agg(
        pl.len().alias("n_siblings")
    )
    result = result.join(sibling_counts, on="branch_name", how="left")

    result = result.with_columns(
        (pl.col("realized_shadow_price").fill_null(0.0) > 0).alias("is_binding"),
        pl.col("realized_shadow_price").fill_null(0.0),
    )

    # Per-tier stats
    tier_stats = {}
    for tier in range(5):
        tier_df = result.filter(pl.col("tier") == tier)
        n = len(tier_df)
        n_bind = int(tier_df["is_binding"].sum()) if n > 0 else 0
        bind_rate = n_bind / n if n > 0 else 0
        mean_sp = float(tier_df["realized_shadow_price"].mean()) if n > 0 else 0
        tier_stats[f"tier_{tier}"] = {
            "n": n, "n_binding": n_bind,
            "binding_rate": bind_rate, "mean_sp": mean_sp,
        }

    # da_rank_value range check
    da_rank_vals = result["da_rank_value"].to_numpy()
    da_rank_max = float(np.max(da_rank_vals))

    n_unique_branches = result["branch_name"].n_unique()
    max_siblings = int(result["n_siblings"].max())

    diag = {
        "n_constraints": len(result),
        "n_unique_branches": n_unique_branches,
        "max_siblings": max_siblings,
        "per_tier": tier_stats,
        "da_rank_value_max": da_rank_max,
        "da_rank_value_gt_1": int((da_rank_vals > 1.0).sum()),
    }
    logger.info("Published signal: %d constraints, %d unique branches, max siblings=%d",
                len(result), n_unique_branches, max_siblings)
    for t, s in tier_stats.items():
        logger.info("  %s: %d constraints, %d binding (%.1f%%), mean SP=%.0f",
                     t, s["n"], s["n_binding"], s["binding_rate"] * 100, s["mean_sp"])

    return result, diag


def build_loss_waterfall(
    da_merged: pl.DataFrame,
    training_map: pl.DataFrame,
    published_cstr: pl.DataFrame,
    ctype: str,
) -> tuple[pl.DataFrame, dict]:
    """Step 5: 4-stage SP funnel."""
    total_sp = float(da_merged["realized_sp"].sum())
    total_cids = len(da_merged)

    # Stage 1: total DA
    # Stage 2: mapped to branches
    mapped = da_merged.filter(pl.col("mapping_status") == "mapped")
    mapped_sp = float(mapped["realized_sp"].sum())
    mapped_branches = mapped["branch_name"].n_unique()

    # Stage 3: mapped branch in model universe
    universe_branches = set(training_map["branch_name"].to_list())
    mapped_in_universe = mapped.filter(
        pl.col("branch_name").is_in(list(universe_branches))
    )
    in_universe_sp = float(mapped_in_universe["realized_sp"].sum())
    in_universe_branches = mapped_in_universe["branch_name"].n_unique()

    # Stage 4: published branch
    pub_branches = set(published_cstr["branch_name"].to_list())
    mapped_published = mapped_in_universe.filter(
        pl.col("branch_name").is_in(list(pub_branches))
    )
    published_sp = float(mapped_published["realized_sp"].sum())
    published_branches = mapped_published["branch_name"].n_unique()

    # Build waterfall rows
    waterfall = []
    for row in da_merged.iter_rows(named=True):
        cid = row["constraint_id"]
        sp = row["realized_sp"]
        branch = row["branch_name"]
        status = row["mapping_status"]

        if status == "unmapped":
            stage = "unmapped_cid"
        elif status == "ambiguous_dropped":
            stage = "ambiguous_dropped"
        elif branch and branch not in universe_branches:
            stage = "outside_model_universe"
        elif branch and branch not in pub_branches:
            stage = "in_universe_not_published"
        else:
            stage = "published"

        waterfall.append({
            "constraint_id": cid,
            "branch_name": branch,
            "realized_sp": sp,
            "loss_reason": stage,
        })

    waterfall_df = pl.DataFrame(waterfall)

    # Print waterfall
    print()
    print(f"{'Stage':<35} {'CIDs':>6} {'Branches':>9} {'SP':>14} {'%':>7} {'Lost_SP':>14}")
    print("-" * 90)
    print(f"{'1. Total DA':.<35} {total_cids:>6} {'-':>9} {total_sp:>14,.0f} {'100.0%':>7}")
    lost_unmapped = total_sp - mapped_sp
    print(f"{'2. Mapped to branches':.<35} {len(mapped):>6} {mapped_branches:>9} {mapped_sp:>14,.0f} {mapped_sp/total_sp*100:>6.1f}% {lost_unmapped:>14,.0f}")
    lost_universe = mapped_sp - in_universe_sp
    print(f"{'3. In model universe':.<35} {len(mapped_in_universe):>6} {in_universe_branches:>9} {in_universe_sp:>14,.0f} {in_universe_sp/total_sp*100:>6.1f}% {lost_universe:>14,.0f}")
    lost_pub = in_universe_sp - published_sp
    print(f"{'4. Published':.<35} {len(mapped_published):>6} {published_branches:>9} {published_sp:>14,.0f} {published_sp/total_sp*100:>6.1f}% {lost_pub:>14,.0f}")
    print()

    diag = {
        "stage_1_total": {"cids": total_cids, "sp": total_sp},
        "stage_2_mapped": {"cids": len(mapped), "branches": mapped_branches, "sp": mapped_sp, "pct": mapped_sp / total_sp * 100},
        "stage_3_in_universe": {"cids": len(mapped_in_universe), "branches": in_universe_branches, "sp": in_universe_sp, "pct": in_universe_sp / total_sp * 100},
        "stage_4_published": {"cids": len(mapped_published), "branches": published_branches, "sp": published_sp, "pct": published_sp / total_sp * 100},
        "loss_unmapped": lost_unmapped,
        "loss_out_of_universe": lost_universe,
        "loss_publication_filter": lost_pub,
    }

    return waterfall_df, diag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--py", required=True)
    parser.add_argument("--aq", required=True)
    parser.add_argument("--class-type", required=True, choices=["onpeak", "offpeak"])
    args = parser.parse_args()

    from pbase.config.ray import init_ray
    init_ray()

    py, aq, ctype = args.py, args.aq, args.class_type
    slug = f"{py}_{aq}_{ctype}"
    out_dir = OUTPUT_ROOT / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info("=== V7 Verification: %s/%s/%s ===", py, aq, ctype)

    # Load published constraints
    pub_path = f"{CONSTRAINT_ROOT}/{py}/{aq}/{ctype}/signal.parquet"
    published_cstr = pl.read_parquet(pub_path)
    logger.info("Published constraints loaded: %d rows", len(published_cstr))

    # Step 2: real_da_merged
    logger.info("--- Step 2: real_da_merged ---")
    da_merged, da_diag = build_real_da_merged(py, aq, ctype)
    da_merged.write_parquet(str(out_dir / "real_da_merged.parquet"))
    logger.info("Saved real_da_merged.parquet (%d rows)", len(da_merged))

    # Step 3: training_universe_map
    logger.info("--- Step 3: training_universe_map ---")
    training_map, train_diag = build_training_universe_map(
        py, aq, ctype, da_merged, published_cstr,
    )
    training_map.write_parquet(str(out_dir / "training_universe_map.parquet"))
    logger.info("Saved training_universe_map.parquet (%d rows)", len(training_map))

    # Step 4: published_signal_map
    logger.info("--- Step 4: published_signal_map ---")
    pub_map, pub_diag = build_published_signal_map(
        py, aq, ctype, published_cstr, training_map,
    )
    pub_map.write_parquet(str(out_dir / "published_signal_map.parquet"))
    logger.info("Saved published_signal_map.parquet (%d rows)", len(pub_map))

    # Step 5: loss_waterfall
    logger.info("--- Step 5: loss_waterfall ---")
    waterfall, wf_diag = build_loss_waterfall(
        da_merged, training_map, published_cstr, ctype,
    )
    waterfall.write_parquet(str(out_dir / "loss_waterfall.parquet"))
    logger.info("Saved loss_waterfall.parquet (%d rows)", len(waterfall))

    # Summary
    summary = {
        "slice": f"{py}/{aq}/{ctype}",
        "da_merge": da_diag,
        "training_universe": train_diag,
        "published_signal": pub_diag,
        "loss_waterfall": wf_diag,
        "walltime_s": round(time.time() - t0, 1),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved summary.json (%.1fs total)", time.time() - t0)


if __name__ == "__main__":
    main()
