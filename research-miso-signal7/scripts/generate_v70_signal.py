#!/usr/bin/env python
"""Generate V7.0 MISO FTR constraint signal for one auction month.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-miso-signal7/scripts/generate_v70_signal.py \
        --auction-month 2026-03

V7.0 replaces V6.2B formula scoring for f0/f1 with LightGBM LambdaRank.
All other period types (f2, f3, q2-q4) are exact V6.2B passthrough.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Add project root to path (for v70 package)
_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

# Stage5 is added by v70 modules automatically
from pbase.data.dataset.signal.general import ConstraintsSignal, ShiftFactorSignal

from v70.cache import REALIZED_DA_CACHE, ensure_realized_da_cache
from v70.inference import load_all_binding_sets, score_ml_inference
from v70.signal_writer import available_ptypes, compute_rank_tier

V62B_SIGNAL = "TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1"
V70_SIGNAL = "TEST.TEST.Signal.MISO.SPICE_F0P_V7.0.R1"

ML_PTYPES = ["f0", "f1"]
CLASS_TYPES = ["onpeak", "offpeak"]


def generate_v70_signal(
    auction_month: str,
    signal_name: str = V70_SIGNAL,
    dry_run: bool = False,
) -> dict[str, str]:
    """Generate V7.0 signal for one auction month.

    Returns dict of {ptype/ctype: status} for logging.
    """
    t0 = time.time()
    results: dict[str, str] = {}
    ts = pd.Timestamp(auction_month)

    # ── Step 1: Preflight — ensure realized DA cache ──
    print(f"\n{'='*60}")
    print(f"V7.0 Signal Generation: {auction_month}")
    print(f"{'='*60}")
    ensure_realized_da_cache(auction_month, ptypes=ML_PTYPES, ctypes=CLASS_TYPES)

    # ── Step 2: Load binding sets (shared across all ML slices) ──
    bs = {ct: load_all_binding_sets(peak_type=ct, cache_dir=REALIZED_DA_CACHE) for ct in CLASS_TYPES}
    for ct in CLASS_TYPES:
        print(f"[main] {ct} binding sets: {len(bs[ct])} months loaded")

    # ── Step 3: Process each slice ──
    ptypes = available_ptypes(auction_month)
    print(f"[main] Available ptypes for {auction_month}: {ptypes}")

    for ptype in ptypes:
        for ctype in CLASS_TYPES:
            key = f"{ptype}/{ctype}"
            t1 = time.time()

            # Load V6.2B source (pandas DataFrame with string index)
            v62b_df = ConstraintsSignal(
                "miso", V62B_SIGNAL, ptype, ctype
            ).load_data(ts)

            if ptype in ML_PTYPES:
                # Extract V6.2B rank_ori per constraint_id (before overwrite)
                v62b_cids = v62b_df.index.str.split("|").str[0]
                v62b_rank_map = pd.Series(
                    v62b_df["rank_ori"].values, index=v62b_cids
                ).groupby(level=0).first()

                # ML scoring
                cids, scores = score_ml_inference(
                    auction_month, ptype, ctype, bs[ctype]
                )

                # V6.2B rank_ori as tie-breaker (lower = more binding)
                v62b_rank_for_cids = v62b_rank_map.reindex(cids).fillna(1.0).values
                rank, tier = compute_rank_tier(scores, v62b_rank_for_cids)

                # Join on constraint_id, NOT positional assignment
                score_df = pd.DataFrame({
                    "constraint_id": cids,
                    "_rank_ori": scores,
                    "_rank": rank,
                    "_tier": tier,
                }).set_index("constraint_id")

                v62b_df["rank_ori"] = score_df.loc[v62b_cids.values, "_rank_ori"].values
                v62b_df["rank"] = score_df.loc[v62b_cids.values, "_rank"].values
                v62b_df["tier"] = score_df.loc[v62b_cids.values, "_tier"].values

                # SO_MW_Transfer exception: force tier=1
                if "branch_name" in v62b_df.columns:
                    so_mask = v62b_df["branch_name"] == "SO_MW_Transfer"
                    if so_mask.any():
                        v62b_df.loc[so_mask, "tier"] = 1
                        print(f"[main] {key}: SO_MW_Transfer -> tier=1 ({so_mask.sum()} rows)")

                status = f"ML scored ({len(cids)} constraints, {time.time()-t1:.1f}s)"
            else:
                # Passthrough: V6.2B unchanged
                status = f"passthrough ({len(v62b_df)} constraints)"

            # Write constraints signal
            ConstraintsSignal(
                "miso", signal_name, ptype, ctype
            ).save_data(v62b_df, ts, dry_run=dry_run)

            # Copy shift factors from V6.2B
            sf_df = ShiftFactorSignal(
                "miso", V62B_SIGNAL, ptype, ctype
            ).load_data(ts)
            ShiftFactorSignal(
                "miso", signal_name, ptype, ctype
            ).save_data(sf_df, ts, dry_run=dry_run)

            results[key] = status
            print(f"[main] {key}: {status}")

    elapsed = time.time() - t0
    print(f"\n[main] Done in {elapsed:.1f}s")
    for k, v in results.items():
        print(f"  {k}: {v}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate V7.0 signal")
    parser.add_argument("--auction-month", required=True, help="Auction month (YYYY-MM)")
    parser.add_argument("--signal-name", default=V70_SIGNAL, help="Output signal name")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to NFS")
    args = parser.parse_args()

    generate_v70_signal(args.auction_month, args.signal_name, args.dry_run)


if __name__ == "__main__":
    main()
