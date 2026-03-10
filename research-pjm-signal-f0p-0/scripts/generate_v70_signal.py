#!/usr/bin/env python
"""Generate V7.0b PJM FTR constraint signal for one auction month.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0/scripts/generate_v70_signal.py \
        --auction-month 2026-03

V7.0b replaces V6.2B formula scoring for f0/f1 with LightGBM LambdaRank.
All other period types (f2-f11) are exact V6.2B passthrough.
PJM has 3 class types: onpeak, dailyoffpeak, wkndonpeak.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from pbase.data.dataset.signal.general import ConstraintsSignal, ShiftFactorSignal

from v70.cache import REALIZED_DA_CACHE, ensure_realized_da_cache
from v70.inference import load_all_binding_sets, score_ml_inference
from v70.signal_writer import available_ptypes, compute_rank_tier
from ml.config import PJM_CLASS_TYPES

V62B_SIGNAL = "TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1"
V70_SIGNAL = "TEST.TEST.Signal.PJM.SPICE_F0P_V7.0B.R1"

ML_PTYPES = ["f0", "f1"]
CLASS_TYPES = PJM_CLASS_TYPES  # onpeak, dailyoffpeak, wkndonpeak


def generate_v70_signal(
    auction_month: str,
    signal_name: str = V70_SIGNAL,
    dry_run: bool = False,
    verify: bool = False,
) -> dict[str, str]:
    """Generate V7.0b signal for one auction month.

    Returns dict of {ptype/ctype: status} for logging.
    """
    t0 = time.time()
    results: dict[str, str] = {}
    ts = pd.Timestamp(auction_month)

    # -- Step 1: Preflight — ensure realized DA cache --
    print(f"\n{'='*60}")
    print(f"V7.0b PJM Signal Generation: {auction_month}")
    print(f"{'='*60}")
    ensure_realized_da_cache(auction_month, ptypes=ML_PTYPES, ctypes=CLASS_TYPES)

    # -- Step 2: Load binding sets (shared across all ML slices) --
    bs = {ct: load_all_binding_sets(peak_type=ct, cache_dir=REALIZED_DA_CACHE) for ct in CLASS_TYPES}
    for ct in CLASS_TYPES:
        print(f"[main] {ct} binding sets: {len(bs[ct])} months loaded")

    # -- Step 3: Process each slice --
    ptypes = available_ptypes(auction_month)
    print(f"[main] Available ptypes for {auction_month}: {ptypes}")

    for ptype in ptypes:
        for ctype in CLASS_TYPES:
            key = f"{ptype}/{ctype}"
            t1 = time.time()

            v62b_df = ConstraintsSignal(
                "pjm", V62B_SIGNAL, ptype, ctype
            ).load_data(ts)

            if len(v62b_df) == 0:
                results[key] = "SKIP (V6.2B source empty)"
                print(f"[main] {key}: {results[key]}")
                continue

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
                assert len(cids) == len(set(cids)), (
                    f"Duplicate constraint_ids in ML output for {key}"
                )

                # V6.2B rank_ori as tie-breaker
                v62b_rank_for_cids = v62b_rank_map.reindex(cids).fillna(1.0).values
                rank, tier = compute_rank_tier(scores, v62b_rank_for_cids)

                # Join on constraint_id
                score_df = pd.DataFrame({
                    "constraint_id": cids,
                    "_rank_ori": scores,
                    "_rank": rank,
                    "_tier": tier,
                }).set_index("constraint_id")

                v62b_df["rank_ori"] = score_df.loc[v62b_cids.values, "_rank_ori"].values
                v62b_df["rank"] = score_df.loc[v62b_cids.values, "_rank"].values
                v62b_df["tier"] = score_df.loc[v62b_cids.values, "_tier"].values

                # PJM: no SO_MW_Transfer exception (MISO-specific)

                status = f"ML scored ({len(cids)} constraints, {time.time()-t1:.1f}s)"
            else:
                # Passthrough: V6.2B unchanged
                status = f"passthrough ({len(v62b_df)} constraints)"

            # Write constraints signal
            ConstraintsSignal(
                "pjm", signal_name, ptype, ctype
            ).save_data(v62b_df, ts, dry_run=dry_run)

            # Copy shift factors from V6.2B
            sf_df = ShiftFactorSignal(
                "pjm", V62B_SIGNAL, ptype, ctype
            ).load_data(ts)
            ShiftFactorSignal(
                "pjm", signal_name, ptype, ctype
            ).save_data(sf_df, ts, dry_run=dry_run)

            results[key] = status
            print(f"[main] {key}: {status}")

    elapsed = time.time() - t0
    written = [k for k, v in results.items() if not v.startswith("SKIP")]

    print(f"\n[main] Done in {elapsed:.1f}s")
    for k, v in results.items():
        print(f"  {k}: {v}")

    if not written:
        print(f"\n[main] ERROR: no slices were written for {auction_month}")

    # -- Verification --
    if verify:
        _verify_output(auction_month, signal_name, ptypes)

    return results


def _verify_output(auction_month: str, signal_name: str, ptypes: list[str]) -> None:
    """Verify V7.0b output against V6.2B source."""
    import polars as pl

    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")

    ts = pd.Timestamp(auction_month)
    changed_cols = {"rank", "rank_ori", "tier"}

    for ptype in ptypes:
        for ctype in CLASS_TYPES:
            key = f"{ptype}/{ctype}"
            try:
                old_pd = ConstraintsSignal("pjm", V62B_SIGNAL, ptype, ctype).load_data(ts)
                new_pd = ConstraintsSignal("pjm", signal_name, ptype, ctype).load_data(ts)
            except Exception as e:
                print(f"  {key}: SKIP ({e})")
                continue

            if len(old_pd) == 0 or len(new_pd) == 0:
                print(f"  {key}: SKIP (empty)")
                continue

            # Row count
            if len(old_pd) != len(new_pd):
                print(f"  {key}: FAIL — row count {len(old_pd)} vs {len(new_pd)}")
                continue

            old = pl.from_pandas(old_pd.reset_index())
            new = pl.from_pandas(new_pd.reset_index())

            # Immutable columns check
            immutable = [c for c in old.columns if c not in changed_cols and c in new.columns]
            all_ok = True
            for col in immutable:
                if old[col].to_list() != new[col].to_list():
                    print(f"  {key}: FAIL — column '{col}' changed but should be immutable")
                    all_ok = False
                    break

            if ptype in ML_PTYPES:
                if all_ok:
                    print(f"  {key}: {len(new)} rows, {len(immutable)} immutable cols verified — PASS")
            else:
                # Passthrough should be identical
                if all_ok and old.frame_equal(new):
                    print(f"  {key}: passthrough — bit-identical PASS")
                elif all_ok:
                    print(f"  {key}: passthrough — columns match but values differ FAIL")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PJM V7.0b signal")
    parser.add_argument("--auction-month", required=True, help="Auction month (YYYY-MM)")
    parser.add_argument("--signal-name", default=V70_SIGNAL, help="Output signal name")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to NFS")
    parser.add_argument("--verify", action="store_true", help="Run verification after generation")
    args = parser.parse_args()

    results = generate_v70_signal(args.auction_month, args.signal_name, args.dry_run, args.verify)
    written = [k for k, v in results.items() if not v.startswith("SKIP")]
    if not written:
        sys.exit(1)


if __name__ == "__main__":
    main()
