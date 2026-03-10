#!/usr/bin/env python
"""Live head-to-head comparison: V6.2B vs V7.0 on realized DA ground truth.

Loads both signals via pbase ConstraintsSignal, joins realized DA,
computes VC@K / Recall@K / NDCG from scratch. No pre-computed tables.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-miso-signal7/scripts/compare_v62b_vs_v70.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from pbase.data.dataset.signal.general import ConstraintsSignal

# Add project root to path for v70 package
_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))
from v70.cache import REALIZED_DA_CACHE

V62B = "TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1"
V70 = "TEST.TEST.Signal.MISO.SPICE_F0P_V7.0.R1"

DA_CACHE = Path(REALIZED_DA_CACHE)

# Holdout: 2024-01 through 2025-12
HOLDOUT_MONTHS = [f"{y:04d}-{m:02d}" for y in (2024, 2025) for m in range(1, 13)]
ML_PTYPES = ["f0", "f1"]
CLASS_TYPES = ["onpeak", "offpeak"]

# MISO auction schedule (which ptypes exist per calendar month)
MISO_SCHEDULE = {
    1: ["f0", "f1", "q4"], 2: ["f0", "f1", "f2", "f3"],
    3: ["f0", "f1", "f2"], 4: ["f0", "f1"],
    5: ["f0"], 6: ["f0"],
    7: ["f0", "f1", "q2", "q3", "q4"], 8: ["f0", "f1", "f2", "f3"],
    9: ["f0", "f1", "f2"], 10: ["f0", "f1", "q3", "q4"],
    11: ["f0", "f1", "f2", "f3"], 12: ["f0", "f1", "f2"],
}

PERIOD_OFFSETS = {"f0": 0, "f1": 1, "f2": 2, "f3": 3}


def load_da(delivery_month: str, ctype: str) -> dict[str, float]:
    """Load realized DA -> {constraint_id: realized_sp}."""
    suffix = f"_{ctype}" if ctype == "offpeak" else ""
    fpath = DA_CACHE / f"{delivery_month}{suffix}.parquet"
    if not fpath.exists():
        return {}
    df = pl.read_parquet(str(fpath))
    return dict(zip(
        df["constraint_id"].to_list(),
        df["realized_sp"].to_list(),
    ))


def vc_at_k(realized: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Value capture @ k: fraction of total realized value in top-k by score."""
    total = realized.sum()
    if total <= 0:
        return 0.0
    k = min(k, len(scores))
    top_k = np.argsort(scores)[::-1][:k]
    return float(realized[top_k].sum() / total)


def recall_at_k(realized: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Recall @ k: fraction of binding constraints captured in top-k."""
    n_binding = (realized > 0).sum()
    if n_binding == 0:
        return 0.0
    k = min(k, len(scores))
    top_k = np.argsort(scores)[::-1][:k]
    return float((realized[top_k] > 0).sum() / n_binding)


def ndcg(realized: np.ndarray, scores: np.ndarray) -> float:
    """Normalized discounted cumulative gain."""
    order = np.argsort(scores)[::-1]
    dcg = np.sum(realized[order] / np.log2(np.arange(len(realized)) + 2))
    ideal = np.sort(realized)[::-1]
    idcg = np.sum(ideal / np.log2(np.arange(len(realized)) + 2))
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def score_signal(signal_df: pd.DataFrame) -> np.ndarray:
    """Extract scores from signal df. Use -rank (lower rank = better = higher score)."""
    return -signal_df["rank"].values.astype(np.float64)


def evaluate_month(
    auction_month: str, ptype: str, ctype: str,
) -> dict[str, dict[str, float]] | None:
    """Evaluate V6.2B vs V7.0 for one (month, ptype, ctype) slice.

    Returns {"v62b": {metrics}, "v70": {metrics}} or None if data missing.
    """
    ts = pd.Timestamp(auction_month)
    offset = PERIOD_OFFSETS.get(ptype, 0)
    delivery = (ts + pd.DateOffset(months=offset)).strftime("%Y-%m")

    # Load realized DA
    da_map = load_da(delivery, ctype)
    if not da_map:
        return None

    results = {}
    for label, sig_name in [("v62b", V62B), ("v70", V70)]:
        try:
            df = ConstraintsSignal("miso", sig_name, ptype, ctype).load_data(ts)
        except (FileNotFoundError, OSError):
            return None

        if len(df) == 0:
            return None

        # Extract constraint_id from index (format: "constraint_id|flow_direction")
        cids = df.index.str.split("|").str[0].values

        # Map realized DA to each row
        realized = np.array([da_map.get(cid, 0.0) for cid in cids], dtype=np.float64)
        scores = score_signal(df)

        results[label] = {
            "VC@20": vc_at_k(realized, scores, 20),
            "VC@50": vc_at_k(realized, scores, 50),
            "VC@100": vc_at_k(realized, scores, 100),
            "Recall@20": recall_at_k(realized, scores, 20),
            "Recall@100": recall_at_k(realized, scores, 100),
            "NDCG": ndcg(realized, scores),
            "n_constraints": len(df),
            "n_binding": int((realized > 0).sum()),
        }

    return results


def main():
    t0 = time.time()
    print("=" * 80)
    print("LIVE COMPARISON: V6.2B vs V7.0 (from pbase ConstraintsSignal)")
    print("Ground truth: realized DA shadow prices")
    print("=" * 80)

    all_results: dict[str, list[dict]] = {}

    for ptype in ML_PTYPES:
        for ctype in CLASS_TYPES:
            key = f"{ptype}/{ctype}"
            all_results[key] = []

            for month in HOLDOUT_MONTHS:
                month_num = pd.Timestamp(month).month
                if ptype not in MISO_SCHEDULE.get(month_num, ["f0"]):
                    continue

                result = evaluate_month(month, ptype, ctype)
                if result is None:
                    continue

                all_results[key].append({
                    "month": month,
                    **{f"v62b_{k}": v for k, v in result["v62b"].items()},
                    **{f"v70_{k}": v for k, v in result["v70"].items()},
                })

    # Print per-slice summary
    metrics = ["VC@20", "VC@50", "VC@100", "Recall@20", "Recall@100", "NDCG"]

    for key, rows in all_results.items():
        if not rows:
            print(f"\n{key}: no data")
            continue

        print(f"\n{'='*70}")
        print(f"  {key}  ({len(rows)} months)")
        print(f"{'='*70}")

        # Per-month detail
        print(f"{'Month':<10} ", end="")
        for m in metrics[:3]:
            print(f"{'V6.2B':>8} {'V7.0':>8} {'delta':>7}  ", end="")
        print()
        print("-" * 90)

        for r in rows:
            print(f"{r['month']:<10} ", end="")
            for m in metrics[:3]:
                v62b = r[f"v62b_{m}"]
                v70 = r[f"v70_{m}"]
                delta = v70 - v62b
                sign = "+" if delta >= 0 else ""
                print(f"{v62b:>8.4f} {v70:>8.4f} {sign}{delta:>6.4f}  ", end="")
            print()

        # Aggregate means
        print("-" * 90)
        print(f"{'MEAN':<10} ", end="")
        for m in metrics[:3]:
            v62b_mean = np.mean([r[f"v62b_{m}"] for r in rows])
            v70_mean = np.mean([r[f"v70_{m}"] for r in rows])
            delta = v70_mean - v62b_mean
            sign = "+" if delta >= 0 else ""
            print(f"{v62b_mean:>8.4f} {v70_mean:>8.4f} {sign}{delta:>6.4f}  ", end="")
        print()

        # Full metrics summary
        print(f"\n{'Metric':<12} {'V6.2B':>10} {'V7.0':>10} {'Diff':>10} {'% Change':>10}")
        print("-" * 55)
        for m in metrics:
            v62b_mean = np.mean([r[f"v62b_{m}"] for r in rows])
            v70_mean = np.mean([r[f"v70_{m}"] for r in rows])
            diff = v70_mean - v62b_mean
            pct = 100 * diff / v62b_mean if v62b_mean != 0 else 0
            winner = "***" if diff > 0.005 else ("   " if diff > -0.005 else "<<<")
            print(f"{m:<12} {v62b_mean:>10.4f} {v70_mean:>10.4f} {diff:>+10.4f} {pct:>+9.1f}% {winner}")

    # Grand summary
    print(f"\n{'='*70}")
    print("GRAND SUMMARY (mean VC@20 across holdout)")
    print(f"{'='*70}")
    print(f"{'Slice':<16} {'V6.2B':>10} {'V7.0':>10} {'Improvement':>12} {'Months':>8}")
    print("-" * 60)
    for key, rows in all_results.items():
        if not rows:
            continue
        v62b_vc20 = np.mean([r["v62b_VC@20"] for r in rows])
        v70_vc20 = np.mean([r["v70_VC@20"] for r in rows])
        pct = 100 * (v70_vc20 - v62b_vc20) / v62b_vc20 if v62b_vc20 != 0 else 0
        print(f"{key:<16} {v62b_vc20:>10.4f} {v70_vc20:>10.4f} {pct:>+11.1f}% {len(rows):>8}")

    # V7.0 win rate
    print(f"\n{'Slice':<16} {'V7.0 Wins':>10} {'Ties':>6} {'V6.2B Wins':>12}")
    print("-" * 50)
    for key, rows in all_results.items():
        if not rows:
            continue
        wins = sum(1 for r in rows if r["v70_VC@20"] > r["v62b_VC@20"] + 0.001)
        ties = sum(1 for r in rows if abs(r["v70_VC@20"] - r["v62b_VC@20"]) <= 0.001)
        losses = len(rows) - wins - ties
        print(f"{key:<16} {wins:>10} {ties:>6} {losses:>12}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
