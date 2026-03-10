#!/usr/bin/env python
"""Tier-0 focused comparison: V6.2B vs V7.0.

Production uses tier 0 as a whole (top 20% of constraints).
This script answers: within that tier 0 set, is V7.0 actually better?

Metrics computed:
  1. Tier0 Value Capture: sum(realized_sp in tier0) / sum(realized_sp total)
  2. Tier0 Binding Rate: n_binding_in_tier0 / tier0_size
  3. Tier0 Binding Recall: n_binding_in_tier0 / n_binding_total
  4. Tier0 Mean Realized SP: mean(realized_sp for binding constraints in tier0)
  5. Tier0 Overlap: |V6.2B_tier0 ∩ V7.0_tier0| / |V6.2B_tier0 ∪ V7.0_tier0|
  6. Unique pick value: realized SP in V7.0-only tier0 vs V6.2B-only tier0
  7. Within-tier0 ranking quality (NDCG within tier0 only)

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-miso-signal7/scripts/compare_tier0.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats as sp_stats

from pbase.data.dataset.signal.general import ConstraintsSignal

# Add project root to path for v70 package
_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))
from v70.cache import REALIZED_DA_CACHE

V62B = "TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1"
V70 = "TEST.TEST.Signal.MISO.SPICE_F0P_V7.0.R1"

DA_CACHE = Path(REALIZED_DA_CACHE)

HOLDOUT_MONTHS = [f"{y:04d}-{m:02d}" for y in (2024, 2025) for m in range(1, 13)]
ML_PTYPES = ["f0", "f1"]
CLASS_TYPES = ["onpeak", "offpeak"]

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
    suffix = f"_{ctype}" if ctype == "offpeak" else ""
    fpath = DA_CACHE / f"{delivery_month}{suffix}.parquet"
    if not fpath.exists():
        return {}
    df = pl.read_parquet(str(fpath))
    return dict(zip(df["constraint_id"].to_list(), df["realized_sp"].to_list()))


def ndcg_subset(realized: np.ndarray) -> float:
    """NDCG of a pre-ordered array (position 0 = rank 1)."""
    if realized.sum() <= 0:
        return 0.0
    dcg = np.sum(realized / np.log2(np.arange(len(realized)) + 2))
    ideal = np.sort(realized)[::-1]
    idcg = np.sum(ideal / np.log2(np.arange(len(realized)) + 2))
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_tier0(
    auction_month: str, ptype: str, ctype: str,
) -> dict | None:
    """Compare tier 0 sets between V6.2B and V7.0 for one month."""
    ts = pd.Timestamp(auction_month)
    offset = PERIOD_OFFSETS.get(ptype, 0)
    delivery = (ts + pd.DateOffset(months=offset)).strftime("%Y-%m")

    da_map = load_da(delivery, ctype)
    if not da_map:
        return None

    signals = {}
    for label, sig_name in [("v62b", V62B), ("v70", V70)]:
        try:
            df = ConstraintsSignal("miso", sig_name, ptype, ctype).load_data(ts)
        except (FileNotFoundError, OSError):
            return None
        if len(df) == 0:
            return None
        signals[label] = df

    # Both signals should have the same constraint universe; verify
    if len(signals["v62b"]) != len(signals["v70"]):
        return None

    total_n = len(signals["v62b"])

    # Build per-constraint realized SP mapping (at constraint_id level, not row level)
    # Rows are indexed as "constraint_id|flow_direction"
    v62b_df = signals["v62b"]
    cids = v62b_df.index.str.split("|").str[0].values
    realized_full = np.array([da_map.get(cid, 0.0) for cid in cids], dtype=np.float64)
    total_sp = realized_full.sum()
    total_binding = int((realized_full > 0).sum())

    if total_sp <= 0 or total_binding == 0:
        return None

    result = {"month": auction_month, "total_n": total_n,
              "total_sp": total_sp, "total_binding": total_binding}

    for label in ["v62b", "v70"]:
        df = signals[label]
        tier = df["tier"].values.astype(int)
        rank = df["rank"].values.astype(np.float64)

        t0_mask = tier == 0
        t0_size = int(t0_mask.sum())
        t0_cids_set = set(cids[t0_mask])

        # Realized SP for tier 0 rows
        t0_realized = realized_full[t0_mask]
        t0_sp = float(t0_realized.sum())
        t0_binding = int((t0_realized > 0).sum())

        # Value capture = tier0_sp / total_sp
        t0_vc = t0_sp / total_sp

        # Binding rate = binding_in_tier0 / tier0_size
        t0_bind_rate = t0_binding / t0_size if t0_size > 0 else 0.0

        # Binding recall = binding_in_tier0 / total_binding
        t0_bind_recall = t0_binding / total_binding if total_binding > 0 else 0.0

        # Mean realized SP per binding constraint in tier0
        t0_mean_sp = float(t0_realized[t0_realized > 0].mean()) if t0_binding > 0 else 0.0

        # Within-tier0 NDCG (how well ordered within tier0)
        # Sort tier0 rows by rank ascending (lower rank = higher priority)
        t0_indices = np.where(t0_mask)[0]
        t0_rank_order = np.argsort(rank[t0_mask])  # ascending rank
        t0_realized_ordered = t0_realized[t0_rank_order]
        t0_ndcg = ndcg_subset(t0_realized_ordered)

        result[f"{label}_t0_size"] = t0_size
        result[f"{label}_t0_vc"] = t0_vc
        result[f"{label}_t0_binding"] = t0_binding
        result[f"{label}_t0_bind_rate"] = t0_bind_rate
        result[f"{label}_t0_bind_recall"] = t0_bind_recall
        result[f"{label}_t0_mean_sp"] = t0_mean_sp
        result[f"{label}_t0_ndcg"] = t0_ndcg
        result[f"{label}_t0_cids"] = t0_cids_set

    # Overlap analysis (at constraint_id level)
    v62b_t0 = result["v62b_t0_cids"]
    v70_t0 = result["v70_t0_cids"]
    overlap = v62b_t0 & v70_t0
    union = v62b_t0 | v70_t0
    only_v62b = v62b_t0 - v70_t0
    only_v70 = v70_t0 - v62b_t0

    result["overlap_n"] = len(overlap)
    result["overlap_jaccard"] = len(overlap) / len(union) if union else 0.0
    result["only_v62b_n"] = len(only_v62b)
    result["only_v70_n"] = len(only_v70)

    # Value of unique picks
    cid_to_sp = dict(zip(cids, realized_full))
    result["overlap_sp"] = sum(cid_to_sp.get(c, 0.0) for c in overlap)
    result["only_v62b_sp"] = sum(cid_to_sp.get(c, 0.0) for c in only_v62b)
    result["only_v70_sp"] = sum(cid_to_sp.get(c, 0.0) for c in only_v70)
    result["only_v62b_binding"] = sum(1 for c in only_v62b if cid_to_sp.get(c, 0.0) > 0)
    result["only_v70_binding"] = sum(1 for c in only_v70 if cid_to_sp.get(c, 0.0) > 0)

    # Clean up set fields before returning
    del result["v62b_t0_cids"], result["v70_t0_cids"]

    return result


def print_slice(key: str, rows: list[dict]) -> None:
    n = len(rows)
    print(f"\n{'='*100}")
    print(f"  {key}  ({n} months)")
    print(f"{'='*100}")

    # ── Table 1: Tier 0 Value Capture (the money question) ──
    print(f"\n--- Tier 0 Value Capture (realized SP in tier0 / total realized SP) ---")
    print(f"{'Month':<10} {'Size':>6} {'V6.2B_VC':>10} {'V7.0_VC':>10} {'Delta':>10} "
          f"{'V6.2B_bind':>10} {'V7.0_bind':>10}")
    print("-" * 72)
    for r in rows:
        d = r["v70_t0_vc"] - r["v62b_t0_vc"]
        sign = "+" if d >= 0 else ""
        print(f"{r['month']:<10} {r['v62b_t0_size']:>6} "
              f"{r['v62b_t0_vc']:>10.4f} {r['v70_t0_vc']:>10.4f} {sign}{d:>9.4f} "
              f"{r['v62b_t0_binding']:>10} {r['v70_t0_binding']:>10}")
    print("-" * 72)
    v62b_mean = np.mean([r["v62b_t0_vc"] for r in rows])
    v70_mean = np.mean([r["v70_t0_vc"] for r in rows])
    d = v70_mean - v62b_mean
    v62b_bind = np.mean([r["v62b_t0_binding"] for r in rows])
    v70_bind = np.mean([r["v70_t0_binding"] for r in rows])
    avg_size = np.mean([r["v62b_t0_size"] for r in rows])
    print(f"{'MEAN':<10} {avg_size:>6.0f} {v62b_mean:>10.4f} {v70_mean:>10.4f} "
          f"{'+' if d>=0 else ''}{d:>9.4f} {v62b_bind:>10.1f} {v70_bind:>10.1f}")

    # ── Table 2: Comprehensive metrics ──
    metrics = [
        ("T0 Value Capture", "t0_vc"),
        ("T0 Binding Rate", "t0_bind_rate"),
        ("T0 Binding Recall", "t0_bind_recall"),
        ("T0 Mean SP (bind)", "t0_mean_sp"),
        ("T0 Within-NDCG", "t0_ndcg"),
    ]
    print(f"\n--- Aggregated Metrics (mean over {n} months) ---")
    print(f"{'Metric':<22} {'V6.2B':>10} {'V7.0':>10} {'Diff':>10} {'%Chg':>8} {'Winner':>8}")
    print("-" * 72)
    for name, suffix in metrics:
        v62b_vals = [r[f"v62b_{suffix}"] for r in rows]
        v70_vals = [r[f"v70_{suffix}"] for r in rows]
        v62b_m = np.mean(v62b_vals)
        v70_m = np.mean(v70_vals)
        diff = v70_m - v62b_m
        pct = 100 * diff / v62b_m if v62b_m != 0 else 0
        winner = "V7.0" if diff > 0.001 else ("V6.2B" if diff < -0.001 else "TIE")
        print(f"{name:<22} {v62b_m:>10.4f} {v70_m:>10.4f} {diff:>+10.4f} {pct:>+7.1f}% {winner:>8}")

    # ── Table 3: Overlap & unique picks ──
    print(f"\n--- Tier 0 Set Overlap ---")
    avg_overlap = np.mean([r["overlap_n"] for r in rows])
    avg_only62 = np.mean([r["only_v62b_n"] for r in rows])
    avg_only70 = np.mean([r["only_v70_n"] for r in rows])
    avg_jaccard = np.mean([r["overlap_jaccard"] for r in rows])
    print(f"  Mean Jaccard overlap:    {avg_jaccard:.3f}")
    print(f"  Mean shared constraints: {avg_overlap:.0f}")
    print(f"  Mean V6.2B-only:         {avg_only62:.0f}")
    print(f"  Mean V7.0-only:          {avg_only70:.0f}")

    print(f"\n--- Unique Pick Quality ---")
    print(f"{'Month':<10} {'shared_SP':>10} {'v62b_only_SP':>12} {'v70_only_SP':>12} "
          f"{'v62b_only_#b':>12} {'v70_only_#b':>12}")
    print("-" * 72)
    for r in rows:
        print(f"{r['month']:<10} {r['overlap_sp']:>10.1f} {r['only_v62b_sp']:>12.1f} "
              f"{r['only_v70_sp']:>12.1f} {r['only_v62b_binding']:>12} {r['only_v70_binding']:>12}")
    print("-" * 72)
    shared_sp = sum(r["overlap_sp"] for r in rows)
    only62_sp = sum(r["only_v62b_sp"] for r in rows)
    only70_sp = sum(r["only_v70_sp"] for r in rows)
    only62_bind = sum(r["only_v62b_binding"] for r in rows)
    only70_bind = sum(r["only_v70_binding"] for r in rows)
    print(f"{'TOTAL':<10} {shared_sp:>10.1f} {only62_sp:>12.1f} {only70_sp:>12.1f} "
          f"{only62_bind:>12} {only70_bind:>12}")
    if only62_sp + only70_sp > 0:
        print(f"\n  Where the signals DISAGREE on tier 0:")
        print(f"    V6.2B unique picks captured: {only62_sp:>10.1f} realized SP ({only62_bind} binding)")
        print(f"    V7.0  unique picks captured: {only70_sp:>10.1f} realized SP ({only70_bind} binding)")
        diff_pct = 100 * (only70_sp - only62_sp) / (only62_sp + only70_sp) if (only62_sp + only70_sp) > 0 else 0
        winner = "V7.0" if only70_sp > only62_sp else "V6.2B"
        print(f"    => {winner} unique picks are worth more ({diff_pct:+.1f}% relative advantage)")

    # ── Win/Loss ──
    wins = sum(1 for r in rows if r["v70_t0_vc"] > r["v62b_t0_vc"] + 0.001)
    ties = sum(1 for r in rows if abs(r["v70_t0_vc"] - r["v62b_t0_vc"]) <= 0.001)
    losses = n - wins - ties
    print(f"\n  Tier0 VC win rate: V7.0 wins {wins}, ties {ties}, V6.2B wins {losses} out of {n} months")

    # ── Statistical significance ──
    v62b_vc = [r["v62b_t0_vc"] for r in rows]
    v70_vc = [r["v70_t0_vc"] for r in rows]
    diffs = [v7 - v6 for v7, v6 in zip(v70_vc, v62b_vc)]
    t_stat, p_value = sp_stats.ttest_rel(v70_vc, v62b_vc)
    _, wilcoxon_p = sp_stats.wilcoxon(diffs, alternative="two-sided")
    print(f"\n  Paired t-test: t={t_stat:.3f}, p={p_value:.4f} "
          f"{'(significant)' if p_value < 0.05 else '(NOT significant)'}")
    print(f"  Wilcoxon signed-rank: p={wilcoxon_p:.4f} "
          f"{'(significant)' if wilcoxon_p < 0.05 else '(NOT significant)'}")

    # ── Worst-case months ──
    worst = sorted(rows, key=lambda r: r["v70_t0_vc"] - r["v62b_t0_vc"])[:3]
    best = sorted(rows, key=lambda r: r["v70_t0_vc"] - r["v62b_t0_vc"], reverse=True)[:3]
    print(f"\n  3 worst months for V7.0 (vs V6.2B tier0 VC):")
    for r in worst:
        d = r["v70_t0_vc"] - r["v62b_t0_vc"]
        print(f"    {r['month']}: V6.2B={r['v62b_t0_vc']:.4f} V7.0={r['v70_t0_vc']:.4f} delta={d:+.4f}")
    print(f"  3 best months for V7.0:")
    for r in best:
        d = r["v70_t0_vc"] - r["v62b_t0_vc"]
        print(f"    {r['month']}: V6.2B={r['v62b_t0_vc']:.4f} V7.0={r['v70_t0_vc']:.4f} delta={d:+.4f}")


def main():
    t0 = time.time()
    print("=" * 100)
    print("TIER 0 DEEP DIVE: V6.2B vs V7.0")
    print("Question: within the top-20% tier used in production, is V7.0 actually better?")
    print("=" * 100)

    all_results: dict[str, list[dict]] = {}

    for ptype in ML_PTYPES:
        for ctype in CLASS_TYPES:
            key = f"{ptype}/{ctype}"
            all_results[key] = []

            for month in HOLDOUT_MONTHS:
                month_num = pd.Timestamp(month).month
                if ptype not in MISO_SCHEDULE.get(month_num, ["f0"]):
                    continue
                result = evaluate_tier0(month, ptype, ctype)
                if result is not None:
                    all_results[key].append(result)

    for key, rows in all_results.items():
        if rows:
            print_slice(key, rows)

    # ── Grand Summary ──
    print(f"\n{'='*100}")
    print("GRAND SUMMARY: Tier 0 Value Capture")
    print(f"{'='*100}")
    print(f"{'Slice':<16} {'V6.2B':>10} {'V7.0':>10} {'Diff':>10} {'%Chg':>8} "
          f"{'Wins':>6} {'Losses':>8} {'p-val':>8}")
    print("-" * 80)
    for key, rows in all_results.items():
        if not rows:
            continue
        v62b_m = np.mean([r["v62b_t0_vc"] for r in rows])
        v70_m = np.mean([r["v70_t0_vc"] for r in rows])
        diff = v70_m - v62b_m
        pct = 100 * diff / v62b_m if v62b_m != 0 else 0
        wins = sum(1 for r in rows if r["v70_t0_vc"] > r["v62b_t0_vc"] + 0.001)
        losses = sum(1 for r in rows if r["v62b_t0_vc"] > r["v70_t0_vc"] + 0.001)
        _, p = sp_stats.ttest_rel([r["v70_t0_vc"] for r in rows], [r["v62b_t0_vc"] for r in rows])
        print(f"{key:<16} {v62b_m:>10.4f} {v70_m:>10.4f} {diff:>+10.4f} {pct:>+7.1f}% "
              f"{wins:>6} {losses:>8} {p:>8.4f}")

    print(f"\n{'='*100}")
    print("GRAND SUMMARY: Unique Pick Value (where V6.2B and V7.0 DISAGREE)")
    print(f"{'='*100}")
    print(f"{'Slice':<16} {'V6.2B-only SP':>14} {'V7.0-only SP':>14} {'V7.0 edge':>12} {'V7.0 better?':>14}")
    print("-" * 75)
    for key, rows in all_results.items():
        if not rows:
            continue
        v62_sp = sum(r["only_v62b_sp"] for r in rows)
        v70_sp = sum(r["only_v70_sp"] for r in rows)
        edge = v70_sp - v62_sp
        winner = "YES" if edge > 0 else "NO"
        print(f"{key:<16} {v62_sp:>14.1f} {v70_sp:>14.1f} {edge:>+12.1f} {winner:>14}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
