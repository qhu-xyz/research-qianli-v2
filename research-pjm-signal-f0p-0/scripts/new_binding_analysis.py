#!/usr/bin/env python
"""V7.0b vs V6.2B: Signal distinctiveness and new-binding analysis for PJM.

Replicates the MISO analysis from research-miso-signal7/docs/new-binding-analysis.md.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl
from dateutil.relativedelta import relativedelta

from ml.config import HOLDOUT_MONTHS, V62B_SIGNAL_BASE

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_DIR = ROOT / "holdout"


# -- Helpers ------------------------------------------------------------------

def load_v62b_raw(auction_month: str, period_type: str, class_type: str) -> pl.DataFrame:
    """Load raw V6.2B signal (rank_ori only, no enrichment)."""
    path = Path(V62B_SIGNAL_BASE) / auction_month / period_type / class_type
    df = pl.read_parquet(str(path))
    return df.select(
        pl.col("constraint_id").cast(pl.String),
        pl.col("branch_name").cast(pl.String),
        pl.col("rank_ori").cast(pl.Float64),
    )


def load_v70b_results(auction_month: str, period_type: str, class_type: str) -> pl.DataFrame:
    """Load V7.0b holdout results (rank column from holdout predictions)."""
    holdout_pred_dir = HOLDOUT_DIR / period_type / class_type / "v2" / "predictions"
    pred_path = holdout_pred_dir / f"{auction_month}.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"No V7.0b predictions for {auction_month}: {pred_path}")
    return pl.read_parquet(str(pred_path)).select(
        pl.col("constraint_id").cast(pl.String),
        pl.col("rank").cast(pl.Float64),
    )


def load_realized_binding(month: str, peak_type: str) -> set:
    """Return set of branch_names that bound in this month."""
    from ml.realized_da import load_realized_da
    try:
        da = load_realized_da(month, peak_type=peak_type)
        return set(da.filter(pl.col("realized_sp") > 0)["branch_name"].to_list())
    except FileNotFoundError:
        return set()


def compute_auc(ranks: np.ndarray, labels: np.ndarray) -> float:
    """AUC: P(rank_binding < rank_nonbinding). Lower rank = higher priority."""
    binding = ranks[labels == 1]
    nonbinding = ranks[labels == 0]
    if len(binding) == 0 or len(nonbinding) == 0:
        return 0.5
    auc = np.mean([np.mean(nonbinding > b) for b in binding])
    return auc


def rank_to_tier(rank: float) -> int:
    """Convert rank (0-1) to tier (0-4). T0 = top 20%."""
    if rank <= 0.20: return 0
    if rank <= 0.40: return 1
    if rank <= 0.60: return 2
    if rank <= 0.80: return 3
    return 4


# -- Main analysis ------------------------------------------------------------

def main():
    period_type = "f0"
    class_type = "onpeak"
    months = sorted(HOLDOUT_MONTHS)

    print("=" * 80)
    print("V7.0b vs V6.2B: Signal Distinctiveness Analysis (PJM)")
    print(f"Scope: {period_type}/{class_type}, {len(months)} holdout months")
    print("=" * 80)

    # -- Study 1 & 2: Overall discrimination --
    print("\n## Study 1: Overall Discrimination")
    print(f"{'Month':>8} {'#Bound':>7} {'V6.2B avgR':>11} {'V7.0b avgR':>11} "
          f"{'Delta':>7} {'V6.2B T20':>9} {'V7.0b T20':>9} "
          f"{'V6.2B AUC':>9} {'V7.0b AUC':>9}")
    print("-" * 95)

    total_v62b_t20, total_v70b_t20 = 0, 0
    all_v62b_auc, all_v70b_auc = [], []

    for m in months:
        try:
            v62b = load_v62b_raw(m, period_type, class_type)
            v70b = load_v70b_results(m, period_type, class_type)
        except FileNotFoundError as e:
            print(f"{m:>8} SKIP: {e}")
            continue

        from ml.config import delivery_month
        gt_month = delivery_month(m, period_type)
        binding_branches = load_realized_binding(gt_month, class_type)

        merged = v62b.join(v70b, on="constraint_id", how="inner")
        merged = merged.with_columns(
            pl.col("branch_name").is_in(list(binding_branches)).cast(pl.Int8).alias("bound")
        )

        n_bound = merged.filter(pl.col("bound") == 1).height
        if n_bound == 0:
            continue

        binding_df = merged.filter(pl.col("bound") == 1)
        v62b_avg = binding_df["rank_ori"].mean()
        v70b_avg = binding_df["rank"].mean()

        v62b_t20 = binding_df.filter(pl.col("rank_ori") <= 0.20).height
        v70b_t20 = binding_df.filter(pl.col("rank") <= 0.20).height
        total_v62b_t20 += v62b_t20
        total_v70b_t20 += v70b_t20

        ranks_62b = merged["rank_ori"].to_numpy()
        ranks_70b = merged["rank"].to_numpy()
        labels = merged["bound"].to_numpy()
        auc_62b = compute_auc(ranks_62b, labels)
        auc_70b = compute_auc(ranks_70b, labels)
        all_v62b_auc.append(auc_62b)
        all_v70b_auc.append(auc_70b)

        print(f"{m:>8} {n_bound:>7} {v62b_avg:>11.4f} {v70b_avg:>11.4f} "
              f"{v62b_avg - v70b_avg:>+7.3f} {v62b_t20:>9} {v70b_t20:>9} "
              f"{auc_62b:>9.3f} {auc_70b:>9.3f}")

    print(f"\nAggregate Top-20 captures: V6.2B={total_v62b_t20}, V7.0b={total_v70b_t20} "
          f"(V7.0b {total_v70b_t20/max(total_v62b_t20,1)*100 - 100:+.0f}%)")
    if all_v62b_auc:
        print(f"Mean AUC: V6.2B={np.mean(all_v62b_auc):.3f}, V7.0b={np.mean(all_v70b_auc):.3f}")

    # -- Study 3: New binder early alarm --
    print("\n## Study 3: New Binder Early Alarm")
    print("Constraints that first bind during holdout with NO prior binding history.")

    from ml.config import delivery_month as dm
    pre_holdout_binding: dict[str, set[str]] = {}
    first_holdout = months[0]
    all_months_dirs = sorted(Path(V62B_SIGNAL_BASE).iterdir())
    for mp in all_months_dirs:
        m_str = mp.name
        if m_str >= first_holdout:
            break
        gt = dm(m_str, period_type)
        for bn in load_realized_binding(gt, class_type):
            pre_holdout_binding.setdefault(bn, set()).add(m_str)

    new_binder_events = []
    holdout_binding: dict[str, str] = {}
    for m in months:
        gt = dm(m, period_type)
        binding = load_realized_binding(gt, class_type)
        for bn in binding:
            if bn not in pre_holdout_binding and bn not in holdout_binding:
                holdout_binding[bn] = m
                new_binder_events.append((bn, m))

    print(f"Total truly new binders in holdout: {len(new_binder_events)}")

    for lead in [1, 2, 3, 6]:
        v62b_t0, v70b_t0, v62b_t01, v70b_t01, n_obs = 0, 0, 0, 0, 0
        for bn, first_month in new_binder_events:
            dt = datetime.datetime.strptime(first_month, "%Y-%m")
            check_dt = dt - relativedelta(months=lead)
            check_month = check_dt.strftime("%Y-%m")
            if check_month not in months and check_month < months[0]:
                continue

            try:
                v62b = load_v62b_raw(check_month, period_type, class_type)
                v70b = load_v70b_results(check_month, period_type, class_type)
            except FileNotFoundError:
                continue

            v62b_rows = v62b.filter(pl.col("branch_name") == bn)
            if v62b_rows.height == 0:
                continue

            cids = v62b_rows["constraint_id"].to_list()
            v70b_rows = v70b.filter(pl.col("constraint_id").is_in(cids))
            if v70b_rows.height == 0:
                continue

            best_v62b = v62b_rows["rank_ori"].min()
            best_v70b = v70b_rows["rank"].min()

            n_obs += 1
            if rank_to_tier(best_v62b) == 0: v62b_t0 += 1
            if rank_to_tier(best_v70b) == 0: v70b_t0 += 1
            if rank_to_tier(best_v62b) <= 1: v62b_t01 += 1
            if rank_to_tier(best_v70b) <= 1: v70b_t01 += 1

        if n_obs > 0:
            winner = "V6.2B" if v62b_t0 > v70b_t0 else ("V7.0b" if v70b_t0 > v62b_t0 else "TIE")
            print(f"  Lead={lead}mo: n={n_obs}, V6.2B T0={v62b_t0/n_obs:.1%}, "
                  f"V7.0b T0={v70b_t0/n_obs:.1%}, "
                  f"V6.2B T0+T1={v62b_t01/n_obs:.1%}, "
                  f"V7.0b T0+T1={v70b_t01/n_obs:.1%} [{winner}]")

    # -- Study 4: Recurring binder early alarm --
    print("\n## Study 4: Recurring Binder Early Alarm")
    print("Constraints that bound, stopped 3+ months, then re-bound during holdout.")

    full_binding: dict[str, set[str]] = {}
    for bn, ms in pre_holdout_binding.items():
        full_binding.setdefault(bn, set()).update(ms)
    for m in months:
        gt = dm(m, period_type)
        for bn in load_realized_binding(gt, class_type):
            full_binding.setdefault(bn, set()).add(m)

    gap_events = []
    for bn, bound_months in full_binding.items():
        sorted_months = sorted(bound_months)
        for i in range(1, len(sorted_months)):
            prev = datetime.datetime.strptime(sorted_months[i-1], "%Y-%m")
            curr = datetime.datetime.strptime(sorted_months[i], "%Y-%m")
            gap = (curr.year - prev.year) * 12 + curr.month - prev.month
            if gap >= 4 and sorted_months[i] in months:
                gap_events.append((bn, sorted_months[i]))

    print(f"Total gap-resume events: {len(gap_events)}")

    v62b_t0, v70b_t0, v62b_t01, v70b_t01, n_obs = 0, 0, 0, 0, 0
    for bn, resume_month in gap_events:
        dt = datetime.datetime.strptime(resume_month, "%Y-%m")
        check_dt = dt - relativedelta(months=1)
        check_month = check_dt.strftime("%Y-%m")

        try:
            v62b = load_v62b_raw(check_month, period_type, class_type)
            v70b = load_v70b_results(check_month, period_type, class_type)
        except FileNotFoundError:
            continue

        v62b_rows = v62b.filter(pl.col("branch_name") == bn)
        if v62b_rows.height == 0:
            continue
        cids = v62b_rows["constraint_id"].to_list()
        v70b_rows = v70b.filter(pl.col("constraint_id").is_in(cids))
        if v70b_rows.height == 0:
            continue

        best_v62b = v62b_rows["rank_ori"].min()
        best_v70b = v70b_rows["rank"].min()

        n_obs += 1
        if rank_to_tier(best_v62b) == 0: v62b_t0 += 1
        if rank_to_tier(best_v70b) == 0: v70b_t0 += 1
        if rank_to_tier(best_v62b) <= 1: v62b_t01 += 1
        if rank_to_tier(best_v70b) <= 1: v70b_t01 += 1

    if n_obs > 0:
        print(f"  1mo before re-bind: n={n_obs}, V6.2B T0={v62b_t0/n_obs:.1%}, "
              f"V7.0b T0={v70b_t0/n_obs:.1%}, "
              f"V6.2B T0+T1={v62b_t01/n_obs:.1%}, "
              f"V7.0b T0+T1={v70b_t01/n_obs:.1%}")

    print("\n## Summary written to docs/new-binding-analysis.md")


if __name__ == "__main__":
    main()
