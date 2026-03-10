"""Reproduce V6.2B rank from parquet columns and verify exact match.

Usage:
  /home/xyz/workspace/pmodel/.venv/bin/python -m ml.reproduce_v62b --month 2020-12
  /home/xyz/workspace/pmodel/.venv/bin/python -m ml.reproduce_v62b --default-eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from ml.config import V62B_SIGNAL_BASE, _DEFAULT_EVAL_MONTHS
from ml.v62b_formula import v62b_rank_from_columns


def _load_month(month: str, period_type: str, class_type: str) -> pl.DataFrame:
    path = Path(V62B_SIGNAL_BASE) / month / period_type / class_type
    if not path.exists():
        raise FileNotFoundError(f"Missing V6.2B parquet: {path}")
    return pl.read_parquet(
        str(path),
        columns=["rank", "da_rank_value", "density_mix_rank_value", "density_ori_rank_value"],
    )


def reproduce_month(month: str, period_type: str = "f0", class_type: str = "onpeak") -> float:
    df = _load_month(month, period_type, class_type)
    rank_true = df["rank"].to_numpy().astype(float)
    rank_calc = v62b_rank_from_columns(
        da_rank_value=df["da_rank_value"].to_numpy(),
        density_mix_rank_value=df["density_mix_rank_value"].to_numpy(),
        density_ori_rank_value=df["density_ori_rank_value"].to_numpy(),
    )
    max_abs = float(np.max(np.abs(rank_calc - rank_true))) if len(rank_true) else 0.0
    return max_abs


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce V6.2B rank and verify.")
    parser.add_argument("--month", default=None, help="Auction month YYYY-MM")
    parser.add_argument("--default-eval", action="store_true", help="Verify all default eval months")
    parser.add_argument("--ptype", default="f0", help="Period type")
    parser.add_argument("--class-type", default="onpeak", help="Class type")
    args = parser.parse_args()

    if not args.month and not args.default_eval:
        raise SystemExit("Provide --month or --default-eval")

    months = [args.month] if args.month else list(_DEFAULT_EVAL_MONTHS)
    worst = 0.0
    worst_month = None
    for m in months:
        diff = reproduce_month(m, period_type=args.ptype, class_type=args.class_type)
        print(f"{m}: max_abs_diff={diff:g}")
        if diff > worst:
            worst = diff
            worst_month = m

    if worst == 0.0:
        print("OK: exact match for all checked months.")
    else:
        raise SystemExit(f"Mismatch: worst {worst:g} on {worst_month}")


if __name__ == "__main__":
    main()

