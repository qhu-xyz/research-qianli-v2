"""Phase 1.2 — One-shot coverage audit for all PJM annual signal data sources.

Emits per-(PY, round) coverage tables for density, bridge, limit, SF, and DA.
Run: cd /home/xyz/workspace/pmodel && source .venv/bin/activate && python /home/xyz/workspace/research-qianli-v2/research-annual-signal-pjm/scripts/audit_coverage.py
"""
from __future__ import annotations

import os
from pathlib import Path

import polars as pl

# ── Paths ──────────────────────────────────────────────────────────────
SPICE_BASE = "/opt/data/xyz-dataset/spice_data/pjm"
DA_BASE = "/opt/data/xyz-dataset/modeling_data/pjm/PJM_DA_SHADOW_PRICE.parquet"

DATASETS = {
    "density": "PJM_SPICE_DENSITY_DISTRIBUTION.parquet",
    "bridge": "PJM_SPICE_CONSTRAINT_INFO.parquet",
    "limit": "PJM_SPICE_CONSTRAINT_LIMIT.parquet",
    "sf": "PJM_SPICE_SF.parquet",
}

ROOTS = {
    "legacy": "network_model=miso/spice_version=v6/auction_type=annual",
    "newer": "spice_version=v6/auction_type=annual",
}

ALL_PYS = ["2019-06", "2020-06", "2021-06", "2022-06", "2023-06", "2024-06", "2025-06"]
ALL_ROUNDS = [1, 2, 3, 4]
ALL_QUARTERS = ["aq1", "aq2", "aq3", "aq4"]


def _dir_exists(path: str) -> bool:
    return Path(path).is_dir()


# ── Density coverage ──────────────────────────────────────────────────
def audit_density() -> pl.DataFrame:
    """Per (PY, root, round): count market_months and unique CIDs."""
    rows = []
    ds = DATASETS["density"]
    for root_name, root_path in ROOTS.items():
        for py in ALL_PYS:
            base = f"{SPICE_BASE}/{ds}/{root_path}/auction_month={py}"
            if not _dir_exists(base):
                continue
            # Count market_months
            market_months = sorted(
                d.name.split("=")[1]
                for d in Path(base).iterdir()
                if d.is_dir() and d.name.startswith("market_month=")
            )
            n_months = len(market_months)
            # Read first market_month to get rounds
            if n_months == 0:
                continue
            first_mm = market_months[0]
            try:
                df = pl.read_parquet(f"{base}/market_month={first_mm}/")
                rounds_present = sorted(df["market_round"].unique().to_list())
                n_cids = df["constraint_id"].n_unique()
            except Exception as e:
                rounds_present = []
                n_cids = 0
            rows.append({
                "dataset": "density",
                "root": root_name,
                "py": py,
                "n_months": n_months,
                "rounds": str(rounds_present),
                "cids_sample": n_cids,
            })
    return pl.DataFrame(rows)


# ── Bridge coverage ───────────────────────────────────────────────────
def audit_bridge() -> pl.DataFrame:
    """Per (PY, root, round, quarter): whether partition exists and CID/branch counts."""
    rows = []
    ds = DATASETS["bridge"]
    for root_name, root_path in ROOTS.items():
        for py in ALL_PYS:
            for r in ALL_ROUNDS:
                for q in ALL_QUARTERS:
                    path = f"{SPICE_BASE}/{ds}/{root_path}/auction_month={py}/market_round={r}/period_type={q}"
                    exists = _dir_exists(path)
                    n_cids, n_branches, n_rows = 0, 0, 0
                    if exists:
                        try:
                            df = pl.read_parquet(f"{path}/")
                            n_rows = len(df)
                            n_cids = df["constraint_id"].n_unique()
                            n_branches = df["branch_name"].n_unique()
                        except Exception:
                            pass
                    rows.append({
                        "dataset": "bridge",
                        "root": root_name,
                        "py": py,
                        "round": r,
                        "quarter": q,
                        "exists": exists,
                        "n_rows": n_rows,
                        "n_cids": n_cids,
                        "n_branches": n_branches,
                    })
    return pl.DataFrame(rows)


# ── Limit coverage ────────────────────────────────────────────────────
def audit_limit() -> pl.DataFrame:
    """Per (PY, root): count market_months and rounds."""
    rows = []
    ds = DATASETS["limit"]
    for root_name, root_path in ROOTS.items():
        for py in ALL_PYS:
            base = f"{SPICE_BASE}/{ds}/{root_path}/auction_month={py}"
            if not _dir_exists(base):
                continue
            market_months = sorted(
                d.name.split("=")[1]
                for d in Path(base).iterdir()
                if d.is_dir() and d.name.startswith("market_month=")
            )
            # Check rounds in first month
            rounds_found = set()
            if market_months:
                mm_path = f"{base}/market_month={market_months[0]}"
                for d in Path(mm_path).iterdir():
                    if d.is_dir() and d.name.startswith("market_round="):
                        rounds_found.add(int(d.name.split("=")[1]))
            rows.append({
                "dataset": "limit",
                "root": root_name,
                "py": py,
                "n_months": len(market_months),
                "rounds": str(sorted(rounds_found)),
            })
    return pl.DataFrame(rows)


# ── SF coverage ───────────────────────────────────────────────────────
def audit_sf() -> pl.DataFrame:
    """Per (PY, root): count market_months and column count."""
    rows = []
    ds = DATASETS["sf"]
    for root_name, root_path in ROOTS.items():
        for py in ALL_PYS:
            base = f"{SPICE_BASE}/{ds}/{root_path}/auction_month={py}"
            if not _dir_exists(base):
                continue
            market_months = sorted(
                d.name.split("=")[1]
                for d in Path(base).iterdir()
                if d.is_dir() and d.name.startswith("market_month=")
            )
            n_cols = 0
            if market_months:
                try:
                    df = pl.read_parquet(f"{base}/market_month={market_months[0]}/", n_rows=1)
                    n_cols = len(df.columns)
                except Exception:
                    pass
            rows.append({
                "dataset": "sf",
                "root": root_name,
                "py": py,
                "n_months": len(market_months),
                "n_constraint_cols": n_cols - 1,  # minus pnode_id
            })
    return pl.DataFrame(rows)


# ── DA coverage ───────────────────────────────────────────────────────
def audit_da() -> pl.DataFrame:
    """Per year: row count and month range."""
    rows = []
    for d in sorted(Path(DA_BASE).iterdir()):
        if not d.is_dir() or not d.name.startswith("year="):
            continue
        year = d.name.split("=")[1]
        try:
            df = pl.scan_parquet(f"{d}/").select(
                pl.col("month").min().alias("min_month"),
                pl.col("month").max().alias("max_month"),
                pl.len().alias("n_rows"),
            ).collect()
            rows.append({
                "dataset": "da",
                "year": int(year),
                "min_month": int(df["min_month"][0]),
                "max_month": int(df["max_month"][0]),
                "n_rows": int(df["n_rows"][0]),
            })
        except Exception as e:
            rows.append({
                "dataset": "da",
                "year": int(year),
                "min_month": 0,
                "max_month": 0,
                "n_rows": 0,
            })
    return pl.DataFrame(rows)


def main():
    print("=" * 80)
    print("PJM Annual Signal — Data Coverage Audit")
    print("=" * 80)

    print("\n── DENSITY ─────────────────────────────────────────")
    df = audit_density()
    print(df.to_pandas().to_string(index=False))

    print("\n── BRIDGE (summary: exists per PY × round × quarter) ──")
    df_bridge = audit_bridge()
    # Summarize: per (root, py, round), count quarters that exist
    summary = (
        df_bridge.filter(pl.col("exists"))
        .group_by(["root", "py", "round"])
        .agg(
            pl.col("quarter").count().alias("n_quarters"),
            pl.col("n_cids").mean().alias("avg_cids"),
            pl.col("n_branches").mean().alias("avg_branches"),
        )
        .sort(["py", "round", "root"])
    )
    print(summary.to_pandas().to_string(index=False))

    print("\n── LIMIT ───────────────────────────────────────────")
    df = audit_limit()
    print(df.to_pandas().to_string(index=False))

    print("\n── SF ──────────────────────────────────────────────")
    df = audit_sf()
    print(df.to_pandas().to_string(index=False))

    print("\n── DA SHADOW PRICE ─────────────────────────────────")
    df = audit_da()
    print(df.to_pandas().to_string(index=False))

    print("\n" + "=" * 80)
    print("Audit complete.")


if __name__ == "__main__":
    main()
