from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from pbase.data.dataset.signal.general import ConstraintsSignal, ShiftFactorSignal

logger = logging.getLogger(__name__)

SPICE_BASE = "/opt/data/xyz-dataset/spice_data/pjm"
DA_BASE = "/opt/data/xyz-dataset/modeling_data/pjm/PJM_DA_SHADOW_PRICE.parquet"
V46_BASE = "/opt/data/xyz-dataset/signal_data/pjm/constraints"
ANNUAL_ROOT = "network_model=miso/spice_version=v6/auction_type=annual"
DENSITY_DS = "PJM_SPICE_DENSITY_DISTRIBUTION.parquet"
BRIDGE_DS = "PJM_SPICE_CONSTRAINT_INFO.parquet"
SF_DS = "PJM_SPICE_SF.parquet"
LIMIT_DS = "PJM_SPICE_CONSTRAINT_LIMIT.parquet"

PUBLISH_SIGNAL_PREFIX = "TEST.Signal.PJM.SPICE_ANNUAL_V7.0B"
BRIDGE_CTYPE = {"onpeak": "onpeak", "dailyoffpeak": "dailyoffpeak", "wkndonpeak": "onpeak"}

PUBLISHED_BINS = ["0", "60", "65", "70", "75", "80", "85", "90", "95", "100"]
RIGHT_TAIL_BINS = ["60", "65", "70", "75", "80", "85", "90", "95", "100"]
DEVIATION_WEIGHTS = np.array([7**i for i in range(len(RIGHT_TAIL_BINS))], dtype=float)
CONVENTION_THRESHOLD = 10
CONSTRAINT_INDEX_COLUMN = "__index_level_0__"
SF_INDEX_COLUMN = "pnode_id"
CONSTRAINT_COLUMNS = [
    "0_max",
    "60_max",
    "65_max",
    "70_max",
    "75_max",
    "80_max",
    "85_max",
    "90_max",
    "95_max",
    "100_max",
    "0_sum",
    "60_sum",
    "65_sum",
    "70_sum",
    "75_sum",
    "80_sum",
    "85_sum",
    "90_sum",
    "95_sum",
    "100_sum",
    "deviation_max",
    "deviation_sum",
    "flow_direction",
    "equipment",
    "convention",
    "shadow_sign",
    "shadow_price",
    "shadow_price_da",
    "constraint",
    "deviation_max_rank",
    "deviation_sum_rank",
    "shadow_rank",
    "rank",
    "tier",
    "constraint_limit",
]
REQUIRED_NON_NULL_COLUMNS = [
    "constraint",
    "equipment",
    "convention",
    "flow_direction",
    "shadow_sign",
    "shadow_price",
    "tier",
    "constraint_limit",
]


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_off_peak_day(the_date: date) -> bool:
    if the_date.weekday() > 4:
        return True
    if the_date.month == 1 and the_date.day == 1:
        return True
    if the_date.month == 1 and the_date.day == 2 and the_date.weekday() == 0:
        return True
    if the_date.month == 5 and the_date.weekday() == 0:
        return (the_date + timedelta(weeks=1)).month != 5
    if the_date.month == 7 and the_date.day == 4:
        return True
    if the_date.month == 7 and the_date.day == 5 and the_date.weekday() == 0:
        return True
    if the_date.month == 9 and the_date.weekday() == 0 and the_date.day < 8:
        return True
    if the_date.month == 11 and the_date.weekday() == 3 and (the_date.day - 1) // 7 + 1 == 4:
        return True
    if the_date.month == 12 and the_date.day == 25:
        return True
    if the_date.month == 12 and the_date.day == 26 and the_date.weekday() == 0:
        return True
    return False


def get_off_peak_dates(start_year: int, end_year: int) -> list[date]:
    out: list[date] = []
    cur = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    while cur <= end:
        if is_off_peak_day(cur):
            out.append(cur)
        cur += timedelta(days=1)
    return out


def load_da_filtered(
    start_year: int, start_month: int, end_year: int, end_month: int, class_type: str,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for year in range(start_year, end_year + 1):
        path = f"{DA_BASE}/year={year}/"
        if not Path(path).is_dir():
            continue
        table = ds.dataset(path, format="parquet").to_table(
            columns=["datetime_beginning_utc", "monitored_facility", "shadow_price"]
        )
        df = pl.from_arrow(table)
        ts_dtype = df.schema["datetime_beginning_utc"]
        if ts_dtype == pl.Datetime("ns"):
            df = df.with_columns(
                pl.col("datetime_beginning_utc").cast(pl.Datetime("us"))
            )
        elif ts_dtype == pl.Datetime("us", "US/Eastern"):
            df = df.with_columns(
                pl.col("datetime_beginning_utc").dt.replace_time_zone(None)
            )
        else:
            df = df.with_columns(
                pl.col("datetime_beginning_utc")
                .cast(pl.Datetime("us", "US/Eastern"))
                .dt.replace_time_zone(None)
            )
        frames.append(df)
    if not frames:
        return pl.DataFrame(
            {"datetime_beginning_utc": [], "monitored_facility": [], "shadow_price": [], "month": []}
        ).cast({"shadow_price": pl.Float64})

    da = pl.concat(frames)
    da = da.with_columns(pl.col("datetime_beginning_utc").dt.month().alias("month"))
    if start_year == end_year:
        da = da.filter((pl.col("month") >= start_month) & (pl.col("month") <= end_month))
    else:
        da = da.filter(
            ((pl.col("datetime_beginning_utc").dt.year() == start_year) & (pl.col("month") >= start_month))
            | ((pl.col("datetime_beginning_utc").dt.year() == end_year) & (pl.col("month") <= end_month))
            | (
                (pl.col("datetime_beginning_utc").dt.year() > start_year)
                & (pl.col("datetime_beginning_utc").dt.year() < end_year)
            )
        )
    off_peak_dates = get_off_peak_dates(start_year, end_year)
    da = da.with_columns(
        [
            pl.col("datetime_beginning_utc").dt.hour().alias("hb"),
            pl.col("datetime_beginning_utc").dt.weekday().alias("dow"),
            pl.col("datetime_beginning_utc").dt.date().alias("market_date"),
        ]
    )
    if class_type == "onpeak":
        da = da.filter(
            (pl.col("dow") <= 5)
            & (~pl.col("market_date").is_in(off_peak_dates))
            & (pl.col("hb") >= 7)
            & (pl.col("hb") <= 22)
        )
    elif class_type == "dailyoffpeak":
        da = da.filter((pl.col("hb") >= 23) | (pl.col("hb") <= 6))
    elif class_type == "wkndonpeak":
        da = da.filter(
            pl.col("market_date").is_in(off_peak_dates)
            & (pl.col("hb") >= 7)
            & (pl.col("hb") <= 22)
        )
    return da.drop(["hb", "dow", "market_date"])


def quarter_names() -> list[str]:
    return ["aq1", "aq2", "aq3", "aq4"]


def annual_market_months(py: str) -> list[str]:
    year = int(py[:4])
    return [f"{year}-{m:02d}" for m in range(6, 13)] + [f"{year + 1}-{m:02d}" for m in range(1, 6)]


def build_candidate_constraints(planning_year: str, market_round: int, class_type: str) -> pl.DataFrame:
    bridge_ctype = BRIDGE_CTYPE[class_type]
    quarter_frames: list[pl.DataFrame] = []
    for quarter in quarter_names():
        path = (
            f"{SPICE_BASE}/{BRIDGE_DS}/{ANNUAL_ROOT}/auction_month={planning_year}/"
            f"market_round={market_round}/period_type={quarter}/"
        )
        if not Path(path).is_dir():
            continue
        df = pl.read_parquet(path).filter(
            (pl.col("class_type") == bridge_ctype) & (pl.col("convention") < CONVENTION_THRESHOLD)
        )
        if len(df) == 0:
            continue
        quarter_frames.append(
            df.select(["constraint_id", "branch_name", "convention", "limit"])
            .rename({"limit": "bridge_limit"})
            .with_columns(
                pl.col("constraint_id").str.split(":").list.first().alias("constraint"),
                pl.col("constraint_id")
                .str.split(":")
                .list.first()
                .map_elements(normalize_ws, return_dtype=pl.Utf8)
                .alias("monitored"),
            )
        )
    if not quarter_frames:
        return pl.DataFrame(
            {
                "constraint_id": [],
                "branch_name": [],
                "convention": [],
                "bridge_limit": [],
                "constraint": [],
                "monitored": [],
                "constraint_limit": [],
            }
        )
    candidates = pl.concat(quarter_frames).unique(subset=["branch_name", "constraint_id"])
    limit_map = load_constraint_limits(planning_year, market_round)
    if len(limit_map) > 0:
        candidates = candidates.join(limit_map, on="constraint_id", how="left")
    else:
        candidates = candidates.with_columns(pl.lit(None).cast(pl.Float64).alias("limit"))
    return candidates.with_columns(
        pl.coalesce([pl.col("bridge_limit"), pl.col("limit")]).alias("constraint_limit")
    ).drop("limit")


def load_constraint_limits(planning_year: str, market_round: int) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for market_month in annual_market_months(planning_year):
        path = (
            f"{SPICE_BASE}/{LIMIT_DS}/{ANNUAL_ROOT}/auction_month={planning_year}/"
            f"market_month={market_month}/market_round={market_round}/"
        )
        if not Path(path).is_dir():
            continue
        df = pl.read_parquet(path).select(["constraint_id", "limit"])
        if len(df) > 0:
            frames.append(df)
    if not frames:
        return pl.DataFrame({"constraint_id": [], "limit": []}).cast({"limit": pl.Float64})
    return pl.concat(frames).group_by("constraint_id").agg(pl.col("limit").max())


def build_union_lookup(candidates: pl.DataFrame) -> dict[str, str]:
    unique_pairs = (
        candidates.group_by("monitored")
        .agg(pl.col("branch_name").n_unique().alias("n"), pl.col("branch_name").first().alias("branch_name"))
        .filter(pl.col("n") == 1)
    )
    return {row["monitored"]: row["branch_name"] for row in unique_pairs.iter_rows(named=True)}


def load_density_metrics(planning_year: str, market_round: int) -> pl.DataFrame:
    month_frames: list[pl.DataFrame] = []
    for market_month in annual_market_months(planning_year):
        path = (
            f"{SPICE_BASE}/{DENSITY_DS}/{ANNUAL_ROOT}/auction_month={planning_year}/market_month={market_month}/"
        )
        if not Path(path).is_dir():
            continue
        df = pl.read_parquet(path).filter(pl.col("market_round") == market_round)
        if len(df) == 0:
            continue
        month_frames.append(df.select(["constraint_id"] + PUBLISHED_BINS))
    if not month_frames:
        return pl.DataFrame({"constraint_id": []})

    density = pl.concat(month_frames)
    agg_exprs = []
    for b in PUBLISHED_BINS:
        agg_exprs.append(pl.col(b).max().alias(f"{b}_max"))
        agg_exprs.append(pl.col(b).sum().alias(f"{b}_sum"))
    metrics = density.group_by("constraint_id").agg(agg_exprs)

    max_score = np.zeros(len(metrics), dtype=float)
    sum_score = np.zeros(len(metrics), dtype=float)
    for i, b in enumerate(RIGHT_TAIL_BINS):
        max_score += metrics[f"{b}_max"].to_numpy() * DEVIATION_WEIGHTS[i]
        sum_score += metrics[f"{b}_sum"].to_numpy() * DEVIATION_WEIGHTS[i]

    metrics = metrics.with_columns(
        [
            pl.Series("deviation_max", np.log1p(max_score)),
            pl.Series("deviation_sum", np.log1p(sum_score)),
        ]
    )
    n = len(metrics)
    return metrics.with_columns(
        [
            (pl.col("deviation_max").rank(descending=True) / n).alias("deviation_max_rank"),
            (pl.col("deviation_sum").rank(descending=True) / n).alias("deviation_sum_rank"),
        ]
    )


def build_branch_scores(candidates: pl.DataFrame, density_metrics: pl.DataFrame, branch_da: pl.DataFrame) -> pl.DataFrame:
    cid_density = candidates.select(["constraint_id", "branch_name"]).join(density_metrics, on="constraint_id", how="left")
    branch_density = cid_density.group_by("branch_name").agg(pl.col("deviation_max").max().fill_null(0.0).alias("legacy_density"))
    merged = branch_density.join(branch_da.rename({"sp_abs": "da_sp"}), on="branch_name", how="full", coalesce=True)
    merged = merged.with_columns([pl.col("legacy_density").fill_null(0.0), pl.col("da_sp").fill_null(0.0)])
    n = max(len(merged), 1)
    merged = merged.with_columns(
        [
            (pl.col("legacy_density").rank(descending=True) / n).alias("density_rank"),
            (pl.col("da_sp").rank(descending=True) / n).alias("da_rank"),
        ]
    )
    merged = merged.with_columns(
        (pl.col("density_rank") * 0.3 + pl.col("da_rank") * 0.7).alias("signal_rank")
    )
    return merged.select(["branch_name", "signal_rank"])


def build_historical_da_metadata(
    planning_year: str, class_type: str, union_lookup: dict[str, str]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    year = int(planning_year[:4])
    da = load_da_filtered(year - 2, 6, year, 3, class_type)
    if len(da) == 0:
        empty_mon = pl.DataFrame({"monitored": [], "shadow_price_da": [], "signed_shadow_price_da": []}).cast(
            {"shadow_price_da": pl.Float64, "signed_shadow_price_da": pl.Float64}
        )
        empty_branch = pl.DataFrame({"branch_name": [], "sp_abs": []}).cast({"sp_abs": pl.Float64})
        return empty_mon, empty_branch
    da = da.with_columns(
        pl.col("monitored_facility")
        .map_elements(lambda x: normalize_ws(x.strip()), return_dtype=pl.Utf8)
        .alias("monitored")
    )
    monitored = da.group_by("monitored").agg(
        pl.col("shadow_price").abs().sum().alias("shadow_price_da"),
        pl.col("shadow_price").sum().alias("signed_shadow_price_da"),
    )
    branches = [union_lookup.get(m) for m in monitored["monitored"].to_list()]
    branch_da = monitored.with_columns(pl.Series("branch_name", branches)).filter(pl.col("branch_name").is_not_null())
    branch_da = branch_da.group_by("branch_name").agg(pl.col("shadow_price_da").sum().alias("sp_abs"))
    return monitored, branch_da


def load_v46_metadata(planning_year: str, market_round: int, class_type: str) -> pl.DataFrame:
    path = f"{V46_BASE}/TEST.Signal.PJM.SPICE_ANNUAL_V4.6.R{market_round}/{planning_year}/a/{class_type}/"
    if not Path(path).is_dir():
        return pl.DataFrame(
            {"constraint_id": [], "v46_shadow_sign": [], "v46_constraint": [], "v46_equipment": [], "v46_convention": []}
        )
    df = pl.read_parquet(path)
    return (
        df.with_columns(pl.col(CONSTRAINT_INDEX_COLUMN).str.split("|").list.first().alias("constraint_id"))
        .select(
            [
                "constraint_id",
                pl.col("shadow_sign").alias("v46_shadow_sign"),
                pl.col("constraint").alias("v46_constraint"),
                pl.col("equipment").alias("v46_equipment"),
                pl.col("convention").alias("v46_convention"),
            ]
        )
        .unique(subset=["constraint_id"])
    )


def select_candidates(
    candidates: pl.DataFrame,
    density_metrics: pl.DataFrame,
    monitored_da: pl.DataFrame,
    branch_scores: pl.DataFrame,
    v46_meta: pl.DataFrame,
    tier_sizes: list[int],
    branch_cap: int,
) -> pl.DataFrame:
    target = sum(tier_sizes)
    df = (
        candidates.join(density_metrics, on="constraint_id", how="left")
        .join(monitored_da, on="monitored", how="left")
        .join(branch_scores, on="branch_name", how="left")
        .join(v46_meta, on="constraint_id", how="left")
        .with_columns(
            [
                pl.col("shadow_price_da").fill_null(0.0),
                pl.col("signed_shadow_price_da").fill_null(0.0),
                pl.col("signal_rank").fill_null(1.0),
            ]
        )
    )
    sign_expr = (
        pl.when(pl.col("signed_shadow_price_da") > 0)
        .then(pl.lit(1))
        .when(pl.col("signed_shadow_price_da") < 0)
        .then(pl.lit(-1))
        .when(pl.col("v46_shadow_sign").is_not_null())
        .then(pl.col("v46_shadow_sign"))
        .otherwise(pl.lit(-1))
    )
    df = df.with_columns(
        [
            sign_expr.alias("shadow_sign"),
            (-sign_expr).alias("flow_direction"),
            sign_expr.cast(pl.Float64).alias("shadow_price"),
            (pl.col("branch_name") + "," + pl.col("constraint")).alias("equipment"),
        ]
    )
    n = max(len(df), 1)
    df = df.with_columns(
        [
            (pl.col("shadow_price_da").rank(descending=True) / n).alias("shadow_rank"),
            (pl.col("signal_rank").rank() / n).alias("rank"),
        ]
    )
    # Return candidates sorted by rank with branch_cap applied, but with a generous
    # buffer (3x target) so finalize_publication can replace SF-dropped rows.
    sort_df = df.sort(["rank", "shadow_price_da", "constraint_id"], descending=[False, True, False])
    buffer = target * 3
    rows: list[dict] = []
    branch_counts: dict[str, int] = {}
    for row in sort_df.iter_rows(named=True):
        if len(rows) >= buffer:
            break
        branch = row["branch_name"]
        if branch_counts.get(branch, 0) >= branch_cap:
            continue
        rows.append(row)
        branch_counts[branch] = branch_counts.get(branch, 0) + 1
    return pl.DataFrame(rows)


def load_annual_sf(planning_year: str, market_round: int, constraint_ids: list[str]) -> pd.DataFrame:
    sums: pd.DataFrame | None = None
    counts: dict[str, int] = {cid: 0 for cid in constraint_ids}
    for market_month in annual_market_months(planning_year):
        month_round_path = (
            f"{SPICE_BASE}/{SF_DS}/{ANNUAL_ROOT}/auction_month={planning_year}/"
            f"market_month={market_month}/market_round={market_round}/"
        )
        if not Path(month_round_path).is_dir():
            continue
        files = sorted(Path(month_round_path).glob("outage_date=*/*.parquet"))
        for file_path in files:
            schema_names = pq.ParquetFile(file_path).schema.names
            available = [c for c in constraint_ids if c in schema_names]
            if not available:
                continue
            table = pq.read_table(file_path, columns=[SF_INDEX_COLUMN] + available)
            outage_df = table.to_pandas().set_index(SF_INDEX_COLUMN)
            if sums is None:
                sums = outage_df
            else:
                sums = sums.add(outage_df, fill_value=0.0)
            for cid in available:
                counts[cid] += 1
    if sums is None:
        raise ValueError("No SF data found for requested constraints.")
    for cid in list(sums.columns):
        if counts.get(cid, 0) == 0:
            sums.drop(columns=[cid], inplace=True)
        else:
            sums[cid] = sums[cid] / counts[cid]
    sums = sums.astype("float64")
    sums.index.name = SF_INDEX_COLUMN
    return sums


def finalize_publication(
    selected: pl.DataFrame,
    sf_avg: pd.DataFrame,
    tier_sizes: list[int],
    branch_cap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sort_df = selected.sort(["rank", "shadow_price_da", "constraint_id"], descending=[False, True, False])
    rows: list[dict] = []
    branch_counts: dict[str, int] = {}
    target = sum(tier_sizes)
    for row in sort_df.iter_rows(named=True):
        if len(rows) >= target:
            break
        cid = row["constraint_id"]
        branch = row["branch_name"]
        if cid not in sf_avg.columns:
            continue
        if float(sf_avg[cid].abs().sum()) == 0.0:
            continue
        if branch_counts.get(branch, 0) >= branch_cap:
            continue
        rows.append(row)
        branch_counts[branch] = branch_counts.get(branch, 0) + 1
    published = pl.DataFrame(rows)
    if len(published) == 0:
        raise ValueError("No publishable constraints after SF filtering.")
    tiers = []
    idx = 0
    count = 0
    for _ in range(len(published)):
        tiers.append(idx)
        count += 1
        if idx < len(tier_sizes) and count >= tier_sizes[idx]:
            idx += 1
            count = 0
    published = published.with_columns(
        [
            pl.Series("tier", tiers).cast(pl.Int64),
            (pl.col("constraint_id") + "|" + pl.col("shadow_sign").cast(pl.Utf8) + "|spice").alias(
                CONSTRAINT_INDEX_COLUMN
            ),
        ]
    )
    for col in CONSTRAINT_COLUMNS:
        if col not in published.columns:
            published = published.with_columns(pl.lit(None).alias(col))
    constraints_out = published.select(CONSTRAINT_COLUMNS + [CONSTRAINT_INDEX_COLUMN]).to_pandas()
    constraints_out = constraints_out.set_index(CONSTRAINT_INDEX_COLUMN)
    for col in REQUIRED_NON_NULL_COLUMNS:
        if constraints_out[col].isna().any():
            raise ValueError(f"Nulls found in required column {col}.")
    sf_cols = published["constraint_id"].to_list()
    sf_out = sf_avg[sf_cols].copy()
    rename_map = dict(zip(published["constraint_id"].to_list(), published[CONSTRAINT_INDEX_COLUMN].to_list()))
    sf_out = sf_out.rename(columns=rename_map)
    sf_out.index.name = SF_INDEX_COLUMN
    return constraints_out, sf_out


def publish_signal(
    planning_year: str,
    market_round: int,
    class_type: str,
    tier_sizes: list[int] | None = None,
    branch_cap: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if tier_sizes is None:
        tier_sizes = [200, 200, 200, 200, 200]
    candidates = build_candidate_constraints(planning_year, market_round, class_type)
    if len(candidates) == 0:
        raise ValueError(f"No candidate constraints for {planning_year} R{market_round} {class_type}.")
    union_lookup = build_union_lookup(candidates)
    density_metrics = load_density_metrics(planning_year, market_round)
    monitored_da, branch_da = build_historical_da_metadata(planning_year, class_type, union_lookup)
    branch_scores = build_branch_scores(candidates, density_metrics, branch_da)
    v46_meta = load_v46_metadata(planning_year, market_round, class_type)
    selected = select_candidates(
        candidates=candidates,
        density_metrics=density_metrics,
        monitored_da=monitored_da,
        branch_scores=branch_scores,
        v46_meta=v46_meta,
        tier_sizes=tier_sizes,
        branch_cap=branch_cap,
    )
    probe_ids = selected["constraint_id"].to_list()
    sf_avg = load_annual_sf(planning_year, market_round, probe_ids)
    return finalize_publication(selected, sf_avg, tier_sizes, branch_cap)


def save_signal(
    planning_year: str,
    market_round: int,
    class_type: str,
    constraints_df: pd.DataFrame,
    sf_df: pd.DataFrame,
    dry_run: bool = False,
    is_dev: bool = False,
) -> tuple[str, str]:
    signal_name = f"{PUBLISH_SIGNAL_PREFIX}.R{market_round}"
    auction_month = pd.Timestamp(planning_year)
    constraints_loader = ConstraintsSignal(
        rto="pjm", signal_name=signal_name, period_type="a", class_type=class_type, is_dev=is_dev
    )
    sf_loader = ShiftFactorSignal(
        rto="pjm", signal_name=signal_name, period_type="a", class_type=class_type, is_dev=is_dev
    )
    constraints_path = constraints_loader.save_data(constraints_df, auction_month=auction_month, dry_run=dry_run)
    sf_path = sf_loader.save_data(sf_df, auction_month=auction_month, dry_run=dry_run)
    return constraints_path, sf_path
