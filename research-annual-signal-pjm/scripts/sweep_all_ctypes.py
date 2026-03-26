"""Full sweep: all PYs × rounds × class types, ctype-filtered DA, March cutoff.

Quarter-aware mapping: DA and GT are mapped per quarter through that quarter's
bridge partition, then summed to annual at the branch level. This is correct
because PJM bridge membership varies by aq1-aq4.

Class types (EPT hour_beginning):
  onpeak:       weekday HB07-HB22
  dailyoffpeak: every day HB23 + HB00-HB06
  wkndonpeak:   weekend HB07-HB22 (NERC holidays NOT yet handled — known approximation)

Signals evaluated:
  baseline_v69: 2-component density + DA blend
  candidate_v70: 3-component flow-stress max + flow-stress persistence + DA pressure

Run: cd /home/xyz/workspace/pmodel && source .venv/bin/activate && python /home/xyz/workspace/research-qianli-v2/research-annual-signal-pjm/scripts/sweep_all_ctypes.py
"""
from __future__ import annotations

import re
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

SPICE_BASE = "/opt/data/xyz-dataset/spice_data/pjm"
DA_BASE = "/opt/data/xyz-dataset/modeling_data/pjm/PJM_DA_SHADOW_PRICE.parquet"
V46_BASE = "/opt/data/xyz-dataset/signal_data/pjm/constraints"
DENSITY_DS = "PJM_SPICE_DENSITY_DISTRIBUTION.parquet"
BRIDGE_DS = "PJM_SPICE_CONSTRAINT_INFO.parquet"
ANNUAL_ROOT = "network_model=miso/spice_version=v6/auction_type=annual"

RIGHT_TAIL_BINS = ["60", "65", "70", "75", "80", "85", "90", "95", "100"]
DEVIATION_WEIGHTS = np.array([7**i for i in range(len(RIGHT_TAIL_BINS))])
CONVENTION_THRESHOLD = 10
FLOW_STRESS_MAX_BINS = ["90", "95", "100"]
FLOW_STRESS_PERSISTENCE_BINS = ["80", "85", "90", "95", "100"]

# Bridge class_type: wkndonpeak uses the onpeak bridge
BRIDGE_CTYPE = {"onpeak": "onpeak", "dailyoffpeak": "dailyoffpeak", "wkndonpeak": "onpeak"}

# Quarter → settlement months
def get_quarter_months(py: str) -> dict[str, list[tuple[int, int]]]:
    year = int(py[:4])
    return {
        "aq1": [(year, 6), (year, 7), (year, 8)],
        "aq2": [(year, 9), (year, 10), (year, 11)],
        "aq3": [(year, 12), (year + 1, 1), (year + 1, 2)],
        "aq4": [(year + 1, 3), (year + 1, 4), (year + 1, 5)],
    }


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def is_off_peak_day(the_date: date) -> bool:
    """Mirror pbase.utils.hours.is_off_peak_day() exactly for PJM."""
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
    off_peak_dates: list[date] = []
    cur = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    while cur <= end:
        if is_off_peak_day(cur):
            off_peak_dates.append(cur)
        cur += timedelta(days=1)
    return off_peak_dates


# ── DA loading with peak-type filter ──────────────────────────────────
def load_da_filtered(
    start_year: int, start_month: int, end_year: int, end_month: int, class_type: str,
) -> pl.DataFrame:
    """Load DA filtered to specific class type hours. Returns raw rows."""
    frames = []
    for year in range(start_year, end_year + 1):
        path = f"{DA_BASE}/year={year}/"
        if not Path(path).is_dir():
            continue
        table = pq.read_table(path, columns=[
            "datetime_beginning_utc", "monitored_facility", "shadow_price", "month",
        ])
        df = pl.from_arrow(table)
        if year == start_year:
            df = df.filter(pl.col("month") >= start_month)
        if year == end_year:
            df = df.filter(pl.col("month") <= end_month)
        frames.append(df)
    if not frames:
        return pl.DataFrame({
            "monitored_facility": [], "shadow_price": [], "month": [],
            "datetime_beginning_utc": [],
        }).cast({"shadow_price": pl.Float64})
    normalized = []
    for f in frames:
        f = f.with_columns(
            pl.col("datetime_beginning_utc").cast(pl.Datetime("us", "US/Eastern"))
        )
        normalized.append(f)
    da = pl.concat(normalized)
    off_peak_dates = get_off_peak_dates(start_year, end_year)
    da = da.with_columns([
        pl.col("datetime_beginning_utc").dt.hour().alias("hb"),
        pl.col("datetime_beginning_utc").dt.weekday().alias("dow"),  # 1=Mon..7=Sun
        pl.col("datetime_beginning_utc").dt.date().alias("market_date"),
    ])
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


# ── Bridge: per-quarter monitored → branch lookup ─────────────────────
def build_quarter_lookup(py: str, rnd: int, quarter: str, class_type: str) -> dict[str, str]:
    """Build monitored → branch for ONE quarter."""
    bridge_ctype = BRIDGE_CTYPE[class_type]
    path = f"{SPICE_BASE}/{BRIDGE_DS}/{ANNUAL_ROOT}/auction_month={py}/market_round={rnd}/period_type={quarter}/"
    if not Path(path).is_dir():
        return {}
    df = pl.read_parquet(path)
    df = df.filter(
        (pl.col("class_type") == bridge_ctype) & (pl.col("convention") < CONVENTION_THRESHOLD)
    )
    df = df.with_columns(
        pl.col("constraint_id").str.split(":").list.first().str.strip_chars().alias("monitored")
    )
    mon_br = df.group_by("monitored").agg(
        pl.col("branch_name").n_unique().alias("n"),
        pl.col("branch_name").first().alias("branch_name"),
    )
    return {
        normalize_ws(row["monitored"]): row["branch_name"]
        for row in mon_br.filter(pl.col("n") == 1).iter_rows(named=True)
    }


def build_union_lookup(py: str, rnd: int, class_type: str) -> dict[str, str]:
    """Union of all quarter lookups. For density mapping (not quarter-specific).

    Verified: all 4 quarters have identical monitored → branch mappings (6431 each,
    0 conflicts, 0 quarter-only lines for PY=2024-06 R2 onpeak). Quarter sensitivity
    is at the constraint_id level (different monitored:contingency pairs per quarter),
    not at the monitored-line → branch level. So union is exact for density.
    """
    union: dict[str, str] = {}
    for q in ["aq1", "aq2", "aq3", "aq4"]:
        qlookup = build_quarter_lookup(py, rnd, q, class_type)
        for k, v in qlookup.items():
            if k not in union:
                union[k] = v
    return union


# ── Quarter-aware DA → branch mapping ─────────────────────────────────
def map_da_quarterly(
    da: pl.DataFrame, py: str, rnd: int, class_type: str,
) -> pl.DataFrame:
    """Map DA to branches per quarter using that quarter's bridge, then sum."""
    quarter_months = get_quarter_months(py)
    branch_frames = []
    for q, ym_list in quarter_months.items():
        # Filter DA to this quarter's months
        # Build per-month filter (handles year boundary at aq3/aq4)
        q_da = da.filter(pl.lit(False))  # start empty
        for y, m in ym_list:
            chunk = da.filter(
                (pl.col("datetime_beginning_utc").dt.year() == y)
                & (pl.col("month") == m)
            )
            if len(chunk) > 0:
                q_da = pl.concat([q_da, chunk])
        if len(q_da) == 0:
            continue
        # Map through this quarter's bridge
        qlookup = build_quarter_lookup(py, rnd, q, class_type)
        if not qlookup:
            continue
        branches = [
            qlookup.get(normalize_ws(m.strip()))
            for m in q_da["monitored_facility"].to_list()
        ]
        q_da = q_da.with_columns(pl.Series("branch_name", branches))
        q_da = q_da.filter(pl.col("branch_name").is_not_null())
        if len(q_da) > 0:
            branch_agg = q_da.group_by("branch_name").agg(
                pl.col("shadow_price").abs().sum().alias("sp")
            )
            branch_frames.append(branch_agg)
    if not branch_frames:
        return pl.DataFrame({"branch_name": [], "sp": []}).cast({"branch_name": pl.Utf8, "sp": pl.Float64})
    # Sum across quarters
    all_branches = pl.concat(branch_frames)
    return all_branches.group_by("branch_name").agg(pl.col("sp").sum())


def map_da_union(da: pl.DataFrame, union_lookup: dict[str, str]) -> pl.DataFrame:
    """Map DA to branches through the evaluation cell's monitored-line union lookup.

    Used for historical DA features. The March-cutoff lookback window does not align to the
    evaluation planning year's aq1-aq4 settlement months, so quarter-month filtering would
    incorrectly drop the entire lookback history.
    """
    if len(da) == 0 or not union_lookup:
        return pl.DataFrame({"branch_name": [], "sp": []}).cast({"branch_name": pl.Utf8, "sp": pl.Float64})
    branches = [
        union_lookup.get(normalize_ws(m.strip()))
        for m in da["monitored_facility"].to_list()
    ]
    mapped = da.with_columns(pl.Series("branch_name", branches)).filter(pl.col("branch_name").is_not_null())
    if len(mapped) == 0:
        return pl.DataFrame({"branch_name": [], "sp": []}).cast({"branch_name": pl.Utf8, "sp": pl.Float64})
    return mapped.group_by("branch_name").agg(pl.col("shadow_price").abs().sum().alias("sp"))


# ── Density → branch (uses union lookup, not quarter-specific) ────────
def compute_density_branches(py: str, rnd: int, union_lookup: dict[str, str]) -> pl.DataFrame:
    year = int(py[:4])
    frames = []
    for y, months in [(year, range(6, 13)), (year + 1, range(1, 6))]:
        for m in months:
            path = f"{SPICE_BASE}/{DENSITY_DS}/{ANNUAL_ROOT}/auction_month={py}/market_month={y}-{m:02d}/"
            if not Path(path).is_dir():
                continue
            df = pl.read_parquet(path).filter(pl.col("market_round") == rnd)
            if len(df) > 0:
                frames.append(df)
    if not frames:
        return pl.DataFrame({"branch_name": [], "dev_max": []}).cast({"dev_max": pl.Float64})
    density = pl.concat(frames)
    agg_exprs = [pl.col(b).max().alias(f"{b}_max") for b in RIGHT_TAIL_BINS]
    cid_agg = density.group_by("constraint_id").agg(agg_exprs)

    legacy_scores = np.zeros(len(cid_agg))
    for i, b in enumerate(RIGHT_TAIL_BINS):
        legacy_scores += cid_agg[f"{b}_max"].to_numpy() * DEVIATION_WEIGHTS[i]

    flow_stress_max = np.max(
        np.column_stack([cid_agg[f"{b}_max"].to_numpy() for b in FLOW_STRESS_MAX_BINS]),
        axis=1,
    )
    flow_stress_persistence = np.sum(
        np.column_stack([cid_agg[f"{b}_max"].to_numpy() for b in FLOW_STRESS_PERSISTENCE_BINS]),
        axis=1,
    )

    cid_agg = cid_agg.with_columns([
        pl.Series("legacy_density", np.log1p(legacy_scores)),
        pl.Series("flow_stress_max", flow_stress_max),
        pl.Series("flow_stress_persistence", np.log1p(flow_stress_persistence)),
    ])
    cid_agg = cid_agg.with_columns(
        pl.col("constraint_id").str.split(":").list.first().str.strip_chars().alias("monitored")
    )
    branches = [union_lookup.get(normalize_ws(m)) for m in cid_agg["monitored"].to_list()]
    cid_agg = cid_agg.with_columns(pl.Series("branch_name", branches))
    mapped = cid_agg.filter(pl.col("branch_name").is_not_null())
    return mapped.group_by("branch_name").agg(
        pl.col("legacy_density").max(),
        pl.col("flow_stress_max").max(),
        pl.col("flow_stress_persistence").max(),
    )


# ── Signals ───────────────────────────────────────────────────────────
def _rank_pct_desc(expr: pl.Expr, n: int, alias: str) -> pl.Expr:
    return (expr.rank(descending=True) / n).alias(alias)


def score_baseline_v69(branch_density: pl.DataFrame, branch_da: pl.DataFrame) -> pl.DataFrame:
    merged = branch_density.join(
        branch_da.rename({"sp": "da_sp"}), on="branch_name", how="full", coalesce=True,
    )
    merged = merged.with_columns([
        pl.col("legacy_density").fill_null(0.0), pl.col("da_sp").fill_null(0.0),
    ])
    n = len(merged)
    merged = merged.with_columns([
        _rank_pct_desc(pl.col("legacy_density"), n, "density_rank"),
        _rank_pct_desc(pl.col("da_sp"), n, "da_rank"),
    ])
    merged = merged.with_columns(
        (pl.col("density_rank") * 0.3 + pl.col("da_rank") * 0.7).alias("signal_rank")
    )
    return merged.with_columns(
        pl.col("signal_rank").rank().cast(pl.Int32).alias("rank")
    ).sort("rank")


def score_candidate_v70(branch_density: pl.DataFrame, branch_da: pl.DataFrame) -> pl.DataFrame:
    merged = branch_density.join(
        branch_da.rename({"sp": "da_sp"}), on="branch_name", how="full", coalesce=True,
    )
    merged = merged.with_columns([
        pl.col("flow_stress_max").fill_null(0.0),
        pl.col("flow_stress_persistence").fill_null(0.0),
        pl.col("da_sp").fill_null(0.0),
    ])
    n = len(merged)
    merged = merged.with_columns([
        _rank_pct_desc(pl.col("flow_stress_max"), n, "flow_stress_max_rank"),
        _rank_pct_desc(pl.col("flow_stress_persistence"), n, "flow_stress_persistence_rank"),
        _rank_pct_desc(pl.col("da_sp"), n, "da_pressure_rank"),
    ])
    merged = merged.with_columns(
        (
            pl.col("flow_stress_max_rank") * 0.3
            + pl.col("flow_stress_persistence_rank") * 0.2
            + pl.col("da_pressure_rank") * 0.5
        ).alias("signal_rank")
    )
    return merged.with_columns(
        pl.col("signal_rank").rank().cast(pl.Int32).alias("rank")
    ).sort("rank")


# ── V4.6 ──────────────────────────────────────────────────────────────
def load_v46(py: str, rnd: int, class_type: str) -> pl.DataFrame | None:
    path = f"{V46_BASE}/TEST.Signal.PJM.SPICE_ANNUAL_V4.6.R{rnd}/{py}/a/{class_type}/"
    if not Path(path).is_dir():
        return None
    df = pl.read_parquet(path)
    df = df.with_columns(
        pl.col("equipment").str.split(",").list.first().alias("branch_name")
    )
    return df.with_columns(pl.col("rank").rank().cast(pl.Int32).alias("v46_rank")).sort("v46_rank")


# ── Metric ────────────────────────────────────────────────────────────
def abs_sp_at_k(gt: pl.DataFrame, branches: list[str], total_sp: float) -> float:
    if total_sp <= 0:
        return 0.0
    return float(gt.filter(pl.col("branch_name").is_in(branches))["sp"].sum()) / total_sp


def gt_branch_recall(gt: pl.DataFrame, signal_branches: set[str]) -> float:
    """Fraction of GT branches that exist in the signal's branch universe (GT branch recall)."""
    if len(gt) == 0:
        return 0.0
    return len(set(gt["branch_name"].to_list()) & signal_branches) / len(gt)


# ── Main ──────────────────────────────────────────────────────────────
def main():
    pys = ["2019-06", "2020-06", "2021-06", "2022-06", "2023-06", "2024-06", "2025-06"]
    rounds = [1, 2, 3, 4]
    ctypes = ["onpeak", "dailyoffpeak", "wkndonpeak"]
    results = []
    signal_builders = {
        "baseline_v69": score_baseline_v69,
        "candidate_v70": score_candidate_v70,
    }

    for ctype in ctypes:
        print(f"\n{'#'*90}")
        print(f"  CLASS TYPE: {ctype}")
        print(f"{'#'*90}")
        for py in pys:
            year = int(py[:4])
            for rnd in rounds:
                t0 = time.time()

                # Union lookup for density (not quarter-specific)
                union_lookup = build_union_lookup(py, rnd, ctype)
                if not union_lookup:
                    continue

                # Density features
                bd = compute_density_branches(py, rnd, union_lookup)
                if len(bd) == 0:
                    continue

                # DA feature: 2-year lookback, ctype-filtered, March cutoff.
                # Historical DA is mapped through the evaluation cell's monitored-line
                # union lookup; quarter-month filtering is only correct for GT.
                da_feat = load_da_filtered(year - 2, 6, year, 3, ctype)
                branch_da = map_da_union(da_feat, union_lookup)

                # GT: ctype-filtered DA for PY settlement months
                # Same quarter-aware mapping
                gt_da = load_da_filtered(year, 6, year + 1, 5, ctype)
                total_sp = float(gt_da["shadow_price"].abs().sum())
                gt = map_da_quarterly(gt_da, py, rnd, ctype)
                if total_sp == 0:
                    continue

                mapped_sp = float(gt["sp"].sum())
                gt_coverage = mapped_sp / total_sp if total_sp > 0 else 0.0

                # V4.6
                v46 = load_v46(py, rnd, ctype)
                v46_lbl_cov = 0.0
                if v46 is not None:
                    v46_branches = set(v46["branch_name"].to_list())
                    v46_lbl_cov = gt_branch_recall(gt, v46_branches)

                for signal_name, builder in signal_builders.items():
                    signal = builder(bd, branch_da)
                    signal_branches = set(signal["branch_name"].to_list())
                    our_lbl_cov = gt_branch_recall(gt, signal_branches)

                    for k in [200, 400]:
                        ours = abs_sp_at_k(gt, signal.head(k)["branch_name"].to_list(), total_sp)
                        v46s = 0.0
                        if v46 is not None:
                            v46s = abs_sp_at_k(gt, v46.head(k)["branch_name"].to_list(), total_sp)
                        results.append({
                            "signal_name": signal_name,
                            "ctype": ctype,
                            "py": py,
                            "round": rnd,
                            "k": k,
                            "ours": ours * 100,
                            "v46": v46s * 100,
                            "delta": (ours - v46s) * 100,
                            "total_sp": total_sp,
                            "gt_coverage": gt_coverage * 100,
                            "our_lbl_cov": our_lbl_cov * 100,
                            "v46_lbl_cov": v46_lbl_cov * 100,
                            "signal_n": len(signal),
                            "gt_branches": len(gt),
                        })

                elapsed = time.time() - t0
                summary_signal = score_candidate_v70(bd, branch_da)
                summary_lbl_cov = gt_branch_recall(gt, set(summary_signal["branch_name"].to_list()))
                print(
                    f"  {py} R{rnd} {ctype}: {elapsed:.1f}s  "
                    f"gt_cov={gt_coverage:.1%}  v70_lbl={summary_lbl_cov:.1%}  "
                    f"v46_lbl={v46_lbl_cov:.1%}"
                )

    df = pl.DataFrame(results)

    for signal_name in signal_builders:
        print(f'\n{"#"*100}')
        print(f"  SIGNAL: {signal_name}")
        print(f"{'#'*100}")
        for ctype in ctypes:
            for k in [200, 400]:
                kdf = df.filter(
                    (pl.col("signal_name") == signal_name)
                    & (pl.col("k") == k)
                    & (pl.col("ctype") == ctype)
                )
                print(f'\n{"="*100}')
                print(
                    f"Abs_SP@{k} — {ctype.upper()} — {signal_name} vs V4.6  "
                    "(gt_cov=GT mapping %, lbl=GT branch recall %)"
                )
                print(f'{"="*100}')
                for py in pys:
                    pydf = kdf.filter(pl.col("py") == py)
                    line = f"{py}  "
                    for rnd in rounds:
                        rdf = pydf.filter(pl.col("round") == rnd)
                        if len(rdf) == 0:
                            line += f"R{rnd}: ---            "
                        else:
                            r = rdf.row(0, named=True)
                            line += (
                                f"R{rnd}: {r['ours']:5.1f} vs {r['v46']:5.1f} ({r['delta']:+5.1f})  "
                            )
                    print(line)

    # Coverage summary table
    print(f'\n{"="*100}')
    print("COVERAGE SUMMARY (per signal × ctype, averaged across PYs × rounds)")
    print(f'{"="*100}')
    for signal_name in signal_builders:
        for ctype in ctypes:
            cdf = df.filter(
                (pl.col("signal_name") == signal_name)
                & (pl.col("ctype") == ctype)
                & (pl.col("k") == 200)
            )
            print(
                f"  {signal_name:14s} {ctype:15s}  gt_mapping={cdf['gt_coverage'].mean():.1f}%  "
                f"our_recall={cdf['our_lbl_cov'].mean():.1f}%  "
                f"v46_recall={cdf['v46_lbl_cov'].mean():.1f}%  "
                f"avg_gt_branches={cdf['gt_branches'].mean():.0f}"
            )

    # Summary: average delta per signal × ctype × K
    print(f'\n{"="*100}')
    print("SUMMARY: Average delta (signal - V4.6) across all PYs × rounds")
    print(f'{"="*100}')
    for signal_name in signal_builders:
        for ctype in ctypes:
            for k in [200, 400]:
                kdf = df.filter(
                    (pl.col("signal_name") == signal_name)
                    & (pl.col("k") == k)
                    & (pl.col("ctype") == ctype)
                    & (pl.col("v46") > 0)
                )
                avg_delta = kdf["delta"].mean() if len(kdf) > 0 else 0
                n_cells = len(kdf)
                wins = (kdf["delta"] > 0).sum()
                print(
                    f"  {signal_name:14s} {ctype:15s} @{k}: "
                    f"avg delta={avg_delta:+5.1f}pp  wins={wins}/{n_cells}"
                )

    # Direct comparison: candidate_v70 minus baseline_v69
    print(f'\n{"="*100}')
    print("SUMMARY: candidate_v70 minus baseline_v69 (positive = improvement)")
    print(f'{"="*100}')
    for ctype in ctypes:
        for k in [200, 400]:
            v70 = df.filter(
                (pl.col("signal_name") == "candidate_v70")
                & (pl.col("ctype") == ctype)
                & (pl.col("k") == k)
            ).sort(["py", "round"])
            v69 = df.filter(
                (pl.col("signal_name") == "baseline_v69")
                & (pl.col("ctype") == ctype)
                & (pl.col("k") == k)
            ).sort(["py", "round"])
            if len(v70) == 0 or len(v69) == 0:
                continue
            delta_vs_baseline = float((v70["ours"] - v69["ours"]).mean())
            wins_vs_baseline = int((v70["ours"] > v69["ours"]).sum())
            n_cells = min(len(v70), len(v69))
            print(
                f"  {ctype:15s} @{k}: "
                f"avg improvement={delta_vs_baseline:+5.1f}pp  wins={wins_vs_baseline}/{n_cells}"
            )


if __name__ == "__main__":
    main()
