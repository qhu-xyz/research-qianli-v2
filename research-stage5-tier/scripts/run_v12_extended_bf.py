#!/usr/bin/env python
"""V12: Extended BF — blend partial-month data INTO existing bf windows.

Instead of adding bf_partial as a separate feature (v11a), this modifies
bf_1..bf_15 to include the first N days of the cutoff month as a fractional
month contribution. This makes existing features strictly more informative.

Key insight: bf_partial as a standalone feature got only 3% importance because
the model couldn't figure out how to combine it with bf_1. Blending the partial
data directly into bf_1..bf_15 avoids this dilution problem.

Formula:
  bf_ext = (full_months_bound + days_binding/30) / (n_full_months + n_days/30)

Where days_binding = # of partial days the constraint bound (out of n_days),
and 30 is the normalizing constant (approximate days per month).

v12a: 12-day partial window, extended bf_1..bf_15 (same 9 features as v10e)
Baseline: v10e-lag1 (9 features, lag=1)
"""
from __future__ import annotations

import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    REALIZED_DA_CACHE, LTRConfig, PipelineConfig,
    _FULL_EVAL_MONTHS, _SCREEN_EVAL_MONTHS,
    collect_usable_months, has_period_type,
)
from ml.data_loader import load_v62b_month
from ml.evaluate import aggregate_months, evaluate_ltr
from ml.features import compute_query_groups, prepare_features
from ml.realized_da import fetch_partial_month_da
from ml.registry_paths import registry_root, holdout_root
from ml.train import predict_scores, train_ltr_model

ROOT = Path(__file__).resolve().parent.parent

BLEND_WEIGHTS: dict[tuple[str, str], tuple[float, float, float]] = {
    ("f0", "onpeak"): (0.85, 0.00, 0.15),
    ("f0", "offpeak"): (0.85, 0.00, 0.15),
    ("f1", "onpeak"): (0.70, 0.00, 0.30),
    ("f1", "offpeak"): (0.80, 0.00, 0.20),
}
V7_DA, V7_DMIX, V7_DORI = 0.85, 0.00, 0.15

# Same 9 features as v10e-lag1 — no new features, just better BF values
V12_FEATURES = [
    "binding_freq_1", "binding_freq_3", "binding_freq_6", "binding_freq_12",
    "binding_freq_15", "v7_formula_score", "prob_exceed_110", "constraint_limit",
    "da_rank_value",
]
V12_MONOTONE = [1, 1, 1, 1, 1, -1, 1, 0, -1]

HOLDOUT_MONTHS = [f"{y:04d}-{m:02d}" for y in (2024, 2025) for m in range(1, 13)]

PARTIAL_N_DAYS = 12
DAYS_PER_MONTH = 30.0  # normalizing constant for fractional month weight


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def prev_month(m: str) -> str:
    ts = pd.Timestamp(m)
    return (ts - pd.DateOffset(months=1)).strftime("%Y-%m")


# ── Binding data loaders ──

def load_all_binding_sets(
    peak_type: str = "onpeak",
    cache_dir: str = REALIZED_DA_CACHE,
) -> dict[str, set[str]]:
    """Load monthly binding sets (full months)."""
    binding_sets: dict[str, set[str]] = {}
    if peak_type == "onpeak":
        pattern = "[0-9][0-9][0-9][0-9]-[0-9][0-9].parquet"
    else:
        pattern = f"*_{peak_type}.parquet"
    for f in sorted(Path(cache_dir).glob(pattern)):
        if "_partial_" in f.name:
            continue
        df = pl.read_parquet(str(f))
        month = f.stem.replace(f"_{peak_type}", "")
        binding_sets[month] = set(df.filter(pl.col("realized_sp") > 0)["constraint_id"].to_list())
    print(f"[bf] Loaded {len(binding_sets)} months of {peak_type} full-month binding sets")
    return binding_sets


def load_all_partial_binding(
    n_days: int = PARTIAL_N_DAYS,
    peak_type: str = "onpeak",
    cache_dir: str = REALIZED_DA_CACHE,
) -> dict[str, tuple[dict[str, int], int]]:
    """Load partial-month binding data.

    Returns {month: ({constraint_id: days_binding}, days_total)}.
    """
    partial: dict[str, tuple[dict[str, int], int]] = {}
    suffix = "" if peak_type == "onpeak" else f"_{peak_type}"
    pattern = f"*_partial_d{n_days}{suffix}.parquet"
    for f in sorted(Path(cache_dir).glob(pattern)):
        month = f.stem.split("_partial_")[0]
        df = pl.read_parquet(str(f))
        if len(df) == 0:
            partial[month] = ({}, n_days)
        else:
            counts = dict(zip(
                df["constraint_id"].to_list(),
                df["days_binding"].to_list(),
            ))
            days_total = df["days_total"][0]
            partial[month] = (counts, days_total)
    print(f"[bf] Loaded {len(partial)} months of {peak_type} partial-month binding (d{n_days})")
    return partial


def ensure_partial_cache(
    months_needed: set[str],
    n_days: int = PARTIAL_N_DAYS,
    peak_type: str = "onpeak",
    cache_dir: str = REALIZED_DA_CACHE,
) -> None:
    """Fetch and cache any missing partial-month DA data."""
    suffix = "" if peak_type == "onpeak" else f"_{peak_type}"
    cached = set()
    for f in Path(cache_dir).glob(f"*_partial_d{n_days}{suffix}.parquet"):
        cached.add(f.stem.split("_partial_")[0])

    missing = sorted(months_needed - cached)
    if not missing:
        print(f"[partial_da] All {len(months_needed)} partial months cached (d{n_days})")
        return

    print(f"[partial_da] Need to fetch {len(missing)} partial months (d{n_days}): {missing[:5]}...")
    for m in missing:
        try:
            fetch_partial_month_da(m, n_days=n_days, peak_type=peak_type, cache_dir=cache_dir)
        except Exception as e:
            print(f"[partial_da] WARNING: failed to fetch {m}: {e}")


# ── Extended BF computation ──

def compute_bf_extended(
    cids: list[str],
    cutoff: str,
    bs: dict[str, set[str]],
    lookback: int,
    partial_data: dict[str, tuple[dict[str, int], int]],
) -> np.ndarray:
    """Compute binding frequency with partial-month extension.

    Blends the first N days of the cutoff month into the BF window:
      bf_ext = (full_months_bound + days_binding/30) / (n_full + n_days/30)

    The partial month (= cutoff itself) is BEYOND the months < cutoff boundary,
    but its first N days are already realized at bid submission time.

    Parameters
    ----------
    cids : constraint IDs
    cutoff : BF cutoff month (e.g., M-1 for lag=1). Partial data is for THIS month.
    bs : full-month binding sets {month: set(cid)}
    lookback : number of full months to look back
    partial_data : {month: ({cid: days_binding}, days_total)}
    """
    prior = [m for m in sorted(bs.keys()) if m < cutoff][-lookback:]
    n_full = len(prior)

    # Get partial contribution for the cutoff month itself
    entry = partial_data.get(cutoff)

    if n_full == 0 and entry is None:
        return np.zeros(len(cids), dtype=np.float64)

    # Count full months bound
    freq = np.zeros(len(cids), dtype=np.float64)
    for m in prior:
        s = bs[m]
        for i, cid in enumerate(cids):
            if cid in s:
                freq[i] += 1

    if entry is not None:
        binding_counts, days_total = entry
        partial_weight = days_total / DAYS_PER_MONTH  # fraction of a full month

        # Each constraint's partial contribution: days_binding / 30
        # (equivalent to partial_weight * (days_binding / days_total))
        partial_contrib = np.array(
            [binding_counts.get(cid, 0) / DAYS_PER_MONTH for cid in cids],
            dtype=np.float64,
        )
        return (freq + partial_contrib) / (n_full + partial_weight)
    else:
        return freq / n_full if n_full > 0 else freq


def compute_bf_plain(
    cids: list[str], cutoff: str, bs: dict[str, set[str]], lookback: int,
) -> np.ndarray:
    """Original BF (no partial extension). For baseline comparison."""
    prior = [m for m in sorted(bs.keys()) if m < cutoff][-lookback:]
    n = len(prior)
    if n == 0:
        return np.zeros(len(cids), dtype=np.float64)
    freq = np.zeros(len(cids), dtype=np.float64)
    for m in prior:
        s = bs[m]
        for i, cid in enumerate(cids):
            if cid in s:
                freq[i] += 1
    return freq / n


def enrich_df(
    df: pl.DataFrame,
    month: str,
    bs: dict[str, set[str]],
    partial_data: dict[str, tuple[dict[str, int], int]] | None = None,
    lag: int = 1,
    blend_weights: tuple[float, float, float] | None = None,
    use_extended: bool = True,
) -> pl.DataFrame:
    """Add features: bf_1..bf_15 (extended or plain), v7_formula_score."""
    cutoff = month
    for _ in range(lag):
        cutoff = prev_month(cutoff)

    w_da, w_dmix, w_dori = blend_weights or (V7_DA, V7_DMIX, V7_DORI)
    cids = df["constraint_id"].to_list()

    df = df.with_columns(
        (w_da * pl.col("da_rank_value")
         + w_dmix * pl.col("density_mix_rank_value")
         + w_dori * pl.col("density_ori_rank_value")
        ).alias("v7_formula_score")
    )

    for lb in [1, 3, 6, 12, 15]:
        col_name = f"binding_freq_{lb}"
        if col_name not in df.columns:
            if use_extended and partial_data is not None:
                freq = compute_bf_extended(cids, cutoff, bs, lb, partial_data)
            else:
                freq = compute_bf_plain(cids, cutoff, bs, lb)
            df = df.with_columns(pl.Series(col_name, freq))

    return df


# ── Run variant ──

def run_variant(
    label: str,
    features: list[str],
    monotone: list[int],
    eval_months: list[str],
    bs: dict[str, set[str]],
    partial_data: dict[str, tuple[dict[str, int], int]] | None,
    lag: int,
    class_type: str,
    period_type: str,
    use_extended: bool = True,
) -> dict[str, dict]:
    blend = BLEND_WEIGHTS.get((period_type, class_type), (V7_DA, V7_DMIX, V7_DORI))
    ext_label = "extended" if use_extended else "plain"
    print(f"\n[{label}] {len(features)}f, {len(eval_months)} months, lag={lag}, "
          f"ptype={period_type}, ctype={class_type}, bf={ext_label}")

    cfg = PipelineConfig(
        ltr=LTRConfig(
            features=features,
            monotone_constraints=monotone,
            backend="lightgbm",
            label_mode="tiered",
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
        ),
        train_months=8,
        val_months=0,
    )

    per_month: dict[str, dict] = {}

    for m in eval_months:
        t0 = time.time()

        train_month_strs = collect_usable_months(m, period_type, n_months=8)
        if not train_month_strs:
            print(f"  {m}: SKIP (insufficient training history)")
            continue
        train_month_strs = list(reversed(train_month_strs))

        parts = []
        for tm in train_month_strs:
            try:
                part = load_v62b_month(tm, period_type, class_type)
                part = part.with_columns(pl.lit(tm).alias("query_month"))
                part = enrich_df(part, tm, bs, partial_data=partial_data,
                                 lag=lag, blend_weights=blend, use_extended=use_extended)
                parts.append(part)
            except FileNotFoundError:
                print(f"    {tm}: missing V6.2B data, skip")
        if not parts:
            print(f"  {m}: SKIP (no training data)")
            continue
        train_df = pl.concat(parts)

        try:
            test_df = load_v62b_month(m, period_type, class_type)
        except FileNotFoundError:
            print(f"  {m}: SKIP (no GT data)")
            continue
        test_df = test_df.with_columns(pl.lit(m).alias("query_month"))
        test_df = enrich_df(test_df, m, bs, partial_data=partial_data,
                            lag=lag, blend_weights=blend, use_extended=use_extended)

        train_df = train_df.sort("query_month")
        X_train, _ = prepare_features(train_df, cfg.ltr)
        y_train = train_df["realized_sp"].to_numpy().astype(np.float64)
        groups = compute_query_groups(train_df)

        model = train_ltr_model(X_train, y_train, groups, cfg.ltr)
        X_test, _ = prepare_features(test_df, cfg.ltr)
        scores = predict_scores(model, X_test)
        actual = test_df["realized_sp"].to_numpy().astype(np.float64)

        metrics = evaluate_ltr(actual, scores)
        per_month[m] = metrics

        if hasattr(model, "feature_importance"):
            imp = model.feature_importance(importance_type="gain")
            metrics["_fi"] = dict(zip(cfg.ltr.features, [float(x) for x in imp]))

        elapsed = time.time() - t0
        n_bind = int((actual > 0).sum())
        if m == eval_months[0] or m == eval_months[-1]:
            print(f"  {m}: VC@20={metrics['VC@20']:.4f} VC@100={metrics['VC@100']:.4f} "
                  f"binding={n_bind} train={len(train_df)} ({elapsed:.1f}s)")

        del train_df, test_df, parts, X_train, y_train, groups, model, X_test, actual, scores
        gc.collect()

    clean = {m: {k: v for k, v in met.items() if not k.startswith("_")}
             for m, met in per_month.items()}
    agg = aggregate_months(clean)
    means = agg["mean"]
    print(f"  => VC@20={means.get('VC@20', 0):.4f} VC@50={means.get('VC@50', 0):.4f} "
          f"VC@100={means.get('VC@100', 0):.4f} R@20={means.get('Recall@20', 0):.4f} "
          f"NDCG={means.get('NDCG', 0):.4f} Spearman={means.get('Spearman', 0):.4f}")

    # Feature importance avg
    fi_sums: dict[str, float] = {}
    fi_n: dict[str, int] = {}
    for met in per_month.values():
        for name, val in met.get("_fi", {}).items():
            fi_sums[name] = fi_sums.get(name, 0) + val
            fi_n[name] = fi_n.get(name, 0) + 1
    if fi_sums:
        fi_avg = {n: fi_sums[n] / fi_n[n] for n in fi_sums}
        total = sum(fi_avg.values())
        print("  Feature importance:")
        for name, avg in sorted(fi_avg.items(), key=lambda x: x[1], reverse=True):
            print(f"    {name:<25s} {avg:>8.1f} ({100 * avg / total:.1f}%)")

    return per_month


def save_results(
    label: str, per_month: dict, eval_months: list[str], dest_dir: Path,
    *,
    class_type: str, period_type: str, lag: int = 1,
    blend_weights: tuple[float, float, float] | None = None,
    features: list[str] | None = None,
) -> None:
    clean = {m: {k: v for k, v in met.items() if not k.startswith("_")}
             for m, met in per_month.items()}
    agg = aggregate_months(clean)
    d = dest_dir / label
    d.mkdir(parents=True, exist_ok=True)

    actual_months = sorted(clean.keys())
    skipped = sorted(set(eval_months) - set(actual_months))

    result = {
        "eval_config": {
            "eval_months": actual_months, "class_type": class_type,
            "period_type": period_type, "lag": lag,
            "note": f"v12a: extended BF (d{PARTIAL_N_DAYS}, blend partial into bf windows)",
        },
        "per_month": clean, "aggregate": agg, "n_months": len(clean),
        "n_months_requested": len(eval_months),
        "skipped_months": skipped,
    }
    with open(d / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    config = {
        "method": "lightgbm_lambdarank_tiered",
        "features": features or list(V12_FEATURES),
        "lag": lag,
        "period_type": period_type,
        "class_type": class_type,
        "label_mode": "tiered",
        "train_months": 8,
        "backend": "lightgbm",
        "partial_n_days": PARTIAL_N_DAYS,
        "bf_mode": "extended",
    }
    if blend_weights:
        config["blend_weights"] = {"da": blend_weights[0], "dmix": blend_weights[1], "dori": blend_weights[2]}
    with open(d / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Saved to {d}/")


def print_comparison(labels: dict[str, dict], title: str) -> None:
    metrics_list = ["VC@10", "VC@20", "VC@25", "VC@50", "VC@100", "VC@200",
                    "Recall@10", "Recall@20", "Recall@50", "Recall@100", "NDCG", "Spearman"]

    print(f"\n{'=' * 90}")
    print(title)
    print(f"{'=' * 90}")
    header = f"{'Metric':<12}"
    for vid in labels:
        header += f" {vid:>14}"
    print(header)
    print("-" * (12 + 15 * len(labels)))

    for met in metrics_list:
        vals = {vid: m.get(met, 0) for vid, m in labels.items()}
        best = max(vals.values())
        row = f"{met:<12}"
        for vid in labels:
            v = vals[vid]
            star = "*" if v == best else " "
            row += f" {v:>13.4f}{star}"
        print(row)


def collect_partial_months_needed(
    eval_months: list[str],
    period_type: str,
    lag: int = 1,
) -> set[str]:
    """Determine which partial months need to be cached."""
    needed: set[str] = set()
    for m in eval_months:
        cutoff = m
        for _ in range(lag):
            cutoff = prev_month(cutoff)
        needed.add(cutoff)

        train_months = collect_usable_months(m, period_type, n_months=8)
        for tm in train_months:
            cutoff = tm
            for _ in range(lag):
                cutoff = prev_month(cutoff)
            needed.add(cutoff)

    return needed


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="V12: Extended BF (partial blended into windows)")
    parser.add_argument("--class-type", required=True, choices=["onpeak", "offpeak"])
    parser.add_argument("--ptype", required=True, help="Period type (f0, f1)")
    parser.add_argument("--screen", action="store_true", help="Screen on 4 months only")
    parser.add_argument("--dev-only", action="store_true", help="Skip holdout")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip fetching partial DA (assume cache is populated)")
    parser.add_argument("--baseline", action="store_true",
                        help="Also run v10e baseline (plain BF) for direct comparison")
    args = parser.parse_args()

    class_type = args.class_type
    period_type = args.ptype
    # Production lag: fN needs lag = N+1 (see CLAUDE.md temporal leakage rules)
    ptype_n = int(period_type[1:])  # "f0" -> 0, "f1" -> 1
    lag = ptype_n + 1

    t_start = time.time()
    print(f"[main] V12 extended-BF, ptype={period_type}, ctype={class_type}, "
          f"partial_d={PARTIAL_N_DAYS}, mem={mem_mb():.0f} MB")

    # Load full-month binding sets
    bs = load_all_binding_sets(peak_type=class_type)

    # Determine eval months
    if args.screen:
        dev_eval = [m for m in _SCREEN_EVAL_MONTHS if has_period_type(m, period_type)]
    else:
        dev_eval = [m for m in _FULL_EVAL_MONTHS if has_period_type(m, period_type)]
    holdout_eval = [m for m in HOLDOUT_MONTHS if has_period_type(m, period_type)]

    print(f"[main] {period_type}: {len(dev_eval)} dev months, {len(holdout_eval)} holdout months")

    # Fetch partial-month DA if needed
    if not args.skip_fetch:
        import os
        os.environ.setdefault("RAY_ADDRESS", "ray://10.8.0.36:10001")
        from pbase.config.ray import init_ray
        import pmodel
        init_ray(extra_modules=[pmodel])

        all_months = dev_eval if args.dev_only else dev_eval + holdout_eval
        needed = collect_partial_months_needed(all_months, period_type, lag=lag)
        print(f"[main] Need partial data for {len(needed)} months")
        ensure_partial_cache(needed, n_days=PARTIAL_N_DAYS, peak_type=class_type)

    # Load partial binding data
    partial_data = load_all_partial_binding(n_days=PARTIAL_N_DAYS, peak_type=class_type)

    reg_slice = registry_root(period_type, class_type, base_dir=ROOT / "registry")
    ho_slice = holdout_root(period_type, class_type, base_dir=ROOT / "holdout")

    version_id = "v12a"

    # ── Optionally run baseline (plain BF, no partial) ──
    baseline_pm = None
    if args.baseline:
        baseline_pm = run_variant(
            "v10e-lag1 (baseline)", V12_FEATURES, V12_MONOTONE,
            dev_eval, bs, partial_data=None, lag=lag,
            class_type=class_type, period_type=period_type,
            use_extended=False,
        )

    # ── V12a: extended BF ──
    dev_pm = run_variant(
        version_id, V12_FEATURES, V12_MONOTONE,
        dev_eval, bs, partial_data=partial_data, lag=lag,
        class_type=class_type, period_type=period_type,
        use_extended=True,
    )
    bw = BLEND_WEIGHTS.get((period_type, class_type), (V7_DA, V7_DMIX, V7_DORI))
    save_results(version_id, dev_pm, dev_eval, reg_slice, class_type=class_type,
                 lag=lag, period_type=period_type, blend_weights=bw, features=list(V12_FEATURES))

    # ── Holdout ──
    holdout_pm = None
    if not args.dev_only and not args.screen:
        holdout_pm = run_variant(
            f"{version_id}-holdout", V12_FEATURES, V12_MONOTONE,
            holdout_eval, bs, partial_data=partial_data, lag=lag,
            class_type=class_type, period_type=period_type,
            use_extended=True,
        )
        save_results(version_id, holdout_pm, holdout_eval, ho_slice, class_type=class_type,
                     lag=lag, period_type=period_type, blend_weights=bw, features=list(V12_FEATURES))

    # ── Comparison ──
    comparison = {}

    # Load v10e-lag1 from registry if exists
    v10e_path = reg_slice / "v10e-lag1" / "metrics.json"
    if v10e_path.exists():
        comparison["v10e-lag1 (reg)"] = json.load(open(v10e_path))["aggregate"]["mean"]

    if baseline_pm:
        bl_clean = {m: {k: v for k, v in met.items() if not k.startswith("_")}
                    for m, met in baseline_pm.items()}
        comparison["v10e-lag1 (run)"] = aggregate_months(bl_clean)["mean"]

    # v11a if exists
    v11a_path = reg_slice / "v11a" / "metrics.json"
    if v11a_path.exists():
        comparison["v11a"] = json.load(open(v11a_path))["aggregate"]["mean"]

    dev_clean = {m: {k: v for k, v in met.items() if not k.startswith("_")}
                 for m, met in dev_pm.items()}
    comparison[version_id] = aggregate_months(dev_clean)["mean"]

    if comparison:
        print_comparison(comparison,
                         f"DEV COMPARISON ({period_type}/{class_type}, {len(dev_eval)} months)")

    # Holdout comparison
    if holdout_pm:
        ho_comparison = {}
        v10e_ho = ho_slice / "v10e-lag1" / "metrics.json"
        if v10e_ho.exists():
            ho_comparison["v10e-lag1"] = json.load(open(v10e_ho))["aggregate"]["mean"]
        v11a_ho = ho_slice / "v11a" / "metrics.json"
        if v11a_ho.exists():
            ho_comparison["v11a"] = json.load(open(v11a_ho))["aggregate"]["mean"]
        ho_clean = {m: {k: v for k, v in met.items() if not k.startswith("_")}
                    for m, met in holdout_pm.items()}
        ho_comparison[version_id] = aggregate_months(ho_clean)["mean"]
        if ho_comparison:
            print_comparison(ho_comparison,
                             f"HOLDOUT COMPARISON ({period_type}/{class_type}, {len(holdout_eval)} months)")

    print(f"\n[main] Done in {time.time() - t_start:.1f}s, mem={mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
