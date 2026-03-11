#!/usr/bin/env python
"""V4 ML model for PJM: 14-feature LightGBM LambdaRank with v0b blend.

Key difference from v3b: uses v0b blend weights (0.80/0.15/0.05) for
v7_formula_score, not the old smoothed per-ptype blends.

Evaluated against v0b baseline with 12 Group A gates including NewBind metrics.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
    python scripts/run_v4_ml.py --ptype f0 --class-type onpeak  # single slice
    python scripts/run_v4_ml.py                                  # all 6 slices
"""
from __future__ import annotations

import argparse
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
    V3_FEATURES, V3_MONOTONE,
    _FULL_EVAL_MONTHS, HOLDOUT_MONTHS, PJM_CLASS_TYPES,
    has_period_type, collect_usable_months,
)
from ml.data_loader import load_v62b_month, clear_month_cache, compute_new_mask
from ml.evaluate import aggregate_months, evaluate_ltr
from ml.features import compute_query_groups, prepare_features
from ml.registry_paths import registry_root, holdout_root
from ml.train import predict_scores, train_ltr_model

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "registry"
HOLDOUT_DIR = ROOT / "holdout"

# v0b blend weights for all slices (0.80/0.15/0.05)
V0B_BLEND = (0.80, 0.15, 0.05)


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_all_binding_sets(
    peak_type: str = "onpeak",
    cache_dir: str = REALIZED_DA_CACHE,
) -> dict[str, set[str]]:
    binding_sets: dict[str, set[str]] = {}
    if peak_type == "onpeak":
        pattern = "[0-9][0-9][0-9][0-9]-[0-9][0-9].parquet"
    else:
        pattern = f"*_{peak_type}.parquet"
    for f in sorted(Path(cache_dir).glob(pattern)):
        df = pl.read_parquet(str(f))
        month = f.stem.replace(f"_{peak_type}", "")
        binding_sets[month] = set(
            df.filter(pl.col("realized_sp") > 0)["branch_name"].to_list()
        )
    print(f"[bf] Loaded {len(binding_sets)} months of {peak_type} binding sets")
    return binding_sets


def compute_bf(
    branch_names: list[str], month: str,
    bs: dict[str, set[str]], lookback: int,
) -> np.ndarray:
    prior = [m for m in sorted(bs.keys()) if m < month][-lookback:]
    n = len(prior)
    if n == 0:
        return np.zeros(len(branch_names), dtype=np.float64)
    freq = np.zeros(len(branch_names), dtype=np.float64)
    for m in prior:
        s = bs[m]
        for i, bn in enumerate(branch_names):
            if bn in s:
                freq[i] += 1
    return freq / n


def prev_month(m: str) -> str:
    return (pd.Timestamp(m) - pd.DateOffset(months=1)).strftime("%Y-%m")


def enrich_df(
    df: pl.DataFrame, month: str, bs: dict[str, set[str]],
) -> pl.DataFrame:
    """Add binding_freq and v0b formula score features."""
    cutoff = prev_month(month)
    w_da, w_dmix, w_dori = V0B_BLEND
    branch_names = df["branch_name"].to_list()

    df = df.with_columns(
        (w_da * pl.col("da_rank_value")
         + w_dmix * pl.col("density_mix_rank_value")
         + w_dori * pl.col("density_ori_rank_value")
        ).alias("v7_formula_score")
    )
    for lb in [1, 3, 6, 12, 15]:
        col_name = f"binding_freq_{lb}"
        if col_name not in df.columns:
            freq = compute_bf(branch_names, cutoff, bs, lb)
            df = df.with_columns(pl.Series(col_name, freq))
    return df


def run_variant(
    label: str,
    eval_months: list[str],
    bs: dict[str, set[str]],
    class_type: str,
    period_type: str,
) -> dict[str, dict]:
    print(f"\n[{label}] 14f/tiered, {len(eval_months)} months, ptype={period_type}, "
          f"ctype={class_type}, blend=v0b({V0B_BLEND})")

    cfg = PipelineConfig(
        ltr=LTRConfig(
            features=list(V3_FEATURES),
            monotone_constraints=list(V3_MONOTONE),
            backend="lightgbm",
            label_mode="tiered",
        ),
        train_months=8,
        val_months=0,
    )

    per_month: dict[str, dict] = {}

    for m in eval_months:
        t0 = time.time()
        train_month_strs = collect_usable_months(m, period_type, n_months=8)
        if not train_month_strs:
            print(f"  {m}: SKIP (insufficient history)")
            continue
        train_month_strs = list(reversed(train_month_strs))

        parts = []
        for tm in train_month_strs:
            try:
                part = load_v62b_month(tm, period_type, class_type)
                part = part.with_columns(pl.lit(tm).alias("query_month"))
                part = enrich_df(part, tm, bs)
                parts.append(part)
            except FileNotFoundError:
                pass
        if not parts:
            continue
        train_df = pl.concat(parts)

        try:
            test_df = load_v62b_month(m, period_type, class_type)
        except FileNotFoundError:
            continue
        test_df = test_df.with_columns(pl.lit(m).alias("query_month"))
        test_df = enrich_df(test_df, m, bs)

        train_df = train_df.sort("query_month")
        X_train, _ = prepare_features(train_df, cfg.ltr)
        y_train = train_df["realized_sp"].to_numpy().astype(np.float64)
        groups = compute_query_groups(train_df)

        model = train_ltr_model(X_train, y_train, groups, cfg.ltr)
        X_test, _ = prepare_features(test_df, cfg.ltr)
        scores = predict_scores(model, X_test)
        actual = test_df["realized_sp"].to_numpy().astype(np.float64)

        # Compute new_mask (BF-zero)
        test_branches = test_df["branch_name"].to_list()
        new_mask = compute_new_mask(test_branches, m, bs, lookback=6)

        metrics = evaluate_ltr(actual, scores, new_mask=new_mask)
        per_month[m] = metrics

        if hasattr(model, "feature_importance"):
            imp = model.feature_importance(importance_type="gain")
            metrics["_fi"] = dict(zip(cfg.ltr.features, [float(x) for x in imp]))

        elapsed = time.time() - t0
        n_bind = int((actual > 0).sum())
        n_new = metrics.get("n_new", 0)
        print(f"  {m}: VC@20={metrics['VC@20']:.4f} VC@50={metrics['VC@50']:.4f} "
              f"R@20={metrics['Recall@20']:.3f} NB_R@50={metrics.get('NewBind_Recall@50', 0):.3f} "
              f"binding={n_bind} new={n_new} ({elapsed:.1f}s)")

        del train_df, test_df, parts, X_train, y_train, groups, model, X_test, actual, scores, new_mask
        gc.collect()

    if per_month:
        clean = {m: {k: v for k, v in met.items() if not k.startswith("_")}
                 for m, met in per_month.items()}
        agg = aggregate_months(clean)
        means = agg["mean"]
        print(f"\n  => Aggregate ({len(per_month)} months):")
        for met in ["VC@20", "VC@50", "VC@100", "Recall@20", "Recall@50", "NDCG", "Spearman",
                     "NewBind_Recall@50", "NewBind_Recall@100", "NewBind_VC@50", "NewBind_VC@100"]:
            print(f"     {met:<22} {means.get(met, 0):.4f}")
        print(f"  [mem] {mem_mb():.0f} MB")

    return per_month


def save_results(label, per_month, eval_months, dest_dir, class_type, period_type):
    clean = {m: {k: v for k, v in met.items() if not k.startswith("_")}
             for m, met in per_month.items()}
    agg = aggregate_months(clean)
    d = dest_dir / label
    d.mkdir(parents=True, exist_ok=True)

    result = {
        "eval_config": {"eval_months": sorted(clean.keys()), "class_type": class_type,
                        "period_type": period_type, "lag": 1},
        "per_month": clean, "aggregate": agg, "n_months": len(clean),
        "n_months_requested": len(eval_months),
        "skipped_months": sorted(set(eval_months) - set(clean.keys())),
    }
    with open(d / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    config = {
        "method": "lightgbm_lambdarank_tiered",
        "features": list(V3_FEATURES),
        "label_mode": "tiered",
        "lag": 1, "period_type": period_type, "class_type": class_type,
        "blend_weights": dict(zip(["da", "dmix", "dori"], V0B_BLEND)),
    }
    with open(d / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved to {d}/")

    # Feature importance summary
    fi_months = {m: met["_fi"] for m, met in per_month.items() if "_fi" in met}
    if fi_months:
        avg_fi = {}
        for fi in fi_months.values():
            for feat, val in fi.items():
                avg_fi[feat] = avg_fi.get(feat, 0) + val
        for feat in avg_fi:
            avg_fi[feat] /= len(fi_months)
        sorted_fi = sorted(avg_fi.items(), key=lambda x: x[1], reverse=True)
        print(f"  Feature importance (avg gain):")
        for feat, val in sorted_fi:
            print(f"    {feat:<30} {val:.0f}")


def print_comparison(ptypes, ctypes):
    """Print comparison: v0b vs v2 vs v4 on all Group A metrics."""
    METRICS = ["VC@20", "VC@50", "VC@100", "Recall@20", "Recall@50", "Recall@100",
               "NDCG", "Spearman", "NewBind_Recall@50", "NewBind_Recall@100",
               "NewBind_VC@50", "NewBind_VC@100"]
    slices = [(p, c) for p in ptypes for c in ctypes]

    def load_agg(path):
        if path.exists():
            return json.load(open(path))["aggregate"]["mean"]
        return {}

    for mode, base in [("DEV", REGISTRY), ("HOLDOUT", HOLDOUT_DIR)]:
        print(f"\n{'='*95}")
        print(f"  {mode}: v0b vs v2 vs v4")
        print(f"{'='*95}")
        for metric in METRICS:
            print(f"\n  {metric}:")
            print(f"    {'Slice':<22} {'v0b':>8} {'v2':>8} {'v4':>8} {'v0b→v4':>10}")
            print(f"    {'-'*62}")
            vals = {v: [] for v in ["v0b", "v2", "v4"]}
            for pt, ct in slices:
                root = base / pt / ct if mode == "HOLDOUT" else registry_root(pt, ct, base_dir=base)
                if mode == "HOLDOUT":
                    root = holdout_root(pt, ct, base_dir=base)
                else:
                    root = registry_root(pt, ct, base_dir=base)
                for ver in ["v0b", "v2", "v4"]:
                    agg = load_agg(root / ver / "metrics.json")
                    vals[ver].append(agg.get(metric, 0))
                v0b_v = vals["v0b"][-1]
                v4_v = vals["v4"][-1]
                delta = (v4_v / v0b_v - 1) * 100 if v0b_v != 0 else 0
                print(f"    {pt}/{ct:<18} {v0b_v:>8.4f} {vals['v2'][-1]:>8.4f} "
                      f"{v4_v:>8.4f} {delta:>+9.1f}%")
            for ver in vals:
                if vals[ver]:
                    vals[ver].append(sum(vals[ver]) / len(vals[ver]))
            v0b_m = vals["v0b"][-1]
            v4_m = vals["v4"][-1]
            delta_m = (v4_m / v0b_m - 1) * 100 if v0b_m != 0 else 0
            print(f"    {'MEAN':<22} {v0b_m:>8.4f} {vals['v2'][-1]:>8.4f} "
                  f"{v4_m:>8.4f} {delta_m:>+9.1f}%")


def print_temporal_segmentation(ptypes, ctypes):
    """Print 2024 vs 2025 holdout breakdown."""
    METRICS = ["VC@20", "VC@50", "VC@100", "Recall@20", "NDCG",
               "NewBind_Recall@50", "NewBind_VC@50"]

    for pt in ptypes:
        for ct in ctypes:
            ho_dir = holdout_root(pt, ct, base_dir=HOLDOUT_DIR)
            for ver in ["v0b", "v4"]:
                path = ho_dir / ver / "metrics.json"
                if not path.exists():
                    continue
                data = json.load(open(path))
                pm = data["per_month"]

                print(f"\n  [{ver}] {pt}/{ct} temporal segmentation:")
                for year in ["2024", "2025"]:
                    year_pm = {m: v for m, v in pm.items() if m.startswith(year)}
                    if not year_pm:
                        continue
                    year_agg = aggregate_months(year_pm)
                    means = year_agg["mean"]
                    print(f"    {year} ({len(year_pm)} months):", end="")
                    for metric in METRICS:
                        print(f"  {metric}={means.get(metric, 0):.4f}", end="")
                    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptype", default=None)
    parser.add_argument("--class-type", default=None)
    parser.add_argument("--dev-only", action="store_true")
    parser.add_argument("--holdout-only", action="store_true")
    args = parser.parse_args()

    ptypes = [args.ptype] if args.ptype else ["f0", "f1"]
    ctypes = [args.class_type] if args.class_type else PJM_CLASS_TYPES
    t_start = time.time()

    print(f"\n{'#'*70}")
    print(f"  V4 ML: 14 features, tiered labels, v0b blend (0.80/0.15/0.05)")
    print(f"{'#'*70}")

    for ptype in ptypes:
        for ctype in ctypes:
            print(f"\n{'='*70}")
            print(f"v4: {ptype}/{ctype}")
            print(f"{'='*70}")

            bs = load_all_binding_sets(peak_type=ctype)

            if not args.holdout_only:
                dev_eval = [m for m in _FULL_EVAL_MONTHS if has_period_type(m, ptype)]
                reg_slice = registry_root(ptype, ctype, base_dir=REGISTRY)

                dev_pm = run_variant("v4", dev_eval, bs, class_type=ctype, period_type=ptype)
                if dev_pm:
                    save_results("v4", dev_pm, dev_eval, reg_slice,
                                 class_type=ctype, period_type=ptype)

            if not args.dev_only:
                ho_eval = [m for m in HOLDOUT_MONTHS if has_period_type(m, ptype)]
                ho_slice = holdout_root(ptype, ctype, base_dir=HOLDOUT_DIR)
                clear_month_cache()
                gc.collect()

                ho_pm = run_variant("v4-holdout", ho_eval, bs, class_type=ctype, period_type=ptype)
                if ho_pm:
                    save_results("v4", ho_pm, ho_eval, ho_slice,
                                 class_type=ctype, period_type=ptype)

            clear_month_cache()
            gc.collect()

    # Print comparison and temporal segmentation
    print_comparison(ptypes, ctypes)
    if not args.dev_only:
        print(f"\n{'='*95}")
        print(f"  TEMPORAL SEGMENTATION (Holdout 2024 vs 2025)")
        print(f"{'='*95}")
        print_temporal_segmentation(ptypes, ctypes)

    print(f"\n[main] All done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
