#!/usr/bin/env python
"""V5 Two-Model experiment: predictive-only vs full, with BF-aware blending.

Hypothesis: binding_freq dominates the full model, drowning out predictive
features. For BF-zero constraints (27% of binders), the full model is
essentially guessing. A dedicated predictive model trained WITHOUT any
historical info should rank new constraints much better.

Models:
  v5p  — 5 predictive features (ori_mean, binding_prob, pred_sp, p100, limit)
  v5f  — 14 full features (same as v4, includes BF + DA history)
  v5   — BF-aware blend: full model for BF-positive, pred model for BF-zero

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
    python scripts/run_v5_two_model.py --ptype f0 --class-type onpeak
    python scripts/run_v5_two_model.py                                   # all 6
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
    V3_FEATURES, V3_MONOTONE, V5P_FEATURES, V5P_MONOTONE,
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


def run_two_model(
    eval_months: list[str],
    bs: dict[str, set[str]],
    class_type: str,
    period_type: str,
    label_prefix: str = "",
) -> dict[str, dict[str, dict]]:
    """Train pred-only and full models, evaluate both + blend.

    Returns {variant: {month: metrics}} for v5p, v5f, v5.
    """
    cfg_pred = PipelineConfig(
        ltr=LTRConfig(
            features=list(V5P_FEATURES),
            monotone_constraints=list(V5P_MONOTONE),
            backend="lightgbm",
            label_mode="tiered",
        ),
        train_months=8, val_months=0,
    )
    cfg_full = PipelineConfig(
        ltr=LTRConfig(
            features=list(V3_FEATURES),
            monotone_constraints=list(V3_MONOTONE),
            backend="lightgbm",
            label_mode="tiered",
        ),
        train_months=8, val_months=0,
    )

    results: dict[str, dict[str, dict]] = {"v5p": {}, "v5f": {}, "v5": {}}

    print(f"\n[{label_prefix}] Two-model: pred(5f) + full(14f) + blend")
    print(f"  Pred features: {V5P_FEATURES}")
    print(f"  Full features: {V3_FEATURES}")
    print(f"  {len(eval_months)} eval months, {period_type}/{class_type}")

    for m in eval_months:
        t0 = time.time()
        train_month_strs = collect_usable_months(m, period_type, n_months=8)
        if not train_month_strs:
            print(f"  {m}: SKIP (insufficient history)")
            continue
        train_month_strs = list(reversed(train_month_strs))

        # Load + enrich training data
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
        train_df = pl.concat(parts).sort("query_month")

        # Load + enrich test data
        try:
            test_df = load_v62b_month(m, period_type, class_type)
        except FileNotFoundError:
            continue
        test_df = test_df.with_columns(pl.lit(m).alias("query_month"))
        test_df = enrich_df(test_df, m, bs)

        actual = test_df["realized_sp"].to_numpy().astype(np.float64)
        test_branches = test_df["branch_name"].to_list()
        new_mask = compute_new_mask(test_branches, m, bs, lookback=6)

        # --- Train PRED model (8 features, no BF) ---
        y_train = train_df["realized_sp"].to_numpy().astype(np.float64)
        groups = compute_query_groups(train_df)

        X_train_pred, _ = prepare_features(train_df, cfg_pred.ltr)
        model_pred = train_ltr_model(X_train_pred, y_train, groups, cfg_pred.ltr)
        X_test_pred, _ = prepare_features(test_df, cfg_pred.ltr)
        scores_pred = predict_scores(model_pred, X_test_pred)

        # --- Train FULL model (14 features, with BF) ---
        X_train_full, _ = prepare_features(train_df, cfg_full.ltr)
        model_full = train_ltr_model(X_train_full, y_train, groups, cfg_full.ltr)
        X_test_full, _ = prepare_features(test_df, cfg_full.ltr)
        scores_full = predict_scores(model_full, X_test_full)

        # --- BF-aware blend: full for BF-positive, pred for BF-zero ---
        scores_blend = np.where(new_mask, scores_pred, scores_full)

        # Evaluate all three
        met_pred = evaluate_ltr(actual, scores_pred, new_mask=new_mask)
        met_full = evaluate_ltr(actual, scores_full, new_mask=new_mask)
        met_blend = evaluate_ltr(actual, scores_blend, new_mask=new_mask)

        results["v5p"][m] = met_pred
        results["v5f"][m] = met_full
        results["v5"][m] = met_blend

        # Feature importance
        if hasattr(model_pred, "feature_importance"):
            imp = model_pred.feature_importance(importance_type="gain")
            met_pred["_fi"] = dict(zip(cfg_pred.ltr.features, [float(x) for x in imp]))
        if hasattr(model_full, "feature_importance"):
            imp = model_full.feature_importance(importance_type="gain")
            met_full["_fi"] = dict(zip(cfg_full.ltr.features, [float(x) for x in imp]))

        n_bind = int((actual > 0).sum())
        n_new = met_blend.get("n_new", 0)
        n_bf_zero = int(new_mask.sum())
        elapsed = time.time() - t0

        print(f"  {m}: bind={n_bind} new={n_new} bf0_rows={n_bf_zero}/{len(actual)}")
        print(f"    pred: VC@20={met_pred['VC@20']:.4f} NB_R@50={met_pred.get('NewBind_Recall@50',0):.3f} "
              f"NB_VC@50={met_pred.get('NewBind_VC@50',0):.3f}")
        print(f"    full: VC@20={met_full['VC@20']:.4f} NB_R@50={met_full.get('NewBind_Recall@50',0):.3f} "
              f"NB_VC@50={met_full.get('NewBind_VC@50',0):.3f}")
        print(f"    blend:VC@20={met_blend['VC@20']:.4f} NB_R@50={met_blend.get('NewBind_Recall@50',0):.3f} "
              f"NB_VC@50={met_blend.get('NewBind_VC@50',0):.3f}  ({elapsed:.1f}s)")

        del train_df, test_df, parts, X_train_pred, X_train_full, X_test_pred, X_test_full
        del model_pred, model_full, y_train, groups, actual, scores_pred, scores_full, scores_blend
        gc.collect()

    # Print aggregates
    for variant in ["v5p", "v5f", "v5"]:
        pm = results[variant]
        if not pm:
            continue
        clean = {m: {k: v for k, v in met.items() if not k.startswith("_")}
                 for m, met in pm.items()}
        agg = aggregate_months(clean)
        means = agg["mean"]
        print(f"\n  [{variant}] Aggregate ({len(pm)} months):")
        for met in ["VC@20", "VC@50", "VC@100", "Recall@20", "Recall@50", "NDCG", "Spearman",
                     "NewBind_Recall@20", "NewBind_Recall@50", "NewBind_VC@20", "NewBind_VC@50"]:
            print(f"    {met:<22} {means.get(met, 0):.4f}")

    print(f"  [mem] {mem_mb():.0f} MB")
    return results


def save_variant(variant, per_month, eval_months, dest_dir, class_type, period_type, features, label_mode="tiered"):
    clean = {m: {k: v for k, v in met.items() if not k.startswith("_")}
             for m, met in per_month.items()}
    agg = aggregate_months(clean)
    d = dest_dir / variant
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
        "method": f"lightgbm_lambdarank_{label_mode}",
        "variant": variant,
        "features": features,
        "label_mode": label_mode,
        "lag": 1, "period_type": period_type, "class_type": class_type,
        "blend_weights": dict(zip(["da", "dmix", "dori"], V0B_BLEND)),
    }
    if variant == "v5":
        config["blend_strategy"] = "hard_switch: full model for BF-positive, pred model for BF-zero"
        config["pred_features"] = list(V5P_FEATURES)
        config["full_features"] = list(V3_FEATURES)
    with open(d / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved {variant} to {d}/")

    # Feature importance
    fi_months = {m: met["_fi"] for m, met in per_month.items() if "_fi" in met}
    if fi_months:
        avg_fi = {}
        for fi in fi_months.values():
            for feat, val in fi.items():
                avg_fi[feat] = avg_fi.get(feat, 0) + val
        for feat in avg_fi:
            avg_fi[feat] /= len(fi_months)
        sorted_fi = sorted(avg_fi.items(), key=lambda x: x[1], reverse=True)
        print(f"  [{variant}] Feature importance (avg gain):")
        for feat, val in sorted_fi:
            print(f"    {feat:<30} {val:.0f}")


def print_comparison(ptypes, ctypes):
    """Print comparison: v0b vs v5p vs v5f vs v5 on key metrics."""
    METRICS = ["VC@20", "VC@50", "VC@100", "Recall@50", "NDCG", "Spearman",
               "NewBind_Recall@20", "NewBind_Recall@50", "NewBind_VC@20", "NewBind_VC@50"]
    slices = [(p, c) for p in ptypes for c in ctypes]

    def load_agg(path):
        if path.exists():
            return json.load(open(path))["aggregate"]["mean"]
        return {}

    for mode, base in [("DEV", REGISTRY), ("HOLDOUT", HOLDOUT_DIR)]:
        print(f"\n{'='*110}")
        print(f"  {mode}: v0b vs v5p (pred-only) vs v5f (full) vs v5 (blend)")
        print(f"{'='*110}")
        for metric in METRICS:
            print(f"\n  {metric}:")
            print(f"    {'Slice':<22} {'v0b':>8} {'v5p':>8} {'v5f':>8} {'v5':>8} {'v0b→v5':>10}")
            print(f"    {'-'*72}")
            vals = {v: [] for v in ["v0b", "v5p", "v5f", "v5"]}
            for pt, ct in slices:
                if mode == "HOLDOUT":
                    root = holdout_root(pt, ct, base_dir=base)
                else:
                    root = registry_root(pt, ct, base_dir=base)
                for ver in vals:
                    agg = load_agg(root / ver / "metrics.json")
                    vals[ver].append(agg.get(metric, 0))
                v0b_v = vals["v0b"][-1]
                v5_v = vals["v5"][-1]
                delta = (v5_v / v0b_v - 1) * 100 if v0b_v != 0 else 0
                print(f"    {pt}/{ct:<18} {v0b_v:>8.4f} {vals['v5p'][-1]:>8.4f} "
                      f"{vals['v5f'][-1]:>8.4f} {v5_v:>8.4f} {delta:>+9.1f}%")
            for ver in vals:
                if vals[ver]:
                    vals[ver].append(sum(vals[ver]) / len(vals[ver]))
            v0b_m = vals["v0b"][-1]
            v5_m = vals["v5"][-1]
            delta_m = (v5_m / v0b_m - 1) * 100 if v0b_m != 0 else 0
            print(f"    {'MEAN':<22} {v0b_m:>8.4f} {vals['v5p'][-1]:>8.4f} "
                  f"{vals['v5f'][-1]:>8.4f} {v5_m:>8.4f} {delta_m:>+9.1f}%")


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
    print(f"  V5 Two-Model: pred(5f, no history) + full(14f) + BF-aware blend")
    print(f"{'#'*70}")

    for ptype in ptypes:
        for ctype in ctypes:
            print(f"\n{'='*70}")
            print(f"v5: {ptype}/{ctype}")
            print(f"{'='*70}")

            bs = load_all_binding_sets(peak_type=ctype)

            if not args.holdout_only:
                dev_eval = [m for m in _FULL_EVAL_MONTHS if has_period_type(m, ptype)]
                reg_slice = registry_root(ptype, ctype, base_dir=REGISTRY)

                dev_results = run_two_model(dev_eval, bs, class_type=ctype,
                                            period_type=ptype, label_prefix="dev")
                for variant, features in [("v5p", V5P_FEATURES), ("v5f", V3_FEATURES), ("v5", V3_FEATURES)]:
                    if dev_results[variant]:
                        save_variant(variant, dev_results[variant], dev_eval,
                                     reg_slice, class_type=ctype, period_type=ptype,
                                     features=features)

            if not args.dev_only:
                ho_eval = [m for m in HOLDOUT_MONTHS if has_period_type(m, ptype)]
                ho_slice = holdout_root(ptype, ctype, base_dir=HOLDOUT_DIR)
                clear_month_cache()
                gc.collect()

                ho_results = run_two_model(ho_eval, bs, class_type=ctype,
                                           period_type=ptype, label_prefix="holdout")
                for variant, features in [("v5p", V5P_FEATURES), ("v5f", V3_FEATURES), ("v5", V3_FEATURES)]:
                    if ho_results[variant]:
                        save_variant(variant, ho_results[variant], ho_eval,
                                     ho_slice, class_type=ctype, period_type=ptype,
                                     features=features)

            clear_month_cache()
            gc.collect()

    print_comparison(ptypes, ctypes)
    print(f"\n[main] All done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
