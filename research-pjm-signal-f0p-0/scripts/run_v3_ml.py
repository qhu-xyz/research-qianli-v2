#!/usr/bin/env python
"""V3 value-aware ML experiments for PJM.

4 variants:
  v3a: 9 features (V10E) + log_value labels  (isolate label change)
  v3b: 14 features (V3) + tiered labels      (isolate feature change)
  v3c: 14 features (V3) + log_value labels   (combined)
  v3d: v3c + two-stage hybrid                (formula top-K preserved)

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
    python scripts/run_v3_ml.py --variant v3a --ptype f0 --class-type onpeak  # quick screen
    python scripts/run_v3_ml.py --variant v3c  # all 6 slices, dev + holdout
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
    V10E_FEATURES, V10E_MONOTONE, V3_FEATURES, V3_MONOTONE,
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

# Same blend weights as v2 (smoothed per-ptype)
BLEND_WEIGHTS: dict[tuple[str, str], tuple[float, float, float]] = {
    ("f0", "onpeak"): (0.00, 0.20, 0.80),
    ("f0", "dailyoffpeak"): (0.00, 0.20, 0.80),
    ("f0", "wkndonpeak"): (0.00, 0.20, 0.80),
    ("f1", "onpeak"): (0.00, 0.90, 0.10),
    ("f1", "dailyoffpeak"): (0.00, 0.90, 0.10),
    ("f1", "wkndonpeak"): (0.00, 0.90, 0.10),
}
_DEFAULT_BLEND = (0.85, 0.00, 0.15)

# Variant configs: (features, monotone, label_mode, two_stage)
VARIANT_CONFIGS = {
    "v3a": (list(V10E_FEATURES), list(V10E_MONOTONE), "log_value", False),
    "v3b": (list(V3_FEATURES), list(V3_MONOTONE), "tiered", False),
    "v3c": (list(V3_FEATURES), list(V3_MONOTONE), "log_value", False),
    "v3d": (list(V3_FEATURES), list(V3_MONOTONE), "log_value", True),
}


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
    blend_weights: tuple[float, float, float],
) -> pl.DataFrame:
    cutoff = prev_month(month)
    w_da, w_dmix, w_dori = blend_weights
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


def apply_two_stage(scores: np.ndarray, test_df: pl.DataFrame, k: int = 30) -> np.ndarray:
    """Two-stage hybrid: preserve formula ranking for top-K, ML for rest."""
    scores = scores.copy()
    formula_scores = -test_df["v7_formula_score"].to_numpy().astype(np.float64)
    formula_order = np.argsort(formula_scores)[::-1]
    top_k_idx = set(formula_order[:k].tolist())
    ml_max = scores.max()
    for idx in top_k_idx:
        scores[idx] = ml_max + 1.0 + formula_scores[idx]
    return scores


def run_variant(
    variant: str,
    eval_months: list[str],
    bs: dict[str, set[str]],
    class_type: str,
    period_type: str,
) -> dict[str, dict]:
    features, monotone, label_mode, two_stage = VARIANT_CONFIGS[variant]
    blend = BLEND_WEIGHTS.get((period_type, class_type), _DEFAULT_BLEND)

    print(f"\n[{variant}] {len(features)}f/{label_mode}, {len(eval_months)} months, "
          f"ptype={period_type}, ctype={class_type}, blend={blend}, 2stage={two_stage}")

    cfg = PipelineConfig(
        ltr=LTRConfig(
            features=features,
            monotone_constraints=monotone,
            backend="lightgbm",
            label_mode=label_mode,
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
                part = enrich_df(part, tm, bs, blend)
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
        test_df = enrich_df(test_df, m, bs, blend)

        train_df = train_df.sort("query_month")
        X_train, _ = prepare_features(train_df, cfg.ltr)
        y_train = train_df["realized_sp"].to_numpy().astype(np.float64)
        groups = compute_query_groups(train_df)

        model = train_ltr_model(X_train, y_train, groups, cfg.ltr)
        X_test, _ = prepare_features(test_df, cfg.ltr)
        scores = predict_scores(model, X_test)
        actual = test_df["realized_sp"].to_numpy().astype(np.float64)

        if two_stage:
            scores = apply_two_stage(scores, test_df)

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
        print(f"  {m}: VC@20={metrics['VC@20']:.4f} VC@50={metrics['VC@50']:.4f} "
              f"R@20={metrics['Recall@20']:.3f} binding={n_bind} ({elapsed:.1f}s)")

        del train_df, test_df, parts, X_train, y_train, groups, model, X_test, actual, scores, new_mask
        gc.collect()

    if per_month:
        clean = {m: {k: v for k, v in met.items() if not k.startswith("_")}
                 for m, met in per_month.items()}
        agg = aggregate_months(clean)
        means = agg["mean"]
        print(f"\n  => Aggregate ({len(per_month)} months):")
        for met in ["VC@20", "VC@50", "VC@100", "Recall@20", "Recall@50", "NDCG", "Spearman"]:
            print(f"     {met:<12} {means.get(met, 0):.4f}")
        print(f"  [mem] {mem_mb():.0f} MB")

    return per_month


def save_results(variant, per_month, eval_months, dest_dir, class_type, period_type, label_mode):
    clean = {m: {k: v for k, v in met.items() if not k.startswith("_")}
             for m, met in per_month.items()}
    agg = aggregate_months(clean)
    d = dest_dir / variant
    d.mkdir(parents=True, exist_ok=True)

    features, monotone, lm, two_stage = VARIANT_CONFIGS[variant]

    result = {
        "eval_config": {"eval_months": sorted(clean.keys()), "class_type": class_type,
                        "period_type": period_type, "lag": 1, "variant": variant},
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
        "two_stage": two_stage,
        "lag": 1, "period_type": period_type, "class_type": class_type,
        "blend_weights": dict(zip(["da", "dmix", "dori"],
                                  BLEND_WEIGHTS.get((period_type, class_type), _DEFAULT_BLEND))),
    }
    with open(d / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved to {d}/")


def print_comparison(variant, ptypes, ctypes):
    """Print full multi-metric comparison: v0 vs v2 vs variant."""
    METRICS = ["VC@20", "VC@50", "VC@100", "Recall@20", "Recall@50", "Recall@100", "NDCG", "Spearman"]
    slices = [(p, c) for p in ptypes for c in ctypes]

    def load_agg(path):
        if path.exists():
            return json.load(open(path))["aggregate"]["mean"]
        return {}

    for mode, base in [("DEV", REGISTRY), ("HOLDOUT", HOLDOUT_DIR)]:
        print(f"\n{'='*90}")
        print(f"  {mode}: v0 vs v2 vs {variant}")
        print(f"{'='*90}")
        for metric in METRICS:
            print(f"\n  {metric}:")
            print(f"    {'Slice':<22} {'v0':>8} {'v2':>8} {variant:>8} {'v0→'+variant:>10}")
            print(f"    {'-'*60}")
            vals = {v: [] for v in ["v0", "v2", variant]}
            for pt, ct in slices:
                for ver in ["v0", "v2", variant]:
                    agg = load_agg(base / pt / ct / ver / "metrics.json")
                    vals[ver].append(agg.get(metric, 0))
                v0_v = vals["v0"][-1]
                v2_v = vals["v2"][-1]
                v3_v = vals[variant][-1]
                delta = (v3_v / v0_v - 1) * 100 if v0_v != 0 else 0
                print(f"    {pt}/{ct:<18} {v0_v:>8.4f} {v2_v:>8.4f} {v3_v:>8.4f} {delta:>+9.1f}%")
            for ver in vals:
                if vals[ver]:
                    mean = sum(vals[ver]) / len(vals[ver])
                    vals[ver].append(mean)
            v0_m = vals["v0"][-1]
            v3_m = vals[variant][-1]
            delta_m = (v3_m / v0_m - 1) * 100 if v0_m != 0 else 0
            print(f"    {'MEAN':<22} {v0_m:>8.4f} {vals['v2'][-1]:>8.4f} {v3_m:>8.4f} {delta_m:>+9.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANT_CONFIGS.keys()))
    parser.add_argument("--ptype", default=None)
    parser.add_argument("--class-type", default=None)
    parser.add_argument("--dev-only", action="store_true")
    parser.add_argument("--holdout-only", action="store_true")
    args = parser.parse_args()

    variant = args.variant
    ptypes = [args.ptype] if args.ptype else ["f0", "f1"]
    ctypes = [args.class_type] if args.class_type else PJM_CLASS_TYPES
    t_start = time.time()

    features, monotone, label_mode, two_stage = VARIANT_CONFIGS[variant]
    print(f"\n{'#'*70}")
    print(f"  V3 Experiment: {variant}")
    print(f"  Features: {len(features)} ({label_mode} labels, 2stage={two_stage})")
    print(f"{'#'*70}")

    for ptype in ptypes:
        for ctype in ctypes:
            print(f"\n{'='*70}")
            print(f"{variant}: {ptype}/{ctype}")
            print(f"{'='*70}")

            bs = load_all_binding_sets(peak_type=ctype)

            if not args.holdout_only:
                dev_eval = [m for m in _FULL_EVAL_MONTHS if has_period_type(m, ptype)]
                reg_slice = registry_root(ptype, ctype, base_dir=REGISTRY)

                dev_pm = run_variant(variant, dev_eval, bs, class_type=ctype, period_type=ptype)
                if dev_pm:
                    save_results(variant, dev_pm, dev_eval, reg_slice,
                                 class_type=ctype, period_type=ptype, label_mode=label_mode)

            if not args.dev_only:
                ho_eval = [m for m in HOLDOUT_MONTHS if has_period_type(m, ptype)]
                ho_slice = holdout_root(ptype, ctype, base_dir=HOLDOUT_DIR)
                clear_month_cache()
                gc.collect()

                ho_pm = run_variant(variant, ho_eval, bs, class_type=ctype, period_type=ptype)
                if ho_pm:
                    save_results(variant, ho_pm, ho_eval, ho_slice,
                                 class_type=ctype, period_type=ptype, label_mode=label_mode)

            clear_month_cache()
            gc.collect()

    # Print comparison table
    print_comparison(variant, ptypes, ctypes)
    print(f"\n[main] All done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
