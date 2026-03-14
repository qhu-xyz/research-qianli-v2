#!/usr/bin/env python
"""V13: New-binder-aware features — address V7.0's blind spot for new constraints.

V7.0 relies heavily on binding_freq (50-70% importance), giving 0% T0 alarm
for constraints that have never bound. This experiment adds features to help
the model score new/latent binders:

v13a: Add shadow_price_da as raw feature (not just da_rank_value rank)
v13b: v13a + has_bf_signal binary indicator (lets model learn separate paths)
v13c: v13b + da_rank_no_bf interaction (da_rank_value * (1 - bf_6))

All variants use the same 8mo train, lag=N+1, LambdaRank tiered setup as v10e-lag1.

Baseline: v10e-lag1 (9 features) — the current V7.0 champion.
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
    _FULL_EVAL_MONTHS,
    collect_usable_months, has_period_type,
)
from ml.data_loader import load_v62b_month
from ml.evaluate import aggregate_months, evaluate_ltr
from ml.features import compute_query_groups, prepare_features
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

# ── Feature sets for each variant ──

# v13a: baseline + shadow_price_da (10 features)
V13A_FEATURES = [
    "binding_freq_1", "binding_freq_3", "binding_freq_6", "binding_freq_12",
    "binding_freq_15", "v7_formula_score", "prob_exceed_110", "constraint_limit",
    "da_rank_value", "shadow_price_da",
]
V13A_MONOTONE = [1, 1, 1, 1, 1, -1, 1, 0, -1, 1]  # higher shadow_price_da = more binding

# v13b: v13a + has_bf_signal (11 features)
V13B_FEATURES = V13A_FEATURES + ["has_bf_signal"]
V13B_MONOTONE = V13A_MONOTONE + [0]  # no monotone constraint on binary indicator

# v13c: v13b + da_rank_no_bf interaction (12 features)
V13C_FEATURES = V13B_FEATURES + ["da_rank_no_bf"]
V13C_MONOTONE = V13B_MONOTONE + [-1]  # lower da_rank when bf=0 → more binding

VARIANTS = {
    "v13a": (V13A_FEATURES, V13A_MONOTONE),
    "v13b": (V13B_FEATURES, V13B_MONOTONE),
    "v13c": (V13C_FEATURES, V13C_MONOTONE),
}

HOLDOUT_MONTHS = [f"{y:04d}-{m:02d}" for y in (2024, 2025) for m in range(1, 13)]


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def prev_month(m: str) -> str:
    ts = pd.Timestamp(m)
    return (ts - pd.DateOffset(months=1)).strftime("%Y-%m")


# ── Binding freq computation ──

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


def compute_binding_freq(
    constraint_ids: list[str],
    cutoff_month: str,
    windows: list[int],
    binding_sets: dict[str, set[str]],
) -> dict[str, list[float]]:
    """Compute binding frequency features for given constraints up to cutoff_month."""
    ts = pd.Timestamp(cutoff_month)
    results: dict[str, list[float]] = {f"binding_freq_{w}": [] for w in windows}

    for cid in constraint_ids:
        for w in windows:
            bound_count = 0
            total = 0
            for i in range(1, w + 1):
                m = (ts - pd.DateOffset(months=i)).strftime("%Y-%m")
                bs = binding_sets.get(m)
                if bs is not None:
                    total += 1
                    if cid in bs:
                        bound_count += 1
            freq = bound_count / total if total > 0 else 0.0
            results[f"binding_freq_{w}"].append(freq)

    return results


def add_new_binder_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add the new-binder-awareness features to a dataframe.

    Adds:
    - has_bf_signal: 1 if any of bf_1..bf_6 > 0, else 0
    - da_rank_no_bf: da_rank_value * (1 - bf_6) — amplifies da_rank when BF is absent
    """
    # has_bf_signal: any short-window BF > 0
    bf_cols = [c for c in df.columns if c.startswith("binding_freq_")]
    short_bf = [c for c in bf_cols if int(c.split("_")[-1]) <= 6]

    if short_bf:
        has_bf_expr = pl.lit(0.0)
        for c in short_bf:
            has_bf_expr = pl.when(pl.col(c) > 0).then(1.0).otherwise(has_bf_expr)
        df = df.with_columns(has_bf_expr.alias("has_bf_signal"))
    else:
        df = df.with_columns(pl.lit(0.0).alias("has_bf_signal"))

    # da_rank_no_bf: interaction — amplifies da_rank when no BF signal
    if "binding_freq_6" in df.columns and "da_rank_value" in df.columns:
        df = df.with_columns(
            (pl.col("da_rank_value") * (1.0 - pl.col("binding_freq_6")))
            .alias("da_rank_no_bf")
        )
    else:
        df = df.with_columns(pl.lit(0.0).alias("da_rank_no_bf"))

    return df


def load_and_enrich_month(
    auction_month: str,
    period_type: str,
    class_type: str,
    cutoff_month: str,
    binding_sets: dict[str, set[str]],
    blend_weights: tuple[float, float, float],
) -> pl.DataFrame:
    """Load V6.2B month, add BF features, formula score, and new-binder features."""
    df = load_v62b_month(auction_month, period_type, class_type)
    if df is None or len(df) == 0:
        return pl.DataFrame()

    cids = df["constraint_id"].to_list()
    bf = compute_binding_freq(cids, cutoff_month, [1, 3, 6, 12, 15], binding_sets)

    for col_name, vals in bf.items():
        df = df.with_columns(pl.Series(name=col_name, values=vals))

    w_da, w_dmix, w_dori = blend_weights
    if all(c in df.columns for c in ["da_rank_value", "density_mix_rank_value", "density_ori_rank_value"]):
        df = df.with_columns(
            (w_da * pl.col("da_rank_value")
             + w_dmix * pl.col("density_mix_rank_value")
             + w_dori * pl.col("density_ori_rank_value")
            ).alias("v7_formula_score")
        )

    # Add new-binder features
    df = add_new_binder_features(df)

    return df


def build_train_test(
    eval_month: str,
    period_type: str,
    class_type: str,
    binding_sets: dict[str, set[str]],
    blend_weights: tuple[float, float, float],
    n_train: int = 8,
) -> tuple[pl.DataFrame, pl.DataFrame] | None:
    """Build training and test datasets with proper lag."""
    ptype_n = int(period_type[1:])
    lag = ptype_n + 1

    lagged_month = eval_month
    for _ in range(lag):
        lagged_month = prev_month(lagged_month)

    train_months = collect_usable_months(lagged_month, period_type, n_months=n_train)
    if not train_months:
        return None

    # BF cutoff for training: each train month uses its own lag-adjusted cutoff
    train_dfs = []
    for tm in train_months:
        bf_cutoff = prev_month(tm)
        df = load_and_enrich_month(tm, period_type, class_type, bf_cutoff, binding_sets, blend_weights)
        if len(df) > 0:
            df = df.with_columns(pl.lit(tm).alias("query_month"))
            train_dfs.append(df)

    if not train_dfs:
        return None

    train_df = pl.concat(train_dfs)

    # Test: BF cutoff for test month uses lagged cutoff
    bf_cutoff_test = prev_month(lagged_month)
    test_df = load_and_enrich_month(eval_month, period_type, class_type, bf_cutoff_test, binding_sets, blend_weights)
    if len(test_df) == 0:
        return None

    test_df = test_df.with_columns(pl.lit(eval_month).alias("query_month"))

    return train_df, test_df


def run_variant(
    variant: str,
    features: list[str],
    monotone: list[int],
    period_type: str,
    class_type: str,
    eval_months: list[str],
    binding_sets: dict[str, set[str]],
    blend_weights: tuple[float, float, float],
    save: bool = False,
) -> dict:
    """Run one variant on one slice."""
    cfg = LTRConfig(
        features=list(features),
        monotone_constraints=list(monotone),
        backend="lightgbm",
        label_mode="tiered",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
    )

    month_results: dict[str, dict] = {}
    feature_importances = []
    t0 = time.time()

    for em in eval_months:
        if not has_period_type(em, period_type):
            continue

        result = build_train_test(em, period_type, class_type, binding_sets, blend_weights)
        if result is None:
            continue

        train_df, test_df = result
        X_train, _ = prepare_features(train_df, cfg)
        y_train = train_df["realized_sp"].fill_null(0.0).to_numpy()
        groups_train = compute_query_groups(train_df)

        model = train_ltr_model(X_train, y_train, groups_train, cfg)

        X_test, _ = prepare_features(test_df, cfg)
        scores = predict_scores(model, X_test)
        metrics = evaluate_ltr(test_df["realized_sp"].fill_null(0.0).to_numpy(), scores)
        month_results[em] = metrics

        # Feature importance
        if hasattr(model, "feature_importance"):
            imp = model.feature_importance(importance_type="gain")
            feature_importances.append(dict(zip(cfg.features, imp)))

    elapsed = time.time() - t0
    agg = aggregate_months(month_results)
    agg["walltime_s"] = round(elapsed, 1)
    agg["n_months"] = len(month_results)

    # Avg feature importance
    if feature_importances:
        avg_imp = {}
        for feat in cfg.features:
            vals = [fi.get(feat, 0) for fi in feature_importances]
            avg_imp[feat] = round(np.mean(vals), 1)
        total = sum(avg_imp.values())
        if total > 0:
            avg_imp_pct = {k: f"{v/total*100:.1f}%" for k, v in sorted(avg_imp.items(), key=lambda x: -x[1])}
            agg["feature_importance"] = avg_imp_pct

    if save:
        is_holdout = eval_months == HOLDOUT_MONTHS
        root_fn = holdout_root if is_holdout else registry_root
        out_dir = root_fn(period_type, class_type) / variant
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = out_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"[save] {metrics_path}")

        config_path = out_dir / "config.json"
        pcfg = PipelineConfig(ltr=cfg)
        with open(config_path, "w") as f:
            json.dump(pcfg.to_dict(), f, indent=2)

    return agg


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS.keys()) + ["all"], default="all")
    parser.add_argument("--ptype", default="f0")
    parser.add_argument("--ctype", default="onpeak")
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    period_type = args.ptype
    class_type = args.ctype
    blend_weights = BLEND_WEIGHTS.get((period_type, class_type), (0.85, 0.00, 0.15))

    eval_months = HOLDOUT_MONTHS if args.holdout else _FULL_EVAL_MONTHS

    # Load binding sets
    binding_sets = load_all_binding_sets(class_type)

    variants_to_run = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]

    for variant in variants_to_run:
        features, monotone = VARIANTS[variant]
        print(f"\n{'='*70}")
        print(f"Running {variant} on {period_type}/{class_type} ({'holdout' if args.holdout else 'dev'})")
        print(f"Features ({len(features)}): {features}")
        print(f"{'='*70}")

        agg = run_variant(
            variant, features, monotone,
            period_type, class_type, eval_months,
            binding_sets, blend_weights,
            save=args.save,
        )

        print(f"\n{variant} results:")
        for k, v in agg.items():
            if k not in ("monthly_detail", "feature_importance"):
                print(f"  {k}: {v}")
        if "feature_importance" in agg:
            print(f"  feature_importance: {agg['feature_importance']}")

    print(f"\n[mem] Peak: {mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
