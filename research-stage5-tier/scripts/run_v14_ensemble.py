#!/usr/bin/env python
"""V14: Two-model ensemble — address V7.0's new-binder blind spot.

Model A: V7.0 champion (9 features including BF) — strong for known binders
Model B: Structural model (NO BF features) — relies on da_rank, shadow_price_da,
         spice6, formula score. Trained on ALL constraints (not just BF-zero).

Blending rule:
  - If constraint has any BF signal (bf_1 or bf_3 or bf_6 > 0): use Model A score
  - If constraint has NO BF signal: use Model B score
  - v14b: soft blend — score = alpha * A + (1-alpha) * B, where alpha = min(1, bf_6 * K)

This gives BF-zero constraints a fighting chance via structural features,
without degrading the strong BF-driven signal for known binders.

Baseline: v10e-lag1 (9 features, lag=1) — the current V7.0 champion.
"""
from __future__ import annotations

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

# Model A: V7.0 champion features (9)
MODEL_A_FEATURES = [
    "binding_freq_1", "binding_freq_3", "binding_freq_6", "binding_freq_12",
    "binding_freq_15", "v7_formula_score", "prob_exceed_110", "constraint_limit",
    "da_rank_value",
]
MODEL_A_MONOTONE = [1, 1, 1, 1, 1, -1, 1, 0, -1]

# Model B: Structural features only (NO BF) — 5 features
MODEL_B_FEATURES = [
    "da_rank_value", "shadow_price_da", "v7_formula_score",
    "prob_exceed_110", "constraint_limit",
]
MODEL_B_MONOTONE = [-1, 1, -1, 1, 0]

HOLDOUT_MONTHS = [f"{y:04d}-{m:02d}" for y in (2024, 2025) for m in range(1, 13)]


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def prev_month(m: str) -> str:
    ts = pd.Timestamp(m)
    return (ts - pd.DateOffset(months=1)).strftime("%Y-%m")


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


def load_and_enrich_month(
    auction_month: str,
    period_type: str,
    class_type: str,
    cutoff_month: str,
    binding_sets: dict[str, set[str]],
    blend_weights: tuple[float, float, float],
) -> pl.DataFrame:
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

    return df


def build_train_test(
    eval_month: str,
    period_type: str,
    class_type: str,
    binding_sets: dict[str, set[str]],
    blend_weights: tuple[float, float, float],
    n_train: int = 8,
) -> tuple[pl.DataFrame, pl.DataFrame] | None:
    ptype_n = int(period_type[1:])
    lag = ptype_n + 1

    lagged_month = eval_month
    for _ in range(lag):
        lagged_month = prev_month(lagged_month)

    train_months = collect_usable_months(lagged_month, period_type, n_months=n_train)
    if not train_months:
        return None

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

    bf_cutoff_test = prev_month(lagged_month)
    test_df = load_and_enrich_month(eval_month, period_type, class_type, bf_cutoff_test, binding_sets, blend_weights)
    if len(test_df) == 0:
        return None

    test_df = test_df.with_columns(pl.lit(eval_month).alias("query_month"))
    return train_df, test_df


def run_ensemble(
    variant: str,
    period_type: str,
    class_type: str,
    eval_months: list[str],
    binding_sets: dict[str, set[str]],
    blend_weights: tuple[float, float, float],
    soft_blend_k: float = 5.0,
    save: bool = False,
) -> dict:
    """Run two-model ensemble."""
    cfg_a = LTRConfig(
        features=list(MODEL_A_FEATURES),
        monotone_constraints=list(MODEL_A_MONOTONE),
        backend="lightgbm", label_mode="tiered",
        n_estimators=100, learning_rate=0.05, num_leaves=31,
    )
    cfg_b = LTRConfig(
        features=list(MODEL_B_FEATURES),
        monotone_constraints=list(MODEL_B_MONOTONE),
        backend="lightgbm", label_mode="tiered",
        n_estimators=100, learning_rate=0.05, num_leaves=31,
    )

    month_results: dict[str, dict] = {}
    imp_a_list, imp_b_list = [], []
    ensemble_stats = {"n_bf_zero": 0, "n_bf_pos": 0, "n_bf_zero_bound": 0, "n_bf_pos_bound": 0}
    t0 = time.time()

    for em in eval_months:
        if not has_period_type(em, period_type):
            continue

        result = build_train_test(em, period_type, class_type, binding_sets, blend_weights)
        if result is None:
            continue

        train_df, test_df = result

        # Train Model A (full features)
        X_train_a, _ = prepare_features(train_df, cfg_a)
        y_train = train_df["realized_sp"].fill_null(0.0).to_numpy()
        groups_train = compute_query_groups(train_df)
        model_a = train_ltr_model(X_train_a, y_train, groups_train, cfg_a)

        # Train Model B (structural features only, same training data)
        X_train_b, _ = prepare_features(train_df, cfg_b)
        model_b = train_ltr_model(X_train_b, y_train, groups_train, cfg_b)

        # Predict with both models on test
        X_test_a, _ = prepare_features(test_df, cfg_a)
        X_test_b, _ = prepare_features(test_df, cfg_b)
        scores_a = predict_scores(model_a, X_test_a)
        scores_b = predict_scores(model_b, X_test_b)

        # Determine BF signal presence
        bf1 = test_df["binding_freq_1"].fill_null(0.0).to_numpy()
        bf3 = test_df["binding_freq_3"].fill_null(0.0).to_numpy()
        bf6 = test_df["binding_freq_6"].fill_null(0.0).to_numpy()
        has_bf = (bf1 > 0) | (bf3 > 0) | (bf6 > 0)

        actual = test_df["realized_sp"].fill_null(0.0).to_numpy()
        is_bound = actual > 0

        # Track stats
        ensemble_stats["n_bf_zero"] += (~has_bf).sum()
        ensemble_stats["n_bf_pos"] += has_bf.sum()
        ensemble_stats["n_bf_zero_bound"] += ((~has_bf) & is_bound).sum()
        ensemble_stats["n_bf_pos_bound"] += (has_bf & is_bound).sum()

        if variant == "v14a":
            # Hard switch: BF+ → Model A, BF-zero → Model B
            scores = np.where(has_bf, scores_a, scores_b)
        elif variant == "v14b":
            # Soft blend: alpha = min(1, bf6 * K)
            alpha = np.minimum(1.0, bf6 * soft_blend_k)
            scores = alpha * scores_a + (1.0 - alpha) * scores_b
        elif variant == "v14c":
            # Model A only (baseline comparison)
            scores = scores_a
        else:
            raise ValueError(f"Unknown variant: {variant}")

        metrics = evaluate_ltr(actual, scores)
        month_results[em] = metrics

        # Feature importance
        if hasattr(model_a, "feature_importance"):
            imp = model_a.feature_importance(importance_type="gain")
            imp_a_list.append(dict(zip(cfg_a.features, imp)))
        if hasattr(model_b, "feature_importance"):
            imp = model_b.feature_importance(importance_type="gain")
            imp_b_list.append(dict(zip(cfg_b.features, imp)))

    elapsed = time.time() - t0
    agg = aggregate_months(month_results)
    agg["walltime_s"] = round(elapsed, 1)
    agg["n_months"] = len(month_results)
    agg["ensemble_stats"] = ensemble_stats

    # Avg feature importance for both models
    for label, imp_list, cfg in [("model_a_importance", imp_a_list, cfg_a),
                                  ("model_b_importance", imp_b_list, cfg_b)]:
        if imp_list:
            avg_imp = {}
            for feat in cfg.features:
                vals = [fi.get(feat, 0) for fi in imp_list]
                avg_imp[feat] = round(np.mean(vals), 1)
            total = sum(avg_imp.values())
            if total > 0:
                agg[label] = {k: f"{v/total*100:.1f}%" for k, v in sorted(avg_imp.items(), key=lambda x: -x[1])}

    if save:
        is_holdout = eval_months == HOLDOUT_MONTHS
        root_fn = holdout_root if is_holdout else registry_root
        out_dir = root_fn(period_type, class_type) / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(agg, f, indent=2)
        print(f"[save] {out_dir / 'metrics.json'}")

    return agg


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["v14a", "v14b", "v14c", "all"], default="all")
    parser.add_argument("--ptype", default="f0")
    parser.add_argument("--ctype", default="onpeak")
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--blend-k", type=float, default=5.0, help="Soft blend sharpness for v14b")
    args = parser.parse_args()

    period_type = args.ptype
    class_type = args.ctype
    blend_weights = BLEND_WEIGHTS.get((period_type, class_type), (0.85, 0.00, 0.15))
    eval_months = HOLDOUT_MONTHS if args.holdout else _FULL_EVAL_MONTHS

    binding_sets = load_all_binding_sets(class_type)

    variants = ["v14a", "v14b", "v14c"] if args.variant == "all" else [args.variant]

    for variant in variants:
        print(f"\n{'='*70}")
        desc = {
            "v14a": "Hard switch (BF+ → Model A, BF-zero → Model B)",
            "v14b": f"Soft blend (alpha = min(1, bf6 * {args.blend_k}))",
            "v14c": "Model A only (sanity check = v10e-lag1)",
        }
        print(f"Running {variant}: {desc[variant]}")
        print(f"  {period_type}/{class_type}, {'holdout' if args.holdout else 'dev'}")
        print(f"{'='*70}")

        agg = run_ensemble(
            variant, period_type, class_type, eval_months,
            binding_sets, blend_weights,
            soft_blend_k=args.blend_k, save=args.save,
        )

        print(f"\n{variant} results:")
        vc20 = agg.get("mean", {}).get("VC@20", "?")
        recall20 = agg.get("mean", {}).get("Recall@20", "?")
        spearman = agg.get("mean", {}).get("Spearman", "?")
        ndcg = agg.get("mean", {}).get("NDCG", "?")
        print(f"  VC@20={vc20:.4f}  Recall@20={recall20:.4f}  NDCG={ndcg:.4f}  Spearman={spearman:.4f}")
        print(f"  walltime: {agg.get('walltime_s', '?')}s, months: {agg.get('n_months', '?')}")

        stats = agg.get("ensemble_stats", {})
        if stats:
            n_bfz = stats["n_bf_zero"]
            n_bfp = stats["n_bf_pos"]
            n_bfz_b = stats["n_bf_zero_bound"]
            n_bfp_b = stats["n_bf_pos_bound"]
            print(f"  BF-zero: {n_bfz} constraints ({n_bfz_b} bound, {n_bfz_b/max(n_bfz,1)*100:.1f}% bind rate)")
            print(f"  BF-pos:  {n_bfp} constraints ({n_bfp_b} bound, {n_bfp_b/max(n_bfp,1)*100:.1f}% bind rate)")

        if "model_a_importance" in agg:
            print(f"  Model A importance: {agg['model_a_importance']}")
        if "model_b_importance" in agg:
            print(f"  Model B importance: {agg['model_b_importance']}")

    print(f"\n[mem] Peak: {mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
