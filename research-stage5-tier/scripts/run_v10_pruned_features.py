#!/usr/bin/env python
"""DEPRECATED: Results archived to archive/registry/. Superseded by v10e-lag1.

V10 experiment: pruned features + multi-window binding frequency.

v10: 6 features — bf_3, bf_6, bf_12, v7_formula, prob_exceed_110, constraint_limit
v10b: 8 features — v10 + da_rank_value + prob_exceed_100 (check if pruning hurt)

Expected timing: ~15s for 36-month eval (fewer features = faster training).
"""
from __future__ import annotations

import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    REALIZED_DA_CACHE,
    V62B_SIGNAL_BASE,
    LTRConfig,
    PipelineConfig,
    _FULL_EVAL_MONTHS,
)
from ml.data_loader import load_train_val_test
from ml.evaluate import aggregate_months, evaluate_ltr
from ml.features import compute_query_groups, prepare_features
from ml.realized_da import load_realized_da
from ml.train import predict_scores, train_ltr_model

REGISTRY = Path(__file__).resolve().parent.parent / "registry"

V7_DA = 0.85
V7_DMIX = 0.00
V7_DORI = 0.15


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ═══════════════════════════════════════════════════════════════════════════════
# Binding frequency
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_binding_sets(cache_dir: str = REALIZED_DA_CACHE) -> dict[str, set[str]]:
    binding_sets: dict[str, set[str]] = {}
    for f in sorted(Path(cache_dir).glob("*.parquet")):
        df = pl.read_parquet(str(f))
        binding_sets[f.stem] = set(df.filter(pl.col("realized_sp") > 0)["constraint_id"].to_list())
    print(f"[binding_freq] Loaded binding sets for {len(binding_sets)} months")
    return binding_sets


def compute_binding_freq(
    cids: list[str],
    target_month: str,
    binding_sets: dict[str, set[str]],
    lookback: int,
) -> np.ndarray:
    all_months = sorted(binding_sets.keys())
    prior = [m for m in all_months if m < target_month][-lookback:]
    if not prior:
        return np.zeros(len(cids), dtype=np.float64)
    freq = np.zeros(len(cids), dtype=np.float64)
    for m in prior:
        bs = binding_sets[m]
        for i, cid in enumerate(cids):
            if cid in bs:
                freq[i] += 1
    return freq / len(prior)


def add_all_bf_columns(
    df: pl.DataFrame,
    month: str,
    binding_sets: dict[str, set[str]],
) -> pl.DataFrame:
    cids = df["constraint_id"].to_list()
    for lb in [3, 6, 12]:
        freq = compute_binding_freq(cids, month, binding_sets, lb)
        df = df.with_columns(pl.Series(f"binding_freq_{lb}", freq))
    return df


def add_v7_formula(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (V7_DA * pl.col("da_rank_value")
         + V7_DMIX * pl.col("density_mix_rank_value")
         + V7_DORI * pl.col("density_ori_rank_value")
        ).alias("v7_formula_score")
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Run ML variant
# ═══════════════════════════════════════════════════════════════════════════════

def run_variant(
    version_id: str,
    features: list[str],
    monotone: list[int],
    eval_months: list[str],
    binding_sets: dict[str, set[str]],
) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    print(f"\n[{version_id}] ML regression, {len(features)}f, {len(eval_months)} months")
    print(f"  Features: {features}")

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

    import pandas as pd
    per_month = {}
    per_month_scores = {}

    for m in eval_months:
        t0 = time.time()
        train_df, _, test_df = load_train_val_test(m, cfg.train_months, cfg.val_months, "f0", "onpeak")

        # Add v7 formula
        train_df = add_v7_formula(train_df)
        test_df = add_v7_formula(test_df)

        # Add binding_freq per training month
        eval_ts = pd.Timestamp(m)
        train_month_strs = [(eval_ts - pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(8, 0, -1)]
        train_parts = []
        for tm in train_month_strs:
            part = train_df.filter(pl.col("query_month") == tm)
            if len(part) > 0:
                part = add_all_bf_columns(part, tm, binding_sets)
                train_parts.append(part)
        train_df = pl.concat(train_parts) if train_parts else train_df

        test_df = add_all_bf_columns(test_df, m, binding_sets)

        train_df = train_df.sort("query_month")
        X_train, _ = prepare_features(train_df, cfg.ltr)
        y_train = train_df["realized_sp"].to_numpy().astype(np.float64)
        groups_train = compute_query_groups(train_df)

        model = train_ltr_model(X_train, y_train, groups_train, cfg.ltr)
        X_test, _ = prepare_features(test_df, cfg.ltr)
        scores = predict_scores(model, X_test)
        actual = test_df["realized_sp"].to_numpy().astype(np.float64)

        metrics = evaluate_ltr(actual, scores)
        per_month[m] = metrics
        per_month_scores[m] = scores

        if hasattr(model, "feature_importance"):
            importance = model.feature_importance(importance_type="gain")
            metrics["_feature_importance"] = {
                name: float(imp)
                for name, imp in sorted(
                    zip(cfg.ltr.features, importance),
                    key=lambda x: x[1],
                    reverse=True,
                )
            }

        elapsed = time.time() - t0
        n_binding = int((actual > 0).sum())
        print(f"  {m}: VC@20={metrics['VC@20']:.4f}, VC@100={metrics['VC@100']:.4f}, "
              f"binding={n_binding} ({elapsed:.1f}s)")

        del train_df, test_df, train_parts, X_train, y_train, groups_train, model, X_test, actual
        gc.collect()

    return per_month, per_month_scores


def save_version(version_id: str, per_month: dict, eval_months: list[str],
                 config_extra: dict | None = None) -> None:
    clean = {}
    for m, metrics in per_month.items():
        clean[m] = {k: v for k, v in metrics.items() if not k.startswith("_")}

    agg = aggregate_months(clean)
    version_dir = REGISTRY / version_id
    version_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "eval_config": {"eval_months": eval_months, "class_type": "onpeak",
                        "period_type": "f0", "mode": "eval"},
        "per_month": clean,
        "aggregate": agg,
        "n_months": len(clean),
    }
    with open(version_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    if config_extra:
        with open(version_dir / "config.json", "w") as f:
            json.dump(config_extra, f, indent=2)

    means = agg["mean"]
    print(f"\n[{version_id}] Saved to {version_dir}/")
    print(f"  VC@20={means['VC@20']:.4f}  VC@100={means['VC@100']:.4f}  "
          f"R@20={means['Recall@20']:.4f}  NDCG={means['NDCG']:.4f}  "
          f"Spearman={means['Spearman']:.4f}")


def print_importance(per_month: dict) -> None:
    feat_sums: dict[str, float] = {}
    feat_counts: dict[str, int] = {}
    for m, metrics in per_month.items():
        fi = metrics.get("_feature_importance", {})
        for name, imp in fi.items():
            feat_sums[name] = feat_sums.get(name, 0) + imp
            feat_counts[name] = feat_counts.get(name, 0) + 1
    if not feat_sums:
        return
    feat_avg = {n: feat_sums[n] / feat_counts[n] for n in feat_sums}
    total = sum(feat_avg.values())
    print("\n  Feature importance (avg gain):")
    for name, avg in sorted(feat_avg.items(), key=lambda x: x[1], reverse=True):
        print(f"    {name:<25s} {avg:>8.1f} ({100 * avg / total:.1f}%)")


def main() -> None:
    eval_months = _FULL_EVAL_MONTHS
    t_start = time.time()

    print(f"[main] V10 pruned features experiment, {len(eval_months)} months, mem={mem_mb():.0f} MB")

    binding_sets = load_all_binding_sets()

    # ── V10: 6 features (pruned) ──
    v10_features = [
        "binding_freq_3",
        "binding_freq_6",
        "binding_freq_12",
        "v7_formula_score",
        "prob_exceed_110",
        "constraint_limit",
    ]
    v10_monotone = [1, 1, 1, -1, 1, 0]

    v10_pm, v10_scores = run_variant("v10", v10_features, v10_monotone, eval_months, binding_sets)
    save_version("v10", v10_pm, eval_months, config_extra={
        "method": "lightgbm_regression",
        "features": v10_features,
        "monotone": v10_monotone,
        "label_mode": "tiered",
        "rationale": "Pruned to top features: 3 binding_freq windows + formula + best 2 spice6",
    })
    print_importance(v10_pm)

    # ── V10b: 8 features (v10 + da_rank_value + prob_exceed_100) ──
    v10b_features = [
        "binding_freq_3",
        "binding_freq_6",
        "binding_freq_12",
        "v7_formula_score",
        "da_rank_value",
        "prob_exceed_110",
        "prob_exceed_100",
        "constraint_limit",
    ]
    v10b_monotone = [1, 1, 1, -1, -1, 1, 1, 0]

    v10b_pm, v10b_scores = run_variant("v10b", v10b_features, v10b_monotone, eval_months, binding_sets)
    save_version("v10b", v10b_pm, eval_months, config_extra={
        "method": "lightgbm_regression",
        "features": v10b_features,
        "monotone": v10b_monotone,
        "label_mode": "tiered",
        "rationale": "v10 + da_rank_value + prob_exceed_100 to check if pruning hurt",
    })
    print_importance(v10b_pm)

    # ── Comparison ──
    # Load v9 for reference
    v9_data = json.load(open(REGISTRY / "v9" / "metrics.json"))
    v9_means = v9_data["aggregate"]["mean"]

    v10_agg = aggregate_months({m: {k: v for k, v in met.items() if not k.startswith("_")} for m, met in v10_pm.items()})
    v10b_agg = aggregate_months({m: {k: v for k, v in met.items() if not k.startswith("_")} for m, met in v10b_pm.items()})

    print(f"\n{'='*70}")
    print("COMPARISON (36-month dev)")
    print(f"{'='*70}")
    metrics_list = ["VC@10", "VC@20", "VC@25", "VC@50", "VC@100", "VC@200",
                    "Recall@10", "Recall@20", "Recall@50", "Recall@100", "NDCG", "Spearman"]
    print(f"{'Metric':<12} {'v9 (14f)':>10} {'v10 (6f)':>10} {'v10b (8f)':>10}")
    print("-" * 45)
    for met in metrics_list:
        v9v = v9_means.get(met, 0)
        v10v = v10_agg["mean"].get(met, 0)
        v10bv = v10b_agg["mean"].get(met, 0)
        best = max(v9v, v10v, v10bv)
        v9s = f"{v9v:.4f}{'*' if v9v == best else ' '}"
        v10s = f"{v10v:.4f}{'*' if v10v == best else ' '}"
        v10bs = f"{v10bv:.4f}{'*' if v10bv == best else ' '}"
        print(f"{met:<12} {v9s:>10} {v10s:>10} {v10bs:>10}")

    total = time.time() - t_start
    print(f"\n[main] All done in {total:.1f}s, mem={mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
