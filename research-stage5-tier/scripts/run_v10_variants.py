#!/usr/bin/env python
"""DEPRECATED: Results archived to archive/registry/. Superseded by v10e-lag1.

V10 variants: find a feature set that beats v9 on ALL metrics.

v10 (6f) beats v9 on VC@50+, Spearman, NDCG but loses on VC@20 by 3%.
Goal: sharpen top-k precision while keeping broader gains.

Variants:
  v10c: 8f — v10 + bf_1 (last month) + bf_15
  v10d: 9f — v10c + binding_recency (1/months_since_last_binding)
  v10e: 9f — v10c + da_rank_value (was 6.1% importance in v9)
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
    REALIZED_DA_CACHE,
    LTRConfig,
    PipelineConfig,
    _FULL_EVAL_MONTHS,
)
from ml.data_loader import load_train_val_test
from ml.evaluate import aggregate_months, evaluate_ltr
from ml.features import compute_query_groups, prepare_features
from ml.train import predict_scores, train_ltr_model

REGISTRY = Path(__file__).resolve().parent.parent / "registry"
V7_DA, V7_DMIX, V7_DORI = 0.85, 0.00, 0.15


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ═══════════════════════════════════════════════════════════════════════════════
# Binding frequency features
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_binding_sets(cache_dir: str = REALIZED_DA_CACHE) -> dict[str, set[str]]:
    binding_sets: dict[str, set[str]] = {}
    for f in sorted(Path(cache_dir).glob("*.parquet")):
        df = pl.read_parquet(str(f))
        binding_sets[f.stem] = set(df.filter(pl.col("realized_sp") > 0)["constraint_id"].to_list())
    print(f"[bf] Loaded {len(binding_sets)} months of binding sets")
    return binding_sets


def compute_bf(cids: list[str], month: str, bs: dict[str, set[str]], lookback: int) -> np.ndarray:
    """Binding frequency: fraction of prior N months where constraint was binding."""
    prior = [m for m in sorted(bs.keys()) if m < month][-lookback:]
    n = len(prior)
    if n == 0:
        return np.zeros(len(cids), dtype=np.float64)
    freq = np.zeros(len(cids), dtype=np.float64)
    for m in prior:
        s = bs[m]
        for i, cid in enumerate(cids):
            if cid in s:
                freq[i] += 1
    return freq / n  # always normalize by actual months available


def compute_binding_recency(cids: list[str], month: str, bs: dict[str, set[str]]) -> np.ndarray:
    """Months since last binding. Returns 1/(months_since+1) so higher=more recent. 0 if never."""
    prior = [m for m in sorted(bs.keys()) if m < month]
    recency = np.zeros(len(cids), dtype=np.float64)
    for i, cid in enumerate(cids):
        for offset, m in enumerate(reversed(prior)):
            if cid in bs[m]:
                recency[i] = 1.0 / (offset + 1)
                break
    return recency


def enrich_df(df: pl.DataFrame, month: str, bs: dict[str, set[str]], feature_set: str) -> pl.DataFrame:
    """Add all computed features for a given feature_set."""
    cids = df["constraint_id"].to_list()

    # v7 formula
    df = df.with_columns(
        (V7_DA * pl.col("da_rank_value")
         + V7_DMIX * pl.col("density_mix_rank_value")
         + V7_DORI * pl.col("density_ori_rank_value")
        ).alias("v7_formula_score")
    )

    # Binding frequencies — always compute all, the feature list controls what's used
    for lb in [1, 3, 6, 12, 15]:
        col_name = f"binding_freq_{lb}"
        if col_name not in df.columns:
            freq = compute_bf(cids, month, bs, lb)
            df = df.with_columns(pl.Series(col_name, freq))

    # Binding recency
    if "binding_recency" not in df.columns:
        rec = compute_binding_recency(cids, month, bs)
        df = df.with_columns(pl.Series("binding_recency", rec))

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Run variant
# ═══════════════════════════════════════════════════════════════════════════════

def run_variant(
    version_id: str,
    features: list[str],
    monotone: list[int],
    eval_months: list[str],
    bs: dict[str, set[str]],
) -> dict[str, dict]:
    print(f"\n[{version_id}] {len(features)}f: {features}")

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
        train_df, _, test_df = load_train_val_test(m, cfg.train_months, cfg.val_months, "f0", "onpeak")

        # Enrich per training month (each gets its own temporal window)
        eval_ts = pd.Timestamp(m)
        train_month_strs = [(eval_ts - pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(8, 0, -1)]
        parts = []
        for tm in train_month_strs:
            part = train_df.filter(pl.col("query_month") == tm)
            if len(part) > 0:
                part = enrich_df(part, tm, bs, version_id)
                parts.append(part)
        train_df = pl.concat(parts)
        test_df = enrich_df(test_df, m, bs, version_id)

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

        # Feature importance
        if hasattr(model, "feature_importance"):
            imp = model.feature_importance(importance_type="gain")
            metrics["_fi"] = dict(zip(cfg.ltr.features, [float(x) for x in imp]))

        elapsed = time.time() - t0
        if m == eval_months[0] or m == eval_months[-1]:
            print(f"  {m}: VC@20={metrics['VC@20']:.4f} VC@100={metrics['VC@100']:.4f} ({elapsed:.1f}s)")

        del train_df, test_df, parts, X_train, y_train, groups, model, X_test, actual, scores
        gc.collect()

    # Aggregate
    clean = {m: {k: v for k, v in met.items() if not k.startswith("_")} for m, met in per_month.items()}
    agg = aggregate_months(clean)
    means = agg["mean"]
    print(f"  => VC@20={means['VC@20']:.4f} VC@50={means['VC@50']:.4f} VC@100={means['VC@100']:.4f} "
          f"R@20={means['Recall@20']:.4f} NDCG={means['NDCG']:.4f} Spearman={means['Spearman']:.4f}")

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


def save_version(vid: str, per_month: dict, eval_months: list[str], config_extra: dict) -> None:
    clean = {m: {k: v for k, v in met.items() if not k.startswith("_")} for m, met in per_month.items()}
    agg = aggregate_months(clean)
    d = REGISTRY / vid
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "metrics.json", "w") as f:
        json.dump({"eval_config": {"eval_months": eval_months, "class_type": "onpeak",
                                    "period_type": "f0"}, "per_month": clean,
                    "aggregate": agg, "n_months": len(clean)}, f, indent=2)
    with open(d / "config.json", "w") as f:
        json.dump(config_extra, f, indent=2)


def main() -> None:
    eval_months = _FULL_EVAL_MONTHS
    t_start = time.time()
    print(f"[main] V10 variants, {len(eval_months)} months, mem={mem_mb():.0f} MB")

    bs = load_all_binding_sets()

    # ── Define variants ──

    # v10c: v10 + bf_1 + bf_15
    v10c_f = ["binding_freq_1", "binding_freq_3", "binding_freq_6", "binding_freq_12",
              "binding_freq_15", "v7_formula_score", "prob_exceed_110", "constraint_limit"]
    v10c_m = [1, 1, 1, 1, 1, -1, 1, 0]

    # v10d: v10c + binding_recency
    v10d_f = v10c_f + ["binding_recency"]
    v10d_m = v10c_m + [1]

    # v10e: v10c + da_rank_value
    v10e_f = v10c_f + ["da_rank_value"]
    v10e_m = v10c_m + [-1]

    # v10f: v10d + da_rank_value (kitchen sink of useful features)
    v10f_f = v10c_f + ["binding_recency", "da_rank_value"]
    v10f_m = v10c_m + [1, -1]

    # v10g: v10f + ori_mean (flow feature, was 0.4% but might help for NEW binding)
    v10g_f = v10f_f + ["ori_mean"]
    v10g_m = v10f_m + [1]

    variants = {
        "v10c": (v10c_f, v10c_m),
        "v10d": (v10d_f, v10d_m),
        "v10e": (v10e_f, v10e_m),
        "v10f": (v10f_f, v10f_m),
        "v10g": (v10g_f, v10g_m),
    }

    results = {}
    for vid, (feat, mono) in variants.items():
        pm = run_variant(vid, feat, mono, eval_months, bs)
        results[vid] = pm
        save_version(vid, pm, eval_months, {"features": feat, "monotone": mono, "method": "lightgbm_regression"})

    # ── Comparison table ──
    # Load v9 and v10 for reference
    v9_means = json.load(open(REGISTRY / "v9" / "metrics.json"))["aggregate"]["mean"]
    v10_means = json.load(open(REGISTRY / "v10" / "metrics.json"))["aggregate"]["mean"]

    all_versions = {"v9 (14f)": v9_means, "v10 (6f)": v10_means}
    for vid, pm in results.items():
        clean = {m: {k: v for k, v in met.items() if not k.startswith("_")} for m, met in pm.items()}
        all_versions[vid] = aggregate_months(clean)["mean"]

    metrics_list = ["VC@10", "VC@20", "VC@25", "VC@50", "VC@100", "VC@200",
                    "Recall@10", "Recall@20", "Recall@50", "Recall@100", "NDCG", "Spearman"]

    print(f"\n{'=' * 100}")
    print("COMPARISON (36-month dev)")
    print(f"{'=' * 100}")
    header = f"{'Metric':<12}"
    for vid in all_versions:
        header += f" {vid:>12}"
    print(header)
    print("-" * (12 + 13 * len(all_versions)))

    for met in metrics_list:
        vals = {vid: m.get(met, 0) for vid, m in all_versions.items()}
        best = max(vals.values())
        row = f"{met:<12}"
        for vid in all_versions:
            v = vals[vid]
            star = "*" if v == best else " "
            row += f" {v:>11.4f}{star}"
        print(row)

    # Delta vs v9
    print(f"\nDelta vs v9:")
    print(f"{'Metric':<12}", end="")
    for vid in all_versions:
        print(f" {vid:>12}", end="")
    print()
    for met in ["VC@20", "VC@50", "VC@100", "Recall@20", "NDCG", "Spearman"]:
        base = v9_means.get(met, 0)
        row = f"{met:<12}"
        for vid in all_versions:
            v = all_versions[vid].get(met, 0)
            pct = (v - base) / base * 100 if base > 0 else 0
            row += f" {pct:>+11.1f}%"
        print(row)

    print(f"\n[main] Done in {time.time() - t_start:.1f}s, mem={mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
