#!/usr/bin/env python
"""V10e holdout: run on 2024-2025 (24 months) and save to holdout/v10e/."""
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

from ml.config import REALIZED_DA_CACHE, LTRConfig, PipelineConfig
from ml.data_loader import load_train_val_test
from ml.evaluate import aggregate_months, evaluate_ltr
from ml.features import compute_query_groups, prepare_features
from ml.train import predict_scores, train_ltr_model

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "registry"
HOLDOUT = ROOT / "holdout"

V7_DA, V7_DMIX, V7_DORI = 0.85, 0.00, 0.15

# v10e config: 9 features
V10E_FEATURES = [
    "binding_freq_1", "binding_freq_3", "binding_freq_6", "binding_freq_12",
    "binding_freq_15", "v7_formula_score", "prob_exceed_110", "constraint_limit",
    "da_rank_value",
]
V10E_MONOTONE = [1, 1, 1, 1, 1, -1, 1, 0, -1]

HOLDOUT_MONTHS = [f"{y:04d}-{m:02d}" for y in (2024, 2025) for m in range(1, 13)]


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_all_binding_sets(cache_dir: str = REALIZED_DA_CACHE) -> dict[str, set[str]]:
    binding_sets: dict[str, set[str]] = {}
    for f in sorted(Path(cache_dir).glob("*.parquet")):
        df = pl.read_parquet(str(f))
        binding_sets[f.stem] = set(df.filter(pl.col("realized_sp") > 0)["constraint_id"].to_list())
    print(f"[bf] Loaded {len(binding_sets)} months of binding sets")
    return binding_sets


def compute_bf(cids: list[str], month: str, bs: dict[str, set[str]], lookback: int) -> np.ndarray:
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
    return freq / n


def enrich_df(df: pl.DataFrame, month: str, bs: dict[str, set[str]]) -> pl.DataFrame:
    cids = df["constraint_id"].to_list()
    df = df.with_columns(
        (V7_DA * pl.col("da_rank_value")
         + V7_DMIX * pl.col("density_mix_rank_value")
         + V7_DORI * pl.col("density_ori_rank_value")
        ).alias("v7_formula_score")
    )
    for lb in [1, 3, 6, 12, 15]:
        col_name = f"binding_freq_{lb}"
        if col_name not in df.columns:
            freq = compute_bf(cids, month, bs, lb)
            df = df.with_columns(pl.Series(col_name, freq))
    return df


def run_holdout(bs: dict[str, set[str]]) -> dict[str, dict]:
    print(f"\n[v10e holdout] 9f, {len(HOLDOUT_MONTHS)} months")

    cfg = PipelineConfig(
        ltr=LTRConfig(
            features=V10E_FEATURES,
            monotone_constraints=V10E_MONOTONE,
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

    for m in HOLDOUT_MONTHS:
        t0 = time.time()
        train_df, _, test_df = load_train_val_test(m, cfg.train_months, cfg.val_months, "f0", "onpeak")

        eval_ts = pd.Timestamp(m)
        train_month_strs = [(eval_ts - pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(8, 0, -1)]
        parts = []
        for tm in train_month_strs:
            part = train_df.filter(pl.col("query_month") == tm)
            if len(part) > 0:
                part = enrich_df(part, tm, bs)
                parts.append(part)
        train_df = pl.concat(parts)
        test_df = enrich_df(test_df, m, bs)

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
        print(f"  {m}: VC@20={metrics['VC@20']:.4f} VC@100={metrics['VC@100']:.4f} "
              f"binding={n_bind} ({elapsed:.1f}s)")

        del train_df, test_df, parts, X_train, y_train, groups, model, X_test, actual, scores
        gc.collect()

    return per_month


def save_holdout(per_month: dict[str, dict]) -> None:
    clean = {m: {k: v for k, v in met.items() if not k.startswith("_")} for m, met in per_month.items()}
    agg = aggregate_months(clean)

    d = HOLDOUT / "v10e"
    d.mkdir(parents=True, exist_ok=True)
    result = {
        "eval_config": {
            "eval_months": HOLDOUT_MONTHS,
            "class_type": "onpeak",
            "period_type": "f0",
            "mode": "holdout",
        },
        "per_month": clean,
        "aggregate": agg,
        "n_months": len(clean),
    }
    with open(d / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    means = agg["mean"]
    print(f"\n[v10e holdout] Saved to {d}/")
    print(f"  VC@20={means['VC@20']:.4f}  VC@50={means['VC@50']:.4f}  VC@100={means['VC@100']:.4f}")
    print(f"  R@20={means['Recall@20']:.4f}  NDCG={means['NDCG']:.4f}  Spearman={means['Spearman']:.4f}")


def print_comparison() -> None:
    versions = {}
    for vdir in sorted(HOLDOUT.iterdir()):
        mf = vdir / "metrics.json"
        if mf.exists():
            data = json.load(open(mf))
            versions[vdir.name] = data["aggregate"]["mean"]

    if not versions:
        return

    metrics_list = ["VC@10", "VC@20", "VC@25", "VC@50", "VC@100", "VC@200",
                    "Recall@10", "Recall@20", "Recall@50", "Recall@100", "NDCG", "Spearman"]

    print(f"\n{'=' * 100}")
    print("HOLDOUT COMPARISON (24 months, 2024-2025)")
    print(f"{'=' * 100}")
    header = f"{'Metric':<12}"
    for vid in versions:
        header += f" {vid:>10}"
    print(header)
    print("-" * (12 + 11 * len(versions)))

    for met in metrics_list:
        vals = {vid: m.get(met, 0) for vid, m in versions.items()}
        best = max(vals.values())
        row = f"{met:<12}"
        for vid in versions:
            v = vals[vid]
            star = "*" if v == best else " "
            row += f" {v:>9.4f}{star}"
        print(row)

    # Delta vs v0
    if "v0" in versions:
        v0 = versions["v0"]
        print(f"\nDelta vs v0 (formula):")
        for met in ["VC@20", "VC@50", "VC@100", "Recall@20", "NDCG", "Spearman"]:
            base = v0.get(met, 0)
            row = f"  {met:<12}"
            for vid in versions:
                v = versions[vid].get(met, 0)
                pct = (v - base) / base * 100 if base > 0 else 0
                row += f" {pct:>+9.1f}%"
            print(row)


def main() -> None:
    t_start = time.time()
    print(f"[main] V10e holdout, mem={mem_mb():.0f} MB")

    bs = load_all_binding_sets()
    per_month = run_holdout(bs)
    save_holdout(per_month)
    print_comparison()

    print(f"\n[main] Done in {time.time() - t_start:.1f}s, mem={mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
