#!/usr/bin/env python
"""Stage6: v10e-style model with partial-month historical rebuild.

This keeps the safe row boundary from `v10e-lag1`:

- training rows stop at `M-2`

and recovers recent information only through feature rebuilding:

- for target month `M`, the previous month `M-1` contributes through
  `look_back_days`
- older months remain full-month

Important:

- `look_back_days=31` is a feature-rebuild equivalence check, not a full
  reproduction of leaky `v10e`, because the `M-1` training row is still
  excluded on purpose.
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

from ml.config import LTRConfig, PipelineConfig, _FULL_EVAL_MONTHS
from ml.data_loader import load_train_val_test, load_v62b_month
from ml.evaluate import aggregate_months, evaluate_ltr
from ml.features import compute_query_groups, prepare_features
from ml.lookback_history import add_binding_freq_columns_asof
from ml.train import predict_scores, train_ltr_model

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "registry"
HOLDOUT = ROOT / "holdout"

V7_DA, V7_DMIX, V7_DORI = 0.85, 0.00, 0.15
HOLDOUT_MONTHS = [f"{y:04d}-{m:02d}" for y in (2024, 2025) for m in range(1, 13)]

FEATURES = [
    "binding_freq_1", "binding_freq_3", "binding_freq_6", "binding_freq_12",
    "binding_freq_15", "v7_formula_score", "prob_exceed_110", "constraint_limit",
    "da_rank_value",
]
MONOTONE = [1, 1, 1, 1, 1, -1, 1, 0, -1]


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def enrich_df(df: pl.DataFrame, month: str, look_back_days: int) -> pl.DataFrame:
    df = df.with_columns(
        (V7_DA * pl.col("da_rank_value")
         + V7_DMIX * pl.col("density_mix_rank_value")
         + V7_DORI * pl.col("density_ori_rank_value")
        ).alias("v7_formula_score")
    )
    return add_binding_freq_columns_asof(df, month, look_back_days=look_back_days)


def run_variant(label: str, eval_months: list[str], look_back_days: int) -> dict[str, dict]:
    print(f"\n[{label}] {len(eval_months)} months, look_back_days={look_back_days}")
    cfg = PipelineConfig(
        ltr=LTRConfig(
            features=FEATURES,
            monotone_constraints=MONOTONE,
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
        eval_ts = pd.Timestamp(m)

        # Keep the safe row boundary from lag1: train on M-9 .. M-2.
        shifted_eval = (eval_ts - pd.DateOffset(months=1)).strftime("%Y-%m")
        train_df, _, _ = load_train_val_test(shifted_eval, cfg.train_months, cfg.val_months, "f0", "onpeak")
        train_months = [
            (eval_ts - pd.DateOffset(months=i)).strftime("%Y-%m")
            for i in range(9, 1, -1)
        ]

        parts = []
        for tm in train_months:
            part = train_df.filter(pl.col("query_month") == tm)
            if len(part) > 0:
                parts.append(enrich_df(part, tm, look_back_days))
        if not parts:
            print(f"  {m}: SKIP (no training data)")
            continue

        train_df = pl.concat(parts).sort("query_month")
        test_df = load_v62b_month(m, "f0", "onpeak").with_columns(pl.lit(m).alias("query_month"))
        test_df = enrich_df(test_df, m, look_back_days)

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

        if m == eval_months[0] or m == eval_months[-1]:
            print(f"  {m}: VC@20={metrics['VC@20']:.4f} VC@100={metrics['VC@100']:.4f} "
                  f"train={len(train_df)} ({time.time()-t0:.1f}s)")

        del train_df, test_df, parts, X_train, y_train, groups, model, X_test, actual, scores
        gc.collect()

    return per_month


def save_results(label: str, per_month: dict[str, dict], eval_months: list[str], dest_dir: Path, look_back_days: int) -> None:
    clean = {m: {k: v for k, v in met.items() if not k.startswith("_")} for m, met in per_month.items()}
    agg = aggregate_months(clean)
    d = dest_dir / label
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "metrics.json", "w") as f:
        json.dump({
            "eval_config": {
                "eval_months": eval_months,
                "class_type": "onpeak",
                "period_type": "f0",
                "look_back_days": look_back_days,
                "note": "safe M-2 row boundary with partial M-1 feature rebuild",
            },
            "per_month": clean,
            "aggregate": agg,
            "n_months": len(clean),
        }, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--look-back-days", type=int, default=12)
    parser.add_argument("--mode", choices=["dev", "holdout"], default="dev")
    args = parser.parse_args()

    months = _FULL_EVAL_MONTHS if args.mode == "dev" else HOLDOUT_MONTHS
    label = f"v10e-lookback{args.look_back_days}"
    if args.mode == "holdout":
        label += "-holdout"

    print(f"[main] {label}, mem={mem_mb():.0f} MB")
    per_month = run_variant(label, months, args.look_back_days)
    dest = REGISTRY if args.mode == "dev" else HOLDOUT
    save_results(label.replace("-holdout", ""), per_month, months, dest, args.look_back_days)


if __name__ == "__main__":
    main()
