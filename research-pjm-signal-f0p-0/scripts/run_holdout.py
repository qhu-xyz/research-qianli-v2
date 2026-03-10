#!/usr/bin/env python
"""Holdout evaluation for PJM V7.0b: v2 ML on 2024-2025.

Saves metrics and per-month predictions (constraint_id, rank) for Task 14 analysis.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
    python scripts/run_holdout.py
    python scripts/run_holdout.py --ptype f0 --class-type onpeak
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
    REALIZED_DA_CACHE, LTRConfig, PipelineConfig, V10E_FEATURES, V10E_MONOTONE,
    HOLDOUT_MONTHS, PJM_CLASS_TYPES,
    has_period_type, collect_usable_months,
)
from ml.data_loader import load_v62b_month
from ml.evaluate import aggregate_months, evaluate_ltr
from ml.features import compute_query_groups, prepare_features
from ml.registry_paths import registry_root, holdout_root
from ml.train import predict_scores, train_ltr_model

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "registry"
HOLDOUT_DIR = ROOT / "holdout"


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def get_blend_weights(ptype: str, ctype: str) -> tuple[float, float, float]:
    """Read optimized blend from registry config, fall back to defaults."""
    config_path = registry_root(ptype, ctype, base_dir=REGISTRY) / "v2" / "config.json"
    if config_path.exists():
        cfg = json.load(open(config_path))
        bw = cfg.get("blend_weights", {})
        if bw:
            return (bw.get("da", 0.85), bw.get("dmix", 0.0), bw.get("dori", 0.15))
    # Defaults
    defaults = {
        ("f0", "onpeak"): (0.85, 0.00, 0.15),
        ("f0", "dailyoffpeak"): (0.85, 0.00, 0.15),
        ("f0", "wkndonpeak"): (0.85, 0.00, 0.15),
        ("f1", "onpeak"): (0.70, 0.00, 0.30),
        ("f1", "dailyoffpeak"): (0.80, 0.00, 0.20),
        ("f1", "wkndonpeak"): (0.80, 0.00, 0.20),
    }
    return defaults.get((ptype, ctype), (0.85, 0.00, 0.15))


def load_all_binding_sets(peak_type: str, cache_dir: str = REALIZED_DA_CACHE) -> dict[str, set[str]]:
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
    return binding_sets


def compute_bf(branch_names: list[str], month: str,
               bs: dict[str, set[str]], lookback: int) -> np.ndarray:
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


def enrich_df(df: pl.DataFrame, month: str, bs: dict[str, set[str]],
              blend: tuple[float, float, float]) -> pl.DataFrame:
    cutoff = prev_month(month)
    w_da, w_dmix, w_dori = blend
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


def run_holdout_slice(ptype: str, ctype: str) -> dict[str, dict]:
    blend = get_blend_weights(ptype, ctype)
    ho_months = [m for m in HOLDOUT_MONTHS if has_period_type(m, ptype)]
    print(f"\n{'='*60}")
    print(f"[holdout] {ptype}/{ctype} ({len(ho_months)} months, blend={blend})")
    print(f"{'='*60}")

    bs = load_all_binding_sets(peak_type=ctype)

    cfg = PipelineConfig(
        ltr=LTRConfig(
            features=list(V10E_FEATURES),
            monotone_constraints=list(V10E_MONOTONE),
            backend="lightgbm",
            label_mode="tiered",
        ),
        train_months=8,
        val_months=0,
    )

    per_month: dict[str, dict] = {}
    pred_dir = holdout_root(ptype, ctype, base_dir=HOLDOUT_DIR) / "v2" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for m in ho_months:
        t0 = time.time()
        train_month_strs = collect_usable_months(m, ptype, n_months=8)
        if not train_month_strs:
            print(f"  {m}: SKIP (insufficient history)")
            continue
        train_month_strs = list(reversed(train_month_strs))

        parts = []
        for tm in train_month_strs:
            try:
                part = load_v62b_month(tm, ptype, ctype)
                part = part.with_columns(pl.lit(tm).alias("query_month"))
                part = enrich_df(part, tm, bs, blend)
                parts.append(part)
            except FileNotFoundError:
                pass
        if not parts:
            continue
        train_df = pl.concat(parts)

        try:
            test_df = load_v62b_month(m, ptype, ctype)
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

        metrics = evaluate_ltr(actual, scores)
        per_month[m] = metrics

        # Save per-month predictions (constraint_id, rank)
        order = np.argsort(-scores)  # descending
        rank = np.empty(len(scores), dtype=np.float64)
        rank[order] = (np.arange(len(scores)) + 1) / len(scores)
        pred_df = pl.DataFrame({
            "constraint_id": test_df["constraint_id"].cast(pl.String),
            "rank": rank,
        })
        pred_df.write_parquet(str(pred_dir / f"{m}.parquet"))

        elapsed = time.time() - t0
        n_bind = int((actual > 0).sum())
        print(f"  {m}: VC@20={metrics['VC@20']:.4f} binding={n_bind} ({elapsed:.1f}s)")

        del train_df, test_df, parts, X_train, y_train, groups, model, X_test, actual, scores
        gc.collect()

    # Save aggregate metrics
    if per_month:
        agg = aggregate_months(per_month)
        ho_dir = holdout_root(ptype, ctype, base_dir=HOLDOUT_DIR) / "v2"
        ho_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "eval_config": {"eval_months": sorted(per_month.keys()), "class_type": ctype,
                            "period_type": ptype, "mode": "holdout", "lag": 1},
            "per_month": per_month, "aggregate": agg,
            "n_months": len(per_month), "n_months_requested": len(ho_months),
            "skipped_months": sorted(set(ho_months) - set(per_month.keys())),
        }
        with open(ho_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)

        means = agg["mean"]
        print(f"\n[holdout] Aggregate ({len(per_month)} months):")
        for met in ["VC@20", "VC@50", "VC@100", "Recall@20", "NDCG", "Spearman"]:
            print(f"  {met:<12} {means.get(met, 0):.4f}")

    return per_month


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptype", default=None)
    parser.add_argument("--class-type", default=None)
    args = parser.parse_args()

    ptypes = [args.ptype] if args.ptype else ["f0", "f1"]
    ctypes = [args.class_type] if args.class_type else PJM_CLASS_TYPES
    t_start = time.time()

    for ptype in ptypes:
        for ctype in ctypes:
            run_holdout_slice(ptype, ctype)

    # Print summary table
    print(f"\n{'='*70}")
    print("HOLDOUT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Slice':<25} {'v0 HO':>8} {'v2 HO':>8} {'Delta':>8}")
    print("-" * 50)
    for ptype in ptypes:
        for ctype in ctypes:
            v0_path = holdout_root(ptype, ctype, base_dir=HOLDOUT_DIR) / "v0" / "metrics.json"
            v2_path = holdout_root(ptype, ctype, base_dir=HOLDOUT_DIR) / "v2" / "metrics.json"
            v0_vc = json.load(open(v0_path))["aggregate"]["mean"]["VC@20"] if v0_path.exists() else 0
            v2_vc = json.load(open(v2_path))["aggregate"]["mean"]["VC@20"] if v2_path.exists() else 0
            delta = (v2_vc / v0_vc - 1) * 100 if v0_vc > 0 else 0
            print(f"{ptype}/{ctype:<20} {v0_vc:>8.4f} {v2_vc:>8.4f} {delta:>+7.1f}%")

    print(f"\n[main] All done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
