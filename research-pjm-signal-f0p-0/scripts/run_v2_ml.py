#!/usr/bin/env python
"""V2 ML model for PJM: LightGBM LambdaRank with 9 features.

Runs dev (36mo) and holdout (24mo) for all 6 ML slices.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
    python scripts/run_v2_ml.py --ptype f0 --class-type onpeak
    python scripts/run_v2_ml.py  # all 6 slices
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
    _FULL_EVAL_MONTHS, HOLDOUT_MONTHS, PJM_CLASS_TYPES,
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

# Per-(period_type, class_type) blend weights for v7_formula_score.
# Start with MISO defaults. Will be tuned by blend search script.
BLEND_WEIGHTS: dict[tuple[str, str], tuple[float, float, float]] = {
    ("f0", "onpeak"): (0.85, 0.00, 0.15),
    ("f0", "dailyoffpeak"): (0.85, 0.00, 0.15),
    ("f0", "wkndonpeak"): (0.85, 0.00, 0.15),
    ("f1", "onpeak"): (0.70, 0.00, 0.30),
    ("f1", "dailyoffpeak"): (0.80, 0.00, 0.20),
    ("f1", "wkndonpeak"): (0.80, 0.00, 0.20),
}
_DEFAULT_BLEND = (0.85, 0.00, 0.15)


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_all_binding_sets(
    peak_type: str = "onpeak",
    cache_dir: str = REALIZED_DA_CACHE,
) -> dict[str, set[str]]:
    """Load all cached DA into {month: set(branch_names)}.

    PJM-specific: keys are branch_name, not constraint_id.
    """
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
    """Compute binding frequency for branch_names."""
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
    ts = pd.Timestamp(m)
    return (ts - pd.DateOffset(months=1)).strftime("%Y-%m")


def enrich_df(
    df: pl.DataFrame, month: str, bs: dict[str, set[str]],
    blend_weights: tuple[float, float, float],
) -> pl.DataFrame:
    """Add binding_freq and formula score features. BF_LAG=1 always."""
    cutoff = prev_month(month)  # BF sees months strictly before M-1
    w_da, w_dmix, w_dori = blend_weights

    # PJM-specific: use branch_name for BF, not constraint_id
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
    blend = BLEND_WEIGHTS.get((period_type, class_type), _DEFAULT_BLEND)
    print(f"\n[{label}] 9f, {len(eval_months)} months, ptype={period_type}, "
          f"class_type={class_type}, blend={blend}")

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

        metrics = evaluate_ltr(actual, scores)
        per_month[m] = metrics

        if hasattr(model, "feature_importance"):
            imp = model.feature_importance(importance_type="gain")
            metrics["_fi"] = dict(zip(cfg.ltr.features, [float(x) for x in imp]))

        elapsed = time.time() - t0
        n_bind = int((actual > 0).sum())
        print(f"  {m}: VC@20={metrics['VC@20']:.4f} binding={n_bind} ({elapsed:.1f}s)")

        del train_df, test_df, parts, X_train, y_train, groups, model, X_test, actual, scores
        gc.collect()

    if per_month:
        clean = {m: {k: v for k, v in met.items() if not k.startswith("_")}
                 for m, met in per_month.items()}
        agg = aggregate_months(clean)
        means = agg["mean"]
        print(f"  => VC@20={means['VC@20']:.4f} VC@100={means['VC@100']:.4f} "
              f"NDCG={means['NDCG']:.4f} Spearman={means['Spearman']:.4f}")

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
        "features": list(V10E_FEATURES),
        "lag": 1, "period_type": period_type, "class_type": class_type,
        "blend_weights": dict(zip(["da", "dmix", "dori"],
                                  BLEND_WEIGHTS.get((period_type, class_type), _DEFAULT_BLEND))),
    }
    with open(d / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved to {d}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptype", default=None)
    parser.add_argument("--class-type", default=None)
    parser.add_argument("--dev-only", action="store_true")
    args = parser.parse_args()

    ptypes = [args.ptype] if args.ptype else ["f0", "f1"]
    ctypes = [args.class_type] if args.class_type else PJM_CLASS_TYPES
    t_start = time.time()

    for ptype in ptypes:
        for ctype in ctypes:
            print(f"\n{'='*70}")
            print(f"V2 ML: {ptype}/{ctype}")
            print(f"{'='*70}")

            bs = load_all_binding_sets(peak_type=ctype)

            dev_eval = [m for m in _FULL_EVAL_MONTHS if has_period_type(m, ptype)]
            reg_slice = registry_root(ptype, ctype, base_dir=REGISTRY)

            dev_pm = run_variant("v2", dev_eval, bs, class_type=ctype, period_type=ptype)
            if dev_pm:
                save_results("v2", dev_pm, dev_eval, reg_slice,
                             class_type=ctype, period_type=ptype)

            if not args.dev_only:
                ho_eval = [m for m in HOLDOUT_MONTHS if has_period_type(m, ptype)]
                ho_slice = holdout_root(ptype, ctype, base_dir=HOLDOUT_DIR)
                ho_pm = run_variant("v2-holdout", ho_eval, bs,
                                    class_type=ctype, period_type=ptype)
                if ho_pm:
                    save_results("v2", ho_pm, ho_eval, ho_slice,
                                 class_type=ctype, period_type=ptype)

            # Print comparison vs v0
            v0_path = reg_slice / "v0" / "metrics.json"
            if v0_path.exists() and dev_pm:
                v0_agg = json.load(open(v0_path))["aggregate"]["mean"]
                clean = {m: {k: v for k, v in met.items() if not k.startswith("_")}
                         for m, met in dev_pm.items()}
                v2_agg = aggregate_months(clean)["mean"]
                print(f"\n  {'Metric':<12} {'v0':>8} {'v2':>8} {'delta':>8}")
                for met in ["VC@20", "VC@50", "VC@100", "Recall@20", "NDCG"]:
                    v0_v = v0_agg.get(met, 0)
                    v2_v = v2_agg.get(met, 0)
                    delta = (v2_v / v0_v - 1) * 100 if v0_v > 0 else 0
                    print(f"  {met:<12} {v0_v:>8.4f} {v2_v:>8.4f} {delta:>+7.1f}%")

    print(f"\n[main] All done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
