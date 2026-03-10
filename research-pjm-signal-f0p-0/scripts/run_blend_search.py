#!/usr/bin/env python
"""Blend search for PJM V7.0b: find optimal (w_da, w_dmix, w_dori) per slice.

Explores the full simplex in 0.05 increments. Updates BLEND_WEIGHTS in run_v2_ml.py
and saves winning blend to registry/{ptype}/{ctype}/v2/config.json.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
    python scripts/run_blend_search.py --ptype f0 --class-type onpeak
    python scripts/run_blend_search.py  # all 6 slices
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    REALIZED_DA_CACHE, LTRConfig, PipelineConfig, V10E_FEATURES, V10E_MONOTONE,
    _DEFAULT_EVAL_MONTHS, PJM_CLASS_TYPES,
    has_period_type, collect_usable_months,
)
from ml.data_loader import load_v62b_month
from ml.evaluate import evaluate_ltr
from ml.features import compute_query_groups, prepare_features
from ml.registry_paths import registry_root
from ml.train import predict_scores, train_ltr_model

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "registry"


def generate_simplex_grid(step: float = 0.05) -> list[tuple[float, float, float]]:
    """Generate all (w_da, w_dmix, w_dori) on simplex with given step."""
    grid = []
    n_steps = int(round(1.0 / step))
    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            k = n_steps - i - j
            w_da = round(i * step, 2)
            w_dmix = round(j * step, 2)
            w_dori = round(k * step, 2)
            grid.append((w_da, w_dmix, w_dori))
    return grid


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


def eval_blend(
    blend: tuple[float, float, float],
    eval_months: list[str],
    bs: dict[str, set[str]],
    class_type: str,
    period_type: str,
) -> float:
    """Evaluate a single blend, return mean VC@20."""
    w_da, w_dmix, w_dori = blend

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

    vc20_values = []
    for m in eval_months:
        train_month_strs = collect_usable_months(m, period_type, n_months=8)
        if not train_month_strs:
            continue
        train_month_strs = list(reversed(train_month_strs))

        parts = []
        for tm in train_month_strs:
            try:
                part = load_v62b_month(tm, period_type, class_type)
                part = part.with_columns(pl.lit(tm).alias("query_month"))
                branch_names = part["branch_name"].to_list()
                part = part.with_columns(
                    (w_da * pl.col("da_rank_value")
                     + w_dmix * pl.col("density_mix_rank_value")
                     + w_dori * pl.col("density_ori_rank_value")
                    ).alias("v7_formula_score")
                )
                cutoff = prev_month(tm)
                for lb in [1, 3, 6, 12, 15]:
                    col_name = f"binding_freq_{lb}"
                    if col_name not in part.columns:
                        freq = compute_bf(branch_names, cutoff, bs, lb)
                        part = part.with_columns(pl.Series(col_name, freq))
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
        branch_names = test_df["branch_name"].to_list()
        test_df = test_df.with_columns(
            (w_da * pl.col("da_rank_value")
             + w_dmix * pl.col("density_mix_rank_value")
             + w_dori * pl.col("density_ori_rank_value")
            ).alias("v7_formula_score")
        )
        cutoff = prev_month(m)
        for lb in [1, 3, 6, 12, 15]:
            col_name = f"binding_freq_{lb}"
            if col_name not in test_df.columns:
                freq = compute_bf(branch_names, cutoff, bs, lb)
                test_df = test_df.with_columns(pl.Series(col_name, freq))

        train_df = train_df.sort("query_month")
        X_train, _ = prepare_features(train_df, cfg.ltr)
        y_train = train_df["realized_sp"].to_numpy().astype(np.float64)
        groups = compute_query_groups(train_df)

        model = train_ltr_model(X_train, y_train, groups, cfg.ltr)
        X_test, _ = prepare_features(test_df, cfg.ltr)
        scores = predict_scores(model, X_test)
        actual = test_df["realized_sp"].to_numpy().astype(np.float64)

        metrics = evaluate_ltr(actual, scores)
        vc20_values.append(metrics["VC@20"])

        del train_df, test_df, parts, X_train, y_train, groups, model, X_test, actual, scores
        gc.collect()

    return np.mean(vc20_values) if vc20_values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptype", default=None)
    parser.add_argument("--class-type", default=None)
    parser.add_argument("--step", type=float, default=0.10,
                        help="Simplex step size (default 0.10, use 0.05 for fine search)")
    args = parser.parse_args()

    ptypes = [args.ptype] if args.ptype else ["f0", "f1"]
    ctypes = [args.class_type] if args.class_type else PJM_CLASS_TYPES
    grid = generate_simplex_grid(args.step)
    print(f"[blend] Simplex grid: {len(grid)} points (step={args.step})")

    for ptype in ptypes:
        for ctype in ctypes:
            print(f"\n{'='*60}")
            print(f"Blend search: {ptype}/{ctype}")
            print(f"{'='*60}")

            eval_months = [m for m in _DEFAULT_EVAL_MONTHS if has_period_type(m, ptype)]
            if not eval_months:
                print(f"  No eval months for {ptype}")
                continue

            bs = load_all_binding_sets(peak_type=ctype)

            best_blend = (0.85, 0.00, 0.15)
            best_vc20 = 0.0

            t0 = time.time()
            for i, blend in enumerate(grid):
                vc20 = eval_blend(blend, eval_months, bs, ctype, ptype)
                if vc20 > best_vc20:
                    best_vc20 = vc20
                    best_blend = blend
                    print(f"  [{i+1}/{len(grid)}] {blend} -> VC@20={vc20:.4f} ** NEW BEST")
                elif (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{len(grid)}] {blend} -> VC@20={vc20:.4f}")

            elapsed = time.time() - t0
            print(f"\n  Best blend: {best_blend} -> VC@20={best_vc20:.4f} ({elapsed:.0f}s)")

            # Save to config
            reg_slice = registry_root(ptype, ctype, base_dir=REGISTRY)
            v2_dir = reg_slice / "v2"
            v2_dir.mkdir(parents=True, exist_ok=True)
            config_path = v2_dir / "config.json"
            if config_path.exists():
                config = json.load(open(config_path))
            else:
                config = {}
            config["blend_weights"] = {
                "da": best_blend[0], "dmix": best_blend[1], "dori": best_blend[2],
            }
            config["blend_search_vc20"] = best_vc20
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"  Saved to {config_path}")

    print("\n[blend] All done.")


if __name__ == "__main__":
    main()
