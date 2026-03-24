"""Round-aware V7.1B comparison: v0c R1/R2/R3 vs V4.4 R1/R2/R3.

Builds model tables from the round-aware pipeline (NOT data/nb_cache/).
Scores v0c on round-specific features. Loads matched V4.4.R{N}.
Reports at base grain (py, aq, ctype, round).

Requires Ray for build_class_model_table (pbase loaders).

Usage:
    source /home/xyz/workspace/pmodel/.venv/bin/activate
    PYTHONPATH=. uv run python scripts/round_comparison.py
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("RAY_ADDRESS", "ray://10.8.0.36:10001")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import polars as pl

V44_BASE = "/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V4.4"

# Full eval scope: all PYs with V4.4 coverage
EVAL_PYS = ["2019-06", "2020-06", "2021-06", "2022-06", "2023-06", "2024-06", "2025-06"]
AQS = ["aq1", "aq2", "aq3"]
CTYPES = ["onpeak", "offpeak"]
ROUNDS = [1, 2, 3]
K_LEVELS = [200, 400]


def _minmax(arr):
    mn, mx = arr.min(), arr.max()
    return np.full_like(arr, 0.5) if mx == mn else (arr - mn) / (mx - mn)


def score_v0c(table):
    da = table["da_rank_value"].to_numpy().astype(np.float64)
    bf_col = "bf_12" if table["class_type"][0] == "onpeak" else "bfo_12"
    bf = table[bf_col].to_numpy().astype(np.float64)
    rt = np.max(np.column_stack([
        table["bin_80_cid_max"].to_numpy(),
        table["bin_90_cid_max"].to_numpy(),
        table["bin_100_cid_max"].to_numpy(),
        table["bin_110_cid_max"].to_numpy(),
    ]), axis=1).astype(np.float64)
    return 0.40 * (1.0 - _minmax(da)) + 0.30 * _minmax(rt) + 0.30 * _minmax(bf)


def load_v44(eval_py, aq, ct, market_round):
    """Load V4.4 for matched round. Returns (ranked_branches, universe_size)."""
    path = f"{V44_BASE}.R{market_round}/{eval_py}/{aq}/{ct}/"
    if not os.path.exists(path):
        return [], 0
    df = pl.read_parquet(path).filter(pl.col("equipment") != "").sort("rank")
    return df["equipment"].to_list(), len(df)


def compute_metrics(sp, idx_arr, is_dormant, is_nb, K, universe_size):
    mask = np.zeros(len(sp), dtype=bool)
    bounded = idx_arr[idx_arr < len(sp)]
    mask[bounded] = True
    actual_k = int(mask.sum())
    total_sp = sp.sum()
    n_bind = int((sp > 0).sum())

    return {
        "K": K, "actual_k": actual_k, "universe": universe_size,
        "sp": float(sp[mask].sum()),
        "binders": int((sp[mask] > 0).sum()),
        "precision": float((sp[mask] > 0).sum() / actual_k) if actual_k > 0 else 0.0,
        "vc": float(sp[mask].sum() / total_sp) if total_sp > 0 else 0.0,
        "recall": float((sp[mask] > 0).sum() / n_bind) if n_bind > 0 else 0.0,
        "nb_in": int((mask & is_dormant).sum()),
        "nb_binders": int((mask & is_nb).sum()),
        "nb_sp": float(sp[mask & is_nb].sum()),
        "d20_hit": int((sp[mask] > 20000).sum()),
        "d20_total": int((sp > 20000).sum()),
    }


def compute_v44_metrics(v44_topk, branches, sp, is_dormant, is_nb, K):
    branch_to_idx = {}
    for i, b in enumerate(branches):
        if b not in branch_to_idx:
            branch_to_idx[b] = i

    mask = np.zeros(len(sp), dtype=bool)
    labeled = 0
    for b in v44_topk:
        if b in branch_to_idx:
            mask[branch_to_idx[b]] = True
            labeled += 1

    total_sp = sp.sum()
    n_bind = int((sp > 0).sum())
    return {
        "K": K, "actual_k": int(mask.sum()), "universe": -1,
        "sp": float(sp[mask].sum()),
        "binders": int((sp[mask] > 0).sum()),
        "precision": float((sp[mask] > 0).sum() / K) if K > 0 else 0.0,
        "vc": float(sp[mask].sum() / total_sp) if total_sp > 0 else 0.0,
        "recall": float((sp[mask] > 0).sum() / n_bind) if n_bind > 0 else 0.0,
        "nb_in": int((mask & is_dormant).sum()),
        "nb_binders": int((mask & is_nb).sum()),
        "nb_sp": float(sp[mask & is_nb].sum()),
        "d20_hit": int((sp[mask] > 20000).sum()),
        "d20_total": int((sp > 20000).sum()),
        "labeled": labeled, "unlabeled": len(v44_topk) - labeled,
    }


def main():
    from pbase.config.ray import init_ray
    import pmodel
    init_ray(extra_modules=[pmodel])

    from ml.phase6.features import build_class_model_table

    t0 = time.time()
    all_results = []

    for eval_py in EVAL_PYS:
        for ct in CTYPES:
            for market_round in ROUNDS:
                print(f"\n{'='*100}")
                print(f"  {eval_py} / {ct} / R{market_round}")
                print(f"{'='*100}")

                # Build round-aware model table (NOT from nb_cache)
                t1 = time.time()
                try:
                    table = build_class_model_table(eval_py, "aq1", ct, market_round=market_round)
                except Exception as e:
                    print(f"  BUILD FAILED: {e}")
                    continue
                build_time = time.time() - t1

                N = len(table)
                bf_col = "bf_12" if ct == "onpeak" else "bfo_12"

                # Score v0c
                scores = score_v0c(table)

                print(f"  Built {N} branches in {build_time:.0f}s")

                for aq in AQS:
                    # Rebuild per quarter (universe may differ)
                    try:
                        aq_table = build_class_model_table(eval_py, aq, ct, market_round=market_round)
                    except Exception as e:
                        print(f"  {aq} BUILD FAILED: {e}")
                        continue

                    sp = aq_table["realized_shadow_price"].to_numpy().astype(np.float64)
                    bf = aq_table[bf_col].to_numpy().astype(np.float64)
                    branches = aq_table["branch_name"].to_list()
                    is_dormant = bf == 0
                    is_nb = is_dormant & (sp > 0)
                    aq_scores = score_v0c(aq_table)
                    aq_N = len(aq_table)

                    # V4.4 for matched round
                    v44_branches, v44_univ = load_v44(eval_py, aq, ct, market_round)

                    for K in K_LEVELS:
                        # v0c native
                        v0c_idx = np.argsort(aq_scores)[::-1][:K]
                        m_v0c = compute_metrics(sp, v0c_idx, is_dormant, is_nb, K, aq_N)
                        m_v0c.update(eval_py=eval_py, aq=aq, ct=ct, market_round=market_round, model="v0c")
                        all_results.append(m_v0c)

                        # V4.4 native (matched round)
                        v44_topk = v44_branches[:K]
                        m_v44 = compute_v44_metrics(v44_topk, branches, sp, is_dormant, is_nb, K)
                        m_v44.update(eval_py=eval_py, aq=aq, ct=ct, market_round=market_round, model="V4.4")
                        all_results.append(m_v44)

    # ── Summary tables ──
    print(f"\n{'='*120}")
    print("  NATIVE TOP-K: v0c vs V4.4, per (PY, ctype, round, K)")
    print(f"{'='*120}")
    print(f"  {'PY':<8} {'CT':<8} {'R':>2} {'K':>4}  {'v0c SP':>12} {'V44 SP':>12} {'Delta':>12}  {'v0c NB$':>10} {'V44 NB$':>10} {'V44 unlbl':>9}")
    print(f"  {'-'*100}")

    for eval_py in EVAL_PYS:
        for ct in CTYPES:
            for K in K_LEVELS:
                for market_round in ROUNDS:
                    v0c_rows = [r for r in all_results
                                if r.get("eval_py") == eval_py and r.get("ct") == ct
                                and r.get("model") == "v0c" and r["K"] == K
                                and r.get("market_round") == market_round]
                    v44_rows = [r for r in all_results
                                if r.get("eval_py") == eval_py and r.get("ct") == ct
                                and r.get("model") == "V4.4" and r["K"] == K
                                and r.get("market_round") == market_round]

                    if not v0c_rows or not v44_rows:
                        continue

                    v0c_sp = sum(r["sp"] for r in v0c_rows) / len(v0c_rows)
                    v44_sp = sum(r["sp"] for r in v44_rows) / len(v44_rows)
                    v0c_nb = sum(r["nb_sp"] for r in v0c_rows) / len(v0c_rows)
                    v44_nb = sum(r["nb_sp"] for r in v44_rows) / len(v44_rows)
                    v44_unlbl = sum(r.get("unlabeled", 0) for r in v44_rows) / len(v44_rows)

                    print(
                        f"  {eval_py:<8} {ct:<8} R{market_round} {K:>4}  "
                        f"${v0c_sp:>11,.0f} ${v44_sp:>11,.0f} ${v0c_sp - v44_sp:>+11,.0f}  "
                        f"${v0c_nb:>9,.0f} ${v44_nb:>9,.0f} {v44_unlbl:>9.0f}"
                    )

    # ── Round delta table ──
    print(f"\n{'='*120}")
    print("  ROUND DELTA: v0c R2-R1 and R3-R1 SP difference")
    print(f"{'='*120}")
    print(f"  {'PY':<8} {'CT':<8} {'K':>4}  {'R1 SP':>12} {'R2 SP':>12} {'R2-R1':>12} {'R3 SP':>12} {'R3-R1':>12}")
    print(f"  {'-'*90}")

    for eval_py in EVAL_PYS:
        for ct in CTYPES:
            for K in K_LEVELS:
                sps = {}
                for market_round in ROUNDS:
                    rows = [r for r in all_results
                            if r.get("eval_py") == eval_py and r.get("ct") == ct
                            and r.get("model") == "v0c" and r["K"] == K
                            and r.get("market_round") == market_round]
                    if rows:
                        sps[market_round] = sum(r["sp"] for r in rows) / len(rows)

                if len(sps) == 3:
                    print(
                        f"  {eval_py:<8} {ct:<8} {K:>4}  "
                        f"${sps[1]:>11,.0f} ${sps[2]:>11,.0f} ${sps[2] - sps[1]:>+11,.0f} "
                        f"${sps[3]:>11,.0f} ${sps[3] - sps[1]:>+11,.0f}"
                    )

    # Save
    reg_path = "registry/miso/annual/comparisons/round_comparison_v1"
    os.makedirs(reg_path, exist_ok=True)
    with open(f"{reg_path}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Normalized spec.json
    import subprocess
    code_commit = subprocess.check_output(["git", "log", "--oneline", "-1", "--format=%h"]).decode().strip()
    spec = {
        "spec_type": "comparison",
        "comparison_id": "round_comparison_v1",
        "market": "miso",
        "product": "annual",
        "models": [
            {
                "model_id": "miso_annual_v0c_formula_v1",
                "universe_id": "miso_annual_branch_active_v1",
                "feature_recipe_id": "miso_annual_v0c_features_v1",
                "rank_direction": "descending",
                "round_sensitivity": "round_aware",
                "pipeline": "build_class_model_table (round-aware)",
            },
            {
                "benchmark_id": "miso_annual_v44_published_v1",
                "universe_id": "miso_annual_v44_published_v1",
                "signal_path_pattern": "TEST.Signal.MISO.SPICE_ANNUAL_V4.4.R{round}",
                "rank_direction": "ascending",
                "round_sensitivity": "round_aware",
            },
        ],
        "eval_pys": EVAL_PYS,
        "eval_quarters": AQS,
        "eval_ctypes": CTYPES,
        "eval_rounds": ROUNDS,
        "k_levels": K_LEVELS,
        "base_grain": "planning_year/aq_quarter/class_type/market_round",
        "code_commit": code_commit,
    }
    with open(f"{reg_path}/spec.json", "w") as f:
        json.dump(spec, f, indent=2)

    # Normalized metrics.json
    cells = []
    for r in all_results:
        cells.append({
            "planning_year": r["eval_py"], "aq_quarter": r["aq"],
            "class_type": r["ct"], "market_round": r["market_round"],
            "model": r["model"], "K": r["K"],
            "sp": r["sp"], "binders": r["binders"],
            "precision": r["precision"], "vc": r["vc"], "recall": r["recall"],
            "nb_in": r["nb_in"], "nb_binders": r["nb_binders"], "nb_sp": r["nb_sp"],
            "d20_hit": r.get("d20_hit"), "d20_total": r.get("d20_total"),
            "labeled": r.get("labeled"), "unlabeled": r.get("unlabeled"),
        })
    with open(f"{reg_path}/metrics.json", "w") as f:
        json.dump({"base_grain": "planning_year/aq_quarter/class_type/market_round", "cells": cells}, f, indent=2)

    import ray
    ray.shutdown()

    print(f"\nSaved to {reg_path}/")
    print(f"Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
