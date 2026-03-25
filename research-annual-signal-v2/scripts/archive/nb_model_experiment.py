"""Blended tier experiment: v0c + V4.4 reserved slots.

Tier 0 (200): 170 v0c + 30 V4.4 NB-hist-12
Tier 0+1 (400): 300 v0c + 100 V4.4 NB-hist-12

Compare against pure v0c (200/400) and pure V4.4 (200/400).

Usage:
    source /home/xyz/workspace/pmodel/.venv/bin/activate
    RAY_ADDRESS=ray://10.8.0.36:10001 PYTHONPATH=. uv run python scripts/nb_model_experiment.py
"""
from __future__ import annotations

import os
import re
import sys
import time

os.environ.setdefault("RAY_ADDRESS", "ray://10.8.0.36:10001")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import polars as pl
from ml.features import build_model_table
from ml.phase6.scoring import _minmax

V44_BASE = "/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V4.4.R1"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()) if s else ""


def load_v44_branches(py: str, aq: str) -> dict[str, dict]:
    """Load V4.4 signal features keyed by equipment (branch name)."""
    data = {}
    for ct in ["onpeak"]:
        path = f"{V44_BASE}/{py}/{aq}/{ct}/"
        if not os.path.exists(path):
            continue
        v44 = pl.read_parquet(path)
        for r in v44.iter_rows(named=True):
            equip = r.get("equipment", "")
            if equip:
                data[equip] = r
    return data


def evaluate(sp: np.ndarray, scores: np.ndarray, total_da_sp: float, n_bind_total: int, k: int) -> dict:
    total_sp = sp.sum()
    if total_sp == 0 or n_bind_total == 0:
        return {}
    topk = np.argsort(scores)[::-1][:min(k, len(sp))]
    mask = np.zeros(len(sp), dtype=bool)
    mask[topk] = True
    return {
        f"VC@{k}": float(sp[mask].sum() / total_sp),
        f"Abs@{k}": float(sp[mask].sum() / total_da_sp) if total_da_sp > 0 else 0,
        f"Rec@{k}": float((sp[mask] > 0).sum() / n_bind_total),
        f"Bind@{k}": int((sp[mask] > 0).sum()),
        f"NB_in@{k}": 0,  # filled below
        f"NB_bind@{k}": 0,
        f"NB_SP@{k}": 0.0,
    }


def main():
    from pbase.config.ray import init_ray
    init_ray()

    t0 = time.time()
    py = "2025-06"

    print("Building model tables and V4.4 features...")

    for aq in ["aq1", "aq2", "aq3"]:
        table = build_model_table(py, aq, market_round=1)
        v44_data = load_v44_branches(py, aq)

        sp = table["realized_shadow_price"].to_numpy().astype(np.float64)
        branches = table["branch_name"].to_list()
        total_sp = sp.sum()
        total_da = float(table["total_da_sp_quarter"][0])
        n_bind = (sp > 0).sum()
        bf = table["bf_combined_12"].to_numpy().astype(np.float64)

        # v0c scores
        da = table["da_rank_value"].to_numpy().astype(np.float64)
        rt = table.select(pl.max_horizontal(
            "bin_80_cid_max", "bin_90_cid_max", "bin_100_cid_max", "bin_110_cid_max"
        )).to_series().to_numpy().astype(np.float64)
        v0c = 0.40 * (1.0 - _minmax(da)) + 0.30 * _minmax(rt) + 0.30 * _minmax(bf)

        # V4.4 rank scores (lower rank = better → invert to score)
        v44_scores = np.zeros(len(branches))
        nb_hist12 = bf == 0  # NB-hist-12 population
        has_v44 = np.zeros(len(branches), dtype=bool)
        for i, b in enumerate(branches):
            if b in v44_data:
                has_v44[i] = True
                v44_scores[i] = 1.0 - v44_data[b].get("rank", 0.5)

        # NB-hist-12 flags
        nb_binding = nb_hist12 & (sp > 0)
        total_nb_sp = sp[nb_binding].sum()

        n_branches = len(branches)

        print(f"\n=== {aq} ({n_branches} branches, {n_bind} binding, {nb_hist12.sum()} NB-hist-12, {nb_binding.sum()} NB-bind) ===")

        # Strategy configs: (name, tier0_v0c, tier0_v44, tier01_v0c, tier01_v44)
        configs = [
            ("pure_v0c", 200, 0, 400, 0),
            ("pure_v44", 0, 200, 0, 400),
            ("blend_170_30", 170, 30, 300, 100),
            ("blend_150_50", 150, 50, 250, 150),
        ]

        print(f"{'Config':<16} {'K':>4} {'VC':>7} {'Abs':>7} {'Rec':>7} {'Bind':>5} {'NB_in':>6} {'NB_bind':>7} {'NB_SP':>10}")
        print("-" * 80)

        for config_name, t0_v0c, t0_v44, t01_v0c, t01_v44 in configs:
            for K, n_v0c, n_v44 in [(200, t0_v0c, t0_v44), (400, t01_v0c, t01_v44)]:
                # Select top n_v0c by v0c
                v0c_order = np.argsort(v0c)[::-1]
                selected = set()

                # v0c picks (skip NB-hist-12 if we're reserving slots for V4.4)
                v0c_count = 0
                for idx in v0c_order:
                    if v0c_count >= n_v0c:
                        break
                    selected.add(idx)
                    v0c_count += 1

                # V4.4 picks from NB-hist-12 population only (not already selected)
                if n_v44 > 0:
                    # Score NB-hist-12 branches by V4.4 rank
                    nb_candidates = []
                    for i in range(n_branches):
                        if nb_hist12[i] and has_v44[i] and i not in selected:
                            nb_candidates.append((i, v44_scores[i]))
                    nb_candidates.sort(key=lambda x: -x[1])

                    v44_count = 0
                    for idx, _ in nb_candidates:
                        if v44_count >= n_v44:
                            break
                        selected.add(idx)
                        v44_count += 1

                # Evaluate
                mask = np.zeros(n_branches, dtype=bool)
                for idx in selected:
                    mask[idx] = True

                captured = sp[mask].sum()
                vc = captured / total_sp if total_sp > 0 else 0
                abs_sp = captured / total_da if total_da > 0 else 0
                rec = (sp[mask] > 0).sum() / n_bind if n_bind > 0 else 0
                bind_count = (sp[mask] > 0).sum()
                nb_in = (mask & nb_hist12).sum()
                nb_bind_count = (mask & nb_binding).sum()
                nb_sp_captured = sp[mask & nb_binding].sum()

                print(f"{config_name:<16} {K:>4} {vc:>7.4f} {abs_sp:>7.4f} {rec:>7.4f} {bind_count:>5} {nb_in:>6} {nb_bind_count:>7} {nb_sp_captured:>10,.0f}")

            print()

    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
