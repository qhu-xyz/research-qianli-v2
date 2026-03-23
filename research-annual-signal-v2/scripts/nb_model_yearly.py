"""NB-hist-12 model: rolling window with per-PY breakdown + full-universe combos.

Rolling CV:
  Eval 2023-06: train 2021-2022
  Eval 2024-06: train 2021-2023
  Eval 2025-06: train 2021-2024

Part 1: NB-hist-12 only metrics, per PY and aggregate, for ML_nb / V4.4 / blend_05
Part 2: Full universe with reserved NB slots (v0c + NB scorer)

Usage:
    source /home/xyz/workspace/pmodel/.venv/bin/activate
    RAY_ADDRESS=ray://10.8.0.36:10001 PYTHONPATH=. uv run python scripts/nb_model_yearly.py
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("RAY_ADDRESS", "ray://10.8.0.36:10001")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import spearmanr

from ml.features import build_model_table
from ml.phase6.scoring import _minmax

V44_BASE = "/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V4.4.R1"

NB_FEATURE_COLS = [
    "bin_80_max", "bin_90_max", "bin_100_max", "bin_110_max",
    "rt_max", "count_active_cids",
    "shadow_price_da", "da_rank_value",
    "dev_max", "dev_sum", "pct100", "pct90",
    "shadow_rank_v44", "dev_max_rank",
]


def assign_tiers(sp):
    labels = np.zeros(len(sp), dtype=np.int32)
    pos = sp > 0
    if pos.sum() > 0:
        r = sp[pos].argsort().argsort()
        n = len(r)
        labels[pos] = np.where(r < n // 3, 1, np.where(r < 2 * n // 3, 2, 3))
    return labels


def build_all_data(pys, aqs):
    """Build full branch data for given PYs."""
    all_rows = []
    for py in pys:
        for aq in aqs:
            try:
                table = build_model_table(py, aq)
            except Exception as e:
                print(f"  {py}/{aq}: SKIP ({e})")
                continue

            v44 = {}
            path = f"{V44_BASE}/{py}/{aq}/onpeak/"
            if os.path.exists(path):
                for r in pl.read_parquet(path).iter_rows(named=True):
                    e = r.get("equipment", "")
                    if e:
                        v44[e] = r

            for r in table.iter_rows(named=True):
                b = r["branch_name"]
                v = v44.get(b, {})
                rt = max(r["bin_80_cid_max"], r["bin_90_cid_max"],
                         r["bin_100_cid_max"], r["bin_110_cid_max"])
                bf = r["bf_combined_12"]
                all_rows.append({
                    "branch": b, "py": py, "aq": aq,
                    "sp": r["realized_shadow_price"],
                    "bf_combined_12": bf,
                    "is_nb_hist12": bf == 0,
                    "da_rank_value": r["da_rank_value"],
                    "rt_max": rt,
                    "shadow_price_da": r.get("shadow_price_da", 0),
                    "total_da_sp": float(r.get("total_da_sp_quarter", 0)),
                    "bin_80_max": r["bin_80_cid_max"],
                    "bin_90_max": r["bin_90_cid_max"],
                    "bin_100_max": r["bin_100_cid_max"],
                    "bin_110_max": r["bin_110_cid_max"],
                    "count_active_cids": r["count_active_cids"],
                    "dev_max": v.get("deviation_max", 0),
                    "dev_sum": v.get("deviation_sum", 0),
                    "pct100": v.get("100_max", 0),
                    "pct90": v.get("90_max", 0),
                    "shadow_rank_v44": v.get("shadow_rank", 1.0),
                    "dev_max_rank": v.get("deviation_max_rank", 1.0),
                    "has_v44": b in v44,
                    "v44_rank": v.get("rank", 1.0) if b in v44 else 1.0,
                    "v44_tier": v.get("tier", -1) if b in v44 else -1,
                })
    return pl.DataFrame(all_rows)


def main():
    from pbase.config.ray import init_ray
    init_ray()

    t0 = time.time()
    all_pys = ["2021-06", "2022-06", "2023-06", "2024-06", "2025-06"]
    aqs = ["aq1", "aq2", "aq3"]

    print("Building all data...")
    df = build_all_data(all_pys, aqs)
    print(f"Total: {len(df)} rows, {df.filter(pl.col('sp')>0).height} binding, "
          f"{df.filter(pl.col('is_nb_hist12')).height} NB-hist-12")
    print(f"Build time: {time.time()-t0:.0f}s\n")

    # Rolling CV — only eval 2023-2025 (skip 2022: only 1 train year)
    eval_configs = [
        ("2023-06", ["2021-06", "2022-06"]),
        ("2024-06", ["2021-06", "2022-06", "2023-06"]),
        ("2025-06", ["2021-06", "2022-06", "2023-06", "2024-06"]),
    ]

    nb_results = []
    combo_results = []

    for eval_py, train_pys in eval_configs:
        train = df.filter(pl.col("py").is_in(train_pys) & pl.col("has_v44") & pl.col("is_nb_hist12"))
        eval_full = df.filter(pl.col("py") == eval_py)

        if len(train) < 50:
            print(f"  {eval_py}: only {len(train)} train rows, skipping")
            continue

        # Train NB model
        X_train = train.select(NB_FEATURE_COLS).to_numpy().astype(np.float64)
        y_train = assign_tiers(train["sp"].to_numpy().astype(np.float64))
        groups_train = train.group_by(["py", "aq"], maintain_order=True).len()["len"].to_numpy()

        ds = lgb.Dataset(X_train, label=y_train, group=groups_train, free_raw_data=False)
        nb_model = lgb.train(
            {"objective": "lambdarank", "metric": "ndcg",
             "num_leaves": 15, "learning_rate": 0.05,
             "min_child_samples": 5, "subsample": 0.8,
             "colsample_bytree": 0.8, "num_threads": 4, "verbose": -1},
            ds, num_boost_round=150,
        )

        for aq in aqs:
            eval_q = eval_full.filter(pl.col("aq") == aq)
            if len(eval_q) == 0:
                continue

            sp_all = eval_q["sp"].to_numpy().astype(np.float64)
            bf_all = eval_q["bf_combined_12"].to_numpy().astype(np.float64)
            da_all = eval_q["da_rank_value"].to_numpy().astype(np.float64)
            rt_all = eval_q["rt_max"].to_numpy().astype(np.float64)
            br_all = eval_q["branch"].to_list()
            total_sp = sp_all.sum()
            total_da = float(eval_q["total_da_sp"][0])
            n_bind = (sp_all > 0).sum()
            nb_mask = bf_all == 0
            nb_bind_mask = nb_mask & (sp_all > 0)

            if n_bind == 0:
                continue

            # v0c scores
            v0c = 0.40 * (1.0 - _minmax(da_all)) + 0.30 * _minmax(rt_all) + 0.30 * _minmax(bf_all)

            # NB model scores
            nb_eval = eval_q.filter(pl.col("is_nb_hist12") & pl.col("has_v44"))
            nb_scores = np.full(len(eval_q), -999.0)
            if len(nb_eval) > 0:
                X_nb = nb_eval.select(NB_FEATURE_COLS).to_numpy().astype(np.float64)
                nb_pred = nb_model.predict(X_nb)
                nb_branches = set(nb_eval["branch"].to_list())
                j = 0
                for i, b in enumerate(br_all):
                    if b in nb_branches and nb_mask[i]:
                        nb_scores[i] = nb_pred[j]
                        j += 1

            # V4.4 scores
            v44_rank = eval_q["v44_rank"].to_numpy()
            v44_scores = 1.0 - v44_rank

            # blend_05 = 0.5 * ML_nb + 0.5 * V4.4 (normalized)
            blend_scores = np.full(len(eval_q), -999.0)
            valid = (nb_scores > -999) & (v44_scores > -0.5)
            if valid.sum() > 0:
                ml_n = _minmax(nb_scores[valid])
                v44_n = _minmax(v44_scores[valid])
                blend_scores[valid] = 0.5 * ml_n + 0.5 * v44_n

            # === Part 1: NB-only metrics ===
            nb_sp = sp_all[nb_mask]
            nb_total = nb_sp.sum()
            nb_n_bind = (nb_sp > 0).sum()

            if nb_n_bind > 0 and nb_total > 0:
                for sname, scores_full in [("ML_nb", nb_scores), ("V4.4", v44_scores), ("blend_05", blend_scores), ("v0c", v0c)]:
                    sc = scores_full[nb_mask]
                    for K in [30, 50, 100]:
                        topk = np.argsort(sc)[::-1][:min(K, len(sc))]
                        mk = np.zeros(len(sc), dtype=bool)
                        mk[topk] = True
                        nb_results.append({
                            "eval_py": eval_py, "aq": aq, "scorer": sname, "K": K,
                            "vc": nb_sp[mk].sum() / nb_total,
                            "rec": (nb_sp[mk] > 0).sum() / nb_n_bind,
                            "sp_captured": nb_sp[mk].sum(),
                            "bind_captured": int((nb_sp[mk] > 0).sum()),
                            "n_nb": len(sc), "n_nb_bind": nb_n_bind,
                        })

            # === Part 2: Full universe combos ===
            # (config_name, nb_scorer_name, nb_scorer, n_v0c@200, n_nb@200, n_v0c@400, n_nb@400)
            configs = [
                ("pure_v0c",       "none",     None,         200,   0, 400,   0),
                ("R30_v44",        "v44",      v44_scores,   170,  30, 350,  50),
                ("R30_blend",      "blend",    blend_scores, 170,  30, 350,  50),
                ("R50_v44",        "v44",      v44_scores,   150,  50, 300, 100),
                ("R50_blend",      "blend",    blend_scores, 150,  50, 300, 100),
                ("R100_v44",       "v44",      v44_scores,   100, 100, 200, 200),
                ("R100_blend",     "blend",    blend_scores, 100, 100, 200, 200),
            ]

            for cname, nb_sname, nb_scorer, nv200, nn200, nv400, nn400 in configs:
                for K, nv, nn in [(200, nv200, nn200), (400, nv400, nn400)]:
                    selected = set()
                    # v0c picks
                    for idx in np.argsort(v0c)[::-1]:
                        if len(selected) >= nv:
                            break
                        selected.add(int(idx))
                    # NB picks from dormant population
                    if nn > 0 and nb_scorer is not None:
                        candidates = [(i, nb_scorer[i]) for i in range(len(br_all))
                                      if nb_mask[i] and i not in selected and nb_scorer[i] > -999]
                        candidates.sort(key=lambda x: -x[1])
                        for idx, _ in candidates[:nn]:
                            selected.add(idx)

                    mask = np.zeros(len(sp_all), dtype=bool)
                    for idx in selected:
                        mask[idx] = True

                    combo_results.append({
                        "eval_py": eval_py, "aq": aq, "config": cname, "K": K,
                        "vc": sp_all[mask].sum() / total_sp if total_sp > 0 else 0,
                        "abs_sp": sp_all[mask].sum() / total_da if total_da > 0 else 0,
                        "rec": (sp_all[mask] > 0).sum() / n_bind,
                        "bind": int((sp_all[mask] > 0).sum()),
                        "nb_in": int((mask & nb_mask).sum()),
                        "nb_bind": int((mask & nb_bind_mask).sum()),
                        "nb_sp": float(sp_all[mask & nb_bind_mask].sum()),
                    })

    # ==========================================================================
    # REPORTING
    # ==========================================================================
    nb_df = pl.DataFrame(nb_results)
    combo_df = pl.DataFrame(combo_results)

    print("=" * 110)
    print("PART 1: NB-hist-12 Only — Per Eval Year")
    print("=" * 110)

    scorers = ["ML_nb", "V4.4", "blend_05", "v0c"]
    for eval_py in ["2023-06", "2024-06", "2025-06"]:
        rows_py = nb_df.filter(pl.col("eval_py") == eval_py)
        if len(rows_py) == 0:
            continue
        n_groups = len(rows_py.filter(pl.col("scorer") == scorers[0]).filter(pl.col("K") == 30))
        print(f"\n  === Eval {eval_py} ({n_groups} quarters) ===")
        for K in [30, 50, 100]:
            print(f"\n    K={K}:")
            print(f"    {'Scorer':<12} {'VC':>8} {'Rec':>8} {'SP_cap':>10} {'Bind':>6}")
            print(f"    {'-'*50}")
            for s in scorers:
                r = rows_py.filter((pl.col("scorer") == s) & (pl.col("K") == K))
                if len(r) == 0:
                    continue
                print(f"    {s:<12} {r['vc'].mean():>8.4f} {r['rec'].mean():>8.4f} "
                      f"{r['sp_captured'].mean():>10,.0f} {r['bind_captured'].mean():>6.1f}")

    # Aggregate across all years
    print(f"\n  === AGGREGATE (all eval years) ===")
    for K in [30, 50, 100]:
        print(f"\n    K={K}:")
        print(f"    {'Scorer':<12} {'Grp':>4} {'VC':>8} {'Rec':>8} {'SP_cap':>10} {'Bind':>6}")
        print(f"    {'-'*50}")
        for s in scorers:
            r = nb_df.filter((pl.col("scorer") == s) & (pl.col("K") == K))
            if len(r) == 0:
                continue
            print(f"    {s:<12} {len(r):>4} {r['vc'].mean():>8.4f} {r['rec'].mean():>8.4f} "
                  f"{r['sp_captured'].mean():>10,.0f} {r['bind_captured'].mean():>6.1f}")

    print(f"\n{'='*110}")
    print("PART 2: Full Universe — Reserved NB Slots")
    print("=" * 110)

    config_order = ["pure_v0c", "R30_v44", "R30_blend", "R50_v44", "R50_blend", "R100_v44", "R100_blend"]

    for eval_py in ["2023-06", "2024-06", "2025-06"]:
        rows_py = combo_df.filter(pl.col("eval_py") == eval_py)
        if len(rows_py) == 0:
            continue
        print(f"\n  === Eval {eval_py} ===")
        for K in [200, 400]:
            print(f"\n    K={K}:")
            print(f"    {'Config':<16} {'VC':>7} {'Abs':>7} {'Rec':>7} {'Bind':>5} {'NB_in':>6} {'NB_bind':>7} {'NB_SP':>10}")
            print(f"    {'-'*75}")
            for c in config_order:
                r = rows_py.filter((pl.col("config") == c) & (pl.col("K") == K))
                if len(r) == 0:
                    continue
                print(f"    {c:<16} {r['vc'].mean():>7.4f} {r['abs_sp'].mean():>7.4f} "
                      f"{r['rec'].mean():>7.4f} {r['bind'].mean():>5.0f} "
                      f"{r['nb_in'].mean():>6.0f} {r['nb_bind'].mean():>7.1f} {r['nb_sp'].mean():>10,.0f}")

    # Aggregate
    print(f"\n  === AGGREGATE (all eval years) ===")
    for K in [200, 400]:
        print(f"\n    K={K}:")
        print(f"    {'Config':<16} {'Grp':>4} {'VC':>7} {'Abs':>7} {'Rec':>7} {'Bind':>5} {'NB_in':>6} {'NB_bind':>7} {'NB_SP':>10}")
        print(f"    {'-'*80}")
        for c in config_order:
            r = combo_df.filter((pl.col("config") == c) & (pl.col("K") == K))
            if len(r) == 0:
                continue
            print(f"    {c:<16} {len(r):>4} {r['vc'].mean():>7.4f} {r['abs_sp'].mean():>7.4f} "
                  f"{r['rec'].mean():>7.4f} {r['bind'].mean():>5.0f} "
                  f"{r['nb_in'].mean():>6.0f} {r['nb_bind'].mean():>7.1f} {r['nb_sp'].mean():>10,.0f}")

    # Delta table: each config vs pure_v0c
    print(f"\n{'='*110}")
    print("PART 3: Delta vs pure_v0c (AGGREGATE)")
    print("=" * 110)

    for K in [200, 400]:
        base = combo_df.filter((pl.col("config") == "pure_v0c") & (pl.col("K") == K))
        if len(base) == 0:
            continue
        base_vc = base["vc"].mean()
        base_nb_sp = base["nb_sp"].mean()
        base_nb_bind = base["nb_bind"].mean()
        print(f"\n    K={K} (base VC={base_vc:.4f}, base NB_SP=${base_nb_sp:,.0f}, base NB_bind={base_nb_bind:.1f}):")
        print(f"    {'Config':<16} {'dVC':>8} {'dNB_SP':>10} {'dNB_bind':>9}")
        print(f"    {'-'*45}")
        for c in config_order[1:]:
            r = combo_df.filter((pl.col("config") == c) & (pl.col("K") == K))
            if len(r) == 0:
                continue
            print(f"    {c:<16} {r['vc'].mean() - base_vc:>+8.4f} "
                  f"${r['nb_sp'].mean() - base_nb_sp:>+9,.0f} "
                  f"{r['nb_bind'].mean() - base_nb_bind:>+9.1f}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
