"""Native standalone comparison: v0c vs tiered_top2 (R30) vs V4.4.

Each model picks its own top-K from its own universe. GT attached by branch name.
Reports 4 tables (2024/2025 × on/off) with all metrics per the metric contract.

Uses cached class-specific tables from data/nb_cache/ (build with nb_v3_ablation.py first).

Usage:
    source /home/xyz/workspace/pmodel/.venv/bin/activate
    PYTHONPATH=. uv run python scripts/nb_native_comparison.py
"""
from __future__ import annotations

import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from nb_v3_ablation import (
    _minmax, compute_v0c, assign_tiers_per_group, compute_tiered_weights,
    allocate_reserved_slots, BASELINE_FEATURES, LGB_RANK_PARAMS,
)
from ml.config import CLASS_BF_COL

V44_BASE = "/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V4.4.R1"
CACHE_DIR = "data/nb_cache"
FEAT_TOP2 = BASELINE_FEATURES + ["top2_bin_80", "top2_bin_90", "top2_bin_100", "top2_bin_110"]
ALL_PYS = ["2020-06", "2021-06", "2022-06", "2023-06", "2024-06", "2025-06"]
AQS = ["aq1", "aq2", "aq3"]
EVAL_CONFIGS = [
    ("2024-06", ["2020-06", "2021-06", "2022-06", "2023-06"]),
    ("2025-06", ["2020-06", "2021-06", "2022-06", "2023-06", "2024-06"]),
]


def load_cached_data():
    """Load all cached class-specific tables + top2 into per-ctype DataFrames."""
    frames = {"onpeak": [], "offpeak": []}
    for py in ALL_PYS:
        for aq in AQS:
            for ct in ["onpeak", "offpeak"]:
                p = f"{CACHE_DIR}/{py}_{aq}_{ct}.parquet"
                if not os.path.exists(p):
                    continue
                bf_col = CLASS_BF_COL[ct]
                t = pl.read_parquet(p).with_columns(
                    pl.lit(py).alias("py"), pl.lit(aq).alias("aq"),
                    pl.col(bf_col).alias("bf"),
                    pl.max_horizontal(
                        "bin_80_cid_max", "bin_90_cid_max",
                        "bin_100_cid_max", "bin_110_cid_max"
                    ).alias("rt_max"),
                ).rename({
                    "branch_name": "branch", "realized_shadow_price": "sp",
                    "bin_80_cid_max": "bin_80_max", "bin_90_cid_max": "bin_90_max",
                    "bin_100_cid_max": "bin_100_max", "bin_110_cid_max": "bin_110_max",
                })
                t2p = f"{CACHE_DIR}/{py}_{aq}_top2.parquet"
                if os.path.exists(t2p):
                    t2 = pl.read_parquet(t2p).rename({"branch_name": "branch"})
                    t = t.join(t2, on="branch", how="left")
                    for c in ["top2_bin_80", "top2_bin_90", "top2_bin_100", "top2_bin_110"]:
                        if c in t.columns:
                            t = t.with_columns(pl.col(c).fill_null(0.0))
                cols = (
                    ["branch", "py", "aq", "sp", "bf", "bin_80_max", "bin_90_max",
                     "bin_100_max", "bin_110_max", "rt_max", "count_active_cids",
                     "shadow_price_da", "da_rank_value", "total_da_sp_quarter"]
                    + [c for c in ["top2_bin_80", "top2_bin_90", "top2_bin_100", "top2_bin_110"]
                       if c in t.columns]
                )
                frames[ct].append(t.select([c for c in cols if c in t.columns]))

    df_class = {ct: pl.concat(fs, how="diagonal") for ct, fs in frames.items()}
    for ct in df_class:
        for c in ["top2_bin_80", "top2_bin_90", "top2_bin_100", "top2_bin_110"]:
            if c in df_class[ct].columns:
                df_class[ct] = df_class[ct].with_columns(pl.col(c).fill_null(0.0))
    return df_class


def main():
    t0 = time.time()
    df_class = load_cached_data()
    print(f"Data load: {time.time()-t0:.1f}s")

    for eval_py, train_pys in EVAL_CONFIGS:
        for ct in ["onpeak", "offpeak"]:
            # Train tiered_top2 model
            train = df_class[ct].filter(pl.col("py").is_in(train_pys) & (pl.col("bf") == 0))
            feats_tt = [f for f in FEAT_TOP2 if f in train.columns]
            X = train.select(feats_tt).to_numpy().astype(np.float64)
            groups = train.group_by(["py", "aq"], maintain_order=True).len()["len"].to_numpy()
            sp_train = train["sp"].to_numpy().astype(np.float64)
            y = assign_tiers_per_group(sp_train, groups)
            w = compute_tiered_weights(sp_train, y)
            ds = lgb.Dataset(X, label=y, group=groups, weight=w,
                             feature_name=feats_tt, free_raw_data=False)
            m_tt = lgb.train(LGB_RANK_PARAMS, ds, num_boost_round=150)

            print(f"\n{'='*130}")
            print(f"{eval_py} / {ct}")
            print("=" * 130)

            rows_by_model = {m: [] for m in ["v0c", "tiered_top2_R30", "V4.4"]}

            for aq in AQS:
                eq = df_class[ct].filter((pl.col("py") == eval_py) & (pl.col("aq") == aq))
                if len(eq) == 0:
                    continue
                N = len(eq)
                sp = eq["sp"].to_numpy().astype(np.float64)
                da = eq["da_rank_value"].to_numpy().astype(np.float64)
                bf = eq["bf"].to_numpy().astype(np.float64)
                rt = eq["rt_max"].to_numpy().astype(np.float64)
                branches = eq["branch"].to_list()
                is_dormant = bf == 0
                is_nb_binder = is_dormant & (sp > 0)
                total_da = float(eq["total_da_sp_quarter"][0]) if "total_da_sp_quarter" in eq.columns else float(sp.sum())
                total_sp = sp.sum()
                n_bind = int((sp > 0).sum())
                gt_map = {b: float(sp[i]) for i, b in enumerate(branches)}
                n_d20 = int((sp > 20000).sum())
                n_d50 = int((sp > 50000).sum())

                v0c = compute_v0c(da, rt, bf)

                # TT scores for dormant
                didx = np.where(is_dormant)[0]
                tt_scores = np.full(N, -np.inf)
                if len(didx) > 0:
                    tt_scores[didx] = m_tt.predict(
                        eq[didx.tolist()].select(feats_tt).to_numpy().astype(np.float64)
                    )

                # V4.4 data
                v44_path = f"{V44_BASE}/{eval_py}/{aq}/{ct}/"
                v44_branches_ranked = []
                if os.path.exists(v44_path):
                    v44_df = pl.read_parquet(v44_path).filter(
                        pl.col("equipment") != ""
                    ).sort("rank")
                    v44_branches_ranked = v44_df["equipment"].to_list()

                for K in [200, 400]:
                    # v0c native top-K
                    v0c_idx = np.argsort(v0c)[::-1][:K]
                    rows_by_model["v0c"].append(_compute_metrics(
                        sp, v0c_idx, is_dormant, is_nb_binder, total_sp, total_da, n_bind, n_d20, n_d50, K, K
                    ))

                    # tiered_top2 R30
                    nv = 170 if K == 200 else 350
                    nn = 30 if K == 200 else 50
                    sel_tt, nf = allocate_reserved_slots(v0c, tt_scores, is_dormant, nv, nn)
                    mask_tt = np.array(sorted(sel_tt))
                    rows_by_model["tiered_top2_R30"].append(_compute_metrics(
                        sp, mask_tt, is_dormant, is_nb_binder, total_sp, total_da, n_bind, n_d20, n_d50, K, K
                    ))

                    # V4.4 native top-K
                    v44_topk = v44_branches_ranked[:K]
                    v44_metrics = _compute_v44_metrics(
                        v44_topk, gt_map, branches, is_dormant, is_nb_binder, sp,
                        total_sp, total_da, n_bind, n_d20, n_d50, K, len(v44_branches_ranked)
                    )
                    rows_by_model["V4.4"].append(v44_metrics)

            # Print tables
            for K in [200, 400]:
                tier = "Tier 0" if K == 200 else "Tier 0+1"
                print(f"\n  K={K} ({tier}) — averaged across {len(AQS)} quarters")
                print(f"  {'Model':<20} {'Univ':>5} {'SP':>12} {'Bind':>5} {'Prec':>5} {'VC':>6} {'Abs':>6} {'Rec':>6}"
                      f" {'NB_in':>5} {'NB_b':>4} {'NB_SP':>10} {'D20':>7} {'D50':>5} {'LblCov':>7}")
                print(f"  {'-'*125}")
                for mname in ["v0c", "tiered_top2_R30", "V4.4"]:
                    mrs = [r for r in rows_by_model[mname] if r["K"] == K]
                    if not mrs:
                        continue
                    _print_row(mname, mrs, K)

            # NB-only
            print(f"\n  NB-only (dormant universe):")
            print(f"  {'Model':<20} {'K':>4} {'NB_VC':>7} {'NB_SP':>10} {'NB_Bind':>8} {'NB_Rec':>7} {'NB_Prec':>8}")
            print(f"  {'-'*70}")
            for K_nb in [50, 100]:
                for mname in ["v0c", "tiered_top2", "V4.4"]:
                    nb_vcs = []; nb_sps = []; nb_binds = []; nb_recs = []; nb_precs = []
                    for aq in AQS:
                        eq = df_class[ct].filter((pl.col("py") == eval_py) & (pl.col("aq") == aq))
                        if len(eq) == 0:
                            continue
                        sp_q = eq["sp"].to_numpy().astype(np.float64)
                        bf_q = eq["bf"].to_numpy().astype(np.float64)
                        branches_q = eq["branch"].to_list()
                        is_dorm = bf_q == 0
                        nb_sp_q = sp_q[is_dorm]
                        nb_total = nb_sp_q.sum()
                        nb_n_bind = int((nb_sp_q > 0).sum())
                        if nb_total <= 0 or nb_n_bind <= 0:
                            continue

                        if mname == "v0c":
                            da_q = eq["da_rank_value"].to_numpy().astype(np.float64)
                            rt_q = eq["rt_max"].to_numpy().astype(np.float64)
                            sc = compute_v0c(da_q, rt_q, bf_q)[is_dorm]
                        elif mname == "tiered_top2":
                            d_idx = np.where(is_dorm)[0]
                            sc = m_tt.predict(
                                eq[d_idx.tolist()].select(feats_tt).to_numpy().astype(np.float64)
                            ) if len(d_idx) > 0 else np.array([])
                        else:
                            v44p = f"{V44_BASE}/{eval_py}/{aq}/{ct}/"
                            v44d = {}
                            if os.path.exists(v44p):
                                for r in pl.read_parquet(v44p).iter_rows(named=True):
                                    e = r.get("equipment", "")
                                    if e:
                                        v44d[e] = r
                            sc = np.full(int(is_dorm.sum()), -np.inf)
                            dorm_br = [branches_q[i] for i in range(len(branches_q)) if is_dorm[i]]
                            for j, b in enumerate(dorm_br):
                                if b in v44d:
                                    sc[j] = 1.0 - v44d[b].get("rank", 1.0)

                        if len(sc) == 0:
                            continue
                        topk = np.argsort(sc)[::-1][:min(K_nb, len(sc))]
                        mk = np.zeros(len(sc), dtype=bool)
                        mk[topk] = True
                        nb_vcs.append(nb_sp_q[mk].sum() / nb_total)
                        nb_sps.append(nb_sp_q[mk].sum())
                        nb_binds.append((nb_sp_q[mk] > 0).sum())
                        nb_recs.append((nb_sp_q[mk] > 0).sum() / nb_n_bind)
                        nb_precs.append((nb_sp_q[mk] > 0).sum() / K_nb)

                    if nb_vcs:
                        print(f"  {mname:<20} {K_nb:>4} {np.mean(nb_vcs):>7.4f} ${np.mean(nb_sps):>9,.0f} "
                              f"{np.mean(nb_binds):>8.1f} {np.mean(nb_recs):>7.4f} {np.mean(nb_precs):>8.4f}")

            # Coverage
            print(f"\n  Coverage:")
            our_n = np.mean([len(df_class[ct].filter((pl.col("py") == eval_py) & (pl.col("aq") == aq))) for aq in AQS])
            v44_ns = []
            for aq in AQS:
                v44p = f"{V44_BASE}/{eval_py}/{aq}/{ct}/"
                if os.path.exists(v44p):
                    v44_ns.append(len(pl.read_parquet(v44p).filter(pl.col("equipment") != "")))
            v44_n = np.mean(v44_ns) if v44_ns else 0
            print(f"  v0c / tiered_top2: {our_n:.0f} branches")
            print(f"  V4.4: {v44_n:.0f} branches ({100*v44_n/our_n:.0f}% of ours)")

    print(f"\nTotal: {time.time()-t0:.0f}s")


def _compute_metrics(sp, idx, is_dormant, is_nb_binder, total_sp, total_da, n_bind, n_d20, n_d50, K, label_cov):
    mask = np.zeros(len(sp), dtype=bool)
    mask[idx] = True
    return {
        "K": K, "universe": len(sp), "sp": float(sp[mask].sum()),
        "binders": int((sp[mask] > 0).sum()),
        "prec": float((sp[mask] > 0).sum() / K),
        "vc": float(sp[mask].sum() / total_sp) if total_sp > 0 else 0,
        "abs_sp": float(sp[mask].sum() / total_da) if total_da > 0 else 0,
        "rec": float((sp[mask] > 0).sum() / n_bind) if n_bind > 0 else 0,
        "nb_in": int((mask & is_dormant).sum()),
        "nb_b": int((mask & is_nb_binder).sum()),
        "nb_sp": float(sp[mask & is_nb_binder].sum()),
        "d20": int((sp[mask] > 20000).sum()), "d20_tot": n_d20,
        "d50": int((sp[mask] > 50000).sum()), "d50_tot": n_d50,
        "label_cov": label_cov,
    }


def _compute_v44_metrics(v44_topk, gt_map, branches, is_dormant, is_nb_binder, sp,
                         total_sp, total_da, n_bind, n_d20, n_d50, K, v44_universe):
    branch_set = set(branches)
    labeled = 0; v44_sp = 0; v44_binders = 0; v44_nb_in = 0; v44_nb_b = 0; v44_nb_sp = 0
    v44_d20 = 0; v44_d50 = 0
    for b in v44_topk:
        if b in branch_set:
            labeled += 1
            idx = branches.index(b)
            v44_sp += sp[idx]
            if sp[idx] > 0:
                v44_binders += 1
            if is_dormant[idx]:
                v44_nb_in += 1
            if is_nb_binder[idx]:
                v44_nb_b += 1
                v44_nb_sp += sp[idx]
            if sp[idx] > 20000:
                v44_d20 += 1
            if sp[idx] > 50000:
                v44_d50 += 1
    return {
        "K": K, "universe": v44_universe, "sp": float(v44_sp),
        "binders": v44_binders,
        "prec": float(v44_binders / labeled) if labeled > 0 else 0,
        "vc": float(v44_sp / total_sp) if total_sp > 0 else 0,
        "abs_sp": float(v44_sp / total_da) if total_da > 0 else 0,
        "rec": float(v44_binders / n_bind) if n_bind > 0 else 0,
        "nb_in": v44_nb_in, "nb_b": v44_nb_b, "nb_sp": float(v44_nb_sp),
        "d20": v44_d20, "d20_tot": n_d20, "d50": v44_d50, "d50_tot": n_d50,
        "label_cov": labeled, "unlabeled": K - labeled,
    }


def _print_row(mname, mrs, K):
    n = len(mrs)
    def avg(key): return sum(r[key] for r in mrs) / n
    d20s = f"{avg('d20'):.0f}/{avg('d20_tot'):.0f}"
    d50s = f"{avg('d50'):.0f}/{avg('d50_tot'):.0f}"
    lbl = f"{avg('label_cov'):.0f}/{K}"
    if any("unlabeled" in r for r in mrs):
        unlbl = sum(r.get("unlabeled", 0) for r in mrs) / n
        if unlbl > 0:
            lbl = f"{avg('label_cov'):.0f}/{K} ({unlbl:.0f}?)"
    print(f"  {mname:<20} {avg('universe'):>5.0f} ${avg('sp'):>11,.0f} {avg('binders'):>5.0f} "
          f"{avg('prec'):>5.3f} {avg('vc'):>6.3f} {avg('abs_sp'):>6.3f} {avg('rec'):>6.3f}"
          f" {avg('nb_in'):>5.0f} {avg('nb_b'):>4.0f} ${avg('nb_sp'):>9,.0f}"
          f" {d20s:>7} {d50s:>5} {lbl:>7}")


if __name__ == "__main__":
    main()
