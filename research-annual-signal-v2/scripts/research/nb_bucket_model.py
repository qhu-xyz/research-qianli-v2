"""Bucket_6_20: danger-aware unified LambdaRank model.

5-tier severity labels with aggressive weights on dangerous binders:
  tier 0: SP = 0          (weight 1)
  tier 1: 0 < SP <= 200   (weight 1)
  tier 2: 200 < SP <= 5K  (weight 2)
  tier 3: 5K < SP <= 20K  (weight 6)
  tier 4: SP > 20K        (weight 20)

Trains on ALL branches (not just dormant), evaluated per class type.
Saves registry artifacts + comparison vs V4.4.

Usage:
    source /home/xyz/workspace/pmodel/.venv/bin/activate
    PYTHONPATH=. uv run python scripts/nb_bucket_model.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lightgbm as lgb
import numpy as np
import polars as pl

from ml.config import CLASS_BF_COL

V44_BASE = "/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V4.4.R1"
CACHE_DIR = "data/nb_cache"

UNIFIED_FEATURES = [
    "da_rank_value", "shadow_price_da", "bf", "count_active_cids",
    "bin_80_max", "bin_90_max", "bin_100_max", "bin_110_max", "rt_max",
    "top2_bin_80", "top2_bin_90", "top2_bin_100", "top2_bin_110",
]

LGB_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 15, "learning_rate": 0.05,
    "min_child_samples": 5, "subsample": 0.8,
    "colsample_bytree": 0.8, "num_threads": 4, "verbose": -1,
}

BUCKET_BOUNDS = [0, 200, 5000, 20000]  # tier boundaries
BUCKET_WEIGHTS = {0: 1, 1: 1, 2: 2, 3: 6, 4: 20}

ALL_PYS = ["2018-06", "2019-06", "2020-06", "2021-06", "2022-06", "2023-06", "2024-06", "2025-06"]
AQS = ["aq1", "aq2", "aq3"]
EVAL_CONFIGS = [
    ("2022-06", [py for py in ALL_PYS if py < "2022-06"]),
    ("2023-06", [py for py in ALL_PYS if py < "2023-06"]),
    ("2024-06", [py for py in ALL_PYS if py < "2024-06"]),
    ("2025-06", [py for py in ALL_PYS if py < "2025-06"]),
]


def assign_bucket_labels(sp):
    """Assign 5-tier severity labels."""
    labels = np.zeros(len(sp), dtype=np.int32)
    labels[sp > 0] = 1
    labels[sp > BUCKET_BOUNDS[1]] = 2
    labels[sp > BUCKET_BOUNDS[2]] = 3
    labels[sp > BUCKET_BOUNDS[3]] = 4
    return labels


def assign_bucket_weights(labels):
    return np.array([BUCKET_WEIGHTS[int(l)] for l in labels])


def load_cached_data():
    """Load all cached class-specific tables + top2."""
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
                cols = ["branch", "py", "aq", "sp", "bf",
                        "bin_80_max", "bin_90_max", "bin_100_max", "bin_110_max",
                        "rt_max", "count_active_cids", "shadow_price_da", "da_rank_value",
                        "top2_bin_80", "top2_bin_90", "top2_bin_100", "top2_bin_110"]
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
    print(f"Data loaded: {time.time()-t0:.1f}s")

    results = []

    for eval_py, train_pys in EVAL_CONFIGS:
        for ct in ["onpeak", "offpeak"]:
            feats = [f for f in UNIFIED_FEATURES if f in df_class[ct].columns]
            train = df_class[ct].filter(pl.col("py").is_in(train_pys))
            X_train = train.select(feats).to_numpy().astype(np.float64)
            groups = train.group_by(["py", "aq"], maintain_order=True).len()["len"].to_numpy()
            sp_t = train["sp"].to_numpy().astype(np.float64)

            labels = assign_bucket_labels(sp_t)
            weights = assign_bucket_weights(labels)
            ds = lgb.Dataset(X_train, label=labels, group=groups, weight=weights,
                             feature_name=feats, free_raw_data=False)
            model = lgb.train(LGB_PARAMS, ds, num_boost_round=150)

            for K in [200, 400]:
                for aq in AQS:
                    eq = df_class[ct].filter(
                        (pl.col("py") == eval_py) & (pl.col("aq") == aq)
                    )
                    if len(eq) == 0:
                        continue
                    N = len(eq)
                    sp = eq["sp"].to_numpy().astype(np.float64)
                    bf = eq["bf"].to_numpy().astype(np.float64)
                    branches = eq["branch"].to_list()
                    is_dormant = bf == 0
                    is_nb_binder = is_dormant & (sp > 0)

                    scores = model.predict(eq.select(feats).to_numpy().astype(np.float64))
                    topk = np.argsort(scores)[::-1][:K]
                    mask = np.zeros(N, dtype=bool)
                    mask[topk] = True

                    # V4.4 native
                    v44p = f"{V44_BASE}/{eval_py}/{aq}/{ct}/"
                    v44_sp = 0; v44_bind = 0; v44_nb_b = 0; v44_nb_sp = 0
                    if os.path.exists(v44p):
                        v44_df = pl.read_parquet(v44p).filter(
                            pl.col("equipment") != ""
                        ).sort("rank")
                        v44_top = v44_df["equipment"].to_list()[:K]
                        br_set = set(branches)
                        v44_mask = np.zeros(N, dtype=bool)
                        for b in v44_top:
                            if b in br_set:
                                v44_mask[branches.index(b)] = True
                        v44_sp = float(sp[v44_mask].sum())
                        v44_bind = int((sp[v44_mask] > 0).sum())
                        v44_nb_b = int((v44_mask & is_nb_binder).sum())
                        v44_nb_sp = float(sp[v44_mask & is_nb_binder].sum())

                    results.append({
                        "eval_py": eval_py, "aq": aq, "ct": ct, "K": K,
                        "bk_sp": float(sp[mask].sum()),
                        "bk_bind": int((sp[mask] > 0).sum()),
                        "bk_prec": float((sp[mask] > 0).sum() / K),
                        "bk_nb_b": int((mask & is_nb_binder).sum()),
                        "bk_nb_sp": float(sp[mask & is_nb_binder].sum()),
                        "v44_sp": v44_sp,
                        "v44_bind": v44_bind,
                        "v44_nb_b": v44_nb_b,
                        "v44_nb_sp": v44_nb_sp,
                    })

    rdf = pl.DataFrame(results)

    # Print comparison
    for ct in ["onpeak", "offpeak"]:
        for K in [200, 400]:
            tier = "Tier 0" if K == 200 else "Tier 0+1"
            print(f"\n  {ct} / K={K} ({tier}):")
            print(f"  {'Year':<8} {'Bkt SP':>12} {'V44 SP':>12} {'dSP':>12} {'Bkt NB$':>10} {'V44 NB$':>10} {'dNB$':>10}")
            print(f"  {'-'*70}")
            for eval_py in ["2022-06", "2023-06", "2024-06", "2025-06"]:
                r = rdf.filter(
                    (pl.col("eval_py") == eval_py) & (pl.col("ct") == ct) & (pl.col("K") == K)
                )
                if len(r) == 0:
                    continue
                bsp = r["bk_sp"].mean()
                vsp = r["v44_sp"].mean()
                bnb = r["bk_nb_sp"].mean()
                vnb = r["v44_nb_sp"].mean()
                print(f"  {eval_py:<8} ${bsp:>11,.0f} ${vsp:>11,.0f} ${bsp-vsp:>+11,.0f} ${bnb:>9,.0f} ${vnb:>9,.0f} ${bnb-vnb:>+9,.0f}")

    # Save registry
    print("\nSaving registry...")
    for ct in ["onpeak", "offpeak"]:
        path = f"registry/{ct}/bucket_6_20"
        os.makedirs(path, exist_ok=True)

        config = {
            "model": "bucket_6_20",
            "class_type": ct,
            "features": UNIFIED_FEATURES,
            "bucket_bounds": BUCKET_BOUNDS,
            "bucket_weights": BUCKET_WEIGHTS,
            "lgb_params": LGB_PARAMS,
            "training_pys": ALL_PYS,
            "description": "5-tier danger-aware LambdaRank on all branches. "
                           "Tier 4 (SP>20K) gets 20x weight.",
        }
        with open(f"{path}/config.json", "w") as f:
            json.dump(config, f, indent=2)

        metrics = rdf.filter(pl.col("ct") == ct).to_dicts()
        with open(f"{path}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"  {path}/ saved")

    # Feature importance (last trained model)
    imp = model.feature_importance(importance_type="gain")
    total = imp.sum()
    print(f"\nFeature importance ({ct}):")
    for f, i in sorted(zip(feats, imp), key=lambda x: -x[1]):
        print(f"  {f:<22} {100*i/total:.1f}%")

    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
