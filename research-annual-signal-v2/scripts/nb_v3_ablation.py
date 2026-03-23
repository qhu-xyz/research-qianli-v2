"""NB V3 ablation: 9 variants testing 2020 data, class-specific train, labels, features.

Variants:
  0: V2_baseline (no 2020, combined train, tertile, baseline 8 features)
  1: +2020
  2: +class_train (class-specific shadow_price_da / da_rank_value)
  3: +log1p (scaled 0-255 continuous relevance)
  4: +binary_sqrt (binary objective, sqrt(SP) weights)
  5: +tiered_wt (tertile labels with [1,1,3,10] weights)
  6: -count (drop count_active_cids from best label winner)
  7: +ratio (replace count_active_cids with active_ratio)
  8: +top2 (add top2_mean for density bins)

Usage:
    source /home/xyz/workspace/pmodel/.venv/bin/activate
    RAY_ADDRESS=ray://10.8.0.36:10001 PYTHONPATH=. uv run python scripts/nb_v3_ablation.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAY_ADDRESS", "ray://10.8.0.36:10001")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lightgbm as lgb
import numpy as np
import polars as pl

from ml.config import (
    DENSITY_PATH, BRIDGE_PATH, get_market_months, CLASS_BF_COL,
    SELECTED_BINS, RIGHT_TAIL_BINS,
)

LGB_RANK_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 15, "learning_rate": 0.05,
    "min_child_samples": 5, "subsample": 0.8,
    "colsample_bytree": 0.8, "num_threads": 4, "verbose": -1,
}

LGB_BINARY_PARAMS = {
    "objective": "binary", "metric": "auc",
    "num_leaves": 15, "learning_rate": 0.05,
    "min_child_samples": 5, "subsample": 0.8,
    "colsample_bytree": 0.8, "num_threads": 4, "verbose": -1,
}

BASELINE_FEATURES = [
    "bin_80_max", "bin_90_max", "bin_100_max", "bin_110_max",
    "rt_max", "count_active_cids", "shadow_price_da", "da_rank_value",
]


# ── Helpers ────────────────────────────────────────────────────────────

def _minmax(arr):
    mn, mx = arr.min(), arr.max()
    return np.full_like(arr, 0.5) if mx == mn else (arr - mn) / (mx - mn)


def compute_v0c(da, rt, bf):
    return 0.40 * (1.0 - _minmax(da)) + 0.30 * _minmax(rt) + 0.30 * _minmax(bf)


def assign_tiers_per_group(sp, groups):
    labels = np.zeros(len(sp), dtype=np.int32)
    offset = 0
    for g in groups:
        sl = slice(offset, offset + g)
        sp_g = sp[sl]; pos = sp_g > 0
        if pos.sum() > 0:
            r = sp_g[pos].argsort().argsort(); n = len(r)
            labels[sl][pos] = np.where(r < n // 3, 1, np.where(r < 2 * n // 3, 2, 3))
        offset += g
    return labels


def assign_scaled_log1p_per_group(sp, groups):
    """Bounded 0-30 continuous relevance from log1p(SP), per group.

    LambdaRank max label = 2*num_leaves for label mapping. With num_leaves=15,
    max safe label = 30. Scale log1p(SP) to 1-30 for binders, 0 for non-binders.
    """
    MAX_LABEL = 30
    labels = np.zeros(len(sp), dtype=np.int32)
    offset = 0
    for g in groups:
        sl = slice(offset, offset + g)
        sp_g = sp[sl]; pos = sp_g > 0
        if pos.sum() > 0:
            log_sp = np.log1p(sp_g[pos])
            max_log = log_sp.max()
            if max_log > 0:
                scaled = np.clip(np.round(log_sp / max_log * (MAX_LABEL - 1) + 1), 1, MAX_LABEL).astype(np.int32)
            else:
                scaled = np.ones(pos.sum(), dtype=np.int32)
            labels[sl][pos] = scaled
        offset += g
    return labels


def compute_tiered_weights(sp, labels):
    """Sample weights [1, 1, 3, 10] for tiers 0/1/2/3."""
    wt_map = {0: 1.0, 1: 1.0, 2: 3.0, 3: 10.0}
    return np.array([wt_map[int(l)] for l in labels])


def compute_binary_sqrt_weights(sp, labels):
    """Binary weights: sqrt(SP) for binders × class imbalance correction."""
    w = np.ones(len(sp))
    pos = labels > 0
    n0, n1 = (labels == 0).sum(), pos.sum()
    if n1 > 0:
        w[pos] = np.sqrt(sp[pos]) * (n0 / max(n1, 1))
    return w


def allocate_reserved_slots(v0c_scores, nb_scores, is_dormant, n_v0c, n_nb):
    K = n_v0c + n_nb
    selected = set()
    v0c_order = np.argsort(v0c_scores)[::-1]
    for idx in v0c_order:
        if len(selected) >= n_v0c: break
        selected.add(int(idx))
    nb_filled = 0
    if n_nb > 0:
        candidates = [(i, nb_scores[i]) for i in range(len(v0c_scores))
                      if is_dormant[i] and i not in selected and np.isfinite(nb_scores[i])]
        candidates.sort(key=lambda x: -x[1])
        for idx, _ in candidates[:n_nb]:
            selected.add(idx); nb_filled += 1
    if len(selected) < K:
        for idx in v0c_order:
            if len(selected) >= K: break
            selected.add(int(idx))
    return selected, nb_filled


def compute_top2_mean(planning_year, aq_quarter, bins=("80", "90", "100", "110")):
    """Compute mean of top-2 CID values per branch for each bin."""
    market_months = get_market_months(planning_year, aq_quarter)
    frames = []
    for mm in market_months:
        p = f"{DENSITY_PATH}/spice_version=v6/auction_type=annual/auction_month={planning_year}/market_month={mm}/market_round=1/"
        if Path(p).exists():
            frames.append(pl.read_parquet(p))
    if not frames:
        return pl.DataFrame(schema={"branch_name": pl.Utf8})
    raw = pl.concat(frames, how="diagonal")

    # Level 1: mean across outage dates per CID
    cid_level = raw.group_by("constraint_id").agg(
        [pl.col(b).mean().alias(b) for b in bins]
    )

    # Join bridge
    bp = f"{BRIDGE_PATH}/spice_version=v6/auction_type=annual/auction_month={planning_year}/"
    if not Path(bp).exists():
        return pl.DataFrame(schema={"branch_name": pl.Utf8})
    bridge = pl.read_parquet(bp)
    if "convention" in bridge.columns:
        bridge = bridge.filter(pl.col("convention") < 10)
    bridge = bridge.select(["constraint_id", "branch_name"]).unique()
    cid_with_branch = cid_level.join(bridge, on="constraint_id", how="inner")

    # Level 2: top2_mean per branch per bin
    result_frames = []
    for b in bins:
        # Sort CID values descending within each branch, take top 2, mean
        ranked = cid_with_branch.select(["branch_name", b]).sort(b, descending=True)
        top2 = ranked.group_by("branch_name").head(2)
        top2_agg = top2.group_by("branch_name").agg(
            pl.col(b).mean().alias(f"top2_bin_{b}")
        )
        result_frames.append(top2_agg)

    result = result_frames[0]
    for f in result_frames[1:]:
        result = result.join(f, on="branch_name", how="outer_coalesce")
    return result


# ── Variant definitions ────────────────────────────────────────────────

class Variant:
    def __init__(self, name, use_2020, class_specific_train, label_fn, objective,
                 weight_fn, features, lgb_params):
        self.name = name
        self.use_2020 = use_2020
        self.class_specific_train = class_specific_train
        self.label_fn = label_fn
        self.objective = objective
        self.weight_fn = weight_fn
        self.features = features
        self.lgb_params = lgb_params


def make_variants(best_label_name="tertile"):
    """Build the 9-variant matrix. Rows 6-8 use the best label from rows 3-5."""
    bl_fn = assign_tiers_per_group
    bl_obj = "lambdarank"
    bl_wfn = None
    bl_params = LGB_RANK_PARAMS

    if best_label_name == "log1p":
        bl_fn = assign_scaled_log1p_per_group
    elif best_label_name == "binary_sqrt":
        bl_fn = lambda sp, groups: (sp > 0).astype(np.int32)  # binary labels
        bl_obj = "binary"
        bl_wfn = compute_binary_sqrt_weights
        bl_params = LGB_BINARY_PARAMS
    elif best_label_name == "tiered_wt":
        bl_wfn = compute_tiered_weights

    feat_no_count = [f for f in BASELINE_FEATURES if f != "count_active_cids"]
    feat_ratio = [f for f in BASELINE_FEATURES if f != "count_active_cids"] + ["active_ratio"]
    feat_top2 = BASELINE_FEATURES + ["top2_bin_80", "top2_bin_90", "top2_bin_100", "top2_bin_110"]

    return [
        Variant("V2_baseline", False, False, assign_tiers_per_group, "lambdarank", None, BASELINE_FEATURES, LGB_RANK_PARAMS),
        Variant("+2020", True, False, assign_tiers_per_group, "lambdarank", None, BASELINE_FEATURES, LGB_RANK_PARAMS),
        Variant("+class_train", True, True, assign_tiers_per_group, "lambdarank", None, BASELINE_FEATURES, LGB_RANK_PARAMS),
        Variant("+log1p", True, True, assign_scaled_log1p_per_group, "lambdarank", None, BASELINE_FEATURES, LGB_RANK_PARAMS),
        Variant("+binary_sqrt", True, True, lambda sp, g: (sp > 0).astype(np.int32), "binary", compute_binary_sqrt_weights, BASELINE_FEATURES, LGB_BINARY_PARAMS),
        Variant("+tiered_wt", True, True, assign_tiers_per_group, "lambdarank", compute_tiered_weights, BASELINE_FEATURES, LGB_RANK_PARAMS),
        Variant("-count", True, True, bl_fn, bl_obj, bl_wfn, feat_no_count, bl_params),
        Variant("+ratio", True, True, bl_fn, bl_obj, bl_wfn, feat_ratio, bl_params),
        Variant("+top2", True, True, bl_fn, bl_obj, bl_wfn, feat_top2, bl_params),
    ]


# ── Main ───────────────────────────────────────────────────────────────

def main():
    from pbase.config.ray import init_ray
    init_ray()

    t0 = time.time()
    all_pys_full = ["2020-06", "2021-06", "2022-06", "2023-06", "2024-06", "2025-06"]
    all_pys_no2020 = ["2021-06", "2022-06", "2023-06", "2024-06", "2025-06"]
    aqs = ["aq1", "aq2", "aq3"]

    eval_configs = [
        ("2023-06", {"with2020": ["2020-06", "2021-06", "2022-06"], "no2020": ["2021-06", "2022-06"]}),
        ("2024-06", {"with2020": ["2020-06", "2021-06", "2022-06", "2023-06"], "no2020": ["2021-06", "2022-06", "2023-06"]}),
        ("2025-06", {"with2020": ["2020-06", "2021-06", "2022-06", "2023-06", "2024-06"], "no2020": ["2021-06", "2022-06", "2023-06", "2024-06"]}),
    ]

    # ── Build data ─────────────────────────────────────────────────────
    print("Building data (combined + class-specific)...")
    from ml.features import build_model_table
    from ml.phase6.features import build_class_model_table

    # Combined data (for V2_baseline, +2020 variants)
    combined_rows = []
    for py in all_pys_full:
        for aq in aqs:
            try:
                t = build_model_table(py, aq)
            except Exception as e:
                print(f"  {py}/{aq} combined: SKIP ({e})")
                continue
            for r in t.iter_rows(named=True):
                combined_rows.append({
                    "branch": r["branch_name"], "py": py, "aq": aq,
                    "sp_onpeak": r["onpeak_sp"], "sp_offpeak": r["offpeak_sp"],
                    "bf_12": r["bf_12"], "bfo_12": r["bfo_12"],
                    "bin_80_max": r["bin_80_cid_max"], "bin_90_max": r["bin_90_cid_max"],
                    "bin_100_max": r["bin_100_cid_max"], "bin_110_max": r["bin_110_cid_max"],
                    "rt_max": max(r["bin_80_cid_max"], r["bin_90_cid_max"],
                                  r["bin_100_cid_max"], r["bin_110_cid_max"]),
                    "count_active_cids": r["count_active_cids"],
                    "count_cids": r.get("count_cids", r["count_active_cids"]),
                    "shadow_price_da": r["shadow_price_da"],
                    "da_rank_value": r["da_rank_value"],
                })
    df_combined = pl.DataFrame(combined_rows)
    df_combined = df_combined.with_columns(
        (pl.col("count_active_cids") / pl.col("count_cids").clip(lower_bound=1)).alias("active_ratio")
    )

    # Class-specific data (for +class_train and later variants)
    class_rows = {ct: [] for ct in ["onpeak", "offpeak"]}
    for py in all_pys_full:
        for aq in aqs:
            for ct in ["onpeak", "offpeak"]:
                try:
                    ct_t = build_class_model_table(py, aq, ct)
                except Exception as e:
                    print(f"  {py}/{aq}/{ct}: SKIP ({e})")
                    continue
                bf_col = CLASS_BF_COL[ct]
                for r in ct_t.iter_rows(named=True):
                    class_rows[ct].append({
                        "branch": r["branch_name"], "py": py, "aq": aq,
                        "sp": r["realized_shadow_price"],
                        "bf": r[bf_col],
                        "bin_80_max": r["bin_80_cid_max"], "bin_90_max": r["bin_90_cid_max"],
                        "bin_100_max": r["bin_100_cid_max"], "bin_110_max": r["bin_110_cid_max"],
                        "rt_max": max(r["bin_80_cid_max"], r["bin_90_cid_max"],
                                      r["bin_100_cid_max"], r["bin_110_cid_max"]),
                        "count_active_cids": r["count_active_cids"],
                        "count_cids": r.get("count_cids", r["count_active_cids"]),
                        "shadow_price_da": r["shadow_price_da"],
                        "da_rank_value": r["da_rank_value"],
                    })
    df_class = {ct: pl.DataFrame(rows) for ct, rows in class_rows.items()}
    for ct in df_class:
        df_class[ct] = df_class[ct].with_columns(
            (pl.col("count_active_cids") / pl.col("count_cids").clip(lower_bound=1)).alias("active_ratio")
        )

    # top2_mean features
    print("Computing top2_mean features...")
    top2_cache = {}
    for py in all_pys_full:
        for aq in aqs:
            try:
                top2_cache[(py, aq)] = compute_top2_mean(py, aq)
            except Exception as e:
                print(f"  top2 {py}/{aq}: SKIP ({e})")
                top2_cache[(py, aq)] = pl.DataFrame(schema={"branch_name": pl.Utf8})

    # Join top2 to combined and class data
    for py in all_pys_full:
        for aq in aqs:
            t2 = top2_cache.get((py, aq))
            if t2 is None or len(t2) == 0:
                continue
            t2_renamed = t2.rename({"branch_name": "branch"})
            mask = (df_combined["py"] == py) & (df_combined["aq"] == aq)
            # Join via temporary df
    # Simpler: add top2 columns to combined df via left join on (py, aq, branch)
    top2_all = []
    for (py, aq), t2 in top2_cache.items():
        if len(t2) == 0:
            continue
        t2 = t2.with_columns(pl.lit(py).alias("py"), pl.lit(aq).alias("aq"))
        t2 = t2.rename({"branch_name": "branch"})
        top2_all.append(t2)
    if top2_all:
        top2_df = pl.concat(top2_all, how="diagonal")
        df_combined = df_combined.join(top2_df, on=["py", "aq", "branch"], how="left")
        for col in ["top2_bin_80", "top2_bin_90", "top2_bin_100", "top2_bin_110"]:
            if col in df_combined.columns:
                df_combined = df_combined.with_columns(pl.col(col).fill_null(0.0))
        # Also join to class dfs
        for ct in df_class:
            df_class[ct] = df_class[ct].join(top2_df, on=["py", "aq", "branch"], how="left")
            for col in ["top2_bin_80", "top2_bin_90", "top2_bin_100", "top2_bin_110"]:
                if col in df_class[ct].columns:
                    df_class[ct] = df_class[ct].with_columns(pl.col(col).fill_null(0.0))

    print(f"Data build: {time.time()-t0:.0f}s, combined={len(df_combined)}, onpeak={len(df_class['onpeak'])}, offpeak={len(df_class['offpeak'])}")

    # ── Pre-cache eval tables (same for all variants) ────────────────
    print("Pre-caching eval tables...")
    eval_tables = {}
    for eval_py, _ in eval_configs:
        for aq in aqs:
            for ct in ["onpeak", "offpeak"]:
                try:
                    eval_tables[(eval_py, aq, ct)] = build_class_model_table(eval_py, aq, ct)
                except Exception as e:
                    print(f"  eval {eval_py}/{aq}/{ct}: SKIP ({e})")
    print(f"  Cached {len(eval_tables)} eval tables ({time.time()-t0:.0f}s)")

    # ── Run variants ───────────────────────────────────────────────────
    # First pass: rows 0-5 to find best label
    variants = make_variants("tertile")  # initial pass
    all_results = []

    for vi, var in enumerate(variants[:6]):
        print(f"\n--- Variant {vi}: {var.name} ---")
        for eval_py, train_py_map in eval_configs:
            train_pys = train_py_map["with2020"] if var.use_2020 else train_py_map["no2020"]

            for ct in ["onpeak", "offpeak"]:
                dormant_col = "bf_12" if ct == "onpeak" else "bfo_12"
                target_col = "sp_onpeak" if ct == "onpeak" else "sp_offpeak"

                # Select training data
                if var.class_specific_train:
                    src = df_class[ct]
                    train = src.filter(pl.col("py").is_in(train_pys) & (pl.col("bf") == 0))
                    sp_arr = train["sp"].to_numpy().astype(np.float64)
                else:
                    train = df_combined.filter(
                        pl.col("py").is_in(train_pys) & (pl.col(dormant_col) == 0)
                    )
                    sp_arr = train[target_col].to_numpy().astype(np.float64)

                if len(train) < 50:
                    continue

                features = [f for f in var.features if f in train.columns]
                X = train.select(features).to_numpy().astype(np.float64)
                groups = train.group_by(["py", "aq"], maintain_order=True).len()["len"].to_numpy()

                # Labels
                y = var.label_fn(sp_arr, groups)

                # Weights
                w = var.weight_fn(sp_arr, y) if var.weight_fn else None

                # Train
                if var.objective == "binary":
                    ds = lgb.Dataset(X, label=y, weight=w, feature_name=features, free_raw_data=False)
                    model = lgb.train(var.lgb_params, ds, num_boost_round=150)
                else:
                    ds = lgb.Dataset(X, label=y, group=groups, weight=w, feature_name=features, free_raw_data=False)
                    model = lgb.train(var.lgb_params, ds, num_boost_round=150)

                # Eval per quarter
                for aq in aqs:
                    ct_table = eval_tables.get((eval_py, aq, ct))
                    if ct_table is None:
                        continue

                    bf_col_eval = CLASS_BF_COL[ct]
                    branches = ct_table["branch_name"].to_list()
                    N = len(ct_table)
                    sp = ct_table["realized_shadow_price"].to_numpy().astype(np.float64)
                    da = ct_table["da_rank_value"].to_numpy().astype(np.float64)
                    bf = ct_table[bf_col_eval].to_numpy().astype(np.float64)
                    rt = ct_table.select(
                        pl.max_horizontal("bin_80_cid_max", "bin_90_cid_max",
                                          "bin_100_cid_max", "bin_110_cid_max")
                    ).to_series().to_numpy().astype(np.float64)
                    total_da = float(ct_table["total_da_sp_quarter"][0])
                    total_sp = sp.sum()
                    n_bind = int((sp > 0).sum())
                    is_dormant = bf == 0
                    is_nb_binder = is_dormant & (sp > 0)
                    total_nb_sp = float(sp[is_nb_binder].sum())
                    n_nb_binder = int(is_nb_binder.sum())

                    if n_bind == 0:
                        continue

                    v0c = compute_v0c(da, rt, bf)

                    # NB model scores for dormant branches
                    nb_scores = np.full(N, -np.inf)
                    if var.class_specific_train:
                        src_q = df_class[ct].filter((pl.col("py") == eval_py) & (pl.col("aq") == aq))
                    else:
                        src_q = df_combined.filter((pl.col("py") == eval_py) & (pl.col("aq") == aq))
                    br_map = {r["branch"]: i for i, r in enumerate(src_q.iter_rows(named=True))}

                    dormant_idx = [i for i in range(N) if is_dormant[i]]
                    feat_rows = []
                    valid_idx = []
                    for di in dormant_idx:
                        br = branches[di]
                        if br in br_map:
                            feat_rows.append(src_q[br_map[br]].select(features))
                            valid_idx.append(di)
                    if feat_rows:
                        X_pred = pl.concat(feat_rows).to_numpy().astype(np.float64)
                        preds = model.predict(X_pred)
                        for di, pred in zip(valid_idx, preds):
                            nb_scores[di] = pred

                    # NB-only metrics
                    nb_sp = sp[is_dormant]
                    nb_total = nb_sp.sum()
                    nb_n_bind = int((nb_sp > 0).sum())
                    if nb_n_bind > 0 and nb_total > 0:
                        for K in [50, 100]:
                            sc = nb_scores[is_dormant]
                            topk = np.argsort(sc)[::-1][:min(K, len(sc))]
                            mk = np.zeros(len(sc), dtype=bool)
                            mk[topk] = True
                            all_results.append({
                                "variant": var.name, "eval_py": eval_py, "aq": aq, "ct": ct,
                                "metric_type": "nb_only", "K": K,
                                "vc": float(nb_sp[mk].sum() / nb_total),
                                "nb_sp": float(nb_sp[mk].sum()),
                                "nb_bind": int((nb_sp[mk] > 0).sum()),
                            })

                    # R30 full universe
                    for K, nv, nn in [(200, 170, 30), (400, 350, 50)]:
                        selected, nb_filled = allocate_reserved_slots(v0c, nb_scores, is_dormant, nv, nn)
                        mask = np.zeros(N, dtype=bool)
                        for idx in selected:
                            mask[idx] = True
                        all_results.append({
                            "variant": var.name, "eval_py": eval_py, "aq": aq, "ct": ct,
                            "metric_type": f"R30@{K}", "K": K,
                            "vc": float(sp[mask].sum() / total_sp) if total_sp > 0 else 0,
                            "nb_sp": float(sp[mask & is_nb_binder].sum()),
                            "nb_bind": int((mask & is_nb_binder).sum()),
                        })

    # Determine best label from rows 3-5
    rdf = pl.DataFrame(all_results)
    label_candidates = ["+log1p", "+binary_sqrt", "+tiered_wt"]
    baseline_r30_400 = rdf.filter(
        (pl.col("variant") == "+class_train") & (pl.col("metric_type") == "R30@400")
    )
    best_label = "+class_train"  # default to tertile if none improve
    best_nb_sp = baseline_r30_400["nb_sp"].mean() if len(baseline_r30_400) > 0 else 0

    for lc in label_candidates:
        lc_r30 = rdf.filter((pl.col("variant") == lc) & (pl.col("metric_type") == "R30@400"))
        if len(lc_r30) == 0:
            continue
        lc_vc = lc_r30["vc"].mean()
        base_vc = baseline_r30_400["vc"].mean() if len(baseline_r30_400) > 0 else 0
        lc_nb = lc_r30["nb_sp"].mean()
        if lc_nb > best_nb_sp and lc_vc >= base_vc - 0.01:
            best_label = lc
            best_nb_sp = lc_nb

    label_map = {
        "+class_train": "tertile",
        "+log1p": "log1p",
        "+binary_sqrt": "binary_sqrt",
        "+tiered_wt": "tiered_wt",
    }
    best_label_key = label_map.get(best_label, "tertile")
    print(f"\nBest label from rows 3-5: {best_label} (key={best_label_key})")

    # Run rows 6-8 with best label
    variants_6_8 = make_variants(best_label_key)[6:9]
    for vi, var in enumerate(variants_6_8, start=6):
        print(f"\n--- Variant {vi}: {var.name} (label={best_label_key}) ---")
        for eval_py, train_py_map in eval_configs:
            train_pys = train_py_map["with2020"]
            for ct in ["onpeak", "offpeak"]:
                src = df_class[ct]
                train = src.filter(pl.col("py").is_in(train_pys) & (pl.col("bf") == 0))
                sp_arr = train["sp"].to_numpy().astype(np.float64)
                if len(train) < 50:
                    continue

                features = [f for f in var.features if f in train.columns]
                X = train.select(features).to_numpy().astype(np.float64)
                groups = train.group_by(["py", "aq"], maintain_order=True).len()["len"].to_numpy()
                y = var.label_fn(sp_arr, groups)
                w = var.weight_fn(sp_arr, y) if var.weight_fn else None

                if var.objective == "binary":
                    ds = lgb.Dataset(X, label=y, weight=w, feature_name=features, free_raw_data=False)
                else:
                    ds = lgb.Dataset(X, label=y, group=groups, weight=w, feature_name=features, free_raw_data=False)
                model = lgb.train(var.lgb_params, ds, num_boost_round=150)

                for aq in aqs:
                    ct_table = eval_tables.get((eval_py, aq, ct))
                    if ct_table is None:
                        continue

                    bf_col_eval = CLASS_BF_COL[ct]
                    branches = ct_table["branch_name"].to_list()
                    N = len(ct_table)
                    sp = ct_table["realized_shadow_price"].to_numpy().astype(np.float64)
                    da = ct_table["da_rank_value"].to_numpy().astype(np.float64)
                    bf = ct_table[bf_col_eval].to_numpy().astype(np.float64)
                    rt = ct_table.select(
                        pl.max_horizontal("bin_80_cid_max", "bin_90_cid_max",
                                          "bin_100_cid_max", "bin_110_cid_max")
                    ).to_series().to_numpy().astype(np.float64)
                    total_sp = sp.sum()
                    n_bind = int((sp > 0).sum())
                    is_dormant = bf == 0
                    is_nb_binder = is_dormant & (sp > 0)
                    total_nb_sp = float(sp[is_nb_binder].sum())
                    n_nb_binder = int(is_nb_binder.sum())
                    if n_bind == 0:
                        continue

                    v0c = compute_v0c(da, rt, bf)
                    nb_scores = np.full(N, -np.inf)
                    src_q = df_class[ct].filter((pl.col("py") == eval_py) & (pl.col("aq") == aq))
                    br_map = {r["branch"]: i for i, r in enumerate(src_q.iter_rows(named=True))}
                    dormant_idx = [i for i in range(N) if is_dormant[i]]
                    feat_rows = []; valid_idx = []
                    for di in dormant_idx:
                        br = branches[di]
                        if br in br_map:
                            feat_rows.append(src_q[br_map[br]].select(features))
                            valid_idx.append(di)
                    if feat_rows:
                        X_pred = pl.concat(feat_rows).to_numpy().astype(np.float64)
                        preds = model.predict(X_pred)
                        for di, pred in zip(valid_idx, preds):
                            nb_scores[di] = pred

                    nb_sp_arr = sp[is_dormant]
                    nb_total = nb_sp_arr.sum()
                    nb_n_bind = int((nb_sp_arr > 0).sum())
                    if nb_n_bind > 0 and nb_total > 0:
                        for K in [50, 100]:
                            sc = nb_scores[is_dormant]
                            topk = np.argsort(sc)[::-1][:min(K, len(sc))]
                            mk = np.zeros(len(sc), dtype=bool)
                            mk[topk] = True
                            all_results.append({
                                "variant": var.name, "eval_py": eval_py, "aq": aq, "ct": ct,
                                "metric_type": "nb_only", "K": K,
                                "vc": float(nb_sp_arr[mk].sum() / nb_total),
                                "nb_sp": float(nb_sp_arr[mk].sum()),
                                "nb_bind": int((nb_sp_arr[mk] > 0).sum()),
                            })

                    for K, nv, nn in [(200, 170, 30), (400, 350, 50)]:
                        selected, nb_filled = allocate_reserved_slots(v0c, nb_scores, is_dormant, nv, nn)
                        mask = np.zeros(N, dtype=bool)
                        for idx in selected:
                            mask[idx] = True
                        all_results.append({
                            "variant": var.name, "eval_py": eval_py, "aq": aq, "ct": ct,
                            "metric_type": f"R30@{K}", "K": K,
                            "vc": float(sp[mask].sum() / total_sp) if total_sp > 0 else 0,
                            "nb_sp": float(sp[mask & is_nb_binder].sum()),
                            "nb_bind": int((mask & is_nb_binder).sum()),
                        })

    # ── Report ─────────────────────────────────────────────────────────
    rdf = pl.DataFrame(all_results)
    var_order = ["V2_baseline", "+2020", "+class_train", "+log1p", "+binary_sqrt",
                 "+tiered_wt", "-count", "+ratio", "+top2"]

    print(f"\n{'='*100}")
    print("NB V3 ABLATION RESULTS")
    print("=" * 100)

    for mt in ["nb_only", "R30@200", "R30@400"]:
        for ct in ["onpeak", "offpeak"]:
            print(f"\n  === {mt} / {ct} ===")

            # Per year
            for eval_py in ["2024-06", "2025-06"]:
                rows = rdf.filter(
                    (pl.col("metric_type") == mt) & (pl.col("ct") == ct) & (pl.col("eval_py") == eval_py)
                )
                if len(rows) == 0:
                    continue
                K_val = int(rows["K"][0])
                print(f"\n    {eval_py} (K={K_val}):")
                print(f"    {'Variant':<16} {'VC':>7} {'NB_SP':>10} {'NB_bind':>8}")
                print(f"    {'-'*45}")
                for v in var_order:
                    r = rows.filter(pl.col("variant") == v)
                    if len(r) == 0:
                        continue
                    print(f"    {v:<16} {r['vc'].mean():>7.4f} {r['nb_sp'].mean():>10,.0f} {r['nb_bind'].mean():>8.1f}")

            # Aggregate
            rows = rdf.filter((pl.col("metric_type") == mt) & (pl.col("ct") == ct))
            if len(rows) == 0:
                continue
            K_val = int(rows["K"][0])
            print(f"\n    AGGREGATE (K={K_val}):")
            print(f"    {'Variant':<16} {'Grp':>4} {'VC':>7} {'NB_SP':>10} {'NB_bind':>8}")
            print(f"    {'-'*50}")
            for v in var_order:
                r = rows.filter(pl.col("variant") == v)
                if len(r) == 0:
                    continue
                print(f"    {v:<16} {len(r):>4} {r['vc'].mean():>7.4f} {r['nb_sp'].mean():>10,.0f} {r['nb_bind'].mean():>8.1f}")

    # Delta vs V2_baseline
    print(f"\n{'='*100}")
    print("DELTA vs V2_baseline (aggregate)")
    print("=" * 100)
    for mt in ["R30@200", "R30@400"]:
        for ct in ["onpeak", "offpeak"]:
            base = rdf.filter(
                (pl.col("variant") == "V2_baseline") & (pl.col("metric_type") == mt) & (pl.col("ct") == ct)
            )
            if len(base) == 0:
                continue
            bvc = base["vc"].mean()
            bnb = base["nb_sp"].mean()
            print(f"\n    {mt} / {ct} (base VC={bvc:.4f}, NB_SP=${bnb:,.0f}):")
            print(f"    {'Variant':<16} {'dVC':>8} {'dNB_SP':>10}")
            print(f"    {'-'*35}")
            for v in var_order[1:]:
                r = rdf.filter(
                    (pl.col("variant") == v) & (pl.col("metric_type") == mt) & (pl.col("ct") == ct)
                )
                if len(r) == 0:
                    continue
                print(f"    {v:<16} {r['vc'].mean()-bvc:>+8.4f} ${r['nb_sp'].mean()-bnb:>+9,.0f}")

    # Save all variants to registry
    print("\nSaving registry artifacts...")
    for ct in ["onpeak", "offpeak"]:
        base_path = f"registry/{ct}/nb_v3"
        os.makedirs(base_path, exist_ok=True)

        for v in var_order:
            v_path = f"{base_path}/{v.replace('+','plus_').replace('-','minus_')}"
            os.makedirs(v_path, exist_ok=True)
            v_metrics = rdf.filter((pl.col("variant") == v) & (pl.col("ct") == ct)).to_dicts()
            with open(f"{v_path}/metrics.json", "w") as f:
                json.dump(v_metrics, f, indent=2)

        # Comparison summary
        comparison = {}
        for v in var_order:
            for mt in ["R30@200", "R30@400"]:
                r = rdf.filter(
                    (pl.col("variant") == v) & (pl.col("metric_type") == mt) & (pl.col("ct") == ct)
                )
                if len(r) == 0:
                    continue
                comparison[f"{v}/{mt}"] = {
                    "vc": float(r["vc"].mean()),
                    "nb_sp": float(r["nb_sp"].mean()),
                    "nb_bind": float(r["nb_bind"].mean()),
                }
        with open(f"{base_path}/comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

        print(f"  {base_path}/ saved ({len(var_order)} variants)")

    print(f"\nBest label: {best_label} (key={best_label_key})")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
