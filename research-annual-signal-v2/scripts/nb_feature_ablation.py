"""NB feature ablation: test density-derived feature families for dormant branch prediction.

Computes tail probability and expected excess features from raw 77-bin density distribution,
then runs the same per-ctype NB experiment (V2 framework) with 5 feature sets:

  A: baseline (current 8 features)
  B: +tail_probs (prob_above_90, prob_above_100, prob_above_110)
  C: +expected_excess (excess_above_100, excess_above_110)
  D: +extreme_bins_limits (bin_120, bin_150, bin_-100, limit_mean, limit_std)
  E: all of the above

Reports NB-only and full-universe metrics per (ctype, year, feature_set).

Usage:
    source /home/xyz/workspace/pmodel/.venv/bin/activate
    RAY_ADDRESS=ray://10.8.0.36:10001 PYTHONPATH=. uv run python scripts/nb_feature_ablation.py
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

from ml.config import (
    DENSITY_PATH, BRIDGE_PATH, get_market_months, UNIVERSE_THRESHOLD,
    RIGHT_TAIL_BINS,
)
from ml.data_loader import load_collapsed
from ml.bridge import map_cids_to_branches

# Row sums = 20 in raw density. Normalize to probability.
NORM_FACTOR = 20.0

# Bins above thresholds (from the 77-bin schema)
BINS_ABOVE_90 = ["90", "95", "100", "105", "110", "115", "120", "125", "130",
                 "135", "140", "145", "150", "160", "180", "200", "220", "240",
                 "260", "280", "300"]
BINS_ABOVE_100 = ["100", "105", "110", "115", "120", "125", "130", "135", "140",
                  "145", "150", "160", "180", "200", "220", "240", "260", "280",
                  "300"]
BINS_ABOVE_110 = ["110", "115", "120", "125", "130", "135", "140", "145", "150",
                  "160", "180", "200", "220", "240", "260", "280", "300"]

# Bin centers for expected excess calculation
BIN_CENTERS_ABOVE_100 = {
    "100": 100, "105": 105, "110": 110, "115": 115, "120": 120,
    "125": 125, "130": 130, "135": 135, "140": 140, "145": 145,
    "150": 150, "160": 160, "180": 180, "200": 200, "220": 220,
    "240": 240, "260": 260, "280": 280, "300": 300,
}
BIN_CENTERS_ABOVE_110 = {k: v for k, v in BIN_CENTERS_ABOVE_100.items() if v >= 110}


def compute_density_features(planning_year: str, aq_quarter: str) -> pl.DataFrame:
    """Compute tail probability and expected excess features per branch.

    Returns DataFrame with branch_name + new density features.
    """
    market_months = get_market_months(planning_year, aq_quarter)

    # Load raw density
    frames = []
    for mm in market_months:
        path = (
            f"{DENSITY_PATH}/spice_version=v6/auction_type=annual"
            f"/auction_month={planning_year}/market_month={mm}/market_round=1/"
        )
        from pathlib import Path
        if not Path(path).exists():
            continue
        df = pl.read_parquet(path)
        frames.append(df)

    if not frames:
        return pl.DataFrame(schema={"branch_name": pl.Utf8})

    raw = pl.concat(frames, how="diagonal")

    # Per-row: compute tail probabilities (normalized by NORM_FACTOR)
    raw = raw.with_columns(
        (pl.sum_horizontal([pl.col(b) for b in BINS_ABOVE_90]) / NORM_FACTOR).alias("prob_above_90"),
        (pl.sum_horizontal([pl.col(b) for b in BINS_ABOVE_100]) / NORM_FACTOR).alias("prob_above_100"),
        (pl.sum_horizontal([pl.col(b) for b in BINS_ABOVE_110]) / NORM_FACTOR).alias("prob_above_110"),
    )

    # Per-row: expected excess above 100 and 110 (E[flow - threshold | flow > threshold])
    # = sum(bin_weight/20 * (center - threshold)) for centers above threshold
    excess_100_expr = pl.sum_horizontal([
        pl.col(b) / NORM_FACTOR * (center - 100)
        for b, center in BIN_CENTERS_ABOVE_100.items()
    ])
    excess_110_expr = pl.sum_horizontal([
        pl.col(b) / NORM_FACTOR * (center - 110)
        for b, center in BIN_CENTERS_ABOVE_110.items()
    ])
    raw = raw.with_columns(
        excess_100_expr.alias("excess_above_100"),
        excess_110_expr.alias("excess_above_110"),
    )

    # Level 1: mean across outage dates per CID
    feature_cols = ["prob_above_90", "prob_above_100", "prob_above_110",
                    "excess_above_100", "excess_above_110"]
    cid_level = raw.group_by("constraint_id").agg(
        [pl.col(f).mean().alias(f) for f in feature_cols]
    )

    # Join bridge to get branch_name
    bridge_path = (
        f"{BRIDGE_PATH}/spice_version=v6/auction_type=annual"
        f"/auction_month={planning_year}/"
    )
    if not Path(bridge_path).exists():
        return pl.DataFrame(schema={"branch_name": pl.Utf8})

    bridge = pl.read_parquet(bridge_path)
    if "convention" in bridge.columns:
        bridge = bridge.filter(pl.col("convention") < 10)
    bridge = bridge.select(["constraint_id", "branch_name"]).unique()

    cid_with_branch = cid_level.join(bridge, on="constraint_id", how="inner")

    # Level 2: max and mean across CIDs per branch
    agg_exprs = []
    for f in feature_cols:
        agg_exprs.append(pl.col(f).max().alias(f"{f}_max"))
        agg_exprs.append(pl.col(f).mean().alias(f"{f}_mean"))

    return cid_with_branch.group_by("branch_name").agg(agg_exprs)


# ── Feature sets ───────────────────────────────────────────────────────

BASELINE_FEATURES = [
    "bin_80_max", "bin_90_max", "bin_100_max", "bin_110_max",
    "rt_max", "count_active_cids", "shadow_price_da", "da_rank_value",
]

TAIL_PROB_FEATURES = [
    "prob_above_90_max", "prob_above_100_max", "prob_above_110_max",
    "prob_above_90_mean", "prob_above_100_mean", "prob_above_110_mean",
]

EXCESS_FEATURES = [
    "excess_above_100_max", "excess_above_110_max",
    "excess_above_100_mean", "excess_above_110_mean",
]

EXTREME_BIN_LIMIT_FEATURES = [
    "bin_120_max", "bin_150_max", "bin_-100_max",
    "limit_mean", "limit_std",
]

FEATURE_SETS = {
    "A_baseline": BASELINE_FEATURES,
    "B_+tail": BASELINE_FEATURES + TAIL_PROB_FEATURES,
    "C_+excess": BASELINE_FEATURES + EXCESS_FEATURES,
    "D_+ext_lim": BASELINE_FEATURES + EXTREME_BIN_LIMIT_FEATURES,
    "E_all": BASELINE_FEATURES + TAIL_PROB_FEATURES + EXCESS_FEATURES + EXTREME_BIN_LIMIT_FEATURES,
}


# ── Core functions (from nb_experiment_v2.py) ──────────────────────────

def _minmax(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.full_like(arr, 0.5)
    return (arr - mn) / (mx - mn)


def compute_v0c(da, rt, bf):
    return 0.40 * (1.0 - _minmax(da)) + 0.30 * _minmax(rt) + 0.30 * _minmax(bf)


def assign_tiers_per_group(sp, groups):
    labels = np.zeros(len(sp), dtype=np.int32)
    offset = 0
    for g in groups:
        sl = slice(offset, offset + g)
        sp_g = sp[sl]
        pos = sp_g > 0
        if pos.sum() > 0:
            r = sp_g[pos].argsort().argsort()
            n = len(r)
            tier = np.where(r < n // 3, 1, np.where(r < 2 * n // 3, 2, 3))
            labels[sl][pos] = tier
        offset += g
    return labels


def allocate_reserved_slots(v0c_scores, nb_scores, is_dormant, n_v0c, n_nb):
    K = n_v0c + n_nb
    selected = set()
    v0c_order = np.argsort(v0c_scores)[::-1]
    for idx in v0c_order:
        if len(selected) >= n_v0c:
            break
        selected.add(int(idx))
    nb_filled = 0
    if n_nb > 0:
        candidates = [(i, nb_scores[i]) for i in range(len(v0c_scores))
                      if is_dormant[i] and i not in selected and np.isfinite(nb_scores[i])]
        candidates.sort(key=lambda x: -x[1])
        for idx, _ in candidates[:n_nb]:
            selected.add(idx)
            nb_filled += 1
    if len(selected) < K:
        for idx in v0c_order:
            if len(selected) >= K:
                break
            selected.add(int(idx))
    return selected, nb_filled


LGB_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 15, "learning_rate": 0.05,
    "min_child_samples": 5, "subsample": 0.8,
    "colsample_bytree": 0.8, "num_threads": 4, "verbose": -1,
}


def main():
    from pbase.config.ray import init_ray
    init_ray()

    t0 = time.time()
    all_pys = ["2021-06", "2022-06", "2023-06", "2024-06", "2025-06"]
    aqs = ["aq1", "aq2", "aq3"]

    # ── Phase 1: Build data with new density features ──────────────────
    print("Phase 1: Building data + density features...")
    from ml.features import build_model_table

    all_rows = []
    for py in all_pys:
        for aq in aqs:
            try:
                table = build_model_table(py, aq)
            except Exception as e:
                print(f"  {py}/{aq}: SKIP ({e})")
                continue

            # Compute new density features
            try:
                new_feats = compute_density_features(py, aq)
            except Exception as e:
                print(f"  {py}/{aq} density features: SKIP ({e})")
                new_feats = pl.DataFrame(schema={"branch_name": pl.Utf8})

            # Join new features to table
            if len(new_feats) > 0:
                table = table.join(new_feats, on="branch_name", how="left")
                # Fill nulls for new features
                for col in new_feats.columns:
                    if col != "branch_name" and col in table.columns:
                        table = table.with_columns(pl.col(col).fill_null(0.0))

            for r in table.iter_rows(named=True):
                b = r["branch_name"]
                row = {
                    "branch": b, "py": py, "aq": aq,
                    "sp_onpeak": r["onpeak_sp"],
                    "sp_offpeak": r["offpeak_sp"],
                    "sp_combined": r["realized_shadow_price"],
                    "bf_12": r["bf_12"], "bfo_12": r["bfo_12"],
                    "bf_combined_12": r["bf_combined_12"],
                    # Baseline features
                    "bin_80_max": r["bin_80_cid_max"],
                    "bin_90_max": r["bin_90_cid_max"],
                    "bin_100_max": r["bin_100_cid_max"],
                    "bin_110_max": r["bin_110_cid_max"],
                    "rt_max": max(r["bin_80_cid_max"], r["bin_90_cid_max"],
                                  r["bin_100_cid_max"], r["bin_110_cid_max"]),
                    "count_active_cids": r["count_active_cids"],
                    "shadow_price_da": r["shadow_price_da"],
                    "da_rank_value": r["da_rank_value"],
                    # New: tail probs
                    "prob_above_90_max": r.get("prob_above_90_max", 0.0),
                    "prob_above_100_max": r.get("prob_above_100_max", 0.0),
                    "prob_above_110_max": r.get("prob_above_110_max", 0.0),
                    "prob_above_90_mean": r.get("prob_above_90_mean", 0.0),
                    "prob_above_100_mean": r.get("prob_above_100_mean", 0.0),
                    "prob_above_110_mean": r.get("prob_above_110_mean", 0.0),
                    # New: expected excess
                    "excess_above_100_max": r.get("excess_above_100_max", 0.0),
                    "excess_above_110_max": r.get("excess_above_110_max", 0.0),
                    "excess_above_100_mean": r.get("excess_above_100_mean", 0.0),
                    "excess_above_110_mean": r.get("excess_above_110_mean", 0.0),
                    # New: extreme bins + limits
                    "bin_120_max": r.get("bin_120_cid_max", 0.0),
                    "bin_150_max": r.get("bin_150_cid_max", 0.0),
                    "bin_-100_max": r.get("bin_-100_cid_max", 0.0),
                    "limit_mean": r.get("limit_mean", 0.0) if r.get("limit_mean") is not None else 0.0,
                    "limit_std": r.get("limit_std", 0.0) if r.get("limit_std") is not None else 0.0,
                }
                all_rows.append(row)

    df = pl.DataFrame(all_rows)
    print(f"  {len(df)} rows, build time {time.time()-t0:.0f}s")

    # Sanity: check new features are nonzero
    for f in ["prob_above_100_max", "excess_above_100_max"]:
        nz = df.filter(pl.col(f) > 0).height
        print(f"  {f}: {nz}/{len(df)} nonzero ({100*nz/len(df):.1f}%)")

    # ── Phase 2: Ablation per feature set ──────────────────────────────
    print(f"\nPhase 2: Running ablation ({len(FEATURE_SETS)} feature sets × 2 ctypes × 3 eval years)...")

    eval_configs = [
        ("2023-06", ["2021-06", "2022-06"]),
        ("2024-06", ["2021-06", "2022-06", "2023-06"]),
        ("2025-06", ["2021-06", "2022-06", "2023-06", "2024-06"]),
    ]

    from ml.phase6.features import build_class_model_table
    from ml.config import CLASS_BF_COL

    results = []

    for fset_name, fset_cols in FEATURE_SETS.items():
        for eval_py, train_pys in eval_configs:
            for ct in ["onpeak", "offpeak"]:
                dormant_col = "bf_12" if ct == "onpeak" else "bfo_12"
                target_col = "sp_onpeak" if ct == "onpeak" else "sp_offpeak"

                # Train NB model
                train = df.filter(
                    pl.col("py").is_in(train_pys) & (pl.col(dormant_col) == 0)
                )
                if len(train) < 50:
                    continue

                X = train.select(fset_cols).to_numpy().astype(np.float64)
                groups = train.group_by(["py", "aq"], maintain_order=True).len()["len"].to_numpy()
                y = assign_tiers_per_group(train[target_col].to_numpy().astype(np.float64), groups)

                ds = lgb.Dataset(X, label=y, group=groups, free_raw_data=False)
                model = lgb.train(LGB_PARAMS, ds, num_boost_round=150)

                # Eval per quarter
                for aq in aqs:
                    try:
                        ct_table = build_class_model_table(eval_py, aq, ct)
                    except Exception:
                        continue

                    bf_col = CLASS_BF_COL[ct]
                    branches = ct_table["branch_name"].to_list()
                    N = len(ct_table)
                    sp = ct_table["realized_shadow_price"].to_numpy().astype(np.float64)
                    da = ct_table["da_rank_value"].to_numpy().astype(np.float64)
                    bf = ct_table[bf_col].to_numpy().astype(np.float64)
                    rt = ct_table.select(
                        pl.max_horizontal("bin_80_cid_max", "bin_90_cid_max",
                                          "bin_100_cid_max", "bin_110_cid_max")
                    ).to_series().to_numpy().astype(np.float64)
                    total_da = float(ct_table["total_da_sp_quarter"][0])
                    total_sp = sp.sum()
                    n_bind = int((sp > 0).sum())
                    is_dormant = bf == 0
                    is_nb_binder = is_dormant & (sp > 0)
                    n_nb_binder = int(is_nb_binder.sum())
                    total_nb_sp = float(sp[is_nb_binder].sum())

                    if n_bind == 0:
                        continue

                    # v0c (class-specific)
                    v0c = compute_v0c(da, rt, bf)

                    # NB model scores
                    nb_scores = np.full(N, -np.inf)
                    combined_q = df.filter((pl.col("py") == eval_py) & (pl.col("aq") == aq))
                    nb_br_map = {r["branch"]: i for i, r in enumerate(combined_q.iter_rows(named=True))}
                    dormant_idx = [i for i in range(N) if is_dormant[i]]
                    if dormant_idx:
                        feat_rows = []
                        valid_idx = []
                        for di in dormant_idx:
                            br = branches[di]
                            if br in nb_br_map:
                                row_i = nb_br_map[br]
                                feat_rows.append(combined_q[row_i].select(fset_cols))
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
                            results.append({
                                "fset": fset_name, "eval_py": eval_py, "aq": aq, "ct": ct,
                                "metric_type": "nb_only", "K": K,
                                "vc": float(nb_sp[mk].sum() / nb_total),
                                "rec": float((nb_sp[mk] > 0).sum() / nb_n_bind),
                                "nb_sp": float(nb_sp[mk].sum()),
                                "nb_bind": int((nb_sp[mk] > 0).sum()),
                            })

                    # Full universe: R30 config
                    for K, nv, nn in [(200, 170, 30), (400, 350, 50)]:
                        selected, nb_filled = allocate_reserved_slots(v0c, nb_scores, is_dormant, nv, nn)
                        mask = np.zeros(N, dtype=bool)
                        for idx in selected:
                            mask[idx] = True
                        results.append({
                            "fset": fset_name, "eval_py": eval_py, "aq": aq, "ct": ct,
                            "metric_type": f"R30@{K}", "K": K,
                            "vc": float(sp[mask].sum() / total_sp) if total_sp > 0 else 0,
                            "rec": float((sp[mask] > 0).sum() / n_bind),
                            "nb_sp": float(sp[mask & is_nb_binder].sum()),
                            "nb_bind": int((mask & is_nb_binder).sum()),
                        })

    # ── Phase 3: Report ────────────────────────────────────────────────
    rdf = pl.DataFrame(results)
    fset_order = list(FEATURE_SETS.keys())

    print(f"\n{'='*120}")
    print("ABLATION RESULTS")
    print("=" * 120)

    for metric_type in ["nb_only", "R30@200", "R30@400"]:
        for ct in ["onpeak", "offpeak"]:
            print(f"\n  === {metric_type} / {ct} ===")

            # Per year
            for eval_py in ["2024-06", "2025-06"]:
                rows = rdf.filter(
                    (pl.col("metric_type") == metric_type) & (pl.col("ct") == ct) & (pl.col("eval_py") == eval_py)
                )
                if len(rows) == 0:
                    continue
                K_val = int(rows["K"][0])
                print(f"\n    {eval_py} (K={K_val}):")
                print(f"    {'Feature Set':<14} {'VC':>7} {'Rec':>7} {'NB_SP':>10} {'NB_bind':>8}")
                print(f"    {'-'*50}")
                for fs in fset_order:
                    r = rows.filter(pl.col("fset") == fs)
                    if len(r) == 0:
                        continue
                    print(f"    {fs:<14} {r['vc'].mean():>7.4f} {r['rec'].mean():>7.4f} "
                          f"{r['nb_sp'].mean():>10,.0f} {r['nb_bind'].mean():>8.1f}")

            # Aggregate
            rows = rdf.filter(
                (pl.col("metric_type") == metric_type) & (pl.col("ct") == ct)
            )
            if len(rows) == 0:
                continue
            K_val = int(rows["K"][0])
            print(f"\n    AGGREGATE (K={K_val}):")
            print(f"    {'Feature Set':<14} {'Grp':>4} {'VC':>7} {'Rec':>7} {'NB_SP':>10} {'NB_bind':>8}")
            print(f"    {'-'*55}")
            for fs in fset_order:
                r = rows.filter(pl.col("fset") == fs)
                if len(r) == 0:
                    continue
                print(f"    {fs:<14} {len(r):>4} {r['vc'].mean():>7.4f} {r['rec'].mean():>7.4f} "
                      f"{r['nb_sp'].mean():>10,.0f} {r['nb_bind'].mean():>8.1f}")

    # Delta table: each feature set vs baseline
    print(f"\n{'='*120}")
    print("DELTA vs A_baseline (aggregate)")
    print("=" * 120)
    for metric_type in ["nb_only", "R30@200", "R30@400"]:
        for ct in ["onpeak", "offpeak"]:
            base = rdf.filter(
                (pl.col("metric_type") == metric_type) & (pl.col("ct") == ct) & (pl.col("fset") == "A_baseline")
            )
            if len(base) == 0:
                continue
            bvc = base["vc"].mean()
            bnb = base["nb_sp"].mean()
            K_val = int(base["K"][0])
            print(f"\n    {metric_type} / {ct} (K={K_val}, base VC={bvc:.4f}, NB_SP=${bnb:,.0f}):")
            print(f"    {'Feature Set':<14} {'dVC':>8} {'dNB_SP':>10}")
            print(f"    {'-'*35}")
            for fs in fset_order[1:]:
                r = rdf.filter(
                    (pl.col("metric_type") == metric_type) & (pl.col("ct") == ct) & (pl.col("fset") == fs)
                )
                if len(r) == 0:
                    continue
                print(f"    {fs:<14} {r['vc'].mean()-bvc:>+8.4f} ${r['nb_sp'].mean()-bnb:>+9,.0f}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
