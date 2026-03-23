"""NB-hist-12 Experiment V2: per-ctype models, class-specific v0c, V4.4 benchmark.

Two NB models (onpeak: bf_12==0, offpeak: bfo_12==0), each with per-ctype target.
Evaluation uses build_class_model_table for fully class-specific v0c.
V4.4 loaded per-ctype as benchmark scorer.
All reserved-slot configs maintain fixed K via v0c backfill.

Usage:
    source /home/xyz/workspace/pmodel/.venv/bin/activate
    RAY_ADDRESS=ray://10.8.0.36:10001 PYTHONPATH=. uv run python scripts/nb_experiment_v2.py
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("RAY_ADDRESS", "ray://10.8.0.36:10001")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lightgbm as lgb
import numpy as np
import polars as pl

V44_BASE = "/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V4.4.R1"

NB_FEATURES = [
    "bin_80_max", "bin_90_max", "bin_100_max", "bin_110_max",
    "rt_max", "count_active_cids", "shadow_price_da", "da_rank_value",
]

CONFIGS = [
    # (name, n_v0c@200, n_nb@200, n_v0c@400, n_nb@400, scorer_key)
    ("pure_v0c",  200,  0, 400,   0, None),
    ("pure_v44",    0,  0,   0,   0, "v44"),     # special: all by V4.4
    ("R30_nb",   170, 30, 350,  50, "nb"),
    ("R50_nb",   150, 50, 300, 100, "nb"),
    ("R30_v44",  170, 30, 350,  50, "v44_nb"),   # V4.4 as NB scorer
    ("R50_v44",  150, 50, 300, 100, "v44_nb"),   # V4.4 as NB scorer
]

LGB_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 15, "learning_rate": 0.05,
    "min_child_samples": 5, "subsample": 0.8,
    "colsample_bytree": 0.8, "num_threads": 4, "verbose": -1,
}


# ── Core functions (tested) ────────────────────────────────────────────

def _minmax(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.full_like(arr, 0.5)
    return (arr - mn) / (mx - mn)


def compute_v0c(da: np.ndarray, rt: np.ndarray, bf: np.ndarray) -> np.ndarray:
    return 0.40 * (1.0 - _minmax(da)) + 0.30 * _minmax(rt) + 0.30 * _minmax(bf)


def assign_tiers_per_group(sp: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Assign 0/1/2/3 tier labels within each LambdaRank group."""
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
    """Select n_v0c by v0c + n_nb by NB from dormant. Backfill with v0c if underfill.

    Returns (selected: set[int], nb_filled: int).
    """
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


def load_v44(py: str, aq: str, class_type: str) -> dict[str, dict]:
    """Load V4.4 signal keyed by equipment for the given class type."""
    path = f"{V44_BASE}/{py}/{aq}/{class_type}/"
    if not os.path.exists(path):
        return {}
    data = {}
    for r in pl.read_parquet(path).iter_rows(named=True):
        e = r.get("equipment", "")
        if e:
            data[e] = r
    return data


# ── Main ───────────────────────────────────────────────────────────────

def main():
    from pbase.config.ray import init_ray
    init_ray()

    t0 = time.time()
    all_pys = ["2021-06", "2022-06", "2023-06", "2024-06", "2025-06"]
    aqs = ["aq1", "aq2", "aq3"]

    # ── Phase 1: Build training data (combined builder) ────────────────
    print("Phase 1: Building training data (combined builder)...")
    from ml.features import build_model_table

    all_rows = []
    for py in all_pys:
        for aq in aqs:
            try:
                table = build_model_table(py, aq)
            except Exception as e:
                print(f"  {py}/{aq}: SKIP ({e})")
                continue
            for r in table.iter_rows(named=True):
                b = r["branch_name"]
                all_rows.append({
                    "branch": b, "py": py, "aq": aq,
                    "sp_onpeak": r["onpeak_sp"],
                    "sp_offpeak": r["offpeak_sp"],
                    "sp_combined": r["realized_shadow_price"],
                    "bf_12": r["bf_12"],
                    "bfo_12": r["bfo_12"],
                    "bf_combined_12": r["bf_combined_12"],
                    "bin_80_max": r["bin_80_cid_max"],
                    "bin_90_max": r["bin_90_cid_max"],
                    "bin_100_max": r["bin_100_cid_max"],
                    "bin_110_max": r["bin_110_cid_max"],
                    "rt_max": max(r["bin_80_cid_max"], r["bin_90_cid_max"],
                                  r["bin_100_cid_max"], r["bin_110_cid_max"]),
                    "count_active_cids": r["count_active_cids"],
                    "shadow_price_da": r["shadow_price_da"],
                    "da_rank_value": r["da_rank_value"],
                })
    df = pl.DataFrame(all_rows)

    # Invariant: per-ctype target sums to combined
    diff = (df["sp_onpeak"] + df["sp_offpeak"] - df["sp_combined"]).abs().max()
    assert diff < 0.01, f"Target invariant violated: max diff = {diff}"

    print(f"  {len(df)} rows, build time {time.time()-t0:.0f}s")

    # ── Phase 2: Train 2 NB models per eval year ──────────────────────
    print("\nPhase 2: Training per-ctype NB models...")
    eval_configs = [
        ("2023-06", ["2021-06", "2022-06"]),
        ("2024-06", ["2021-06", "2022-06", "2023-06"]),
        ("2025-06", ["2021-06", "2022-06", "2023-06", "2024-06"]),
    ]

    nb_models = {}
    for eval_py, train_pys in eval_configs:
        for ct in ["onpeak", "offpeak"]:
            dormant_col = "bf_12" if ct == "onpeak" else "bfo_12"
            target_col = "sp_onpeak" if ct == "onpeak" else "sp_offpeak"

            train = df.filter(
                pl.col("py").is_in(train_pys) & (pl.col(dormant_col) == 0)
            )
            if len(train) < 50:
                print(f"  {eval_py}/{ct}: only {len(train)} rows, skipping")
                continue

            X = train.select(NB_FEATURES).to_numpy().astype(np.float64)
            groups = train.group_by(["py", "aq"], maintain_order=True).len()["len"].to_numpy()
            y = assign_tiers_per_group(train[target_col].to_numpy().astype(np.float64), groups)

            ds = lgb.Dataset(X, label=y, group=groups, free_raw_data=False)
            model = lgb.train(LGB_PARAMS, ds, num_boost_round=150)
            nb_models[(eval_py, ct)] = model
            print(f"  {eval_py}/{ct}: {len(train)} train rows, {(y>0).sum()} binders")

    # ── Phase 3+4: Eval per (eval_py, aq, class_type) ─────────────────
    print("\nPhase 3+4: Per-ctype evaluation...")
    from ml.phase6.features import build_class_model_table
    from ml.config import CLASS_BF_COL

    combo_results = []
    nb_only_results = []
    case_studies = []

    for eval_py, train_pys in eval_configs:
        for aq in aqs:
            for ct in ["onpeak", "offpeak"]:
                # Build class-specific table
                try:
                    ct_table = build_class_model_table(eval_py, aq, ct)
                except Exception as e:
                    print(f"  {eval_py}/{aq}/{ct}: SKIP ({e})")
                    continue

                bf_col = CLASS_BF_COL[ct]
                branches = ct_table["branch_name"].to_list()
                N = len(ct_table)

                # Class-specific arrays from class builder
                sp = ct_table["realized_shadow_price"].to_numpy().astype(np.float64)
                da = ct_table["da_rank_value"].to_numpy().astype(np.float64)
                bf = ct_table[bf_col].to_numpy().astype(np.float64)
                rt = ct_table.select(
                    pl.max_horizontal("bin_80_cid_max", "bin_90_cid_max",
                                      "bin_100_cid_max", "bin_110_cid_max")
                ).to_series().to_numpy().astype(np.float64)
                label_tier = ct_table["label_tier"].to_numpy().astype(np.float64)

                assert ct_table["total_da_sp_quarter"].n_unique() == 1
                total_da = float(ct_table["total_da_sp_quarter"][0])
                total_sp = sp.sum()
                n_bind = int((sp > 0).sum())

                # Two masks: dormant vs dormant-binder
                is_dormant = bf == 0
                is_nb_binder = is_dormant & (sp > 0)
                n_dormant = int(is_dormant.sum())
                n_nb_binder = int(is_nb_binder.sum())
                total_nb_sp = float(sp[is_nb_binder].sum())

                if n_bind == 0:
                    continue

                # v0c (class-specific)
                v0c = compute_v0c(da, rt, bf)

                # NB model scores
                nb_scores = np.full(N, -np.inf)
                model_key = (eval_py, ct)
                if model_key in nb_models:
                    # Score ALL branches, model was trained on dormant but can score any
                    # We only USE scores for dormant branches in allocation
                    combined_row = df.filter(
                        (pl.col("py") == eval_py) & (pl.col("aq") == aq)
                    )
                    nb_br_map = {r["branch"]: i for i, r in enumerate(
                        combined_row.iter_rows(named=True)
                    )}
                    # Score dormant branches
                    dormant_idx = [i for i in range(N) if is_dormant[i]]
                    if dormant_idx:
                        dormant_branches = [branches[i] for i in dormant_idx]
                        # Get features from combined df for these branches
                        feat_rows = []
                        valid_dormant_idx = []
                        for di, br_name in zip(dormant_idx, dormant_branches):
                            if br_name in nb_br_map:
                                row_idx = nb_br_map[br_name]
                                feat_rows.append(combined_row[row_idx].select(NB_FEATURES))
                                valid_dormant_idx.append(di)
                        if feat_rows:
                            X_pred = pl.concat(feat_rows).to_numpy().astype(np.float64)
                            preds = nb_models[model_key].predict(X_pred)
                            for di, pred in zip(valid_dormant_idx, preds):
                                nb_scores[di] = pred

                # V4.4 scores (per-ctype)
                v44_data = load_v44(eval_py, aq, ct)
                v44_scores = np.full(N, -np.inf)
                for i, b in enumerate(branches):
                    if b in v44_data:
                        v44_scores[i] = 1.0 - v44_data[b].get("rank", 1.0)

                # ── NB-only metrics ────────────────────────────────────
                nb_sp = sp[is_dormant]
                nb_total = nb_sp.sum()
                nb_n_bind_local = int((nb_sp > 0).sum())
                if nb_n_bind_local > 0 and nb_total > 0:
                    for sname, sc_full in [("ML_nb", nb_scores), ("V4.4", v44_scores), ("v0c", v0c)]:
                        sc = sc_full[is_dormant]
                        for K in [30, 50, 100]:
                            topk = np.argsort(sc)[::-1][:min(K, len(sc))]
                            mk = np.zeros(len(sc), dtype=bool)
                            mk[topk] = True
                            nb_only_results.append({
                                "eval_py": eval_py, "aq": aq, "ct": ct,
                                "scorer": sname, "K": K,
                                "vc": float(nb_sp[mk].sum() / nb_total),
                                "rec": float((nb_sp[mk] > 0).sum() / nb_n_bind_local),
                                "sp_captured": float(nb_sp[mk].sum()),
                                "bind_captured": int((nb_sp[mk] > 0).sum()),
                                "n_dormant": n_dormant, "n_nb_binder": nb_n_bind_local,
                            })

                # ── Full universe configs ──────────────────────────────
                for cname, nv200, nn200, nv400, nn400, scorer_key in CONFIGS:
                    for K, nv, nn in [(200, nv200, nn200), (400, nv400, nn400)]:
                        if cname == "pure_v44":
                            # All K by V4.4
                            order = np.argsort(v44_scores)[::-1][:K]
                            selected = set(int(x) for x in order)
                            # Backfill if V4.4 doesn't cover K branches
                            if len(selected) < K:
                                for idx in np.argsort(v0c)[::-1]:
                                    if len(selected) >= K:
                                        break
                                    selected.add(int(idx))
                            nb_filled = 0
                        elif scorer_key is None:
                            # pure_v0c
                            selected = set(int(x) for x in np.argsort(v0c)[::-1][:K])
                            nb_filled = 0
                        else:
                            # Reserved-slot configs
                            if scorer_key == "nb":
                                scorer = nb_scores
                            else:  # v44_nb
                                scorer = v44_scores
                            selected, nb_filled = allocate_reserved_slots(
                                v0c, scorer, is_dormant, nv, nn
                            )

                        assert len(selected) == K, f"{cname} K={K}: got {len(selected)}"

                        mask = np.zeros(N, dtype=bool)
                        for idx in selected:
                            mask[idx] = True

                        # Dangerous
                        dang20 = sp > 20000
                        dang50 = sp > 50000

                        # NDCG
                        from sklearn.metrics import ndcg_score as _ndcg
                        try:
                            ndcg = float(_ndcg(
                                label_tier[mask].reshape(1, -1),
                                sp[mask].reshape(1, -1),
                            )) if mask.sum() > 0 else 0.0
                        except Exception:
                            ndcg = 0.0

                        combo_results.append({
                            "eval_py": eval_py, "aq": aq, "ct": ct,
                            "config": cname, "K": K,
                            "vc": float(sp[mask].sum() / total_sp) if total_sp > 0 else 0,
                            "abs_sp": float(sp[mask].sum() / total_da) if total_da > 0 else 0,
                            "rec": float((sp[mask] > 0).sum() / n_bind),
                            "prec": float((sp[mask] > 0).sum() / K),
                            "bind": int((sp[mask] > 0).sum()),
                            "sp_cap": float(sp[mask].sum()),
                            "nb_in": int((mask & is_dormant).sum()),
                            "nb_bind": int((mask & is_nb_binder).sum()),
                            "nb_sp": float(sp[mask & is_nb_binder].sum()),
                            "nb_vc": float(sp[mask & is_nb_binder].sum() / total_nb_sp) if total_nb_sp > 0 else 0,
                            "nb_rec": float((mask & is_nb_binder).sum() / n_nb_binder) if n_nb_binder > 0 else 0,
                            "dang20_cap": int((mask & dang20).sum()),
                            "dang20_tot": int(dang20.sum()),
                            "dang50_cap": int((mask & dang50).sum()),
                            "dang50_tot": int(dang50.sum()),
                            "nb_filled": nb_filled,
                            "nb_requested": nn,
                            "ndcg": ndcg,
                        })

                # ── Case studies ───────────────────────────────────────
                dang_threshold = 20000
                dang_idx = [i for i in range(N) if sp[i] > dang_threshold]
                if dang_idx:
                    v0c_rank = np.argsort(np.argsort(v0c)[::-1]) + 1
                    v44_rank_arr = np.full(N, N, dtype=int)
                    v44_finite = np.isfinite(v44_scores)
                    if v44_finite.sum() > 0:
                        fi = np.where(v44_finite)[0]
                        order = np.argsort(v44_scores[v44_finite])[::-1]
                        for rk, oi in enumerate(order):
                            v44_rank_arr[fi[oi]] = rk + 1
                    nb_rank_arr = np.full(N, N, dtype=int)
                    nb_finite = np.isfinite(nb_scores)
                    if nb_finite.sum() > 0:
                        fi = np.where(nb_finite)[0]
                        order = np.argsort(nb_scores[nb_finite])[::-1]
                        for rk, oi in enumerate(order):
                            nb_rank_arr[fi[oi]] = rk + 1

                    for i in sorted(dang_idx, key=lambda x: -sp[x]):
                        case_studies.append({
                            "eval_py": eval_py, "aq": aq, "ct": ct,
                            "branch": branches[i], "sp": float(sp[i]),
                            "is_dormant": bool(is_dormant[i]),
                            "v0c_rk": int(v0c_rank[i]),
                            "v44_rk": int(v44_rank_arr[i]),
                            "nb_rk": int(nb_rank_arr[i]),
                            "N": N,
                        })

    # ── Phase 5: Reporting ─────────────────────────────────────────────
    combo_df = pl.DataFrame(combo_results)
    nb_df = pl.DataFrame(nb_only_results) if nb_only_results else None
    cs_df = pl.DataFrame(case_studies) if case_studies else None

    config_order = ["pure_v0c", "pure_v44", "R30_nb", "R30_v44", "R50_nb", "R50_v44"]

    def print_combo_table(df_slice, title):
        print(f"\n  === {title} ===")
        hdr = (f"    {'Config':<12} {'VC':>6} {'Abs':>6} {'Rec':>6} {'Prec':>5} {'Bind':>5} {'NDCG':>6}"
               f" {'NB_in':>5} {'NB_b':>4} {'NB_SP':>9} {'NB_VC':>6} {'NB_R':>5}"
               f" {'D20':>7} {'D50':>5} {'Fill':>7}")
        print(hdr)
        print(f"    {'-'*115}")
        for c in config_order:
            r = df_slice.filter(pl.col("config") == c)
            if len(r) == 0:
                continue
            d20 = f"{r['dang20_cap'].mean():.0f}/{r['dang20_tot'].mean():.0f}"
            d50 = f"{r['dang50_cap'].mean():.0f}/{r['dang50_tot'].mean():.0f}"
            fill = f"{r['nb_filled'].mean():.0f}/{r['nb_requested'].mean():.0f}" if r['nb_requested'].mean() > 0 else "-"
            print(f"    {c:<12} {r['vc'].mean():>6.3f} {r['abs_sp'].mean():>6.3f} "
                  f"{r['rec'].mean():>6.3f} {r['prec'].mean():>5.3f} {r['bind'].mean():>5.0f} "
                  f"{r['ndcg'].mean():>6.3f}"
                  f" {r['nb_in'].mean():>5.0f} {r['nb_bind'].mean():>4.1f} {r['nb_sp'].mean():>9,.0f}"
                  f" {r['nb_vc'].mean():>6.3f} {r['nb_rec'].mean():>5.3f}"
                  f" {d20:>7} {d50:>5} {fill:>7}")

    # Part 1: NB-only
    if nb_df is not None:
        print("=" * 120)
        print("PART 1: NB-only metrics (per-ctype dormant universe)")
        print("=" * 120)
        scorers = ["ML_nb", "V4.4", "v0c"]
        for ct in ["onpeak", "offpeak"]:
            for eval_py in ["2024-06", "2025-06"]:
                rows = nb_df.filter((pl.col("ct") == ct) & (pl.col("eval_py") == eval_py))
                if len(rows) == 0:
                    continue
                print(f"\n  {eval_py} / {ct}:")
                for K in [30, 50, 100]:
                    print(f"    K={K}: ", end="")
                    for s in scorers:
                        r = rows.filter((pl.col("scorer") == s) & (pl.col("K") == K))
                        if len(r) == 0:
                            continue
                        print(f"  {s}: VC={r['vc'].mean():.3f} Rec={r['rec'].mean():.3f} Bind={r['bind_captured'].mean():.1f}", end="")
                    print()

    # Part 2: 8 year-ctype-K tables
    print(f"\n{'='*120}")
    print("PART 2: Full universe — (year × ctype × K)")
    print("=" * 120)
    for eval_py in ["2024-06", "2025-06"]:
        for ct in ["onpeak", "offpeak"]:
            for K in [200, 400]:
                tier = "Tier 0" if K == 200 else "Tier 0+1"
                sl = combo_df.filter(
                    (pl.col("eval_py") == eval_py) & (pl.col("ct") == ct) & (pl.col("K") == K)
                )
                if len(sl) == 0:
                    continue
                print_combo_table(sl, f"{eval_py} / {ct} / K={K} ({tier})")

    # Part 3: 4 aggregate tables
    print(f"\n{'='*120}")
    print("PART 3: Aggregate (ctype × K)")
    print("=" * 120)
    for ct in ["onpeak", "offpeak"]:
        for K in [200, 400]:
            sl = combo_df.filter((pl.col("ct") == ct) & (pl.col("K") == K))
            if len(sl) == 0:
                continue
            n_groups = len(sl) // len(config_order) if len(config_order) > 0 else 0
            print_combo_table(sl, f"AGG / {ct} / K={K} ({n_groups} quarter-groups)")

    # Part 4: Delta vs pure_v0c
    print(f"\n{'='*120}")
    print("PART 4: Delta vs pure_v0c")
    print("=" * 120)
    for ct in ["onpeak", "offpeak"]:
        for K in [200, 400]:
            base = combo_df.filter(
                (pl.col("ct") == ct) & (pl.col("K") == K) & (pl.col("config") == "pure_v0c")
            )
            if len(base) == 0:
                continue
            bvc = base["vc"].mean()
            bnb = base["nb_sp"].mean()
            print(f"\n    {ct} K={K} (base VC={bvc:.4f}, NB_SP=${bnb:,.0f}):")
            print(f"    {'Config':<12} {'dVC':>8} {'dNB_SP':>10} {'Fill':>7}")
            print(f"    {'-'*40}")
            for c in config_order[1:]:
                r = combo_df.filter(
                    (pl.col("ct") == ct) & (pl.col("K") == K) & (pl.col("config") == c)
                )
                if len(r) == 0:
                    continue
                fill = f"{r['nb_filled'].mean():.0f}/{r['nb_requested'].mean():.0f}" if r['nb_requested'].mean() > 0 else "-"
                print(f"    {c:<12} {r['vc'].mean()-bvc:>+8.4f} "
                      f"${r['nb_sp'].mean()-bnb:>+9,.0f} {fill:>7}")

    # Part 5: Case studies
    if cs_df is not None and len(cs_df) > 0:
        print(f"\n{'='*120}")
        print("PART 5: Case studies (SP > $20K, per-ctype)")
        print("=" * 120)
        for eval_py in ["2024-06", "2025-06"]:
            for ct in ["onpeak", "offpeak"]:
                rows = cs_df.filter((pl.col("eval_py") == eval_py) & (pl.col("ct") == ct))
                if len(rows) == 0:
                    continue
                print(f"\n  {eval_py} / {ct} ({len(rows)} branches):")
                print(f"  {'Branch':<35} {'SP':>10} {'Dorm':>5} {'v0c':>5} {'v44':>5} {'ML':>5} {'N':>5}")
                print(f"  {'-'*75}")
                for r in rows.sort("sp", descending=True).head(20).iter_rows(named=True):
                    d = "NB" if r["is_dormant"] else ""
                    print(f"  {r['branch']:<35} ${r['sp']:>9,.0f} {d:>5} "
                          f"{r['v0c_rk']:>5} {r['v44_rk']:>5} {r['nb_rk']:>5} {r['N']:>5}")

    # ── Phase 6: Registry save ─────────────────────────────────────────
    print("\nSaving registry artifacts...")
    for ct in ["onpeak", "offpeak"]:
        path = f"registry/{ct}/nb_v2"
        os.makedirs(path, exist_ok=True)

        config_obj = {
            "version": "nb_v2",
            "class_type": ct,
            "dormant_col": "bf_12" if ct == "onpeak" else "bfo_12",
            "target_col": "onpeak_sp" if ct == "onpeak" else "offpeak_sp",
            "features": NB_FEATURES,
            "lgb_params": LGB_PARAMS,
            "eval_configs": {ep: tps for ep, tps in eval_configs},
            "configs": [(c[0], c[1], c[2], c[3], c[4]) for c in CONFIGS],
        }
        with open(f"{path}/config.json", "w") as f:
            json.dump(config_obj, f, indent=2)

        ct_metrics = combo_df.filter(pl.col("ct") == ct).to_dicts()
        with open(f"{path}/metrics.json", "w") as f:
            json.dump(ct_metrics, f, indent=2)

        print(f"  {path}/ saved ({len(ct_metrics)} entries)")

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
