"""Champion confirmation: 3-way v0c vs Bucket_6_20 vs V4.4.

Reports per (eval_year x ctype x K) grain:
  - Native top-K: SP, Binders, Prec, VC, Abs_SP, Rec, NB metrics, D20, D50
  - V4.4 label coverage per cell
  - Deployment eval: R30/R50 reserved-slot policies (v0c + Bucket_6_20 dormant)

Usage:
    source /home/xyz/workspace/pmodel/.venv/bin/activate
    PYTHONPATH=. uv run python scripts/champion_confirmation.py
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

# ── Paths ──────────────────────────────────────────────────────────────
V44_BASE = "/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V4.4.R1"
CACHE_DIR = "data/nb_cache"

# ── Bucket_6_20 config ─────────────────────────────────────────────────
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

BUCKET_BOUNDS = [0, 200, 5000, 20000]
BUCKET_WEIGHTS = {0: 1, 1: 1, 2: 2, 3: 6, 4: 20}

# ── Eval config ────────────────────────────────────────────────────────
ALL_PYS = ["2018-06", "2019-06", "2020-06", "2021-06", "2022-06", "2023-06", "2024-06", "2025-06"]
AQS = ["aq1", "aq2", "aq3"]
EVAL_CONFIGS = [
    ("2022-06", [py for py in ALL_PYS if py < "2022-06"]),
    ("2023-06", [py for py in ALL_PYS if py < "2023-06"]),
    ("2024-06", [py for py in ALL_PYS if py < "2024-06"]),
    ("2025-06", [py for py in ALL_PYS if py < "2025-06"]),
]

K_LEVELS = [200, 400]

# Deployment reserved-slot configs: (name, n_v0c_200, n_bkt_200, n_v0c_400, n_bkt_400)
DEPLOY_CONFIGS = [
    ("R30", 170, 30, 350, 50),
    ("R50", 150, 50, 300, 100),
]


# ── Helpers ────────────────────────────────────────────────────────────

def _minmax(arr):
    mn, mx = arr.min(), arr.max()
    return np.full_like(arr, 0.5) if mx == mn else (arr - mn) / (mx - mn)


def compute_v0c(da, rt, bf):
    return 0.40 * (1.0 - _minmax(da)) + 0.30 * _minmax(rt) + 0.30 * _minmax(bf)


def assign_bucket_labels(sp):
    labels = np.zeros(len(sp), dtype=np.int32)
    labels[sp > 0] = 1
    labels[sp > BUCKET_BOUNDS[1]] = 2
    labels[sp > BUCKET_BOUNDS[2]] = 3
    labels[sp > BUCKET_BOUNDS[3]] = 4
    return labels


def assign_bucket_weights(labels):
    return np.array([BUCKET_WEIGHTS[int(l)] for l in labels])


def allocate_reserved_slots(v0c_scores, bkt_scores, is_dormant, n_v0c, n_bkt):
    """Select n_v0c by v0c rank + n_bkt dormant branches by Bucket score."""
    K = n_v0c + n_bkt
    selected = set()
    v0c_order = np.argsort(v0c_scores)[::-1]
    for idx in v0c_order:
        if len(selected) >= n_v0c:
            break
        selected.add(int(idx))
    bkt_filled = 0
    if n_bkt > 0:
        candidates = [
            (i, bkt_scores[i]) for i in range(len(v0c_scores))
            if is_dormant[i] and i not in selected and np.isfinite(bkt_scores[i])
        ]
        candidates.sort(key=lambda x: -x[1])
        for idx, _ in candidates[:n_bkt]:
            selected.add(idx)
            bkt_filled += 1
    # backfill with v0c if not enough dormant
    if len(selected) < K:
        for idx in v0c_order:
            if len(selected) >= K:
                break
            selected.add(int(idx))
    return selected, bkt_filled


# ── Data loading ───────────────────────────────────────────────────────

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
                cols = [
                    "branch", "py", "aq", "sp", "bf",
                    "bin_80_max", "bin_90_max", "bin_100_max", "bin_110_max",
                    "rt_max", "count_active_cids", "shadow_price_da", "da_rank_value",
                    "top2_bin_80", "top2_bin_90", "top2_bin_100", "top2_bin_110",
                ]
                frames[ct].append(t.select([c for c in cols if c in t.columns]))

    df_class = {ct: pl.concat(fs, how="diagonal") for ct, fs in frames.items()}
    for ct in df_class:
        for c in ["top2_bin_80", "top2_bin_90", "top2_bin_100", "top2_bin_110"]:
            if c in df_class[ct].columns:
                df_class[ct] = df_class[ct].with_columns(pl.col(c).fill_null(0.0))
    return df_class


def load_v44(eval_py, aq, ct):
    """Load V4.4 signal, return (ranked_branches, universe_size)."""
    v44p = f"{V44_BASE}/{eval_py}/{aq}/{ct}/"
    if not os.path.exists(v44p):
        return [], 0
    v44_df = pl.read_parquet(v44p).filter(pl.col("equipment") != "").sort("rank")
    return v44_df["equipment"].to_list(), len(v44_df)


# ── Metric computation ─────────────────────────────────────────────────

def compute_cell_metrics(sp, idx_arr, is_dormant, is_nb_binder, K, universe_size):
    """Compute metrics for a set of selected indices."""
    total_sp = sp.sum()
    n_bind = int((sp > 0).sum())
    n_d20 = int((sp > 20000).sum())
    n_d50 = int((sp > 50000).sum())

    mask = np.zeros(len(sp), dtype=bool)
    idx_bounded = idx_arr[idx_arr < len(sp)]
    mask[idx_bounded] = True
    actual_k = int(mask.sum())

    captured = float(sp[mask].sum())
    binders = int((sp[mask] > 0).sum())
    nb_in = int((mask & is_dormant).sum())
    nb_b = int((mask & is_nb_binder).sum())
    nb_sp = float(sp[mask & is_nb_binder].sum())
    d20 = int((sp[mask] > 20000).sum())
    d50 = int((sp[mask] > 50000).sum())

    return {
        "K": K, "actual_k": actual_k, "universe": universe_size,
        "sp": captured, "binders": binders,
        "prec": binders / actual_k if actual_k > 0 else 0.0,
        "vc": captured / total_sp if total_sp > 0 else 0.0,
        "rec": binders / n_bind if n_bind > 0 else 0.0,
        "nb_in": nb_in, "nb_b": nb_b, "nb_sp": nb_sp,
        "d20": d20, "d20_tot": n_d20, "d50": d50, "d50_tot": n_d50,
    }


def compute_v44_cell_metrics(v44_topk, branches, sp, is_dormant, is_nb_binder, K):
    """Compute metrics for V4.4 picks projected onto our GT."""
    N = len(sp)
    total_sp = sp.sum()
    n_bind = int((sp > 0).sum())
    n_d20 = int((sp > 20000).sum())
    n_d50 = int((sp > 50000).sum())

    branch_to_idx = {}
    for i, b in enumerate(branches):
        if b not in branch_to_idx:
            branch_to_idx[b] = i

    mask = np.zeros(N, dtype=bool)
    labeled = 0
    unlabeled = 0
    for b in v44_topk:
        if b in branch_to_idx:
            mask[branch_to_idx[b]] = True
            labeled += 1
        else:
            unlabeled += 1

    actual_k = int(mask.sum())
    captured = float(sp[mask].sum())
    binders = int((sp[mask] > 0).sum())
    nb_in = int((mask & is_dormant).sum())
    nb_b = int((mask & is_nb_binder).sum())
    nb_sp = float(sp[mask & is_nb_binder].sum())
    d20 = int((sp[mask] > 20000).sum())
    d50 = int((sp[mask] > 50000).sum())

    return {
        "K": K, "actual_k": actual_k, "universe": -1,  # external
        "sp": captured, "binders": binders,
        "prec": binders / K if K > 0 else 0.0,
        "vc": captured / total_sp if total_sp > 0 else 0.0,
        "rec": binders / n_bind if n_bind > 0 else 0.0,
        "nb_in": nb_in, "nb_b": nb_b, "nb_sp": nb_sp,
        "d20": d20, "d20_tot": n_d20, "d50": d50, "d50_tot": n_d50,
        "labeled": labeled, "unlabeled": unlabeled,
    }


# ── Printing ───────────────────────────────────────────────────────────

HEADER = (
    f"  {'Model':<22} {'Univ':>5} {'SP':>12} {'Bind':>5} {'Prec':>5} {'VC':>6} "
    f"{'Rec':>6} {'NB_in':>5} {'NB_b':>4} {'NB_SP':>10} {'D20':>7} {'D50':>5} {'LblCov':>7}"
)
SEP = f"  {'-'*120}"


def _fmt_row(name, mrs, K):
    def avg(key):
        vals = [r[key] for r in mrs if key in r]
        return sum(vals) / len(vals) if vals else 0

    d20s = f"{avg('d20'):.0f}/{avg('d20_tot'):.0f}"
    d50s = f"{avg('d50'):.0f}/{avg('d50_tot'):.0f}"
    unlbl = avg("unlabeled") if any("unlabeled" in r for r in mrs) else 0
    lbl = f"{avg('actual_k'):.0f}/{K}" + (f" ({unlbl:.0f}?)" if unlbl > 0 else "")

    return (
        f"  {name:<22} {avg('universe'):>5.0f} ${avg('sp'):>11,.0f} {avg('binders'):>5.0f} "
        f"{avg('prec'):>5.3f} {avg('vc'):>6.3f} "
        f"{avg('rec'):>6.3f} {avg('nb_in'):>5.0f} {avg('nb_b'):>4.0f} ${avg('nb_sp'):>9,.0f} "
        f"{d20s:>7} {d50s:>5} {lbl:>7}"
    )


def print_section(title, model_results, K, aqs_used):
    """Print one comparison section."""
    print(f"\n  {title} — averaged across {len(aqs_used)} quarters")
    print(HEADER)
    print(SEP)
    for model_name, rows in model_results.items():
        filtered = [r for r in rows if r["K"] == K]
        if filtered:
            print(_fmt_row(model_name, filtered, K))


# ── Main ───────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    df_class = load_cached_data()
    print(f"Data loaded: {time.time()-t0:.1f}s")

    all_results = []  # flat list for registry

    for eval_py, train_pys in EVAL_CONFIGS:
        for ct in ["onpeak", "offpeak"]:
            # ── Train Bucket_6_20 ──
            feats = [f for f in UNIFIED_FEATURES if f in df_class[ct].columns]
            train = df_class[ct].filter(pl.col("py").is_in(train_pys))
            X_train = train.select(feats).to_numpy().astype(np.float64)
            groups = train.group_by(["py", "aq"], maintain_order=True).len()["len"].to_numpy()
            sp_t = train["sp"].to_numpy().astype(np.float64)
            labels = assign_bucket_labels(sp_t)
            weights = assign_bucket_weights(labels)
            ds = lgb.Dataset(X_train, label=labels, group=groups, weight=weights,
                             feature_name=feats, free_raw_data=False)
            bkt_model = lgb.train(LGB_PARAMS, ds, num_boost_round=150)

            # ── Collect per-quarter metrics ──
            native_results = {"v0c": [], "Bucket_6_20": [], "V4.4": []}
            deploy_results = {name: [] for name, *_ in DEPLOY_CONFIGS}

            print(f"\n{'='*130}")
            print(f"  {eval_py} / {ct}  (train on {len(train_pys)} PYs: {train_pys[0]}..{train_pys[-1]})")
            print(f"{'='*130}")

            for aq in AQS:
                eq = df_class[ct].filter(
                    (pl.col("py") == eval_py) & (pl.col("aq") == aq)
                )
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

                # v0c scores
                v0c_scores = compute_v0c(da, rt, bf)

                # Bucket_6_20 scores
                bkt_scores = bkt_model.predict(
                    eq.select(feats).to_numpy().astype(np.float64)
                )

                # V4.4 data
                v44_branches, _v44_univ = load_v44(eval_py, aq, ct)

                for K in K_LEVELS:
                    # v0c native
                    v0c_idx = np.argsort(v0c_scores)[::-1][:K]
                    m_v0c = compute_cell_metrics(sp, v0c_idx, is_dormant, is_nb_binder, K, N)
                    m_v0c.update(eval_py=eval_py, aq=aq, ct=ct, model="v0c")
                    native_results["v0c"].append(m_v0c)

                    # Bucket_6_20 native
                    bkt_idx = np.argsort(bkt_scores)[::-1][:K]
                    m_bkt = compute_cell_metrics(sp, bkt_idx, is_dormant, is_nb_binder, K, N)
                    m_bkt.update(eval_py=eval_py, aq=aq, ct=ct, model="Bucket_6_20")
                    native_results["Bucket_6_20"].append(m_bkt)

                    # V4.4 native
                    v44_topk = v44_branches[:K]
                    m_v44 = compute_v44_cell_metrics(
                        v44_topk, branches, sp, is_dormant, is_nb_binder, K
                    )
                    m_v44.update(eval_py=eval_py, aq=aq, ct=ct, model="V4.4")
                    native_results["V4.4"].append(m_v44)

                # ── Deployment eval: reserved slots ──
                for dname, nv200, nb200, nv400, nb400 in DEPLOY_CONFIGS:
                    for K, nv, nb in [(200, nv200, nb200), (400, nv400, nb400)]:
                        sel, filled = allocate_reserved_slots(
                            v0c_scores, bkt_scores, is_dormant, nv, nb
                        )
                        sel_arr = np.array(sorted(sel))
                        m_dep = compute_cell_metrics(
                            sp, sel_arr, is_dormant, is_nb_binder, K, N
                        )
                        m_dep.update(
                            eval_py=eval_py, aq=aq, ct=ct, model=dname,
                            nb_filled=filled, nb_requested=nb,
                        )
                        deploy_results[dname].append(m_dep)

            # ── Print native comparison ──
            for K in K_LEVELS:
                tier = "Tier 0" if K == 200 else "Tier 0+1"
                print_section(f"Native K={K} ({tier})", native_results, K, AQS)

            # ── Print deployment comparison ──
            for dname, *_ in DEPLOY_CONFIGS:
                for K in K_LEVELS:
                    tier = "Tier 0" if K == 200 else "Tier 0+1"
                    combined = {"v0c": native_results["v0c"], dname: deploy_results[dname]}
                    print_section(f"Deploy {dname} K={K} ({tier})", combined, K, AQS)

            # ── Per-quarter detail table ──
            print(f"\n  Per-quarter detail (K=200):")
            print(f"  {'AQ':<5} {'Model':<16} {'SP':>12} {'Bind':>5} {'VC':>6} {'NB_b':>4} {'NB_SP':>10} {'D20':>4} {'LblCov':>7}")
            print(f"  {'-'*80}")
            for aq in AQS:
                for mname in ["v0c", "Bucket_6_20", "V4.4"]:
                    rows = [r for r in native_results[mname] if r.get("aq") == aq and r["K"] == 200]
                    if not rows:
                        continue
                    r = rows[0]
                    unlbl = r.get("unlabeled", 0)
                    lbl = f"{r['actual_k']}/200" + (f" ({unlbl}?)" if unlbl > 0 else "")
                    print(
                        f"  {aq:<5} {mname:<16} ${r['sp']:>11,.0f} {r['binders']:>5} "
                        f"{r['vc']:>6.3f} {r['nb_b']:>4} ${r['nb_sp']:>9,.0f} "
                        f"{r['d20']:>4} {lbl:>7}"
                    )

            # Collect for registry
            for mname, rows in native_results.items():
                all_results.extend(rows)
            for dname, rows in deploy_results.items():
                all_results.extend(rows)

    # ── Summary: year-level SP wins ──
    print(f"\n{'='*130}")
    print("  SUMMARY: SP wins by year × ctype × K (native)")
    print(f"{'='*130}")
    print(f"  {'Year':<8} {'CType':<10} {'K':>4}  {'v0c SP':>12} {'Bkt SP':>12} {'V44 SP':>12}  {'Winner':<12} {'Delta':>10}")
    print(f"  {'-'*85}")

    winner_counts = {"v0c": 0, "Bucket_6_20": 0, "V4.4": 0}
    for eval_py, _ in EVAL_CONFIGS:
        for ct in ["onpeak", "offpeak"]:
            for K in K_LEVELS:
                def _avg_sp(model):
                    rows = [r for r in all_results
                            if r.get("eval_py") == eval_py and r.get("ct") == ct
                            and r.get("model") == model and r["K"] == K
                            and r.get("aq") in AQS]
                    return sum(r["sp"] for r in rows) / len(rows) if rows else 0

                v0c_sp = _avg_sp("v0c")
                bkt_sp = _avg_sp("Bucket_6_20")
                v44_sp = _avg_sp("V4.4")

                sps = {"v0c": v0c_sp, "Bucket_6_20": bkt_sp, "V4.4": v44_sp}
                winner = max(sps, key=lambda k: sps[k])
                second = sorted(sps.values(), reverse=True)[1]
                delta = sps[winner] - second
                winner_counts[winner] += 1

                print(
                    f"  {eval_py:<8} {ct:<10} {K:>4}  "
                    f"${v0c_sp:>11,.0f} ${bkt_sp:>11,.0f} ${v44_sp:>11,.0f}  "
                    f"{winner:<12} ${delta:>+9,.0f}"
                )

    print(f"\n  Win counts: {winner_counts}")

    # ── Summary: NB_SP wins ──
    print(f"\n  NB_SP wins by year × ctype × K (native):")
    print(f"  {'Year':<8} {'CType':<10} {'K':>4}  {'v0c NB$':>10} {'Bkt NB$':>10} {'V44 NB$':>10}  {'Winner':<12}")
    print(f"  {'-'*75}")

    nb_winner_counts = {"v0c": 0, "Bucket_6_20": 0, "V4.4": 0}
    for eval_py, _ in EVAL_CONFIGS:
        for ct in ["onpeak", "offpeak"]:
            for K in K_LEVELS:
                def _avg_nb(model):
                    rows = [r for r in all_results
                            if r.get("eval_py") == eval_py and r.get("ct") == ct
                            and r.get("model") == model and r["K"] == K
                            and r.get("aq") in AQS]
                    return sum(r["nb_sp"] for r in rows) / len(rows) if rows else 0

                v0c_nb = _avg_nb("v0c")
                bkt_nb = _avg_nb("Bucket_6_20")
                v44_nb = _avg_nb("V4.4")

                nbs = {"v0c": v0c_nb, "Bucket_6_20": bkt_nb, "V4.4": v44_nb}
                winner = max(nbs, key=lambda k: nbs[k])
                nb_winner_counts[winner] += 1

                print(
                    f"  {eval_py:<8} {ct:<10} {K:>4}  "
                    f"${v0c_nb:>9,.0f} ${bkt_nb:>9,.0f} ${v44_nb:>9,.0f}  "
                    f"{winner:<12}"
                )

    print(f"\n  NB win counts: {nb_winner_counts}")

    # ── Save registry ──
    print("\nSaving registry...")
    reg_path = "registry/champion_confirmation"
    os.makedirs(reg_path, exist_ok=True)

    # Save full results
    with open(f"{reg_path}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save config
    config = {
        "models": ["v0c", "Bucket_6_20", "V4.4"],
        "deploy_configs": {name: {"200": (nv2, nb2), "400": (nv4, nb4)}
                          for name, nv2, nb2, nv4, nb4 in DEPLOY_CONFIGS},
        "bucket_features": UNIFIED_FEATURES,
        "bucket_params": LGB_PARAMS,
        "bucket_bounds": BUCKET_BOUNDS,
        "bucket_weights": BUCKET_WEIGHTS,
        "eval_pys": [e[0] for e in EVAL_CONFIGS],
        "k_levels": K_LEVELS,
    }
    with open(f"{reg_path}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Saved to {reg_path}/")
    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
