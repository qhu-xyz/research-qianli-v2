"""7.2b full-grid candidate generation — branch-level parity + aggregate metrics.

Produces:
  1. registry/7.2b/results.parquet — aggregate metrics (target: 1,380 rows)
  2. registry/7.2b/branch_level.parquet — per-branch ranks (target: 368,808 rows)
  3. registry/7.2b/parity_summary.json — comparison vs research reference

Research reference artifacts:
  /opt/tmp/qianli/miso/trash/annual_candidate_7.2b/results.parquet
  /opt/tmp/qianli/miso/trash/annual_candidate_7.2b/branch_level.parquet

Usage:
    source /home/xyz/workspace/pmodel/.venv/bin/activate
    PYTHONPATH=. python scripts/run_72b_full_grid.py
"""
from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time

os.environ.setdefault("RAY_ADDRESS", "ray://10.8.0.36:10001")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lightgbm as lgb
import numpy as np
import polars as pl

from ml.markets.miso.release_candidate import (
    BASE_FEATURES, BASE_LGB, BASE_BOOST_ROUNDS,
    STAGE2_FEATS, STAGE2_LGB, STAGE2_BOOST_ROUNDS,
    assign_bucket_labels, assign_bucket_weights,
    score_v0c, make_reserved_selection, compute_metrics,
    EVAL_CONFIGS, ROUNDS, K_LEVELS, R_AT_K, AQS_FULL,
    build_eval_table,
)

REF_RESULTS = "/opt/tmp/qianli/miso/trash/annual_candidate_7.2b/results.parquet"
REF_BRANCH = "/opt/tmp/qianli/miso/trash/annual_candidate_7.2b/branch_level.parquet"
OUT_DIR = "registry/7.2b"

# External signal paths
SIGNAL_TEMPLATES = {
    "V4.4": "/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V4.4.R{round}",
    "V7.1B": "/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V7.1B.R{round}",
}


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_external_signal(signal_name: str, eval_py: str, aq: str,
                         ct: str, market_round: int) -> dict[str, int]:
    """Load external signal branch->rank mapping."""
    template = SIGNAL_TEMPLATES.get(signal_name, "")
    if not template:
        return {}
    base = template.format(round=market_round)
    path = f"{base}/{eval_py}/{aq}/{ct}/"
    if not os.path.exists(path):
        return {}
    df = pl.read_parquet(path).filter(pl.col("equipment") != "")
    t0_rank = df.filter(pl.col("tier") == 0)["rank"].mean() if "tier" in df.columns else None
    t4_rank = df.filter(pl.col("tier") == 4)["rank"].mean() if "tier" in df.columns else None
    if t0_rank is not None and t4_rank is not None:
        rank_desc = t0_rank > t4_rank
        df = df.sort(["tier", "rank"], descending=[False, rank_desc])
    else:
        df = df.sort("rank")
    rank_map: dict[str, int] = {}
    for i, row in enumerate(df.iter_rows(named=True)):
        if row["equipment"] not in rank_map:
            rank_map[row["equipment"]] = i + 1
    return rank_map


# In-memory cache for eval tables (shared across rounds within a ctype)
_cache: dict[tuple, pl.DataFrame] = {}


def get_table(py: str, aq: str, ct: str, rnd: int) -> pl.DataFrame:
    key = (py, aq, ct, rnd)
    if key not in _cache:
        _cache[key] = build_eval_table(py, aq, ct, rnd)
    return _cache[key]


def main():
    t0 = time.time()

    from pbase.config.ray import init_ray
    import pmodel
    init_ray(extra_modules=[pmodel])

    all_rows: list[dict] = []
    all_branch: list[dict] = []
    cells_done = 0
    cells_failed = 0

    for market_round in ROUNDS:
        print(f"\n{'='*80}")
        print(f"  ROUND {market_round} (mem={mem_mb():.0f} MB)")
        print(f"{'='*80}")
        round_t0 = time.time()

        for eval_py, train_pys, eval_aqs in EVAL_CONFIGS:
            for ct in ["onpeak", "offpeak"]:
                cell_t0 = time.time()

                # ── Build training data ──
                train_frames = []
                for tpy in train_pys:
                    for taq in AQS_FULL:
                        try:
                            t = get_table(tpy, taq, ct, market_round)
                            t = t.with_columns(
                                pl.lit(tpy).alias("py"),
                                pl.lit(taq).alias("aq_label"),
                            )
                            train_frames.append(t)
                        except Exception as e:
                            print(f"  WARN: skip train {tpy}/{taq}/{ct}/R{market_round}: {e}")
                if not train_frames:
                    continue
                train_all = pl.concat(train_frames, how="diagonal")

                # ── Stage 1: Base model ──
                base_feats = [f for f in BASE_FEATURES if f in train_all.columns]
                sp_t = train_all["sp"].to_numpy().astype(np.float64)
                labels_t = assign_bucket_labels(sp_t)
                weights_t = assign_bucket_weights(labels_t)
                groups_t = train_all.group_by(
                    ["py", "aq_label"], maintain_order=True
                ).len()["len"].to_numpy()
                ds_base = lgb.Dataset(
                    train_all.select(base_feats).to_numpy().astype(np.float64),
                    label=labels_t, group=groups_t, weight=weights_t,
                    feature_name=base_feats, free_raw_data=False,
                )
                model_base = lgb.train(BASE_LGB, ds_base, num_boost_round=BASE_BOOST_ROUNDS)

                # ── Stage 2: Specialist ──
                s2_train_frames = []
                for tpy in train_pys:
                    for taq in AQS_FULL:
                        try:
                            t = get_table(tpy, taq, ct, market_round)
                            bf_s = t["bf"].to_numpy()
                            bf_c = t["bf_cross"].to_numpy()
                            t_td = t.filter(pl.Series((bf_s == 0) & (bf_c == 0)))
                            if len(t_td) > 0:
                                s2_train_frames.append(t_td)
                        except Exception:
                            pass

                if not s2_train_frames:
                    continue
                s2_train = pl.concat(s2_train_frames, how="diagonal")
                s2_feats = [f for f in STAGE2_FEATS if f in s2_train.columns]
                s2_sp = s2_train["sp"].to_numpy().astype(np.float64)
                s2_lab = (s2_sp > 0).astype(np.int32)
                s2_wt = np.ones(len(s2_lab))
                s2_wt[s2_sp > 200] = 3.0
                s2_wt[s2_sp > 5000] = 10.0
                ds_s2 = lgb.Dataset(
                    s2_train.select(s2_feats).to_numpy().astype(np.float64),
                    label=s2_lab, weight=s2_wt, feature_name=s2_feats, free_raw_data=False,
                )
                model_s2 = lgb.train(STAGE2_LGB, ds_s2, num_boost_round=STAGE2_BOOST_ROUNDS)

                # ── Evaluate each quarter ──
                for aq in eval_aqs:
                    try:
                        eq = get_table(eval_py, aq, ct, market_round)
                    except Exception as e:
                        print(f"  SKIP eval {eval_py}/{aq}/{ct}/R{market_round}: {e}")
                        cells_failed += 1
                        continue

                    N = len(eq)
                    sp = eq["sp"].to_numpy().astype(np.float64)
                    bf = eq["bf"].to_numpy().astype(np.float64)
                    bfc = eq["bf_cross"].to_numpy().astype(np.float64)
                    branches = eq["branch"].to_list()
                    is_td = (bf == 0) & (bfc == 0)
                    is_nb12 = eq["is_nb12"].to_numpy().astype(bool)
                    total_sp = float(eq["total_da_sp_quarter"][0]) if "total_da_sp_quarter" in eq.columns else float(sp.sum())

                    # Score
                    base_scores = model_base.predict(eq.select(base_feats).to_numpy().astype(np.float64))
                    v0c_scores = score_v0c(eq)
                    s2_scores = model_s2.predict(eq.select(s2_feats).to_numpy().astype(np.float64))

                    base_order = np.argsort(base_scores)[::-1]
                    v0c_order = np.argsort(v0c_scores)[::-1]
                    td_idx = np.where(is_td)[0]
                    td_order = td_idx[np.argsort(s2_scores[td_idx])[::-1]]

                    # External signals
                    v44_map = load_external_signal("V4.4", eval_py, aq, ct, market_round)
                    v71b_map = load_external_signal("V7.1B", eval_py, aq, ct, market_round)
                    br_set = {b: i for i, b in enumerate(branches)}

                    # Rank maps
                    base_rmap = {int(base_order[i]): i + 1 for i in range(N)}
                    v0c_rmap = {int(v0c_order[i]): i + 1 for i in range(N)}
                    td_rmap = {int(td_order[i]): i + 1 for i in range(len(td_order))}

                    # Candidate selections
                    cand_sel = {}
                    cand_rank = {}
                    for K in K_LEVELS:
                        R = R_AT_K[K]
                        sel = make_reserved_selection(base_order, td_order, K, R)
                        cand_sel[K] = set(sel)
                        sel_sorted = sorted(sel, key=lambda j: base_scores[j], reverse=True)
                        cand_rank[K] = {j: i + 1 for i, j in enumerate(sel_sorted)}

                    # Branch-level output
                    for idx in range(N):
                        branch = branches[idx]
                        all_branch.append({
                            "eval_py": eval_py, "aq": aq, "ct": ct,
                            "market_round": market_round,
                            "branch": branch,
                            "sp": float(sp[idx]),
                            "is_nb12": bool(is_nb12[idx]),
                            "is_td": bool(is_td[idx]),
                            "base_rank": base_rmap.get(idx, -1),
                            "v0c_rank": v0c_rmap.get(idx, -1),
                            "cand_rank_200": cand_rank[200].get(idx, -1),
                            "cand_rank_400": cand_rank[400].get(idx, -1),
                            "s2_rank": td_rmap.get(idx, -1) if is_td[idx] else -1,
                            "s2_denom": len(td_idx) if is_td[idx] else -1,
                            "v44_rank": v44_map.get(branch, -1),
                            "v71b_rank": v71b_map.get(branch, -1),
                            "in_cand_200": idx in cand_sel[200],
                            "in_cand_400": idx in cand_sel[400],
                            "in_base_200": idx in set(base_order[:200].tolist()),
                            "in_base_400": idx in set(base_order[:400].tolist()),
                        })

                    # Aggregate metrics
                    meta = {
                        "eval_py": eval_py, "aq": aq, "ct": ct,
                        "market_round": market_round,
                    }
                    for K in K_LEVELS:
                        R = R_AT_K[K]
                        # v0c
                        m = compute_metrics(sp, is_nb12, list(v0c_order[:K]), total_sp)
                        all_rows.append({**meta, "K": K, "model": "v0c", **m})
                        # base
                        m = compute_metrics(sp, is_nb12, list(base_order[:K]), total_sp)
                        all_rows.append({**meta, "K": K, "model": "base", **m})
                        # candidate (base+R)
                        m = compute_metrics(sp, is_nb12, list(cand_sel[K]), total_sp)
                        all_rows.append({**meta, "K": K, "model": f"base+R{R}", **m})
                        # V4.4
                        if v44_map:
                            v44_top = sorted(v44_map, key=lambda b: v44_map[b])[:K]
                            v44_idx = [br_set[b] for b in v44_top if b in br_set]
                            m = compute_metrics(sp, is_nb12, v44_idx, total_sp)
                            all_rows.append({**meta, "K": K, "model": "V4.4", **m})
                        # V7.1B
                        if v71b_map:
                            v71b_top = sorted(v71b_map, key=lambda b: v71b_map[b])[:K]
                            v71b_idx = [br_set[b] for b in v71b_top if b in br_set]
                            m = compute_metrics(sp, is_nb12, v71b_idx, total_sp)
                            all_rows.append({**meta, "K": K, "model": "V7.1B", **m})

                    cells_done += 1

                del train_all, s2_train
                gc.collect()
                print(f"  {eval_py}/{ct}/R{market_round}: {len(eval_aqs)} quarters, "
                      f"{time.time()-cell_t0:.0f}s (mem={mem_mb():.0f} MB)")

        # Clear cache per round to limit memory
        _cache.clear()
        gc.collect()
        print(f"  Round {market_round} total: {time.time()-round_t0:.0f}s")

    # ── Save ──
    os.makedirs(OUT_DIR, exist_ok=True)
    rdf = pl.DataFrame(all_rows)
    bdf = pl.DataFrame(all_branch)
    rdf.write_parquet(f"{OUT_DIR}/results.parquet")
    bdf.write_parquet(f"{OUT_DIR}/branch_level.parquet")
    print(f"\nSaved: results.parquet {rdf.shape}, branch_level.parquet {bdf.shape}")
    print(f"Cells done: {cells_done}, failed: {cells_failed}")

    # ── Parity comparison ──
    parity = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # Row counts
    parity["results_rows"] = rdf.shape[0]
    parity["results_expected"] = 1380
    parity["results_match"] = rdf.shape[0] == 1380
    parity["branch_rows"] = bdf.shape[0]
    parity["branch_expected"] = 368808
    parity["branch_match"] = bdf.shape[0] == 368808

    # Compare against research reference if available
    if os.path.exists(REF_RESULTS):
        ref_r = pl.read_parquet(REF_RESULTS)
        parity["ref_results_rows"] = ref_r.shape[0]

        # Join on key columns and compare SP
        shared_cols = ["eval_py", "aq", "ct", "market_round", "K", "model"]
        joined = rdf.join(ref_r, on=shared_cols, how="inner", suffix="_ref")
        parity["results_joined"] = joined.shape[0]

        if "sp_ref" in joined.columns and "sp" in joined.columns:
            sp_diff = (joined["sp"] - joined["sp_ref"]).abs()
            parity["sp_mean_abs_diff"] = float(sp_diff.mean())
            parity["sp_max_abs_diff"] = float(sp_diff.max())
            parity["sp_corr"] = float(joined.select(pl.corr("sp", "sp_ref"))[0, 0])
        if "vc_ref" in joined.columns and "vc" in joined.columns:
            vc_diff = (joined["vc"] - joined["vc_ref"]).abs()
            parity["vc_mean_abs_diff"] = float(vc_diff.mean())
            parity["vc_max_abs_diff"] = float(vc_diff.max())

    if os.path.exists(REF_BRANCH):
        ref_b = pl.read_parquet(REF_BRANCH)
        parity["ref_branch_rows"] = ref_b.shape[0]

        # Compare branch-level ranks on shared key
        shared_cols = ["eval_py", "aq", "ct", "market_round", "branch"]
        joined_b = bdf.join(ref_b, on=shared_cols, how="inner", suffix="_ref")
        parity["branch_joined"] = joined_b.shape[0]

        if "base_rank_ref" in joined_b.columns:
            rank_diff = (joined_b["base_rank"] - joined_b["base_rank_ref"]).abs()
            parity["base_rank_mean_diff"] = float(rank_diff.mean())
            parity["base_rank_exact_match_pct"] = float((rank_diff == 0).mean() * 100)

        # Selected-set overlap: per-cell intersection / K, averaged across cells
        # NOT row-wise boolean (which is dominated by true-negatives)
        cell_keys = joined_b.select(["eval_py", "aq", "ct", "market_round"]).unique()
        for K in [200, 400]:
            col = f"in_cand_{K}"
            if f"{col}_ref" not in joined_b.columns:
                continue
            overlaps = []
            for row in cell_keys.iter_rows(named=True):
                cell = joined_b.filter(
                    (pl.col("eval_py") == row["eval_py"]) &
                    (pl.col("aq") == row["aq"]) &
                    (pl.col("ct") == row["ct"]) &
                    (pl.col("market_round") == row["market_round"])
                )
                prod_sel = set(cell.filter(pl.col(col))["branch"].to_list())
                ref_sel = set(cell.filter(pl.col(f"{col}_ref"))["branch"].to_list())
                overlaps.append(len(prod_sel & ref_sel) / K * 100 if K > 0 else 0)
            parity[f"cand_{K}_selected_set_overlap_pct"] = float(np.mean(overlaps))
            parity[f"cand_{K}_min_overlap_pct"] = float(np.min(overlaps))
            parity[f"cand_{K}_max_overlap_pct"] = float(np.max(overlaps))

    with open(f"{OUT_DIR}/parity_summary.json", "w") as f:
        json.dump(parity, f, indent=2, default=str)

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"  7.2b FULL GRID SUMMARY")
    print(f"{'='*60}")
    for k, v in parity.items():
        print(f"  {k}: {v}")
    print(f"  Total time: {time.time()-t0:.0f}s")

    import ray
    ray.shutdown()


if __name__ == "__main__":
    main()
