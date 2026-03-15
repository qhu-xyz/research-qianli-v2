"""Phase 5: Re-evaluation under new metric framework (150/200/300/400).

Computes true solo baselines, sweeps hard-R and blend configs as paired
(K_lo, K_hi) candidates, ranks by paired scorecard, saves to registry.

Usage:
    # Dev sweep + holdout validation
    PYTHONPATH=. uv run python scripts/run_phase5_reeval.py

    # Holdout only for a specific config
    PYTHONPATH=. uv run python scripts/run_phase5_reeval.py \
        --holdout-only --config C1_a0.05 --version phase5_blend_a005
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from ml.config import (
    EVAL_SPLITS, DEV_GROUPS, HOLDOUT_GROUPS, AQ_QUARTERS,
    DANGEROUS_THRESHOLD, PHASE5_K_LEVELS,
    get_bf_cutoff_month,
)
from ml.features import build_model_table_all
from ml.features_trackb import compute_recency_features
from ml.history_features import compute_history_features
from ml.train import train_and_predict
from ml.evaluate import evaluate_group
from ml.merge import merge_tracks
from ml.registry import save_experiment
from scripts.archive.run_phase4a_experiment import compute_v0c_scores, compute_sample_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NB_FEATURES = [
    'bin_80_cid_max', 'bin_70_cid_max', 'bin_90_cid_max',
    'count_active_cids', 'bin_60_cid_max', 'bin_100_cid_max', 'bin_110_cid_max',
    'bin_-50_cid_max', 'bin_120_cid_max', 'bin_-100_cid_max', 'bin_150_cid_max',
    'shadow_price_da', 'da_rank_value', 'historical_max_sp',
]

W_SCORECARD = {"VC": 0.4, "Recall": 0.2, "Dang_Recall": 0.2, "NB12_SP": 0.2}


def _weight_tiered(sp: np.ndarray) -> np.ndarray:
    """Tiered sample weights: equal thirds 1/3/10."""
    w = np.ones(len(sp))
    pos = sp > 0
    if pos.sum() == 0:
        return w
    ranks = sp[pos].argsort().argsort()
    n = len(ranks)
    w[pos] = np.where(ranks < n // 3, 1.0, np.where(ranks < 2 * n // 3, 3.0, 10.0))
    return w


def _weight_sqrt(sp: np.ndarray) -> np.ndarray:
    """Sqrt(SP) sample weights: sublinear, value-proportional."""
    w = np.ones(len(sp))
    pos = sp > 0
    if pos.sum() > 0:
        w[pos] = np.sqrt(sp[pos])
    return w


WEIGHT_FNS = {"tiered": _weight_tiered, "sqrt": _weight_sqrt}


def train_nb(train_df: pl.DataFrame, weight_scheme: str = "tiered") -> lgb.Booster:
    X = train_df.select(NB_FEATURES).to_numpy().astype(np.float64)
    sp = train_df["realized_shadow_price"].to_numpy().astype(np.float64)
    y = (sp > 0).astype(int)
    w = WEIGHT_FNS[weight_scheme](sp)
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    cw = np.where(y == 1, n0 / max(n1, 1), 1.0)
    ds = lgb.Dataset(X, label=y, weight=cw * w, feature_name=NB_FEATURES, free_raw_data=False)
    return lgb.train({"objective": "binary", "num_leaves": 15, "learning_rate": 0.03,
                       "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 5,
                       "num_threads": 4, "verbose": -1}, ds, num_boost_round=200)


def eval_at_k(gdf: pl.DataFrame, topk_mask: np.ndarray, K: int) -> dict:
    sp = gdf["realized_shadow_price"].to_numpy().astype(np.float64)
    total_sp = sp.sum()
    n_bind = (sp > 0).sum()
    total_da = float(gdf["total_da_sp_quarter"][0])
    is_nb12 = gdf["is_nb_12"].to_numpy() if "is_nb_12" in gdf.columns else np.zeros(len(sp), dtype=bool)
    dang = sp > DANGEROUS_THRESHOLD
    cohorts = gdf["cohort"].to_list()
    is_dorm = np.array([c == "history_dormant" for c in cohorts])

    return {
        f"VC@{K}": sp[topk_mask].sum() / total_sp if total_sp > 0 else 0,
        f"Recall@{K}": (sp[topk_mask] > 0).sum() / n_bind if n_bind > 0 else 0,
        f"Abs_SP@{K}": sp[topk_mask].sum() / total_da if total_da > 0 else 0,
        f"NB12_Count@{K}": int(is_nb12[topk_mask].sum()),
        f"NB12_SP@{K}": sp[topk_mask & is_nb12].sum() / sp[is_nb12].sum() if sp[is_nb12].sum() > 0 else 0,
        f"Dang_Recall@{K}": (topk_mask & dang).sum() / dang.sum() if dang.sum() > 0 else 0,
        f"Dang_SP_Ratio@{K}": sp[topk_mask & dang].sum() / sp[dang].sum() if sp[dang].sum() > 0 else 0,
        f"Dang_Count@{K}": int((topk_mask & dang).sum()),
        "Dang_Total": int(dang.sum()),
        f"Dorm_inK@{K}": int((topk_mask & is_dorm).sum()),
        f"Dorm_SP@{K}": float(sp[topk_mask & is_dorm].sum()),
    }


def make_topk_mask(scores: np.ndarray, K: int) -> np.ndarray:
    n = len(scores)
    topk = np.argsort(scores)[::-1][:min(K, n)]
    mask = np.zeros(n, dtype=bool)
    mask[topk] = True
    return mask


def run_solo(gdf, base_scores, K):
    return eval_at_k(gdf, make_topk_mask(base_scores, K), K)


def run_hard_r(gdf, base_scores, K, R, nb_model):
    cohorts = gdf["cohort"].to_list()
    estab_idx = [i for i, c in enumerate(cohorts) if c == "established"]
    dorm_idx = [i for i, c in enumerate(cohorts) if c == "history_dormant"]

    estab_scores = base_scores[estab_idx]
    r_actual = min(R, len(dorm_idx))
    n_a = min(K - r_actual, len(estab_idx))
    a_top = np.array(estab_idx)[np.argsort(estab_scores)[::-1][:n_a]]

    if r_actual > 0 and len(dorm_idx) > 0:
        dorm_df = gdf[dorm_idx]
        X_b = dorm_df.select(NB_FEATURES).to_numpy().astype(np.float64)
        nb_scores = nb_model.predict(X_b)
        b_top = np.array(dorm_idx)[np.argsort(nb_scores)[::-1][:r_actual]]
    else:
        b_top = np.array([], dtype=int)

    mask = np.zeros(len(gdf), dtype=bool)
    mask[a_top] = True
    mask[b_top] = True
    return eval_at_k(gdf, mask, K)


def run_hard_r_tau(gdf, base_scores, K, R, nb_model, tau):
    """Hard two-track with tau threshold — only insert NB picks scoring >= tau."""
    cohorts = gdf["cohort"].to_list()
    estab_idx = [i for i, c in enumerate(cohorts) if c == "established"]
    dorm_idx = [i for i, c in enumerate(cohorts) if c == "history_dormant"]

    if len(dorm_idx) > 0:
        dorm_df = gdf[dorm_idx]
        X_b = dorm_df.select(NB_FEATURES).to_numpy().astype(np.float64)
        nb_scores = nb_model.predict(X_b)
        qualified = nb_scores >= tau
        r_actual = min(R, int(qualified.sum()))
        qualified_idx = np.array(dorm_idx)[qualified]
        qualified_scores = nb_scores[qualified]
        b_top = qualified_idx[np.argsort(qualified_scores)[::-1][:r_actual]]
    else:
        r_actual = 0
        b_top = np.array([], dtype=int)

    n_a = min(K - r_actual, len(estab_idx))
    a_top = np.array(estab_idx)[np.argsort(base_scores[estab_idx])[::-1][:n_a]]

    mask = np.zeros(len(gdf), dtype=bool)
    mask[a_top] = True
    mask[b_top] = True
    return eval_at_k(gdf, mask, K)


def run_blend(gdf, base_scores, K, alpha, nb_model):
    cohorts = gdf["cohort"].to_list()
    is_dorm = np.array([c == "history_dormant" for c in cohorts])

    final = base_scores.copy()
    if alpha > 0 and is_dorm.sum() > 0:
        dorm_idx = np.where(is_dorm)[0]
        dorm_df = gdf[dorm_idx.tolist()]
        X_b = dorm_df.select(NB_FEATURES).to_numpy().astype(np.float64)
        nb_raw = nb_model.predict(X_b)
        base_range = base_scores.max() - base_scores.min()
        nb_range = nb_raw.max() - nb_raw.min()
        if nb_range > 0 and base_range > 0:
            nb_norm = (nb_raw - nb_raw.min()) / nb_range * base_range
            final[dorm_idx] += alpha * nb_norm

    return eval_at_k(gdf, make_topk_mask(final, K), K)


def paired_score(avgs_lo: dict, avgs_hi: dict, k_lo: int, k_hi: int) -> float:
    score = 0.0
    for level, K, avgs in [("lo", k_lo, avgs_lo), ("hi", k_hi, avgs_hi)]:
        score += 0.5 * (
            W_SCORECARD["VC"] * avgs.get(f"VC@{K}", 0)
            + W_SCORECARD["Recall"] * avgs.get(f"Recall@{K}", 0)
            + W_SCORECARD["Dang_Recall"] * avgs.get(f"Dang_Recall@{K}", 0)
            + W_SCORECARD["NB12_SP"] * avgs.get(f"NB12_SP@{K}", 0)
        )
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-only", action="store_true")
    parser.add_argument("--config", default=None)
    parser.add_argument("--version", default=None)
    args = parser.parse_args()

    t0 = time.time()
    V3A_FEATURES = json.load(open("registry/archive/v3a/config.json"))["features"]

    # Build data
    all_needed = set()
    for si in EVAL_SPLITS.values():
        for py in si["train_pys"] + si["eval_pys"]:
            for aq in AQ_QUARTERS:
                all_needed.add(f"{py}/{aq}")
    all_needed.discard("2025-06/aq4")
    mt = build_model_table_all(sorted(all_needed))

    # Enrich
    enriched_parts = []
    for (py, aq), part in mt.group_by(["planning_year", "aq_quarter"], maintain_order=True):
        dormant = part.filter(part["cohort"] == "history_dormant")["branch_name"].to_list()
        if dormant:
            cutoff = get_bf_cutoff_month(str(py))
            _, mb = compute_history_features(str(py), str(aq), part["branch_name"].to_list())
            rec = compute_recency_features(mb, dormant, cutoff)
            part = part.join(rec, on="branch_name", how="left")
            for c in rec.columns:
                if c != "branch_name" and c in part.columns:
                    part = part.with_columns(pl.col(c).fill_null(0 if "month" in c or "binding" in c else 0.0))
        enriched_parts.append(part)
    mt = pl.concat(enriched_parts, how="diagonal")
    logger.info("Data built (%.1fs)", time.time() - t0)

    # Paired configs
    CONFIGS = []
    CONFIGS.append(("A1_v0c_solo", "v0c", "solo", {}))
    CONFIGS.append(("A2_v3a_solo", "v3a", "solo", {}))

    for ta in ["v0c", "v3a"]:
        tag = "B1" if ta == "v0c" else "B2"
        for r_lo, r_hi in [(5, 10), (10, 15), (10, 20), (15, 20), (15, 30), (20, 30)]:
            CONFIGS.append((f"{tag}_R{r_lo}_{r_hi}", ta, "hard", {"r_pairs": (r_lo, r_hi)}))

    for ta in ["v0c", "v3a"]:
        tag = "C1" if ta == "v0c" else "C2"
        for alpha in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
            CONFIGS.append((f"{tag}_a{alpha:.2f}", ta, "blend", {"alpha": alpha, "weight": "tiered"}))

    # Sqrt weight blend variants
    for ta in ["v0c", "v3a"]:
        tag = "S1" if ta == "v0c" else "S2"
        for alpha in [0.05, 0.1, 0.2]:
            CONFIGS.append((f"{tag}_sqrt_a{alpha:.2f}", ta, "blend", {"alpha": alpha, "weight": "sqrt"}))

    # Adaptive-R variants (v0c only, tau from NB score percentiles)
    for r_lo, r_hi in [(10, 20), (15, 30)]:
        for tau_pctile in [70, 80, 90]:
            CONFIGS.append((f"D1_R{r_lo}_{r_hi}_p{tau_pctile}", "v0c", "adaptive",
                            {"r_pairs": (r_lo, r_hi), "tau_pctile": tau_pctile}))

    PAIRS = [(150, 300), (200, 400)]

    for target_split in (["holdout"] if args.holdout_only else ["dev", "holdout"]):
        eval_groups = DEV_GROUPS if target_split == "dev" else HOLDOUT_GROUPS

        # Cache Track A scores and NB models (both weight schemes)
        v0c_cache, v3a_cache, group_cache = {}, {}, {}
        nb_cache: dict[tuple[str, str], lgb.Booster] = {}  # (eval_key, weight_scheme) -> model
        v3a_avail = [f for f in V3A_FEATURES if f in mt.columns]

        for ek, si in EVAL_SPLITS.items():
            if si["split"] != target_split:
                continue
            scored, _ = train_and_predict(mt, si["train_pys"], si["eval_pys"], v3a_avail)
            for row_py in si["eval_pys"]:
                for aq in AQ_QUARTERS:
                    v3a_group = scored.filter(
                        (pl.col("planning_year") == row_py) & (pl.col("aq_quarter") == aq))
                    if len(v3a_group) > 0:
                        sm = dict(zip(v3a_group["branch_name"].to_list(), v3a_group["score"].to_list()))
                        gdf = mt.filter((pl.col("planning_year") == row_py) & (pl.col("aq_quarter") == aq))
                        v3a_cache[(row_py, aq)] = np.array([sm.get(b, 0.0) for b in gdf["branch_name"].to_list()])
            nb_train = mt.filter(pl.col("planning_year").is_in(si["train_pys"])
                                 & (pl.col("cohort") == "history_dormant"))
            for ws in WEIGHT_FNS:
                nb_cache[(ek, ws)] = train_nb(nb_train, weight_scheme=ws)

        for (py, aq), gdf in mt.group_by(["planning_year", "aq_quarter"], maintain_order=True):
            key = f"{py}/{aq}"
            if key not in eval_groups:
                continue
            group_cache[(str(py), str(aq))] = gdf
            v0c_cache[(str(py), str(aq))] = compute_v0c_scores(gdf)

        logger.info("Caches built for %s (%.1fs)", target_split, time.time() - t0)

        # Evaluate all configs
        all_results = {}  # cfg_name -> {(k_lo,k_hi): {"lo": {group: m}, "hi": {group: m}}}

        for cfg_name, track_a, mode, params in CONFIGS:
            if args.config and cfg_name != args.config:
                continue
            all_results[cfg_name] = {}
            for k_lo, k_hi in PAIRS:
                res_lo, res_hi = {}, {}
                for ek, si in EVAL_SPLITS.items():
                    if si["split"] != target_split:
                        continue
                    ws = params.get("weight", "tiered")
                    nb_model = nb_cache[(ek, ws)]
                    for py in si["eval_pys"]:
                        for aq in AQ_QUARTERS:
                            key = f"{py}/{aq}"
                            if key not in eval_groups:
                                continue
                            gdf = group_cache[(py, aq)]
                            base = v0c_cache[(py, aq)] if track_a == "v0c" else v3a_cache[(py, aq)]

                            for K, res in [(k_lo, res_lo), (k_hi, res_hi)]:
                                R_lo, R_hi = params.get("r_pairs", (0, 0))
                                R = R_lo if K == k_lo else R_hi
                                if mode == "solo":
                                    m = run_solo(gdf, base, K)
                                elif mode == "hard":
                                    m = run_hard_r(gdf, base, K, R, nb_model)
                                elif mode == "blend":
                                    m = run_blend(gdf, base, K, params["alpha"], nb_model)
                                elif mode == "adaptive":
                                    # Compute tau from training split NB scores
                                    nb_train_df = mt.filter(
                                        pl.col("planning_year").is_in(si["train_pys"])
                                        & (pl.col("cohort") == "history_dormant"))
                                    X_train_nb = nb_train_df.select(NB_FEATURES).to_numpy().astype(np.float64)
                                    train_nb_scores = nb_model.predict(X_train_nb)
                                    tau = float(np.percentile(train_nb_scores, params["tau_pctile"]))
                                    m = run_hard_r_tau(gdf, base, K, R, nb_model, tau)
                                res[key] = m

                all_results[cfg_name][(k_lo, k_hi)] = {"lo": res_lo, "hi": res_hi}

        logger.info("All configs evaluated (%.1fs)", time.time() - t0)

        # Scorecard
        for k_lo, k_hi in PAIRS:
            # Solo baselines for gates
            solo_best = {}
            for sn in ["A1_v0c_solo", "A2_v3a_solo"]:
                if sn not in all_results:
                    continue
                for level, K in [("lo", k_lo), ("hi", k_hi)]:
                    vals = list(all_results[sn][(k_lo, k_hi)][level].values())
                    if vals:
                        for mk in [f"VC@{K}", f"Dang_Recall@{K}"]:
                            avg = sum(v.get(mk, 0) for v in vals) / len(vals)
                            if mk not in solo_best or avg > solo_best[mk]:
                                solo_best[mk] = avg

            print(f"\n{'='*170}")
            print(f"  {target_split.upper()} — PAIR ({k_lo}, {k_hi})")
            print(f"{'='*170}")
            header = (f"  {'Config':<25} "
                      f"{'VC_lo':>7} {'Rec_lo':>7} {'DgR_lo':>7} {'NB_SP_lo':>8} {'Drm_lo':>7} "
                      f"{'VC_hi':>7} {'Rec_hi':>7} {'DgR_hi':>7} {'NB_SP_hi':>8} {'Drm_hi':>7} "
                      f"{'Score':>7} {'Gate':>5}")
            print(header)
            print(f"  {'-'*155}")

            scored_list = []
            for cfg_name in all_results:
                pair_data = all_results[cfg_name].get((k_lo, k_hi))
                if not pair_data:
                    continue
                avgs = {}
                for level in ["lo", "hi"]:
                    vals = list(pair_data[level].values())
                    if vals:
                        K = k_lo if level == "lo" else k_hi
                        ng = len(vals)
                        avgs[level] = {}
                        for mk in vals[0]:
                            avgs[level][mk] = sum(v[mk] for v in vals) / ng
                if "lo" not in avgs or "hi" not in avgs:
                    continue

                gate = True
                for K, level in [(k_lo, "lo"), (k_hi, "hi")]:
                    vc_gate = solo_best.get(f"VC@{K}", 0) - 0.02
                    dg_gate = solo_best.get(f"Dang_Recall@{K}", 0) - 0.05
                    if avgs[level].get(f"VC@{K}", 0) < vc_gate:
                        gate = False
                    if avgs[level].get(f"Dang_Recall@{K}", 0) < dg_gate:
                        gate = False

                score = paired_score(avgs["lo"], avgs["hi"], k_lo, k_hi)

                print(f"  {cfg_name:<25} "
                      f"{avgs['lo'].get(f'VC@{k_lo}',0):>7.4f} {avgs['lo'].get(f'Recall@{k_lo}',0):>7.4f} "
                      f"{avgs['lo'].get(f'Dang_Recall@{k_lo}',0):>7.4f} {avgs['lo'].get(f'NB12_SP@{k_lo}',0):>8.4f} "
                      f"{avgs['lo'].get(f'Dorm_inK@{k_lo}',0):>7.1f} "
                      f"{avgs['hi'].get(f'VC@{k_hi}',0):>7.4f} {avgs['hi'].get(f'Recall@{k_hi}',0):>7.4f} "
                      f"{avgs['hi'].get(f'Dang_Recall@{k_hi}',0):>7.4f} {avgs['hi'].get(f'NB12_SP@{k_hi}',0):>8.4f} "
                      f"{avgs['hi'].get(f'Dorm_inK@{k_hi}',0):>7.1f} "
                      f"{score:>7.4f} {'PASS' if gate else 'fail':>5}")

                scored_list.append((cfg_name, score, gate, avgs))

            passing = [(n, s, a) for n, s, g, a in scored_list if g]
            passing.sort(key=lambda x: -x[1])
            print(f"\n  TOP 5 (gate-passing):")
            for i, (name, score, avgs) in enumerate(passing[:5]):
                print(f"    #{i+1} {name:<25} score={score:.4f}")

            # Save winner to registry if holdout + version specified
            if target_split == "holdout" and args.version and passing:
                winner_name, winner_score, winner_avgs = passing[0]
                pair_data = all_results[winner_name][(k_lo, k_hi)]
                config = {
                    "version": args.version,
                    "phase": "5",
                    "champion": winner_name,
                    "paired_score": winner_score,
                    "k_pair": [k_lo, k_hi],
                    "scorecard_weights": W_SCORECARD,
                    "nb_features": NB_FEATURES,
                }
                metrics = {
                    "per_group_lo": pair_data["lo"],
                    "per_group_hi": pair_data["hi"],
                    "mean_lo": winner_avgs["lo"],
                    "mean_hi": winner_avgs["hi"],
                }
                version_id = f"{args.version}_{k_lo}_{k_hi}"
                save_experiment(version_id, config, metrics)
                logger.info("Saved %s to registry", version_id)

    logger.info("Done (%.1fs)", time.time() - t0)


if __name__ == "__main__":
    main()
