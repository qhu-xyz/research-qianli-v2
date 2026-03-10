"""v17: Partial-month binding frequency for annual signal.

Adds first-12-days-of-April binding data that is available at bid submission
but currently discarded by the months < YYYY-04 cutoff.

Two new feature types:
  bf_partial / bfo_partial — single-year current April partial (recency)
  bf_april / bfo_april     — multi-year April BF (structural seasonal signal)

Variants:
  v16 (baseline)        — 7f champion: shadow_price_da, da_rank_value, bf_6/12/15, bfo_6/12
  v17a (+april only)    — v16 + bf_april + bfo_april (9f)
  v17b (+partial only)  — v16 + bf_partial + bfo_partial (9f)
  v17c (+all partial)   — v16 + bf_partial + bf_april + bfo_partial + bfo_april (11f)
"""
import gc
import json
import resource
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import polars as pl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import (
    PipelineConfig, LTRConfig, EVAL_SPLITS, AQ_ROUNDS,
    DEFAULT_EVAL_GROUPS,
)
from ml.data_loader import load_v61_enriched
from ml.evaluate import evaluate_ltr, aggregate_months
from ml.ground_truth import get_ground_truth
from ml.features import prepare_features, compute_query_groups
from ml.train import train_ltr_model, predict_scores
from ml.binding_freq import (
    enrich_with_binding_freq,
    enrich_with_offpeak_bf,
    enrich_with_partial_bf,
)


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ── Constants ──

HOLDOUT_GROUPS = ["2025-06/aq1", "2025-06/aq2", "2025-06/aq3"]

GROUP_A = ["VC@20", "VC@50", "VC@100", "Recall@20", "Recall@50", "Recall@100", "NDCG"]
GROUP_B = ["Spearman", "Tier0-AP", "Tier01-AP"]

PARTIAL_N_DAYS = 12

# v16 champion (baseline)
V16_FEATURES = ["shadow_price_da", "da_rank_value", "bf_6", "bf_12", "bf_15", "bfo_6", "bfo_12"]
V16_MONOTONE = [1, -1, 1, 1, 1, 1, 1]

VARIANTS = {
    "v16_baseline": {
        "features": V16_FEATURES,
        "monotone": V16_MONOTONE,
        "use_partial": False,
    },
    "v17a_april": {
        "features": V16_FEATURES + ["bf_april", "bfo_april"],
        "monotone": V16_MONOTONE + [1, 1],
        "use_partial": True,
    },
    "v17b_partial": {
        "features": V16_FEATURES + ["bf_partial", "bfo_partial"],
        "monotone": V16_MONOTONE + [1, 1],
        "use_partial": True,
    },
    "v17c_all": {
        "features": V16_FEATURES + ["bf_partial", "bf_april", "bfo_partial", "bfo_april"],
        "monotone": V16_MONOTONE + [1, 1, 1, 1],
        "use_partial": True,
    },
}

# Year-level grouping for dev
_year_groups: dict[str, list[str]] = OrderedDict()
for _g in DEFAULT_EVAL_GROUPS:
    _y = _g.split("/")[0]
    _year_groups.setdefault(_y, []).append(_g)


# ── Data loading ──

_DATA_CACHE: dict[tuple, pl.DataFrame] = {}


def load_group(planning_year: str, aq_round: str, use_partial: bool = False) -> pl.DataFrame:
    cache_key = (planning_year, aq_round, use_partial)
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    df = load_v61_enriched(planning_year, aq_round)
    group_id = f"{planning_year}/{aq_round}"
    df = df.with_columns(pl.lit(group_id).alias("query_group"))
    df = get_ground_truth(planning_year, aq_round, df, cache=True)

    # v16 champion uses backfill + offpeak
    floor = "2017-04"
    df = enrich_with_binding_freq(df, planning_year, aq_round, floor_month=floor)
    df = enrich_with_offpeak_bf(df, planning_year, aq_round, floor_month=floor)

    if use_partial:
        df = enrich_with_partial_bf(
            df, planning_year, aq_round,
            n_days=PARTIAL_N_DAYS, include_offpeak=True, floor_year=2020,
        )

    _DATA_CACHE[cache_key] = df
    return df


def _get_train_groups(eval_year: str) -> list[str]:
    for split_def in EVAL_SPLITS.values():
        if split_def["eval_year"] == eval_year:
            return [f"{y}/{aq}" for y in split_def["train_years"] for aq in AQ_ROUNDS]
    raise ValueError(f"No split for year: {eval_year}")


# ── Training and eval ──

def run_variant(name: str, spec: dict,
                eval_groups: list[str] | None = None,
                holdout: bool = False) -> dict:
    features = spec["features"]
    monotone = spec["monotone"]
    use_partial = spec["use_partial"]

    config = PipelineConfig(
        ltr=LTRConfig(
            features=features,
            monotone_constraints=monotone,
            backend="lightgbm",
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            label_mode="tiered",
        ),
    )

    if eval_groups is None:
        eval_groups = DEFAULT_EVAL_GROUPS

    # Dev evaluation: train per eval-year
    per_group = {}
    model_cache = {}

    for year, groups_in_year in _year_groups.items():
        if year not in model_cache:
            train_ids = _get_train_groups(year)
            dfs = []
            for gid in train_ids:
                py, aq = gid.split("/")
                try:
                    dfs.append(load_group(py, aq, use_partial))
                except FileNotFoundError:
                    pass
            train_df = pl.concat(dfs, how="diagonal").sort("query_group")
            X_train, _ = prepare_features(train_df, config.ltr)
            y_train = train_df["realized_shadow_price"].to_numpy().astype(np.float64)
            groups_train = compute_query_groups(train_df)
            model_cache[year] = train_ltr_model(X_train, y_train, groups_train, config.ltr)
            del train_df, X_train, y_train, groups_train

        model = model_cache[year]
        for gid in groups_in_year:
            py, aq = gid.split("/")
            df = load_group(py, aq, use_partial)
            X, _ = prepare_features(df, config.ltr)
            actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
            scores = predict_scores(model, X)
            per_group[gid] = evaluate_ltr(actual, scores)

    last_model = list(model_cache.values())[-1]
    feat_imp = {}
    if hasattr(last_model, "feature_importance"):
        importance = last_model.feature_importance(importance_type="gain")
        feat_imp = dict(zip(features, [float(x) for x in importance]))

    dev_agg = aggregate_months(per_group)

    # Holdout
    holdout_per_group = {}
    holdout_agg = {}
    holdout_feat_imp = {}
    if holdout:
        holdout_train_ids = _get_train_groups("2025-06")
        dfs = []
        for gid in holdout_train_ids:
            py, aq = gid.split("/")
            try:
                dfs.append(load_group(py, aq, use_partial))
            except FileNotFoundError:
                pass
        train_df = pl.concat(dfs, how="diagonal").sort("query_group")
        X_train, _ = prepare_features(train_df, config.ltr)
        y_train = train_df["realized_shadow_price"].to_numpy().astype(np.float64)
        groups_train = compute_query_groups(train_df)
        ho_model = train_ltr_model(X_train, y_train, groups_train, config.ltr)
        del train_df, X_train, y_train, groups_train

        for gid in HOLDOUT_GROUPS:
            py, aq = gid.split("/")
            df = load_group(py, aq, use_partial)
            X, _ = prepare_features(df, config.ltr)
            actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
            scores = predict_scores(ho_model, X)
            holdout_per_group[gid] = evaluate_ltr(actual, scores)

        holdout_agg = aggregate_months(holdout_per_group)

        if hasattr(ho_model, "feature_importance"):
            importance = ho_model.feature_importance(importance_type="gain")
            holdout_feat_imp = dict(zip(features, [float(x) for x in importance]))

        del ho_model

    del model_cache
    gc.collect()

    return {
        "dev_per_group": per_group,
        "dev_agg": dev_agg,
        "dev_feat_imp": feat_imp,
        "holdout_per_group": holdout_per_group,
        "holdout_agg": holdout_agg,
        "holdout_feat_imp": holdout_feat_imp,
    }


# ── Display ──

def print_comparison(all_results: dict[str, dict], phase: str = "dev"):
    agg_key = f"{phase}_agg"
    metrics = GROUP_A + GROUP_B
    print(f"\n{'='*130}")
    print(f"  {phase.upper()} RESULTS (mean across groups)")
    print(f"{'='*130}")
    header = f"{'Variant':<25}"
    for m in metrics:
        header += f"  {m:>10}"
    print(header)
    print("-" * 130)

    baseline_mean = all_results.get("v16_baseline", {}).get(agg_key, {}).get("mean", {})
    for vname, res in all_results.items():
        agg = res.get(agg_key, {})
        mean = agg.get("mean", {}) if agg else {}
        if not mean:
            continue
        row = f"{vname:<25}"
        for m in metrics:
            val = mean.get(m, 0)
            row += f"  {val:>10.4f}"
        print(row)

    # Delta line
    if baseline_mean:
        for vname, res in all_results.items():
            if vname == "v16_baseline":
                continue
            agg = res.get(agg_key, {})
            mean = agg.get("mean", {}) if agg else {}
            if not mean:
                continue
            row = f"  {'d:'+vname:<23}"
            for m in metrics:
                val = mean.get(m, 0)
                base = baseline_mean.get(m, 0)
                if base > 0:
                    pct = (val - base) / base * 100
                    row += f"  {pct:>+9.1f}%"
                else:
                    row += f"  {'N/A':>10}"
            print(row)

    print()


def print_feature_importance(all_results: dict[str, dict], phase: str = "dev"):
    imp_key = f"{phase}_feat_imp"
    print(f"\n  {phase.upper()} FEATURE IMPORTANCE (gain)")
    print("-" * 100)
    for vname, res in all_results.items():
        fi = res.get(imp_key, {})
        if not fi:
            continue
        total = sum(fi.values()) or 1
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        parts = [f"{k}={v/total*100:.1f}%" for k, v in sorted_fi]
        print(f"  {vname:<25} {', '.join(parts)}")
    print()


# ── Main ──

def main():
    t0 = time.time()
    print(f"[v17] Starting partial-month BF experiment (d{PARTIAL_N_DAYS})")
    print(f"[v17] Mem: {mem_mb():.0f} MB")

    all_results: dict[str, dict] = {}

    # Phase 1: Dev
    print("\n" + "=" * 60)
    print("  PHASE 1: DEV (12 groups)")
    print("=" * 60)

    for vname, spec in VARIANTS.items():
        t1 = time.time()
        print(f"\n[{vname}] Running {len(spec['features'])}f...")
        all_results[vname] = run_variant(vname, spec, holdout=False)
        print(f"[{vname}] Done in {time.time()-t1:.1f}s, mem={mem_mb():.0f} MB")

    print_comparison(all_results, "dev")
    print_feature_importance(all_results, "dev")

    # Phase 2: Holdout (for variants that improve on dev)
    print("\n" + "=" * 60)
    print("  PHASE 2: HOLDOUT (3 groups)")
    print("=" * 60)

    for vname, spec in VARIANTS.items():
        t1 = time.time()
        print(f"\n[{vname}] Running holdout {len(spec['features'])}f...")
        all_results[vname] = run_variant(vname, spec, holdout=True)
        print(f"[{vname}] Done in {time.time()-t1:.1f}s, mem={mem_mb():.0f} MB")

    print_comparison(all_results, "holdout")
    print_feature_importance(all_results, "holdout")

    # Per-group holdout breakdown
    print(f"\n{'='*130}")
    print("  HOLDOUT PER-GROUP BREAKDOWN")
    print(f"{'='*130}")
    for gid in HOLDOUT_GROUPS:
        print(f"\n  {gid}:")
        header = f"  {'Variant':<25}"
        for m in GROUP_A[:4]:
            header += f"  {m:>10}"
        print(header)
        print("  " + "-" * 80)
        for vname, res in all_results.items():
            pg = res.get("holdout_per_group", {}).get(gid, {})
            if not pg:
                continue
            row = f"  {vname:<25}"
            for m in GROUP_A[:4]:
                row += f"  {pg.get(m, 0):>10.4f}"
            print(row)

    # Save results
    out_dir = _PROJECT_ROOT / "registry" / "v17_partial_bf"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for vname, res in all_results.items():
        dev_agg = res.get("dev_agg", {})
        ho_agg = res.get("holdout_agg", {})
        out[vname] = {
            "dev_mean": dev_agg.get("mean", {}) if dev_agg else {},
            "dev_bottom2": dev_agg.get("bottom_2_mean", {}) if dev_agg else {},
            "holdout_mean": ho_agg.get("mean", {}) if ho_agg else {},
            "holdout_bottom2": ho_agg.get("bottom_2_mean", {}) if ho_agg else {},
            "holdout_per_group": res.get("holdout_per_group", {}),
            "dev_feat_imp": res.get("dev_feat_imp", {}),
            "holdout_feat_imp": res.get("holdout_feat_imp", {}),
        }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[v17] Results saved to {out_dir / 'metrics.json'}")

    elapsed = time.time() - t0
    print(f"\n[v17] Total walltime: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"[v17] Final mem: {mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
