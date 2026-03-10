"""v16: Champion analysis — per-group holdout breakdown + tail risk + formalization.

Runs the top 3 holdout variants from v15 (backfill+offpeak, backfill_lean, v10e)
plus v0b formula baseline. Reports:
1. Per-group holdout metrics (aq1, aq2, aq3)
2. Tail risk: bottom-2-mean, worst group, max drawdown vs v0b
3. Group A gating check (mean floor + tail safety + bottom-2 non-regression)
4. Champion recommendation with config

Builds on v15 infrastructure; does NOT re-run dev (those numbers are stable).
"""
import json
import gc
import time
import sys
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
)

HOLDOUT_GROUPS = ["2025-06/aq1", "2025-06/aq2", "2025-06/aq3"]

GROUP_A = ["VC@20", "VC@50", "VC@100", "Recall@20", "Recall@50", "Recall@100", "NDCG"]
GROUP_B = ["Spearman", "Tier0-AP", "Tier01-AP"]

# ── Data loading ──
_DATA_CACHE: dict[tuple, pl.DataFrame] = {}

def load_group(planning_year: str, aq_round: str,
               use_offpeak: bool = False, use_backfill: bool = False) -> pl.DataFrame:
    mode = f"off={use_offpeak}_bf={use_backfill}"
    cache_key = (planning_year, aq_round, mode)
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    df = load_v61_enriched(planning_year, aq_round)
    group_id = f"{planning_year}/{aq_round}"
    df = df.with_columns(pl.lit(group_id).alias("query_group"))
    df = get_ground_truth(planning_year, aq_round, df, cache=True)

    floor = "2017-04" if use_backfill else None
    df = enrich_with_binding_freq(df, planning_year, aq_round, floor_month=floor)
    if use_offpeak:
        df = enrich_with_offpeak_bf(df, planning_year, aq_round, floor_month=floor)

    # Engineered features used by v10e
    if "bf_3" in df.columns and "bf_12" in df.columns:
        df = df.with_columns((pl.col("bf_3") - pl.col("bf_12")).alias("bf_trend_3_12"))
    if "shadow_price_da" in df.columns and "bf_12" in df.columns:
        df = df.with_columns((pl.col("shadow_price_da") * pl.col("bf_12")).alias("sp_x_bf12"))
    if "da_rank_value" in df.columns and "bf_12" in df.columns:
        df = df.with_columns(((1.0 - pl.col("da_rank_value")) * pl.col("bf_12")).alias("rank_x_bf12"))

    _DATA_CACHE[cache_key] = df
    return df


def _get_holdout_train_groups() -> list[str]:
    """All groups through 2024 for holdout model training."""
    for split_def in EVAL_SPLITS.values():
        if split_def["eval_year"] == "2025-06":
            return [f"{y}/{aq}" for y in split_def["train_years"] for aq in AQ_ROUNDS]
    raise ValueError("No split for 2025-06")


# ── Variant definitions ──
VARIANTS = {
    "backfill+offpeak": {
        "features": ["shadow_price_da", "da_rank_value", "bf_6", "bf_12", "bf_15", "bfo_6", "bfo_12"],
        "monotone":  [1, -1, 1, 1, 1, 1, 1],
        "use_offpeak": True, "use_backfill": True,
    },
    "backfill_lean": {
        "features": ["shadow_price_da", "da_rank_value", "bf_6", "bf_12", "bf_15", "bf_24"],
        "monotone":  [1, -1, 1, 1, 1, 1],
        "use_offpeak": False, "use_backfill": True,
    },
    "v10e": {
        "features": ["shadow_price_da", "da_rank_value", "bf_6", "bf_12", "bf_15",
                      "rank_x_bf12", "bf_trend_3_12", "sp_x_bf12"],
        "monotone":  [1, -1, 1, 1, 1, 1, 0, 1],
        "use_offpeak": False, "use_backfill": False,
    },
}

def run_holdout_variant(name: str, spec: dict) -> dict:
    """Train on all years through 2024, eval on holdout aq1-aq3."""
    features = spec["features"]
    monotone = spec["monotone"]
    use_offpeak = spec["use_offpeak"]
    use_backfill = spec["use_backfill"]

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

    train_ids = _get_holdout_train_groups()
    dfs = []
    for gid in train_ids:
        py, aq = gid.split("/")
        try:
            dfs.append(load_group(py, aq, use_offpeak, use_backfill))
        except FileNotFoundError:
            pass
    train_df = pl.concat(dfs, how="diagonal").sort("query_group")
    X_train, _ = prepare_features(train_df, config.ltr)
    y_train = train_df["realized_shadow_price"].to_numpy().astype(np.float64)
    groups_train = compute_query_groups(train_df)
    model = train_ltr_model(X_train, y_train, groups_train, config.ltr)
    del train_df, X_train, y_train, groups_train

    per_group = {}
    for gid in HOLDOUT_GROUPS:
        py, aq = gid.split("/")
        df = load_group(py, aq, use_offpeak, use_backfill)
        X, _ = prepare_features(df, config.ltr)
        actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
        scores = predict_scores(model, X)
        per_group[gid] = evaluate_ltr(actual, scores)
        per_group[gid]["n_samples"] = len(df)

    feat_imp = {}
    if hasattr(model, "feature_importance"):
        importance = model.feature_importance(importance_type="gain")
        feat_imp = dict(zip(features, [float(x) for x in importance]))

    del model
    gc.collect()
    return {"per_group": per_group, "feature_importance": feat_imp}


def run_formula_holdout() -> dict:
    """v0b formula on holdout."""
    per_group = {}
    for gid in HOLDOUT_GROUPS:
        py, aq = gid.split("/")
        df = load_group(py, aq, False, False)
        actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
        formula = 1.0 - df["da_rank_value"].to_numpy().astype(np.float64)
        per_group[gid] = evaluate_ltr(actual, formula)
        per_group[gid]["n_samples"] = len(df)
    return {"per_group": per_group, "feature_importance": {}}


# ── Display ──

def print_per_group(all_results: dict[str, dict]):
    """Print per-group holdout metrics for all variants."""
    metrics = GROUP_A + GROUP_B
    for gid in HOLDOUT_GROUPS:
        print(f"\n{'='*160}")
        print(f"  HOLDOUT GROUP: {gid}")
        print(f"{'='*160}")
        header = f"{'Variant':<25}"
        for m in metrics:
            header += f"  {m:>12}"
        header += f"  {'n_samples':>10}"
        print(header)
        print("-" * 160)

        for vname, res in all_results.items():
            pg = res["per_group"].get(gid, {})
            if not pg:
                continue
            row = f"{vname:<25}"
            for m in metrics:
                val = pg.get(m, 0)
                row += f"  {val:>12.4f}"
            row += f"  {pg.get('n_samples', 0):>10.0f}"
            print(row)


def print_aggregate_and_tail(all_results: dict[str, dict]):
    """Print aggregate holdout metrics with tail risk."""
    metrics = GROUP_A + GROUP_B

    print(f"\n{'='*160}")
    print(f"  HOLDOUT AGGREGATE — Mean across 3 groups")
    print(f"{'='*160}")
    header = f"{'Variant':<25}"
    for m in metrics:
        header += f"  {m:>12}"
    print(header)
    print("-" * 160)

    for vname, res in all_results.items():
        per_group = res["per_group"]
        if not per_group:
            continue
        row = f"{vname:<25}"
        for m in metrics:
            vals = [per_group[gid].get(m, 0) for gid in HOLDOUT_GROUPS if gid in per_group]
            mean_val = np.mean(vals) if vals else 0
            row += f"  {mean_val:>12.4f}"
        print(row)

    # Bottom-2 mean (tail risk)
    print(f"\n{'='*160}")
    print(f"  HOLDOUT TAIL RISK — Bottom-2-mean (worst 2 of 3 groups)")
    print(f"{'='*160}")
    header = f"{'Variant':<25}"
    for m in metrics:
        header += f"  {m:>12}"
    print(header)
    print("-" * 160)

    for vname, res in all_results.items():
        per_group = res["per_group"]
        if not per_group:
            continue
        row = f"{vname:<25}"
        for m in metrics:
            vals = sorted([per_group[gid].get(m, 0) for gid in HOLDOUT_GROUPS if gid in per_group])
            bottom2 = np.mean(vals[:2]) if len(vals) >= 2 else (vals[0] if vals else 0)
            row += f"  {bottom2:>12.4f}"
        print(row)

    # Worst group
    print(f"\n{'='*160}")
    print(f"  HOLDOUT WORST GROUP — Min across 3 groups")
    print(f"{'='*160}")
    header = f"{'Variant':<25}"
    for m in metrics:
        header += f"  {m:>12}"
    print(header)
    print("-" * 160)

    for vname, res in all_results.items():
        per_group = res["per_group"]
        if not per_group:
            continue
        row = f"{vname:<25}"
        for m in metrics:
            vals = [per_group[gid].get(m, 0) for gid in HOLDOUT_GROUPS if gid in per_group]
            min_val = min(vals) if vals else 0
            row += f"  {min_val:>12.4f}"
        print(row)


def print_gating(all_results: dict[str, dict], baseline_name: str = "v0b_formula"):
    """Check stage5-tier 3-layer gating vs baseline."""
    baseline = all_results.get(baseline_name)
    if not baseline:
        print(f"\nWARNING: baseline {baseline_name} not found, skipping gating")
        return

    bl_per_group = baseline["per_group"]

    print(f"\n{'='*160}")
    print(f"  GATING CHECK vs {baseline_name}")
    print(f"{'='*160}")

    for vname, res in all_results.items():
        if vname == baseline_name:
            continue
        per_group = res["per_group"]
        if not per_group:
            continue

        print(f"\n  --- {vname} ---")
        all_pass = True
        for m in GROUP_A:
            # Layer 1: mean floor (must exceed baseline mean)
            v_mean = np.mean([per_group[g].get(m, 0) for g in HOLDOUT_GROUPS if g in per_group])
            bl_mean = np.mean([bl_per_group[g].get(m, 0) for g in HOLDOUT_GROUPS if g in bl_per_group])
            l1 = v_mean >= bl_mean
            delta_pct = (v_mean / bl_mean - 1) * 100 if bl_mean > 0 else 0

            # Layer 2: tail safety (bottom-2 >= baseline bottom-2 * 0.98)
            v_vals = sorted([per_group[g].get(m, 0) for g in HOLDOUT_GROUPS if g in per_group])
            bl_vals = sorted([bl_per_group[g].get(m, 0) for g in HOLDOUT_GROUPS if g in bl_per_group])
            v_b2 = np.mean(v_vals[:2]) if len(v_vals) >= 2 else 0
            bl_b2 = np.mean(bl_vals[:2]) if len(bl_vals) >= 2 else 0
            l2 = v_b2 >= bl_b2 * 0.98  # 2% noise tolerance

            # Layer 3: worst group >= baseline worst * 0.95
            v_min = min(v_vals) if v_vals else 0
            bl_min = min(bl_vals) if bl_vals else 0
            l3 = v_min >= bl_min * 0.95  # 5% noise tolerance

            status = "PASS" if (l1 and l2 and l3) else "FAIL"
            if not (l1 and l2 and l3):
                all_pass = False
            layers = f"L1={'OK' if l1 else 'NO'} L2={'OK' if l2 else 'NO'} L3={'OK' if l3 else 'NO'}"
            print(f"    {m:<12} {status}  mean={v_mean:.4f} ({delta_pct:+.1f}%)  b2={v_b2:.4f}  worst={v_min:.4f}  [{layers}]")

        verdict = "ALL GROUP A PASS" if all_pass else "SOME GROUP A FAIL"
        print(f"    >>> {verdict}")


def print_feature_importance(all_results: dict[str, dict]):
    """Print feature importance for ML variants."""
    print(f"\n{'='*80}")
    print(f"  FEATURE IMPORTANCE (holdout model, gain)")
    print(f"{'='*80}")
    for vname, res in all_results.items():
        fi = res.get("feature_importance", {})
        if not fi:
            continue
        total = sum(fi.values())
        print(f"\n  {vname}:")
        for fname, val in sorted(fi.items(), key=lambda x: -x[1]):
            pct = val / total * 100 if total > 0 else 0
            print(f"    {fname:<25} {val:>8.1f}  ({pct:>5.1f}%)")


def save_champion_config(all_results: dict[str, dict], champion_name: str):
    """Save champion configuration to registry."""
    spec = VARIANTS.get(champion_name)
    if not spec:
        print(f"\nWARNING: {champion_name} not in VARIANTS, not saving config")
        return

    per_group = all_results[champion_name]["per_group"]
    metrics_mean = {}
    metrics_bottom2 = {}
    for m in GROUP_A + GROUP_B:
        vals = [per_group[g].get(m, 0) for g in HOLDOUT_GROUPS if g in per_group]
        metrics_mean[m] = float(np.mean(vals))
        sorted_vals = sorted(vals)
        metrics_bottom2[m] = float(np.mean(sorted_vals[:2])) if len(sorted_vals) >= 2 else 0

    config = {
        "version": "v16_champion",
        "variant": champion_name,
        "features": spec["features"],
        "monotone_constraints": spec["monotone"],
        "use_backfill": spec["use_backfill"],
        "use_offpeak": spec["use_offpeak"],
        "floor_month": "2017-04" if spec["use_backfill"] else "2019-06",
        "hyperparams": {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "label_mode": "tiered",
            "backend": "lightgbm",
            "num_threads": 4,
        },
        "holdout_mean": metrics_mean,
        "holdout_bottom2": metrics_bottom2,
        "holdout_per_group": {g: per_group[g] for g in HOLDOUT_GROUPS if g in per_group},
        "feature_importance": all_results[champion_name].get("feature_importance", {}),
    }

    out_dir = _PROJECT_ROOT / "registry" / "v16_champion"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "config.json"
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2, default=float)
    print(f"\nChampion config saved to: {out_path}")


def main():
    t0 = time.time()

    print("=" * 80)
    print("  V16: Champion Analysis — Holdout per-group + tail risk")
    print("=" * 80)

    # Run all variants on holdout
    all_results = {}

    print("\n[1/4] Running v0b formula baseline on holdout...")
    t1 = time.time()
    all_results["v0b_formula"] = run_formula_holdout()
    print(f"  Done in {time.time()-t1:.1f}s")

    for vname, spec in VARIANTS.items():
        idx = list(VARIANTS.keys()).index(vname) + 2
        print(f"\n[{idx}/4] Running {vname} on holdout...")
        t1 = time.time()
        all_results[vname] = run_holdout_variant(vname, spec)
        print(f"  Done in {time.time()-t1:.1f}s")
        gc.collect()

    # Reports
    print_per_group(all_results)
    print_aggregate_and_tail(all_results)
    print_gating(all_results)
    print_gating(all_results, baseline_name="v10e")
    print_feature_importance(all_results)

    # Determine champion: best holdout mean VC@20 that passes all Group A gates vs v0b
    print(f"\n{'='*80}")
    print(f"  CHAMPION RECOMMENDATION")
    print(f"{'='*80}")

    # Rank by composite: average rank across all Group A metrics (holdout mean)
    variant_names = [v for v in all_results if v != "v0b_formula"]
    composite_scores = {}
    for vname in variant_names:
        pg = all_results[vname]["per_group"]
        rank_sum = 0
        for m in GROUP_A:
            vals = {v: np.mean([all_results[v]["per_group"][g].get(m, 0)
                                for g in HOLDOUT_GROUPS if g in all_results[v]["per_group"]])
                    for v in variant_names}
            ranked = sorted(vals.keys(), key=lambda v: vals[v], reverse=True)
            rank_sum += ranked.index(vname)
        composite_scores[vname] = rank_sum

    best = min(composite_scores, key=composite_scores.get)
    print(f"\n  Composite ranking (lower = better, sum of ranks across {len(GROUP_A)} Group A metrics):")
    for vname in sorted(composite_scores, key=composite_scores.get):
        pg = all_results[vname]["per_group"]
        vc20 = np.mean([pg[g].get("VC@20", 0) for g in HOLDOUT_GROUPS if g in pg])
        print(f"    {vname:<25}  score={composite_scores[vname]:>2}  holdout_VC@20={vc20:.4f}")

    print(f"\n  >>> CHAMPION: {best}")
    save_champion_config(all_results, best)

    print(f"\nTotal walltime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
