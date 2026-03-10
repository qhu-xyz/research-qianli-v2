"""v14: Combine ALL signal sources for maximum VC@20.

Current best (v10e, VC@20=0.3389) uses only shadow_price_da + da_rank_value + onpeak BF.
That's 81% importance on just 2 base features. Untapped signals:

1. Offpeak BF (bfo_N): constraints binding in offpeak hours — complementary to onpeak
2. Density simulation (prob_exceed_110/100/90): Monte Carlo exceedance from spice6
3. Structural (constraint_limit): physical limit — never combined with BF
4. Flow features (mean_branch_max, ori_mean, mix_mean): V6.1 base features
5. Formula output (rank_ori): V6.1 formula score as feature

Strategy: systematic feature addition on top of the v10e base to find what stacks.
All use pre-backfill (DA_FLOOR_MONTH="2019-06") since backfill was proven to hurt.
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
    DEFAULT_EVAL_GROUPS, HOLDOUT_EVAL_GROUPS,
)
from ml.data_loader import load_v61_enriched
from ml.evaluate import evaluate_ltr, aggregate_months
from ml.ground_truth import get_ground_truth
from ml.features import prepare_features, compute_query_groups
from ml.train import train_ltr_model, predict_scores
from ml.binding_freq import enrich_with_binding_freq, enrich_with_offpeak_bf

REGISTRY_DIR = _PROJECT_ROOT / "registry"
DISPLAY_METRICS = ["VC@20", "VC@50", "VC@100", "Recall@20", "Recall@50", "Recall@100", "NDCG", "Spearman"]

# ── Year-level training/assessment ──
_year_groups: dict[str, list[str]] = OrderedDict()
for _g in DEFAULT_EVAL_GROUPS:
    _y = _g.split("/")[0]
    _year_groups.setdefault(_y, []).append(_g)


def _get_train_groups(eval_year: str) -> list[str]:
    for split_def in EVAL_SPLITS.values():
        if split_def["eval_year"] == eval_year:
            return [f"{y}/{aq}" for y in split_def["train_years"] for aq in AQ_ROUNDS]
    raise ValueError(f"No split for year: {eval_year}")


# ── Data loading ──

_DATA_CACHE: dict[tuple[str, str], pl.DataFrame] = {}


def load_group_full(planning_year: str, aq_round: str) -> pl.DataFrame:
    """Load data with ALL features: base V6.1 + spice6 density + onpeak BF + offpeak BF."""
    cache_key = (planning_year, aq_round)
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    # Base enriched data (V6.1 + spice6 density + structural)
    df = load_v61_enriched(planning_year, aq_round)
    group_id = f"{planning_year}/{aq_round}"
    df = df.with_columns(pl.lit(group_id).alias("query_group"))
    df = get_ground_truth(planning_year, aq_round, df, cache=True)

    # Onpeak BF (pre-backfill floor)
    df = enrich_with_binding_freq(df, planning_year, aq_round, floor_month=None)

    # Offpeak BF
    df = enrich_with_offpeak_bf(df, planning_year, aq_round, floor_month=None)

    # Engineered features
    if "bf_3" in df.columns and "bf_12" in df.columns:
        df = df.with_columns((pl.col("bf_3") - pl.col("bf_12")).alias("bf_trend_3_12"))
    if "shadow_price_da" in df.columns and "bf_12" in df.columns:
        df = df.with_columns((pl.col("shadow_price_da") * pl.col("bf_12")).alias("sp_x_bf12"))
    if "da_rank_value" in df.columns and "bf_12" in df.columns:
        df = df.with_columns(((1.0 - pl.col("da_rank_value")) * pl.col("bf_12")).alias("rank_x_bf12"))

    # Offpeak interactions
    if "shadow_price_da" in df.columns and "bfo_12" in df.columns:
        df = df.with_columns((pl.col("shadow_price_da") * pl.col("bfo_12")).alias("sp_x_bfo12"))
    if "bf_12" in df.columns and "bfo_12" in df.columns:
        df = df.with_columns((pl.col("bf_12") - pl.col("bfo_12")).alias("bf_peak_spread_12"))

    _DATA_CACHE[cache_key] = df
    return df


# ── Training ──

def train_and_assess(variant_name: str, features: list[str], monotone: list[int],
                     n_estimators: int = 200, learning_rate: float = 0.03,
                     num_leaves: int = 31) -> dict:
    """Train and assess a variant across all 12 dev groups."""
    config = PipelineConfig(
        ltr=LTRConfig(
            features=features,
            monotone_constraints=monotone,
            backend="lightgbm",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=0.8,
            colsample_bytree=0.8,
            label_mode="tiered",
        ),
    )

    per_group = {}
    model_cache = {}

    for year, groups_in_year in _year_groups.items():
        if year not in model_cache:
            train_group_ids = _get_train_groups(year)
            dfs = []
            for gid in train_group_ids:
                py, aq = gid.split("/")
                try:
                    dfs.append(load_group_full(py, aq))
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
            df = load_group_full(py, aq)
            X, _ = prepare_features(df, config.ltr)
            actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
            scores = predict_scores(model, X)
            per_group[gid] = evaluate_ltr(actual, scores)

    # Feature importance from last model
    last_model = list(model_cache.values())[-1]
    feat_imp = {}
    if hasattr(last_model, "feature_importance"):
        importance = last_model.feature_importance(importance_type="gain")
        feat_imp = dict(zip(config.ltr.features, [float(x) for x in importance]))

    agg = aggregate_months(per_group)
    del model_cache
    gc.collect()

    return {
        "per_month": per_group,
        "aggregate": agg,
        "feature_importance": feat_imp,
        "n_months": len(per_group),
        "config": config.to_dict(),
    }


def _minmax(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.full_like(arr, 0.5)
    return (arr - mn) / (mx - mn)


def run_blending(best_name: str, best_features: list[str], best_monotone: list[int],
                 best_params: dict) -> dict:
    """Run formula blending for best variant."""
    config = PipelineConfig(
        ltr=LTRConfig(
            features=best_features,
            monotone_constraints=best_monotone,
            backend="lightgbm",
            label_mode="tiered",
            **best_params,
        ),
    )

    predictions: dict[str, np.ndarray] = {}
    test_data: dict[str, pl.DataFrame] = {}
    model_cache = {}

    for year, groups_in_year in _year_groups.items():
        if year not in model_cache:
            train_group_ids = _get_train_groups(year)
            dfs = []
            for gid in train_group_ids:
                py, aq = gid.split("/")
                try:
                    dfs.append(load_group_full(py, aq))
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
            df = load_group_full(py, aq)
            test_data[gid] = df
            X, _ = prepare_features(df, config.ltr)
            predictions[gid] = predict_scores(model, X)

    del model_cache
    gc.collect()

    blend_results = {}
    for alpha_pct in [40, 50, 60, 70, 80, 90, 100]:
        alpha = alpha_pct / 100.0
        name = f"blend_{best_name}_a{alpha_pct}"
        per_month = {}
        for gid, df in test_data.items():
            actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
            formula = 1.0 - df["da_rank_value"].to_numpy().astype(np.float64)
            blended = alpha * _minmax(predictions[gid]) + (1.0 - alpha) * _minmax(formula)
            per_month[gid] = evaluate_ltr(actual, blended)
        agg = aggregate_months(per_month)
        blend_results[name] = {"per_month": per_month, "aggregate": agg, "n_months": len(per_month)}

    # v0b reference
    per_month = {}
    for gid, df in test_data.items():
        actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
        formula = 1.0 - df["da_rank_value"].to_numpy().astype(np.float64)
        per_month[gid] = evaluate_ltr(actual, formula)
    blend_results["v0b_ref"] = {"per_month": per_month, "aggregate": aggregate_months(per_month), "n_months": len(per_month)}

    del test_data
    gc.collect()
    return blend_results


def print_results(results: dict[str, dict]):
    print(f"\n{'='*160}")
    print("  V14 COMBINED SIGNALS RESULTS (mean over 12 dev groups)")
    print(f"{'='*160}")

    header = f"{'Variant':<50} {'#f':>3}"
    for m in DISPLAY_METRICS:
        header += f"  {m:>10}"
    print(header)
    print("-" * 160)

    entries = sorted(results.items(), key=lambda x: x[1]["aggregate"]["mean"].get("VC@20", 0), reverse=True)

    for name, res in entries:
        means = res["aggregate"]["mean"]
        fi = res.get("feature_importance", {})
        nf = len(fi) if fi else "?"
        row = f"{name:<50} {nf:>3}"
        for m in DISPLAY_METRICS:
            row += f"  {means.get(m, 0):>10.4f}"

        if fi:
            total = sum(fi.values())
            if total > 0:
                top3 = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:3]
                fi_str = ", ".join(f"{k}={v/total*100:.0f}%" for k, v in top3)
                row += f"  [{fi_str}]"
        print(row)

    print(f"\n--- Reference ---")
    for vid in ["v0b", "v8b"]:
        p = REGISTRY_DIR / vid / "metrics.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            means = d["aggregate"]["mean"]
            row = f"{vid + ' (registry)':<50} {'':>3}"
            for m in DISPLAY_METRICS:
                row += f"  {means.get(m, 0):>10.4f}"
            print(row)


def main():
    t_total = time.time()
    all_results = {}

    # ── v10e reference (reproduce) ──
    v10e_f = ["shadow_price_da", "da_rank_value", "bf_6", "bf_12", "bf_15",
              "rank_x_bf12", "bf_trend_3_12", "sp_x_bf12"]
    v10e_m = [1, -1, 1, 1, 1, 1, 0, 1]

    print(f"\n{'='*80}\n  v14_ref: reproduce v10e baseline\n{'='*80}")
    t0 = time.time()
    r = train_and_assess("v14_ref", v10e_f, v10e_m)
    all_results["v14_ref_v10e"] = r
    print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── A: v10e + offpeak BF ──
    name = "v14a_v10e+offpeak"
    features = v10e_f + ["bfo_6", "bfo_12", "bfo_24"]
    monotone = v10e_m + [1, 1, 1]
    print(f"\n{'='*80}\n  {name}: add offpeak BF\n{'='*80}")
    t0 = time.time()
    r = train_and_assess(name, features, monotone)
    all_results[name] = r
    print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── B: v10e + offpeak interactions ──
    name = "v14b_v10e+offpeak_int"
    features = v10e_f + ["bfo_12", "sp_x_bfo12", "bf_peak_spread_12"]
    monotone = v10e_m + [1, 1, 0]
    print(f"\n{'='*80}\n  {name}: add offpeak BF + interactions\n{'='*80}")
    t0 = time.time()
    r = train_and_assess(name, features, monotone)
    all_results[name] = r
    print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── C: v10e + density features ──
    name = "v14c_v10e+density"
    features = v10e_f + ["prob_exceed_110", "prob_exceed_100", "prob_exceed_90"]
    monotone = v10e_m + [1, 1, 1]
    print(f"\n{'='*80}\n  {name}: add density exceedance\n{'='*80}")
    t0 = time.time()
    r = train_and_assess(name, features, monotone)
    all_results[name] = r
    print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── D: v10e + structural (constraint_limit) ──
    name = "v14d_v10e+structural"
    features = v10e_f + ["constraint_limit"]
    monotone = v10e_m + [0]
    print(f"\n{'='*80}\n  {name}: add constraint_limit\n{'='*80}")
    t0 = time.time()
    r = train_and_assess(name, features, monotone)
    all_results[name] = r
    print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── E: v10e + flow features ──
    name = "v14e_v10e+flows"
    features = v10e_f + ["mean_branch_max", "ori_mean", "mix_mean"]
    monotone = v10e_m + [1, 1, 1]
    print(f"\n{'='*80}\n  {name}: add flow features\n{'='*80}")
    t0 = time.time()
    r = train_and_assess(name, features, monotone)
    all_results[name] = r
    print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── F: v10e + rank_ori (formula as feature) ──
    name = "v14f_v10e+formula"
    features = v10e_f + ["rank_ori"]
    monotone = v10e_m + [-1]
    print(f"\n{'='*80}\n  {name}: add V6.1 formula output\n{'='*80}")
    t0 = time.time()
    r = train_and_assess(name, features, monotone)
    all_results[name] = r
    print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── G: v10e + density_rank features ──
    name = "v14g_v10e+density_ranks"
    features = v10e_f + ["density_mix_rank_value", "density_ori_rank_value"]
    monotone = v10e_m + [-1, -1]
    print(f"\n{'='*80}\n  {name}: add density rank values\n{'='*80}")
    t0 = time.time()
    r = train_and_assess(name, features, monotone)
    all_results[name] = r
    print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── H: Kitchen sink — everything ──
    name = "v14h_kitchen_sink"
    features = v10e_f + [
        "bfo_6", "bfo_12", "bfo_24", "sp_x_bfo12", "bf_peak_spread_12",
        "prob_exceed_110", "prob_exceed_100",
        "constraint_limit",
        "mean_branch_max",
        "rank_ori",
    ]
    monotone = v10e_m + [1, 1, 1, 1, 0, 1, 1, 0, 1, -1]
    print(f"\n{'='*80}\n  {name}: all features combined\n{'='*80}")
    t0 = time.time()
    r = train_and_assess(name, features, monotone)
    all_results[name] = r
    print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── I: Best additive features only ──
    # After seeing A-H, combine only the features that helped
    # For now, pre-compute with likely winners: offpeak + density
    name = "v14i_offpeak+density"
    features = v10e_f + ["bfo_12", "bfo_24", "prob_exceed_110"]
    monotone = v10e_m + [1, 1, 1]
    print(f"\n{'='*80}\n  {name}: offpeak + density\n{'='*80}")
    t0 = time.time()
    r = train_and_assess(name, features, monotone)
    all_results[name] = r
    print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── J: Lean SP + offpeak BF (test if offpeak alone adds signal) ──
    name = "v14j_lean+offpeak"
    features = ["shadow_price_da", "da_rank_value", "bf_6", "bf_12", "bfo_6", "bfo_12"]
    monotone = [1, -1, 1, 1, 1, 1]
    print(f"\n{'='*80}\n  {name}: lean SP + onpeak/offpeak BF\n{'='*80}")
    t0 = time.time()
    r = train_and_assess(name, features, monotone)
    all_results[name] = r
    print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── K: SP + density only (no BF, test density value) ──
    name = "v14k_sp+density"
    features = ["shadow_price_da", "da_rank_value", "prob_exceed_110", "prob_exceed_100", "prob_exceed_90"]
    monotone = [1, -1, 1, 1, 1]
    print(f"\n{'='*80}\n  {name}: SP + density only (no BF)\n{'='*80}")
    t0 = time.time()
    r = train_and_assess(name, features, monotone)
    all_results[name] = r
    print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── L: Hyperparameter sweep on v10e ──
    for n_est, lr, nl in [(300, 0.02, 31), (200, 0.03, 63), (400, 0.01, 31), (200, 0.05, 15)]:
        name = f"v14l_v10e_t{n_est}_lr{lr}_nl{nl}"
        print(f"\n{'='*80}\n  {name}\n{'='*80}")
        t0 = time.time()
        r = train_and_assess(name, v10e_f, v10e_m, n_estimators=n_est,
                             learning_rate=lr, num_leaves=nl)
        all_results[name] = r
        print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── Print all results ──
    print_results(all_results)

    # ── Find best and run blending ──
    best_name = max(all_results.keys(), key=lambda k: all_results[k]["aggregate"]["mean"]["VC@20"])
    best_vc20 = all_results[best_name]["aggregate"]["mean"]["VC@20"]
    ref_vc20 = all_results["v14_ref_v10e"]["aggregate"]["mean"]["VC@20"]
    print(f"\n  BEST ML: {best_name} (VC@20={best_vc20:.4f}, vs v10e ref={ref_vc20:.4f}, delta={best_vc20-ref_vc20:+.4f})")

    # Run blending for best
    best_r = all_results[best_name]
    best_cfg = best_r.get("config", {}).get("ltr", {})
    best_features = best_cfg.get("features", v10e_f)
    best_monotone = best_cfg.get("monotone_constraints", v10e_m)

    print(f"\n{'='*80}\n  BLENDING: {best_name} + v0b formula\n{'='*80}")
    blend_results = run_blending(
        best_name, best_features, best_monotone,
        {
            "n_estimators": best_cfg.get("n_estimators", 200),
            "learning_rate": best_cfg.get("learning_rate", 0.03),
            "num_leaves": best_cfg.get("num_leaves", 31),
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    )

    print(f"\n{'='*140}")
    print(f"  BLENDING RESULTS")
    print(f"{'='*140}")
    header = f"{'Blend':<50}"
    for m in DISPLAY_METRICS:
        header += f"  {m:>10}"
    print(header)
    print("-" * 140)
    for name, res in sorted(blend_results.items(),
                            key=lambda x: x[1]["aggregate"]["mean"]["VC@20"], reverse=True):
        means = res["aggregate"]["mean"]
        row = f"{name:<50}"
        for m in DISPLAY_METRICS:
            row += f"  {means.get(m, 0):>10.4f}"
        print(row)

    # Per-group comparison for top 3 + blends
    top3 = sorted(all_results.keys(), key=lambda k: all_results[k]["aggregate"]["mean"]["VC@20"], reverse=True)[:3]
    print(f"\n{'='*120}")
    print(f"  PER-GROUP VC@20 (top 3 ML vs v10e ref)")
    print(f"{'='*120}")
    header = f"{'Group':<15} {'v10e_ref':>12}"
    for n in top3:
        short = n.replace("v14", "").replace("_", " ").strip()[:12]
        header += f" {short:>12}"
    print(header)
    print("-" * 120)

    for gid in sorted(DEFAULT_EVAL_GROUPS):
        row = f"{gid:<15}"
        ref_val = all_results["v14_ref_v10e"]["per_month"][gid]["VC@20"]
        row += f" {ref_val:>12.4f}"
        for n in top3:
            val = all_results[n]["per_month"][gid]["VC@20"]
            delta = val - ref_val
            marker = "+" if delta > 0.01 else ("-" if delta < -0.01 else " ")
            row += f" {val:>9.4f}{marker:>3}"
        print(row)

    total = time.time() - t_total
    print(f"\n[main] Total walltime: {total:.1f}s")

    # Save
    save_dir = REGISTRY_DIR / "v14_combined_signals"
    save_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, res in {**all_results, **blend_results}.items():
        summary[name] = {
            "mean": res["aggregate"]["mean"],
            "bottom_2_mean": res["aggregate"]["bottom_2_mean"],
            "feature_importance": res.get("feature_importance", {}),
        }
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Also save best overall
    best_blend_name = max(blend_results.keys(), key=lambda k: blend_results[k]["aggregate"]["mean"]["VC@20"])
    best_blend_vc20 = blend_results[best_blend_name]["aggregate"]["mean"]["VC@20"]
    overall_best = best_blend_name if best_blend_vc20 > best_vc20 else best_name
    overall_vc20 = max(best_vc20, best_blend_vc20)
    print(f"\n  OVERALL BEST: {overall_best} (VC@20={overall_vc20:.4f})")
    print(f"  vs v0b baseline (0.2997): +{(overall_vc20-0.2997)/0.2997*100:.1f}%")
    print(f"  vs v10e (0.3389): {overall_vc20-0.3389:+.4f}")
    print(f"\n[main] Results saved to {save_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
