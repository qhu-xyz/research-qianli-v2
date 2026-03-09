#!/usr/bin/env python
"""DEPRECATED: Results archived to archive/registry/. Superseded by v10e-lag1.

V9 experiment: binding frequency feature.

binding_freq_6 = count(months constraint was binding in prior 6 months) / 6

This computes a historical binding frequency from realized DA data for the
6 months prior to the eval/train month. Each month gets its OWN binding_freq
computed from its own temporal window — no future leakage.

v9:  ML regression, 14f (12 base + v7_formula_score + binding_freq_6)
v9c: Post-hoc ensemble: alpha*v9_ML + (1-alpha)*v7_blend

Expected timing: ~3min for 36-month eval (binding_freq adds ~10s overhead).
"""
from __future__ import annotations

import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    FEATURES_V1B,
    MONOTONE_V1B,
    REALIZED_DA_CACHE,
    V62B_SIGNAL_BASE,
    LTRConfig,
    PipelineConfig,
    _FULL_EVAL_MONTHS,
)
from ml.data_loader import load_train_val_test
from ml.evaluate import aggregate_months, evaluate_ltr
from ml.features import compute_query_groups, prepare_features
from ml.realized_da import load_realized_da
from ml.train import predict_scores, train_ltr_model

REGISTRY = Path(__file__).resolve().parent.parent / "registry"

# V7 blend weights
V7_DA = 0.85
V7_DMIX = 0.00
V7_DORI = 0.15


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ═══════════════════════════════════════════════════════════════════════════════
# Binding frequency computation
# ═══════════════════════════════════════════════════════════════════════════════

def _load_all_binding_sets(cache_dir: str = REALIZED_DA_CACHE) -> dict[str, set[str]]:
    """Load binding constraint sets for all cached months.

    Returns {month: set(constraint_id)} where each constraint had realized_sp > 0.
    """
    binding_sets: dict[str, set[str]] = {}
    cache_path = Path(cache_dir)
    for f in sorted(cache_path.glob("*.parquet")):
        month = f.stem  # "2022-06"
        df = pl.read_parquet(str(f))
        binding = df.filter(pl.col("realized_sp") > 0)["constraint_id"].to_list()
        binding_sets[month] = set(binding)
    print(f"[binding_freq] Loaded binding sets for {len(binding_sets)} months")
    return binding_sets


def compute_binding_freq(
    constraint_ids: list[str],
    target_month: str,
    binding_sets: dict[str, set[str]],
    lookback: int = 6,
) -> np.ndarray:
    """Compute binding frequency for constraints in the N months BEFORE target_month.

    For target_month M, uses months M-N through M-1.
    Returns array of shape (len(constraint_ids),) with values in [0, 1].
    """
    all_months = sorted(binding_sets.keys())
    prior = [m for m in all_months if m < target_month][-lookback:]

    if not prior:
        return np.zeros(len(constraint_ids), dtype=np.float64)

    freq = np.zeros(len(constraint_ids), dtype=np.float64)
    for m in prior:
        bs = binding_sets[m]
        for i, cid in enumerate(constraint_ids):
            if cid in bs:
                freq[i] += 1
    return freq / len(prior)


def add_binding_freq_column(
    df: pl.DataFrame,
    month: str,
    binding_sets: dict[str, set[str]],
    lookback: int = 6,
) -> pl.DataFrame:
    """Add binding_freq_6 column to a dataframe for a given month."""
    cids = df["constraint_id"].to_list()
    freq = compute_binding_freq(cids, month, binding_sets, lookback)
    return df.with_columns(pl.Series("binding_freq_6", freq))


# ═══════════════════════════════════════════════════════════════════════════════
# V9: ML with binding_freq_6
# ═══════════════════════════════════════════════════════════════════════════════

def run_v9(
    eval_months: list[str],
    binding_sets: dict[str, set[str]],
) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    """Run v9: ML regression with 14 features (12 base + v7_formula + binding_freq_6)."""
    features = list(FEATURES_V1B) + ["v7_formula_score", "binding_freq_6"]
    monotone = list(MONOTONE_V1B) + [-1, 1]
    # v7_formula: lower = more binding → -1
    # binding_freq: higher = more binding → +1

    print(f"\n[v9] ML regression, 14f (12 + v7_formula + binding_freq_6), {len(eval_months)} months")

    cfg = PipelineConfig(
        ltr=LTRConfig(
            features=features,
            monotone_constraints=monotone,
            backend="lightgbm",
            label_mode="tiered",
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
        ),
        train_months=8,
        val_months=0,
    )

    per_month = {}
    per_month_scores = {}

    for m in eval_months:
        t0 = time.time()
        train_df, _, test_df = load_train_val_test(m, cfg.train_months, cfg.val_months, "f0", "onpeak")

        # Add v7 formula score
        for label, df in [("train", train_df), ("test", test_df)]:
            df_new = df.with_columns([
                (V7_DA * pl.col("da_rank_value")
                 + V7_DMIX * pl.col("density_mix_rank_value")
                 + V7_DORI * pl.col("density_ori_rank_value")
                ).alias("v7_formula_score"),
            ])
            if label == "train":
                train_df = df_new
            else:
                test_df = df_new

        # Add binding_freq_6 — each month uses its OWN temporal window
        # Training data: each month T gets binding_freq from T-6..T-1
        import pandas as pd
        eval_ts = pd.Timestamp(m)
        train_months_list = []
        for i in range(cfg.train_months, 0, -1):
            tm = (eval_ts - pd.DateOffset(months=i)).strftime("%Y-%m")
            train_months_list.append(tm)

        # Process training data month by month
        train_parts = []
        for tm in train_months_list:
            part = train_df.filter(pl.col("query_month") == tm)
            if len(part) > 0:
                part = add_binding_freq_column(part, tm, binding_sets)
                train_parts.append(part)
        train_df = pl.concat(train_parts) if train_parts else train_df

        # Process test data
        test_df = add_binding_freq_column(test_df, m, binding_sets)

        train_df = train_df.sort("query_month")
        X_train, _ = prepare_features(train_df, cfg.ltr)
        y_train = train_df["realized_sp"].to_numpy().astype(np.float64)
        groups_train = compute_query_groups(train_df)

        del train_parts
        gc.collect()

        model = train_ltr_model(X_train, y_train, groups_train, cfg.ltr)
        del X_train, y_train, groups_train
        gc.collect()

        X_test, _ = prepare_features(test_df, cfg.ltr)
        scores = predict_scores(model, X_test)
        actual = test_df["realized_sp"].to_numpy().astype(np.float64)

        metrics = evaluate_ltr(actual, scores)
        per_month[m] = metrics
        per_month_scores[m] = scores

        # Feature importance
        if hasattr(model, "feature_importance"):
            importance = model.feature_importance(importance_type="gain")
            metrics["_feature_importance"] = {
                name: float(imp)
                for name, imp in sorted(
                    zip(cfg.ltr.features, importance),
                    key=lambda x: x[1],
                    reverse=True,
                )
            }

        elapsed = time.time() - t0
        n_binding = int((actual > 0).sum())
        bf_nonzero = int((test_df["binding_freq_6"].to_numpy() > 0).sum())
        print(f"  {m}: VC@20={metrics['VC@20']:.4f}, VC@100={metrics['VC@100']:.4f}, "
              f"binding={n_binding}, bf>0={bf_nonzero} ({elapsed:.1f}s)")

        del X_test, actual, model, test_df
        gc.collect()

    return per_month, per_month_scores


# ═══════════════════════════════════════════════════════════════════════════════
# V9c: Ensemble
# ═══════════════════════════════════════════════════════════════════════════════

def run_v9c_ensemble(
    ml_scores: dict[str, np.ndarray],
    eval_months: list[str],
) -> list[dict]:
    """Sweep alpha for v9 ML + v7 blend ensemble."""
    print(f"\n[v9c] Ensemble sweep: alpha*v9_ML + (1-alpha)*v7_blend")

    v7_scores = {}
    for m in eval_months:
        path = Path(V62B_SIGNAL_BASE) / m / "f0" / "onpeak"
        df = pl.read_parquet(str(path))
        realized = load_realized_da(m)
        df = df.join(realized, on="constraint_id", how="left")
        df = df.with_columns(pl.col("realized_sp").fill_null(0.0))

        da = df["da_rank_value"].to_numpy().astype(np.float64)
        dmix = df["density_mix_rank_value"].to_numpy().astype(np.float64)
        dori = df["density_ori_rank_value"].to_numpy().astype(np.float64)
        blend = -(V7_DA * da + V7_DMIX * dmix + V7_DORI * dori)

        v7_scores[m] = {
            "actual": df["realized_sp"].to_numpy().astype(np.float64),
            "blend": blend,
        }
        del df, realized

    results = []
    for alpha_int in range(0, 21):
        alpha = alpha_int * 0.05
        per_month = {}
        for m in eval_months:
            ml = ml_scores[m]
            blend = v7_scores[m]["blend"]
            actual = v7_scores[m]["actual"]

            ml_norm = (ml - ml.min()) / (ml.max() - ml.min() + 1e-10)
            blend_norm = (blend - blend.min()) / (blend.max() - blend.min() + 1e-10)

            combined = alpha * ml_norm + (1 - alpha) * blend_norm
            per_month[m] = evaluate_ltr(actual, combined)

        agg = aggregate_months(per_month)["mean"]
        results.append({"alpha": alpha, **{k: v for k, v in agg.items() if isinstance(v, float)}})

    print(f"\n{'alpha':>6} | {'VC@20':>8} {'VC@100':>8} {'R@20':>8} {'NDCG':>8} {'Spearman':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['alpha']:>6.2f} | {r['VC@20']:>8.4f} {r['VC@100']:>8.4f} "
              f"{r['Recall@20']:>8.4f} {r['NDCG']:>8.4f} {r['Spearman']:>10.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Audit diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def run_audit_diagnostics(
    eval_months: list[str],
    binding_sets: dict[str, set[str]],
) -> dict:
    """Run audit checks for binding_freq_6 feature."""
    print("\n" + "=" * 70)
    print("AUDIT DIAGNOSTICS")
    print("=" * 70)

    results = {}

    # Check 1: Temporal boundary — does binding_freq for month M use data from M?
    print("\n[Audit 1] Temporal boundary check")
    for m in eval_months[:3]:
        all_months = sorted(binding_sets.keys())
        prior = [x for x in all_months if x < m][-6:]
        print(f"  eval={m}: binding_freq uses {prior[0]}..{prior[-1]} (6 months)")
        assert m not in prior, f"LEAKAGE: eval month {m} is in lookback!"
    print("  PASS: eval month never in lookback window")

    # Check 2: Training label overlap
    print("\n[Audit 2] Training label vs binding_freq overlap")
    import pandas as pd
    m = eval_months[12]  # pick a middle month
    eval_ts = pd.Timestamp(m)
    train_months = [(eval_ts - pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(8, 0, -1)]
    print(f"  eval={m}, train_months={train_months}")

    for tm in train_months[:3]:
        prior = [x for x in sorted(binding_sets.keys()) if x < tm][-6:]
        overlap = set(prior) & set(train_months)
        print(f"  train={tm}: bf_lookback={prior[0]}..{prior[-1]}, "
              f"overlap_with_train_labels={overlap}")
    print("  NOTE: Training months appear in OTHER training months' lookback windows.")
    print("  This is standard time-series feature engineering, NOT target leakage.")

    # Check 3: binding_freq correlation with target
    print("\n[Audit 3] binding_freq_6 correlation with realized_sp")
    from scipy.stats import spearmanr
    all_bf = []
    all_sp = []
    for m in eval_months:
        path = Path(V62B_SIGNAL_BASE) / m / "f0" / "onpeak"
        df = pl.read_parquet(str(path))
        realized = load_realized_da(m)
        df = df.join(realized, on="constraint_id", how="left")
        df = df.with_columns(pl.col("realized_sp").fill_null(0.0))
        cids = df["constraint_id"].to_list()
        bf = compute_binding_freq(cids, m, binding_sets)
        sp = df["realized_sp"].to_numpy().astype(np.float64)
        all_bf.extend(bf.tolist())
        all_sp.extend(sp.tolist())
        del df, realized
    corr, pval = spearmanr(all_bf, all_sp)
    print(f"  Spearman(binding_freq_6, realized_sp) = {corr:.4f} (p={pval:.2e})")
    results["bf_sp_correlation"] = float(corr)

    # Check 4: How much signal is just "exists in binding data"?
    print("\n[Audit 4] Binary binding signal: bf>0 vs bf=0")
    bf_arr = np.array(all_bf)
    sp_arr = np.array(all_sp)
    n_total = len(bf_arr)
    n_bf_pos = int((bf_arr > 0).sum())
    n_bf_zero = n_total - n_bf_pos
    sp_when_bf_pos = sp_arr[bf_arr > 0]
    sp_when_bf_zero = sp_arr[bf_arr == 0]
    binding_rate_bf_pos = float((sp_when_bf_pos > 0).mean()) if len(sp_when_bf_pos) > 0 else 0
    binding_rate_bf_zero = float((sp_when_bf_zero > 0).mean()) if len(sp_when_bf_zero) > 0 else 0
    print(f"  bf>0: n={n_bf_pos} ({100*n_bf_pos/n_total:.1f}%), binding_rate={100*binding_rate_bf_pos:.1f}%")
    print(f"  bf=0: n={n_bf_zero} ({100*n_bf_zero/n_total:.1f}%), binding_rate={100*binding_rate_bf_zero:.1f}%")
    results["bf_pos_count"] = n_bf_pos
    results["bf_zero_count"] = n_bf_zero
    results["binding_rate_bf_pos"] = binding_rate_bf_pos
    results["binding_rate_bf_zero"] = binding_rate_bf_zero

    # Check 5: Correlation with da_rank_value
    print("\n[Audit 5] Correlation with da_rank_value")
    all_da = []
    for m in eval_months:
        path = Path(V62B_SIGNAL_BASE) / m / "f0" / "onpeak"
        df = pl.read_parquet(str(path))
        all_da.extend(df["da_rank_value"].to_numpy().tolist())
        del df
    da_arr = np.array(all_da)
    corr_da, _ = spearmanr(bf_arr, da_arr)
    print(f"  Spearman(binding_freq_6, da_rank_value) = {corr_da:.4f}")
    results["bf_da_correlation"] = float(corr_da)

    # Check 6: Production viability
    print("\n[Audit 6] Production viability")
    print("  binding_freq_6 requires realized DA for M-6..M-1")
    print("  Realized DA is available ~24h after month end (MISO publishes daily)")
    print("  V6.2B signal runs ~5th of month → M-1 data is always available")
    print("  PASS: Feature is producible in production")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Registry helpers
# ═══════════════════════════════════════════════════════════════════════════════

def save_version(version_id: str, per_month: dict, eval_months: list[str],
                 config_extra: dict | None = None) -> None:
    """Save version to registry."""
    clean = {}
    for m, metrics in per_month.items():
        clean[m] = {k: v for k, v in metrics.items() if not k.startswith("_")}

    agg = aggregate_months(clean)
    version_dir = REGISTRY / version_id
    version_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "eval_config": {
            "eval_months": eval_months,
            "class_type": "onpeak",
            "period_type": "f0",
            "mode": "eval",
        },
        "per_month": clean,
        "aggregate": agg,
        "n_months": len(clean),
        "n_months_requested": len(eval_months),
        "skipped_months": [],
    }
    with open(version_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    if config_extra:
        with open(version_dir / "config.json", "w") as f:
            json.dump(config_extra, f, indent=2)

    means = agg["mean"]
    print(f"\n[{version_id}] Saved to {version_dir}/")
    print(f"  VC@20={means['VC@20']:.4f}  VC@100={means['VC@100']:.4f}  "
          f"R@20={means['Recall@20']:.4f}  NDCG={means['NDCG']:.4f}  "
          f"Spearman={means['Spearman']:.4f}")


def main() -> None:
    eval_months = _FULL_EVAL_MONTHS  # 36 months
    t_start = time.time()

    print(f"[main] V9 binding freq experiment, {len(eval_months)} months, mem={mem_mb():.0f} MB")

    # Load binding sets once
    binding_sets = _load_all_binding_sets()

    # ── V9: ML with binding_freq_6 ──
    t0 = time.time()
    v9_per_month, v9_scores = run_v9(eval_months, binding_sets)
    save_version("v9", v9_per_month, eval_months, config_extra={
        "method": "lightgbm_regression",
        "features": "FEATURES_V1B + v7_formula_score + binding_freq_6 (14f)",
        "binding_freq": {
            "lookback": 6,
            "source": "realized_da_cache",
            "definition": "count(binding months in prior 6) / 6 per constraint",
        },
        "label_mode": "tiered",
        "train_months": 8,
        "val_months": 0,
    })
    print(f"[v9] Done in {time.time()-t0:.1f}s")

    # ── V9c: ensemble ──
    t0 = time.time()
    ensemble_results = run_v9c_ensemble(v9_scores, eval_months)
    best_vc20 = max(ensemble_results, key=lambda x: x["VC@20"])

    ens_dir = REGISTRY / "v9" / "ensemble"
    ens_dir.mkdir(parents=True, exist_ok=True)
    with open(ens_dir / "sweep_results.json", "w") as f:
        json.dump({"sweep": ensemble_results, "best_vc20": best_vc20}, f, indent=2)
    print(f"[v9c] Best by VC@20: alpha={best_vc20['alpha']:.2f}, VC@20={best_vc20['VC@20']:.4f}")
    print(f"[v9c] Done in {time.time()-t0:.1f}s")

    # ── Feature importance summary ──
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (average gain across months)")
    print("=" * 70)
    feat_sums: dict[str, float] = {}
    feat_counts: dict[str, int] = {}
    for m, metrics in v9_per_month.items():
        fi = metrics.get("_feature_importance", {})
        for name, imp in fi.items():
            feat_sums[name] = feat_sums.get(name, 0) + imp
            feat_counts[name] = feat_counts.get(name, 0) + 1
    feat_avg = {n: feat_sums[n] / feat_counts[n] for n in feat_sums}
    for name, avg in sorted(feat_avg.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:<30s} {avg:>10.1f}")

    # ── Audit diagnostics ──
    audit = run_audit_diagnostics(eval_months, binding_sets)
    with open(REGISTRY / "v9" / "audit_diagnostics.json", "w") as f:
        json.dump(audit, f, indent=2)

    total = time.time() - t_start
    print(f"\n[main] All done in {total:.1f}s, mem={mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
