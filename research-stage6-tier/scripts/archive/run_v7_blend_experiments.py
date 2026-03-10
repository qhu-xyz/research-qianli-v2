#!/usr/bin/env python
"""V7+ experiments: optimized blend baseline + ML rebuilds.

v7:  Optimized formula blend (85/0/15) — new baseline
v8a: ML regression, 12f, tiered labels (same as v6a, measured against v7)
v8b: ML regression, 13f with v7 formula as feature (v6b rebuilt with better formula)
v8c: Post-hoc ensemble sweep: alpha*ML + (1-alpha)*v7_blend

Expected timing: v7 <5s, v8a ~2min, v8b ~2min, v8c <1s (reuses v8a scores)
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
    _DEFAULT_EVAL_MONTHS,
    _FULL_EVAL_MONTHS,
)
from ml.data_loader import load_train_val_test
from ml.evaluate import aggregate_months, evaluate_ltr
from ml.features import compute_query_groups, prepare_features
from ml.realized_da import load_realized_da
from ml.train import predict_scores, train_ltr_model

REGISTRY = Path(__file__).resolve().parent.parent / "registry"


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ── V7 blend weights ──
V7_DA = 0.85
V7_DMIX = 0.00
V7_DORI = 0.15


def v7_blend_score(
    da_rank_value: np.ndarray,
    density_mix_rank_value: np.ndarray,
    density_ori_rank_value: np.ndarray,
) -> np.ndarray:
    """V7 optimized blend: 85% da + 0% dmix + 15% dori. Lower = more binding."""
    return V7_DA * np.asarray(da_rank_value, dtype=float) \
         + V7_DMIX * np.asarray(density_mix_rank_value, dtype=float) \
         + V7_DORI * np.asarray(density_ori_rank_value, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
# V7: Optimized blend baseline
# ═══════════════════════════════════════════════════════════════════════════════

def run_v7(eval_months: list[str]) -> dict[str, dict]:
    """Evaluate v7 formula on given months. Returns per_month metrics."""
    print(f"\n[v7] Optimized blend ({V7_DA}/{V7_DMIX}/{V7_DORI}), {len(eval_months)} months")
    per_month = {}
    for m in eval_months:
        path = Path(V62B_SIGNAL_BASE) / m / "f0" / "onpeak"
        df = pl.read_parquet(str(path))
        realized = load_realized_da(m)
        df = df.join(realized, on="constraint_id", how="left")
        df = df.with_columns(pl.col("realized_sp").fill_null(0.0))

        actual = df["realized_sp"].to_numpy().astype(np.float64)
        scores = -v7_blend_score(
            df["da_rank_value"].to_numpy(),
            df["density_mix_rank_value"].to_numpy(),
            df["density_ori_rank_value"].to_numpy(),
        )
        metrics = evaluate_ltr(actual, scores)
        n_binding = int((actual > 0).sum())
        print(f"  {m}: n={len(df)}, binding={n_binding}, VC@20={metrics['VC@20']:.4f}")
        per_month[m] = metrics

        del df, realized, actual, scores
        gc.collect()

    return per_month


# ═══════════════════════════════════════════════════════════════════════════════
# V8a/V8b: ML regression with v7 formula feature
# ═══════════════════════════════════════════════════════════════════════════════

def run_ml_version(
    version_id: str,
    eval_months: list[str],
    use_v7_formula: bool = False,
) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    """Run ML pipeline, return (per_month_metrics, per_month_scores).

    If use_v7_formula=True, uses v7_blend_score as the formula feature
    instead of the default v62b_formula_score.
    """
    features = list(FEATURES_V1B)
    monotone = list(MONOTONE_V1B)

    if use_v7_formula:
        features.append("v7_formula_score")
        monotone.append(-1)  # lower = more binding
        desc = "13f (12 + v7 formula)"
    else:
        desc = "12f"

    print(f"\n[{version_id}] ML regression, {desc}, {len(eval_months)} months")

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

        # Add v7 formula feature if requested
        if use_v7_formula:
            for df_name in ["train", "test"]:
                df = train_df if df_name == "train" else test_df
                if "v7_formula_score" not in df.columns:
                    has_cols = all(c in df.columns for c in
                                  ["da_rank_value", "density_mix_rank_value", "density_ori_rank_value"])
                    if has_cols:
                        if df_name == "train":
                            train_df = df.with_columns(
                                (V7_DA * pl.col("da_rank_value")
                                 + V7_DMIX * pl.col("density_mix_rank_value")
                                 + V7_DORI * pl.col("density_ori_rank_value")
                                ).alias("v7_formula_score")
                            )
                        else:
                            test_df = df.with_columns(
                                (V7_DA * pl.col("da_rank_value")
                                 + V7_DMIX * pl.col("density_mix_rank_value")
                                 + V7_DORI * pl.col("density_ori_rank_value")
                                ).alias("v7_formula_score")
                            )

        train_df = train_df.sort("query_month")
        X_train, _ = prepare_features(train_df, cfg.ltr)
        y_train = train_df["realized_sp"].to_numpy().astype(np.float64)
        groups_train = compute_query_groups(train_df)

        del train_df
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

        elapsed = time.time() - t0
        print(f"  {m}: VC@20={metrics['VC@20']:.4f}, VC@100={metrics['VC@100']:.4f} ({elapsed:.1f}s)")

        # Feature importance for last month
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

        del X_test, actual, model
        gc.collect()

    return per_month, per_month_scores


# ═══════════════════════════════════════════════════════════════════════════════
# V8c: Ensemble sweep
# ═══════════════════════════════════════════════════════════════════════════════

def run_ensemble_sweep(
    ml_scores: dict[str, np.ndarray],
    eval_months: list[str],
) -> list[dict]:
    """Sweep alpha in ML+blend ensemble. Returns list of {alpha, metrics}."""
    print(f"\n[v8c] Ensemble sweep: alpha*ML + (1-alpha)*v7_blend")

    # Preload v7 blend scores for each month
    v7_scores = {}
    for m in eval_months:
        path = Path(V62B_SIGNAL_BASE) / m / "f0" / "onpeak"
        df = pl.read_parquet(str(path))
        realized = load_realized_da(m)
        df = df.join(realized, on="constraint_id", how="left")
        df = df.with_columns(pl.col("realized_sp").fill_null(0.0))

        v7_scores[m] = {
            "actual": df["realized_sp"].to_numpy().astype(np.float64),
            "blend": -v7_blend_score(
                df["da_rank_value"].to_numpy(),
                df["density_mix_rank_value"].to_numpy(),
                df["density_ori_rank_value"].to_numpy(),
            ),
        }
        del df, realized

    results = []
    for alpha_int in range(0, 21):  # 0.00 to 1.00 step 0.05
        alpha = alpha_int * 0.05
        per_month = {}
        for m in eval_months:
            # Normalize both to [0,1] range before blending
            ml = ml_scores[m]
            blend = v7_scores[m]["blend"]
            actual = v7_scores[m]["actual"]

            # Min-max normalize
            ml_norm = (ml - ml.min()) / (ml.max() - ml.min() + 1e-10)
            blend_norm = (blend - blend.min()) / (blend.max() - blend.min() + 1e-10)

            combined = alpha * ml_norm + (1 - alpha) * blend_norm
            per_month[m] = evaluate_ltr(actual, combined)

        agg = aggregate_months(per_month)["mean"]
        results.append({"alpha": alpha, **{k: v for k, v in agg.items() if isinstance(v, float)}})

    # Print results
    print(f"\n{'alpha':>6} | {'VC@20':>8} {'VC@100':>8} {'R@20':>8} {'NDCG':>8} {'Spearman':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['alpha']:>6.2f} | {r['VC@20']:>8.4f} {r['VC@100']:>8.4f} "
              f"{r['Recall@20']:>8.4f} {r['NDCG']:>8.4f} {r['Spearman']:>10.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Registry helpers
# ═══════════════════════════════════════════════════════════════════════════════

def save_version(version_id: str, per_month: dict, eval_months: list[str],
                 config_extra: dict | None = None) -> None:
    """Save version to registry."""
    # Strip feature importance before aggregation
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


def print_comparison(versions: dict[str, dict]) -> None:
    """Print comparison table from {version_id: aggregate_means}."""
    metrics = ["VC@20", "VC@100", "Recall@20", "Recall@50", "NDCG", "Spearman"]
    header = f"{'Metric':<12}"
    for vid in versions:
        header += f" {vid:>14}"
    print(f"\n{'='*70}")
    print("COMPARISON TABLE (36-month dev)")
    print(f"{'='*70}")
    print(header)
    print("-" * (12 + 15 * len(versions)))
    for met in metrics:
        row = f"{met:<12}"
        vals = [versions[vid].get(met, 0) for vid in versions]
        best = max(vals)
        for vid in versions:
            v = versions[vid].get(met, 0)
            marker = " *" if v == best else "  "
            row += f" {v:>12.4f}{marker}"
        print(row)
    print(f"\n* = best for that metric")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    eval_months = _FULL_EVAL_MONTHS  # 36 months
    t_start = time.time()

    print(f"[main] Starting v7+ experiments, {len(eval_months)} months, mem={mem_mb():.0f} MB")
    print(f"[main] Months: {eval_months[0]} to {eval_months[-1]}")

    # ── V7: optimized blend ──
    t0 = time.time()
    v7_per_month = run_v7(eval_months)
    save_version("v7", v7_per_month, eval_months, config_extra={
        "method": "optimized_blend",
        "formula": f"-({V7_DA}*da_rank_value + {V7_DMIX}*density_mix_rank_value + {V7_DORI}*density_ori_rank_value)",
        "rationale": "Grid search over 231 weight triplets on 12-month dev. Density mix has ~0 correlation with realized DA.",
    })
    print(f"[v7] Done in {time.time()-t0:.1f}s")

    # ── V8a: ML regression, 12f (same as v6a, against v7 baseline) ──
    t0 = time.time()
    v8a_per_month, v8a_scores = run_ml_version("v8a", eval_months, use_v7_formula=False)
    save_version("v8a", v8a_per_month, eval_months, config_extra={
        "method": "lightgbm_regression",
        "features": "FEATURES_V1B (12f)",
        "label_mode": "tiered",
        "note": "Same as v6a, re-evaluated on 36 months for comparison with v7 baseline",
    })
    print(f"[v8a] Done in {time.time()-t0:.1f}s")

    # ── V8b: ML regression, 13f with v7 formula as feature ──
    t0 = time.time()
    v8b_per_month, v8b_scores = run_ml_version("v8b", eval_months, use_v7_formula=True)
    save_version("v8b", v8b_per_month, eval_months, config_extra={
        "method": "lightgbm_regression",
        "features": "FEATURES_V1B + v7_formula_score (13f)",
        "v7_formula": f"{V7_DA}*da + {V7_DMIX}*dmix + {V7_DORI}*dori",
        "label_mode": "tiered",
        "note": "v6b rebuilt with optimized v7 formula as feature instead of v62b formula",
    })
    print(f"[v8b] Done in {time.time()-t0:.1f}s")

    # ── V8c: ensemble sweep using v8b scores ──
    t0 = time.time()
    ensemble_results = run_ensemble_sweep(v8b_scores, eval_months)

    # Find best alpha by VC@20
    best_vc20 = max(ensemble_results, key=lambda x: x["VC@20"])
    best_vc100 = max(ensemble_results, key=lambda x: x["VC@100"])

    print(f"\n[v8c] Best by VC@20:  alpha={best_vc20['alpha']:.2f}, VC@20={best_vc20['VC@20']:.4f}")
    print(f"[v8c] Best by VC@100: alpha={best_vc100['alpha']:.2f}, VC@100={best_vc100['VC@100']:.4f}")

    # Save ensemble results
    ens_dir = REGISTRY / "v8c_ensemble"
    ens_dir.mkdir(parents=True, exist_ok=True)
    with open(ens_dir / "sweep_results.json", "w") as f:
        json.dump({"sweep": ensemble_results, "best_vc20": best_vc20, "best_vc100": best_vc100}, f, indent=2)
    print(f"[v8c] Done in {time.time()-t0:.1f}s")

    # ── Comparison table ──
    # Load v0 and v6b from existing registry for reference
    comparison = {}

    # v0 from registry
    v0_path = REGISTRY / "v0" / "metrics.json"
    if v0_path.exists():
        v0_data = json.loads(v0_path.read_text())
        # v0 was 12-month, re-eval on 36 if we have it
        if "v0_36" in [p.name for p in REGISTRY.iterdir() if p.is_dir()]:
            v0_36 = json.loads((REGISTRY / "v0_36" / "metrics.json").read_text())
            comparison["v0"] = v0_36["aggregate"]["mean"]
        else:
            comparison["v0"] = v0_data["aggregate"]["mean"]

    # v6b from registry
    v6b_path = REGISTRY / "v6b_36" / "metrics.json"
    if v6b_path.exists():
        v6b_data = json.loads(v6b_path.read_text())
        comparison["v6b"] = v6b_data["aggregate"]["mean"]

    # New versions
    comparison["v7"] = aggregate_months(
        {m: {k: v for k, v in v.items() if not k.startswith("_")} for m, v in v7_per_month.items()}
    )["mean"]
    comparison["v8a"] = aggregate_months(
        {m: {k: v for k, v in v.items() if not k.startswith("_")} for m, v in v8a_per_month.items()}
    )["mean"]
    comparison["v8b"] = aggregate_months(
        {m: {k: v for k, v in v.items() if not k.startswith("_")} for m, v in v8b_per_month.items()}
    )["mean"]
    comparison["v8c*"] = best_vc20  # best ensemble by VC@20

    print_comparison(comparison)

    total = time.time() - t_start
    print(f"\n[main] All experiments done in {total:.1f}s, mem={mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
