"""Apple-to-apple comparison: v0 (original repo) vs v1 (v0011-based stage 2).

Both pipelines predict on the SAME test data for 12 eval months (f0, onpeak),
scored by the SAME evaluation function.

Phases:
  A — Run original repo (v0) pipeline for each eval month
  B — Run v1 model (v0011 classifier config) for each eval month
  C — Score both through original repo's score_results_df + our evaluate_pipeline
  D — Aggregate and compare

Usage:
  cd /home/xyz/workspace/pmodel && source .venv/bin/activate
  PYTHONPATH=.../research-stage2-shadow:$PYTHONPATH python .../scripts/apple_to_apple.py
"""
from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# ── Paths ─────────────────────────────────────────────────────────────────
STAGE2_ROOT = Path(__file__).resolve().parent.parent
ORIG_REPO_SRC = Path(
    "/home/xyz/workspace/research-qianli-v2"
    "/research-spice-shadow-price-pred-qianli/src"
)
COMPARISON_DIR = STAGE2_ROOT / "registry" / "comparison"
V0_DIR = COMPARISON_DIR / "v0"
V1_DIR = COMPARISON_DIR / "v1"

for d in (V0_DIR, V1_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Ensure both repos are importable
if str(ORIG_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(ORIG_REPO_SRC))
if str(STAGE2_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE2_ROOT))


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ── Eval months from gates.json ──────────────────────────────────────────
EVAL_MONTHS = [
    "2020-09", "2020-11", "2021-01", "2021-03",
    "2021-05", "2021-07", "2021-09", "2021-11",
    "2022-03", "2022-06", "2022-09", "2022-12",
]

CLASS_TYPE = "onpeak"
PERIOD_TYPE = "f0"  # forward month


# ── v1 configs (from v0011) ──────────────────────────────────────────────
V1_CLF_FEATURES = [
    "prob_exceed_110", "prob_exceed_105", "prob_exceed_100",
    "prob_exceed_95", "prob_exceed_90",
    "prob_below_100", "prob_below_95", "prob_below_90",
    "expected_overload", "density_skewness", "density_kurtosis",
    "hist_da", "hist_da_trend",
    "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac",
    "is_interface", "constraint_limit",
    "density_mean", "density_variance", "density_entropy",
    "tail_concentration", "prob_band_95_100", "prob_band_100_105",
    "hist_da_max_season",
    # interaction features (5)
    "band_severity", "sf_exceed_interaction", "hist_seasonal_band",
]  # 29 total

V1_CLF_MONOTONE = [
    1, 1, 1, 1, 1,       # prob_exceed_*
    -1, -1, -1,           # prob_below_*
    1,                     # expected_overload
    0, 0,                  # density_skewness, kurtosis
    1, 1,                  # hist_da, hist_da_trend
    1, 1, 0, 0,           # sf_max_abs, sf_mean_abs, sf_std, sf_nonzero_frac
    0, 0,                  # is_interface, constraint_limit
    0, 0, 0,              # density_mean, variance, entropy
    1,                     # tail_concentration
    0, 0,                  # prob_band_95_100, prob_band_100_105
    1,                     # hist_da_max_season
    0, 0, 0,              # band_severity, sf_exceed_interaction, hist_seasonal_band
]

V1_REG_ADDITIONAL = [
    "prob_exceed_85", "prob_exceed_80",
    "recent_hist_da", "season_hist_da_1", "season_hist_da_2",
]
V1_REG_FEATURES = V1_CLF_FEATURES + V1_REG_ADDITIONAL  # 34 total
V1_REG_ADDITIONAL_MONOTONE = [1, 1, 1, 1, 1]
V1_REG_MONOTONE = V1_CLF_MONOTONE + V1_REG_ADDITIONAL_MONOTONE


# ══════════════════════════════════════════════════════════════════════════
# Phase A: Run v0 (original repo)
# ══════════════════════════════════════════════════════════════════════════

def run_v0_month(
    auction_month_str: str,
) -> pd.DataFrame | None:
    """Run original pipeline for a single eval month.

    Returns results_per_outage DataFrame or None on failure.
    """
    from shadow_price_prediction.config import PredictionConfig
    from shadow_price_prediction.pipeline import ShadowPricePipeline

    parquet_path = V0_DIR / f"{auction_month_str}.parquet"
    if parquet_path.exists():
        print(f"  [v0] {auction_month_str}: cached, loading from {parquet_path}")
        return pd.read_parquet(parquet_path)

    print(f"  [v0] {auction_month_str}: running original pipeline...")
    t0 = time.time()

    try:
        config = PredictionConfig()
        config.class_type = CLASS_TYPE
        pipeline = ShadowPricePipeline(config)

        am = pd.Timestamp(auction_month_str)
        # f0 => market_month = auction_month (same month)
        mm = am
        test_periods = [(am, mm)]

        results_per_outage, _, _, _, _ = pipeline.run(
            test_periods=test_periods,
            class_type=CLASS_TYPE,
            use_parallel=False,
            verbose=False,
        )

        if results_per_outage is None or len(results_per_outage) == 0:
            print(f"  [v0] {auction_month_str}: no results (empty)")
            return None

        results_per_outage.to_parquet(parquet_path)
        elapsed = time.time() - t0
        print(
            f"  [v0] {auction_month_str}: {len(results_per_outage)} rows, "
            f"{elapsed:.1f}s, mem={mem_mb():.0f} MB"
        )
        return results_per_outage

    except Exception as e:
        print(f"  [v0] {auction_month_str}: FAILED — {e}")
        return None
    finally:
        gc.collect()


# ══════════════════════════════════════════════════════════════════════════
# Phase B: Run v1 (our stage-2 model with v0011 config)
# ══════════════════════════════════════════════════════════════════════════

def _load_v1_training_data(
    auction_month_str: str,
    train_months: int = 6,
    val_months: int = 2,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load train/val data for v1 using source repo's MisoDataLoader."""
    from shadow_price_prediction.config import PredictionConfig
    from shadow_price_prediction.data_loader import MisoDataLoader

    pred_config = PredictionConfig()
    pred_config.class_type = CLASS_TYPE
    loader = MisoDataLoader(pred_config)

    auction_ts = pd.Timestamp(auction_month_str)
    # f0 horizon = 0
    lookback = train_months + val_months + 0
    train_start = auction_ts - pd.DateOffset(months=lookback)
    train_end = auction_ts

    print(f"    [v1-data] loading train {train_start:%Y-%m} to {train_end:%Y-%m}")
    train_pd = loader.load_training_data(
        train_start=train_start,
        train_end=train_end,
        required_period_types={PERIOD_TYPE},
    )
    print(f"    [v1-data] loaded {len(train_pd)} rows, mem={mem_mb():.0f} MB")

    if "label" in train_pd.columns and "actual_shadow_price" not in train_pd.columns:
        train_pd = train_pd.rename(columns={"label": "actual_shadow_price"})

    train_data = pl.from_pandas(train_pd)
    del train_pd
    gc.collect()

    # Split: first train_months for fit, last val_months for val
    val_boundary = train_start + pd.DateOffset(months=train_months)
    val_boundary_str = val_boundary.strftime("%Y-%m")

    if "auction_month" in train_data.columns:
        if train_data["auction_month"].dtype != pl.Utf8:
            train_data = train_data.with_columns(
                pl.col("auction_month").cast(pl.Utf8).str.slice(0, 7).alias("auction_month")
            )
        fit_df = train_data.filter(pl.col("auction_month") < val_boundary_str)
        val_df = train_data.filter(pl.col("auction_month") >= val_boundary_str)
    else:
        split = int(len(train_data) * train_months / (train_months + val_months))
        fit_df = train_data[:split]
        val_df = train_data[split:]

    del train_data
    gc.collect()
    print(f"    [v1-data] fit={fit_df.shape}, val={val_df.shape}")
    return fit_df, val_df


def _load_v1_test_data(auction_month_str: str) -> pd.DataFrame | None:
    """Load test data for v1 via source repo's MisoDataLoader."""
    from shadow_price_prediction.config import PredictionConfig
    from shadow_price_prediction.data_loader import MisoDataLoader

    pred_config = PredictionConfig()
    pred_config.class_type = CLASS_TYPE
    loader = MisoDataLoader(pred_config)

    am = pd.Timestamp(auction_month_str)
    mm = am  # f0 => market_month = auction_month
    test_pd = loader.load_test_data(test_periods=[(am, mm)], verbose=False)
    return test_pd


def run_v1_month(auction_month_str: str) -> pd.DataFrame | None:
    """Train v1 model and predict on test data for a single eval month.

    Returns results_per_outage-style DataFrame or None on failure.
    """
    from ml.features import compute_interaction_features

    parquet_path = V1_DIR / f"{auction_month_str}.parquet"
    if parquet_path.exists():
        print(f"  [v1] {auction_month_str}: cached, loading from {parquet_path}")
        return pd.read_parquet(parquet_path)

    print(f"  [v1] {auction_month_str}: training + predicting...")
    t0 = time.time()

    try:
        # --- Load training data ---
        fit_df, val_df = _load_v1_training_data(auction_month_str)

        # --- Compute interaction features on train/val ---
        fit_df = compute_interaction_features(fit_df)
        val_df = compute_interaction_features(val_df)

        # --- Prepare classifier features ---
        clf_cols = V1_CLF_FEATURES
        clf_monotone = tuple(V1_CLF_MONOTONE)

        def _extract_X(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
            return df.select(cols).fill_null(0.0).to_numpy().astype(np.float64)

        X_train_clf = _extract_X(fit_df, clf_cols)
        y_train_clf = (
            fit_df["actual_shadow_price"].to_numpy().astype(np.float64) > 0.0
        ).astype(int)

        X_val_clf = _extract_X(val_df, clf_cols)
        y_val_clf = (
            val_df["actual_shadow_price"].to_numpy().astype(np.float64) > 0.0
        ).astype(int)

        # --- Train classifier (v0011 hyperparams) ---
        from xgboost import XGBClassifier

        scale_pos_weight = max(1.0, (y_train_clf == 0).sum() / max(1, (y_train_clf == 1).sum()))
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.07,
            subsample=0.8,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=10,
            scale_pos_weight=scale_pos_weight,
            monotone_constraints=clf_monotone,
            eval_metric="logloss",
            random_state=42,
        )
        clf.fit(X_train_clf, y_train_clf)
        print(f"    [v1] classifier trained, mem={mem_mb():.0f} MB")

        # --- Optimize threshold (F-beta=0.7 on val set) ---
        from ml.threshold import find_optimal_threshold

        val_proba = clf.predict_proba(X_val_clf)[:, 1]
        threshold = find_optimal_threshold(y_val_clf, val_proba, beta=0.7)
        print(f"    [v1] threshold={threshold:.3f}")

        # --- Train regressor (gated: binding-only samples) ---
        from xgboost import XGBRegressor

        reg_cols = V1_REG_FEATURES
        reg_monotone = tuple(V1_REG_MONOTONE)

        # Gated: use binding-only samples from train set
        binding_mask_train = y_train_clf == 1
        n_binding_train = binding_mask_train.sum()

        if n_binding_train >= 5:
            X_train_reg = _extract_X(fit_df.filter(pl.col("actual_shadow_price") > 0), reg_cols)
            y_train_reg_raw = (
                fit_df.filter(pl.col("actual_shadow_price") > 0)
                ["actual_shadow_price"].to_numpy().astype(np.float64)
            )
        else:
            # Fallback: use all samples
            X_train_reg = _extract_X(fit_df, reg_cols)
            y_train_reg_raw = fit_df["actual_shadow_price"].to_numpy().astype(np.float64)

        y_train_reg = np.log1p(np.maximum(0.0, y_train_reg_raw))

        reg = XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=10,
            monotone_constraints=reg_monotone,
            random_state=42,
        )
        reg.fit(X_train_reg, y_train_reg)
        print(f"    [v1] regressor trained (n_binding_train={n_binding_train}), mem={mem_mb():.0f} MB")

        # Free training data
        del fit_df, val_df, X_train_clf, X_val_clf, X_train_reg
        del y_train_clf, y_val_clf, y_train_reg, y_train_reg_raw
        gc.collect()

        # --- Load test data ---
        test_pd = _load_v1_test_data(auction_month_str)
        if test_pd is None or len(test_pd) == 0:
            print(f"  [v1] {auction_month_str}: no test data")
            return None

        if "label" in test_pd.columns and "actual_shadow_price" not in test_pd.columns:
            test_pd = test_pd.rename(columns={"label": "actual_shadow_price"})

        test_pl = pl.from_pandas(test_pd)

        # Compute interaction features on test data
        test_pl = compute_interaction_features(test_pl)

        # --- Predict ---
        X_test_clf = _extract_X(test_pl, clf_cols)
        pred_proba = clf.predict_proba(X_test_clf)[:, 1]
        pred_binding = (pred_proba >= threshold).astype(int)

        X_test_reg = _extract_X(test_pl, reg_cols)
        raw_reg_pred = reg.predict(X_test_reg)
        pred_shadow_price = np.expm1(np.maximum(0.0, raw_reg_pred))
        # Gate: non-binding predictions get 0 shadow price
        pred_shadow_price = np.where(pred_binding == 1, pred_shadow_price, 0.0)

        # --- Build results_per_outage DataFrame matching v0 schema ---
        actual_sp = test_pl["actual_shadow_price"].to_numpy().astype(np.float64)
        actual_binding = (actual_sp > 0).astype(int)

        result_df = pd.DataFrame({
            "actual_shadow_price": actual_sp,
            "predicted_shadow_price": pred_shadow_price,
            "actual_binding": actual_binding,
            "predicted_binding": pred_binding,
            "binding_probability": pred_proba,
            "model_used": "v1_default",
        })

        # Copy metadata columns from test data
        for col in ["branch_name", "constraint_id", "flow_direction", "outage_date"]:
            if col in test_pd.columns:
                result_df[col] = test_pd[col].values
            else:
                result_df[col] = "unknown"

        result_df.to_parquet(parquet_path)
        elapsed = time.time() - t0
        print(
            f"  [v1] {auction_month_str}: {len(result_df)} rows, "
            f"{elapsed:.1f}s, mem={mem_mb():.0f} MB"
        )
        return result_df

    except Exception as e:
        import traceback
        print(f"  [v1] {auction_month_str}: FAILED — {e}")
        traceback.print_exc()
        return None
    finally:
        gc.collect()


# ══════════════════════════════════════════════════════════════════════════
# Phase C: Score both pipelines
# ══════════════════════════════════════════════════════════════════════════

def score_month(
    v0_df: pd.DataFrame | None,
    v1_df: pd.DataFrame | None,
    month: str,
) -> dict:
    """Score both v0 and v1 predictions for a single month.

    Returns dict with keys "v0" and "v1", each containing:
      - "original_scores": from score_results_df()
      - "ev_scores": from our evaluate_pipeline()
    """
    from shadow_price_prediction.evaluation import score_results_df
    from ml.evaluate import evaluate_pipeline

    result: dict = {"month": month, "v0": None, "v1": None}

    for label, df in [("v0", v0_df), ("v1", v1_df)]:
        if df is None or len(df) == 0:
            continue

        # Original repo scoring
        orig_scores = score_results_df(df)

        # EV scoring through our evaluate_pipeline
        actual_sp = df["actual_shadow_price"].values.astype(float)
        pred_proba = df["binding_probability"].values.astype(float)
        pred_sp = df["predicted_shadow_price"].values.astype(float)
        ev = pred_proba * pred_sp

        ev_metrics = evaluate_pipeline(actual_sp, pred_proba, pred_sp, ev)

        result[label] = {
            "original_scores": orig_scores,
            "ev_scores": ev_metrics,
        }

    return result


# ══════════════════════════════════════════════════════════════════════════
# Phase D: Aggregate and compare
# ══════════════════════════════════════════════════════════════════════════

def _safe_get(d: dict, *keys, default=float("nan")):
    """Navigate nested dict safely."""
    current = d
    for k in keys:
        if not isinstance(current, dict) or k not in current:
            return default
        current = current[k]
    return current


def aggregate_and_compare(per_month_scores: list[dict]) -> dict:
    """Aggregate per-month metrics and produce comparison table."""

    # Define all metrics we want to compare
    metric_defs = [
        # (display_name, path_in_scores, direction)
        # From original repo scoring
        ("AUC-ROC", ["original_scores", "stage1", "auc_roc"], "higher"),
        ("Avg-Precision", ["original_scores", "stage1", "avg_precision"], "higher"),
        ("Brier", ["original_scores", "stage1", "brier_score"], "lower"),
        ("Precision", ["original_scores", "stage1", "precision"], "higher"),
        ("Recall", ["original_scores", "stage1", "recall"], "higher"),
        ("F1", ["original_scores", "stage1", "f1"], "higher"),
        ("MAE-TP", ["original_scores", "stage2", "mae_tp"], "lower"),
        ("RMSE-TP", ["original_scores", "stage2", "rmse_tp"], "lower"),
        ("Spearman-TP", ["original_scores", "stage2", "spearman_tp"], "higher"),
        # Outage-level ranking from original scoring
        ("ValCap@100", ["original_scores", "ranking_outage", "value_capture@100"], "higher"),
        ("ValCap@500", ["original_scores", "ranking_outage", "value_capture@500"], "higher"),
        ("ValCap@1000", ["original_scores", "ranking_outage", "value_capture@1000"], "higher"),
        ("NDCG-outage", ["original_scores", "ranking_outage", "ndcg"], "higher"),
        # EV-based metrics from our evaluate_pipeline
        ("EV-VC@100", ["ev_scores", "EV-VC@100"], "higher"),
        ("EV-VC@500", ["ev_scores", "EV-VC@500"], "higher"),
        ("EV-VC@1000", ["ev_scores", "EV-VC@1000"], "higher"),
        ("EV-NDCG", ["ev_scores", "EV-NDCG"], "higher"),
        ("Spearman", ["ev_scores", "Spearman"], "higher"),
        ("C-RMSE", ["ev_scores", "C-RMSE"], "lower"),
        ("C-MAE", ["ev_scores", "C-MAE"], "lower"),
        ("R-REC@500", ["ev_scores", "R-REC@500"], "higher"),
    ]

    # Collect per-month values
    v0_monthly: dict[str, list[float]] = {m[0]: [] for m in metric_defs}
    v1_monthly: dict[str, list[float]] = {m[0]: [] for m in metric_defs}
    months_used: list[str] = []

    for month_data in per_month_scores:
        month = month_data["month"]
        v0 = month_data.get("v0")
        v1 = month_data.get("v1")

        if v0 is None or v1 is None:
            print(f"  [compare] skipping {month}: v0={'ok' if v0 else 'MISS'}, v1={'ok' if v1 else 'MISS'}")
            continue

        months_used.append(month)
        for name, path, _ in metric_defs:
            v0_monthly[name].append(_safe_get(v0, *path))
            v1_monthly[name].append(_safe_get(v1, *path))

    # Compute aggregates
    comparison: dict = {
        "months_evaluated": months_used,
        "n_months": len(months_used),
        "metrics": {},
    }

    print("\n" + "=" * 90)
    print(f"{'Metric':<20} {'v0 mean':>10} {'v1 mean':>10} {'Delta':>10} {'Winner':>8} {'v0 std':>10} {'v1 std':>10}")
    print("-" * 90)

    v0_wins = 0
    v1_wins = 0
    ties = 0

    for name, _, direction in metric_defs:
        v0_vals = np.array(v0_monthly[name], dtype=float)
        v1_vals = np.array(v1_monthly[name], dtype=float)

        # Filter out NaN
        valid = ~(np.isnan(v0_vals) | np.isnan(v1_vals))
        v0_valid = v0_vals[valid]
        v1_valid = v1_vals[valid]

        if len(v0_valid) == 0:
            comparison["metrics"][name] = {"note": "no valid months"}
            continue

        v0_mean = float(np.mean(v0_valid))
        v1_mean = float(np.mean(v1_valid))
        v0_std = float(np.std(v0_valid, ddof=1)) if len(v0_valid) > 1 else 0.0
        v1_std = float(np.std(v1_valid, ddof=1)) if len(v1_valid) > 1 else 0.0

        delta = v1_mean - v0_mean
        if direction == "lower":
            winner = "v1" if delta < -1e-9 else ("v0" if delta > 1e-9 else "tie")
        else:
            winner = "v1" if delta > 1e-9 else ("v0" if delta < -1e-9 else "tie")

        if winner == "v0":
            v0_wins += 1
        elif winner == "v1":
            v1_wins += 1
        else:
            ties += 1

        comparison["metrics"][name] = {
            "v0_mean": round(v0_mean, 6),
            "v1_mean": round(v1_mean, 6),
            "delta": round(delta, 6),
            "direction": direction,
            "winner": winner,
            "v0_std": round(v0_std, 6),
            "v1_std": round(v1_std, 6),
            "v0_min": round(float(np.min(v0_valid)), 6),
            "v1_min": round(float(np.min(v1_valid)), 6),
            "v0_max": round(float(np.max(v0_valid)), 6),
            "v1_max": round(float(np.max(v1_valid)), 6),
            "v0_per_month": [round(float(v), 6) for v in v0_vals.tolist()],
            "v1_per_month": [round(float(v), 6) for v in v1_vals.tolist()],
        }

        # Print row
        delta_str = f"{delta:+.6f}" if abs(delta) < 100 else f"{delta:+.1f}"
        print(
            f"{name:<20} {v0_mean:>10.4f} {v1_mean:>10.4f} {delta_str:>10} "
            f"{'<-- v0' if winner == 'v0' else ('--> v1' if winner == 'v1' else '  tie'):>8} "
            f"{v0_std:>10.4f} {v1_std:>10.4f}"
        )

    print("-" * 90)
    print(f"Win count:  v0={v0_wins}, v1={v1_wins}, tie={ties}")
    print("=" * 90)

    comparison["summary"] = {
        "v0_wins": v0_wins,
        "v1_wins": v1_wins,
        "ties": ties,
    }

    return comparison


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def _sanitize(obj):
    """Make scores JSON-serializable (handle NaN, numpy types)."""
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Apple-to-apple v0 vs v1 comparison")
    parser.add_argument(
        "--phase", choices=["v0", "v1", "all"], default="all",
        help="Which pipeline(s) to run: v0-only, v1-only, or all (default: all)",
    )
    args = parser.parse_args()

    run_v0 = args.phase in ("v0", "all")
    run_v1 = args.phase in ("v1", "all")

    print("=" * 80)
    print("APPLE-TO-APPLE COMPARISON: v0 (original) vs v1 (v0011 stage-2)")
    print(f"Phase: {args.phase} | Eval months: {len(EVAL_MONTHS)} | class={CLASS_TYPE} | ptype={PERIOD_TYPE}")
    print(f"mem baseline: {mem_mb():.0f} MB")
    print("=" * 80)

    # ── Init Ray once ──
    os.environ.setdefault("RAY_ADDRESS", "ray://10.8.0.36:10001")
    import ray
    if not ray.is_initialized():
        from pbase.config.ray import init_ray
        import pmodel
        init_ray(extra_modules=[pmodel])
    print(f"Ray initialized, mem={mem_mb():.0f} MB\n")

    per_month_scores: list[dict] = []
    t_start = time.time()

    for i, month in enumerate(EVAL_MONTHS):
        print(f"\n{'─' * 60}")
        print(f"[{i+1}/{len(EVAL_MONTHS)}] Processing {month}")
        print(f"{'─' * 60}")

        v0_df = None
        v1_df = None

        if run_v0:
            v0_df = run_v0_month(month)
            gc.collect()

        if run_v1:
            v1_df = run_v1_month(month)
            gc.collect()

        # Phase C: Score (only if we have at least one side)
        if v0_df is not None or v1_df is not None:
            month_scores = score_month(v0_df, v1_df, month)
            per_month_scores.append(month_scores)

        del v0_df, v1_df
        gc.collect()
        print(f"  mem after month {month}: {mem_mb():.0f} MB")

    # Save per-month metrics
    suffix = f"_{args.phase}" if args.phase != "all" else ""
    metrics_path = COMPARISON_DIR / f"per_month_metrics{suffix}.json"
    metrics_path.write_text(json.dumps(_sanitize(per_month_scores), indent=2))
    print(f"\nPer-month metrics saved to {metrics_path}")

    # Phase D: Aggregate and compare (only meaningful with both sides)
    if run_v0 and run_v1:
        print("\n" + "=" * 80)
        print("PHASE D: AGGREGATE AND COMPARE")
        print("=" * 80)
        comparison = aggregate_and_compare(per_month_scores)
        comparison_path = COMPARISON_DIR / "comparison_results.json"
        comparison_path.write_text(json.dumps(_sanitize(comparison), indent=2))
        print(f"\nComparison results saved to {comparison_path}")
    elif run_v0:
        print("\n[v0-only mode] Predictions cached. Run with --phase v1 next, then --phase all for comparison.")
    elif run_v1:
        print("\n[v1-only mode] Predictions cached. Run with --phase all for comparison.")

    # Shutdown Ray
    if ray.is_initialized():
        ray.shutdown()

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed / 60:.1f} min, peak mem={mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
