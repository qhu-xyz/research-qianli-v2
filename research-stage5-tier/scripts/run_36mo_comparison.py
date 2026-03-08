"""Run v0 formula + v5/v6a/v6b/v6c on 24 extra months, then combine with existing 12.

Produces a 36-month comprehensive comparison table.
Expected time: ~15 min (5 variants x 24 months x ~8s/month, parallel 4 workers).
"""
import sys
import gc
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    LTRConfig, PipelineConfig, FEATURES_V1B, MONOTONE_V1B,
    V62B_SIGNAL_BASE, _FULL_EVAL_MONTHS, _DEFAULT_EVAL_MONTHS,
)
from ml.benchmark import run_benchmark
from ml.evaluate import evaluate_ltr, aggregate_months
from ml.realized_da import load_realized_da
from ml.v62b_formula import v62b_score

# 24 months NOT in the original 12-month eval
_EXTRA_MONTHS = sorted(set(_FULL_EVAL_MONTHS) - set(_DEFAULT_EVAL_MONTHS))

# v1b features + formula score
_V6_FEATURES = list(FEATURES_V1B) + ["v62b_formula_score"]
_V6_MONOTONE = list(MONOTONE_V1B) + [-1]


# ── v0 formula (no training, just evaluate) ─────────────────────────────────

def evaluate_v0_month(month: str) -> dict:
    """Evaluate V6.2B formula on one month against realized DA."""
    path = Path(V62B_SIGNAL_BASE) / month / "f0" / "onpeak"
    df = pl.read_parquet(str(path))
    realized = load_realized_da(month)
    df = df.join(realized, on="constraint_id", how="left")
    df = df.with_columns(pl.col("realized_sp").fill_null(0.0))

    actual = df["realized_sp"].to_numpy().astype(np.float64)
    scores = -v62b_score(
        da_rank_value=df["da_rank_value"].to_numpy(),
        density_mix_rank_value=df["density_mix_rank_value"].to_numpy(),
        density_ori_rank_value=df["density_ori_rank_value"].to_numpy(),
    )
    metrics = evaluate_ltr(actual, scores)
    n_binding = int((actual > 0).sum())
    print(f"  v0 {month}: n={len(df)}, binding={n_binding}, VC@20={metrics['VC@20']:.4f}")
    del df, realized, actual, scores
    gc.collect()
    return metrics


def run_v0_extra(months: list[str]) -> dict:
    """Run v0 formula on extra months, save to registry/v0_36/."""
    print(f"\n{'=' * 60}")
    print(f"v0 formula — {len(months)} extra months")
    print(f"{'=' * 60}")

    t0 = time.time()
    per_month = {}
    for m in months:
        per_month[m] = evaluate_v0_month(m)

    agg = aggregate_months(per_month)
    result = {
        "eval_config": {"eval_months": months, "mode": "extra_24"},
        "per_month": per_month,
        "aggregate": agg,
        "n_months": len(per_month),
    }

    out_dir = Path("registry/v0_36")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    elapsed = time.time() - t0
    print(f"v0 done in {elapsed:.0f}s: VC@20={agg['mean']['VC@20']:.4f}")
    return result


# ── ML variants ──────────────────────────────────────────────────────────────

ML_VARIANTS = {
    "v5_36": {
        "features": list(FEATURES_V1B),
        "monotone": list(MONOTONE_V1B),
        "backend": "lightgbm",
        "label_mode": "tiered",
        "desc": "tiered ranking, 12 features",
    },
    "v6a_36": {
        "features": list(FEATURES_V1B),
        "monotone": list(MONOTONE_V1B),
        "backend": "lightgbm_regression",
        "label_mode": "tiered",
        "desc": "regression, 12 features",
    },
    "v6b_36": {
        "features": list(_V6_FEATURES),
        "monotone": list(_V6_MONOTONE),
        "backend": "lightgbm_regression",
        "label_mode": "tiered",
        "desc": "regression, 13f (12+formula)",
    },
    "v6c_36": {
        "features": list(_V6_FEATURES),
        "monotone": list(_V6_MONOTONE),
        "backend": "lightgbm",
        "label_mode": "tiered",
        "desc": "tiered ranking, 13f (12+formula)",
    },
}


def run_ml_variant(vid: str, cfg_dict: dict, months: list[str]) -> dict:
    config = PipelineConfig(
        ltr=LTRConfig(
            features=cfg_dict["features"],
            monotone_constraints=cfg_dict["monotone"],
            backend=cfg_dict["backend"],
            label_mode=cfg_dict["label_mode"],
        ),
        train_months=8,
        val_months=0,
    )

    print(f"\n{'=' * 60}")
    print(f"{vid}: {cfg_dict['desc']}")
    print(f"{'=' * 60}")

    t0 = time.time()
    result = run_benchmark(
        version_id=vid,
        eval_months=months,
        config=config,
        mode="extra_24",
    )
    elapsed = time.time() - t0

    agg = result.get("aggregate", {}).get("mean", {})
    print(f"{vid} done in {elapsed:.0f}s: VC@20={agg.get('VC@20', 0):.4f}")
    return result


# ── Merge + report ───────────────────────────────────────────────────────────

def merge_results(vid_36: str, vid_12: str) -> dict | None:
    """Merge 12-month + 24-month per_month dicts, recompute mean."""
    p12 = Path("registry") / vid_12 / "metrics.json"
    p24 = Path("registry") / vid_36 / "metrics.json"
    if not p12.exists() or not p24.exists():
        return None
    with open(p12) as f:
        d12 = json.load(f)
    with open(p24) as f:
        d24 = json.load(f)

    all_months = {}
    for m, metrics in d12.get("per_month", {}).items():
        # Strip internal keys
        all_months[m] = {k: v for k, v in metrics.items() if not k.startswith("_")}
    for m, metrics in d24.get("per_month", {}).items():
        all_months[m] = {k: v for k, v in metrics.items() if not k.startswith("_")}

    if not all_months:
        return None

    keys = list(next(iter(all_months.values())).keys())
    combined = {}
    for k in keys:
        vals = [all_months[m][k] for m in all_months if k in all_months[m]]
        combined[k] = sum(vals) / len(vals) if vals else 0

    return {"mean": combined, "n_months": len(all_months)}


def print_table(title: str, rows: list[tuple[str, dict]]):
    metrics = ["VC@20", "VC@100", "Recall@20", "Recall@100", "NDCG", "Spearman"]
    print(f"\n{'=' * 78}")
    print(title)
    print(f"{'=' * 78}")
    print(f"{'Version':<14} " + " ".join(f"{m:>10}" for m in metrics) + "    N")
    print("-" * 78)
    for name, agg in rows:
        vals = " ".join(f"{agg.get(m, 0):>10.4f}" for m in metrics)
        n = agg.get("_n", "")
        print(f"{name:<14} {vals}   {n}")


def main():
    t_total = time.time()
    extra_months = _EXTRA_MONTHS
    print(f"24 extra months: {extra_months}")

    # Run v0 formula on extra months
    run_v0_extra(extra_months)

    # Run ML variants on extra months
    for vid, cfg_dict in ML_VARIANTS.items():
        run_ml_variant(vid, cfg_dict, extra_months)

    # ── Report ──
    mapping = [
        ("v0_36", "v0", "v0 (formula)"),
        ("v5_36", "v5", "v5 (rank,12f)"),
        ("v6a_36", "v6a", "v6a (reg,12f)"),
        ("v6b_36", "v6b", "v6b (reg,13f)"),
        ("v6c_36", "v6c", "v6c (rank,13f)"),
    ]

    # Extra 24 months only
    rows_24 = []
    for vid_36, _, label in mapping:
        p = Path("registry") / vid_36 / "metrics.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            agg = d.get("aggregate", {}).get("mean", {})
            agg["_n"] = d.get("n_months", "?")
            rows_24.append((label, agg))
    print_table("EXTRA 24-MONTH RESULTS (out-of-sample)", rows_24)

    # Original 12 months
    rows_12 = []
    for _, vid_12, label in mapping:
        p = Path("registry") / vid_12 / "metrics.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            agg = d.get("aggregate", {}).get("mean", {})
            agg["_n"] = d.get("n_months", "?")
            rows_12.append((label, agg))
    print_table("ORIGINAL 12-MONTH RESULTS (for reference)", rows_12)

    # Combined 36 months
    rows_36 = []
    for vid_36, vid_12, label in mapping:
        merged = merge_results(vid_36, vid_12)
        if merged:
            agg = merged["mean"]
            agg["_n"] = merged["n_months"]
            rows_36.append((label, agg))
    print_table("COMBINED 36-MONTH RESULTS", rows_36)

    total_elapsed = time.time() - t_total
    print(f"\nTotal time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
