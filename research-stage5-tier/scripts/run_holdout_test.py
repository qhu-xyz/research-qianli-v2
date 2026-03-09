#!/usr/bin/env python
"""One-time holdout test: evaluate champions on 2024-2025 data.

Results go to holdout/{version}/metrics.json and are immutable.
Run from research-stage5-tier/ with pmodel venv activated.

Usage:
    python scripts/run_holdout_test.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ml.config import FEATURES_V1B, MONOTONE_V1B, LTRConfig, PipelineConfig
from ml.data_loader import clear_month_cache, load_v62b_month
from ml.evaluate import aggregate_months, evaluate_ltr
from ml.pipeline import run_pipeline
from ml.registry_paths import holdout_root
from ml.v62b_formula import v62b_score

_HOLDOUT_BASE = Path(__file__).resolve().parent.parent / "holdout"
EVAL_MONTHS = [f"{y:04d}-{m:02d}" for y in (2024, 2025) for m in range(1, 13)]

# ── Champion definitions ────────────────────────────────────────────
CHAMPIONS = {
    "v0": {
        "description": "V6.2B formula baseline (no ML)",
        "type": "formula",
    },
    "v5": {
        "description": "LightGBM lambdarank, 12 features, tiered labels",
        "type": "ml",
        "config": PipelineConfig(
            ltr=LTRConfig(
                features=list(FEATURES_V1B),
                monotone_constraints=list(MONOTONE_V1B),
                backend="lightgbm",
                label_mode="tiered",
            )
        ),
    },
    "v6b": {
        "description": "LightGBM regression, 13 features (12 + formula score)",
        "type": "ml",
        "config": PipelineConfig(
            ltr=LTRConfig(
                features=list(FEATURES_V1B) + ["v62b_formula_score"],
                monotone_constraints=list(MONOTONE_V1B) + [-1],
                backend="lightgbm_regression",
                label_mode="tiered",
            )
        ),
    },
    "v6c": {
        "description": "LightGBM lambdarank, 13 features (12 + formula score)",
        "type": "ml",
        "config": PipelineConfig(
            ltr=LTRConfig(
                features=list(FEATURES_V1B) + ["v62b_formula_score"],
                monotone_constraints=list(MONOTONE_V1B) + [-1],
                backend="lightgbm",
                label_mode="tiered",
            )
        ),
    },
}


def run_v0_holdout() -> dict:
    """Run v0 formula on holdout months."""
    per_month = {}
    for month in EVAL_MONTHS:
        df = load_v62b_month(month)
        actual = df["realized_sp"].to_numpy().astype(np.float64)
        scores = -v62b_score(
            da_rank_value=df["da_rank_value"].to_numpy(),
            density_mix_rank_value=df["density_mix_rank_value"].to_numpy(),
            density_ori_rank_value=df["density_ori_rank_value"].to_numpy(),
        )
        metrics = evaluate_ltr(actual, scores)
        per_month[month] = metrics
        n_binding = int((actual > 0).sum())
        print(f"  v0 {month}: n={len(df)}, binding={n_binding}, VC@20={metrics['VC@20']:.4f}")
    return {"aggregate": aggregate_months(per_month), "per_month": per_month}


def run_ml_holdout(version_id: str, config: PipelineConfig) -> dict:
    """Run ML champion on holdout months."""
    per_month = {}
    for month in EVAL_MONTHS:
        result = run_pipeline(
            config=config,
            version_id=version_id,
            eval_month=month,
            period_type="f0",
            class_type="onpeak",
        )
        metrics = result.get("metrics", {})
        metrics.pop("_feature_importance", None)
        per_month[month] = metrics
        vc20 = metrics.get("VC@20", 0)
        print(f"  {version_id} {month}: VC@20={vc20:.4f}")
    return {"aggregate": aggregate_months(per_month), "per_month": per_month}


def write_result(version_id: str, result: dict, description: str, walltime: float,
                 period_type: str = "f0", class_type: str = "onpeak") -> None:
    """Write holdout result. Immutable — refuses to overwrite."""
    out_dir = holdout_root(period_type, class_type, base_dir=_HOLDOUT_BASE) / version_id
    out_path = out_dir / "metrics.json"

    if out_path.exists():
        print(f"  SKIP: {out_path} already exists (immutable)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "version": version_id,
        "description": description,
        "holdout_period": "2024-01 to 2025-12",
        "n_months": len(result["per_month"]),
        "walltime_s": round(walltime, 1),
        "aggregate": result["aggregate"],
        "per_month": result["per_month"],
    }
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"  Wrote {out_path}")


def print_comparison(period_type: str = "f0", class_type: str = "onpeak") -> None:
    """Print side-by-side comparison from all holdout results."""
    ho_dir = holdout_root(period_type, class_type, base_dir=_HOLDOUT_BASE)
    if not ho_dir.exists():
        return
    versions = {}
    for d in sorted(ho_dir.iterdir()):
        mp = d / "metrics.json"
        if mp.exists():
            data = json.loads(mp.read_text())
            versions[data["version"]] = data

    if not versions:
        return

    # Write comparison.json
    comparison = {v: {"description": d["description"], "aggregate": d["aggregate"],
                      "n_months": d["n_months"], "walltime_s": d.get("walltime_s", 0)}
                  for v, d in versions.items()}
    (ho_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))

    keys = ["VC@20", "VC@100", "Recall@20", "Recall@100", "NDCG", "Spearman"]
    print("\n" + "=" * 90)
    print("HOLDOUT TEST RESULTS (2024-01 to 2025-12, 24 months)")
    print("=" * 90)
    header = f"{'Version':<20}" + "".join(f"{k:>12}" for k in keys) + f"{'Time':>8}"
    print(header)
    print("-" * len(header))
    for v, data in versions.items():
        agg = data["aggregate"]
        # aggregate_months returns {"mean": {...}, "std": {...}, ...}
        means = agg.get("mean", agg)  # fallback to flat dict if already flat
        row = f"{v:<20}"
        for k in keys:
            row += f"{means.get(k, 0):>12.4f}"
        row += f"{data.get('walltime_s', 0):>7.0f}s"
        print(row)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="One-time holdout test")
    parser.add_argument("--ptype", default="f0", help="Period type (default: f0)")
    parser.add_argument("--class-type", default="onpeak", choices=["onpeak", "offpeak"])
    args = parser.parse_args()

    period_type = args.ptype
    class_type = args.class_type
    ho_dir = holdout_root(period_type, class_type, base_dir=_HOLDOUT_BASE)

    t_total = time.time()

    for version_id, champ in CHAMPIONS.items():
        out_path = ho_dir / version_id / "metrics.json"
        if out_path.exists():
            print(f"\n{'='*60}")
            print(f"{version_id}: already exists — skipping (immutable)")
            print(f"{'='*60}")
            continue

        print(f"\n{'='*60}")
        print(f"{version_id}: {champ['description']}")
        print(f"{'='*60}")

        clear_month_cache()
        t0 = time.time()
        if champ["type"] == "formula":
            result = run_v0_holdout()
        else:
            result = run_ml_holdout(version_id, champ["config"])
        walltime = time.time() - t0

        agg = result["aggregate"]
        means = agg.get("mean", agg)
        vc20 = means.get("VC@20", 0)
        print(f"{version_id} done in {walltime:.0f}s: VC@20={vc20:.4f}")
        write_result(version_id, result, champ["description"], walltime,
                     period_type=period_type, class_type=class_type)

    print_comparison(period_type=period_type, class_type=class_type)
    print(f"\nTotal holdout time: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
