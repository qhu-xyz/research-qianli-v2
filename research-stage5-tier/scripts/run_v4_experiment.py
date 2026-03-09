"""DEPRECATED: Results archived to archive/registry/. Superseded by v10e-lag1.

v4: Fix the label noise bug + add formula score as feature.

Two changes from v3:
  1. Tiered labels (0/1/2/3) instead of raw rank (~600 levels).
     Raw rank assigns 528 distinct labels to non-binding constraints
     that all have realized_sp=0 — LightGBM wastes capacity on noise.
  2. Formula score (0.60*da + 0.30*dmix + 0.10*dori) as a feature,
     so ML can learn to deviate from it when beneficial.

Also runs v4a (tiered labels only, same 3 features as v3) to isolate
the label fix vs the formula feature.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    LTRConfig, PipelineConfig,
    _SCREEN_EVAL_MONTHS, _DEFAULT_EVAL_MONTHS,
)
from ml.benchmark import run_benchmark

# v4a: Same 3 features as v3, but with tiered labels (isolates the label fix)
_V4A_FEATURES = [
    "da_rank_value",
    "density_mix_rank_value",
    "density_ori_rank_value",
]
_V4A_MONOTONE = [-1, -1, -1]

# v4b: Same 3 features + formula score as 4th feature
_V4B_FEATURES = [
    "da_rank_value",
    "density_mix_rank_value",
    "density_ori_rank_value",
    "v62b_formula_score",
]
_V4B_MONOTONE = [-1, -1, -1, -1]  # lower formula score = more binding


def main():
    # ── v4a: tiered labels, same 3 features ──
    config_4a = PipelineConfig(
        ltr=LTRConfig(
            features=list(_V4A_FEATURES),
            monotone_constraints=list(_V4A_MONOTONE),
            backend="lightgbm",
            label_mode="tiered",
        ),
        train_months=8,
        val_months=0,
    )

    print("=" * 60)
    print("v4a SCREEN — tiered labels, formula's 3 features only")
    print(f"Features: {_V4A_FEATURES}")
    print("=" * 60)
    result_4a = run_benchmark(
        version_id="v4a_screen",
        eval_months=_SCREEN_EVAL_MONTHS,
        config=config_4a,
        mode="screen",
    )
    vc20_4a = result_4a.get("aggregate", {}).get("mean", {}).get("VC@20", 0)
    spearman_4a = result_4a.get("aggregate", {}).get("mean", {}).get("Spearman", 0)
    print(f"\nv4a screen VC@20: {vc20_4a:.4f} (v0: 0.2817, v3: 0.1709)")
    print(f"v4a screen Spearman: {spearman_4a:.4f} (v0: 0.2045, v3: 0.1806)")

    # ── v4b: tiered labels + formula feature ──
    config_4b = PipelineConfig(
        ltr=LTRConfig(
            features=list(_V4B_FEATURES),
            monotone_constraints=list(_V4B_MONOTONE),
            backend="lightgbm",
            label_mode="tiered",
        ),
        train_months=8,
        val_months=0,
    )

    print("\n" + "=" * 60)
    print("v4b SCREEN — tiered labels + formula score feature")
    print(f"Features: {_V4B_FEATURES}")
    print("=" * 60)
    result_4b = run_benchmark(
        version_id="v4b_screen",
        eval_months=_SCREEN_EVAL_MONTHS,
        config=config_4b,
        mode="screen",
    )
    vc20_4b = result_4b.get("aggregate", {}).get("mean", {}).get("VC@20", 0)
    spearman_4b = result_4b.get("aggregate", {}).get("mean", {}).get("Spearman", 0)
    print(f"\nv4b screen VC@20: {vc20_4b:.4f} (v0: 0.2817)")
    print(f"v4b screen Spearman: {spearman_4b:.4f} (v0: 0.2045)")

    # Run full eval for whichever is better
    best_id = "v4a" if vc20_4a >= vc20_4b else "v4b"
    best_config = config_4a if best_id == "v4a" else config_4b
    best_vc20 = max(vc20_4a, vc20_4b)

    print(f"\n{'=' * 60}")
    print(f"Best screen: {best_id} (VC@20={best_vc20:.4f})")

    if best_vc20 > 0.10:
        print(f"Running {best_id} full 12-month eval...")
        print("=" * 60)
        run_benchmark(
            version_id=best_id,
            eval_months=_DEFAULT_EVAL_MONTHS,
            config=best_config,
            mode="eval",
        )

        # Also run full eval for the other one if it's close
        other_id = "v4b" if best_id == "v4a" else "v4a"
        other_vc20 = vc20_4b if best_id == "v4a" else vc20_4a
        other_config = config_4b if best_id == "v4a" else config_4a
        if other_vc20 > 0.10:
            print(f"\nAlso running {other_id} full eval (VC@20={other_vc20:.4f})...")
            run_benchmark(
                version_id=other_id,
                eval_months=_DEFAULT_EVAL_MONTHS,
                config=other_config,
                mode="eval",
            )
    else:
        print("Both < 0.10 — something is still wrong.")


if __name__ == "__main__":
    main()
