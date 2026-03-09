"""DEPRECATED: Results archived to archive/registry/. Superseded by v10e-lag1.

v6: Regression vs ranking comparison.

Tests:
  v6a: LightGBM regression on raw realized_sp (12 features = v1b set)
  v6b: LightGBM regression on raw realized_sp (12 features + formula score)
  v6c: LightGBM tiered ranking (12 features + formula score) — combines v5 + v4b

If regression works better, it means the ranking framework was hurting us.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    LTRConfig, PipelineConfig, FEATURES_V1B, MONOTONE_V1B,
    _SCREEN_EVAL_MONTHS, _DEFAULT_EVAL_MONTHS,
    _HIST_DA_MONOTONE,
)
from ml.benchmark import run_benchmark

# v1b features + formula score
_V6_FEATURES = list(FEATURES_V1B) + ["v62b_formula_score"]
_V6_MONOTONE = list(MONOTONE_V1B) + [-1]  # lower formula score = more binding


def _run_variant(name, features, monotone, backend, label_mode):
    config = PipelineConfig(
        ltr=LTRConfig(
            features=list(features),
            monotone_constraints=list(monotone),
            backend=backend,
            label_mode=label_mode,
        ),
        train_months=8,
        val_months=0,
    )

    print(f"\n{'=' * 60}")
    print(f"{name} SCREEN — {backend}, label_mode={label_mode}")
    print(f"Features ({len(features)}): {features}")
    print("=" * 60)
    result = run_benchmark(
        version_id=f"{name}_screen",
        eval_months=_SCREEN_EVAL_MONTHS,
        config=config,
        mode="screen",
    )
    vc20 = result.get("aggregate", {}).get("mean", {}).get("VC@20", 0)
    spearman = result.get("aggregate", {}).get("mean", {}).get("Spearman", 0)
    print(f"\n{name} screen: VC@20={vc20:.4f}, Spearman={spearman:.4f}")
    return name, vc20, spearman, config


def main():
    results = []

    # v6a: regression, 12 features (same as v5 but regression instead of ranking)
    results.append(_run_variant(
        "v6a", FEATURES_V1B, MONOTONE_V1B,
        backend="lightgbm_regression", label_mode="tiered",  # label_mode ignored for regression
    ))

    # v6b: regression, 12 features + formula score
    results.append(_run_variant(
        "v6b", _V6_FEATURES, _V6_MONOTONE,
        backend="lightgbm_regression", label_mode="tiered",
    ))

    # v6c: tiered ranking, 12 features + formula score
    results.append(_run_variant(
        "v6c", _V6_FEATURES, _V6_MONOTONE,
        backend="lightgbm", label_mode="tiered",
    ))

    # Summary
    print(f"\n{'=' * 60}")
    print("SCREEN SUMMARY (v0 baseline: VC@20=0.2817, Spearman=0.2045)")
    print("=" * 60)
    for name, vc20, spearman, _ in results:
        print(f"  {name}: VC@20={vc20:.4f}, Spearman={spearman:.4f}")

    # Full eval for best
    best_name, best_vc20, _, best_config = max(results, key=lambda x: x[1])
    print(f"\nBest: {best_name} (VC@20={best_vc20:.4f})")

    if best_vc20 > 0.10:
        for name, vc20, _, config in results:
            if vc20 > 0.20:  # run full eval for all promising variants
                print(f"\nRunning {name} full 12-month eval...")
                run_benchmark(
                    version_id=name,
                    eval_months=_DEFAULT_EVAL_MONTHS,
                    config=config,
                    mode="eval",
                )


if __name__ == "__main__":
    main()
