"""DEPRECATED: Results archived to archive/registry/. Superseded by v10e-lag1.

v5: Full feature set (A+B+C = 12 features) with tiered labels.

Same features as v1b but with the label noise fix (tiered labels).
This isolates the label fix effect on the best previous feature set.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    LTRConfig, PipelineConfig, FEATURES_V1B, MONOTONE_V1B,
    _SCREEN_EVAL_MONTHS, _DEFAULT_EVAL_MONTHS,
)
from ml.benchmark import run_benchmark


def main():
    config = PipelineConfig(
        ltr=LTRConfig(
            features=list(FEATURES_V1B),
            monotone_constraints=list(MONOTONE_V1B),
            backend="lightgbm",
            label_mode="tiered",
        ),
        train_months=8,
        val_months=0,
    )

    print("=" * 60)
    print("v5 SCREEN — tiered labels + v1b features (A+B+C, 12 feat)")
    print(f"Features: {FEATURES_V1B}")
    print("=" * 60)
    result = run_benchmark(
        version_id="v5_screen",
        eval_months=_SCREEN_EVAL_MONTHS,
        config=config,
        mode="screen",
    )

    vc20 = result.get("aggregate", {}).get("mean", {}).get("VC@20", 0)
    spearman = result.get("aggregate", {}).get("mean", {}).get("Spearman", 0)
    print(f"\nv5 screen VC@20: {vc20:.4f} (v0: 0.2817, v1b: 0.2440)")
    print(f"v5 screen Spearman: {spearman:.4f} (v0: 0.2045, v1b: 0.1955)")

    if vc20 > 0.10:
        print("\nRunning full 12-month eval...")
        print("=" * 60)
        print("v5 FULL EVAL (12 months)")
        print("=" * 60)
        run_benchmark(
            version_id="v5",
            eval_months=_DEFAULT_EVAL_MONTHS,
            config=config,
            mode="eval",
        )


if __name__ == "__main__":
    main()
