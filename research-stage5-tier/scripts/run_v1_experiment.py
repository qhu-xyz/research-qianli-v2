"""v1: LTR with Groups A+B (11 features, no historical DA signal)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    LTRConfig, PipelineConfig,
    FEATURES_V1, MONOTONE_V1,
    _SCREEN_EVAL_MONTHS, _DEFAULT_EVAL_MONTHS,
)
from ml.benchmark import run_benchmark

def main():
    config = PipelineConfig(
        ltr=LTRConfig(
            features=list(FEATURES_V1),
            monotone_constraints=list(MONOTONE_V1),
            backend="lightgbm",
        ),
        train_months=8,
        val_months=0,
    )

    # Screen first (4 months)
    print("=" * 60)
    print("v1 SCREEN (4 months)")
    print("=" * 60)
    result = run_benchmark(
        version_id="v1_screen",
        eval_months=_SCREEN_EVAL_MONTHS,
        config=config,
        mode="screen",
    )

    vc20_mean = result.get("aggregate", {}).get("mean", {}).get("VC@20", 0)
    print(f"\nv1 screen VC@20 mean: {vc20_mean:.4f} (v0 baseline: 0.2817)")

    if vc20_mean > 0.15:  # reasonable threshold -- even matching v0 is interesting
        print("\nRunning full 12-month eval...")
        print("=" * 60)
        print("v1 FULL EVAL (12 months)")
        print("=" * 60)
        run_benchmark(
            version_id="v1",
            eval_months=_DEFAULT_EVAL_MONTHS,
            config=config,
            mode="eval",
        )
    else:
        print("\nNot promising. Skipping full eval.")

if __name__ == "__main__":
    main()
