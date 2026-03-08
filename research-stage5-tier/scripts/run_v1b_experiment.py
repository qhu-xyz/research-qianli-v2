"""v1b: LTR with Groups A+B+C (12 features, adds da_rank_value)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    LTRConfig, PipelineConfig,
    FEATURES_V1B, MONOTONE_V1B,
    _SCREEN_EVAL_MONTHS, _DEFAULT_EVAL_MONTHS,
)
from ml.benchmark import run_benchmark

def main():
    config = PipelineConfig(
        ltr=LTRConfig(
            features=list(FEATURES_V1B),
            monotone_constraints=list(MONOTONE_V1B),
            backend="lightgbm",
        ),
        train_months=8,
        val_months=0,
    )

    # Screen first (4 months)
    print("=" * 60)
    print("v1b SCREEN (4 months)")
    print("=" * 60)
    result = run_benchmark(
        version_id="v1b_screen",
        eval_months=_SCREEN_EVAL_MONTHS,
        config=config,
        mode="screen",
    )

    vc20_mean = result.get("aggregate", {}).get("mean", {}).get("VC@20", 0)
    print(f"\nv1b screen VC@20 mean: {vc20_mean:.4f} (v0 baseline: 0.2817)")

    if vc20_mean > 0.15:  # reasonable threshold -- even matching v0 is interesting
        print("\nRunning full 12-month eval...")
        print("=" * 60)
        print("v1b FULL EVAL (12 months)")
        print("=" * 60)
        run_benchmark(
            version_id="v1b",
            eval_months=_DEFAULT_EVAL_MONTHS,
            config=config,
            mode="eval",
        )
    else:
        print("\nNot promising. Skipping full eval.")

if __name__ == "__main__":
    main()
