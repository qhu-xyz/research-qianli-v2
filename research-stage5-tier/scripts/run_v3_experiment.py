"""v3: LTR with ONLY the 3 formula inputs (da_rank_value, density_mix_rank_value, density_ori_rank_value).

If ML can't beat v0 formula with the EXACT SAME 3 features, something is wrong.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    LTRConfig, PipelineConfig,
    _SCREEN_EVAL_MONTHS, _DEFAULT_EVAL_MONTHS,
)
from ml.benchmark import run_benchmark

# Exact same 3 features as v0 formula
_V3_FEATURES = [
    "da_rank_value",           # 60% weight in formula
    "density_mix_rank_value",  # 30% weight in formula
    "density_ori_rank_value",  # 10% weight in formula
]
_V3_MONOTONE = [-1, -1, -1]  # all: lower = more binding


def main():
    config = PipelineConfig(
        ltr=LTRConfig(
            features=list(_V3_FEATURES),
            monotone_constraints=list(_V3_MONOTONE),
            backend="lightgbm",
        ),
        train_months=8,
        val_months=0,
    )

    # Screen first (4 months)
    print("=" * 60)
    print("v3 SCREEN (4 months) — formula's 3 features only")
    print(f"Features: {_V3_FEATURES}")
    print("=" * 60)
    result = run_benchmark(
        version_id="v3_screen",
        eval_months=_SCREEN_EVAL_MONTHS,
        config=config,
        mode="screen",
    )

    vc20_mean = result.get("aggregate", {}).get("mean", {}).get("VC@20", 0)
    spearman_mean = result.get("aggregate", {}).get("mean", {}).get("Spearman", 0)
    print(f"\nv3 screen VC@20 mean: {vc20_mean:.4f} (v0 baseline: 0.2817)")
    print(f"v3 screen Spearman mean: {spearman_mean:.4f} (v0 baseline: 0.2045)")

    if vc20_mean > 0.10:  # low bar — we want to see full results regardless
        print("\nRunning full 12-month eval...")
        print("=" * 60)
        print("v3 FULL EVAL (12 months)")
        print("=" * 60)
        run_benchmark(
            version_id="v3",
            eval_months=_DEFAULT_EVAL_MONTHS,
            config=config,
            mode="eval",
        )
    else:
        print("\nVC@20 < 0.10 — something is very wrong. Skipping full eval.")


if __name__ == "__main__":
    main()
