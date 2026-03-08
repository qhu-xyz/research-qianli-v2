"""v1: LightGBM LambdaRank with V6.1 base features (Set A, 6 features).

Same information available to V6.1 formula, but ML-learned weights.
"""
from ml.benchmark import run_benchmark
from ml.config import (
    PipelineConfig, LTRConfig,
    SET_A_FEATURES, SET_A_MONOTONE,
    SCREEN_EVAL_GROUPS, DEFAULT_EVAL_GROUPS,
)
from ml.compare import run_comparison

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--screen", action="store_true")
args = parser.parse_args()

config = PipelineConfig(
    ltr=LTRConfig(
        features=SET_A_FEATURES,
        monotone_constraints=SET_A_MONOTONE,
        backend="lightgbm",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
    ),
)

groups = SCREEN_EVAL_GROUPS if args.screen else DEFAULT_EVAL_GROUPS
mode = "screen" if args.screen else "eval"

run_benchmark(
    version_id="v1",
    eval_groups=groups,
    config=config,
    mode=mode,
)

# Compare against v0
run_comparison(
    batch_id="annual",
    iteration=1,
    output_path=str(Path(__file__).resolve().parent.parent / "reports" / "v1_comparison.md"),
)
