"""v2: LightGBM LambdaRank with V6.1 + spice6 density (Set B, 11 features)."""
from ml.benchmark import run_benchmark
from ml.config import (
    PipelineConfig, LTRConfig,
    SET_B_FEATURES, SET_B_MONOTONE,
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
        features=SET_B_FEATURES,
        monotone_constraints=SET_B_MONOTONE,
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
    version_id="v2",
    eval_groups=groups,
    config=config,
    mode=mode,
)

run_comparison(
    batch_id="annual",
    iteration=2,
    output_path=str(Path(__file__).resolve().parent.parent / "reports" / "v2_comparison.md"),
)
