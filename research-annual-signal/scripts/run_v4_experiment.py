"""v4: Raw rank labels + formula score (rank_ori) as 7th feature (Set AF).

Same label strategy as v1 (raw rank transform), but adds rank_ori (V6.1 formula
output) as a feature. rank_ori encodes domain knowledge: 0.60*da_rank + 0.30*density_mix + 0.10*density_ori.
"""
from ml.benchmark import run_benchmark
from ml.config import (
    PipelineConfig, LTRConfig,
    SET_AF_FEATURES, SET_AF_MONOTONE,
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
        features=SET_AF_FEATURES,
        monotone_constraints=SET_AF_MONOTONE,
        backend="lightgbm",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        label_mode="rank",
    ),
)

groups = SCREEN_EVAL_GROUPS if args.screen else DEFAULT_EVAL_GROUPS
mode = "screen" if args.screen else "eval"

run_benchmark(
    version_id="v4",
    eval_groups=groups,
    config=config,
    mode=mode,
)

# Compare against v1 (champion)
run_comparison(
    batch_id="annual",
    iteration=4,
    output_path=str(Path(__file__).resolve().parent.parent / "reports" / "v4_comparison.md"),
)
