"""v3: Tiered labels with V6.1 base features (Set A, 6 features).

Same features as v1, but uses tiered relevance labels instead of raw rank transform.
0 = non-binding (sp=0), 1-4 = quantile buckets of positive shadow prices.
This avoids assigning distinct fake ranks to ~58% of non-binding constraints.
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
        label_mode="tiered",
    ),
)

groups = SCREEN_EVAL_GROUPS if args.screen else DEFAULT_EVAL_GROUPS
mode = "screen" if args.screen else "eval"

run_benchmark(
    version_id="v3",
    eval_groups=groups,
    config=config,
    mode=mode,
)

# Compare against v1 (champion)
run_comparison(
    batch_id="annual",
    iteration=3,
    output_path=str(Path(__file__).resolve().parent.parent / "reports" / "v3_comparison.md"),
)
