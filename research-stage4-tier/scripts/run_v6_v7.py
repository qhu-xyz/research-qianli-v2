"""v6 and v7 experiments — with da_rank_value restored.

v6: Same features as V6.2B formula (6 features including da_rank_value)
    mean_branch_max, ori_mean, mix_mean, density_mix_rank_value,
    density_ori_rank_value, da_rank_value

v7: v6 + spice6 features (12 features total)
    + prob_exceed_110, prob_exceed_100, prob_exceed_90,
      prob_exceed_85, prob_exceed_80, constraint_limit
"""
import resource
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.benchmark import run_benchmark
from ml.config import (
    LTRConfig,
    PipelineConfig,
    _DEFAULT_EVAL_MONTHS,
    _SPICE6_FEATURES,
    _SPICE6_MONOTONE,
)


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# V6.2B features (now including da_rank_value)
_V62B_FULL_FEATURES = [
    "mean_branch_max",
    "ori_mean",
    "mix_mean",
    "density_mix_rank_value",
    "density_ori_rank_value",
    "da_rank_value",
]
_V62B_FULL_MONOTONE = [1, 1, 1, -1, -1, -1]
# da_rank_value: lower = more binding = negative monotone


def _make_config(features, monotone):
    ltr = LTRConfig(
        features=features,
        monotone_constraints=monotone,
        backend="lightgbm",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        min_child_weight=25,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        early_stopping_rounds=0,
    )
    return PipelineConfig(ltr=ltr, train_months=8, val_months=0)


def run_v6():
    """v6: V6.2B features including da_rank_value (6 features)."""
    cfg = _make_config(
        features=list(_V62B_FULL_FEATURES),
        monotone=list(_V62B_FULL_MONOTONE),
    )
    return run_benchmark(
        version_id="v6",
        eval_months=_DEFAULT_EVAL_MONTHS,
        config=cfg,
        mode="eval",
    )


def run_v7():
    """v7: V6.2B + spice6 features (12 features)."""
    features = list(_V62B_FULL_FEATURES) + list(_SPICE6_FEATURES)
    monotone = list(_V62B_FULL_MONOTONE) + list(_SPICE6_MONOTONE)
    cfg = _make_config(features=features, monotone=monotone)
    return run_benchmark(
        version_id="v7",
        eval_months=_DEFAULT_EVAL_MONTHS,
        config=cfg,
        mode="eval",
    )


def main():
    print(f"[run_v6_v7] mem at start: {mem_mb():.0f} MB")

    r6 = run_v6()
    r7 = run_v7()

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON (12-month means)")
    print("=" * 70)
    metrics = ["VC@20", "VC@100", "Recall@20", "Recall@50", "Recall@100", "NDCG", "Spearman"]
    print(f"{'Metric':>12} {'v6 (6feat)':>12} {'v7 (12feat)':>12}")
    print("-" * 40)
    for m in metrics:
        v6_val = r6["aggregate"]["mean"].get(m, 0)
        v7_val = r7["aggregate"]["mean"].get(m, 0)
        print(f"{m:>12} {v6_val:>12.4f} {v7_val:>12.4f}")

    print(f"\n[run_v6_v7] mem at end: {mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
