"""Run v1 (V6.2B features only) and v2 (V6.2B + spice6) against fair baseline.

v1: 5 features — same as V6.2B forecast components
v2: 11 features — V6.2B + spice6 density
Both use LightGBM lambdarank WITHOUT early stopping (features are low-signal,
early stopping kills training at iteration 1).
"""
from ml.benchmark import run_benchmark
from ml.compare import run_comparison
from ml.config import (
    LTRConfig,
    PipelineConfig,
    _DEFAULT_EVAL_MONTHS,
    _V62B_FEATURES,
    _V62B_MONOTONE,
    _SPICE6_FEATURES,
    _SPICE6_MONOTONE,
)


def run_v1():
    """v1: LightGBM with V6.2B forecast features only (5 features)."""
    print("\n" + "=" * 60)
    print("  v1: V6.2B forecast features only (5 features)")
    print("=" * 60)

    ltr = LTRConfig(
        features=list(_V62B_FEATURES),
        monotone_constraints=list(_V62B_MONOTONE),
        backend="lightgbm",
        n_estimators=100,
        learning_rate=0.05,
        min_child_weight=25,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        early_stopping_rounds=0,  # disabled — low-signal features
    )

    cfg = PipelineConfig(ltr=ltr, train_months=6, val_months=2)
    print(f"Features ({len(cfg.ltr.features)}): {cfg.ltr.features}")

    return run_benchmark(
        version_id="v1",
        eval_months=_DEFAULT_EVAL_MONTHS,
        config=cfg,
        mode="eval",
    )


def run_v2():
    """v2: LightGBM with V6.2B + spice6 features (11 features)."""
    print("\n" + "=" * 60)
    print("  v2: V6.2B + spice6 density features (11 features)")
    print("=" * 60)

    all_features = list(_V62B_FEATURES) + list(_SPICE6_FEATURES)
    all_monotone = list(_V62B_MONOTONE) + list(_SPICE6_MONOTONE)

    ltr = LTRConfig(
        features=all_features,
        monotone_constraints=all_monotone,
        backend="lightgbm",
        n_estimators=100,
        learning_rate=0.05,
        min_child_weight=25,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        early_stopping_rounds=0,  # disabled — low-signal features
    )

    cfg = PipelineConfig(ltr=ltr, train_months=6, val_months=2)
    print(f"Features ({len(cfg.ltr.features)}): {cfg.ltr.features}")

    return run_benchmark(
        version_id="v2",
        eval_months=_DEFAULT_EVAL_MONTHS,
        config=cfg,
        mode="eval",
    )


def main():
    run_v1()
    run_v2()

    print("\n" + "=" * 60)
    print("  Comparison: v0 (baseline) vs v1 vs v2")
    print("=" * 60)

    run_comparison(
        batch_id="fair",
        iteration=1,
        registry_dir="registry",
        output_path="reports/fair_comparison.md",
    )


if __name__ == "__main__":
    main()
