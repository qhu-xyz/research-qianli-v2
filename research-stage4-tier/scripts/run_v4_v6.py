"""Rerun v1-v3 settings with 8mo train / 0 val → v4, v5, v6, then consolidate.

v4: v1 settings (5 features, lr=0.05, 100 trees) + 8mo train, 0 val
v5: v2 settings (11 features, lr=0.05, 100 trees) + 8mo train, 0 val
v6: v3 settings (11 features, lr=0.02, 300 trees, leaves=15) + 8mo train, 0 val
"""
import json
import shutil
from pathlib import Path

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


def _make_config(
    features: list[str],
    monotone: list[int],
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    min_child_weight: int = 25,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 1.0,
    reg_lambda: float = 1.0,
    early_stopping_rounds: int = 0,
    train_months: int = 8,
    val_months: int = 0,
) -> PipelineConfig:
    ltr = LTRConfig(
        features=list(features),
        monotone_constraints=list(monotone),
        backend="lightgbm",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        early_stopping_rounds=early_stopping_rounds,
    )
    return PipelineConfig(ltr=ltr, train_months=train_months, val_months=val_months)


def _run(version_id: str, cfg: PipelineConfig, eval_months: list[str], mode: str = "eval"):
    print(f"\n{'=' * 60}")
    print(f"  {version_id}: {len(cfg.ltr.features)} features, "
          f"lr={cfg.ltr.learning_rate}, trees={cfg.ltr.n_estimators}, "
          f"leaves={cfg.ltr.num_leaves}, "
          f"train={cfg.train_months}mo, val={cfg.val_months}mo")
    print(f"{'=' * 60}")
    print(f"Features: {cfg.ltr.features}")
    return run_benchmark(
        version_id=version_id,
        eval_months=eval_months,
        config=cfg,
        mode=mode,
    )


def _print_summary(version_id: str, result: dict):
    agg = result.get("aggregate", {}).get("mean", {})
    print(f"\n[{version_id}] VC@20={agg.get('VC@20', 0):.4f}  "
          f"VC@100={agg.get('VC@100', 0):.4f}  "
          f"Recall@20={agg.get('Recall@20', 0):.4f}  "
          f"NDCG={agg.get('NDCG', 0):.4f}  "
          f"Spearman={agg.get('Spearman', 0):.4f}")


def _load_agg(version_id: str) -> dict:
    path = Path("registry") / version_id / "metrics.json"
    if path.exists():
        data = json.loads(path.read_text())
        return data.get("aggregate", {}).get("mean", {})
    return {}


def run_v4():
    """v4: v1 settings + 8mo train, 0 val."""
    cfg = _make_config(
        features=list(_V62B_FEATURES),
        monotone=list(_V62B_MONOTONE),
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        min_child_weight=25,
        reg_alpha=1.0,
        reg_lambda=1.0,
    )
    result = _run("v4", cfg, _DEFAULT_EVAL_MONTHS)
    _print_summary("v4", result)
    return result


def run_v5():
    """v5: v2 settings + 8mo train, 0 val."""
    features = list(_V62B_FEATURES) + list(_SPICE6_FEATURES)
    monotone = list(_V62B_MONOTONE) + list(_SPICE6_MONOTONE)
    cfg = _make_config(
        features=features,
        monotone=monotone,
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        min_child_weight=25,
        reg_alpha=1.0,
        reg_lambda=1.0,
    )
    result = _run("v5", cfg, _DEFAULT_EVAL_MONTHS)
    _print_summary("v5", result)
    return result


def run_v6():
    """v6: v3 settings + 8mo train, 0 val."""
    features = list(_V62B_FEATURES) + list(_SPICE6_FEATURES)
    monotone = list(_V62B_MONOTONE) + list(_SPICE6_MONOTONE)
    cfg = _make_config(
        features=features,
        monotone=monotone,
        n_estimators=300,
        learning_rate=0.02,
        num_leaves=15,
        min_child_weight=50,
        reg_alpha=2.0,
        reg_lambda=2.0,
    )
    result = _run("v6", cfg, _DEFAULT_EVAL_MONTHS)
    _print_summary("v6", result)
    return result


def consolidate():
    """Compare all versions and print summary."""
    print(f"\n{'=' * 60}")
    print("  CONSOLIDATION")
    print(f"{'=' * 60}")

    run_comparison(
        batch_id="consolidated",
        iteration=3,
        registry_dir="registry",
        output_path="reports/consolidated_comparison.md",
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for vdir in sorted(Path("registry").glob("v*")):
        agg = _load_agg(vdir.name)
        if agg:
            print(f"  {vdir.name}: VC@20={agg.get('VC@20', 0):.4f}  "
                  f"VC@100={agg.get('VC@100', 0):.4f}  "
                  f"Recall@20={agg.get('Recall@20', 0):.4f}  "
                  f"NDCG={agg.get('NDCG', 0):.4f}  "
                  f"Spearman={agg.get('Spearman', 0):.4f}")

    # Delete v1 (known inferior)
    v1_dir = Path("registry/v1")
    if v1_dir.exists():
        shutil.rmtree(v1_dir)
        print("  Cleaned up v1 (inferior to v0)")


def main():
    # Clean up any old v4/v5/v6 from previous runs
    for vid in ["v4", "v5", "v6"]:
        vdir = Path("registry") / vid
        if vdir.exists():
            shutil.rmtree(vdir)
            print(f"Cleaned up old {vid}")

    run_v4()
    run_v5()
    run_v6()
    consolidate()


if __name__ == "__main__":
    main()
