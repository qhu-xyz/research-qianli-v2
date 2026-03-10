"""Rerun a registered version with leakage guards applied.

This is intended for "clean" re-evaluations when a prior version's config
included leaky features (e.g. da_rank_value from V6.2B parquet).

Usage:
  /home/xyz/workspace/pmodel/.venv/bin/python -m ml.rerun_registry \\
    --src-version v1 --dst-version v1_clean
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.benchmark import run_benchmark
from ml.config import LTRConfig, PipelineConfig


def _infer_backend(ltr_cfg: dict[str, Any]) -> str:
    backend = ltr_cfg.get("backend")
    if backend:
        return str(backend)
    # Older configs stored an explicit XGBoost objective but not "backend".
    if "objective" in ltr_cfg:
        return "xgboost"
    return "lightgbm"


def load_pipeline_config_from_registry(config_path: str | Path) -> tuple[PipelineConfig, dict[str, Any]]:
    """Load PipelineConfig + eval_config from an existing registry config.json."""
    config_path = Path(config_path)
    data = json.loads(config_path.read_text())

    ltr_raw: dict[str, Any] = data.get("ltr", {}) if isinstance(data, dict) else {}
    pipeline_raw: dict[str, Any] = data.get("pipeline", {}) if isinstance(data, dict) else {}
    eval_raw: dict[str, Any] = data.get("eval_config", {}) if isinstance(data, dict) else {}

    features = list(ltr_raw.get("features", []))
    monotone = list(ltr_raw.get("monotone_constraints", [0] * len(features)))

    ltr = LTRConfig(
        features=features,
        monotone_constraints=monotone,
        backend=_infer_backend(ltr_raw),
        n_estimators=int(ltr_raw.get("n_estimators", 100)),
        learning_rate=float(ltr_raw.get("learning_rate", 0.05)),
        min_child_weight=int(ltr_raw.get("min_child_weight", 25)),
        num_leaves=int(ltr_raw.get("num_leaves", 31)),
        subsample=float(ltr_raw.get("subsample", 0.8)),
        colsample_bytree=float(ltr_raw.get("colsample_bytree", 0.8)),
        reg_alpha=float(ltr_raw.get("reg_alpha", 1.0)),
        reg_lambda=float(ltr_raw.get("reg_lambda", 1.0)),
        max_depth=int(ltr_raw.get("max_depth", 5)),
        early_stopping_rounds=int(ltr_raw.get("early_stopping_rounds", 20)),
    )

    cfg = PipelineConfig(
        ltr=ltr,
        train_months=int(pipeline_raw.get("train_months", eval_raw.get("train_months", 6))),
        val_months=int(pipeline_raw.get("val_months", eval_raw.get("val_months", 2))),
    )
    return cfg, eval_raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun a registry version with leakage guards.")
    parser.add_argument("--src-version", required=True, help="Existing version directory (e.g. v1)")
    parser.add_argument("--dst-version", required=True, help="New version directory (e.g. v1_clean)")
    parser.add_argument("--registry-dir", default="registry", help="Registry root directory")
    args = parser.parse_args()

    registry_dir = Path(args.registry_dir)
    src_dir = registry_dir / args.src_version
    dst_version = args.dst_version

    config_path = src_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {config_path}")

    cfg, eval_cfg = load_pipeline_config_from_registry(config_path)

    eval_months = eval_cfg.get("eval_months")
    if not eval_months:
        raise ValueError(f"No eval_months found in {config_path}")

    class_type = str(eval_cfg.get("class_type", "onpeak"))
    period_type = str(eval_cfg.get("period_type", "f0"))
    mode = str(eval_cfg.get("mode", "custom"))

    print(
        f"[rerun_registry] src={args.src_version} dst={dst_version} "
        f"months={len(eval_months)} backend={cfg.ltr.backend}"
    )

    run_benchmark(
        version_id=dst_version,
        eval_months=list(eval_months),
        class_type=class_type,
        period_type=period_type,
        registry_dir=str(registry_dir),
        config=cfg,
        mode=mode,
    )


if __name__ == "__main__":
    main()

