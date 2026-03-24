"""Registry I/O — read/write spec.json and metrics.json.

Handles serialization of spec types and metric cells to/from the
registry directory layout defined in docs/contracts/registry-schema.md.

This module owns the registry file format. It does NOT own model training,
scoring, or evaluation logic — those live in ml/markets/{rto}/.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ml.core.specs import ModelSpec, BenchmarkSpec, PolicySpec
from ml.core.metrics import MetricCell, MetricsResult


REGISTRY_BASE = Path(__file__).resolve().parent.parent.parent / "registry"


def _entry_dir(
    market: str,
    product: str,
    spec_type: str,
    entry_id: str,
    class_type: str | None = None,
) -> Path:
    """Build registry path: registry/{market}/{product}/{type_plural}/{entry_id}/{ctype}/"""
    type_map = {"model": "models", "benchmark": "benchmarks", "policy": "policies"}
    parts = [REGISTRY_BASE, market, product, type_map[spec_type], entry_id]
    if class_type is not None:
        parts.append(class_type)
    return Path(*[str(p) for p in parts])


def save_spec(spec: ModelSpec | BenchmarkSpec | PolicySpec) -> Path:
    """Write spec.json to the correct registry path."""
    entry_id = getattr(spec, "model_id", None) or getattr(spec, "benchmark_id", None) or getattr(spec, "policy_id", None)
    assert entry_id is not None, "Spec must have an ID field"

    out_dir = _entry_dir(
        market=spec.market,
        product=spec.product,
        spec_type=spec.spec_type,
        entry_id=entry_id,
        class_type=spec.class_type,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / "spec.json"

    data = asdict(spec)
    with open(spec_path, "w") as f:
        json.dump(data, f, indent=2)
    return spec_path


def save_metrics(
    spec: ModelSpec | BenchmarkSpec | PolicySpec,
    metrics: MetricsResult,
) -> Path:
    """Write metrics.json to the correct registry path."""
    entry_id = getattr(spec, "model_id", None) or getattr(spec, "benchmark_id", None) or getattr(spec, "policy_id", None)
    assert entry_id is not None

    out_dir = _entry_dir(
        market=spec.market,
        product=spec.product,
        spec_type=spec.spec_type,
        entry_id=entry_id,
        class_type=spec.class_type,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"

    data = asdict(metrics)
    with open(metrics_path, "w") as f:
        json.dump(data, f, indent=2)
    return metrics_path


def load_spec(path: Path) -> dict:
    """Load spec.json as raw dict (caller decides which dataclass to use)."""
    with open(path) as f:
        return json.load(f)


def load_metrics(path: Path) -> dict:
    """Load metrics.json as raw dict."""
    with open(path) as f:
        return json.load(f)


def validate_spec(data: dict) -> list[str]:
    """Return list of missing required fields for the given spec_type."""
    errors = []
    spec_type = data.get("spec_type")
    if spec_type is None:
        return ["missing spec_type"]

    common = ["market", "product", "market_round", "round_sensitivity", "eval_quarters"]
    model_required = common + [
        "model_id", "class_type", "universe_id", "feature_recipe_id",
        "label_recipe_id", "objective", "train_window", "code_commit", "cache_provenance",
    ]
    benchmark_required = common + [
        "benchmark_id", "class_type", "universe_id", "signal_path", "rank_direction",
    ]
    policy_required = common + [
        "policy_id", "primary_model_id", "allocation",
    ]

    required_map = {
        "model": model_required,
        "benchmark": benchmark_required,
        "policy": policy_required,
    }
    required = required_map.get(spec_type, [])
    for field in required:
        if field not in data or data[field] is None:
            errors.append(f"missing {field}")
    return errors


def validate_metrics(data: dict) -> list[str]:
    """Return list of issues in metrics.json."""
    errors = []
    cells = data.get("cells")
    if cells is None:
        return ["missing cells"]
    if not isinstance(cells, list):
        return ["cells is not a list"]

    required_cell_fields = [
        "planning_year", "aq_quarter", "class_type", "market_round",
        "K", "sp", "binders", "precision", "vc", "recall",
        "nb_in", "nb_binders", "nb_sp",
    ]
    for i, cell in enumerate(cells):
        for field in required_cell_fields:
            if field not in cell:
                errors.append(f"cell[{i}] missing {field}")
    return errors
