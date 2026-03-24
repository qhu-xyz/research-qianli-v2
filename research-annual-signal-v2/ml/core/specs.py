"""Spec types for model, benchmark, and policy registry entries.

These are data classes — they define the shape of spec.json, not the
logic for building models. Implementations live in ml/markets/{rto}/.

See docs/contracts/registry-schema.md for the full contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class TrainWindow:
    type: str  # "expanding", "fixed", "rolling"
    train_pys: list[str]
    eval_pys: list[str]


@dataclass(frozen=True)
class CacheProvenance:
    """Which caches were used to build this entry."""
    density_limits: str  # e.g. "data/collapsed/_r1_ via load_collapsed(market_round=1)"
    history: str         # e.g. "data/realized_da/ + /opt/tmp/qianli/realized_da_daily/"
    model_tables: str    # e.g. "data/nb_cache/ (legacy R1-only)"


RoundSensitivity = Literal["r1_only", "round_aware", "round_independent"]
RankDirection = Literal["ascending", "descending"]
SpecType = Literal["model", "benchmark", "policy"]


@dataclass(frozen=True)
class ModelSpec:
    """Identity and provenance for a trained model."""
    spec_type: SpecType = field(default="model", init=False)
    model_id: str
    market: str
    product: str
    class_type: str  # "onpeak" or "offpeak"
    market_round: int
    universe_id: str
    feature_recipe_id: str
    label_recipe_id: str
    objective: str
    train_window: TrainWindow
    eval_quarters: list[str]
    round_sensitivity: RoundSensitivity
    code_commit: str
    cache_provenance: CacheProvenance


@dataclass(frozen=True)
class BenchmarkSpec:
    """Identity for an external benchmark signal."""
    spec_type: SpecType = field(default="benchmark", init=False)
    benchmark_id: str
    market: str
    product: str
    class_type: str
    market_round: int
    universe_id: str
    signal_path: str
    rank_direction: RankDirection
    round_sensitivity: RoundSensitivity
    eval_quarters: list[str]


@dataclass(frozen=True)
class PolicySpec:
    """Identity for a deployment policy (score → selection rule).

    class_type may be None if the policy applies identically to both ctypes
    (e.g., same allocation ratios). If ctype-specific behavior exists,
    create separate PolicySpec entries per ctype.
    """
    spec_type: SpecType = field(default="policy", init=False)
    policy_id: str
    market: str
    product: str
    class_type: str | None  # None = applies to both ctypes identically
    market_round: int
    primary_model_id: str
    secondary_model_id: str | None
    allocation: dict[str, dict[str, int]]  # {"K_200": {"primary": 170, "secondary_dormant": 30}}
    dormant_definition: str | None  # e.g. "class-specific BF_12 == 0"
    round_sensitivity: RoundSensitivity
