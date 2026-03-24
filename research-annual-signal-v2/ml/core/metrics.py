"""Metric cell and aggregate types for registry metrics.json.

Base grain: (planning_year, aq_quarter, class_type, market_round).
See docs/contracts/registry-schema.md for the full contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetricCell:
    """One evaluation result at base grain."""
    planning_year: str
    aq_quarter: str
    class_type: str
    market_round: int
    K: int

    # Required
    sp: float
    binders: int
    precision: float
    vc: float
    recall: float
    nb_in: int
    nb_binders: int
    nb_sp: float

    # Optional
    d20_hit: int | None = None
    d20_total: int | None = None
    d50_hit: int | None = None
    d50_total: int | None = None
    label_coverage: int | None = None
    unlabeled: int | None = None  # benchmark only


@dataclass
class MetricsResult:
    """Full metrics.json content."""
    base_grain: str = field(default="planning_year/aq_quarter/class_type/market_round", init=False)
    cells: list[MetricCell] = field(default_factory=list)
    aggregates: dict | None = None  # optional summary with explicit rule
