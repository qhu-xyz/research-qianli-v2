"""Annual LTR pipeline configuration.

Mirrors stage4 monthly config but adapted for annual auction structure:
- V6.1 annual signal (year/aq partitions) instead of V6.2B monthly
- Expanding-window train (all prior years) instead of rolling 8-month
- Eval groups = (planning_year, aq_round) instead of month
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# -- Data paths --
V61_SIGNAL_BASE = "/opt/data/xyz-dataset/signal_data/miso/constraints/Signal.MISO.SPICE_ANNUAL_V6.1"
SPICE_DATA_BASE = "/opt/data/xyz-dataset/spice_data/miso"

# -- Leakage guard --
# These are output/target columns that MUST NOT be used as features.
# NOTE: da_rank_value and shadow_price_da are HISTORICAL (60-month lookback),
# NOT realized DA. They are legitimate features. See stage5-handoff.md.
_LEAKY_FEATURES: set[str] = {
    "rank", "rank_ori", "tier",          # derived output columns
    "shadow_sign", "shadow_price",       # target-adjacent
    "density_mix_rank",                  # integer duplicate of density_mix_rank_value
    "mean_branch_max_fillna",            # redundant with mean_branch_max
}

# -- Feature Set A: V6.1 base (6 features) --
_V61_FEATURES: list[str] = [
    "shadow_price_da",          # historical DA shadow price (NOT realized)
    "mean_branch_max",          # max branch loading forecast
    "ori_mean",                 # mean flow, baseline scenario
    "mix_mean",                 # mean flow, mixed scenario
    "density_mix_rank_value",   # percentile rank of mix flow (lower = more binding)
    "density_ori_rank_value",   # percentile rank of ori flow (lower = more binding)
]
_V61_MONOTONE: list[int] = [1, 1, 1, 1, -1, -1]

# -- Feature Set B: V6.1 + spice6 density (11 features) --
_SPICE6_FEATURES: list[str] = [
    "prob_exceed_110",
    "prob_exceed_100",
    "prob_exceed_90",
    "prob_exceed_85",
    "prob_exceed_80",
]
_SPICE6_MONOTONE: list[int] = [1, 1, 1, 1, 1]

# -- Feature Set C: Full (13 features) --
_STRUCTURAL_FEATURES: list[str] = [
    "constraint_limit",
    "rate_a",
]
_STRUCTURAL_MONOTONE: list[int] = [0, 0]

# -- Composite feature lists --
SET_A_FEATURES = list(_V61_FEATURES)
SET_A_MONOTONE = list(_V61_MONOTONE)

SET_B_FEATURES = _V61_FEATURES + _SPICE6_FEATURES
SET_B_MONOTONE = _V61_MONOTONE + _SPICE6_MONOTONE

SET_C_FEATURES = _V61_FEATURES + _SPICE6_FEATURES + _STRUCTURAL_FEATURES
SET_C_MONOTONE = _V61_MONOTONE + _SPICE6_MONOTONE + _STRUCTURAL_MONOTONE

# -- Eval groups --
# Each group = (planning_year, aq_round) as "YYYY-06/aqN"
PLANNING_YEARS = [
    "2019-06", "2020-06", "2021-06", "2022-06",
    "2023-06", "2024-06", "2025-06",
]
AQ_ROUNDS = ["aq1", "aq2", "aq3", "aq4"]

# Quarter -> market months mapping
AQ_MARKET_MONTHS: dict[str, list[int]] = {
    "aq1": [6, 7, 8],    # Jun-Aug
    "aq2": [9, 10, 11],   # Sep-Nov
    "aq3": [12, 1, 2],    # Dec-Feb (crosses year boundary)
    "aq4": [3, 4, 5],     # Mar-May
}

# Eval splits (expanding window)
EVAL_SPLITS: dict[str, dict] = {
    "split1": {"train_years": ["2019-06", "2020-06", "2021-06"], "eval_year": "2022-06"},
    "split2": {"train_years": ["2019-06", "2020-06", "2021-06", "2022-06"], "eval_year": "2023-06"},
    "split3": {"train_years": ["2019-06", "2020-06", "2021-06", "2022-06", "2023-06"], "eval_year": "2024-06"},
}
# 2025-06 held out for final validation

SCREEN_EVAL_GROUPS: list[str] = [
    "2022-06/aq1", "2023-06/aq2", "2024-06/aq3", "2024-06/aq4",
]

DEFAULT_EVAL_GROUPS: list[str] = [
    f"{year}/{aq}"
    for year in ["2022-06", "2023-06", "2024-06"]
    for aq in AQ_ROUNDS
]

HOLDOUT_EVAL_GROUPS: list[str] = [
    f"2025-06/{aq}" for aq in AQ_ROUNDS
]


def get_market_months(planning_year: str, aq_round: str) -> list[str]:
    """Return YYYY-MM strings for the 3 market months in a quarter.

    Example: get_market_months("2022-06", "aq1") -> ["2022-06", "2022-07", "2022-08"]
    """
    base_year = int(planning_year.split("-")[0])
    months = AQ_MARKET_MONTHS[aq_round]
    result = []
    for m in months:
        if aq_round == "aq3" and m in (1, 2):
            year = base_year + 1
        elif aq_round == "aq4":
            year = base_year + 1
        else:
            year = base_year
        result.append(f"{year:04d}-{m:02d}")
    return result


@dataclass
class LTRConfig:
    """Learning-to-rank configuration."""
    features: list[str] = field(default_factory=lambda: list(SET_B_FEATURES))
    monotone_constraints: list[int] = field(default_factory=lambda: list(SET_B_MONOTONE))
    backend: str = "lightgbm"
    n_estimators: int = 100
    learning_rate: float = 0.05
    min_child_weight: int = 25
    num_leaves: int = 31
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 1.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 20

    def __post_init__(self) -> None:
        if len(self.monotone_constraints) != len(self.features):
            raise ValueError(
                f"len(monotone) != len(features): "
                f"{len(self.monotone_constraints)} != {len(self.features)}"
            )
        filtered_features = []
        filtered_mono = []
        removed = []
        for feat, mono in zip(self.features, self.monotone_constraints):
            if feat in _LEAKY_FEATURES:
                removed.append(feat)
                continue
            filtered_features.append(feat)
            filtered_mono.append(mono)
        if removed:
            print(f"[config] WARNING: dropped leaky features: {sorted(set(removed))}")
            self.features = filtered_features
            self.monotone_constraints = filtered_mono

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "monotone_constraints": list(self.monotone_constraints),
            "backend": self.backend,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "num_leaves": self.num_leaves,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LTRConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    ltr: LTRConfig = field(default_factory=LTRConfig)

    def to_dict(self) -> dict[str, Any]:
        return {"ltr": self.ltr.to_dict()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineConfig:
        return cls(ltr=LTRConfig.from_dict(d["ltr"]))


@dataclass
class GateConfig:
    """Quality gates loaded from JSON."""
    gates: dict[str, Any] = field(default_factory=dict)
    noise_tolerance: float = 0.0
    tail_max_failures: int = 0

    @classmethod
    def from_json(cls, path: str | Path) -> GateConfig:
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text())
        return cls(
            gates=data.get("gates", {}),
            noise_tolerance=data.get("noise_tolerance", 0.0),
            tail_max_failures=data.get("tail_max_failures", 0),
        )
