"""ML pipeline configuration dataclasses for tier classification.

TierConfig     -- single model config; agentic loop iterates on this.
PipelineConfig -- composition of tier config + pipeline params.
GateConfig     -- quality gates loaded from JSON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Feature sets (inherited from stage-2 regressor) ─────────────────────────

_DEAD_FEATURES: set[str] = {
    "hist_physical_interaction",
    "overload_exceedance_product",
    "band_severity",
    "sf_exceed_interaction",
    "hist_seasonal_band",
}

_V1_CLF_FEATURES: list[str] = [
    "prob_exceed_110",
    "prob_exceed_105",
    "prob_exceed_100",
    "prob_exceed_95",
    "prob_exceed_90",
    "prob_below_100",
    "prob_below_95",
    "prob_below_90",
    "expected_overload",
    "hist_da",
    "hist_da_trend",
    "hist_physical_interaction",
    "overload_exceedance_product",
    "sf_max_abs",
    "sf_mean_abs",
    "sf_std",
    "sf_nonzero_frac",
    "is_interface",
    "constraint_limit",
    "density_mean",
    "density_variance",
    "density_entropy",
    "tail_concentration",
    "prob_band_95_100",
    "prob_band_100_105",
    "hist_da_max_season",
    "band_severity",
    "sf_exceed_interaction",
    "hist_seasonal_band",
]

_V1_CLF_MONOTONE: list[int] = [
    1, 1, 1, 1, 1,   # prob_exceed_*
    -1, -1, -1,       # prob_below_*
    1,                 # expected_overload
    1, 1,             # hist_da, hist_da_trend
    0, 0,             # hist_physical_interaction, overload_exceedance_product
    1, 1, 0, 0,       # sf_max_abs, sf_mean_abs, sf_std, sf_nonzero_frac
    0, 0,             # is_interface, constraint_limit
    0, 0, 0,          # density_mean, density_variance, density_entropy
    1,                 # tail_concentration
    0, 0,             # prob_band_95_100, prob_band_100_105
    1,                 # hist_da_max_season
    0, 0, 0,          # band_severity, sf_exceed_interaction, hist_seasonal_band
]

# Filter dead features
_V1_CLF_FOR_TIER: list[str] = [
    f for f in _V1_CLF_FEATURES if f not in _DEAD_FEATURES
]
_V1_CLF_MONO_FOR_TIER: list[int] = [
    m for f, m in zip(_V1_CLF_FEATURES, _V1_CLF_MONOTONE)
    if f not in _DEAD_FEATURES
]

# All 34 candidate features (same as stage-2 regressor)
_ALL_TIER_FEATURES: list[str] = _V1_CLF_FOR_TIER + [
    "prob_exceed_85",
    "prob_exceed_80",
    "recent_hist_da",
    "season_hist_da_1",
    "season_hist_da_2",
    "density_skewness",
    "density_kurtosis",
    "density_cv",
    "season_hist_da_3",
    "prob_below_85",
]

_ALL_TIER_MONOTONE: list[int] = _V1_CLF_MONO_FOR_TIER + [
    1, 1,     # prob_exceed_85, prob_exceed_80
    1,        # recent_hist_da
    1, 1,     # season_hist_da_1, season_hist_da_2
    0, 0, 0,  # density_skewness, density_kurtosis, density_cv
    1,        # season_hist_da_3
    -1,       # prob_below_85
]

# Default tier bins: match existing SPICE system
_DEFAULT_BINS: list[float] = [float("-inf"), 0, 100, 1000, 3000, float("inf")]
_DEFAULT_MIDPOINTS: list[float] = [4000, 2000, 550, 50, 0]  # tier 0,1,2,3,4
_DEFAULT_CLASS_WEIGHTS: dict[int, float] = {0: 10, 1: 5, 2: 2, 3: 1, 4: 0.5}


# ── TierConfig ───────────────────────────────────────────────────────────────

@dataclass
class TierConfig:
    """Tier classification configuration. Single multi-class XGBoost model."""

    features: list[str] = field(default_factory=lambda: list(_ALL_TIER_FEATURES))
    monotone_constraints: list[int] = field(
        default_factory=lambda: list(_ALL_TIER_MONOTONE)
    )

    # Tier definitions
    bins: list[float] = field(default_factory=lambda: list(_DEFAULT_BINS))
    tier_midpoints: list[float] = field(default_factory=lambda: list(_DEFAULT_MIDPOINTS))
    num_class: int = 5
    class_weights: dict[int, float] = field(
        default_factory=lambda: dict(_DEFAULT_CLASS_WEIGHTS)
    )

    # XGBoost hyperparams
    n_estimators: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 1.0
    reg_lambda: float = 1.0
    min_child_weight: int = 25

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "monotone_constraints": list(self.monotone_constraints),
            "bins": list(self.bins),
            "tier_midpoints": list(self.tier_midpoints),
            "num_class": self.num_class,
            "class_weights": {str(k): v for k, v in self.class_weights.items()},
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_child_weight": self.min_child_weight,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TierConfig:
        cw = d.get("class_weights", _DEFAULT_CLASS_WEIGHTS)
        if isinstance(cw, dict):
            cw = {int(k): float(v) for k, v in cw.items()}
        return cls(
            features=list(d["features"]),
            monotone_constraints=list(d["monotone_constraints"]),
            bins=list(d.get("bins", _DEFAULT_BINS)),
            tier_midpoints=list(d.get("tier_midpoints", _DEFAULT_MIDPOINTS)),
            num_class=d.get("num_class", 5),
            class_weights=cw,
            n_estimators=d["n_estimators"],
            max_depth=d["max_depth"],
            learning_rate=d["learning_rate"],
            subsample=d["subsample"],
            colsample_bytree=d["colsample_bytree"],
            reg_alpha=d["reg_alpha"],
            reg_lambda=d["reg_lambda"],
            min_child_weight=d["min_child_weight"],
        )


# ── PipelineConfig ───────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Full pipeline configuration: tier model + pipeline params."""

    tier: TierConfig = field(default_factory=TierConfig)
    train_months: int = 6
    val_months: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.to_dict(),
            "train_months": self.train_months,
            "val_months": self.val_months,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineConfig:
        return cls(
            tier=TierConfig.from_dict(d["tier"]),
            train_months=d["train_months"],
            val_months=d["val_months"],
        )


# ── GateConfig ───────────────────────────────────────────────────────────────

@dataclass
class GateConfig:
    """Quality gates loaded from a JSON file."""

    gates: dict[str, Any] = field(default_factory=dict)
    eval_months: dict[str, Any] = field(default_factory=dict)
    noise_tolerance: float = 0.0
    tail_max_failures: int = 0
    cascade_stages: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str | Path) -> GateConfig:
        """Load gate config from JSON. Returns empty defaults if file missing."""
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text())
            return cls(
                gates=data.get("gates", {}),
                eval_months=data.get("eval_months", {}),
                noise_tolerance=data.get("noise_tolerance", 0.0),
                tail_max_failures=data.get("tail_max_failures", 0),
                cascade_stages=data.get("cascade_stages", []),
            )
        except (json.JSONDecodeError, KeyError):
            return cls()
