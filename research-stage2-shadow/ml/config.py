"""ML pipeline configuration dataclasses.

ClassifierConfig  -- frozen; locked to stage-1 champion v0006.
RegressorConfig   -- mutable; agentic loop iterates on this.
PipelineConfig    -- composition of classifier + regressor + pipeline params.
GateConfig        -- quality gates loaded from JSON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Step-1 classifier features (13) ──────────────────────────────────────

_CLASSIFIER_FEATURES: list[str] = [
    "prob_exceed_110",
    "prob_exceed_105",
    "prob_exceed_100",
    "prob_exceed_95",
    "prob_exceed_90",
    "prob_below_100",
    "prob_below_95",
    "prob_below_90",
    "expected_overload",
    "density_skewness",
    "density_kurtosis",
    "hist_da",
    "hist_da_trend",
]

_CLASSIFIER_MONOTONE: list[int] = [
    1, 1, 1, 1, 1,   # prob_exceed_*  -> higher prob => more likely spike
    -1, -1, -1,       # prob_below_*   -> higher prob => less likely spike
    1,                 # expected_overload
    0, 0,             # density_skewness, density_kurtosis (unconstrained)
    1, 1,             # hist_da, hist_da_trend
]

# ── Additional step-2 regressor features (11) ────────────────────────────

_ADDITIONAL_FEATURES: list[str] = [
    "prob_exceed_85",
    "prob_exceed_80",
    "tail_concentration",
    "prob_band_95_100",
    "prob_band_100_105",
    "density_mean",
    "density_variance",
    "density_entropy",
    "recent_hist_da",
    "season_hist_da_1",
    "season_hist_da_2",
]

_ADDITIONAL_MONOTONE: list[int] = [
    1, 1,     # prob_exceed_85, prob_exceed_80
    1,        # tail_concentration
    0, 0,     # prob_band_95_100, prob_band_100_105 (unconstrained)
    0, 0, 0,  # density_mean, density_variance, density_entropy (unconstrained)
    1,        # recent_hist_da
    1, 1,     # season_hist_da_1, season_hist_da_2
]

_REGRESSOR_FEATURES: list[str] = _CLASSIFIER_FEATURES + _ADDITIONAL_FEATURES
_REGRESSOR_MONOTONE: list[int] = _CLASSIFIER_MONOTONE + _ADDITIONAL_MONOTONE


# ── ClassifierConfig (frozen — stage-1 champion v0006) ───────────────────

@dataclass(frozen=True)
class ClassifierConfig:
    """Frozen classifier configuration locked to stage-1 champion v0006."""

    features: tuple[str, ...] = tuple(_CLASSIFIER_FEATURES)
    monotone_constraints: tuple[int, ...] = tuple(_CLASSIFIER_MONOTONE)

    # XGBoost hyperparams
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    min_child_weight: int = 10

    # Decision threshold
    threshold_beta: float = 0.7

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "monotone_constraints": list(self.monotone_constraints),
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_child_weight": self.min_child_weight,
            "threshold_beta": self.threshold_beta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ClassifierConfig:
        return cls(
            features=tuple(d["features"]),
            monotone_constraints=tuple(d["monotone_constraints"]),
            n_estimators=d["n_estimators"],
            max_depth=d["max_depth"],
            learning_rate=d["learning_rate"],
            subsample=d["subsample"],
            colsample_bytree=d["colsample_bytree"],
            reg_alpha=d["reg_alpha"],
            reg_lambda=d["reg_lambda"],
            min_child_weight=d["min_child_weight"],
            threshold_beta=d["threshold_beta"],
        )


# ── RegressorConfig (mutable — agentic loop iterates) ────────────────────

@dataclass
class RegressorConfig:
    """Mutable regressor configuration for stage-2 agentic iteration."""

    features: list[str] = field(default_factory=lambda: list(_REGRESSOR_FEATURES))
    monotone_constraints: list[int] = field(
        default_factory=lambda: list(_REGRESSOR_MONOTONE)
    )

    # XGBoost hyperparams
    n_estimators: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    min_child_weight: int = 10

    # Pipeline mode
    unified_regressor: bool = False  # v0: gated mode
    value_weighted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "monotone_constraints": list(self.monotone_constraints),
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_child_weight": self.min_child_weight,
            "unified_regressor": self.unified_regressor,
            "value_weighted": self.value_weighted,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RegressorConfig:
        return cls(
            features=list(d["features"]),
            monotone_constraints=list(d["monotone_constraints"]),
            n_estimators=d["n_estimators"],
            max_depth=d["max_depth"],
            learning_rate=d["learning_rate"],
            subsample=d["subsample"],
            colsample_bytree=d["colsample_bytree"],
            reg_alpha=d["reg_alpha"],
            reg_lambda=d["reg_lambda"],
            min_child_weight=d["min_child_weight"],
            unified_regressor=d["unified_regressor"],
            value_weighted=d["value_weighted"],
        )


# ── PipelineConfig ────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Full pipeline configuration: classifier + regressor + pipeline params."""

    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    regressor: RegressorConfig = field(default_factory=RegressorConfig)
    train_months: int = 10
    val_months: int = 2
    ev_scoring: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "classifier": self.classifier.to_dict(),
            "regressor": self.regressor.to_dict(),
            "train_months": self.train_months,
            "val_months": self.val_months,
            "ev_scoring": self.ev_scoring,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineConfig:
        return cls(
            classifier=ClassifierConfig.from_dict(d["classifier"]),
            regressor=RegressorConfig.from_dict(d["regressor"]),
            train_months=d["train_months"],
            val_months=d["val_months"],
            ev_scoring=d["ev_scoring"],
        )


# ── GateConfig ────────────────────────────────────────────────────────────

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
