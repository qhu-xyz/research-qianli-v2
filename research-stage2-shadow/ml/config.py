"""ML pipeline configuration dataclasses.

ClassifierConfig  -- parameterizable; supports v0 (14-feat) and v1 (29-feat).
RegressorConfig   -- mutable; agentic loop iterates on this.
PipelineConfig    -- composition of classifier + regressor + pipeline params.
GateConfig        -- quality gates loaded from JSON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── v0 classifier features (14) — original pipeline set ──────────────────

_V0_CLF_FEATURES: list[str] = [
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
    "density_cv",
    "hist_da",
    "hist_da_trend",
]

_V0_CLF_MONOTONE: list[int] = [
    1, 1, 1, 1, 1,   # prob_exceed_*
    -1, -1, -1,       # prob_below_*
    1,                 # expected_overload
    0, 0, 0,          # density_skewness, density_kurtosis, density_cv
    1, 1,             # hist_da, hist_da_trend
]

# ── v1 classifier features (29) — from stage-1 v0011 ─────────────────────

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

# ── Regressor features: ALL available (34) ────────────────────────────────

_ALL_REGRESSOR_FEATURES: list[str] = _V1_CLF_FEATURES + [
    "prob_exceed_85",
    "prob_exceed_80",
    "recent_hist_da",
    "season_hist_da_1",
    "season_hist_da_2",
]

_ALL_REGRESSOR_MONOTONE: list[int] = _V1_CLF_MONOTONE + [
    1, 1,     # prob_exceed_85, prob_exceed_80
    1,        # recent_hist_da
    1, 1,     # season_hist_da_1, season_hist_da_2
]


# ── ClassifierConfig ──────────────────────────────────────────────────────

@dataclass
class ClassifierConfig:
    """Classifier configuration. Use preset() for v0/v1 feature sets."""

    features: list[str] = field(default_factory=lambda: list(_V0_CLF_FEATURES))
    monotone_constraints: list[int] = field(default_factory=lambda: list(_V0_CLF_MONOTONE))

    # XGBoost hyperparams (v0 defaults)
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

    @staticmethod
    def v0() -> ClassifierConfig:
        """14-feature classifier (original pipeline feature set)."""
        return ClassifierConfig(
            features=list(_V0_CLF_FEATURES),
            monotone_constraints=list(_V0_CLF_MONOTONE),
        )

    @staticmethod
    def v1() -> ClassifierConfig:
        """29-feature classifier (stage-1 v0011 feature set)."""
        return ClassifierConfig(
            features=list(_V1_CLF_FEATURES),
            monotone_constraints=list(_V1_CLF_MONOTONE),
            n_estimators=300,
            learning_rate=0.07,
            colsample_bytree=0.9,
        )

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
            threshold_beta=d.get("threshold_beta", 0.7),
        )


# ── RegressorConfig ───────────────────────────────────────────────────────

@dataclass
class RegressorConfig:
    """Regressor configuration. Uses ALL available features by default."""

    features: list[str] = field(default_factory=lambda: list(_ALL_REGRESSOR_FEATURES))
    monotone_constraints: list[int] = field(
        default_factory=lambda: list(_ALL_REGRESSOR_MONOTONE)
    )

    # XGBoost hyperparams
    n_estimators: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 5.0
    min_child_weight: int = 25

    # Pipeline mode
    unified_regressor: bool = False  # False = gated mode (binding-only training)
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
            unified_regressor=d.get("unified_regressor", False),
            value_weighted=d.get("value_weighted", False),
        )


# ── PipelineConfig ────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Full pipeline configuration: classifier + regressor + pipeline params."""

    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    regressor: RegressorConfig = field(default_factory=RegressorConfig)
    train_months: int = 6
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
