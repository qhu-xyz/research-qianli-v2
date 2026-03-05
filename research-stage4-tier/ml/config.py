"""LTR pipeline configuration.

LTRConfig      -- XGBoost rank:pairwise model config.
PipelineConfig -- composition of LTR config + pipeline params.
GateConfig     -- quality gates loaded from JSON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── V6.2B columns available as features ──
_V62B_FEATURES: list[str] = [
    "mean_branch_max", "ori_mean", "mix_mean",
    "density_mix_rank_value", "density_ori_rank_value", "da_rank_value",
]

# ── Stage 3 features (34 proven) ──
_STAGE3_FEATURES: list[str] = [
    "prob_exceed_110", "prob_exceed_105", "prob_exceed_100",
    "prob_exceed_95", "prob_exceed_90",
    "prob_below_100", "prob_below_95", "prob_below_90",
    "expected_overload",
    "hist_da", "hist_da_trend",
    "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac",
    "is_interface", "constraint_limit",
    "density_mean", "density_variance", "density_entropy",
    "tail_concentration",
    "prob_band_95_100", "prob_band_100_105",
    "hist_da_max_season",
    "prob_exceed_85", "prob_exceed_80",
    "recent_hist_da",
    "season_hist_da_1", "season_hist_da_2",
    "density_skewness", "density_kurtosis", "density_cv",
    "season_hist_da_3",
    "prob_below_85",
]

_STAGE3_MONOTONE: list[int] = [
    1, 1, 1, 1, 1,    # prob_exceed_110..90
    -1, -1, -1,        # prob_below_100..90
    1,                  # expected_overload
    1, 1,              # hist_da, hist_da_trend
    1, 1, 0, 0,        # sf_*
    0, 0,              # is_interface, constraint_limit
    0, 0, 0,           # density_mean, variance, entropy
    1,                  # tail_concentration
    0, 0,              # prob_band_*
    1,                  # hist_da_max_season
    1, 1,              # prob_exceed_85, 80
    1,                  # recent_hist_da
    1, 1,              # season_hist_da_1, 2
    0, 0, 0,           # density_skewness, kurtosis, cv
    1,                  # season_hist_da_3
    -1,                # prob_below_85
]

_ALL_FEATURES: list[str] = _STAGE3_FEATURES + _V62B_FEATURES
_ALL_MONOTONE: list[int] = _STAGE3_MONOTONE + [0] * len(_V62B_FEATURES)

# ── Eval months: 12 months spread across 2020-06 to 2023-05 ──
_DEFAULT_EVAL_MONTHS: list[str] = [
    "2020-09", "2020-12", "2021-03", "2021-06",
    "2021-09", "2021-12", "2022-03", "2022-06",
    "2022-09", "2022-12", "2023-03", "2023-05",
]

# ── Data paths ──
V62B_SIGNAL_BASE = "/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1"
SPICE6_DENSITY_BASE = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density"
SPICE6_SF_BASE = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/sf"
SPICE6_CI_BASE = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/constraint_info"


@dataclass
class LTRConfig:
    """XGBoost learning-to-rank configuration."""

    features: list[str] = field(default_factory=lambda: list(_ALL_FEATURES))
    monotone_constraints: list[int] = field(default_factory=lambda: list(_ALL_MONOTONE))

    # XGBoost hyperparams
    objective: str = "rank:pairwise"
    n_estimators: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 1.0
    reg_lambda: float = 1.0
    min_child_weight: int = 25
    early_stopping_rounds: int = 50

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "monotone_constraints": list(self.monotone_constraints),
            "objective": self.objective,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_child_weight": self.min_child_weight,
            "early_stopping_rounds": self.early_stopping_rounds,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LTRConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""

    ltr: LTRConfig = field(default_factory=LTRConfig)
    train_months: int = 6
    val_months: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "ltr": self.ltr.to_dict(),
            "train_months": self.train_months,
            "val_months": self.val_months,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineConfig:
        return cls(
            ltr=LTRConfig.from_dict(d["ltr"]),
            train_months=d["train_months"],
            val_months=d["val_months"],
        )


@dataclass
class GateConfig:
    """Quality gates loaded from JSON."""

    gates: dict[str, Any] = field(default_factory=dict)
    eval_months: dict[str, Any] = field(default_factory=dict)
    noise_tolerance: float = 0.0
    tail_max_failures: int = 0

    @classmethod
    def from_json(cls, path: str | Path) -> GateConfig:
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
            )
        except (json.JSONDecodeError, KeyError):
            return cls()
