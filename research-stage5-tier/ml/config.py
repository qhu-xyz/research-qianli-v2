"""LTR pipeline configuration.

LTRConfig      -- LightGBM lambdarank model config (default) or XGBoost rank:pairwise.
PipelineConfig -- composition of LTR config + pipeline params.
GateConfig     -- quality gates loaded from JSON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Leakage guard ──────────────────────────────────────────────────────────────
# da_rank_value is a historical 60-month lookback, legitimate as a feature.
# What IS leaky: rank, rank_ori, tier (derived formula outputs).
_LEAKY_FEATURES: set[str] = {
    # Composite outputs derived from formula (don't use as features)
    "rank",              # = dense_rank(rank_ori), output of formula
    "rank_ori",          # = 0.60*da_rank_value + 0.30*dmix + 0.10*dori
    "tier",              # derived from rank
    # Metadata / signed columns
    "shadow_sign",
    "shadow_price",      # = shadow_price_da * shadow_sign (signed version)
}


# ── V6.2B forecast-only features ──
# These are the legitimate (non-leaky) columns from V6.2B parquet.
# density_*_rank_value are within-month percentile ranks of flow forecasts.
# Lower density_*_rank_value = higher flow = more binding (inverted scale).
_V62B_FEATURES: list[str] = [
    "mean_branch_max",          # max branch loading in constraint path
    "ori_mean",                 # mean flow, baseline scenario
    "mix_mean",                 # mean flow, mixed scenario
    "density_mix_rank_value",   # percentile rank of mix_mean (inverted)
    "density_ori_rank_value",   # percentile rank of ori_mean (inverted)
]
_V62B_MONOTONE: list[int] = [1, 1, 1, -1, -1]
# mean_branch_max, ori_mean, mix_mean: higher flow = more binding = positive
# density_*_rank_value: lower value = more binding = negative monotone

# ── Spice6 density features (loaded from raw parquets) ──
_SPICE6_FEATURES: list[str] = [
    "prob_exceed_110",   # strongest binding signal
    "prob_exceed_100",   # flow at/above limit
    "prob_exceed_90",    # near-limit stress
    "prob_exceed_85",    # moderate stress
    "prob_exceed_80",    # lower stress
    "constraint_limit",  # MW limit
]
_SPICE6_MONOTONE: list[int] = [
    1,   # prob_exceed_110
    1,   # prob_exceed_100
    1,   # prob_exceed_90
    1,   # prob_exceed_85
    1,   # prob_exceed_80
    0,   # constraint_limit
]

# ── Group C: Historical DA signal (from V6.2B parquet) ──
_HIST_DA_FEATURES: list[str] = ["da_rank_value"]
_HIST_DA_MONOTONE: list[int] = [-1]  # lower rank_value = more binding

# ── Group D: ML predictions (from ml_pred/final_results.parquet) ──
_MLPRED_FEATURES: list[str] = [
    "predicted_shadow_price",
    "binding_probability",
    "binding_probability_scaled",
]
_MLPRED_MONOTONE: list[int] = [1, 1, 1]

# ── Composed feature sets ──
# v1: Groups A+B (11 features) — pure forecasts, no historical DA
FEATURES_V1: list[str] = _V62B_FEATURES + _SPICE6_FEATURES
MONOTONE_V1: list[int] = _V62B_MONOTONE + _SPICE6_MONOTONE

# v1b: Groups A+B+C (12 features) — add historical DA signal
FEATURES_V1B: list[str] = _V62B_FEATURES + _SPICE6_FEATURES + _HIST_DA_FEATURES
MONOTONE_V1B: list[int] = _V62B_MONOTONE + _SPICE6_MONOTONE + _HIST_DA_MONOTONE

# v3: Groups A+B+C+D (15 features) — add ML predictions
FEATURES_V3: list[str] = FEATURES_V1B + _MLPRED_FEATURES
MONOTONE_V3: list[int] = MONOTONE_V1B + _MLPRED_MONOTONE

# Default = v1 (Groups A+B)
_ALL_FEATURES: list[str] = FEATURES_V1
_ALL_MONOTONE: list[int] = MONOTONE_V1

# ── Eval months ──
# Screen: 4 months (fast hypothesis test, ~12s with LightGBM)
# Eval: 12 representative months (1/quarter, comprehensive validation)
# Full: 36 rolling months (only if needed for deep analysis)
# Strategy: screen on 4; if promising, eval on 12; else move on.
_SCREEN_EVAL_MONTHS: list[str] = [
    "2020-12", "2021-09", "2022-06", "2023-03",
]

_DEFAULT_EVAL_MONTHS: list[str] = [
    "2020-09", "2020-12", "2021-03", "2021-06",
    "2021-09", "2021-12", "2022-03", "2022-06",
    "2022-09", "2022-12", "2023-03", "2023-05",
]

_FULL_EVAL_MONTHS: list[str] = [
    f"{y:04d}-{m:02d}"
    for y in range(2020, 2024)
    for m in range(1, 13)
    if (y, m) >= (2020, 6) and (y, m) <= (2023, 5)
]

# ── Data paths ──
V62B_SIGNAL_BASE = "/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1"
SPICE6_DENSITY_BASE = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density"
SPICE6_MLPRED_BASE = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/ml_pred"
SPICE6_CI_BASE = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/constraint_info"
REALIZED_DA_CACHE = str(Path(__file__).resolve().parent.parent / "data" / "realized_da")


@dataclass
class LTRConfig:
    """Learning-to-rank configuration (LightGBM lambdarank or XGBoost rank:pairwise)."""

    features: list[str] = field(default_factory=lambda: list(_ALL_FEATURES))
    monotone_constraints: list[int] = field(default_factory=lambda: list(_ALL_MONOTONE))

    # Backend: "lightgbm" (22x faster) or "xgboost"
    backend: str = "lightgbm"

    # Label mode: "rank" (raw rank, ~600 levels — noisy for sparse labels)
    #             "tiered" (4-level: 0=non-binding, 1/2/3=binding tiers)
    label_mode: str = "tiered"

    # Shared hyperparams
    n_estimators: int = 100
    learning_rate: float = 0.05
    min_child_weight: int = 25

    # LightGBM-specific
    num_leaves: int = 31
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 1.0
    reg_lambda: float = 1.0

    # XGBoost-specific (used only when backend="xgboost")
    max_depth: int = 5
    early_stopping_rounds: int = 20

    def __post_init__(self) -> None:
        if len(self.monotone_constraints) != len(self.features):
            raise ValueError(
                "LTRConfig invalid: len(monotone_constraints) must match len(features) "
                f"({len(self.monotone_constraints)} != {len(self.features)})"
            )

        filtered_features: list[str] = []
        filtered_mono: list[int] = []
        removed: list[str] = []

        for feat, mono in zip(self.features, self.monotone_constraints):
            if feat in _LEAKY_FEATURES:
                removed.append(feat)
                continue
            filtered_features.append(feat)
            filtered_mono.append(mono)

        if removed:
            # Keep deterministic order for logs.
            removed_sorted = ", ".join(sorted(set(removed)))
            print(f"[config] WARNING: dropped leaky features: {removed_sorted}")
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
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "early_stopping_rounds": self.early_stopping_rounds,
            "label_mode": self.label_mode,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LTRConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""

    ltr: LTRConfig = field(default_factory=LTRConfig)
    train_months: int = 8
    val_months: int = 0

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
