# ml/config.py
"""PJM LTR pipeline configuration.

Adapted from research-stage5-tier/ml/config.py for PJM market structure.
Key PJM differences:
  - 3 class types: onpeak, dailyoffpeak, wkndonpeak
  - f0 through f11 period types (vs MISO f0-f3)
  - PJM-specific data paths
  - Branch-level DA join (not constraint_id)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── PJM class types ──
PJM_CLASS_TYPES = ["onpeak", "dailyoffpeak", "wkndonpeak"]

# ── Leakage guard ──
_LEAKY_FEATURES: set[str] = {
    "rank", "rank_ori", "tier",
    "shadow_sign", "shadow_price",
}

# ── V10E features (9, same as MISO champion) ──
# ori_mean = density score (prob of exceeding 110% line rating), already in V6.2B.
# Replaces the old prob_exceed_110 which was redundantly loaded from spice6 density.
V10E_FEATURES = [
    "binding_freq_1", "binding_freq_3", "binding_freq_6", "binding_freq_12",
    "binding_freq_15", "v7_formula_score", "ori_mean", "constraint_limit",
    "da_rank_value",
]
V10E_MONOTONE = [1, 1, 1, 1, 1, -1, 1, 0, -1]

# ── V3 features (14, enriched with value-predictive signals) ──
# ori_mean replaces prob_exceed_110 (identical, already in V6.2B).
# prob_exceed_100 comes from spice6 ml_pred (NOT density — density only has one score).
V3_FEATURES = [
    "binding_freq_1", "binding_freq_3", "binding_freq_6", "binding_freq_12",
    "binding_freq_15", "v7_formula_score", "da_rank_value",
    "shadow_price_da", "binding_probability", "predicted_shadow_price",
    "ori_mean", "prob_exceed_100", "constraint_limit", "hist_da",
]
V3_MONOTONE = [1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 0, 1]

# ── Eval months (PJM-specific, based on V6.2B availability from 2017-06) ──
# Dev: 36 months (2020-06 to 2023-05), same range as MISO
_FULL_EVAL_MONTHS: list[str] = [
    f"{y:04d}-{m:02d}"
    for y in range(2020, 2024)
    for m in range(1, 13)
    if (y, m) >= (2020, 6) and (y, m) <= (2023, 5)
]

# Screen: 4 fast months
_SCREEN_EVAL_MONTHS: list[str] = [
    "2020-12", "2021-09", "2022-06", "2023-03",
]

# Default: 12 representative months
_DEFAULT_EVAL_MONTHS: list[str] = [
    "2020-09", "2020-12", "2021-03", "2021-06",
    "2021-09", "2021-12", "2022-03", "2022-06",
    "2022-09", "2022-12", "2023-03", "2023-05",
]

# Holdout: 2024-2025
HOLDOUT_MONTHS: list[str] = [
    f"{y:04d}-{m:02d}" for y in (2024, 2025) for m in range(1, 13)
]

# ── Data paths ──
V62B_SIGNAL_BASE = "/opt/data/xyz-dataset/signal_data/pjm/constraints/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1"
SPICE6_DENSITY_BASE = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/density"
SPICE6_MLPRED_BASE = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/ml_pred"
SPICE6_CI_BASE = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/constraint_info"
REALIZED_DA_CACHE = str(Path(__file__).resolve().parent.parent / "data" / "realized_da")

# ── PJM Auction Schedule ──
# Determined from actual V6.2B data on disk. PJM has f0-f11.
# June exposes all 12 monthly periods; schedule shrinks as planning year progresses.
PJM_AUCTION_SCHEDULE: dict[int, list[str]] = {
    1: ["f0", "f1", "f2", "f3", "f4"],
    2: ["f0", "f1", "f2", "f3"],
    3: ["f0", "f1", "f2"],
    4: ["f0", "f1"],
    5: ["f0"],
    6: ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"],
    7: ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"],
    8: ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"],
    9: ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"],
    10: ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"],
    11: ["f0", "f1", "f2", "f3", "f4", "f5", "f6"],
    12: ["f0", "f1", "f2", "f3", "f4", "f5"],
}


def period_offset(period_type: str) -> int:
    """f0→0, f1→1, ..., f11→11."""
    if not period_type.startswith("f"):
        raise ValueError(f"period_offset only supports fN types, got '{period_type}'")
    return int(period_type[1:])


def delivery_month(auction_month: str, period_type: str) -> str:
    """Compute delivery month = auction_month + period_offset."""
    import pandas as pd
    offset = period_offset(period_type)
    if offset == 0:
        return auction_month
    dt = pd.Timestamp(auction_month) + pd.DateOffset(months=offset)
    return dt.strftime("%Y-%m")


def has_period_type(auction_month: str, period_type: str) -> bool:
    """Check if period type exists for a given auction month."""
    import pandas as pd
    month_num = pd.Timestamp(auction_month).month
    return period_type in PJM_AUCTION_SCHEDULE.get(month_num, ["f0"])


def collect_usable_months(
    target_auction_month: str,
    period_type: str,
    n_months: int = 8,
    min_months: int = 6,
    max_lookback: int = 24,
) -> list[str]:
    """Walk backward from target, collect n_months usable training auction months.

    A month is "usable" if:
      1. The period_type exists for that month (per PJM auction schedule)
      2. delivery_month(month, period_type) <= last_full_known

    last_full_known = target_auction_month - 2 (last complete realized DA at decision time).
    Returns most-recent-first order. Caller should reverse for chronological.
    """
    import pandas as pd

    target_ts = pd.Timestamp(target_auction_month)
    last_full_known = (target_ts - pd.DateOffset(months=2)).strftime("%Y-%m")

    usable = []
    for i in range(1, max_lookback + 1):
        candidate = (target_ts - pd.DateOffset(months=i)).strftime("%Y-%m")
        if not has_period_type(candidate, period_type):
            continue
        dm = delivery_month(candidate, period_type)
        if dm > last_full_known:
            continue
        usable.append(candidate)
        if len(usable) >= n_months:
            break

    if len(usable) < min_months:
        return []
    return usable


@dataclass
class LTRConfig:
    """Learning-to-rank configuration."""

    features: list[str] = field(default_factory=lambda: list(V10E_FEATURES))
    monotone_constraints: list[int] = field(default_factory=lambda: list(V10E_MONOTONE))
    backend: str = "lightgbm"
    label_mode: str = "tiered"
    n_estimators: int = 100
    learning_rate: float = 0.05
    min_child_weight: int = 25
    num_leaves: int = 31
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 1.0
    reg_lambda: float = 1.0
    max_depth: int = 5
    early_stopping_rounds: int = 20

    def __post_init__(self) -> None:
        if len(self.monotone_constraints) != len(self.features):
            raise ValueError(
                f"len(monotone_constraints) must match len(features) "
                f"({len(self.monotone_constraints)} != {len(self.features)})"
            )
        filtered_features: list[str] = []
        filtered_mono: list[int] = []
        for feat, mono in zip(self.features, self.monotone_constraints):
            if feat in _LEAKY_FEATURES:
                print(f"[config] WARNING: dropped leaky feature: {feat}")
                continue
            filtered_features.append(feat)
            filtered_mono.append(mono)
        self.features = filtered_features
        self.monotone_constraints = filtered_mono

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

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
        return {"ltr": self.ltr.to_dict(), "train_months": self.train_months, "val_months": self.val_months}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineConfig:
        return cls(ltr=LTRConfig.from_dict(d["ltr"]), train_months=d["train_months"], val_months=d["val_months"])


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
