"""Configuration classes for shadow price classification pipeline (Stage 1 only)."""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FeatureConfig:
    """Feature column configuration for Stage 1 classifier.

    Each feature is a (name, monotone_constraint) tuple:
      1 = increasing, -1 = decreasing, 0 = no constraint
    """

    step1_features: list[tuple[str, int]] = field(
        default_factory=lambda: [
            # --- Density exceedance probabilities (core 5) ---
            ("prob_exceed_110", 1),
            ("prob_exceed_105", 1),
            ("prob_exceed_100", 1),
            ("prob_exceed_95", 1),
            ("prob_exceed_90", 1),
            # --- Density below-threshold probabilities ---
            ("prob_below_100", -1),
            ("prob_below_95", -1),
            ("prob_below_90", -1),
            # --- Severity signal ---
            ("expected_overload", 1),
            # --- Historical DA shadow price ---
            ("hist_da", 1),
            ("hist_da_trend", 1),
            # --- Interaction features (retained: top 2 of 3) ---
            ("hist_physical_interaction", 1),
            ("overload_exceedance_product", 1),
            # --- Shift factor features (network topology) ---
            ("sf_max_abs", 1),
            ("sf_mean_abs", 1),
            ("sf_std", 0),
            ("sf_nonzero_frac", 0),
            # --- Constraint metadata ---
            ("is_interface", 0),
            ("constraint_limit", 0),
        ]
    )

    @property
    def features(self) -> list[str]:
        """Return list of feature names."""
        return [f[0] for f in self.step1_features]

    def get_monotone_constraints_str(self) -> str:
        """Return monotone constraints as XGBoost-compatible string.

        Example: "(1,1,1,1,1,-1,-1,-1,1,0,0,0,1,1)"
        """
        values = ",".join(str(f[1]) for f in self.step1_features)
        return f"({values})"


@dataclass
class HyperparamConfig:
    """XGBoost hyperparameter configuration."""

    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    min_child_weight: int = 10
    random_state: int = 42

    def to_dict(self) -> dict:
        """Return hyperparameters as a dictionary."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_child_weight": self.min_child_weight,
            "random_state": self.random_state,
        }


@dataclass
class PipelineConfig:
    """Pipeline execution configuration."""

    auction_month: str | None = None
    class_type: str = "onpeak"
    period_type: str = "f0"
    version_id: str | None = None
    train_months: int = 14
    val_months: int = 2
    threshold_beta: float = 0.7
    threshold_scaling_factor: float = 1.0
    scale_pos_weight_auto: bool = True
    registry_dir: str = "registry"


class GateConfig:
    """Gate configuration loaded from registry/gates.json."""

    def __init__(self, gates_path: str | Path = "registry/gates.json"):
        self.gates_path = Path(gates_path)
        self._data: dict | None = None

    def load(self) -> dict:
        """Load gates from JSON file."""
        with open(self.gates_path) as f:
            self._data = json.load(f)
        return self._data

    @property
    def data(self) -> dict:
        """Return cached gate data, loading if needed."""
        if self._data is None:
            self.load()
        return self._data

    @property
    def gates(self) -> dict:
        """Return the gates sub-dictionary."""
        return self.data["gates"]

    @property
    def noise_tolerance(self) -> float:
        """Return the noise tolerance value."""
        return self.data["noise_tolerance"]

    def all_floors_populated(self) -> bool:
        """Check if all gate floors are populated (no None/null values)."""
        for gate_name, gate_def in self.gates.items():
            if gate_def.get("floor") is None:
                return False
        return True

    @property
    def tail_max_failures(self) -> int:
        """Max months allowed below tail_floor per gate."""
        return self.data.get("tail_max_failures", 1)

    @property
    def eval_months(self) -> dict:
        """Primary and stress evaluation months."""
        return self.data.get("eval_months", {"primary": [], "stress": []})

    @property
    def cascade_stages(self) -> list[dict]:
        """Cascade stage definitions."""
        return self.data.get("cascade_stages", [])

    def pending_v0_gates(self) -> list[str]:
        """Return list of gate names that are still pending v0 calibration."""
        return [
            name
            for name, gate_def in self.gates.items()
            if gate_def.get("pending_v0", False)
        ]
