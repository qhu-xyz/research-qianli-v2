#!/usr/bin/env python3
"""Bootstrap the model registry with the legacy baseline.

Migrates versions/legacy_baseline.json into the versioned directory structure
and promotes v000-legacy as the initial champion.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    PYTHONPATH=/home/xyz/workspace/research-qianli-v2/research-spice-shadow-price-pred-qianli/src:$PYTHONPATH \
        python /home/xyz/workspace/research-qianli-v2/research-spice-shadow-price-pred-qianli/scripts/bootstrap_registry.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_DIR = REPO_ROOT / "versions"
LEGACY_JSON = REGISTRY_DIR / "legacy_baseline.json"

sys.path.insert(0, str(REPO_ROOT / "src"))

from shadow_price_prediction.registry import ModelRegistry


def main():
    if not LEGACY_JSON.exists():
        print(f"ERROR: {LEGACY_JSON} not found. Run the legacy baseline first.")
        sys.exit(1)

    with open(LEGACY_JSON) as f:
        legacy = json.load(f)

    reg = ModelRegistry(REGISTRY_DIR)

    # Check if already bootstrapped
    if "v000-legacy-20260220" in reg.list_versions():
        print("v000-legacy-20260220 already exists. Skipping creation.")
        ver = reg.get_version("v000-legacy-20260220")
    else:
        # Create the legacy baseline version
        ver = reg.create_version(
            algo="legacy",
            description="Legacy unmodified pipeline baseline (defaults from commit b32bf6b)",
            source_commit="b32bf6b",
            model_id_override="v000-legacy-20260220",
        )

        # Record config — the legacy pipeline used all defaults, so we serialize
        # the default PredictionConfig but with the legacy differences noted
        legacy_config = {
            "description": "Legacy pipeline defaults (commit b32bf6b)",
            "features": {
                "step1_features": [
                    {"name": "prob_exceed_110", "monotonicity": 1},
                    {"name": "prob_exceed_105", "monotonicity": 1},
                    {"name": "prob_exceed_100", "monotonicity": 1},
                    {"name": "prob_exceed_95", "monotonicity": 1},
                    {"name": "prob_exceed_90", "monotonicity": 1},
                    {"name": "prob_below_95", "monotonicity": -1},
                    {"name": "prob_below_90", "monotonicity": -1},
                    {"name": "density_skewness", "monotonicity": 1},
                ],
                "step2_features": [
                    {"name": "prob_exceed_110", "monotonicity": 1},
                    {"name": "prob_exceed_105", "monotonicity": 1},
                    {"name": "prob_exceed_100", "monotonicity": 1},
                    {"name": "prob_exceed_95", "monotonicity": 1},
                    {"name": "prob_exceed_90", "monotonicity": 1},
                    {"name": "prob_below_95", "monotonicity": -1},
                    {"name": "prob_below_90", "monotonicity": -1},
                    {"name": "density_skewness", "monotonicity": 1},
                ],
                "note": "hist_da was commented out in legacy code",
            },
            "training": {
                "train_months": 12,
                "val_months": 0,
                "test_months": 0,
                "split": "12/0/0 (rolling 12-month window, no val/test split)",
                "min_samples_for_branch_model": 1,
                "min_binding_samples_for_regression": 1,
            },
            "threshold": {
                "threshold_beta": 0.5,
                "note": "F0.5 (precision-weighted), in-sample optimization",
            },
            "models": {
                "classifiers": ["XGBClassifier"],
                "regressors": ["XGBRegressor"],
                "note": "ElasticNet was commented out in legacy code",
            },
            "anomaly_detection": {
                "enabled": True,
                "k_multiplier": 3.0,
            },
        }
        ver.record_config(legacy_config)

        # Record features
        ver.record_features({
            "step1_features": legacy_config["features"]["step1_features"],
            "step2_features": legacy_config["features"]["step2_features"],
            "n_step1": 8,
            "n_step2": 8,
            "note": "8 density features only; hist_da and seasonal features not used",
        })

        print(f"Created version: {ver.model_id}")
        print(f"  Config checksum: {ver.config_checksum}")

    # Record metrics from legacy baseline
    ver.record_metrics(
        gate_values=legacy["gate_values"],
        benchmark_scope=legacy["benchmark_scope"],
        per_run_scores=legacy["per_run_scores"],
    )
    print(f"  Recorded metrics: {len(legacy['per_run_scores'])} runs")

    # Promote as initial champion (force=True since it won't pass gates)
    result = reg.promote(ver.model_id, force=True, reason="Initial baseline — bootstrapped")
    if result.success:
        print(f"\n  Promoted {ver.model_id} as champion")
        print(f"  Previous champion: {result.previous_champion or '(none)'}")
    else:
        print(f"\n  Promotion failed: {result.reason}")

    # Verify integrity
    if reg.verify_active():
        print("  Checksum verification: PASSED")
    else:
        print("  Checksum verification: FAILED")

    # Print gate check (informational — baseline won't pass absolute floors)
    gate_result = reg.check_gates(ver.model_id)
    print(f"\n{gate_result.summary_table()}")

    # Show registry state
    print(f"\nRegistry versions: {reg.list_versions()}")
    print(f"Active champion: {reg.active_version}")


if __name__ == "__main__":
    main()
