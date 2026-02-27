"""Populate v0-relative gate floors from baseline metrics.

For each gate with pending_v0=true:
  - higher direction: floor = v0_metric - v0_offset
  - lower direction (BRIER): floor = v0_metric + v0_offset

Sets pending_v0 to false after populating. Idempotent (no-op if no pending entries).

Usage: python ml/populate_v0_gates.py [--registry-dir registry] [--gates-path registry/gates.json]
"""

import argparse
import json
from pathlib import Path


def populate_v0_gates(
    registry_dir: str = "registry",
    gates_path: str = "registry/gates.json",
) -> dict:
    """Populate v0-relative gate floors.

    Parameters
    ----------
    registry_dir : str
        Path to registry directory (must contain v0/metrics.json).
    gates_path : str
        Path to gates.json file to update.

    Returns
    -------
    gates_data : dict
        Updated gates data.
    """
    registry_dir = Path(registry_dir)
    gates_path = Path(gates_path)

    # Load v0 metrics
    v0_metrics_path = registry_dir / "v0" / "metrics.json"
    if not v0_metrics_path.exists():
        raise FileNotFoundError(
            f"v0 metrics not found at {v0_metrics_path}. "
            "Run the pipeline with --version-id v0 first."
        )
    with open(v0_metrics_path) as f:
        v0_metrics = json.load(f)

    # Load gates
    with open(gates_path) as f:
        gates_data = json.load(f)

    modified = False
    for gate_name, gate_def in gates_data["gates"].items():
        if not gate_def.get("pending_v0", False):
            continue

        v0_value = v0_metrics.get(gate_name)
        if v0_value is None:
            print(f"[populate_v0] WARNING: {gate_name} not found in v0 metrics, skipping")
            continue

        v0_offset = gate_def.get("v0_offset", 0.0)
        direction = gate_def["direction"]

        if direction == "higher":
            # Higher is better: floor = v0 - offset (allow some regression from v0)
            floor = v0_value - v0_offset
        else:
            # Lower is better (e.g., BRIER): floor = v0 + offset (allow some increase from v0)
            floor = v0_value + v0_offset

        gate_def["floor"] = round(floor, 6)
        gate_def["pending_v0"] = False
        modified = True
        print(f"[populate_v0] {gate_name}: floor = {floor:.6f} "
              f"(v0={v0_value}, offset={v0_offset}, direction={direction})")

    if modified:
        with open(gates_path, "w") as f:
            json.dump(gates_data, f, indent=2)
        print(f"[populate_v0] Updated {gates_path}")
    else:
        print(f"[populate_v0] No pending gates to populate")

    return gates_data


def main():
    parser = argparse.ArgumentParser(
        description="Populate v0-relative gate floors from baseline metrics"
    )
    parser.add_argument("--registry-dir", default="registry", help="Registry directory")
    parser.add_argument("--gates-path", default="registry/gates.json", help="Gates JSON path")
    args = parser.parse_args()

    populate_v0_gates(
        registry_dir=args.registry_dir,
        gates_path=args.gates_path,
    )


if __name__ == "__main__":
    main()
