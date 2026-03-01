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
        v0_data = json.load(f)

    # Load gates
    with open(gates_path) as f:
        gates_data = json.load(f)

    # Detect schema: v2 has "aggregate", v1 has flat metrics
    is_v2 = "aggregate" in v0_data
    if is_v2:
        v0_mean = v0_data["aggregate"]["mean"]
        v0_min = v0_data["aggregate"].get("min", {})
        v0_max = v0_data["aggregate"].get("max", {})
    else:
        v0_mean = v0_data  # flat metrics ARE the mean (single month)
        v0_min = v0_data
        v0_max = v0_data

    modified = False
    for gate_name, gate_def in gates_data["gates"].items():
        if not gate_def.get("pending_v0", False):
            continue

        mean_val = v0_mean.get(gate_name)
        if mean_val is None:
            print(f"[populate_v0] WARNING: {gate_name} not in v0 metrics, skipping")
            continue

        v0_offset = gate_def.get("v0_offset", 0.0)
        v0_tail_offset = gate_def.get("v0_tail_offset", v0_offset)
        direction = gate_def["direction"]

        if direction == "higher":
            gate_def["floor"] = round(mean_val - v0_offset, 6)
            extreme = v0_min.get(gate_name, mean_val)
            gate_def["tail_floor"] = round(extreme - v0_tail_offset, 6)
        else:
            gate_def["floor"] = round(mean_val + v0_offset, 6)
            extreme = v0_max.get(gate_name, mean_val)
            gate_def["tail_floor"] = round(extreme + v0_tail_offset, 6)

        gate_def["pending_v0"] = False
        modified = True
        print(f"[populate_v0] {gate_name}: floor={gate_def['floor']}, "
              f"tail_floor={gate_def['tail_floor']} "
              f"(mean={mean_val}, extreme={extreme}, direction={direction})")

    if modified:
        with open(gates_path, "w") as f:
            json.dump(gates_data, f, indent=2)
        print(f"[populate_v0] Updated {gates_path}")
    else:
        print("[populate_v0] No pending gates to populate")

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
