"""Gate comparison system for shadow price classification pipeline.

Loads all registered versions, checks each metric against gate floors,
produces a Markdown comparison table and JSON summary.

CLI: python ml/compare.py --batch-id X --iteration N --output path
"""

import argparse
import json
import resource
from datetime import datetime, timezone
from pathlib import Path


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_all_versions(registry_dir: str | Path) -> dict[str, dict]:
    """Read all registry/v*/metrics.json files.

    Returns {version_id: metrics_dict} sorted by version_id.
    """
    registry = Path(registry_dir)
    versions = {}
    for metrics_path in sorted(registry.glob("v*/metrics.json")):
        version_id = metrics_path.parent.name
        with open(metrics_path) as f:
            versions[version_id] = json.load(f)
    return versions


def check_gates(
    metrics: dict,
    gates: dict,
    champion_metrics: dict | None = None,
    noise_tolerance: float = 0.02,
) -> dict[str, dict]:
    """Check each gate metric against its floor.

    Parameters
    ----------
    metrics : dict
        Metrics for the version being evaluated.
    gates : dict
        Gate definitions from gates.json.
    champion_metrics : dict or None
        Metrics for the current champion (for regression checks).
    noise_tolerance : float
        Tolerance for regression vs champion.

    Returns
    -------
    results : dict
        {gate_name: {"value": float, "floor": float, "direction": str,
                      "passed": bool, "delta": float, "group": str}}
    """
    results = {}
    for gate_name, gate_def in gates.items():
        floor = gate_def.get("floor")
        direction = gate_def["direction"]
        group = gate_def.get("group", "A")
        value = metrics.get(gate_name)

        if floor is None or value is None:
            results[gate_name] = {
                "value": value,
                "floor": floor,
                "direction": direction,
                "passed": None,  # Cannot evaluate
                "delta": None,
                "group": group,
            }
            continue

        import math

        if math.isnan(value):
            results[gate_name] = {
                "value": value,
                "floor": floor,
                "direction": direction,
                "passed": False,
                "delta": None,
                "group": group,
            }
            continue

        if direction == "higher":
            passed = value >= floor
            delta = value - floor
        else:  # lower
            passed = value <= floor
            delta = floor - value  # positive delta means "better" (lower)

        # Check regression vs champion
        if champion_metrics is not None and gate_name in champion_metrics:
            champ_val = champion_metrics[gate_name]
            if champ_val is not None and not (isinstance(champ_val, float) and math.isnan(champ_val)):
                if direction == "higher":
                    regression = value < champ_val - noise_tolerance
                else:
                    regression = value > champ_val + noise_tolerance
                if regression:
                    passed = False

        results[gate_name] = {
            "value": round(value, 4) if isinstance(value, float) else value,
            "floor": floor,
            "direction": direction,
            "passed": passed,
            "delta": round(delta, 4) if delta is not None else None,
            "group": group,
        }
    return results


def evaluate_overall_pass(gate_results: dict[str, dict]) -> tuple[bool, bool]:
    """Evaluate overall pass from gate results, separated by group.

    Returns (group_a_passed, group_b_passed).
    Group A gates block promotion; Group B gates are informational.
    """
    group_a_passed = True
    group_b_passed = True
    for result in gate_results.values():
        passed = result.get("passed")
        group = result.get("group", "A")
        if passed is not True:  # None or False
            if group == "A":
                group_a_passed = False
            else:
                group_b_passed = False
    return group_a_passed, group_b_passed


def build_comparison_table(
    versions: dict[str, dict],
    gates: dict,
    champion_metrics: dict | None = None,
    noise_tolerance: float = 0.02,
) -> str:
    """Build a Markdown comparison table across all versions.

    Returns a Markdown string with one row per version,
    columns for each gate metric with pass/fail indicators.
    """
    gate_names = list(gates.keys())

    # Header row
    header = "| Version | " + " | ".join(gate_names) + " | Pass |"
    separator = "|---------|" + "|".join(["--------"] * len(gate_names)) + "|------|"

    rows = [header, separator]

    for version_id, metrics in sorted(versions.items()):
        gate_results = check_gates(metrics, gates, champion_metrics, noise_tolerance)

        cells = []
        group_a_passed = True
        group_b_passed = True
        for gate_name in gate_names:
            result = gate_results.get(gate_name, {})
            value = result.get("value")
            passed = result.get("passed")
            group = result.get("group", "A")

            if value is None:
                cells.append("--")
            elif isinstance(value, float) and (value != value):  # NaN check
                cells.append("NaN")
                if group == "A":
                    group_a_passed = False
                else:
                    group_b_passed = False
            else:
                mark = "P" if passed else "F"
                if passed is None:
                    mark = "?"
                    if group == "A":
                        group_a_passed = False
                    else:
                        group_b_passed = False
                elif not passed:
                    if group == "A":
                        group_a_passed = False
                    else:
                        group_b_passed = False
                cells.append(f"{value:.4f} {mark}")

        pass_str = "YES" if group_a_passed else "NO"
        group_b_str = "YES" if group_b_passed else "NO"
        row = f"| {version_id} | " + " | ".join(cells) + f" | {pass_str} (B:{group_b_str}) |"
        rows.append(row)

    return "\n".join(rows)


def run_comparison(
    batch_id: str,
    iteration: int,
    registry_dir: str = "registry",
    gates_path: str = "registry/gates.json",
    champion_path: str = "registry/champion.json",
    output_path: str | None = None,
) -> dict:
    """Run full comparison: load versions, check gates, build table, write outputs.

    Parameters
    ----------
    batch_id : str
        Current batch identifier.
    iteration : int
        Current iteration number.
    registry_dir : str
        Path to registry directory.
    gates_path : str
        Path to gates.json.
    champion_path : str
        Path to champion.json.
    output_path : str or None
        Path to write Markdown report. If None, prints to stdout.

    Returns
    -------
    comparison : dict
        Full comparison data including table, per-version gate results.
    """
    print(f"[compare] mem at start: {mem_mb():.0f} MB")

    # Load gates
    with open(gates_path) as f:
        gates_data = json.load(f)
    gates = gates_data["gates"]
    noise_tolerance = gates_data.get("noise_tolerance", 0.02)

    # Load champion
    champion_metrics = None
    with open(champion_path) as f:
        champion_data = json.load(f)
    champion_version = champion_data.get("version")
    if champion_version:
        champ_metrics_path = Path(registry_dir) / champion_version / "metrics.json"
        if champ_metrics_path.exists():
            with open(champ_metrics_path) as f:
                champion_metrics = json.load(f)

    # Load all versions
    versions = load_all_versions(registry_dir)

    # Build comparison table
    table = build_comparison_table(versions, gates, champion_metrics, noise_tolerance)

    # Per-version gate results
    per_version = {}
    per_version_pass = {}
    for version_id, metrics in versions.items():
        gate_results = check_gates(
            metrics, gates, champion_metrics, noise_tolerance
        )
        per_version[version_id] = gate_results
        ga, gb = evaluate_overall_pass(gate_results)
        per_version_pass[version_id] = {
            "group_a_passed": ga,
            "group_b_passed": gb,
            "overall_passed": ga,  # Only Group A blocks
        }

    comparison = {
        "batch_id": batch_id,
        "iteration": iteration,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "champion": champion_version,
        "noise_tolerance": noise_tolerance,
        "versions": per_version,
        "pass_summary": per_version_pass,
        "table": table,
    }

    # Write Markdown report
    if output_path:
        report = f"# Comparison Report\n\n"
        report += f"**Batch:** {batch_id}  **Iteration:** {iteration}\n"
        report += f"**Champion:** {champion_version or 'None'}\n"
        report += f"**Timestamp:** {comparison['timestamp']}\n\n"
        report += table + "\n"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"[compare] wrote report to {output_path}")

    # Write JSON to registry/comparisons/
    comparisons_dir = Path(registry_dir) / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    json_path = comparisons_dir / f"{batch_id}_iter{iteration}.json"
    with open(json_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"[compare] wrote JSON to {json_path}")

    print(f"[compare] mem at end: {mem_mb():.0f} MB")
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Compare all registered versions against gates")
    parser.add_argument("--batch-id", required=True, help="Batch identifier")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number")
    parser.add_argument("--output", default=None, help="Path for Markdown report")
    parser.add_argument("--registry-dir", default="registry", help="Registry directory")
    parser.add_argument("--gates-path", default="registry/gates.json", help="Gates JSON path")
    parser.add_argument("--champion-path", default="registry/champion.json", help="Champion JSON path")
    args = parser.parse_args()

    run_comparison(
        batch_id=args.batch_id,
        iteration=args.iteration,
        registry_dir=args.registry_dir,
        gates_path=args.gates_path,
        champion_path=args.champion_path,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
