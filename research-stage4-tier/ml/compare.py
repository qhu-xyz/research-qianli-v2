"""Gate comparison system for LTR ranking pipeline.

Loads all registered versions, checks each metric against gate floors,
produces a Markdown comparison table and JSON summary.

LTR gate metrics:
  Group A (blocking): VC@20, VC@100, Recall@20, Recall@100, NDCG
  Group B (monitor):  VC@10, VC@25, VC@50, VC@200, Recall@10, Recall@50,
                      Spearman, Tier0-AP, Tier01-AP

All metrics are higher-is-better.

CLI: python ml/compare.py --batch-id X --iteration N --output path
"""

import argparse
import json
import math
import resource
from datetime import datetime, timezone
from pathlib import Path


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_all_versions(registry_dir: str | Path) -> dict[str, dict]:
    """Read all registry/v*/metrics.json files."""
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
    """Check each gate metric against its floor."""
    results = {}
    for gate_name, gate_def in gates.items():
        floor = gate_def.get("floor")
        direction = gate_def["direction"]
        group = gate_def.get("group", "A")
        value = metrics.get(gate_name)

        if floor is None or value is None:
            results[gate_name] = {
                "value": value, "floor": floor, "direction": direction,
                "passed": None, "delta": None, "group": group,
            }
            continue

        if isinstance(value, float) and math.isnan(value):
            results[gate_name] = {
                "value": value, "floor": floor, "direction": direction,
                "passed": False, "delta": None, "group": group,
            }
            continue

        if direction == "higher":
            passed = value >= floor
            delta = value - floor
        else:
            passed = value <= floor
            delta = floor - value

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
            "floor": floor, "direction": direction,
            "passed": passed,
            "delta": round(delta, 4) if delta is not None else None,
            "group": group,
        }
    return results


def evaluate_overall_pass(gate_results: dict[str, dict]) -> tuple[bool, bool]:
    """Returns (group_a_passed, group_b_passed)."""
    group_a_passed = True
    group_b_passed = True
    for result in gate_results.values():
        passed = result.get("passed")
        group = result.get("group", "A")
        if passed is not True:
            if group == "A":
                group_a_passed = False
            else:
                group_b_passed = False
    return group_a_passed, group_b_passed


def check_gates_multi_month(
    per_month: dict[str, dict],
    gates: dict,
    tail_max_failures: int = 1,
    champion_per_month: dict[str, dict] | None = None,
    noise_tolerance: float = 0.02,
) -> dict[str, dict]:
    """Three-layer gate check across multiple evaluation months."""
    months = sorted(per_month.keys())
    results = {}

    for gate_name, gate_def in gates.items():
        floor = gate_def.get("floor")
        tail_floor = gate_def.get("tail_floor")
        direction = gate_def["direction"]
        group = gate_def.get("group", "A")

        values = [per_month[m].get(gate_name) for m in months]
        values = [v for v in values if v is not None and (not isinstance(v, float) or v == v)]

        if not values:
            results[gate_name] = {
                "mean_value": None, "mean_passed": None,
                "tail_failures": None, "tail_passed": None,
                "bottom_2_mean": None, "tail_regression_passed": None,
                "group": group, "overall_passed": None,
            }
            continue

        mean_val = sum(values) / len(values)

        # Layer 1: mean check
        if floor is not None:
            mean_passed = mean_val >= floor if direction == "higher" else mean_val <= floor
        else:
            mean_passed = None

        # Layer 2: tail safety
        tail_failures = 0
        if tail_floor is not None:
            for v in values:
                if direction == "higher" and v < tail_floor:
                    tail_failures += 1
                elif direction == "lower" and v > tail_floor:
                    tail_failures += 1
            tail_passed = tail_failures <= tail_max_failures
        else:
            tail_passed = None

        # Layer 3: tail non-regression
        sorted_vals = sorted(values) if direction == "higher" else sorted(values, reverse=True)
        n_bottom = min(2, len(sorted_vals))
        bottom_2_mean = sum(sorted_vals[:n_bottom]) / n_bottom

        tail_regression_passed = True
        if champion_per_month is not None:
            champ_values = [champion_per_month[m].get(gate_name)
                            for m in months if m in champion_per_month]
            champ_values = [v for v in champ_values if v is not None and (not isinstance(v, float) or v == v)]
            if champ_values:
                champ_sorted = sorted(champ_values) if direction == "higher" else sorted(champ_values, reverse=True)
                cn = min(2, len(champ_sorted))
                champ_bottom_2 = sum(champ_sorted[:cn]) / cn
                if direction == "higher":
                    tail_regression_passed = bottom_2_mean >= champ_bottom_2 - noise_tolerance
                else:
                    tail_regression_passed = bottom_2_mean <= champ_bottom_2 + noise_tolerance

        overall = all(x is not False for x in [mean_passed, tail_passed, tail_regression_passed])

        results[gate_name] = {
            "mean_value": round(mean_val, 4),
            "mean_passed": mean_passed,
            "tail_failures": tail_failures,
            "tail_passed": tail_passed,
            "bottom_2_mean": round(bottom_2_mean, 4),
            "tail_regression_passed": tail_regression_passed,
            "group": group,
            "overall_passed": overall if mean_passed is not None else None,
        }

    return results


def evaluate_overall_pass_multi_month(gate_results: dict[str, dict]) -> tuple[bool, bool]:
    """Returns (group_a_passed, group_b_passed) from multi-month results."""
    group_a_passed = True
    group_b_passed = True
    for result in gate_results.values():
        passed = result.get("overall_passed")
        group = result.get("group", "A")
        if passed is not True:
            if group == "A":
                group_a_passed = False
            else:
                group_b_passed = False
    return group_a_passed, group_b_passed


def _is_v2_metrics(data: dict) -> bool:
    return isinstance(data, dict) and "per_month" in data and "aggregate" in data


def _extract_flat_metrics(data: dict) -> dict:
    if _is_v2_metrics(data):
        return data["aggregate"].get("mean", {})
    return data


def _extract_per_month(data: dict) -> dict[str, dict]:
    if _is_v2_metrics(data):
        return data.get("per_month", {})
    return {}


def build_comparison_table(
    versions: dict[str, dict],
    gates: dict,
    champion_metrics: dict | None = None,
    noise_tolerance: float = 0.02,
    tail_max_failures: int = 1,
) -> str:
    """Build Markdown comparison table across all versions."""
    gate_names = list(gates.keys())

    header = "| Version | " + " | ".join(gate_names) + " | Pass |"
    separator = "|---------|" + "|".join(["--------"] * len(gate_names)) + "|------|"
    rows = [header, separator]

    for version_id, metrics in sorted(versions.items()):
        if _is_v2_metrics(metrics):
            per_month = _extract_per_month(metrics)
            champion_pm = _extract_per_month(champion_metrics) if champion_metrics else None
            mm_results = check_gates_multi_month(
                per_month, gates, tail_max_failures, champion_pm, noise_tolerance
            )
            gate_display = {
                name: (r.get("mean_value"), r.get("overall_passed"), r.get("group", "A"))
                for name, r in mm_results.items()
            }
        else:
            champion_flat = _extract_flat_metrics(champion_metrics) if champion_metrics else None
            flat_results = check_gates(metrics, gates, champion_flat, noise_tolerance)
            gate_display = {
                name: (r.get("value"), r.get("passed"), r.get("group", "A"))
                for name, r in flat_results.items()
            }

        cells = []
        group_a_passed = True
        for gate_name in gate_names:
            value, passed, group = gate_display.get(gate_name, (None, None, "A"))
            if value is None:
                cells.append("--")
                if group == "A":
                    group_a_passed = False
            elif isinstance(value, float) and (value != value):
                cells.append("NaN")
                if group == "A":
                    group_a_passed = False
            else:
                mark = "P" if passed else "F"
                if passed is None:
                    mark = "?"
                    if group == "A":
                        group_a_passed = False
                elif not passed:
                    if group == "A":
                        group_a_passed = False
                cells.append(f"{value:.4f} {mark}")

        pass_str = "YES" if group_a_passed else "NO"
        row = f"| {version_id} | " + " | ".join(cells) + f" | {pass_str} |"
        rows.append(row)

    return "\n".join(rows)


def build_three_layer_detail(
    version_id: str,
    metrics: dict,
    gates: dict,
    champion_metrics: dict | None = None,
    noise_tolerance: float = 0.02,
    tail_max_failures: int = 1,
) -> str:
    """Build detailed three-layer gate analysis for a single version."""
    if not _is_v2_metrics(metrics):
        return ""

    per_month = _extract_per_month(metrics)
    champion_pm = _extract_per_month(champion_metrics) if champion_metrics else None
    mm_results = check_gates_multi_month(
        per_month, gates, tail_max_failures, champion_pm, noise_tolerance
    )

    lines = [f"### {version_id} — Three-Layer Detail\n"]
    lines.append("| Gate | Group | Mean | Floor | L1 | Tail Fail | L2 | Bot2 Mean | L3 | Overall |")
    lines.append("|------|-------|------|-------|----|-----------|----|-----------|----|---------|")

    for gate_name, r in mm_results.items():
        group = r.get("group", "A")
        mean_val = r.get("mean_value")
        floor = gates[gate_name].get("floor")
        l1 = "P" if r.get("mean_passed") else "F" if r.get("mean_passed") is False else "?"
        tail_f = r.get("tail_failures", "?")
        l2 = "P" if r.get("tail_passed") else "F" if r.get("tail_passed") is False else "?"
        b2 = r.get("bottom_2_mean")
        l3 = "P" if r.get("tail_regression_passed") else "F" if r.get("tail_regression_passed") is False else "?"
        overall = "P" if r.get("overall_passed") else "F" if r.get("overall_passed") is False else "?"

        mean_str = f"{mean_val:.4f}" if mean_val is not None else "--"
        floor_str = f"{floor:.4f}" if floor is not None else "--"
        b2_str = f"{b2:.4f}" if b2 is not None else "--"

        lines.append(f"| {gate_name} | {group} | {mean_str} | {floor_str} | {l1} | {tail_f} | {l2} | {b2_str} | {l3} | {overall} |")

    lines.append("")

    # Per-month breakdown for Group A gates
    months = sorted(per_month.keys())
    group_a_gates = [g for g, d in gates.items() if d.get("group", "A") == "A"]

    if months and group_a_gates:
        lines.append("**Per-month (Group A gates):**\n")
        header = "| Month | " + " | ".join(group_a_gates) + " |"
        sep = "|-------|" + "|".join(["--------"] * len(group_a_gates)) + "|"
        lines.append(header)
        lines.append(sep)

        for month in months:
            cells = []
            for gate in group_a_gates:
                v = per_month[month].get(gate)
                tail_floor = gates[gate].get("tail_floor")
                if v is not None:
                    direction = gates[gate]["direction"]
                    below = False
                    if tail_floor is not None:
                        if direction == "higher" and v < tail_floor:
                            below = True
                    mark = " **!**" if below else ""
                    cells.append(f"{v:.4f}{mark}")
                else:
                    cells.append("--")
            lines.append(f"| {month} | " + " | ".join(cells) + " |")

        lines.append("")

    return "\n".join(lines)


def run_comparison(
    batch_id: str,
    iteration: int,
    registry_dir: str = "registry",
    gates_path: str = "registry/gates.json",
    champion_path: str = "registry/champion.json",
    output_path: str | None = None,
) -> dict:
    """Run full comparison: load versions, check gates, build table, write outputs."""
    print(f"[compare] mem at start: {mem_mb():.0f} MB")

    with open(gates_path) as f:
        gates_data = json.load(f)
    gates = gates_data["gates"]
    noise_tolerance = gates_data.get("noise_tolerance", 0.02)
    tail_max_failures = gates_data.get("tail_max_failures", 1)

    champion_metrics = None
    with open(champion_path) as f:
        champion_data = json.load(f)
    champion_version = champion_data.get("version")
    if champion_version:
        champ_metrics_path = Path(registry_dir) / champion_version / "metrics.json"
        if champ_metrics_path.exists():
            with open(champ_metrics_path) as f:
                champion_metrics = json.load(f)

    versions = load_all_versions(registry_dir)
    table = build_comparison_table(
        versions, gates, champion_metrics, noise_tolerance, tail_max_failures
    )

    per_version = {}
    per_version_pass = {}
    for version_id, metrics in versions.items():
        if _is_v2_metrics(metrics):
            per_month = _extract_per_month(metrics)
            champion_pm = _extract_per_month(champion_metrics) if champion_metrics else None
            gate_results = check_gates_multi_month(
                per_month, gates, tail_max_failures, champion_pm, noise_tolerance
            )
            per_version[version_id] = gate_results
            ga, gb = evaluate_overall_pass_multi_month(gate_results)
        else:
            champion_flat = _extract_flat_metrics(champion_metrics) if champion_metrics else None
            gate_results = check_gates(metrics, gates, champion_flat, noise_tolerance)
            per_version[version_id] = gate_results
            ga, gb = evaluate_overall_pass(gate_results)
        per_version_pass[version_id] = {
            "group_a_passed": ga, "group_b_passed": gb, "overall_passed": ga,
        }

    comparison = {
        "batch_id": batch_id, "iteration": iteration,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "champion": champion_version,
        "noise_tolerance": noise_tolerance,
        "versions": per_version, "pass_summary": per_version_pass,
        "table": table,
    }

    detail_sections = []
    for version_id, metrics in sorted(versions.items()):
        if _is_v2_metrics(metrics):
            detail = build_three_layer_detail(
                version_id, metrics, gates, champion_metrics,
                noise_tolerance, tail_max_failures,
            )
            if detail:
                detail_sections.append(detail)

    if output_path:
        report = f"# Comparison Report\n\n"
        report += f"**Batch:** {batch_id}  **Iteration:** {iteration}\n"
        report += f"**Champion:** {champion_version or 'None'}\n"
        report += f"**Timestamp:** {comparison['timestamp']}\n\n"
        report += "## Summary Table\n\n"
        report += table + "\n\n"
        if detail_sections:
            report += "## Three-Layer Gate Detail\n\n"
            report += "\n".join(detail_sections)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"[compare] wrote report to {output_path}")

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
        batch_id=args.batch_id, iteration=args.iteration,
        registry_dir=args.registry_dir, gates_path=args.gates_path,
        champion_path=args.champion_path, output_path=args.output,
    )


if __name__ == "__main__":
    main()
