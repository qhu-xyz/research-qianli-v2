"""Research pipeline for MISO FTR Annual R1 baseline experiments.

Single file with metrics computation, version management, promotion gates,
and CLI commands. See design-planning.md Rev 3 for full specification.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# ─── Constants ─────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
VERSIONS_DIR = ROOT / "versions"
SCHEMA_VERSION = 1
QUARTERS = ["aq1", "aq2", "aq3", "aq4"]
VERSION_RE = re.compile(r"^v\d+$")
def _is_band_part(part: str) -> bool:
    """Check if a part uses band gates (BG0-BG7)."""
    return part.startswith("bands/")


# ─── Helpers ───────────────────────────────────────────────────────────────────


def _sanitize_for_json(obj):
    """Replace NaN/Inf with None for JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically: write to .tmp, fsync, then rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(_sanitize_for_json(data), f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, path)


# ─── Section 1: Metrics ───────────────────────────────────────────────────────


def compute_metrics(mcp: pl.Series, pred: pl.Series, label: str, total_n: int) -> dict:
    """Compute baseline evaluation metrics.

    Same logic as eval_baseline() in scripts/baseline_utils.py.
    Returns dict with stored fields only (drops label, n_dir, n_50, n_100).
    """
    mask = pred.is_not_null() & mcp.is_not_null()
    n = int(mask.sum())

    if n == 0:
        return {
            "n": 0, "coverage_pct": 0.0,
            "bias": 0, "mae": 0, "median_ae": 0, "p95_ae": 0, "p99_ae": 0,
            "dir_all": float("nan"), "dir_50": float("nan"), "dir_100": float("nan"),
            "mae_tiny": float("nan"), "mae_small": float("nan"),
            "mae_med": float("nan"), "mae_large": float("nan"),
        }

    m = mcp.filter(mask)
    b = pred.filter(mask)
    res = m - b
    abs_res = res.abs()
    abs_m = m.abs()

    def dir_acc(ms: pl.Series, bs: pl.Series) -> float:
        valid = (ms != 0) & (bs != 0)
        vn = int(valid.sum())
        if vn == 0:
            return float("nan")
        return round(float((ms.filter(valid).sign() == bs.filter(valid).sign()).mean()) * 100, 1)

    def mae_bin(lo: float, hi: float) -> float:
        bm = (abs_m >= lo) & (abs_m < hi)
        bn = int(bm.sum())
        return round(float(abs_res.filter(bm).mean()), 0) if bn > 100 else float("nan")

    da_all = dir_acc(m, b)

    mask_50 = abs_m > 50
    da_50 = dir_acc(m.filter(mask_50), b.filter(mask_50)) if int(mask_50.sum()) > 0 else float("nan")

    mask_100 = abs_m > 100
    da_100 = dir_acc(m.filter(mask_100), b.filter(mask_100)) if int(mask_100.sum()) > 0 else float("nan")

    return {
        "n": n,
        "coverage_pct": round(n / total_n * 100, 1) if total_n > 0 else 0.0,
        "bias": round(float(res.mean()), 1),
        "mae": round(float(abs_res.mean()), 1),
        "median_ae": round(float(abs_res.median()), 1),
        "p95_ae": round(float(abs_res.quantile(0.95)), 0),
        "p99_ae": round(float(abs_res.quantile(0.99)), 0),
        "dir_all": da_all,
        "dir_50": da_50,
        "dir_100": da_100,
        "mae_tiny": mae_bin(0, 50),
        "mae_small": mae_bin(50, 250),
        "mae_med": mae_bin(250, 1000),
        "mae_large": mae_bin(1000, 999999),
    }


def compute_matched_comparison(
    df: pl.DataFrame, candidate_col: str, promoted_col: str, quarter: str,
    mcp_col: str = "mcp_mean",
) -> dict:
    """Compute head-to-head comparison on matched rows.

    On rows where both candidate_col and promoted_col are non-null:
    - Compute candidate metrics via compute_metrics
    - Compute path-level win rate: % of rows where |candidate_error| < |promoted_error|
      Ties (|cand_err| == |prom_err|) count as 0.5 wins for each side.

    Returns dict with: n_matched, mae, dir_all, win_rate, p99_ae.
    """
    matched = df.filter(
        pl.col(candidate_col).is_not_null()
        & pl.col(promoted_col).is_not_null()
        & pl.col(mcp_col).is_not_null()
    )
    n_matched = matched.height

    if n_matched == 0:
        return {
            "n_matched": 0, "mae": 0, "dir_all": float("nan"),
            "win_rate": float("nan"), "p99_ae": 0,
        }

    metrics = compute_metrics(matched[mcp_col], matched[candidate_col], "", n_matched)

    # Win rate: % of rows where candidate absolute error < promoted absolute error
    cand_err = (matched[mcp_col] - matched[candidate_col]).abs()
    prom_err = (matched[mcp_col] - matched[promoted_col]).abs()
    wins = (cand_err < prom_err).cast(pl.Float64)
    ties = (cand_err == prom_err).cast(pl.Float64) * 0.5
    win_rate = round(float((wins + ties).mean()) * 100, 1)

    return {
        "n_matched": n_matched,
        "mae": metrics["mae"],
        "dir_all": metrics["dir_all"],
        "win_rate": win_rate,
        "p99_ae": metrics["p99_ae"],
    }


def compute_stability(per_py_metrics: dict[str, dict]) -> dict:
    """Compute stability metrics from per-PY breakdown.

    Input: {py_str: {"mae": float, "dir_all": float, ...}} for each PY.
    Returns: mae_cv, worst_py, worst_py_mae, median_py_mae, dir_all_range.
    """
    maes = [
        v["mae"] for v in per_py_metrics.values()
        if isinstance(v.get("mae"), (int, float)) and not math.isnan(v["mae"])
    ]
    dir_alls = [
        v["dir_all"] for v in per_py_metrics.values()
        if isinstance(v.get("dir_all"), (int, float)) and not math.isnan(v["dir_all"])
    ]

    if len(maes) < 2:
        return {
            "mae_cv": float("nan"), "worst_py": "", "worst_py_mae": float("nan"),
            "median_py_mae": float("nan"), "dir_all_range": [float("nan"), float("nan")],
        }

    mae_mean = statistics.mean(maes)
    mae_std = statistics.stdev(maes)
    mae_cv = round(mae_std / mae_mean, 4) if mae_mean > 0 else 0.0

    worst_py = max(
        per_py_metrics.items(),
        key=lambda kv: kv[1]["mae"] if isinstance(kv[1].get("mae"), (int, float)) and not math.isnan(kv[1]["mae"]) else -1,
    )[0]
    worst_py_mae = per_py_metrics[worst_py]["mae"]
    median_py_mae = round(statistics.median(maes), 1)

    dir_all_range = (
        [round(min(dir_alls), 1), round(max(dir_alls), 1)] if dir_alls
        else [float("nan"), float("nan")]
    )

    return {
        "mae_cv": mae_cv,
        "worst_py": worst_py,
        "worst_py_mae": worst_py_mae,
        "median_py_mae": median_py_mae,
        "dir_all_range": dir_all_range,
    }


def compute_full_evaluation(
    df: pl.DataFrame,
    candidate_col: str,
    promoted_col: str | None,
    quarters: list[str],
    mcp_col: str = "mcp_mean",
    quarter_col: str = "period_type",
    py_col: str = "planning_year",
    class_col: str = "class_type",
    h_col: str = "mtm_1st_mean",
) -> dict:
    """Master evaluation function.

    Returns full metrics.json structure with sections:
    - overall: compute_metrics per quarter
    - matched: compute_matched_comparison per quarter (if promoted_col given)
    - per_py: compute_metrics per (quarter, PY) pair
    - per_class: compute_metrics per (quarter, class_type) pair
    - stability: compute_stability per quarter
    - vs_h: MAE improvement % and Dir% improvement pp vs H baseline
    """
    result: dict = {
        "schema_version": SCHEMA_VERSION,
        "overall": {},
        "per_py": {},
        "per_class": {},
        "stability": {},
        "vs_h": {},
    }

    if promoted_col is not None:
        result["matched"] = {
            "description": "Head-to-head on rows where both candidate and promoted have non-null predictions",
            "match_keys": ["source_id", "sink_id", "class_type", "planning_year"],
        }

    for q in quarters:
        qdf = df.filter(pl.col(quarter_col) == q)
        total_n = qdf.height

        # Overall
        result["overall"][q] = compute_metrics(qdf[mcp_col], qdf[candidate_col], "", total_n)

        # Matched
        if promoted_col is not None:
            result["matched"][q] = compute_matched_comparison(
                qdf, candidate_col, promoted_col, q, mcp_col,
            )

        # Per-PY
        pys = sorted(qdf[py_col].unique().to_list())
        py_metrics = {}
        for py in pys:
            py_df = qdf.filter(pl.col(py_col) == py)
            py_metrics[str(py)] = compute_metrics(
                py_df[mcp_col], py_df[candidate_col], "", py_df.height,
            )
        result["per_py"][q] = py_metrics

        # Per-class
        classes = sorted(qdf[class_col].unique().to_list())
        class_metrics = {}
        for cls in classes:
            cls_df = qdf.filter(pl.col(class_col) == cls)
            class_metrics[cls] = compute_metrics(
                cls_df[mcp_col], cls_df[candidate_col], "", cls_df.height,
            )
        result["per_class"][q] = class_metrics

        # Stability
        result["stability"][q] = compute_stability(py_metrics)

        # vs_h
        h_metrics = compute_metrics(qdf[mcp_col], qdf[h_col], "", total_n)
        cand_metrics = result["overall"][q]
        if h_metrics["mae"] > 0:
            mae_imp = round((cand_metrics["mae"] - h_metrics["mae"]) / h_metrics["mae"] * 100, 1)
        else:
            mae_imp = 0.0
        dir_imp = round(cand_metrics["dir_all"] - h_metrics["dir_all"], 1)
        result["vs_h"][q] = {
            "mae_improvement_pct": mae_imp,
            "dir_improvement_pp": dir_imp,
        }

    return result


def save_metrics(metrics: dict, version_dir: Path) -> None:
    """Atomic write metrics.json to version directory."""
    _atomic_write_json(version_dir / "metrics.json", metrics)


# ─── Section 2: Version management ────────────────────────────────────────────


def validate_version_id(version_id: str) -> None:
    """Raises ValueError if version_id doesn't match pattern."""
    if not VERSION_RE.match(version_id):
        raise ValueError(
            f"Invalid version ID '{version_id}'. Must match {VERSION_RE.pattern}"
        )


def create_version(part: str, version_id: str, description: str) -> Path:
    """Create version directory with config.json stub.

    Returns the directory path. Raises FileExistsError if already exists.
    """
    validate_version_id(version_id)
    version_dir = VERSIONS_DIR / part / version_id
    if version_dir.exists():
        raise FileExistsError(f"Version directory already exists: {version_dir}")
    version_dir.mkdir(parents=True)

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
        ).strip()
    except Exception:
        git_hash = "unknown"

    config = {
        "schema_version": SCHEMA_VERSION,
        "version": version_id,
        "description": description,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "part": part,
        "method": {},
        "parameters": {},
        "data_sources": [],
        "environment": {
            "git_hash": git_hash,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "polars_version": pl.__version__,
        },
    }
    _atomic_write_json(version_dir / "config.json", config)
    return version_dir


def list_versions(part: str) -> list[str]:
    """Return sorted list of version directory names matching VERSION_RE."""
    part_dir = VERSIONS_DIR / part
    if not part_dir.exists():
        return []
    return sorted(
        d.name for d in part_dir.iterdir()
        if d.is_dir() and VERSION_RE.match(d.name)
    )


def get_promoted(part: str) -> dict | None:
    """Read promoted.json. Returns dict or None if not found."""
    path = VERSIONS_DIR / part / "promoted.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def promote(part: str, version_id: str, notes: str = "", force: bool = False) -> None:
    """Promote a version. Runs gate check before writing promoted.json.

    - HARD gate failure → refuses unless force=True
    - Stale matched section → refuses unless force=True
    - SOFT/ADVISORY failures print warnings but don't block
    """
    validate_version_id(version_id)
    version_dir = VERSIONS_DIR / part / version_id
    if not version_dir.exists():
        raise FileNotFoundError(f"Version directory not found: {version_dir}")

    metrics_path = version_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")

    with open(metrics_path) as f:
        candidate_metrics = json.load(f)

    current = get_promoted(part)
    if current is not None:
        promoted_dir = VERSIONS_DIR / part / current["version"]
        promoted_metrics_path = promoted_dir / "metrics.json"
        if promoted_metrics_path.exists():
            with open(promoted_metrics_path) as f:
                promoted_metrics = json.load(f)

            # Staleness check (baseline only — bands don't have matched sections)
            if not _is_band_part(part):
                compared_against = candidate_metrics.get("compared_against")
                if compared_against != current["version"]:
                    if not force:
                        print(
                            f"ERROR: Stale matched section. Candidate was compared against "
                            f"'{compared_against}', but current promoted is '{current['version']}'."
                        )
                        print("Re-evaluate or use --force to override.")
                        sys.exit(1)
                    else:
                        print(
                            f"WARNING: Stale matched section (compared against "
                            f"'{compared_against}', not '{current['version']}'). "
                            f"Proceeding with --force."
                        )

            if _is_band_part(part):
                config_path = version_dir / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        cand_config = json.load(f)
                    candidate_metrics["baseline_version"] = cand_config.get("baseline_version")
                gate_results = check_band_gates(candidate_metrics, promoted_metrics)
            else:
                gate_results = check_gates(candidate_metrics, promoted_metrics)
            print_gate_table(gate_results)

            hard_failures = [
                g for g in gate_results
                if g["severity"] == "HARD" and not g["passed"]
            ]
            if hard_failures:
                if not force:
                    print("\nERROR: HARD gate(s) failed. Cannot promote.")
                    print("Use --force to override.")
                    sys.exit(1)
                else:
                    print("\nWARNING: HARD gate(s) failed. Promoting with --force.")
        else:
            print(
                f"WARNING: Promoted version '{current['version']}' has no metrics.json. "
                f"Skipping gate check."
            )
    else:
        # First version: only absolute gates
        if _is_band_part(part):
            # Read config for baseline_version (BG0)
            config_path = version_dir / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    cand_config = json.load(f)
                candidate_metrics["baseline_version"] = cand_config.get("baseline_version")
            gate_results = check_band_gates(candidate_metrics, None)
        else:
            gate_results = check_gates(candidate_metrics, None)
        print_gate_table(gate_results)

    promoted_data = {
        "version": version_id,
        "promoted_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "notes": notes,
    }
    _atomic_write_json(VERSIONS_DIR / part / "promoted.json", promoted_data)
    print(f"\nPromoted {version_id} as current best for {part}.")


# ─── Section 3: Gates ─────────────────────────────────────────────────────────


def check_gates(candidate_metrics: dict, promoted_metrics: dict | None) -> list[dict]:
    """Check all 9 promotion gates. Returns list of gate result dicts.

    If promoted_metrics is None, skips comparative gates (G1/G2/G8/G9).
    Quarter keys read from candidate_metrics["overall"]; all 4 must be present.
    """
    quarters = sorted(
        k for k in candidate_metrics.get("overall", {}).keys()
        if k.startswith("aq")
    )
    if len(quarters) != 4:
        raise ValueError(
            f"Expected 4 quarters in metrics, got {len(quarters)}: {quarters}"
        )

    if promoted_metrics is not None:
        if candidate_metrics.get("schema_version") != promoted_metrics.get("schema_version"):
            raise ValueError("Cannot compare across different schema versions")

    results = []
    has_promoted = promoted_metrics is not None
    has_matched = "matched" in candidate_metrics and has_promoted

    # G1: MAE improvement (HARD, comparative — matched)
    if has_matched and "matched" in (promoted_metrics or {}):
        pass_count = 0
        values = {}
        for q in quarters:
            cand_mae = candidate_metrics["matched"].get(q, {}).get("mae", float("inf"))
            prom_mae = promoted_metrics["matched"].get(q, {}).get("mae", float("inf"))
            passed_q = cand_mae <= prom_mae
            values[q] = f"{cand_mae:.0f} vs {prom_mae:.0f}"
            if passed_q:
                pass_count += 1
        results.append({
            "gate": "G1", "name": "MAE improvement", "severity": "HARD",
            "passed": pass_count == 4, "detail": f"{pass_count}/4 quarters pass",
            "values": values,
        })

    # G2: Direction preserved (HARD, comparative — matched, allow -1.0pp)
    if has_matched and "matched" in (promoted_metrics or {}):
        pass_count = 0
        values = {}
        for q in quarters:
            cand_dir = candidate_metrics["matched"].get(q, {}).get("dir_all", 0)
            prom_dir = promoted_metrics["matched"].get(q, {}).get("dir_all", 0)
            passed_q = cand_dir >= prom_dir - 1.0
            diff = round(cand_dir - prom_dir, 1)
            values[q] = f"{diff:+.1f}pp"
            if passed_q:
                pass_count += 1
        results.append({
            "gate": "G2", "name": "Direction preserved", "severity": "HARD",
            "passed": pass_count == 4, "detail": f"{pass_count}/4 quarters pass",
            "values": values,
        })

    # G3: Coverage floor (HARD, absolute — 95%)
    pass_count = 0
    values = {}
    for q in quarters:
        cov = candidate_metrics["overall"][q].get("coverage_pct", 0)
        passed_q = cov >= 95.0
        values[q] = f"{cov:.1f}%"
        if passed_q:
            pass_count += 1
    min_cov = min(float(v.rstrip("%")) for v in values.values())
    results.append({
        "gate": "G3", "name": "Coverage floor", "severity": "HARD",
        "passed": pass_count == 4, "detail": f"min {min_cov:.1f}%",
        "values": values,
    })

    # G4: Bias sign (SOFT, absolute — bias >= 0)
    pass_count = 0
    values = {}
    for q in quarters:
        bias = candidate_metrics["overall"][q].get("bias", 0)
        passed_q = bias >= 0
        values[q] = f"{bias:+.0f}"
        if passed_q:
            pass_count += 1
    results.append({
        "gate": "G4", "name": "Bias sign", "severity": "SOFT",
        "passed": pass_count == 4, "detail": f"{pass_count}/4 quarters >= 0",
        "values": values,
    })

    # G5: Per-PY stability (SOFT, absolute — mae_cv < 0.30)
    pass_count = 0
    values = {}
    max_cv = -1.0
    for q in quarters:
        cv = candidate_metrics.get("stability", {}).get(q, {}).get("mae_cv")
        if isinstance(cv, (int, float)) and cv is not None:
            passed_q = cv < 0.30
            values[q] = f"{cv:.4f}"
            max_cv = max(max_cv, cv)
            if passed_q:
                pass_count += 1
        else:
            passed_q = False
            values[q] = "n/a"
    results.append({
        "gate": "G5", "name": "Per-PY stability", "severity": "SOFT",
        "passed": pass_count == 4,
        "detail": f"max CV {max_cv:.2f}" if max_cv >= 0 else "n/a",
        "values": values,
    })

    # G6: Worst-PY bound (SOFT, absolute — worst_py_mae < 1.5 * median_py_mae)
    pass_count = 0
    values = {}
    max_ratio = -1.0
    for q in quarters:
        stab = candidate_metrics.get("stability", {}).get(q, {})
        worst = stab.get("worst_py_mae")
        median = stab.get("median_py_mae")
        if (
            isinstance(worst, (int, float)) and worst is not None
            and isinstance(median, (int, float)) and median is not None
            and median > 0
        ):
            ratio = worst / median
            passed_q = ratio < 1.5
            values[q] = f"{ratio:.2f}x"
            max_ratio = max(max_ratio, ratio)
        else:
            passed_q = False
            values[q] = "n/a"
        if passed_q:
            pass_count += 1
    results.append({
        "gate": "G6", "name": "Worst-PY bound", "severity": "SOFT",
        "passed": pass_count == 4,
        "detail": f"max {max_ratio:.2f}x median" if max_ratio >= 0 else "n/a",
        "values": values,
    })

    # G7: Class parity (ADVISORY — |onpeak - offpeak| / avg < 0.40)
    pass_count = 0
    values = {}
    max_gap = 0.0
    for q in quarters:
        pc = candidate_metrics.get("per_class", {}).get(q, {})
        on_mae = pc.get("onpeak", {}).get("mae")
        off_mae = pc.get("offpeak", {}).get("mae")
        if (
            isinstance(on_mae, (int, float)) and on_mae is not None
            and isinstance(off_mae, (int, float)) and off_mae is not None
        ):
            avg_mae = (on_mae + off_mae) / 2
            gap = abs(on_mae - off_mae) / avg_mae if avg_mae > 0 else 0
            passed_q = gap < 0.40
            values[q] = f"{gap:.2f}"
            max_gap = max(max_gap, gap)
        else:
            passed_q = False
            values[q] = "n/a"
        if passed_q:
            pass_count += 1
    results.append({
        "gate": "G7", "name": "Class parity", "severity": "ADVISORY",
        "passed": pass_count == 4,
        "detail": f"max gap {max_gap:.2f}" if max_gap > 0 else "n/a",
        "values": values,
    })

    # G8: Win rate (ADVISORY, comparative — win_rate >= 50%)
    if has_matched:
        pass_count = 0
        values = {}
        min_wr = 100.0
        for q in quarters:
            wr = candidate_metrics.get("matched", {}).get(q, {}).get("win_rate", 0)
            passed_q = wr >= 50.0
            values[q] = f"{wr:.1f}%"
            min_wr = min(min_wr, wr)
            if passed_q:
                pass_count += 1
        results.append({
            "gate": "G8", "name": "Win rate", "severity": "ADVISORY",
            "passed": pass_count == 4, "detail": f"min {min_wr:.1f}%",
            "values": values,
        })

    # G9: Tail risk (ADVISORY, comparative — p99 <= promoted * 1.10)
    if has_matched and "matched" in (promoted_metrics or {}):
        pass_count = 0
        values = {}
        max_pct = -999.0
        for q in quarters:
            cand_p99 = candidate_metrics.get("matched", {}).get(q, {}).get("p99_ae", float("inf"))
            prom_p99 = promoted_metrics.get("matched", {}).get(q, {}).get("p99_ae", 1)
            if prom_p99 and prom_p99 > 0:
                ratio = cand_p99 / prom_p99
                passed_q = ratio <= 1.10
                pct = round((ratio - 1) * 100, 1)
                values[q] = f"{pct:+.1f}%"
                max_pct = max(max_pct, pct)
            else:
                passed_q = True
                values[q] = "n/a"
            if passed_q:
                pass_count += 1
        results.append({
            "gate": "G9", "name": "Tail risk (p99)", "severity": "ADVISORY",
            "passed": pass_count == 4,
            "detail": f"max {max_pct:+.1f}%" if max_pct > -999 else "n/a",
            "values": values,
        })

    return results


def check_band_gates(
    candidate_metrics: dict,
    promoted_metrics: dict | None,
) -> list[dict]:
    """Check band promotion gates BG0-BG7. Returns list of gate result dicts.

    If promoted_metrics is None, skips BG4 (width comparison).
    """
    quarters = QUARTERS
    results = []

    # BG0: Baseline still promoted (HARD)
    # Read candidate config to get baseline_version, check against promoted baseline
    # This is checked via config.json, not metrics — caller validates before calling.
    # We check that config's baseline_version matches current promoted baseline.
    baseline_promoted = get_promoted("baseline")
    config_path = None
    # Find candidate config from metrics' parent directory
    # BG0 is checked in promote()/compare() wrapper — skip here if no config context.
    # Instead, we accept baseline_version as a field in the metrics dict.
    baseline_version = candidate_metrics.get("baseline_version")
    # BG0 only applies when baseline_version is a versioned baseline (e.g., "v3").
    # R2/R3 bands use M (prior round MCP) which isn't a versioned artifact.
    if baseline_promoted and baseline_version and VERSION_RE.match(baseline_version):
        passed = baseline_version == baseline_promoted["version"]
        results.append({
            "gate": "BG0", "name": "Baseline still promoted", "severity": "HARD",
            "passed": passed,
            "detail": f"bands uses {baseline_version}, promoted is {baseline_promoted['version']}",
            "values": {},
        })

    # BG1: P95 coverage accuracy (HARD) — |actual - 95.0| < 3.0pp for all 4 quarters
    pass_count = 0
    values = {}
    for q in quarters:
        cov = candidate_metrics.get("coverage", {}).get(q, {}).get("overall", {}).get("p95", {})
        error = cov.get("error", 99)
        passed_q = abs(error) < 3.0
        values[q] = f"{cov.get('actual', 0):.1f}% (err {error:+.1f}pp)"
        if passed_q:
            pass_count += 1
    results.append({
        "gate": "BG1", "name": "P95 coverage accuracy", "severity": "HARD",
        "passed": pass_count == 4,
        "detail": f"{pass_count}/4 quarters within 3pp",
        "values": values,
    })

    # BG2: P50 coverage accuracy (HARD) — |actual - 50.0| < 5.0pp for all 4 quarters
    pass_count = 0
    values = {}
    for q in quarters:
        cov = candidate_metrics.get("coverage", {}).get(q, {}).get("overall", {}).get("p50", {})
        error = cov.get("error", 99)
        passed_q = abs(error) < 5.0
        values[q] = f"{cov.get('actual', 0):.1f}% (err {error:+.1f}pp)"
        if passed_q:
            pass_count += 1
    results.append({
        "gate": "BG2", "name": "P50 coverage accuracy", "severity": "HARD",
        "passed": pass_count == 4,
        "detail": f"{pass_count}/4 quarters within 5pp",
        "values": values,
    })

    # BG3: Per-bin uniformity P95 (HARD) — all bins within 5pp of target, all 4 quarters
    pass_count = 0
    values = {}
    for q in quarters:
        per_bin = candidate_metrics.get("coverage", {}).get(q, {}).get("per_bin", {})
        all_bins_ok = True
        worst_err = 0
        for bin_label in per_bin:
            p95 = per_bin[bin_label].get("p95", {})
            error = p95.get("error", 99)
            if abs(error) >= 5.0:
                all_bins_ok = False
            worst_err = max(worst_err, abs(error))
        values[q] = f"worst {worst_err:.1f}pp"
        if all_bins_ok:
            pass_count += 1
    results.append({
        "gate": "BG3", "name": "Per-bin uniformity (P95)", "severity": "HARD",
        "passed": pass_count == 4,
        "detail": f"{pass_count}/4 quarters, all bins within 5pp",
        "values": values,
    })

    # BG4: Width narrower or equal (SOFT) — candidate P95 width <= promoted P95 width
    has_promoted = promoted_metrics is not None
    if has_promoted:
        pass_count = 0
        values = {}
        for q in quarters:
            cand_w = candidate_metrics.get("widths", {}).get(q, {}).get("overall", {}).get("p95", {}).get("mean_width")
            prom_w = promoted_metrics.get("widths", {}).get(q, {}).get("overall", {}).get("p95", {}).get("mean_width")
            if cand_w is not None and prom_w is not None:
                passed_q = cand_w <= prom_w
                values[q] = f"{cand_w:.0f} vs {prom_w:.0f}"
            else:
                passed_q = True
                values[q] = "n/a"
            if passed_q:
                pass_count += 1
        results.append({
            "gate": "BG4", "name": "Width narrower or equal", "severity": "SOFT",
            "passed": pass_count == 4,
            "detail": f"{pass_count}/4 quarters",
            "values": values,
        })

    # BG5: Per-PY stability (SOFT) — p95_worst_py_coverage >= 90.0 for all 4 quarters
    pass_count = 0
    values = {}
    for q in quarters:
        stab = candidate_metrics.get("stability", {}).get(q, {})
        worst_cov = stab.get("p95_worst_py_coverage", 0)
        worst_py = stab.get("p95_worst_py", "?")
        passed_q = worst_cov >= 90.0
        values[q] = f"{worst_cov:.1f}% (PY {worst_py})"
        if passed_q:
            pass_count += 1
    results.append({
        "gate": "BG5", "name": "Per-PY stability", "severity": "SOFT",
        "passed": pass_count == 4,
        "detail": f"{pass_count}/4 quarters, worst PY >= 90%",
        "values": values,
    })

    # BG6: Width monotonicity (ADVISORY) — p50 < p70 < p80 < p90 < p95 overall
    pass_count = 0
    values = {}
    level_order = ["p50", "p70", "p80", "p90", "p95"]
    for q in quarters:
        widths_q = candidate_metrics.get("widths", {}).get(q, {}).get("overall", {})
        ws = []
        for lvl in level_order:
            w = widths_q.get(lvl, {}).get("mean_width")
            ws.append(w)
        if all(w is not None for w in ws):
            monotonic = all(ws[i] < ws[i + 1] for i in range(len(ws) - 1))
            values[q] = " < ".join(f"{w:.0f}" for w in ws) if monotonic else "NOT MONO"
        else:
            monotonic = False
            values[q] = "n/a"
        if monotonic:
            pass_count += 1
    results.append({
        "gate": "BG6", "name": "Width monotonicity", "severity": "ADVISORY",
        "passed": pass_count == 4,
        "detail": f"{pass_count}/4 quarters",
        "values": values,
    })

    # BG7: Class parity coverage (ADVISORY) — |onpeak - offpeak| < 5pp at P95
    # This requires per-class coverage data. If not present, skip.
    per_class_cov = candidate_metrics.get("per_class_coverage")
    if per_class_cov:
        pass_count = 0
        values = {}
        for q in quarters:
            qc = per_class_cov.get(q, {})
            on_cov = qc.get("onpeak", {}).get("p95", {}).get("actual")
            off_cov = qc.get("offpeak", {}).get("p95", {}).get("actual")
            if on_cov is not None and off_cov is not None:
                gap = abs(on_cov - off_cov)
                passed_q = gap < 5.0
                values[q] = f"|{on_cov:.1f} - {off_cov:.1f}| = {gap:.1f}pp"
            else:
                passed_q = True
                values[q] = "n/a"
            if passed_q:
                pass_count += 1
        results.append({
            "gate": "BG7", "name": "Class parity coverage", "severity": "ADVISORY",
            "passed": pass_count == 4,
            "detail": f"{pass_count}/4 quarters within 5pp",
            "values": values,
        })

    return results


def print_gate_table(gate_results: list[dict]) -> None:
    """Pretty-print gate pass/fail table."""
    if not gate_results:
        print("No gates to display.")
        return

    print(f"\n{'Gate':<6} {'Check':<22} {'Severity':<10} {'Result':<8} {'Detail'}")
    print(f"{'─' * 6} {'─' * 22} {'─' * 10} {'─' * 8} {'─' * 30}")
    for g in gate_results:
        result_str = "PASS" if g["passed"] else "FAIL"
        print(f"{g['gate']:<6} {g['name']:<22} {g['severity']:<10} {result_str:<8} {g['detail']}")

    hard_fails = sum(1 for g in gate_results if g["severity"] == "HARD" and not g["passed"])
    soft_fails = sum(1 for g in gate_results if g["severity"] == "SOFT" and not g["passed"])
    advisory_warns = sum(1 for g in gate_results if g["severity"] == "ADVISORY" and not g["passed"])

    print()
    if hard_fails:
        print(f"RESULT: {hard_fails} HARD gate(s) FAILED. Cannot promote without --force.")
    elif soft_fails:
        print(f"RESULT: All HARD gates PASS. {soft_fails} SOFT gate(s) failed (needs justification).")
    elif advisory_warns:
        print(f"RESULT: All HARD/SOFT gates PASS. {advisory_warns} advisory warning(s).")
    else:
        print("RESULT: All gates PASS. Auto-promotable.")


def compare(part: str, candidate_id: str, promoted_id: str | None = None) -> None:
    """Load metrics for both versions, check gates, print comparison table.

    If promoted_id is None, reads promoted.json for the same part.
    For baseline part: staleness check on matched section.
    For bands part: uses band-specific gates (BG0-BG7).
    """
    cand_dir = VERSIONS_DIR / part / candidate_id
    cand_metrics_path = cand_dir / "metrics.json"
    if not cand_metrics_path.exists():
        print(f"ERROR: metrics.json not found for {candidate_id}")
        sys.exit(1)
    with open(cand_metrics_path) as f:
        candidate_metrics = json.load(f)

    # For bands, also read config.json to get baseline_version for BG0
    if _is_band_part(part):
        cand_config_path = cand_dir / "config.json"
        if cand_config_path.exists():
            with open(cand_config_path) as f:
                cand_config = json.load(f)
            candidate_metrics["baseline_version"] = cand_config.get("baseline_version")

    if promoted_id is None:
        prom = get_promoted(part)
        if prom is None:
            print("No promoted version. Running absolute gates only.")
            if _is_band_part(part):
                gate_results = check_band_gates(candidate_metrics, None)
            else:
                gate_results = check_gates(candidate_metrics, None)
            print_gate_table(gate_results)
            return
        promoted_id = prom["version"]

    print(f"Comparing {candidate_id} vs {promoted_id}")

    prom_dir = VERSIONS_DIR / part / promoted_id
    prom_metrics_path = prom_dir / "metrics.json"
    if not prom_metrics_path.exists():
        print(f"ERROR: metrics.json not found for {promoted_id}")
        sys.exit(1)
    with open(prom_metrics_path) as f:
        promoted_metrics = json.load(f)

    if _is_band_part(part):
        gate_results = check_band_gates(candidate_metrics, promoted_metrics)
    else:
        # Staleness check for baseline
        compared_against = candidate_metrics.get("compared_against")
        if compared_against is not None and compared_against != promoted_id:
            print(
                f"\nWARNING: matched section was computed against '{compared_against}', "
                f"not '{promoted_id}'."
            )
            print("Re-evaluate to get fresh matched metrics. Only absolute gates (G3-G7) will run.")
            gate_results = check_gates(candidate_metrics, None)
        else:
            gate_results = check_gates(candidate_metrics, promoted_metrics)

    print_gate_table(gate_results)


# ─── Section 4: Validate ──────────────────────────────────────────────────────


def validate(part: str) -> int:
    """Walk all version dirs and check schema compliance.

    Returns 0 if no errors, 1 if any ERROR found. WARNs don't fail.
    """
    part_dir = VERSIONS_DIR / part
    if not part_dir.exists():
        print(f"ERROR: Part directory not found: {part_dir}")
        return 1

    errors = 0
    warns = 0

    version_dirs = sorted(d for d in part_dir.iterdir() if d.is_dir())

    for vdir in version_dirs:
        vid = vdir.name
        prefix = f"  [{vid}]"

        # Version ID format
        if not VERSION_RE.match(vid):
            print(f"{prefix} ERROR: Invalid version ID format (must match {VERSION_RE.pattern})")
            errors += 1
            continue

        # config.json exists + valid JSON
        config_path = vdir / "config.json"
        if not config_path.exists():
            print(f"{prefix} ERROR: config.json missing")
            errors += 1
            continue

        try:
            with open(config_path) as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"{prefix} ERROR: config.json invalid JSON: {e}")
            errors += 1
            continue

        # config schema_version
        if config.get("schema_version") != SCHEMA_VERSION:
            print(f"{prefix} ERROR: config.json schema_version != {SCHEMA_VERSION}")
            errors += 1

        # config required fields
        for field in ["version", "description", "created", "part", "environment"]:
            if field not in config:
                print(f"{prefix} ERROR: config.json missing field '{field}'")
                errors += 1

        # config version match
        if config.get("version") != vid:
            print(
                f"{prefix} ERROR: config.version '{config.get('version')}' "
                f"!= directory name '{vid}'"
            )
            errors += 1

        # metrics.json
        metrics_path = vdir / "metrics.json"
        if not metrics_path.exists():
            print(f"{prefix} WARN: metrics.json missing (not yet evaluated)")
            warns += 1
        else:
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
            except json.JSONDecodeError as e:
                print(f"{prefix} ERROR: metrics.json invalid JSON: {e}")
                errors += 1
                continue

            if _is_band_part(part):
                # Band-specific schema checks
                for section in ["coverage", "widths", "stability"]:
                    if section not in metrics:
                        print(f"{prefix} ERROR: metrics.json missing section '{section}'")
                        errors += 1
                    else:
                        for q in QUARTERS:
                            if q not in metrics[section]:
                                print(f"{prefix} ERROR: metrics.json {section} missing quarter '{q}'")
                                errors += 1

                # Coverage structure checks
                if "coverage" in metrics:
                    for q in QUARTERS:
                        qc = metrics["coverage"].get(q, {})
                        if "overall" not in qc:
                            print(f"{prefix} ERROR: metrics.json coverage.{q} missing 'overall'")
                            errors += 1
                        else:
                            for lvl in ["p50", "p70", "p80", "p90", "p95"]:
                                if lvl not in qc["overall"]:
                                    print(f"{prefix} ERROR: metrics.json coverage.{q}.overall missing '{lvl}'")
                                    errors += 1
                        if "per_bin" not in qc:
                            print(f"{prefix} ERROR: metrics.json coverage.{q} missing 'per_bin'")
                            errors += 1

                # Widths structure checks
                if "widths" in metrics:
                    for q in QUARTERS:
                        qw = metrics["widths"].get(q, {})
                        if "overall" not in qw:
                            print(f"{prefix} ERROR: metrics.json widths.{q} missing 'overall'")
                            errors += 1

                # Config baseline_version check
                if config.get("part") == "bands/r1" and "baseline_version" not in config:
                    print(f"{prefix} WARN: config.json missing 'baseline_version'")
                    warns += 1

            else:
                # Baseline-specific schema checks
                if metrics.get("schema_version") != SCHEMA_VERSION:
                    print(f"{prefix} ERROR: metrics.json schema_version != {SCHEMA_VERSION}")
                    errors += 1

                for section in ["overall", "per_py", "per_class", "stability"]:
                    if section not in metrics:
                        print(f"{prefix} ERROR: metrics.json missing section '{section}'")
                        errors += 1
                    else:
                        for q in QUARTERS:
                            if q not in metrics[section]:
                                print(f"{prefix} ERROR: metrics.json {section} missing quarter '{q}'")
                                errors += 1

                # Field type checks
                if "overall" in metrics:
                    for q in QUARTERS:
                        qm = metrics["overall"].get(q, {})
                        for field in ["mae", "bias", "dir_all", "coverage_pct"]:
                            val = qm.get(field)
                            if val is not None and not isinstance(val, (int, float)):
                                print(
                                    f"{prefix} ERROR: metrics.json overall.{q}.{field} "
                                    f"is not numeric: {type(val).__name__}"
                                )
                                errors += 1

                # matched coherence
                if "matched" in metrics:
                    ca = metrics["matched"].get("compared_against")
                    if ca:
                        ca_dir = part_dir / ca
                        if not ca_dir.exists():
                            print(f"{prefix} WARN: matched.compared_against '{ca}' not found")
                            warns += 1

        # NOTES.md
        if not (vdir / "NOTES.md").exists():
            print(f"{prefix} WARN: NOTES.md missing")
            warns += 1

    # promoted.json coherence
    promoted_path = part_dir / "promoted.json"
    if promoted_path.exists():
        try:
            with open(promoted_path) as f:
                promoted = json.load(f)
            pv = promoted.get("version")
            if pv:
                pv_dir = part_dir / pv
                if not pv_dir.exists():
                    print(f"  [promoted.json] ERROR: version '{pv}' directory not found")
                    errors += 1
                elif not (pv_dir / "metrics.json").exists():
                    print(f"  [promoted.json] ERROR: version '{pv}' has no metrics.json")
                    errors += 1
        except json.JSONDecodeError as e:
            print(f"  [promoted.json] ERROR: invalid JSON: {e}")
            errors += 1

    print(f"\nValidation: {len(version_dirs)} versions, {errors} error(s), {warns} warning(s)")
    if errors == 0:
        print("OK")
    return 1 if errors > 0 else 0


# ─── Section 5: CLI ───────────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline/pipeline.py <command> [args]")
        print("Commands: create, list, compare, promote, validate")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "create":
        if len(sys.argv) < 5:
            print("Usage: pipeline.py create <part> <version_id> <description>")
            sys.exit(1)
        path = create_version(sys.argv[2], sys.argv[3], sys.argv[4])
        print(f"Created {path}")

    elif cmd == "list":
        if len(sys.argv) < 3:
            print("Usage: pipeline.py list <part>")
            sys.exit(1)
        versions = list_versions(sys.argv[2])
        promoted = get_promoted(sys.argv[2])
        pv = promoted["version"] if promoted else None
        for v in versions:
            marker = " <- promoted" if v == pv else ""
            print(f"  {v}{marker}")

    elif cmd == "compare":
        if len(sys.argv) < 4:
            print("Usage: pipeline.py compare <part> <candidate_id> [promoted_id]")
            sys.exit(1)
        pid = sys.argv[4] if len(sys.argv) > 4 else None
        compare(sys.argv[2], sys.argv[3], pid)

    elif cmd == "promote":
        if len(sys.argv) < 4:
            print("Usage: pipeline.py promote <part> <version_id> [--force]")
            sys.exit(1)
        promote(sys.argv[2], sys.argv[3], force="--force" in sys.argv)

    elif cmd == "validate":
        if len(sys.argv) < 3:
            print("Usage: pipeline.py validate <part>")
            sys.exit(1)
        sys.exit(validate(sys.argv[2]))

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
