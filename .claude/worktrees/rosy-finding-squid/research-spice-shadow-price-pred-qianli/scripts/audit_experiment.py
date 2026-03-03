"""Experiment completeness audit — checks all 32 benchmark parquets.

Usage:
    python scripts/audit_experiment.py --version-id v001-threshold-f2 [--output-dir path]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_OUTPUT = "/opt/temp/tmp/pw_data/spice6/experiments"

BENCHMARK_MONTHS = {
    "PY20": [("2020-07", "Summer"), ("2020-10", "Fall"), ("2021-01", "Winter"), ("2021-04", "Spring")],
    "PY21": [("2021-07", "Summer"), ("2021-10", "Fall"), ("2022-01", "Winter"), ("2022-04", "Spring")],
}
CLASS_TYPES = ["onpeak", "offpeak"]
PERIOD_TYPES = ["f0", "f1"]

REQUIRED_COLUMNS = ["actual_binding", "predicted_binding", "predicted_shadow_price"]
CRITICAL_COLUMNS = ["actual_binding", "predicted_binding"]


def expected_parquet(auction_month: str, class_type: str, period_type: str, output_dir: str) -> Path:
    am_compact = auction_month.replace("-", "")
    return Path(output_dir) / f"results_{am_compact}_{class_type}_{period_type}.parquet"


def audit_parquet(pq_path: Path) -> dict:
    """Audit a single parquet file."""
    result = {"path": str(pq_path), "status": "PASS", "issues": [], "n_rows": 0}

    if not pq_path.exists():
        result["status"] = "MISSING"
        result["issues"].append("File not found")
        return result

    try:
        df = pd.read_parquet(pq_path)
        result["n_rows"] = len(df)

        # Row count check
        if len(df) == 0:
            result["issues"].append("Empty DataFrame (0 rows)")
            result["status"] = "FAIL"
            return result
        if len(df) > 500_000:
            result["issues"].append(f"Suspiciously large: {len(df):,} rows")

        # Schema check
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                result["issues"].append(f"Missing column: {col}")
                result["status"] = "FAIL"

        # Null check on critical columns
        for col in CRITICAL_COLUMNS:
            if col in df.columns and df[col].isna().any():
                n_null = int(df[col].isna().sum())
                result["issues"].append(f"Nulls in {col}: {n_null}")
                result["status"] = "WARN"

    except Exception as e:
        result["status"] = "FAIL"
        result["issues"].append(f"Read error: {e}")

    if result["issues"] and result["status"] == "PASS":
        result["status"] = "WARN"

    return result


def main():
    parser = argparse.ArgumentParser(description="Experiment completeness audit")
    parser.add_argument("--version-id", required=True, help="Model version ID")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    output_dir = args.output_dir or str(Path(DEFAULT_BASE_OUTPUT) / args.version_id)

    print(f"Experiment Audit: {args.version_id}")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()

    total = 0
    present = 0
    issues_count = 0

    for py, months in BENCHMARK_MONTHS.items():
        for am_str, season in months:
            for ct in CLASS_TYPES:
                for pt in PERIOD_TYPES:
                    total += 1
                    pq = expected_parquet(am_str, ct, pt, output_dir)
                    result = audit_parquet(pq)

                    if result["status"] == "MISSING":
                        pass  # Count below
                    elif result["status"] == "PASS":
                        present += 1
                    elif result["status"] == "WARN":
                        present += 1
                        issues_count += 1
                    else:
                        present += 1
                        issues_count += 1

                    tag = f"{am_str}/{ct}/{pt}"
                    if result["status"] == "PASS":
                        detail = f"{result['n_rows']:,} rows, schema OK"
                    elif result["status"] == "MISSING":
                        detail = "NOT FOUND"
                    else:
                        detail = f"{result['n_rows']:,} rows, {'; '.join(result['issues'])}"

                    status_char = result["status"]
                    print(f"  {tag:30s} ... {status_char:7s} ({detail})")

    missing = total - present
    print()
    print(f"Parquets: {present}/{total} present", end="")
    if missing:
        print(f" ({missing} missing)")
    else:
        print()

    if missing == 0 and issues_count == 0:
        print(f"\nOverall: PASS ({total}/{total} parquets valid)")
    elif missing > 0:
        print(f"\nOverall: INCOMPLETE ({missing} parquets missing)")
        sys.exit(1)
    else:
        print(f"\nOverall: WARN ({issues_count} issues found)")


if __name__ == "__main__":
    main()
