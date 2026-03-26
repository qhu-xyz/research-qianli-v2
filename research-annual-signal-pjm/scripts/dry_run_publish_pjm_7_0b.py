from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.markets.pjm.signal_publisher import publish_signal


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run one PJM 7.0b publication cell.")
    parser.add_argument("--planning-year", default="2024-06")
    parser.add_argument("--round", type=int, default=2)
    parser.add_argument("--class-type", default="onpeak")
    args = parser.parse_args()

    constraints_df, sf_df = publish_signal(
        planning_year=args.planning_year,
        market_round=args.round,
        class_type=args.class_type,
    )

    summary = {
        "planning_year": args.planning_year,
        "round": args.round,
        "class_type": args.class_type,
        "constraint_rows": int(constraints_df.shape[0]),
        "constraint_columns": list(constraints_df.columns),
        "sf_shape": [int(sf_df.shape[0]), int(sf_df.shape[1])],
        "sf_index_name": sf_df.index.name,
        "tier_counts": {str(k): int(v) for k, v in constraints_df["tier"].value_counts().sort_index().to_dict().items()},
        "first_constraint_keys": constraints_df.index[:5].tolist(),
    }

    out_dir = Path("releases/pjm/annual/7.0b")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dry_run_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
