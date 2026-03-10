"""v0: True V6.2B formula baseline (full formula including da_rank_value).

V6.2B formula:
  score = 0.60 * da_rank_value + 0.30 * density_mix_rank_value + 0.10 * density_ori_rank_value
  rank  = dense_rank(score) / max(dense_rank(score))

da_rank_value is a historical DA shadow price rank (NOT leakage — it's a persistent
cross-month signal with ~65-70% overlap between adjacent months).

This script evaluates the formula ranking against shadow_price_da for each eval month
and saves as the v0 benchmark in the registry.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import V62B_SIGNAL_BASE, _DEFAULT_EVAL_MONTHS
from ml.evaluate import aggregate_months, evaluate_ltr
from ml.v62b_formula import v62b_score


def evaluate_formula_month(auction_month: str) -> dict:
    """Evaluate full V6.2B formula ranking for one month."""
    path = Path(V62B_SIGNAL_BASE) / auction_month / "f0" / "onpeak"
    df = pl.read_parquet(str(path))

    actual = df["shadow_price_da"].to_numpy().astype(np.float64)
    # V6.2B rank_value columns: lower = more binding.
    # Formula score: lower = more binding. Negate so higher = more binding
    # for evaluate_ltr which expects higher score = better.
    scores = -v62b_score(
        da_rank_value=df["da_rank_value"].to_numpy(),
        density_mix_rank_value=df["density_mix_rank_value"].to_numpy(),
        density_ori_rank_value=df["density_ori_rank_value"].to_numpy(),
    )

    return evaluate_ltr(actual, scores)


def main():
    eval_months = list(_DEFAULT_EVAL_MONTHS)
    per_month = {}
    for m in eval_months:
        print(f"Evaluating V6.2B formula on {m}...")
        per_month[m] = evaluate_formula_month(m)

    agg = aggregate_months(per_month)
    result = {
        "eval_config": {
            "eval_months": eval_months,
            "class_type": "onpeak",
            "period_type": "f0",
            "model": "v62b_formula",
            "note": "Full V6.2B formula: 0.60*da_rank + 0.30*dmix + 0.10*dori",
        },
        "per_month": per_month,
        "aggregate": agg,
        "n_months": len(per_month),
        "n_months_requested": len(eval_months),
        "skipped_months": [],
    }

    # Save to registry
    v0_dir = Path("registry/v0")
    v0_dir.mkdir(parents=True, exist_ok=True)
    with open(v0_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(v0_dir / "config.json", "w") as f:
        json.dump({
            "model": "v62b_formula",
            "note": "Full V6.2B formula (da_rank_value is historical, NOT leakage)",
            "formula": "score = 0.60*da_rank_value + 0.30*density_mix_rank_value + 0.10*density_ori_rank_value",
        }, f, indent=2)
    with open(v0_dir / "meta.json", "w") as f:
        json.dump({"version_id": "v0", "model": "v62b_formula"}, f, indent=2)

    # Recalibrate gates from full formula baseline
    gates = {}
    group_a = ["VC@20", "VC@100", "Recall@20", "Recall@50", "Recall@100", "NDCG"]
    group_b = ["VC@10", "VC@25", "VC@50", "VC@200",
               "Recall@10", "Spearman", "Tier0-AP", "Tier01-AP"]

    for metric in group_a + group_b:
        group = "A" if metric in group_a else "B"
        mean_val = agg["mean"].get(metric, 0)
        min_val = agg["min"].get(metric, 0)
        gates[metric] = {
            "floor": round(0.9 * mean_val, 6),
            "tail_floor": round(min_val, 6),
            "direction": "higher",
            "group": group,
        }

    gates_data = {
        "version": 2,
        "note": "Calibrated from full V6.2B formula baseline (da_rank_value included)",
        "noise_tolerance": 0.02,
        "tail_max_failures": 1,
        "eval_months": {"primary": eval_months},
        "gates": gates,
    }

    registry_dir = Path("registry")
    with open(registry_dir / "gates.json", "w") as f:
        json.dump(gates_data, f, indent=2)
    with open(registry_dir / "champion.json", "w") as f:
        json.dump({"version": "v0"}, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("V6.2B FORMULA BASELINE (v0)")
    print("=" * 60)
    print("Formula: score = 0.60*da_rank + 0.30*dmix + 0.10*dori")
    print()
    for metric in group_a:
        mean = agg["mean"].get(metric, 0)
        mn = agg["min"].get(metric, 0)
        mx = agg["max"].get(metric, 0)
        print(f"  {metric:>12}: mean={mean:.4f}  min={mn:.4f}  max={mx:.4f}")
    print()
    for metric in group_b:
        mean = agg["mean"].get(metric, 0)
        print(f"  {metric:>12}: mean={mean:.4f}")
    print()
    print("Gates recalibrated from formula baseline. Champion set to v0.")


if __name__ == "__main__":
    main()
