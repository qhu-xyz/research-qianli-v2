"""Grid search over V6.1 formula weights + alternative baselines.

Sweeps alpha/beta in:
  rank_ori = alpha * da_rank_value + beta * density_mix_rank_value + (1-alpha-beta) * density_ori_rank_value

Also evaluates standalone baselines:
  - raw_shadow_price_da: raw shadow_price_da column (higher = more binding)
  - da_x_branch_max: shadow_price_da * mean_branch_max product

Saves best formula weights as v0b, raw shadow_price_da as v0c.
"""
import json
import gc
import time
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import DEFAULT_EVAL_GROUPS
from ml.data_loader import load_v61_enriched
from ml.evaluate import evaluate_ltr, aggregate_months
from ml.ground_truth import get_ground_truth

REGISTRY_DIR = _PROJECT_ROOT / "registry"

# Key metrics to display
DISPLAY_METRICS = ["VC@20", "VC@100", "Recall@20", "Recall@100", "NDCG", "Spearman"]


def load_all_groups() -> dict[str, pl.DataFrame]:
    """Load all 12 dev eval groups from cache."""
    groups = {}
    for group_id in DEFAULT_EVAL_GROUPS:
        planning_year, aq_round = group_id.split("/")
        v61 = load_v61_enriched(planning_year, aq_round)
        v61 = get_ground_truth(planning_year, aq_round, v61, cache=True)
        groups[group_id] = v61
    return groups


def evaluate_variant(
    groups: dict[str, pl.DataFrame],
    score_fn,
    name: str,
) -> dict:
    """Evaluate a scoring function across all groups."""
    per_month = {}
    for group_id, df in groups.items():
        actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
        scores = score_fn(df)
        metrics = evaluate_ltr(actual, scores)
        per_month[group_id] = metrics

    agg = aggregate_months(per_month)
    return {
        "name": name,
        "per_month": per_month,
        "aggregate": agg,
        "n_months": len(per_month),
    }


def formula_score_fn(df: pl.DataFrame, alpha: float, beta: float) -> np.ndarray:
    """Compute formula rank and return inverted score (higher = more binding)."""
    gamma = 1.0 - alpha - beta
    da_rank = df["da_rank_value"].to_numpy().astype(np.float64)
    mix_rank = df["density_mix_rank_value"].to_numpy().astype(np.float64)
    ori_rank = df["density_ori_rank_value"].to_numpy().astype(np.float64)
    rank_ori = alpha * da_rank + beta * mix_rank + gamma * ori_rank
    return 1.0 - rank_ori  # lower rank = more binding -> invert


def run_grid_search(groups: dict[str, pl.DataFrame]) -> list[dict]:
    """Sweep alpha/beta grid and return sorted results."""
    results = []
    alphas = np.arange(0.50, 1.01, 0.05)
    betas = np.arange(0.00, 0.41, 0.05)

    total = sum(1 for a in alphas for b in betas if a + b <= 1.0)
    print(f"\n[grid] Sweeping {total} alpha/beta combinations...")

    for alpha in alphas:
        for beta in betas:
            alpha = round(alpha, 2)
            beta = round(beta, 2)
            gamma = round(1.0 - alpha - beta, 2)
            if gamma < -0.001:
                continue

            name = f"formula_a{alpha:.2f}_b{beta:.2f}_g{gamma:.2f}"
            result = evaluate_variant(
                groups,
                lambda df, a=alpha, b=beta: formula_score_fn(df, a, b),
                name,
            )
            result["alpha"] = alpha
            result["beta"] = beta
            result["gamma"] = gamma
            results.append(result)

    # Sort by mean VC@20 descending
    results.sort(key=lambda r: r["aggregate"]["mean"]["VC@20"], reverse=True)
    return results


def run_alternative_baselines(groups: dict[str, pl.DataFrame]) -> list[dict]:
    """Evaluate alternative baselines."""
    results = []

    # 1. Raw shadow_price_da (higher = more binding)
    result = evaluate_variant(
        groups,
        lambda df: df["shadow_price_da"].to_numpy().astype(np.float64),
        "raw_shadow_price_da",
    )
    results.append(result)

    # 2. shadow_price_da * mean_branch_max
    result = evaluate_variant(
        groups,
        lambda df: (
            df["shadow_price_da"].to_numpy().astype(np.float64)
            * df["mean_branch_max"].to_numpy().astype(np.float64)
        ),
        "da_x_branch_max",
    )
    results.append(result)

    return results


def print_results_table(results: list[dict], title: str, top_n: int = 25):
    """Print top-N results as a formatted table."""
    print(f"\n{'=' * 120}")
    print(f"  {title} (top {min(top_n, len(results))} by mean VC@20)")
    print(f"{'=' * 120}")

    header = f"{'Rank':>4}  {'Name':<40}"
    for m in DISPLAY_METRICS:
        header += f"  {m:>10}"
    print(header)
    print("-" * 120)

    for i, r in enumerate(results[:top_n]):
        means = r["aggregate"]["mean"]
        row = f"{i+1:>4}  {r['name']:<40}"
        for m in DISPLAY_METRICS:
            row += f"  {means.get(m, 0.0):>10.4f}"
        print(row)


def save_registry_version(
    version_name: str,
    result: dict,
    config: dict,
):
    """Save metrics.json and config.json to registry."""
    version_dir = REGISTRY_DIR / version_name
    version_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "eval_config": {
            "eval_groups": DEFAULT_EVAL_GROUPS,
            "mode": "eval",
        },
        "per_month": result["per_month"],
        "aggregate": result["aggregate"],
        "n_months": result["n_months"],
        "n_months_requested": len(DEFAULT_EVAL_GROUPS),
        "skipped_months": [],
    }

    with open(version_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(version_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"[save] Saved {version_name} -> {version_dir}")


def main():
    t0 = time.time()

    # Load data
    print("[main] Loading all 12 dev eval groups from cache...")
    t_load = time.time()
    groups = load_all_groups()
    print(f"[main] Loaded {len(groups)} groups in {time.time() - t_load:.1f}s")

    # Grid search formula weights
    t_grid = time.time()
    grid_results = run_grid_search(groups)
    print(f"[main] Grid search done in {time.time() - t_grid:.1f}s")

    # Alternative baselines
    t_alt = time.time()
    alt_results = run_alternative_baselines(groups)
    print(f"[main] Alternative baselines done in {time.time() - t_alt:.1f}s")

    # Print grid results
    print_results_table(grid_results, "FORMULA WEIGHT GRID SEARCH")

    # Print v0 reference (alpha=0.60, beta=0.30)
    v0_match = [r for r in grid_results if abs(r["alpha"] - 0.60) < 0.001 and abs(r["beta"] - 0.30) < 0.001]
    if v0_match:
        v0_means = v0_match[0]["aggregate"]["mean"]
        print(f"\n  >>> v0 reference (a=0.60, b=0.30, g=0.10): VC@20={v0_means['VC@20']:.4f}")

    # Print alternative baselines
    print(f"\n{'=' * 120}")
    print("  ALTERNATIVE BASELINES")
    print(f"{'=' * 120}")
    header = f"{'Name':<40}"
    for m in DISPLAY_METRICS:
        header += f"  {m:>10}"
    print(header)
    print("-" * 120)
    for r in alt_results:
        means = r["aggregate"]["mean"]
        row = f"{r['name']:<40}"
        for m in DISPLAY_METRICS:
            row += f"  {means.get(m, 0.0):>10.4f}"
        print(row)

    # Best formula
    best = grid_results[0]
    print(f"\n[result] Best formula: alpha={best['alpha']:.2f}, beta={best['beta']:.2f}, gamma={best['gamma']:.2f}")
    print(f"[result] Best mean VC@20: {best['aggregate']['mean']['VC@20']:.4f}")

    # Compare with v0
    if v0_match:
        v0_vc20 = v0_match[0]["aggregate"]["mean"]["VC@20"]
        best_vc20 = best["aggregate"]["mean"]["VC@20"]
        delta_pct = 100 * (best_vc20 - v0_vc20) / v0_vc20 if v0_vc20 > 0 else 0
        print(f"[result] vs v0 (a=0.60, b=0.30): {delta_pct:+.1f}%")

    # Save v0b: best formula weights
    save_registry_version(
        "v0b",
        best,
        {
            "formula": f"rank_ori = {best['alpha']:.2f}*da_rank_value + {best['beta']:.2f}*density_mix_rank_value + {best['gamma']:.2f}*density_ori_rank_value",
            "alpha": best["alpha"],
            "beta": best["beta"],
            "gamma": best["gamma"],
            "score": "1 - rank_ori (inverted so higher = more binding)",
            "note": "Best formula weights from grid search over alpha/beta.",
        },
    )

    # Save v0c: raw shadow_price_da
    raw_da = [r for r in alt_results if r["name"] == "raw_shadow_price_da"][0]
    save_registry_version(
        "v0c",
        raw_da,
        {
            "score": "shadow_price_da (raw, higher = more binding)",
            "note": "Standalone historical DA shadow price as ranking signal.",
        },
    )

    # Save full grid results summary
    grid_dir = REGISTRY_DIR / "v0b_grid_search"
    grid_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for r in grid_results:
        entry = {
            "name": r["name"],
            "alpha": r.get("alpha"),
            "beta": r.get("beta"),
            "gamma": r.get("gamma"),
        }
        for m in DISPLAY_METRICS:
            entry[f"mean_{m}"] = r["aggregate"]["mean"].get(m)
            entry[f"bottom2_{m}"] = r["aggregate"]["bottom_2_mean"].get(m)
        summary.append(entry)
    # Also add alt baselines
    for r in alt_results:
        entry = {"name": r["name"], "alpha": None, "beta": None, "gamma": None}
        for m in DISPLAY_METRICS:
            entry[f"mean_{m}"] = r["aggregate"]["mean"].get(m)
            entry[f"bottom2_{m}"] = r["aggregate"]["bottom_2_mean"].get(m)
        summary.append(entry)

    with open(grid_dir / "all_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] Grid summary -> {grid_dir / 'all_results.json'}")

    # Per-group comparison: best formula vs v0 vs raw_da
    print(f"\n{'=' * 120}")
    print("  PER-GROUP COMPARISON: v0 vs best_formula vs raw_shadow_price_da (VC@20)")
    print(f"{'=' * 120}")
    print(f"{'Group':<20}  {'v0':>10}  {'best':>10}  {'raw_da':>10}")
    print("-" * 60)
    if v0_match:
        for gid in DEFAULT_EVAL_GROUPS:
            v0_val = v0_match[0]["per_month"][gid]["VC@20"]
            best_val = best["per_month"][gid]["VC@20"]
            raw_val = raw_da["per_month"][gid]["VC@20"]
            print(f"{gid:<20}  {v0_val:>10.4f}  {best_val:>10.4f}  {raw_val:>10.4f}")

    total_time = time.time() - t0
    print(f"\n[main] Total walltime: {total_time:.1f}s")

    gc.collect()


if __name__ == "__main__":
    main()
