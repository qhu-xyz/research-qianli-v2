# Baseline Recalibration + Bug Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 3 audit issues, grid-search optimal formula weights for v0b baseline, and re-evaluate ML versions against the stronger baseline.

**Architecture:** All work uses cached data (no Ray needed). Bug fixes are small edits to evaluate.py and ground_truth.py. Grid search sweeps alpha/beta weights and evaluates each combo across 12 dev groups. If v0b is materially better, re-calibrate gates and re-check ML gate results.

**Tech Stack:** Python, polars, numpy, existing ml/ modules

**Key context:**
- v6_exploration already shows `da_raw` (raw shadow_price_da) beats v0 by 28.6% on VC@20
- Current v0: `rank_ori = 0.60*da_rank + 0.30*dmix + 0.10*dori`, VC@20=0.2329
- All data cached in `cache/enriched/` and `cache/ground_truth/`
- Working dir: `/home/xyz/workspace/research-qianli-v2/research-annual-signal`
- Activate venv: `cd /home/xyz/workspace/pmodel && source .venv/bin/activate`

---

### Task 1: Fix Recall@100 tie-breaking in evaluate.py

**Files:**
- Modify: `ml/evaluate.py:26-31`

**Step 1: Fix recall_at_k to cap true set to positive-value rows only**

In `ml/evaluate.py`, replace the `recall_at_k` function:

```python
def recall_at_k(actual: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Of the true top-k by actual value, how many are in model's top-k.

    When fewer than k rows have positive actual value, the true set is
    capped to only positive rows (avoids arbitrary tie-breaking among zeros).
    """
    k = min(k, len(scores))
    sorted_idx = np.argsort(actual)[::-1]
    # Cap true set to rows with positive value
    n_positive = int((actual > 0).sum())
    effective_k = min(k, n_positive)
    if effective_k == 0:
        return 0.0
    true_top_k = set(sorted_idx[:effective_k].tolist())
    pred_top_k = set(np.argsort(scores)[::-1][:k].tolist())
    return len(true_top_k & pred_top_k) / k if k > 0 else 0.0
```

**Step 2: Verify fix doesn't break existing evaluation**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal && python -c "
from ml.evaluate import recall_at_k
import numpy as np
# 5 items, only 2 positive: true top-3 should be capped to 2
actual = np.array([10.0, 5.0, 0.0, 0.0, 0.0])
scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
r = recall_at_k(actual, scores, 3)
print(f'recall@3 = {r:.4f}')  # expect 2/3 = 0.6667
assert abs(r - 2/3) < 1e-6, f'Expected 0.6667, got {r}'

# All positive: behaves normally
actual2 = np.array([10.0, 5.0, 3.0, 1.0, 0.5])
scores2 = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
r2 = recall_at_k(actual2, scores2, 3)
print(f'recall@3 (all pos) = {r2:.4f}')  # expect 1.0
assert abs(r2 - 1.0) < 1e-6
print('PASS')
"`

Expected: PASS

---

### Task 2: Commit ground_truth.py fix + evaluate.py fix

**Files:**
- Already modified: `ml/ground_truth.py` (partition-filtered mapping, already in working copy)
- Modified in Task 1: `ml/evaluate.py`

**Step 1: Commit both fixes**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-annual-signal
git add ml/ground_truth.py ml/evaluate.py
git commit -m "fix: partition-filtered GT mapping + Recall@K tie-breaking

- ground_truth.py: filter MISO_SPICE_CONSTRAINT_INFO to target partition
  (auction_type, auction_month, period_type, class_type) to avoid cross-
  partition fan-out. Cache was already generated with corrected code.
- evaluate.py: cap recall_at_k true set to positive-value rows only,
  avoiding arbitrary tie-breaking among zero-valued constraints."
```

---

### Task 3: Grid search over formula weights

**Files:**
- Create: `scripts/run_baseline_grid_search.py`

**Step 1: Write the grid search script**

This script:
1. Loads all 12 dev eval groups from cache
2. For each (alpha, beta) combo where alpha + beta <= 1.0:
   - Computes `score = alpha * shadow_price_da_rank + beta * density_mix_rank + (1-alpha-beta) * density_ori_rank`
   - Note: V6.1 uses `da_rank_value` (lower = more binding), so score = 1 - rank_ori for eval
   - Evaluates all 13 metrics per group
3. Also evaluates raw `shadow_price_da` as a standalone ranker
4. Also evaluates `shadow_price_da * mean_branch_max` product
5. Prints comparison table sorted by mean VC@20
6. Saves best weights + full results to `registry/v0b/`

```python
"""Grid search over V6.1 formula weights + alternative baselines.

Sweeps alpha/beta in rank_ori = alpha*da_rank + beta*dmix + (1-a-b)*dori
and compares against raw shadow_price_da and product baselines.

No Ray needed — all from cache.
"""
import json
import gc
import time
from pathlib import Path
from itertools import product

import numpy as np
import polars as pl

from ml.config import DEFAULT_EVAL_GROUPS
from ml.data_loader import load_v61_enriched
from ml.evaluate import evaluate_ltr, aggregate_months
from ml.ground_truth import get_ground_truth


REGISTRY_DIR = Path(__file__).resolve().parent.parent / "registry"


def eval_formula(groups_data: dict, alpha: float, beta: float) -> dict:
    """Evaluate formula: score = alpha*da_rank + beta*dmix + (1-a-b)*dori."""
    gamma = 1.0 - alpha - beta
    per_group = {}
    for group_id, (v61, actual) in groups_data.items():
        da_rank = v61["da_rank_value"].to_numpy().astype(np.float64)
        dmix = v61["density_mix_rank_value"].to_numpy().astype(np.float64)
        dori = v61["density_ori_rank_value"].to_numpy().astype(np.float64)
        rank_ori = alpha * da_rank + beta * dmix + gamma * dori
        # Lower rank_ori = more binding, so invert for eval
        scores = 1.0 - rank_ori
        per_group[group_id] = evaluate_ltr(actual, scores)
    return per_group


def eval_raw_da(groups_data: dict) -> dict:
    """Evaluate raw shadow_price_da as ranker (higher = more binding)."""
    per_group = {}
    for group_id, (v61, actual) in groups_data.items():
        scores = v61["shadow_price_da"].to_numpy().astype(np.float64)
        per_group[group_id] = evaluate_ltr(actual, scores)
    return per_group


def eval_product(groups_data: dict) -> dict:
    """Evaluate shadow_price_da * mean_branch_max."""
    per_group = {}
    for group_id, (v61, actual) in groups_data.items():
        da = v61["shadow_price_da"].to_numpy().astype(np.float64)
        mbm = v61["mean_branch_max"].to_numpy().astype(np.float64)
        scores = da * mbm
        per_group[group_id] = evaluate_ltr(actual, scores)
    return per_group


def main():
    t0 = time.time()

    # Load all eval groups
    groups_data = {}
    for group_id in DEFAULT_EVAL_GROUPS:
        planning_year, aq_round = group_id.split("/")
        v61 = load_v61_enriched(planning_year, aq_round)
        v61 = get_ground_truth(planning_year, aq_round, v61, cache=True)
        actual = v61["realized_shadow_price"].to_numpy().astype(np.float64)
        groups_data[group_id] = (v61, actual)
    print(f"Loaded {len(groups_data)} groups in {time.time()-t0:.1f}s\n")

    # --- Grid search ---
    alphas = [round(a, 2) for a in np.arange(0.50, 1.01, 0.05)]
    betas = [round(b, 2) for b in np.arange(0.00, 0.45, 0.05)]

    results = []

    for alpha, beta in product(alphas, betas):
        if alpha + beta > 1.0 + 1e-9:
            continue
        gamma = round(1.0 - alpha - beta, 2)
        per_group = eval_formula(groups_data, alpha, beta)
        agg = aggregate_months(per_group)
        vc20 = agg["mean"]["VC@20"]
        results.append({
            "name": f"a={alpha:.2f}_b={beta:.2f}_g={gamma:.2f}",
            "alpha": alpha, "beta": beta, "gamma": gamma,
            "type": "formula",
            "per_group": per_group,
            "agg": agg,
            "vc20": vc20,
        })

    # --- Alternative baselines ---
    # Raw shadow_price_da
    per_group = eval_raw_da(groups_data)
    agg = aggregate_months(per_group)
    results.append({
        "name": "raw_shadow_price_da",
        "type": "raw_da",
        "per_group": per_group,
        "agg": agg,
        "vc20": agg["mean"]["VC@20"],
    })

    # Product: da * branch_max
    per_group = eval_product(groups_data)
    agg = aggregate_months(per_group)
    results.append({
        "name": "da_x_branch_max",
        "type": "product",
        "per_group": per_group,
        "agg": agg,
        "vc20": agg["mean"]["VC@20"],
    })

    # --- Sort and report ---
    results.sort(key=lambda r: r["vc20"], reverse=True)

    print("=" * 100)
    print(f"{'Rank':>4}  {'Name':>30}  {'VC@20':>8}  {'VC@100':>8}  {'R@20':>8}  {'R@100':>8}  {'NDCG':>8}  {'Spearman':>8}")
    print("-" * 100)
    for i, r in enumerate(results[:25]):
        m = r["agg"]["mean"]
        print(f"{i+1:>4}  {r['name']:>30}  {m['VC@20']:.4f}  {m['VC@100']:.4f}  {m['Recall@20']:.4f}  {m['Recall@100']:.4f}  {m['NDCG']:.4f}  {m['Spearman']:.4f}")

    # Current v0 for reference
    v0_result = next((r for r in results if r.get("alpha") == 0.60 and r.get("beta") == 0.30), None)
    if v0_result:
        print(f"\nCurrent v0 (a=0.60, b=0.30, g=0.10): VC@20={v0_result['vc20']:.4f}")

    best = results[0]
    print(f"\nBest overall: {best['name']}, VC@20={best['vc20']:.4f}")
    if v0_result:
        pct = 100 * (best["vc20"] - v0_result["vc20"]) / v0_result["vc20"]
        print(f"Improvement vs v0: {pct:+.1f}%")

    # --- Save best formula as v0b ---
    best_formula = next((r for r in results if r["type"] == "formula"), None)
    if best_formula:
        v0b_dir = REGISTRY_DIR / "v0b"
        v0b_dir.mkdir(parents=True, exist_ok=True)

        metrics_out = {
            "eval_config": {"eval_groups": DEFAULT_EVAL_GROUPS, "mode": "eval"},
            "per_month": best_formula["per_group"],
            "aggregate": best_formula["agg"],
            "n_months": len(best_formula["per_group"]),
            "n_months_requested": len(DEFAULT_EVAL_GROUPS),
            "skipped_months": [],
        }
        with open(v0b_dir / "metrics.json", "w") as f:
            json.dump(metrics_out, f, indent=2)

        config_out = {
            "formula": f"rank_ori = {best_formula['alpha']:.2f}*da_rank_value + {best_formula['beta']:.2f}*density_mix_rank_value + {best_formula['gamma']:.2f}*density_ori_rank_value",
            "alpha": best_formula["alpha"],
            "beta": best_formula["beta"],
            "gamma": best_formula["gamma"],
            "score": "1 - rank_ori (inverted so higher = more binding)",
            "note": "Grid-search optimized formula weights on 12 dev eval groups.",
        }
        with open(v0b_dir / "config.json", "w") as f:
            json.dump(config_out, f, indent=2)

        print(f"\nSaved v0b to {v0b_dir}")
        print(f"  Formula: {config_out['formula']}")

    # --- Save raw_da as v0c ---
    raw_da = next((r for r in results if r["type"] == "raw_da"), None)
    if raw_da:
        v0c_dir = REGISTRY_DIR / "v0c"
        v0c_dir.mkdir(parents=True, exist_ok=True)

        metrics_out = {
            "eval_config": {"eval_groups": DEFAULT_EVAL_GROUPS, "mode": "eval"},
            "per_month": raw_da["per_group"],
            "aggregate": raw_da["agg"],
            "n_months": len(raw_da["per_group"]),
            "n_months_requested": len(DEFAULT_EVAL_GROUPS),
            "skipped_months": [],
        }
        with open(v0c_dir / "metrics.json", "w") as f:
            json.dump(metrics_out, f, indent=2)

        config_out = {
            "formula": "raw shadow_price_da (higher = more binding)",
            "note": "Uses raw historical DA shadow price as standalone ranker.",
        }
        with open(v0c_dir / "config.json", "w") as f:
            json.dump(config_out, f, indent=2)

        print(f"Saved v0c (raw_da) to {v0c_dir}")

    # --- Full results JSON ---
    grid_dir = REGISTRY_DIR / "v0b_grid_search"
    grid_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for r in results:
        entry = {
            "name": r["name"], "type": r["type"], "vc20": r["vc20"],
            "agg_mean": {k: round(v, 4) for k, v in r["agg"]["mean"].items() if isinstance(v, float)},
        }
        if r["type"] == "formula":
            entry["alpha"] = r["alpha"]
            entry["beta"] = r["beta"]
            entry["gamma"] = r["gamma"]
        summary.append(entry)

    with open(grid_dir / "all_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved full grid results to {grid_dir / 'all_results.json'}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
```

**Step 2: Run the grid search**

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-annual-signal
python scripts/run_baseline_grid_search.py
```

Expected: prints top-25 weight combos + saves v0b and v0c to registry. Should take <30s (all from cache, no training).

---

### Task 4: Re-run comparison with v0b in registry

**Step 1: Run the compare module**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-annual-signal
python -m ml.compare --batch-id baseline_recal --iteration 1 \
  --output reports/baseline_recalibration.md
```

This picks up v0, v0b, v0c, v1-v5 and builds a comparison table against gates.

**Step 2: Analyze results and update mem.md**

Check:
- Does v0b beat v0? By how much?
- Does v0b close the gap with v1 (ML)?
- Does raw_da (v0c) beat the formula entirely?
- Does v1 still pass all gates when v0b is used as baseline?

Update `mem.md` with:
- Best formula weights and their metrics
- Whether ML still adds value over recalibrated formula
- Raw-da standalone performance

---

### Task 5: Commit grid search results

**Step 1: Commit the new script + results**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-annual-signal
git add scripts/run_baseline_grid_search.py registry/v0b/ registry/v0c/ \
  registry/v0b_grid_search/ reports/baseline_recalibration.md
git commit -m "baseline recalibration: grid search over formula weights

- Grid search alpha/beta weights for rank_ori formula
- v0b: best formula weights (registered)
- v0c: raw shadow_price_da standalone baseline
- Comparison report against all versions"
```

---

### Task 6: Re-calibrate gates from v0b and re-check ML (conditional)

**Only if v0b is materially better than v0 (>5% VC@20 improvement).**

If v0b is indeed much stronger, we need to:
1. Re-calibrate gates from v0b (replace gates.json)
2. Re-check if v1-v5 still pass gates against the higher baseline
3. Update champion.json if appropriate

This can be done by modifying `run_v0_baseline.py` to accept custom formula weights, or by adding a `--recalibrate-from v0b` flag to compare.py.

**Step 1: If needed, recalibrate gates**

```python
# Quick inline recalibration:
python -c "
import json
from pathlib import Path
from ml.config import DEFAULT_EVAL_GROUPS

v0b = json.loads(Path('registry/v0b/metrics.json').read_text())
agg = v0b['aggregate']

blocking = ['VC@20', 'VC@100', 'Recall@20', 'Recall@50', 'Recall@100', 'NDCG']
monitor = ['Spearman', 'Tier0-AP', 'Tier01-AP']
gates = {}
for m in blocking + monitor:
    mean_val = agg['mean'].get(m)
    min_val = agg['min'].get(m)
    if mean_val is None: continue
    gates[m] = {
        'floor': round(0.9 * mean_val, 4),
        'tail_floor': round(min_val, 4) if min_val else None,
        'direction': 'higher',
        'group': 'A' if m in blocking else 'B',
    }
data = {'gates': gates, 'noise_tolerance': 0.02, 'tail_max_failures': 1, 'calibrated_from': 'v0b'}
Path('registry/gates_v0b.json').write_text(json.dumps(data, indent=2))
print('Saved gates_v0b.json')
"
```

**Step 2: Compare ML versions against v0b gates**

```bash
python -m ml.compare --batch-id baseline_recal --iteration 2 \
  --gates-path registry/gates_v0b.json \
  --output reports/baseline_recalibration_v0b_gates.md
```

**Step 3: Update mem.md with final assessment**

---

### Task 7: Final commit + mem.md update

**Step 1: Update mem.md with grid search findings**

Add a section with:
- Best formula weights
- raw_da vs formula vs ML comparison
- Whether recalibrated baseline changes the ML narrative

**Step 2: Final commit**

```bash
git add mem.md registry/gates_v0b.json reports/
git commit -m "baseline recalibration: update mem + gates from v0b"
```
