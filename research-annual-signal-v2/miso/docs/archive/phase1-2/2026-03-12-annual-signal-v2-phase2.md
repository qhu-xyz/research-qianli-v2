# Annual Signal v2 — Phase 2 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan.

**Goal:** Train LightGBM LambdaRank models and find the best feature set that beats v0c baseline.

**Architecture:** Expanding-window LambdaRank with tiered labels (0/1/2/3). Each eval PY trains on all prior PYs. Feature groups added incrementally: history → density → limits → metadata. Pruning at the end.

**Tech Stack:** LightGBM, Polars, pytest

---

## Holdout Policy (Plan-Wide)

All model-selection and keep/drop decisions use **dev metrics only**. Holdout is reported for monitoring but NEVER used for go/stop decisions. This applies to every task below.

## Hurdle

v0c (authoritative baseline from `registry/baseline_contract.json`):
- Dev: VC@50=0.3382, NDCG=0.8457
- Holdout: VC@50=0.3547, NDCG=0.8599

---

## Milestone 1: Infrastructure (Tasks 1–2) — DONE

### Task 1: LambdaRank training module — DONE
- `ml/train.py` with `tiered_labels()`, `build_query_groups()`, `train_and_predict()`
- Returns `(scored_df, train_info)` where train_info has feature_importance, walltime, n_train_rows

### Task 2: Experiment runner — DONE
- `scripts/run_ml_experiment.py` with feature groups, importance aggregation, gate checking
- `tests/test_train.py` with 4 passing tests

---

## Milestone 2: v2a History-Only ML (Task 3)

### Task 3: Train v2a (history features only)

**Features:** HISTORY_FEATURES from `ml/config.py` (8 features: da_rank_value, bf_6, bf_12, bf_15, bfo_6, bfo_12, bf_combined_6, bf_combined_12)

- [ ] **Step 1: Run v2a experiment**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2
source /home/xyz/workspace/pmodel/.venv/bin/activate
python scripts/run_ml_experiment.py --version v2a --features history
```

- [ ] **Step 2: Analyze results**

Record per-group metrics for ALL 15 eval groups (12 dev + 3 holdout).
Report: VC@50, VC@100, Recall@50, NDCG, NB12_Recall@50, Abs_SP@50, n_binding.
Report dev mean and holdout mean separately.
Report feature importance (dev-averaged for decisions, all-split for monitoring).
Report gate check vs v0c.

- [ ] **Step 3: Decision gate (dev-only)**

**GO condition:** dev mean VC@50 > v0c dev VC@50 (0.3382).
If GO: proceed to Milestone 3.
If NO-GO: diagnose using dev feature importance and per-group breakdown, iterate on history features before adding density.

- [ ] **Step 4: Commit**

```bash
git add registry/v2a/
git commit -m "[AB#XXXX] v2a: history-only ML baseline, dev VC@50=X.XXXX"
```

**→ HALT for review after Task 3.**

---

## Milestone 3: Density Experiments (Tasks 4–5)

### Task 4: Train v2b (history + density_core)

**Features:** HISTORY_FEATURES + density_core (5 bins: bin_80_cid_max, bin_100_cid_max, bin_110_cid_max, bin_120_cid_max, bin_150_cid_max)

- [ ] **Step 1: Run v2b experiment**

```bash
python scripts/run_ml_experiment.py --version v2b --features history,density_core
```

- [ ] **Step 2: Record full results**

Same reporting as Task 3.

- [ ] **Step 3: Analyze density contribution (dev-only)**

Compare v2b vs v2a dev means:
- If density_core bins collectively have dev-averaged importance > 10%: density is contributing.
- If density_core bins collectively have dev-averaged importance < 2%: density is dead weight.
- Record per-bin importance to see which bins matter.

- [ ] **Step 4: Commit**

```bash
git add registry/v2b/
git commit -m "[AB#XXXX] v2b: history+density_core, dev VC@50=X.XXXX"
```

### Task 5: Diagnostic comparison table

- [ ] **Step 1: Create comparison table**

Build a side-by-side table of v0a, v0b, v0c, v2a, v2b with:
- Dev mean: VC@50, VC@100, Recall@50, NDCG, NB12_Recall@50, Abs_SP@50
- Holdout mean (for monitoring only, NOT for decisions)
- Feature count
- Dev-averaged feature importance for v2a and v2b (top features)

- [ ] **Step 2: Assess density signal**

Using dev metrics only:
- Is v2b dev mean VC@50 > v2a dev mean VC@50? By how much?
- Which density bins have the highest dev-averaged importance?
- If density helps: proceed to Milestone 4.
- If density hurts or is neutral: stop density expansion, proceed to pruning in Milestone 4.

**→ HALT for review after Task 5.**

---

## Milestone 4: Feature Expansion, Pruning & Champion (Tasks 6–9)

### Task 6: Conditional density expansion

**Only run if density_core showed value in Task 5 (dev-only assessment).**

- [ ] **Step 6a: Add density_core_std (v2c)**

density_core_std uses the same 5 bins as density_core: bin_80_cid_std, bin_100_cid_std, bin_110_cid_std, bin_120_cid_std, bin_150_cid_std.

**IMPORTANT:** These cid_std columns may not exist in the cached data. Before running:
1. Check if cid_std columns exist in `build_model_table("2024-06", "aq1")`
2. If missing: update `ml/data_loader.py` to include cid_std in Level-2 collapse, delete `data/` cache, rebuild

```bash
python scripts/run_ml_experiment.py --version v2c --features history,density_core,density_core_std
```

- [ ] **Step 6b: Add density_counter (v2d)**

```bash
python scripts/run_ml_experiment.py --version v2d --features history,density_core,density_counter
```

- [ ] **Step 6c: Add limits (v2e)**

```bash
python scripts/run_ml_experiment.py --version v2e --features history,density_core,limits
```

- [ ] **Step 6d: Kitchen-sink (v2f)**

All groups: history, density_core, density_core_std, density_counter, limits, metadata.

```bash
python scripts/run_ml_experiment.py --version v2f --features history,density_core,density_core_std,density_counter,limits,metadata
```

- [ ] **Step 6e: Compare and select best pre-pruning version**

Using dev metrics only: compare v2c/v2d/v2e/v2f. Pick the version with best dev VC@50 as the pruning candidate.

- [ ] **Step 6f: Commit all results**

```bash
git add registry/v2c/ registry/v2d/ registry/v2e/ registry/v2f/
git commit -m "[AB#XXXX] density expansion experiments v2c-v2f"
```

### Task 7: Skip if density was dead in Task 5

If density was dead weight, the pruning candidate is v2a. Skip Task 6 entirely.

### Task 8: Monotone constraints experiment

- [ ] **Step 1: Run monotone version of best candidate**

```bash
python scripts/run_ml_experiment.py --version v2X_mono --features <best_features> --monotone
```

- [ ] **Step 2: Compare vs non-monotone (dev-only)**

If monotone dev VC@50 >= non-monotone: keep monotone.
Otherwise: drop monotone.

### Task 9: Feature pruning

- [ ] **Step 1: Identify pruning candidates**

From the best version's dev-averaged feature importance:
- Features with dev-averaged importance < 2%: candidates for removal.
- Features with dev-averaged importance > 10%: safe, keep.

- [ ] **Step 2: Run pruned version**

Use `--custom-features` to specify exact columns (minus pruned features):

```bash
python scripts/run_ml_experiment.py --version v2_pruned --custom-features feat1,feat2,...
```

- [ ] **Step 3: Compare pruned vs unpruned (dev-only)**

If pruned dev VC@50 >= unpruned (within 0.005): keep pruned (simpler is better).
If pruned dev VC@50 drops > 0.005: keep unpruned.

- [ ] **Step 4: Declare champion**

The winner of pruning comparison becomes the champion candidate.
Run final gate check vs v0c using BOTH dev and holdout.
Save as `v2_champion`.

```bash
git add registry/v2_pruned/ registry/v2_champion/ registry/v2X_mono/
git commit -m "[AB#XXXX] v2 champion: <features>, dev VC@50=X.XXXX, holdout VC@50=X.XXXX"
```

**→ HALT for review after Task 9.**

---

## Decision Checkpoints Summary

| Checkpoint | Decision | Metric | Scope |
|---|---|---|---|
| After Task 3 | GO/NO-GO on ML | dev mean VC@50 > 0.3382 | dev only |
| After Task 5 | Density helps? | dev-averaged importance + dev mean delta | dev only |
| After Task 6e | Best pre-pruning version | dev mean VC@50 | dev only |
| After Task 8 | Keep monotone? | dev mean VC@50 | dev only |
| After Task 9 | Champion selection | dev mean VC@50 + holdout gate | dev for selection, holdout for confirmation |

## Infrastructure Notes

- **Model table scope:** 27 groups = 7 PYs (2019-06..2025-06) × 4 quarters (aq1..aq4) minus 2025-06/aq4 (incomplete). This includes training-only PYs (2019-06 through 2021-06) that are NOT in DEV_GROUPS or HOLDOUT_GROUPS.
- **Eval groups:** 15 total = 12 DEV_GROUPS + 3 HOLDOUT_GROUPS. The runner filters to these after prediction.
- **Feature importance:** train-row-weighted average across expanding-window splits. Dev-averaged (dev splits only) for all decisions; all-split averaged for monitoring.
- **Cache:** `data/` directory. Delete if data_loader changes (e.g. adding cid_std columns).
