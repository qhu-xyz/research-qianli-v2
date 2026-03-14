# Signal Quality Redesign: Beyond VC@20 + New Constraint Detection

**Date**: 2026-03-11
**Status**: Proposal — pending review

---

## Problem Statement

Two fundamental gaps in how we evaluate and gate signal quality:

1. **Metric tunnel vision**: We've optimized almost exclusively for VC@20. The gates already have 12 metrics across Group A/B, but development decisions (feature selection, model choice, architecture) were driven by VC@20 movement. VC@50, VC@100, Recall@50, NDCG, and Spearman are equally important for signal consumers.

2. **New constraint detection is unmonitored**: No gate, no metric, no evaluation for how the signal handles constraints without binding history. This is a first-class signal quality dimension — ~25% of binding events each month come from constraints binding for the first time, and V7.0 systematically underranks them.

---

## Part 1: Broadening Metric Coverage

### Current State

| Metric | Group | Gate floor | Used in development? |
|--------|:-----:|:---------:|:---:|
| VC@20 | A | 0.2535 | **PRIMARY** — all decisions driven by this |
| VC@100 | A | 0.5407 | Checked but not optimized for |
| Recall@20 | A | 0.165 | Checked |
| Recall@50 | A | 0.207 | Checked |
| NDCG | A | 0.3981 | Checked |
| VC@10 | B | 0.1705 | Rarely checked |
| VC@25 | B | 0.290 | Rarely checked |
| **VC@50** | **B** | 0.4188 | **Rarely checked — should be Group A** |
| VC@200 | B | 0.7384 | Rarely checked |
| Recall@10 | B | 0.1725 | Rarely checked |
| Recall@100 | B | 0.2152 | Demoted due to tie issues |
| **Spearman** | **B** | 0.1841 | **Rarely checked — should be Group A** |

### Why VC@20 Alone Is Insufficient

VC@20 measures: "of all binding value, how much is captured by the top 20 constraints?"

Problems:
- **N=20 is tiny**: With 500-900 constraints per month, top 20 is the top 2-4%. A model can get great VC@20 by being excellent at the extreme top and terrible everywhere else.
- **Ignores depth**: A trader who looks at the top 50 or 100 constraints gets no signal quality guarantee from VC@20.
- **Ignores ordering**: Two models with identical VC@20 can have very different ranking quality throughout the list. Spearman and NDCG capture this.
- **Ignores stability**: VC@20 can be dominated by a single high-value constraint. If that one constraint happens to be ranked #1, VC@20 looks great even if the rest of the ranking is noise.

### Proposed Group A Expansion

Promote to Group A:

| Metric | Rationale |
|--------|-----------|
| **VC@50** | Most traders look at 50+ constraints. This is the "working list" metric. |
| **Spearman** | Measures full-ranking correlation. Catches models that are good at extremes but bad in the middle. |

This gives Group A 7 metrics across 3 depth levels (20, 50, 100) plus full-ranking quality (NDCG, Spearman).

### Evaluation Discipline

For every experiment going forward, report ALL of these in a table:

```
         VC@20  VC@50  VC@100  Recall@20  Recall@50  NDCG  Spearman
v0       0.184  0.419  0.541    0.165      0.207    0.398   0.184
v2       0.353  0.612  0.732    0.283      0.384    0.551   0.378
delta   +92%   +46%   +35%    +72%       +86%     +38%    +105%
```

A version that improves VC@20 but regresses VC@50 or Spearman should be investigated, not celebrated.

---

## Part 2: Defining "New Constraint"

### The Definition Problem

"New constraint" conflates several distinct situations:

| Type | Definition | Prevalence | V7.0 behavior |
|------|-----------|:---:|------|
| **Type 1: History-zero** | Never bound in realized DA cache (2017-04 to present) | ~24% of V6.2B universe | bf=0, model has no binding signal |
| **Type 2: BF-zero** | Not bound in recent windows (bf_1/3/6 = 0) but may have ancient history | ~55-65% of V6.2B | bf_1/3/6=0, bf_12/15 may be >0 |
| **Type 3: Lapsed** | Bound before, then stopped for 6+ months | Subset of Type 2 | bf_6=0 but bf_12/15 >0, partial signal |
| **Type 4: Unmapped** | FG-style constraint_id that can't join to realized DA | ~8% of V6.2B | ALWAYS bf=0 regardless of true binding |

### The Mapping Issue (Type 4)

**Finding: ~41 FG-style constraint_ids per month in V6.2B cannot join to realized DA.**

These constraints use non-numeric IDs (e.g., `1024FG`, `FG29234 13808 A NIP13030`) that don't match the numeric MISO IDs in the realized DA cache. Some of them share a `branch_name` with a numeric V6.2B constraint that DOES have binding history:

| FG constraint_id | Numeric counterpart | Branch name | Numeric binding months |
|-----------------|:---:|-------------|:---:|
| `FG29234 13808 A NIP13030` | 79047 | `13808 A` | 9 months |
| `FG27862 ELDORWELSB69_1 1 ALW16118` | 300342 | `ELDORWELSB69_1 1` | 10 months |
| `FG27809 KEO____WMPEHV8 A EA50022` | 310654 | `KEO____WMPEHV8 A` | 6 months |
| `1024FG` | 315320 | `8VOLUN_PHIPP 1` | 1 month |

**Impact**: 6 of 41 FG constraints have a numeric counterpart with real binding history. These get:
- `realized_sp = 0` in training (treated as non-binding even if their physical counterpart binds)
- `binding_freq = 0` (no binding history matches their ID)

This is a small but real data quality issue (~1.2% of constraints). The remaining 35 FG constraints have no binding counterpart — they may represent forward-modeled constraints that MISO DA doesn't track.

### Recommended Operational Definition

For gating and evaluation purposes, define "new constraint" as:

**BF-zero**: `max(bf_1, bf_3, bf_6) == 0`

This is the right boundary because:
- It's what determines V7.0 model behavior (78% importance from BF features)
- It captures all three problematic types: history-zero, lapsed (>6mo), and unmapped
- It's computable at signal generation time (no need for separate bridge tables)
- It cleanly separates the two information regimes (empirical vs structural signal)

For deeper analysis, sub-categorize:
- **BF-zero/history-zero**: bf_6=0 AND never bound in full DA history
- **BF-zero/lapsed**: bf_6=0 BUT bf_12 or bf_15 > 0 (ancient memory exists)
- **BF-zero/unmapped**: FG-style ID that can't join to DA (always bf=0)
- **BF-positive**: bf_1 or bf_3 or bf_6 > 0

### Fix the Mapping Issue

Before adding gates, fix the FG mapping:

```python
def resolve_fg_mapping(df: pl.DataFrame, binding_sets: dict) -> pl.DataFrame:
    """For FG-style constraint_ids, try to inherit binding history
    from a numeric constraint sharing the same branch_name."""
    fg_mask = ~df["constraint_id"].str.contains(r"^\d+$")
    fg_rows = df.filter(fg_mask)

    for row in fg_rows.iter_rows(named=True):
        bn = row["branch_name"]
        # Find numeric constraints with same branch_name
        numeric_match = df.filter(
            (df["branch_name"] == bn) &
            df["constraint_id"].str.contains(r"^\d+$")
        )
        if len(numeric_match) > 0:
            # Inherit binding history from numeric counterpart
            numeric_id = numeric_match["constraint_id"][0]
            # Transfer BF values from numeric to FG row
            ...
    return df
```

Or more simply: when computing binding_freq for FG constraints, look up by `branch_name` instead of `constraint_id`.

---

## Part 3: New-Constraint Detection Gates

### Proposed Metrics

Add three new metrics to `evaluate_ltr()`:

```python
def evaluate_ltr_extended(
    actual: np.ndarray,
    scores: np.ndarray,
    has_bf: np.ndarray,  # boolean: True if bf_1/3/6 > 0
) -> dict:
    """Extended metrics including new-constraint detection."""
    base = evaluate_ltr(actual, scores)

    # --- New-constraint metrics ---
    bf_zero = ~has_bf
    bf_zero_binding = bf_zero & (actual > 0)
    n_bf_zero_binders = bf_zero_binding.sum()

    if n_bf_zero_binders > 0:
        n = len(scores)
        # Percentile rank for each constraint (0=best, 1=worst)
        ranks = np.argsort(np.argsort(-scores)) / n

        bf_zero_binder_ranks = ranks[bf_zero_binding]

        # NewBind-VC@50: Value capture of BF-zero binding value in model's top-50
        bf_zero_actual = actual.copy()
        bf_zero_actual[~bf_zero] = 0  # zero out BF-positive
        base["NewBind-VC@50"] = value_capture_at_k(bf_zero_actual, scores, 50)

        # NewBind-Recall@50: of BF-zero binders in actual top-50, how many in model's top-50
        # (measures: can the model find new binders at reasonable depth?)
        base["NewBind-Recall@50"] = recall_at_k(bf_zero_actual, scores, 50)

        # NewBind-T01: fraction of BF-zero binders in top 40% (T0+T1)
        base["NewBind-T01"] = float((bf_zero_binder_ranks <= 0.4).mean())

        # NewBind-T34: fraction of BF-zero binders in bottom 40% (T3+T4) — lower is better
        # Invert to make it higher-is-better: 1 - T34_fraction
        base["NewBind-AvoidT34"] = float(1.0 - (bf_zero_binder_ranks >= 0.6).mean())

        # NewBind-AvgRank: average percentile rank of BF-zero binders (lower rank = better)
        # Invert: 1 - avg_rank to make higher-is-better
        base["NewBind-AvgRank"] = float(1.0 - bf_zero_binder_ranks.mean())
    else:
        base["NewBind-VC@50"] = float("nan")
        base["NewBind-Recall@50"] = float("nan")
        base["NewBind-T01"] = float("nan")
        base["NewBind-AvoidT34"] = float("nan")
        base["NewBind-AvgRank"] = float("nan")

    base["n_bf_zero"] = int(bf_zero.sum())
    base["n_bf_zero_binding"] = int(n_bf_zero_binders)

    return base
```

### Gate Calibration

To set gate floors, we need baselines. From the holdout analysis:

**V6.2B (formula baseline) on holdout, f0/onpeak:**
- BF-zero binders: T0=13.9%, T1=13.3% → T0+T1 = **27.2%**
- BF-zero binders in T3+T4: 60.6% → AvoidT34 = **39.4%**
- BF-zero binder avg rank: ~0.58 → AvgRank = **0.42**

**V7.0 (current champion) on holdout, f0/onpeak:**
- BF-zero binders: T0=3.0%, T1=13.3% → T0+T1 = **16.3%**
- BF-zero binders in T3+T4: 53.3% → AvoidT34 = **46.7%**
- BF-zero binder avg rank: ~0.55 → AvgRank = **0.45**

Note: V7.0 has slightly better AvoidT34 and AvgRank than V6.2B (it clusters BF-zero in the middle rather than extreme tails), but much worse T0+T1 (V6.2B gets more into the top tiers).

**Proposed gate floors** (calibrated from V6.2B baseline):

| Metric | Floor | Tail floor | Group | Rationale |
|--------|:-----:|:----------:|:-----:|-----------|
| NewBind-T01 | 0.25 | 0.10 | **A** | Must beat V6.2B's ~27% T0+T1 rate (floor slightly below) |
| NewBind-AvoidT34 | 0.35 | 0.20 | **A** | At most 65% of BF-zero binders in bottom 40% |
| NewBind-VC@50 | — | — | B | Monitor only — depends heavily on constraint count |
| NewBind-AvgRank | 0.40 | 0.25 | B | Monitor — avg rank should be above median |

### Why Two NewBind Gates in Group A

- **NewBind-T01** catches the "zero T0" problem: a model that puts 0% of BF-zero binders in T0+T1 would fail this gate. V7.0's 16.3% is below the 25% floor — it would fail this gate today.

- **NewBind-AvoidT34** catches the "dumped at bottom" problem: a model that assigns all BF-zero constraints to T3-T4 would fail. Both V6.2B (39.4%) and V7.0 (46.7%) pass, but a degenerate model that puts 100% of BF-zero binders at the bottom would fail.

Together they require: (a) meaningful top-tier representation AND (b) not dumping everything to the bottom.

---

## Part 4: What the Winning Signal Looks Like

### Requirements

A signal that passes ALL gates must simultaneously:

1. **Overall discrimination**: VC@20 ≥ 0.25, VC@50 ≥ 0.42, VC@100 ≥ 0.54 (beat V6.2B floor)
2. **Full-ranking quality**: NDCG ≥ 0.40, Spearman ≥ 0.18 (meaningful correlation throughout)
3. **Recall at depth**: Recall@20 ≥ 0.17, Recall@50 ≥ 0.21 (find true top constraints)
4. **New-constraint detection**: NewBind-T01 ≥ 0.25 (25%+ of BF-zero binders in T0+T1)

### Why Single-Model V7.0 Cannot Pass

V7.0 champion (v10e-lag1) achieves:
- VC@20 = 0.353, VC@50 ≈ 0.59, VC@100 = 0.73 → passes all VC gates
- NDCG = 0.55, Spearman = 0.38 → passes
- Recall@20 = 0.28, Recall@50 = 0.38 → passes
- **NewBind-T01 ≈ 0.16 → FAILS** (below 0.25 floor)

The single-model architecture cannot pass both overall discrimination AND new-constraint gates because BF feature dominance suppresses structural signal for BF-zero constraints.

### Candidate Architectures

**Option A: Two-model ensemble (v14-style)**
- Model A (9 features with BF) for BF-positive constraints
- Model B (5 structural features) for BF-zero constraints
- Hard switch based on bf signal presence
- Pro: cleanly addresses both populations
- Con: VC@20 drops ~10% (may still pass gate floor, but loses headroom)
- NewBind-T01 on holdout: 49.1% (passes easily)

**Option B: Stratified tier allocation**
- Run Model A for all constraints
- Reserve X% of T0+T1 slots for BF-zero constraints
- Rank BF-zero constraints among themselves using Model B or V6.2B formula
- Pro: guarantees representation, no VC@20 loss for BF-positive
- Con: adds post-processing complexity, tier boundaries become artificial

**Option C: Population-aware LambdaRank**
- Single model but with custom loss that upweights ranking errors for BF-zero constraints
- Sample weighting: give BF-zero binders 3-5x weight in the loss
- Pro: single model, clean deployment
- Con: may not be enough — LightGBM still learns BF as primary splitter

**Option D: Blended scores**
- `final_score = α * model_score + (1-α) * v62b_formula_score`
- For BF-zero constraints, α is lower (more formula influence)
- For BF-positive constraints, α is higher (more model influence)
- Pro: simple, tunable
- Con: need to find right α schedule

### Recommended Path

1. **Implement the extended evaluation** (Part 3 metrics) in `evaluate.py`
2. **Recalibrate gates** with NewBind metrics included
3. **Run V6.2B, V7.0, and v14a ensemble** through the new evaluation to establish baselines
4. **Test Options A-D** against the full gate set
5. **Declare champion** based on composite pass across ALL Group A gates (including NewBind)

---

## Part 5: Implementation Plan

### Step 1: Extend evaluate.py

Add `has_bf` parameter to `evaluate_ltr()` (optional, backward-compatible).
Add NewBind metrics. Modify `aggregate_months()` to handle NaN values.

### Step 2: Extend gates.json

Add NewBind-T01 and NewBind-AvoidT34 to Group A for all 4 slices.
Promote VC@50 and Spearman to Group A.

### Step 3: Extend experiment scripts

All run scripts need to pass `has_bf` array to `evaluate_ltr()`.
This requires computing BF before evaluation (already done in v10e/v14 scripts).

### Step 4: Baseline evaluation

Run v0 (formula), v2 (V7.0 champion), v14a (ensemble) through the new evaluation.
Generate the full multi-metric comparison table.

### Step 5: Fix FG mapping

Resolve FG-style constraint_ids by branch_name lookup before computing BF.
Re-run evaluation to see if this changes any results.

### Step 6: Develop winning signal

Iterate on Options A-D until all Group A gates pass (including NewBind).

---

## Appendix: Data Summary

### Monthly new-binder prevalence (holdout, all realized DA binding)

| Month | Total binding | First-time binders | % first-time |
|-------|:---:|:---:|:---:|
| 2024-01 | 291 | 68 | 23% |
| 2024-04 | 383 | 115 | 30% |
| 2024-10 | 421 | 106 | 25% |
| 2025-06 | 261 | 76 | 29% |
| 2025-10 | 451 | 156 | 35% |
| **Average** | **312** | **80** | **25%** |

### V6.2B constraint universe composition (f0/onpeak, 2025-01)

| Category | Count | % | Bind rate |
|----------|:---:|:---:|:---:|
| Ever bound in DA history (numeric ID) | 329 | 67% | — |
| Never bound, numeric ID | 78 | 16% | — |
| FG-style ID (never matches DA) | 41 | 8% | 0% (by construction) |
| Special (SO_MW_Transfer etc.) | ~1 | <1% | — |
| **Total** | 489 | 100% | 11% |

### FG-style mapping audit

- 41 FG-style IDs in V6.2B per month
- 6 have a numeric counterpart (same branch_name) with real binding history
- 35 have no binding counterpart in realized DA
- All 41 get realized_sp=0 and bf=0 regardless of true binding status
