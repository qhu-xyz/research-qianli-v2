# Annual Signal Framework Redesign Report

**Date**: 2026-03-11
**Status**: Proposal for independent review
**Scope**: Framework changes only. Modeling changes deferred to separate plan.

---

## Table of Contents

1. [Current Framework Inventory](#1-current-framework-inventory)
2. [Diagnosis: What's Wrong](#2-diagnosis-whats-wrong)
3. [Universe Options](#3-universe-options)
4. [Metrics & Gates Redesign](#4-metrics--gates-redesign)
5. [Constraint Cohorts & Failure Mode Analysis](#5-constraint-cohorts--failure-mode-analysis)
6. [Ground Truth Mapping Impact](#6-ground-truth-mapping-impact)
7. [Implementation Order](#7-implementation-order)
8. [Risk Register](#8-risk-register)

---

## 1. Current Framework Inventory

### Two Data Systems

The pipeline draws from two distinct data systems:

| System | Purpose | Path prefix | Granularity |
|--------|---------|-------------|-------------|
| **Signal data** | Pre-assembled curated signals with ranked features | `/opt/data/xyz-dataset/signal_data/miso/constraints/` | One row per (constraint, year, quarter, class_type) |
| **Spice6 pipeline data** | Raw ML pipeline outputs: density distributions, scores, limits | `/opt/data/xyz-dataset/spice_data/miso/` | One row per (constraint, outage_date, flow_direction) |

**Signal data** contains the constraint universe and pre-ranked features (shadow_rank, deviation_rank, percentile features). This defines who we rank.

**Spice6 pipeline data** provides enrichment features — `density_signal_score` (binding probability), 77-bin density distribution, constraint limits — that can be joined onto the signal universe. The bridge table (`MISO_SPICE_CONSTRAINT_INFO`) also lives here.

**Data lineage**: Spice6 density_distribution (rawest, 77 bins per constraint per outage_date) → density_signal_score (derived, 1 binding probability per constraint) → V4.4 signal (most aggregated, 20 percentile features = compressed density distribution + shadow/deviation ranks).

> **DATA SOURCE DECISION**: A teammate suggested using `MISO_SPICE_DENSITY_SIGNAL_SCORE` directly via the pbase loader. Investigation resolved this: density_signal_score has only 3 columns (constraint_id, flow_direction, score) — it cannot serve as the primary signal. V4.4 contains the same information in compressed form (20 percentile features) plus shadow/deviation ranks. We use V4.4 as the universe + base features, and density_signal_score as enrichment (joined as an additional feature) and optionally as a universe expansion filter. See `data-tracing.md` "Resolution" section for full analysis.

### Data Flow

```
Signal data (V6.1 parquet)
  -> ml/data_loader.py :: load_v61_enriched()
     Loads {V61_SIGNAL_BASE}/{planning_year}/{aq_round}/onpeak
     Joins spice6 density features on constraint_id
  -> ml/binding_freq.py :: enrich_with_binding_freq()
     Loads realized DA parquet (monthly, keyed by constraint_id)
     Maps constraint_id -> branch_name via MISO_SPICE_CONSTRAINT_INFO bridge table
     Computes bf_1/3/6/12/15 per branch_name
  -> ml/ground_truth.py :: get_ground_truth()
     Fetches realized DA shadow prices for 3 market months in target quarter
     Maps DA constraint_id -> branch_name via same bridge table
     LEFT JOINs onto V6.1 universe on branch_name
     Constraints not binding get realized_shadow_price = 0.0
  -> ml/train.py :: train_ltr_model()
     LightGBM LambdaRank with tiered labels (0/1/2/3)
  -> ml/evaluate.py :: evaluate_ltr()
     Computes VC@k, Recall@k, NDCG, Spearman, Tier-AP
```

### Split Logic (config.py)

- **Expanding window**: train on all prior years, eval on target year
- **Dev**: 12 groups (2022-06 through 2024-06, 4 quarters each)
- **Holdout**: 4 groups (2025-06/aq1-4)
- **Train once per eval year**, reuse model across 4 quarters

### Row Universe

- **Row definer**: V6.1 signal (`Signal.MISO.SPICE_ANNUAL_V6.1`)
- **Unit**: one row per (constraint_id, planning_year, aq_round)
- **Size**: 286-632 constraints per group (varies by year/quarter)
- **Join key to ground truth**: `branch_name` (via bridge table from constraint_id)

### Current Champion: v16

- **Features (7)**: shadow_price_da, da_rank_value, bf_6, bf_12, bf_15, bfo_6, bfo_12
- **Holdout VC@20**: 0.3920 (+68% vs v0b pure formula baseline)
- **Holdout metrics**: VC@50=0.5935, Recall@50=0.4800, NDCG=0.7093, Spearman=0.5794

### Baseline Contract

- **v0**: V6.1 formula (0.60*da_rank + 0.30*dens_mix + 0.10*dens_ori). Holdout VC@20=0.2329
- **v0b**: Pure da_rank_value only. Holdout VC@20=0.2997
- All ML versions compared against v0b as the operational baseline

---

## 2. Diagnosis: What's Wrong

### 2.1 Metric Tunnel Vision

All 17 experiment versions (v0-v17) were optimized primarily for **VC@20**. This is wrong:

- VC@50 and VC@100 matter equally for portfolio construction (we don't just trade top-20)
- Recall@50/100 measures completeness — missing a top binder anywhere hurts
- NDCG measures full-ranking quality, not just head
- Spearman measures ordinal agreement across entire list
- **A model that wins VC@20 but loses Recall@100 is not better** — it concentrates value in fewer positions while missing more opportunities

The current gate framework (Group A = blocking, Group B = monitoring) puts VC@20 at the top with equal weight to VC@100 and Recall@20, but in practice experiment selection was driven by VC@20 improvements.

### 2.2 Universe Is Too Small

V6.1 has the **smallest universe** of all available annual signals:

| Signal | aq1 size | aq2 size | aq3 size |
|--------|:--------:|:--------:|:--------:|
| **V6.1 (current)** | **395** | **385** | **286** |
| V4.4.R1 | 1227 | 1483 | 964 |
| V4.5.R1 | 736 | 889 | 578 |

**Binding coverage** (2025-06 holdout, DA binding constraints visible via bridge table):

| Signal | aq1 | aq2 | aq3 |
|--------|:---:|:---:|:---:|
| **V6.1** | 144/546 (26%) = **38% SP** | 193/814 (24%) = **42% SP** | 124/657 (19%) = **41% SP** |
| **V4.4** | 248/546 (45%) = **68% SP** | 366/814 (45%) = **68% SP** | 242/657 (37%) = **61% SP** |

V6.1 can only see ~40% of binding value. V4.4 sees ~65%. This is not a model problem — **the remaining 60% of binding value is invisible to any model built on V6.1**.

### 2.3 Key Constraints Missing from V6.1

| Constraint | V6.1 | V4.4 |
|------------|:-----:|:-----:|
| **BENTON** | Not present in any quarter | Present in aq3 |
| **MNTCELO** | aq3 only, ranked 45th percentile | aq1/aq2/aq3, with flow deviation signal |

### 2.4 "New Constraint" Blind Spot

Constraints with all binding frequency features = 0 (bf_6=bf_12=bf_15=bfo_6=bfo_12=0):

- These constraints have **zero historical binding signal** and rely entirely on shadow_price_da and da_rank_value
- When shadow_price_da is also 0 (truly new), the model has **no signal at all**
- Example: COO-OLI1 ranked 287/395 (72nd percentile) with zero signal, bound for $1930 SP
- **No metric currently measures performance on this population separately**

### 2.5 Overfit Risk with Small Holdout

- Only **4 holdout groups** (2025-06/aq1-4)
- Aggregate metrics across 4 groups can swing heavily on a single group
- Gate decisions based on mean-of-4 are noisy — one outlier quarter can flip a gate
- **Per-quarter breakdowns and distribution reporting** needed, not just aggregates

---

## 3. Universe Options

V6.1 is too small (§2.2). The question is not just "which pre-built signal to adopt" but "what is the right universe and feature source?" There are three tiers of ambition:

| Tier | Universe | Features | SP coverage | Engineering cost |
|------|----------|----------|:-----------:|:----------------:|
| **A: V4.4 as-is** | 1,227 constraints | 35 pre-computed columns | ~65% | Low — loader + join key |
| **B: V4.4 + score-gated expansion** | ~1,926 constraints | V4.4 features + density_signal_score | ~72.5% | Medium — need spice6 join for expanded rows |
| **C: Custom from raw spice6** | Flexible (score-filtered subset of 12,841) | Custom aggregation of 77-bin density + BF + DA history | ~72.5%+ | High — aggregation design, daily→annual collapse |

**Phase 1 uses Tier A** (V4.4 as-is) because it's ready to use and establishes baselines fast. But the data loader and ground truth pipeline must be built to support Tier B/C without rebuild. V4.4 is scaffolding, not the destination.

**Why not jump to Tier C?** V4.4's pre-computed percentile features already show strong monotonic tier separation (see feature quality table below). We need evidence that custom re-aggregation beats V4.4's compression before investing the engineering. Phase 2 tests this directly.

### V4.4 vs V6.1 Comparison

| Aspect | V6.1 (current) | V4.4 (proposed) |
|--------|:---:|:---:|
| Signal path | `Signal.MISO.SPICE_ANNUAL_V6.1` | `TEST.Signal.MISO.SPICE_ANNUAL_V4.4.R1` |
| Universe size | 286-632 | 700-1483 |
| Years available | 2019-2025 | 2019-2025 |
| Quarters | aq1-aq4 | aq1-aq4 |
| Class types | onpeak + offpeak | onpeak + offpeak |
| Identifier column | `branch_name`, `constraint_id` | `equipment` (≈ branch_name), constraint_id parsed from `__index_level_0__` |
| DA rank feature | `da_rank_value` | `shadow_rank` (Spearman ρ=0.83 with da_rank_value) |
| Flow features | `density_mix_rank_value`, `density_ori_rank_value`, `ori_mean`, `mix_mean` | `deviation_max_rank`, `deviation_sum_rank`, 20 percentile columns |
| `shadow_price_da` scale | Monthly average (~791) | Cumulative sum (~8459) |

### Universe Overlap (2025-06/aq1 onpeak)

```
V6.1:  333 unique equipment names
V4.4: 1227 unique equipment names

V6.1 ∩ V4.4:  272  (82% of V6.1 is in V4.4)
Only V6.1:      61  (18% of V6.1 is NOT in V4.4)
Only V4.4:     453  (V4.4 brings 453 new constraints)
V4.5 ⊂ V4.4:  True (V4.5 is a strict subset of V4.4)
```

**61 V6.1 constraints are not in V4.4.** The union adds only +1-2pp coverage over V4.4 alone, so these 61 are low-value. But this needs verification.

### Feature Independence

V4.4's deviation features are **nearly independent** from V6.1's DA features on shared constraints:

| Pair | Spearman ρ |
|------|:---:|
| da_rank_value ↔ shadow_rank | 0.83 (same underlying signal) |
| da_rank_value ↔ deviation_max_rank | **0.08** (independent) |
| da_rank_value ↔ deviation_sum_rank | **0.02** (independent) |
| density_mix_rank ↔ deviation_max_rank | **0.18** (mostly independent) |

This means V4.4 brings both **more constraints AND genuinely new signal**.

### V4.4 Feature Quality by Tier

| Tier | N | deviation_max | deviation_sum | shadow_da | 100_max |
|------|---|:---:|:---:|:---:|:---:|
| 0 (most binding) | 147 | 15.71 | 13.13 | 35,612 | 0.987 |
| 1 | 196 | 15.57 | 11.73 | 6,002 | 0.909 |
| 2 | 245 | 14.94 | 10.07 | 5,957 | 0.570 |
| 3 | 295 | 13.64 | 8.06 | 6,039 | 0.158 |
| 4 (least binding) | 344 | 10.86 | 4.75 | 2,112 | 0.004 |

Strong monotonic separation — deviation and percentile features are meaningful signal.

### What Needs to Change in Code

1. **`ml/config.py`**: Add `V44_SIGNAL_BASE` path, new feature sets for V4.4 columns
2. **`ml/data_loader.py`**: New `load_v44_enriched()` function
   - Parse `constraint_id` from `__index_level_0__` (format: `"296436|-1|spice"`)
   - Rename `equipment` → `branch_name` (or update all downstream to use `equipment`)
   - Handle `shadow_price_da` scale difference (cumulative vs monthly avg)
3. **`ml/ground_truth.py`**: Update join key — V4.4 uses `equipment` not `branch_name`
   - Bridge table maps constraint_id → branch_name; V4.4 uses equipment which IS the branch_name
   - Verify: `V4.4.equipment == bridge.branch_name` for all shared constraints
4. **`ml/binding_freq.py`**: No change needed if join key alignment is handled in data_loader
5. **`ml/evaluate.py`**: No change needed (metric computation is universe-agnostic)
6. **`ml/config.py`**: Update `_LEAKY_FEATURES` set for V4.4-specific columns

### CRITICAL: Backward Compatibility

- **v0-v16 results become non-comparable** after universe change (different N, different constraints)
- Must re-run v0b baseline on V4.4 to establish new baseline
- Must re-run v16 champion on V4.4 to see if BF features still dominate
- **Registry needs a version break** — new registry namespace (e.g., `registry/v44/`)

### Beyond V4.4: Score-Gated Expansion (Tier B)

V4.4 is a curated 1,227-constraint subset of the 12,841 constraints spice6 models. Adding excluded constraints with `density_signal_score > 0.001` yields ~1,926 constraints at ~72.5% SP coverage. Key findings:

- 48 excluded constraints with score>0.001 actually bind for $162k SP — real missed value
- The score filter keeps universe manageable (1,926 vs 12,841) while capturing most binding value
- Full 12,841 is counterproductive (97.5% non-binding → extreme class imbalance)
- density_signal_score is independent from V4.4's features (Spearman ρ=0.48 with rank, -0.23 with shadow_rank)

The expanded constraints lack V4.4's pre-computed features (percentiles, deviation ranks). They'd have: density_signal_score, constraint_limit, BF features (from realized DA), and optionally custom-aggregated density bins. This is why Tier B requires the spice6 join infrastructure.

### Beyond V4.4: Custom Aggregation from Raw Density (Tier C)

V4.4's 20 percentile features are a pre-computed compression of spice6's 77-bin density distribution (flow probability at each shadow price threshold, aggregated across outage dates by max and sum). We could re-aggregate differently:

- Different bin boundaries or tail statistics (skewness, kurtosis)
- Different temporal aggregation (mean vs max vs specific months, seasonal weighting)
- Constraint-specific aggregation (e.g., weight summer outage dates higher for aq1)

**But**: V4.4's percentiles already show strong monotonic tier separation. Custom aggregation is only worth pursuing if Phase 2 shows V4.4's percentiles are the bottleneck. This is a hypothesis to test, not an assumption to build on.

---

## 4. Metrics & Gates Redesign

### Current Problems

1. **VC@20 dominance**: All experiment decisions driven by VC@20
2. **Group A/B split unclear**: VC@50 is "monitoring" but matters just as much as VC@100 (blocking)
3. **No per-quarter reporting**: Aggregate mean across 4 holdout groups hides quarter-specific failures
4. **No cohort metrics**: No measurement of performance on new vs. established constraints
5. **Gate math with N=4 is noisy**: Mean of 4 numbers is not a reliable decision criterion

### Proposed Metric Framework

#### Tier 1: Blocking Gates (must pass to promote)

These metrics are evaluated on **every holdout group individually** (not just mean):

| Metric | What it measures | Why blocking |
|--------|-----------------|--------------|
| VC@50 | Value captured in top-50 | Core portfolio metric — top 50 is realistic bid size |
| VC@100 | Value captured in top-100 | Broader portfolio coverage |
| Recall@50 | Completeness of top-50 | Missing a top binder is costly |
| Recall@100 | Completeness of top-100 | Broader completeness |
| NDCG | Full ranking quality | Penalizes any misranking, not just head |

**Gate rule**: Candidate must beat baseline on **at least 3 of 4 holdout groups** for each metric, AND mean must be ≥ baseline mean. This prevents a single outlier quarter from flipping the gate.

#### Tier 2: Monitoring (reported, not gated)

| Metric | What it measures |
|--------|-----------------|
| VC@20 | Head precision (still useful, just not the only metric) |
| VC@200 | Tail coverage |
| Recall@20 | Head completeness |
| Spearman | Full-list ordinal agreement |
| Tier0-AP | Precision on top-20% constraints |
| Tier01-AP | Precision on top-40% constraints |

#### Tier 3: Cohort Contribution Metrics (new — see Section 5)

Measures each cohort's contribution to the **global** top-K, not rank-within-subgroup:

| Metric | What it measures |
|--------|-----------------|
| Cohort-in-Top50 | How many of each cohort's constraints appear in global top-50 |
| Cohort-SP-in-Top50 | Binding SP from each cohort's constraints in global top-50 |
| Cohort-Recall@50 | Fraction of cohort's total binding SP captured in global top-50 |
| Cohort-Miss-Rate | Fraction of cohort's binders ranked below position 100 |

#### Reporting Changes

Every experiment report must include:

1. **Per-group breakdown** for all 4 holdout quarters (not just mean)
2. **Per-quarter min/max/spread** to show stability
3. **Cohort breakdown**: metrics split by new/dormant/established
4. **Per-aq aggregation**: aq1/aq2/aq3/aq4 means separately (seasonality check)

### Why Remove VC@20 from Blocking

- VC@20 is **highly concentrated**: with ~300 constraints and ~50 binders, the top-20 slots are dominated by a few high-value constraints
- A single constraint moving in/out of top-20 can swing VC@20 by 5-10%
- With only 4 holdout groups, VC@20 gate decisions are essentially random coin flips
- VC@50 is more stable and still captures head performance

---

## 5. Constraint Cohorts & Failure Mode Analysis

### Why Cohorts Matter

The v1 champion (v16) gives 80%+ feature importance to historical features (BF + shadow_rank). This means the model is essentially blind to constraints without recent binding history. We need to measure WHERE the model succeeds and fails, which requires defining populations by **what signal the model has available**, not just by label.

### Cohort Definitions

The old "never-bound" definition conflated two very different failure modes. Revised:

| Cohort | Definition | Model signal available | Why distinct |
|--------|-----------|----------------------|-------------|
| **History-zero** | `shadow_price_da = 0 AND bf_15 = 0 AND bfo_12 = 0` | Only prediction features (deviation, percentiles) | No historical DA binding in any lookback — model's historical features are all zero |
| **History-only dormant** | `shadow_price_da > 0 AND bf_12 = 0` | shadow_rank + prediction features, but no recent BF | Has DA history (shadow_rank works), but nothing recent. Long-dormant constraints that re-emerge. |
| **Cross-peak signal** | `bf_12 = 0 AND bfo_12 > 0` | Offpeak BF + prediction features | Never bound onpeak recently, but offpeak binding provides structural signal |
| **Established** | `bf_12 > 0` | All features — BF is strong | The "easy" population where BF dominates |

**On the "history-zero" definition**: `shadow_price_da = 0` means zero historical DA shadow price in V4.4's lookback window (which is cumulative, so this covers all months V4.4 considered). Combined with `bf_15 = 0 AND bfo_12 = 0`, this means: no DA binding in V4.4's lookback AND no binding in the 15-month BF window AND no offpeak binding in 12 months. However, `shadow_price_da = 0` does NOT guarantee "never in history" — a constraint that bound 3+ years ago but not since may have `shadow_price_da = 0` in V4.4's window. This is acceptable for our purpose: the cohort identifies constraints **where the model has no historical signal**, regardless of whether ancient history exists. The model can't use what it can't see.

**IMPORTANT**: This cohort should NOT be used as the sole gate for dual-model decisions. Phase 1 must also measure:
- How many history-zero constraints actually bind (if <5 per quarter, the cohort is too small to justify architecture changes)
- Whether long-dormant constraints (bound >3 years ago, `shadow_price_da = 0` now) contaminate the cohort — check by cross-referencing against the full realized DA cache (2017-04+)
- **Monthly cross-reference**: A constraint with `shadow_price_da = 0` in annual signals may still have binding history in **monthly** realized DA or monthly signals (V6.2B). Phase 1 must cross-reference each annual history-zero constraint against monthly realized DA (2017-04+) and V6.2B monthly signal data. A constraint that is annual-history-zero but monthly-binding-active is a "bring in monthly history as feature" candidate, not a prediction-model candidate.

**Two lookback windows for history-zero classification**: Rather than a single binary definition, Phase 1 should compute `months_since_last_binding` across ALL realized DA (annual + monthly combined) for each constraint and plot the distribution. Natural clusters in this distribution determine two thresholds:
- **Short window** (~6-12 months): "recently inactive" — may re-emerge, monthly cross-reference most relevant
- **Long window** (~24-36 months): "structurally dormant" — genuinely new or permanently inactive
The exact thresholds should be set from the data distribution, not guessed. This produces a richer cohort taxonomy than a single binary split.

**Key distinction**: "History-zero" has zero signal from every historical feature — the model must rely entirely on prediction features. "History-only dormant" has shadow_rank signal (the #1 or #2 feature) but no BF. These have fundamentally different detection difficulty.

For non-overlapping reporting, assign each constraint to exactly one cohort using priority: established > cross-peak > history-only dormant > history-zero. (Cross-peak can overlap with history-only dormant when `shadow_price_da > 0 AND bf_12 = 0 AND bfo_12 > 0`; priority resolves this.)

### Measurement: Global Shortlist Contribution

**The old plan was wrong.** Computing VC@50 within each subgroup (e.g., "what fraction of binding SP among the 200 never-bound constraints is captured in the top-50 of those 200?") answers the wrong question. A trader doesn't have separate shortlists per cohort — they have one global shortlist.

**The right question**: How many constraints from each cohort appear in the model's **global top-K**?

For each experiment, report:

```python
def cohort_contribution(df, scores, actual, K=50):
    """How much does each cohort contribute to the global top-K?"""
    top_k_idx = np.argsort(-scores)[:K]
    top_k_actual = actual[top_k_idx]
    top_k_cohort = df_cohort_labels[top_k_idx]  # assigned via priority above

    for cohort in ['history_zero', 'history_dormant', 'cross_peak', 'established']:
        mask = top_k_cohort == cohort
        n_in_topk = mask.sum()
        sp_in_topk = top_k_actual[mask].sum()
        total_sp_for_cohort = actual[df_cohort_labels == cohort].sum()
        print(f"{cohort}: {n_in_topk}/{K} slots, "
              f"${sp_in_topk:.0f} SP captured, "
              f"{sp_in_topk/total_sp_for_cohort:.1%} of cohort's binding value")
```

This tells us: "Of the model's top-50 picks, how many came from each population, and how much value did they capture?" This is what matters for trading.

**Additional cohort metrics** (all computed against the global ranking):

| Metric | What it measures |
|--------|-----------------|
| `Cohort-in-Top50` | Count of cohort members in global top-50 |
| `Cohort-SP-in-Top50` | Binding SP from cohort members in global top-50 |
| `Cohort-Recall@50` | Fraction of cohort's total binding SP captured in global top-50 |
| `Cohort-Miss-Rate` | Fraction of cohort's binders ranked below position 100 |

**New Binder metrics** (computed for history-zero cohort, the Model A target population):

| Metric | What it measures |
|--------|-----------------|
| `NB_Capture@K` | Fraction of history-zero binders found in global top-K (K=50, 100) |
| `NB_Count@K` | Raw count of history-zero binders in global top-K |
| `NB_SP@K` | Dollar value of history-zero binders surfaced in global top-K |
| `NB_Total` | How many history-zero constraints actually bound this quarter (denominator) |
| `NB_Median_Rank` | Median rank position of history-zero binders in global ranking. Even when NB_Capture@50=0, this shows whether they're at position 60 (almost made it) or 500 (model is blind). |

**Dual-model complementarity metrics** (Phase 2 experiment 6 only):

| Metric | What it measures |
|--------|-----------------|
| `Complement_A_not_B@K` | Binders in Model A's top-K that are NOT in Model B's top-K |
| `Complement_B_not_A@K` | Binders in Model B's top-K that are NOT in Model A's top-K |
| `Blend_NB_Capture@K` | NB_Capture@K for the blended output (must exceed both A and B individually) |
| `Head_to_Head_NB` | For each history-zero binder: rank in Model A vs rank in Model B |

These metrics are what stage5-tier's v13/v14 experiments were missing — they built dual models but only measured aggregate VC@20/NDCG, making it impossible to tell whether the structural model actually helped rank new binders.

### Why This Matters for Architecture Decisions

If Phase 2 shows:
- **History-zero binders exist AND are badly ranked AND collectively valuable (>$50k SP/quarter)** → dual model justified (Model A can boost them)
- **History-zero binders are rare (<5/quarter) or low-value** → dual model is not justified; focus on single model
- **History-dormant constraints appear but are mispriced** → shadow_rank needs augmentation, not a separate model
- **Established constraints dominate top-50** → single model is fine, dual model is unnecessary complexity
- **Cross-peak constraints are well-handled** → offpeak BF already solves this (v1 finding confirmed)

The cohort analysis drives architecture decisions. It's not an afterthought — it's the diagnostic that determines whether we need Model A/B or just a better single model. But the cohort definition alone is not sufficient — **the decision also requires measuring cohort size, binding rate, and total SP value** before committing to architecture.

### Interaction with Universe Change

Switching to V4.4 **increases the history-zero population** (more constraints in the universe, many with shadow_price_da=0 and bf=0 because they've never appeared in DA). This could:
- **Dilute BF features**: More rows where bf=0, making BF less discriminating
- **Increase the value of prediction features**: More constraints where percentile/deviation features are the only signal
- **Require recalibration**: The ratio of established-to-new changes, affecting tier boundaries

---

## 6. Ground Truth Mapping Impact

### Current Ground Truth Pipeline (ground_truth.py)

```
Step 1: Fetch realized DA shadow prices for target quarter
  - DA returns (constraint_id, shadow_price) per market month
  - Aggregate: sum(abs(shadow_price)) per constraint_id → "per_cid"

Step 2: Map DA constraint_id → branch_name via bridge table
  - Bridge: MISO_SPICE_CONSTRAINT_INFO (filtered to auction_type='annual', target partition)
  - LEFT JOIN per_cid to bridge on constraint_id
  - **CRITICAL**: Unmapped constraint_ids are DROPPED (line 112)
  - Multiple constraint_ids can map to the same branch_name → aggregated

Step 3: Attach to signal universe
  - LEFT JOIN realized DA (by branch_name) onto signal DataFrame
  - Constraints without a match get realized_shadow_price = 0.0
```

### The False Negative Problem

The DA gap (34% of binding SP from 186 branches with no bridge entry) creates **false negatives in training labels**:

1. A V4.4 constraint (equipment="FOO") actually binds in DA under constraint_id=12345
2. But constraint_id 12345 has no entry in the bridge table for this partition
3. The binding event is dropped at Step 2
4. "FOO" gets labeled `realized_shadow_price = 0` — a false negative

**Why the model is still usable despite this:**

| Factor | Explanation |
|--------|-------------|
| **Consistent noise** | Both train and eval use the same GT pipeline → relative rankings are valid |
| **V4.4 has good bridge coverage** | V4.4 constraints are built from spice6, which IS the bridge table source |
| **Bounded impact** | 34% unmapped SP is across ALL DA; subset affecting V4.4 is smaller |
| **Ceiling-limited, not corrupted** | Reported metrics underestimate true performance |

**Remaining risk**: A constraint that binds every quarter but whose DA constraint_id never has a bridge entry gets persistently mislabeled as "never binds". This is a systematic false negative. Should be rare but is worth diagnosing.

**v2 mitigation**: Add a diagnostic that measures the unmapped rate **within V4.4's universe** specifically. Log which V4.4 equipment names get zero realized SP but might be binding via unmapped DA constraint_ids.

### With V4.4

The mapping chain is the same (V4.4 also uses `equipment` which equals `branch_name`), but the larger universe means:

1. **More constraints get ground truth labels** — V4.4 covers 68% of binding SP vs V6.1's 38%
2. **More constraints get realized_shadow_price = 0** — V4.4 has 1227 rows but only ~250 bind (via bridge). The other ~977 get 0.
3. **Class imbalance increases** — ~20% binding rate (250/1227) vs ~36% in V6.1 (144/395). This affects tiered label construction.

### Universe Expansion Potential

See §3 "Beyond V4.4" for the score-gated expansion analysis (V4.4 + score>0.001 = 1,926 constraints, ~72.5% SP coverage) and §7 Phase 3a for the implementation plan. This is gated by Phase 2 evidence.

---

## 7. Implementation Order

The phases are ordered by information value: each phase produces evidence that determines whether the next phase is needed and how it should be scoped.

### Phase 1: Foundation (V4.4 baseline + metric framework)

**Goal**: Get running on V4.4 with proper metrics. Establish baselines. Diagnose cohort-level failures.

**No model architecture decisions yet** — Phase 1 produces the diagnostic data that drives Phase 2/3 decisions.

1. **Build data loader** (`load_v44_enriched()`)
   - Parse constraint_id from `__index_level_0__`
   - Map equipment → branch_name equivalence
   - Verify ground truth join works on V4.4
   - **Design for extensibility**: the loader should accept a universe DataFrame so that Tier B/C universes can plug in later without rebuild

2. **Run baselines on V4.4**
   - **v0b**: pure shadow_rank only (establishes what DA history alone achieves)
   - **v16-equiv**: shadow_rank + BF features (bf_6/12/15, bfo_6/12)
   - Verify BF computation works with V4.4's larger universe
   - Run on both dev (12 groups) and holdout (4 groups)

3. **Implement metric framework**
   - Tier 1 blocking gates (VC@50/100, Recall@50/100, NDCG)
   - Tier 2 monitoring (VC@20, Spearman, etc.)
   - **Tier 3 cohort contribution** (global shortlist analysis — see §5)
   - Per-group breakdown for all holdout quarters
   - Coverage ceiling metric (fraction of DA binding SP in universe)

4. **Run cohort diagnostics on v16-equiv**
   - Assign every constraint to a cohort (history-zero, history-dormant, cross-peak, established)
   - Measure cohort contribution to global top-50 and top-100
   - **Monthly cross-reference**: For each annual history-zero constraint, check monthly realized DA (2017-04+) and V6.2B monthly signals. Classify as "truly zero" vs "annual-zero-but-monthly-active".
   - **Lookback window analysis**: Compute `months_since_last_binding` (across annual + monthly DA combined) for all constraints. Plot the distribution, identify natural clusters, and set two thresholds (short ~6-12mo, long ~24-36mo) from data rather than guessing.
   - **This is the key decision input**: if history-zero constraints are badly ranked, dual model is justified. If they're rare and low-value, single model suffices.

5. **GT false-negative diagnostic**
   - Measure unmapped rate within V4.4's universe
   - Log which V4.4 equipment names get zero realized SP but might be binding via unmapped DA constraint_ids

**Phase 1 outputs**: V4.4 baseline numbers on all metrics, cohort-level failure analysis, GT quality assessment. These determine Phase 2 scope.

### Phase 2: Feature Exploration

**Goal**: Determine which features matter, whether V4.4's pre-computed features are sufficient, and whether prediction features can rank constraints independently of history.

#### Feature Classification

| Type | Features | Count | Signal for history-zero? |
|------|----------|:-----:|:-:|
| **Historical** | `shadow_rank` (ranked historical DA shadow price) | 1 | No (shadow_price_da = 0 by definition) |
| **Historical (engineered)** | `bf_6`, `bf_12`, `bf_15`, `bfo_6`, `bfo_12` | 5 | No (zero by definition) |
| **Predictive (V4.4)** | `{0,60,65,70,75,80,85,90,95,100}_{max,sum}` (percentile features from SPICE flow simulations) | 20 | **Yes** |
| **Predictive (V4.4)** | `deviation_max_rank`, `deviation_sum_rank`, `deviation` | 3 | **Yes** |
| **Predictive (spice6 join)** | `density_signal_score`, `constraint_limit` | 2 | **Yes** |

V4.4's percentile features are normalized probabilities [0, 1] from forward-looking power flow simulations — a compressed version of spice6's 77-bin density distribution. They exist for **every** constraint regardless of binding history.

#### Experiments (in order)

1. **Single model with all V4.4 features + BF + density_signal_score**
   - ~30 features total. Does a single model with V4.4's richer feature set still drown prediction features?
   - Compare feature importance: if prediction features get >20% importance (vs <5% in v1), the "BF drowns everything" problem may be solved by having 23 prediction features instead of V6.1's 2.
   - **If prediction features get meaningful importance → single model may suffice, dual model unnecessary**

2. **Prediction-only model (Model A standalone)**
   - ~24 features: deviation + percentiles + density_signal_score + constraint_limit. Zero BF, zero shadow_rank.
   - **Key test**: does Model A beat pure shadow_rank baseline? If yes, SPICE flow predictions have real ranking power.
   - Run cohort contribution analysis — does Model A place history-zero binders in global top-50?

3. **V4.4 percentiles vs raw spice6 density aggregation**
   - Take V4.4's 20 percentile features vs custom-aggregated features from raw 77-bin density
   - Custom options: tail statistics (p95, p99 of binding probability), skewness, seasonal weighting
   - **If V4.4 percentiles ≈ custom → no need for Tier C infrastructure**
   - **If custom >> V4.4 → worth the engineering for Tier C**

4. **Feature selection among the 20 percentile columns**
   - Likely highly correlated (e.g., 90_max ≈ 95_max). Measure VIF or mutual information.
   - Prune to ~8-12 features if redundancy is high.

5. **Monthly binding frequency as cross-domain feature**
   - Compute `monthly_bf_N` (has this constraint bound in monthly realized DA in last N months?) using short and long lookback windows from Phase 1's distribution analysis.
   - This is cheap to add (realized DA already loaded) and directly addresses annual-history-zero constraints that have monthly signal.
   - Compare feature importance and cohort contribution with/without `monthly_bf`.

6. **Dual model: prediction-only Model A + full Model B → blend**
   - **This is a definite experiment**, not conditional on other Phase 2 results. The goal is a model that can rank constraints independently of annual binding history, specifically to find new binders.
   - **Model A (prediction-only)**: ~24 features — deviation, percentiles, density_signal_score, constraint_limit, monthly_bf. Zero annual BF, zero shadow_rank.
   - **Model B (full)**: All features including annual BF and shadow_rank (the single-model from experiment 1).
   - **Blend**: Start with rank-average, then test cohort-weighted (weight Model A higher for history-zero/history-dormant).
   - **Evaluation**: Head-to-head ranking of individual history-zero binders (where does Model A rank them vs Model B?). Complementarity analysis: count binders in Model A's top-50 that are NOT in Model B's top-50. Global shortlist contribution by cohort for the blended output.
   - **Decision criterion**: Blend must improve Cohort-Recall@50 for history-zero WITHOUT degrading overall VC@50 by >5%.
   - **Note**: Model A's aggregate metrics (VC@50, NDCG) may be poor — that's expected. Judge it by per-cohort metrics and complementarity, not by global rank quality.

**Phase 2 outputs**: Feature importance analysis, Model A standalone performance, dual-model blend results, monthly_bf value assessment, V4.4 vs custom aggregation comparison, cohort contribution with/without prediction features. These determine Phase 3 scope.

### Phase 3: Universe Expansion & Custom Features (evidence-driven)

**Conditional on Phase 2 results.** Only proceed with the experiments that Phase 2 justifies. Note: dual model is now a Phase 2 experiment (experiment 6), not conditional here.

#### 3a: Score-Gated Universe Expansion (Tier B)

Expand V4.4's 1,227 constraints to ~1,926 via density_signal_score > 0.001 filter.

**The problem**: Expanded rows lack V4.4's 20 percentile features and 3 deviation features. Training a single model on all 1,926 rows would require imputing 23 features for 699 rows — a major distribution mismatch that could produce OOD-scoring artifacts.

**Evaluation contract** (must be followed to produce interpretable results):

1. **Train on V4.4 rows only** (1,227 constraints with full features). Do NOT train on expanded rows.
2. **Score expanded rows with explicit missing-value encoding**:
   - Expanded rows have: density_signal_score, constraint_limit, BF features (bf_6/12/15, bfo_6/12), shadow_rank (if the constraint has any DA history — many will have shadow_rank=0 legitimately).
   - Expanded rows are **missing**: all 20 V4.4 percentile features and 3 deviation features (23 columns total).
   - **Encoding**: Use `np.nan` (NOT zero) for the 23 missing features. LightGBM routes NaN to the optimal child at each split, which is fundamentally different from encoding 0 (which LightGBM treats as a real value and routes deterministically). Using 0 would create false signal — many V4.4 rows have legitimate 0 values in percentile columns, so the model would conflate "missing because not in V4.4" with "in V4.4 but zero probability."
   - **Monitor**: Log how many splits in the trained model use the 23 missing features. If >50% of tree splits rely on features that are NaN for expanded rows, the model is not generalizing to them.
3. **Evaluate on two populations separately**:
   - **V4.4 population** (1,227): standard metrics (VC@50, Recall@50, etc.) — must not regress vs Phase 1 baseline
   - **Expanded population** (699): count of binders found, SP captured, and where they rank in the global merged list
4. **The gate is NOT a single ">$50k" number.** The gate is: (a) V4.4 population metrics do not regress, AND (b) expanded population contributes ≥N binders in the global top-100 that were previously invisible.

This separates real signal gain (finding new binders) from scoring artifacts (expanded rows getting high scores due to missing feature patterns).

#### 3b: Custom Aggregation (Tier C — only if Phase 2 shows custom >> V4.4 percentiles)

If Phase 2 experiment 3 shows meaningful improvement from custom density aggregation:

- Build aggregation pipeline from raw spice6 density (77 bins × outage_dates → annual features)
- Design choices: temporal weighting, tail statistics, constraint-specific vs uniform
- Apply to both V4.4 constraints and expanded universe
- This is the highest-engineering-cost path — only justified by clear Phase 2 evidence

### Decision Tree Summary

```
Phase 1 → cohort diagnostics + monthly cross-reference + lookback windows
  ├── History-zero binders rare & low-value → focus on established ranking only
  └── History-zero binders valuable → proceed to Phase 2
        │
Phase 2 → feature exploration + dual model (always run)
  ├── Experiments 1-5: feature importance, Model A standalone, monthly_bf, custom vs V4.4
  ├── Experiment 6: dual model blend (definite — run regardless of experiment 1-5 results)
  │     └── Evaluate by complementarity + per-cohort metrics, not aggregate metrics
  └── Evidence from experiments 1-6 determines Phase 3 scope:
        ├── Universe expansion worthwhile → Phase 3a (Tier B)
        └── Custom aggregation >> V4.4 percentiles → Phase 3b (Tier C)
```

Phase 3 is purely about universe and feature infrastructure — architecture (single vs dual model) is resolved in Phase 2.

---

## 8. Risk Register

| Risk | Impact | Mitigation | Phase |
|------|--------|------------|:-----:|
| V4.4 is a TEST signal — may not be production-stable | Results not deployable | Verify with signal production team before Phase 2 | 1 |
| 61 V6.1 constraints not in V4.4 — could include important binders | Lost coverage | Quantify binding value of the 61 before dropping V6.1 | 1 |
| V4.4's larger universe dilutes BF features | Established-cohort accuracy drops | Phase 1 cohort diagnostic measures this directly | 1 |
| shadow_price_da scale difference (cumulative vs avg) | Feature importance shifts | Use shadow_rank (ranked, scale-invariant) — never raw shadow_price_da | 1 |
| 4 holdout groups still too few for reliable gating | Overfit to 2025 pattern | Report per-group; use 3-of-4 passing rule | 1 |
| Ground truth mapping gap (34% of binding SP unmapped) | Ceiling on metrics + false negative labels | Phase 1 GT diagnostic; report ceiling alongside metrics; data infra issue | 1 |
| Dual model adds complexity for marginal gain | Blend may not improve over single model | Phase 2 experiment 6 measures complementarity directly; drop blend if no new binders found | 2 |
| 20 V4.4 percentile features highly correlated | Model instability, overfitting | Phase 2 feature selection / VIF analysis before architecture decisions | 2 |
| Custom aggregation not better than V4.4 percentiles | Tier C engineering wasted | Phase 2 head-to-head comparison gates Tier C decision | 2 |
| Score-gated expansion rows lack V4.4 features | Heterogeneous feature space; must use NaN not zero | Phase 3a scoring contract specifies NaN encoding + split monitoring | 3 |
| Backward incompatibility — v0-v16 results non-comparable | Can't track progress across universe change | Clean registry break; re-run key baselines on V4.4 | 1 |

---

## Appendix A: Key File Paths

### Signal Data (curated, pre-ranked)

| Component | Path |
|-----------|------|
| V6.1 signal (current) | `/opt/data/xyz-dataset/signal_data/miso/constraints/Signal.MISO.SPICE_ANNUAL_V6.1/` |
| V4.4 signal (proposed) | `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V4.4.R1/` |
| V4.5 signal (subset of V4.4) | `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V4.5.R1/` |
| V6.2B monthly | `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1` |

### Spice6 Pipeline Data (raw ML outputs)

| Component | Path | Schema |
|-----------|------|--------|
| Bridge table | `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_INFO.parquet` | 29 cols, partitioned by spice_version/auction_type/auction_month/market_round |
| Density signal score | `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_DENSITY_SIGNAL_SCORE.parquet` | 3 cols (constraint_id, flow_direction, score), 12,841 constraints |
| Density distribution | `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_DENSITY_DISTRIBUTION.parquet` | 78 cols (constraint_id + 77 price bins), 12,841 constraints |
| Constraint limits | `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_LIMIT.parquet` | 2 cols (constraint_id, limit), 14,007 constraints |

**pbase loaders**: `MisoSpiceDensitySignalScoreOutput`, `MisoSpiceDensityDistributionOutput`, `MisoSpiceConstraintLimitOutput` in `pbase.data.dataset.spice.pipeline`. Partitioned by `spice_version/auction_type/auction_month/market_month/market_round/outage_date`.

### Other

| Component | Path |
|-----------|------|
| Spice6 density (legacy) | `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/` |
| Spice6 constraint info (legacy) | `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/constraint_info/` |
| Realized DA cache | `/home/xyz/workspace/research-qianli-v2/research-stage5-tier/data/realized_da/` |
| Config | `research-annual-signal/ml/config.py` |
| Data loader | `research-annual-signal/ml/data_loader.py` |
| Ground truth | `research-annual-signal/ml/ground_truth.py` |
| Binding frequency | `research-annual-signal/ml/binding_freq.py` |
| Evaluation | `research-annual-signal/ml/evaluate.py` |
| v16 champion config | `research-annual-signal/registry/v16_champion/config.json` |

## Appendix B: density_signal_score Independence Analysis

The spice6 `density_signal_score` is NOT already captured by V4.4's existing features:

| Pair | Spearman ρ | Interpretation |
|------|:---:|---|
| score ↔ V4.4 rank | 0.48 | Moderate — not redundant |
| score ↔ 100_max | 0.42 | Moderate — different facet of binding probability |
| score ↔ shadow_rank | -0.23 | Weak negative — genuinely independent |

V4.4's formula: `rank ≈ 0.25*shadow_rank + 0.38*deviation_max_rank + 0.57*deviation_sum_rank` (R²=0.9815). Adding `score` does NOT improve R² — confirming score is not already in V4.4.

This means density_signal_score could serve as an additional feature in Model A (prediction-only) or as a filter for universe expansion.
