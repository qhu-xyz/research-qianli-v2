# Class-Specific Annual Signal Pipeline Design

**Date**: 2026-03-15
**Status**: Draft
**Depends on**: Phase 5 combined-ctype baseline (archived), V6.1 class-type analysis

---

## 1. Problem Statement

All Phase 3-5 models were trained on combined class_type data:
- Target: `realized_shadow_price = onpeak_sp + offpeak_sp`
- BF: `bf_combined_12` (bound in either ctype)
- Cohort: dormant = `bf_combined_12 == 0`

V6.1 production signals are **genuinely class-specific** (verified):
- `shadow_price_da` differs up to $7,318 for same constraint across onpeak/offpeak
- Different constraint sets: 425 onpeak vs 413 offpeak (78% overlap) for 2024-06/aq1
- Different rankings and tiers: up to 3 tier difference for same constraint
- Root cause: `shadow_price_da` computed from historical DA separately per hour type

The combined-ctype model cannot reproduce this. We need separate onpeak and offpeak
pipelines to match production behavior and capture class-specific binding patterns.

## 2. What Changes vs What Stays

### 2.1 Must Change (class-specific)

| Component | Combined (current) | Onpeak | Offpeak |
|---|---|---|---|
| **Target** | `onpeak_sp + offpeak_sp` | `onpeak_sp` | `offpeak_sp` |
| **BF for formula** | `bf_combined_12` | `bf_12` | `bfo_12` |
| **Cohort: dormant** | `bf_combined_12 == 0` | `bf_12 == 0` | `bfo_12 == 0` |
| **NB flags** | `is_nb_12` (combined) | `nb_onpeak_12` | `nb_offpeak_12` |
| **V6.1 metadata** | N/A (loaded flat) | V6.1 onpeak partition | V6.1 offpeak partition |
| **shadow_price_da** | From V6.2B (flat) | Class-specific from V6.1 or DA | Class-specific |
| **da_rank_value** | Rank of combined spda | Rank of onpeak spda | Rank of offpeak spda |
| **Published signal** | N/A | Separate onpeak parquet | Separate offpeak parquet |

### 2.2 Stays the Same (class-agnostic)

| Component | Why unchanged |
|---|---|
| Density features (`bin_*_cid_max`) | SPICE simulation is outage-based, not hour-specific |
| `ori_mean`, `mix_mean` | Same as density |
| SF matrix | Physically determined; identical across class types |
| Bridge mapping (CID → branch) | Structural, not hour-specific |
| `count_cids`, `count_active_cids` | Structural |
| Universe threshold | Same SPICE density for both |
| Dedup logic | Same SF-based dedup |

### 2.3 Cross-Class Dormancy (new opportunity)

With class-specific cohorts, some branches change status:

| Cohort | Count (2024-06/aq1) |
|---|---|
| Both dormant (on + off) | 1,554 |
| Onpeak-dormant, offpeak-active | 60 |
| Offpeak-dormant, onpeak-active | 126 |
| Neither dormant | 599 |

186 branches (60 + 126) are dormant in one class but active in the other. The combined
model treats all of these as established — the class-specific model correctly puts them
in Track B for the dormant class, giving the NB model a chance to find them.

## 3. Cross-Class Features (new opportunity)

Cross-class history is almost as predictive as same-class history (verified):

| Feature → Target | Spearman |
|---|:---:|
| offpeak BF → offpeak SP (same-class) | 0.4368 |
| **onpeak BF → offpeak SP (cross-class)** | **0.4297** |
| onpeak BF → onpeak SP (same-class) | 0.4233 |
| **offpeak BF → onpeak SP (cross-class)** | **0.4270** |
| onpeak SP ↔ offpeak SP | **0.7958** |

Among class-dormant branches, cross-class binding is highly informative:
- 82.5% of offpeak-dormant binders also bind onpeak
- 62.0% of onpeak-dormant binders also bind offpeak
- 126 offpeak-dormant branches have `bf_12 > 0` (onpeak active)

**Feature set for each class-specific model**:

| Feature | Onpeak model | Offpeak model | Why |
|---|---|---|---|
| Same-class BF (`bf_12` / `bfo_12`) | `bf_12` | `bfo_12` | Primary BF signal |
| **Cross-class BF** | `bfo_12` | `bf_12` | Almost equal predictive power |
| Same-class `shadow_price_da` | onpeak spda | offpeak spda | Class-specific DA history |
| **Cross-class `shadow_price_da`** | offpeak spda | onpeak spda | Cross-class congestion signal |
| Same-class `da_rank_value` | onpeak rank | offpeak rank | Rank of same-class spda |
| `bf_combined_12` | Yes | Yes | Combined BF (backward compat) |
| Density bins | Same | Same | Class-agnostic |
| `ori_mean`, `mix_mean` | Same | Same | Class-agnostic |

**For the NB model**: cross-class BF is especially valuable. A branch that is offpeak-
dormant but onpeak-active (`bfo_12 == 0, bf_12 > 0`) is a much better NB candidate
than one dormant in both classes. Include cross-class BF as a feature.

## 4. v0c Formula Changes

Current (combined):
```python
score = 0.40 × norm(1 - da_rank_value) + 0.30 × norm(rt_max) + 0.30 × norm(bf_combined_12)
```

Onpeak:
```python
score = 0.40 × norm(1 - da_rank_value_onpeak) + 0.30 × norm(rt_max) + 0.30 × norm(bf_12)
```

Offpeak:
```python
score = 0.40 × norm(1 - da_rank_value_offpeak) + 0.30 × norm(rt_max) + 0.30 × norm(bfo_12)
```

`da_rank_value` becomes class-specific because it's the rank of class-specific
`shadow_price_da`. `rt_max` stays the same (density is class-agnostic).

## 4. NB Model Changes

Separate NB model per class_type:

**Onpeak NB model**:
- Population: `bf_12 == 0` AND `has_hist_da == True`
- Target: `onpeak_sp > 0`
- Sample weights: sqrt(onpeak_sp) or tiered by onpeak_sp
- Features (~17):
  - Density bins (11): class-agnostic, same as before
  - Same-class: `shadow_price_da_onpeak`, `da_rank_value_onpeak`, `historical_max_sp_onpeak`
  - **Cross-class: `bfo_12`, `shadow_price_da_offpeak`** — a branch active in offpeak
    is a strong NB candidate for onpeak (82.5% of offpeak-dormant binders also bind onpeak)
  - `count_active_cids`

**Offpeak NB model**:
- Population: `bfo_12 == 0` AND `has_hist_da == True`
- Target: `offpeak_sp > 0`
- Same feature structure with offpeak-specific values + onpeak cross-class features
- **Cross-class: `bf_12`, `shadow_price_da_onpeak`**

## 5. shadow_price_da Construction

Two options:

### Option A: Inherit from V6.1
- For constraints in V6.1's universe, use V6.1's class-specific `shadow_price_da`
- Pro: exact match to production baseline
- Con: only available for V6.1's ~400 constraints per (aq, ctype)

### Option B: Compute from realized DA
- For each constraint, aggregate historical realized DA per class_type
- `shadow_price_da_onpeak = mean(abs(realized_sp_onpeak))` across historical months
- Pro: available for all constraints in our universe
- Con: may differ slightly from V6.1's SPICE-computed value

**Recommendation**: Option B for training (full universe coverage), with Option A
values for verification (confirm our computation matches V6.1 where they overlap).

## 6. Evaluation Changes

### 6.1 Separate Champion Selection

Run Phase 5 evaluation separately for onpeak and offpeak:
- Onpeak: v0c_onpeak + NB_onpeak blend, evaluated on onpeak_sp target
- Offpeak: v0c_offpeak + NB_offpeak blend, evaluated on offpeak_sp target

The blend α may differ between class types (onpeak might need α=0.05, offpeak α=0.10).

### 6.2 Metrics

Same metric framework as Phase 5:
- VC@K, Recall@K, NB12_SP@K, Dang_Recall@K at K=150/200/300/400
- Paired scorecard per class_type
- Dangerous threshold: $25k per class (half of combined $50k, since SP is split)

### 6.3 Ground Truth

For onpeak evaluation:
```python
realized_shadow_price = onpeak_sp  # NOT combined
label_tier = tiered_labels(onpeak_sp)  # tiers based on onpeak SP only
```

## 7. Implementation Plan

### 7.1 Module Changes

| File | Change |
|------|--------|
| `ml/config.py` | Add `CLASS_TYPES = ["onpeak", "offpeak"]`, class-specific BF column mapping |
| `ml/features.py` | Parameterize `build_model_table(class_type=...)` — use class-specific BF, target, cohort |
| `ml/ground_truth.py` | Add `class_type` param — return `onpeak_sp` or `offpeak_sp` as target |
| `ml/history_features.py` | Already has `bf_12` and `bfo_12` — just needs routing |
| `ml/nb_detection.py` | Already has `nb_onpeak_12` and `nb_offpeak_12` — just needs routing |
| `ml/evaluate.py` | No change — evaluation is target-agnostic |

### 7.2 New Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_phase6_class_specific.py` | Phase 6: class-specific champion selection |

### 7.3 Experiment Structure

```
For each class_type in [onpeak, offpeak]:

    Phase A: Baselines (rebuild from scratch)
    ─────────────────────────────────────────
    1. Build class-specific model table
       └─ target = {ctype}_sp
       └─ BF: same-class + cross-class (bf_12 + bfo_12)
       └─ cohort = bf_12==0 or bfo_12==0
       └─ shadow_price_da = class-specific from V6.1 or DA
       └─ cross-class features included

    2. v0 baselines:
       └─ v0a: da_rank_value_{ctype} only
       └─ v0c: 0.40×da_rank + 0.30×rt_max + 0.30×bf_{ctype}

    Phase B: ML + NB models
    ──────────────────────────
    3. v3a: LambdaRank with class-specific + cross-class features

    4. NB model: class-specific dormant, with cross-class BF/spda as features

    5. Blend sweep: v0c + α×NB for α ∈ {0.05, 0.10, 0.15, 0.20}

    Phase C: Evaluation + champion selection
    ─────────────────────────────────────────
    6. Evaluate all configs at K=150/200/300/400
       └─ class-specific target (onpeak_sp or offpeak_sp)
       └─ class-specific dangerous threshold ($25k per class)

    7. Paired scorecard per class_type → champion per class_type
```

### 7.4 Metrics (class-specific)

Dangerous threshold should be scaled: $25k per class_type (half of combined $50k,
since SP is split). Or keep $50k if the team considers that the meaningful threshold
regardless of class split — this is a business decision.

Evaluation target:
- Onpeak: `realized_shadow_price = onpeak_sp` (NOT combined)
- Offpeak: `realized_shadow_price = offpeak_sp`
- `label_tier` recomputed per class: tertiles of class-specific SP among binders

### 7.4 Reusable from Current Repo

| Component | Reusable? |
|---|---|
| `ml/data_loader.py` | Yes — density is class-agnostic |
| `ml/bridge.py` | Yes — structural |
| `ml/realized_da.py` | Yes — already loads per class_type |
| `ml/merge.py` | Yes — merge logic is target-agnostic |
| `ml/evaluate.py` | Yes — metric computation is target-agnostic |
| `ml/features_trackb.py` | Partially — `historical_max_sp` needs class-specific recompute |
| `scripts/run_phase5_reeval.py` | Template — fork and parameterize by class_type |

## 8. Success Criteria

1. **Onpeak champion** passes all gates vs onpeak v0c solo at (150, 300) and (200, 400)
2. **Offpeak champion** passes all gates vs offpeak v0c solo at same K pairs
3. Class-specific rankings differ meaningfully (not just noise)
4. Published onpeak and offpeak signals have different constraint sets and tiers
   (matching V6.1 behavior)

## 9. Relationship to Production Port

This design feeds directly into Project 1 (signal publication):

```
Class-Specific Pipeline (this doc)     →  Project 1 (publication)
─────────────────────────────────         ────────────────────────
1. Train onpeak model                     1. Score branches (onpeak v0c + NB)
2. Train offpeak model                    2. Expand to constraints
3. Evaluate separately                    3. Attach class-specific V6.1 metadata
4. Champion per class_type                4. Assign tiers
                                          5. Build SF from MISO_SPICE_SF.parquet
                                          6. Dedup
                                          7. Publish per (aq, class_type)
```

The current combined-ctype Phase 5 results are archived as baseline — not discarded,
but not used for production.
