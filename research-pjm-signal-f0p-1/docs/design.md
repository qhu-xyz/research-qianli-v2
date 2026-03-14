# PJM V7.1 — Expanded Universe Design

## 1. Problem Statement

V6.2B filters ~11,600 raw density constraints down to ~475 branches, missing **65% of binding constraints** and **53% of binding value**. No ML model can recover constraints that aren't in the universe. The biggest opportunity is expanding the universe, not improving the model.

### What This Project Changes vs f0p-0

| Aspect | f0p-0 (V7.0b) | f0p-1 (V7.1) |
|--------|----------------|---------------|
| Universe source | V6.2B signal (~475 branches) | Raw spice6 density + ml_pred (~3,100 branches) |
| Features | V6.2B columns + spice6 enrichment | Built from raw spice6 data directly |
| Metrics | VC@k on fixed universe | Cross-universe comparable metrics |
| NB tracking | BF-zero cohort only | Formal NB metrics suite |
| Mapping quality | Assumed good | Explicitly tracked and reported |

## 2. Data Pipeline

### 2.1 Universe Construction

For each (auction_month, period_type, class_type):

```
Step 1: Load raw constraint universe
  - score.parquet: ~11,600 constraint_ids × 2 flow_dirs = ~23,200 rows
  - Aggregate across 11 outage_dates: mean(score) per (constraint_id, flow_direction)

Step 2: Map to branches via constraint_info
  - constraint_info: ~18,000 cids → ~4,800 branches
  - Many-to-one: multiple constraint_ids share one branch_name
  - Aggregate per branch: take max score across contingencies
  - Result: ~3,100 branches with density score

Step 3: Enrich with features
  - ml_pred: join on (constraint_id, flow_direction), aggregate to branch level
    Available features (SAFE): binding_probability, predicted_shadow_price,
    hist_da, prob_exceed_{80,85,90,95,100,105,110}
    DO NOT USE: actual_shadow_price, actual_binding, error, abs_error
  - limit.parquet: constraint_limit, join on constraint_id, aggregate to branch
  - density.parquet: extract custom percentile features if needed

Step 4: Join realized DA ground truth (for training only)
  - Existing cache: data/realized_da/ (86 months, branch_name → realized_sp)
  - Join on branch_name, fill missing with 0.0

Step 5: Compute binding frequency features
  - Same as f0p-0: BF at windows {1, 3, 6, 12, 15} months
  - Lag rules per CLAUDE.md (fN → lag N+1)
```

### 2.2 Feature Candidates

#### From density (score.parquet, density.parquet):
| Feature | Description | Source |
|---------|-------------|--------|
| `density_score` | Mean binding probability across outage_dates | score.parquet |
| `density_score_max` | Max binding probability across outage_dates | score.parquet |
| `density_ev` | Expected thermal loading from distribution | density.parquet |
| `density_std` | Std of thermal loading distribution | density.parquet |
| `density_pe100` | P(exceed 100% limit) from distribution | density.parquet |

#### From ml_pred:
| Feature | Description | Source |
|---------|-------------|--------|
| `binding_probability` | ML-predicted probability of binding | ml_pred |
| `predicted_shadow_price` | ML-predicted shadow price | ml_pred |
| `hist_da` | Historical DA shadow price metric | ml_pred |
| `prob_exceed_100` | P(flow exceeds 100% of limit) | ml_pred |
| `prob_exceed_110` | P(flow exceeds 110% of limit) | ml_pred |

#### From limit.parquet:
| Feature | Description | Source |
|---------|-------------|--------|
| `constraint_limit` | Physical thermal limit (MW) | limit.parquet |

#### Computed features (historical — need lag):
| Feature | Description | Source |
|---------|-------------|--------|
| `bf_{1,3,6,12,15}` | Binding frequency at N-month windows | realized_da cache |
| `hist_shadow_price_da` | Historical avg shadow price from DA | ml_pred `hist_da` |

### 2.3 Aggregation Strategy (KEY DESIGN QUESTION)

The constraint universe has ~3,100 branches. That's 7x the V6.2B universe. Adding all to training is fine for LightGBM (it handles sparse data well), but raises concerns:

**Option A: Full universe (~3,100 branches)**
- Pro: Maximum coverage, no filtering bias
- Pro: Model learns which branches are irrelevant (score 0)
- Con: ~88% of branches never bind — lots of noise
- Con: Training set is 8 months × 3,100 = ~25,000 rows (still manageable)

**Option B: Filtered universe (~1,000-2,000 branches)**
- Filter: keep branches where `density_score > 1e-10` OR `ml_pred.binding_probability > 0.01` OR branch has any historical binding
- Pro: Removes obviously irrelevant branches (zero-score, zero-probability)
- Con: Risk filtering out surprise binders
- Con: Filter threshold is arbitrary

**Option C: Tiered universe**
- Core tier (~500): V6.2B-equivalent branches — always included, high confidence
- Extended tier (~1,000-2,000): branches with some signal but not in V6.2B
- Use different training strategies per tier (e.g., more regularization for extended)

**Recommendation: Start with Option A, then experiment with B.**
At 25k rows per training fold, LightGBM trains in <1 second. The cost of including noise branches is minimal. We can always filter at inference time (only output tiers for branches with score > threshold).

### 2.4 V6.2B Feature Reconstruction

To do apples-to-apples comparison, we need V6.2B features on the expanded universe.
V6.2B formula: `rank_ori = 0.60 * da_rank_value + 0.30 * density_mix_rank_value + 0.10 * density_ori_rank_value`

**Problem**: `da_rank_value`, `density_mix_rank_value`, `density_ori_rank_value` are V6.2B-computed ranks. We don't have them for the ~2,600 branches outside V6.2B.

**Solution**: Reconstruct from raw data:
- `density_ori_rank_value` ≈ rank of `density_score` (= ori_mean in V6.2B)
- `da_rank_value` ≈ rank of `hist_da` from ml_pred (= shadow_price_da in V6.2B)
- `density_mix_rank_value` — unclear source, investigate. May be a blend of density scores.

For cross-universe comparison, we use the **raw features** not the ranks. Ranks are universe-dependent (rank 5 out of 475 ≠ rank 5 out of 3,100).

## 3. Metrics Design

### 3.1 Core Problem: Cross-Universe Comparison

When universe sizes differ (475 vs 3,100 branches), absolute metrics like VC@20 aren't directly comparable:
- VC@20 on 475 branches: model selects top 20 out of 475
- VC@20 on 3,100 branches: model selects top 20 out of 3,100 (harder)

And VC@20 means different things:
- On V6.2B universe: captures X% of **V6.2B-visible** value
- On expanded universe: captures X% of **all** value (more total value available)

### 3.2 Metric Categories

#### Category 1: Universe-Quality Metrics (not model-dependent)
Measure how much binding reality the universe captures, before any model scoring.

| Metric | Definition | Purpose |
|--------|-----------|---------|
| `Univ_BinderCoverage` | # binders in universe / # total binders in DA | How many binding branches does the universe contain? |
| `Univ_ValueCoverage` | sum(realized_sp for branches in universe) / sum(all DA realized_sp) | How much binding $ does the universe contain? |
| `Univ_Size` | # branches in universe | Size tracking |
| `Univ_BinderRate` | # binders / # branches | Density of signal |

#### Category 2: Model-Quality Metrics (universe-independent)
Evaluate model ranking quality, normalized to be comparable across universe sizes.

| Metric | Definition | Purpose |
|--------|-----------|---------|
| `VC@20` | Value captured by top-20 model picks / total value in universe | Standard value capture (top-20) |
| `VC@50` | Value captured by top-50 / total value in universe | Standard value capture (top-50) |
| `VC@k%` | Value captured by top-k% of universe / total value | **Percentile-based** — comparable across sizes (e.g., VC@5% means top-5% of universe) |
| `Recall@20` | Overlap of model top-20 with true top-20 / 20 | Precision at fixed k |
| `Recall@k%` | Overlap of model top-k% with true top-k% / k% | Percentile-based recall |
| `NDCG` | Standard NDCG | Full-ranking quality |
| `Spearman` | Spearman rank correlation | Monotonic relationship |
| `AP@20` | Average Precision for "is this in the true top-20?" | Ranking quality at head |

#### Category 3: Absolute-Value Metrics (the ones that matter for trading)
What $ value does the model actually capture? Not relative to universe — absolute.

| Metric | Definition | Purpose |
|--------|-----------|---------|
| `AbsVC@20` | sum(realized_sp for model's top-20 picks) | Absolute $ captured by top-20 |
| `AbsVC@50` | sum(realized_sp for model's top-50 picks) | Absolute $ captured by top-50 |
| `AbsVC@20_share` | AbsVC@20 / sum(all DA realized_sp) | Top-20 capture as share of ALL DA value |
| `AbsVC@50_share` | AbsVC@50 / sum(all DA realized_sp) | Top-20 capture as share of ALL DA value |

These are the most meaningful for cross-universe comparison:
- V6.2B VC@20 = 47% of V6.2B value, but V6.2B only sees 47% of total → **actual capture = 22%**
- Expanded VC@20 = 35% of expanded value, expanded sees 89% of total → **actual capture = 31%**

The expanded universe wins despite lower VC@20 because it sees more value.

#### Category 4: New-Binding (NB) Metrics
Track model performance on constraints with NO binding history.

| Metric | Definition | Purpose |
|--------|-----------|---------|
| `NB_n` | Count of BF-zero branches in universe | Size of NB cohort |
| `NB_n_binders` | Count of NB branches that actually bind | How many surprises? |
| `NB_value_share` | NB binding value / total binding value | How much $ comes from surprises? |
| `NB_VC@20` | Value captured by model's top-20 NB picks / total NB value | Can model rank NB branches? |
| `NB_Recall@20` | Overlap of model NB-top-20 with true NB-top-20 / 20 | NB ranking precision |
| `NB_AbsVC@20` | Absolute $ captured by model's top-20 NB picks | Real NB value captured |

NB is defined as BF-zero at lookback=6 months (same as f0p-0).

#### Category 5: Mapping Quality / Data Quality
Track how well our branch mapping and data pipeline work.

| Metric | Definition | Purpose |
|--------|-----------|---------|
| `Map_DA_coverage` | # DA binders mapped to a branch / # total DA binders | Branch mapping completeness |
| `Map_DA_value_coverage` | mapped DA value / total DA value | Branch mapping value coverage |
| `Map_CI_coverage` | # density cids with constraint_info match / # total density cids | constraint_info completeness |
| `Map_MLpred_coverage` | # branches with ml_pred features / # branches in universe | ml_pred feature availability |
| `Map_Limit_coverage` | # branches with limit data / # branches in universe | limit data availability |

### 3.3 Comparison Framework

To compare Model A (on Universe X) vs Model B (on Universe Y):

**Primary comparison**: AbsVC@20_share, AbsVC@50_share (absolute value captured, normalized by total DA value). These are directly comparable regardless of universe size.

**Secondary comparison**: Universe-quality metrics (does the bigger universe actually help?) + VC@k% (model quality within its universe).

**Reporting format** (per month):
```
Month: 2023-06, f0/onpeak
                  V6.2B+v0b    Expanded+v0    Expanded+ML
Univ_Size         475          3100           3100
Univ_BinderCov    33%          89%            89%
Univ_ValueCov     47%          89%            89%
VC@20             0.47         0.18           0.32
AbsVC@20          $45,200      $32,100        $57,500
AbsVC@20_share    22%          29%            51%
NB_n_binders      --           95             95
NB_AbsVC@20       --           $3,200         $8,100
```

## 4. Experiment Plan

### Phase 0: Data Pipeline & Mapping Audit
1. Build expanded universe loader (score.parquet → constraint_info → branch-level)
2. Measure mapping quality metrics for every month
3. Verify realized DA coverage on expanded universe
4. Symlink realized DA cache from f0p-0 (don't rebuild)

### Phase 1: Baseline Comparisons
1. **v0-v62b**: V6.2B formula on V6.2B universe (reproduce f0p-0 v0b results)
2. **v0-expanded**: V6.2B-equivalent formula on expanded universe (rank by density_score)
3. **v0-expanded-da**: Rank by hist_da on expanded universe
4. Compare using AbsVC metrics — does universe expansion help even with a dumb model?

### Phase 2: ML on Expanded Universe
1. **v1**: LightGBM LambdaRank with V10E-equivalent features on expanded universe
2. **v2**: Add density distribution features (ev, std, pe100)
3. **v3**: Feature selection / ablation
4. Compare v1-v3 against v0-v62b using AbsVC metrics

### Phase 3: NB-Focused Analysis
1. Measure NB metrics for all versions
2. Test two-model blend (full model for BF-positive, pred-only for BF-zero)
3. Test universe filtering strategies (Option B/C from Section 2.3)

### Phase 4: Aggregation Experiments
1. Test branch-level vs constraint-level training
2. Test filtered universes (various thresholds)
3. Test tiered training strategies
4. Find the sweet spot: maximum coverage with minimum noise

## 5. Key Risks & Open Questions

### R1: Mapping Quality Degradation
constraint_info maps ~18k cids to ~4,800 branches, but ml_pred only covers ~3,100 branches. The gap may mean some branches lack features. Need to quantify per-month.

### R2: density_score ≠ ori_mean
V6.2B's `ori_mean` and raw `score` have Spearman ~0.85 (not identical). The aggregation method differs (V6.2B may weight outage_dates differently). Need to understand the gap.

### R3: Training Set Size
Full universe × 8 months = ~25,000 rows. Still manageable for LightGBM, but training signal is sparser (more zero-binding rows). Tiered labels should handle this.

### R4: hist_da Availability
`hist_da` from ml_pred is the strongest single feature (= shadow_price_da in V6.2B). It's available for ~3,100 branches (ml_pred coverage), not the full ~4,800 from constraint_info. Branches without hist_da will have weaker features.

### R5: Realized DA Cache Reuse
f0p-0's realized DA cache (86 months) is branch-level, mapped via constraint_info. We can symlink it — no need to rebuild. But we should verify that the branch mapping is consistent between f0p-0 and the expanded universe construction.

### Q1: What is `mix_mean` / `density_mix_rank_value`?
V6.2B has `mix_mean` (30% weight in formula). This appears to be a blend of density scores, possibly averaging across outage_dates differently than `ori_mean`. Need to investigate the density_multi.parquet and the aggregation logic.

### Q2: Should we filter by flow_direction?
score.parquet has both flow_direction=1 and flow_direction=-1. V6.2B appears to keep both. ml_pred has both. Realized DA is direction-agnostic (aggregated by branch). How to handle: keep the max-scoring flow_direction per branch? Or keep both as separate candidates?

### Q3: Score aggregation across outage_dates
11 outage_dates per month. We take mean(score), but should we take max? Or a weighted blend? This is the "aggregation" question the user flagged.

## 6. Directory Structure

```
research-pjm-signal-f0p-1/
├── CLAUDE.md
├── docs/
│   ├── design.md              (this file)
│   └── knowledge.md           (data pipeline knowledge)
├── human-input/               (user context, requirements)
├── ml/                        (ML pipeline modules)
│   ├── config.py              (features, paths, schedules)
│   ├── universe_loader.py     (expanded universe construction)
│   ├── data_loader.py         (training data: universe + GT)
│   ├── features.py            (feature extraction & prep)
│   ├── train.py               (LightGBM training)
│   ├── evaluate.py            (new metrics suite)
│   ├── compare.py             (cross-universe comparison)
│   ├── branch_mapping.py      (constraint_id → branch_name)
│   ├── realized_da.py         (DA cache, reuse from f0p-0)
│   └── registry_paths.py      (versioned results)
├── scripts/                   (experiment scripts)
├── registry/                  (versioned results)
└── data/
    └── realized_da -> symlink to f0p-0/data/realized_da
```
