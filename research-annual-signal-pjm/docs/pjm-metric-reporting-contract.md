# PJM Annual Signal — Metric and Reporting Contract

**Date**: 2026-03-25
**Status**: Proposed reporting contract for Phase 4 and Phase 5
**Goal**: use the same reporting discipline as MISO annual-signal while keeping PJM evaluation honest about GT coverage, universe coverage, and benchmark overlap

---

## 1. Reporting Principles

This contract follows the same structure already used in MISO:

- separate native performance from overlap-only ranking diagnostics
- separate GT mapping coverage from model-universe coverage
- report branch-level results, not constraint-level results
- evaluate with total DA SP as the honest denominator
- keep benchmark comparison and model comparison on the same annual publish cell
- evaluate reproducible baselines by economic capture, not closeness to V4.6 internals

All reported results are branch-level.

---

## 2. Base Grain

### 2.1 Publish / benchmark grain

All headline PJM results must be reported at:

- `planning_year`
- `market_round`
- `class_type`

This is the annual publish surface.

### 2.2 Internal diagnostic grain

Coverage and GT diagnostics must also retain:

- `aq_quarter`

This is required because PJM bridge mapping is quarter-sensitive.

Rule:

- annual metrics may be published at `(planning_year, market_round, class_type)`
- quarterly diagnostics must remain available underneath

PJM adjustment versus MISO:

- do not make quarter the user-facing default result grain
- do not force all reports through a quarter-first presentation just because GT is quarter-built internally

---

## 3. Mandatory Reporting Views

### A. Native Standalone View

Question:

- if this model selects its own top-K from its own native branch universe, how much real PJM DA SP does it capture?

Use for:

- model-vs-model comparison
- model-vs-V4.6 comparison
- champion selection

### B. Overlap-Only View

Question:

- where both models can score the same branches, whose ranking is better after reranking only on the shared overlap set?

Use for:

- pure ranking analysis between models
- diagnosing whether performance differences are ranking quality or universe coverage

### C. Deployment View

Question:

- if we apply our actual shortlist policy on our branch universe, what do we capture?

Use for:

- shipping decisions
- release candidate comparison
- policy/backfill/reserved-slot evaluation if introduced later

### D. Coverage View

Question:

- how much SP is lost before the model even gets a chance to rank?

Use for:

- data quality explanation
- honest ceiling analysis

---

## 4. K Levels

The default PJM reporting K levels should match the later MISO annual-signal reports, not the early `@50/@100` exploration.

### Primary K levels

- `K=200`
- `K=400`

These are the headline K levels for:

- native standalone
- overlap-only
- deployment

### Secondary monitoring K levels

- `K=20`
- `K=50`
- `K=100`
- `K=150`
- `K=300`

These are useful for curves and debugging, but not the primary decision points.

### NB-only K levels

If PJM later adds an NB specialist track, use:

- `K=50`
- `K=100`

on the NB-only universe.

---

## 5. Mandatory Metrics

### 5.1 Headline metrics at K=200 and K=400

For every annual cell `(planning_year, market_round, class_type)`:

- `VC@K`
  captured SP divided by mapped in-universe branch SP
- `Abs_SP@K`
  captured SP divided by total DA SP
- `Recall@K`
  captured positive-binding branches divided by all positive-binding branches in the branch universe
- `SP_Captured@K`
  raw dollar/SP amount captured by the shortlist
- `Binders@K`
  count of positive-binding branches in shortlist
- `Precision@K`
  `Binders@K / K`

Rule:

- `Abs_SP@K` is the primary honest metric
- `VC@K` is conditional and must never be shown alone

PJM adjustment:

- baseline gates should use annual-cell `Abs_SP@200/400` and coverage metrics first
- MISO-style heavy use of `NDCG` as a gate is less important than GT integrity in PJM's early phases

### 5.2 Dangerous branch metrics

MISO uses dangerous-branch reporting and PJM should too.

For every annual cell at `K=200` and `K=400`:

- `Dang_Recall@K`
  fraction of dangerous binders captured
- `Dang_SP_Ratio@K`
  dangerous SP captured divided by total dangerous SP
- `Dang_Count@K`
  count of dangerous branches in top-K
- `Dang_Total`
  total dangerous branches in the eval cell

PJM must define one explicit dangerous threshold in config. Until calibrated, use a clearly named provisional constant.

### 5.3 NB / dormant branch metrics

Even before an NB specialist exists, PJM should track whether general models ever surface dormant/NB branches.

For every annual cell at `K=200` and `K=400`:

- `NB12_Count@K`
- `NB12_Recall@K`
- `NB12_SP@K`

If the relevant cohort definitions differ for PJM, preserve the metric names but document the cohort logic explicitly.

If PJM later adds additional cohorts, also include:

- `NB6_Recall@K`
- `NB24_Recall@K`

PJM adjustment:

- treat NB metrics as monitoring metrics in the baseline branch ranker
- promote them to selection metrics only if PJM later adopts an NB-specialist or reserved-slot policy

### 5.4 Ranking diagnostics

These are monitoring metrics, not the main business decision metrics:

- `NDCG`
- `Spearman`

Keep them in `metrics.json`, but do not lead the report with them.

### 5.5 Benchmark delta metrics

Because the project goal is to beat V4.6 with reproducible inputs, every serious report should include direct deltas versus V4.6:

- `dSP_Captured@200`
- `dSP_Captured@400`
- `dAbs_SP@200`
- `dAbs_SP@400`
- `dDang_Recall@200`
- `dDang_Recall@400`
- `dNB12_SP@200`
- `dNB12_SP@400`

These are reported as:

- model minus V4.6
- on the same annual cell

---

## 6. Coverage Metrics

Coverage must be reported separately from ranking quality.

For every annual cell:

- `gt_mapping_coverage`
  `mapped_da_sp / total_da_sp`
- `model_universe_coverage`
  `in_universe_mapped_sp / mapped_da_sp`
- `combined_capture_ceiling`
  `in_universe_mapped_sp / total_da_sp`

Also include raw numerators / denominators:

- `total_da_sp`
- `mapped_da_sp`
- `in_universe_mapped_sp`
- `unmapped_da_sp`
- `filtered_out_sp`
- `annual_bridge_recovered_sp`
- `monthly_f0_recovered_sp`
- `fallback_recovered_sp`
- `ambiguous_sp`

These belong in diagnostics and `analysis.json`, and the three percentage coverages belong in report tables.

---

## 7. Native vs Overlap Reporting

This is the main lesson from MISO's `metric-contract.md`.

### 7.1 Native standalone table

For each model:

- native universe size
- label coverage
- `SP_Captured@200`
- `SP_Captured@400`
- `Abs_SP@200`
- `Abs_SP@400`
- `Binders@200`
- `Binders@400`
- `NB12_SP@200`
- `NB12_SP@400`
- `Dang_Recall@200`
- `Dang_Recall@400`

This is the table to use for real selector quality.

### 7.2 Overlap-only reranking table

For each pair of models:

- overlap branch count
- overlap SP coverage
- `SP_Captured@200_overlap`
- `SP_Captured@400_overlap`
- `Binders@200_overlap`
- `Binders@400_overlap`
- `NB12_SP@200_overlap`
- `NB12_SP@400_overlap`
- `avg_rank_overlap_topN_binders`

Use this table only for pure ranking analysis on the shared set.

Rule:

- do not present overlap-only numbers as deployment numbers

---

## 8. Case Study Tables

The report should include branch-level case studies, following the MISO pattern from the NB reports.

### 8.1 Most dangerous branches

For the top dangerous branches in the eval slice, include:

- `branch_name`
- `realized_shadow_price`
- `is_dangerous`
- `native_rank_model_a`
- `native_rank_model_b`
- `native_rank_v46`
- `overlap_rank_model_a`
- `overlap_rank_model_b`
- selected at `K=200` and `K=400` by each model

Purpose:

- show which high-value branches each model finds or misses

### 8.2 Most dangerous NB branches

For the top dangerous dormant/NB branches:

- same columns as above
- plus cohort flags such as `is_nb_12`

Purpose:

- answer the exact question: which model surfaces the most dangerous dormant branches, and at what K level?

### 8.3 Miss tables

Include:

- top dangerous branches missed by all models
- top V4.6-only hits
- top our-model-only hits

This is often more useful than aggregate deltas.

---

## 9. Required Report Sections

Every serious PJM comparison report should include these sections in order:

1. Coverage summary
   GT mapping, universe coverage, combined ceiling
2. Native standalone results at `K=200/400`
3. Direct delta versus V4.6 at `K=200/400`
4. Overlap-only reranking results
5. Dangerous branch analysis
6. Dangerous NB-branch analysis
7. Model-vs-V4.6 branch case studies
8. Annual-to-quarter drilldown for any anomalies

This keeps coverage failures from being mistaken for ranking failures.

---

## 10. Artifact Layout

### 10.1 `metrics.json`

Include:

- per annual cell headline metrics
- aggregated dev / holdout means
- `K=200` and `K=400` results explicitly
- secondary K monitoring values if available

### 10.2 `analysis.json`

Include:

- coverage breakdown
- overlap-set comparison results
- dangerous branch case studies
- dangerous NB branch case studies
- benchmark overlap analysis
- quarter-level anomaly notes

### 10.3 Optional case-study artifacts

If the report becomes too large, emit:

- `case_studies_dangerous.json`
- `case_studies_nb_dangerous.json`
- `overlap_comparison.json`

---

## 11. Immediate PJM Recommendation

For the first baseline comparison against V4.6, report exactly this:

### Headline annual table

Per `(planning_year, market_round, class_type)`:

- `SP_Captured@200`
- `SP_Captured@400`
- `Abs_SP@200`
- `Abs_SP@400`
- `VC@200`
- `VC@400`
- `Binders@200`
- `Binders@400`
- `Dang_Recall@200`
- `Dang_Recall@400`
- `NB12_SP@200`
- `NB12_SP@400`
- `Label_Coverage@200`
- `Label_Coverage@400`

### Delta-vs-V4.6 table

Per `(planning_year, market_round, class_type)`:

- `dSP_Captured@200`
- `dSP_Captured@400`
- `dAbs_SP@200`
- `dAbs_SP@400`
- `dDang_Recall@200`
- `dDang_Recall@400`

### Separate coverage table

- `gt_mapping_coverage`
- `model_universe_coverage`
- `combined_capture_ceiling`

### Case-study appendix

- top 20 dangerous branches
- top 20 dangerous NB branches
- for each branch: ranks and in-top-K flags for our model and V4.6

That is the minimum reporting surface that matches the mature MISO annual-signal style.

---

## 12. Bottom Line

Replicate MISO's reporting structure.
Do not replicate MISO's exact metric priorities blindly.

For PJM baseline work:

- annual-cell `Abs_SP@200/400` is the primary scorecard
- coverage tables are first-class and must appear beside model metrics
- quarter metrics support diagnosis, not headline reporting
- dangerous/NB branch case studies remain mandatory because they explain why models differ beyond averages
- direct deltas versus V4.6 must appear in every baseline benchmark report
