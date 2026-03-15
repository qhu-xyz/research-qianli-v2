# Implementation Handoff Prompt

Copy everything below this line and feed it to the implementing AI.

---

You are implementing the MISO Annual FTR Constraint Ranking ML Pipeline from scratch. No code exists yet — only documentation. Your job is to build the entire `ml/` library and experiment scripts following the specs precisely.

## Documents to Read (in this exact order)

1. **`docs/superpowers/specs/2026-03-12-miso-annual-constraint-ranking-design.md`** — the approved design spec. This is your primary implementation guide. It defines all module boundaries, interfaces, data flows, metric definitions, and the phased plan. Read it fully before writing any code.

2. **`docs/implementer-guide.md`** — the exhaustive domain spec (1,738 lines). Contains data paths, column names, bin semantics, code snippets, partition layouts, and 26 numbered traps. When the design spec says "see implementer-guide SS8.3", go read that section. The design spec is the WHAT; the implementer guide is the HOW and WHERE.

3. **`docs/test-specification.md`** — validation targets for each module. Use this to write assertions and sanity checks as you build each module.

4. **`docs/bridge-table-gap-analysis.md`** — historical context only. Read if you need to understand why 2025-06 has 26% Stage 1 loss or why monthly bridge fallback matters. Do NOT use this to make design decisions — those are already made.

## Architecture Summary

12 Python modules in `ml/`, thin experiment scripts in `scripts/`, results in `registry/`.

```
ml/config.py           — paths, constants, UNIVERSE_THRESHOLD, feature lists, splits, params
ml/bridge.py           — annual+monthly bridge loading, convention < 10, both-ctype UNION
ml/realized_da.py      — DA cache loading, combined/onpeak/offpeak aggregation
ml/data_loader.py      — density + limits -> universe filter -> Level 1+2 collapse -> branch features
ml/history_features.py — BF family (7) + da_rank_value + has_hist_da flag
ml/ground_truth.py     — continuous SP target + tiered labels + per-ctype split + coverage diagnostics
ml/nb_detection.py     — NB6/NB12/NB24 + per-ctype NB12 flags (branch-level)
ml/features.py         — joins all into ONE model table, cohort assignment, monotone vector
ml/train.py            — expanding-window LambdaRank training + prediction
ml/evaluate.py         — Tier 1/2/3 metrics, NB metrics, cohort contribution, gate checking
ml/registry.py         — write config.json + metrics.json to registry/{version}/
```

Experiment scripts are thin — they import from `ml/`, specify a feature list and params, call train/evaluate, write to registry. The scripts ARE the configs. No YAML/JSON config framework.

## Critical Rules (violations cause silent bugs)

### Data Rules
- **Use polars, not pandas.** Memory budget is 128 GiB with ~40 GiB for scripts.
- **pbase loaders are BROKEN for annual data.** Use `pl.read_parquet()` with partition-specific paths. See implementer-guide SS4.2.
- **Bridge table: use partition-specific paths** (not hive scan). Different partitions have different schemas (`device_type` column mismatch). See Trap 5.
- **Density bins sum to 20.0 per row.** They are density weights, NOT probabilities. Values can exceed 1.0. NOT monotone across thresholds. Do NOT set monotone_constraints on density features.
- **Realized DA cache stores `abs(sum(shadow_price))` per (constraint_id, month, ctype).** Netting already happened within-month-within-ctype. When aggregating to quarter GT, just sum these nonnegative values. No re-netting.

### Leakage Prevention (THE most dangerous bug class)
- **BF/da_rank_value lookback cutoff = March of submission year.** For PY 2025-06 (submitted ~April 10, 2025): use months 2017-04 through 2025-03. NEVER include April.
- **For any realized-data-derived feature, ask: "at submission time, is this data available?"**

### Training Rules
- **LightGBM num_threads=4.** Container has 64 CPUs. Auto-detection causes 57s training time instead of 0.1s.
- **Sort by (planning_year, aq_quarter, branch_name) before building query group sizes.** Misalignment is silent.
- **Eval-only scoring.** Never predict on training rows.
- **Monotone constraint order must exactly match feature_cols.** Assert this.

### Module Boundary Rules
- **Everything after Level 2 collapse is branch-level.** No downstream module sees a constraint_id.
- **bridge.py is the single source of truth** for bridge loading AND ambiguity handling. It exposes `map_cids_to_branches()` which handles both-ctype UNION, convention < 10, ambiguity detection+logging+dropping. All three consumers (data_loader, ground_truth, history_features) use this single function. No module implements its own bridge-join logic.
- **realized_da.py is the single source of truth** for DA cache access. Same rule.
- **features.py owns the model table contract.** It joins data_loader + history_features + ground_truth + nb_detection into ONE DataFrame. It also attaches `total_da_sp_quarter` from GT coverage diagnostics as a group-level column. train.py and evaluate.py receive this table.
- **evaluate.py gets everything from the model table.** The `Abs_SP@K` denominator (`total_da_sp_quarter`) is in the table — no side-channel needed.
- **registry.py is independent of evaluate.py.** It takes a metrics dict + config dict — no import from evaluate.

### Code Quality Rules (from CLAUDE.md)
- **No silent fallbacks.** Never use fillna or default values for missing data. Raise errors.
- **No try/except to swallow errors.** Let things fail loudly.
- **Assert early.** Validate data shapes, column presence, parameter values at every stage.
- **No default parameters for critical arguments.** If class_type, period_type, etc. must be explicitly provided, make them required.
- **Use Ray for any parallelism.** Never multiprocessing, concurrent.futures, joblib, dask.

## Three Decisions Already Made

1. **Universe threshold (SS5.4)**: Run elbow analysis from scratch in Step 1.2. The SS5.4 tables in the implementer guide are planning targets from the old filter, NOT validated raw-only numbers. Produce a calibration artifact (threshold, plot, regenerated table, rationale note) and freeze in config.py.

2. **Level-2 density second stat (SS6.2)**: Deferred to Phase 2 empirics. Scaffold max+min as default, but test max+std and max-only in Steps 2b/2d. Keep/drop based on improvement to VC@50, Abs_SP@50, NB12_Recall@50 — must beat noise across dev groups.

3. **Per-ctype NB monitoring (SS11.1)**: Compute nb_onpeak_12 and nb_offpeak_12 only (12-month window). Monitoring only, not gated. Per-ctype NB uses per-ctype target binding (onpeak_sp > 0 / offpeak_sp > 0), not combined target.

## Implementation Order

Follow the phased plan in design spec SS11 exactly:

**Phase 1** (foundation — no ML yet):
1. `scripts/fetch_realized_da.py` — build DA cache first (requires Ray)
2. `ml/config.py` + `ml/bridge.py` + `ml/realized_da.py` — shared infrastructure
3. `scripts/calibrate_threshold.py` — universe threshold elbow analysis
4. `ml/data_loader.py` — density collapse pipeline
5. `ml/ground_truth.py` — GT with annual + monthly bridge fallback
6. `ml/history_features.py` — BF + da_rank_value via monthly binding table
7. `ml/nb_detection.py` — NB flags (branch-level)
8. `ml/features.py` — model table assembly
9. `ml/evaluate.py` + `ml/registry.py` — metrics + persistence
10. Formula baselines (v0a, v0b, v0c) + baseline contract freeze

**Phase 2** (ML build-up — add feature groups one at a time):
- 2a: historical features only (must beat v0c)
- 2b: + core density max (key metric: NB12_Recall@50)
- 2c: + counter-flow/mid-range
- 2d: test min vs std variants
- 2e: + structural features
- 2f: final pruning -> candidate champion
- 2g: density-only diagnostic

**Phase 3** (exploration, only if density shows signal)

## Key Details Easy to Miss

1. **count_active_cids fix**: Compute is_active flag for ALL cids BEFORE filtering. Then count_cids = total mapped cids on branch (active + inactive), count_active_cids = active only. If you filter first then count, both features are identical and useless.

2. **Constraint limits need Level 1 aggregation**: mean(limit) across dates/months per cid, THEN Level 2 branch aggregation (min/mean/max/std).

3. **Bridge ambiguity**: After UNION of onpeak+offpeak bridge, some cids map to multiple branch_names. Detect, log count + SP, and drop ambiguous cids. Do not silently include them. This rule is enforced by the shared `bridge.map_cids_to_branches()` function — used by data_loader, ground_truth, AND history_features. Same ambiguity policy everywhere.

4. **v0c formula uses bf_combined_12** (not onpeak-only bf_12). The signal is class-type agnostic.

5. **Cohorts use has_hist_da flag** (cumulative_sp > 0), NOT da_rank_value == sentinel. The flag is exported from history_features.py.

6. **BF denominator is always fixed N**, even if fewer months of history exist. bf_6 divides by 6 even with only 4 months available.

7. **da_rank_value is ranked within the current (PY, quarter) universe**, not globally. Dense rank over positive-history branches. Zero-history branches get rank = n_positive + 1.

8. **Ground truth returns raw coverage diagnostics** (unmapped_cids, unmapped_sp, total_cids, total_sp) as a separate dict alongside the DataFrame. Not just percentages.

9. **Abs_SP@50 denominator is ALL DA binding SP** for the quarter — including constraints outside the model universe. This is the cross-universe metric. The denominator (`total_da_sp_quarter`) is propagated from GT coverage diagnostics through `features.py` into the model table as a group-level column. `evaluate.py` reads it directly from the table — no side-channel.

10. **history_features.py builds a monthly branch-binding table first** (month, branch, onpeak_bound, offpeak_bound, combined_bound, onpeak_sp, offpeak_sp, combined_sp), then derives both BF and da_rank_value from it. Single DA scan, no duplication. **Bridge partition rule for historical months**: always use the **eval PY's** annual bridge (not the PY that contains historical month M). Monthly fallback uses f0 monthly bridge for month M itself. This keeps the mapping consistent with GT and data_loader for the same PY.

11. **NB detection uses the monthly binding table from history_features as input.** It does NOT re-scan DA. The interface is: `compute_nb_flags(branches, PY, quarter, gt_df, monthly_binding_table) -> DataFrame`.

12. **Gate metric is NB12_Recall@50** (not NB_Recall@50). The "12" must be explicit in the name.

13. **Baseline contract freeze** happens after formula baselines run (Step 1.9). After this point, the authoritative baseline for promotion (v0c) and exact metric names/denominators are frozen. No drift.

## Data Paths

| Data | Path |
|------|------|
| Density distribution | `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_DENSITY_DISTRIBUTION.parquet` |
| Constraint limits | `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_LIMIT.parquet` |
| Bridge table | `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_INFO.parquet` |
| Realized DA cache (build your own) | `data/realized_da/` (this project) |
| v1 code (reference only) | `/home/xyz/workspace/research-qianli-v2/research-annual-signal/` |

## Environment

- Python via pmodel venv: `cd /home/xyz/workspace/pmodel && source .venv/bin/activate`
- Ray cluster: `ray://10.8.0.36:10001`
- polars 1.31.0, lightgbm, numpy
- 128 GiB pod, 64 CPUs
- Today's date: 2026-03-12
