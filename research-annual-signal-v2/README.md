# research-annual-signal-v2

MISO annual constraint ranking signal: branch-level scoring for annual FTR auctions (R1) with NB (new binder) detection for dormant constraint discovery.

## Current Status (2026-03-23)

**Champion**: v0c formula (`0.40×da_rank + 0.30×rt_max + 0.30×bf`) — established in Phase 5-10.

**Active work**: NB-hist-12 model to augment v0c with dormant branch detection.

**Published signals**:
- V7.0.R1 — 54 files at `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1/`
- V7.0B.R1 — 54 files (supplement key matching + zero-SF filter) at `TEST.Signal.MISO.SPICE_ANNUAL_V7.0B.R1/`

## Project Timeline

### Phase 1-5: Combined-Ctype Pipeline (archived)
Built branch-level model table, formula baselines, LambdaRank models, NB detection framework. v0c emerged as champion — simple formula that beats all ML attempts on the general population. Results in `registry/archive/`.

### Phase 6: Class-Specific Pipeline
Separated onpeak/offpeak features and targets. Confirmed class type is noise for annual — combined model performs as well. Pipeline in `ml/phase6/`.

### Phase 7-10: Attempted ML Improvements (all negative)
Multiple rounds of ML models, blends, cross-class features, two-population approaches all failed to beat v0c. Confirmed v0c as ceiling for the general population.

### V7.0 Signal Publication (2026-03-18 to 2026-03-21)
- Published V7.0 and V7.0B constraint-level signals for all PYs (2019-2025)
- V7.0B adds supplement key matching (recovers 86/129 unmapped DA CIDs) and zero-SF filter
- Full pipeline: build branch model table → score → map back to SPICE CIDs → publish parquets
- See `docs/v7.0b-release-report.md` for comparison results

### NB-hist-12 Model Experiment V2 (2026-03-23) — CURRENT

**Problem**: v0c is structurally blind to dormant branches. Per-ctype dormant populations differ (onpeak: `bf_12==0`, offpeak: `bfo_12==0`).

**Approach (V2)**: Two per-ctype NB LambdaRank models (8 class-agnostic features, no V4.4 features). Evaluation uses `build_class_model_table` for fully class-specific v0c. V4.4 as per-ctype benchmark only. Fixed K via v0c backfill.

**Key findings**:
- At K=400 onpeak, R30_nb (+0.4pp VC) and R30_v44 (+0.5pp VC) are a near-free lunch
- At K=400 offpeak, all reservations cost VC (-0.5pp to -3.2pp)
- ML_nb dominates V4.4 on onpeak NB-only (VC@50: 0.096 vs 0.013 in 2024)
- V4.4 is inconsistent — near-zero on 2024 for both ctypes

**V1 superseded**: V1 had combined v0c (wrong BF + da_rank_value), onpeak-only V4.4, combined NB model. Archived to `scripts/archive/`.

**Full results**: `docs/2026-03-23-nb-v2-experiment-report.md`

## Repo Structure

### Core ML Pipeline (`ml/`)

| Module | Purpose |
|--------|---------|
| `config.py` | Constants, eval splits, feature lists, gate metrics |
| `data_loader.py` | Density loading, CID mapping, branch-level collapse |
| `ground_truth.py` | Realized DA ground truth with supplement fallback + tiered labels |
| `history_features.py` | Binding frequency (bf_6/12/15, bfo_6/12) + da_rank_value |
| `bridge.py` | Constraint-to-branch mapping (MISO bridge + supplement keys) |
| `realized_da.py` | Realized DA shadow price loading |
| `nb_detection.py` | NB12/NB6/NB24 flag computation |
| `features.py` | Combined model table assembly |
| `signal_publisher.py` | Constraint-level signal publication (V7.0/V7.0B) |
| `evaluate.py` | Metrics: VC@K, Abs_SP@K, Rec@K, NB metrics, dangerous branch |
| `train.py` | LambdaRank training |
| `merge.py` | Two-track merge with tau support |
| `registry.py` | Experiment save/load |

### Class-Specific (`ml/phase6/`)

| Module | Purpose |
|--------|---------|
| `features.py` | Class-specific model table with `build_model_table()` |
| `scoring.py` | v0c formula, `_minmax()` normalization |

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/nb_experiment_v2.py` | **NB V2 experiment**: 2 per-ctype models, class-specific v0c, V4.4 benchmark, 6-part report |
| `scripts/archive/` | Historical scripts (V1 NB experiment, phase 3-5) |

### Documents

| Document | Description |
|----------|-------------|
| `docs/2026-03-23-nb-v2-experiment-report.md` | **NB V2 experiment**: per-ctype models, 8 tables + aggregates, case studies |
| `docs/v7.0b-release-report.md` | V7.0B comparison results (supplement matching) |
| `docs/2026-03-18-data-quality-audit.md` | Coverage gap analysis |
| `docs/2026-03-19-v7-verification-report.md` | V7.0 verification (loss waterfall, zero-SF) |
| `docs/2026-03-20-coverage-investigation-handoff.md` | Teammate handoff: unmapped DA CIDs |
| `docs/coverage-analysis-runbook.md` | Runbook with supplement method |
| `docs/supplement-matching-implementation-plan.md` | Supplement key matching design |
| `docs/repo-reorg-plan.md` | MISO+PJM reorg plan (not yet executed) |
| `docs/model-comparison-report.md` | Phase 5 combined-ctype model comparison |
| `docs/pipeline-and-production-port.md` | Production port plan |
| `docs/signal-generation-procedure.md` | SPICE V6.2B signal generation reference |

### Registry

| Directory | Contents |
|-----------|---------|
| `registry/archive/` | 31 combined-ctype entries (Phase 3-5) |
| `registry/onpeak/`, `registry/offpeak/` | Placeholders for class-specific results |

## Key Design Decisions

1. **Branch-level modeling**: One row = (branch_name, planning_year, quarter). Target = total DA SP for the branch. Multiple SPICE CIDs map to the same branch — collapsed at Level 2.

2. **Combined onpeak+offpeak**: Annual density features are class-type agnostic. Using combined target + separate BF features (bf_* for onpeak, bfo_* for offpeak) outperforms split models.

3. **v0c as general scorer**: Formula `0.40×(1-minmax(da_rank)) + 0.30×minmax(rt_max) + 0.30×minmax(bf)` is the champion after 10 phases of ML experiments. Simple, stable, and hard to beat.

4. **Reserved-slot combination** (not blended scores): Instead of blending v0c and NB scores into one ranking, we allocate fixed slots — e.g., 170 v0c + 30 NB at K=200. This preserves v0c's ranking integrity while guaranteeing NB exposure.

5. **V4.4 as feature, not scorer**: V4.4 deviation features are valuable for NB detection but V4.4's standalone ranking is too inconsistent across years (near-zero in 2024). The ML model with V4.4 features is more stable.
