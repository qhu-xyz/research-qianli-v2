# research-annual-signal-v2

MISO annual constraint ranking signal: branch-level scoring for annual FTR auctions (R1) with NB (new binder) detection for dormant constraint discovery.

## Current Status (2026-03-24)

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

### Bucket_6_20: Danger-Aware Unified Model (2026-03-24) — CANDIDATE

**Candidate: Bucket_6_20** — Single unified LambdaRank model on ALL branches with 5-tier severity labels [0/1/2/3/4] weighted [1/1/2/6/20]. 13 features (history + density + top2_mean), trained on 2018-2025.

**Key results** (2022-2025, native standalone vs V4.4):
- Beats V4.4 on total SP in **all 16 (year × ctype × K) cells** (+$14K to +$270K)
- At K=200: also wins NB_SP in 5/8 cells (all 2024-2025)
- At K=400: V4.4 wins NB_SP in 7/8 cells by packing more dormant branches at cost of overall SP
- Fully reproducible — no V4.4 dependency
- **Caveat**: V4.4 SP numbers are conservative — V4.4 picks outside our universe get zero credit (5-10 branches per quarter unlabeled). Not yet compared side-by-side with v0c or Opt3 in the same script.

**Status**: Beats V4.4 benchmark. Not yet confirmed as overall champion vs v0c/Opt3. Needs deployment-style eval (R30/R50).

**Reports**: `docs/2026-03-24-bucket-model-report.md` | `docs/metric-contract.md`
**Evolution**: v0c → Opt3 → feature ablation → top-tail variants → Bucket_6_20

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
| `scripts/nb_bucket_model.py` | **Bucket_6_20 champion**: train + eval + registry save |
| `scripts/nb_native_comparison.py` | Native standalone comparison (multi-model) |
| `scripts/nb_v3_ablation.py` | V3 ablation: 9 variants (+2020, labels, features) |
| `scripts/nb_experiment_v2.py` | V2 baseline experiment |
| `scripts/nb_feature_ablation.py` | Density feature expansion ablation |
| `scripts/publish_annual_signal.py` | V7.0/V7.0B signal publication |
| `scripts/archive/` | Historical scripts (V1 NB, phase 3-5, v7 verification) |

### Documents

| Document | Description |
|----------|-------------|
| `docs/2026-03-23-nb-v3-final-report.md` | **NB V3 final report**: tiered_top2 champion, native standalone comparison vs V4.4 |
| `docs/metric-contract.md` | **Metric contract**: cross-model comparison methodology (native/overlap/deployment views) |
| `docs/2026-03-23-nb-v2-experiment-report.md` | NB V2 experiment (superseded by V3) |
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
