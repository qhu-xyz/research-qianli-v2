# research-annual-signal-v2

MISO annual constraint ranking signal: ML-based scoring of annual FTR branches
with NB (new binder) detection for dormant constraint discovery.

## Status

**Phase 5 (combined-ctype)**: Complete. Results archived in `registry/archive/`.
**Phase 6 (class-specific)**: In progress. Building separate onpeak/offpeak pipelines.

### Combined-Ctype Baseline (archived)

v0c + NB blend (α=0.05) was the combined-ctype holdout champion. These results
serve as the baseline — not used for production because V6.1 signals are
class-specific. See `docs/model-comparison-report.md` for full comparison.

**Script**: `scripts/run_phase5_reeval.py`
**Registry**: `registry/archive/phase5_final_150_300/`, `registry/archive/phase5_final_200_400/`

## Repo Structure

### Shared Infrastructure (`ml/`)

Reused by both combined and class-specific pipelines.

| Module | Purpose |
|--------|---------|
| `config.py` | Constants, eval splits, feature lists, gate metrics, class-type mappings |
| `data_loader.py` | Density loading, CID mapping, branch-level collapse |
| `ground_truth.py` | Realized DA ground truth with tiered labels |
| `history_features.py` | Binding frequency + da_rank_value |
| `nb_detection.py` | NB12/NB6/NB24 flag computation |
| `bridge.py` | Constraint-to-branch mapping via MISO bridge tables |
| `realized_da.py` | Realized DA shadow price loading |
| `features.py` | Combined-ctype model table assembly |
| `features_trackb.py` | Track B features (recency, shape) |
| `train.py` | LambdaRank training |
| `evaluate.py` | Metrics at K=150/200/300/400 + dangerous branch |
| `merge.py` | Two-track merge with tau support |
| `registry.py` | Experiment save/load |

### Class-Specific Pipeline (`ml/phase6/`)

New modules for onpeak/offpeak separation. Uses shared infrastructure above.

| Module | Purpose |
|--------|---------|
| `features.py` | Class-specific model table (class-specific target, BF, cohort + cross-class features) |
| `scoring.py` | Class-specific v0c formula + NB model + blend |

### Scripts

| Directory | Contents |
|-----------|---------|
| `scripts/phase6/` | Class-specific experiment scripts (v0a, v0c, blend per class) |
| `scripts/` | Active shared scripts (Phase 5 reeval, v0c baseline, utilities) |
| `scripts/archive/` | Historical combined-ctype scripts (Phase 3, 4a, 4b, etc.) |

### Registry

| Directory | Contents |
|-----------|---------|
| `registry/onpeak/` | Class-specific onpeak results (M1 GT, M2 models, M3 blend) |
| `registry/offpeak/` | Class-specific offpeak results |
| `registry/archive/` | 31 combined-ctype entries (Phase 3-5, v0a-v3g, etc.) |

### Docs

| Document | Status |
|----------|--------|
| `superpowers/specs/class-specific-pipeline-design.md` | **Active — Phase 6 design + milestone verification** |
| `superpowers/specs/project1-annual-signal-publication.md` | **Active — constraint-level publication** |
| `superpowers/specs/project2-path-rating-band-validation.md` | **Active — path rating** |
| `superpowers/specs/project1-test-specification.md` | **Active — 90-case test plan** |
| `model-comparison-report.md` | Combined-ctype model comparison (baseline reference) |
| `pipeline-and-production-port.md` | Production port plan + V6.1 artifact map |
| `signal-generation-procedure.md` | SPICE V6.2B signal generation reference |
| `superpowers/findings/nb-two-track-consolidated.md` | Combined-ctype findings |
| `superpowers/specs/phase5-reeval-design.md` | Phase 5 design (executed, combined-ctype) |
| `archive/` | Historical docs (phase1-4 designs, old reviews) |

## Key Findings (Combined-Ctype Baseline)

> All results below are from the combined-ctype pipeline. Class-specific results
> will be produced in Phase 6.

1. **NB blend beats hard two-track** — soft score boosting preserves v0c's natural
   dormant inclusion while adding targeted NB picks
2. **v3a fails DangR gate** at (200,400) — v0c captures more dangerous branches
3. **K=300+ is the NB sweet spot** — at K≤150, NB costs VC; at K≥300, blend is free
4. **Cross-class BF is nearly as predictive as same-class BF** (Spearman 0.43 vs 0.44)
5. **V6.1 is genuinely class-specific** — shadow_price_da differs up to $7,318 across onpeak/offpeak

## Implementation Milestones

| Milestone | Deliverable | Status |
|-----------|------------|--------|
| M1 | Class-specific GT + model table | In progress |
| M2 | Formula ladder → NB → blend champion per class | Planned |
| M3 | Constraint-level publication pipeline | Planned (metadata gap open) |
| M4 | Path rating + band validation | Planned |
