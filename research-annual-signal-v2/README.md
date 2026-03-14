# research-annual-signal-v2

MISO annual constraint ranking signal: ML-based scoring of annual FTR branches
with NB (new binder) detection for dormant constraint discovery.

## Current Champion

**v0c + NB blend (α=0.05)** — verified holdout champion at both K pairs.

| K Pair | Champion Weight | NB12_SP | Dang_Recall | VC |
|---|---|:---:|:---:|:---:|
| (150, 300) | sqrt SP | 21.6%@300 | 87.6%@300 | 0.724@300 |
| (200, 400) | tiered 1/3/10 | 29.2%@400 | 93.9%@400 | 0.788@400 |

How it works: score all branches with v0c formula, add 0.05 × normalized NB model
prediction for dormant branches only. No forced slot reservation.

**Script**: `scripts/run_phase5_reeval.py`
**Registry**: `registry/phase5_final_150_300/`, `registry/phase5_final_200_400/`

## Repo Structure

### ML Library (`ml/`)
| Module | Purpose |
|--------|---------|
| `config.py` | Constants, eval splits, feature lists, gate metrics |
| `features.py` | Model table assembly (universe + GT + history + NB flags) |
| `features_trackb.py` | Phase 4b Track B features (recency, shape) |
| `data_loader.py` | Density loading, CID mapping, branch-level collapse |
| `ground_truth.py` | Realized DA ground truth with tiered labels |
| `history_features.py` | Binding frequency + da_rank_value |
| `nb_detection.py` | NB12/NB6/NB24 flag computation |
| `bridge.py` | Constraint-to-branch mapping via MISO bridge tables |
| `realized_da.py` | Realized DA shadow price loading |
| `train.py` | LambdaRank training for Track A (v3a) |
| `evaluate.py` | Metrics at K=20/50/100/150/200/300/400 + dangerous branch |
| `merge.py` | Two-track merge with tau support |
| `registry.py` | Experiment save/load |
| `signal_publisher.py` | (Project 1 — not yet implemented) |

### Scripts (`scripts/`)
| Script | Status | Purpose |
|--------|--------|---------|
| `run_phase5_reeval.py` | **Active** | Phase 5 full candidate matrix evaluation |
| `run_v0c_full_blend.py` | Active | v0c formula baseline |
| `run_ml_experiment.py` | Active | v3a and other ML experiments |
| `publish_annual_signal.py` | Planned | Project 1 signal publication |
| `run_phase4a_experiment.py` | Superseded | Phase 4a (old @50/@100 metrics) |
| `run_phase4b_regression.py` | Negative | Phase 4b regression (negative result) |
| `run_two_track_experiment.py` | Superseded | Phase 3 two-track (old metrics) |
| `run_nb_analysis.py` | Historical | Phase 3.1 NB analysis |
| `run_nb_supplement.py` | Historical | Phase 3 NB supplement metrics |
| `run_track_b_experiment.py` | Historical | Phase 3.2 Track B model dev |
| `calibrate_threshold.py` | Utility | Universe threshold calibration |
| `fetch_realized_da.py` | Utility | DA data fetching |
| `run_ml_diagnostics.py` | Utility | ML model diagnostics |

### Registry (`registry/`)
| Version | Status | Description |
|---------|--------|-------------|
| `phase5_final_150_300/` | **Champion** | Blend holdout results for (150,300) |
| `phase5_final_200_400/` | **Champion** | Blend holdout results for (200,400) |
| `v0c/` | Baseline | v0c formula champion (no NB) |
| `v3a/` | Baseline | v3a LambdaRank (13 features) |
| `p4a_tiered_lgbm_r5_r15/` | Historical | Phase 4a best Track B model |
| `phase5_verified_*/` | Superseded | Earlier Phase 5 run (same results as final) |
| `phase5_champ_*/` | Superseded | First Phase 5 run |
| `tt_v0c_r*/` | Superseded | Phase 3 two-track experiments |
| `p4b_reg_r5_r15/` | Negative | Phase 4b regression (negative result) |
| Others (v0a/v0b/v2a*/v3b/v3e_nb/v3f_nbonly/v3g/d1/d2/d4) | Historical | Earlier experiments |

### Docs (`docs/`)
| Document | Status |
|----------|--------|
| `specs/2026-03-14-project1-annual-signal-publication.md` | **Active — next implementation** |
| `specs/2026-03-14-project2-path-rating-band-validation.md` | **Active — next implementation** |
| `specs/2026-03-14-phase5-reeval-design.md` | Executed |
| `findings/2026-03-14-nb-two-track-consolidated.md` | **Current findings** |
| `specs/2026-03-13-phase4a-weighted-track-b-design.md` | Completed |
| `specs/2026-03-14-phase4b-value-aware-track-b-design.md` | Negative result |
| `plans/2026-03-13-phase3-nb-two-track-design.md` | Historical (misnamed — is a spec) |
| Others | Historical |

## Key Findings

1. **NB blend beats hard two-track** — soft score boosting preserves v0c's natural
   dormant inclusion while adding targeted NB picks
2. **v3a fails DangR gate** at (200,400) — v0c captures more dangerous branches
3. **Constraint propagation infeasible** — CID mapping is 1-to-1 (zero shared CIDs)
4. **Phase 4b regression negative** — worse than Phase 4a tiered binary
5. **K=300+ is the NB sweet spot** — at K≤150, NB costs VC; at K≥300, blend is free

## Next Steps

- **Project 1**: Publish constraint-level annual signal (V7.0)
- **Project 2**: Path rating + annual-band validation
