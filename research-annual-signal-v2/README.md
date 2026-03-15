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
| `features_trackb.py` | Track B features (recency, shape) for dormant branches |
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

### Scripts (`scripts/`)
| Script | Status | Purpose |
|--------|--------|---------|
| `run_phase5_reeval.py` | **Active** | Phase 5 full candidate matrix evaluation |
| `run_v0c_full_blend.py` | Active | v0c formula baseline |
| `run_ml_experiment.py` | Active | v3a and other ML experiments |
| `run_phase4a_experiment.py` | Historical | Phase 4a (old @50/@100 metrics) |
| `run_phase4b_regression.py` | Historical | Phase 4b regression (negative result) |
| `run_two_track_experiment.py` | Historical | Phase 3 two-track (old metrics) |
| `calibrate_threshold.py` | Utility | Universe threshold calibration |
| `fetch_realized_da.py` | Utility | DA data fetching |

### Registry (`registry/`)
| Version | Status | Description |
|---------|--------|-------------|
| `phase5_final_150_300/` | **Champion** | Blend holdout results for (150,300) |
| `phase5_final_200_400/` | **Champion** | Blend holdout results for (200,400) |
| `v0c/` | Baseline | v0c formula (no NB) |
| `v3a/` | Baseline | v3a LambdaRank (13 features) |
| `p4a_tiered_lgbm_r5_r15/` | Historical | Phase 4a best Track B model |
| `p4b_reg_r5_r15/` | Negative | Phase 4b regression (negative result) |
| Others | Historical | Earlier experiments (v0a/v0b/v2a*/v3b/etc., tt_v0c_r*, phase5_champ/verified) |

### Docs (`docs/`)
| Document | Status |
|----------|--------|
| `model-comparison-report.md` | **Current** — all models compared with holdout tables |
| `pipeline-and-production-port.md` | **Current** — full pipeline + V6.1/V7.0 artifact map + gaps |
| `signal-generation-procedure.md` | Reference — SPICE V6.2B signal generation procedure |
| `superpowers/specs/project1-annual-signal-publication.md` | **Active — next implementation** |
| `superpowers/specs/project2-path-rating-band-validation.md` | **Active — next implementation** |
| `superpowers/specs/project1-test-specification.md` | **Active — 90-case test plan** |
| `superpowers/specs/phase5-reeval-design.md` | Executed |
| `superpowers/findings/nb-two-track-consolidated.md` | **Current findings** |
| `archive/` | Historical (phase1-2 plans, phase3-4 designs, old reviews) |

## Key Findings

1. **NB blend beats hard two-track** — soft score boosting preserves v0c's natural
   dormant inclusion while adding targeted NB picks
2. **v3a fails DangR gate** at (200,400) — v0c captures more dangerous branches
3. **Constraint propagation infeasible** — CID mapping is 1-to-1 (zero shared CIDs)
4. **Phase 4b regression negative** — worse than Phase 4a tiered binary
5. **K=300+ is the NB sweet spot** — at K≤150, NB costs VC; at K≥300, blend is free

## Next Steps

- **Project 1**: Publish constraint-level annual signal (V7.0) — see `docs/superpowers/specs/`
- **Project 2**: Path rating + annual-band validation — see `docs/superpowers/specs/`
