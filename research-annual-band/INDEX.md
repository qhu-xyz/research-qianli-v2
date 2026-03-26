# research-annual-band — Repository Index

## Quick Reference

| What | Where |
|------|-------|
| CLAUDE.md (rules) | `./CLAUDE.md` |
| Port plan (MISO) | `porting-preparation/miso-annual-port-plan.md` |
| Port differences | `porting-preparation/miso-vs-pjm-port-differences.md` |
| Test spec | `porting-preparation/v2-annual-test-spec.md` |
| PJM report | `pjm/docs/pjm-consolidated-report.md` |
| MISO R1 report | `miso/docs/r1-v2-consolidated-report.md` |
| MISO R2/R3 report | `miso/docs/r2-r3-consolidated-report.md` |
| Revenue features | `miso/docs/revenue-features.md` |

## Directory Structure

```
research-annual-band/
├── CLAUDE.md                          # Project rules (Pxx definition, scale, reporting)
├── INDEX.md                           # This file
├── README.md                          # Project overview
├── human-input.md                     # User instructions log
├── task_plan.md                       # Task tracking
│
├── miso/                              # MISO annual research
│   ├── docs/
│   │   ├── r1-v2-consolidated-report.md   # R1 champion: frozen spec, holdout results
│   │   ├── r2-r3-consolidated-report.md   # R2/R3 champion: pure MTM, hybrid cells
│   │   ├── revenue-features.md            # 1(rev) definition, quarter-month mapping
│   │   └── data-verification.md           # Data pipeline verification
│   ├── scripts/
│   │   ├── load_canonical.py              # Build canonical_annual_paths.parquet (Ray)
│   │   ├── run_r1_v2.py                   # Frozen R1 V2 pipeline (legacy two-sided)
│   │   └── run_r2r3.py                    # Frozen R2/R3 pipeline (one-sided Pxx)
│   ├── data/                              # Cached parquets (not in git)
│   │   ├── canonical_annual_paths.parquet  # 2.39M paths, R1-R3, PY2018-2025
│   │   ├── r1_1rev_option_b.parquet       # R1 1(rev), 637K rows
│   │   └── all_rounds_1rev.parquet        # All-rounds 1(rev), 2.39M rows
│   └── archive_v1/                        # Legacy research (pre-V2)
│       ├── findings.md                    # Original research findings
│       ├── runbook.md                     # Original runbook
│       ├── scripts/
│       │   ├── run_v9_bands.py            # V9 banding (imported by V2 scripts)
│       │   ├── run_aq1_experiment.py      # aq1 baseline research
│       │   ├── run_aq3_experiment.py      # aq3 baseline (nodal_f0 source code)
│       │   ├── run_aq4_experiment.py      # aq4 baseline (year_map logic)
│       │   └── ...                        # Other experiment scripts
│       ├── docs/
│       │   └── prod-port.md               # Original production port plan
│       └── versions/                      # Versioned experiment results
│
├── pjm/                               # PJM annual research
│   ├── docs/
│   │   └── pjm-consolidated-report.md     # PJM champion: R1 blend, R2-R4 pure
│   ├── scripts/
│   │   ├── load_canonical.py              # Build PJM canonical paths (Ray)
│   │   ├── run_pjm.py                     # Frozen PJM pipeline (one-sided Pxx)
│   │   └── validate_nodal_f0q_parity.py   # nodal_f0_q parity gate script
│   ├── data/                              # Cached parquets (not in git)
│   │   ├── canonical_annual_paths.parquet  # 2.51M paths, R1-R4, PY2017-2025
│   │   ├── pjm_recent1.parquet            # March DA × 12, 93.9% coverage
│   │   └── pjm_1rev.parquet               # Trailing 12mo DA (not used in champion)
│   ├── plan.md                            # PJM research plan
│   └── archive_v1/                        # Legacy PJM research
│
└── porting-preparation/               # Production port docs
    ├── miso-annual-port-plan.md           # MISO port plan (active, updated)
    ├── miso-vs-pjm-port-differences.md    # 46-item difference audit
    ├── v2-annual-integration-plan.md      # V2 integration architecture
    ├── v2-annual-port-schema.md           # Config schema spec
    └── v2-annual-test-spec.md             # Test & validation spec
```

## Key Data Files (NFS, not in git)

| File | Location | Rows | Purpose |
|------|----------|-----:|---------|
| MISO canonical paths | `miso/data/canonical_annual_paths.parquet` | 2.39M | All MISO annual trades R1-R3 |
| MISO R1 1(rev) | `miso/data/r1_1rev_option_b.parquet` | 638K | R1 DA revenue feature |
| MISO all-rounds 1(rev) | `miso/data/all_rounds_1rev.parquet` | 2.39M | All-rounds DA revenue |
| PJM canonical paths | `pjm/data/canonical_annual_paths.parquet` | 2.51M | All PJM annual trades R1-R4 |
| PJM recent_1 | `pjm/data/pjm_recent1.parquet` | — | March DA × 12 |
| Research nodal_f0 baselines | `/opt/temp/qianli/annual_research/crossproduct_work/aq*_all_baselines.parquet` | 584K | Research f0 stitch (non-standard alias expansion) |
| MISO training (live-built) | `/opt/temp/qianli/miso_annual_training_v3/` | 192 parts | Production training data |
| PJM training | `/opt/temp/qianli/pjm_annual_training/` | 108 parts | Production training data |

## Production Code (in pmodel)

| File | Purpose |
|------|---------|
| `pmodel/.../pipeline/band/annual.py` | Core engine: quantile calibration, winsorization, caps |
| `pmodel/.../pipeline/band/generator.py` | Routing, feature dispatch, `_compute_1rev`, `_compute_nodal_f0_q`, `_compute_recent_1` |
| `pmodel/.../config/versions/miso_aq*/param1.json` | 8 MISO configs |
| `pmodel/.../config/versions/pjm_a_*/param1.json` | 3 PJM configs |
| `pmodel/scripts/build_miso_annual_training.py` | MISO training builder (Ray, self-contained) |
| `pmodel/scripts/build_pjm_annual_training.py` | PJM training builder (Ray) |
| `pmodel/tests/.../test_annual.py` | 67 tests |
| `pmodel/docs/ftr24_v2_annual_direct_handoff.md` | Teammate handoff doc |

## Champion Specs (frozen)

### MISO R1
- Baseline: `w(flow,bin) × nodal_f0_q + (1-w) × 1_rev`
- nodal_f0_q: prior-year same-quarter f0 stitch (`_get_f0_stitch_months`)
- 1_rev: prior-PY same-quarter DA congestion (aq4 production: Mar N + Apr-May N-1)
- Calibration: (bin, flow, class), min 300 rows
- Winsorization: nodal_f0_q [-5800, 11600], 1_rev [-6200, 11800]
- Weights: prevail q1-q4=0.85, q5=0.50; counter q1=0.85...q5=0.60

### MISO R2/R3
- Baseline: pure `mtm_1st_period`
- Calibration: hybrid q1-q3 (bin,flow), q4-q5 (bin only), min 300 rows
- No blend, no winsorization

### PJM R1
- Baseline: `w(flow,bin) × mtm_1st_period + (1-w) × recent_1`
- recent_1: March N DA × 12
- Calibration: (bin, flow, class), min 200 rows

### PJM R2-R4
- Baseline: pure `mtm_1st_period`
- Calibration: (bin, flow, class), min 200 rows

## Known Limitations

1. **aq4 weights not re-estimated** — 1_rev mapping changed from research
2. **~1.3% MISO R1 trades dropped** — both nodal_f0_q and 1_rev NaN
3. **nodal_f0_q differs from research** — live compute uses canonical pbase backward fill, research used non-standard alias expansion
4. **aq4 May f0 borderline** — f0 clears ~mid-April, same time as annual auction
5. **No temporal leakage** — all features use prior-year data available at auction time

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Live nodal_f0_q (not research lookup) | Research lookup had 1.3% coverage gaps on real trades |
| Canonical `load_data_with_replacement` | Consistent pbase pattern, proper backward fill |
| `missing_base_action: "feature_only"` | ~2% of R1 paths lack f0 data, use 1_rev as fallback |
| Raise-only R1 (no class pooling) | PJM precedent, MISO has only 2 classes (larger cells) |
| Prior-year f0 stitch (not delivery months) | No temporal leakage, matches research archive code |
