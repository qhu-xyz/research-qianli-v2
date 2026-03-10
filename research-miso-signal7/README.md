# research-miso-signal7

V7.0 MISO FTR constraint signal — ML replacement for V6.2B formula scoring.

## What This Does

Replaces V6.2B's static formula (`0.60 × da_rank + 0.30 × density_mix + 0.10 × density_ori`) with a LightGBM LambdaRank model for **f0** and **f1** period types. All other period types pass through V6.2B unchanged.

**Result**: +43% to +92% VC@20 improvement on holdout (2024–2025).

## Quick Start

```bash
# Activate pmodel venv
cd /home/xyz/workspace/pmodel && source .venv/bin/activate

# Generate signal for one month
python /path/to/research-miso-signal7/scripts/generate_v70_signal.py --auction-month 2026-04

# Validate
python /path/to/research-miso-signal7/scripts/validate_v70.py --months 2026-04

# Full holdout validation
python /path/to/research-miso-signal7/scripts/validate_v70.py --full-holdout
```

## Repo Structure

```
research-miso-signal7/
│
├── v70/                          # Core package
│   ├── inference.py              # ML training + scoring (LambdaRank, 9 features)
│   ├── signal_writer.py          # Row-percentile rank/tier with V6.2B tie-breaking
│   ├── cache.py                  # Realized DA cache preflight (labels + BF lookback)
│   └── loader.py                 # V7.0 loader with V6.2B fallback
│
├── scripts/
│   ├── generate_v70_signal.py    # Generate V7.0 for one auction month
│   └── validate_v70.py           # Automated validation gates (A–G)
│
└── docs/
    ├── v70-summary.md            # What V7.0 is, performance, design decisions
    ├── v70-design-choices.md     # 10 design decisions with rationale
    ├── v70-deployment-handoff.md # Full implementation spec for deployment
    ├── v70-validation-plan.md    # 13 validation gates with code + criteria
    └── validation-output.txt     # Saved run: 76/76 gates passed
```

## Dependencies

- **`research-stage5-tier/ml/`** — ML pipeline (config, data loading, training, features). Added to `sys.path` at runtime.
- **pmodel venv** — Python environment with pbase, LightGBM, polars, pandas.
- **Ray** — Required for realized DA cache fetching (not for signal generation itself if cache is warm).

## Signal Paths

| Signal | Path |
|--------|------|
| V6.2B (source) | `signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1/` |
| V7.0 (output) | `signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V7.0.R1/` |
| Realized DA cache | `/opt/temp/qianli/spice_data/miso/realized_da/` |

## Coverage

- **f0**: 75 months (2020-01 → 2026-03)
- **f1**: 63 months (2020-01 → 2026-03, no May/Jun per MISO schedule)
- **f2/f3/q2–q4**: exact V6.2B passthrough

## Docs

Start with **[v70-summary.md](docs/v70-summary.md)** for a complete overview. See **[v70-design-choices.md](docs/v70-design-choices.md)** for why specific decisions were made. **[v70-deployment-handoff.md](docs/v70-deployment-handoff.md)** has the full implementation spec.
