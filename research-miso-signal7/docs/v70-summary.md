# V7.0 MISO FTR Constraint Signal — Summary

## What V7.0 Is

V7.0 replaces the V6.2B formula-based constraint ranking for **f0** (front-month) and **f1** (second-month) with a LightGBM LambdaRank model trained on realized DA binding outcomes. All other period types (f2, f3, q2–q4) are **exact V6.2B passthrough** — unchanged, bit-identical.

V7.0 is a **plug-and-play replacement** for V6.2B. Same signal path structure, same columns, same dtypes, same index format. Downstream code (pmodel ftr22/ftr23) requires zero changes.

---

## Why

V6.2B uses a static formula:

```
rank_ori = 0.60 × da_rank_value + 0.30 × density_mix + 0.10 × density_ori
```

This was the best available approach without realized DA ground truth. V7.0 leverages realized DA outcomes (which constraints actually bound) to train a model that learns which constraints are most likely to bind in future months.

---

## Performance (Holdout, 2024–2025)

Walk-forward evaluation: each month trains a fresh model on trailing 8 months, scores the current month with no future data.

| Slice | V6.2B VC@20 | V7.0 VC@20 | Improvement |
|-------|:-----------:|:----------:|:-----------:|
| f0/onpeak (24 mo) | 0.1835 | **0.3529** | +92% |
| f0/offpeak (24 mo) | 0.2075 | **0.3780** | +82% |
| f1/onpeak (19 mo) | 0.2209 | **0.3677** | +66% |
| f1/offpeak (19 mo) | 0.2492 | **0.3561** | +43% |

VC@20 = fraction of total binding value captured by the top 20 ranked constraints. These numbers are identical to research-stage5-tier holdout metrics — confirmed per-month to 4 decimal places.

### Early planning year (July/August f1)

V7.0 also wins in the hardest months — early planning year where less delivery-month history is available:

| Month | V6.2B VC@20 | V7.0 VC@20 | Improvement | Win Rate |
|-------|:-----------:|:----------:|:-----------:|:--------:|
| July | 0.4031 | 0.4896 | +21.5% | 9/10 |
| August | 0.3314 | 0.3997 | +20.6% | 8/10 |

---

## What Was Built

### Code (`research-miso-signal7/`)

| File | Purpose |
|------|---------|
| `v70/inference.py` | Train LightGBM on history, score target month. Returns `(constraint_ids, scores)`. |
| `v70/signal_writer.py` | Row-percentile rank/tier conversion with V6.2B tie-breaking. |
| `v70/cache.py` | Realized DA cache preflight — ensures all required months (labels + BF lookback) are cached before training. |
| `v70/loader.py` | Fallback loader: tries V7.0, falls back to V6.2B on `FileNotFoundError` only. |
| `scripts/generate_v70_signal.py` | CLI entrypoint: generates V7.0 for one auction month. |
| `scripts/validate_v70.py` | Automated validation: gates A–G (holdout reproduction, improvement, passthrough, schema, rank/tier, scores, shift factors). |

### Generated Signal

- **Path**: `signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V7.0.R1/`
- **f0 coverage**: 75 months (2020-01 → 2026-03), 71 ML-scored, 4 passthrough
- **f1 coverage**: 63 months (2020-01 → 2026-03), 55 ML-scored, 8 passthrough
- **f2/f3/q2–q4**: exact V6.2B passthrough wherever they exist
- **Shift factors**: bit-identical to V6.2B for all period types

### Docs

| Document | Content |
|----------|---------|
| `docs/v70-summary.md` | This file. |
| `docs/v70-design-choices.md` | 10 design decisions with alternatives considered and rationale. |
| `docs/v70-deployment-handoff.md` | Full implementation spec: format, lag rules, ML config, data dependencies. |
| `docs/v70-validation-plan.md` | 13 validation gates (A–M) with code snippets and pass criteria. |
| `docs/validation-output.txt` | Saved output from `validate_v70.py --full-holdout`: 76/76 pass. |

---

## ML Model

### Features (9)

| Feature | Description |
|---------|-------------|
| `binding_freq_1` | Fraction of months bound in last 1 month |
| `binding_freq_3` | ... last 3 months |
| `binding_freq_6` | ... last 6 months |
| `binding_freq_12` | ... last 12 months |
| `binding_freq_15` | ... last 15 months |
| `v7_formula_score` | Optimized V6.2B-style blend (da_rank + density_ori, weights vary per slice) |
| `prob_exceed_110` | Spice6 exceedance probability at 110% of thermal limit |
| `constraint_limit` | MW thermal limit |
| `da_rank_value` | Historical DA shadow price percentile (from V6.2B) |

Binding frequency features are the primary signal (~50–70% importance). They measure how often a constraint actually bound in realized DA over different lookback windows.

### Training

- **Backend**: LightGBM LambdaRank
- **Labels**: Tiered (4 levels: non-binding / bottom-50% binding / top-50% / top-20%)
- **Training window**: 8 months, walk-forward retrain per eval month
- **Validation**: 0 months (8-train/0-val beats 6-train/2-val)
- **Params**: 100 trees, lr=0.05, 31 leaves, num_threads=4, seed=42

### Temporal Safety

All features and labels respect the submission-time cutoff: signal for month M is submitted ~mid M−1, so only realized DA through M−2 is used. This was validated by comparing against the pre-lag version (v10e vs v10e-lag1), which showed 6–20% inflation without proper lag.

---

## Key Design Decisions

1. **Row-percentile tiering** (not dense-rank/K): ML produces ~55% unique scores. Dense-rank/K gives tier 0 = 11%, tier 4 = 45%. Row-percentile guarantees ~20% per tier, with ties broken by V6.2B rank_ori then index.

2. **V6.2B formula score as feature**: The formula is included as one of 9 features with optimized blend weights (not the original 60/30/10). density_mix gets zero weight — it adds noise.

3. **Join on constraint_id, not positional**: ML inference and signal assembly load V6.2B through different code paths (polars vs pandas). Joining by constraint_id prevents silent row misalignment.

4. **BF_LAG=1 always**: Binding frequency cutoff is keyed on auction month, not delivery month. Same for both f0 and f1.

5. **SO_MW_Transfer exception**: Preserved from V6.2B — forced to tier 1 regardless of ML score.

6. **Passthrough for f2+**: ML gains uncertain for longer horizons (sparser data, shorter history). Rather than risk degradation, pass through V6.2B unchanged.

Full rationale for all 10 decisions: `docs/v70-design-choices.md`

---

## What Changes for Downstream

**Nothing.** V7.0 is consumed identically to V6.2B:

```python
# Before (V6.2B)
df = ConstraintsSignal("miso", "...V6.2B.R1", ptype, ctype).load_data(month)

# After (V7.0)
df = ConstraintsSignal("miso", "...V7.0.R1", ptype, ctype).load_data(month)
```

Same columns, same dtypes, same index format, same row count. The `tier`, `rank`, and `equipment` columns (used by ftr22/ftr23) all follow the same conventions.

**One caveat**: `rank_ori` in ML slices now contains raw LightGBM scores (higher = more binding), which is **opposite polarity** from V6.2B (lower = more binding). Downstream code does not appear to read `rank_ori` directly — it uses `tier` and `rank` — but this should be confirmed before deployment.

### Fallback

For months where V7.0 is not available, use the fallback loader:

```python
from v70.loader import load_constraints
df, source = load_constraints("f0", "onpeak", pd.Timestamp("2025-01"))
# source = "...V7.0.R1" or "...V6.2B.R1"
```

---

## Validation

All validation is automated via `scripts/validate_v70.py`:

```bash
# Full holdout + sample months
python scripts/validate_v70.py --full-holdout --months 2025-01 2026-03

# Specific months
python scripts/validate_v70.py --months 2025-07 2025-08
```

### Gates (76/76 passed)

| Gate | What | Mandatory? |
|------|------|:----------:|
| A | ML beats V6.2B by >20–40% per slice | Yes |
| B | Per-month VC@20 matches stage5 holdout exactly (±0.001) | Yes |
| C | f2/f3/q2–q4 bit-identical to V6.2B | Yes |
| D | Columns, dtypes, index, row count match V6.2B | Yes |
| E | Rank range (0,1], ~20% per tier, tier formula, monotonicity | Yes |
| F | No NaN/Inf scores, correct polarity | Yes |
| G | Shift factors bit-identical to V6.2B | Yes |

---

## How to Generate V7.0 for a New Month

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate

# Generate
python /home/xyz/workspace/research-qianli-v2/research-miso-signal7/scripts/generate_v70_signal.py \
    --auction-month 2026-04

# Validate
python /home/xyz/workspace/research-qianli-v2/research-miso-signal7/scripts/validate_v70.py \
    --months 2026-04
```

Runtime: ~3–5 seconds per month (all slices). Requires Ray for realized DA fetching (if cache is cold).

---

## Repository Structure

```
research-miso-signal7/
├── v70/
│   ├── __init__.py
│   ├── inference.py       # ML training + scoring (9 features, LambdaRank)
│   ├── signal_writer.py   # rank/tier computation
│   ├── cache.py           # realized DA cache preflight
│   └── loader.py          # V7.0-with-fallback loader
├── scripts/
│   ├── generate_v70_signal.py  # CLI: generate signal for one month
│   └── validate_v70.py         # CLI: run validation gates
├── docs/
│   ├── v70-summary.md              # this file
│   ├── v70-design-choices.md        # 10 design decisions
│   ├── v70-deployment-handoff.md    # full implementation spec
│   ├── v70-validation-plan.md       # 13 gate definitions
│   └── validation-output.txt        # saved validation run (76/76 pass)
└── .gitignore
```

### Dependency: `research-stage5-tier/ml/`

V7.0 imports from `research-stage5-tier/ml/` for:
- `config.py`: auction schedule, month math, training month selection
- `data_loader.py`: V6.2B + spice6 + realized DA loading
- `features.py`: feature matrix preparation, query groups
- `train.py`: LightGBM training and prediction
- `spice6_loader.py`: Spice6 density loading (handles both old/new schema)

These are added to `sys.path` at import time. For production deployment, this dependency should be packaged properly.
