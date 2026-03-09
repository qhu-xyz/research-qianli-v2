# Stage 5 — Working Memory

## Status: ML Champion Across All 4 Slices (f0/f1 x onpeak/offpeak) — 2026-03-09

---

## Research Summary

### What we built

A LightGBM LambdaRank model with tiered labels (v10e-lag1 for f0, v2 for f1) that predicts
which MISO transmission constraints will bind in the day-ahead market, using 9 features.
It replaces the V6.2B production formula (v0) which uses a hand-tuned 3-feature weighted sum.
Covers all 4 slices: f0/f1 x onpeak/offpeak.

### What it does

For each auction month M, rank ~600-800 constraints by predicted binding likelihood.
The top-ranked constraints are where we expect congestion (and therefore FTR value).

### How well it works

**Cross-Slice Holdout VC@20 (2024-2025, out-of-sample):**

| Slice | v0 (Formula) | v1 (Blend) | ML | ML vs v0 |
|-------|:---:|:---:|:---:|:---:|
| f0/onpeak (24mo) | 0.1835 | -- | **0.3529** | +92% |
| f0/offpeak (24mo) | 0.2075 | -- | **0.3780** | +82% |
| f1/onpeak (19mo) | 0.2209 | 0.2337 | **0.3677** | +66% |
| f1/offpeak (19mo) | 0.2492 | 0.2747 | **0.3561** | +43% |

**Cross-Slice Dev VC@20:**

| Slice | v0 | ML | ML vs v0 |
|-------|:---:|:---:|:---:|
| f0/onpeak (36mo) | 0.2817 | **0.4137** | +47% |
| f0/offpeak (36mo) | 0.3438 | **0.5200** | +51% |
| f1/onpeak (30mo) | 0.3413 | **0.4557** | +34% |
| f1/offpeak (30mo) | 0.3579 | **0.4499** | +26% |

**f0/onpeak detailed (flagship slice):**

| Metric | v0 (formula) | v6b (prev best) | v10e-lag1 (current) | v10e-lag1 vs v0 |
|--------|-------------|-----------------|--------------------|-----------------|
| VC@20 | 0.2817 | 0.3503 (+24%) | **0.4137** | **+47%** |
| VC@50 | 0.4653 | — | **0.5631** | **+21%** |
| VC@100 | 0.6008 | 0.6241 (+4%) | **0.7195** | **+20%** |
| Recall@20 | 0.1833 | 0.2375 (+30%) | **0.3278** | **+79%** |
| NDCG | 0.4423 | 0.5567 (+26%) | **0.5837** | **+32%** |
| Spearman | 0.2045 | 0.1712 (-16%) | **0.2989** | **+46%** |

**f0/onpeak holdout:**

| Metric | v0 (formula) | v6b (prev best) | v10e-lag1 (current) | v10e-lag1 vs v0 |
|--------|-------------|-----------------|--------------------|-----------------|
| VC@20 | 0.1835 | 0.2709 (+48%) | **0.3529** | **+92%** |
| VC@50 | 0.3947 | 0.4183 (+6%) | **0.5442** | **+38%** |
| VC@100 | 0.5924 | 0.5854 (-1%) | **0.6807** | **+15%** |
| Recall@20 | 0.1500 | 0.1937 (+29%) | **0.3021** | **+101%** |
| NDCG | 0.4224 | 0.4462 (+6%) | **0.5497** | **+30%** |
| Spearman | 0.1946 | 0.1891 (-3%) | **0.3226** | **+66%** |

---

## The Model: v10e-lag1

### Features (9)

| Feature | Source | Importance | Monotone | What it captures |
|---------|--------|-----------|----------|-----------------|
| binding_freq_12 | Realized DA cache | 36.4% | +1 | Fraction of prior 12 months constraint was binding |
| v7_formula_score | V6.2B parquet | 19.4% | -1 | 0.85*da_rank_value + 0.15*density_ori_rank_value |
| binding_freq_15 | Realized DA cache | 16.3% | +1 | Fraction of prior 15 months (seasonal patterns) |
| da_rank_value | V6.2B parquet | 10.0% | -1 | Percentile rank of 60-month historical DA shadow price |
| binding_freq_6 | Realized DA cache | 9.0% | +1 | Fraction of prior 6 months |
| binding_freq_1 | Realized DA cache | 2.9% | +1 | Did it bind 2 months ago? (lagged) |
| binding_freq_3 | Realized DA cache | 2.7% | +1 | Fraction of prior 3 months |
| prob_exceed_110 | Spice6 density | 2.3% | +1 | Probability flow exceeds 110% of thermal limit |
| constraint_limit | Spice6 limit | 0.9% | 0 | MW thermal limit of constraint |

### Config

- **Method**: LightGBM LambdaRank (`backend="lightgbm"`), tiered labels (0/1/2/3)
- **Training**: 8 months rolling, 0 validation, production lag = N+1 (f0: lag 1, f1: lag 2)
- **Hyperparams**: lr=0.05, 31 leaves, 100 trees, num_threads=4
- **Blend weights** (da_rank / density_mix / density_ori):
  - f0: (0.85 / 0.00 / 0.15)
  - f1/onpeak: (0.70 / 0.00 / 0.30)
  - f1/offpeak: (0.80 / 0.00 / 0.20)

### v7_formula_score decomposition

`v7_formula_score` is NOT an independent feature. It's computed by us as a weighted sum
of two V6.2B parquet columns:

```
v7_formula_score = 0.85 * da_rank_value + 0.15 * density_ori_rank_value
```

Since `da_rank_value` also appears as its own feature (10%), the total model dependence
on `da_rank_value` is approximately: 10% (direct) + 0.85 × 19.4% (via formula) ≈ **26.5%**.

---

## Leakage Audit

### The two core leakage dimensions

1. **Feature leak**: Are features for target month M using information unavailable at
   decision time?
2. **Row leak**: Are training rows/labels including months whose outcomes would not yet
   be known at decision time?

### Production timing constraint

General rule: for period type fN, auction month M is submitted **~mid of month M-1**.
The production lag is **N+1** months.

| Period | Lag | Training window | bf cutoff | Auction timing |
|--------|:---:|-----------------|-----------|----------------|
| f0 | 1 | M-9..M-2 | months < M-1 | mid(M-1) for market M |
| f1 | 2 | M-10..M-3 | months < M-2 | mid(M-1) for market M+1 |
| f2 | 3 | M-11..M-4 | months < M-3 | mid(M-1) for market M+2 |
| f3 | 4 | M-12..M-5 | months < M-4 | mid(M-1) for market M+3 |

Why: fN's training month T has GT for T+N (delivery month). With lag L, T_max = M-L-1.
For no leakage: T_max + N < M-1, so L >= N+1.

### v9 through v10e: LEAKY (do not use for production estimates)

These versions had both leak types:
- **Feature leak**: `binding_freq` for test month M used `months < M`, which includes M-1.
  At mid of M-1, we don't have M-1's complete realized DA.
- **Row leak**: Training included month M-1 with its `realized_sp` label. At mid of M-1,
  that label doesn't exist yet.

**Impact**: v10e results were inflated by 6-20% depending on metric.
- bf_1 was especially inflated: "did it bind LAST month" (strong signal) vs
  "did it bind 2 months ago" (weaker signal)

### v10e-lag1: FIXED on both dimensions

For eval month M with lag=1:

| Component | Leaky (v10e) | Fixed (v10e-lag1) | Available at mid(M-1)? |
|-----------|-------------|-------------------|----------------------|
| Training months | M-8..M-1 | M-9..M-2 | M-2 is complete ✓ |
| Training labels | realized_sp for M-8..M-1 | realized_sp for M-9..M-2 | M-2 is complete ✓ |
| bf for training month T | months < T | months < T-1 | T-2 is complete ✓ |
| bf for test month M | months < M (includes M-1) | months < M-1 (through M-2) | M-2 is complete ✓ |
| V6.2B test features | M parquet | M parquet (unchanged) | See caveat below |
| Spice6 test features | M parquet | M parquet (unchanged) | See caveat below |
| Test label | M realized_sp | M realized_sp (unchanged) | Ground truth ✓ |

**Concrete example**: eval month 2025-03, submitted mid-February 2025.
- Training: 2024-06 through 2025-01 (features + labels). No 2025-02 data anywhere.
- bf for training month 2025-01: uses realized DA months < 2024-12 (through Nov 2024)
- bf for test month 2025-03: uses realized DA months < 2025-02 (through Jan 2025)
- 2025-02 realized DA is NOT used — not as label, not as feature, not in bf.

**Why v10e-lag1 is lower than v10e**: it is not worse because it is wrong — it is worse
because it is more realistic. You lose the freshest training month and binding_freq_1
becomes "2 months ago" instead of "last month." Short-horizon persistence signals weaken.
That is the correct cost of respecting production timing.

### Remaining caveat: V6.2B/Spice6 signal parquet provenance

The lag fix covers all realized-DA-derived features (67.3% of model importance) and
training labels. But 32.7% of importance comes from V6.2B parquet and Spice6 features
loaded for the test month:

| Feature | Source | Importance | Risk |
|---------|--------|-----------|------|
| binding_freq_* (5) | Our realized DA cache | 67.3% | **CLEAN** — lag verified |
| v7_formula_score | V6.2B parquet for M | 19.4% | **Unverified** |
| da_rank_value | V6.2B parquet for M | 10.0% | **Unverified** |
| prob_exceed_110 | Spice6 density for M | 2.3% | **Unverified** |
| constraint_limit | Spice6 limit for M | 0.9% | **Unverified** |

All V6.2B and Spice6 parquets were written **2025-11-12** (bulk backfill). At backfill
time, the pipeline had access to all data through November 2025. We cannot verify from
the data alone whether the pipeline:
- **(a)** Correctly respects point-in-time (uses only inputs available at original
  submission) — no leakage
- **(b)** Uses data available at backfill time — potential leakage in `shadow_price_da`
  and flow forecasts

Evidence is ambiguous: `shadow_price_da` for overlapping constraints changes 77.6% between
adjacent months, which is high for a "60-month historical lookback." But
Spearman(shadow_price_da, same-month realized_sp) is only 0.18, suggesting it's not
directly reflecting current-month outcomes.

**These are production signal features.** If V6.2B runs in production before each auction,
then by definition these features are available at submission time. The backfill should
reproduce production behavior. But "should" is not "verified" — **this needs confirmation
from the pipeline team.**

---

## Version History

### Phase 1: Baseline and ML foundations (v0-v6c)

- **v0** (formula): `rank_ori = 0.60*da_rank + 0.30*density_mix_rank + 0.10*density_ori_rank`.
  Exact reproduction of V6.2B production signal. Baseline for all comparisons.
- **v1-v4**: Feature set and label experiments. Key finding: tiered labels (0/1/2/3) fix
  rank-transform noise (+36% VC@20). Formula-as-feature helps (+10-18% VC@20).
- **v5** (lambdarank, 12f): comparable to formula on VC@20, better on coverage metrics
- **v6b** (regression, 13f): +5% VC@20 over formula on dev, +48% on holdout.
  Previous champion. Regression > lambdarank for top-k precision on sparse targets.
- **v6c** (lambdarank, 13f): +3.5% VC@100, best coverage. Underperforms v6b on holdout.

### Phase 2: Binding frequency breakthrough (v9-v10e)

- **v9** (14f, +bf_6): Single binding_freq_6 feature added. +34% VC@20 dev.
  Feature importance: bf_6=73.6%, everything else combined=26.4%.
  **LEAKY** — used M-1 data not available at submission time.
- **v10** (6f): Pruned 8 features with <2% importance. Multi-window bf (3/6/12).
- **v10e** (9f): Added bf_1, bf_15, da_rank_value back. Beat v9 on ALL metrics.
  **LEAKY** — same timing issue as v9.

### Phase 3: Temporal leakage fix (v10e-lag1)

- **v10e-lag1** (9f, lag=1): Production-safe version. Costs 6-20% vs leaky v10e
  but still +47-92% vs formula. Current champion.

### Feature importance evolution

| Feature | v9 (leaky) | v10e (leaky) | v10e-lag1 (safe) |
|---------|-----------|-------------|-----------------|
| binding_freq_6 | 73.6% | ~8% | 9.0% |
| binding_freq_1 | — | ~44% | 2.9% |
| binding_freq_12 | — | ~12% | **36.4%** |
| binding_freq_15 | — | ~4% | **16.3%** |
| v7_formula_score | 14.9% | ~6% | **19.4%** |
| da_rank_value | 6.1% | ~2% | **10.0%** |

With the lag, longer-window features (bf_12, bf_15) and the formula become much more
important. bf_1 collapses from 44% to 3% because "2 months ago" is much weaker than
"last month." This is the correct behavior — the model adapts to the information actually
available at decision time.

---

## Ground Truth and Data

### Ground truth = Realized DA shadow prices
- Source: `MisoApTools().tools.get_da_shadow_by_peaktype()`
- Cached: `data/realized_da/{YYYY-MM}.parquet` (79 months, 2019-06 to 2025-12)
- Each parquet has columns: `constraint_id`, `realized_sp` (= abs sum of DA shadow prices)
- NOT `shadow_price_da` from V6.2B (that's a historical lookback feature, not ground truth)

### V6.2B signal data
- Path: `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1/{month}/{ptype}/{class_type}`
- Period types: f0 (106mo), f1 (89mo), f2 (54mo), f3 (27mo)
- ~550-780 constraints per month, 21 columns
- Key columns used: `da_rank_value`, `density_ori_rank_value`, `constraint_id`
- Production formula output: `rank_ori` (verified exact match with our v0 computation)
- `shadow_price_da` ≡ `da_rank_value` (Spearman = -1.0, identical information)

### Spice6 density data
- Path: `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/auction_month={month}/...`
- Monte Carlo simulation output: probability of flow exceeding various thresholds
- Aggregated across outage date scenarios (mean per constraint)
- Key columns used: `prob_exceed_110`, `constraint_limit`

### Constraint universe stats
- ~600-800 V6.2B constraints per month
- ~300-400 DA binding constraints per month (total MISO)
- ~68-85 overlap (V6.2B ∩ DA binding) — this is what we predict
- ~200-250 DA binding constraints NOT in V6.2B (coverage gap, not addressable)
- ~12% base binding rate within V6.2B universe

---

## V6.2B Formula (Verified Exact)

```
rank_ori = 0.60 * da_rank_value + 0.30 * density_mix_rank_value + 0.10 * density_ori_rank_value
```

- Verified via reproduce_v62b.py: max_abs_diff = 0.0 for all months
- All three rank columns: lower = more binding (inverted percentile ranks)
- v7 optimized weights vary by slice (see Config above); density_mix gets zero weight everywhere

---

## Key ML Learnings

- **binding_freq is the #1 feature**: multi-window (1/3/6/12/15) better than single bf_6
- **Production lag matters**: 1-month lag costs 6-20% but is mandatory for honest evaluation
- **Longer bf windows are more robust to lag**: bf_12 and bf_15 become dominant with lag
- 8mo train / 0 val >> 6mo train / 2 val
- LightGBM LambdaRank with tiered labels (backend="lightgbm") is the final production method
- Note: v6b used regression and beat lambdarank at the time, but tiered labels fixed that gap
- Tiered labels (0/1/2/3) fix rank-transform noise: +36% VC@20
- Formula-as-feature: consistently helpful across all versions
- Feature pruning works: 9 features beats 14 features (less noise)
- Simple params (lr=0.05, 31 leaves, 100 trees) beat tuned params
- LightGBM deadlocks with 64 threads — always num_threads=4

---

## Bug Fixes

- **Temporal leakage in v9-v10e**: Used M-1 realized DA not available at f0 submission.
  Fixed with 1-month lag in v10e-lag1.
- **Label noise**: `_rank_transform_labels()` gave ~528 distinct levels to non-binding.
  Fixed with `_tiered_labels()` (0/1/2/3).
- **LightGBM threading deadlock**: Container reports 64 CPUs → `num_threads=4`.
- **Fork deadlock**: ProcessPoolExecutor copies broken lock state → use sequential.
- **NFS bottleneck**: In-memory `_MONTH_CACHE` cuts NFS reads by ~87%.
- **BLEND_WEIGHTS keyed wrong**: Was keyed by ptype only; f1/offpeak got onpeak's weights.
  Fixed by re-keying to `(ptype, class_type)`.
- **champion.json key mismatch**: f0/onpeak used `"champion"` key, compare.py read `"version"`.
  Fixed with fallback: `get("version") or get("champion")`.
- **Backend label wrong**: Config.json said "lightgbm_regression" but actual backend is
  "lightgbm" (LambdaRank). Fixed in save_results() and all existing config.json.
- **`_has_gt()` class_type-blind**: Only checked `{month}.parquet`, not `{month}_offpeak.parquet`.
  Fixed to check class_type-specific filename.
- **Holdout eval_config listed unevaluated months**: 2025-12 (f1, no GT for 2026-01) appeared
  in eval_months. Fixed by recording actual_months + skipped_months.

---

## Production Migration Assessment

See `production-migration/assessment.md`. Key findings:
- v0 reproduces 100% of production V6.2B signal for f0
- ML versions can produce same output format (replace rank_ori with ML score)
- pmodel consumes: `shadow_price`, `shadow_sign`, `tier`, `rank_ori` + `constraint_id`
- Gaps: tier assignment (small), score normalization (small), output writer (medium)

---

## File Index

### Registry (per-slice: `registry/{ptype}/{ctype}/`)
- `registry/f0/onpeak/v10e-lag1/` — f0/onpeak champion: metrics, config, notes
- `registry/f0/offpeak/v10e-lag1/` — f0/offpeak champion
- `registry/f1/onpeak/v0/`, `v1/`, `v2/` — f1/onpeak version ladder
- `registry/f1/offpeak/v0/`, `v1/`, `v2/` — f1/offpeak version ladder
- `registry/{ptype}/{ctype}/gates.json` — Gate thresholds (calibrated from v0)
- `registry/{ptype}/{ctype}/champion.json` — Current champion pointer
- `archive/registry/` — Legacy experiments (v1-v10d)

### Holdout (immutable: `holdout/{ptype}/{ctype}/`)
- `holdout/f0/onpeak/v10e-lag1/` — f0/onpeak holdout
- `holdout/f0/offpeak/v10e-lag1/` — f0/offpeak holdout
- `holdout/f1/{onpeak,offpeak}/v0/`, `v1/`, `v2/` — f1 holdout results

### Scripts
- `scripts/run_v0_formula_baseline.py` — Formula baseline (supports `--ptype`, `--class-type`)
- `scripts/run_v10e_lagged.py` — ML experiment runner (supports `--ptype`, `--class-type`)
- `scripts/run_f1_blend_search.py` — Blend weight optimizer for f1
- `scripts/gen_comparison_table.py` — Cross-slice comparison tables
- `scripts/run_v10_variants.py` — v10c-v10g feature search (f0 only)
- `scripts/run_v9_binding_freq.py` — Original v9 experiment (leaky, reference)

### Docs
- `CLAUDE.md` — Contains temporal leakage warning for future work
- `audit.md` — 19-point pipeline audit (pre-binding-freq)
- `docs/audit-report.md` — Standalone peer audit report
- `multi-period-extension.md` — f1/f2/f3 extension design and leakage analysis

## What's Next

- Verify V6.2B/Spice6 backfill uses point-in-time inputs (check pipeline source code)
- f2/f3 extension (same approach as f1, with lag=3 and lag=4 respectively)
- Production migration: implement tier assignment + output writer
- Partial recent data plan (see docs/plans/2026-03-09-partial-recent-data.md)
