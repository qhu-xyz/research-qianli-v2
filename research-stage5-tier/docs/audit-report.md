# Stage 5 Audit Report: ML Constraint Ranking for MISO FTR Tier Signal

**Date:** 2026-03-09
**Author:** Research team (automated + human review)
**Audience:** Teammates familiar with V6.2B production formula, auditing the ML replacement
**Status:** v10e-lag1 is the production-safe champion. Ready for peer audit.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What V0 (Production Formula) Does](#2-what-v0-production-formula-does)
3. [What We Changed and Why](#3-what-we-changed-and-why)
4. [Ground Truth: How We Measure "Right"](#4-ground-truth-how-we-measure-right)
5. [Evaluation Methodology](#5-evaluation-methodology)
6. [Metric Definitions](#6-metric-definitions)
7. [Results](#7-results)
8. [The Temporal Leakage Discovery](#8-the-temporal-leakage-discovery)
9. [Feature Audit](#9-feature-audit)
10. [Open Questions for Auditor](#10-open-questions-for-auditor)
11. [How to Reproduce](#11-how-to-reproduce)
12. [File Reference](#12-file-reference)

---

## 1. Executive Summary

We built a LightGBM model that ranks MISO transmission constraints by predicted binding
severity. It replaces the hand-tuned V6.2B formula (v0) for f0/onpeak tier assignment.

**Bottom line:**

| Metric | v0 (formula) | v10e-lag1 (ML) | Improvement |
|--------|-------------|---------------|-------------|
| VC@20 (dev) | 0.2817 | 0.4137 | **+47%** |
| VC@20 (holdout) | 0.1835 | 0.3529 | **+92%** |
| Recall@20 (holdout) | 0.1500 | 0.3021 | **+101%** |
| NDCG (holdout) | 0.4224 | 0.5497 | **+30%** |

The key new signal is **binding frequency** — how often each constraint has actually bound
in recent months, computed from realized day-ahead market data. This single concept drives
67% of model importance. The remaining 33% comes from V6.2B/Spice6 features (the same
data the formula already uses).

A temporal leakage bug was discovered and fixed during development (see Section 8). The
reported results are from the **fixed** version (v10e-lag1). Earlier versions (v9, v10e)
had inflated results and should not be used for production estimates.

---

## 2. What V0 (Production Formula) Does

V0 is an exact reproduction of the V6.2B production signal. It ranks constraints using:

```
rank_ori = 0.60 * da_rank_value + 0.30 * density_mix_rank_value + 0.10 * density_ori_rank_value
```

Where:
- **da_rank_value** (60% weight): Percentile rank of `shadow_price_da`, a 60-month
  historical lookback of day-ahead shadow prices. Lower = more historically congested.
  This is NOT the realized DA shadow price for the delivery month — it's a backward-looking
  statistic computed by the V6.2B pipeline before submission.
- **density_mix_rank_value** (30% weight): Percentile rank of Spice6 Monte Carlo flow
  simulation output (mixed outage scenarios). Lower = higher predicted flow.
- **density_ori_rank_value** (10% weight): Same as above but for original (base) outage
  scenarios.

All three are within-month percentile ranks where **lower = more likely to bind**.

**Verification:** We reproduced v0's `rank_ori` with `max_abs_diff = 0.0` against the
production V6.2B parquet output. The formula is exact.

---

## 3. What We Changed and Why

### The core insight: binding frequency

The V6.2B formula uses only *predictive* features — flow simulations and historical shadow
prices. It does not use the most obvious signal: **did this constraint actually bind in
recent months?**

We compute `binding_freq` from realized day-ahead shadow prices:

```
binding_freq_N(constraint, month M) = (# months in last N where constraint bound) / N
```

Where "bound" means the constraint had nonzero realized DA shadow price in that month.

Example: if constraint 72691 bound in 4 of the last 12 months, `binding_freq_12 = 0.333`.

This feature captures persistence — constraints that bind tend to keep binding due to
physical grid topology and generation patterns. The formula's `da_rank_value` partially
captures this (it's a 60-month lookback), but binding_freq provides a much more direct and
recent signal.

### The model

We use LightGBM regression (not ranking) because:
- The target is sparse (88% of constraints are non-binding in any month)
- Regression handles sparse targets better than lambdarank for top-K precision
- This was empirically validated: regression beat lambdarank on VC@20 by 5-15%

**9 features** (pruned from 14 — fewer features = less noise):

| # | Feature | Source | Importance | What it captures |
|---|---------|--------|-----------|-----------------|
| 1 | binding_freq_12 | Realized DA | 36.4% | Binding rate over prior 12 months |
| 2 | v7_formula_score | V6.2B cols | 19.4% | 0.85 * da_rank_value + 0.15 * density_ori_rank_value |
| 3 | binding_freq_15 | Realized DA | 16.3% | Binding rate over prior 15 months (seasonal) |
| 4 | da_rank_value | V6.2B parquet | 10.0% | 60-month historical DA shadow price rank |
| 5 | binding_freq_6 | Realized DA | 9.0% | Binding rate over prior 6 months |
| 6 | binding_freq_1 | Realized DA | 2.9% | Did it bind 2 months ago? |
| 7 | binding_freq_3 | Realized DA | 2.7% | Binding rate over prior 3 months |
| 8 | prob_exceed_110 | Spice6 density | 2.3% | P(flow > 110% thermal limit) |
| 9 | constraint_limit | Spice6 density | 0.9% | MW thermal limit |

Note: `v7_formula_score` is computed by us as `0.85 * da_rank_value + 0.15 * density_ori_rank_value`
(optimized weights from v7 experiments). Since `da_rank_value` also appears as its own
feature (#4), total model dependence on `da_rank_value` is ~26.5%.

**Training config:**
- 8-month rolling window, no validation set
- 1-month production lag (see Section 8)
- Tiered labels: 0 (non-binding), 1 (bottom 50% binding), 2 (top 50%), 3 (top 20%)
- LightGBM: lr=0.05, 31 leaves, 100 trees, num_threads=4
- Monotone constraints: all binding_freq features constrained to positive effect

---

## 4. Ground Truth: How We Measure "Right"

### What is NOT the ground truth

`shadow_price_da` from the V6.2B parquet is **not** realized data. It is a backward-looking
60-month statistic computed by the pipeline. Stage 4 of this research used it as ground
truth, which created a circular evaluation (Spearman = -1.0 between `da_rank_value` and
`shadow_price_da` because one is the rank of the other). Stage 4 was abandoned for this reason.

### What IS the ground truth

**Realized day-ahead shadow prices** for the delivery month, fetched independently:

```python
MisoApTools().tools.get_da_shadow_by_peaktype(start, end, peak_type="onpeak")
```

Aggregation: `abs(sum(shadow_price))` per constraint per month. This nets opposing hours
within a month, then takes the absolute value. Only constraints with nonzero aggregate
shadow price are considered "binding."

**Cached**: 79 months (2019-06 to 2025-12) in `data/realized_da/{YYYY-MM}.parquet`.

**Join to V6.2B**: Left join on `constraint_id` (String type). Non-matching V6.2B
constraints get `realized_sp = 0`. Typically ~68-81 constraints match per month out of
~550-780 V6.2B constraints (~12% binding rate).

**Verification (from 19-point audit):**
- `Spearman(shadow_price_da, realized_sp) = 0.22` — low, confirming these are different data
- `Spearman(da_rank_value, realized_sp) = -0.22` — moderate, as expected for a useful-but-imperfect feature
- If shadow_price_da were the target, correlation would be ~1.0. It is not.

---

## 5. Evaluation Methodology

### Train/test split: rolling window

For each evaluation month M:
1. **Training**: Load V6.2B features + realized DA labels for months M-9 through M-2
   (8 months, with 1-month production lag)
2. **Test**: Load V6.2B features for month M, evaluate against month M's realized DA

Each training month has its own independent realized DA labels. There is no label sharing
or future information in training.

### Dev evaluation period

**36 months: 2020-06 through 2023-05.** Used during development for all model comparisons.

In practice, we evaluate on 12 quarterly sample months (2020-09, 2020-12, ..., 2023-03,
2023-05) for iteration speed, then validate on all 36 for final numbers. The 12-month
sample captures quarterly variation; the full 36 confirms no cherry-picking.

### Holdout period

**24 months: 2024-01 through 2025-12.** Reserved as a one-time unseen test. Results are
computed once and frozen — we do not iterate on holdout performance.

The holdout period is harder than dev: ~800 constraints/month vs ~600, and the constraint
universe shifts over time (new lines built, old constraints retired).

### Why these periods?

- Dev starts at 2020-06 because we need 8 training months before the first eval, and
  realized DA cache starts at 2019-06.
- Dev ends at 2023-05 to leave a >6 month gap before holdout (2024-01), avoiding temporal
  proximity effects.
- Holdout covers the most recent data (2024-2025) to test generalization to current
  market conditions.

---

## 6. Metric Definitions

All metrics are computed per-month, then averaged across evaluation months. Higher = better
for all metrics.

### VC@K — Value Capture at K

**What it measures**: If you could only look at K constraints, what fraction of total
binding value would you capture?

```
VC@K = sum(realized_sp[top_K_by_score]) / sum(realized_sp[all])
```

Sort constraints by model score (descending), take top K, sum their realized shadow prices,
divide by total. VC@20 = "how much of the action do we capture in our top 20 picks?"

**Why it matters**: We trade a limited number of FTR paths. VC@20 directly measures whether
our top picks contain the highest-value constraints.

### Recall@K — Overlap at K

**What it measures**: Of the K truly most valuable constraints, how many does the model
put in its top K?

```
Recall@K = |true_top_K ∩ predicted_top_K| / K
```

**Why it matters**: Complementary to VC@K. VC@K can be high by catching one very large
constraint; Recall@K requires catching many of the top constraints.

### NDCG — Normalized Discounted Cumulative Gain

**What it measures**: Full-ranking quality with position-weighted scoring. Items ranked
higher get exponentially more credit.

```
DCG = Σ realized_sp[rank_i] / log2(i + 2)
NDCG = DCG / ideal_DCG
```

**Why it matters**: Measures the entire ranking, not just top K. Penalizes putting good
constraints far down the list.

### Spearman Rank Correlation

**What it measures**: Monotonic agreement between predicted scores and actual values across
all constraints.

```
Spearman = rank_correlation(scores, realized_sp)
```

**Why it matters**: Tests whether the model's ordering agrees with reality globally. Less
sensitive to top-K performance; more about overall calibration.

---

## 7. Results

### Dev evaluation (36 months, 2020-06 to 2023-05)

| Metric | v0 (formula) | v10e-lag1 (ML) | Δ vs v0 |
|--------|-------------|---------------|---------|
| VC@20 | 0.2817 | **0.4137** | +47% |
| VC@50 | 0.4653 | **0.5631** | +21% |
| VC@100 | 0.6008 | **0.7195** | +20% |
| Recall@20 | 0.1833 | **0.3278** | +79% |
| NDCG | 0.4423 | **0.5837** | +32% |
| Spearman | 0.2045 | **0.2989** | +46% |

### Holdout (24 months, 2024-01 to 2025-12)

| Metric | v0 (formula) | v10e-lag1 (ML) | Δ vs v0 |
|--------|-------------|---------------|---------|
| VC@20 | 0.1835 | **0.3529** | +92% |
| VC@50 | 0.3947 | **0.5442** | +38% |
| VC@100 | 0.5924 | **0.6807** | +15% |
| Recall@20 | 0.1500 | **0.3021** | +101% |
| NDCG | 0.4224 | **0.5497** | +30% |
| Spearman | 0.1946 | **0.3226** | +66% |

### Interpreting these numbers

- **VC@20 = 0.35 (holdout)**: Our top 20 constraints capture 35% of total binding value,
  vs 18% for the formula. Nearly double.
- **Recall@20 = 0.30 (holdout)**: We correctly identify 6 of the top 20 binding constraints,
  vs 3 for the formula.
- **Holdout < dev**: Expected. 2024-2025 has more constraints (~800 vs ~600) and the
  constraint universe shifted. The model still generalizes well — improvements are consistent
  across all metrics.
- **VC@100 = 0.68 (holdout)**: With 100 picks we capture 68% of binding value. The
  remaining 32% is spread across ~250+ constraints not in V6.2B's universe at all.

### Month-level variance

These are averages. Individual months vary significantly:
- Best months: VC@20 > 0.60 (few dominant constraints, model identifies them)
- Worst months: VC@20 < 0.10 (binding spread across many constraints, or novel constraints)
- The ML model's worst months are generally no worse than the formula's worst months

---

## 8. The Temporal Leakage Discovery

### The problem

For f0 (front-month), signal for auction month M is submitted **~mid of month M-1**.
At that moment:

| Data | Available? |
|------|-----------|
| Realized DA through M-2 | Yes (complete month) |
| Realized DA for M-1 | **No** (only ~12 days, incomplete) |
| Realized DA for M | No (hasn't happened yet) |
| V6.2B parquet for M | Yes (generated by pipeline before submission) |
| Spice6 density for M | Yes (generated by pipeline before submission) |

### What went wrong (v9, v10, v10e)

The original code computed binding_freq using `months < M`, which includes M-1. This
created **two layers of leakage**:

1. **Feature leak**: `binding_freq` for test month M used M-1's realized DA to compute
   "did it bind last month?" — but at mid-M-1, we don't have that complete data.
2. **Row leak**: Training included month M-1 as a training row with its `realized_sp`
   label. At mid-M-1, that label doesn't exist yet (M-1 hasn't finished).

The `binding_freq_1` feature was especially inflated — "did it bind last month?" is a
strong signal, but "did it bind 2 months ago?" is much weaker. With leakage, bf_1 had
44% feature importance. Without, it drops to 3%.

### The fix (v10e-lag1)

Apply a 1-month production lag to everything derived from realized DA:

| Component | Leaky (v10e) | Fixed (v10e-lag1) |
|-----------|-------------|-------------------|
| Training months | M-8 .. M-1 | M-9 .. M-2 |
| Training labels | realized_sp for M-8 .. M-1 | realized_sp for M-9 .. M-2 |
| bf for training month T | months < T | months < T-1 |
| bf for test month M | months < M | months < M-1 |

**Concrete example: eval month 2025-03 (submitted mid-February 2025)**

- Training months: 2024-06 through 2025-01 (NOT 2025-02)
- bf for training month 2025-01: uses realized DA through Nov 2024 (months < Dec 2024)
- bf for test month 2025-03: uses realized DA through Jan 2025 (months < Feb 2025)
- **2025-02 data is not used anywhere** — not as label, not as training feature, not as test feature

### Cost of the fix

| Metric | v10e (leaky) | v10e-lag1 (safe) | Drop |
|--------|-------------|-----------------|------|
| VC@20 dev | 0.4536 | 0.4137 | -8.8% |
| VC@20 holdout | 0.4230 | 0.3529 | -16.6% |
| Recall@20 holdout | 0.3792 | 0.3021 | -20.3% |

The lag costs 6-20% depending on metric. This is the cost of honesty — the signal is real,
just less fresh. All reported results in Section 7 are from the **fixed** version.

---

## 9. Feature Audit

### Audit framework

Every feature must pass: **"At the moment we submit this signal (~mid M-1), is this data
actually available?"**

| Feature | Source | Available at mid(M-1)? | Risk | Importance |
|---------|--------|----------------------|------|-----------|
| binding_freq_1 | Realized DA, months < M-1 | Yes — uses through M-2 | **Clean** (lag verified) | 2.9% |
| binding_freq_3 | Realized DA, months < M-1 | Yes — uses through M-2 | **Clean** (lag verified) | 2.7% |
| binding_freq_6 | Realized DA, months < M-1 | Yes — uses through M-2 | **Clean** (lag verified) | 9.0% |
| binding_freq_12 | Realized DA, months < M-1 | Yes — uses through M-2 | **Clean** (lag verified) | 36.4% |
| binding_freq_15 | Realized DA, months < M-1 | Yes — uses through M-2 | **Clean** (lag verified) | 16.3% |
| v7_formula_score | V6.2B parquet for M | Probably — production pipeline generates before submission | **Unverified** | 19.4% |
| da_rank_value | V6.2B parquet for M | Probably — same as above | **Unverified** | 10.0% |
| prob_exceed_110 | Spice6 density for M | Probably — same as above | **Unverified** | 2.3% |
| constraint_limit | Spice6 density for M | Probably — same as above | **Unverified** | 0.9% |

**Summary**: 67.3% of model importance is from features we fully control and have verified
the lag on. 32.7% comes from V6.2B/Spice6 parquets.

### The V6.2B/Spice6 provenance question

The V6.2B and Spice6 parquets we use were written **2025-11-12** as a bulk backfill. At
backfill time, the pipeline had access to all data through November 2025. We cannot verify
from the data alone whether:

- **(a)** The backfill correctly reproduces point-in-time behavior (uses only data available
  at original submission time) — **no leakage**
- **(b)** The backfill used data available at backfill time — **potential leakage in
  `shadow_price_da` and flow forecasts**

These are production signal features — the production pipeline generates them before each
auction, so they SHOULD be available at submission time. The backfill SHOULD reproduce
production behavior. But "should" ≠ "verified."

**Evidence is mixed:**
- `shadow_price_da` changes 77.6% between adjacent months — high for a "60-month lookback"
- But `Spearman(shadow_price_da, same-month realized_sp) = 0.18` — too low for direct leakage
- Adjacent months share ~65% identical `shadow_price_da` values — consistent with a slowly-updating lookback

**Recommendation**: Verify with the pipeline team that the backfill respected point-in-time
inputs. If not, the 32.7% unverified importance may be partially inflated, but the 67.3%
from binding_freq is clean regardless.

---

## 10. Open Questions for Auditor

1. **V6.2B backfill point-in-time**: Does the V6.2B pipeline's backfill reproduce what
   would have been generated at original submission time? Or does it use data available
   at backfill time (2025-11)?

2. **Spice6 density backfill**: Same question for Spice6 Monte Carlo simulations. Were
   the outage scenarios and flow forecasts computed with point-in-time inputs?

3. **binding_freq stability**: The model relies heavily on binding persistence. Are there
   known regime changes (new transmission lines, market rule changes) that could break
   this assumption? 2024-2025 holdout suggests it still works, but future regimes may differ.

4. **Constraint universe drift**: V6.2B covers ~550-780 constraints per month, but ~200-250
   DA binding constraints are NOT in V6.2B at all. Is the V6.2B constraint selection process
   stable, or could it shift and invalidate historical binding_freq patterns?

5. **Tier assignment**: The current model outputs a continuous score. Converting to 5 tiers
   (as consumed by pmodel) requires a mapping. This has not been implemented yet.

---

## 11. How to Reproduce

### Environment setup

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-stage5-tier
```

### Run the champion experiment (v10e-lag1)

```bash
python scripts/run_v10e_lagged.py
```

This runs both dev (36 months) and holdout (24 months) evaluation. Expected runtime: ~3-5
minutes. Output includes per-month metrics and aggregate comparisons.

### Run the formula baseline (v0)

```bash
python ml/benchmark.py --version-id v0 --full
```

### Verify the V6.2B formula reproduction

```bash
python scripts/reproduce_v62b.py
```

Should print `max_abs_diff = 0.0` for all months.

### Check the pipeline audit

```bash
python scripts/run_pipeline_audit.py
```

Runs the 19-point audit checking ground truth, joins, labels, and score direction.

---

## 12. File Reference

| File | Purpose |
|------|---------|
| `registry/f0/onpeak/v10e-lag1/` | Champion: config, metrics, detailed notes |
| `registry/f0/onpeak/v10e-lag1/NOTES.md` | Full leakage analysis with concrete examples |
| `registry/f0/onpeak/v0/` | Formula baseline metrics |
| `holdout/f0/onpeak/v10e-lag1/metrics.json` | Holdout results (immutable) |
| `audit.md` | 19-point pipeline audit (pre-binding-freq) |
| `experiment-setup.md` | Glossary, problem statement, data paths |
| `mem.md` | Working memory with version history and learnings |
| `ml/evaluate.py` | Metric implementations (VC@K, Recall@K, NDCG, Spearman) |
| `ml/data_loader.py` | Train/test split logic, rolling window, month cache |
| `ml/train.py` | LightGBM training, tiered labels, monotone constraints |
| `ml/realized_da.py` | Ground truth fetching and caching |
| `ml/v62b_formula.py` | V6.2B formula computation |
| `scripts/run_v10e_lagged.py` | Champion experiment script |
| `scripts/run_v9_binding_freq.py` | Original binding_freq experiment (leaky) |
| `CLAUDE.md` | Development rules including temporal leakage warning |
