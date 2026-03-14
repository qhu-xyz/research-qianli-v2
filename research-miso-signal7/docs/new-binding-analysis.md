# V7.0 vs V6.2B: Early Alarms & Signal Distinctiveness

**Date**: 2026-03-10
**Scope**: f0 onpeak/offpeak, holdout 2024-01 to 2025-12 (24 months)
**Ground truth**: Realized DA binding outcomes from cached parquet files

---

## Executive Summary

V7.0 is overwhelmingly better at discriminating binding from non-binding constraints (+57% top-20 captures, AUC 0.784 vs 0.668). However, V6.2B retains a structural advantage for **truly new constraints** that have never bound before, and for **persistent signal** during quiet periods when binding frequency decays.

The root cause is V7.0's heavy reliance on binding_freq (bf_1..bf_15), which is zero for constraints with no binding history. V6.2B's da_rank_value carries forward-looking historical congestion information that can flag constraints before they ever bind in realized DA.

---

## Study 1: Overall Discrimination

V7.0 wins **24/24 months** on average rank of binding constraints across the entire holdout.

### f0/onpeak (24 months)

| Month | #Bound | V6.2B avg rank | V7.0 avg rank | Delta | Top-20: V6.2B | Top-20: V7.0 |
|-------|-------:|---------------:|:-------------:|:-----:|:-------------:|:-------------:|
| 2024-01 | 78 | 0.3595 | 0.3032 | +0.056 | 29 | 44 |
| 2024-02 | 80 | 0.3518 | 0.2309 | +0.121 | 30 | 49 |
| 2024-03 | 92 | 0.3445 | 0.2134 | +0.131 | 37 | 58 |
| 2024-04 | 102 | 0.3617 | 0.2761 | +0.086 | 36 | 58 |
| 2024-05 | 99 | 0.3441 | 0.2563 | +0.088 | 37 | 55 |
| 2024-06 | 70 | 0.3998 | 0.2834 | +0.116 | 23 | 34 |
| 2024-07 | 56 | 0.2845 | 0.2201 | +0.064 | 25 | 33 |
| 2024-08 | 64 | 0.3507 | 0.2762 | +0.074 | 21 | 35 |
| 2024-09 | 92 | 0.3697 | 0.2843 | +0.085 | 38 | 47 |
| 2024-10 | 130 | 0.4209 | 0.2782 | +0.143 | 41 | 70 |
| 2024-11 | 103 | 0.3861 | 0.3333 | +0.053 | 35 | 47 |
| 2024-12 | 77 | 0.3700 | 0.3072 | +0.063 | 30 | 42 |
| 2025-01 | 54 | 0.3413 | 0.2486 | +0.093 | 20 | 33 |
| 2025-02 | 87 | 0.3368 | 0.2308 | +0.106 | 33 | 54 |
| 2025-03 | 87 | 0.3486 | 0.2260 | +0.123 | 36 | 56 |
| 2025-04 | 86 | 0.3440 | 0.2217 | +0.122 | 35 | 55 |
| 2025-05 | 78 | 0.2872 | 0.1826 | +0.105 | 38 | 55 |
| 2025-06 | 60 | 0.3384 | 0.2062 | +0.132 | 24 | 43 |
| 2025-07 | 44 | 0.3026 | 0.2578 | +0.045 | 16 | 25 |
| 2025-08 | 59 | 0.3653 | 0.2228 | +0.143 | 21 | 37 |
| 2025-09 | 84 | 0.3145 | 0.2247 | +0.090 | 34 | 51 |
| 2025-10 | 106 | 0.3680 | 0.2563 | +0.112 | 33 | 59 |
| 2025-11 | 80 | 0.3995 | 0.2430 | +0.157 | 25 | 47 |
| 2025-12 | 74 | 0.3662 | 0.2375 | +0.129 | 23 | 43 |

### Aggregated Summary

| Metric | V6.2B | V7.0 | V7.0 edge |
|--------|:-----:|:----:|:---------:|
| **Onpeak** | | | |
| Avg rank of binding constraints | 0.3523 | 0.2509 | **+29% better** |
| Top-20% captures (24mo sum) | 720 | 1,130 | **+57%** |
| Avg T0 concentration | 37.3% | 58.6% | **+21pp** |
| Avg T0+T1 concentration | 61.6% | 75.9% | **+14pp** |
| **Offpeak** | | | |
| Avg rank of binding constraints | 0.3392 | 0.2279 | **+33% better** |
| Top-20% captures (24mo sum) | 653 | 1,112 | **+70%** |
| Avg T0 concentration | 37.3% | 63.6% | **+26pp** |
| Avg T0+T1 concentration | 63.6% | 78.4% | **+15pp** |

---

## Study 2: Signal Distinctiveness (AUC)

AUC measures how well each model separates binding from non-binding constraints.
1.0 = perfect separation, 0.5 = random.

V7.0 wins **24/24 months** on AUC (f0/onpeak).

| Month | V6.2B AUC | V7.0 AUC | V6.2B rank sep | V7.0 rank sep |
|-------|:---------:|:--------:|:--------------:|:-------------:|
| 2024-01 | 0.657 | 0.719 | 0.157 | 0.219 |
| 2024-02 | 0.665 | 0.798 | 0.165 | 0.298 |
| 2024-03 | 0.673 | 0.819 | 0.173 | 0.319 |
| 2024-04 | 0.659 | 0.758 | 0.159 | 0.258 |
| 2024-05 | 0.678 | 0.778 | 0.178 | 0.278 |
| 2024-06 | 0.616 | 0.749 | 0.116 | 0.249 |
| 2024-07 | 0.740 | 0.812 | 0.240 | 0.312 |
| 2024-08 | 0.671 | 0.755 | 0.171 | 0.255 |
| 2024-09 | 0.648 | 0.744 | 0.148 | 0.244 |
| 2024-10 | 0.594 | 0.762 | 0.094 | 0.262 |
| 2024-11 | 0.637 | 0.699 | 0.137 | 0.199 |
| 2024-12 | 0.651 | 0.724 | 0.151 | 0.224 |
| 2025-01 | 0.680 | 0.784 | 0.180 | 0.284 |
| 2025-02 | 0.686 | 0.806 | 0.186 | 0.306 |
| 2025-03 | 0.671 | 0.809 | 0.171 | 0.309 |
| 2025-04 | 0.678 | 0.817 | 0.178 | 0.317 |
| 2025-05 | 0.737 | 0.853 | 0.237 | 0.353 |
| 2025-06 | 0.680 | 0.827 | 0.180 | 0.327 |
| 2025-07 | 0.717 | 0.767 | 0.217 | 0.267 |
| 2025-08 | 0.654 | 0.817 | 0.154 | 0.317 |
| 2025-09 | 0.711 | 0.812 | 0.211 | 0.312 |
| 2025-10 | 0.657 | 0.789 | 0.157 | 0.289 |
| 2025-11 | 0.621 | 0.808 | 0.121 | 0.308 |
| 2025-12 | 0.655 | 0.804 | 0.155 | 0.304 |

**Summary**:
- V6.2B avg AUC: **0.668** (modest discrimination)
- V7.0 avg AUC: **0.784** (strong discrimination)
- V7.0's rank separation is **69% wider** than V6.2B (0.284 vs 0.168)

---

## Study 3: Early Alarm for Truly New Binders

For constraints that **never bound in any prior month** and first bind during holdout:
- Total: 2,010 constraints
- All 2,010 are truly new (0% had any prior binding history in the DA cache)

Tier assignment N months **before** first binding (f0/onpeak):

| Lead time | #Obs | V6.2B T0 | V7.0 T0 | V6.2B T0+T1 | V7.0 T0+T1 | Winner |
|:---------:|:----:|:--------:|:-------:|:-----------:|:----------:|:------:|
| 1 month | 113 | **9.7%** | 0.0% | **19.5%** | 11.5% | **V6.2B** |
| 2 months | 92 | **8.7%** | 0.0% | **25.0%** | 8.7% | **V6.2B** |
| 3 months | 66 | **19.7%** | 0.0% | **25.8%** | 15.2% | **V6.2B** |
| 6 months | 41 | **14.6%** | 2.4% | **29.3%** | 22.0% | **V6.2B** |

**V6.2B dominates for new binders at every lead time.**

### Root Cause

V7.0's top features by importance are binding_freq windows (bf_1, bf_3, bf_6, bf_12, bf_15), comprising ~50-70% of model importance. For a constraint that has **never bound**, all bf features are zero. The model has no discriminative signal and defaults to a low-priority score.

V6.2B's da_rank_value is derived from historical DA shadow prices (not realized binding). It captures **structural congestion potential** — a constraint can have high shadow_price_da (reflecting historical flow patterns) even if it has never actually bound in our realized DA history. This forward-looking information is unavailable to V7.0's BF-dominated model.

### Implication

V7.0 has a **blind spot** for emerging constraints. Any constraint entering the binding universe for the first time is likely to be ranked tier 3-4 by V7.0, regardless of its structural congestion potential.

---

## Study 4: Early Alarm for Recurring Binders

For constraints that **previously bound**, stopped for 3+ months, then re-bound during holdout:
- Total gap-resume events: 2,182 (of which 500 had signal data in both V6.2B and V7.0)

Tier assignment **1 month before** re-binding:

| | V6.2B | V7.0 | Winner |
|---|:---:|:---:|:---:|
| T0 | 26.6% | **42.8%** | **V7.0 (+16pp)** |
| T0+T1 | 53.4% | **68.8%** | **V7.0 (+15pp)** |

**V7.0 is 60% better at flagging recurring binders before they re-bind.** The bf_12 and bf_15 windows retain memory of historical binding even after multi-month gaps, giving V7.0 signal where V6.2B's formula has already decayed.

---

## Study 5: MNTCELO TR6 Case Study (f0/onpeak)

MNTCELO TR6 is a heavily-binding onpeak constraint with 14+ binding months. It illustrates the core trade-off between persistence (V6.2B) and sharpness (V7.0).

| Month | V6.2B rank | V6.2B tier | V7.0 rank | V7.0 tier | Bound? | Notes |
|-------|:----------:|:----------:|:---------:|:---------:|:------:|-------|
| 2023-09 | 0.293 | **T1** | 0.794 | T3 | YES | V6.2B catches it, V7.0 misses |
| 2023-10 | 0.269 | **T1** | 0.629 | T3 | | V6.2B holds signal |
| 2023-11 | 0.304 | T1 | 0.170 | **T0** | | V7.0 catches up as BF kicks in |
| 2023-12 | 0.532 | T2 | 0.128 | **T0** | | V7.0 stronger |
| 2024-01 | 0.593 | T2 | 0.176 | **T0** | YES | V7.0 at T0, V6.2B only T2 |
| 2024-02 | 0.425 | T2 | 0.210 | **T1** | YES | V7.0 still ahead |
| 2024-03 | 0.235 | T1 | 0.233 | T1 | YES | Tied |
| 2024-04 | 0.265 | T1 | 0.277 | T1 | | Tied |
| 2024-05 | 0.248 | **T1** | 0.691 | T3 | | BF decays, V7.0 drops sharply |
| 2024-09 | 0.313 | **T1** | 0.936 | T4 | | V6.2B steady, V7.0 at bottom |
| 2024-11 | 0.336 | T1 | 0.243 | T1 | | Tied after BF partially recovers |
| 2024-12 | 0.507 | T2 | 0.267 | **T1** | | V7.0 edges ahead |
| 2025-01 | 0.454 | T2 | 0.372 | **T1** | | V7.0 marginally ahead |
| 2025-05 | 0.261 | **T1** | 0.625 | T3 | | Same pattern: BF decays again |
| 2025-06 | 0.593 | T2 | 0.912 | T4 | | V7.0 at rock bottom |
| 2025-08 | 0.609 | T3 | 0.807 | T4 | | Both miss |
| 2025-10 | 0.105 | **T0** | 0.387 | T1 | YES | V6.2B at T0 as shadow_price surges |
| 2025-11 | 0.051 | **T0** | 0.267 | T1 | YES | V6.2B stays T0 |
| 2025-12 | 0.007 | T0 | 0.041 | T0 | YES | Both at T0 |

### Key Observations

1. **V6.2B is persistent**: Maintains T0-T1 through quiet months (2024-05 through 2024-09, 2025-05 through 2025-08) based on da_rank_value / shadow_price_da. Never drops below T2 in the months shown.

2. **V7.0 is volatile**: Swings from T0 to T4 as binding frequency decays. When BF is strong (post-binding), V7.0 gives sharper signal (T0 rank 0.13-0.18). When BF decays (3+ months after binding), V7.0 drops to T3-T4 (rank 0.69-0.94).

3. **V6.2B's late-2025 T0 is from shadow_price_da surge**: In 2025-10/11, shadow_price_da for MNTCELO jumped to 420-2886 (from ~56 earlier), driving da_rank_value down to 0.31-0.11. This structural signal is not available to V7.0's BF-dominated model at the same strength.

4. **Neither model gives truly early alarm**: Both miss MNTCELO's initial binding in 2023-09 at the T0 level. V6.2B has T1 (catches it at a moderate level), V7.0 has T3 (misses entirely). The signal only strengthens after binding occurs.

---

## Synthesis: Two Different Alarm Philosophies

| Dimension | V6.2B | V7.0 |
|-----------|:-----:|:----:|
| **Overall discrimination** | AUC 0.668 | **AUC 0.784** |
| **Top-20% binding captures** | 720 | **1,130 (+57%)** |
| **Recurring binder early alarm** | T0 = 27% | **T0 = 43%** |
| **New binder early alarm** | **T0 = 10-20%** | T0 = 0% |
| **Signal persistence** | **Holds T0-T1 through quiet periods** | Decays to T3-T4 |
| **Signal volatility** | Low (rank stable ±0.15) | High (rank swings ±0.50) |
| **Alarm sharpness when active** | Moderate (rank ~0.10-0.25) | **Sharp (rank ~0.04-0.18)** |

### V6.2B's Philosophy: Structural Awareness

V6.2B ranks constraints by **historical congestion potential** (shadow_price_da, density features). This signal is persistent and forward-looking — it reflects the physical topology and flow patterns, not just whether binding has been observed. This makes V6.2B good at:
- Maintaining awareness of constraints during quiet periods
- Flagging constraints with structural congestion potential before they first bind
- Providing a stable, low-volatility ranking

### V7.0's Philosophy: Empirical Evidence

V7.0 ranks constraints by **observed binding behavior** (binding_freq dominates at 50-70% importance). This signal is sharp and evidence-based — if a constraint has been binding recently, V7.0 ranks it very high. If not, V7.0 is skeptical. This makes V7.0 good at:
- Discriminating binding vs non-binding with high accuracy (AUC 0.784)
- Capturing much more binding value in the top tiers (+57% top-20 captures)
- Giving confident, high-conviction tier assignments for active constraints

### The Gap

V7.0's blind spot is **latent constraints** — those with structural congestion potential but no recent binding history. These constraints are rare individually but collectively meaningful:
- 2,010 constraints first bound during the 24-month holdout
- V7.0 assigns 0% of them to T0 before their first binding
- V6.2B assigns 10-20% to T0 (modest but non-zero early warning)

The ideal signal would combine V7.0's empirical discrimination with V6.2B's structural awareness. The `da_rank_value` feature is already inside V7.0 as one of 9 features, but the BF-dominated model underweights it for the new-binder detection task.

---

## Methodology

- **Signal data**: V6.2B from `SPICE_F0P_V6.2B.R1`, V7.0 from `SPICE_F0P_V7.0.R1`
- **Ground truth**: Realized DA shadow prices from cached parquet files (`data/realized_da/*.parquet`), constraint is "binding" if `abs(realized_sp) > 0`
- **Rank convention**: Lower rank = higher priority (more likely to bind). Rank is percentile from 0 to 1.
- **Tier convention**: T0 = top 20%, T1 = 20-40%, T2 = 40-60%, T3 = 60-80%, T4 = bottom 20%
- **AUC**: Probability that a randomly chosen binding constraint has lower rank than a randomly chosen non-binding constraint
- **"New binder"**: Constraint with no binding in any month in the realized DA cache (2017-04 through 2026-02) prior to its first binding in holdout
- **"Recurring binder"**: Constraint that bound, then had 3+ consecutive months without binding, then bound again during holdout
- **Top-20% capture**: Number of binding constraints with rank <= 0.20

---

## Part II: Fixing V7.0's New-Binder Blind Spot

V7.0's 0% T0 rate for new binders is a meaningful gap. We tested three approaches to fix it,
ranging from simple post-processing to model architecture changes. All experiments use
f0/onpeak dev eval (36 months) with the same walk-forward protocol as the V7.0 champion.

---

## Approach 1: Asymmetric Floor (Post-Processing)

**Idea**: If V7.0 assigns tier 3-4 but V6.2B assigns tier 0-1, cap V7.0 at tier 2.
This is a pure post-processing rule — no model retraining needed.

**Rule**: `if v70_tier >= 3 AND v62b_tier <= 1 → set tier = 2`

### Results (f0/onpeak holdout, 24 months)

| Metric | Value |
|--------|------:|
| Constraints floored | 987 |
| Of those, actually bound | 45 (4.6% precision) |
| Binders rescued to T0-T2 | +45 |
| False positives (non-binding floored to T2) | 942 |

**Tier distribution change for binding constraints**:

| Metric | V7.0 baseline | V7.0 + Floor |
|--------|:---:|:---:|
| T0+T1+T2 capture | 87.0% | 89.1% (+2.1pp) |
| T0+T1 capture | 75.9% | 75.9% (unchanged) |
| T0 capture | 58.6% | 58.6% (unchanged) |

### Verdict

The floor rescues 45 binding constraints from T3-T4 to T2, but it does **not** move them
into T0 or T1 — it just prevents the worst misses. The 4.6% precision means 21 false
positives per true rescue. The approach is cheap and safe but does not solve the underlying
problem: V7.0 still cannot **distinguish** which BF-zero constraints are likely to bind.

---

## Approach 2: Feature Additions (v13a/b/c)

**Idea**: Add new features that give the model signal for BF-zero constraints, so it can
learn to rank them without relying on binding_freq.

### Variants

| Variant | Features added | Total features |
|---------|---------------|:-:|
| v13a | `shadow_price_da` (raw historical DA shadow price) | 10 |
| v13b | v13a + `has_bf_signal` (binary: any bf_1..bf_6 > 0) | 11 |
| v13c | v13b + `da_rank_no_bf` (da_rank_value × (1 - bf_6)) | 12 |

The rationale: `shadow_price_da` carries structural congestion information that V6.2B uses
via `da_rank_value`. Adding it as a raw feature gives the model access to the magnitude,
not just the rank. `has_bf_signal` lets the model learn separate scoring paths for BF-zero
vs BF-positive. `da_rank_no_bf` amplifies `da_rank_value` when BF is absent.

### Results (f0/onpeak dev, 36 months)

| Variant | VC@20 | vs Baseline | Key feature importance |
|---------|:-----:|:-----------:|----------------------|
| **Baseline (v10e-lag1)** | **0.4137** | — | bf_6: 26%, bf_12: 18%, bf_15: 14% |
| v13a (+shadow_price_da) | 0.4040 | -2.3% | shadow_price_da: **1.6%** |
| v13b (+has_bf_signal) | 0.3976 | -3.9% | has_bf_signal: **0.4%** |
| v13c (+da_rank_no_bf) | 0.4033 | -2.5% | da_rank_no_bf: **45.1%** |

### Why v13c's 45% importance didn't help

`da_rank_no_bf = da_rank_value × (1 - bf_6)` got 45% feature importance because it
**cannibalized** the binding_freq features (bf_6 dropped from 26% to 8%). The model
effectively replaced BF with a noisier proxy. For BF-positive constraints, the interaction
term is near zero (unhelpful). For BF-zero constraints, it equals da_rank_value (same as
what V6.2B already uses). Net result: no improvement.

### Verdict

**All three variants perform worse than baseline.** The model buries structural features
because binding_freq is overwhelmingly more predictive in training data. Adding features
that help the 21% BF-zero population (6.8% bind rate) cannot overcome the 79% BF-positive
population (29.1% bind rate) that dominates training signal.

This is a fundamental limitation: **you cannot fix a population-level blind spot by adding
features to a single model that is dominated by a different signal.**

---

## Approach 3: Two-Model Ensemble (v14a/b/c)

**Idea**: Train two separate models, each specialized for its population:
- **Model A** (BF-positive): V7.0 champion (9 features including all BF windows)
- **Model B** (BF-zero): Structural-only model (5 features, no BF at all)

Model B features: `da_rank_value`, `shadow_price_da`, `v7_formula_score`, `prob_exceed_110`, `constraint_limit`

### Variants

| Variant | Strategy |
|---------|----------|
| v14a | Hard switch: BF-positive → Model A scores, BF-zero → Model B scores |
| v14b | Soft blend: α × Model A + (1-α) × Model B, where α = min(bf_6, 1.0) |
| v14c | Model A only (sanity check = baseline) |

### VC@20 Results

| Variant | Dev VC@20 | Holdout VC@20 | vs Baseline |
|---------|:---------:|:------------:|:-----------:|
| v14c (baseline) | 0.4147 | 0.3267 | — |
| v14a (hard switch) | 0.3794 | 0.2946 | **-9.8%** |
| v14b (soft blend) | 0.3795 | 0.2946 | **-9.8%** |

**VC@20 drops ~10%.** This is expected: promoting BF-zero constraints into higher tiers
necessarily displaces BF-positive binders that capture more binding value.

### But VC@20 Is Not the Whole Story

The ensemble's purpose is to improve tier assignment for BF-zero binders specifically.
Here is the tier distribution analysis on holdout:

**BF-zero binders** (constraints with no binding history that actually bind):

| Tier | V7.0 Baseline | Ensemble (v14a) | Change |
|------|:---:|:---:|:---:|
| T0 | 10.8% | 21.2% | **+10.4pp** |
| T1 | 26.6% | 27.9% | +1.3pp |
| T0+T1 | 37.4% | 49.1% | **+11.7pp** |
| T4 (worst) | 19.5% | 10.7% | **-8.8pp** |

**BF-positive binders** (constraints with binding history that bind again):

| Tier | V7.0 Baseline | Ensemble (v14a) | Change |
|------|:---:|:---:|:---:|
| T0 | 86.6% | 68.1% | -18.5pp |
| T0+T1 | 93.2% | 85.3% | -7.9pp |

### Model B Feature Importance

| Feature | Importance |
|---------|:---------:|
| v7_formula_score | 60.0% |
| prob_exceed_110 | 14.7% |
| da_rank_value | 9.9% |
| shadow_price_da | 9.2% |
| constraint_limit | 6.2% |

Model B learns to use the V6.2B formula score (which includes da_rank_value and
density_ori) as its primary signal, supplemented by spice6 exceedance probability.
This matches V6.2B's ranking philosophy but with ML-optimized weighting.

### Verdict

The ensemble **works for its intended purpose**: BF-zero binders get dramatically better
tier assignments (T0+T1: 37% → 49%, T4: 20% → 11%). The cost is a ~10% VC@20 drop because
BF-positive binders lose some top-tier slots to the BF-zero population.

Whether this trade-off is acceptable depends on the use case:
- If maximizing total binding value in top-20 is paramount → stay with V7.0 baseline
- If catching emerging constraints early matters → the ensemble is worth the VC@20 cost

---

## Synthesis: Honest Assessment

### What V7.0 Does Well (Keep)

- **Overall discrimination**: AUC 0.784 vs 0.668 (+17%), wins 24/24 months
- **Top-tier concentration**: 57% more binding in top-20 than V6.2B
- **Recurring binder detection**: 60% better T0 rate for constraints resuming after gaps
- **Sharp conviction**: When BF is strong, V7.0's signal is unambiguous

### What V7.0 Gets Wrong (The Blind Spot)

- **0% T0 for truly new binders** (V6.2B gives 10-20%)
- **47% of actual binding comes from BF-zero constraints** on holdout
- **Signal volatility**: T0 → T4 swings as BF decays after binding stops

### What We Tried to Fix It

| Approach | VC@20 Impact | New-binder T0+T1 | Practical? |
|----------|:---:|:---:|:---:|
| 1. Asymmetric floor | 0% (unchanged) | +2.1pp (T012 only) | Yes, but minimal |
| 2. Feature additions | -2 to -4% | Not measured (VC@20 already worse) | No |
| 3. Two-model ensemble | **-10%** | **+11.7pp** | Yes, with trade-off |

### Recommended Path

**For production V7.0**: Ship the current model as-is. The +57% top-20 capture over V6.2B
is too valuable to dilute. The blind spot is real but affects a minority of total binding value.

**For V7.1 (future improvement)**: The two-model ensemble (Approach 3) is the most
promising direction. Key refinements to explore:
1. **Score calibration**: Instead of raw score mixing, calibrate Model A and Model B scores
   to the same probability scale before combining
2. **Adaptive blending**: Use a smoother transition between models based on BF magnitude
   (v14b attempted this but with a naive α = bf_6 blend)
3. **Separate tier allocation**: Give BF-zero and BF-positive populations their own tier
   budgets (e.g., reserve 15% of T0 slots for BF-zero constraints)
4. **Richer structural features**: Add more forward-looking features to Model B beyond the
   5 currently used (e.g., topology-based features, seasonal patterns)

The fundamental insight is that **a single model cannot serve two populations with
fundamentally different information regimes**. BF-positive constraints have strong empirical
signal; BF-zero constraints have only structural signal. The ensemble architecture
acknowledges this reality.

---

## Appendix: Data Split Statistics

### BF-zero vs BF-positive on Holdout (f0/onpeak)

| Population | % of constraints | Bind rate | % of total binding |
|------------|:---:|:---:|:---:|
| BF-zero (no prior binding) | 79% | 6.8% | 47% |
| BF-positive (has prior binding) | 21% | 29.1% | 53% |

Despite being only 21% of constraints, BF-positive constraints produce 53% of binding
value — they bind at 4× the rate and tend to have higher shadow prices. This explains
why VC@20 favors the BF-focused model: concentrating on BF-positive constraints is
the most efficient path to capturing binding value.

But the 47% of binding from BF-zero constraints is not negligible. These are the
constraints that V7.0 systematically underranks, and where V6.2B's structural awareness
provides genuine value.
