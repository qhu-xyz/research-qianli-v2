# V7.0 New-Constraint Detection Problem

**Date**: 2026-03-11
**Status**: Open problem, proposed solutions tested, none yet production-ready
**Severity**: High — affects ~47% of actual binding value on holdout

---

## 1. What Is a "New Constraint"?

The term "new constraint" is imprecise. In the MISO constraint universe, there are three distinct populations that V7.0 handles differently:

### Population A: Never-Bound Constraints (BF-zero, structurally present)

These are constraints that exist in the V6.2B/V7.0 signal universe — they have a `constraint_id`, `da_rank_value`, `shadow_price_da`, flow forecasts, and spice6 density features — but they have **never appeared as binding** in realized Day-Ahead shadow prices. Their binding_freq features (bf_1, bf_3, bf_6, bf_12, bf_15) are all zero.

This does NOT mean they are "new to the system." Many have been in the signal data for years, sitting at moderate `da_rank_value` (0.3-0.6) with non-trivial `shadow_price_da` ($50-$800), indicating structural congestion potential. The physical topology, flow patterns, and thermal limits have always made them plausible binding candidates — they simply haven't reached the threshold in realized DA yet.

**Example: Constraint 357512** — First appears in V7.0 signal well before 2024. Has `da_rank_value=0.414` and `shadow_price_da=$853` (indicating historical DA congestion). V6.2B assigns T0 (rank=0.112) based on this structural signal. V7.0 assigns T3 (rank=0.788) because all BF features are zero. The constraint binds in 2024-01.

### Population B: Lapsed Binders (BF decayed to near-zero)

Constraints that bound months or years ago, then went quiet. Their bf_1/bf_3/bf_6 have decayed to zero, but bf_12 or bf_15 may retain a faint signal. V7.0 ranks them low because the short-window BF dominates.

**Example: BENTON3 TR9 (constraint_id=362693)** — Bound once in 2022-02 (realized_sp=$598). By 2024, all BF windows have decayed to zero. V7.0 consistently assigns T4 (rank=0.87-0.94) throughout 2024-2025. V6.2B holds at T2 (rank=0.50-0.56) based on `da_rank_value=0.659` and `shadow_price_da=$49.8`. Then in 2026-01, BENTON3 explodes: realized_sp=$8,830 onpeak, $11,494 offpeak. V7.0 had it at the bottom of the ranking for 2+ years before this event.

### Population C: Truly New Entrants

Constraints that literally do not exist in the signal data until they appear for the first time. These are new transmission elements, re-configurations, or newly-modeled paths. They have no signal data at all in any prior month.

V7.0 cannot help with Population C — if there's no signal row, there's no ranking. This is a data coverage issue, not a model issue.

**The critical insight**: Population A and B together comprise **79% of all constraints** in the signal universe, with a 6.8% bind rate. Collectively, they produce **47% of actual binding events** on holdout. V7.0 systematically underranks them because all BF features are zero or near-zero.

---

## 2. The Problem in Numbers

Aggregated across 24 holdout months (2024-01 through 2025-12), f0/onpeak:

### Never-bound binders that ARE in the signal (165 events)

| Tier | V7.0 | V6.2B | Interpretation |
|------|:----:|:-----:|----------------|
| T0 (top 20%) | **3.0%** | **13.9%** | V6.2B catches 4.6x more at T0 |
| T1 (20-40%) | 13.3% | 13.3% | Tied |
| T2 (40-60%) | 30.3% | 12.1% | V7.0 clusters them in the middle |
| T3 (60-80%) | 24.2% | 26.7% | Similar |
| T4 (bottom 20%) | **29.1%** | **33.9%** | Both miss many, V7.0 slightly better |

V7.0 puts 53.3% of these binders in T3-T4 (the bottom 40%). V6.2B puts 60.6% there. But V6.2B gets nearly 5x more into T0, which is what matters for the alarm signal.

### Previously-bound binders (1,777 events)

| Tier | V7.0 | Interpretation |
|------|:----:|----------------|
| T0 | **63.3%** | V7.0 dominates for known binders |
| T1 | 17.6% | |
| T0+T1 | **80.9%** | 4 out of 5 caught in top 40% |
| T4 | 3.7% | Very few misses |

This is where V7.0 shines. For constraints with BF signal, it is dramatically better than V6.2B.

### The combined picture

Every month, ~20-30% of binding events come from constraints that have never bound before:

| Month | Total binding | New binders | % new |
|-------|:---:|:---:|:---:|
| 2024-01 | 291 | 68 | 23% |
| 2024-04 | 383 | 115 | **30%** |
| 2024-10 | 421 | 106 | 25% |
| 2025-06 | 261 | 76 | **29%** |
| 2025-10 | 451 | 156 | **35%** |
| **Avg** | **312** | **80** | **25%** |

One in four binding events is a constraint binding for the first time. V7.0 is near-blind to these.

---

## 3. Case Studies

### Case Study 1: BENTON3 TR9 — The Dormant Giant

Constraint 362693 (`BENTON3 TR9`) bound once in 2022-02 (sp=$598), then went completely quiet.

| Month | V7.0 | V6.2B | Bound? |
|-------|------|-------|:------:|
| 2023-01 | T2 rk=0.537 | T2 rk=0.498 | |
| 2023-02 | T3 rk=0.617 | T2 rk=0.559 | |
| 2023-12 | **T4 rk=0.944** | T2 rk=0.536 | |
| 2024-01 | **T4 rk=0.871** | T2 rk=0.563 | |
| 2024-02 | T3 rk=0.763 | T2 rk=0.527 | |
| 2025-01 | **T4 rk=0.924** | T3 rk=0.609 | |
| 2025-02 | T2 rk=0.590 | T1 rk=0.367 | |
| **2026-01** | **???** | **???** | **YES (sp=$8,830!)** |

V7.0 had BENTON3 at T4 (rank 0.87-0.94) for nearly every month from late 2023 through early 2025. V6.2B held it at T2 (rank ~0.5) based on `da_rank_value=0.659` and `shadow_price_da=$49.8`.

Then in January 2026, BENTON3 bound with a realized shadow price of **$8,830** (onpeak) and **$11,494** (offpeak). This is one of the highest binding values in the dataset. V7.0's BF-driven model had this constraint ranked in the bottom 10% of the universe for 2+ years leading up to this event.

This is the worst possible failure mode: a constraint that V7.0 confidently dismisses ends up producing extreme binding value.

### Case Study 2: MNTCELO TR6 — The Signal Volatility Problem

Constraint 1973 (`MNTCELO TR6`) is a recurring binder that illustrates V7.0's signal volatility.

| Month | V7.0 | V6.2B | Bound? |
|-------|------|-------|:------:|
| 2023-09 | **T3 rk=0.794** | T1 rk=0.293 | **YES** |
| 2023-11 | T0 rk=0.170 | T1 rk=0.304 | |
| 2023-12 | T0 rk=0.128 | T2 rk=0.532 | |
| 2024-01 | T0 rk=0.176 | T2 rk=0.593 | **YES** |
| 2024-02 | T1 rk=0.210 | T2 rk=0.425 | **YES** |
| 2024-03 | T0 rk=0.158 | T1 rk=0.245 | |
| 2024-11 | T1 rk=0.243 | T1 rk=0.336 | |
| 2025-01 | T1 rk=0.372 | T2 rk=0.454 | |
| 2025-06 | **T4 rk=0.855** | T2 rk=0.594 | |
| 2025-08 | **T4 rk=0.807** | T3 rk=0.609 | |

**When BF is strong** (post-binding in 2023-11 through 2024-03): V7.0 ranks T0 (0.13-0.18), sharper than V6.2B. This is excellent discrimination.

**When BF decays** (2025-06, 2025-08): V7.0 collapses to T4 (0.81-0.86). V6.2B holds at T2-T3 because `da_rank_value` provides structural persistence.

The swing from T0 (rank 0.13) to T4 (rank 0.86) is a rank delta of 0.73 — spanning nearly the entire ranking. This volatility makes V7.0's signal unreliable for downstream consumers who expect signal persistence.

### Case Study 3: Constraint 357512 — Structural Signal Ignored

Constraint 357512 has `da_rank_value=0.414` and `shadow_price_da=$853` — strong structural congestion indicators. It has never bound before January 2024.

| Month | V7.0 | V6.2B | Bound? |
|-------|------|-------|:------:|
| 2024-01 | **T3 rk=0.788** | **T0 rk=0.112** | **YES** |

V6.2B ranks this constraint #55 out of 489 (T0) based on its high historical shadow price. V7.0 ranks it #386 out of 489 (T3) because it has zero binding history. The $853 shadow_price_da is invisible to V7.0's BF-dominated model.

---

## 4. Root Cause Analysis

### Why V7.0 Is Blind to New Binders

V7.0 uses LightGBM LambdaRank with 9 features. Feature importance across the holdout:

| Feature | Avg importance |
|---------|:---:|
| binding_freq_6 | 26% |
| binding_freq_12 | 18% |
| binding_freq_15 | 14% |
| binding_freq_1 | 12% |
| binding_freq_3 | 8% |
| **BF total** | **78%** |
| da_rank_value | 10% |
| v7_formula_score | 6% |
| prob_exceed_110 | 4% |
| constraint_limit | 2% |

78% of model importance comes from binding_freq windows. For a constraint with zero BF, 78% of the model's discriminative power is useless. The remaining 22% (da_rank_value, formula score, spice6) does provide some signal, but it's drowned out by the BF-zero-means-low-risk learned pattern.

### The Walk-Forward Training Creates the Bias

V7.0 uses walk-forward training: for eval month M, it trains on months M-9 through M-2 (with lag). In each training month, the vast majority of binding events come from BF-positive constraints (they bind at 29.1% vs 6.8% for BF-zero). The model learns that BF > 0 is the strongest predictor of binding, which is statistically correct but creates a systematic blind spot.

### The Fundamental Tension

The model optimizes a single objective (LambdaRank NDCG) over a mixed population. The optimal strategy for maximizing NDCG is to use BF as the primary signal — it's the most predictive feature by far. But this strategy sacrifices discrimination within the BF-zero population, which has a lower base rate but produces ~47% of binding events by count.

**A single model cannot serve two populations with fundamentally different information regimes.**

---

## 5. Why This Must Be Fixed

### Trading Impact

For FTR (Financial Transmission Rights) trading, the tier assignment directly influences:

1. **Portfolio construction**: Constraints at T0-T1 are candidates for inclusion in the trading portfolio. T3-T4 constraints are excluded.

2. **Risk management**: The tier determines position sizing. A constraint incorrectly placed at T4 receives zero allocation — even if it ends up producing $8,830 in binding value (as BENTON3 TR9 did).

3. **Signal consumption**: Downstream systems (auction bidding, position management) consume the tier as a categorical signal. If a trader sees T4, they do not investigate further. The ranking is the gateway to attention.

### The 47% Problem

If ~47% of actual binding comes from BF-zero constraints, and V7.0 puts 53% of BF-zero binders in T3-T4, then roughly **25% of all binding value is systematically hidden** from the trading system. This is not a marginal issue — it's a structural gap in the information pipeline.

### The BENTON3 Scenario

A constraint going from T4 to binding with sp=$8,830 is a worst-case event:
- The trading system had no exposure to BENTON3
- The binding creates a large unhedged exposure
- The signal that could have flagged this (da_rank_value=0.659, shadow_price_da=$49.8) was available but ignored by the model

---

## 6. Code: Reproducing the Problem

All code below runs from `research-stage5-tier/` with the pmodel venv active.

### Loading and comparing V7.0 vs V6.2B tiers

```python
import polars as pl
import os

V70_BASE = "/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V7.0.R1"
V62_BASE = "/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1"
DA_CACHE = "data/realized_da"

def load_signal(base: str, month: str, ptype: str = "f0", ctype: str = "onpeak") -> pl.DataFrame:
    """Load signal parquet for a given month/ptype/ctype."""
    d = os.path.join(base, month, ptype, ctype)
    files = os.listdir(d)
    return pl.read_parquet(os.path.join(d, files[0]))

def load_binding_set(month: str) -> set[str]:
    """Load set of binding constraint_ids for a month from realized DA cache."""
    f = os.path.join(DA_CACHE, f"{month}.parquet")
    df = pl.read_parquet(f)
    return set(df.filter(pl.col("realized_sp") > 0)["constraint_id"].to_list())

# Example: compare models for 2025-01
month = "2025-01"
df70 = load_signal(V70_BASE, month)
df62 = load_signal(V62_BASE, month)
bound = load_binding_set(month)

# Tag binding status
df70 = df70.with_columns(pl.col("constraint_id").is_in(list(bound)).alias("bound"))

# Show binding constraints in V7.0 T3-T4
missed = df70.filter(pl.col("bound") & (pl.col("tier") >= 3))
print(f"V7.0 T3-T4 binding constraints in {month}: {len(missed)}")
print(missed.select(["constraint_id", "branch_name", "tier", "rank", "da_rank_value"]))
```

### Building binding_freq and identifying BF-zero population

```python
def build_binding_history(cache_dir: str = DA_CACHE) -> dict[str, set[str]]:
    """Load all months of realized DA binding sets."""
    binding_sets = {}
    for f in sorted(os.listdir(cache_dir)):
        if f.endswith(".parquet") and "_partial_" not in f:
            month = f.replace(".parquet", "")
            df = pl.read_parquet(os.path.join(cache_dir, f))
            binding_sets[month] = set(
                df.filter(pl.col("realized_sp") > 0)["constraint_id"].to_list()
            )
    return binding_sets

def has_any_bf(constraint_id: str, eval_month: str, binding_sets: dict, windows=(1,3,6)) -> bool:
    """Check if a constraint has ANY binding in bf_1/3/6 windows before eval_month."""
    import pandas as pd
    ts = pd.Timestamp(eval_month)
    max_window = max(windows)
    for i in range(1, max_window + 1):
        m = (ts - pd.DateOffset(months=i)).strftime("%Y-%m")
        if m in binding_sets and constraint_id in binding_sets[m]:
            return True
    return False

# Partition signal into BF-zero and BF-positive
binding_sets = build_binding_history()
signal_ids = df70["constraint_id"].to_list()

bf_zero = [cid for cid in signal_ids if not has_any_bf(cid, month, binding_sets)]
bf_pos = [cid for cid in signal_ids if has_any_bf(cid, month, binding_sets)]
print(f"BF-zero: {len(bf_zero)} ({len(bf_zero)/len(signal_ids)*100:.0f}%)")
print(f"BF-pos:  {len(bf_pos)} ({len(bf_pos)/len(signal_ids)*100:.0f}%)")
```

### Tracking a constraint across months (early alarm analysis)

```python
def track_constraint(constraint_id: str, months: list[str]):
    """Track a constraint's V7.0 and V6.2B tier across months."""
    for em in months:
        try:
            df70 = load_signal(V70_BASE, em)
            df62 = load_signal(V62_BASE, em)
        except (FileNotFoundError, IndexError):
            continue

        r70 = df70.filter(pl.col("constraint_id") == constraint_id)
        r62 = df62.filter(pl.col("constraint_id") == constraint_id)

        v70 = f"T{r70['tier'][0]} rk={r70['rank'][0]:.3f}" if len(r70) > 0 else "N/A"
        v62 = f"T{r62['tier'][0]} rk={r62['rank'][0]:.3f}" if len(r62) > 0 else "N/A"

        bound = constraint_id in binding_sets.get(em, set())
        flag = " <-- BINDS" if bound else ""
        print(f"  {em} | V7.0: {v70:15s} | V6.2B: {v62:15s}{flag}")

# Track BENTON3 TR9
months = [f"{y}-{m:02d}" for y in range(2023, 2027) for m in range(1, 13)]
track_constraint("362693", months)
```

### Running the two-model ensemble experiment (Approach 3)

```bash
# From research-stage5-tier/
source /home/xyz/workspace/pmodel/.venv/bin/activate

# Dev evaluation (36 months, 2020-06 to 2023-05)
python scripts/run_v14_ensemble.py --variant all --ptype f0 --ctype onpeak

# Holdout evaluation (24 months, 2024-2025)
python scripts/run_v14_ensemble.py --variant all --ptype f0 --ctype onpeak --holdout

# Save results to registry
python scripts/run_v14_ensemble.py --variant v14a --holdout --save
```

---

## 7. Walk-Forward vs Holdout: Could Validation Strategy Help?

### Current Setup

V7.0 already uses **walk-forward training** — for each eval month M, a fresh model is trained on M-9 through M-2. This is correct for production deployment: every month gets a model trained on the most recent available data.

The **dev set** (2020-06 through 2023-05, 36 months) was used for all model development and feature selection. The **holdout set** (2024-01 through 2025-12, 24 months) was reserved for final evaluation and was never tuned on.

### Why Holdout Matters for This Problem

The new-binder problem is particularly well-suited to holdout evaluation because:

1. **New binders are non-stationary**: The set of constraints that have "never bound before" changes every month as the historical binding set grows. By 2025, fewer constraints are truly "new" than in 2021, because we have 4 more years of history. Dev-period statistics on new binders may not reflect holdout-period behavior.

2. **Dev-period overfitting risk**: If we tune model features or architecture on dev-period new-binder performance, we risk fitting to the specific constraints that were new during 2020-2023. These are different physical constraints than those that are new during 2024-2025.

3. **Holdout gives honest signal**: On holdout, we found that 25% of binding events come from new binders (consistent with dev). The V7.0 tier distribution for these binders (3% T0, 53% T3-T4) is the ground truth of the problem severity.

### Could Walk-Forward Validation Help With the Blind Spot?

The short answer is **no, not by itself**. Here's why:

Walk-forward is a training strategy — it determines what data the model sees during training. The new-binder problem is not caused by the training window or data split. It's caused by **feature dominance**: binding_freq is so predictive that the model allocates 78% of its capacity to BF features, leaving only 22% for structural features that could help BF-zero constraints.

Changing from walk-forward to expanding-window (training on all history up to M-2) would give the model more training examples, but the BF-dominant pattern would remain. In fact, expanding windows might make it worse: with more history, the BF features become even more reliable for the BF-positive population.

### What Would Actually Help

The approaches that address the structural cause:

1. **Two-model ensemble** (tested as v14a/v14b): Separate models for BF-zero and BF-positive populations. BF-zero binder T0+T1 improves from 37% to 49% on holdout. Cost: ~10% VC@20 drop.

2. **Stratified tier allocation** (not yet tested): Reserve a fixed percentage of T0-T1 slots for BF-zero constraints. Rank BF-zero constraints among themselves using Model B, and BF-positive constraints among themselves using Model A. This preserves V7.0's strong discrimination for known binders while guaranteeing representation for new binders.

3. **Walk-forward with population-aware loss** (not yet tested): Modify the LambdaRank loss to upweight ranking errors for BF-zero constraints. This is the "single model" approach that might work, by telling the optimizer to care more about getting BF-zero rankings right, even at the cost of slightly worse BF-positive rankings.

4. **Post-hoc V6.2B floor** (tested as Approach 1): Cheap and safe — if V7.0 assigns T3-T4 but V6.2B assigns T0-T1, cap at T2. Rescues 45 binders at 4.6% precision. Minimal impact but better than nothing.

### Recommended Validation Framework

For any future fix, the evaluation should include:

```python
# 1. Overall VC@20 (must not degrade catastrophically)
vc20 = evaluate_ltr(actual, scores)["VC@20"]

# 2. BF-zero binder tier distribution (the PRIMARY metric for this problem)
bf_zero_mask = ~has_bf_signal
bf_zero_bound = (bf_zero_mask) & (actual > 0)
bf_zero_binder_tiers = compute_tier_distribution(scores, bf_zero_bound)
# Target: T0+T1 >= 40% (currently 16% for V7.0, 27% for V6.2B)

# 3. BF-positive binder tier distribution (must remain strong)
bf_pos_bound = has_bf_signal & (actual > 0)
bf_pos_binder_tiers = compute_tier_distribution(scores, bf_pos_bound)
# Target: T0 >= 60% (currently 63% for V7.0)

# 4. Walk-forward consistency (month-to-month stability)
# For each constraint, compute tier variance across consecutive months
tier_volatility = compute_signal_volatility(all_month_tiers)
# Target: avg tier change < 1.0 per month for BF-zero constraints
```

Run these metrics on **both dev and holdout** — the new-binder problem was first detected on holdout, and dev-only evaluation missed it.

---

## 8. Summary

| Aspect | Status |
|--------|--------|
| **Problem severity** | High — 47% of binding from BF-zero constraints |
| **Root cause** | BF features dominate (78% importance), leaving no capacity for structural signal |
| **Best fix tested** | Two-model ensemble (v14a): +11.7pp T0+T1 for BF-zero binders, -10% VC@20 |
| **Production recommendation** | Ship V7.0 as-is, add V6.2B floor as safety net, develop ensemble for V7.1 |
| **Walk-forward impact** | Does not address root cause — this is a feature dominance problem, not a training window problem |
| **Key constraints to watch** | BENTON3 TR9 (dormant giant), MNTCELO TR6 (signal volatility), all high-da_rank/zero-BF constraints |
