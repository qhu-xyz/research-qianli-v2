# V7.0 Signal Validation Plan

## Overview

Validates V7.0 signal correctness against two standards:
1. **Consistency with research-stage5-tier**: ML slices must reproduce exact holdout VC@20
2. **Substantive improvement**: ML slices must beat V6.2B formula by concrete margins
3. **Passthrough fidelity**: f2/f3/q2-q4 must be bit-identical to V6.2B

Run `scripts/validate_v70.py` after generating signals. All checks are automated.

---

## Gate A: Improvement Over V6.2B (MANDATORY)

V7.0 must demonstrate clear superiority over V6.2B formula for ML slices.
These are hard gates — if any fails, the signal is not deployed.

### Holdout VC@20 improvement gates

| Slice | V6.2B (v0) | V7.0 (ML) | Min improvement | Min absolute |
|-------|-----------|-----------|-----------------|--------------|
| f0/onpeak | 0.1835 | 0.3529 | **+40%** | **>0.28** |
| f0/offpeak | 0.2075 | 0.3780 | **+40%** | **>0.30** |
| f1/onpeak | 0.2209 | 0.3677 | **+30%** | **>0.30** |
| f1/offpeak | 0.2492 | 0.3561 | **+20%** | **>0.30** |

"Min improvement" is relative to V6.2B: `(ml - v0) / v0`. Set conservatively
below observed gains (92%/82%/66%/43%) to allow for minor numerical differences.

### Per-month regression guard

No individual holdout month should be worse than the V6.2B formula's worst month.
V6.2B worst months: f0/on 0.0202, f0/off 0.0401, f1/on 0.0373, f1/off 0.0253.
ML worst months: f0/on 0.0682, f0/off 0.1253, f1/on 0.1359, f1/off 0.1779.

Gate: ML worst-month VC@20 must be > 0.05 for all slices.

### Multi-metric improvement

Not just VC@20 — also check VC@100, Recall@20, NDCG:

| Metric | Required direction | Rationale |
|--------|--------------------|-----------|
| VC@20 | ML > V6.2B | Primary: value captured by top picks |
| VC@100 | ML > V6.2B | Breadth: value in top 100 |
| Recall@20 | ML > V6.2B | Precision: binding constraints in top 20 |
| NDCG | ML > V6.2B | Overall ranking quality |

All four metrics must improve for every ML slice.

---

## Gate B: Exact Holdout Reproduction (MANDATORY)

V7.0 must reproduce the exact VC@20 from research-stage5-tier holdout runs.
Same code, same seed (42), same data → same results.

### Full per-month VC@20 targets

#### f0/onpeak (v10e-lag1, 24 months)

| Month | VC@20 | Month | VC@20 |
|-------|-------|-------|-------|
| 2024-01 | 0.1350 | 2025-01 | 0.7401 |
| 2024-02 | 0.3689 | 2025-02 | 0.6269 |
| 2024-03 | 0.0682 | 2025-03 | 0.3729 |
| 2024-04 | 0.5810 | 2025-04 | 0.3042 |
| 2024-05 | 0.3544 | 2025-05 | 0.3915 |
| 2024-06 | 0.2979 | 2025-06 | 0.3969 |
| 2024-07 | 0.2675 | 2025-07 | 0.1502 |
| 2024-08 | 0.3367 | 2025-08 | 0.3030 |
| 2024-09 | 0.5004 | 2025-09 | 0.2434 |
| 2024-10 | 0.3260 | 2025-10 | 0.3920 |
| 2024-11 | 0.3932 | 2025-11 | 0.2571 |
| 2024-12 | 0.3089 | 2025-12 | 0.3521 |
| **Mean** | **0.3529** | | |

#### f0/offpeak (v10e-lag1, 24 months)

| Month | VC@20 | Month | VC@20 |
|-------|-------|-------|-------|
| 2024-01 | 0.2142 | 2025-01 | 0.9051 |
| 2024-02 | 0.4416 | 2025-02 | 0.7248 |
| 2024-03 | 0.3823 | 2025-03 | 0.4201 |
| 2024-04 | 0.3338 | 2025-04 | 0.3126 |
| 2024-05 | 0.3741 | 2025-05 | 0.2255 |
| 2024-06 | 0.3606 | 2025-06 | 0.3372 |
| 2024-07 | 0.1253 | 2025-07 | 0.2045 |
| 2024-08 | 0.3519 | 2025-08 | 0.2285 |
| 2024-09 | 0.6278 | 2025-09 | 0.1839 |
| 2024-10 | 0.3907 | 2025-10 | 0.4891 |
| 2024-11 | 0.3304 | 2025-11 | 0.2535 |
| 2024-12 | 0.3401 | 2025-12 | 0.5154 |
| **Mean** | **0.3780** | | |

#### f1/onpeak (v2, 19 months)

| Month | VC@20 | Month | VC@20 |
|-------|-------|-------|-------|
| 2024-01 | 0.3886 | 2025-01 | 0.4742 |
| 2024-02 | 0.5047 | 2025-02 | 0.4094 |
| 2024-03 | 0.1359 | 2025-03 | 0.2579 |
| 2024-04 | 0.4227 | 2025-04 | 0.3888 |
| 2024-07 | 0.2753 | 2025-07 | 0.2918 |
| 2024-08 | 0.3063 | 2025-08 | 0.2124 |
| 2024-09 | 0.3646 | 2025-09 | 0.1936 |
| 2024-10 | 0.5981 | 2025-10 | 0.3467 |
| 2024-11 | 0.3585 | 2025-11 | 0.3595 |
| 2024-12 | 0.6969 | | |
| **Mean** | **0.3677** | | |

#### f1/offpeak (v2, 19 months)

| Month | VC@20 | Month | VC@20 |
|-------|-------|-------|-------|
| 2024-01 | 0.3555 | 2025-01 | 0.5187 |
| 2024-02 | 0.1779 | 2025-02 | 0.5852 |
| 2024-03 | 0.2963 | 2025-03 | 0.3373 |
| 2024-04 | 0.3803 | 2025-04 | 0.2394 |
| 2024-07 | 0.2587 | 2025-07 | 0.2592 |
| 2024-08 | 0.1786 | 2025-08 | 0.1892 |
| 2024-09 | 0.4765 | 2025-09 | 0.1896 |
| 2024-10 | 0.4745 | 2025-10 | 0.2975 |
| 2024-11 | 0.3643 | 2025-11 | 0.3570 |
| 2024-12 | 0.8309 | | |
| **Mean** | **0.3561** | | |

### Tolerance

**±0.0001 per month.** Same seed, same code, same training data = deterministic.
Any larger deviation means a bug in feature computation, training config, or
rank conversion. If a month is off, diff the raw ML scores before rank/tier
conversion against scores from `run_v10e_lagged.py` for that month.

---

## Gate C: Passthrough Bit-Identity (MANDATORY)

Every non-ML ptype must be byte-for-byte identical to V6.2B.

### What to test

| Ptype | Test months | Has class_types |
|-------|-------------|-----------------|
| f2 | 2026-02, 2026-03 | onpeak, offpeak |
| f3 | 2026-02 | onpeak, offpeak |
| q4 | 2026-01 | onpeak, offpeak |
| q3 | 2025-10 | onpeak, offpeak |
| q2 | 2025-07 | onpeak, offpeak |

### Method

```python
for ptype in passthrough_ptypes:
    for ctype in ["onpeak", "offpeak"]:
        v62b = ConstraintsSignal("miso", V62B_SIGNAL, ptype, ctype).load_data(ts)
        v70  = ConstraintsSignal("miso", V70_SIGNAL,  ptype, ctype).load_data(ts)
        pd.testing.assert_frame_equal(v62b, v70, check_exact=True)
        # Also check SF
        sf_v62b = ShiftFactorSignal("miso", V62B_SIGNAL, ptype, ctype).load_data(ts)
        sf_v70  = ShiftFactorSignal("miso", V70_SIGNAL,  ptype, ctype).load_data(ts)
        pd.testing.assert_frame_equal(sf_v62b, sf_v70, check_exact=True)
```

**Pass criteria**: `assert_frame_equal(check_exact=True)` for both constraints AND
shift factors. No floating-point tolerance — exact match.

---

## Gate D: Schema & Structure (MANDATORY)

### D1: Column and dtype parity

Every V7.0 slice (ML and passthrough) must have identical columns, dtypes, and
index format to V6.2B.

```python
assert v70.columns.tolist() == v62b.columns.tolist()
assert v70.index.dtype == v62b.index.dtype
assert all(v70[c].dtype == v62b[c].dtype for c in v62b.columns)
for idx in v70.index:
    parts = idx.split("|")
    assert len(parts) == 3 and parts[2] == "spice"
```

### D2: Row count match

`v70.shape[0] == v62b.shape[0]` for every slice. ML scoring must not drop or add
constraints.

### D3: ML column isolation

For ML slices (f0, f1), only `rank_ori`, `rank`, `tier` should differ from V6.2B.
All 17 other columns must be identical.

```python
unchanged = [c for c in v62b.columns if c not in ("rank_ori", "rank", "tier")]
for col in unchanged:
    pd.testing.assert_series_equal(v62b[col], v70[col], check_names=False)
```

### D4: Index preservation

V7.0 index must be in the same order as V6.2B (not re-sorted by rank).

```python
assert v70.index.tolist() == v62b.index.tolist()
```

---

## Gate E: Rank & Tier Invariants (MANDATORY)

### E1: Rank range

```python
assert v70["rank"].min() > 0        # row-percentile starts at 1/n
assert v70["rank"].max() == 1.0     # max rank = n/n = 1
```

### E2: Score diagnostics (DIAGNOSTIC — log only)

With tiered labels (4 relevance levels, ~88% label=0), LightGBM assigns
identical leaf paths to many non-binding constraints, producing ~55% unique
raw scores. This is **correct behavior** — the model appropriately does
not differentiate within non-binding constraints. Ties are broken
deterministically using V6.2B rank_ori as secondary key and original index
as tertiary, so all ranks are unique despite score ties.

Score degeneracy (NaN, Inf, near-constant std) is already checked by F1.

```python
scores = v70["rank_ori"].values
n_unique_scores = len(np.unique(scores))
n_unique_rank = v70["rank"].nunique()
print(f"  unique scores: {n_unique_scores}/{len(v70)} ({100*n_unique_scores/len(v70):.0f}%)")
print(f"  unique ranks:  {n_unique_rank}/{len(v70)} (should be 100%)")
print(f"  score std:     {np.std(scores):.4f}")
assert n_unique_rank == len(v70), "ranks should be unique (row-percentile)"
```

### E3: Tier distribution (MANDATORY)

Row-percentile ranking guarantees ~20% per tier (±1 constraint due to
rounding). Each tier should contain 18-22% of constraints.

```python
tier_counts = v70["tier"].value_counts()
assert set(tier_counts.index) == {0, 1, 2, 3, 4}, "missing tiers"
for t in range(5):
    n_tier = tier_counts[t]
    frac = n_tier / len(v70)
    print(f"  tier {t}: {frac:.1%} ({n_tier} constraints)")
    assert 0.18 <= frac <= 0.22, f"tier {t} has {frac:.1%}, expected ~20%"
```

### E4: Tier formula

```python
rank = v70["rank"].values
expected_tier = np.clip(np.ceil(rank * 5).astype(int) - 1, 0, 4)
so_mask = v70["branch_name"] == "SO_MW_Transfer"
np.testing.assert_array_equal(v70["tier"].values[~so_mask], expected_tier[~so_mask])
```

### E5: Tier-rank monotonicity

Tier 0 constraints must have the lowest rank values.

```python
# Exclude SO_MW_Transfer from monotonicity check (tier forced to 1 regardless of rank)
v70_no_so = v70[~so_mask]
for t in range(4):
    tier_t = v70_no_so.loc[v70_no_so["tier"] == t, "rank"]
    tier_next = v70_no_so.loc[v70_no_so["tier"] == t + 1, "rank"]
    if len(tier_t) > 0 and len(tier_next) > 0:
        assert tier_t.max() <= tier_next.min(), (
            f"tier {t} max rank {tier_t.max():.6f} > tier {t+1} min rank {tier_next.min():.6f}"
        )
```

### E6: SO_MW_Transfer exception

```python
if so_mask.any():
    assert v70.loc[so_mask, "tier"].values[0] == 1
```

---

## Gate F: Score Quality

F1 and F2 are mandatory (correctness invariants). F3 and F4 are recommended
diagnostics — they detect likely problems but can legitimately be violated,
so they should warn rather than block deployment.

### F1: No degenerate scores (MANDATORY)

```python
scores = v70["rank_ori"].values  # raw ML scores in V7.0
assert not np.any(np.isnan(scores)), "NaN scores"
assert not np.any(np.isinf(scores)), "Inf scores"
assert np.std(scores) > 0.01, "near-constant scores"
```

### F2: Score polarity (MANDATORY)

Higher ML score = more binding = lower rank value.

```python
# Top 20 by rank (lowest rank) should have highest mean ML score
top20_mask = v70["rank"] <= v70["rank"].nsmallest(20).max()
bot20_mask = v70["rank"] >= v70["rank"].nlargest(20).min()
assert v70.loc[top20_mask, "rank_ori"].mean() > v70.loc[bot20_mask, "rank_ori"].mean()
```

### F3: ML vs V6.2B differentiation (RECOMMENDED — warn only)

V7.0 should meaningfully rerank constraints, not just reproduce V6.2B.
This is a diagnostic, not an invariant — a valid model could produce stronger
or weaker reranking depending on the month.

```python
from scipy.stats import spearmanr
rho, _ = spearmanr(v62b["rank"].values, v70["rank"].values)
if not (0.2 < rho < 0.95):
    print(f"WARNING: V7.0 vs V6.2B rank correlation {rho:.3f} outside expected range (0.2, 0.95)")
```

### F4: Feature importance sanity (RECOMMENDED — warn only)

After training, binding_freq features typically dominate importance (~50-70%).
Significant deviation may indicate a feature computation bug but is not
inherently invalid.

```python
if hasattr(model, "feature_importance"):
    imp = dict(zip(features, model.feature_importance(importance_type="gain")))
    bf_total = sum(v for k, v in imp.items() if k.startswith("binding_freq"))
    all_total = sum(imp.values())
    bf_frac = bf_total / all_total
    if bf_frac < 0.3:
        print(f"WARNING: binding_freq importance {bf_frac:.1%} below expected range (>30%)")
```

---

## Gate G: Shift Factor Parity (MANDATORY)

For ALL ptypes (including f0, f1), SF must be bit-identical to V6.2B.

```python
for ptype in available_ptypes(month):
    for ctype in ["onpeak", "offpeak"]:
        sf_v62b = ShiftFactorSignal("miso", V62B_SIGNAL, ptype, ctype).load_data(ts)
        sf_v70  = ShiftFactorSignal("miso", V70_SIGNAL,  ptype, ctype).load_data(ts)
        pd.testing.assert_frame_equal(sf_v62b, sf_v70, check_exact=True)
```

---

## Gate H: Temporal Integrity (MANDATORY)

### H1: Training window audit

For each ML month, log the training months used and verify:
- No training month overlaps the eval month
- No training month's delivery month has realized DA newer than M-2
- At least 6 usable months (otherwise should skip)

### H2: Binding freq cutoff audit

For the target month M, binding_freq must use months strictly < M-1.
Log the cutoff month and verify it equals `prev_month(M)`.

### H3: No future data in features

After enrichment, verify that no feature column contains data from the target
month or later. Spot-check: for month 2025-06, binding_freq_1 should reflect
binding in months before 2025-05 (not 2025-05 or 2025-06).

---

## Gate I: Determinism (RECOMMENDED)

Generate V7.0 for the same month twice. Output must be bit-identical.

```python
generate_v70_signal("2025-01")
v70_a = ConstraintsSignal("miso", V70_SIGNAL, "f0", "onpeak").load_data(ts)

generate_v70_signal("2025-01")
v70_b = ConstraintsSignal("miso", V70_SIGNAL, "f0", "onpeak").load_data(ts)

pd.testing.assert_frame_equal(v70_a, v70_b, check_exact=True)
```

LightGBM with seed=42 and num_threads=4 should be deterministic.

---

## Gate J: Forward Month / Inference-Only (MANDATORY)

Generate V7.0 for the latest month where no ground truth exists. Must test
**all 4 ML slices** — f0 and f1 have different delivery-month logic, and
onpeak/offpeak use separate realized DA cache files.

**Test month**: `2026-03` (has f0, f1, f2 — see auction schedule for Mar).
- f0: delivery=2026-03, no realized DA → exercises inference-only path
- f1: delivery=2026-04, no realized DA → exercises f1 delivery offset + inference path
- f2: passthrough → verify still bit-identical to V6.2B

```python
generate_v70_signal("2026-03")

# All 4 ML slices must succeed
for ptype in ["f0", "f1"]:
    for ctype in ["onpeak", "offpeak"]:
        v70 = ConstraintsSignal("miso", V70_SIGNAL, ptype, ctype).load_data(
            pd.Timestamp("2026-03")
        )
        assert v70.shape[0] > 400, f"{ptype}/{ctype}: too few constraints"
        assert v70["tier"].between(0, 4).all(), f"{ptype}/{ctype}: invalid tiers"
        assert not v70["rank_ori"].isna().any(), f"{ptype}/{ctype}: NaN scores"

        # Rank/tier sanity (same checks as Gate E)
        rank = v70["rank"].values
        assert rank.min() > 0 and rank.max() == 1.0
        tier_counts = v70["tier"].value_counts()
        assert set(tier_counts.index) == {0, 1, 2, 3, 4}

# Passthrough slice must be bit-identical
for ctype in ["onpeak", "offpeak"]:
    v62b = ConstraintsSignal("miso", V62B_SIGNAL, "f2", ctype).load_data(
        pd.Timestamp("2026-03")
    )
    v70 = ConstraintsSignal("miso", V70_SIGNAL, "f2", ctype).load_data(
        pd.Timestamp("2026-03")
    )
    pd.testing.assert_frame_equal(v62b, v70, check_exact=True)
```

No VC@20 check (no GT). But verify:
- No errors during generation for any of the 4 ML slices
- Schema valid for all slices
- Rank/tier distributions look normal
- f2 passthrough is bit-identical to V6.2B

---

## Gate K: Multi-Month Stability (RECOMMENDED)

### K1: Universe size stability

Constraint count should be stable across adjacent months (±15%).

```python
months = ["2025-10", "2025-11", "2025-12", "2026-01"]
sizes = {m: len(load_v70(m, "f0", "onpeak")) for m in months}
for i in range(len(months) - 1):
    ratio = sizes[months[i+1]] / sizes[months[i]]
    assert 0.85 < ratio < 1.15
```

### K2: Tier-0 turnover

Tier-0 set (top ~100 constraints) should have 30-70% overlap between adjacent
months. Too low = unstable signal. Too high = model not learning month-specific
patterns.

```python
tier0_sets = {m: set(v70[v70["tier"] == 0].index) for m, v70 in ...}
for i in range(len(months) - 1):
    a, b = tier0_sets[months[i]], tier0_sets[months[i + 1]]
    overlap = len(a & b) / max(len(a), len(b))
    assert 0.15 < overlap < 0.85
```

### K3: Score distribution stability

Mean and std of ML scores should not vary by more than 3x across months.

---

## Gate L: Cross-Slice Consistency (RECOMMENDED)

### L1: Onpeak vs offpeak correlation

For the same month and ptype, onpeak and offpeak rankings should be positively
correlated (same physical constraints, similar binding patterns).

```python
on = load_v70(month, "f0", "onpeak").set_index("constraint_id")["rank"]
off = load_v70(month, "f0", "offpeak").set_index("constraint_id")["rank"]
common = on.index.intersection(off.index)
rho, _ = spearmanr(on[common], off[common])
assert rho > 0.3, f"onpeak/offpeak rank correlation: {rho:.3f}"
```

### L2: f0 vs f1 correlation

For the same month, f0 and f1 rankings should be moderately correlated
(overlapping constraint universe, but different delivery months).

```python
f0_df = load_v70(month, "f0", "onpeak").set_index("constraint_id")["rank"]
f1_df = load_v70(month, "f1", "onpeak").set_index("constraint_id")["rank"]
common = f0_df.index.intersection(f1_df.index)
rho, _ = spearmanr(f0_df[common], f1_df[common])
assert rho > 0.2, f"f0/f1 rank correlation: {rho:.3f}"
```

---

## Gate M: Diff Analysis (RECOMMENDED)

### M1: Biggest movers

For each ML slice, identify the 20 constraints that moved most (by tier change)
between V6.2B and V7.0. Log them with their feature values for manual inspection.

```python
v62b_tier = v62b.set_index("constraint_id")["tier"]
v70_tier = v70.set_index("constraint_id")["tier"]
diff = (v62b_tier - v70_tier).abs().sort_values(ascending=False)
print("Top 20 biggest tier movers:")
print(diff.head(20))
```

Sanity check: constraints that moved from tier 4→0 should have high binding_freq.
Constraints that moved from tier 0→4 should have low binding_freq.

### M2: Tier migration matrix

```python
migration = pd.crosstab(v62b["tier"], v70["tier"], margins=True)
```

Expected pattern: diagonal dominance (most constraints stay in same or adjacent
tier), but with meaningful off-diagonal movement (if 95%+ stays unchanged, the
ML model isn't adding value).

---

## Execution Summary

| Gate | Type | Scope | Est. time | Blocks deployment? |
|------|------|-------|-----------|-------------------|
| A | Improvement | f0/f1 | ~5s | YES |
| B | Exact repro | f0/f1, all holdout months | ~60s | YES |
| C | Passthrough | f2/f3/q* | ~3s | YES |
| D | Schema | all | ~2s | YES |
| E1,E3-E6 | Rank/tier invariants | f0/f1 | ~1s | YES |
| E2 | Score diagnostics | f0/f1 | ~1s | No (diagnostic) |
| F1-F2 | Score validity | f0/f1 | ~1s | YES |
| F3-F4 | Score diagnostics | f0/f1 | ~1s | No (warn only) |
| G | SF parity | all | ~3s | YES |
| H | Temporal | f0/f1 | ~5s | YES |
| I | Determinism | f0 x 1mo | ~6s | No (recommended) |
| J | Forward month | 2026-03, all 4 ML slices + f2 | ~5s | YES |
| K | Stability | f0 x 4mo | ~12s | No (recommended) |
| L | Cross-slice | f0/f1 x on/off | ~3s | No (recommended) |
| M | Diff analysis | f0/f1 | ~2s | No (recommended) |

Total mandatory: ~85s. Total with recommended: ~110s.

---

## Failure Response

| Gate | Failure | Most likely cause | Fix |
|------|---------|------------------|-----|
| A | ML not better than V6.2B | Wrong features, wrong blend weights, or training bug | Diff config vs run_v10e_lagged.py |
| B | VC@20 doesn't match holdout | Feature computation diverged | Compare raw scores pre-rank against run_v10e_lagged output |
| C | Passthrough differs | Accidentally modified non-ML slice | Check ptype routing logic |
| D | Schema mismatch | Column dropped or dtype changed | Check DataFrame construction |
| E | Tier formula wrong | Using floor instead of ceil, or missing SO_MW exception | Use `ceil(rank*5)-1` |
| F1-F2 | Degenerate/inverted scores | Empty features, NaN propagation, or score polarity bug | Check enrich_df output, verify -scores in dense_rank |
| F3-F4 | Diagnostic warning | Unusual reranking strength or feature shift | Investigate but do not block; compare vs holdout |
| G | SF differs | Wrong source signal or modified during copy | Check ShiftFactorSignal path |
| H | Temporal leak | Training on future data | Audit collect_usable_months + bf cutoff |
| I | Non-deterministic | Thread count > 1 or missing seed | Verify num_threads=4, seed=42 |
| J | Forward month fails | load_v62b_month requires GT, or f1 delivery offset wrong, or offpeak cache missing | Verify require_gt=False, delivery_month(M, f1), offpeak cache files |
| K | Unstable rankings | Overfitting to training noise | Check training window, feature stability |
| L | Slices uncorrelated | Bug in per-slice blend weights or bf computation | Check BLEND_WEIGHTS dict |
| M | No meaningful movers | Model reproducing V6.2B instead of improving | Check binding_freq features are populated |
