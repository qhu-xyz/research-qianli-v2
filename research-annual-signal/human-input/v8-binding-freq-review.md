# V8 Binding Frequency Review (Updated)

Date: 2026-03-10
Reviewer: Claude (stage5-tier agent)
Reviewed: `docs/plan-v8-binding-freq.md`, `ml/binding_freq.py`, `scripts/run_v8_binding_freq.py`, `ml/config.py`, `ml/data_loader.py`, `ml/train.py`, `ml/ground_truth.py`, `ml/features.py`, registry artifacts, and post-backfill results.

---

## Verdict

No temporal leakage. Implementation correct. Backfill data is valid. But the annual agent is drawing wrong conclusions from the results — see §4 and §5.

---

## 1. Temporal Leakage Check: PASS

- Each planning year `YYYY-06` uses BF cutoff `< YYYY-04` (through March)
- `enrich_with_binding_freq()` asserts `auction_month.endswith("-06")`
- Training groups use their OWN year's cutoff, not the eval year's
- Ground truth and BF never overlap (minimum 3-month gap for aq1, 12-month for aq4)
- Bridge table is correctly partition-filtered to `(auction_type='annual', auction_month, period_type, class_type='onpeak')`

---

## 2. Backfill: What We Did and What It Changed

### Action taken
Fetched 26 months of realized DA (2017-04 through 2019-05) for both onpeak and offpeak. Cache now covers 107 months (2017-04 through 2026-02) for both peak types.

### What actually changed in features

**Eval groups (2022-06, 2023-06, 2024-06): ZERO change.** BF uses `available[-window:]` (the most recent N months before cutoff). Adding older months does not change the lookback window for any group that already had sufficient data.

**Training groups only:**

| Planning Year | bf_1..bf_6 | bf_12 | bf_24 |
|:---:|:---:|:---:|:---:|
| 2019-06 | NaN → computed | NaN → computed | NaN → computed |
| 2020-06 | unchanged | NaN → computed | NaN → computed |
| 2021-06 | unchanged | unchanged | NaN → computed |
| 2022-06+ | unchanged | unchanged | unchanged |

### Backfill data quality: VERIFIED
- Bridge coverage: 78.4% of 2019 V6.1 branches found in binding sets (comparable to 2024)
- BF distribution: mean bf_12 = 0.261 for 2019 (vs 0.255 for 2024) — similar
- Schema and aggregation consistent with existing cache files

---

## 3. Post-Backfill Results: The Annual Agent's Conclusions Are Wrong

The annual agent reported that backfill hurt VC@20 and concluded the model has hit a ceiling. Three problems with this analysis:

### Problem 1: "Adding more data shouldn't hurt" — correct intuition

The backfill provides strictly more information to the model. If VC@20 dropped, the most likely cause is **Issue B below (feature set drift)**, not the backfill itself. The code's `_BF_FEATURES` was expanded from 5 to 8 windows + `bf_months_avail` AFTER the original experiment. If the re-run picked up the new feature definitions, the comparison is not apples-to-apples.

**The annual agent must confirm**: did the post-backfill run use 7 features (original v8b) or 11 features (current code)?

If the feature set changed, the backfill comparison is invalid and must be re-run with the original 7 features to isolate the backfill effect.

### Problem 2: Holdout must be run

Dev eval (12 groups, 2022-2024) is not sufficient to declare a ceiling. The holdout (4 groups, 2025-06) was specifically set aside for this purpose. Results from dev eval alone can be misleading — the stage5-tier monthly research showed cases where dev and holdout disagreed.

Run the holdout for ALL variants (v8b, v10e, best blend) and compare. Only then can you draw conclusions about whether the backfill helps or hurts generalization.

### Problem 3: VC@20 is ONE metric — use the full gate system

The stage5-tier monthly research uses **12 metrics across 2 groups** (see `registry/f0/onpeak/gates.json`):

**Group A (blocking)**:
- VC@20, VC@100
- Recall@20, Recall@50
- NDCG

**Group B (monitoring)**:
- VC@10, VC@25, VC@50, VC@200
- Recall@10, Recall@100
- Spearman

Plus **tail risk**: `bottom_2_mean` for each metric (worst 2 groups).

The annual agent already noted that Spearman improved (0.486 → 0.500) and Recall@100 improved (0.562 → 0.576). A version that improves Spearman, Recall@100, and possibly NDCG/VC@100 but slightly drops VC@20 may still be a BETTER model overall — it ranks more broadly correct even if the top-20 concentration shifts slightly.

**The annual agent should**:
1. Print ALL 12 metrics for pre-backfill vs post-backfill (same feature set!)
2. Check tail risk (bottom_2_mean) — does the backfill reduce variance?
3. Only then decide which version is truly better

Declaring a "ceiling" based on one metric (VC@20) while ignoring improvements in 2+ other metrics is incomplete analysis.

---

## 4. Issue B (CRITICAL): Code-to-Registry Feature Mismatch

The code now defines:
```python
_BF_FEATURES = ["bf_1", "bf_3", "bf_6", "bf_12", "bf_15", "bf_24", "bf_36", "bf_48"]  # 8 features
SET_V8_LEAN_FEATURES = ["shadow_price_da", "da_rank_value"] + _BF_FEATURES + ["bf_months_avail"]  # 11 features
```

The original v8b experiment used:
```python
features: ["shadow_price_da", "da_rank_value", "bf_1", "bf_3", "bf_6", "bf_12", "bf_24"]  # 7 features
```

If the post-backfill run used the expanded 11-feature set, the VC@20 drop could be entirely from feature bloat (4 extra features on small data), not from the backfill.

**Action required**: Check `registry/v8b/config.json` after the re-run. If it shows 11 features, the comparison is invalid.

---

## 5. Recommended Actions for the Annual Agent

1. **Pin features**: Before re-running, ensure `SET_V8_LEAN_FEATURES` matches the original 7-feature set (or explicitly test both)
2. **Re-run with backfill + original features**: Isolate the backfill effect from the feature set change
3. **Run holdout**: For v8b, v10e, and best blend — dev eval alone is insufficient
4. **Report ALL metrics**: Print the full 12-metric table (VC@10/20/25/50/100/200, Recall@10/20/50/100, NDCG, Spearman) plus bottom_2_mean for tail risk
5. **Compare properly**: A version that trades -1% VC@20 for +3% Spearman and +2% Recall@100 may be genuinely better for the trading use case

---

## 6. What Was Already Correct

- Cutoff logic: `< YYYY-04` for annual auction
- Bridge table filtering: partition-filtered per (auction_month, period_type)
- bf=0 as informative signal (never bound = unlikely to bind)
- Per-group bridge selection (each training year uses its own bridge)
- Tiered labels: 5 levels (0=non-binding, 1-4=quantile buckets)
- Query groups: data sorted by query_group, compute_query_groups correct
- Memory management: del + gc.collect() between stages
- Preliminary validation: bf_12 beats shadow_price_da 12/12 quarters (+33.5% Spearman)
