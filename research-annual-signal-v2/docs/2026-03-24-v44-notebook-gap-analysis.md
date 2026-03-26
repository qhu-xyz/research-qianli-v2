# 2026-03-24 V4.4 Notebook Gap Analysis

## Purpose

This note captures what we learned from the recovered V4.4 source notebook:

- source notebook: `human-inputs/se-signal-annual.ipynb`
- key question: why has our internal annual/NB work repeatedly struggled to beat V4.4 on the top dormant branches?

## Bottom Line

V4.4 is not winning because it is a better-tuned version of our current model family.
It is winning because it uses a different signal family:

- richer forward-looking flow-stress features
- stronger emphasis on extreme tail behavior
- constraint-first ranking before branch/equipment dedup
- only modest reliance on historical DA
- round-aware DA cutoff logic

Our current annual models are much more history-driven and branch-collapsed, which is exactly the wrong bias for dormant outlier binders.

## What The Notebook Actually Does

The core generation logic is in notebook cell 3.

### 1. Build flow exceedance profiles from perturbation flows

For each market month in the annual quarter window, the notebook:

- loads `flow_memo.parquet`
- filters to useful constraints with `limit >= 10`
- drops columns where the flow percentage is always too extreme
- keeps only numeric constraint ids
- samples hours `[0, 5, 12, 18]`

Then it builds exceedance counts for thresholds:

- `60, 65, 70, ..., 100`

and does this in two variants:

- `max` style: whether a threshold is hit
- `sum` style: how often/how broadly it is hit

### 2. Convert those profiles into deviation distances

The notebook converts the exceedance histograms into a distance-from-baseline score using:

- `constraint_prediction.constraints_flow_deviation_distance(...)`

That function is implemented in:

- `/home/xyz/workspace/pbase2/src/pbase/analysis/constraints_prediction/v1/base.py`

Operationally, it is a weighted Wasserstein-style distance from a baseline distribution concentrated at bin 0.

The notebook uses weights:

- `weight = np.array([7**i for i in range(len(total_cols) + 1)])`

So higher-flow bins are emphasized very aggressively.

### 3. Add a small amount of DA history

The notebook also loads about two years of DA shadow history up to a round-specific `run_at` date:

- R1: April 7
- R2: April 24
- R3: May 1

It aggregates DA by:

- `equipment`
- `convention`
- `shadow_sign`

and forms a percentile-style history term:

- `shadow_rank`

### 4. Final ranking formula

The final rank is:

- `0.3 * deviation_max_rank`
- `0.5 * deviation_sum_rank`
- `0.2 * shadow_rank`

This means 80% of the final rank is flow-deviation based, and only 20% is DA-history based.

### 5. Constraint-first, then dedup by equipment

The notebook ranks constraints first, then dedups by:

- `equipment`

keeping the best-ranked representative per equipment/branch-like object.

This is important because our current annual pipeline collapses to branch much earlier.

## Why This Beats Our Current Annual/NB Models On Dormant Branches

### A. V4.4 uses a stronger forward-looking signal

Our current annual work mostly uses:

- BF/history
- DA rank / DA cumulative SP
- a narrow branch-level density summary (`80/90/100/110`, `rt_max`, `top2`)

V4.4 instead measures:

- how strongly a constraint lives in high-flow states
- across outages
- across months
- with heavy emphasis on the right tail

That is much closer to the real mechanism behind dormant but dangerous binders.

### B. V4.4 keeps more shape information

We mostly keep a collapsed branch-level tail summary.

V4.4 keeps:

- many exceedance thresholds from `60` through `100`
- both `max` and `sum` versions
- a distance calculation over the whole shape

That is much richer than our current `rt_max + top2` view.

### C. V4.4 is not strongly history-biased

Dormant branches are exactly where BF/history underperform.

Our models have often been dominated by:

- `bf`
- `da_rank_value`
- `shadow_price_da`

V4.4 only gives history 20% of the final score, so a dormant branch can still rank well if its flow profile is dangerous.

### D. V4.4 is constraint-first

V4.4 lets individual constraints compete before deduping by equipment.

Our pipeline collapses to branch much earlier, which likely loses:

- within-branch heterogeneity
- single dangerous CID behavior
- one-direction exposure asymmetry

This is especially damaging for dormant branches where one CID may be the entire story.

### E. V4.4 is already round-aware

The notebook has different `run_at` dates for R1/R2/R3 and reads round-specific flow paths.

Our current annual code is still effectively:

- `market_round=1` on auction-side data
- month-level March cutoff on history

So V4.4 has a structurally better alignment to the actual annual submission rounds.

## Concrete Examples From Current Data

### `MNTCELO  TR6__2`

In our cache for `2025-06/aq2/onpeak`:

- `bf_12 = 0`
- `da_rank_value = 2101`
- `shadow_price_da = 0.05`
- `rt_max = 1.307565`

This is hard for a history-driven model to rank well.

In V4.4:

- `deviation_max = 15.465499`
- `deviation_sum = 12.821697`

So V4.4 sees a very strong flow-deviation signal even though historical DA is weak.

### `13866 A`

In our cache for `2025-06/aq2/offpeak`:

- `bfo_12 = 0`
- `da_rank_value = 1357`
- `shadow_price_da = 223.17`
- `rt_max = 0.024219`

This is nearly invisible to our current density summary.

In V4.4:

- `deviation_max = 14.711204`
- `deviation_sum = 9.586741`

Again, V4.4 is seeing a strong dangerous-flow signature that our current branch-level summary does not capture.

## What This Implies

The dormant-branch gap is mostly a feature/representation gap, not just a tuning gap.

If we want to close it, the most promising directions are:

1. Add V4.4-style flow deviation features from raw perturbation flow distributions.
2. Preserve more CID/constraint-level structure before branch collapse.
3. Make annual features round-aware.
4. Reduce overreliance on BF/history in the dormant specialist.

## What This Does Not Prove

This note does not prove that V4.4 is globally better than our system.

It only explains why V4.4 has been hard to beat on:

- top dormant branches
- especially rare, high-impact outliers

Our broader deployment stack can still outperform it on:

- total shortlist SP
- production coverage
- internal reproducibility

## Recommended Follow-Up

Before the next annual model cycle:

1. Turn the notebook logic into a clean feature specification:
   - threshold grid
   - max/sum exceedance features
   - deviation distance transform
   - round-aware cutoff behavior

2. Build one internal experimental model that adds these features without changing the rest of the evaluation framework.

3. Compare it directly against:
   - `v0c`
   - `Bucket_6_20`
   - published V4.4
   on:
   - native top-K
   - overlap-only dormant ranking
   - deployment R30/R50
