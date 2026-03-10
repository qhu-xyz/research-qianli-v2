# V7.0 Signal Deployment -- Implementation Plan

## Background

### What signals are and how they're consumed

The MISO FTR trading pipeline uses **constraint signals** to decide which transmission
constraints to trade. A signal is a set of parquet files, one per
(auction_month, period_type, class_type), stored at a well-known path. Downstream code
(pmodel ftr22/ftr23) loads a signal by name and uses three columns to select and rank
constraints: `tier` (0-4, lower = more likely to bind), `rank` (0-1 normalized score),
and `equipment` (for position grouping).

The current production signal is **V6.2B**, which uses a formula:

```
rank_ori = 0.60 * da_rank_value + 0.30 * density_mix_rank_value + 0.10 * density_ori_rank_value
rank     = dense_rank(rank_ori) / K          # K = number of unique dense ranks, NOT row count n
tier     = ceil(rank * 5) - 1, clamped to 0..4  # right-inclusive quintile bucket
```

- `da_rank_value`: historical DA shadow price percentile (60-month lookback, lower = more binding)
- `density_mix_rank_value`: Spice6 mixed-scenario flow forecast percentile (lower = more binding)
- `density_ori_rank_value`: Spice6 baseline-scenario flow forecast percentile (lower = more binding)

### What V7.0 replaces

**V7.0** replaces the formula scoring for `f0` (front-month) and `f1` (second-month) with
an ML model trained on realized DA outcomes. All other period types (f2, f3, q2-q4) remain
exact V6.2B passthrough.

The ML model (stage5 v10e-lag1 for f0, v2 for f1) uses LightGBM LambdaRank with 9 features
including binding frequency from realized DA history, the V6.2B formula score as a feature,
Spice6 density exceedance probability, and the historical DA rank. It produces a raw score
per constraint, which gets converted to rank/tier in the same format as V6.2B.

### Why this matters (performance)

Holdout evaluation (2024-2025, walk-forward retrain -- each eval month trains a fresh
model on trailing 8 months, so no data from the eval month or later is used):

| Slice | v0 (formula) VC@20 | ML VC@20 | Delta | Months |
|-------|-------------------|----------|-------|--------|
| f0/onpeak | 0.1835 | 0.3529 | +92% | 24 |
| f0/offpeak | 0.2075 | 0.3780 | +82% | 24 |
| f1/onpeak | 0.2209 | 0.3677 | +66% | 19 |
| f1/offpeak | 0.2492 | 0.3561 | +43% | 19 |

Source: `research-stage5-tier/holdout/{ptype}/{ctype}/{version}/metrics.json`

VC@20 = value capture at top 20 constraints (what fraction of total binding value is
captured by the top 20 ranked constraints). Higher is better. These gains are
production-safe: temporal leakage was identified and fixed (see Temporal Lag section).

---

## Signal Output Format (must match V6.2B exactly)

### Constraints parquet

Path: `{data_root}/signal_data/miso/constraints/{signal_name}/{auction_month}/{period_type}/{class_type}/`

Index: `{constraint_id}|{flow_direction}|spice` (string, e.g. `232780|-1|spice`)

Downstream code (`pmodel/base/ftr22/base.py:1008`) extracts `shadow_sign` and
`monitored_line_id` by splitting the index on `|`. The index format must be exact.

Required columns (20):

| Column | Type | V7.0 ML slices | V7.0 passthrough slices |
|--------|------|----------------|------------------------|
| constraint_id | String | Pass through | Pass through |
| flow_direction | Int64 | Pass through | Pass through |
| shadow_price_da | Float64 | Pass through | Pass through |
| shadow_price | Float64 | Pass through | Pass through |
| shadow_sign | Int64 | Pass through | Pass through |
| da_rank_value | Float64 | Pass through | Pass through |
| density_mix_rank_value | Float64 | Pass through | Pass through |
| density_ori_rank_value | Float64 | Pass through | Pass through |
| density_mix_rank | Float64 | Pass through | Pass through |
| rank_ori | Float64 | **ML score** | Pass through |
| rank | Float64 | **Computed** | Pass through |
| tier | Int64 | **Computed** | Pass through |
| ori_mean | Float64 | Pass through | Pass through |
| mix_mean | Float64 | Pass through | Pass through |
| mean_branch_max | Float64 | Pass through | Pass through |
| mean_branch_max_fillna | Float64 | Pass through | Pass through |
| branch_name | String | Pass through | Pass through |
| bus_key | String | Pass through | Pass through |
| bus_key_group | String | Pass through | Pass through |
| equipment | String | Pass through | Pass through |

**Only 3 columns change** for ML slices: rank_ori, rank, tier. Everything else is V6.2B.

**rank_ori semantics change**: In V6.2B, rank_ori is a weighted blend (range ~0-1,
lower = more binding). In V7.0 ML slices, rank_ori will contain raw LightGBM prediction
scores (arbitrary range, higher = more binding -- opposite polarity). Downstream code
(`pmodel/base/ftr22/base.py`) uses `tier`, `rank`, `shadow_price`, and `equipment` but
does NOT appear to use `rank_ori` directly. This must be confirmed before deployment --
if any consumer reads rank_ori, the sign flip will silently produce wrong results.

### Shift factor parquet

Path: `{data_root}/signal_data/miso/sf/{signal_name}/{auction_month}/{period_type}/{class_type}/`

Shape: (n_nodes, n_constraints). Columns = constraint index strings. Rows = node names.

**Pass through unchanged from V6.2B.** SF depends on the constraint universe (which
constraints exist), not on scoring. V7.0 has the same constraints as V6.2B.

### SO_MW_Transfer exception

In V6.2B, `SO_MW_Transfer` has rank_ori near 0 (most extreme) but tier = 1 (not 0).
This is a hardcoded upstream exception. V7.0 must preserve: after ML scoring, if
`SO_MW_Transfer` is present, force tier = 1.

---

## MISO Auction Schedule

Which period types exist for each calendar month:

| Month | Period types |
|-------|-------------|
| Jan | f0, f1, q4 |
| Feb | f0, f1, f2, f3 |
| Mar | f0, f1, f2 |
| Apr | f0, f1 |
| May | f0 only |
| Jun | f0 only |
| Jul | f0, f1, q2, q3, q4 |
| Aug | f0, f1, f2, f3 |
| Sep | f0, f1, f2 |
| Oct | f0, f1, q3, q4 |
| Nov | f0, f1, f2, f3 |
| Dec | f0, f1, f2 |

V7.0 uses ML for f0 and f1. Everything else is V6.2B passthrough.

---

## Temporal Lag Rules (CRITICAL -- data leakage prevention)

For period type fN, the signal for auction month M is submitted **~mid of month M-1**.
At submission time, only realized DA through month **M-2** is complete (M-1 is partial).

This means any feature derived from realized DA must be lagged. There are two separate
mechanisms, and they are easy to confuse:

**1. Training row selection** (which auction months to include as training data):
`collect_usable_months(M, fN, n_months=8)` walks backward from M and only includes
months whose delivery month has complete realized DA. For f0, delivery_month = auction_month,
so the latest training month is M-2. For f1, delivery_month = auction_month + 1, so the
latest training month is M-3 (because its delivery month M-2 must have complete DA).

**2. Binding freq cutoff** (per-row feature, same for both f0 and f1):
`BF_LAG = 1` always. For any row with auction month T, bf features use realized DA
strictly before `prev_month(T)`, i.e., months < T-1. This is the decision-time cutoff
(at ~mid of T-1, only DA through T-2 is complete). The lag is 1 for BOTH f0 and f1 --
it is keyed on auction month, not delivery month.

**3. Target month**: does NOT need realized DA. The target is what we're scoring (inference
only). Ground truth for the target is not available at decision time.

Without these lags, results are inflated by 6-20%. The production-safe versions are
v10e-lag1 (f0) and v2 (f1).

---

## Deployment Flow

For one auction month:

```
1. PREFLIGHT: ensure realized DA cache has all required months
2. Load binding sets from cache (shared across all ML slices)
3. For each (period_type, class_type) available this month:
   a. If f0 or f1:
      - Collect 8 usable training months (respecting lag + auction schedule)
      - Load training data: V6.2B features + spice6 + realized DA ground truth
      - Enrich with binding_freq and formula score features
      - Train LightGBM LambdaRank model
      - Load target month: V6.2B features + spice6 (NO realized DA needed)
      - Enrich target with binding_freq + formula score
      - Score target with trained model
      - Convert scores to rank/tier
      - Replace rank_ori, rank, tier in V6.2B parquet
      - Apply SO_MW_Transfer exception
   b. If other ptype: copy V6.2B parquet unchanged
   c. Write constraints parquet via ConstraintsSignal.save_data()
   d. Copy shift factor parquet from V6.2B to V7.0 path
```

---

## Gap 1: Realized DA Cache Migration (REQUIRED, OPERATIONAL)

### Current state

Cache lives inside the git repo at `research-stage5-tier/data/realized_da/`.
162 files (81 months x 2 peak types), covering 2019-06 through 2026-02.
No auto-refresh. Manual `fetch_and_cache_month()` calls required.

### What's needed

**Move cache to shared persistent location**: `/opt/temp/qianli/spice_data/miso/realized_da/`

Make path configurable via environment variable:
```python
REALIZED_DA_CACHE = os.environ.get(
    "REALIZED_DA_CACHE",
    "/opt/temp/qianli/spice_data/miso/realized_da",
)
```

**Add explicit preflight** (`ensure_realized_da_cache`):

1. Compute all required realized DA months for the target auction month, across all
   ML period types (f0, f1) and class types (onpeak, offpeak).
2. Check which months are missing from cache (check per peak_type -- they are separate files).
3. Fetch missing months via `fetch_and_cache_month()` (one Ray API call per month, ~10s each).
4. Only then proceed with train + score.

This is a **top-level preflight step**, not hidden inside model code. It runs once before
any ML work, and if it fails (e.g., API unreachable), the job stops cleanly.

### Required months computation

For auction month M, period type P:

- **Training labels**: each training month T (from `collect_usable_months(M, P)`) needs
  realized DA for `delivery_month(T, P)`. This is usually the binding constraint on the
  latest required month.
- **Binding frequency**: bf features use realized DA for months in the lookback window.
  The bf cutoff is `prev_month(auction_month_of_row)` -- i.e., `BF_LAG=1`, keyed on
  auction month, same for both f0 and f1 (see Temporal Lag section). Lookback goes up to
  15 months before cutoff. In practice, training label requirements already cover these
  months, so bf does not independently extend the required set.

In practice, the binding set cache covers 2019-06+, so the main gap is always the most
recent 1-3 months. The preflight typically fetches 0-2 new months.

### Example: what months are needed

| Auction Month | f0 needs DA through | f1 needs DA through | As of 2026-03 |
|---------------|---------------------|---------------------|---------------|
| 2026-03 | 2026-01 | 2025-12 | Covered |
| 2026-04 | 2026-02 | 2026-02 | Covered |
| 2026-05 | 2026-03 | 2026-03 | **Preflight must fetch** |
| 2026-06 | 2026-04 | 2026-04 | **Preflight must fetch** |

### Safety

- **Locking**: write to temp file, rename atomically. Prevents partial writes if two jobs
  run concurrently. NOTE: current `fetch_and_cache_month()` in `realized_da.py:119` does a
  direct `write_parquet()` with no temp+rename -- this must be implemented, not just documented.
- **Peak type independence**: onpeak = `{month}.parquet`, offpeak = `{month}_offpeak.parquet`.
  Check and fetch each independently.
- **Centralize logic**: one function `required_realized_da_months(auction_month, ptype, ctype)`
  answers "which months are needed?". Do not duplicate.

### Source code reference

Existing code in `research-stage5-tier/ml/realized_da.py`:
- `load_realized_da(month, peak_type)` -- reads one cached parquet
- `fetch_and_cache_month(month, peak_type)` -- fetches from MISO API via Ray, writes parquet

Existing code in `research-stage5-tier/ml/config.py`:
- `collect_usable_months(target, ptype, n_months)` -- computes training month list
- `delivery_month(auction_month, ptype)` -- auction_month + period_offset

---

## Gap 2: Inference-Only Scoring Path (REQUIRED)

### Current state

`run_v10e_lagged.py:run_variant()` requires realized DA ground truth for the test month
to compute evaluation metrics. When GT is missing (the month hasn't happened yet), it skips
the month entirely. This is fine for research evaluation but blocks deployment.

### What's needed

A scoring function that:
1. Collects training months via `collect_usable_months()` (same as now)
2. Loads and enriches training data with `load_v62b_month()` + `enrich_df()` (same as now)
3. Computes query groups via `compute_query_groups()` (required by `train_ltr_model`)
4. Trains LightGBM model (same as now)
5. Loads target month V6.2B + spice6 **without** requiring realized DA
6. Enriches target month with binding_freq + formula score (same `enrich_df`)
7. Scores with trained model
8. Returns (constraint_ids, raw_scores) as aligned pairs -- NOT a bare scores array

**Alignment hazard**: the scoring function loads V6.2B internally via `load_v62b_month`
(polars DataFrame) while the assembly step (Gap 4) loads V6.2B separately via
`ConstraintsSignal.load_data()` (pandas DataFrame with string index). Row ordering is
not guaranteed to match between the two loads. To avoid this:
- The inference function should return `(constraint_ids, scores)` pairs, and
- The assembly step should join on constraint_id, not assign positionally.
- Alternatively, the inference function can accept the already-loaded DataFrame and
  return scores in the same row order, avoiding the double load entirely.

The key change: `load_v62b_month()` currently always joins realized DA as ground truth
for the target month. For inference, we need to skip that join. Options:
- Add `require_gt=False` parameter to `load_v62b_month()`
- Or write a lightweight `load_v62b_features_only()` that loads V6.2B + spice6 without GT

### Source code reference

Key functions in `research-stage5-tier/scripts/run_v10e_lagged.py`:
- `prev_month(m)` -- month arithmetic
- `load_all_binding_sets(peak_type)` -- loads all cached DA into {month: set(constraint_ids)}
- `compute_bf(cids, month, bs, lookback)` -- binding frequency for one lookback window
- `enrich_df(df, month, bs, lag, blend_weights)` -- adds bf_1/3/6/12/15 + v7_formula_score
- `run_variant(label, eval_months, bs, lag, class_type, period_type)` -- full loop

Key functions in `research-stage5-tier/ml/features.py`:
- `prepare_features(df, cfg)` -- extract feature matrix from DataFrame
- `compute_query_groups(df)` -- compute LTR query group sizes from query_month column
  (REQUIRED by `train_ltr_model`, must not be omitted)

### Effort

Small. Extract train+predict from `run_variant()`, add GT-optional loading.

---

## Gap 3: Rank/Tier Generation (REQUIRED)

### Current state

Not implemented. ML produces raw scores (higher = more binding), not rank/tier.

### Design choice: row-percentile tiering with deterministic tie-breaking

V6.2B uses `dense_rank_normalized` (divides by K = number of unique values). This
works for V6.2B because its formula produces nearly all unique scores (K ≈ n), giving
even ~20% quintile tiers. But ML with tiered labels produces ~55% unique scores (K ≈ 267
for n ≈ 489), causing uneven tiers: tier 4 holds 45-65% of constraints.

**V7.0 instead uses row-percentile ranking with deterministic tie-breaking:**

1. Sort constraints by ML score descending (higher = more binding)
2. Break ties with V6.2B rank_ori ascending (lower = more binding)
3. Final tie-break: original index ascending
4. `rank = row_position / n` (row-percentile, all ranks unique)
5. `tier = ceil(rank * 5) - 1` (guarantees ~20% per tier)

**Rationale:**
- Raw ML score is the real model output; tier is downstream packaging
- If two constraints have identical ML score, the model says "I can't distinguish them"
- V6.2B rank_ori is a stable, informative secondary signal
- Downstream consumers need usable, stable ~20% bucket sizes
- Dense-rank/K semantics under heavy ties produce operationally unstable tier sizes

```python
def compute_rank_tier(
    scores: np.ndarray,
    v62b_rank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(scores)
    order = np.lexsort((np.arange(n), v62b_rank, -scores))
    rank = np.empty(n, dtype=np.float64)
    rank[order] = (np.arange(n) + 1) / n
    tier = np.clip(np.ceil(rank * 5).astype(int) - 1, 0, 4)
    return rank, tier
```

**This is a product decision**, not a correctness fix. ML slices no longer mimic V6.2B's
dense-rank semantics exactly. This only affects f0/f1 (ML slices); f2+ stays as V6.2B
passthrough with the original dense_rank/K tiering.

### Effort

Small (~15 lines of code).

---

## Gap 4: Signal Assembly + Writing (REQUIRED)

### What's needed

For one (auction_month, period_type, class_type):

1. Load V6.2B constraints parquet via `ConstraintsSignal.load_data()` (returns pandas DataFrame
   with string index already in `{cid}|{fd}|spice` format)
2. If f0 or f1:
   - Run ML inference (Gap 2) to get raw scores per constraint
   - Compute rank/tier (Gap 3)
   - Replace `rank_ori`, `rank`, `tier` columns in the DataFrame
   - Apply SO_MW_Transfer exception (force tier=1 if present)
3. If other ptype: no changes needed
4. Write via `ConstraintsSignal("miso", v70_signal_name, ptype, ctype).save_data(df, auction_month)`
5. Load SF from V6.2B via `ShiftFactorSignal`, save under V7.0 name

### ConstraintsSignal API (from pbase)

```python
from pbase.data.dataset.signal.general import ConstraintsSignal, ShiftFactorSignal

# Constructor: ConstraintsSignal(rto, signal_name, period_type, class_type, is_dev=False)
# load_data(auction_month: pd.Timestamp) -> pd.DataFrame
# save_data(data: pd.DataFrame, auction_month: pd.Timestamp) -> str

# Signal names:
V62B_SIGNAL = "TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1"
V70_SIGNAL  = "TEST.TEST.Signal.MISO.SPICE_F0P_V7.0.R1"
```

`save_data` internally calls `update_parquet` which writes to the signal data path on NFS.
The DataFrame index and columns must match V6.2B exactly.

### Effort

Medium. Mostly wiring, no research.

---

## Not a Gap: Spice6 Schema Change

**Already fixed** in `research-stage5-tier/ml/spice6_loader.py`. From 2026-01 onward,
upstream renamed `score_df.parquet` (columns: 110, 100, 90...) to `score.parquet`
(single `score` column). The loader auto-detects the format. New `score` column is
functionally equivalent to old `prob_exceed_110` (Spearman = 0.9994).

## Not a Gap: Speed

In an observed benchmark run (2026-01, all 4 slices), total wall time was ~2.5s for
train + score across f0/f1 x onpeak/offpeak. Individual slices ranged from ~0.4-0.8s.
This is with the LightGBM backend (`num_threads=4`); earlier XGBoost runs were much
slower (~30-70s per slice). Actual deployment times may vary with data size and I/O
conditions -- treat 2.5s as an indicative estimate, not a guarantee.

After signal generation, consumers just read parquets -- zero ML at consumption time.

---

## ML Configuration Reference

### Features (9)

```
V10E_FEATURES = [
    "binding_freq_1",    # fraction of months bound in last 1 month
    "binding_freq_3",    # ... last 3 months
    "binding_freq_6",    # ... last 6 months
    "binding_freq_12",   # ... last 12 months
    "binding_freq_15",   # ... last 15 months
    "v7_formula_score",  # V6.2B-style weighted blend (used as feature, not as output)
    "prob_exceed_110",   # Spice6 exceedance probability at 110% of limit
    "constraint_limit",  # MW thermal limit
    "da_rank_value",     # historical DA shadow price percentile (from V6.2B parquet)
]
V10E_MONOTONE = [1, 1, 1, 1, 1, -1, 1, 0, -1]
```

Binding freq features (bf_1 through bf_15) are the primary signal. They measure how
often a constraint actually bound in realized DA outcomes over different lookback windows.

`v7_formula_score` is a blend of V6.2B features, **not** the V6.2B rank_ori directly.
The blend weights vary per slice:

| Slice | da_rank_value weight | density_mix weight | density_ori weight |
|-------|---------------------|-------------------|-------------------|
| f0/onpeak | 0.85 | 0.00 | 0.15 |
| f0/offpeak | 0.85 | 0.00 | 0.15 |
| f1/onpeak | 0.70 | 0.00 | 0.30 |
| f1/offpeak | 0.80 | 0.00 | 0.20 |

### LightGBM config (all slices)

```
objective:        lambdarank
label_mode:       tiered (4 levels per train.py:37-65:
                    0=non-binding (realized_sp==0, ~88% of rows),
                    1=binding bottom 50%, 2=binding top 50%, 3=binding top 20%)
n_estimators:     100
learning_rate:    0.05
num_leaves:       31
num_threads:      4       (CRITICAL: container has 64 CPUs, LightGBM deadlocks with auto-detect)
subsample:        0.8
colsample_bytree: 0.8
min_data_in_leaf: 25      (config field: min_child_weight, mapped in train.py:102)
reg_alpha:        1.0     (L1 regularization, LTRConfig default)
reg_lambda:       1.0     (L2 regularization, LTRConfig default)
seed:             42      (hardcoded in train.py:110, not in LTRConfig)
```

### Training config per slice

| Slice | Train months | Row selection | BF lag |
|-------|-------------|---------------|--------|
| f0 | 8 usable (via collect_usable_months) | collect_usable_months handles delivery month cutoff | 1 (months < M-1) |
| f1 | 8 usable (via collect_usable_months) | collect_usable_months handles delivery month cutoff | 1 (months < M-1) |

`collect_usable_months` walks backward from the target, skipping months where:
- The period type doesn't exist for that calendar month (e.g., f1 doesn't exist in May/Jun)
- The delivery month's realized DA isn't available yet

---

## Data Dependencies

| Source | Path | Coverage | Refresh? |
|--------|------|----------|----------|
| V6.2B signal | `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1` | through 2026-03 | No (upstream) |
| V6.2B shift factors | `/opt/data/xyz-dataset/signal_data/miso/sf/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1` | same | No (upstream) |
| Spice6 density | `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/` | through 2026-03 | No (upstream) |
| Realized DA cache | `/opt/temp/qianli/spice_data/miso/realized_da/` **(new location)** | through 2026-02 | **Yes, monthly (preflight)** |

### Realized DA cache file naming

- Onpeak: `{YYYY-MM}.parquet` (e.g. `2026-01.parquet`)
- Offpeak: `{YYYY-MM}_offpeak.parquet` (e.g. `2026-01_offpeak.parquet`)

Each file has columns: `constraint_id` (String), `realized_sp` (Float64).
`realized_sp` = abs(sum(shadow_price)) per constraint_id per month.

Fetched via: `MisoApTools().tools.get_da_shadow_by_peaktype(st, et_ex, peak_type)`.
Requires Ray to be initialized.

---

## Source Code Reference (research-stage5-tier)

All ML code lives in `research-stage5-tier/ml/`:

| Module | Key functions | Role |
|--------|--------------|------|
| `config.py` | `delivery_month()`, `collect_usable_months()`, `has_period_type()`, `MISO_AUCTION_SCHEDULE` | Paths, schedule, month math |
| `data_loader.py` | `load_v62b_month()` | Load V6.2B + spice6 + realized DA |
| `spice6_loader.py` | `load_spice6_density()` | Load Spice6 density (handles both schemas) |
| `realized_da.py` | `load_realized_da()`, `fetch_and_cache_month()` | Cache read/write |
| `features.py` | `prepare_features()`, `compute_query_groups()` | Feature matrix extraction |
| `train.py` | `train_ltr_model()`, `predict_scores()` | LightGBM training + inference |
| `v62b_formula.py` | `v62b_score()`, `dense_rank_normalized()` | Formula + rank utilities |

Scoring script: `research-stage5-tier/scripts/run_v10e_lagged.py`:

| Function | Role |
|----------|------|
| `load_all_binding_sets(peak_type)` | Load all cached DA into {month: set(cids)} |
| `compute_bf(cids, month, bs, lookback)` | Binding freq for one window |
| `enrich_df(df, month, bs, lag, blend_weights)` | Add bf_1/3/6/12/15 + formula score |
| `run_variant(label, eval_months, bs, lag, class_type, period_type)` | Full train+eval loop |
| `BLEND_WEIGHTS` | Per-slice blend weight dict |
| `V10E_FEATURES`, `V10E_MONOTONE` | Feature list + monotone constraints |

---

## Deployment Script Outline

```python
def generate_v70_signal(
    auction_month: str,
    signal_name: str = "TEST.TEST.Signal.MISO.SPICE_F0P_V7.0.R1",
):
    """Generate V7.0 signal for one auction month.

    Steps:
    1. Preflight: ensure realized DA cache coverage
    2. Load binding sets from cache
    3. For each available (ptype, ctype): score or passthrough, write signal + SF
    """
    ml_ptypes = ["f0", "f1"]
    class_types = ["onpeak", "offpeak"]

    # ── Step 1: Preflight ──
    ensure_realized_da_cache(auction_month, ptypes=ml_ptypes, ctypes=class_types)

    # ── Step 2: Load binding sets (shared across all ML slices) ──
    bs = {ct: load_all_binding_sets(peak_type=ct) for ct in class_types}

    # ── Step 3: Process each slice ──
    for ptype in available_ptypes(auction_month):
        for ctype in class_types:
            # Load V6.2B source (pandas DataFrame with string index)
            v62b_df = ConstraintsSignal(
                "miso", V62B_SIGNAL, ptype, ctype
            ).load_data(pd.Timestamp(auction_month))

            if ptype in ml_ptypes:
                # Extract V6.2B rank_ori for tie-breaking (before overwrite)
                v62b_cids = v62b_df.index.str.split("|").str[0]
                v62b_rank_map = pd.Series(
                    v62b_df["rank_ori"].values, index=v62b_cids
                ).groupby(level=0).first()

                # ML scoring: train on history, score target.
                cids, scores = score_ml_inference(
                    auction_month, ptype, ctype, bs[ctype]
                )

                # Row-percentile rank with V6.2B tie-breaking
                v62b_rank_for_cids = v62b_rank_map.reindex(cids).fillna(1.0).values
                rank, tier = compute_rank_tier(scores, v62b_rank_for_cids)

                # Join on constraint_id, NOT positional assignment
                score_df = pd.DataFrame({
                    "constraint_id": cids, "_rank_ori": scores,
                    "_rank": rank, "_tier": tier,
                }).set_index("constraint_id")
                v62b_df["rank_ori"] = score_df.loc[v62b_cids, "_rank_ori"].values
                v62b_df["rank"] = score_df.loc[v62b_cids, "_rank"].values
                v62b_df["tier"] = score_df.loc[v62b_cids, "_tier"].values

                # SO_MW_Transfer exception
                if "branch_name" in v62b_df.columns:
                    mask = v62b_df["branch_name"] == "SO_MW_Transfer"
                    if mask.any():
                        v62b_df.loc[mask, "tier"] = 1

            # Write constraints
            ConstraintsSignal(
                "miso", signal_name, ptype, ctype
            ).save_data(v62b_df, pd.Timestamp(auction_month))

            # Copy SF from V6.2B
            sf_df = ShiftFactorSignal(
                "miso", V62B_SIGNAL, ptype, ctype
            ).load_data(pd.Timestamp(auction_month))
            ShiftFactorSignal(
                "miso", signal_name, ptype, ctype
            ).save_data(sf_df, pd.Timestamp(auction_month))
```

---

## New Code to Write

| File | Purpose |
|------|---------|
| `ml/cache.py` | `required_realized_da_months()`, `ensure_realized_da_cache()` |
| `ml/inference.py` | `load_v62b_features_only()`, `score_ml_inference()` |
| `ml/signal_writer.py` | `compute_rank_tier()`, `available_ptypes()` |
| `scripts/generate_v70_signal.py` | CLI entrypoint |

Estimated total new code: ~200-300 lines. Most logic is extracted/reused from existing
`run_v10e_lagged.py` and `ml/` modules.

---

## Validation Plan

Before declaring V7.0 ready:

1. **Schema match**: V7.0 output has exact same columns, dtypes, index format as V6.2B
2. **Passthrough check**: f2/f3/q* partitions are bit-identical to V6.2B
3. **ML sanity**: f0/f1 rank_ori values differ from V6.2B, but:
   - rank values in (0, 1], normalized by K (unique dense ranks), matching V6.2B convention
   - tier distribution is roughly equal quintiles (0-4)
   - tier 0 constraints have highest ML scores
   - tier assignments match `ceil(rank * 5) - 1` clamped to [0, 4] (validated across 7 months)
4. **Round-trip**: `ConstraintsSignal.load_data()` on V7.0 returns valid DataFrame
5. **pmodel integration**: load V7.0 in ftr22 signal loading path, verify no errors
6. **Backtest**: generate V7.0 for a historical month where we have GT, verify ML
   rank/tier produce expected VC@20 improvement over V6.2B

---

## Open Questions

1. **V6.2B signal name**: is `TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1` the exact
   production name, or is there a non-TEST variant?
2. **V7.0 signal name**: confirm `TEST.TEST.Signal.MISO.SPICE_F0P_V7.0.R1` or different.
3. **SO_MW_Transfer**: is this still relevant in current months? Should the exception
   be parameterized?
4. **Upstream dependency**: V6.2B signal parquet and Spice6 density are generated by
   an upstream pipeline. What is the SLA for availability relative to auction timing?
