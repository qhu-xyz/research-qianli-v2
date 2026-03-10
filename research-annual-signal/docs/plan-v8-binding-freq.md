# V8: Add Monthly Binding Frequency Features to Annual Signal

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Augment the annual MISO FTR constraint ranking model with monthly binding frequency features derived from realized DA shadow prices, then evaluate whether they improve ranking quality over the current best (blend_v7d_a70).

**Architecture:** Build a new module (`ml/binding_freq.py`) that maps realized DA constraint_ids to V6.1 annual branch_names via the MISO_SPICE_CONSTRAINT_INFO bridge table, then computes multi-window binding frequency per branch. Integrate into the existing data enrichment pipeline. Run experiments with the new features added to the v7d model.

**Tech Stack:** Python 3.13, Polars, LightGBM LambdaRank, NumPy. Data on NFS parquet. Ray required for fetching new realized DA months (existing cache covers 2019-06 through 2026-02).

---

## Background: What This Project Does

### The Business Problem

We trade **MISO annual Financial Transmission Rights (FTRs)** — contracts that pay out based on transmission congestion. Each year (~April submission, June start), MISO runs an annual FTR auction covering 4 quarters (aq1=Jun-Aug, aq2=Sep-Nov, aq3=Dec-Feb, aq4=Mar-May).

Before the auction, we need to **rank ~300-800 transmission constraints** by how likely they are to be congested (i.e., have high realized shadow prices). Better ranking = better trade selection = higher returns.

### The Signal Pipeline

The existing **V6.1 annual signal** is produced by an upstream SPICE simulation pipeline. It gives us, for each constraint in each quarter:
- A **constraint universe** (which constraints to consider)
- **Features** for each constraint (described below)

Our research builds ML models on top of this signal to improve the ranking.

### Current Best: blend_v7d_a70

The current best approach blends two signals:
1. **v7d (ML model):** LightGBM LambdaRank with 7 features, 4-tier relevance labels
2. **v0b (formula):** Pure `da_rank_value` (rank of historical DA shadow prices)

Blended as: `0.70 * minmax(ML_scores) + 0.30 * minmax(formula_scores)`

**Performance:**
- Dev eval (12 quarters, 2022-2024): VC@20 = 0.3113
- Holdout (4 quarters, 2025): VC@20 = 0.2513 (+15.4% vs v0b formula)

### The Problem with Current Features

Feature importance analysis reveals the ML model is dominated by historical features:

| Feature | Type | Gain% |
|---------|------|------:|
| shadow_price_da | Historical | 81.2% |
| da_rank_value | Historical (rank of above) | 14.8% |
| mix_mean | Prediction (SPICE simulation) | 3.3% |
| mean_branch_max | Prediction (SPICE simulation) | 0.7% |
| density_mix_rank_value | Prediction | 0.1% |
| ori_mean | Prediction | 0.0% |
| density_ori_rank_value | Prediction | 0.0% |

96% of the model's gain comes from one signal: historical DA shadow prices. The expensive SPICE simulation features add almost nothing.

### Why Binding Frequency Should Help

In the **monthly** signal research (research-stage5-tier), binding frequency is the **#1 feature** — multi-window bf features (bf_1, bf_3, bf_6, bf_12, bf_15) dominate the LightGBM model.

**Preliminary analysis on annual data confirms this:**

| Feature | Spearman with realized shadow price |
|---------|:---:|
| shadow_price_da | 0.3678 (mean over 12 quarters) |
| **bf_12** | **0.4912 (+33.5%)** |

bf_12 beats shadow_price_da in **all 12 dev quarters** (12/12). Even more striking: constraints with high bf_12 but near-zero shadow_price_da have 89% binding rate vs 45% base rate — bf captures recent binding patterns that the historical average misses.

---

## Data Architecture

### Identifier Mapping Challenge

The annual and monthly systems use different constraint identifiers:

```
Realized DA shadow prices:  constraint_id (numeric MISO IDs, e.g., "72691")
V6.2B monthly signal:       constraint_id (same as DA — direct join works)
V6.1 annual signal:         branch_name (e.g., "ADDIS__WILBT_4 A")
```

**We cannot directly join** realized DA to V6.1 annual. We need a bridge:

```
Realized DA constraint_id
    → MISO_SPICE_CONSTRAINT_INFO (bridge table)
    → branch_name
    → V6.1 annual signal
```

The bridge table lives at `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_INFO.parquet` and must be **partition-filtered** to `(auction_type='annual', auction_month, period_type, class_type='onpeak')` to avoid fan-out (one DA constraint_id can map to many branch_names across different partitions).

### Data Sources

| Source | Path | Key Columns |
|--------|------|-------------|
| V6.1 annual signal | `/opt/data/xyz-dataset/signal_data/miso/constraints/Signal.MISO.SPICE_ANNUAL_V6.1/{year}/{aq}/onpeak/` | branch_name, shadow_price_da, da_rank_value, ... |
| Realized DA cache | `/home/xyz/workspace/research-qianli-v2/research-stage5-tier/data/realized_da/{month}.parquet` | constraint_id, realized_sp |
| Bridge table | `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_INFO.parquet` | constraint_id, branch_name, auction_type, auction_month, period_type, class_type |

### Realized DA Cache Coverage

The stage5-tier monthly research has already cached realized DA shadow prices:
- **81 onpeak months:** 2019-06 through 2026-02
- Format: `{month}.parquet` with columns `[constraint_id: String, realized_sp: Float64]`
- Binding = `realized_sp > 0`

### Timing / Leakage Rules

The annual auction is submitted **~early April** for a planning year starting June.

```
Planning year 2024: submitted ~April 2024
  → Can use realized DA through March 2024 (months < "2024-04")
  → CANNOT use April 2024 or later
  → 12-month lookback: April 2023 through March 2024
```

**Rule:** For planning year `YYYY-06`, the binding freq cutoff is `YYYY-04` (strictly `< YYYY-04`).

This is analogous to the monthly signal's lag=1 rule, but adapted for annual timing.

---

## Binding Frequency Definition

For a constraint identified by `branch_name` in planning year `YYYY-06`:

```python
binding_freq_N = count(months in lookback window where branch was binding) / N
```

Where:
- "binding" = the branch_name appears in the realized DA with `realized_sp > 0` for that month
- Lookback window = the last N months before the cutoff (`YYYY-04`)
- Example: bf_12 for 2024-06 uses months 2023-04 through 2024-03

### Windows

| Window | Lookback | What It Captures |
|--------|----------|------------------|
| bf_1 | 1 month | Very recent binding (March only) |
| bf_3 | 3 months | Recent quarter (Jan-Mar) |
| bf_6 | 6 months | Recent half-year |
| bf_12 | 12 months | Full year cycle — captures seasonality |
| bf_24 | 24 months | Two-year trend — structural vs transient |

### Coverage (from preliminary analysis)

| Window | V6.1 constraints with bf > 0 |
|--------|------------------------------|
| bf_1 | 23.4% |
| bf_3 | 37.7% |
| bf_6 | 50.7% |
| bf_12 | 66.4% |
| bf_24 | 80.4% |

Constraints with bf=0 are informative: they haven't bound recently, so they're less likely to bind in the future.

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `ml/binding_freq.py` | Core module: load bridge table, build monthly binding sets by branch_name, compute multi-window binding frequency. Caches intermediate results. |
| `scripts/run_v8_binding_freq.py` | Experiment script: run v8 variants (v7d + bf features), compare against v0b and blend_v7d_a70. Dev eval + holdout. |

### Modified Files

| File | Change |
|------|--------|
| `ml/config.py` | Add BF feature names and monotone constraints to feature set definitions. |
| `ml/data_loader.py` | Add `enrich_with_binding_freq()` call in `load_v61_enriched()`. |

### Unchanged Files (for reference)

| File | Why It Matters |
|------|---------------|
| `ml/ground_truth.py` | Uses same bridge table + mapping approach — reference implementation. |
| `ml/pipeline.py` | Training/evaluation loop — unchanged, just sees new features. |
| `ml/train.py` | LightGBM training — unchanged. |
| `ml/evaluate.py` | Metrics computation — unchanged. |
| `research-stage5-tier/ml/realized_da.py` | Monthly realized DA loader — we READ its cache, don't modify it. |
| `research-stage5-tier/data/realized_da/*.parquet` | The cached monthly realized DA data. |

---

## Chunk 1: Core Binding Frequency Module

### Task 1: Create `ml/binding_freq.py` — Bridge Table Loading

**Files:**
- Create: `ml/binding_freq.py`

The bridge table maps realized DA `constraint_id` to V6.1 `branch_name`. We must partition-filter it to avoid fan-out.

- [ ] **Step 1: Create `ml/binding_freq.py` with bridge table loader**

```python
"""Monthly binding frequency features for annual constraints.

Maps realized DA shadow prices (keyed by constraint_id) to V6.1 annual
signal (keyed by branch_name) via MISO_SPICE_CONSTRAINT_INFO bridge table,
then computes multi-window binding frequency per branch.

Data flow:
  Realized DA (constraint_id, realized_sp)
    → Bridge table (constraint_id → branch_name), partition-filtered
    → Monthly binding sets: {month: set(branch_name)}
    → Binding frequency: count(months bound) / window_size
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ml.config import SPICE_DATA_BASE

# Stage5-tier's realized DA cache — we read from it, don't write to it.
REALIZED_DA_CACHE = Path("/home/xyz/workspace/research-qianli-v2/research-stage5-tier/data/realized_da")

# Local cache for precomputed monthly binding sets (by branch_name)
_BF_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "binding_freq"

# Default binding frequency windows
BF_WINDOWS = [1, 3, 6, 12, 24]


def _load_bridge(auction_month: str, period_type: str) -> pl.DataFrame:
    """Load constraint_id → branch_name mapping from MISO_SPICE_CONSTRAINT_INFO.

    Partition-filtered to (auction_type='annual', auction_month, period_type,
    class_type='onpeak') to avoid fan-out where one constraint_id maps to
    different branch_names across partitions.

    Returns unique (constraint_id, branch_name) pairs.
    """
    info_path = Path(SPICE_DATA_BASE) / "MISO_SPICE_CONSTRAINT_INFO.parquet"
    bridge = (
        pl.scan_parquet(str(info_path))
        .filter(
            (pl.col("auction_type") == "annual")
            & (pl.col("auction_month") == auction_month)
            & (pl.col("period_type") == period_type)
            & (pl.col("class_type") == "onpeak")
        )
        .select(["constraint_id", "branch_name"])
        .collect()
        .unique()
    )
    return bridge
```

- [ ] **Step 2: Verify bridge table loads correctly**

Run a quick sanity check:
```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-annual-signal
python -c "
from ml.binding_freq import _load_bridge
b = _load_bridge('2024-06', 'aq1')
print(f'Bridge 2024-06/aq1: {len(b)} rows, {b[\"constraint_id\"].n_unique()} cids, {b[\"branch_name\"].n_unique()} branches')
"
```
Expected: ~32000 rows, ~14000 cids, ~6900 branches.

---

### Task 2: Build Monthly Binding Sets by Branch Name

- [ ] **Step 3: Add `build_monthly_binding_sets()` to `ml/binding_freq.py`**

This function reads each month from the stage5-tier realized DA cache, maps constraint_ids to branch_names via the bridge, and returns `{month: set(branch_name)}`.

```python
def build_monthly_binding_sets(
    auction_month: str,
    period_type: str,
    cutoff_month: str,
) -> dict[str, set[str]]:
    """Build monthly binding sets: {month: set(branch_name that bound)}.

    Parameters
    ----------
    auction_month : str
        Planning year (e.g., "2024-06"). Used to select bridge table partition.
    period_type : str
        Quarter round (e.g., "aq1"). Used to select bridge table partition.
    cutoff_month : str
        Strict upper bound for months to include (e.g., "2024-04").
        Only months < cutoff_month are used.

    Returns
    -------
    dict mapping month string to set of branch_names that were binding.
    """
    cache_path = _BF_CACHE_DIR / f"binding_sets_{auction_month}_{period_type}.json"
    # No caching for now — computation is fast (~2 seconds for 80 months)

    bridge = _load_bridge(auction_month, period_type)

    binding_sets: dict[str, set[str]] = {}
    onpeak_files = sorted(REALIZED_DA_CACHE.glob("*.parquet"))

    for f in onpeak_files:
        # Skip offpeak files
        if "_offpeak" in f.stem:
            continue
        month = f.stem
        if month >= cutoff_month:
            continue

        da = pl.read_parquet(str(f)).filter(pl.col("realized_sp") > 0)
        if len(da) == 0:
            binding_sets[month] = set()
            continue

        mapped = da.join(bridge, on="constraint_id", how="inner")
        binding_sets[month] = set(mapped["branch_name"].to_list())

    return binding_sets
```

- [ ] **Step 4: Verify binding sets build correctly**

```bash
python -c "
from ml.binding_freq import build_monthly_binding_sets
bs = build_monthly_binding_sets('2024-06', 'aq1', '2024-04')
print(f'Months loaded: {len(bs)}')
print(f'Range: {sorted(bs.keys())[0]} .. {sorted(bs.keys())[-1]}')
m = '2024-03'
print(f'{m}: {len(bs.get(m, set()))} binding branches')
"
```
Expected: ~58 months (2019-06 through 2024-03), ~80-120 binding branches per month.

---

### Task 3: Compute Binding Frequency Vectors

- [ ] **Step 5: Add `compute_binding_freq()` to `ml/binding_freq.py`**

```python
def compute_binding_freq(
    branch_names: list[str],
    binding_sets: dict[str, set[str]],
    cutoff_month: str,
    window: int,
) -> np.ndarray:
    """Compute binding frequency for a list of branch_names.

    Parameters
    ----------
    branch_names : list[str]
        Constraint identifiers to compute frequency for.
    binding_sets : dict[str, set[str]]
        Output of build_monthly_binding_sets().
    cutoff_month : str
        Strict upper bound (same as used to build binding_sets).
    window : int
        Number of prior months to look back.

    Returns
    -------
    np.ndarray of shape (len(branch_names),) with values in [0, 1].
    """
    available = sorted(m for m in binding_sets.keys() if m < cutoff_month)
    lookback = available[-window:] if len(available) >= window else available
    n = len(lookback)
    if n == 0:
        return np.zeros(len(branch_names), dtype=np.float64)

    freq = np.zeros(len(branch_names), dtype=np.float64)
    for m in lookback:
        s = binding_sets.get(m, set())
        for i, bn in enumerate(branch_names):
            if bn in s:
                freq[i] += 1
    return freq / n


def enrich_with_binding_freq(
    df: pl.DataFrame,
    auction_month: str,
    period_type: str,
    windows: list[int] | None = None,
) -> pl.DataFrame:
    """Add binding frequency columns to a V6.1 DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        V6.1 data with 'branch_name' column.
    auction_month : str
        Planning year (e.g., "2024-06").
    period_type : str
        Quarter round (e.g., "aq1").
    windows : list[int]
        Lookback windows. Default: [1, 3, 6, 12, 24].

    Returns
    -------
    pl.DataFrame with bf_1, bf_3, bf_6, bf_12, bf_24 columns added.
    """
    if windows is None:
        windows = list(BF_WINDOWS)

    # Cutoff: annual auction submitted ~April of planning year
    py = int(auction_month.split("-")[0])
    cutoff = f"{py}-04"

    binding_sets = build_monthly_binding_sets(auction_month, period_type, cutoff)
    branch_names = df["branch_name"].to_list()

    for w in windows:
        col_name = f"bf_{w}"
        freq = compute_binding_freq(branch_names, binding_sets, cutoff, w)
        df = df.with_columns(pl.Series(col_name, freq))

    return df
```

- [ ] **Step 6: Verify enrichment on sample data**

```bash
python -c "
import polars as pl
from ml.binding_freq import enrich_with_binding_freq

df = pl.read_parquet('/opt/data/xyz-dataset/signal_data/miso/constraints/Signal.MISO.SPICE_ANNUAL_V6.1/2024-06/aq1/onpeak/')
df = enrich_with_binding_freq(df, '2024-06', 'aq1')
print(df.select(['branch_name', 'bf_1', 'bf_3', 'bf_6', 'bf_12', 'bf_24']).describe())
for w in [1, 3, 6, 12, 24]:
    nonzero = (df[f'bf_{w}'] > 0).sum()
    print(f'bf_{w}: {nonzero}/{len(df)} ({100*nonzero/len(df):.1f}%) nonzero, mean={df[f\"bf_{w}\"].mean():.4f}')
"
```
Expected: bf_12 should have ~66% nonzero, mean ~0.23.

- [ ] **Step 7: Commit core module**

```bash
git add ml/binding_freq.py
git commit -m "feat: add binding frequency module for annual constraints"
```

---

## Chunk 2: Integration and Feature Configuration

### Task 4: Add BF Features to Config

**Files:**
- Modify: `ml/config.py`

- [ ] **Step 8: Add BF feature definitions to `ml/config.py`**

Add after the existing feature set definitions (after line ~76):

```python
# -- Binding frequency features (from realized DA monthly data) --
_BF_FEATURES: list[str] = [
    "bf_1", "bf_3", "bf_6", "bf_12", "bf_24",
]
_BF_MONOTONE: list[int] = [1, 1, 1, 1, 1]  # higher freq = more likely to bind

# -- V8 feature sets: V6.1 + binding frequency --
SET_V8_FEATURES = list(_V61_FEATURES) + ["da_rank_value"] + _BF_FEATURES  # 12 features
SET_V8_MONOTONE = list(_V61_MONOTONE) + [-1] + _BF_MONOTONE

SET_V8_LEAN_FEATURES = ["shadow_price_da", "da_rank_value"] + _BF_FEATURES  # 7 features
SET_V8_LEAN_MONOTONE = [1, -1] + _BF_MONOTONE

SET_V8_BF_ONLY_FEATURES = _BF_FEATURES  # 5 features — pure BF, no V6.1
SET_V8_BF_ONLY_MONOTONE = list(_BF_MONOTONE)
```

- [ ] **Step 9: Commit config changes**

```bash
git add ml/config.py
git commit -m "feat: add v8 binding frequency feature set definitions"
```

---

### Task 5: Integrate BF Enrichment into Data Loader

**Files:**
- Modify: `ml/data_loader.py`

The data loader already enriches V6.1 with spice6 density features. We add BF enrichment as an additional step. The enriched cache must be invalidated (new cache key) since the schema changes.

- [ ] **Step 10: Add BF enrichment to `load_v61_enriched()`**

Modify `ml/data_loader.py` to optionally enrich with binding frequency. The key change: add a `with_bf` parameter and use a separate cache directory for BF-enriched data to avoid invalidating the existing cache.

```python
# Add import at top of data_loader.py:
from ml.binding_freq import enrich_with_binding_freq

# Add new cache dir:
_BF_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "enriched_bf"

# Add new function (don't modify load_v61_enriched — keep it backward compatible):
def load_v61_enriched_bf(planning_year: str, aq_round: str) -> pl.DataFrame:
    """Load V6.1 data enriched with spice6 density AND binding frequency features.

    Caches separately from non-BF enriched data.
    """
    cache_path = _BF_CACHE_DIR / f"{planning_year}_{aq_round}.parquet"
    if cache_path.exists():
        df = pl.read_parquet(str(cache_path))
        print(f"[data_loader] loaded from BF cache: {cache_path.name}")
        return df

    # Start from existing enriched data (has spice6 features)
    df = load_v61_enriched(planning_year, aq_round)

    # Add binding frequency features
    df = enrich_with_binding_freq(df, planning_year, aq_round)
    n_bf = (df["bf_12"] > 0).sum()
    print(f"[data_loader] bf enrichment: {n_bf}/{len(df)} with bf_12 > 0")

    _BF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(cache_path))
    print(f"[data_loader] cached to {cache_path.name}")
    return df
```

- [ ] **Step 11: Verify BF-enriched loading**

```bash
python -c "
from ml.data_loader import load_v61_enriched_bf
df = load_v61_enriched_bf('2024-06', 'aq1')
print(f'Shape: {df.shape}')
print(f'BF columns: {[c for c in df.columns if c.startswith(\"bf_\")]}'  )
print(f'bf_12 nonzero: {(df[\"bf_12\"] > 0).sum()}/{len(df)}')
"
```

- [ ] **Step 12: Commit data loader changes**

```bash
git add ml/data_loader.py
git commit -m "feat: add BF-enriched data loader with separate cache"
```

---

## Chunk 3: Experiment Script and Evaluation

### Task 6: Create V8 Experiment Script

**Files:**
- Create: `scripts/run_v8_binding_freq.py`

This script runs multiple v8 variants to answer:
1. Do BF features improve over v7d (current best ML)?
2. Which feature set works best (full v8, lean, bf-only)?
3. Does blending v8 + v0b formula beat blend_v7d_a70?
4. Holdout validation on 2025 data.

- [ ] **Step 13: Create experiment script**

The script should follow the same pattern as `scripts/run_v7_ml_and_blending.py`:
- Load all test groups with BF-enriched data
- Train/evaluate each variant on 12 dev groups (2022-2024)
- Run best variant on 4 holdout groups (2025)
- Compute blends with v0b formula
- Save metrics to registry
- Print comparison tables

**Variants to test:**

| Version | Features | Count | Hypothesis |
|---------|----------|-------|------------|
| v8a | v7d features + bf_1,3,6,12,24 | 12 | Full: all V6.1 + da_rank + all BF windows |
| v8b | shadow_price_da + da_rank_value + bf_1,3,6,12,24 | 7 | Lean: only historical + BF (drop SPICE predictions) |
| v8c | bf_1,3,6,12,24 only | 5 | Pure BF: how good is BF alone? |
| v8d | shadow_price_da + bf_12 | 2 | Minimal: best historical + best BF window |
| blend_v8a_a70 | 0.70 * v8a + 0.30 * v0b | - | Blend best v8 with formula |

Key implementation notes:
- Use `load_v61_enriched_bf()` instead of `load_v61_enriched()` for data loading
- The training pipeline (`train_for_year`) needs to use the BF-enriched data — you may need a parallel `train_for_year_bf()` or pass a loader function
- The cutoff logic is already handled inside `enrich_with_binding_freq()` — no leakage risk as long as you call it correctly
- Compare all v8 variants against v0b AND blend_v7d_a70 baselines

**Critical: the pipeline's `train_for_year()` currently calls `load_v61_enriched()`. For v8, we need it to call `load_v61_enriched_bf()` instead.** The cleanest approach: pass a `loader` function parameter, or just duplicate the training logic inline in the experiment script (as v7 scripts do).

- [ ] **Step 14: Run the experiment script**

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-annual-signal
python scripts/run_v8_binding_freq.py
```

Expected runtime: ~5-10 minutes (BF computation adds ~2s per group, 28 groups total).

- [ ] **Step 15: Save results to registry**

The script should save to:
- `registry/v8a/metrics.json`
- `registry/v8b/metrics.json`
- `registry/v8c/metrics.json`
- `registry/v8d/metrics.json`
- `registry/blend_v8a_a70/metrics.json` (or whichever v8 variant blends best)

- [ ] **Step 16: Run holdout evaluation for best v8 variant**

Train on 2019-2024, evaluate on 2025/aq1-aq4. Save to `registry/v8X_holdout/metrics.json`.

- [ ] **Step 17: Commit experiment and results**

```bash
git add scripts/run_v8_binding_freq.py registry/v8*/
git commit -m "feat: v8 binding frequency experiments — dev and holdout results"
```

---

## Chunk 4: Documentation and Analysis

### Task 7: Update Documentation

- [ ] **Step 18: Update `reports/v7_consolidated_findings.md`**

Add a new section "## V8: Binding Frequency Features" with:
- Feature descriptions and correlation analysis
- Dev eval comparison table (v0b, blend_v7d_a70, v8 variants)
- Holdout comparison table
- Feature importance analysis (has BF displaced shadow_price_da?)
- Conclusion: does BF justify the added complexity?

- [ ] **Step 19: Update `mem.md`**

Add v8 results to the running performance table and update the recommendation.

- [ ] **Step 20: Commit documentation**

```bash
git add reports/ mem.md
git commit -m "docs: v8 binding frequency results and analysis"
```

---

## Verification Checklist

Before declaring v8 complete:

- [ ] No temporal leakage: binding_freq uses months strictly `< YYYY-04` for planning year YYYY-06
- [ ] Bridge table is partition-filtered (not using all partitions which causes fan-out)
- [ ] BF features are cached (not recomputed on every run)
- [ ] All v8 variants compared against BOTH v0b formula AND blend_v7d_a70
- [ ] Holdout results (2025) confirm dev eval findings
- [ ] Feature importance analysis shows whether BF displaces or complements shadow_price_da
- [ ] Memory usage reported at each stage (CLAUDE.md requirement)
- [ ] Registry metrics saved in standard format
- [ ] All scripts run within 40 GiB memory budget

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Bridge table fan-out inflates BF | False positives in binding | Partition-filter bridge table per (auction_month, period_type) |
| BF coverage only 66% at bf_12 | 34% of constraints get bf=0 | bf=0 is informative (never bound = likely won't bind); also include bf_24 (80% coverage) |
| Temporal leakage via cutoff | Inflated results | Cutoff = `YYYY-04` strictly; verified in preliminary analysis |
| Stage5-tier cache not up to date | Missing recent months | Check cache range; fetch missing months with Ray if needed |
| BF just duplicates shadow_price_da | No incremental value | Preliminary Spearman analysis shows +33% improvement; bf captures recent patterns that 60-month historical average misses |
| Memory pressure from 81 parquet reads | OOM risk | Each file is ~10KB; total < 1MB. Bridge table scan is the expensive part (~200MB); cached after first load. |
