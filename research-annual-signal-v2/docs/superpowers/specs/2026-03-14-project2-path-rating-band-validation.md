# Project 2: Path Rating + Annual Band Validation

**Date**: 2026-03-14
**Status**: Draft (rev 2 — review fixes applied)
**Depends on**: Project 1 (constraint-level signal contract)

---

## 1. Goal

For any given path, compute a canonical rating from the annual constraint signal, then
validate whether `research-annual-band` conclusions (baseline, band coverage, width)
hold across rating segments.

## 2. Canonical Path Rating

### 2.1 Definition (frozen)

**Step 1: Nodal replacement**

```python
from pbase.data.dataset.replacement import MisoNodalReplacement

nr = MisoNodalReplacement()
node_map = nr.load_data(auction_month)
# Filter effective_start_date <= auction_month, drop terminated/expired
# Resolve chains via simplify_node_replacement
source_replaced = node_map.get(source_pnode, source_pnode)
sink_replaced = node_map.get(sink_pnode, sink_pnode)
```

**Step 2: Per-constraint sign-aligned exposure**

```python
# sf: pnode × constraint matrix from published signal
path_sf = sf.loc[sink_replaced] - sf.loc[source_replaced]  # per-constraint
aligned_exposure = shadow_sign * path_sf  # sign-adjusted
```

**Step 3: Continuous path rating score**

```python
TIER_WEIGHTS = {0: 1.0, 1: 0.5, 2: 0.0, 3: 0.0, 4: 0.0}

path_rating_score = sum(
    max(0, aligned_exposure[c]) * TIER_WEIGHTS[tier[c]] * normalized_constraint_score[c]
    for c in constraints
)
```

Where `normalized_constraint_score` is the published `rank` column min-max normalized
to [0, 1] within the signal.

**Step 4: Binary segment**

```python
high_rated = any(
    aligned_exposure[c] > 0.1
    for c in constraints
    if tier[c] == 0
)
```

A path is "highly rated" if it has **sign-aligned** exposure > 0.1 against ANY tier-0
constraint. This is direction-aware and uses the published tier, not raw |SF|.

### 2.2 Why This Definition

- **Sign-aligned**: respects path direction (source→sink vs sink→source are different)
- **Tier-weighted**: uses the published annual ranking, not generic SF magnitude
- **Threshold 0.1**: matches pmodel's `abs_sf_thres` for path generation
- **No mixing**: one canonical metric (sign-adjusted exposure), not raw SF or |SF|

## 3. Join Keys Into Annual-Band Data

### 3.1 Row-Level Join

Annual-band data is path-level. The exact join keys:

```python
join_keys = ["source_id", "sink_id", "class_type", "planning_year"]
```

Confirmed from `research-annual-band/pipeline/pipeline.py:247`:
```python
"match_keys": ["source_id", "sink_id", "class_type", "planning_year"]
```

### 3.2 Partition Keys

`period_type` (aq1-aq4) and `round` (R1/R2/R3) are used to filter/partition the data
BEFORE the matched comparison (`run_v5_bands.py:1548, 1583`). Include them explicitly
in the analysis table:

```python
analysis_keys = ["planning_year", "round", "period_type", "class_type", "source_id", "sink_id"]
```

### 3.3 trade_type

- **NOT needed** for baseline / residual / coverage / width validation
- **Needed** only for clearing-prob / bid-edge analysis (segmented by trade side,
  `v10-annual-band-port-plan.md:35`)
- Exclude for now; add later if clearing-prob analysis is requested

### 3.4 Data Source

Project 2 needs **row-level** baseline/residual/band datasets from annual-band, NOT
the summary `metrics.json` aggregates. These are the per-path DataFrames produced by
the band pipeline.

Location: `/home/xyz/workspace/research-qianli-v2/research-annual-band/versions/bands/`

## 4. Segmentation

Rate all paths, then segment:

| Segment | Definition |
|---------|-----------|
| **High** | `high_rated == True` (any tier-0 aligned_exposure > 0.1) |
| **Medium** | Not high, but any tier-0 aligned_exposure > 0.05 OR any tier-1 aligned_exposure > 0.1 |
| **Low** | All aligned_exposures ≤ 0.05 for tier-0 and ≤ 0.1 for tier-1 |
| **Unrated** | Path nodes not in SF (after nodal replacement) |

## 5. Metrics Per Segment

For each segment, compute from annual-band row-level data:

| Metric | Definition |
|--------|-----------|
| Baseline MAE | mean |mcp - baseline| per segment |
| Baseline directional accuracy | % of paths where sign(residual) = sign(baseline_residual) |
| Band coverage | % of realized mcp within [lower, upper] band |
| Band width | mean (upper - lower) / |mcp| |
| P&L mean, median, P10, P90 | distribution of profit_per_MW |

Compare across segments: do high-rated paths have different (better? worse?)
banding performance than low-rated paths?

## 6. Pipeline

```
Step 1: Load annual signal (constraints + SF)
  └─ From Project 1 published signal, or inline for research

Step 2: Load path pool
  └─ From annual-band row-level data or MisoPathPoolV6Loader

Step 3: Apply nodal replacement to path nodes
  └─ from pbase.data.dataset.replacement import MisoNodalReplacement
  └─ Resolve chains, replace source_id / sink_id

Step 4: Compute path ratings
  └─ Per path: aligned_exposure = shadow_sign × (sf[sink] - sf[source])
  └─ Continuous score: tier-weighted sum
  └─ Binary: high_rated = any tier-0 aligned_exposure > 0.1

Step 5: Segment paths

Step 6: Join to annual-band row-level data
  └─ Keys: [source_id, sink_id, class_type, planning_year]
  └─ Partition by: period_type, round

Step 7: Compute per-segment metrics

Step 8: Compare segments
```

## 7. Verification

- For a sample of paths, manually compute SF exposure and verify rating matches
- Cross-check: known high-DA-congestion paths should rate high
- Segment sizes should be reasonable (not degenerate — all high or all low)
- The union of segments should equal the full path set
- Reproduce a known result from annual-band on the full set, then verify it
  decomposes correctly into segment contributions

## 8. Implementation

| File | Description |
|------|-------------|
| `scripts/rate_paths.py` | NEW — compute path ratings |
| `scripts/validate_bands_by_rating.py` | NEW — segment + compare band metrics |
| `ml/path_rating.py` | NEW — rating computation + nodal replacement |

### Dependencies
- `pbase.data.dataset.signal.general.ConstraintsSignal` (load)
- `pbase.data.dataset.signal.general.ShiftFactorSignal` (load)
- `pbase.data.dataset.replacement.MisoNodalReplacement` (nodal replacement)
- `pbase.utils.tools.Tools.simplify_node_replacement` (chain resolution)
- `research-annual-band` row-level data (NOT metrics.json)

## 9. Dependency on Project 1

| Mode | Dependency | Use case |
|------|-----------|----------|
| Research (inline) | None — compute scores inline | Rapid iteration, no publication needed |
| Production-matching | Project 1 published signal | Ensures same constraint universe + tiers |

**Recommended**: Start inline for research, switch to published signal for final validation.
