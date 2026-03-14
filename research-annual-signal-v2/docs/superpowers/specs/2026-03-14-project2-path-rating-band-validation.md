# Project 2: Path Rating + Annual Band Validation

**Date**: 2026-03-14
**Status**: Draft
**Depends on**: Project 1 (constraint-level signal contract, not necessarily full publication)

---

## 1. Goal

For any given path (source_pnode, sink_pnode), use the annual constraint signal to
compute a "rating" that measures how exposed the path is to highly-ranked constraints.
Then verify: for paths with high ratings (e.g., paths touching tier-0 constraints with
SF > 0.1), do the conclusions from `research-annual-band` (baseline accuracy, band
coverage, band width) still hold?

## 2. Path Rating

### 2.1 Definition

A path's rating measures its exposure to dangerous/high-value annual constraints:

```python
# For path (source, sink):
path_sf = sf[sink_pnode] - sf[source_pnode]  # per-constraint shift factor
path_exposure = path_sf * shadow_sign         # signed exposure

# Rating: max exposure against tier-0 constraints
path_rating = max(path_exposure[tier_0_constraints])

# Or: weighted sum across tiers
path_rating = sum(
    tier_weight[tier] * path_exposure[constraint]
    for constraint in constraints
    if abs(path_exposure[constraint]) > 0.05  # SF threshold
)
```

The exact rating formula is a design choice. Options:
- **Max SF against tier-0**: simple, interpretable. A path with any tier-0 constraint SF > 0.1 is "highly rated."
- **Weighted exposure sum**: tier0 × 5 + tier1 × 3 + tier2 × 1 (matches pmodel's `scale_exposure`)
- **Count of tier-0 exposures above threshold**: how many tier-0 constraints does this path touch?

### 2.2 Nodal Replacement

**Required for production-matching results.** Before computing path SFs:

```python
from pbase.data.dataset.miso import MisoNodalReplacement

# Load replacement mapping
node_map = MisoNodalReplacement.load_data(auction_month)
# Filter: effective_start_date <= auction_month, not terminated/expired
# Resolve chains: A→B→C becomes A→C
node_map = simplify_node_replacement(node_map)

# Apply to path nodes
source_replaced = node_map.get(source_pnode, source_pnode)
sink_replaced = node_map.get(sink_pnode, sink_pnode)

# Then compute SF using replaced nodes
path_sf = sf[sink_replaced] - sf[source_replaced]
```

If we skip this, paths referencing retired/renamed nodes will have missing SF values
and the rating will be wrong. This is the same replacement pmodel applies at
`base.py:2311` during trade generation.

### 2.3 SF Loading

```python
from pbase.data.dataset.signal.general import ConstraintsSignal, ShiftFactorSignal

cstrs = ConstraintsSignal(
    rto="miso", signal_name="TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1",
    period_type="aq1", class_type="onpeak",
).load_data(auction_month=pd.Timestamp("2025-06"))

sf = ShiftFactorSignal(
    rto="miso", signal_name="TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1",
    period_type="aq1", class_type="onpeak",
).load_data(auction_month=pd.Timestamp("2025-06"))

# sf.index = pnode_ids
# sf.columns = constraint index (same as cstrs.index)
# cstrs["tier"] = integer 0-4
```

## 3. Band Validation

### 3.1 What We're Checking

`research-annual-band` established baseline and banding parameters (coverage, width)
for annual FTR paths. The question: **do these parameters hold for paths that our
annual signal rates as high-exposure?**

Specifically:
- For paths with tier-0 constraint SF > 0.1: is baseline accuracy still valid?
- For paths with high weighted exposure: are band widths appropriate?
- Do high-rated paths have different P&L distributions than low-rated paths?

### 3.2 Segmentation

Rate all paths in the annual path pool, then segment:

| Segment | Definition |
|---------|-----------|
| **High-rated** | max tier-0 SF > 0.1 |
| **Medium-rated** | max tier-0 SF ∈ (0.05, 0.1] OR max tier-1 SF > 0.1 |
| **Low-rated** | max tier-0 SF ≤ 0.05 AND max tier-1 SF ≤ 0.1 |
| **Unrated** | no constraint exposure (path nodes not in SF) |

### 3.3 Metrics Per Segment

For each segment, compute from `research-annual-band` data:
- **Baseline accuracy**: mean |residual| / mcp
- **Band coverage**: what % of realized outcomes fall within the band?
- **Band width**: mean (upper - lower) / mcp
- **P&L distribution**: mean, median, P10, P90 of profit per MW

### 3.4 Research-Annual-Band Data

Location: `/home/xyz/workspace/research-qianli-v2/research-annual-band/`

This contains:
- Band parameters per path/version
- Baseline predictions
- Historical coverage analysis

We need to join our path ratings to this data by (source_id, sink_id) after nodal
replacement.

## 4. Pipeline

```
Step 1: Load annual signal (constraints + SF)
  └─ From Project 1 output or inline computation

Step 2: Load path pool
  └─ From MisoPathPoolV6Loader or research-annual-band paths

Step 3: Apply nodal replacement to path nodes
  └─ MisoNodalReplacement → simplify chains → replace source/sink

Step 4: Compute path ratings
  └─ For each path: sf[replaced_sink] - sf[replaced_source] per constraint
  └─ Aggregate by tier weighting

Step 5: Segment paths by rating
  └─ High / Medium / Low / Unrated

Step 6: Join to research-annual-band data
  └─ By (source_id, sink_id) after replacement
  └─ Pull baseline, band, P&L data per path

Step 7: Compute per-segment statistics
  └─ Baseline accuracy, band coverage, band width, P&L distribution

Step 8: Compare segments
  └─ Do high-rated paths have better/worse/different banding performance?
```

## 5. Verification

### 5.1 Path Rating Verification
- For a sample of paths, manually compute SF exposure and verify rating matches
- Cross-check: paths with known high DA congestion should rate high
- Check nodal replacement: compare path count before/after replacement

### 5.2 Band Validation Verification
- Reproduce a known result from research-annual-band on the full path set
- Then split by rating segment and check if the aggregate numbers decompose correctly
- Spot-check: the union of segments should equal the full set

### Reference data:
- research-annual-band versions at `/home/xyz/workspace/research-qianli-v2/research-annual-band/versions/`
- Path pool at pmodel/pbase path pool loaders
- Nodal replacement via `MisoNodalReplacement`

## 6. Implementation

| File | Description |
|------|-------------|
| `scripts/rate_paths.py` | NEW — compute path ratings from annual signal |
| `scripts/validate_bands_by_rating.py` | NEW — segment paths, compare band parameters |
| `ml/path_rating.py` | NEW — path rating computation + nodal replacement |

### Dependencies
- `pbase.data.dataset.signal.general.ConstraintsSignal` (load)
- `pbase.data.dataset.signal.general.ShiftFactorSignal` (load)
- `pbase.data.dataset.miso.MisoNodalReplacement` (nodal replacement)
- `pbase.utils.tools.Tools.simplify_node_replacement` (chain resolution)
- `research-annual-band` data (band parameters, baseline, P&L)

## 7. Dependency on Project 1

**For research**: Project 2 can compute scores inline from the existing research
pipeline (branch scores → constraint expansion → inline SF) without needing the full
published signal. This allows rapid iteration.

**For production-matching results**: Project 2 should consume Project 1's published
signal artifact to ensure constraint selection, tier assignment, and SF are identical
to what pmodel would use.

**Recommended approach**: Start with inline computation for research validation, then
switch to reading the published signal once Project 1 is done.
