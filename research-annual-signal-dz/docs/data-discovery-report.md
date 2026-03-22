# DYZER Annual Signal — Data Discovery Report

**Date**: 2026-03-22
**Source repo**: `research-annual-signal-dz`
**Based on**: live data exploration of Dayzer outputs, mapping tables, SF, and DA overlap

---

## 1. Raw Data

### Location
```
/opt/data/xyz-dayzer/dayzer_output/{scenario_name}/RESULTS_CONSTRAINTS.parquet/year={Y}/month={M}
```

### Schema (9 columns)

| Column | Type | Description |
|--------|------|-------------|
| `scenario_name` | String | e.g., `miso_auc24annual_base` |
| `result_date` | Datetime | Simulation date |
| `result_hour` | Int8 | Hour 1-24 |
| `constraint_id` | Int64 | Dayzer constraint ID |
| `flows` | Float64 | Flow on the constraint (MW) |
| `shadow_price` | Float64 | Hourly shadow price ($/MWh) |
| `min_flow_limit` | Float64 | Lower flow limit |
| `max_flow_limit` | Float64 | Upper flow limit |
| `day` | Int64 | Day of month |

### Scenario naming convention
```
miso_auc{YY}annual_{suffix}_{round}
```
Example: `miso_auc24annual_base_r3` = PY 2024, base scenario, round 3

### Coverage

| PY | Round | Suffixes | Months | Rows per suffix (approx) |
|----|:---:|------|:---:|---:|
| 2022 | r3 | base, P1*, P2*, PUG1, PUG2, PZD1, PZD2 | 12 | ~8.9M |
| 2023 | r3 | base, P1, P2, PUG1, PUG2, PZD1, PZD2 | 12 | ~8.9M |
| 2024 | r3 | base, P1, P2, PUG1, PUG2, PZD1, PZD2 | 12 | ~8.9M |
| 2025 | r3 | base, P1, P2, PUG1, PUG2, PZD1, PZD2 | 11* | ~8.1M |

*PY 2022 P1/P2: incomplete (missing some months). PY 2025: missing 2026-05.

Only round r3 is available for the `miso_auc` series. Earlier `annual_base_at_*` and `miso_annual_base_at_*` scenarios also exist (PY 2020-2025) but use a different naming convention.

### Data size

Per PY/suffix/12 months: **~8.9M rows**. Per month: 630K-800K rows, 5,000-7,000 unique DZ CIDs, ~90-155 hours.

---

## 2. DZ CID to Branch Mapping

### Location
```
/opt/data/xyz-dataset/dayzer_modeling_data/miso/constraint_mapping_dz_to_iso.parquet
```

### Schema (11 columns)

| Column | Description | Unique values |
|--------|-------------|:---:|
| `CID` | Dayzer constraint ID | 11,162 |
| `ISOConstraint_dz` | DZ constraint name (DZ namespace) | 14,859 |
| `ISOContingency_dz` | DZ contingency name | 3,051 |
| `_ISOConstraint` | Normalized constraint name | 14,300 |
| `_ISOContingency` | Normalized contingency name | 3,034 |
| `ISO_CID` | Mapped ISO constraint ID | 11,417 |
| `ISOConstraint_iso` | ISO constraint name | 10,382 |
| `ISOContingency_iso` | ISO contingency name | 2,917 |
| `ISOBranch` | **Mapped branch name (DA format)** | 3,944 |
| `auction_year` | Year | 11 |
| `auction_month` | Month | 12 |

### Coverage

| Metric | Value |
|--------|-------|
| Total mapping rows | 1,207,302 |
| Rows with ISOBranch | 897,580 (74.3%) |
| Rows missing ISOBranch | 309,722 (25.7%) |
| Unique mapped DZ CIDs | 9,785 |
| Unique ISOBranches | 3,944 |
| DZ CIDs per branch | min=1, max=1,901, median=1 |

### Branch namespace

**Critical finding**: `ISOBranch` uses the **DA branch_name format**, NOT the SPICE internal format.

```
DZ ISOBranch: "ABBOTT ABBOTTRAER16_1 1 (LN/ALTW/ALTW)"
DA branch_name: "ABBOTT ABBOTTRAER16_1 1 (LN/ALTW/ALTW)"
SPICE branch:   "ABBOTTRAER16_1 1"
```

DZ branches and DA branches are in the **same namespace**. No bridge translation needed — direct string match.

### DA overlap

In one sampled month (2024-10/offpeak):
- DA binding branches: 238
- DZ mapped branches: 3,944
- **Overlap: 208 (87% of DA branches)**

This means 87% of DA binding value can be attributed to branches we can model with DZ features.

---

## 3. Shift Factors

### Location
```
/opt/data/shared/sf/dz/miso/avg5_m4/{YYYY-MM}/{YYYY-MM}.parquet
```

### Structure

- Rows: pnodes (~2,759-2,764)
- Columns: DZ CIDs (~27,000-27,400 per month)
- Coverage: 2016-01 through 2026-04 (monthly)
- Partitioned by delivery month (not by PY/quarter like SPICE)

### PY 2025 gap

Missing `2026-05` SF (same as noted in README).

---

## 4. Ground Truth Feasibility

GT is branch-level realized DA shadow price — same as SPICE pipeline.

Since DZ `ISOBranch` = DA `branch_name` format:
1. Load DA shadow prices for the delivery months
2. DA `branch_name` matches DZ `ISOBranch` directly
3. Sum DA SP per branch = GT

**No bridge translation needed.** This is simpler than the SPICE pipeline where we needed CID→branch bridge + supplement key matching.

The SPICE supplement key matching logic (`key1+key3` for XF, `key2+key3` for LN) is not needed here because the DZ mapping already provides DA-format branch names.

---

## 5. Feature Candidates

### From DZ raw data (per CID, aggregate across hours/days/months)

| Feature | Derivation | Analogous SPICE feature |
|---------|-----------|------------------------|
| `binding_count_{suffix}` | Count of hours where `\|shadow_price\| > 0` | density right-tail |
| `binding_fraction_{suffix}` | `binding_count / total_hours` | density bin values |
| `sp_sum_{suffix}` | `sum(\|shadow_price\|)` | — |
| `sp_mean_{suffix}` | `mean(\|shadow_price\|)` where binding | — |
| `sp_max_{suffix}` | `max(\|shadow_price\|)` | — |
| `utilization_mean` | `mean(\|flows\| / max_flow_limit)` | — |
| `utilization_max` | `max(\|flows\| / max_flow_limit)` | — |
| `onpeak_sp_sum` | SP sum during peak hours | class-specific SP |
| `offpeak_sp_sum` | SP sum during off-peak hours | class-specific SP |
| `consensus_count` | Number of suffixes predicting binding | — |
| `cross_suffix_sp_var` | Variance of SP across suffixes | — |

### After branch collapse (multiple DZ CIDs → one branch)

- `max` of each CID-level feature
- `mean` of each CID-level feature
- `count_dz_cids` (how many DZ CIDs map to this branch)
- `count_binding_dz_cids` (how many had any binding)

### DA history features (reuse from SPICE)

Since branches are in the same namespace as DA:
- `da_rank_value` — rank of cumulative historical DA SP
- `bf_12` / `bfo_12` — binding frequency windows
- `shadow_price_da` — cumulative DA SP

These can be computed using the same `history_features.py` logic, with DZ branch names matching DA branch names directly.

---

## 6. Publication

### What to publish
- DZ CIDs (not DA CIDs, not SPICE CIDs)
- DZ SF matrix (pnode × DZ CID)

### Dedup
Yes — up to 1,901 DZ CIDs per branch. Need SF-based dedup + branch cap similar to SPICE.

### Flow direction
Not available from a density signal score (unlike SPICE). Derive from the sign distribution of `shadow_price` across hours for each DZ CID.

---

## 7. Key Differences from SPICE Pipeline

| Aspect | SPICE | DYZER |
|--------|-------|-------|
| Universe source | Density distribution | RESULTS_CONSTRAINTS |
| Universe filter | `right_tail_max >= threshold` | Has valid ISOBranch mapping |
| Branch namespace | SPICE internal (`FORMAFORMN11_1 1`) | DA format (`ABBOTT ABBOTTRAER16_1 1 (LN/...)`) |
| DA→branch mapping | CID bridge + supplement keys | Direct branch_name match |
| Features | Density bins, limits | Scenario shadow prices, utilization |
| Hourly data | No (pre-aggregated) | Yes (must aggregate) |
| Class derivation | Pre-labeled | Derive from `result_hour` |
| flow_direction | From density signal score | Derive from shadow_price sign |
| SF source | SPICE SF parquet | `/opt/data/shared/sf/dz/miso/avg5_m4/` |
| SF size | ~14K constraints | ~27K constraints |

---

## 8. Reusable from SPICE Pipeline

| Module | Reusable? | Notes |
|--------|:---------:|-------|
| `evaluate.py` | Yes | Metrics are RTO/signal-agnostic |
| `train.py` | Yes | LambdaRank training is generic |
| `registry.py` | Yes | Save/load JSON |
| `ground_truth.py` | Partial | GT logic same, but no bridge needed (direct branch match) |
| `history_features.py` | Partial | BF/da_rank same, but bridge calls need DZ branch namespace |
| `config.py` | No | All paths, PYs, splits different |
| `bridge.py` | No | DZ mapping is different structure |
| `data_loader.py` | No | DZ raw data format is completely different |
| `signal_publisher.py` | Partial | Dedup/tier logic reusable, metadata/SF loading different |

---

## 9. Implementation Sequence

1. `ml/config.py` — paths, PYs (2022-2025), suffixes, peak-hour rules, eval splits
2. `ml/dz_bridge.py` — load mapping, normalize branches, diagnostics
3. `ml/dz_data_loader.py` — load RESULTS_CONSTRAINTS, aggregate to CID-level features, cache
4. `ml/ground_truth.py` — DA SP per branch (direct match, no bridge)
5. `ml/history_features.py` — BF, da_rank (adapt to DZ branch namespace)
6. `ml/features.py` — branch collapse, join DZ features + GT + history
7. `ml/scoring.py` — baseline formula scorer
8. `ml/dz_signal_publisher.py` — branch→DZ CID expansion, SF, dedup, tiers
