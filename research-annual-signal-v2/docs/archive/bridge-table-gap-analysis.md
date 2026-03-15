# Bridge Table Mapping Gap Analysis

> **HISTORICAL DOCUMENT — V4.4 ERA**
>
> This analysis was conducted when V4.4 was the planned universe. V4.4 has since been **abandoned** (see `implementer-guide.md` §3). However, the core findings about the DA-to-model mapping gap remain valid for ANY universe built on SPICE bridge table mapping:
>
> - **Stage 1 loss** (no bridge entry) is structural and applies to all universes: 1-3% for 2021-2024, 26.3% for 2025-06 (outlier).
> - **Stage 2 loss** (bridge hit but not in universe) is universe-dependent. The numbers in this doc are for V4.4's 1,227 constraints. The v2 universe (~2,400-3,900 per quarter) will have different Stage 2 numbers.
> - The three naming systems (DA long-format, SPICE abbreviated, bridge) and 0% direct match still apply.
> - The bridge table filtering rules (4-column filter) still apply.
>
> **Do not import the V4.4-specific SP coverage ceilings (~77%) or phase guidance from §7 into the v2 design.** The v2 universe has different coverage characteristics. See `implementer-guide.md` §5 for v2-specific numbers.

**Date**: 2026-03-11
**Context**: MISO Annual FTR signal — ground truth mapping from realized DA shadow prices (originally analyzed for V4.4 universe)
**Status**: Historical reference — core mapping mechanics still valid, V4.4-specific numbers superseded by v2 analysis

---

## 1. Problem Statement

To evaluate our ranking model, we need ground truth: which V4.4 constraints actually bound in the DA market, and how much? The mapping chain is:

```
DA shadow_price (per constraint_id)
   → MISO_SPICE_CONSTRAINT_INFO bridge table (constraint_id → branch_name)
   → V4.4 equipment column (branch_name = equipment)
   → ground truth label
```

The concern: **a significant fraction of DA binding shadow price comes from constraint_ids that have NO entry in the bridge table**, making that binding activity invisible to any V4.4-based model.

---

## 2. Three Naming Systems

MISO uses three distinct constraint naming conventions. They do **not** match each other:

| System | Format | Example |
|--------|--------|---------|
| **DA (realized market)** | Long human-readable with zone suffix | `ADAMS_I ADAMSHAYWA16_1 1 (LN/ALTW/ALTW)` |
| **V4.4 / SPICE (model)** | Short abbreviated | `08CLO08STI13_1 1` |
| **Bridge table** | Same as V4.4 | `LEONIPLYMO13_1 1` |

**Direct match between DA `branch_name` and V4.4 `equipment`: 0 out of 423 (0.0%).** The bridge table is the only mapping path.

### Verification code

```python
from pbase.analysis.tools.all_positions import MisoApTools
import polars as pl

aptools = MisoApTools()
# Load DA
da = aptools.tools.get_da_shadow_by_peaktype(
    st='2025-07-01', et_ex='2025-08-01', peak_type='onpeak'
)
# DA branch_name examples:
#   'ESTBMRK ESTBMBISMA11_1 1 (LN/MDU/WAUE)'
#   'MURPHYCR MURPHHAYWA16_1 1 (LN/SMP/ALTW)'

# V4.4 equipment examples:
#   '0304           1'
#   '08CLO08STI13_1 1'

da_branches = set(da['branch_name'].dropna().unique())
v44_equips = set(v44['equipment'].to_list())
direct_match = da_branches & v44_equips
# Result: 0 matches out of 423 DA branch_names
```

---

## 3. Quantified Gap: Two Planning Years

Investigation ran on two quarters to assess stability.

### Mapping chain results

```python
# Bridge table: MISO_SPICE_CONSTRAINT_INFO.parquet
# Filter: auction_type='annual', auction_month=PY, period_type='aq1', class_type='onpeak'
bridge = (
    pl.scan_parquet(f'{SPICE_DATA}/MISO_SPICE_CONSTRAINT_INFO.parquet')
    .filter(
        (pl.col('auction_type') == 'annual')
        & (pl.col('auction_month') == planning_year)
        & (pl.col('period_type') == 'aq1')
        & (pl.col('class_type') == 'onpeak')
    )
    .select(['constraint_id', 'branch_name'])
    .collect().unique()
)
```

### Full results: 5 planning years, aq1/onpeak

| PY | V4.4 size | DA binding cids | No bridge (cids) | No bridge (SP) | Bridge-not-V4.4 (SP) | **V4.4 captured SP** | **Total lost** |
|---|---|---|---|---|---|---|---|
| **2021-06** | 1,089 | 572 | 21 (3.7%) | 1.5% | 30.4% | **68.1%** | 31.9% |
| **2022-06** | 1,094 | 588 | 32 (5.4%) | 3.2% | 15.8% | **81.0%** | 19.0% |
| **2023-06** | 1,202 | 598 | 43 (7.2%) | 3.1% | 22.0% | **74.9%** | 25.1% |
| **2024-06** | 1,227 | 633 | 22 (3.5%) | 1.4% | 14.5% | **84.1%** | 15.9% |
| **2025-06** | 1,227 | 593 | 227 (38.3%) | 26.3% | 9.6% | **64.0%** | 36.0% |

Detailed breakdown for each year:

| | **2021-06** | **2022-06** | **2023-06** | **2024-06** | **2025-06** |
|---|:---:|:---:|:---:|:---:|:---:|
| Total binding \|SP\| | $1,362,905 | $1,670,463 | $1,218,647 | $991,291 | $1,092,433 |
| Mapped \|SP\| | $1,343,053 | $1,616,221 | $1,181,234 | $977,542 | $804,870 |
| Unmapped \|SP\| | $19,852 | $54,242 | $37,413 | $13,749 | $287,563 |
| V4.4-captured \|SP\| | $928,238 | $1,352,497 | $913,138 | $833,911 | $699,646 |

### Key observations

**Stage 1 (no bridge entry) is normally small but 2025-06 is a severe outlier.** In 2021-2024, only 1.4-3.2% of binding SP is unmapped. In 2025-06, it's 26.3%. The bridge table size is stable (~14,000 cids across all years), so the problem is that 2025 DA binding included many constraint_ids not in the SPICE annual bridge table.

The unmapped 2025 constraint_ids are NOT "new" (all are below the bridge table's max ID of 5,002,513,935). They exist in MISO's constraint universe but were not included in the SPICE annual bridge table for 2025-06. **This is worth flagging to the data team — possible bridge table build issue for 2025-06.**

**Stage 2 (bridge hit but not in V4.4) is the bigger consistent loss.** It accounts for 9.6-30.4% of SP across all years. This is the cost of V4.4's aggressive filtering from ~14,000 bridge constraints down to ~1,200. Phase 3a universe expansion directly targets this gap.

**V4.4 captures 64-84% of total binding SP.** The average across 2021-2024 is ~77%. 2025-06 drags this down due to its Stage 1 outlier.

### Where the loss happens (two stages)

```
Total binding DA SP
  ├── Stage 1: No bridge entry (constraint_id not in MISO_SPICE_CONSTRAINT_INFO)
  │     2021-2024 avg: ~2.3%  |  2025: 26.3% (outlier)
  ├── Stage 2: Bridge hit but branch_name not in V4.4's ~1,200 equipment
  │     2021-2024 avg: ~20.7% |  2025: 9.6%
  └── V4.4-captured
        2021-2024 avg: ~77%   |  2025: 64.0%
```

---

## 4. V4.4 Side: Bridge Coverage of Our Universe

From the model's perspective (V4.4 looking outward), coverage is excellent:

```python
v44_cids = set(v44['constraint_id'].to_list())  # parsed from __index_level_0__
bridge_cids = set(bridge['constraint_id'].to_list())
v44_in_bridge = v44_cids & bridge_cids
# Result: 1,196 / 1,227 = 97.5% of V4.4 cids have bridge entries

v44_equips = set(v44['equipment'].to_list())
bridge_branches = set(bridge['branch_name'].to_list())
equip_match = v44_equips & bridge_branches
# Result: 1,214 / 1,227 = 98.9% of V4.4 equipment names appear in bridge
```

| Metric | Value |
|--------|-------|
| V4.4 constraint_ids with bridge entry | 1,196 / 1,227 (97.5%) |
| V4.4 equipment names in bridge branch_name | 1,214 / 1,227 (98.9%) |

**Interpretation**: Almost all V4.4 constraints CAN receive ground truth labels. The gap is on the DA side — binding DA constraints that never appear in V4.4's universe.

---

## 5. Production pbase Code Comparison

The production codebase uses the **same mapping pattern** and accepts the gap.

### `MisoTools.get_equipment_mapping()` (pbase/analysis/tools/miso.py:608)

```python
@staticmethod
def get_equipment_mapping(
    auction_month_str, auction_round, period_type,
    class_type="onpeak", by="constraint_id",
    cons_info_dir="/opt/data/tmp/pw_data/spice3/prod_f0p_model_{rto}",
):
    # Reads spice3 constraint_info partitioned parquet
    # Filters convention < 10, maps constraint_id → branch_name → equipment
    mapping = cons[cons["convention"] < 10][
        ["constraint_id", "branch_name", "convention", "type"]
    ].rename(columns={"branch_name": "equipment"})
    return mapping
```

### `MisoTools.get_da_sp_by_equipment()` (miso.py:640)

```python
@staticmethod
def get_da_sp_by_equipment(da, mapping, da_sp_col="shadow_price"):
    # Left join — unmapped DA constraint_ids get NaN equipment
    da_w_equ = da_by_consid.merge(mapping, on="constraint_id", how="left")
    da_by_equ = da_w_equ.groupby("equipment")[da_sp_col].sum().reset_index()
    return da_by_equ, da_w_equ
```

**Note**: The spice3 constraint_info path (`/opt/data/tmp/pw_data/spice3/prod_f0p_model_miso/constraint_info/`) only has `period_type=f0` for annual auction months — it does NOT have `aq1/aq2/aq3/aq4`. Only the spice6 bridge table (`MISO_SPICE_CONSTRAINT_INFO.parquet`) has annual period types.

### Alternative mapping: Panorama branch mapping

`get_da_shadow_by_peaktype()` (base.py:13285) has a `case_name` parameter that uses `MisoConstraintsBranchMapping` — a network-model-based mapping independent of SPICE. However:
- It maps DA constraint_id → network model equipment names
- These names come from state estimation, not SPICE's naming convention
- They would not match V4.4's equipment column
- Not a viable bridge bypass for our pipeline

---

## 6. Root Cause Assessment

The gap has two components:

### 6a. No bridge entry (Stage 1)

DA binds on constraint_ids that SPICE did not include in its annual bridge table. Causes:
- **Constraint universe refresh timing**: The bridge table is built when the SPICE model runs (before the delivery quarter). New constraints added to MISO's network after that point won't appear.
- **SPICE filtering**: SPICE may exclude constraints below certain thresholds or outside its modeling scope.
- **Year-to-year variability**: 2025-06 has 38% unmapped vs 2024-06 at 3.5% — suggests the 2025 bridge was less complete or 2025 DA had more out-of-universe binding.

### 6b. Bridge hit but not in V4.4 (Stage 2)

The bridge has ~14,000 constraint_ids but V4.4 only has 1,227. V4.4 applies aggressive filtering (shadow_rank cutoff, flow direction). Constraints with bridge entries but excluded from V4.4 are a known reduction step.

---

## 7. Implications for the ML Pipeline

### What this means (general — applies to any SPICE-based universe)

| Aspect | Implication |
|--------|-------------|
| **Ground truth labels** | Constraints with bridge coverage get reliable labels. Coverage depends on universe size. |
| **Model ceiling** | Total binding SP captured depends on universe choice. Stage 1 loss (no bridge entry: 1-3% normally, 26.3% for 2025-06) is the hard floor regardless of universe. |
| **NB metrics** | Must report "universe new binders" separately from "total DA new binders". A constraint that binds but has no bridge entry cannot be a false negative — it was never mappable. |
| **Year-to-year noise** | The variable gap (1.4% to 26.3% unmapped SP) means ground truth quality differs across eval periods. 2025-06 labels are noisier than 2024-06. |

> **V4.4-SPECIFIC (historical)**: The numbers below were for V4.4's 1,227 constraints. The v2 universe (~2,400-3,900 per quarter) will have different Stage 2 loss. See `implementer-guide.md` §5 and §8 for v2-specific guidance.

### What to do about it (general principles)

1. **Accept Stage 1**: The gap is structural and production-accepted. No universe change can fix no-bridge-entry constraints.
2. **Track coverage per eval group**: Report total DA SP, universe-captured SP, and coverage % for each (planning_year, aq_round).
3. **Within-universe metrics: do NOT normalize by total DA SP** — use universe-visible binding SP as the denominator. **However**, the v2 guide introduces `Abs_SP@50` (§10.6) as a deliberate cross-universe metric that DOES normalize by total DA SP — this is correct for comparing models across different universe sizes.
4. **Flag 2025-06 as noisy**: The 38% bridge gap in 2025-06 means ground truth labels for that year miss more binding activity than usual.
5. **99.2% of density constraints have bridge entries** (verified for 2024-06): the expanded v2 universe has excellent bridge coverage.

---

## 8. Appendix: Data Paths and Bridge Table Schema

### Data paths

| Data | Path |
|------|------|
| V4.4 signal | `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V4.4.R1/{PY}/{aq}/onpeak` |
| Bridge table (spice6) | `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_INFO.parquet` |
| Bridge table (spice3, f0 only) | `/opt/data/tmp/pw_data/spice3/prod_f0p_model_miso/constraint_info/` |
| DA shadow prices | Via `MisoApTools().tools.get_da_shadow_by_peaktype()` (requires Ray) |

### Bridge table schema (MISO_SPICE_CONSTRAINT_INFO)

Partition columns: `auction_type`, `auction_month`, `market_round` (=`period_type`), `period_type`, `class_type`, `spice_version`

Key columns for mapping: `constraint_id` (join key to DA), `branch_name` (join key to V4.4 equipment)

### V4.4 constraint_id format

Parsed from `__index_level_0__` column: `"296436|-1|spice"` → constraint_id = `"296436"`

### DA shadow price columns

`constraint_id`, `constraint_name`, `branch_name`, `contingency_description`, `shadow_price`, `monitored_line`, `year`, `month`, `day`

---

## 9. Relevant Skills and Tools

| Skill | How it relates |
|-------|---------------|
| `cross-signal-constraint-matching` | Documents MISO naming conventions (SPICE abbreviated vs DA long-format). Recommends SF correlation for definitive per-constraint matching. |
| `signal-loading` | Loading V4.4 signal data via `ConstraintsSignal` / `ShiftFactorSignal` |
| `pbase-knowledge` | `MisoTools.get_equipment_mapping()`, `MisoConstraintsBranchMapping`, bridge table loaders |
| `parallel-with-ray` | DA shadow price loading requires Ray |
