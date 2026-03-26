# Agents: Errors, Corrections, and Surprising Findings

This file is the durable log for errors and significant corrections discovered during the PJM annual signal port. Every agent must append here immediately upon discovery.

---

## Format

```
### YYYY-MM-DD — <short title>

**What was wrong / surprising:**
<description>

**Correct understanding:**
<description>

**Impact:**
<what downstream work is affected>
```

---

<!-- Entries below -->

### 2026-03-25 — V4.6 rank direction is LOW = BEST (not high = best)

**What was wrong / surprising:**
Initial Explore agent reported V4.6 rank direction as "rank 1.0 = BEST". This is wrong. Verified from actual parquet data for PY 2024-06 R1 onpeak:
- rank ≈ 0.001 → shadow_price_da = 602,672 (NOTTINGHM, most important constraint)
- rank ≈ 1.000 → shadow_price_da = 0.0 (least important)
- Correlation(rank, shadow_price_da) = -0.28

**Correct understanding:**
- **V4.6 rank: low = best** (rank near 0 = most important constraint)
- **V4.6 tier: 0 = best** (tier 0 assigned to rank <= 0.12, the top 12%)
- Tier assignment code: `tier=0` for `rank <= 0.12`, `tier=4` for the rest (up to rank <= 1.0)
- This is the SAME direction as MISO V4.4 (both: low rank = best)
- V7.1B uses the OPPOSITE convention (high rank = best)

**Impact:**
All benchmark comparisons against V4.6 must use low-rank-is-best convention. If our model outputs high-rank-is-best, invert before comparing. Affects Phase 4 and Phase 5.

### 2026-03-25 — R1 missing PY 2025-06 in V4.6

**What was wrong / surprising:**
V4.6 R1 has planning years 2019-06 through 2024-06 only (6 PYs). R2/R3/R4 include 2025-06 (7 PYs). R1 is missing the current holdout year.

**Correct understanding:**
R1 benchmark comparison cannot include PY 2025-06. Only R2-R4 have 2025-06 coverage. This may be because R1 was generated before 2025-06 data was available.

**Impact:**
When benchmarking baseline model against V4.6, R1 evaluation is limited to PYs 2019-06..2024-06. Holdout comparison for R1 requires a different approach or accepting the gap.

### 2026-03-25 — PJM bridge uses aq1-aq4 quarterly partitions (not just period_type=a)

**What was wrong / surprising:**
Phase 0 contract note stated "There is NO PJM equivalent of MISO's aq1-aq4 quarterly partition." This is wrong for the underlying data. The V4.6 *publish* surface uses `period_type=a`, but the PJM bridge (constraint_info) data is partitioned by `period_type=aq1/aq2/aq3/aq4`, exactly like MISO.

Verified on disk:
```
PJM_SPICE_CONSTRAINT_INFO.parquet/.../auction_month=2019-06/market_round=1/period_type=aq1/
PJM_SPICE_CONSTRAINT_INFO.parquet/.../auction_month=2019-06/market_round=1/period_type=aq2/
...
```

**Correct understanding:**
- The bridge CID→branch mapping varies by quarter (aq1-aq4). Different constraints may be active in different quarters.
- The density data does NOT have aq partitions — it has per-market_month data (12 months per PY).
- The V4.6 publish collapses quarterly data into a single `a` output — likely aggregating across quarters.
- For GT mapping, we need to use the quarterly bridge to map DA SP to branches per quarter, then aggregate to annual.

**Impact:**
- Phase 2 bridge module must support quarterly bridge loading (same as MISO)
- Phase 3 GT mapping must be done per-quarter, not as a single annual pass
- The "no quarterly partition" statement in the contract note is misleading — corrected below

### 2026-03-25 — PJM density is a migration-style split across two roots

**What was wrong / surprising:**
Initial analysis treated the two density roots as partial duplicates. They are NOT duplicates — they hold different rounds for overlapping PYs.

**Correct understanding (verified from data):**

Legacy root (`network_model=miso/spice_version=v6/auction_type=annual/`):
| PY | Rounds | Months |
|----|--------|--------|
| 2019-06 | R1, R2, R3, R4 | 12/12 |
| 2020-06 | R1, R2, R3, R4 | 12/12 |
| 2021-06 | R1, R2, R3, R4 | 12/12 |
| 2023-06 | R1, R2, R3, R4 | 12/12 |
| 2024-06 | **R2, R3, R4 only** | 12/12 |

Newer root (`spice_version=v6/auction_type=annual/`):
| PY | Rounds | Months |
|----|--------|--------|
| 2024-06 | **R1 only** | 12/12 |
| 2025-06 | R1, R2, R3, R4 | 12/12 |

Key facts:
- `market_round` is a column inside the parquet, AND effectively a partition dimension across roots
- PY 2022-06 is **missing from both roots entirely**
- PY 2024-06 is **split by round across roots**: R1 in newer, R2-R4 in legacy
- PY 2025-06 is **only in newer root** with all 4 rounds
- Any loader that scans only one root will be wrong

**Impact:**
- `data_loader.py` must implement root-resolution logic that selects the correct root per (PY, round)
- Cannot train/eval on PY 2022-06 — reduces training set by ~1 year
- Phase 1 must build an explicit root/round resolution table before any data loading

### 2026-03-25 — Contract note prematurely asserted DA loader

**What was wrong / surprising:**
Phase 0 contract note section 8 listed `PjmApTools.get_da_shadow_by_peaktype()` as the definitive DA loader. The handoff explicitly says the canonical DA loader must be confirmed from code and data in Phase 1, not assumed.

**Correct understanding:**
The notebook uses `get_da_shadow_by_peaktype()`, which delegates to `DaShadowPrice.load_data()`, which reads from `PJM_DA_SHADOW_PRICE.parquet`. This is a *candidate* loader, not yet confirmed as canonical for the annual pipeline. Phase 1 must verify:
- Whether daily vs monthly aggregation is needed
- Whether the loader returns constraint-level or branch-level data
- Whether replacement/backfill affects usable date range

**Impact:**
Contract note section 8 reworded to mark DA loader as "candidate, pending Phase 1 confirmation."

### 2026-03-25 — Principal review: corrected density mental model and answer precision

**What was wrong / surprising:**
Principal review of the 25-question answers identified several issues:
1. Q5 described `market_round` as "just an in-file column" — wrong. It is a column inside the parquet AND the dimension that splits data across the two roots. The two roots are not duplicates.
2. Q6 listed the two roots but did not clearly state that overlapping PY 2024-06 is split by round (legacy=R2-R4, newer=R1).
3. Q7 deferred to "Phase 1 must verify" when the round-per-root split was already observable from the data.
4. Q1 was loose — did not distinguish that GT/eval needs an internal quarterly layer even though the publish grain is annual.
5. Q8-9, Q14-17 were correctly flagged by the principal as still Phase 1 decisions, not answered yet. The answers should not have presented candidate loaders as if semi-confirmed.

**Correct understanding:**
- Density: think of it as a migration-style split, not one clean root. For any (PY, round), exactly one root has the data.
- Contract note and answers now corrected: Q1 distinguishes publish vs internal eval grain, Q5-Q7 use the verified round-per-root table, Q8-Q17 clearly marked as pending Phase 1.
- Any loader that scans only one root will silently miss rounds.

**Impact:**
- Phase 1 density loader design is now unambiguous: must resolve root per (PY, round)
- Future agents must not treat the two roots as interchangeable or redundant

### 2026-03-25 — DA year=2022 has zero rows; PY 2021-06 GT partially affected

**What was wrong / surprising:**
Coverage audit script found DA `year=2022` partition has 0 rows. This is not just a density gap — there is literally no DA shadow price data for calendar year 2022.

**Correct understanding:**
- PY 2022-06 (Jun 2022–May 2023): no density, AND no DA for Jun–Dec 2022. Jan–May 2023 DA exists (year=2023 has data). So PY 2022-06 GT is at best 5/12 months.
- PY 2021-06 (Jun 2021–May 2022): DA exists for Jun–Dec 2021 (year=2021), but Jan–May 2022 is in year=2022 which is empty. So PY 2021-06 GT is at best 7/12 months.
- All other PYs have complete DA (2010 through 2021 and 2023 through 2025 are populated; 2026 through Mar).

**Impact:**
- PY 2022-06 is completely unusable (no density, no DA for most months)
- PY 2021-06 GT evaluation is compromised — missing 5 months of DA (Jan–May 2022)
- Usable PYs for full training: 2019-06, 2020-06, 2023-06, 2024-06 (4 PYs). PY 2021-06 usable with caveat. PY 2025-06 holdout with 10/12 months DA.

### 2026-03-25 — PJM DA shadow prices are always negative (convention, not error)

**What was wrong / surprising:**
All PJM DA shadow prices across all years (2010-2026) are strictly negative. Zero positive values in any year, confirmed in both `modeling_data` and `yesenergy_data` sources.

**Correct understanding:**
- This is the **PJM convention** — DA shadow prices are stored as negative values representing the cost of the constraint
- When computing total |SP| for GT, always use `abs(shadow_price)`
- The `shadow_sign` in V4.6 output comes from the SPICE model's `flow_direction`, NOT from the sign of DA shadow price
- MISO may use a different convention — do not assume cross-RTO sign parity

**Impact:**
- GT module must use `abs(shadow_price)` everywhere
- No sign-based grouping of DA data is meaningful (all negative)
- The DA→GT mapping is by CID identity, not by sign

### 2026-03-25 — aq4 GT recovery drops to 74% (vs 96-99% for aq1-aq3)

**What was wrong / surprising:**
GT mapping coverage for PY=2024-06, R2, onpeak shows strong recovery for aq1-aq3 (96.6%, 98.7%, 96.5%) but aq4 drops to **74.0%**. The 124 unmapped aq4 CIDs account for 416k |SP| (26% of aq4 total).

**Correct understanding:**
The unmapped aq4 CIDs use **natural-language facility names** (e.g., "Jordan - WFrankfort E 138kV") that don't match the bridge's coded format (e.g., "02BRIM  138 KV  02B-BTAP:BASE"). This is a naming convention mismatch, not a missing-data problem. The top 5 unmapped aq4 CIDs account for ~255k |SP|.

Annual recovery is 91.7% overall, but the aq4 drop is a structural issue that a fallback ladder (fuzzy matching or supplement keys) could partially address.

**Impact:**
- Phase 2 bridge module should include a name-normalization or fuzzy-matching fallback for PJM CIDs
- aq4 evaluation will be less accurate than aq1-aq3 without a fallback
- Superseded by the multi-year sweep below — the aq4 drop is not the primary pattern

### 2026-03-25 — Multi-year GT recovery sweep: drops are NOT aq4-specific; 2025-06 collapses due to DA CID naming convention change

**What was wrong / surprising:**
Swept GT mapping recovery across PYs 2019-2025, R2, onpeak. The aq4 drop seen in 2024-06 is NOT a universal quarterly pattern. Results:

| PY | aq1 | aq2 | aq3 | aq4 | Annual | Worst Q |
|----|-----|-----|-----|-----|--------|---------|
| 2019-06 | 93.3% | 88.5% | 92.7% | 85.2% | 90.0% | aq4 |
| 2020-06 | 96.1% | 92.3% | 93.1% | 78.7% | 89.6% | aq4 |
| 2021-06 | 90.4% | 81.4% | 81.4% | 93.8% | 86.3% | aq2/aq3 |
| 2023-06 | 99.0% | 93.9% | 91.4% | 94.3% | 94.4% | aq3 |
| 2024-06 | 96.6% | 98.7% | 96.5% | 74.0% | 91.7% | aq4 |
| **2025-06** | **70.4%** | **51.4%** | **26.7%** | **44.2%** | **45.6%** | **all** |

**2025-06 (holdout) is catastrophic**: 45.6% annual recovery. Only 28-30 CIDs overlap out of 430-612 DA CIDs per quarter.

**Root cause for 2025-06:** DA CID naming convention changed. New DA CIDs use underscores and device-type suffixes:
- Old (bridge): `02BRIM  138 KV  02B-BTAP:BASE`
- New (2025-06 DA): `BEDINGTO_T1_500TRAN2_XF:L500.Bedington-Doubs`

The new format is: `{FACILITY}_{DEVICE}_{VOLTAGE}{SHORT}_{CIRCUIT}_{TYPE}:{CONTINGENCY}` where TYPE is `_LN` (line), `_XF` (transformer).

**Root cause for older PY drops (85-78%):** smaller naming mismatches (natural-language names like "Jordan - WFrankfort E 138kV") affect individual high-SP constraints sporadically.

**No monthly f0 bridge fallback available**: monthly bridge only exists for 2026-01..2026-04 (not historical months).

**Correct understanding:**
- Historical PYs 2019-2024: 86-94% recovery with direct CID match. Drops are sporadic, not quarter-systematic.
- 2025-06: systematic naming convention change breaks direct CID matching. Need a normalization layer or separate mapping table.
- The bridge `device_name` column may contain the key needed to match new-format DA CIDs.

**Impact:**
- For historical PYs (training), direct CID match is sufficient (~90% recovery)
- For 2025-06 (holdout), a CID normalization or device_name-based mapping is mandatory
- Phase 2 bridge module must include a fallback for new-format CIDs
- The 2025-06 problem must be solved before holdout evaluation is meaningful

### 2026-03-25 — f0 fallback helps 2025-06 aq3/aq4 but not aq1/aq2; no effect on historical PYs

**What was wrong / surprising:**
f0 monthly bridge fallback (like MISO's `_try_monthly_bridge`) was tested across all PYs. Results:
- Historical PYs 2019-2024: **zero effect** — no monthly bridge data exists for those settlement months
- 2025-06 aq3: 26.7% → **68.4%** (268 CIDs recovered via Jan+Feb 2026 f0 bridge)
- 2025-06 aq4: 44.2% → **91.7%** (148 CIDs recovered via Mar+Apr 2026 f0 bridge)
- 2025-06 aq1/aq2: no change (no monthly bridge for Jun-Nov 2025)
- 2025-06 annual: 45.6% → **66.8%**

**Correct understanding:**
- f0 fallback is essential for 2025-06 aq3/aq4 — recovers ~42-48pp
- But it cannot help 2025-06 aq1/aq2 or any historical PY
- The remaining ~33% loss in 2025-06 is due to: (a) Dec 2025 having no monthly bridge, (b) new-format CIDs in aq1/aq2 settlement months with no bridge coverage at all
- For historical PYs, the 5-20% loss per quarter is solely naming mismatches in the annual bridge — no fallback available

**Impact:**
- Phase 2 bridge module must implement f0 fallback (essential for current PY)
- 2025-06 holdout evaluation will be capped at ~67% recovery even with f0 — honest evaluation must report this
- For historical PYs, a device_name-based fuzzy matcher is the only remaining lever

### 2026-03-25 — Correct GT mapping approach: match by monitored line, not full constraint_id

**What was wrong / surprising:**
Previous GT mapping scripts matched DA to bridge on full `constraint_id` (monitored:contingency). This is wrong. The MISO repo maps many CIDs → one branch (many-to-one collapse). The bridge key is effectively the **monitored line**, not the full CID. For PJM, 99.9% of monitored lines in the bridge map to exactly 1 branch. The 0.1% exceptions are generic interface names (APSouth, WEST, AEP-DOM).

Earlier "monitored-line matching" was treated as a *fallback*. It is actually the **primary and correct** matching strategy.

**Correct understanding:**
- The model is branch-level. Both SPICE features and DA GT collapse many constraints onto the same branch.
- PJM `constraint_id` = `monitored_facility:contingency`. The contingency identifies the outage scenario but doesn't change the physical branch.
- Mapping should be: `DA.monitored_facility` → (normalize whitespace) → bridge monitored part → `branch_name`.
- Full-CID matching is a subset of this — it works when the bridge happens to have the exact same contingency. Monitored-line matching is strictly more general.

**Recovery with monitored-line matching (R2, onpeak):**

| PY | aq1 | aq2 | aq3 | aq4 | Annual |
|----|-----|-----|-----|-----|--------|
| 2019-06 | 93.8% | 90.1% | 92.7% | 90.4% | ~92% |
| 2020-06 | 96.1% | 92.3% | 93.5% | 80.2% | ~90% |
| 2021-06 | 90.4% | 81.4% | 81.4% | 94.2% | ~87% |
| 2023-06 | 99.0% | 93.9% | 91.7% | 94.3% | ~95% |
| 2024-06 | 96.7% | 98.7% | 96.5% | 79.7% | ~93% |
| 2025-06 | 89.1% | 65.0% | 29.0% | 44.7% | ~57% |

- Historical PYs: **87-95%** — the 5-13% loss is DA monitored lines that genuinely don't exist in the bridge.
- 2025-06 aq1: **89.1%** (good — monitored-line matching recovers most of aq1)
- 2025-06 aq2-aq4: **29-65%** — a `monitored_facility` format change in recent DA data (`BEDINGTO_T1_500TRAN2_XF` vs bridge `BEDINGTO500 KV  BEDINGTO.T1`) breaks even monitored-line matching. Same physical equipment, different string encoding.

**f0 monthly fallback (on top of monitored-line matching):**
- Still essential for 2025-06 aq3/aq4 where monthly bridge data exists (Jan-Apr 2026)
- Expected to push 2025-06 aq3 from 29% → ~70% and aq4 from 45% → ~92%
- No help for aq1/aq2 (no monthly bridge for those months)

**Impact:**
- Phase 2 bridge module should use monitored-line → branch as the primary mapping, not full CID
- The f0 fallback remains valuable for 2025-06 where monthly bridge exists
- 2025-06 aq2/aq3 without f0 coverage is capped by the format change — needs either bridge regeneration or a normalization layer
- The previous agents.md entries about "naming convention mismatch" and "fuzzy matching" are superseded by this entry — the root cause is understood, not speculative

### 2026-03-25 — CRITICAL: initial blend signal experiment has two methodological errors

**What was wrong / surprising:**
The `build_blend_signal.py` experiment and all comparison numbers reported in this session have two errors that invalidate the results:

**Error 1: DA not filtered by peak type.**
Both the DA feature (2-year lookback) and the GT label use all-hours DA — onpeak + offpeak + weekend combined. V4.6's onpeak signal uses only onpeak DA hours via `get_da_shadow_by_peaktype(peak_type='onpeak')`. We were comparing an all-hours model against V4.6's onpeak-only model, evaluated on all-hours GT. This inflates our apparent performance because:
- Our DA feature includes offpeak SP that V4.6 deliberately excludes
- Our GT denominator includes offpeak SP, making the capture fraction look different
- The comparison is not apples-to-apples

**Error 2: DA lookback cutoff uses April, not March.**
The script uses `da_end_year=year, da_end_month=4` (April) as the lookback end. But the DA lookback must respect the round close date. For R2, close is April 11. Since we load by month granularity (no day-level filtering), we cannot safely include any April data — partial-month April would leak post-submission DA. The safe cutoff at month granularity is **March** (last full month before any round's close date). From `ml/core/calendars.py`: all PJM rounds close in April, so `get_history_cutoff_month()` returns March for all rounds.

The notebook also uses April as the lookback end, but it loads via `get_da_shadow_by_peaktype(et_ex=run_at+1day)` which does day-level cutoff. Our month-level loading cannot replicate this safely.

**Correct understanding:**
- DA features must be loaded with `peak_type` filtering (onpeak-only for the onpeak signal)
- GT must be computed from onpeak-only DA hours
- DA lookback end must be March (not April) when using month-level granularity
- All comparison numbers in this session (Abs_SP@200, @400, top-10 tables, NB-12 tables) are preliminary and must be re-run after fixing both errors

**Impact:**
- ALL benchmark numbers reported so far are invalid for production use
- The blend signal script must be rewritten with:
  1. Peak-type hour filtering for DA (both features and GT)
  2. March cutoff for DA lookback
  3. Separate onpeak and dailyoffpeak signal builds
- The qualitative findings (density+DA blend > pure DA, coverage analysis, bridge mapping approach) remain valid
- The quantitative deltas vs V4.6 will change and may be smaller after correction

### 2026-03-25 — PJM onpeak hour definition needed for DA filtering

**What was wrong / surprising:**
To filter DA by peak type, we need PJM's exact onpeak hour definition. The DA parquet has `datetime_beginning_utc` (hourly timestamps in US/Eastern). pbase's `add_class_type_to_df()` and `get_peak_hourly_df_with_extended_offpeak()` handle this, but our standalone scripts need to replicate the logic.

**Correct understanding:**
PJM class types and their hour definitions:
- `onpeak`: weekday (Mon-Fri, excluding NERC holidays), hours ending 8-23 (hour_beginning 7-22 EPT)
- `dailyoffpeak`: every day, hours ending 24-7 + 23 (hour_beginning 23, 0-6 EPT). The notebook passes `offpeak_hrs=(22, 9)` which means offpeak hours start at HB22 and end before HB9.
- `wkndonpeak`: weekend + NERC holidays, hours ending 8-23

For the V4.6 notebook, onpeak DA is loaded via:
```python
da = aptools.tools.get_da_shadow_by_peaktype(st=st, et_ex=et, peak_type='onpeak', offpeak_hrs=(22, 9))
```

For our scripts, we must either:
1. Use `PjmApTools.get_da_shadow_by_peaktype()` directly (requires Ray)
2. Or replicate the hour/weekday filter on raw DA data (no Ray, but must match exactly)

**Impact:**
- Must determine whether to use pbase loader (accurate but needs Ray) or standalone filter (no Ray but risk of mismatch)
- NERC holiday calendar must be accounted for — weekday-only filtering without holidays will slightly overcount onpeak hours

### 2026-03-25 — DA lookback safe cutoff is March for all PJM annual rounds

**What was wrong / surprising:**
PJM annual round close dates from `ml/core/calendars.py`:
- R1: April 4
- R2: April 11
- R3: April 18
- R4: April 25

All rounds close in April. When loading DA at month granularity, April is NOT safe for any round — even R4 (April 25) has 5 days of April that are post-submission.

**Correct understanding:**
- `get_history_cutoff_month()` returns March for all PJM rounds (the last FULL safe month)
- DA lookback must end at March (inclusive), not April
- If day-level DA loading is later implemented, the cutoff can be extended to the actual close date minus 1 day
- The V4.6 notebook uses day-level cutoff (`et_ex=run_at+1day`), so it can safely include early April. Our month-level approach cannot.

**Impact:**
- DA lookback window for all signals: `[PY_year-2 June, PY_year March]` (inclusive)
- This loses ~10 days of April DA that V4.6 has access to — a small disadvantage but necessary for correctness
- All existing scripts using `da_end_month=4` must be changed to `da_end_month=3` (inclusive) or `da_end_month=4` with `filter(month < 4)`

### 2026-03-25 — Corrected benchmark results: density+DA blend beats V4.6 @200 across all class types

**What was wrong / surprising:**
After fixing both errors (peak-type DA filtering + March cutoff), re-ran full sweep across 7 PYs × 4 rounds × 3 class types. Results in `docs/baseline-benchmark-results.md`.

**Correct understanding:**
Corrected results (Abs_SP, each model picks own top-K, class-type-specific GT denominator):

| Class type | @200 avg Δ | @200 wins | @400 avg Δ | @400 wins |
|-----------|-----------|-----------|-----------|-----------|
| onpeak | +4.9pp | 25/25 | +3.6pp | 21/25 |
| wkndonpeak | +3.0pp | 22/25 | -1.2pp | 14/25 |
| dailyoffpeak | +1.9pp | 18/25 | -1.3pp | 19/25 |

Key year findings:
- 2021-06 is our systematic weakness @400 (DA year=2022 gap)
- 2022-06 now included (data backfilled since initial audit)
- 2025-06 holdout: +4-9pp vs V4.6 on onpeak

**Impact:**
- The density+DA blend is a viable baseline for the 7.0 signal
- Onpeak is the strongest class type; dailyoffpeak needs tuning
- The superseded v1 build_blend_signal.py (all-hours bug) and v2 (single-ctype) have been removed
- `sweep_all_ctypes.py` is now the canonical benchmark script

### 2026-03-25 — PY 2022-06 data backfilled since initial audit

**What was wrong / surprising:**
Initial audit (earlier in this session) found PY 2022-06 density missing and DA year=2022 with 0 rows. Both have since been backfilled:
- Density: 12 months now present under `network_model=miso` root
- DA year=2022: 73,065 rows

**Correct understanding:**
- PY 2022-06 is now usable for R1 and R2 (R3/R4 bridge still missing)
- The data landscape is not static — future workers should re-run `audit_coverage.py` before assuming any gaps from earlier reports

**Impact:**
- Training set now has 6 usable PYs (2019-2024, with 2021 caveated and 2022 R1+R2 only)
- Earlier coverage report and agents.md entries about 2022-06 being "completely unusable" are superseded

### 2026-03-25 — Principal review: three high-priority doc/script issues + ctype-specific model requirement

**What was wrong / surprising:**

**Issue 1: GT coverage scripts and report mislabeled as onpeak.**
`gt_mapping_coverage.py`, `gt_recovery_sweep.py`, and `phase1-coverage-report.md` present results as "R2, onpeak" but the DA loader never applies any peak-hour filter. All coverage numbers are all-hours, not onpeak-specific. The report even admits this at the bottom ("Peak-type filtering: GT scripts currently use all hours") but the tables above that note present the numbers without qualification.

**Issue 2: Annual root-resolution stale — 2025-06 and 2024-06 newer root disappeared.**
`gt_recovery_sweep.py:32` routes 2025-06 to `.../spice_version=v6/auction_type=annual` but this path no longer exists on NFS. All data (including 2024-06 R1 and 2025-06) has moved to the legacy root `.../network_model=miso/...`. The same stale assumption is in `phase1-data-source-contract.md` and `phase0-pjm-base-grain-contract.md`.

**Issue 3: 2022-06 stale claims persist in multiple docs.**
`phase1-data-source-contract.md`, `phase1-coverage-report.md`, and `baseline-benchmark-results.md` still contain "2022-06 MISSING" claims. Live NFS now has density (12 months), bridge (R1-R2), limit, SF, and DA (73k rows, months 1-12) for 2022-06.

**Issue 4 (new requirement): Model must be fully class-type-specific.**
Principal requirement:
- DA features: ctype-specific (already done in `sweep_all_ctypes.py`)
- Density features: ctype-specific via bridge mapping (already done — bridge filters by class_type, so branch sets differ per ctype, though raw density values are the same)
- GT: ctype-specific (already done in `sweep_all_ctypes.py`)
- Metrics: ctype-specific — top-10 binders, NB-12, total SP all differ by ctype
- Benchmarks: compare our onpeak against V4.6 onpeak, our dailyoffpeak against V4.6 dailyoffpeak, etc.
- Note: raw density from `PJM_SPICE_DENSITY_DISTRIBUTION.parquet` has no class_type dimension. V4.6 had separate `flow_onpeak.parquet`/`flow_offpeak.parquet`. We cannot replicate this split from our data. Density class-type specificity comes only from the bridge mapping path.

**Correct understanding:**
- Phase 1 coverage scripts must be either re-run with peak-type filtering or clearly relabeled as "all-hours"
- Root resolution rule must be updated: all PJM annual data is now under the legacy `network_model=miso` root
- All doc references to "2022-06 missing" or "newer root for 2024-06/2025-06" must be corrected
- The model is already class-type-specific in the canonical benchmark script (`sweep_all_ctypes.py`)

**Impact:**
- `phase1-data-source-contract.md` root resolution table needs rewrite
- `phase0-pjm-base-grain-contract.md` section 8.6 root table needs rewrite
- `phase1-coverage-report.md` should either be re-run with ctype filter or relabeled
- `gt_mapping_coverage.py` and `gt_recovery_sweep.py` should add peak-type filtering or be marked as all-hours diagnostic tools
- Future NB-12 and top-10 case studies must be run per class type separately

### 2026-03-25 — CRITICAL: quarter-aware mapping reverses benchmark results — we LOSE to V4.6

**What was wrong / surprising:**
All previous benchmark results used a union lookup built by scanning aq1-aq4 and keeping the first match per monitored line. This is wrong because PJM bridge membership is quarter-sensitive — a constraint may map to different branches in different quarters, or may only exist in certain quarters.

After fixing to map DA per quarter through that quarter's bridge then sum at branch level:

| Class type | @200 avg Δ | @200 wins | @400 avg Δ | @400 wins |
|-----------|-----------|-----------|-----------|-----------|
| onpeak | **-7.9pp** | 4/25 | **-3.0pp** | 5/25 |
| dailyoffpeak | **-7.7pp** | 5/25 | **-1.7pp** | 5/25 |
| wkndonpeak | **-7.0pp** | 5/25 | **-3.1pp** | 11/25 |

We now lose on average by 7-8pp @200 and 2-3pp @400 across all class types. The only wins are PY 2021-06 (where V4.6 itself is weak) and 2025-06 holdout.

**Root cause of the reversal:**
The union lookup was leaking branch mappings across quarters. If a DA constraint bound in aq4 but its monitored line only existed in aq1's bridge, the union lookup would still map it (using aq1's branch), while the correct per-quarter lookup would leave it unmapped. This inflated our branch-level DA feature and GT, giving us fake coverage advantage.

**Correct understanding:**
- V4.6 is a materially stronger signal than our density+DA blend
- V4.6's advantage comes from its flow-based deviation (computed from raw hourly flow simulations), which our density bins cannot replicate
- Our density signal captures the right tail of the flow distribution but loses the temporal/binary structure that V4.6's deviation_max/deviation_sum encode
- V4.6 has only ~1,049 branches but ranks them much better (label coverage ~56-63% vs our ~94%)
- Our higher label coverage does not compensate for worse ranking quality

**Label coverage context (new metric, now reported):**
- Our signal: ~94% of GT branches are in our universe
- V4.6: ~56-63% of GT branches are in V4.6's universe
- Despite seeing fewer GT branches, V4.6 captures more SP because its ranking is more accurate

**Impact:**
- ALL previous benchmark numbers in this session are invalid (this is the third invalidation)
- `baseline-benchmark-results.md` must be rewritten entirely with the corrected numbers
- The density+DA blend in its current form is not competitive with V4.6
- Next steps should focus on improving ranking quality, not coverage

### 2026-03-25 — Density union lookup verified as exact for monitored-line → branch mapping

**What was wrong / surprising:**
Review flagged that density still uses an annual union lookup across aq1-aq4, while GT/DA were fixed to be quarter-aware. Investigated whether this is a real bug.

**Correct understanding:**
Verified empirically for PY=2024-06 R2 onpeak: all 4 quarters have **identical** monitored → branch mappings (6,431 each, 0 conflicts, 0 quarter-only lines). The quarter sensitivity in the bridge is at the `constraint_id` level (different `monitored:contingency` pairs per quarter), not at the `monitored-line → branch` level.

Since density maps through monitored line (not full CID), the union is exact — no information is lost. DA/GT map through full CIDs (via the `monitored_facility` column in DA matching the monitored part of `constraint_id` in the per-quarter bridge), so they correctly needed the per-quarter fix.

**Impact:**
- Density union lookup is correct and does not need the per-quarter treatment
- Added verification comment to `sweep_all_ctypes.py` documenting this
- Renamed "label_coverage" to "gt_branch_recall" to avoid misinterpretation

### 2026-03-25 — NERC holiday list for PJM peak-type classification

**What was wrong / surprising:**
Our peak-type filter uses weekday/weekend only for onpeak vs wkndonpeak. PJM treats NERC holidays as off-peak days (same as weekends). pbase has the canonical implementation at `pbase/utils/hours.py:is_off_peak_day()`.

**Correct understanding:**
PJM off-peak day = weekend OR any of these 6 NERC holidays:
1. New Year's Day (Jan 1; if Sunday, observed Monday Jan 2)
2. Memorial Day (last Monday in May)
3. Independence Day (Jul 4; if Sunday, observed Monday Jul 5)
4. Labor Day (first Monday in September)
5. Thanksgiving Day (4th Thursday in November)
6. Christmas Day (Dec 25; if Sunday, observed Monday Dec 26)

On an off-peak day, HB07-22 = `wkndonpeak`, HB23+HB00-06 = `dailyoffpeak`.
On a non-off-peak weekday, HB07-22 = `onpeak`, HB23+HB00-06 = `dailyoffpeak`.

PJM onpeak hours: `7 <= hour_beginning < 23` on non-off-peak weekdays (confirmed from `pbase/utils/hours.py` line 67).

Impact: ~6 weekday holidays per PY × 16 hours = ~96 hours misclassified (2.4% of onpeak). Our current script classifies these as onpeak instead of wkndonpeak. Small but fixable.

**Impact:**
- The fix is to call `is_off_peak_day()` or replicate its logic in the polars filter
- Current benchmark numbers have a ~2.4% hour-classification error for onpeak/wkndonpeak
- dailyoffpeak is unaffected (it's defined by hour only, applies every day including holidays)
- Subtle: pbase only implements Sunday→Monday observation (Jan 1 on Sunday → observe Monday Jan 2). It does NOT implement Saturday→Friday observation. If we copy pbase exactly, a holiday falling on Saturday is NOT an off-peak day — consistent with the rest of the stack even if debatable

### 2026-03-25 — CRITICAL: DA lookback feature was zeroed out by quarter-month filtering

**What was wrong / surprising:**
After the quarter-aware GT fix, benchmark results showed we lost badly to V4.6. Reviewing the implementation found a second major bug: the DA lookback feature was being passed through `map_da_quarterly()`.

That function filters rows to the evaluation planning year's settlement quarters:
- `aq1 = Jun-Aug PY`
- `aq2 = Sep-Nov PY`
- `aq3 = Dec-Feb PY+1`
- `aq4 = Mar-May PY+1`

But the DA feature window is historical:
- `Jun (PY-2)` through `Mar PY`

These windows do not overlap. Result: `branch_da` was empty for nearly every cell, so both `baseline_v69` and `candidate_v70` were effectively density-only signals while being reported as density+DA blends.

Verified examples after debugging:
- `2024-06 R2 onpeak`: raw DA rows `76,602`, branch DA rows `0` before fix
- `2023-06 R1 onpeak`: raw DA rows `69,482`, branch DA rows `0` before fix

**Correct understanding:**
- GT must remain quarter-aware because settlement DA belongs to the evaluation year's aq1-aq4 slices
- Historical DA features must NOT be quarter-month filtered
- Historical DA should map through the evaluation cell's monitored-line lookup directly

Implemented fix in `sweep_all_ctypes.py`:
- added `map_da_union()`
- DA feature now uses `map_da_union(da_feat, union_lookup)`
- GT still uses `map_da_quarterly()`

**Corrected benchmark summary after the fix:**

`baseline_v69` vs V4.6:
- onpeak: `+4.8pp @200`, `+3.6pp @400`
- dailyoffpeak: `+1.9pp @200`, `-1.3pp @400`
- wkndonpeak: `+2.7pp @200`, `-1.2pp @400`

`candidate_v70` vs V4.6:
- onpeak: `+4.3pp @200`, `+3.4pp @400`
- dailyoffpeak: `-0.7pp @200`, `-2.1pp @400`
- wkndonpeak: `+1.7pp @200`, `-1.6pp @400`

`candidate_v70` vs `baseline_v69`:
- worse on average in every class type at both `@200` and `@400`

**Impact:**
- The earlier "we lose broadly to V4.6" conclusion is invalid
- The fixed 2-component blend (`baseline_v69`) is the current reproducible leader
- The 3-component hand-weighted mimic (`candidate_v70`) should not be the initial release baseline

### 2026-03-25 — Naming clarification: `V4.6` is the real baseline

**What was wrong / surprising:**
Internal experiment labels like `baseline_v69` and `candidate_v70` started to blur the actual project objective in discussion.

**Correct understanding:**
- The real baseline is released `V4.6`
- `baseline_v69` is just the current best reproducible challenger
- `candidate_v70` is an internal experiment and not a target state
- All future summaries should answer: "does this beat `V4.6`?" before discussing internal challenger-vs-challenger comparisons

**Impact:**
- Docs should frame `V4.6` as the benchmark baseline to beat
- Internal challenger labels are still useful, but only as secondary comparisons

### 2026-03-25 — Implemented NERC holiday handling and reran full benchmark

**What was wrong / surprising:**
The benchmark script previously treated `wkndonpeak` as weekends only and `onpeak` as plain weekdays. That disagreed with `pbase.utils.hours.is_off_peak_day()`, which treats NERC holidays as off-peak days.

**Correct understanding:**
Implemented the exact `pbase` holiday logic in `sweep_all_ctypes.py`:
- weekends are off-peak days
- New Year's Day, Memorial Day, Independence Day, Labor Day, Thanksgiving, Christmas
- Monday observation only where the holiday falls on Sunday, matching `pbase`

This changes only the `HB07-HB22` daytime split:
- holiday daytime hours move from `onpeak` to `wkndonpeak`
- `dailyoffpeak` remains unchanged

**Rerun result:**
After the holiday-aware rerun:

`baseline_v69` vs V4.6:
- onpeak: `+4.8pp @200`, `+3.5pp @400`
- dailyoffpeak: `+1.9pp @200`, `-1.2pp @400`
- wkndonpeak: `+2.6pp @200`, `-1.2pp @400`

`candidate_v70` vs V4.6:
- onpeak: `+4.1pp @200`, `+3.4pp @400`
- dailyoffpeak: `-0.6pp @200`, `-2.0pp @400`
- wkndonpeak: `+1.5pp @200`, `-1.8pp @400`

**Impact:**
- The holiday fix is easy and is now implemented
- Aggregate results moved only slightly; the ranking story did not change
- `baseline_v69` remains the strongest internal challenger to released `V4.6`

### 2026-03-25 — Publication contract tightened for PJM 7.0b

**What was wrong / surprising:**
The first publication contract draft still left three release-critical items underspecified:
- whether publication is the full expanded CID universe or only a post-dedup subset
- the exact tier policy
- the exact metadata provenance for `constraint`, `equipment`, `convention`, `shadow_sign`, `flow_direction`, and `constraint_limit`

**Verified from sibling / released signals:**
- MISO v2 publishes the final post-dedup subset, not all expanded CIDs
- MISO `signal_publisher.py` enforces branch-cap selection before write
- PJM `V4.6` publishes a `34`-column annual constraint table plus parquet index
- PJM `V4.6` equipment format is `"{branch_name},{constraint}"`
- PJM `V4.6` uses `flow_direction = -shadow_sign` and `shadow_price = shadow_sign`
- PJM older DA annual monitor families publish simpler DA-derived sign metadata, which is a useful model for sign provenance

**Correct understanding:**
- `pjm 7.0b` should publish the final post-dedup subset only
- tier contract is `5` tiers, `200` constraints each, with `tier 0` strongest
- branch cap should be `3` for the initial release, mirroring MISO's first-pass discipline
- published schema is `V4.6-compatible-plus-limit`
- metadata provenance is now frozen in `pjm-annual-publication-contract.md`

**Impact:**
- publisher implementation can now target a concrete release contract
- remaining uncertainty is smaller and mostly validation-oriented:
  - prove sign parity on overlapping `V4.6` rows
  - prove the proxy `deviation_*` fields are acceptable on the published surface

### 2026-03-25 — First PJM 7.0b publisher dry run succeeded on one real cell

**What was wrong / surprising:**
The first implementation pass exposed two real production issues:
- PJM DA year partitions use mixed timestamp conventions: 2019-2022 are naive Eastern-local values, while 2024-2025 are explicit `US/Eastern`
- annual SF assembly is the main bottleneck; the raw SF parquet is partitioned by both `market_round` and `outage_date`, and broad scans are too expensive

**Correct understanding:**
- DA normalization should treat naive timestamps as already Eastern-local values and strip tz from the newer partitions to the same local-clock representation before hour filtering
- annual SF loading must read only the requested `market_round` and must prune columns file-by-file
- `constraint_limit` needs an explicit fallback from `PJM_SPICE_CONSTRAINT_LIMIT.parquet`; bridge inline limits are not sufficient

**Dry-run result (`2024-06 / R2 / onpeak`):**
- constraints parquet built successfully
- rows: `772`
- columns: contracted `V4.6 + constraint_limit` surface
- SF shape: `11,337 x 772`
- SF row index name: `pnode_id`
- tier counts: `200 / 200 / 200 / 172`
- overlap with released `V4.6`: `104` rows
- overlap parity on those rows:
  - `constraint`: `100%`
  - `equipment`: `100%`
  - `convention`: `100%`
  - `shadow_sign`: `100%`

**Impact:**
- publisher skeleton is real and can build a valid annual PJM cell
- we are not ready for full release publication yet because only one cell has been smoked
- the next gating work is multi-cell smoke coverage and performance hardening for annual SF assembly

### 2026-03-25 — PY 2022-06 R3-R4 data now fully available

**What was wrong / surprising:**
Multiple docs and manifest stated 2022-06 R3-R4 was unpublishable due to missing density and bridge data. Verified on live NFS: all four datasets (density, bridge, limit, SF) now have R1-R4 for 2022-06.

**Correct understanding:**
- 2022-06 is fully publishable across all 4 rounds
- NFS data continues to be backfilled — earlier gaps are no longer current
- The only PY with round-level publishing constraints is 2025-06 R1 (no V4.6 benchmark, but data exists)

**Impact:**
- Updated manifest.json: 2022-06 now shows rounds [1,2,3,4], no unpublishable_rounds
- Updated phase1-data-source-contract.md: 2022-06 row changed from R1-R2 to R1-R4
- Updated phase1-coverage-report.md: 2022-06 status changed from "R1-R2 only" to full training
- Updated pjm-annual-publication-contract.md: removed "2022-06 R3-R4 unpublishable" from section 8.7
- Total publishable cells: 7 PYs × 4 rounds × 3 ctypes = 84 (all cells now publishable)

### 2026-03-25 — Multi-cell smoke test: 5/5 cells pass after two publisher bug fixes

**What was wrong / surprising:**
Two bugs found during multi-cell smoke testing:

1. **SF float32 dtype**: `load_annual_sf()` returned SF columns in the source parquet's native dtype (often float32). The publication contract requires float64.
2. **Duplicate __index_level_0__ column**: `finalize_publication()` used `set_index(CONSTRAINT_INDEX_COLUMN, drop=False)`, keeping the index key as both the pandas index AND a data column. When saving to parquet via `update_parquet()`, pyarrow wrote both, causing "Multiple matches for FieldRef.Name(__index_level_0__)" on load.

**Correct understanding:**
- Fix 1: Added `.astype("float64")` at the end of `load_annual_sf()`.
- Fix 2: Changed to `set_index(CONSTRAINT_INDEX_COLUMN)` (default `drop=True`). The index key is still the parquet index; it no longer appears as a duplicate data column.

**Multi-cell smoke results after fixes:**

| Cell | Rows | SF Shape | Tiers | V4.6 Overlap | Parity (c/e/conv/sign) | RT |
|------|------|----------|-------|-------------|------------------------|-----|
| 2024-06/R1/onpeak | 773 | 11319x773 | 200/200/200/173 | 103 | 100/100/100/99.0 | OK |
| 2024-06/R2/dailyoffpeak | 779 | 11337x779 | 200/200/200/179 | 107 | 100/100/99.1/97.2 | OK |
| 2024-06/R3/wkndonpeak | 782 | 11333x782 | 200/200/200/182 | 107 | 100/100/100/100 | OK |
| 2021-06/R2/onpeak | 819 | 10943x819 | 200/200/200/200/19 | 95 | 100/100/100/97.9 | OK |
| 2025-06/R4/onpeak | 788 | 11263x788 | 200/200/200/188 | 95 | 100/100/100/98.9 | OK |

All cells: schema valid, no nulls, index unique, SF float64, rank low=best, tiers contiguous, save/load round-trip passes.

**Impact:**
- Publisher is now functionally correct for multi-cell publication
- Row counts 773-819 are below the 1000 target — see separate analysis below
- shadow_sign parity 97-100% is expected (our sign is DA-derived, V4.6 uses SPICE flow_direction)
- convention parity 99-100% — minor difference on 1 dailyoffpeak constraint

### 2026-03-25 — Row count analysis: 773-819 rows vs 1000 target

**What was wrong / surprising:**
All cells produce 773-819 published constraints, well below the 1000 target (5 tiers x 200). The limiting factor is NOT branch_cap or zero-SF exclusion alone.

**Correct understanding:**
The pipeline bottleneck is the intersection of:
1. **Candidate expansion**: bridge provides ~6400 unique monitored lines, but many CIDs are filtered by convention < 10
2. **Branch collapse**: many CIDs share the same branch_name; branch_cap=3 eliminates siblings
3. **Zero-SF exclusion**: constraints with all-zero SF vectors are dropped
4. **Two-pass selection**: `select_candidates()` pre-selects up to 1000 rows, then `finalize_publication()` re-walks with SF checks, potentially losing more rows without replacement from the broader candidate pool

The two-pass design is slightly suboptimal: candidates eliminated by zero-SF in the second pass cannot be replaced because the first pass already stopped at the target. This is not a correctness bug — the contract allows short trailing tiers — but it explains why we consistently fall short. A single-pass design (combining branch_cap + zero-SF in one walk over the full candidate list) would produce slightly more rows.

However, V4.6 publishes ~1043-1091 constraints for the same cells. The gap (773 vs 1049) suggests our candidate universe is genuinely smaller, likely because density coverage of the bridge universe is narrower than V4.6's flow-based coverage.

**Impact:**
- 772-819 rows is acceptable for the initial 7.0b release (contract allows short tiers)
- To close the gap toward 1000, consider: (a) merging the two selection passes, (b) lowering convention threshold, (c) expanding density coverage
- This is not a blocker for publication
