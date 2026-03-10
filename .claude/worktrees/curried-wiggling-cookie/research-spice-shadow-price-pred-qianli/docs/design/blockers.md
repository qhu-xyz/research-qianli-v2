# Blockers: SPICE_F0P_V6.7B.R1 Signal Generation

## Summary

3 blockers preventing full coverage. MISO monthly/quarterly onpeak+offpeak is unblocked and ready to run.

---

## Blocker 1: MISO Annual Periods (aq1–aq4) — Missing Density Data

**Status**: Blocked on upstream data generation

**Problem**: Annual quarter periods (aq1–aq4) are auctioned in June. Each spans 3 market months:
- aq1: Jun–Aug
- aq2: Sep–Nov
- aq3: Dec–Feb
- aq4: Mar–May

The ML pipeline needs density data for each market month. But June's density directory only contains 1 market month:

```
/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/
  auction_month=2024-06/
    market_month=2024-06/    ← only this exists
    market_month=2024-07/    ← MISSING
    market_month=2024-08/    ← MISSING
    ...
```

Compare to July, which has 11 market months (enough for f0, f1, q2, q3, q4):
```
  auction_month=2024-07/
    market_month=2024-07/
    market_month=2024-08/
    ...
    market_month=2025-05/
```

**Impact**: Cannot generate signals for aq1, aq2, aq3, aq4 for any June auction month.

**Resolution**: Density model needs to be run for June with all market months (Jun through May+1). This is an upstream dependency — whoever runs the `prod_f0p_model_miso` density pipeline needs to extend the June output.

**Workaround**: None. The ML pipeline requires density features as input; there is no way to predict without them.

---

## Blocker 2: PJM Non-Onpeak Class Types — Hardcoded Constraint Path

**Status**: Needs investigation / code change

**Problem**: The PJM config in `iso_configs.py` hardcodes `class_type=onpeak` in the constraint path template:

```python
# src/shadow_price_prediction/iso_configs.py line 98-102
constraint_path_template=(
    "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/constraint_info/"
    "auction_month={auction_month}/market_round={market_round}/"
    "period_type={period_type}/class_type=onpeak"   # ← hardcoded
),
```

Yet the existing PJM 6.7B signal on disk has 3 class types:
```
TEST.TEST.Signal.PJM.SPICE_F0P_V6.7B.R1/2025-06/f0/
  dailyoffpeak/
  onpeak/
  wkndonpeak/
```

And the PJM constraint data directory only has onpeak:
```
prod_f0p_model_pjm/constraint_info/.../period_type=f0/
  class_type=onpeak/    ← only this exists
```

**Impact**: Cannot generate PJM signals for dailyoffpeak or wkndonpeak.

**Questions to resolve**:
1. How was the existing PJM 6.7B generated for dailyoffpeak/wkndonpeak? Was a different constraint set used?
2. Does PJM use the same constraints across all class types (unlike MISO which has separate constraint sets)?
3. If yes, should we just use `class_type=onpeak` constraints for all PJM class types?

**Resolution options**:
- A: Confirm PJM constraints are class-type-agnostic, then parameterize the template: `class_type={class_type}` and always load onpeak constraints regardless
- B: Generate separate constraint data for dailyoffpeak/wkndonpeak upstream
- C: Skip non-onpeak for PJM (but this doesn't match existing 6.7B coverage)

---

## Blocker 3: PJM Notebook — Not Yet Parameterized

**Status**: Implementation needed

**Problem**: The current `notebook/generate_signal_67b.ipynb` is hardcoded for MISO:
- `RTO = 'miso'`
- `SIGNAL_NAME = 'TEST.TEST.Signal.MISO.SPICE_F0P_V6.7B.R1'`
- `CLASS_TYPES = ['onpeak', 'offpeak']`
- Uses `MisoApTools`

PJM differs in:
- Signal name: `TEST.TEST.Signal.PJM.SPICE_F0P_V6.7B.R1`
- Class types: `onpeak`, `dailyoffpeak`, `wkndonpeak`
- Period types: f0–f11 (monthly only, no quarterly)
- AP Tools: `PjmApTools`
- ISO config: `PJM_ISO_CONFIG`

**Impact**: Cannot run PJM signal generation without notebook changes.

**Resolution**: Either:
- A: Parameterize the existing notebook with an RTO selector cell at the top
- B: Create a separate `generate_signal_67b_pjm.ipynb`

This is straightforward once Blocker 2 is resolved. The `signal_generator.py` module is already RTO-agnostic.

---

## What's Unblocked

| RTO | Period Types | Class Types | Status |
|-----|-------------|-------------|--------|
| MISO | f0, f1, f2, f3 (monthly) | onpeak, offpeak | Ready to run |
| MISO | q2, q3, q4 (quarterly) | onpeak, offpeak | Ready to run |
| MISO | aq1–aq4 (annual) | onpeak, offpeak | **Blocked** (#1) |
| PJM | f0–f11 (monthly) | onpeak | Ready after notebook parameterization (#3) |
| PJM | f0–f11 (monthly) | dailyoffpeak, wkndonpeak | **Blocked** (#2) |
