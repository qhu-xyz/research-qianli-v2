# PJM Annual Band Research

## Status
Baseline research complete. Banding phase starting.

## Key Decisions
- **Baseline:** `mtm_1st_mean * 12` (annual scale) for all 4 rounds
- **R1 source:** Long-term yr1 Round 5 MCP (~March, non-leaky)
- **R2-R4 source:** Prior annual round MCP (same April auction event)
- **Classes:** `onpeak`, `dailyoffpeak`, `wkndonpeak`
- **Filter:** `hedge_type == 'obligation'` only
- **Scale:** Annual total (`mcp` as target, `mtm_1st_mean * 12` as baseline)

## Files
- `knowledge.md` — Full research log with all findings and data
- `versions/v1/r{1-4}/metrics.json` — Preliminary band calibration results

## Data
- Raw: `/opt/temp/qianli/annual_research/pjm_annual_cleared_all.parquet` (238 MB)
- With MTM: `/opt/temp/qianli/annual_research/pjm_annual_with_mcp.parquet` (403 MB)
- Nodal f0 lookup: `/opt/temp/qianli/annual_research/pjm_nodal_f0_lookup.parquet` (not used — nodal f0 is worse than mtm_1st)

## Next
1. Write PJM band script (adapt from MISO `miso/scripts/run_v9_bands.py`)
2. Full dev run with temporal CV
3. Holdout validation (PY 2025)
4. Comprehensive report
