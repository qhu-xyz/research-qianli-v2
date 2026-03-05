# Progress Log

## Session: 2026-03-04

### Phase 1: Data Pull — COMPLETE
- Loaded annual R4 + monthly f0-f11 for PY 2020, 2021, 2022
- Validated data structure, units, row counts
- MCP is total $ (not $/MWh), no hourly column available
- ~13.5k-14.1k nodes per year, consistent inner joins

### Phase 2: Analysis — COMPLETE (multiple iterations)
- **Iteration 1** (scripts 01-05): Node-level system sum → misleading (Oct=17%, Jan=-8%)
- **Iteration 2** (scripts 07, 11-12): Hub-to-hub path tracing → showed path-dependency
- **Iteration 3** (scripts 14, 16): Annual cleared paths split by MTM sign → final result

### Phase 3: Audit — COMPLETE (script 15)
- Obligation MCPs verified: cleared.mcp matches node[sink]-node[source] for R4 obligations
- Options have different pricing → filtered to Obligation only
- Hub-to-hub cross-verified between scripts (diffs < 0.05%)
- All percentage sums verified = 100.000%

### Phase 4: Documentation — COMPLETE
- findings.md updated with final results, methodology, error corrections
- Final script: `16_final.py`
