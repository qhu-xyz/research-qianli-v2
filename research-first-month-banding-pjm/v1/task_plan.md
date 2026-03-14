# PJM First Monthly Auction MCP Distribution Research

## Goal
Derive a percentage distribution of MCP across f0–f11 for PJM's June monthly auction, replacing the naive `annual_mcp / 12` approach.

## Phases

### Phase 1: Data Pull & Validation `complete`
- [x] Load annual auction MCP (last round) for PY 2020, 2021, 2022
- [x] Load June monthly auction MCP for f0–f11 for the same years
- [x] Validate data: row counts, columns, spot-check values

### Phase 2: Analysis `complete`
- [x] For each path: compute monthly MCP as % of sum(f0..f11)
- [x] Compare sum(f0..f11) vs annual MCP to see divergence
- [x] Aggregate across paths to get average distribution
- [x] Check if distribution is stable across years

### Phase 3: Report `in_progress`
- [x] Summarize findings with tables
- [ ] Report to user
- [ ] Update mem.md with actual results

## Key Decisions
| Decision | Choice | Reason |
|----------|--------|--------|
| Annual round | Last round (round 4) | Most recent info before monthly |
| Years | 2020-2022 | Per user request |
| Auction month | June only | First monthly of planning year |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
