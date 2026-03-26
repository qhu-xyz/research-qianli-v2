# INDEX — research-annual-signal-pjm

Living table of contents. Update whenever new files are created.

## Handoff / Context

- `human-input/2026-03-25-pjm-handoff.md` — Phase 0-6 handoff from principal
- `human-input/pjm-spice-signal-annual.ipynb` — PJM exploratory notebook (V4.x signal generation)

## Docs — Contracts and Policies

- `docs/pjm-reconstruction-charter.md` — Source policy and modeling goal: beat released `V4.6` with a reproducible density+DA signal
- `docs/phase0-pjm-base-grain-contract.md` — PJM evaluation/publish grain definition
- `docs/phase1-data-source-contract.md` — Canonical paths, schemas, root-resolution rules
- `docs/pjm-metric-reporting-contract.md` — Metric suite and report structure
- `docs/pjm-annual-signal-implementation-plan.md` — Build plan matching MISO repo shape
- `docs/pjm-annual-publication-contract.md` — Publish schema, release surface, and unresolved publisher decisions for `pjm 7.0b`

## Docs — Analysis and Results

- `docs/phase1-coverage-report.md` — GT mapping + universe coverage analysis
- `docs/baseline-benchmark-results.md` — Current internal challengers vs released `V4.6` across all PYs × rounds × class types
- `docs/pjm-7.0b-publication-checklist.md` — Concrete pre-implementation checklist for `pjm 7.0b`
- `docs/agents.md` — Error and correction log (append-only)

## Scripts — Data Audit (Phase 1)

- `scripts/audit_coverage.py` — One-shot coverage audit for density, bridge, limit, SF, DA
- `scripts/gt_mapping_coverage.py` — GT mapping (monitored-line → branch) per quarter for checkpoint slice
- `scripts/universe_coverage.py` — Model-universe coverage, separate from GT
- `scripts/gt_recovery_sweep.py` — GT recovery sweep across all PYs with monitored-line matching + f0 fallback

## Scripts — Signal and Benchmark (Phase A-B)

- `scripts/sweep_all_ctypes.py` — Full sweep: all PYs × rounds × class types, peak-filtered DA, March cutoff
- `scripts/smoke_test_multi_cell.py` — Multi-cell smoke test for PJM 7.0b publisher (5 cells, validates schema/SF/overlap/round-trip)

## Config

- `CLAUDE.md` — Repo-level agent instructions

## Releases

- `releases/pjm/annual/7.0b/manifest.json` — Draft release manifest for `pjm 7.0b`
- `releases/pjm/annual/7.0b/smoke_test.json` — Smoke-test contract and results for `pjm 7.0b` (5/5 cells pass)
- `releases/pjm/annual/7.0b/smoke_test_results.json` — Detailed per-cell smoke test output
- `releases/pjm/annual/7.0b/dry_run_summary.json` — Original single-cell dry run output

## Implementation

- `ml/markets/pjm/signal_publisher.py` — PJM 7.0b publisher: candidate expansion, scoring, SF loading, tier assignment, validation
