# Progress

## Current State
- **Champion**: v0 (baseline, unchanged)
- **Iterations completed**: 0 successful (all prior failed due to infrastructure bug, now fixed)
- **Version counter**: next_id=5 (leaked 4 times from failed iterations)
- **Next batch**: tier-fe-2 (FE only, 3 iterations)

## Root Cause of All Prior Failures

ALL worker failures (tier-fe-1 iter1-2, tier-fe-2 iter1) had the same root cause:
**Uncommitted changes to HUMAN-WRITE-ONLY files** caused the pre-merge guard to reject worker output.

The guard in `run_single_iter.sh` (line 116-121) diffs main working tree vs worktree.
If main has uncommitted edits to evaluate.py or gates.json, the worktree (branched from HEAD)
has the old version, and the guard sees a diff → rejects the worker.

**Fix**: All changes to evaluate.py, gates.json, and registry/v0/ are now committed to main (commit a2a38c5).

## Metric Redesign (v2 gates) — COMMITTED

**Group A (blocking)** — all tier-count invariant:
- Tier-VC@100, Tier-VC@500, Tier0-AP (new), Tier01-AP (new)

**Group B (monitor)** — no hard gates:
- Tier-NDCG, QWK, Macro-F1, Value-QWK, Tier-Recall@0, Tier-Recall@1

**Removed**: Tier-Accuracy, Adjacent-Accuracy

## Priority Improvement Areas
1. Tier-VC@100 below floor (0.071 vs 0.075) — only Group A gate failing Layer 1
2. Tier0-AP mean 0.306 — high variance (0.114 to 0.594), worst months late 2022
3. Tier01-AP mean 0.311 — barely passing, worst months 2022-06 (0.195), 2022-12 (0.194)
4. Tier-Recall@1 catastrophically low (0.047) — missing most strongly binding constraints
5. High variance across months — 2022-06 is the worst month across most metrics
