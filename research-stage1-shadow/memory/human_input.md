# Human Input — First Batch

## Objective
Run the shadow price classifier pipeline in SMOKE_TEST mode to verify the full agentic loop works end-to-end. This is an infrastructure validation batch, not a real experiment.

## What to test this batch
- Iteration 1: Run the pipeline with default hyperparameters (same as v0 baseline). The goal is NOT to beat v0 — it's to verify that every component works: version allocation, pipeline execution, gate comparison, reviews, and synthesis.
- If the worker produces identical metrics to v0, that's fine — it confirms determinism.
- Focus the review on whether the infrastructure is correct, not on ML improvements.

## Context
- We are running on synthetic data (100 rows, SMOKE_TEST=true)
- v0 baseline exists with populated gate floors
- This is the first real batch — there is no prior experiment history
- Auction month: 2021-07, class: onpeak, period: f0
