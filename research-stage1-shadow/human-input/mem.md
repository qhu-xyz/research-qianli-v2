# Intention Memo (Compressed)

## Core Goal
Build an agentic pipeline for Stage-1 shadow-price classification with a 3-iteration autonomous loop:
1. Orchestrator plans
2. Worker implements + runs tests/eval
3. Claude + Codex reviewers critique results and architecture
4. Orchestrator synthesizes and plans next iteration

After 3 iterations, force HUMAN_SYNC and produce an executive summary.

## Must-Have Constraints
- Keep comparisons apple-to-apple inside each 3-iteration batch.
- Keep `v0` always present as baseline.
- Gate/metric policy can be suggested by reviewers but only changed at HUMAN_SYNC.
- Same data source across iterations in a batch.
- Preserve traceability (versions, reports, critiques).
- Use subscriptions: Claude Opus 4.6 for orchestrator/worker/Claude reviewer; Codex for one reviewer.
- Operate safely inside repo; no destructive deletes of historical artifacts.

## Open Questions I Care About
- Is my intention fully clear from docs?
- Do model artifacts consume significant disk, and what storage policy should we enforce?
- Do we need explicit handshake contracts between agents (especially Codex reviewer handoff)?
- What is the clean CLI invocation strategy given `-p` differences between Claude and Codex?
- What is the fastest reliable smoke test to validate the full loop?

## Checklist
- Confirm both Claude and Codex subscriptions are used as intended.
- Confirm shell scripts support both scenarios:
  1. Human-triggered command -> full loop -> orchestrator receives reviewer output -> next iteration planning.
  2. Continue mode using prior round context/planning without fresh human input.

## Note
Detailed technical addenda belong in `docs/plans/...`, not in this memo.

## answer those questions one by one:
- now does the reviewers focus not only on recent code changes (the codering agent should produce a summary), but also on the metrics and see if they are too strict, too loose, or just do not provide enough info and need to be changed? this is important -> i think the thresholds groups make sense but we need to run and see the results to decide further.

1. have we set up the CLIs for claude and codex? you wanna quickly test it?
2. after your first implementation, are we gonna test it to make sure that each component in the pipeline works? I want to test the integrity of the pipeline. 
3. do we have cron jobs? what do they do? can we quickly test them ?
4. memory: how are the memory system designed? for a claude code coder, what info will be passed into his context?
5. do we have two reviewers, one codex, one claude?