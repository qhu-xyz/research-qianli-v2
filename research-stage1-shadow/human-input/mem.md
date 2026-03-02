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

## ML pipeline setup

- check /home/xyz/workspace/research-qianli-v2/research-spice-shadow-price-pred-qianli to see how we are testing each round. 
- for testing and getting results:
  1. do you think we need train/val/test split?
  2. one thought (not final plz argue with me if you dont agree) for any (auction month, ctype, ptype, ) use previous 12 months as train + val (10 + 2), then use that specific month to get results. do this for many months to get benchmarks
  - if we are only evaluating f0 + f1, maybe this is not enough. how about this?
    - we use gates in steps; a. we use some months, spanning year 2020-2022 to test only f0's gates and metrics. if the new version passes, go on to f1; if f1 passes go on to next periods. test f0 most extensively and decreasing by ptypes
    - total validation months should not exceed 32 months. 
    - after we've finalized this, we need to re-organize the gates & promotions. we need to re-calculate baselines and get new floors, right?
    - also we need to consider using **monitor gates not hard gates for some** this is a LOT of gates to pass, and if there are some outliers i think it is very difficult to pass all gates even if the newer version of model is superior.
Remark:
- this is NOT about running our pre-designed pipeline for 3-iter-per-report. this is about setting it up
- however, if you think agent team is valuable, install claude agent team and run tasks in parallel.

1. also, this version v000 baseline has threshold tuned right? i think this is not necessary for further rounds. what if in version next, version v001, we now use threshold 0.5, and version v002 works best with threshold 0.7? if we just compare them brute-force, this is comparing apples-to-oranges.
2. okay i agree with your proposal. let's use the strict cascade.
3. rolling window - which months do u propose?

### Benchmark results
1. why does f1 skipped 2021-08, why val empty?
2. for any period type, if we use 10 trn + 2 val, why not trace back the most recent 12 months available? does this fix f2's issue or even longer period types?

- why "Skips if market_month >= train_end (2021-08)" is present? my understand of the data, is the below correct?
each row contains columns:
auction month | ptype, ctype | constraint name, features | target --> target tracks whether in the future, in market month, whether this constraint will bind.

if so: 
1. why do we have this market_month >= train_end guard? and why train_end in your f1 example is the same as target month?
2. for f2 for example, if target month is 2025-03 (assume 2025-03 has f2), then why cannot we use 2025-02, 2025-01 for training?

My reasoning to skip the guard: 
The guard prevents training on labels that don't exist yet at prediction time. ==> is this a valid concern? I am not sure. for example, for the same ptype, ctype, different auction month, they do not interfere with each other because their market month is different. so even though yes, the target of the market month in val may go beyond what we have during inference
