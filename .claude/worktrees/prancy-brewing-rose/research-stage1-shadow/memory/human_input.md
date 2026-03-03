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

## r1
on structure
- this is a sandbox environment right? meaning that once the round gets triggered i can close this session and all the modules in pipeline works
- what is the exact command for me to begin a 3-iter-per-report?
- where do i see the summary reports from the rounds?
- what does cron job currently do?
- so we are ONLY using both subscriptions from claude and openAI right, I DON"T want to use api
- now does the pipeline support these two scenarios? 
1. i type in command: "i think we should generate more useful features" then orchestrator picks up and gimme the worker command -> coder codes -> 2 reviewers review *independently* -> review summary given by orchestrator, who gives out a final opinion 
2. if we running in 3-iter-per-report: orchestrator picks up last rounds report and gimme the worker command -> coder codes -> 2 reviewers review *independently* -> review summary given by orchestrator, who gives out a final opinion
- do u think adding cron job helps?
- do u think we can set the whole process, the controller script in tmux by default?
- "Not worth it until we build actual recovery logic." -> why don't we do that?
- also, is there **any risk** that the agents stuck in infinite loop? any risk that we burn tokens without knowing when and how to quit?


- now compare your implementation with the design md file and the implementation md file. have you implemented everything? (or have you implemented everything that makes sense? we even added something in our current design, so you need to compare **everything** and update to the latest. if sth is in design or implementation file but not done, do it.)
    - my teammates may want to build a similar closed loop involving **pipeline, state, multiple reviewers from miltiple agents** your file should help them in that they will run into no friction in knowing the essences and easy pitfalls.
- also, **create a runbook.md** for this whole repo, so next time when I talk with any claude code agent, they know how the pipeline is roughly set up and how to evoke it.

On the ML pipeline itself
- what are the dataset used gimme details
- what are the gates used currently? promotion is defined by how?
