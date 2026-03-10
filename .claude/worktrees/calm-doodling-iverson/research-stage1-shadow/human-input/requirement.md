# Agentic system pipeline

## purpose
- fork from https://github.com/xyzpower/research-spice-shadow-price-pred. There are two pipelines. Identify and extract only the first Machine-Learning model. Use similar ideas given in this reference.md to set up an iterative, agent-driven process to improve model while keeping versions tracable. Minimize human work using 3-iteration-per-report strategy while keeping track of the process using ample reviews from AIs in a systematic/RALPH LOOP manner. You are effectively porting the "Software Engineering" agent pattern (OpenClaw) into a "Machine Learning Research" agent pattern. You are replacing "Code PRs" with "Model Iterations," and replacing "UI Screenshots" with "Evaluation Reports."
- Set up a promotion and model registry pipeline similar to (this pipeline forks from the same source but includes 2 stages. we are focusing ONLY on the first stage, classification). Difference: I will interact with the agents via claude Code and CLI, not using any apps. ** usage example: 
I type in claude code: "please go to this repo, and lower threshold to 0.4 to see what happens." Claude code agent will then process my command and start this iterative pipeline involving multiple agents and run for 3 runs (which includes thinking -> action -> observe -> review -> provide feedback)
** end of usage example
- set up metrics/gates. set up ample testing. push code using /checkin and use cron job to periodically review code and results and reports (this is not 100% the same as the reference.md. Define rules for a worker's work to be marked as "pass" to include reports, tests, codes, pushes, etc)
- evaluation metrics
  - goal: metrics should be consistent. we should only compare apple-to-apple metrics. for example, we should not compare precision of two models if they use different thresholds.

### things to note
- use claude CLI and codex CLI to use subscriptions. saves money
### coding agent
- use claude CLI subscription 4.6 opus (use worktree and tmux if you want like the reference)
### metrics. some metrics to consider:
- precision & recall for each threshold, F1 score
- capture@K. for the top-100 heaviest binding constraint (the constraints with the most shadow prices), how many do our model capture at each threshold?
- learn from repo /home/xyz/workspace/research-qianli-v2/research-spice-shadow-price-pred-qianli to see which metrics are used **reviewers should not only review results but also architecture. basicaly reviewers can say anything about this repo**
- Risk: If a Reviewer suggests lowering the "Pass" threshold because the model is performing poorly, you lose your "apple-to-apple" comparison. The version0 (benchmark) should ALWAYs be compared against. if you are introducing new methods and the comparison is not 100% the same between models (it can happen), make sure you find the best method to compare at least (curr_iter, best_iter, v0). Fix: Reviewers can suggest new metrics, but the Champion Gate can only be updated during the Human-in-the-loop Sync after Iteration 3. This is vital as sometimes after metrics are altered, there are significant reruns of older models involved.
### data
- we should set up beforehand the data. 
  - **each iteration should use the same data source**
  - if the reviewers see fit, they can point out that currently train/val split is not ideal and we need to change **but this should be carefully done and should provide after 3 iterations**

### model usage
In my mind this is how each iteration should go on:
1. if human input: orchestrator reads files and human input to understand the intent. writes files.
2. coders code according to orchestrator's guideline, write files
3. reviews (claude code CLI  + codex CLI review), write files
4. orchestrator process reviews and write next steps.
#### orchestrator 
- It manages the task registries, reads the Memory Files, and read reviewers comments. and decides which hyperparameter or architecture change to test next.
#### worker. we need to make sure we define a worker's completion by:
- command "claude --worktree" learn from the reference.md to see how to set up worktree and tmux sessions.
- it has implemented suggested changes, and ran the pipeline
- it has written tests and there are no compiling bugs
- we need to define standardized evaluation and gates beforehand. 
- it has run the updated codes and tests, and has compared metrics to decide if this new iteration is a pass/fail against metrics/gates
- it should produce a json indicating that it has finished.
#### reviewers
- use claude 4.6 opus and codex's best model to review.
  - **The reviewers has the authority and should always point out that the promotions gates are not sufficient and need to be changed if needed. We should not stick to stale promotion gates and metrics**
  - need to review: a. methodologies and whether the coding agent has successfully implmented last round's suggest changes (or direct human input). b. the results and if they are within expectation c. whether there are bugs, issues to fix, or is the conclusion reached correct d. provide feedback and suggestions for the next period
  - this is heavy work. since we are spinning up new AI reviewers each time, we need to **set up enough memory so that** they understand context and current work done.
#### 3-iter-per-report this process runs 3 iterations autonumously. After 3 iterations, have the Orchestrator force a "Human-in-the-loop" sync. And have the orchestrator check the entire repo and clean up scripts/codes. It should compile the reports into one "Executive Summary" and ping me.

### memory management
  - daily memory files 
  - and overall files: champion, daily logs and experiement_log, critique, runbook, learning, progress, (you may choose some of them depending on how you see fit and assign each AI to their respective files to maintain) 
### cron job:
- check if tmux session is alive, check which step are we in the process (is it orchestrator/coder/reviewer/orchestrator done?)
- Health Check: The Cron job checks the timestamp of the last log entry in the tmux session to see if max_seconds for each task (there should be **different max_seconds for each agent depending on job.**) Triage: If stuck, the Orchestrator spawns a "Debugger Agent" to read the tmux history, find the OOM or deadlock, and fix the code before restarting the iteration.
- example: If the Worker is stuck in a tmux session for 2 hours on a 10-minute training task, the Cron job kills the session and alerts the Orchestrator to "Fix Direction."
- reviewers/orchestrator should be able to point out and suggest alterations of any setting/params, such as max time out setting, etcetc
### things to note
- we don't need to checkin and push to remote. as long as we have complete loop between agents, we are fine. 
- orchestrator can adjust interval length of each cron job based on how long it takes for agents to complete: write,execute,run, compare, review, further suggestion from reviewers. orchestrator also need cron job to fix stuck agents or error of directions
- security risk: **the agents should only operate inside folder. they should not delete older versions**
  - No Deletions: The CLAUDE.md and runbook.md explicitly forbid rm -rf or overwriting old results_vN.json. Everything is additive.
  - **except for the cleaning up after 3 rounds are completed**
- how human interact:
  - i will mostly use claude code to interact. once the pipeline is built, i want to be able to do 2 things: a. interact with claude code, asking questions and ask claude code to modify certain parts of the pipeline. in this process, i do not want to trigger the entire pipeline b. say "let's start this 3-iter-per-report" process, and the process runs and gimmes results.

