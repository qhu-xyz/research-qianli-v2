## Accumulated Learning

### Worker Reliability (2 consecutive failures)
1. **Phantom completion is the dominant failure mode**: Both iterations had worker write `worker_done.json` claiming "done" while producing zero artifacts. The handoff signal is not trustworthy — must verify artifacts exist.
2. **Workers can make unauthorized changes**: Iter 1 (ralph-v1) worker modified the frozen ClassifierConfig, evaluate.py (HUMAN-WRITE-ONLY), and 6 other files despite direction saying "no code changes needed". This leaves the codebase dirty.
3. **Codebase must be clean-checked before each iteration**: Uncommitted changes from failed workers contaminate the next iteration. Add `git checkout -- ml/` to pre-worker cleanup.
4. **Direction complexity correlates with failure**: smoke-test direction required code edits → failed. ralph-v1 direction required only hyperparameter overrides → worker still failed but also went off-script. Both directions may have been too long / information-dense.
5. **Simplify radically for iter 2**: Single hypothesis, exact copy-paste commands, explicit file prohibition list, verification gates.

### Pipeline Architecture
1. **Pipeline.py requires code modification for value_weighted**: `train.py` accepts `sample_weight`, `config.py` has the flag, but `pipeline.py` Phase 4 does NOT wire it. Defer this hypothesis until override-only hypotheses are exhausted.

### v0 Baseline Characteristics
- EV-VC@100 mean=0.069, extreme variance (0.0001 to 0.194), catastrophic months: 2021-05, 2022-06
- EV-VC@500 mean=0.216, less variance but 2022-06 still weak (0.076)
- Spearman mean=0.393, tail months 2021-11 (0.265) and 2022-06 (0.273)
- 2022-06 is the universal worst month across nearly all metrics
- Gate floors are set to v0 means — the bar is "don't regress" not "improve"
