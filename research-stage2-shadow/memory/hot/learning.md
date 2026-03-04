## Accumulated Learning

### Iter 1 — Worker Failure Lessons
1. **Pipeline.py requires code modification for value_weighted**: The `train.py` already accepts `sample_weight`, and `config.py` has the `value_weighted` flag, but `pipeline.py` Phase 4 does NOT compute or pass sample weights. This is the critical wiring gap.
2. **Worker phantom completion is a real failure mode**: Worker wrote handoff JSON claiming "done" with artifact_path, but no artifacts existed. Future directions must include explicit verification steps.
3. **Direction complexity may contribute to failure**: Iter 1 direction asked for a code edit (pipeline.py) + 4 hyperparameter changes simultaneously. Simpler directions may have higher execution success rate.
4. **v0 baseline characteristics** (from champion.md):
   - EV-VC@100 mean=0.069, extreme variance (0.0001 to 0.194), 2 catastrophic months (2021-05, 2022-06)
   - EV-VC@500 mean=0.216, less variance but 2022-06 still weak (0.076)
   - Spearman mean=0.393, tail months 2021-11 (0.265) and 2022-06 (0.273)
   - 2022-06 is the universal worst month across nearly all metrics
