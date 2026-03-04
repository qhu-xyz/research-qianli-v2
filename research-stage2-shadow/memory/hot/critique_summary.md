## Critique Summary

### Iter 2 — feat-eng-3-20260304-121042 (v0012, WORKER SUCCESS, PROMOTED)

**Claude Review**:
- All Group A gates pass all 3 layers. EV-VC@500 +3.5%, tail failure eliminated (0.0527→0.0720). EV-VC@500 no longer the binding constraint.
- EV-VC@100 -5.3% is real (7/12 months degrade), concentrated in 2022-12 (-21.8%) and 2022-03 (-15.9%). +14.2% margin absorbs it.
- Weak month improvements on EV-VC@500 are the desired pattern: floor lifted without pulling down ceiling.
- EV-NDCG +0.2% (7/12 months), Spearman +0.4% (8/12 months) — consistent, not outlier-driven.
- Code review: PASS. Minimal 2-line HP change + test assertions, correct.
- **Recommends next iter**: (1) mcw reduction 25→15-20 for top-100 sharpening, (2) value_weighted=True for high-$ emphasis, (3) Do NOT touch features or lr/n_estimators further.
- **Gate calibration**: noise_tolerance still non-scale-aware (repeated), EV-VC@100 tail_floor still non-protective (repeated). No recalibration needed this iter.

**Codex Review**:
- Gates pass, frozen-classifier verified. Net EV-VC@500 improvement is partly concentrated (largest gains: 2021-11, 2021-05, 2022-09) rather than uniform uplift.
- **HIGH code finding**: gated regressor trains on true labels (`y_train_binary==1`) not classifier predictions — train/inference mismatch, target-aware selection risk. Location: pipeline.py:195, 208. Repeated from prior iter.
- **LOW**: _apply_config_overrides docstring claims classifier overrides supported but code ignores them. Repeated.
- Frozen-classifier check: PASS. v0012 classifier block identical to v0011.
- **Recommends next iter**: (1) Fix gated-training semantics (HIGH priority), (2) Add per-version feature importance output, (3) If EV-VC@100 recovery needed, tune mcw/depth/colsample, (4) Unified regressor mode as controlled ablation after fixing gated semantics.
- **Gate calibration**: Do not recalibrate downward. Switch to scale-aware noise_tolerance per metric.

**Synthesis**:
1. **Both agree**: v0012 is promotable, achieves the primary EV-VC@500 recovery objective
2. **Both agree**: EV-VC@100 -5.3% is real but absorbed by margin; mcw or value_weighted are next levers
3. **Both agree**: Scale-aware noise_tolerance needed (3rd consecutive iteration flagging this); no gate recalibration
4. **Key divergence**: Claude focuses on HP tuning specifics (mcw, value_weighted); Codex emphasizes fixing gated-training semantics first (HIGH finding, pipeline.py:195-208)
5. **Code debt (accumulated, unfixed)**: train-inference mismatch (pipeline.py), dead feature importance, data-split leakage audit, pipeline.py docstring mismatch
6. **Neither reviewer** suggests classifier changes (correctly respecting freeze)
7. **New this iter**: Both note colsample=0.9 did not help — breadth recovery required more ensemble rounds, not more features per tree

### Prior Iter Critique (Iter 1, preserved for context)

Both reviewers agreed v0011 was promotable with precision-vs-breadth tradeoff. EV-VC@500 tail margins were flagged as critical risk — correctly identified, now resolved by v0012.
