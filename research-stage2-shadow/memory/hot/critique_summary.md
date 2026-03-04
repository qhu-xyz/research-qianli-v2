## Critique Summary

### Iter 1 — ralph-v2-20260304-031811 (v0005, WORKER SUCCESS)

**Claude Review**:
- v0005 is a genuine, consistent improvement: EV-VC@100 +6.5%, EV-VC@500 +5.9%, C-RMSE -7.2%
- 9/12 months improved on EV-VC@100; gains largest on weak months (consistent with L2 hypothesis)
- Spearman -0.0008 is noise (t-stat ≈ -0.5, all per-month deltas within ±0.003)
- Gate calibration is dysfunctional: floors at v0 exact mean, v0 fails its own gates on EV-VC@100 and EV-NDCG
- Hypothesis B diagnosis: 0.6 subsampling starved trees of signal. Moderate 0.7 could be retested later.
- Recommends: gate recalibration (priority 1), then lr/trees interaction with L2, moderate subsampling, value-weighted
- Clean code review: 2 files, 4 lines, no bugs, no scope violations

**Codex Review**:
- Similar assessment: EV-VC gates improved, Spearman L1 blocks by 0.0008
- **HIGH code finding**: Train-inference mismatch in gated mode — regressor trains on true binding labels (`y_train_binary == 1`) but inferences on classifier predictions. Selection leakage that could bias fitting.
- **MEDIUM code findings**: Feature importance pipeline wired but never populated; frozen-classifier guardrails are weak in code/tests
- Spearman degraded in 9/12 months (opposite pattern to EV-VC improving in 9/12) — L2 compresses predictions, helping value capture but slightly hurting rank correlation
- Recommends: narrow reg_lambda/mcw sweep, fix gated training leakage, explicit feature importance export
- Gate calibration: noise_tolerance=0.02 is not scale-aware (C-RMSE at ~3000 scale makes 0.02 meaningless)

**Synthesis**:
1. **Both agree**: v0005 is a clear improvement, Spearman failure is a calibration artifact, gates need recalibration
2. **Key divergence**: Codex surfaces a pipeline-level train-inference mismatch (gated mode trains on true labels, infers on classifier predictions). This is a real issue but affects v0 equally — it's structural, not a v0005 regression. Out of scope for config-only changes.
3. **Codex's Spearman pattern observation** (degraded 9/12 months) is sharper than Claude's (only noted the mean shift). The L2-compresses-predictions → helps-value-capture → hurts-rank-correlation mechanism is a real tradeoff. Future directions should monitor this.
4. **Neither reviewer** suggests classifier changes (correctly respecting the freeze)
5. **Action items for code**: Feature importance export, classifier freeze guardrails, train-inference mismatch fix — all are pipeline improvements for future batches, not this iteration
