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
2. **Key divergence**: Codex surfaces a pipeline-level train-inference mismatch (gated mode trains on true labels, infers on classifier predictions). Real issue but affects v0 equally — structural, not a regression.
3. **Codex's Spearman pattern observation** (degraded 9/12 months) is sharper than Claude's. The L2-compresses-predictions → helps-value-capture → hurts-rank-correlation mechanism is a real tradeoff.
4. **Neither reviewer** suggests classifier changes (correctly respecting the freeze)

---

### Iter 2 — ralph-v2-20260304-031811 (v0006, WORKER SUCCESS but CONFIG BUG)

**Claude Review**:
- **CRITICAL**: Identified v0006 full benchmark ran with wrong config (reg_alpha=0.1 instead of 1.0). Evidence: config.json records 0.1; per-month values exactly match v0005.
- Screen data IS valid. Neither hypothesis recovered Spearman meaningfully. Both degraded EV-VC@100 on weak month.
- Recommends: Fix config bug, value-weighted training, escalate gate calibration urgently.

**Codex Review**:
- Independently confirmed provenance inconsistency. Suggests artifact-level integrity checks.
- **MEDIUM**: Test suite misaligned (13/24 vs actual 14/34).
- Recommends: unified_regressor, max_depth=4 or mcw/reg_lambda relaxation.

**Synthesis**:
1. **Both agree**: v0006 invalid (config bug), regularization axis exhausted, gates dysfunctional
2. **Key divergence**: Codex suggests unified_regressor; Claude warns against it. Claude's reasoning stronger — gated mode aligns with business objective.

---

### Iter 3 — ralph-v2-20260304-031811 (v0007, FIRST PROMOTABLE VERSION)

**Claude Review**:
- v0007 is the first promotable version. Spearman margin razor-thin (+0.0004).
- EV-VC@500 is the standout metric (+6.2%, 9/12 months improved, robust).
- Per-month Spearman: 5 improved, 7 degraded — mean improvement driven by early months.
- Decomposition confirmed: reg_lambda compresses predictions → hurts Spearman; mcw is orthogonal.
- Code: clean. Flagged stale business_context.md (14 clf/34 reg vs documented 13/24), removed frozen-dataclass test needs verification.
- Recommends: promote, recalibrate gates, then value-weighting / mcw sweep / feature pruning.

**Codex Review**:
- v0007 promotable but "narrow-pass" — Spearman and EV-NDCG each at tail-failure limit (1 bad month).
- **HIGH**: Potential temporal leakage in data_loader.py (train_end inclusive?) — structural, not v0007-specific.
- **MEDIUM**: Feature importance still dead; value_weighted still unwired.
- Classifier freeze: **PASS** (config.json verified).
- Next should prioritize tail robustness, not mean chasing.

**Synthesis**:
1. **Both agree**: v0007 promotable, Spearman margin is noise-level, EV-VC@500 is the real improvement
2. **Both agree**: Gate calibration urgently needs HUMAN_SYNC (3rd iteration flagging this)
3. **Both agree**: Classifier freeze respected, no code bugs in this iteration
4. **Key divergence**: Codex raises temporal leakage concern (data_loader.py train_end inclusive?). Valid audit point but structural — affects all versions equally.
5. **Claude more emphatic on gate recalibration**, Codex more focused on tail robustness as next priority. Both valid — recalibrate gates AND focus on tail months.
6. **Accumulated code debt** (for future batches): train-inference mismatch, dead feature importance, stale business_context.md, test assertions corrected but frozen-dataclass test removed, temporal leakage audit needed.
