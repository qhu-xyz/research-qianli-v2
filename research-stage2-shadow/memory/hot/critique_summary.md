## Critique Summary

### Iter 1 — feat-eng-3-20260304-102111 (v0009, WORKER SUCCESS, DUPLICATE of v0008)

**Claude Review**:
- All Group A gates pass all 3 layers. EV-VC@100 +9.0% driven by 3 large wins (2020-09, 2020-11, 2021-09); 5/12 months degraded. Improvement is real but concentrated/fragile.
- v0009 is byte-identical to v0008 — duplicate registration, no new empirical data. Process issue: worker should detect duplicates.
- 5 of 39 features are zero-filled (hist_physical_interaction, overload_exceedance_product, band_severity, sf_exceed_interaction, hist_seasonal_band). Effective feature count is 34, not 39.
- Weak months cluster in late spring/summer and fall. 2021-05 and 2022-06 appear structurally classifier-limited — regressor improvements won't rescue these.
- Spearman is the binding constraint (4.7% margin to floor). Future iterations with worse Spearman could fail L1.
- Recommends: (1) Remove zero-filled features, (2) HP tuning for 34 effective features (colsample, mcw, n_estimators), (3) Value-weighted training.

**Codex Review**:
- Gates pass, frozen-classifier verified. Positive mean deltas are real but non-uniform. EV gains concentrated in few months; monthly direction mixed.
- **HIGH code finding** (repeated from prior batch): Gated regressor trains on true binding labels, not classifier-predicted positives. Train-inference mismatch at pipeline.py:195-209. Structural issue affecting all versions equally.
- **MEDIUM**: R-REC@500 computed from ev_scores, not regressor-only ranking — metric definition mismatch.
- **MEDIUM**: Feature importance pipeline dead code — prevents quantitative FE decisions.
- **MEDIUM**: Potential data-split leakage risk in data_loader.py:172-214 (train_end inclusive?).
- noise_tolerance=0.02 not scale-aware: makes L3 trivially easy for EV-VC@100 (bot2~0.007) and meaningless for C-RMSE (~5000).
- Recommends: (1) Fix gated-mode semantics, (2) Fix R-REC@500 metric, (3) Add anti-leakage assertions, (4) Enable feature importance emission, (5) Scale-aware noise tolerance.

**Synthesis**:
1. **Both agree**: v0009 is promotable, improvement is real but concentrated, duplicate version is a process issue
2. **Both agree**: Zero-filled features should be pruned, feature importance needed for data-driven FE decisions
3. **Both agree**: Spearman is the binding gate constraint; noise_tolerance needs scale-awareness
4. **Key divergence**: Codex surfaces train-inference mismatch (HIGH priority) and R-REC@500 definition issue — these are structural pipeline debts. Claude focuses more on next-iteration strategy (HP tuning, value-weighting).
5. **Neither reviewer** suggests classifier changes (correctly respecting freeze)
6. **Accumulated code debt**: train-inference mismatch (pipeline.py), dead feature importance, R-REC@500 definition, data-split leakage audit, stale business_context.md
