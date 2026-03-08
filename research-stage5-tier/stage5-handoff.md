# Stage 5 Handoff

**All content consolidated into [`experiment-setup.md`](./experiment-setup.md).**

That document covers:
- Why stage 4 failed (circular evaluation)
- Complete data landscape (V6.2B, spice6 density, ml_pred, constraint_info, V6.4B)
- Constraint space alignment (V6.2B = rows, spice6 = 100% overlap, V6.4B = 23% — don't use)
- Clean feature set (18 features across 4 groups, no leakage)
- Ground truth (realized DA via `get_da_shadow_by_peaktype()`)
- Versioning plan (v0-v4+)
- Evaluation framework, gates, implementation checklist
- Reusable code from stage 4

## Current Status (2026-03-08)

### Completed
1. **V6.2B formula reproduction** — exact match (`max_abs_diff=0`, all 12 eval months)
2. **Realized DA ground truth** — verified via `get_da_shadow_by_peaktype()`, ~72 binding constraints/month
3. **v0 baseline** — V6.2B formula evaluated against realized DA (non-circular):
   - VC@20=0.28, VC@100=0.60, Recall@20=0.18, Spearman=0.20
   - These are the TRUE numbers (stage 4 showed 0.52/0.82/0.69/0.91 due to circularity)

### Next: Build ML Pipeline with Correct Ground Truth
- Modify `ml/pipeline.py` to use realized DA instead of `shadow_price_da`
- Cache realized DA to parquet (avoid repeated Ray calls)
- Run v1 (Groups A+B, 11 features, no da_rank_value)
