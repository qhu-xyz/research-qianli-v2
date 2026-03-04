# Experiment Log

## Batch: ralph-v2-20260304-031811

### Iter 1
- **Screen months**: 2022-06 (weak), 2022-12 (strong)
- **Hypothesis A**: L2 reg (reg_lambda=5, mcw=25)
- **Hypothesis B**: L2 + subsample (reg_lambda=5, mcw=25, subsample=0.6, colsample=0.6)
- **Screen results**:
  - A: mean EV-VC@100=0.1069 (weak=0.0150, strong=0.1988)
  - B: mean EV-VC@100=0.0815 (weak=0.0159, strong=0.1470)
  - **Winner: A** (difference 0.0254 >> 0.002 threshold; B degraded strong month -24%)
- **Full benchmark v0005 (Hypothesis A)**:
  - EV-VC@100: mean=0.0735 (+6.5%), bot2=0.0084 (+23%)
  - EV-VC@500: mean=0.2287 (+5.9%), bot2=0.0689 (+23%)
  - EV-NDCG: mean=0.7501 (+0.4%), bot2=0.6458 (-0.3%)
  - Spearman: mean=0.3920 (-0.2%), bot2=0.2669 (-0.7%)
  - C-RMSE: mean=2907 (-7.2%), C-MAE: mean=1150 (-0.7%)
  - 9/12 months improved on EV-VC@100
- **Gate result**: NOT PROMOTABLE — Spearman L1 fails by 0.0008 (floor=0.3928, mean=0.3920). Calibration artifact: floor equals v0 exact mean.
- **Promoted**: No

### Iter 2
- **Screen months**: 2021-11 (weak Spearman), 2022-12 (strong EV-VC)
- **Hypothesis A**: Depth reduction (max_depth=5→4), keeping L2=5/mcw=25
- **Hypothesis B**: L1 regularization (reg_alpha=0.1→1.0), keeping L2=5/mcw=25
- **Screen results**:
  - A: mean Spearman=0.3249, mean EV-VC@100=0.1142 (weak month EV-VC@100 dropped -42%)
  - B: mean Spearman=0.3247, mean EV-VC@100=0.1154 (weak month EV-VC@100 dropped -23%)
  - **Winner: B** (override rule — A dropped EV-VC@100 by 0.0205 on 2021-11, >0.01 threshold)
- **⚠️ CONFIG BUG**: Full 12-month benchmark ran with reg_alpha=0.1 (old default), NOT 1.0. Evidence: config.json records 0.1; all per-month metrics identical to v0005. Both reviewers independently flagged.
- **Full benchmark v0006 (= v0005 due to bug)**:
  - All metrics identical to v0005 — no valid comparison possible
  - EV-VC@100: mean=0.0735, Spearman: mean=0.3920
- **Gate result**: NOT PROMOTABLE — Spearman L1 fails by 0.0008 (same as v0005)
- **Valid signal from screen only**: Neither depth=4 nor L1=1.0 recovered Spearman. Both degraded EV-VC@100 on weak month. Regularization axis exhausted for Spearman recovery.
- **Promoted**: No
